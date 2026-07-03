"""Train the MVP-6 RHEED temporal morphology condition encoder."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rheed2morph.generative.common import display_path, read_csv_rows, read_json, resolve_repo_path, resolve_torch_device, set_seed, write_csv_rows, write_json
from rheed2morph.generative.models.rheed_mae import load_mae_encoder_state
from rheed2morph.generative.models.rheed_temporal_encoder import RheedTemporalMorphologyEncoder


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RHEED temporal morphology encoder v2.")
    parser.add_argument("--paired-index", type=Path, required=True)
    parser.add_argument("--condition-schema", type=Path, required=True)
    parser.add_argument("--mvp5-root", type=Path, default=Path("reports/afm_prior_v4_height_calibrated/20260703_064826"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--frame-encoder", type=str, default="small_cnn", choices=["small_cnn", "resnet18", "resnet50", "mae_encoder"])
    parser.add_argument("--frame-mae-checkpoint", type=Path, default=None)
    parser.add_argument("--freeze-frame-encoder", type=str_to_bool, default=False)
    parser.add_argument("--temporal-pooling", type=str, default="attention", choices=["mean", "max", "attention", "gru", "transformer", "final"])
    parser.add_argument("--use-visual", type=str_to_bool, default=True)
    parser.add_argument("--use-handcrafted", type=str_to_bool, default=True)
    parser.add_argument("--use-metadata", type=str_to_bool, default=True)
    parser.add_argument("--predict-uncertainty", type=str_to_bool, default=False)
    parser.add_argument("--target-schema", type=str, default="v3", choices=["v3", "v4", "shared"])
    parser.add_argument("--loss", type=str, default="mse", choices=["mse", "heteroscedastic"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--label-fraction", type=float, default=1.0)
    parser.add_argument("--shuffle-labels", type=str_to_bool, default=False)
    parser.add_argument("--shuffle-videos", type=str_to_bool, default=False)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _load_schema(path: Path, paired_index: Path) -> dict[str, Any]:
    schema = read_json(resolve_repo_path(path))
    data_dir = resolve_repo_path(paired_index).parent
    local = data_dir / "condition_schema_v3_mvp6.json"
    if local.is_file():
        enriched = read_json(local)
        if list(enriched.get("condition_columns", [])) == list(schema.get("condition_columns", enriched.get("condition_columns", []))):
            schema.update({key: value for key, value in enriched.items() if key.startswith("rheed_") or key == "metadata_columns"})
    if "rheed_feature_columns" not in schema:
        feature_rows = read_csv_rows(data_dir / "rheed_handcrafted_features.csv") if (data_dir / "rheed_handcrafted_features.csv").is_file() else []
        if feature_rows:
            schema["rheed_feature_columns"] = [key for key in feature_rows[0] if key not in {"pair_id", "row_id", "sample_id", "group_id", "split"}]
        else:
            schema["rheed_feature_columns"] = []
    schema.setdefault("metadata_columns", ["source_frame_count", "frames_used", "image_size", "final_fraction"])
    return schema


def _stats(rows: list[dict[str, str]], columns: list[str], train_ids: set[str], id_key: str = "pair_id") -> tuple[dict[str, float], dict[str, float]]:
    if not columns:
        return {}, {}
    selected = [row for row in rows if row.get(id_key, "") in train_ids] or rows
    matrix = np.asarray([[finite_float(row.get(col, "nan"), float("nan")) for col in columns] for row in selected], dtype=np.float64)
    med = np.nanmedian(matrix, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    for col in range(matrix.shape[1]):
        mask = ~np.isfinite(matrix[:, col])
        matrix[mask, col] = med[col]
    mean = matrix.mean(axis=0)
    std = np.where(matrix.std(axis=0) > 1e-8, matrix.std(axis=0), 1.0)
    return ({col: float(mean[i]) for i, col in enumerate(columns)}, {col: float(std[i]) for i, col in enumerate(columns)})


class RheedMorphologyDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, str]],
        features_by_pair: dict[str, dict[str, str]],
        schema: dict[str, Any],
        split: str,
        scaler: dict[str, Any],
        use_visual: bool,
        visual_mode: str = "all",
        limit: int | None = None,
        label_fraction: float = 1.0,
        seed: int = 42,
        shuffle_labels: bool = False,
        shuffle_videos: bool = False,
    ) -> None:
        selected = [row for row in rows if row.get("split") == split]
        if split == "train" and label_fraction < 1.0 and selected:
            rng = np.random.default_rng(int(seed))
            keep = max(1, int(math.ceil(len(selected) * float(label_fraction))))
            indices = sorted(rng.choice(len(selected), size=keep, replace=False).tolist())
            selected = [selected[i] for i in indices]
        if limit is not None:
            selected = selected[: int(limit)]
        self.rows = selected
        self.features_by_pair = features_by_pair
        self.schema = schema
        self.scaler = scaler
        self.use_visual = bool(use_visual)
        self.visual_mode = str(visual_mode)
        self.feature_columns = list(schema.get("rheed_feature_columns", []))
        self.metadata_columns = list(schema.get("metadata_columns", []))
        self.target_columns = list(schema["condition_columns"])
        self.descriptor_columns = list(schema["descriptor_columns"])
        self.shuffle_labels = bool(shuffle_labels)
        self.shuffle_videos = bool(shuffle_videos)
        rng = np.random.default_rng(int(seed) + 17)
        self.label_order = rng.permutation(len(self.rows)).tolist() if self.shuffle_labels and self.rows else list(range(len(self.rows)))
        self.video_order = rng.permutation(len(self.rows)).tolist() if self.shuffle_videos and self.rows else list(range(len(self.rows)))

    def __len__(self) -> int:
        return len(self.rows)

    def _video(self, row: dict[str, str]) -> torch.Tensor:
        frames = np.asarray(np.load(resolve_repo_path(Path(row["cached_tensor_path"])))["frames"], dtype=np.float32)
        if self.visual_mode == "final":
            frames = frames[-1:,:,:,:]
        elif self.visual_mode == "final_average":
            tail = frames[max(0, len(frames) - min(4, len(frames))) :]
            frames = np.repeat(tail.mean(axis=0, keepdims=True), repeats=min(4, len(tail)), axis=0)
        return torch.from_numpy(frames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        label_row = self.rows[self.label_order[index]]
        video_row = self.rows[self.video_order[index]]
        video = self._video(video_row) if self.use_visual else torch.zeros((0,), dtype=torch.float32)
        feature_row = self.features_by_pair.get(row["pair_id"], {})
        f_mean = self.scaler.get("feature_mean", {})
        f_std = self.scaler.get("feature_std", {})
        handcrafted = torch.tensor(
            [(finite_float(feature_row.get(col, "nan")) - f_mean.get(col, 0.0)) / max(f_std.get(col, 1.0), 1e-8) for col in self.feature_columns],
            dtype=torch.float32,
        )
        m_mean = self.scaler.get("metadata_mean", {})
        m_std = self.scaler.get("metadata_std", {})
        metadata = torch.tensor(
            [(finite_float(row.get(col, "nan")) - m_mean.get(col, 0.0)) / max(m_std.get(col, 1.0), 1e-8) for col in self.metadata_columns],
            dtype=torch.float32,
        )
        target = torch.tensor([finite_float(label_row.get(col, "nan")) for col in self.target_columns], dtype=torch.float32)
        raw = torch.tensor([finite_float(label_row.get(col, "nan")) for col in self.descriptor_columns], dtype=torch.float32)
        proto_text = label_row.get("prototype_id", "")
        proto = int(float(proto_text)) if proto_text != "" else -1
        return {
            "video": video,
            "handcrafted": handcrafted,
            "metadata": metadata,
            "target": target,
            "raw_descriptor": raw,
            "prototype": torch.tensor(proto, dtype=torch.long),
            "row_id": row.get("row_id", ""),
            "pair_id": row.get("pair_id", ""),
            "sample_id": row.get("sample_id", ""),
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("handcrafted", "metadata", "target", "raw_descriptor", "prototype"):
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    out["video"] = torch.stack([item["video"] for item in batch], dim=0) if batch and batch[0]["video"].numel() else None
    for key in ("row_id", "pair_id", "sample_id"):
        out[key] = [item[key] for item in batch]
    return out


def descriptor_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.size == 0:
        return {"descriptor_mse": float("nan"), "descriptor_mae": float("nan"), "descriptor_rmse": float("nan"), "descriptor_r2": float("nan"), "descriptor_spearman": float("nan")}
    err = y_pred - y_true
    mse = float(np.mean(err * err))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - np.sum(err * err) / denom) if denom > 1e-12 else float("nan")
    spearman_values: list[float] = []
    try:
        from scipy.stats import spearmanr

        for col in range(y_true.shape[1]):
            value = spearmanr(y_true[:, col], y_pred[:, col]).correlation
            if value is not None and np.isfinite(value):
                spearman_values.append(float(value))
    except Exception:
        pass
    return {
        "descriptor_mse": mse,
        "descriptor_mae": mae,
        "descriptor_rmse": float(math.sqrt(mse)),
        "descriptor_r2": r2,
        "descriptor_spearman": float(np.mean(spearman_values)) if spearman_values else float("nan"),
    }


def prototype_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: int) -> dict[str, float]:
    mask = y_true >= 0
    if classes <= 0 or not np.any(mask):
        return {"prototype_accuracy": float("nan"), "prototype_macro_f1": float("nan")}
    true = y_true[mask]
    pred = y_pred[mask]
    acc = float(np.mean(true == pred))
    f1s = []
    for cls in range(classes):
        tp = float(np.sum((true == cls) & (pred == cls)))
        fp = float(np.sum((true != cls) & (pred == cls)))
        fn = float(np.sum((true == cls) & (pred != cls)))
        denom = 2.0 * tp + fp + fn
        f1s.append(0.0 if denom <= 0 else 2.0 * tp / denom)
    return {"prototype_accuracy": acc, "prototype_macro_f1": float(np.mean(f1s))}


def _make_loaders(args: argparse.Namespace, visual_mode: str = "all") -> tuple[dict[str, DataLoader], dict[str, Any], dict[str, Any]]:
    paired_index = resolve_repo_path(args.paired_index)
    rows = read_csv_rows(paired_index)
    schema = _load_schema(args.condition_schema, paired_index)
    feature_path = paired_index.parent / "rheed_handcrafted_features.csv"
    feature_rows = read_csv_rows(feature_path) if feature_path.is_file() else []
    features_by_pair = {row["pair_id"]: row for row in feature_rows}
    train_ids = {row["pair_id"] for row in rows if row.get("split") == "train"}
    feature_mean, feature_std = _stats(feature_rows, list(schema.get("rheed_feature_columns", [])), train_ids)
    metadata_mean, metadata_std = _stats(rows, list(schema.get("metadata_columns", [])), train_ids)
    scaler = {"feature_mean": feature_mean, "feature_std": feature_std, "metadata_mean": metadata_mean, "metadata_std": metadata_std}
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = RheedMorphologyDataset(
            rows,
            features_by_pair,
            schema,
            split,
            scaler,
            use_visual=bool(args.use_visual),
            visual_mode=visual_mode,
            limit=args.limit if not args.quick else (args.limit or 16),
            label_fraction=float(args.label_fraction) if split == "train" else 1.0,
            seed=int(args.seed),
            shuffle_labels=bool(args.shuffle_labels) and split == "train",
            shuffle_videos=bool(args.shuffle_videos) and split == "train",
        )
        if len(dataset) == 0 and split == "val":
            dataset = RheedMorphologyDataset(rows, features_by_pair, schema, "train", scaler, bool(args.use_visual), visual_mode, limit=min(args.limit or 8, 8), seed=int(args.seed))
        loaders[split] = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=(split == "train"), num_workers=int(args.num_workers), collate_fn=_collate)
    return loaders, schema, scaler


def _loss(pred: dict[str, torch.Tensor], target: torch.Tensor, prototype: torch.Tensor, proto_weight: float, mode: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode == "heteroscedastic" and "log_variance" in pred:
        logvar = pred["log_variance"]
        desc_loss = 0.5 * (torch.exp(-logvar) * (pred["descriptor"] - target).pow(2) + logvar).mean()
    else:
        desc_loss = F.mse_loss(pred["descriptor"], target)
    proto_loss = torch.tensor(0.0, device=target.device)
    if "prototype_logits" in pred and torch.any(prototype >= 0):
        mask = prototype >= 0
        proto_loss = F.cross_entropy(pred["prototype_logits"][mask], prototype[mask])
    return desc_loss + proto_weight * proto_loss, desc_loss, proto_loss


def _run_epoch(model: RheedTemporalMorphologyEncoder, loader: DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None, scaler: torch.amp.GradScaler | None, amp_enabled: bool, config: dict[str, Any], proto_weight: float) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total = {"loss": 0.0, "descriptor_loss": 0.0, "prototype_ce": 0.0}
    count = 0
    for batch in loader:
        video = batch["video"].to(device) if config["use_visual"] and batch["video"] is not None else None
        handcrafted = batch["handcrafted"].to(device) if config["use_handcrafted"] else None
        metadata = batch["metadata"].to(device) if config["use_metadata"] and batch["metadata"].shape[1] > 0 else None
        target = batch["target"].to(device)
        proto = batch["prototype"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred = model(video=video, handcrafted=handcrafted, metadata=metadata)
                loss, desc_loss, proto_loss = _loss(pred, target, proto, proto_weight, str(config.get("loss", "mse")))
        if not torch.isfinite(loss):
            raise RuntimeError("RHEED morphology encoder loss became non-finite.")
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        batch_size = int(target.shape[0])
        count += batch_size
        total["loss"] += float(loss.detach().cpu()) * batch_size
        total["descriptor_loss"] += float(desc_loss.detach().cpu()) * batch_size
        total["prototype_ce"] += float(proto_loss.detach().cpu()) * batch_size
    return {key: value / max(count, 1) for key, value in total.items()}


@torch.no_grad()
def predict_arrays(model: RheedTemporalMorphologyEncoder, loader: DataLoader, device: torch.device, config: dict[str, Any]) -> dict[str, Any]:
    model.eval()
    true: list[np.ndarray] = []
    pred: list[np.ndarray] = []
    raw: list[np.ndarray] = []
    logvar: list[np.ndarray] = []
    proto_true: list[np.ndarray] = []
    proto_pred: list[np.ndarray] = []
    row_ids: list[str] = []
    pair_ids: list[str] = []
    sample_ids: list[str] = []
    for batch in loader:
        video = batch["video"].to(device) if config["use_visual"] and batch["video"] is not None else None
        handcrafted = batch["handcrafted"].to(device) if config["use_handcrafted"] else None
        metadata = batch["metadata"].to(device) if config["use_metadata"] and batch["metadata"].shape[1] > 0 else None
        out = model(video=video, handcrafted=handcrafted, metadata=metadata)
        true.append(batch["target"].numpy())
        pred.append(out["descriptor"].detach().cpu().numpy())
        raw.append(batch["raw_descriptor"].numpy())
        if "log_variance" in out:
            logvar.append(out["log_variance"].detach().cpu().numpy())
        proto_true.append(batch["prototype"].numpy())
        if "prototype_logits" in out:
            proto_pred.append(torch.argmax(out["prototype_logits"], dim=1).detach().cpu().numpy())
        else:
            proto_pred.append(np.full((batch["target"].shape[0],), -1, dtype=np.int64))
        row_ids.extend(batch["row_id"])
        pair_ids.extend(batch["pair_id"])
        sample_ids.extend(batch["sample_id"])
    return {
        "y_true": np.concatenate(true, axis=0) if true else np.zeros((0, 0), dtype=np.float32),
        "y_pred": np.concatenate(pred, axis=0) if pred else np.zeros((0, 0), dtype=np.float32),
        "raw_true": np.concatenate(raw, axis=0) if raw else np.zeros((0, 0), dtype=np.float32),
        "log_variance": np.concatenate(logvar, axis=0) if logvar else np.zeros((0, 0), dtype=np.float32),
        "proto_true": np.concatenate(proto_true, axis=0) if proto_true else np.zeros((0,), dtype=np.int64),
        "proto_pred": np.concatenate(proto_pred, axis=0) if proto_pred else np.zeros((0,), dtype=np.int64),
        "row_ids": row_ids,
        "pair_ids": pair_ids,
        "sample_ids": sample_ids,
    }


def _write_scatter(path: Path, y_true: np.ndarray, y_pred: np.ndarray, columns: list[str]) -> None:
    if y_true.size == 0:
        return
    count = min(6, y_true.shape[1])
    fig, axes = plt.subplots(2, int(math.ceil(count / 2)), figsize=(3.2 * int(math.ceil(count / 2)), 5.8), dpi=150, squeeze=False)
    for index in range(count):
        axis = axes[index // int(math.ceil(count / 2)), index % int(math.ceil(count / 2))]
        axis.scatter(y_true[:, index], y_pred[:, index], s=18, alpha=0.8)
        lo = float(min(np.min(y_true[:, index]), np.min(y_pred[:, index])))
        hi = float(max(np.max(y_true[:, index]), np.max(y_pred[:, index])))
        axis.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        axis.set_title(columns[index], fontsize=8)
        axis.set_xlabel("true")
        axis.set_ylabel("pred")
    for index in range(count, axes.size):
        axes.flat[index].axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _write_curves(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    fig, axis = plt.subplots(figsize=(5, 3), dpi=150)
    epochs = [row["epoch"] for row in history]
    axis.plot(epochs, [row["train_loss"] for row in history], label="train")
    axis.plot(epochs, [row["val_loss"] for row in history], label="val")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_checkpoint(path: Path, model: RheedTemporalMorphologyEncoder, optimizer: torch.optim.Optimizer, epoch: int, best_val: float, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": int(epoch), "best_val_loss": float(best_val), "config": config}, path)


def load_rheed_morphology_checkpoint(path: Path, device_name: str = "auto") -> tuple[RheedTemporalMorphologyEncoder, dict[str, Any]]:
    device = resolve_torch_device(device_name)
    payload = torch.load(resolve_repo_path(path), map_location=device)
    config = dict(payload["config"])
    model = RheedTemporalMorphologyEncoder(
        descriptor_dim=int(config["descriptor_dim"]),
        handcrafted_dim=int(config.get("handcrafted_dim", 0)),
        metadata_dim=int(config.get("metadata_dim", 0)),
        prototype_classes=int(config.get("prototype_classes", 0)),
        frame_encoder=str(config.get("frame_encoder", "small_cnn")),
        temporal_pooling=str(config.get("temporal_pooling", "attention")),
        visual_embedding_dim=int(config.get("visual_embedding_dim", 256)),
        hidden_dim=int(config.get("hidden_dim", 256)),
        use_visual=bool(config.get("use_visual", True)),
        use_handcrafted=bool(config.get("use_handcrafted", True)),
        use_metadata=bool(config.get("use_metadata", True)),
        predict_uncertainty=bool(config.get("predict_uncertainty", False)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def train_encoder(args: argparse.Namespace, variant_name: str = "trained_model", visual_mode: str = "all") -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = 1 if args.quick else int(args.epochs)
    loaders, schema, scaler = _make_loaders(args, visual_mode=visual_mode)
    device = resolve_torch_device(args.device)
    descriptor_dim = len(schema["condition_columns"])
    feature_dim = len(schema.get("rheed_feature_columns", []))
    metadata_dim = len(schema.get("metadata_columns", [])) if bool(args.use_metadata) else 0
    proto_classes = int(schema.get("prototype_count", 0))
    if proto_classes <= 0:
        proto_classes = 0
    model = RheedTemporalMorphologyEncoder(
        descriptor_dim=descriptor_dim,
        handcrafted_dim=feature_dim,
        metadata_dim=metadata_dim,
        prototype_classes=proto_classes,
        frame_encoder=str(args.frame_encoder),
        temporal_pooling=str(args.temporal_pooling),
        use_visual=bool(args.use_visual),
        use_handcrafted=bool(args.use_handcrafted),
        use_metadata=bool(args.use_metadata) and metadata_dim > 0,
        predict_uncertainty=bool(args.predict_uncertainty) or str(args.loss) == "heteroscedastic",
    ).to(device)
    mae_loaded = False
    if bool(args.use_visual) and args.frame_mae_checkpoint is not None and model.frame_encoder is not None and resolve_repo_path(args.frame_mae_checkpoint).is_file():
        mae_loaded = load_mae_encoder_state(model.frame_encoder, resolve_repo_path(args.frame_mae_checkpoint).as_posix(), strict=False)
    if bool(args.freeze_frame_encoder) and model.frame_encoder is not None:
        for parameter in model.frame_encoder.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=float(args.weight_decay))
    amp_enabled = bool(args.amp and device.type == "cuda")
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    config: dict[str, Any] = {
        "paired_index": display_path(resolve_repo_path(args.paired_index)),
        "condition_schema": display_path(resolve_repo_path(args.condition_schema)),
        "mvp5_root": display_path(resolve_repo_path(args.mvp5_root)),
        "frames": int(args.frames),
        "image_size": int(args.image_size),
        "frame_encoder": str(args.frame_encoder),
        "temporal_pooling": str(args.temporal_pooling),
        "visual_mode": visual_mode,
        "use_visual": bool(args.use_visual),
        "use_handcrafted": bool(args.use_handcrafted),
        "use_metadata": bool(args.use_metadata) and metadata_dim > 0,
        "predict_uncertainty": bool(args.predict_uncertainty) or str(args.loss) == "heteroscedastic",
        "loss": str(args.loss),
        "descriptor_dim": descriptor_dim,
        "handcrafted_dim": feature_dim,
        "metadata_dim": metadata_dim,
        "prototype_classes": proto_classes,
        "condition_columns": list(schema["condition_columns"]),
        "descriptor_columns": list(schema["descriptor_columns"]),
        "scaler": scaler,
        "mae_checkpoint_loaded": mae_loaded,
        "freeze_frame_encoder": bool(args.freeze_frame_encoder),
        "label_fraction": float(args.label_fraction),
        "shuffle_labels": bool(args.shuffle_labels),
        "shuffle_videos": bool(args.shuffle_videos),
        "variant_name": variant_name,
    }
    write_json(out_dir / "config.json", config)
    history: list[dict[str, float]] = []
    best_val = float("inf")
    proto_weight = 0.5 if proto_classes > 0 else 0.0
    for epoch in range(1, epochs + 1):
        train = _run_epoch(model, loaders["train"], device, optimizer, grad_scaler, amp_enabled, config, proto_weight)
        val = _run_epoch(model, loaders["val"], device, None, None, False, config, proto_weight)
        row = {"epoch": float(epoch), "train_loss": train["loss"], "train_descriptor_loss": train["descriptor_loss"], "val_loss": val["loss"], "val_descriptor_loss": val["descriptor_loss"]}
        history.append(row)
        if val["loss"] < best_val:
            best_val = val["loss"]
            save_checkpoint(out_dir / "checkpoints" / "best.pt", model, optimizer, epoch, best_val, config)
        save_checkpoint(out_dir / "checkpoints" / "last.pt", model, optimizer, epoch, best_val, config)
    model, payload = load_rheed_morphology_checkpoint(out_dir / "checkpoints" / "best.pt", str(device))
    config = dict(payload["config"])
    val_arrays = predict_arrays(model, loaders["val"], device, config)
    test_arrays = predict_arrays(model, loaders["test"], device, config) if len(loaders["test"].dataset) else None
    train_targets = []
    for batch in loaders["train"]:
        train_targets.append(batch["target"].numpy())
    train_target_array = np.concatenate(train_targets, axis=0) if train_targets else val_arrays["y_true"]
    mean_condition = np.mean(train_target_array, axis=0, keepdims=True) if train_target_array.size else np.zeros_like(val_arrays["y_true"][:1])
    mean_pred = np.repeat(mean_condition, val_arrays["y_true"].shape[0], axis=0) if val_arrays["y_true"].size else val_arrays["y_true"]
    val_metrics = descriptor_metrics(val_arrays["y_true"], val_arrays["y_pred"])
    baseline = descriptor_metrics(val_arrays["y_true"], mean_pred)
    proto = prototype_metrics(val_arrays["proto_true"], val_arrays["proto_pred"], proto_classes)
    _write_scatter(out_dir / "descriptor_scatter_top_targets.png", val_arrays["y_true"], val_arrays["y_pred"], list(schema["condition_columns"]))
    _write_curves(out_dir / "training_curves.png", history)
    metrics: dict[str, Any] = {
        "variant": variant_name,
        "history": history,
        "split": "val",
        "row_count": int(val_arrays["y_true"].shape[0]),
        **val_metrics,
        **proto,
        "mean_condition_baseline": baseline,
        "beats_mean_condition_mse": bool(np.isfinite(val_metrics["descriptor_mse"]) and val_metrics["descriptor_mse"] < baseline["descriptor_mse"]),
        "best_checkpoint": display_path(out_dir / "checkpoints" / "best.pt"),
        "last_checkpoint": display_path(out_dir / "checkpoints" / "last.pt"),
        "mae_checkpoint_loaded": mae_loaded,
    }
    if test_arrays is not None:
        metrics["test"] = {**descriptor_metrics(test_arrays["y_true"], test_arrays["y_pred"]), **prototype_metrics(test_arrays["proto_true"], test_arrays["proto_pred"], proto_classes), "row_count": int(test_arrays["y_true"].shape[0])}
    if val_arrays["log_variance"].size:
        err = np.mean((val_arrays["y_pred"] - val_arrays["y_true"]) ** 2, axis=1)
        var = np.mean(np.exp(val_arrays["log_variance"]), axis=1)
        metrics["uncertainty_error_pearson"] = float(np.corrcoef(err, var)[0, 1]) if err.size > 1 and np.std(err) > 1e-12 and np.std(var) > 1e-12 else float("nan")
        uncertainty_rows = [{"row_id": rid, "mean_squared_error": float(e), "predicted_variance": float(v)} for rid, e, v in zip(val_arrays["row_ids"], err, var)]
        write_csv_rows(out_dir / "uncertainty_validation.csv", uncertainty_rows)
    write_json(out_dir / "metrics.json", metrics)
    write_csv_rows(out_dir / "ablation_metrics.csv", [{"variant": variant_name, "split": "val", **{key: metrics.get(key, "") for key in ("row_count", "descriptor_mse", "descriptor_mae", "descriptor_rmse", "descriptor_r2", "descriptor_spearman", "prototype_accuracy", "prototype_macro_f1", "beats_mean_condition_mse")}}])
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    metrics = train_encoder(args)
    print(f"Wrote RHEED morphology encoder v2 outputs to {display_path(resolve_repo_path(args.out))}")
    print(f"val_descriptor_mse={metrics['descriptor_mse']:.6f} mean_baseline_mse={metrics['mean_condition_baseline']['descriptor_mse']:.6f}")


if __name__ == "__main__":
    main()
