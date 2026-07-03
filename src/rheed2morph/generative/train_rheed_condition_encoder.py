"""Train a RHEED-to-AFM-condition encoder."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rheed2morph.generative.common import display_path, read_csv_rows, read_json, resolve_repo_path, resolve_torch_device, set_seed, write_csv_rows, write_json
from rheed2morph.generative.models.rheed_condition_encoder import RheedConditionEncoder, build_rheed_condition_encoder


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = value.lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RHEED condition encoder.")
    parser.add_argument("--paired-index", type=Path, required=True)
    parser.add_argument("--condition-schema", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--visual-backbone", type=str, default="small_cnn", choices=["small_cnn", "resnet18", "resnet50"])
    parser.add_argument("--temporal-pooling", type=str, default="attention", choices=["attention", "mean"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-visual", type=str_to_bool, default=True)
    parser.add_argument("--use-handcrafted", type=str_to_bool, default=True)
    parser.add_argument("--use-metadata", type=str_to_bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


class RheedConditionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        paired_rows: list[dict[str, str]],
        feature_rows_by_pair: dict[str, dict[str, str]],
        schema: dict[str, Any],
        split: str,
        use_visual: bool,
        feature_mean: dict[str, float],
        feature_std: dict[str, float],
        metadata_mean: dict[str, float],
        metadata_std: dict[str, float],
        limit: int | None = None,
    ) -> None:
        self.rows = [row for row in paired_rows if row.get("split") == split]
        if limit is not None:
            self.rows = self.rows[:limit]
        self.feature_rows_by_pair = feature_rows_by_pair
        self.schema = schema
        self.use_visual = use_visual
        self.feature_columns = list(schema["rheed_feature_columns"])
        self.target_columns = list(schema["condition_columns"])
        self.metadata_columns = list(schema.get("metadata_columns", []))
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.metadata_mean = metadata_mean
        self.metadata_std = metadata_std

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if self.use_visual:
            video = np.asarray(np.load(resolve_repo_path(Path(row["cached_tensor_path"])))["frames"], dtype=np.float32)
        else:
            video = np.zeros((0,), dtype=np.float32)
        feature_row = self.feature_rows_by_pair[row["pair_id"]]
        handcrafted = np.asarray(
            [
                (float(feature_row[col]) - self.feature_mean.get(col, 0.0)) / max(self.feature_std.get(col, 1.0), 1e-8)
                for col in self.feature_columns
            ],
            dtype=np.float32,
        )
        metadata = np.asarray(
            [
                (float(row.get(col, 0.0) or 0.0) - self.metadata_mean.get(col, 0.0)) / max(self.metadata_std.get(col, 1.0), 1e-8)
                for col in self.metadata_columns
            ],
            dtype=np.float32,
        )
        target = np.asarray([float(row[col]) for col in self.target_columns], dtype=np.float32)
        proto_text = row.get("prototype_id", "")
        proto = int(float(proto_text)) if proto_text != "" else -1
        return {
            "video": torch.from_numpy(video),
            "handcrafted": torch.from_numpy(handcrafted),
            "metadata": torch.from_numpy(metadata),
            "target": torch.from_numpy(target),
            "prototype": torch.tensor(proto, dtype=torch.long),
            "row_id": row["row_id"],
            "sample_id": row.get("sample_id", ""),
        }


def _feature_stats(feature_rows: list[dict[str, str]], feature_columns: list[str], train_pair_ids: set[str]) -> tuple[dict[str, float], dict[str, float]]:
    train = [row for row in feature_rows if row["pair_id"] in train_pair_ids]
    if not train:
        train = feature_rows
    matrix = np.asarray([[float(row[col]) for col in feature_columns] for row in train], dtype=np.float32)
    mean = matrix.mean(axis=0) if matrix.size else np.zeros((len(feature_columns),), dtype=np.float32)
    std = matrix.std(axis=0) if matrix.size else np.ones((len(feature_columns),), dtype=np.float32)
    std = np.where(std > 1e-8, std, 1.0)
    return ({col: float(mean[i]) for i, col in enumerate(feature_columns)}, {col: float(std[i]) for i, col in enumerate(feature_columns)})


def _metadata_stats(rows: list[dict[str, str]], columns: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    train = [row for row in rows if row.get("split") == "train"] or rows
    if not columns:
        return {}, {}
    matrix = np.asarray([[float(row.get(col, 0.0) or 0.0) for col in columns] for row in train], dtype=np.float32)
    mean = matrix.mean(axis=0)
    std = np.where(matrix.std(axis=0) > 1e-8, matrix.std(axis=0), 1.0)
    return ({col: float(mean[i]) for i, col in enumerate(columns)}, {col: float(std[i]) for i, col in enumerate(columns)})


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("handcrafted", "metadata", "target", "prototype"):
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    if batch[0]["video"].numel() > 0:
        out["video"] = torch.stack([item["video"] for item in batch], dim=0)
    else:
        out["video"] = None
    out["row_id"] = [item["row_id"] for item in batch]
    out["sample_id"] = [item["sample_id"] for item in batch]
    return out


def descriptor_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.size == 0:
        return {"descriptor_mse": float("nan"), "descriptor_mae": float("nan"), "descriptor_r2": float("nan"), "descriptor_spearman": float("nan")}
    err = y_pred - y_true
    mse = float(np.mean(err * err))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2))
    r2 = float(1.0 - float(np.sum(err * err)) / denom) if denom > 1e-12 else float("nan")
    spearman_values: list[float] = []
    try:
        from scipy.stats import spearmanr

        for col in range(y_true.shape[1]):
            corr = spearmanr(y_true[:, col], y_pred[:, col]).correlation
            if np.isfinite(corr):
                spearman_values.append(float(corr))
    except Exception:
        pass
    spearman = float(np.mean(spearman_values)) if spearman_values else float("nan")
    return {"descriptor_mse": mse, "descriptor_mae": mae, "descriptor_r2": r2, "descriptor_spearman": spearman}


def prototype_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: int) -> dict[str, float]:
    mask = y_true >= 0
    if classes <= 0 or not np.any(mask):
        return {"prototype_accuracy": float("nan"), "prototype_macro_f1": float("nan")}
    true = y_true[mask]
    pred = y_pred[mask]
    acc = float(np.mean(true == pred))
    f1s: list[float] = []
    for cls in range(classes):
        tp = float(np.sum((true == cls) & (pred == cls)))
        fp = float(np.sum((true != cls) & (pred == cls)))
        fn = float(np.sum((true == cls) & (pred != cls)))
        denom = 2.0 * tp + fp + fn
        f1s.append(0.0 if denom <= 0 else 2.0 * tp / denom)
    return {"prototype_accuracy": acc, "prototype_macro_f1": float(np.mean(f1s))}


def _run_epoch(
    model: RheedConditionEncoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    use_visual: bool,
    use_handcrafted: bool,
    use_metadata: bool,
    prototype_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "descriptor_mse": 0.0, "prototype_ce": 0.0}
    count = 0
    for batch in loader:
        video = batch["video"].to(device) if use_visual and batch["video"] is not None else None
        handcrafted = batch["handcrafted"].to(device) if use_handcrafted else None
        metadata = batch["metadata"].to(device) if use_metadata and batch["metadata"].shape[1] > 0 else None
        target = batch["target"].to(device)
        prototype = batch["prototype"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred = model(video, handcrafted, metadata)
                descriptor_loss = F.mse_loss(pred["descriptor"], target)
                proto_loss = torch.tensor(0.0, device=device)
                if "prototype_logits" in pred and torch.any(prototype >= 0):
                    proto_loss = F.cross_entropy(pred["prototype_logits"][prototype >= 0], prototype[prototype >= 0])
                loss = descriptor_loss + prototype_weight * proto_loss
        if not torch.isfinite(loss):
            raise RuntimeError("RHEED condition encoder loss became non-finite.")
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
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["descriptor_mse"] += float(descriptor_loss.detach().cpu()) * batch_size
        totals["prototype_ce"] += float(proto_loss.detach().cpu()) * batch_size
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def predict_arrays(model: RheedConditionEncoder, loader: DataLoader, device: torch.device, use_visual: bool, use_handcrafted: bool, use_metadata: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    proto_true: list[np.ndarray] = []
    proto_pred: list[np.ndarray] = []
    row_ids: list[str] = []
    for batch in loader:
        video = batch["video"].to(device) if use_visual and batch["video"] is not None else None
        handcrafted = batch["handcrafted"].to(device) if use_handcrafted else None
        metadata = batch["metadata"].to(device) if use_metadata and batch["metadata"].shape[1] > 0 else None
        out = model(video, handcrafted, metadata)
        y_true.append(batch["target"].numpy())
        y_pred.append(out["descriptor"].detach().cpu().numpy())
        proto_true.append(batch["prototype"].numpy())
        if "prototype_logits" in out:
            proto_pred.append(torch.argmax(out["prototype_logits"], dim=1).detach().cpu().numpy())
        else:
            proto_pred.append(np.full((batch["target"].shape[0],), -1, dtype=np.int64))
        row_ids.extend(batch["row_id"])
    return (
        np.concatenate(y_true, axis=0) if y_true else np.zeros((0, 0), dtype=np.float32),
        np.concatenate(y_pred, axis=0) if y_pred else np.zeros((0, 0), dtype=np.float32),
        np.concatenate(proto_true, axis=0) if proto_true else np.zeros((0,), dtype=np.int64),
        np.concatenate(proto_pred, axis=0) if proto_pred else np.zeros((0,), dtype=np.int64),
        row_ids,
    )


def _write_scatter(path: Path, y_true: np.ndarray, y_pred: np.ndarray, columns: list[str]) -> None:
    if y_true.shape[0] == 0:
        return
    count = min(4, y_true.shape[1])
    fig, axes = plt.subplots(1, count, figsize=(3.2 * count, 3.0), dpi=150, squeeze=False)
    for index in range(count):
        axis = axes[0, index]
        axis.scatter(y_true[:, index], y_pred[:, index], s=18, alpha=0.8)
        lo = float(min(np.min(y_true[:, index]), np.min(y_pred[:, index])))
        hi = float(max(np.max(y_true[:, index]), np.max(y_pred[:, index])))
        axis.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        axis.set_title(columns[index], fontsize=8)
        axis.set_xlabel("true")
        axis.set_ylabel("pred")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _write_confusion(path: Path, true: np.ndarray, pred: np.ndarray, classes: int) -> None:
    if classes <= 0 or true.size == 0:
        return
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for t, p in zip(true, pred):
        if 0 <= t < classes and 0 <= p < classes:
            matrix[int(t), int(p)] += 1
    fig, axis = plt.subplots(figsize=(4, 4), dpi=150)
    axis.imshow(matrix, cmap="Blues")
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    axis.set_title("Prototype Confusion")
    for y in range(classes):
        for x in range(classes):
            axis.text(x, y, str(matrix[y, x]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_checkpoint(path: Path, model: RheedConditionEncoder, optimizer: torch.optim.Optimizer, epoch: int, best_val_loss: float, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_val_loss": float(best_val_loss),
            "config": config,
        },
        path,
    )


def load_rheed_condition_checkpoint(path: Path, device_name: str = "auto") -> tuple[RheedConditionEncoder, dict[str, Any]]:
    device = resolve_torch_device(device_name)
    payload = torch.load(resolve_repo_path(path), map_location=device)
    config = dict(payload["config"])
    model = build_rheed_condition_encoder(
        descriptor_dim=int(config["descriptor_dim"]),
        handcrafted_dim=int(config["handcrafted_dim"]),
        metadata_dim=int(config.get("metadata_dim", 0)),
        prototype_classes=int(config.get("prototype_classes", 0)),
        visual_backbone=str(config.get("visual_backbone", "small_cnn")),
        temporal_pooling=str(config.get("temporal_pooling", "attention")),
        use_visual=bool(config.get("use_visual", True)),
        use_handcrafted=bool(config.get("use_handcrafted", True)),
        use_metadata=bool(config.get("use_metadata", False)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def make_dataloaders(
    paired_index: Path,
    condition_schema: Path,
    use_visual: bool,
    use_handcrafted: bool,
    use_metadata: bool,
    batch_size: int,
    limit: int | None,
    num_workers: int,
) -> tuple[dict[str, DataLoader], dict[str, Any], dict[str, Any]]:
    paired_rows = read_csv_rows(paired_index)
    schema = read_json(condition_schema)
    feature_rows = read_csv_rows(condition_schema.parent / "rheed_handcrafted_features.csv")
    feature_by_pair = {row["pair_id"]: row for row in feature_rows}
    train_pair_ids = {row["pair_id"] for row in paired_rows if row.get("split") == "train"}
    feature_mean, feature_std = _feature_stats(feature_rows, list(schema["rheed_feature_columns"]), train_pair_ids)
    metadata_cols = list(schema.get("metadata_columns", [])) if use_metadata else []
    metadata_mean, metadata_std = _metadata_stats(paired_rows, metadata_cols)
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        dataset = RheedConditionDataset(
            paired_rows,
            feature_by_pair,
            schema,
            split,
            use_visual=use_visual,
            feature_mean=feature_mean,
            feature_std=feature_std,
            metadata_mean=metadata_mean,
            metadata_std=metadata_std,
            limit=limit,
        )
        if len(dataset) == 0 and split == "val":
            dataset = RheedConditionDataset(
                paired_rows,
                feature_by_pair,
                schema,
                "train",
                use_visual=use_visual,
                feature_mean=feature_mean,
                feature_std=feature_std,
                metadata_mean=metadata_mean,
                metadata_std=metadata_std,
                limit=min(limit or 8, 8),
            )
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers, collate_fn=_collate)
    scaler = {"feature_mean": feature_mean, "feature_std": feature_std, "metadata_mean": metadata_mean, "metadata_std": metadata_std}
    return loaders, schema, scaler


def train_encoder(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = 1 if args.quick else int(args.epochs)
    limit = args.limit if not args.quick else (args.limit or 16)
    device = resolve_torch_device(args.device)
    loaders, schema, scaler = make_dataloaders(
        resolve_repo_path(args.paired_index),
        resolve_repo_path(args.condition_schema),
        bool(args.use_visual),
        bool(args.use_handcrafted),
        bool(args.use_metadata),
        int(args.batch_size),
        limit,
        int(args.num_workers),
    )
    feature_dim = len(schema["rheed_feature_columns"])
    metadata_dim = len(schema.get("metadata_columns", [])) if args.use_metadata else 0
    descriptor_dim = len(schema["condition_columns"])
    proto_classes = int(schema.get("prototype_count", 0)) if bool(schema.get("prototype_label_exists", False)) else 0
    model = build_rheed_condition_encoder(
        descriptor_dim=descriptor_dim,
        handcrafted_dim=feature_dim,
        metadata_dim=metadata_dim,
        prototype_classes=proto_classes,
        visual_backbone=str(args.visual_backbone),
        temporal_pooling=str(args.temporal_pooling),
        use_visual=bool(args.use_visual),
        use_handcrafted=bool(args.use_handcrafted),
        use_metadata=bool(args.use_metadata) and metadata_dim > 0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    amp_enabled = bool(args.amp and device.type == "cuda")
    grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    config = {
        "paired_index": display_path(resolve_repo_path(args.paired_index)),
        "condition_schema": display_path(resolve_repo_path(args.condition_schema)),
        "frames": int(args.frames),
        "image_size": int(args.image_size),
        "visual_backbone": str(args.visual_backbone),
        "temporal_pooling": str(args.temporal_pooling),
        "epochs": int(epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "amp": bool(args.amp),
        "device": str(device),
        "use_visual": bool(args.use_visual),
        "use_handcrafted": bool(args.use_handcrafted),
        "use_metadata": bool(args.use_metadata) and metadata_dim > 0,
        "descriptor_dim": descriptor_dim,
        "handcrafted_dim": feature_dim,
        "metadata_dim": metadata_dim,
        "prototype_classes": proto_classes,
        "condition_columns": schema["condition_columns"],
        "scaler": scaler,
    }
    write_json(out_dir / "config.json", config)
    best_val = float("inf")
    history: list[dict[str, float]] = []
    prototype_weight = 0.5 if proto_classes > 0 else 0.0
    for epoch in range(1, epochs + 1):
        train = _run_epoch(model, loaders["train"], device, optimizer, grad_scaler, amp_enabled, bool(args.use_visual), bool(args.use_handcrafted), bool(args.use_metadata) and metadata_dim > 0, prototype_weight)
        val = _run_epoch(model, loaders["val"], device, None, None, False, bool(args.use_visual), bool(args.use_handcrafted), bool(args.use_metadata) and metadata_dim > 0, prototype_weight)
        row = {"epoch": float(epoch), "train_loss": train["loss"], "train_descriptor_mse": train["descriptor_mse"], "val_loss": val["loss"], "val_descriptor_mse": val["descriptor_mse"]}
        history.append(row)
        if val["loss"] < best_val:
            best_val = val["loss"]
            save_checkpoint(out_dir / "checkpoints" / "best.pt", model, optimizer, epoch, best_val, config)
        save_checkpoint(out_dir / "checkpoints" / "last.pt", model, optimizer, epoch, best_val, config)
    y_true, y_pred, proto_true, proto_pred, _row_ids = predict_arrays(model, loaders["val"], device, bool(args.use_visual), bool(args.use_handcrafted), bool(args.use_metadata) and metadata_dim > 0)
    desc_metrics = descriptor_metrics(y_true, y_pred)
    proto_metrics = prototype_metrics(proto_true, proto_pred, proto_classes)
    train_targets = []
    for batch in loaders["train"]:
        train_targets.append(batch["target"].numpy())
    train_target_array = np.concatenate(train_targets, axis=0) if train_targets else y_true
    mean_condition = np.mean(train_target_array, axis=0, keepdims=True) if train_target_array.size else np.zeros_like(y_true[:1])
    mean_pred = np.repeat(mean_condition, y_true.shape[0], axis=0) if y_true.size else y_true
    baseline = descriptor_metrics(y_true, mean_pred)
    _write_scatter(out_dir / "descriptor_scatter_top_targets.png", y_true, y_pred, list(schema["condition_columns"]))
    _write_confusion(out_dir / "prototype_confusion.png", proto_true, proto_pred, proto_classes)
    final = history[-1]
    metrics: dict[str, Any] = {
        "history": history,
        "train_loss": final["train_loss"],
        "train_descriptor_mse": final["train_descriptor_mse"],
        "val_loss": final["val_loss"],
        "val_descriptor_mse": final["val_descriptor_mse"],
        **desc_metrics,
        **proto_metrics,
        "mean_condition_baseline": baseline,
        "beats_mean_condition_mse": bool(np.isfinite(desc_metrics["descriptor_mse"]) and desc_metrics["descriptor_mse"] < baseline["descriptor_mse"]),
        "best_checkpoint": display_path(out_dir / "checkpoints" / "best.pt"),
        "last_checkpoint": display_path(out_dir / "checkpoints" / "last.pt"),
    }
    write_json(out_dir / "metrics.json", metrics)
    write_csv_rows(
        out_dir / "ablation_metrics.csv",
        [
            {"variant": "trained_model", **{key: metrics.get(key, "") for key in ("descriptor_mse", "descriptor_mae", "descriptor_r2", "descriptor_spearman", "prototype_accuracy", "prototype_macro_f1")}},
            {"variant": "mean_condition_baseline", **{key: baseline.get(key, "") for key in ("descriptor_mse", "descriptor_mae", "descriptor_r2", "descriptor_spearman")}},
        ],
    )
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    metrics = train_encoder(args)
    print(f"Wrote RHEED condition encoder outputs to {display_path(resolve_repo_path(args.out))}")
    print(f"val_descriptor_mse={metrics['val_descriptor_mse']:.6f} mean_baseline_mse={metrics['mean_condition_baseline']['descriptor_mse']:.6f}")


if __name__ == "__main__":
    main()
