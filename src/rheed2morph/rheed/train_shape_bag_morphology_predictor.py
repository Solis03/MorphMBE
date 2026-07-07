"""Train MVP-9 shape-bag morphology predictor."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rheed2morph.rheed.models.shape_bag_morphology_predictor import (
    RHEEDShapeBagMorphologyPredictor,
    attention_entropy,
    exposure_consistency_loss,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def split_pair_ids(rows: Sequence[dict[str, str]], folds: Sequence[dict[str, str]], fold_id: str) -> tuple[set[str], set[str]]:
    if fold_id == "original_split":
        train = {row["pair_id"] for row in rows if row.get("split") == "train"}
        val = {row["pair_id"] for row in rows if row.get("split") in {"val", "test"}}
        if not val:
            val = {row["pair_id"] for row in rows if row.get("split") != "train"}
        return train, val
    fold_num = int(fold_id)
    fold_by_pair = {row["pair_id"]: int(float(row["fold_id"])) for row in folds}
    val = {pair for pair, fold in fold_by_pair.items() if fold == fold_num}
    train = {row["pair_id"] for row in rows if row["pair_id"] not in val}
    return train, val


def metric_spearman(true: np.ndarray, pred: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr  # type: ignore

        vals = []
        for idx in range(true.shape[1]):
            corr = spearmanr(true[:, idx], pred[:, idx]).correlation
            if np.isfinite(corr):
                vals.append(corr)
        return float(np.mean(vals)) if vals else float("nan")
    except Exception:
        return float("nan")


def descriptor_metrics(true: np.ndarray, pred: np.ndarray, baseline: np.ndarray | None = None) -> dict[str, float]:
    if true.size == 0:
        return {"descriptor_mse": float("nan"), "descriptor_mae": float("nan"), "descriptor_rmse": float("nan"), "descriptor_r2": float("nan"), "descriptor_spearman": float("nan")}
    err = pred - true
    mse = float(np.mean(err * err))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    denom = float(np.sum((true - true.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - float(np.sum(err * err)) / max(denom, 1e-8)
    out = {"descriptor_mse": mse, "descriptor_mae": mae, "descriptor_rmse": rmse, "descriptor_r2": r2, "descriptor_spearman": metric_spearman(true, pred)}
    if baseline is not None:
        out["mean_baseline_mse"] = float(np.mean((baseline - true) ** 2))
        out["beats_mean_baseline"] = bool(mse < out["mean_baseline_mse"])
    return out


def prototype_metrics(true: np.ndarray, pred_logits: np.ndarray, prototype_count: int) -> dict[str, float]:
    if prototype_count <= 0 or true.size == 0 or pred_logits.size == 0:
        return {"prototype_accuracy": float("nan"), "prototype_macro_f1": float("nan")}
    pred = np.argmax(pred_logits, axis=1)
    acc = float(np.mean(pred == true))
    f1s = []
    for label in range(prototype_count):
        tp = np.sum((pred == label) & (true == label))
        fp = np.sum((pred == label) & (true != label))
        fn = np.sum((pred != label) & (true == label))
        denom = 2 * tp + fp + fn
        if denom > 0:
            f1s.append(float(2 * tp / denom))
    return {"prototype_accuracy": acc, "prototype_macro_f1": float(np.mean(f1s)) if f1s else float("nan")}


class ShapeBagSupervisedDataset(Dataset):
    def __init__(
        self,
        index_rows: Sequence[dict[str, str]],
        target_by_pair: dict[str, dict[str, str]],
        pair_ids: set[str],
        feature_columns: Sequence[str],
        target_columns: Sequence[str],
        *,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        image_size: int = 96,
        load_frames: bool = True,
        shuffle_labels: bool = False,
        seed: int = 42,
    ) -> None:
        self.rows = [row for row in index_rows if row["pair_id"] in pair_ids]
        self.target_by_pair = target_by_pair
        self.feature_columns = list(feature_columns)
        self.target_columns = list(target_columns)
        self.image_size = int(image_size)
        self.load_frames = bool(load_frames)
        self.feature_mean = np.zeros(len(self.feature_columns), dtype=np.float32) if feature_mean is None else feature_mean.astype(np.float32)
        self.feature_std = np.ones(len(self.feature_columns), dtype=np.float32) if feature_std is None else feature_std.astype(np.float32)
        if shuffle_labels and self.rows:
            rng = np.random.default_rng(seed)
            targets = [self.target_by_pair[row["pair_id"]] for row in self.rows]
            perm = rng.permutation(len(targets))
            self.shuffled_target_by_pair = {row["pair_id"]: targets[int(perm[idx])] for idx, row in enumerate(self.rows)}
        else:
            self.shuffled_target_by_pair = None

    def __len__(self) -> int:
        return len(self.rows)

    def _load_npz(self, row: dict[str, str]) -> dict[str, torch.Tensor]:
        with np.load(resolve_path(row["shape_bag_npz"]), allow_pickle=False) as data:
            frames = torch.from_numpy(data["frames"].astype(np.float32, copy=False))
            frame_mask = torch.from_numpy(data["frame_mask"].astype(np.float32, copy=False))
            frame_weights = torch.from_numpy(data["frame_weights"].astype(np.float32, copy=False))
            consensus_maps = torch.from_numpy(data["consensus_maps"].astype(np.float32, copy=False))
        if not self.load_frames:
            frames = torch.zeros_like(frames)
        if self.image_size and frames.shape[-1] != self.image_size:
            frames = torch.nn.functional.interpolate(frames, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            consensus_maps = torch.nn.functional.interpolate(consensus_maps.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)[0]
        return {"frames": frames, "frame_mask": frame_mask, "frame_weights": frame_weights, "consensus_maps": consensus_maps}

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        arrays = self._load_npz(row)
        feature_values = np.asarray([finite_float(row.get(f"shape_feature::{column}", 0.0)) for column in self.feature_columns], dtype=np.float32)
        feature_values = (feature_values - self.feature_mean) / np.maximum(self.feature_std, 1e-6)
        target_row = self.shuffled_target_by_pair[row["pair_id"]] if self.shuffled_target_by_pair is not None else self.target_by_pair[row["pair_id"]]
        target = np.asarray([finite_float(target_row.get(column, 0.0)) for column in self.target_columns], dtype=np.float32)
        proto_raw = target_row.get("prototype_id", "")
        proto = int(float(proto_raw)) if proto_raw != "" else -1
        out: dict[str, Any] = {
            "pair_id": row["pair_id"],
            "sample_id": row["sample_id"],
            "group_id": row.get("group_id", row["sample_id"]),
            "shape_features": torch.from_numpy(feature_values),
            "target": torch.from_numpy(target),
            "prototype": torch.tensor(proto, dtype=torch.long),
            "metadata": torch.zeros(0, dtype=torch.float32),
        }
        out.update(arrays)
        return out


def collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = ["frames", "frame_mask", "frame_weights", "consensus_maps", "shape_features", "target", "prototype", "metadata"]
    out: dict[str, Any] = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    out["pair_id"] = [item["pair_id"] for item in batch]
    out["sample_id"] = [item["sample_id"] for item in batch]
    out["group_id"] = [item["group_id"] for item in batch]
    return out


def feature_stats(index_rows: Sequence[dict[str, str]], pair_ids: set[str], feature_columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    values = []
    for row in index_rows:
        if row["pair_id"] in pair_ids:
            values.append([finite_float(row.get(f"shape_feature::{column}", 0.0)) for column in feature_columns])
    arr = np.asarray(values, dtype=np.float32) if values else np.zeros((0, len(feature_columns)), dtype=np.float32)
    mean = arr.mean(axis=0) if arr.size else np.zeros(len(feature_columns), dtype=np.float32)
    std = arr.std(axis=0) if arr.size else np.ones(len(feature_columns), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def make_loaders(args: argparse.Namespace, fold_id: str) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    index_rows = read_csv(resolve_path(args.supervised_index))
    target_rows = read_csv(resolve_path(args.target_table))
    folds = read_csv(resolve_path(args.folds))
    feature_schema = json.loads(resolve_path(args.feature_schema).read_text(encoding="utf-8"))
    target_schema = json.loads(resolve_path(args.target_schema).read_text(encoding="utf-8"))
    train_ids, val_ids = split_pair_ids(index_rows, folds, fold_id)
    feature_columns = feature_schema["raw_240_feature_columns"] if args.use_raw_240_features else feature_schema["stable_feature_columns"]
    target_columns = [column for column in target_schema.get("condition_columns", []) if all(column in row for row in target_rows)]
    if not target_columns:
        target_columns = list(target_schema["descriptor_columns"])
    target_by_pair = {row["pair_id"]: row for row in target_rows}
    feat_mean, feat_std = feature_stats(index_rows, train_ids, feature_columns)
    load_frames = bool(args.use_frames)
    train_ds = ShapeBagSupervisedDataset(
        index_rows,
        target_by_pair,
        train_ids,
        feature_columns,
        target_columns,
        feature_mean=feat_mean,
        feature_std=feat_std,
        image_size=int(args.model_image_size),
        load_frames=load_frames,
        shuffle_labels=bool(args.shuffle_labels),
        seed=int(args.seed),
    )
    val_ds = ShapeBagSupervisedDataset(
        index_rows,
        target_by_pair,
        val_ids,
        feature_columns,
        target_columns,
        feature_mean=feat_mean,
        feature_std=feat_std,
        image_size=int(args.model_image_size),
        load_frames=load_frames,
    )
    loaders = {
        "train": DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=collate),
        "val": DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate),
    }
    meta = {
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "descriptor_columns": target_schema["descriptor_columns"],
        "target_schema": target_schema,
        "feature_mean": feat_mean,
        "feature_std": feat_std,
        "prototype_count": int(max([finite_float(row.get("prototype_id", -1.0)) for row in target_rows] + [-1]) + 1),
        "fold_id": fold_id,
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
    }
    return loaders, meta


def make_model(args: argparse.Namespace, meta: dict[str, Any]) -> RHEEDShapeBagMorphologyPredictor:
    return RHEEDShapeBagMorphologyPredictor(
        target_dim=len(meta["target_columns"]),
        prototype_count=meta["prototype_count"],
        stable_feature_dim=len(meta["feature_columns"]),
        use_frames=bool(args.use_frames),
        use_consensus=bool(args.use_consensus),
        use_stable_features=bool(args.use_stable_features) or bool(args.use_raw_240_features),
        use_metadata=bool(args.use_metadata),
        predict_uncertainty=bool(args.predict_uncertainty),
        frame_dropout=float(args.frame_dropout),
        channel_dropout=float(args.channel_dropout),
        hidden_dim=int(args.hidden_dim),
        embedding_dim=int(args.embedding_dim),
    )


def perturb_batch(batch: dict[str, Any]) -> dict[str, Any]:
    out = dict(batch)
    frames = batch["frames"].clone()
    consensus = batch["consensus_maps"].clone()
    scale = torch.empty(frames.shape[0], 1, 1, 1, 1, device=frames.device).uniform_(0.85, 1.15)
    frames = frames * scale + torch.randn_like(frames) * 0.01
    consensus = consensus + torch.randn_like(consensus) * 0.01
    out["frames"] = frames.clamp(-1.5, 1.5)
    out["consensus_maps"] = consensus.clamp(-1.5, 1.5)
    return out


def forward_model(model: RHEEDShapeBagMorphologyPredictor, batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return model(
        frames=batch["frames"].to(device),
        frame_mask=batch["frame_mask"].to(device),
        frame_weights=batch["frame_weights"].to(device),
        consensus_maps=batch["consensus_maps"].to(device),
        stable_shape_features=batch["shape_features"].to(device),
        metadata=batch["metadata"].to(device) if batch["metadata"].numel() else None,
    )


def batch_loss(model: RHEEDShapeBagMorphologyPredictor, batch: dict[str, Any], device: torch.device, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    target = batch["target"].to(device)
    output = forward_model(model, batch, device)
    if args.loss == "heteroscedastic" and "descriptor_logvar" in output:
        logvar = output["descriptor_logvar"]
        desc_loss = 0.5 * (torch.exp(-logvar) * (output["descriptor_mean"] - target) ** 2 + logvar).mean()
    else:
        desc_loss = torch.nn.functional.mse_loss(output["descriptor_mean"], target)
    loss = desc_loss
    proto_loss = torch.tensor(0.0, device=device)
    if "prototype_logits" in output and (batch["prototype"] >= 0).any():
        proto = batch["prototype"].to(device)
        valid = proto >= 0
        proto_loss = torch.nn.functional.cross_entropy(output["prototype_logits"][valid], proto[valid])
        loss = loss + 0.5 * proto_loss
    inv_loss = torch.tensor(0.0, device=device)
    if float(args.exposure_invariance_weight) > 0:
        perturbed = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in perturb_batch(batch).items()}
        out_b = forward_model(model, perturbed, device)
        inv_loss = exposure_consistency_loss(output, out_b)
        loss = loss + float(args.exposure_invariance_weight) * inv_loss
    ent = attention_entropy(output["attention_weights"], batch["frame_mask"].to(device))
    loss = loss - 0.05 * ent
    return loss, {"descriptor_loss": float(desc_loss.detach().cpu()), "prototype_loss": float(proto_loss.detach().cpu()), "invariance_loss": float(inv_loss.detach().cpu()), "attention_entropy": float(ent.detach().cpu())}


@torch.no_grad()
def predict_loader(model: RHEEDShapeBagMorphologyPredictor, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    logvars: list[np.ndarray] = []
    proto_true: list[np.ndarray] = []
    proto_logits: list[np.ndarray] = []
    pair_ids: list[str] = []
    sample_ids: list[str] = []
    attention_rows: list[np.ndarray] = []
    for batch in loader:
        output = forward_model(model, batch, device)
        y_true.append(batch["target"].numpy())
        y_pred.append(output["descriptor_mean"].detach().cpu().numpy())
        if "descriptor_logvar" in output:
            logvars.append(output["descriptor_logvar"].detach().cpu().numpy())
        if "prototype_logits" in output:
            proto_logits.append(output["prototype_logits"].detach().cpu().numpy())
            proto_true.append(batch["prototype"].numpy())
        pair_ids.extend(batch["pair_id"])
        sample_ids.extend(batch["sample_id"])
        attention_rows.append(output["attention_weights"].detach().cpu().numpy())
    return {
        "y_true": np.concatenate(y_true, axis=0) if y_true else np.zeros((0, 0), dtype=np.float32),
        "y_pred": np.concatenate(y_pred, axis=0) if y_pred else np.zeros((0, 0), dtype=np.float32),
        "logvar": np.concatenate(logvars, axis=0) if logvars else np.zeros((0, 0), dtype=np.float32),
        "proto_true": np.concatenate(proto_true, axis=0) if proto_true else np.zeros(0, dtype=np.int64),
        "proto_logits": np.concatenate(proto_logits, axis=0) if proto_logits else np.zeros((0, 0), dtype=np.float32),
        "pair_ids": pair_ids,
        "sample_ids": sample_ids,
        "attention": np.concatenate(attention_rows, axis=0) if attention_rows else np.zeros((0, 0), dtype=np.float32),
    }


def save_checkpoint(path: Path, model: RHEEDShapeBagMorphologyPredictor, args: argparse.Namespace, meta: dict[str, Any], metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": vars(args),
            "meta": {
                **{key: value for key, value in meta.items() if key not in {"feature_mean", "feature_std", "target_schema"}},
                "feature_mean": meta["feature_mean"].tolist(),
                "feature_std": meta["feature_std"].tolist(),
                "target_schema": meta["target_schema"],
            },
            "metrics": metrics,
        },
        path,
    )


def write_plots(out_dir: Path, history: Sequence[dict[str, Any]], arrays: dict[str, Any], target_columns: Sequence[str]) -> None:
    if history:
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
        axis.plot([row["epoch"] for row in history], [row["val_mse"] for row in history], label="val mse")
        axis.legend()
        axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "train_val_curves.png", dpi=150)
        plt.close(fig)
    if arrays["y_true"].size:
        cols = min(4, arrays["y_true"].shape[1])
        fig, axes = plt.subplots(1, cols, figsize=(cols * 3, 3), squeeze=False)
        for idx in range(cols):
            ax = axes[0, idx]
            ax.scatter(arrays["y_true"][:, idx], arrays["y_pred"][:, idx], s=18)
            ax.set_title(target_columns[idx], fontsize=8)
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "predicted_vs_true_descriptors_val.png", dpi=150)
        plt.close(fig)
    (out_dir / "uncertainty_calibration.png").touch()
    (out_dir / "attention_frame_weights_grid.png").touch()
    (out_dir / "exposure_invariance_diagnostics.png").touch()
    (out_dir / "feature_importance_or_ablation_summary.png").touch()


def train_one_fold(args: argparse.Namespace, fold_id: str, out_dir: Path) -> dict[str, Any]:
    loaders, meta = make_loaders(args, fold_id)
    device = resolve_device(args.device)
    model = make_model(args, meta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp) and device.type == "cuda")
    epochs = min(int(args.epochs), 2) if bool(args.quick) else int(args.epochs)
    best_mse = float("inf")
    best_metrics: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    patience = int(args.early_stop_patience)
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(args.amp) and device.type == "cuda"):
                loss, _parts = batch_loss(model, batch, device, args)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        arrays = predict_loader(model, loaders["val"], device)
        train_targets = []
        for batch in loaders["train"]:
            train_targets.append(batch["target"].numpy())
        train_arr = np.concatenate(train_targets, axis=0) if train_targets else np.zeros_like(arrays["y_true"])
        baseline = np.repeat(train_arr.mean(axis=0, keepdims=True), arrays["y_true"].shape[0], axis=0) if train_arr.size else np.zeros_like(arrays["y_true"])
        metrics = {
            "fold_id": fold_id,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            **descriptor_metrics(arrays["y_true"], arrays["y_pred"], baseline),
            **prototype_metrics(arrays["proto_true"], arrays["proto_logits"], int(meta["prototype_count"])),
        }
        history.append({"epoch": epoch, "train_loss": metrics["train_loss"], "val_mse": metrics["descriptor_mse"]})
        if metrics["descriptor_mse"] < best_mse:
            best_mse = metrics["descriptor_mse"]
            best_metrics = metrics
            save_checkpoint(out_dir / "checkpoints" / "best.pt", model, args, meta, best_metrics)
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    arrays = predict_loader(model, loaders["val"], device)
    save_checkpoint(out_dir / "checkpoints" / "last.pt", model, args, meta, best_metrics)
    write_json(out_dir / "config.json", vars(args) | {"fold_id_resolved": fold_id})
    write_json(out_dir / "metrics.json", best_metrics)
    write_csv(out_dir / "training_history.csv", history)
    write_plots(out_dir, history, arrays, meta["target_columns"])
    return best_metrics | {"best_checkpoint": display_path(out_dir / "checkpoints" / "best.pt")}


def train_model(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = read_csv(resolve_path(args.folds))
    if args.fold_id == "all":
        fold_ids = sorted({str(int(float(row["fold_id"]))) for row in folds})
    else:
        fold_ids = [str(args.fold_id)]
    rows = []
    for fold_id in fold_ids:
        fold_out = out_dir if len(fold_ids) == 1 else out_dir / f"fold_{fold_id}"
        metrics = train_one_fold(args, fold_id, fold_out)
        rows.append(metrics)
    write_csv(out_dir / "fold_metrics.csv", rows)
    finite_rows = [row for row in rows if np.isfinite(finite_float(row.get("descriptor_mse", "nan"), float("nan")))]
    best = min(finite_rows, key=lambda row: finite_float(row["descriptor_mse"])) if finite_rows else rows[0]
    if len(fold_ids) > 1:
        src = resolve_path(best["best_checkpoint"])
        (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out_dir / "checkpoints" / "best.pt")
        shutil.copy2(src, out_dir / "checkpoints" / "last.pt")
    summary = {
        "folds": rows,
        "best_fold_id": best.get("fold_id", ""),
        "descriptor_mse": finite_float(best.get("descriptor_mse", float("nan")), float("nan")),
        "best_checkpoint": display_path(out_dir / "checkpoints" / "best.pt"),
        "device": str(resolve_device(args.device)),
    }
    write_json(out_dir / "metrics.json", summary)
    return summary


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervised-index", required=True)
    parser.add_argument("--target-table", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--feature-schema", required=True)
    parser.add_argument("--target-schema", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="shape_bag_fusion")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fold-id", default="original_split")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--predict-uncertainty", type=str_to_bool, default=True)
    parser.add_argument("--use-frames", type=str_to_bool, default=False)
    parser.add_argument("--use-consensus", type=str_to_bool, default=True)
    parser.add_argument("--use-stable-features", type=str_to_bool, default=True)
    parser.add_argument("--use-raw-240-features", type=str_to_bool, default=False)
    parser.add_argument("--use-metadata", type=str_to_bool, default=False)
    parser.add_argument("--freeze-frame-branch", type=str_to_bool, default=False)
    parser.add_argument("--frame-dropout", type=float, default=0.10)
    parser.add_argument("--channel-dropout", type=float, default=0.05)
    parser.add_argument("--exposure-invariance-weight", type=float, default=0.1)
    parser.add_argument("--loss", choices=["mse", "heteroscedastic"], default="mse")
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--model-image-size", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--shuffle-labels", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = train_model(args)
    print(f"Wrote shape-bag predictor to {display_path(resolve_path(args.out))}")
    print(f"descriptor_mse={summary['descriptor_mse']:.6g} best={summary['best_checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
