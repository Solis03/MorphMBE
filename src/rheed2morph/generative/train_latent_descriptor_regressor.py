"""Train a latent descriptor/prototype regressor for condition guidance."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rheed2morph.generative.common import display_path, read_csv_rows, read_json, resolve_repo_path, resolve_torch_device, set_seed, write_json
from rheed2morph.generative.condition_control_v3_utils import build_condition_matrix_v3, finite_float
from rheed2morph.generative.afm_prior_v2_utils import write_training_curves


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train latent descriptor regressor.")
    parser.add_argument("--latents-dir", type=Path, required=True)
    parser.add_argument("--condition-table", type=Path, required=True)
    parser.add_argument("--condition-schema", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


class LatentDescriptorRegressor(nn.Module):
    def __init__(self, latent_channels: int, descriptor_dim: int, prototype_count: int = 0, hidden: int = 128) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.descriptor_dim = int(descriptor_dim)
        self.prototype_count = int(prototype_count)
        self.encoder = nn.Sequential(
            nn.Conv2d(latent_channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, descriptor_dim))
        self.prototype_head = nn.Linear(hidden, prototype_count) if prototype_count > 0 else None

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.encoder(latent)
        descriptor = self.head(features)
        proto = self.prototype_head(features) if self.prototype_head is not None else None
        return descriptor, proto


class LatentRegressorDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]):
    def __init__(self, latents_dir: Path, split: str, rows: list[dict[str, str]], schema: dict[str, Any], limit: int | None = None) -> None:
        payload = np.load(latents_dir / f"latents_{split}.npz", allow_pickle=True)
        latents = np.asarray(payload["latents"], dtype=np.float32)
        row_ids = np.asarray(payload["row_ids"])
        if limit is not None:
            latents = latents[:limit]
            row_ids = row_ids[:limit]
        conditions = build_condition_matrix_v3(rows, row_ids, schema)
        descriptor_dim = len(schema["condition_columns"])
        proto_labels = []
        by_row = {row["row_id"]: row for row in rows}
        for row_id in row_ids.tolist():
            value = by_row[str(row_id)].get("prototype_id", "")
            proto_labels.append(int(float(value)) if value != "" else -1)
        self.latents = torch.from_numpy(latents)
        self.descriptors = torch.from_numpy(conditions[:, :descriptor_dim].astype(np.float32))
        self.prototypes = torch.from_numpy(np.asarray(proto_labels, dtype=np.int64))
        self.row_ids = [str(value) for value in row_ids.tolist()]

    def __len__(self) -> int:
        return int(self.latents.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        return self.latents[index], self.descriptors[index], self.prototypes[index], self.row_ids[index]


def save_regressor_checkpoint(path: Path, model: LatentDescriptorRegressor, optimizer: torch.optim.Optimizer | None, epoch: int, best: float, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model_state_dict": model.state_dict(), "epoch": int(epoch), "best_val_loss": float(best), "config": config}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_regressor_checkpoint(path: Path, device_name: str = "auto") -> tuple[LatentDescriptorRegressor, dict[str, Any]]:
    device = resolve_torch_device(device_name)
    payload = torch.load(resolve_repo_path(path), map_location=device)
    config = dict(payload["config"])
    model = LatentDescriptorRegressor(
        latent_channels=int(config["latent_channels"]),
        descriptor_dim=int(config["descriptor_dim"]),
        prototype_count=int(config.get("prototype_count", 0)),
        hidden=int(config.get("hidden", 128)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _epoch(
    model: LatentDescriptorRegressor,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "descriptor_mse": 0.0, "prototype_ce": 0.0, "prototype_acc": 0.0}
    count = 0
    proto_count = 0
    for latents, descriptors, prototypes, _row_ids in loader:
        latents = latents.to(device)
        descriptors = descriptors.to(device)
        prototypes = prototypes.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred, proto_logits = model(latents)
                descriptor_loss = F.mse_loss(pred, descriptors)
                proto_loss = latents.new_tensor(0.0)
                if proto_logits is not None and torch.any(prototypes >= 0):
                    mask = prototypes >= 0
                    proto_loss = F.cross_entropy(proto_logits[mask], prototypes[mask])
                loss = descriptor_loss + proto_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Latent descriptor regressor loss became non-finite.")
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        batch = int(latents.shape[0])
        totals["loss"] += float(loss.detach().cpu()) * batch
        totals["descriptor_mse"] += float(descriptor_loss.detach().cpu()) * batch
        totals["prototype_ce"] += float(proto_loss.detach().cpu()) * batch
        if proto_logits is not None and torch.any(prototypes >= 0):
            mask = prototypes >= 0
            totals["prototype_acc"] += float((proto_logits[mask].argmax(dim=1) == prototypes[mask]).float().mean().detach().cpu()) * int(torch.sum(mask))
            proto_count += int(torch.sum(mask))
        count += batch
    return {
        "loss": totals["loss"] / max(count, 1),
        "descriptor_mse": totals["descriptor_mse"] / max(count, 1),
        "prototype_ce": totals["prototype_ce"] / max(count, 1),
        "prototype_acc": totals["prototype_acc"] / max(proto_count, 1),
    }


@torch.no_grad()
def _predict(model: LatentDescriptorRegressor, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    p_true: list[np.ndarray] = []
    p_pred: list[np.ndarray] = []
    model.eval()
    for latents, descriptors, prototypes, _row_ids in loader:
        pred, proto_logits = model(latents.to(device))
        y_true.append(descriptors.numpy())
        y_pred.append(pred.detach().cpu().numpy())
        p_true.append(prototypes.numpy())
        if proto_logits is not None:
            p_pred.append(proto_logits.argmax(dim=1).detach().cpu().numpy())
        else:
            p_pred.append(np.full((latents.shape[0],), -1, dtype=np.int64))
    return np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(p_true), np.concatenate(p_pred)


def _write_plots(out_dir: Path, model: LatentDescriptorRegressor, loader: DataLoader, device: torch.device, schema: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_true, y_pred, p_true, p_pred = _predict(model, loader, device)
    names = schema["descriptor_columns"][: min(8, len(schema["descriptor_columns"]))]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), dpi=150, squeeze=False)
    for index, name in enumerate(names):
        ax = axes.ravel()[index]
        ax.scatter(y_true[:, index], y_pred[:, index], s=12, alpha=0.75)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("target std cond")
        ax.set_ylabel("pred std cond")
    fig.tight_layout()
    fig.savefig(out_dir / "descriptor_regression_scatter.png")
    plt.close(fig)
    if int(schema.get("prototype_count", 0)) > 0 and np.any(p_true >= 0):
        count = int(schema["prototype_count"])
        matrix = np.zeros((count, count), dtype=np.int64)
        for true, pred in zip(p_true, p_pred):
            if 0 <= true < count and 0 <= pred < count:
                matrix[int(true), int(pred)] += 1
        fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
        ax.imshow(matrix, cmap="Blues")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title("prototype confusion")
        fig.tight_layout()
        fig.savefig(out_dir / "prototype_confusion.png")
        plt.close(fig)


def train_regressor(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    latents_dir = resolve_repo_path(args.latents_dir)
    rows = read_csv_rows(resolve_repo_path(args.condition_table))
    schema = read_json(resolve_repo_path(args.condition_schema))
    epochs = 2 if args.quick else int(args.epochs)
    limit = 32 if args.quick else None
    train_data = LatentRegressorDataset(latents_dir, "train", rows, schema, limit=limit)
    val_data = LatentRegressorDataset(latents_dir, "val", rows, schema, limit=limit)
    if len(val_data) == 0:
        val_data = train_data
    device = resolve_torch_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    latent_shape = tuple(int(value) for value in train_data.latents.shape[1:])
    config = {
        "latents_dir": display_path(latents_dir),
        "condition_table": display_path(resolve_repo_path(args.condition_table)),
        "condition_schema": display_path(resolve_repo_path(args.condition_schema)),
        "latent_shape": list(latent_shape),
        "latent_channels": latent_shape[0],
        "descriptor_dim": len(schema["condition_columns"]),
        "prototype_count": int(schema.get("prototype_count", 0)),
        "hidden": 128,
        "epochs": epochs,
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "amp": bool(args.amp),
        "device": str(device),
    }
    model = LatentDescriptorRegressor(latent_shape[0], len(schema["condition_columns"]), int(schema.get("prototype_count", 0))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_loader = DataLoader(train_data, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    best = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train = _epoch(model, train_loader, device, optimizer, scaler, amp_enabled)
        val = _epoch(model, val_loader, device, None, None, False)
        row = {"epoch": float(epoch), "train_loss": train["loss"], "val_loss": val["loss"], "val_descriptor_mse": val["descriptor_mse"], "val_prototype_acc": val["prototype_acc"]}
        history.append(row)
        if val["loss"] < best:
            best = val["loss"]
            save_regressor_checkpoint(out_dir / "checkpoints" / "best.pt", model, optimizer, epoch, best, config)
        save_regressor_checkpoint(out_dir / "checkpoints" / "last.pt", model, optimizer, epoch, best, config)
    y_train = train_data.descriptors.numpy()
    mean_baseline = np.mean(y_train, axis=0, keepdims=True)
    y_val = val_data.descriptors.numpy()
    baseline_mse = float(np.mean((y_val - mean_baseline) ** 2))
    _write_plots(out_dir, model, val_loader, device, schema)
    write_training_curves(out_dir / "training_curves.png", history, ["train_loss", "val_loss", "val_descriptor_mse"])
    metrics = {
        "history": history,
        "train_loss": history[-1]["train_loss"],
        "val_loss": history[-1]["val_loss"],
        "val_descriptor_mse": history[-1]["val_descriptor_mse"],
        "mean_condition_baseline_mse": baseline_mse,
        "beats_mean_condition_baseline": bool(history[-1]["val_descriptor_mse"] < baseline_mse),
        "val_prototype_acc": history[-1]["val_prototype_acc"],
        "best_checkpoint": display_path(out_dir / "checkpoints" / "best.pt"),
        "last_checkpoint": display_path(out_dir / "checkpoints" / "last.pt"),
        "descriptor_regression_scatter": display_path(out_dir / "descriptor_regression_scatter.png"),
    }
    write_json(out_dir / "config.json", config)
    write_json(out_dir / "metrics.json", metrics)
    report = [
        "# Latent Descriptor Regressor Report",
        "",
        f"Validation descriptor MSE: `{metrics['val_descriptor_mse']:.6f}`",
        f"Mean-condition baseline MSE: `{baseline_mse:.6f}`",
        f"Beats baseline: `{metrics['beats_mean_condition_baseline']}`",
        f"Prototype accuracy: `{metrics['val_prototype_acc']:.6f}`",
    ]
    (out_dir / "latent_descriptor_regressor_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    metrics = train_regressor(args)
    print(f"Wrote latent descriptor regressor to {display_path(resolve_repo_path(args.out))}")
    print(f"val_descriptor_mse={metrics['val_descriptor_mse']:.6f} baseline_mse={metrics['mean_condition_baseline_mse']:.6f}")


if __name__ == "__main__":
    main()
