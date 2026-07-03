"""Train descriptor-conditioned AFM latent diffusion."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rheed2morph.generative.common import (
    display_path,
    load_height_array,
    read_csv_rows,
    read_json,
    resolve_repo_path,
    resolve_torch_device,
    set_seed,
    write_json,
)
from rheed2morph.generative.diffusion import GaussianDiffusion
from rheed2morph.generative.models.latent_unet import LatentUNet
from rheed2morph.generative.train_afm_autoencoder import load_autoencoder_checkpoint
from rheed2morph.generative.visualization import write_panel_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AFM descriptor-conditioned latent diffusion.")
    parser.add_argument("--latents-dir", type=Path, required=True)
    parser.add_argument("--autoencoder-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cond-dropout", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


class LatentConditionDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(self, latents: np.ndarray, conditions: np.ndarray, row_ids: np.ndarray) -> None:
        self.latents = torch.from_numpy(latents.astype(np.float32))
        self.conditions = torch.from_numpy(conditions.astype(np.float32))
        self.row_ids = [str(value) for value in row_ids.tolist()]

    def __len__(self) -> int:
        return int(self.latents.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        return self.latents[index], self.conditions[index], self.row_ids[index]


def condition_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    return [key for key in rows[0] if key.startswith("cond_")]


def build_condition_matrix(
    condition_rows: list[dict[str, str]],
    selected_row_ids: np.ndarray,
    columns: list[str] | None = None,
    prototype_count: int | None = None,
) -> tuple[np.ndarray, list[str], int]:
    by_row = {row["row_id"]: row for row in condition_rows}
    cols = columns or condition_columns(condition_rows)
    if not cols:
        raise RuntimeError("condition_table.csv has no standardized descriptor columns with prefix cond_.")
    prototype_values = []
    for row in condition_rows:
        value = row.get("prototype_id", "")
        if value != "":
            prototype_values.append(int(float(value)))
    proto_count = int(prototype_count if prototype_count is not None else ((max(prototype_values) + 1) if prototype_values else 0))
    matrix: list[list[float]] = []
    for row_id in selected_row_ids.tolist():
        row = by_row[str(row_id)]
        values = [float(row[col]) for col in cols]
        if proto_count > 0:
            one_hot = [0.0] * proto_count
            value = row.get("prototype_id", "")
            if value != "":
                index = int(float(value))
                if 0 <= index < proto_count:
                    one_hot[index] = 1.0
            values.extend(one_hot)
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float32), cols, proto_count


def load_split_dataset(
    latents_dir: Path,
    split: str,
    condition_rows: list[dict[str, str]],
    columns: list[str] | None = None,
    prototype_count: int | None = None,
    limit: int | None = None,
) -> tuple[LatentConditionDataset, list[str], int]:
    payload = np.load(latents_dir / f"latents_{split}.npz", allow_pickle=True)
    latents = np.asarray(payload["latents"], dtype=np.float32)
    row_ids = np.asarray(payload["row_ids"])
    if limit is not None:
        latents = latents[:limit]
        row_ids = row_ids[:limit]
    conditions, cols, proto_count = build_condition_matrix(condition_rows, row_ids, columns, prototype_count)
    return LatentConditionDataset(latents, conditions, row_ids), cols, proto_count


def save_diffusion_checkpoint(
    path: Path,
    model: LatentUNet,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_val_loss: float,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_diffusion_checkpoint(path: Path, device_name: str = "auto") -> tuple[LatentUNet, dict[str, Any]]:
    device = resolve_torch_device(device_name)
    payload = torch.load(resolve_repo_path(path), map_location=device)
    config = dict(payload["config"])
    model = LatentUNet(
        latent_channels=int(config["latent_channels"]),
        condition_dim=int(config["condition_dim"]),
        base_channels=int(config.get("base_channels", 64)),
        emb_dim=int(config.get("emb_dim", 256)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _epoch(
    model: LatentUNet,
    loader: DataLoader,
    diffusion: GaussianDiffusion,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    amp_enabled: bool,
    cond_dropout: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for latents, conditions, _row_ids in loader:
        latents = latents.to(device, non_blocking=True)
        conditions = conditions.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                loss = diffusion.training_loss(model, latents, conditions, cond_dropout if training else 0.0)
        if not torch.isfinite(loss):
            raise RuntimeError("Diffusion loss became non-finite.")
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        batch = int(latents.shape[0])
        total += float(loss.detach().cpu()) * batch
        count += batch
    return total / max(count, 1)


def _write_training_sample_grid(
    model: LatentUNet,
    diffusion: GaussianDiffusion,
    latents_dir: Path,
    condition_rows: list[dict[str, str]],
    condition_columns_used: list[str],
    prototype_count: int,
    autoencoder_checkpoint: Path,
    out_path: Path,
    device: torch.device,
) -> dict[str, float]:
    payload = np.load(latents_dir / "latents_val.npz", allow_pickle=True)
    if int(payload["latents"].shape[0]) == 0:
        payload = np.load(latents_dir / "latents_train.npz", allow_pickle=True)
    count = min(4, int(payload["latents"].shape[0]))
    if count == 0:
        return {"sample_pixel_std": 0.0}
    row_ids = np.asarray(payload["row_ids"][:count])
    standardized_latents = np.asarray(payload["latents"][:count], dtype=np.float32)
    conditions, _cols, _proto = build_condition_matrix(condition_rows, row_ids, condition_columns_used, prototype_count)
    latent_stats = np.load(latents_dir / "latent_standardization.npz")
    latent_mean = torch.from_numpy(np.asarray(latent_stats["latent_mean"], dtype=np.float32)).to(device)
    latent_std = torch.from_numpy(np.asarray(latent_stats["latent_std"], dtype=np.float32)).to(device)
    ae, _ae_payload = load_autoencoder_checkpoint(autoencoder_checkpoint, str(device))
    ae.to(device).eval()
    cond = torch.from_numpy(conditions).to(device)
    sampled = diffusion.sample_ddim(
        model,
        tuple(standardized_latents.shape),
        cond,
        steps=min(25, diffusion.timesteps),
        guidance_scale=1.5,
    )
    sampled_raw = sampled * latent_std + latent_mean
    true_raw = torch.from_numpy(np.asarray(payload["latents_raw"][:count], dtype=np.float32)).to(device)
    by_row = {row["row_id"]: row for row in condition_rows}
    true_images: list[np.ndarray] = []
    for row_id in row_ids.tolist():
        path_text = by_row[str(row_id)].get("network_input_path", "")
        true_images.append(load_height_array(resolve_repo_path(Path(path_text))) if path_text else np.zeros((128, 128), dtype=np.float32))
    with torch.no_grad():
        recon = ae.decode(true_raw).detach().cpu().numpy()
        generated = ae.decode(sampled_raw).detach().cpu().numpy()
    rows = [[true_images[i], recon[i, 0], generated[i, 0]] for i in range(count)]
    write_panel_grid(out_path, rows, ["true AFM", "AE reconstruction", "diffusion sample"], [str(x) for x in row_ids])
    return {"sample_pixel_std": float(np.std(generated)), "sample_pixel_mean": float(np.mean(generated))}


def train_diffusion(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    latents_dir = resolve_repo_path(args.latents_dir)
    condition_rows = read_csv_rows(latents_dir / "condition_table.csv")
    epochs = 1 if args.quick else int(args.epochs)
    limit = args.limit
    if args.quick and limit is None:
        limit = 16
    train_dataset, cols, proto_count = load_split_dataset(latents_dir, "train", condition_rows, limit=limit)
    val_dataset, _cols, _proto = load_split_dataset(latents_dir, "val", condition_rows, cols, proto_count, limit=limit)
    if len(val_dataset) == 0:
        val_dataset = train_dataset
    if len(train_dataset) == 0:
        raise RuntimeError(f"No train latents found in {latents_dir}")
    device = resolve_torch_device(args.device)
    latent_shape = tuple(int(value) for value in train_dataset.latents.shape[1:])
    condition_dim = int(train_dataset.conditions.shape[1])
    model = LatentUNet(latent_channels=latent_shape[0], condition_dim=condition_dim).to(device)
    diffusion = GaussianDiffusion(timesteps=int(args.timesteps), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    config = {
        "latents_dir": display_path(latents_dir),
        "autoencoder_checkpoint": display_path(resolve_repo_path(args.autoencoder_checkpoint)),
        "latent_shape": list(latent_shape),
        "latent_channels": int(latent_shape[0]),
        "condition_dim": condition_dim,
        "condition_columns": cols,
        "prototype_count": int(proto_count),
        "timesteps": int(args.timesteps),
        "epochs": int(epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "cond_dropout": float(args.cond_dropout),
        "seed": int(args.seed),
        "amp": bool(args.amp),
        "device": str(device),
        "base_channels": 64,
        "emb_dim": 256,
    }
    write_json(out_dir / "config.json", config)
    best_val = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_loss = _epoch(model, train_loader, diffusion, device, optimizer, scaler, amp_enabled, float(args.cond_dropout))
        val_loss = _epoch(model, val_loader, diffusion, device, None, None, False, 0.0)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            save_diffusion_checkpoint(out_dir / "checkpoints" / "best.pt", model, optimizer, epoch, best_val, config)
        save_diffusion_checkpoint(out_dir / "checkpoints" / "last.pt", model, optimizer, epoch, best_val, config)
    sample_stats = _write_training_sample_grid(
        model,
        diffusion,
        latents_dir,
        condition_rows,
        cols,
        proto_count,
        resolve_repo_path(args.autoencoder_checkpoint),
        out_dir / "sample_grid_val.png",
        device,
    )
    metrics = {
        "history": history,
        "train_loss": history[-1]["train_loss"],
        "val_loss": history[-1]["val_loss"],
        "best_val_loss": best_val,
        "sample_grid_val": display_path(out_dir / "sample_grid_val.png"),
        "sample_pixel_mean": sample_stats.get("sample_pixel_mean", 0.0),
        "sample_pixel_std": sample_stats.get("sample_pixel_std", 0.0),
        "generated_nonconstant": bool(sample_stats.get("sample_pixel_std", 0.0) > 1e-4),
    }
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    metrics = train_diffusion(args)
    print(f"Wrote latent diffusion outputs to {display_path(resolve_repo_path(args.out))}")
    print(f"val_loss={metrics['val_loss']:.6f} sample_std={metrics['sample_pixel_std']:.6f}")


if __name__ == "__main__":
    main()
