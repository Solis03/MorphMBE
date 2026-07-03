"""Train descriptor-conditioned AFM latent diffusion v2."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rheed2morph.generative.afm_prior_v2_utils import build_condition_matrix_v2, write_training_curves
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
from rheed2morph.generative.diffusion_v2 import GaussianDiffusionV2
from rheed2morph.generative.models.latent_unet import LatentUNet
from rheed2morph.generative.train_afm_autoencoder_v2 import load_autoencoder_v2_checkpoint
from rheed2morph.generative.visualization import write_panel_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AFM latent diffusion v2.")
    parser.add_argument("--latents-dir", type=Path, required=True)
    parser.add_argument("--autoencoder-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--prediction-target", choices=["epsilon", "v"], default="epsilon")
    parser.add_argument("--cond-dropout", type=float, default=0.15)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--sample-every", type=int, default=25)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


class LatentConditionDatasetV2(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(self, latents: np.ndarray, conditions: np.ndarray, row_ids: np.ndarray) -> None:
        self.latents = torch.from_numpy(latents.astype(np.float32))
        self.conditions = torch.from_numpy(conditions.astype(np.float32))
        self.row_ids = [str(value) for value in row_ids.tolist()]

    def __len__(self) -> int:
        return int(self.latents.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        return self.latents[index], self.conditions[index], self.row_ids[index]


class EMAModel:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for shadow_param, model_param in zip(self.shadow.parameters(), model.parameters()):
            shadow_param.data.mul_(self.decay).add_(model_param.data, alpha=1.0 - self.decay)
        for shadow_buffer, model_buffer in zip(self.shadow.buffers(), model.buffers()):
            shadow_buffer.data.copy_(model_buffer.data)


def load_split_dataset_v2(
    latents_dir: Path,
    split: str,
    condition_rows: list[dict[str, str]],
    schema: dict[str, Any],
    limit: int | None = None,
) -> LatentConditionDatasetV2:
    payload = np.load(latents_dir / f"latents_{split}.npz", allow_pickle=True)
    latents = np.asarray(payload["latents"], dtype=np.float32)
    row_ids = np.asarray(payload["row_ids"])
    if limit is not None:
        latents = latents[:limit]
        row_ids = row_ids[:limit]
    conditions = build_condition_matrix_v2(condition_rows, row_ids, schema)
    return LatentConditionDatasetV2(latents, conditions, row_ids)


def save_diffusion_v2_checkpoint(
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


def load_diffusion_v2_checkpoint(path: Path, device_name: str = "auto") -> tuple[LatentUNet, dict[str, Any]]:
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
    diffusion: GaussianDiffusionV2,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    cond_dropout: float,
    ema: EMAModel | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    for latents, conditions, _row_ids in loader:
        latents = latents.to(device, non_blocking=True)
        conditions = conditions.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                loss = diffusion.training_loss(model, latents, conditions, cond_dropout if training else 0.0)
        if not torch.isfinite(loss):
            raise RuntimeError("Latent diffusion v2 loss became non-finite.")
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if ema is not None:
                ema.update(model)
        batch = int(latents.shape[0])
        total += float(loss.detach().cpu()) * batch
        count += batch
    return total / max(count, 1)


def _sample_grid(
    model: LatentUNet,
    diffusion: GaussianDiffusionV2,
    latents_dir: Path,
    condition_rows: list[dict[str, str]],
    schema: dict[str, Any],
    autoencoder_checkpoint: Path,
    out_path: Path,
    device: torch.device,
    steps: int,
) -> dict[str, float]:
    payload_path = latents_dir / "latents_val.npz"
    payload = np.load(payload_path, allow_pickle=True)
    if int(payload["latents"].shape[0]) == 0:
        payload = np.load(latents_dir / "latents_train.npz", allow_pickle=True)
    count = min(4, int(payload["latents"].shape[0]))
    if count == 0:
        return {"sample_pixel_std": 0.0, "sample_pixel_mean": 0.0}
    row_ids = np.asarray(payload["row_ids"][:count])
    conditions = build_condition_matrix_v2(condition_rows, row_ids, schema)
    standardization = np.load(latents_dir / "latent_standardization_v2.npz")
    latent_mean = torch.from_numpy(np.asarray(standardization["latent_mean"], dtype=np.float32)).to(device)
    latent_std = torch.from_numpy(np.asarray(standardization["latent_std"], dtype=np.float32)).to(device)
    latent_shape = tuple(int(value) for value in payload["latents"].shape[1:])
    autoencoder, _payload = load_autoencoder_v2_checkpoint(autoencoder_checkpoint, str(device))
    autoencoder.to(device).eval()
    cond = torch.from_numpy(conditions).to(device)
    sampled = diffusion.sample_ddim(model, (count, *latent_shape), cond, steps=steps, guidance_scale=1.5)
    sampled_raw = sampled * latent_std + latent_mean
    true_raw = torch.from_numpy(np.asarray(payload["latents_raw"][:count], dtype=np.float32)).to(device)
    by_row = {row["row_id"]: row for row in condition_rows}
    true_images: list[np.ndarray] = []
    for row_id in row_ids.tolist():
        path_text = by_row[str(row_id)].get("network_input_path", "")
        true_images.append(load_height_array(resolve_repo_path(Path(path_text))) if path_text else np.zeros((128, 128), dtype=np.float32))
    with torch.no_grad():
        recon = autoencoder.decode(true_raw).detach().cpu().numpy()
        generated = autoencoder.decode(sampled_raw).detach().cpu().numpy()
    rows = [[true_images[i], recon[i, 0], generated[i, 0]] for i in range(count)]
    write_panel_grid(out_path, rows, ["true AFM", "AE reconstruction", "diffusion sample"], [str(row_id) for row_id in row_ids])
    return {"sample_pixel_mean": float(np.mean(generated)), "sample_pixel_std": float(np.std(generated))}


def train_diffusion(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    latents_dir = resolve_repo_path(args.latents_dir)
    condition_rows = read_csv_rows(latents_dir / "condition_table_v2.csv")
    schema = read_json(latents_dir / "condition_schema_v2.json")
    epochs = 1 if args.quick else int(args.epochs)
    limit = int(args.limit) if args.limit is not None else (24 if args.quick else None)
    train_dataset = load_split_dataset_v2(latents_dir, "train", condition_rows, schema, limit=limit)
    val_dataset = load_split_dataset_v2(latents_dir, "val", condition_rows, schema, limit=limit)
    if len(val_dataset) == 0:
        val_dataset = train_dataset
    if len(train_dataset) == 0:
        raise RuntimeError(f"No train latents found in {latents_dir}")
    device = resolve_torch_device(args.device)
    latent_shape = tuple(int(value) for value in train_dataset.latents.shape[1:])
    condition_dim = int(train_dataset.conditions.shape[1])
    model = LatentUNet(latent_channels=latent_shape[0], condition_dim=condition_dim, base_channels=64, emb_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    start_epoch = 1
    best_val = float("inf")
    if args.resume is not None:
        payload = torch.load(resolve_repo_path(args.resume), map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_val = float(payload.get("best_val_loss", best_val))
    diffusion = GaussianDiffusionV2(
        timesteps=int(args.timesteps),
        beta_schedule=str(args.beta_schedule),
        prediction_target=str(args.prediction_target),
        device=device,
    )
    ema = EMAModel(model) if args.ema else None
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
        "condition_schema": schema,
        "condition_columns": schema["condition_columns"],
        "descriptor_columns": schema["descriptor_columns"],
        "prototype_count": int(schema.get("prototype_count", 0)),
        "timesteps": int(args.timesteps),
        "beta_schedule": str(args.beta_schedule),
        "prediction_target": str(args.prediction_target),
        "epochs": int(epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "cond_dropout": float(args.cond_dropout),
        "seed": int(args.seed),
        "amp": bool(args.amp),
        "ema": bool(args.ema),
        "device": str(device),
        "base_channels": 64,
        "emb_dim": 256,
    }
    write_json(out_dir / "config.json", config)
    history: list[dict[str, float]] = []
    for epoch in range(start_epoch, epochs + 1):
        train_loss = _epoch(model, train_loader, diffusion, device, optimizer, scaler, amp_enabled, float(args.cond_dropout), ema)
        val_loss = _epoch(model, val_loader, diffusion, device, None, None, False, 0.0, None)
        ema_val_loss = float("nan")
        if ema is not None:
            ema_val_loss = _epoch(ema.shadow, val_loader, diffusion, device, None, None, False, 0.0, None)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss, "ema_val_loss": ema_val_loss})
        monitor = ema_val_loss if np.isfinite(ema_val_loss) else val_loss
        if monitor < best_val:
            best_val = monitor
            save_diffusion_v2_checkpoint(out_dir / "checkpoints" / "best.pt", model, optimizer, epoch, best_val, config)
        save_diffusion_v2_checkpoint(out_dir / "checkpoints" / "last.pt", model, optimizer, epoch, best_val, config)
        if ema is not None:
            save_diffusion_v2_checkpoint(out_dir / "checkpoints" / "ema_last.pt", ema.shadow, None, epoch, best_val, config)
        if int(args.sample_every) > 0 and (epoch == epochs or epoch % int(args.sample_every) == 0):
            sample_model = ema.shadow if ema is not None else model
            _sample_grid(
                sample_model,
                diffusion,
                latents_dir,
                condition_rows,
                schema,
                resolve_repo_path(args.autoencoder_checkpoint),
                out_dir / f"sample_grid_oracle_val_epoch{epoch}.png",
                device,
                steps=min(25, int(args.timesteps)),
            )
    sample_model = ema.shadow if ema is not None else model
    sample_stats = _sample_grid(
        sample_model,
        diffusion,
        latents_dir,
        condition_rows,
        schema,
        resolve_repo_path(args.autoencoder_checkpoint),
        out_dir / "sample_grid_oracle_val_epochfinal.png",
        device,
        steps=min(25, int(args.timesteps)),
    )
    write_training_curves(out_dir / "training_curves.png", history, ["train_loss", "val_loss", "ema_val_loss"])
    metrics = {
        "history": history,
        "train_loss": history[-1]["train_loss"],
        "val_loss": history[-1]["val_loss"],
        "ema_val_loss": history[-1].get("ema_val_loss"),
        "best_val_loss": best_val,
        "sample_grid": display_path(out_dir / "sample_grid_oracle_val_epochfinal.png"),
        "sample_pixel_mean": sample_stats["sample_pixel_mean"],
        "sample_pixel_std": sample_stats["sample_pixel_std"],
        "generated_nonconstant": bool(sample_stats["sample_pixel_std"] > 1e-4),
        "ema_checkpoint": display_path(out_dir / "checkpoints" / "ema_last.pt") if args.ema else "",
    }
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    metrics = train_diffusion(args)
    print(f"Wrote latent diffusion v2 outputs to {display_path(resolve_repo_path(args.out))}")
    print(f"val_loss={metrics['val_loss']:.6f} sample_std={metrics['sample_pixel_std']:.6f}")


if __name__ == "__main__":
    main()
