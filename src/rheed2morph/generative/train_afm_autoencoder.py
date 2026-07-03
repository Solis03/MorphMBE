"""Train the spatial-latent AFM autoencoder."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from rheed2morph.generative.common import (
    AFMImageDataset,
    display_path,
    load_data_index,
    read_json,
    resolve_repo_path,
    resolve_torch_device,
    set_seed,
    write_json,
)
from rheed2morph.generative.losses import reconstruction_loss
from rheed2morph.generative.models.afm_autoencoder import AFMAutoencoder, build_afm_autoencoder
from rheed2morph.generative.visualization import write_panel_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an AFM spatial-latent autoencoder.")
    parser.add_argument("--data-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def save_autoencoder_checkpoint(
    path: Path,
    model: AFMAutoencoder,
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


def load_autoencoder_checkpoint(path: Path, device_name: str = "auto") -> tuple[AFMAutoencoder, dict[str, Any]]:
    device = resolve_torch_device(device_name)
    payload = torch.load(resolve_repo_path(path), map_location=device)
    config = dict(payload.get("config", {}))
    model = build_afm_autoencoder(
        image_size=int(config.get("image_size", payload.get("image_size", 128))),
        latent_channels=int(config.get("latent_channels", payload.get("latent_channels", 8))),
    ).to(device)
    state = payload.get("model_state_dict", payload.get("model_state", payload))
    model.load_state_dict(state)
    model.eval()
    return model, payload


def _loader(records: list[Any], batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(AFMImageDataset(records), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def _run_epoch(
    model: AFMAutoencoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    amp_enabled: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "l1": 0.0, "gradient_l1": 0.0, "psd_l1": 0.0, "roughness_error": 0.0}
    count = 0
    for images, _metadata in loader:
        images = images.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                recon, _latent = model(images)
                loss, parts = reconstruction_loss(recon, images)
        if not torch.isfinite(loss):
            raise RuntimeError("Autoencoder loss became non-finite.")
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        batch_size = int(images.shape[0])
        count += batch_size
        for key in totals:
            value = parts[key] if key in parts else loss.detach()
            totals[key] += float(value.detach().cpu()) * batch_size
    if count == 0:
        return {key: float("nan") for key in totals}
    return {key: value / count for key, value in totals.items()}


def _write_recon_grid(model: AFMAutoencoder, records: list[Any], device: torch.device, path: Path) -> dict[str, float]:
    if not records:
        return {"target_pixel_mean": 0.0, "target_pixel_std": 0.0, "recon_pixel_mean": 0.0, "recon_pixel_std": 0.0}
    loader = _loader(records[:8], batch_size=min(8, len(records)), shuffle=False, num_workers=0)
    images, metadata = next(iter(loader))
    with torch.no_grad():
        recon, _latent = model(images.to(device))
    inputs = images.cpu().numpy()
    outputs = recon.cpu().numpy()
    residuals = np.abs(outputs - inputs)
    rows = [[inputs[i, 0], outputs[i, 0], residuals[i, 0]] for i in range(inputs.shape[0])]
    row_titles = [str(value) for value in metadata["sample_id"]]
    write_panel_grid(path, rows, ["original AFM", "reconstruction", "absolute residual"], row_titles, ["viridis", "viridis", "magma"])
    return {
        "target_pixel_mean": float(np.mean(inputs)),
        "target_pixel_std": float(np.std(inputs)),
        "recon_pixel_mean": float(np.mean(outputs)),
        "recon_pixel_std": float(np.std(outputs)),
    }


def _latent_stats(model: AFMAutoencoder, records: list[Any], device: torch.device, batch_size: int) -> dict[str, Any]:
    if not records:
        return {"latent_count": 0}
    loader = _loader(records, batch_size=batch_size, shuffle=False, num_workers=0)
    latents: list[np.ndarray] = []
    with torch.no_grad():
        for images, _metadata in loader:
            latent = model.encode(images.to(device)).detach().cpu().numpy().astype(np.float32)
            latents.append(latent)
    array = np.concatenate(latents, axis=0)
    return {
        "latent_count": int(array.shape[0]),
        "latent_shape": list(array.shape[1:]),
        "latent_mean": float(np.mean(array)),
        "latent_std": float(np.std(array)),
        "latent_min": float(np.min(array)),
        "latent_max": float(np.max(array)),
    }


def train_autoencoder(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_index = resolve_repo_path(args.data_index)
    epochs = 1 if args.quick else int(args.epochs)
    limit = args.limit
    if args.quick and limit is None:
        limit = 16
    train_records = load_data_index(data_index, split="train", limit=limit)
    val_records = load_data_index(data_index, split="val", limit=limit)
    if not val_records:
        val_records = train_records[: min(8, len(train_records))]
    if not train_records:
        raise RuntimeError(f"No train records found in {data_index}")
    device = resolve_torch_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    model = build_afm_autoencoder(image_size=int(args.image_size), latent_channels=int(args.latent_channels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    config = {
        "data_index": display_path(data_index),
        "image_size": int(args.image_size),
        "latent_channels": int(args.latent_channels),
        "epochs": int(epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "amp": bool(args.amp),
        "device": str(device),
        "quick": bool(args.quick),
    }
    write_json(out_dir / "config.json", config)
    train_loader = _loader(train_records, int(args.batch_size), shuffle=True, num_workers=int(args.num_workers))
    val_loader = _loader(val_records, int(args.batch_size), shuffle=False, num_workers=int(args.num_workers))
    best_val = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device, optimizer, scaler, amp_enabled)
        val_metrics = _run_epoch(model, val_loader, device, None, None, False)
        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_l1": val_metrics["l1"],
            "val_gradient_l1": val_metrics["gradient_l1"],
            "val_psd_l1": val_metrics["psd_l1"],
            "val_roughness_error": val_metrics["roughness_error"],
        }
        history.append(row)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_autoencoder_checkpoint(out_dir / "checkpoints" / "best.pt", model, optimizer, epoch, best_val, config)
        save_autoencoder_checkpoint(out_dir / "checkpoints" / "last.pt", model, optimizer, epoch, best_val, config)
    grid_stats = _write_recon_grid(model, val_records, device, out_dir / "recon_grid_val.png")
    latent_stats = _latent_stats(model, train_records + val_records, device, int(args.batch_size))
    write_json(out_dir / "latent_stats.json", latent_stats)
    collapse_warning = bool(grid_stats["recon_pixel_std"] < 1e-4)
    final = history[-1]
    metrics: dict[str, Any] = {
        "history": history,
        "train_loss": final["train_loss"],
        "val_loss": final["val_loss"],
        "val_l1": final["val_l1"],
        "val_gradient_l1": final["val_gradient_l1"],
        "val_psd_l1": final["val_psd_l1"],
        "val_roughness_error": final["val_roughness_error"],
        "target_pixel_mean": grid_stats["target_pixel_mean"],
        "target_pixel_std": grid_stats["target_pixel_std"],
        "reconstructed_pixel_mean": grid_stats["recon_pixel_mean"],
        "reconstructed_pixel_std": grid_stats["recon_pixel_std"],
        "reconstructed/pixel_mean": grid_stats["recon_pixel_mean"],
        "reconstructed/pixel_std": grid_stats["recon_pixel_std"],
        "collapse_warning": collapse_warning,
        "recon_grid_val": display_path(out_dir / "recon_grid_val.png"),
        "best_checkpoint": display_path(out_dir / "checkpoints" / "best.pt"),
        "last_checkpoint": display_path(out_dir / "checkpoints" / "last.pt"),
    }
    if collapse_warning:
        metrics["quality_warning"] = "reconstruction_std_near_zero"
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    metrics = train_autoencoder(args)
    print(f"Wrote AFM autoencoder outputs to {display_path(resolve_repo_path(args.out))}")
    print(f"val_loss={metrics['val_loss']:.6f} reconstructed_std={metrics['reconstructed_pixel_std']:.6f}")


if __name__ == "__main__":
    main()
