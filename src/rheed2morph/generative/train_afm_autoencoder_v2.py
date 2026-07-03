"""Train AFM autoencoder v2."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rheed2morph.generative.afm_prior_v2_utils import (
    AFMPriorV2Dataset,
    V2_DESCRIPTOR_NAMES,
    compute_afm_descriptors_v2,
    load_v2_index,
    write_training_curves,
)
from rheed2morph.generative.common import (
    display_path,
    load_height_array,
    read_csv_rows,
    resolve_repo_path,
    resolve_torch_device,
    set_seed,
    write_csv_rows,
    write_json,
)
from rheed2morph.generative.losses import gradient_l1, log_psd_l1, roughness_consistency
from rheed2morph.generative.models.afm_autoencoder_v2 import AFMAutoencoderV2, build_afm_autoencoder_v2
from rheed2morph.generative.visualization import write_panel_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AFM autoencoder v2.")
    parser.add_argument("--data-index", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--patch-mode", type=str, default=None)
    parser.add_argument("--patches-per-image", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def histogram_loss(recon: torch.Tensor, target: torch.Tensor, bins: int = 32) -> torch.Tensor:
    losses = []
    edges = torch.linspace(-1.0, 1.0, bins + 1, device=recon.device)
    for image_a, image_b in zip(recon.float(), target.float()):
        hist_a = torch.histc(image_a, bins=bins, min=-1.0, max=1.0)
        hist_b = torch.histc(image_b, bins=bins, min=-1.0, max=1.0)
        hist_a = hist_a / torch.clamp(hist_a.sum(), min=1.0)
        hist_b = hist_b / torch.clamp(hist_b.sum(), min=1.0)
        losses.append(F.l1_loss(hist_a, hist_b))
    _ = edges
    return torch.stack(losses).mean() if losses else recon.new_tensor(0.0)


def multiscale_l1(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    total = F.l1_loss(recon, target)
    for scale in (2, 4):
        total = total + F.l1_loss(F.avg_pool2d(recon, scale), F.avg_pool2d(target, scale))
    return total / 3.0


def reconstruction_loss_v2(recon: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    l1 = F.l1_loss(recon, target)
    grad = gradient_l1(recon, target)
    psd = log_psd_l1(recon, target)
    rough = roughness_consistency(recon, target)
    hist = histogram_loss(recon, target)
    multi = multiscale_l1(recon, target)
    total = l1 + 0.25 * grad + 0.10 * psd + 0.10 * rough + 0.05 * hist + 0.05 * multi
    return total, {
        "loss": total.detach(),
        "l1": l1.detach(),
        "gradient_l1": grad.detach(),
        "psd_l1": psd.detach(),
        "roughness_error": rough.detach(),
        "histogram_loss": hist.detach(),
        "multiscale_l1": multi.detach(),
    }


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


def save_autoencoder_v2_checkpoint(
    path: Path,
    model: AFMAutoencoderV2,
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


def load_autoencoder_v2_checkpoint(path: Path, device_name: str = "auto") -> tuple[AFMAutoencoderV2, dict[str, Any]]:
    device = resolve_torch_device(device_name)
    payload = torch.load(resolve_repo_path(path), map_location=device)
    config = dict(payload.get("config", {}))
    model = build_afm_autoencoder_v2(
        image_size=int(config.get("image_size", 128)),
        latent_channels=int(config.get("latent_channels", 16)),
        latent_size=int(config.get("latent_size", 16)),
        base_channels=int(config.get("base_channels", 32)),
        dropout=float(config.get("dropout", 0.05)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _loader(records: list[Any], batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(AFMPriorV2Dataset(records), batch_size=batch_size, shuffle=shuffle, num_workers=workers)


def _run_epoch(
    model: AFMAutoencoderV2,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    ema: EMAModel | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {key: 0.0 for key in ("loss", "l1", "gradient_l1", "psd_l1", "roughness_error", "histogram_loss", "multiscale_l1")}
    count = 0
    for images, _metadata in loader:
        images = images.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                recon, _latent = model(images)
                loss, parts = reconstruction_loss_v2(recon, images)
        if not torch.isfinite(loss):
            raise RuntimeError("Autoencoder v2 loss became non-finite.")
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
        batch = int(images.shape[0])
        count += batch
        for key in totals:
            totals[key] += float(parts[key].detach().cpu()) * batch
    return {key: value / max(count, 1) for key, value in totals.items()}


def _psd_image(image: np.ndarray) -> np.ndarray:
    centered = image - float(np.mean(image))
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2).astype(np.float32)


def _write_recon_grid(model: AFMAutoencoderV2, records: list[Any], device: torch.device, path: Path) -> dict[str, float]:
    if not records:
        return {"original_mean": 0.0, "original_std": 0.0, "reconstructed_mean": 0.0, "reconstructed_std": 0.0}
    loader = _loader(records[: min(6, len(records))], min(6, len(records)), False, 0)
    images, metadata = next(iter(loader))
    with torch.no_grad():
        recon, _latent = model(images.to(device))
    inputs = images.cpu().numpy()
    outputs = recon.cpu().numpy()
    rows = []
    for i in range(inputs.shape[0]):
        original = inputs[i, 0]
        reconstructed = outputs[i, 0]
        rows.append([original, reconstructed, np.abs(original - reconstructed), _psd_image(original), _psd_image(reconstructed)])
    write_panel_grid(
        path,
        rows,
        ["original", "reconstruction", "residual", "original PSD", "recon PSD"],
        [str(value) for value in metadata["sample_id"]],
        ["viridis", "viridis", "magma", "magma", "magma"],
    )
    return {
        "original_mean": float(np.mean(inputs)),
        "original_std": float(np.std(inputs)),
        "reconstructed_mean": float(np.mean(outputs)),
        "reconstructed_std": float(np.std(outputs)),
    }


def _latent_stats(model: AFMAutoencoderV2, records: list[Any], device: torch.device, batch_size: int) -> dict[str, Any]:
    if not records:
        return {"latent_count": 0}
    loader = _loader(records, batch_size, False, 0)
    latents: list[np.ndarray] = []
    with torch.no_grad():
        for images, _metadata in loader:
            latents.append(model.encode(images.to(device)).detach().cpu().numpy().astype(np.float32))
    array = np.concatenate(latents, axis=0)
    return {
        "latent_count": int(array.shape[0]),
        "latent_shape": list(array.shape[1:]),
        "latent_mean": float(np.mean(array)),
        "latent_std": float(np.std(array)),
        "latent_min": float(np.min(array)),
        "latent_max": float(np.max(array)),
    }


def _descriptor_reconstruction_scatter(
    model: AFMAutoencoderV2,
    records: list[Any],
    device: torch.device,
    path: Path,
) -> None:
    if not records:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loader = _loader(records, min(16, len(records)), False, 0)
    original_desc: list[dict[str, float]] = []
    recon_desc: list[dict[str, float]] = []
    with torch.no_grad():
        for images, _metadata in loader:
            recon, _latent = model(images.to(device))
            originals = images.cpu().numpy()
            outputs = recon.cpu().numpy()
            for index in range(outputs.shape[0]):
                original_desc.append(compute_afm_descriptors_v2(originals[index, 0]))
                recon_desc.append(compute_afm_descriptors_v2(outputs[index, 0]))
    names = ["rq", "ra", "psd_slope", "autocorrelation_length_px"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), dpi=150)
    for axis, name in zip(axes.ravel(), names):
        x = [row[name] for row in original_desc]
        y = [row[name] for row in recon_desc]
        axis.scatter(x, y, s=12, alpha=0.8)
        axis.set_title(name)
        axis.set_xlabel("original")
        axis.set_ylabel("reconstruction")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def train_autoencoder(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_index = resolve_repo_path(args.data_index)
    epochs = 1 if args.quick else int(args.epochs)
    limit = int(args.limit) if args.limit is not None else (24 if args.quick else None)
    train_records = load_v2_index(data_index, split="train", limit=limit)
    val_records = load_v2_index(data_index, split="val", limit=limit)
    test_records = load_v2_index(data_index, split="test", limit=limit)
    if not train_records:
        raise RuntimeError(f"No train rows in {data_index}")
    if not val_records:
        val_records = train_records[: min(8, len(train_records))]
    device = resolve_torch_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    config = {
        "data_index": display_path(data_index),
        "descriptors": display_path(resolve_repo_path(args.descriptors)),
        "image_size": int(args.image_size),
        "latent_channels": int(args.latent_channels),
        "latent_size": int(args.latent_size),
        "base_channels": 32,
        "dropout": 0.05,
        "epochs": int(epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "amp": bool(args.amp),
        "ema": bool(args.ema),
        "device": str(device),
        "loss": "1.0*l1 + 0.25*gradient_l1 + 0.10*log_psd_l1 + 0.10*roughness + 0.05*histogram + 0.05*multiscale_l1",
    }
    model = build_afm_autoencoder_v2(
        image_size=int(args.image_size),
        latent_channels=int(args.latent_channels),
        latent_size=int(args.latent_size),
        base_channels=32,
        dropout=0.05,
    ).to(device)
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
    ema = EMAModel(model) if args.ema else None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_loader = _loader(train_records, int(args.batch_size), True, int(args.num_workers))
    val_loader = _loader(val_records, int(args.batch_size), False, int(args.num_workers))
    write_json(out_dir / "config.json", config)
    history: list[dict[str, float]] = []
    patience_left = int(args.early_stop_patience)
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = _run_epoch(model, train_loader, device, optimizer, scaler, amp_enabled, ema)
        val_metrics = _run_epoch(model, val_loader, device, None, None, False, None)
        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_l1": val_metrics["l1"],
            "val_gradient_l1": val_metrics["gradient_l1"],
            "val_psd_l1": val_metrics["psd_l1"],
            "val_histogram_loss": val_metrics["histogram_loss"],
            "val_roughness_error": val_metrics["roughness_error"],
        }
        history.append(row)
        improved = val_metrics["loss"] < best_val
        if improved:
            best_val = val_metrics["loss"]
            patience_left = int(args.early_stop_patience)
            save_autoencoder_v2_checkpoint(out_dir / "checkpoints" / "best.pt", model, optimizer, epoch, best_val, config)
            if ema is not None:
                save_autoencoder_v2_checkpoint(out_dir / "checkpoints" / "ema_best.pt", ema.shadow, None, epoch, best_val, config)
        else:
            patience_left -= 1
        save_autoencoder_v2_checkpoint(out_dir / "checkpoints" / "last.pt", model, optimizer, epoch, best_val, config)
        if ema is not None:
            save_autoencoder_v2_checkpoint(out_dir / "checkpoints" / "ema_last.pt", ema.shadow, None, epoch, best_val, config)
        if patience_left <= 0 and not args.quick:
            break
    eval_model = ema.shadow if ema is not None else model
    val_grid_stats = _write_recon_grid(eval_model, val_records, device, out_dir / "recon_grid_val.png")
    _write_recon_grid(eval_model, test_records or val_records, device, out_dir / "recon_grid_test.png")
    _descriptor_reconstruction_scatter(eval_model, val_records, device, out_dir / "descriptor_reconstruction_scatter.png")
    latent_stats = _latent_stats(eval_model, train_records + val_records, device, int(args.batch_size))
    write_json(out_dir / "latent_stats_preview.json", latent_stats)
    write_training_curves(out_dir / "training_curves.png", history, ["train_loss", "val_loss", "val_l1", "val_gradient_l1", "val_psd_l1"])
    metrics = {
        "history": history,
        "train_loss": history[-1]["train_loss"],
        "val_loss": history[-1]["val_loss"],
        "best_val_loss": best_val,
        "val_l1": history[-1]["val_l1"],
        "val_gradient_l1": history[-1]["val_gradient_l1"],
        "val_psd_l1": history[-1]["val_psd_l1"],
        "val_histogram_loss": history[-1]["val_histogram_loss"],
        "val_roughness_error": history[-1]["val_roughness_error"],
        "val_psd_slope_error": "",
        "val_autocorr_length_error": "",
        **val_grid_stats,
        "collapse_warning": bool(val_grid_stats["reconstructed_std"] < 1e-4),
        "recon_grid_val": display_path(out_dir / "recon_grid_val.png"),
        "recon_grid_test": display_path(out_dir / "recon_grid_test.png"),
        "descriptor_reconstruction_scatter": display_path(out_dir / "descriptor_reconstruction_scatter.png"),
        "best_checkpoint": display_path(out_dir / "checkpoints" / "best.pt"),
        "last_checkpoint": display_path(out_dir / "checkpoints" / "last.pt"),
        "ema_best_checkpoint": display_path(out_dir / "checkpoints" / "ema_best.pt") if args.ema else "",
    }
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    metrics = train_autoencoder(args)
    print(f"Wrote AFM autoencoder v2 outputs to {display_path(resolve_repo_path(args.out))}")
    print(f"val_loss={metrics['val_loss']:.6f} reconstructed_std={metrics['reconstructed_std']:.6f}")


if __name__ == "__main__":
    main()
