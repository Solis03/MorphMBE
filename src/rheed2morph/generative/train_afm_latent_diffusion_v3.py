"""Train condition-control AFM latent diffusion v3."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from rheed2morph.generative.afm_prior_v2_utils import write_training_curves
from rheed2morph.generative.common import display_path, read_csv_rows, read_json, resolve_repo_path, resolve_torch_device, set_seed, write_json
from rheed2morph.generative.condition_control_v3_utils import build_condition_matrix_v3
from rheed2morph.generative.diffusion_v2 import GaussianDiffusionV2
from rheed2morph.generative.models.latent_unet import LatentUNet
from rheed2morph.generative.train_afm_autoencoder_v2 import load_autoencoder_v2_checkpoint
from rheed2morph.generative.train_latent_descriptor_regressor import load_regressor_checkpoint
from rheed2morph.generative.visualization import write_panel_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AFM latent diffusion v3.")
    parser.add_argument("--latents-dir", type=Path, required=True)
    parser.add_argument("--autoencoder-checkpoint", type=Path, required=True)
    parser.add_argument("--condition-table", type=Path, required=True)
    parser.add_argument("--condition-schema", type=Path, required=True)
    parser.add_argument("--latent-descriptor-regressor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--prediction-target", choices=["epsilon", "v"], default="v")
    parser.add_argument("--cond-dropout", type=float, default=0.10)
    parser.add_argument("--descriptor-mask-prob", type=float, default=0.10)
    parser.add_argument("--aux-cond-loss-weight", type=float, default=0.10)
    parser.add_argument("--prototype-balance", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--sample-every", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


class V3LatentDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]]):
    def __init__(self, latents_dir: Path, split: str, rows: list[dict[str, str]], schema: dict[str, Any], limit: int | None = None) -> None:
        payload = np.load(latents_dir / f"latents_{split}.npz", allow_pickle=True)
        latents = np.asarray(payload["latents"], dtype=np.float32)
        row_ids = np.asarray(payload["row_ids"])
        if limit is not None:
            latents = latents[:limit]
            row_ids = row_ids[:limit]
        conditions = build_condition_matrix_v3(rows, row_ids, schema)
        by_row = {row["row_id"]: row for row in rows}
        prototypes = []
        for row_id in row_ids.tolist():
            value = by_row[str(row_id)].get("prototype_id", "")
            prototypes.append(int(float(value)) if value != "" else -1)
        self.latents = torch.from_numpy(latents)
        self.conditions = torch.from_numpy(conditions)
        self.prototypes = torch.from_numpy(np.asarray(prototypes, dtype=np.int64))
        self.row_ids = [str(value) for value in row_ids.tolist()]

    def __len__(self) -> int:
        return int(self.latents.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        return self.latents[index], self.conditions[index], self.prototypes[index], self.row_ids[index]


class EMAModel:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for shadow, param in zip(self.shadow.parameters(), model.parameters()):
            shadow.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
        for shadow_buffer, buffer in zip(self.shadow.buffers(), model.buffers()):
            shadow_buffer.data.copy_(buffer.data)


def save_diffusion_v3_checkpoint(path: Path, model: LatentUNet, optimizer: torch.optim.Optimizer | None, epoch: int, best: float, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model_state_dict": model.state_dict(), "epoch": int(epoch), "best_val_loss": float(best), "config": config}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_diffusion_v3_checkpoint(path: Path, device_name: str = "auto") -> tuple[LatentUNet, dict[str, Any]]:
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


def _mask_condition(condition: torch.Tensor, descriptor_dim: int, cond_dropout: float, descriptor_mask_prob: float) -> torch.Tensor:
    output = condition
    batch = condition.shape[0]
    if cond_dropout > 0:
        keep = (torch.rand(batch, device=condition.device) >= cond_dropout).float().view(batch, 1)
        output = output * keep
    if descriptor_mask_prob > 0 and descriptor_dim > 0:
        mask = (torch.rand(batch, descriptor_dim, device=condition.device) >= descriptor_mask_prob).float()
        output = output.clone()
        output[:, :descriptor_dim] = output[:, :descriptor_dim] * mask
    return output


def _pred_x0(diffusion: GaussianDiffusionV2, noisy: torch.Tensor, timesteps: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
    eps = diffusion.predict_epsilon(noisy, timesteps, model_output)
    sqrt_alpha = diffusion.sqrt_alpha_bars.to(noisy.device).gather(0, timesteps).view(noisy.shape[0], 1, 1, 1)
    sqrt_one_minus = diffusion.sqrt_one_minus_alpha_bars.to(noisy.device).gather(0, timesteps).view(noisy.shape[0], 1, 1, 1)
    return (noisy - sqrt_one_minus * eps) / sqrt_alpha


def _epoch(
    model: LatentUNet,
    regressor: torch.nn.Module,
    loader: DataLoader,
    diffusion: GaussianDiffusionV2,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    descriptor_dim: int,
    cond_dropout: float,
    descriptor_mask_prob: float,
    aux_weight: float,
    ema: EMAModel | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    regressor.eval()
    totals = {"loss": 0.0, "denoising_loss": 0.0, "aux_condition_loss": 0.0}
    count = 0
    for latents, conditions, _prototypes, _row_ids in loader:
        latents = latents.to(device)
        conditions = conditions.to(device)
        batch = int(latents.shape[0])
        timesteps = torch.randint(0, diffusion.timesteps, (batch,), device=device, dtype=torch.long)
        noise = torch.randn_like(latents)
        noisy = diffusion.q_sample(latents, timesteps, noise)
        target = diffusion._target(latents, timesteps, noise)
        model_condition = _mask_condition(conditions, descriptor_dim, cond_dropout if training else 0.0, descriptor_mask_prob if training else 0.0)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                predicted = model(noisy, timesteps, model_condition)
                denoise_loss = F.mse_loss(predicted, target)
                aux_loss = latents.new_tensor(0.0)
                if aux_weight > 0:
                    x0 = _pred_x0(diffusion, noisy, timesteps, predicted)
                    pred_desc, _proto = regressor(x0)
                    aux_loss = F.mse_loss(pred_desc, conditions[:, :descriptor_dim])
                loss = denoise_loss + float(aux_weight) * aux_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Diffusion v3 loss became non-finite.")
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
        totals["loss"] += float(loss.detach().cpu()) * batch
        totals["denoising_loss"] += float(denoise_loss.detach().cpu()) * batch
        totals["aux_condition_loss"] += float(aux_loss.detach().cpu()) * batch
        count += batch
    return {key: value / max(count, 1) for key, value in totals.items()}


def _make_loader(dataset: V3LatentDataset, batch_size: int, balance: bool) -> DataLoader:
    if balance and torch.any(dataset.prototypes >= 0):
        labels = dataset.prototypes.numpy()
        counts = {label: max(int(np.sum(labels == label)), 1) for label in set(labels.tolist()) if label >= 0}
        weights = np.asarray([1.0 / counts.get(int(label), len(labels)) for label in labels], dtype=np.float64)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


@torch.no_grad()
def _sample_grid(
    model: LatentUNet,
    diffusion: GaussianDiffusionV2,
    autoencoder_checkpoint: Path,
    latents_dir: Path,
    dataset: V3LatentDataset,
    out_path: Path,
    device: torch.device,
    steps: int,
) -> dict[str, float]:
    if len(dataset) == 0:
        return {"sample_pixel_std": 0.0, "sample_latent_std": 0.0}
    indices = list(range(min(4, len(dataset))))
    latents = dataset.latents[indices].to(device)
    conditions = dataset.conditions[indices].to(device)
    latent_shape = tuple(int(value) for value in latents.shape[1:])
    sampled = diffusion.sample_ddim(model, (len(indices), *latent_shape), conditions, steps=steps, guidance_scale=2.0)
    stats = np.load(latents_dir / "latent_standardization_v2.npz")
    latent_mean = torch.from_numpy(np.asarray(stats["latent_mean"], dtype=np.float32)).to(device)
    latent_std = torch.from_numpy(np.asarray(stats["latent_std"], dtype=np.float32)).to(device)
    ae, _payload = load_autoencoder_v2_checkpoint(autoencoder_checkpoint, str(device))
    ae.to(device).eval()
    true_raw = latents * latent_std + latent_mean
    sampled_raw = sampled * latent_std + latent_mean
    recon = ae.decode(true_raw).detach().cpu().numpy()
    generated = ae.decode(sampled_raw).detach().cpu().numpy()
    rows = [[recon[i, 0], generated[i, 0]] for i in range(len(indices))]
    write_panel_grid(out_path, rows, ["AE recon", "v3 sample"], [dataset.row_ids[i] for i in indices])
    return {"sample_pixel_std": float(np.std(generated)), "sample_latent_std": float(torch.std(sampled).detach().cpu())}


def train_diffusion_v3(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    latents_dir = resolve_repo_path(args.latents_dir)
    rows = read_csv_rows(resolve_repo_path(args.condition_table))
    schema = read_json(resolve_repo_path(args.condition_schema))
    epochs = 1 if args.quick else int(args.epochs)
    limit = 32 if args.quick and args.limit is None else args.limit
    train_data = V3LatentDataset(latents_dir, "train", rows, schema, limit=limit)
    val_data = V3LatentDataset(latents_dir, "val", rows, schema, limit=limit)
    if len(val_data) == 0:
        val_data = train_data
    device = resolve_torch_device(args.device)
    descriptor_dim = len(schema["condition_columns"])
    latent_shape = tuple(int(value) for value in train_data.latents.shape[1:])
    condition_dim = int(train_data.conditions.shape[1])
    model = LatentUNet(latent_channels=latent_shape[0], condition_dim=condition_dim, base_channels=64, emb_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    start_epoch = 1
    best = float("inf")
    if args.resume is not None:
        payload = torch.load(resolve_repo_path(args.resume), map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        if "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best = float(payload.get("best_val_loss", best))
    regressor, _reg_payload = load_regressor_checkpoint(args.latent_descriptor_regressor, str(device))
    for param in regressor.parameters():
        param.requires_grad_(False)
    diffusion = GaussianDiffusionV2(int(args.timesteps), str(args.beta_schedule), str(args.prediction_target), device)
    ema = EMAModel(model) if args.ema else None
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_loader = _make_loader(train_data, int(args.batch_size), bool(args.prototype_balance))
    val_loader = DataLoader(val_data, batch_size=int(args.batch_size), shuffle=False, num_workers=0)
    config = {
        "latents_dir": display_path(latents_dir),
        "autoencoder_checkpoint": display_path(resolve_repo_path(args.autoencoder_checkpoint)),
        "condition_table": display_path(resolve_repo_path(args.condition_table)),
        "condition_schema": schema,
        "condition_schema_path": display_path(resolve_repo_path(args.condition_schema)),
        "latent_descriptor_regressor": display_path(resolve_repo_path(args.latent_descriptor_regressor)),
        "latent_shape": list(latent_shape),
        "latent_channels": latent_shape[0],
        "condition_dim": condition_dim,
        "descriptor_dim": descriptor_dim,
        "prototype_count": int(schema.get("prototype_count", 0)),
        "timesteps": int(args.timesteps),
        "beta_schedule": str(args.beta_schedule),
        "prediction_target": str(args.prediction_target),
        "cond_dropout": float(args.cond_dropout),
        "descriptor_mask_prob": float(args.descriptor_mask_prob),
        "aux_cond_loss_weight": float(args.aux_cond_loss_weight),
        "prototype_balance": bool(args.prototype_balance),
        "epochs": epochs,
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "amp": bool(args.amp),
        "ema": bool(args.ema),
        "seed": int(args.seed),
        "device": str(device),
        "base_channels": 64,
        "emb_dim": 256,
        "conditioning_injection": "LatentUNet conditional embedding modulates every residual block with scale/shift FiLM-style parameters.",
    }
    write_json(out_dir / "config.json", config)
    history: list[dict[str, float]] = []
    for epoch in range(start_epoch, epochs + 1):
        train = _epoch(model, regressor, train_loader, diffusion, device, optimizer, scaler, amp_enabled, descriptor_dim, float(args.cond_dropout), float(args.descriptor_mask_prob), float(args.aux_cond_loss_weight), ema)
        val = _epoch(model, regressor, val_loader, diffusion, device, None, None, False, descriptor_dim, 0.0, 0.0, float(args.aux_cond_loss_weight), None)
        row = {
            "epoch": float(epoch),
            "train_loss": train["loss"],
            "train_denoising_loss": train["denoising_loss"],
            "train_aux_condition_loss": train["aux_condition_loss"],
            "val_loss": val["loss"],
            "val_denoising_loss": val["denoising_loss"],
            "val_aux_condition_loss": val["aux_condition_loss"],
        }
        history.append(row)
        if val["loss"] < best:
            best = val["loss"]
            save_diffusion_v3_checkpoint(out_dir / "checkpoints" / "best.pt", model, optimizer, epoch, best, config)
        save_diffusion_v3_checkpoint(out_dir / "checkpoints" / "last.pt", model, optimizer, epoch, best, config)
        if ema is not None:
            save_diffusion_v3_checkpoint(out_dir / "checkpoints" / "ema_last.pt", ema.shadow, None, epoch, best, config)
        if int(args.sample_every) > 0 and (epoch == epochs or epoch % int(args.sample_every) == 0):
            sample_model = ema.shadow if ema is not None else model
            stats = _sample_grid(sample_model, diffusion, resolve_repo_path(args.autoencoder_checkpoint), latents_dir, val_data, out_dir / f"sample_grid_v3_oracle_val_epoch{epoch}.png", device, min(25, int(args.timesteps)))
            write_json(out_dir / f"condition_control_metrics_epoch{epoch}.json", stats)
    sample_model = ema.shadow if ema is not None else model
    sample_stats = _sample_grid(sample_model, diffusion, resolve_repo_path(args.autoencoder_checkpoint), latents_dir, val_data, out_dir / "sample_grid_v3_oracle_val_epochfinal.png", device, min(25, int(args.timesteps)))
    write_training_curves(out_dir / "training_curves.png", history, ["train_loss", "val_loss", "train_aux_condition_loss", "val_aux_condition_loss"])
    metrics = {
        "history": history,
        "train_loss": history[-1]["train_loss"],
        "val_loss": history[-1]["val_loss"],
        "train_denoising_loss": history[-1]["train_denoising_loss"],
        "val_denoising_loss": history[-1]["val_denoising_loss"],
        "train_aux_condition_loss": history[-1]["train_aux_condition_loss"],
        "val_aux_condition_loss": history[-1]["val_aux_condition_loss"],
        "best_val_loss": best,
        "sample_pixel_std": sample_stats["sample_pixel_std"],
        "sample_latent_std": sample_stats["sample_latent_std"],
        "generated_nonconstant": bool(sample_stats["sample_pixel_std"] > 1e-4),
        "ema_checkpoint": display_path(out_dir / "checkpoints" / "ema_last.pt") if args.ema else "",
    }
    write_json(out_dir / "metrics.json", metrics)
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    metrics = train_diffusion_v3(args)
    print(f"Wrote latent diffusion v3 to {display_path(resolve_repo_path(args.out))}")
    print(f"val_loss={metrics['val_loss']:.6f} sample_std={metrics['sample_pixel_std']:.6f}")


if __name__ == "__main__":
    main()
