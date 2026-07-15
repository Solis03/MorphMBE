from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .afm_autoencoder import SmallResidualAFMAutoencoder, architecture_summary
from .afm_dataset import load_unit_shapes
from .afm_evaluation import reconstruction_metrics, summarize_metrics, group_level_metrics
from .afm_losses import afm_loss
from .common import display_path, repo_path, sha256_object, write_csv, write_json


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def device_for_training() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def group_balanced_epoch_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    groups = np.asarray(groups).astype(str)
    chosen = []
    for group in sorted(np.unique(groups)):
        candidates = np.where(groups == group)[0]
        chosen.append(int(rng.choice(candidates)))
    rng.shuffle(chosen)
    return np.asarray(chosen, dtype=int)


def validation_groups(train_groups: list[str], fraction: float, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    groups = np.asarray(sorted(train_groups))
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * fraction)))
    return sorted(groups[:n_val].tolist())


def tensor_batches(x: np.ndarray, indices: np.ndarray, batch_size: int, device: torch.device) -> list[torch.Tensor]:
    batches = []
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        arr = x[idx, None, :, :]
        batches.append(torch.from_numpy(arr).float().to(device))
    return batches


def train_one_model(
    X: np.ndarray,
    manifest: pd.DataFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    latent_dim: int,
    loss_preset: str,
    resolution: int,
    config: dict[str, Any],
    seed: int,
    checkpoint_path: Path,
) -> tuple[SmallResidualAFMAutoencoder, pd.DataFrame, dict[str, Any]]:
    set_seed(seed)
    device = device_for_training()
    model = SmallResidualAFMAutoencoder(latent_dim=latent_dim, resolution=resolution, base_channels=int(config["architecture"]["base_channels"])).to(device)
    opt = AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1)
    train_groups = manifest.loc[train_mask, "growth_run_id"].astype(str).to_numpy()
    global_train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    rng = np.random.default_rng(seed)
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    history_rows = []
    epochs = int(config["stage_a_epochs"] if resolution == int(config["input_resolutions"]["stage_a"]) else config["stage_b_epochs"])
    patience = int(config["early_stopping_patience"])
    stale = 0
    for epoch in range(epochs):
        model.train()
        local_balanced = group_balanced_epoch_indices(train_groups, rng)
        train_indices = global_train_indices[local_balanced]
        train_losses = []
        for batch in tensor_batches(X, train_indices, int(config["batch_size"]), device):
            opt.zero_grad(set_to_none=True)
            pred, _ = model(batch)
            loss, _ = afm_loss(pred, batch, loss_preset)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tensor_batches(X, val_indices, int(config["batch_size"]), device):
                pred, _ = model(batch)
                loss, _ = afm_loss(pred, batch, loss_preset)
                val_losses.append(float(loss.detach().cpu()))
        val_loss = float(np.mean(val_losses)) if val_losses else float(np.mean(train_losses))
        scheduler.step(val_loss)
        history_rows.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss, "validation_groups_excluded_from_gradient_update": True, "train_group_count": int(len(set(train_groups))), "val_group_count": int(len(set(manifest.loc[val_mask, 'growth_run_id'].astype(str))))})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "architecture": architecture_summary(model), "config_hash": sha256_object(config), "best_epoch": best_epoch, "best_val_loss": best_val}, checkpoint_path)
    return model, pd.DataFrame(history_rows), {"best_epoch": best_epoch, "best_val_loss": best_val, **architecture_summary(model)}


def reconstruct_with_model(model: SmallResidualAFMAutoencoder, X: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    recons, latents = [], []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size, None, :, :]).float().to(device)
            pred, z = model(batch)
            recons.append(pred[:, 0].cpu().numpy().astype(np.float32))
            latents.append(z.cpu().numpy().astype(np.float32))
    return np.concatenate(recons, axis=0), np.concatenate(latents, axis=0)


def run_autoencoder_cv(manifest: pd.DataFrame, config: dict[str, Any], split: pd.DataFrame, resolution: int, latent_dims: list[int] | None = None, loss_presets: list[str] | None = None, seeds: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_root = repo_path(config["output_root"])
    ckpt_root = output_root / "autoencoder_checkpoints"
    recon_root = output_root / "autoencoder_oof_reconstructions"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    recon_root.mkdir(parents=True, exist_ok=True)
    X = load_unit_shapes(manifest, resolution)
    groups = manifest["growth_run_id"].astype(str).to_numpy()
    latent_dims = latent_dims or [int(x) for x in config["latent_dimensions"]]
    loss_presets = loss_presets or list(config["loss_presets"])
    seeds = seeds or [int(config["screening_seed"])]
    scan_rows, registry_rows, history_rows = [], [], []
    for latent_dim in latent_dims:
        for loss_preset in loss_presets:
            for seed in seeds:
                model_id = f"ae_z{latent_dim}_{loss_preset}_{resolution}px_seed{seed}"
                full_recon = np.zeros_like(X, dtype=np.float32)
                full_latent = np.zeros((len(X), latent_dim), dtype=np.float32)
                for fold in sorted(split["fold"].unique()):
                    test_groups = set(split.query("fold == @fold and split == 'test'")["growth_run_id"].astype(str))
                    test_mask = np.asarray([g in test_groups for g in groups])
                    outer_train_groups = sorted(set(groups[~test_mask]))
                    val_groups = validation_groups(outer_train_groups, float(config["validation_split"]["group_fraction"]), int(config["validation_split"]["random_seed"]) + fold + seed)
                    val_mask = np.asarray([g in val_groups for g in groups])
                    train_mask = (~test_mask) & (~val_mask)
                    ckpt_path = ckpt_root / f"{model_id}_fold{fold}.pt"
                    model, history, summary = train_one_model(X, manifest, train_mask, val_mask, latent_dim, loss_preset, resolution, config, seed + fold, ckpt_path)
                    recon, latent = reconstruct_with_model(model, X[test_mask], int(config["batch_size"]))
                    full_recon[test_mask] = recon
                    full_latent[test_mask] = latent
                    for rec in history.to_dict("records"):
                        history_rows.append({"model_id": model_id, "fold": fold, "seed": seed, "latent_dim": latent_dim, "loss_preset": loss_preset, "resolution": resolution, **rec})
                    registry_rows.append(
                        {
                            "model_id": model_id,
                            "fold": fold,
                            "resolution": resolution,
                            "latent_dim": latent_dim,
                            "loss_preset": loss_preset,
                            "seed": seed,
                            "fit_scope": "outer_train_groups_only",
                            "validation_group_ids": json.dumps(val_groups),
                            "test_group_ids": json.dumps(sorted(test_groups)),
                            "checkpoint_path": display_path(ckpt_path),
                            **summary,
                            "global_transductive_development_model": False,
                        }
                    )
                recon_path = recon_root / f"{model_id}_oof_reconstructions.npz"
                np.savez_compressed(recon_path, reconstructions=full_recon, latents=full_latent, sample_ids=manifest["sample_id"].astype(str).to_numpy(), afm_file_ids=manifest["afm_file_id"].astype(str).to_numpy(), resolution=resolution, latent_dim=latent_dim, loss_preset=loss_preset, global_transductive_development_model=False)
                for i, row in manifest.reset_index(drop=True).iterrows():
                    metrics = reconstruction_metrics(X[i], full_recon[i], float(row["rq_nm"]))
                    scan_rows.append(
                        {
                            "model_family": "Autoencoder",
                            "model_id": model_id,
                            "resolution": resolution,
                            "latent_dim": latent_dim,
                            "loss_preset": loss_preset,
                            "seed": seed,
                            "sample_id": row["sample_id"],
                            "growth_run_id": row["growth_run_id"],
                            "afm_file_id": row["afm_file_id"],
                            "true_q_used_for_afm_side_decoder_evaluation": True,
                            **metrics,
                        }
                    )
    scan_metrics = pd.DataFrame(scan_rows)
    oof_metrics = summarize_metrics(scan_metrics, ["model_id", "resolution", "latent_dim", "loss_preset", "seed"])
    group_metrics = group_level_metrics(scan_metrics, ["model_id", "resolution", "latent_dim", "loss_preset", "seed"])
    registry = pd.DataFrame(registry_rows)
    history = pd.DataFrame(history_rows)
    write_csv(scan_metrics, output_root / "autoencoder_scan_metrics.csv")
    write_csv(oof_metrics, output_root / "autoencoder_oof_metrics.csv")
    write_csv(group_metrics, output_root / "autoencoder_group_metrics.csv")
    write_csv(registry, output_root / "decoder_model_registry.csv")
    write_csv(history, output_root / "autoencoder_training_history.csv")
    return scan_metrics, oof_metrics, group_metrics, registry


def train_global_development_model(manifest: pd.DataFrame, config: dict[str, Any], resolution: int, latent_dim: int, loss_preset: str, seed: int) -> tuple[SmallResidualAFMAutoencoder, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    output_root = repo_path(config["output_root"]) / "global_development_model"
    output_root.mkdir(parents=True, exist_ok=True)
    X = load_unit_shapes(manifest, resolution)
    groups = manifest["growth_run_id"].astype(str).to_numpy()
    all_groups = sorted(set(groups))
    val_groups = validation_groups(all_groups, float(config["validation_split"]["group_fraction"]), int(config["validation_split"]["random_seed"]) + seed + 99)
    val_mask = np.asarray([g in val_groups for g in groups])
    train_mask = ~val_mask
    local_config = dict(config)
    local_config["stage_b_epochs"] = int(config["global_epochs"])
    ckpt_path = output_root / f"global_ae_z{latent_dim}_{loss_preset}_{resolution}px_seed{seed}.pt"
    model, history, summary = train_one_model(X, manifest, train_mask, val_mask, latent_dim, loss_preset, resolution, local_config, seed, ckpt_path)
    recon, latents = reconstruct_with_model(model, X, int(config["batch_size"]))
    recon_path = output_root / "global_reconstructions_and_latents.npz"
    np.savez_compressed(recon_path, reconstructions=recon, latents=latents, sample_ids=manifest["sample_id"].astype(str).to_numpy(), afm_file_ids=manifest["afm_file_id"].astype(str).to_numpy(), resolution=resolution, latent_dim=latent_dim, loss_preset=loss_preset, global_transductive_development_model=True)
    rows = []
    for i, row in manifest.reset_index(drop=True).iterrows():
        rows.append({"model_family": "Autoencoder", "model_id": f"global_ae_z{latent_dim}_{loss_preset}_{resolution}px_seed{seed}", "global_transductive_development_model": True, "sample_id": row["sample_id"], "growth_run_id": row["growth_run_id"], "afm_file_id": row["afm_file_id"], **reconstruction_metrics(X[i], recon[i], float(row["rq_nm"]))})
    metrics = pd.DataFrame(rows)
    registry = pd.DataFrame([{**summary, "model_id": f"global_ae_z{latent_dim}_{loss_preset}_{resolution}px_seed{seed}", "resolution": resolution, "latent_dim": latent_dim, "loss_preset": loss_preset, "seed": seed, "fit_scope": "all_valid_1um_afm_transductive_development", "checkpoint_path": display_path(ckpt_path), "reconstruction_latent_path": display_path(recon_path), "global_transductive_development_model": True}])
    write_csv(history.assign(model_id=registry.iloc[0]["model_id"]), output_root / "training_history.csv")
    write_csv(metrics, output_root / "global_development_metrics.csv")
    write_csv(registry, output_root / "global_development_model_registry.csv")
    return model, metrics, registry, recon, latents
