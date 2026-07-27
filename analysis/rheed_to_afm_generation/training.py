from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from analysis.rheed_video_afm_story.afm_evaluation import reconstruction_metrics
from analysis.rheed_video_afm_story.afm_losses import afm_loss
from analysis.rheed_video_afm_story.common import repo_path, write_csv, write_json

from .data import AFMConditionDataset, ConditionScaler
from .model import ConditionalAFMVAE, architecture_summary, gaussian_kl


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_unit_map(row: pd.Series, resolution: int) -> np.ndarray:
    paths = json.loads(str(row["unit_shape_paths"]))
    return np.load(repo_path(paths[str(resolution)]), allow_pickle=False).astype(
        np.float32
    )


def _prior_validation_score(
    model: ConditionalAFMVAE,
    rows: pd.DataFrame,
    predicted_conditions: dict[str, np.ndarray],
    predicted_raw_conditions: dict[str, np.ndarray],
    condition_scaler: ConditionScaler,
    resolution: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metric_rows: list[dict[str, float]] = []
    diversity_ratios: list[float] = []
    with torch.no_grad():
        for group_id, group_rows in rows.groupby("growth_run_id"):
            group = str(group_id)
            condition = torch.as_tensor(
                predicted_conditions[group][None], dtype=torch.float32, device=device
            )
            generated = (
                model.generate(condition, use_prior_mean=True)
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )
            for _, row in group_rows.iterrows():
                target = _load_unit_map(row, resolution)
                metrics = reconstruction_metrics(
                    target, generated, float(row["rq_nm"])
                )
                metrics["growth_run_id"] = group
                metric_rows.append(metrics)
            sample_condition = condition.repeat(4, 1)
            generator = torch.Generator(device=device).manual_seed(
                90_000 + int(group)
            )
            generated_samples = (
                model.generate(sample_condition, generator=generator)
                .squeeze(1)
                .detach()
                .cpu()
                .numpy()
            )
            real_samples = [
                _load_unit_map(row, resolution)
                for _, row in group_rows.iterrows()
            ]

            def pairwise_l1(samples: list[np.ndarray] | np.ndarray) -> float:
                values = [
                    float(np.mean(np.abs(samples[i] - samples[j])))
                    for i in range(len(samples))
                    for j in range(i + 1, len(samples))
                ]
                return float(np.median(values)) if values else 0.0

            diversity_ratios.append(
                pairwise_l1(generated_samples)
                / max(pairwise_l1(real_samples), 1e-8)
            )
    frame = pd.DataFrame(metric_rows)
    group_metrics = frame.groupby("growth_run_id").median(numeric_only=True)
    log_rq_position = condition_scaler.columns.index("log_rq_nm")
    rq_errors = []
    for group, group_rows in rows.groupby("growth_run_id"):
        predicted_rq = float(
            np.exp(predicted_raw_conditions[str(group)][log_rq_position])
        )
        rq_errors.append(abs(predicted_rq - float(group_rows["rq_nm"].median())))
    composite = float(group_metrics["composite_score"].median())
    rq_mae = float(np.mean(rq_errors))
    diversity_ratio = float(np.median(diversity_ratios))
    diversity_penalty = float(
        abs(np.log(np.clip(diversity_ratio, 1e-4, 1e4)))
    )
    return {
        "val_prior_composite": composite,
        "val_rq_mae_nm": rq_mae,
        "val_selection_score": composite
        + 0.10 * rq_mae
        + 0.20 * diversity_penalty,
        "val_prior_ssim": float(group_metrics["ssim"].median()),
        "val_prior_psd_log_distance": float(
            group_metrics["normalized_psd_log_distance"].median()
        ),
        "val_diversity_ratio": diversity_ratio,
        "val_diversity_log_penalty": diversity_penalty,
    }


def _reconstruction_validation(
    model: ConditionalAFMVAE,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: list[float] = []
    reconstructions: list[float] = []
    kls: list[float] = []
    with torch.no_grad():
        for images, conditions, _, _ in loader:
            images = images.to(device)
            conditions = conditions.to(device)
            output = model(images, conditions)
            reconstruction, _ = afm_loss(
                output["reconstruction"], images, "physics_shape"
            )
            kl = gaussian_kl(
                output["posterior_mean"],
                output["posterior_logvar"],
                output["prior_mean"],
                output["prior_logvar"],
            )
            total = reconstruction + 0.02 * kl
            totals.append(float(total.detach().cpu()))
            reconstructions.append(float(reconstruction.detach().cpu()))
            kls.append(float(kl.detach().cpu()))
    return {
        "val_elbo_total": float(np.mean(totals)),
        "val_reconstruction": float(np.mean(reconstructions)),
        "val_kl": float(np.mean(kls)),
    }


@dataclass
class TrainingResult:
    checkpoint_path: Path
    history_path: Path
    metrics_path: Path
    best_epoch: int
    best_selection_score: float
    runtime_seconds: float


def train_conditional_vae(
    *,
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    condition_scaler: ConditionScaler,
    validation_predicted_conditions: dict[str, np.ndarray],
    validation_predicted_raw_conditions: dict[str, np.ndarray],
    output_dir: str | Path,
    config: dict[str, Any],
    epochs: int,
    device_name: str = "auto",
) -> TrainingResult:
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(device_name)
    output = repo_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = AFMConditionDataset(
        train_rows,
        condition_scaler,
        int(config["resolution"]),
        augment=True,
    )
    validation_dataset = AFMConditionDataset(
        validation_rows,
        condition_scaler,
        int(config["resolution"]),
        augment=False,
    )
    group_counts = train_rows["growth_run_id"].astype(str).value_counts()
    weights = train_rows["growth_run_id"].astype(str).map(
        lambda group: 1.0 / float(group_counts[group])
    )
    sampler_generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights.to_numpy(float).copy(), dtype=torch.double),
        num_samples=max(len(train_rows), 64),
        replacement=True,
        generator=sampler_generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        sampler=sampler,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    model = ConditionalAFMVAE(
        condition_dim=len(condition_scaler.columns),
        latent_dim=int(config["latent_dim"]),
        resolution=int(config["resolution"]),
        base_channels=int(config["base_channels"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    history: list[dict[str, float | int]] = []
    best_score = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_path = checkpoint_dir / "best.pt"
    started = time.perf_counter()
    interval = int(config["validation_interval"])
    patience = int(config["early_stopping_patience"])
    kl_beta_max = float(config["kl_beta_max"])
    warmup = max(int(config["kl_warmup_epochs"]), 1)

    for epoch in range(1, int(epochs) + 1):
        model.train()
        beta = kl_beta_max * min(1.0, epoch / warmup)
        epoch_total: list[float] = []
        epoch_reconstruction: list[float] = []
        epoch_kl: list[float] = []
        epoch_diversity: list[float] = []
        epoch_condition_delta: list[float] = []
        for images, conditions, _, _ in train_loader:
            images = images.to(device)
            conditions = conditions.to(device)
            optimizer.zero_grad(set_to_none=True)
            output_batch = model(images, conditions)
            reconstruction, _ = afm_loss(
                output_batch["reconstruction"], images, "physics_shape"
            )
            kl = gaussian_kl(
                output_batch["posterior_mean"],
                output_batch["posterior_logvar"],
                output_batch["prior_mean"],
                output_batch["prior_logvar"],
            )
            prior_mean = output_batch["prior_mean"]
            prior_logvar = output_batch["prior_logvar"]
            prior_sample_a = model.reparameterize(prior_mean, prior_logvar)
            prior_sample_b = model.reparameterize(prior_mean, prior_logvar)
            generated_a = model.decode(prior_sample_a, conditions)
            generated_b = model.decode(prior_sample_b, conditions)
            latent_delta = torch.mean(torch.abs(generated_a - generated_b))
            diversity_regularizer = torch.relu(
                torch.as_tensor(
                    float(config["diversity_floor_l1"]),
                    dtype=latent_delta.dtype,
                    device=latent_delta.device,
                )
                - latent_delta
            )
            if len(conditions) > 1:
                permuted_conditions = torch.roll(conditions, shifts=1, dims=0)
                correct_condition = model.decode(prior_mean, conditions)
                wrong_condition = model.decode(prior_mean, permuted_conditions)
                condition_delta = torch.mean(
                    torch.abs(correct_condition - wrong_condition)
                )
                condition_regularizer = torch.relu(
                    torch.as_tensor(
                        float(config["condition_floor_l1"]),
                        dtype=condition_delta.dtype,
                        device=condition_delta.device,
                    )
                    - condition_delta
                )
            else:
                condition_delta = torch.zeros((), device=device)
                condition_regularizer = torch.zeros((), device=device)
            total = (
                reconstruction
                + beta * kl
                + float(config["diversity_regularizer_weight"])
                * diversity_regularizer
                + float(config["condition_regularizer_weight"])
                * condition_regularizer
            )
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"non-finite training loss at epoch {epoch}"
                )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_total.append(float(total.detach().cpu()))
            epoch_reconstruction.append(float(reconstruction.detach().cpu()))
            epoch_kl.append(float(kl.detach().cpu()))
            epoch_diversity.append(float(latent_delta.detach().cpu()))
            epoch_condition_delta.append(float(condition_delta.detach().cpu()))

        row: dict[str, float | int] = {
            "epoch": epoch,
            "kl_beta": beta,
            "train_total": float(np.mean(epoch_total)),
            "train_reconstruction": float(np.mean(epoch_reconstruction)),
            "train_kl": float(np.mean(epoch_kl)),
            "train_generated_pair_l1": float(np.mean(epoch_diversity)),
            "train_condition_swap_l1": float(np.mean(epoch_condition_delta)),
        }
        should_validate = epoch == 1 or epoch % interval == 0 or epoch == epochs
        if should_validate:
            row.update(_reconstruction_validation(model, validation_loader, device))
            row.update(
                _prior_validation_score(
                    model,
                    validation_rows,
                    validation_predicted_conditions,
                    validation_predicted_raw_conditions,
                    condition_scaler,
                    int(config["resolution"]),
                    device,
                )
            )
            score = float(row["val_selection_score"])
            if score < best_score - 1e-6:
                best_score = score
                best_epoch = epoch
                stale_epochs = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "condition_scaler": condition_scaler.to_dict(),
                        "architecture": architecture_summary(model),
                        "best_epoch": best_epoch,
                        "best_selection_score": best_score,
                        "train_groups": sorted(
                            train_rows["growth_run_id"].astype(str).unique()
                        ),
                        "validation_groups": sorted(
                            validation_rows["growth_run_id"].astype(str).unique()
                        ),
                    },
                    best_path,
                )
            else:
                stale_epochs += interval
        history.append(row)
        if stale_epochs >= patience:
            break

    runtime = time.perf_counter() - started
    history_frame = pd.DataFrame(history)
    history_path = output / "training_history.csv"
    write_csv(history_frame, history_path)
    metrics = {
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "epochs_requested": int(epochs),
        "epochs_completed": int(history_frame["epoch"].max()),
        "runtime_seconds": runtime,
        "device": str(device),
        "architecture": architecture_summary(model),
        "train_scan_count": len(train_rows),
        "train_group_count": train_rows["growth_run_id"].nunique(),
        "validation_scan_count": len(validation_rows),
        "validation_group_count": validation_rows["growth_run_id"].nunique(),
        "checkpoint_path": str(best_path.relative_to(repo_path("."))),
    }
    metrics_path = output / "training_metrics.json"
    write_json(metrics, metrics_path)
    if not best_path.exists():
        raise RuntimeError("training completed without a best checkpoint")
    return TrainingResult(
        checkpoint_path=best_path,
        history_path=history_path,
        metrics_path=metrics_path,
        best_epoch=best_epoch,
        best_selection_score=best_score,
        runtime_seconds=runtime,
    )


def load_model_checkpoint(
    checkpoint_path: str | Path,
    device_name: str = "auto",
) -> tuple[ConditionalAFMVAE, dict[str, Any], torch.device]:
    device = resolve_device(device_name)
    payload = torch.load(
        repo_path(checkpoint_path), map_location=device, weights_only=False
    )
    config = payload["config"]
    condition_scaler = payload["condition_scaler"]
    model = ConditionalAFMVAE(
        condition_dim=len(condition_scaler["columns"]),
        latent_dim=int(config["latent_dim"]),
        resolution=int(config["resolution"]),
        base_channels=int(config["base_channels"]),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload, device
