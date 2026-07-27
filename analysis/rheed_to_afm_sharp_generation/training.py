from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from analysis.rheed_video_afm_story.afm_descriptors import radial_psd
from analysis.rheed_video_afm_story.common import repo_path, write_csv, write_json

from analysis.rheed_to_afm_generation.data import ConditionScaler
from analysis.rheed_to_afm_generation.training import resolve_device, seed_everything

from .adversarial import (
    MorphologyConditionLoss,
    PhysicsSeededAFMRefiner,
    ProjectionDiscriminator,
    diff_augment,
    gradient_statistics,
    initialize_orthogonal,
)
from .spectral import ConditionalSpectralModel, load_unit_map


@dataclass
class AdversarialTrainingResult:
    checkpoint_path: Path
    history_path: Path
    best_step: int
    best_validation_score: float
    runtime_seconds: float


def _training_tensors(
    *,
    train_rows: pd.DataFrame,
    condition_scaler: ConditionScaler,
    spectral_model: ConditionalSpectralModel,
    resolution: int,
    seeds_per_group: int,
    seed: int,
    iaaft_iterations: int,
) -> TensorDataset:
    real_images: list[np.ndarray] = []
    spectral_seeds: list[np.ndarray] = []
    conditions: list[np.ndarray] = []
    for group_index, (group_id, group_rows) in enumerate(
        train_rows.groupby("growth_run_id")
    ):
        raw_condition = (
            group_rows[condition_scaler.columns].median().to_numpy(float)
        )
        condition = condition_scaler.transform(
            raw_condition[None], clip=False
        )[0]
        rows = [row for _, row in group_rows.iterrows()]
        for draw in range(seeds_per_group):
            real = load_unit_map(rows[draw % len(rows)], resolution)
            generated_seed = spectral_model.generate(
                condition,
                seed=seed + group_index * 10_000 + draw,
                iterations=iaaft_iterations,
            )
            real_images.append(real[None])
            spectral_seeds.append(generated_seed[None])
            conditions.append(condition)
    return TensorDataset(
        torch.from_numpy(np.stack(real_images).astype(np.float32)),
        torch.from_numpy(np.stack(spectral_seeds).astype(np.float32)),
        torch.from_numpy(np.stack(conditions).astype(np.float32)),
    )


def _radial_log_psd_error(real: np.ndarray, generated: np.ndarray) -> float:
    _, real_power = radial_psd(real)
    _, generated_power = radial_psd(generated)
    real_power /= max(float(real_power.sum()), 1e-12)
    generated_power /= max(float(generated_power.sum()), 1e-12)
    return float(
        np.mean(
            np.abs(
                np.log(real_power + 1e-9) - np.log(generated_power + 1e-9)
            )
        )
    )


@torch.no_grad()
def _validation_score(
    *,
    generator: PhysicsSeededAFMRefiner,
    spectral_model: ConditionalSpectralModel,
    validation_rows: pd.DataFrame,
    predicted_conditions: dict[str, np.ndarray],
    device: torch.device,
    seed: int,
    iaaft_iterations: int,
) -> dict[str, float]:
    generator.eval()
    psd_errors: list[float] = []
    gradient_relative_errors: list[float] = []
    boundary_penalties: list[float] = []
    for group_index, (group_id, rows) in enumerate(
        validation_rows.groupby("growth_run_id")
    ):
        group = str(group_id)
        condition = predicted_conditions[group]
        spectral_fields = [
            spectral_model.generate(
                condition,
                seed=seed + group_index * 1000 + draw,
                iterations=iaaft_iterations,
            )
            for draw in range(4)
        ]
        tensor = torch.from_numpy(
            np.stack(spectral_fields)[:, None].astype(np.float32)
        ).to(device)
        condition_tensor = torch.from_numpy(
            np.repeat(condition[None], len(spectral_fields), axis=0).astype(
                np.float32
            )
        ).to(device)
        generated = generator(tensor, condition_tensor).cpu().numpy()[:, 0]
        real = [
            load_unit_map(row, spectral_model.resolution)
            for _, row in rows.iterrows()
        ]
        real_psd_reference = real[len(real) // 2]
        psd_errors.append(
            float(
                np.median(
                    [
                        _radial_log_psd_error(real_psd_reference, image)
                        for image in generated
                    ]
                )
            )
        )

        def gradient(array: np.ndarray) -> float:
            gy, gx = np.gradient(array)
            return float(np.mean(np.hypot(gx, gy)))

        real_gradient = float(np.median([gradient(image) for image in real]))
        generated_gradient = float(
            np.median([gradient(image) for image in generated])
        )
        gradient_relative_errors.append(
            abs(generated_gradient - real_gradient) / max(real_gradient, 1e-8)
        )
        boundary_values = []
        for image in generated:
            gy, gx = np.gradient(image)
            magnitude = np.hypot(gx, gy)
            border = np.concatenate(
                [
                    magnitude[:8].ravel(),
                    magnitude[-8:].ravel(),
                    magnitude[:, :8].ravel(),
                    magnitude[:, -8:].ravel(),
                ]
            )
            interior = magnitude[8:-8, 8:-8]
            ratio = float(np.mean(border) / max(float(np.mean(interior)), 1e-8))
            boundary_values.append(abs(np.log(max(ratio, 1e-6))))
        boundary_penalties.append(float(np.median(boundary_values)))
    psd = float(np.median(psd_errors))
    gradient = float(np.median(gradient_relative_errors))
    boundary = float(np.median(boundary_penalties))
    return {
        "val_fft_log_mae": psd,
        "val_gradient_relative_error": gradient,
        "val_boundary_log_penalty": boundary,
        "val_selection_score": psd + 0.75 * gradient + 0.25 * boundary,
    }


def train_adversarial_refiner(
    *,
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    condition_scaler: ConditionScaler,
    spectral_model: ConditionalSpectralModel,
    validation_predicted_conditions: dict[str, np.ndarray],
    config: dict[str, Any],
    output_dir: str | Path,
    smoke: bool,
    device_name: str,
) -> AdversarialTrainingResult:
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(device_name)
    output = repo_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best_refiner.pt"
    steps = int(config["gan_smoke_steps"] if smoke else config["gan_steps"])
    dataset = _training_tensors(
        train_rows=train_rows,
        condition_scaler=condition_scaler,
        spectral_model=spectral_model,
        resolution=int(config["resolution"]),
        seeds_per_group=int(
            config[
                "gan_smoke_seeds_per_group"
                if smoke
                else "gan_seeds_per_group"
            ]
        ),
        seed=seed,
        iaaft_iterations=int(config["spectral_iaaft_iterations"]),
    )
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["gan_batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=loader_generator,
    )
    generator = PhysicsSeededAFMRefiner(
        len(condition_scaler.columns),
        channels=int(config["gan_generator_channels"]),
        residual_scale=float(config["gan_residual_scale"]),
    ).to(device)
    discriminator = ProjectionDiscriminator(
        len(condition_scaler.columns),
        base_channels=int(config["gan_discriminator_channels"]),
    ).to(device)
    initialize_orthogonal(generator)
    initialize_orthogonal(discriminator)
    torch.nn.init.zeros_(generator.output.convolution.weight)
    torch.nn.init.zeros_(generator.output.convolution.bias)
    ema = deepcopy(generator).eval()
    ema_decay = float(config["gan_ema_decay"])
    morphology_loss = MorphologyConditionLoss(
        condition_scaler, int(config["resolution"])
    ).to(device)
    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=float(config["gan_generator_lr"]),
        betas=(0.0, 0.999),
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(config["gan_discriminator_lr"]),
        betas=(0.0, 0.999),
    )
    iterator = iter(loader)
    history: list[dict[str, float | int]] = []
    best_score = float("inf")
    best_step = 0
    validation_interval = int(config["gan_validation_interval"])
    started = time.perf_counter()
    for step in range(1, steps + 1):
        try:
            real, spectral_seed, condition = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            real, spectral_seed, condition = next(iterator)
        real = real.to(device)
        spectral_seed = spectral_seed.to(device)
        condition = condition.to(device)

        discriminator_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake = generator(spectral_seed, condition)
        real_score = discriminator(diff_augment(real), condition)
        fake_score = discriminator(diff_augment(fake), condition)
        discriminator_loss = (
            torch.relu(1.0 - real_score).mean()
            + torch.relu(1.0 + fake_score).mean()
        )
        discriminator_loss.backward()
        discriminator_optimizer.step()

        generator_optimizer.zero_grad(set_to_none=True)
        fake = generator(spectral_seed, condition)
        fake_score, fake_features = discriminator(
            diff_augment(fake), condition, return_features=True
        )
        with torch.no_grad():
            _, real_features = discriminator(
                diff_augment(real), condition, return_features=True
            )
        adversarial_loss = -fake_score.mean()
        condition_loss = morphology_loss(fake, condition)
        gradient_loss = torch.nn.functional.smooth_l1_loss(
            torch.log1p(gradient_statistics(fake)),
            torch.log1p(gradient_statistics(real)),
        )
        feature_matching = torch.nn.functional.l1_loss(
            fake_features.mean(dim=0), real_features.mean(dim=0)
        )
        residual_loss = torch.mean(torch.abs(fake - spectral_seed))
        generator_loss = (
            adversarial_loss
            + float(config["gan_condition_loss_weight"]) * condition_loss
            + float(config["gan_gradient_loss_weight"]) * gradient_loss
            + float(config["gan_feature_matching_weight"]) * feature_matching
            + float(config["gan_residual_regularizer_weight"]) * residual_loss
        )
        if not torch.isfinite(generator_loss + discriminator_loss):
            raise FloatingPointError(f"non-finite GAN loss at step {step}")
        generator_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
        generator_optimizer.step()
        with torch.no_grad():
            for ema_parameter, parameter in zip(
                ema.parameters(), generator.parameters()
            ):
                ema_parameter.lerp_(parameter, 1.0 - ema_decay)

        record: dict[str, float | int] = {
            "step": step,
            "discriminator_loss": float(discriminator_loss.detach().cpu()),
            "generator_loss": float(generator_loss.detach().cpu()),
            "adversarial_loss": float(adversarial_loss.detach().cpu()),
            "condition_loss": float(condition_loss.detach().cpu()),
            "gradient_loss": float(gradient_loss.detach().cpu()),
            "feature_matching_loss": float(feature_matching.detach().cpu()),
            "residual_l1": float(residual_loss.detach().cpu()),
        }
        if (
            step == 1
            or step % validation_interval == 0
            or step == steps
        ):
            validation = _validation_score(
                generator=ema,
                spectral_model=spectral_model,
                validation_rows=validation_rows,
                predicted_conditions=validation_predicted_conditions,
                device=device,
                seed=seed + 50_000,
                iaaft_iterations=int(config["spectral_iaaft_iterations"]),
            )
            record.update(validation)
            score = float(validation["val_selection_score"])
            if score < best_score:
                best_score = score
                best_step = step
                torch.save(
                    {
                        "generator_state_dict": ema.state_dict(),
                        "config": config,
                        "condition_scaler": condition_scaler.to_dict(),
                        "best_step": best_step,
                        "best_validation_score": best_score,
                        "train_groups": sorted(
                            train_rows["growth_run_id"].astype(str).unique()
                        ),
                        "validation_groups": sorted(
                            validation_rows["growth_run_id"]
                            .astype(str)
                            .unique()
                        ),
                        "architecture": {
                            "generator": "circular physics-seeded conditional residual refiner",
                            "discriminator": "spectral-normalized projection discriminator",
                            "upsampling": False,
                            "retrieval_at_inference": False,
                        },
                    },
                    checkpoint_path,
                )
        history.append(record)
    runtime = time.perf_counter() - started
    history_path = output / "training_history.csv"
    write_csv(pd.DataFrame(history), history_path)
    write_json(
        {
            "steps": steps,
            "best_step": best_step,
            "best_validation_score": best_score,
            "runtime_seconds": runtime,
            "device": str(device),
            "training_example_count_after_augmentation": len(dataset),
        },
        output / "training_summary.json",
    )
    return AdversarialTrainingResult(
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        best_step=best_step,
        best_validation_score=best_score,
        runtime_seconds=runtime,
    )


def load_refiner(
    checkpoint_path: str | Path, device_name: str
) -> tuple[PhysicsSeededAFMRefiner, dict[str, Any], torch.device]:
    device = resolve_device(device_name)
    payload = torch.load(
        repo_path(checkpoint_path), map_location=device, weights_only=False
    )
    config = payload["config"]
    scaler = payload["condition_scaler"]
    model = PhysicsSeededAFMRefiner(
        len(scaler["columns"]),
        channels=int(config["gan_generator_channels"]),
        residual_scale=float(config["gan_residual_scale"]),
    ).to(device)
    model.load_state_dict(payload["generator_state_dict"])
    model.eval()
    return model, payload, device
