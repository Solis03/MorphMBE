from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from analysis.rheed_to_afm_generation.data import (
    ConditionScaler,
    aggregate_group_conditions,
)
from analysis.rheed_video_afm_story.afm_descriptors import radial_psd
from analysis.rheed_video_afm_story.common import repo_path
from analysis.rheed_video_afm_story.rq_disentanglement import project_unit_rq_np

QUANTILE_LEVELS = np.linspace(0.01, 0.99, 33, dtype=np.float64)
PSD_BINS = 24


def load_unit_map(row: pd.Series, resolution: int = 128) -> np.ndarray:
    paths = json.loads(str(row["unit_shape_paths"]))
    array = np.load(repo_path(paths[str(resolution)]), allow_pickle=False)
    return project_unit_rq_np(np.asarray(array, dtype=np.float32))


def shape_parameters(array: np.ndarray) -> np.ndarray:
    _, power = radial_psd(project_unit_rq_np(array), bins=PSD_BINS)
    normalized = np.clip(power / max(float(power.sum()), 1e-12), 1e-9, None)
    log_power = np.log(normalized)
    quantiles = np.quantile(array, QUANTILE_LEVELS)
    return np.concatenate([log_power, quantiles]).astype(np.float64)


def group_shape_parameter_table(
    rows: pd.DataFrame,
    *,
    resolution: int,
) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for group_id, group_rows in rows.groupby("growth_run_id"):
        values = np.stack(
            [
                shape_parameters(load_unit_map(row, resolution))
                for _, row in group_rows.iterrows()
            ]
        )
        median = np.median(values, axis=0)
        record: dict[str, float | str] = {"growth_run_id": str(group_id)}
        for index, value in enumerate(median[:PSD_BINS]):
            record[f"log_psd_{index:02d}"] = float(value)
        for index, value in enumerate(median[PSD_BINS:]):
            record[f"quantile_{index:02d}"] = float(value)
        records.append(record)
    return pd.DataFrame(records).set_index("growth_run_id").sort_index()


@dataclass
class ConditionalSpectralModel:
    """Learned conditional random-field model with no AFM retrieval at inference."""

    condition_scaler: ConditionScaler
    output_scaler: StandardScaler
    ridge: Ridge
    alpha: float
    train_groups: list[str]
    resolution: int
    removelist_sample_ids: list[str]

    def predict_parameters(self, standardized_condition: np.ndarray) -> np.ndarray:
        condition = np.asarray(standardized_condition, dtype=np.float64)
        if condition.ndim == 1:
            condition = condition[None]
        scaled = self.ridge.predict(condition)
        return self.output_scaler.inverse_transform(scaled)

    def generate(
        self,
        standardized_condition: np.ndarray,
        *,
        seed: int,
        iterations: int = 40,
    ) -> np.ndarray:
        parameters = self.predict_parameters(standardized_condition)[0]
        return synthesize_random_field(
            parameters,
            resolution=self.resolution,
            seed=seed,
            iterations=iterations,
        )


def _loo_alpha(
    conditions: np.ndarray,
    targets: np.ndarray,
    alphas: Iterable[float],
) -> tuple[float, pd.DataFrame]:
    records: list[dict[str, float]] = []
    for alpha in alphas:
        predictions = np.zeros_like(targets)
        for held in range(len(conditions)):
            keep = np.arange(len(conditions)) != held
            model = Ridge(alpha=float(alpha)).fit(conditions[keep], targets[keep])
            predictions[held] = model.predict(conditions[held : held + 1])[0]
        psd_error = float(
            np.mean(np.abs(predictions[:, :PSD_BINS] - targets[:, :PSD_BINS]))
        )
        quantile_error = float(
            np.mean(np.abs(predictions[:, PSD_BINS:] - targets[:, PSD_BINS:]))
        )
        records.append(
            {
                "alpha": float(alpha),
                "loo_scaled_psd_mae": psd_error,
                "loo_scaled_quantile_mae": quantile_error,
                "selection_score": 0.65 * psd_error + 0.35 * quantile_error,
            }
        )
    table = pd.DataFrame(records).sort_values(
        ["selection_score", "loo_scaled_psd_mae", "alpha"]
    )
    return float(table.iloc[0]["alpha"]), table


def fit_conditional_spectral_model(
    *,
    train_rows: pd.DataFrame,
    condition_scaler: ConditionScaler,
    alphas: Iterable[float],
    resolution: int,
    removelist_sample_ids: Iterable[str],
) -> tuple[ConditionalSpectralModel, pd.DataFrame, pd.DataFrame]:
    group_parameters = group_shape_parameter_table(train_rows, resolution=resolution)
    groups = list(group_parameters.index.astype(str))
    group_conditions = (
        aggregate_group_conditions(
            train_rows,
            condition_scaler.columns,
        )
        .loc[groups]
        .to_numpy(float)
    )
    standardized_conditions = condition_scaler.transform(
        group_conditions, clip=False
    ).astype(np.float64)
    output_scaler = StandardScaler().fit(group_parameters.to_numpy(float))
    targets = output_scaler.transform(group_parameters.to_numpy(float))
    alpha, cv = _loo_alpha(standardized_conditions, targets, alphas)
    ridge = Ridge(alpha=alpha).fit(standardized_conditions, targets)
    model = ConditionalSpectralModel(
        condition_scaler=condition_scaler,
        output_scaler=output_scaler,
        ridge=ridge,
        alpha=alpha,
        train_groups=groups,
        resolution=int(resolution),
        removelist_sample_ids=sorted(map(str, removelist_sample_ids)),
    )
    return model, cv, group_parameters


def _target_amplitude(log_power: np.ndarray, resolution: int) -> np.ndarray:
    power = np.exp(np.clip(np.asarray(log_power, dtype=float), -25.0, 10.0))
    power /= max(float(power.sum()), 1e-12)
    frequency = np.fft.fftfreq(resolution) * resolution
    yy, xx = np.meshgrid(frequency, frequency, indexing="ij")
    radius = np.hypot(yy, xx)
    edges = np.linspace(1.0, np.sqrt(2.0) * resolution / 2.0, PSD_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    radial_power = np.interp(
        radius.ravel(),
        centers,
        power,
        left=power[0],
        right=power[-1],
    ).reshape(radius.shape)
    radial_power[radius < 0.5] = 0.0
    return np.sqrt(np.clip(radial_power, 0.0, None))


def _target_values(quantiles: np.ndarray, resolution: int) -> np.ndarray:
    q = np.maximum.accumulate(np.asarray(quantiles, dtype=float))
    ranks = (np.arange(resolution * resolution, dtype=float) + 0.5) / (
        resolution * resolution
    )
    values = np.interp(ranks, QUANTILE_LEVELS, q, left=q[0], right=q[-1])
    values -= float(values.mean())
    scale = float(np.sqrt(np.mean(values**2)))
    return values / max(scale, 1e-8)


def synthesize_random_field(
    parameters: np.ndarray,
    *,
    resolution: int,
    seed: int,
    iterations: int = 40,
) -> np.ndarray:
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (PSD_BINS + len(QUANTILE_LEVELS),):
        raise ValueError(f"unexpected shape-parameter vector: {parameters.shape}")
    amplitude = _target_amplitude(parameters[:PSD_BINS], resolution)
    sorted_values = _target_values(parameters[PSD_BINS:], resolution)
    rng = np.random.default_rng(int(seed))
    field = rng.normal(size=(resolution, resolution))
    for _ in range(max(int(iterations), 1)):
        spectrum = np.fft.fft2(field)
        phase = spectrum / np.maximum(np.abs(spectrum), 1e-12)
        field = np.fft.ifft2(amplitude * phase).real
        order = np.argsort(field, axis=None)
        ranked = np.empty_like(field.ravel())
        ranked[order] = sorted_values
        field = ranked.reshape(field.shape)
    spectrum = np.fft.fft2(field)
    phase = spectrum / np.maximum(np.abs(spectrum), 1e-12)
    field = np.fft.ifft2(amplitude * phase).real
    order = np.argsort(field, axis=None)
    ranked = np.empty_like(field.ravel())
    ranked[order] = sorted_values
    return project_unit_rq_np(ranked.reshape(field.shape)).astype(np.float32)


def save_spectral_model(model: ConditionalSpectralModel, path: str | Path) -> None:
    target = repo_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)


def load_spectral_model(path: str | Path) -> ConditionalSpectralModel:
    return joblib.load(repo_path(path))
