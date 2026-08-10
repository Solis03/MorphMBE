from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

from .islands import (
    ISLAND_FEATURE_COLUMNS,
    extract_island_features,
    group_island_feature_table,
    scan_island_feature_table,
)


def _mahalanobis(values: np.ndarray, reference: np.ndarray) -> float:
    covariance = LedoitWolf().fit(reference)
    delta = np.asarray(values, dtype=float) - covariance.location_
    return float(
        np.sqrt(
            np.maximum(
                delta @ covariance.precision_ @ delta,
                0.0,
            )
        )
    )


def _real_group_reference_distances(values: np.ndarray) -> np.ndarray:
    distances = []
    for held in range(len(values)):
        keep = np.arange(len(values)) != held
        distances.append(_mahalanobis(values[held], values[keep]))
    return np.asarray(distances, dtype=float)


def evaluate_island_methods(
    *,
    held_rows: pd.DataFrame,
    train_rows: pd.DataFrame,
    generated: dict[str, list[np.ndarray]],
    resolution: int,
) -> pd.DataFrame:
    """Compare generated object topology with held AFM and real-AFM support."""

    train_groups = group_island_feature_table(train_rows, resolution=resolution)
    held_scans = scan_island_feature_table(held_rows, resolution=resolution)
    held_features = held_scans[ISLAND_FEATURE_COLUMNS].median().to_numpy(float)
    train_values = train_groups[ISLAND_FEATURE_COLUMNS].to_numpy(float)
    scaler = StandardScaler().fit(train_values)
    train_z = scaler.transform(train_values)
    held_z = scaler.transform(held_features[None])[0]
    reference_distances = _real_group_reference_distances(train_z)
    held_real_distance = _mahalanobis(held_z, train_z)
    records: list[dict[str, Any]] = []
    for method, maps in generated.items():
        features = (
            pd.DataFrame([extract_island_features(array) for array in maps])[
                ISLAND_FEATURE_COLUMNS
            ]
            .median()
            .to_numpy(float)
        )
        generated_z = scaler.transform(features[None])[0]
        distance = _mahalanobis(generated_z, train_z)
        percentile = (
            100.0
            * (1.0 + float(np.sum(reference_distances >= distance)))
            / (len(reference_distances) + 1.0)
        )
        held_percentile = (
            100.0
            * (1.0 + float(np.sum(reference_distances >= held_real_distance)))
            / (len(reference_distances) + 1.0)
        )
        record: dict[str, Any] = {
            "growth_run_id": str(held_rows.iloc[0]["growth_run_id"]),
            "method": method,
            "island_feature_mae_z": float(np.mean(np.abs(generated_z - held_z))),
            "island_feature_rmse_z": float(
                np.sqrt(np.mean(np.square(generated_z - held_z)))
            ),
            "afm_prior_mahalanobis": distance,
            "held_real_afm_prior_mahalanobis": held_real_distance,
            "afm_likeness_percentile": percentile,
            "held_real_likeness_percentile": held_percentile,
            "boundary_ratio_q70_absolute_error": abs(
                features[ISLAND_FEATURE_COLUMNS.index("boundary_gradient_ratio_q70")]
                - held_features[
                    ISLAND_FEATURE_COLUMNS.index("boundary_gradient_ratio_q70")
                ]
            ),
            "median_area_q70_log_absolute_error": abs(
                features[ISLAND_FEATURE_COLUMNS.index("log_median_area_q70")]
                - held_features[ISLAND_FEATURE_COLUMNS.index("log_median_area_q70")]
            ),
            "component_count_q70_log_absolute_error": abs(
                features[ISLAND_FEATURE_COLUMNS.index("log_component_count_q70")]
                - held_features[ISLAND_FEATURE_COLUMNS.index("log_component_count_q70")]
            ),
        }
        for index, column in enumerate(ISLAND_FEATURE_COLUMNS):
            record[f"generated__{column}"] = float(features[index])
            record[f"true__{column}"] = float(held_features[index])
        records.append(record)
    return pd.DataFrame(records)
