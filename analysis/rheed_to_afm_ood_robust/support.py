from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from analysis.rheed_to_afm_functional_morphology.amplitude import (
    CURATED_RHEED_FEATURES,
)


@dataclass(frozen=True)
class SupportDiagnostics:
    nearest_distance: float
    knn_distance: float
    maximum_absolute_robust_z: float
    mahalanobis_distance: float
    density_ood_z: float
    support_confidence: float


def _scaled(
    physics: pd.DataFrame,
    fit_groups: Iterable[str],
    query_groups: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    fit = list(map(str, fit_groups))
    query = list(map(str, query_groups))
    imputer = SimpleImputer(strategy="median").fit(
        physics.loc[fit, CURATED_RHEED_FEATURES]
    )
    scaler = RobustScaler().fit(
        imputer.transform(physics.loc[fit, CURATED_RHEED_FEATURES])
    )
    fit_x = scaler.transform(
        imputer.transform(physics.loc[fit, CURATED_RHEED_FEATURES])
    )
    query_x = scaler.transform(
        imputer.transform(physics.loc[query, CURATED_RHEED_FEATURES])
    )
    return np.asarray(fit_x, dtype=float), np.asarray(query_x, dtype=float)


def _rms_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(
        np.mean(
            np.square(left[:, None, :] - right[None, :, :]),
            axis=2,
        )
    )


def _self_knn_distances(values: np.ndarray, *, neighbors: int) -> np.ndarray:
    count = len(values)
    if count < 3:
        raise ValueError("at least three groups are required for support")
    distances = _rms_distances(values, values)
    distances[np.arange(count), np.arange(count)] = np.inf
    use = min(int(neighbors), count - 1)
    return np.sort(distances, axis=1)[:, :use].mean(axis=1)


def _robust_z(value: float, reference: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    median = float(np.median(reference))
    mad = float(1.4826 * np.median(np.abs(reference - median)))
    floor = max(0.10 * max(abs(median), 1.0), 1e-6)
    return float((float(value) - median) / max(mad, floor))


def query_support(
    physics: pd.DataFrame,
    fit_groups: Iterable[str],
    query_group: str,
    *,
    neighbors: int = 3,
) -> SupportDiagnostics:
    fit = list(map(str, fit_groups))
    query = str(query_group)
    fit_x, query_x = _scaled(physics, fit, [query])
    distances = _rms_distances(query_x, fit_x)[0]
    use = min(int(neighbors), len(fit))
    knn_distance = float(np.sort(distances)[:use].mean())
    self_knn = _self_knn_distances(fit_x, neighbors=neighbors)
    density_z = _robust_z(knn_distance, self_knn)
    covariance = LedoitWolf().fit(fit_x)
    mahalanobis = float(
        np.sqrt(max(float(covariance.mahalanobis(query_x)[0]), 0.0))
    )
    return SupportDiagnostics(
        nearest_distance=float(np.min(distances)),
        knn_distance=knn_distance,
        maximum_absolute_robust_z=float(np.max(np.abs(query_x[0]))),
        mahalanobis_distance=mahalanobis,
        density_ood_z=density_z,
        support_confidence=float(np.exp(-max(density_z, 0.0))),
    )


def density_weights(
    physics: pd.DataFrame,
    groups: Iterable[str],
    *,
    neighbors: int = 3,
    strength: float = 0.5,
    floor: float = 0.25,
) -> pd.DataFrame:
    group_list = list(map(str, groups))
    values, _ = _scaled(physics, group_list, group_list[:1])
    distances = _self_knn_distances(values, neighbors=neighbors)
    median = float(np.median(distances))
    mad = float(1.4826 * np.median(np.abs(distances - median)))
    scale_floor = max(0.10 * max(abs(median), 1.0), 1e-6)
    z = (distances - median) / max(mad, scale_floor)
    weights = np.maximum(
        float(floor),
        np.exp(-float(strength) * np.maximum(z, 0.0)),
    )
    return pd.DataFrame(
        {
            "growth_run_id": group_list,
            "training_knn_distance": distances,
            "training_density_ood_z": z,
            "density_sample_weight": weights,
        }
    )


def leave_one_out_support_audit(
    physics: pd.DataFrame,
    groups: Iterable[str],
    *,
    neighbors: int = 3,
) -> pd.DataFrame:
    """Rank groups using RHEED covariates only; AFM targets are not accepted."""

    group_list = list(map(str, groups))
    records: list[dict[str, float | str]] = []
    for held in group_list:
        fit = [group for group in group_list if group != held]
        diagnostic = query_support(
            physics, fit, held, neighbors=neighbors
        )
        records.append(
            {
                "growth_run_id": held,
                "nearest_rheed_distance": diagnostic.nearest_distance,
                "knn_rheed_distance": diagnostic.knn_distance,
                "maximum_absolute_robust_z": (
                    diagnostic.maximum_absolute_robust_z
                ),
                "ledoit_wolf_mahalanobis": (
                    diagnostic.mahalanobis_distance
                ),
                "density_ood_z": diagnostic.density_ood_z,
                "support_confidence": diagnostic.support_confidence,
            }
        )
    result = pd.DataFrame(records)
    rank_columns = [
        "knn_rheed_distance",
        "maximum_absolute_robust_z",
        "ledoit_wolf_mahalanobis",
    ]
    for column in rank_columns:
        result[f"{column}_percentile_rank"] = result[column].rank(
            pct=True, method="average"
        )
    result["rheed_only_ood_score"] = result[
        [f"{column}_percentile_rank" for column in rank_columns]
    ].mean(axis=1)
    result = result.sort_values(
        ["rheed_only_ood_score", "growth_run_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    result["rheed_only_ood_rank"] = np.arange(1, len(result) + 1)
    for count in (2, 3, 4):
        result[f"excluded_in_top{count}_sensitivity"] = (
            result["rheed_only_ood_rank"] <= count
        )
    result["selection_used_afm_target"] = False
    return result
