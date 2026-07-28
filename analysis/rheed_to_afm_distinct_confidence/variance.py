from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.rheed_to_afm_generation.data import ConditionScaler, predict_groups


FitPredictor = Callable[[list[str], ConditionScaler], object]


@dataclass(frozen=True)
class VarianceCalibrator:
    """Expand cross-fitted condition variance without using the query target.

    Small-data multivariate regressors tend to shrink predictions toward the
    training mean.  The factors are estimated from inner leave-one-growth-group
    predictions only and are capped before they are applied to a query.
    """

    factors: np.ndarray
    cap: float
    minimum_predicted_std: float
    columns: list[str]

    def transform_z(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape[-1] != len(self.columns):
            raise ValueError("condition width does not match calibrator")
        return (array * self.factors).astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "cap": float(self.cap),
            "minimum_predicted_std": float(self.minimum_predicted_std),
            "columns": list(self.columns),
            "factors": {
                column: float(value)
                for column, value in zip(self.columns, self.factors)
            },
            "fit_policy": (
                "truth SD / inner-LOO prediction SD in the enclosing training "
                "scaler; clipped to [1, cap], no query target"
            ),
        }


def fit_variance_calibrator(
    *,
    groups: list[str],
    group_targets: pd.DataFrame,
    enclosing_scaler: ConditionScaler,
    fit_predictor: FitPredictor,
    registry: pd.DataFrame,
    physics: pd.DataFrame,
    cap: float,
    minimum_predicted_std: float = 0.15,
) -> tuple[VarianceCalibrator, pd.DataFrame]:
    """Fit a strictly inner-LOO variance calibrator."""

    groups = list(map(str, groups))
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    records: list[dict[str, float | str]] = []
    for held in groups:
        fit_groups = [group for group in groups if group != held]
        inner_scaler = ConditionScaler.fit(
            group_targets.reset_index(),
            enclosing_scaler.columns,
            set(fit_groups),
        )
        predictor = fit_predictor(fit_groups, inner_scaler)
        raw, _, _ = predict_groups(predictor, [held], registry, physics)
        predicted_z = enclosing_scaler.transform(raw, clip=False)[0]
        true_raw = group_targets.loc[held, enclosing_scaler.columns].to_numpy(
            float
        )
        true_z = enclosing_scaler.transform(true_raw[None], clip=False)[0]
        predictions.append(predicted_z)
        truths.append(true_z)
        record: dict[str, float | str] = {"growth_run_id": held}
        for position, column in enumerate(enclosing_scaler.columns):
            record[f"true__{column}"] = float(true_z[position])
            record[f"predicted__{column}"] = float(predicted_z[position])
        records.append(record)
    predicted = np.stack(predictions)
    truth = np.stack(truths)
    predicted_std = np.std(predicted, axis=0, ddof=0)
    truth_std = np.std(truth, axis=0, ddof=0)
    factors = np.clip(
        truth_std / np.maximum(predicted_std, float(minimum_predicted_std)),
        1.0,
        float(cap),
    )
    calibrator = VarianceCalibrator(
        factors=factors.astype(np.float32),
        cap=float(cap),
        minimum_predicted_std=float(minimum_predicted_std),
        columns=list(enclosing_scaler.columns),
    )
    return calibrator, pd.DataFrame(records)


def condition_prediction_metrics(
    *,
    truth_z: np.ndarray,
    predicted_z: np.ndarray,
    truth_raw: np.ndarray,
    predicted_raw: np.ndarray,
    columns: list[str],
) -> dict[str, float]:
    truth_z = np.asarray(truth_z, dtype=float)
    predicted_z = np.asarray(predicted_z, dtype=float)
    truth_raw = np.asarray(truth_raw, dtype=float)
    predicted_raw = np.asarray(predicted_raw, dtype=float)
    rq_index = columns.index("log_rq_nm")
    correlations = []
    for position in range(len(columns)):
        value = spearmanr(
            truth_z[:, position], predicted_z[:, position]
        ).statistic
        correlations.append(float(value) if np.isfinite(value) else 0.0)
    pairwise = [
        float(np.mean(np.abs(predicted_z[first] - predicted_z[second])))
        for first in range(len(predicted_z))
        for second in range(first)
    ]
    rq_error = np.abs(
        np.exp(truth_raw[:, rq_index]) - np.exp(predicted_raw[:, rq_index])
    )
    return {
        "mean_descriptor_mae_z": float(
            np.mean(np.abs(predicted_z - truth_z))
        ),
        "median_descriptor_mae_z": float(
            np.median(np.mean(np.abs(predicted_z - truth_z), axis=1))
        ),
        "mean_rq_mae_nm": float(np.mean(rq_error)),
        "median_rq_mae_nm": float(np.median(rq_error)),
        "raw_rq_spearman": float(
            spearmanr(
                truth_raw[:, rq_index], predicted_raw[:, rq_index]
            ).statistic
        ),
        "median_shape_spearman": float(
            np.median(
                [
                    value
                    for position, value in enumerate(correlations)
                    if position != rq_index
                ]
            )
        ),
        "condition_sensitivity_z": float(np.median(pairwise)),
    }
