from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

# These are fixed, interpretable RHEED morphology/temporal features.  They
# describe spot roundness/area/count, streak connectivity, skeleton structure,
# anisotropy, diffuse intensity and time evolution.  No AFM target-dependent
# feature search is performed inside the experiment runner.
CURATED_RHEED_FEATURES = [
    "keyframe_1__round_component_fraction_p97_median",
    "keyframe_1__component_eccentricity_mean_p90_median",
    "keyframe_1__component_area_median_p95_median",
    "keyframe_1__component_count_p97_median",
    "keyframe_1__largest_component_area_fraction_p97_median",
    "causal_8__round_component_fraction_p90_first_last_diff",
    "causal_8__component_solidity_mean_p90_slope",
    "selected_16__component_solidity_mean_p90_slope",
    "selected_16__skeleton_branch_point_density_iqr",
    "keyframe_1__skeleton_endpoint_density_median",
    "selected_16__vertical_percolation_fraction_std",
    "selected_16__structure_tensor_coherence_first_last_diff",
    "selected_16__fft_horizontal_vertical_anisotropy_first_last_diff",
    "selected_16__diffuse_to_peak_intensity_ratio_iqr",
    "temporal_brightness_drift",
]

DYNAMIC_NUCLEATION_FEATURES = ["causal_8__component_area_median_p97_first_last_diff"]


def _ridge(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )


@dataclass
class AmplitudeHead:
    morphology_model: Pipeline
    dynamics_model: Pipeline
    morphology_weight: float
    dynamics_weight: float
    target_name: str
    train_groups: list[str]

    def predict_parts(
        self, physics: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        morphology = np.ravel(
            self.morphology_model.predict(physics[CURATED_RHEED_FEATURES])
        )
        dynamics = np.ravel(
            self.dynamics_model.predict(physics[DYNAMIC_NUCLEATION_FEATURES])
        )
        combined = self.morphology_weight * morphology + self.dynamics_weight * dynamics
        return morphology, dynamics, combined


def fit_amplitude_head(
    physics: pd.DataFrame,
    log_target: pd.Series,
    groups: Iterable[str],
    *,
    alpha: float,
    morphology_weight: float,
) -> AmplitudeHead:
    train_groups = list(map(str, groups))
    frame = physics.loc[train_groups]
    target = log_target.loc[train_groups].to_numpy(float)
    morphology = _ridge(alpha).fit(frame[CURATED_RHEED_FEATURES], target)
    dynamics = _ridge(alpha).fit(frame[DYNAMIC_NUCLEATION_FEATURES], target)
    return AmplitudeHead(
        morphology_model=morphology,
        dynamics_model=dynamics,
        morphology_weight=float(morphology_weight),
        dynamics_weight=float(1.0 - morphology_weight),
        target_name=str(log_target.name or "log_target"),
        train_groups=train_groups,
    )


def _query_diagnostics(
    model: AmplitudeHead,
    train_physics: pd.DataFrame,
    query_physics: pd.DataFrame,
    train_target: pd.Series,
) -> tuple[float, dict[str, float]]:
    morphology, dynamics, combined = model.predict_parts(query_physics)
    imputer = model.morphology_model["imputer"]
    scaler = model.morphology_model["scaler"]
    train_x = scaler.transform(imputer.transform(train_physics[CURATED_RHEED_FEATURES]))
    query_x = scaler.transform(imputer.transform(query_physics[CURATED_RHEED_FEATURES]))
    nearest = float(np.min(np.sqrt(np.mean(np.square(train_x - query_x[0]), axis=1))))
    train_m, train_d, train_combined = model.predict_parts(train_physics)
    in_sample_residual = float(
        np.median(
            np.abs(
                train_combined - train_target.loc[model.train_groups].to_numpy(float)
            )
        )
    )
    prediction = float(combined[0])
    return prediction, {
        "predicted_log_target": prediction,
        "head_disagreement_log": float(abs(morphology[0] - dynamics[0])),
        "nearest_training_rheed_distance": nearest,
        "maximum_absolute_robust_z": float(np.max(np.abs(query_x[0]))),
        "training_fit_median_log_residual": in_sample_residual,
    }


def _fit_expected_error(
    diagnostics: np.ndarray,
    errors: np.ndarray,
    query: np.ndarray,
    *,
    floor: float,
) -> float:
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "huber",
                HuberRegressor(alpha=0.1, epsilon=1.35, max_iter=2000),
            ),
        ]
    )
    try:
        model.fit(diagnostics, np.log(errors + float(floor)))
        return float(
            max(
                np.exp(model.predict(query[None])[0]) - float(floor),
                0.0,
            )
        )
    except Exception:
        return float(np.median(errors))


def _higher_quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, float(quantile), method="higher"))


def _range_calibrate(
    reference_predictions: np.ndarray,
    reference_truth: np.ndarray,
    query_prediction: float,
    *,
    scale_cap: float = 1.20,
    blend: float = 0.75,
) -> tuple[float, float]:
    """Expand a compressed positive range using honest reference predictions."""

    predicted_log = np.log(
        np.clip(np.asarray(reference_predictions, dtype=float), 1e-8, None)
    )
    true_log = np.log(np.clip(np.asarray(reference_truth, dtype=float), 1e-8, None))
    query_log = float(np.log(max(float(query_prediction), 1e-8)))
    scale = float(
        np.clip(
            np.std(true_log) / max(float(np.std(predicted_log)), 1e-8),
            1.0 / float(scale_cap),
            float(scale_cap),
        )
    )
    matched = (query_log - float(np.mean(predicted_log))) * scale + float(
        np.mean(true_log)
    )
    calibrated_log = (1.0 - float(blend)) * query_log + float(blend) * matched
    return float(np.exp(calibrated_log)), scale


def crossfit_target(
    *,
    physics: pd.DataFrame,
    log_target: pd.Series,
    alpha: float = 1.0,
    morphology_weight: float = 0.75,
    confidence_alpha: float = 0.10,
    error_floor: float = 0.08,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict leave-one-growth-out predictions and nested error estimates."""

    groups = list(map(str, log_target.index))
    prediction_records: list[dict[str, float | str]] = []
    inner_records: list[dict[str, float | str]] = []
    for held in groups:
        fit_groups = [group for group in groups if group != held]
        model = fit_amplitude_head(
            physics,
            log_target,
            fit_groups,
            alpha=alpha,
            morphology_weight=morphology_weight,
        )
        prediction, diagnostics = _query_diagnostics(
            model,
            physics.loc[fit_groups],
            physics.loc[[held]],
            log_target,
        )
        inner_diagnostics: list[list[float]] = []
        inner_true: list[float] = []
        inner_raw_predictions: list[float] = []
        fold_inner_records: list[dict[str, float | str]] = []
        for inner_held in fit_groups:
            inner_fit = [group for group in fit_groups if group != inner_held]
            inner_model = fit_amplitude_head(
                physics,
                log_target,
                inner_fit,
                alpha=alpha,
                morphology_weight=morphology_weight,
            )
            inner_prediction, inner_diag = _query_diagnostics(
                inner_model,
                physics.loc[inner_fit],
                physics.loc[[inner_held]],
                log_target,
            )
            true_value = float(np.exp(log_target.loc[inner_held]))
            predicted_value = float(np.exp(inner_prediction))
            inner_true.append(true_value)
            inner_raw_predictions.append(predicted_value)
            inner_diagnostics.append(list(inner_diag.values()))
            fold_inner_records.append(
                {
                    "outer_held_growth_group": held,
                    "inner_held_growth_group": inner_held,
                    "true_target": true_value,
                    "raw_predicted_target": predicted_value,
                    **inner_diag,
                }
            )
        inner_true_array = np.asarray(inner_true, dtype=float)
        inner_raw_array = np.asarray(inner_raw_predictions, dtype=float)
        inner_calibrated = np.zeros_like(inner_raw_array)
        for index in range(len(inner_raw_array)):
            keep = np.arange(len(inner_raw_array)) != index
            inner_calibrated[index], _ = _range_calibrate(
                inner_raw_array[keep],
                inner_true_array[keep],
                inner_raw_array[index],
            )
        inner_error_array = np.abs(inner_calibrated - inner_true_array)
        for record, calibrated, error in zip(
            fold_inner_records, inner_calibrated, inner_error_array, strict=False
        ):
            record["predicted_target"] = float(calibrated)
            record["absolute_error"] = float(error)
            inner_records.append(record)
        expected_error = _fit_expected_error(
            np.asarray(inner_diagnostics, dtype=float),
            inner_error_array,
            np.asarray(list(diagnostics.values()), dtype=float),
            floor=error_floor,
        )
        radius = _higher_quantile(inner_error_array, 1.0 - confidence_alpha)
        true_value = float(np.exp(log_target.loc[held]))
        raw_predicted_value = float(np.exp(prediction))
        predicted_value, calibration_scale = _range_calibrate(
            inner_raw_array,
            inner_true_array,
            raw_predicted_value,
        )
        prediction_records.append(
            {
                "growth_run_id": held,
                "true_target": true_value,
                "raw_predicted_target": raw_predicted_value,
                "predicted_target": predicted_value,
                "range_calibration_log_scale": calibration_scale,
                "absolute_error": abs(predicted_value - true_value),
                "predicted_absolute_error": expected_error,
                "interval_lower": max(predicted_value - radius, 0.0),
                "interval_upper": predicted_value + radius,
                "interval_radius": radius,
                "interval_covered": (
                    max(predicted_value - radius, 0.0)
                    <= true_value
                    <= predicted_value + radius
                ),
                "outer_target_used_for_training": False,
                "inner_group_count": len(fit_groups),
                **diagnostics,
            }
        )
    return pd.DataFrame(prediction_records), pd.DataFrame(inner_records)


def predict_external_groups(
    *,
    physics: pd.DataFrame,
    log_target: pd.Series,
    query_groups: list[str],
    crossfit_predictions: pd.DataFrame,
    alpha: float = 1.0,
    morphology_weight: float = 0.75,
    confidence_alpha: float = 0.10,
    error_floor: float = 0.08,
) -> pd.DataFrame:
    """Fit all development groups and predict a pre-existing validation set."""

    train_groups = list(map(str, log_target.index))
    model = fit_amplitude_head(
        physics,
        log_target,
        train_groups,
        alpha=alpha,
        morphology_weight=morphology_weight,
    )
    inner_errors = crossfit_predictions["absolute_error"].to_numpy(float)
    crossfit_truth = crossfit_predictions["true_target"].to_numpy(float)
    crossfit_prediction = crossfit_predictions["predicted_target"].to_numpy(float)
    inner_diag = crossfit_predictions[
        [
            "predicted_log_target",
            "head_disagreement_log",
            "nearest_training_rheed_distance",
            "maximum_absolute_robust_z",
            "training_fit_median_log_residual",
        ]
    ].to_numpy(float)
    radius = _higher_quantile(inner_errors, 1.0 - confidence_alpha)
    records = []
    for group in map(str, query_groups):
        prediction, diagnostics = _query_diagnostics(
            model,
            physics.loc[train_groups],
            physics.loc[[group]],
            log_target,
        )
        raw_predicted_value = float(np.exp(prediction))
        predicted_value, calibration_scale = _range_calibrate(
            crossfit_prediction,
            crossfit_truth,
            raw_predicted_value,
        )
        expected_error = _fit_expected_error(
            inner_diag,
            inner_errors,
            np.asarray(list(diagnostics.values()), dtype=float),
            floor=error_floor,
        )
        records.append(
            {
                "growth_run_id": group,
                "raw_predicted_target": raw_predicted_value,
                "predicted_target": predicted_value,
                "range_calibration_log_scale": calibration_scale,
                "predicted_absolute_error": expected_error,
                "interval_lower": max(predicted_value - radius, 0.0),
                "interval_upper": predicted_value + radius,
                "interval_radius": radius,
                **diagnostics,
            }
        )
    return pd.DataFrame(records)
