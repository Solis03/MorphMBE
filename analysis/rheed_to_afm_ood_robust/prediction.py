from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler, StandardScaler

from analysis.rheed_to_afm_functional_morphology.amplitude import (
    CURATED_RHEED_FEATURES,
    DYNAMIC_NUCLEATION_FEATURES,
    _higher_quantile,
    _range_calibrate,
)

from .support import density_weights, query_support

BASELINE = "M12a_frozen_alpha1"
REGULARIZED = "M14a_regularized_alpha10"
DENSITY_WEIGHTED = "M14b_rheed_density_weighted"
RESIDUAL_WEIGHTED = "M14c_residual_self_paced"
R3D_TEMPORAL = "M14d_r3d_causal_temporal"
MULTIVIEW_20 = "M14e_multiview_curated20_r3d80"
MULTIVIEW_40 = "M14f_multiview_curated40_r3d60"
MULTIVIEW_60 = "M14g_multiview_curated60_r3d40"
NESTED_SELECTOR = "M14h_nested_support_aware_selector"
FINAL_TARGET_SPECIFIC = "M14i_target_specific_robust"

FIXED_METHODS = [
    BASELINE,
    REGULARIZED,
    DENSITY_WEIGHTED,
    RESIDUAL_WEIGHTED,
    R3D_TEMPORAL,
    MULTIVIEW_20,
    MULTIVIEW_40,
    MULTIVIEW_60,
]

CORE_DISAGREEMENT_METHODS = [
    REGULARIZED,
    DENSITY_WEIGHTED,
    R3D_TEMPORAL,
    MULTIVIEW_20,
    MULTIVIEW_40,
    MULTIVIEW_60,
]


@dataclass(frozen=True)
class CandidateConfig:
    density_strength: float = 0.5
    density_floor: float = 0.25
    residual_strength: float = 1.0
    residual_floor: float = 0.25
    r3d_pca_components: int = 5
    ridge_alpha: float = 10.0
    baseline_alpha: float = 1.0
    morphology_weight: float = 0.75


def _preprocess(
    frame: pd.DataFrame,
    fit_groups: list[str],
    query_groups: list[str],
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median").fit(frame.loc[fit_groups, columns])
    scaler = RobustScaler().fit(imputer.transform(frame.loc[fit_groups, columns]))
    fit_x = scaler.transform(imputer.transform(frame.loc[fit_groups, columns]))
    query_x = scaler.transform(imputer.transform(frame.loc[query_groups, columns]))
    return np.asarray(fit_x, dtype=float), np.asarray(query_x, dtype=float)


def _ridge_prediction(
    *,
    physics: pd.DataFrame,
    log_target: pd.Series,
    fit_groups: list[str],
    query_group: str,
    alpha: float,
    morphology_weight: float,
    sample_weights: np.ndarray | None,
) -> float:
    target = log_target.loc[fit_groups].to_numpy(float)
    morphology_x, morphology_query = _preprocess(
        physics,
        fit_groups,
        [query_group],
        CURATED_RHEED_FEATURES,
    )
    morphology = Ridge(alpha=float(alpha)).fit(
        morphology_x, target, sample_weight=sample_weights
    )
    morphology_prediction = float(morphology.predict(morphology_query)[0])
    dynamics_x, dynamics_query = _preprocess(
        physics,
        fit_groups,
        [query_group],
        DYNAMIC_NUCLEATION_FEATURES,
    )
    dynamics = Ridge(alpha=float(alpha)).fit(
        dynamics_x, target, sample_weight=sample_weights
    )
    dynamics_prediction = float(dynamics.predict(dynamics_query)[0])
    weight = float(morphology_weight)
    return weight * morphology_prediction + (1.0 - weight) * dynamics_prediction


def _residual_sample_weights(
    *,
    physics: pd.DataFrame,
    log_target: pd.Series,
    groups: list[str],
    alpha: float,
    strength: float,
    floor: float,
) -> np.ndarray:
    residuals = []
    for held in groups:
        fit = [group for group in groups if group != held]
        prediction = _ridge_prediction(
            physics=physics,
            log_target=log_target,
            fit_groups=fit,
            query_group=held,
            alpha=alpha,
            morphology_weight=1.0,
            sample_weights=None,
        )
        residuals.append(abs(prediction - float(log_target.loc[held])))
    values = np.asarray(residuals, dtype=float)
    median = float(np.median(values))
    mad = float(1.4826 * np.median(np.abs(values - median)))
    z = (values - median) / max(mad, 0.05)
    return np.maximum(
        float(floor),
        np.exp(-float(strength) * np.maximum(z, 0.0)),
    )


def training_weight_audit(
    *,
    physics: pd.DataFrame,
    log_target: pd.Series,
    config: CandidateConfig | None = None,
) -> pd.DataFrame:
    settings = config or CandidateConfig()
    groups = list(map(str, log_target.index))
    density = density_weights(
        physics,
        groups,
        strength=settings.density_strength,
        floor=settings.density_floor,
    )
    residual = _residual_sample_weights(
        physics=physics,
        log_target=log_target,
        groups=groups,
        alpha=settings.ridge_alpha,
        strength=settings.residual_strength,
        floor=settings.residual_floor,
    )
    density["residual_self_paced_weight"] = residual
    density["combined_weight_not_selected"] = (
        density["density_sample_weight"] * residual
    )
    density["weight_computation_used_held_target"] = False
    return density


def _r3d_prediction(
    *,
    embeddings: pd.DataFrame,
    log_target: pd.Series,
    fit_groups: list[str],
    query_group: str,
    alpha: float,
    components: int,
    sample_weights: np.ndarray | None = None,
) -> float:
    fit_values = embeddings.loc[fit_groups].to_numpy(float)
    query_values = embeddings.loc[[query_group]].to_numpy(float)
    scaler = StandardScaler().fit(fit_values)
    fit_scaled = scaler.transform(fit_values)
    query_scaled = scaler.transform(query_values)
    count = min(int(components), len(fit_groups) - 2, fit_scaled.shape[1])
    pca = PCA(n_components=count, svd_solver="full").fit(fit_scaled)
    model = Ridge(alpha=float(alpha)).fit(
        pca.transform(fit_scaled),
        log_target.loc[fit_groups].to_numpy(float),
        sample_weight=sample_weights,
    )
    return float(model.predict(pca.transform(query_scaled))[0])


def _r3d_support(
    *,
    embeddings: pd.DataFrame,
    fit_groups: list[str],
    query_group: str,
    components: int,
) -> dict[str, float]:
    fit_values = embeddings.loc[fit_groups].to_numpy(float)
    query_values = embeddings.loc[[query_group]].to_numpy(float)
    scaler = StandardScaler().fit(fit_values)
    fit_scaled = scaler.transform(fit_values)
    query_scaled = scaler.transform(query_values)
    count = min(int(components), len(fit_groups) - 2, fit_scaled.shape[1])
    pca = PCA(n_components=count, svd_solver="full").fit(fit_scaled)
    fit_x = pca.transform(fit_scaled)
    query_x = pca.transform(query_scaled)[0]
    pairwise = np.sqrt(
        np.mean(
            np.square(fit_x[:, None, :] - fit_x[None, :, :]),
            axis=2,
        )
    )
    np.fill_diagonal(pairwise, np.inf)
    neighbors = min(3, len(fit_groups) - 1)
    self_knn = np.sort(pairwise, axis=1)[:, :neighbors].mean(axis=1)
    query_distances = np.sqrt(np.mean(np.square(fit_x - query_x), axis=1))
    query_knn = float(np.sort(query_distances)[: min(3, len(fit_groups))].mean())
    median = float(np.median(self_knn))
    mad = float(1.4826 * np.median(np.abs(self_knn - median)))
    density_z = float((query_knn - median) / max(mad, 0.10))
    return {
        "r3d_knn_training_distance": query_knn,
        "r3d_density_ood_z": density_z,
        "r3d_support_confidence": float(np.exp(-max(density_z, 0.0))),
    }


def predict_candidates(
    *,
    physics: pd.DataFrame,
    embeddings: pd.DataFrame,
    log_target: pd.Series,
    fit_groups: Iterable[str],
    query_group: str,
    config: CandidateConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    fit = list(map(str, fit_groups))
    held = str(query_group)
    density = density_weights(
        physics,
        fit,
        strength=config.density_strength,
        floor=config.density_floor,
    ).set_index("growth_run_id")["density_sample_weight"]
    density_array = density.loc[fit].to_numpy(float)
    residual_array = _residual_sample_weights(
        physics=physics,
        log_target=log_target,
        groups=fit,
        alpha=config.ridge_alpha,
        strength=config.residual_strength,
        floor=config.residual_floor,
    )
    baseline = _ridge_prediction(
        physics=physics,
        log_target=log_target,
        fit_groups=fit,
        query_group=held,
        alpha=config.baseline_alpha,
        morphology_weight=config.morphology_weight,
        sample_weights=None,
    )
    regularized = _ridge_prediction(
        physics=physics,
        log_target=log_target,
        fit_groups=fit,
        query_group=held,
        alpha=config.ridge_alpha,
        morphology_weight=config.morphology_weight,
        sample_weights=None,
    )
    density_prediction = _ridge_prediction(
        physics=physics,
        log_target=log_target,
        fit_groups=fit,
        query_group=held,
        alpha=config.ridge_alpha,
        morphology_weight=config.morphology_weight,
        sample_weights=density_array,
    )
    residual_prediction = _ridge_prediction(
        physics=physics,
        log_target=log_target,
        fit_groups=fit,
        query_group=held,
        alpha=config.ridge_alpha,
        morphology_weight=config.morphology_weight,
        sample_weights=residual_array,
    )
    temporal = _r3d_prediction(
        embeddings=embeddings,
        log_target=log_target,
        fit_groups=fit,
        query_group=held,
        alpha=config.ridge_alpha,
        components=config.r3d_pca_components,
    )
    predictions = {
        BASELINE: baseline,
        REGULARIZED: regularized,
        DENSITY_WEIGHTED: density_prediction,
        RESIDUAL_WEIGHTED: residual_prediction,
        R3D_TEMPORAL: temporal,
        MULTIVIEW_20: 0.20 * regularized + 0.80 * temporal,
        MULTIVIEW_40: 0.40 * regularized + 0.60 * temporal,
        MULTIVIEW_60: 0.60 * regularized + 0.40 * temporal,
    }
    support = query_support(physics, fit, held)
    core = np.asarray(
        [predictions[method] for method in CORE_DISAGREEMENT_METHODS],
        dtype=float,
    )
    diagnostics = {
        "nearest_training_rheed_distance": support.nearest_distance,
        "knn_training_rheed_distance": support.knn_distance,
        "maximum_absolute_robust_z": (support.maximum_absolute_robust_z),
        "ledoit_wolf_mahalanobis": support.mahalanobis_distance,
        "density_ood_z": support.density_ood_z,
        "support_confidence": support.support_confidence,
        "core_head_disagreement_log_std": float(np.std(core)),
        "core_head_log_range": float(np.ptp(core)),
        **_r3d_support(
            embeddings=embeddings,
            fit_groups=fit,
            query_group=held,
            components=config.r3d_pca_components,
        ),
        "training_target_q25_nm": float(
            np.quantile(np.exp(log_target.loc[fit].to_numpy(float)), 0.25)
        ),
        "training_target_median_nm": float(
            np.median(np.exp(log_target.loc[fit].to_numpy(float)))
        ),
        "training_target_q75_nm": float(
            np.quantile(np.exp(log_target.loc[fit].to_numpy(float)), 0.75)
        ),
        "training_target_iqr_nm": float(
            np.subtract(
                *np.quantile(
                    np.exp(log_target.loc[fit].to_numpy(float)),
                    [0.75, 0.25],
                )
            )
        ),
    }
    return predictions, diagnostics


def _nested_calibrated_errors(
    raw_predictions: np.ndarray,
    truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    calibrated = np.zeros_like(raw_predictions, dtype=float)
    for index in range(len(raw_predictions)):
        keep = np.arange(len(raw_predictions)) != index
        calibrated[index], _ = _range_calibrate(
            raw_predictions[keep],
            truth[keep],
            raw_predictions[index],
        )
    return calibrated, np.abs(calibrated - truth)


def _risk_score(
    diagnostics: pd.DataFrame,
    predictions: np.ndarray,
) -> np.ndarray:
    """Epistemic risk from temporal support and sparse target amplitude."""

    r3d_ood = np.maximum(diagnostics["r3d_density_ood_z"].to_numpy(float), 0.0)
    target_iqr = np.maximum(diagnostics["training_target_iqr_nm"].to_numpy(float), 0.50)
    amplitude_extrapolation = np.maximum(
        (
            np.asarray(predictions, dtype=float)
            - diagnostics["training_target_median_nm"].to_numpy(float)
        )
        / target_iqr,
        0.0,
    )
    return r3d_ood + amplitude_extrapolation


def _expected_error_and_confidence(
    inner_diagnostics: pd.DataFrame,
    inner_errors: np.ndarray,
    inner_predictions: np.ndarray,
    query_diagnostics: dict[str, float],
    query_prediction: float,
) -> tuple[float, float, float, np.ndarray]:
    inner_risk = _risk_score(inner_diagnostics, inner_predictions)
    query_frame = pd.DataFrame([query_diagnostics])
    query_risk = float(
        _risk_score(query_frame, np.asarray([query_prediction], dtype=float))[0]
    )
    risk_scale = max(float(np.quantile(inner_risk, 0.75, method="higher")), 0.50)
    baseline_error = float(np.median(inner_errors))
    predicted_error = float(baseline_error * (1.0 + query_risk / risk_scale))
    confidence = float(np.exp(-query_risk / risk_scale))
    return predicted_error, confidence, query_risk, inner_risk


def crossfit_robust_candidates(
    *,
    physics: pd.DataFrame,
    embeddings: pd.DataFrame,
    log_target: pd.Series,
    config: CandidateConfig | None = None,
    confidence_alpha: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strict outer LOO with candidate assessment inside every outer fold."""

    settings = config or CandidateConfig()
    groups = list(map(str, log_target.index))
    outer_records: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    inner_records: list[dict[str, Any]] = []
    for held in groups:
        fit = [group for group in groups if group != held]
        outer_log, outer_diagnostics = predict_candidates(
            physics=physics,
            embeddings=embeddings,
            log_target=log_target,
            fit_groups=fit,
            query_group=held,
            config=settings,
        )
        inner_log: dict[str, list[float]] = {method: [] for method in FIXED_METHODS}
        inner_truth: list[float] = []
        inner_diagnostics_rows: list[dict[str, float | str]] = []
        for inner_held in fit:
            inner_fit = [group for group in fit if group != inner_held]
            predictions, diagnostics = predict_candidates(
                physics=physics,
                embeddings=embeddings,
                log_target=log_target,
                fit_groups=inner_fit,
                query_group=inner_held,
                config=settings,
            )
            for method in FIXED_METHODS:
                inner_log[method].append(predictions[method])
            inner_truth.append(float(np.exp(log_target.loc[inner_held])))
            inner_diagnostics_rows.append(
                {
                    "outer_held_growth_group": held,
                    "inner_held_growth_group": inner_held,
                    **diagnostics,
                }
            )
        truth_array = np.asarray(inner_truth, dtype=float)
        diagnostics_frame = pd.DataFrame(inner_diagnostics_rows)
        method_errors: dict[str, np.ndarray] = {}
        method_scores: dict[str, float] = {}
        calibrated_by_method: dict[str, np.ndarray] = {}
        for method in FIXED_METHODS:
            raw = np.exp(np.asarray(inner_log[method], dtype=float))
            calibrated, errors = _nested_calibrated_errors(raw, truth_array)
            calibrated_by_method[method] = calibrated
            method_errors[method] = errors
            method_scores[method] = float(np.mean(errors) + 0.25 * np.median(errors))
            for position, inner_group in enumerate(fit):
                inner_records.append(
                    {
                        "outer_held_growth_group": held,
                        "inner_held_growth_group": inner_group,
                        "method": method,
                        "true_target": truth_array[position],
                        "raw_predicted_target": raw[position],
                        "predicted_target": calibrated[position],
                        "absolute_error": errors[position],
                        "inner_selection_score": method_scores[method],
                        **{
                            key: diagnostics_frame.iloc[position][key]
                            for key in outer_diagnostics
                        },
                    }
                )
        selected_method = min(
            method_scores, key=lambda method: (method_scores[method], method)
        )
        for method in FIXED_METHODS:
            raw_outer = float(np.exp(outer_log[method]))
            calibrated_outer, calibration_scale = _range_calibrate(
                np.exp(np.asarray(inner_log[method], dtype=float)),
                truth_array,
                raw_outer,
            )
            (
                expected_error,
                confidence,
                risk,
                inner_risk,
            ) = _expected_error_and_confidence(
                diagnostics_frame,
                method_errors[method],
                calibrated_by_method[method],
                outer_diagnostics,
                calibrated_outer,
            )
            adaptive_scores = method_errors[method] / (1.0 + inner_risk)
            radius = _higher_quantile(adaptive_scores, 1.0 - confidence_alpha) * (
                1.0 + risk
            )
            true_value = float(np.exp(log_target.loc[held]))
            record = {
                "growth_run_id": held,
                "method": method,
                "true_target": true_value,
                "raw_predicted_target": raw_outer,
                "predicted_target": calibrated_outer,
                "range_calibration_log_scale": calibration_scale,
                "absolute_error": abs(calibrated_outer - true_value),
                "predicted_absolute_error": expected_error,
                "confidence": confidence,
                "interval_lower": max(calibrated_outer - radius, 0.0),
                "interval_upper": calibrated_outer + radius,
                "interval_radius": radius,
                "interval_covered": bool(
                    max(calibrated_outer - radius, 0.0)
                    <= true_value
                    <= calibrated_outer + radius
                ),
                "outer_target_used_for_training": False,
                "outer_fit_growth_count": len(fit),
                "inner_selection_score": method_scores[method],
                "selected_by_inner_cv": method == selected_method,
                "uncertainty_risk_score": risk,
                **outer_diagnostics,
            }
            outer_records.append(record)
            if method == selected_method:
                selected_records.append(
                    {
                        **record,
                        "method": NESTED_SELECTOR,
                        "selected_candidate": selected_method,
                    }
                )
    return (
        pd.DataFrame(outer_records),
        pd.DataFrame(selected_records),
        pd.DataFrame(inner_records),
    )
