from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from analysis.rheed_to_afm_functional_morphology.amplitude import (
    _higher_quantile,
    _range_calibrate,
)
from analysis.rheed_to_afm_ood_robust.prediction import (
    CandidateConfig,
    _r3d_prediction,
    predict_candidates,
)


@dataclass(frozen=True)
class StabilitySummary:
    base_prediction: float
    median_prediction: float
    base_to_median_nm: float
    frame_std_nm: float
    roi_std_nm: float
    all_std_nm: float
    all_range_nm: float


def _view_summary(
    predictions: np.ndarray,
    view_names: list[str],
) -> StabilitySummary:
    values = np.asarray(predictions, dtype=float)
    lookup = {name: index for index, name in enumerate(view_names)}
    base = float(values[lookup["base"]])
    frame = values[
        [
            lookup[name]
            for name in (
                "frame_m2",
                "frame_m1",
                "base",
                "frame_p1",
                "frame_p2",
            )
        ]
    ]
    roi = values[
        [
            lookup[name]
            for name in (
                "roi_left",
                "roi_right",
                "roi_up",
                "roi_down",
                "base",
                "roi_tight",
                "roi_wide",
            )
        ]
    ]
    return StabilitySummary(
        base_prediction=base,
        median_prediction=float(np.median(values)),
        base_to_median_nm=float(abs(base - np.median(values))),
        frame_std_nm=float(np.std(frame)),
        roi_std_nm=float(np.std(roi)),
        all_std_nm=float(np.std(values)),
        all_range_nm=float(np.ptp(values)),
    )


def _raw_r3d_views(
    *,
    base_embeddings: pd.DataFrame,
    query_views: np.ndarray,
    log_target: pd.Series,
    fit_groups: list[str],
    query_prefix: str,
    alpha: float,
    components: int,
) -> np.ndarray:
    train = base_embeddings.loc[fit_groups]
    raw = []
    for view_index, embedding in enumerate(query_views):
        query = f"__{query_prefix}_{view_index}__"
        frame = pd.concat(
            [
                train,
                pd.DataFrame(
                    np.asarray(embedding, dtype=float)[None],
                    index=[query],
                ),
            ]
        )
        value = _r3d_prediction(
            embeddings=frame,
            log_target=log_target,
            fit_groups=fit_groups,
            query_group=query,
            alpha=float(alpha),
            components=int(components),
        )
        raw.append(float(np.exp(value)))
    return np.asarray(raw, dtype=float)


def _calibrate_views(
    raw_views: np.ndarray,
    reference_raw: np.ndarray,
    reference_truth: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            _range_calibrate(
                reference_raw,
                reference_truth,
                float(value),
            )[0]
            for value in np.asarray(raw_views, dtype=float)
        ],
        dtype=float,
    )


def _empirical_risk(query: float, reference: np.ndarray) -> float:
    """Smoothed empirical percentile; larger means less stable."""

    values = np.asarray(reference, dtype=float)
    return float((0.5 + np.sum(values <= float(query))) / (len(values) + 1.0))


def combine_tta_and_head_confidence(
    tta_confidence: float | np.ndarray,
    head_agreement_confidence: float | np.ndarray,
    *,
    extreme_quantile: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply head disagreement only as an extreme-conflict veto.

    With 23 scientific samples, ordinary temporal-versus-physics head
    disagreement is too noisy to rank every prediction.  The disagreement
    signal is therefore allowed to reduce TTA confidence only in the most
    extreme empirical tail.  This preserves input-stability ordering while
    catching predictions that are stable under TTA but unsupported by a
    second representation.
    """

    tta = np.asarray(tta_confidence, dtype=float)
    head = np.asarray(head_agreement_confidence, dtype=float)
    tta, head = np.broadcast_arrays(tta, head)
    veto = head <= float(extreme_quantile)
    combined = np.where(veto, np.minimum(tta, head), tta)
    return np.clip(combined, 0.0, 1.0), veto


def angular_coverage_risk(
    tta_centrality_risk: float | np.ndarray,
    rotation_period_risk: float | np.ndarray,
) -> np.ndarray:
    """Combine local instability and angular-window insufficiency."""

    centrality = np.clip(np.asarray(tta_centrality_risk, dtype=float), 0, 1)
    period = np.clip(np.asarray(rotation_period_risk, dtype=float), 0, 1)
    centrality, period = np.broadcast_arrays(centrality, period)
    return np.sqrt(centrality * period)


def _isotonic_expected_error(
    risk: np.ndarray,
    errors: np.ndarray,
    query_risk: float,
) -> float:
    x = np.asarray(risk, dtype=float)
    y = np.asarray(errors, dtype=float)
    if len(np.unique(x)) < 3:
        return float(np.median(y))
    model = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
        y_min=0.0,
    )
    try:
        model.fit(x, y)
        value = float(model.predict([float(query_risk)])[0])
        return max(value, 0.0)
    except Exception:
        return float(np.median(y))


def crossfit_r3d_stability_confidence(
    *,
    perturbation_embeddings: np.ndarray,
    view_names: list[str],
    groups: list[str],
    log_target: pd.Series,
    physics: pd.DataFrame,
    candidate_config: CandidateConfig,
    estimated_period_frames: pd.Series,
    alpha: float = 10.0,
    components: int = 5,
    confidence_alpha: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strict outer LOO R3D predictions with nested TTA confidence.

    For every outer held growth, all error calibration uses only the other
    growths. The held AFM target is appended only after the prediction,
    confidence, and interval have been finalized.

    The primary risk is not the raw TTA variance. It is the displacement of
    the unperturbed prediction from the median of physically plausible
    keyframe/ROI perturbations. This detects a selected input that lies on one
    side of a local prediction transition even when all views form a tight
    cluster.
    """

    tensor = np.asarray(perturbation_embeddings, dtype=np.float32)
    if tensor.ndim != 3 or tensor.shape[:2] != (
        len(groups),
        len(view_names),
    ):
        raise ValueError("perturbation embedding tensor shape is inconsistent")
    if "base" not in view_names:
        raise ValueError("a base perturbation view is required")
    base_index = view_names.index("base")
    base_embeddings = pd.DataFrame(
        tensor[:, base_index],
        index=list(map(str, groups)),
    )
    target = log_target.copy()
    target.index = target.index.astype(str)
    target = target.loc[groups]
    physics_frame = physics.copy()
    physics_frame.index = physics_frame.index.astype(str)
    physics_frame = physics_frame.loc[groups]
    periods = estimated_period_frames.copy()
    periods.index = periods.index.astype(str)
    periods = periods.loc[groups].astype(float)
    if not np.isfinite(periods.to_numpy(float)).all():
        raise ValueError("estimated rotation periods must be finite")
    truth = np.exp(target)
    records: list[dict[str, float | str | bool]] = []
    inner_records: list[dict[str, float | str]] = []

    for held in groups:
        fit = [group for group in groups if group != held]
        inner_raw = np.empty((len(fit), len(view_names)), dtype=float)
        for inner_position, inner_held in enumerate(fit):
            inner_fit = [
                group for group in fit if group != inner_held
            ]
            query_views = tensor[groups.index(inner_held)]
            inner_raw[inner_position] = _raw_r3d_views(
                base_embeddings=base_embeddings,
                query_views=query_views,
                log_target=target,
                fit_groups=inner_fit,
                query_prefix=f"{held}_{inner_held}",
                alpha=alpha,
                components=components,
            )

        fit_truth = truth.loc[fit].to_numpy(float)
        inner_calibrated = np.empty_like(inner_raw)
        inner_summaries: list[StabilitySummary] = []
        inner_diagnostics: list[dict[str, float]] = []
        for inner_position, inner_held in enumerate(fit):
            keep = np.arange(len(fit)) != inner_position
            inner_calibrated[inner_position] = _calibrate_views(
                inner_raw[inner_position],
                inner_raw[keep, base_index],
                fit_truth[keep],
            )
            summary = _view_summary(
                inner_calibrated[inner_position],
                view_names,
            )
            inner_summaries.append(summary)
            inner_fit = [
                group for group in fit if group != inner_held
            ]
            _, diagnostics = predict_candidates(
                physics=physics_frame,
                embeddings=base_embeddings,
                log_target=target,
                fit_groups=inner_fit,
                query_group=inner_held,
                config=candidate_config,
            )
            inner_diagnostics.append(diagnostics)

        inner_base = inner_calibrated[:, base_index]
        inner_errors = np.abs(inner_base - fit_truth)
        inner_centrality = np.asarray(
            [summary.base_to_median_nm for summary in inner_summaries],
            dtype=float,
        )
        inner_risk = np.asarray(
            [
                _empirical_risk(
                    inner_centrality[position],
                    np.delete(inner_centrality, position),
                )
                for position in range(len(inner_centrality))
            ],
            dtype=float,
        )
        fit_periods = periods.loc[fit].to_numpy(float)
        inner_period_risk = np.asarray(
            [
                _empirical_risk(
                    fit_periods[position],
                    np.delete(fit_periods, position),
                )
                for position in range(len(fit_periods))
            ],
            dtype=float,
        )
        inner_angular_tta_risk = angular_coverage_risk(
            inner_risk,
            inner_period_risk,
        )
        inner_amplitude_risk = np.asarray(
            [
                _empirical_risk(
                    inner_base[position],
                    np.delete(inner_base, position),
                )
                for position in range(len(inner_base))
            ],
            dtype=float,
        )
        inner_head_disagreement_array = np.asarray(
            [
                diagnostic["core_head_disagreement_log_std"]
                for diagnostic in inner_diagnostics
            ],
            dtype=float,
        )
        inner_head_risk = np.asarray(
            [
                _empirical_risk(
                    inner_head_disagreement_array[position],
                    np.delete(inner_head_disagreement_array, position),
                )
                for position in range(len(inner_head_disagreement_array))
            ],
            dtype=float,
        )
        inner_range_aware_risk = (
            0.75 * inner_amplitude_risk
            + 0.25 * inner_angular_tta_risk
        )
        inner_confidence, inner_veto = combine_tta_and_head_confidence(
            1.0 - inner_range_aware_risk,
            1.0 - inner_head_risk,
            extreme_quantile=0.10,
        )
        inner_composite_risk = 1.0 - inner_confidence

        outer_raw = _raw_r3d_views(
            base_embeddings=base_embeddings,
            query_views=tensor[groups.index(held)],
            log_target=target,
            fit_groups=fit,
            query_prefix=f"{held}_outer",
            alpha=alpha,
            components=components,
        )
        outer_calibrated = _calibrate_views(
            outer_raw,
            inner_raw[:, base_index],
            fit_truth,
        )
        summary = _view_summary(outer_calibrated, view_names)
        risk = _empirical_risk(
            summary.base_to_median_nm,
            inner_centrality,
        )
        period_risk = _empirical_risk(
            float(periods.loc[held]),
            fit_periods,
        )
        angular_tta_risk = float(
            angular_coverage_risk(risk, period_risk).item()
        )
        tta_confidence = float(
            np.clip(1.0 - angular_tta_risk, 0.0, 1.0)
        )
        _, outer_diagnostics = predict_candidates(
            physics=physics_frame,
            embeddings=base_embeddings,
            log_target=target,
            fit_groups=fit,
            query_group=held,
            config=candidate_config,
        )
        outer_head_disagreement = float(
            outer_diagnostics["core_head_disagreement_log_std"]
        )
        head_risk = _empirical_risk(
            outer_head_disagreement,
            inner_head_disagreement_array,
        )
        head_confidence = float(np.clip(1.0 - head_risk, 0.0, 1.0))
        amplitude_risk = _empirical_risk(
            summary.base_prediction,
            inner_base,
        )
        range_aware_risk = float(
            0.75 * amplitude_risk + 0.25 * angular_tta_risk
        )
        combined, veto = combine_tta_and_head_confidence(
            1.0 - range_aware_risk,
            head_confidence,
            extreme_quantile=0.10,
        )
        confidence = float(combined.item())
        extreme_head_veto = bool(veto.item())
        composite_risk = 1.0 - confidence
        expected_error = _isotonic_expected_error(
            inner_composite_risk,
            inner_errors,
            composite_risk,
        )
        adaptive = inner_errors / np.maximum(
            0.50 + inner_composite_risk,
            0.25,
        )
        radius = _higher_quantile(
            adaptive,
            1.0 - float(confidence_alpha),
        ) * (0.50 + composite_risk)
        predicted = summary.base_prediction
        true_value = float(truth.loc[held])
        records.append(
            {
                "growth_run_id": held,
                "true_target": true_value,
                "predicted_target": predicted,
                "absolute_error": abs(predicted - true_value),
                "predicted_absolute_error": expected_error,
                "confidence": confidence,
                "uncertainty_risk_score": composite_risk,
                "tta_uncertainty_risk_score": risk,
                "rotation_period_risk_score": period_risk,
                "angular_tta_risk_score": angular_tta_risk,
                "predicted_amplitude_risk_score": amplitude_risk,
                "range_aware_risk_score": range_aware_risk,
                "estimated_period_frames": float(periods.loc[held]),
                "head_disagreement_risk_score": head_risk,
                "tta_centrality_confidence": float(
                    np.clip(1.0 - risk, 0.0, 1.0)
                ),
                "angular_coverage_tta_confidence": tta_confidence,
                "legacy_angular_tta_confidence": tta_confidence,
                "head_agreement_confidence": head_confidence,
                "extreme_head_disagreement_veto": extreme_head_veto,
                "core_head_disagreement_log_std": (
                    outer_head_disagreement
                ),
                "core_head_log_range": float(
                    outer_diagnostics["core_head_log_range"]
                ),
                "r3d_density_ood_z": float(
                    outer_diagnostics["r3d_density_ood_z"]
                ),
                "density_ood_z": float(
                    outer_diagnostics["density_ood_z"]
                ),
                "r3d_support_confidence": float(
                    outer_diagnostics["r3d_support_confidence"]
                ),
                "physics_support_confidence": float(
                    outer_diagnostics["support_confidence"]
                ),
                "interval_lower": max(predicted - radius, 0.0),
                "interval_upper": predicted + radius,
                "interval_radius": radius,
                "interval_covered": bool(
                    max(predicted - radius, 0.0)
                    <= true_value
                    <= predicted + radius
                ),
                "base_to_tta_median_nm": summary.base_to_median_nm,
                "tta_frame_std_nm": summary.frame_std_nm,
                "tta_roi_std_nm": summary.roi_std_nm,
                "tta_all_std_nm": summary.all_std_nm,
                "tta_all_range_nm": summary.all_range_nm,
                "outer_target_used_for_training": False,
                "outer_fit_growth_count": len(fit),
                "confidence_calibration_growth_count": len(fit),
                "method": "M15b_auto_r3d_angular_tta",
            }
        )
        for position, inner_held in enumerate(fit):
            inner_records.append(
                {
                    "outer_held_growth_group": held,
                    "inner_held_growth_group": inner_held,
                    "true_target": fit_truth[position],
                    "predicted_target": inner_base[position],
                    "absolute_error": inner_errors[position],
                    "base_to_tta_median_nm": inner_centrality[position],
                    "tta_frame_std_nm": inner_summaries[
                        position
                    ].frame_std_nm,
                    "tta_roi_std_nm": inner_summaries[position].roi_std_nm,
                    "tta_all_std_nm": inner_summaries[position].all_std_nm,
                    "tta_all_range_nm": inner_summaries[position].all_range_nm,
                    "tta_uncertainty_risk_score": inner_risk[position],
                    "rotation_period_risk_score": inner_period_risk[position],
                    "angular_tta_risk_score": inner_angular_tta_risk[
                        position
                    ],
                    "predicted_amplitude_risk_score": (
                        inner_amplitude_risk[position]
                    ),
                    "range_aware_risk_score": (
                        inner_range_aware_risk[position]
                    ),
                    "estimated_period_frames": fit_periods[position],
                    "core_head_disagreement_log_std": (
                        inner_head_disagreement_array[position]
                    ),
                    "core_head_log_range": float(
                        inner_diagnostics[position]["core_head_log_range"]
                    ),
                    "r3d_density_ood_z": float(
                        inner_diagnostics[position]["r3d_density_ood_z"]
                    ),
                    "density_ood_z": float(
                        inner_diagnostics[position]["density_ood_z"]
                    ),
                    "r3d_support_confidence": float(
                        inner_diagnostics[position][
                            "r3d_support_confidence"
                        ]
                    ),
                    "physics_support_confidence": float(
                        inner_diagnostics[position]["support_confidence"]
                    ),
                    "head_disagreement_risk_score": inner_head_risk[position],
                    "extreme_head_disagreement_veto": bool(
                        inner_veto[position]
                    ),
                    "uncertainty_risk_score": inner_composite_risk[position],
                    "confidence": inner_confidence[position],
                    "outer_target_used_for_calibration": False,
                }
            )
    return pd.DataFrame(records), pd.DataFrame(inner_records)
