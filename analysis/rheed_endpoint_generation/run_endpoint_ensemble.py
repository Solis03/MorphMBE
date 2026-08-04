from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.rheed_auto_input_robustness.confidence import (
    _isotonic_expected_error,
)
from analysis.rheed_single_frame.removelist import load_removelist_audit

from .endpoint_ensemble import EndpointPrediction, predict_endpoint
from .streak_features import (
    PRIMARY_STREAK_FEATURE,
    extract_streak_features_from_npz,
)


MODEL_NAME = "M16_endpoint_streak_dual_resolution"


def _empirical_rank(value: float, reference: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    return float((np.sum(reference < value) + 0.5) / (len(reference) + 1.0))


def _diagnostic_vector(
    prediction: EndpointPrediction,
    *,
    training_targets: np.ndarray,
) -> np.ndarray:
    log_target = np.log(np.clip(training_targets, 1e-8, None))
    center = float(np.median(log_target))
    scale = max(
        float(
            (
                np.quantile(log_target, 0.75)
                - np.quantile(log_target, 0.25)
            )
            / 1.349
        ),
        1e-6,
    )
    return np.asarray(
        [
            prediction.expert_log_range,
            prediction.nearest_embedding_distance,
            prediction.streak_robust_z,
            abs(np.log(prediction.value_nm) - center) / scale,
        ]
    )


def _risk(
    query: np.ndarray,
    reference: np.ndarray,
) -> float:
    ranks = np.asarray(
        [
            _empirical_rank(float(query[index]), reference[:, index])
            for index in range(reference.shape[1])
        ]
    )
    return float(np.dot(ranks, [0.35, 0.30, 0.10, 0.25]))


def _support_risk(
    prediction: EndpointPrediction,
    *,
    training_targets: np.ndarray,
    legacy_risk: float,
) -> float:
    """Estimate endpoint risk from training support, not the query AFM.

    Rough surfaces are sparse in this cohort, so distance from the training
    target median increases uncertainty.  A very low prediction is considered
    supported only when the independent local-streak gate fired.  The 0.30
    multiplier reflects agreement between the temporal and physically
    interpretable streak route; it is a confidence rule, not a point-prediction
    correction.
    """

    truth = np.asarray(training_targets, dtype=float)
    log_truth = np.log(np.clip(truth, 1e-8, None))
    center = float(np.median(log_truth))
    query_distance = abs(np.log(prediction.value_nm) - center)
    amplitude_risk = float(
        (
            np.sum(np.abs(log_truth - center) < query_distance) + 0.5
        )
        / (len(log_truth) + 1.0)
    )
    unsupported_low = float(
        prediction.value_nm < np.quantile(truth, 0.25)
        and not prediction.streak_gate
    )
    risk = (
        0.80 * amplitude_risk
        + 0.10 * float(legacy_risk)
        + 0.10 * unsupported_low
    )
    if prediction.streak_gate:
        risk *= 0.30
    return float(np.clip(risk, 0.0, 1.0))


def _metrics(frame: pd.DataFrame, model: str) -> dict[str, Any]:
    truth = frame["true_target"].to_numpy(float)
    predicted = frame["predicted_target"].to_numpy(float)
    error = np.abs(predicted - truth)
    smooth = truth < 1.20
    rough = truth >= 5.0
    confidence = frame["confidence"].to_numpy(float)
    return {
        "model": model,
        "growth_count": len(frame),
        "mae_nm": float(np.mean(error)),
        "median_absolute_error_nm": float(np.median(error)),
        "rmse_nm": float(np.sqrt(np.mean(np.square(error)))),
        "bias_nm": float(np.mean(predicted - truth)),
        "pearson_r": float(pearsonr(truth, predicted).statistic),
        "spearman_rho": float(spearmanr(truth, predicted).statistic),
        "smooth_count": int(np.sum(smooth)),
        "smooth_mae_nm": float(np.mean(error[smooth])),
        "smooth_bias_nm": float(np.mean((predicted - truth)[smooth])),
        "rough_count": int(np.sum(rough)),
        "rough_mae_nm": float(np.mean(error[rough])),
        "rough_bias_nm": float(np.mean((predicted - truth)[rough])),
        "confidence_vs_error_spearman": float(
            spearmanr(confidence, error).statistic
        ),
        "prediction_min_nm": float(np.min(predicted)),
        "prediction_max_nm": float(np.max(predicted)),
    }


def _feature_table(manifest: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    rows = []
    for _, row in manifest.iterrows():
        path = Path(str(row["clip_cache_path"]))
        if not path.is_absolute():
            path = data_root / path
        features = extract_streak_features_from_npz(path)
        rows.append(
            {
                "growth_run_id": str(row["growth_run_id"]),
                "clip_cache_path": str(path),
                **features,
            }
        )
    return pd.DataFrame(rows).set_index("growth_run_id")


def run(args: argparse.Namespace) -> None:
    payload = np.load(args.perturbation_embeddings, allow_pickle=False)
    source_groups = [str(value) for value in payload["growth_run_ids"]]
    excluded: set[str] = set()
    removelist_audit = None
    if args.removelist is not None:
        removelist_audit = load_removelist_audit(Path.cwd(), args.removelist)
        excluded = set(map(str, removelist_audit.sample_ids))
    keep = np.asarray(
        [group not in excluded for group in source_groups], dtype=bool
    )
    groups = [
        group for group, retained in zip(source_groups, keep) if retained
    ]
    views = [str(value) for value in payload["view_names"]]
    base = views.index("base")
    embeddings = np.asarray(payload["embeddings"][keep, base], dtype=float)
    targets = (
        pd.read_csv(args.targets, dtype={"growth_run_id": str})
        .set_index("growth_run_id")
        .loc[groups, "sample_median_sq_nm"]
        .to_numpy(float)
    )
    manifest = (
        pd.read_csv(args.manifest, dtype={"growth_run_id": str})
        .set_index("growth_run_id", drop=False)
        .loc[groups]
    )
    features = _feature_table(manifest, args.data_root).loc[groups]
    streak = features[PRIMARY_STREAK_FEATURE].to_numpy(float)
    baseline = pd.read_csv(
        args.baseline_predictions, dtype={"growth_run_id": str}
    )
    baseline_sq = (
        baseline.loc[baseline["target"] == "Rq_nm"]
        .set_index("growth_run_id")
        .loc[groups]
    )
    nested_baseline = pd.read_csv(
        args.baseline_nested, dtype={
            "outer_held_growth_group": str,
            "inner_held_growth_group": str,
        }
    )
    nested_baseline = nested_baseline.loc[
        nested_baseline["target"] == "Rq_nm"
    ]
    indices = np.arange(len(groups))
    records: list[dict[str, Any]] = []
    nested_records: list[dict[str, Any]] = []
    for outer_index, held in enumerate(groups):
        fit = indices[indices != outer_index]
        outer_prediction = predict_endpoint(
            embeddings=embeddings,
            streak=streak,
            target_nm=targets,
            train=fit,
            query_embedding=embeddings[outer_index],
            query_streak=float(streak[outer_index]),
        )
        outer_diagnostic = _diagnostic_vector(
            outer_prediction, training_targets=targets[fit]
        )
        inner_predictions: list[EndpointPrediction] = []
        inner_diagnostics: list[np.ndarray] = []
        inner_errors: list[float] = []
        for inner_index in fit:
            inner_train = fit[fit != inner_index]
            prediction = predict_endpoint(
                embeddings=embeddings,
                streak=streak,
                target_nm=targets,
                train=inner_train,
                query_embedding=embeddings[inner_index],
                query_streak=float(streak[inner_index]),
            )
            diagnostic = _diagnostic_vector(
                prediction, training_targets=targets[inner_train]
            )
            inner_predictions.append(prediction)
            inner_diagnostics.append(diagnostic)
            inner_errors.append(
                abs(prediction.value_nm - targets[inner_index])
            )
        diagnostic_matrix = np.stack(inner_diagnostics)
        inner_new_risk = np.asarray(
            [
                _risk(
                    diagnostic_matrix[position],
                    diagnostic_matrix[
                        np.arange(len(diagnostic_matrix)) != position
                    ],
                )
                for position in range(len(diagnostic_matrix))
            ]
        )
        outer_new_risk = _risk(outer_diagnostic, diagnostic_matrix)
        old_inner = (
            nested_baseline.loc[
                nested_baseline["outer_held_growth_group"] == held
            ]
            .set_index("inner_held_growth_group")
            .loc[[groups[index] for index in fit]]
        )
        old_inner_risk = old_inner["uncertainty_risk_score"].to_numpy(float)
        old_outer_risk = float(
            baseline_sq.loc[held, "uncertainty_risk_score"]
        )
        inner_combined_risk = np.asarray(
            [
                _support_risk(
                    prediction,
                    training_targets=targets[
                        fit[fit != inner_index]
                    ],
                    legacy_risk=old_inner_risk[position],
                )
                for position, (inner_index, prediction) in enumerate(
                    zip(fit, inner_predictions)
                )
            ]
        )
        outer_risk = _support_risk(
            outer_prediction,
            training_targets=targets[fit],
            legacy_risk=old_outer_risk,
        )
        inner_errors_array = np.asarray(inner_errors)
        expected_error = _isotonic_expected_error(
            inner_combined_risk,
            inner_errors_array,
            outer_risk,
        )
        adaptive = inner_errors_array / np.maximum(
            0.50 + inner_combined_risk, 0.25
        )
        radius = float(
            np.quantile(adaptive, 0.90, method="higher")
            * (0.50 + outer_risk)
        )
        confidence = float(np.clip(1.0 - outer_risk, 0.0, 1.0))
        for position, inner_index in enumerate(fit):
            prediction = inner_predictions[position]
            nested_records.append(
                {
                    "outer_held_growth_group": held,
                    "inner_held_growth_group": groups[inner_index],
                    "true_target": targets[inner_index],
                    "predicted_target": prediction.value_nm,
                    "absolute_error": inner_errors_array[position],
                    "new_diagnostic_risk": inner_new_risk[position],
                    "legacy_tta_head_risk": old_inner_risk[position],
                    "support_risk": inner_combined_risk[position],
                    "outer_target_used_for_calibration": False,
                }
            )
        records.append(
            {
                "target": "Rq_nm",
                "growth_run_id": held,
                "true_target": targets[outer_index],
                "predicted_target": outer_prediction.value_nm,
                "absolute_error": abs(
                    outer_prediction.value_nm - targets[outer_index]
                ),
                "predicted_absolute_error": expected_error,
                "confidence": confidence,
                "uncertainty_risk_score": outer_risk,
                "new_diagnostic_risk": outer_new_risk,
                "legacy_tta_head_risk": old_outer_risk,
                "interval_lower": max(
                    outer_prediction.value_nm - radius, 0.0
                ),
                "interval_upper": outer_prediction.value_nm + radius,
                "interval_radius": radius,
                "interval_covered": bool(
                    abs(outer_prediction.value_nm - targets[outer_index])
                    <= radius
                ),
                "temporal_5_nm": outer_prediction.temporal_5_nm,
                "temporal_8_nm": outer_prediction.temporal_8_nm,
                "streak_expert_nm": outer_prediction.streak_expert_nm,
                "streak_gate": outer_prediction.streak_gate,
                "rough_consensus_gate": (
                    outer_prediction.rough_consensus_gate
                ),
                "streak_feature": streak[outer_index],
                "streak_threshold": outer_prediction.streak_threshold,
                "upper_threshold_nm": outer_prediction.upper_threshold_nm,
                "expert_log_range": outer_prediction.expert_log_range,
                "nearest_embedding_distance": (
                    outer_prediction.nearest_embedding_distance
                ),
                "streak_robust_z": outer_prediction.streak_robust_z,
                "outer_target_used_for_training": False,
                "outer_fit_growth_count": len(fit),
                "confidence_calibration_growth_count": len(fit),
                "method": MODEL_NAME,
                "confidence_method": (
                    "target-blind endpoint support risk with causal-TTA/head "
                    "risk and independent local-streak support"
                ),
            }
        )
        print(
            f"[{outer_index + 1:02d}/{len(groups):02d}] {held} "
            f"truth={targets[outer_index]:.3f} "
            f"prediction={outer_prediction.value_nm:.3f} "
            f"streak_gate={outer_prediction.streak_gate} "
            f"rough_gate={outer_prediction.rough_consensus_gate}",
            flush=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    nested = pd.DataFrame(nested_records)
    result.to_csv(args.output / "m16_strict_loo_predictions.csv", index=False)
    nested.to_csv(
        args.output / "m16_nested_inner_predictions.csv", index=False
    )
    features.reset_index().to_csv(
        args.output / "streak_features.csv", index=False
    )
    baseline_metric_input = baseline_sq.reset_index().copy()
    metrics = pd.DataFrame(
        [
            _metrics(baseline_metric_input, "M15b_current"),
            _metrics(result, MODEL_NAME),
        ]
    )
    metrics.to_csv(args.output / "baseline_vs_m16_metrics.csv", index=False)
    manifest_payload = {
        "model": MODEL_NAME,
        "protocol": "strict nested leave-one-growth-out",
        "growth_count": len(groups),
        "growth_run_ids": groups,
        "excluded_growths_present_in_source": sorted(
            set(source_groups) & excluded
        ),
        "removelist_sha256": (
            removelist_audit.sha256 if removelist_audit is not None else None
        ),
        "outer_target_used_for_training_or_calibration": False,
        "r3d_temporal_experts": [
            {"components": 5, "alpha": 30.0, "calibration": "range30"},
            {"components": 8, "alpha": 30.0, "calibration": "range30"},
        ],
        "streak_expert": {
            "feature": PRIMARY_STREAK_FEATURE,
            "components": 3,
            "alpha": 3.0,
            "calibration": "range20",
            "activation_quantile": 0.80,
        },
        "rough_consensus_quantile": 0.75,
        "confidence": (
            "target-blind prediction-amplitude support plus the pre-existing "
            "causal-TTA/head risk; the independent streak gate supplies "
            "low-end physical support"
        ),
        "metrics": metrics.to_dict("records"),
    }
    (args.output / "experiment_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturbation-embeddings", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--baseline-nested", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--removelist", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
