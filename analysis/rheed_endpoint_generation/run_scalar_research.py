from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from analysis.rheed_to_afm_functional_morphology.run import _physics_table

from .endpoint_models import (
    CALIBRATION_METHODS,
    PHYSICS_ENDPOINT_FEATURES,
    build_model,
    calibrate_positive,
    candidate_grid,
    cross_calibrated_predictions,
    endpoint_objective,
)


def _fit_predict(
    candidate: Any,
    fit: np.ndarray,
    query: int,
    r3d: np.ndarray,
    physics: np.ndarray,
    log_target: np.ndarray,
) -> float:
    model = build_model(candidate).fit(
        r3d[fit],
        physics[fit],
        log_target[fit],
    )
    return float(
        np.exp(model.predict(r3d[[query]], physics[[query]])[0])
    )


def _metrics(rows: pd.DataFrame, method: str) -> dict[str, Any]:
    truth = rows["true_sq_nm"].to_numpy(float)
    predicted = rows["predicted_sq_nm"].to_numpy(float)
    error = np.abs(predicted - truth)
    smooth = truth < 1.20
    rough = truth >= 5.0
    return {
        "method": method,
        "growth_count": len(rows),
        "mae_nm": float(np.mean(error)),
        "rmse_nm": float(np.sqrt(np.mean(np.square(error)))),
        "bias_nm": float(np.mean(predicted - truth)),
        "pearson_r": float(pearsonr(truth, predicted).statistic),
        "spearman_rho": float(spearmanr(truth, predicted).statistic),
        "smooth_count": int(smooth.sum()),
        "smooth_mae_nm": float(np.mean(error[smooth])),
        "smooth_bias_nm": float(np.mean((predicted - truth)[smooth])),
        "rough_count": int(rough.sum()),
        "rough_mae_nm": float(np.mean(error[rough])),
        "rough_bias_nm": float(np.mean((predicted - truth)[rough])),
        "prediction_min_nm": float(np.min(predicted)),
        "prediction_max_nm": float(np.max(predicted)),
    }


def run(args: argparse.Namespace) -> None:
    payload = np.load(args.perturbation_embeddings, allow_pickle=False)
    groups = [str(value) for value in payload["growth_run_ids"]]
    views = [str(value) for value in payload["view_names"]]
    base = views.index("base")
    r3d = np.asarray(payload["embeddings"][:, base], dtype=np.float64)
    targets = (
        pd.read_csv(
            args.targets,
            dtype={"growth_run_id": str},
        )
        .set_index("growth_run_id")
        .loc[groups, "sample_median_sq_nm"]
        .to_numpy(float)
    )
    log_target = np.log(np.clip(targets, 1e-8, None))
    physics_frame = _physics_table(
        pd.read_csv(
            args.physics,
            dtype={"growth_run_id": str, "sample_id": str},
        )
    ).loc[groups]
    missing = sorted(set(PHYSICS_ENDPOINT_FEATURES) - set(physics_frame))
    if missing:
        raise RuntimeError(f"endpoint physics features are missing: {missing}")
    physics = (
        physics_frame[PHYSICS_ENDPOINT_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(float)
    )
    candidates = candidate_grid()
    records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    candidate_outer_records: list[dict[str, Any]] = []
    all_indices = np.arange(len(groups))
    for outer_index, held in enumerate(groups):
        fit = all_indices[all_indices != outer_index]
        truth_fit = targets[fit]
        best: tuple[float, str, str, Any, np.ndarray] | None = None
        for candidate in candidates:
            inner_raw = np.empty(len(fit), dtype=float)
            for position, inner_index in enumerate(fit):
                inner_fit = fit[fit != inner_index]
                inner_raw[position] = _fit_predict(
                    candidate,
                    inner_fit,
                    int(inner_index),
                    r3d,
                    physics,
                    log_target,
                )
            for calibration in CALIBRATION_METHODS:
                inner_prediction = cross_calibrated_predictions(
                    inner_raw,
                    truth_fit,
                    calibration,
                )
                score = endpoint_objective(
                    truth_fit,
                    inner_prediction,
                )
                selection_records.append(
                    {
                        "outer_held_growth": held,
                        "candidate": candidate.identifier,
                        "family": candidate.family,
                        "calibration": calibration,
                        "inner_endpoint_objective": score,
                        "inner_mae_nm": float(
                            np.mean(
                                np.abs(inner_prediction - truth_fit)
                            )
                        ),
                        "inner_smooth_mae_nm": float(
                            np.mean(
                                np.abs(inner_prediction - truth_fit)[
                                    truth_fit
                                    <= np.quantile(truth_fit, 0.20)
                                ]
                            )
                        ),
                        "inner_rough_mae_nm": float(
                            np.mean(
                                np.abs(inner_prediction - truth_fit)[
                                    truth_fit
                                    >= np.quantile(truth_fit, 0.80)
                                ]
                            )
                        ),
                    }
                )
                key = (
                    score,
                    candidate.identifier,
                    calibration,
                    candidate,
                    inner_raw,
                )
                if best is None or key[:3] < best[:3]:
                    best = key
            raw_candidate_outer = _fit_predict(
                candidate,
                fit,
                outer_index,
                r3d,
                physics,
                log_target,
            )
            for calibration in CALIBRATION_METHODS:
                candidate_prediction = calibrate_positive(
                    inner_raw,
                    truth_fit,
                    raw_candidate_outer,
                    calibration,
                )
                candidate_outer_records.append(
                    {
                        "growth_run_id": held,
                        "true_sq_nm": targets[outer_index],
                        "raw_predicted_sq_nm": raw_candidate_outer,
                        "predicted_sq_nm": candidate_prediction,
                        "absolute_error_nm": abs(
                            candidate_prediction - targets[outer_index]
                        ),
                        "candidate": candidate.identifier,
                        "family": candidate.family,
                        "calibration": calibration,
                        "outer_target_used_for_training_or_selection": False,
                    }
                )
        assert best is not None
        score, _, calibration, candidate, inner_raw = best
        raw_outer = _fit_predict(
            candidate,
            fit,
            outer_index,
            r3d,
            physics,
            log_target,
        )
        predicted = calibrate_positive(
            inner_raw,
            truth_fit,
            raw_outer,
            calibration,
        )
        records.append(
            {
                "growth_run_id": held,
                "true_sq_nm": targets[outer_index],
                "raw_predicted_sq_nm": raw_outer,
                "predicted_sq_nm": predicted,
                "absolute_error_nm": abs(predicted - targets[outer_index]),
                "selected_candidate": candidate.identifier,
                "selected_family": candidate.family,
                "selected_calibration": calibration,
                "inner_endpoint_objective": score,
                "outer_target_used_for_training_or_selection": False,
                "outer_fit_growth_count": len(fit),
            }
        )
        print(
            f"[{outer_index + 1:02d}/{len(groups):02d}] {held} "
            f"truth={targets[outer_index]:.3f} pred={predicted:.3f} "
            f"{candidate.family}/{calibration}",
            flush=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(records)
    selections = pd.DataFrame(selection_records)
    result.to_csv(args.output / "nested_loo_predictions.csv", index=False)
    selections.to_csv(
        args.output / "nested_candidate_selection.csv",
        index=False,
    )
    candidate_outer = pd.DataFrame(candidate_outer_records)
    candidate_outer.to_csv(
        args.output / "fixed_candidate_loo_predictions.csv",
        index=False,
    )
    baseline = pd.read_csv(
        args.baseline_predictions,
        dtype={"growth_run_id": str},
    )
    baseline = baseline.loc[baseline["target"] == "Rq_nm"].copy()
    baseline = baseline.rename(
        columns={
            "true_target": "true_sq_nm",
            "predicted_target": "predicted_sq_nm",
        }
    )
    metrics = pd.DataFrame(
        [
            _metrics(baseline, "M15b_current"),
            _metrics(result, "M16_nested_endpoint_aware"),
        ]
    )
    metrics.to_csv(args.output / "baseline_vs_endpoint_metrics.csv", index=False)
    leaderboard = pd.DataFrame(
        [
            _metrics(rows, f"{candidate}/{calibration}")
            for (candidate, calibration), rows in candidate_outer.groupby(
                ["candidate", "calibration"]
            )
        ]
    ).sort_values(
        ["mae_nm", "smooth_mae_nm", "rough_mae_nm"]
    )
    leaderboard.to_csv(
        args.output / "fixed_candidate_leaderboard.csv",
        index=False,
    )
    manifest = {
        "experiment": "nested endpoint-aware scalar research",
        "growth_count": len(groups),
        "physics_features": PHYSICS_ENDPOINT_FEATURES,
        "candidate_count": len(candidates),
        "calibration_methods": list(CALIBRATION_METHODS),
        "outer_target_used_for_training_or_selection": False,
        "result": metrics.to_dict("records"),
    }
    (args.output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))
    print("\nTop fixed candidates:")
    print(leaderboard.head(15).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perturbation-embeddings", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
