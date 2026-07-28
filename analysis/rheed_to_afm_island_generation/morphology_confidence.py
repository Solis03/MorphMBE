from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from analysis.rheed_video_afm_story.common import repo_path, write_csv, write_json


METHOD = "M10_dense_island_spectral_pareto"
FEATURES = ["afm_prior_mahalanobis", "max_training_ssim"]
ALPHAS = [0.3, 1.0, 3.0, 10.0, 30.0]


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    scaler = StandardScaler().fit(train_x)
    model = Ridge(alpha=float(alpha)).fit(
        scaler.transform(train_x), train_y
    )
    return model.predict(scaler.transform(test_x))


def _loo_predictions(
    x: np.ndarray, y: np.ndarray, *, alpha: float
) -> np.ndarray:
    result = np.zeros(len(y), dtype=float)
    for held in range(len(y)):
        keep = np.arange(len(y)) != held
        result[held] = _fit_predict(
            x[keep], y[keep], x[held : held + 1], alpha=alpha
        )[0]
    return result


def _select_alpha(x: np.ndarray, y: np.ndarray) -> tuple[float, pd.DataFrame]:
    records = []
    for alpha in ALPHAS:
        prediction = _loo_predictions(x, y, alpha=alpha)
        records.append(
            {
                "alpha": alpha,
                "loo_mae_z": float(np.mean(np.abs(prediction - y))),
                "loo_rmse_z": float(
                    np.sqrt(np.mean(np.square(prediction - y)))
                ),
            }
        )
    table = pd.DataFrame(records).sort_values(["loo_mae_z", "alpha"])
    return float(table.iloc[0]["alpha"]), table


def _relative_confidence(
    predicted_error: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    # Smoothed survival percentile: smaller expected error -> larger index.
    return np.asarray(
        [
            100.0
            * (1.0 + float(np.sum(reference >= value)))
            / (len(reference) + 1.0)
            for value in predicted_error
        ],
        dtype=np.float32,
    )


def run(args: argparse.Namespace) -> None:
    report = repo_path(args.selected_report)
    output = repo_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    standard = pd.read_csv(
        report / "crossfit" / "standard_per_group.csv",
        dtype={"growth_run_id": str},
    )
    island = pd.read_csv(
        report / "crossfit" / "island_per_group.csv",
        dtype={"growth_run_id": str},
    )
    cross = standard.loc[standard["method"] == METHOD].merge(
        island.loc[island["method"] == METHOD],
        on=["growth_run_id", "method"],
        suffixes=("", "_island"),
    )
    x = cross[FEATURES].to_numpy(float)
    y = cross["island_feature_mae_z"].to_numpy(float)
    predictions = np.zeros(len(cross), dtype=float)
    upper = np.zeros(len(cross), dtype=float)
    selected_alphas = np.zeros(len(cross), dtype=float)
    for held in range(len(cross)):
        keep = np.arange(len(cross)) != held
        alpha, _ = _select_alpha(x[keep], y[keep])
        inner_prediction = _loo_predictions(x[keep], y[keep], alpha=alpha)
        residual = np.abs(inner_prediction - y[keep])
        conformal_radius = float(
            np.quantile(residual, 0.90, method="higher")
        )
        predictions[held] = _fit_predict(
            x[keep], y[keep], x[held : held + 1], alpha=alpha
        )[0]
        upper[held] = predictions[held] + conformal_radius
        selected_alphas[held] = alpha
    cross_output = pd.DataFrame(
        {
            "growth_run_id": cross["growth_run_id"],
            "predicted_island_error_z": predictions,
            "island_error_90_upper_z": upper,
            "realized_island_error_z": y,
            "morphology_confidence_index": _relative_confidence(
                predictions, predictions
            ),
            "selected_nested_ridge_alpha": selected_alphas,
            "afm_prior_mahalanobis": cross["afm_prior_mahalanobis"],
            "max_training_ssim": cross["max_training_ssim"],
        }
    )
    write_csv(cross_output, output / "morphology_confidence_crossfit.csv")

    full_alpha, full_cv = _select_alpha(x, y)
    full_loo = _loo_predictions(x, y, alpha=full_alpha)
    full_radius = float(
        np.quantile(np.abs(full_loo - y), 0.90, method="higher")
    )
    write_csv(full_cv, output / "morphology_confidence_ridge_cv.csv")
    validation_standard = pd.read_csv(
        report / "validation" / "standard" / "per_group_metrics.csv",
        dtype={"growth_run_id": str},
    )
    validation_island = pd.read_csv(
        report / "validation" / "island_per_group.csv",
        dtype={"growth_run_id": str},
    )
    validation = validation_standard.loc[
        validation_standard["method"] == METHOD
    ].merge(
        validation_island.loc[validation_island["method"] == METHOD],
        on=["growth_run_id", "method"],
        suffixes=("", "_island"),
    )
    validation_prediction = _fit_predict(
        x,
        y,
        validation[FEATURES].to_numpy(float),
        alpha=full_alpha,
    )
    validation_output = pd.DataFrame(
        {
            "growth_run_id": validation["growth_run_id"],
            "predicted_island_error_z": validation_prediction,
            "island_error_90_upper_z": validation_prediction + full_radius,
            "realized_island_error_z": validation["island_feature_mae_z"],
            "morphology_confidence_index": _relative_confidence(
                validation_prediction, predictions
            ),
            "selected_ridge_alpha": full_alpha,
            "afm_prior_mahalanobis": validation["afm_prior_mahalanobis"],
            "max_training_ssim": validation["max_training_ssim"],
        }
    )
    write_csv(
        validation_output, output / "morphology_confidence_validation.csv"
    )
    correlation = spearmanr(predictions, y)
    manifest = {
        "method": METHOD,
        "features_available_at_inference": FEATURES,
        "nested_alpha_candidates": ALPHAS,
        "crossfit_predicted_vs_realized_error_spearman": float(
            correlation.statistic
        ),
        "crossfit_predicted_vs_realized_error_pvalue": float(
            correlation.pvalue
        ),
        "crossfit_predicted_error_mae_z": float(
            np.mean(np.abs(predictions - y))
        ),
        "crossfit_90_upper_coverage": float(np.mean(y <= upper)),
        "full_training_alpha": full_alpha,
        "full_training_conformal_radius_z": full_radius,
        "confidence_is_probability": False,
        "confidence_definition": (
            "Smoothed survival percentile of strictly cross-fitted predicted "
            "island error; larger means lower expected morphology error."
        ),
        "historical_test_used": False,
        "validation_used_to_fit_calibrator": False,
    }
    write_json(manifest, output / "morphology_confidence_manifest.json")
    print(json.dumps(manifest, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-report", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
