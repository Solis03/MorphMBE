#!/usr/bin/env python3
"""Run strict 28-fold leave-one-out predictions for the full labeled cohort."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


THIS = Path(__file__).resolve()
PKG = THIS.parents[1]
REPO = next(
    parent
    for parent in THIS.parents
    if (parent / "publication_freeze" / "rheed_afm_single_frame_v1_2026-07-18").is_dir()
)
FREEZE = REPO / "publication_freeze/rheed_afm_single_frame_v1_2026-07-18"
OUTPUT = PKG / "predictions/leave_one_out_28"
MAIN_FIGURES = PKG / "figures/main"
SUPP_FIGURES = PKG / "figures/supplementary"
HISTORICAL_IDS = [
    "6022",
    "6028",
    "6029",
    "6033",
    "6047",
    "6048",
    "6056",
    "6057",
    "6062",
    "6063",
    "6070",
    "6072",
    "6078",
    "6080",
    "6081",
    "6082",
    "6084",
    "6085",
    "6090",
    "6094",
    "6095",
    "6099",
    "6101",
]
ADDED_TRAIN_IDS = ["6358", "6382"]
PROSPECTIVE_TEST_IDS = ["6342", "6389", "6390"]
EXTRA_IDS = ["6342", "6358", "6382", "6389", "6390"]
SAMPLE_IDS = HISTORICAL_IDS + EXTRA_IDS
TARGET_COLUMNS = [
    "T4_second_order_trimmed_mean",
    "T6_quality_weighted_second_order",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_inputs() -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    embedding_path = PKG / "models/encoder/combined_training_and_test_embeddings.npz"
    target_path = PKG / "models/quantitative_model/training_targets.csv"
    ensemble_path = (
        FREEZE
        / "models/quantitative_model/full_cohort_deployment/ensemble_definition.json"
    )
    embedding_bank = np.load(embedding_path, allow_pickle=False)
    training_embedding_ids = [
        str(value) for value in embedding_bank["training_sample_ids"].tolist()
    ]
    test_embedding_ids = [
        str(value).removeprefix("N")
        for value in embedding_bank["test_sample_ids"].tolist()
    ]
    if training_embedding_ids != HISTORICAL_IDS + ADDED_TRAIN_IDS:
        raise RuntimeError(f"Unexpected training embedding order: {training_embedding_ids}")
    if test_embedding_ids != PROSPECTIVE_TEST_IDS:
        raise RuntimeError(f"Unexpected test embedding order: {test_embedding_ids}")
    embedding_map = {
        sample_id: np.asarray(vector, dtype=np.float64)
        for sample_id, vector in zip(
            training_embedding_ids,
            embedding_bank["training_embeddings"],
            strict=True,
        )
    }
    embedding_map.update(
        {
            sample_id: np.asarray(vector, dtype=np.float64)
            for sample_id, vector in zip(
                test_embedding_ids,
                embedding_bank["test_embeddings"],
                strict=True,
            )
        }
    )
    embeddings = np.vstack([embedding_map[sample_id] for sample_id in SAMPLE_IDS])

    training_targets = pd.read_csv(target_path, dtype={"sample_id": str})
    extra_targets = pd.read_csv(
        PKG / "ground_truth_afm/sample_targets.csv",
        dtype={"sample_id": str},
    )
    extra_targets["sample_id"] = extra_targets["sample_id"].str.removeprefix("N")
    training_index = training_targets.set_index("sample_id")
    extra_index = extra_targets.set_index("sample_id")
    for sample_id in ADDED_TRAIN_IDS:
        difference = np.max(
            np.abs(
                training_index.loc[sample_id, TARGET_COLUMNS].to_numpy(float)
                - extra_index.loc[sample_id, TARGET_COLUMNS].to_numpy(float)
            )
        )
        if difference > 1e-12:
            raise RuntimeError(
                f"Added-sample target mismatch for {sample_id}: {difference}"
            )
    target_rows = []
    for sample_id in SAMPLE_IDS:
        source = extra_index if sample_id in EXTRA_IDS else training_index
        target_rows.append(
            {
                "sample_id": sample_id,
                **{
                    column: float(source.loc[sample_id, column])
                    for column in TARGET_COLUMNS
                },
            }
        )
    targets = pd.DataFrame(target_rows)
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Training embeddings contain non-finite values")
    if not np.isfinite(targets[TARGET_COLUMNS].to_numpy(float)).all():
        raise RuntimeError("Training targets contain non-finite values")
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    if ensemble.get("aggregation") != "median" or len(ensemble.get("members", [])) != 5:
        raise RuntimeError("Frozen ensemble definition is not the expected five-member median")
    return embeddings, targets, ensemble


def run_leave_one_out(
    embeddings: np.ndarray, targets: pd.DataFrame, ensemble: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_index = targets.set_index("sample_id")
    prediction_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for held_index, held_out_id in enumerate(SAMPLE_IDS):
        train_mask = np.arange(len(SAMPLE_IDS)) != held_index
        fold_training_ids = [
            sample_id for index, sample_id in enumerate(SAMPLE_IDS) if train_mask[index]
        ]
        X_train = embeddings[train_mask]
        X_test = embeddings[held_index : held_index + 1]
        member_values: list[float] = []

        for member in ensemble["members"]:
            target_variant = member["target_variant"]
            y_train = target_index.loc[fold_training_ids, target_variant].to_numpy(float)
            scaler = StandardScaler().fit(X_train)
            model = Ridge(alpha=1.0).fit(scaler.transform(X_train), y_train)
            predicted = float(model.predict(scaler.transform(X_test))[0])
            member_values.append(predicted)
            member_rows.append(
                {
                    "held_out_sample_id": held_out_id,
                    "member_name": member["name"],
                    "trial_id": member["trial_id"],
                    "target_variant": target_variant,
                    "predicted_rq_nm": predicted,
                    "training_sample_count": int(train_mask.sum()),
                    "held_out_absent_from_fit": held_out_id not in fold_training_ids,
                }
            )

        member_values_array = np.asarray(member_values, dtype=float)
        predicted = float(np.median(member_values_array))
        true_t4 = float(target_index.loc[held_out_id, TARGET_COLUMNS[0]])
        prediction_rows.append(
            {
                "sample_id": held_out_id,
                "display_sample_id": (
                    f"N{held_out_id}" if held_out_id in EXTRA_IDS else held_out_id
                ),
                "sample_source": (
                    "added_extra_five_training"
                    if held_out_id in ADDED_TRAIN_IDS
                    else (
                        "prospective_extra_five_prediction"
                        if held_out_id in PROSPECTIVE_TEST_IDS
                        else "frozen_historical"
                    )
                ),
                "ground_truth_T4_rq_nm": true_t4,
                "ground_truth_T6_rq_nm": float(
                    target_index.loc[held_out_id, TARGET_COLUMNS[1]]
                ),
                "leave_one_out_predicted_rq_nm": predicted,
                "leave_one_out_predicted_rq_nm_clipped_nonnegative": max(0.0, predicted),
                "residual_predicted_minus_true_nm": predicted - true_t4,
                "absolute_error_nm": abs(predicted - true_t4),
                "squared_error_nm2": (predicted - true_t4) ** 2,
                "member_q10_rq_nm": float(np.quantile(member_values_array, 0.10)),
                "member_q90_rq_nm": float(np.quantile(member_values_array, 0.90)),
                "member_min_rq_nm": float(np.min(member_values_array)),
                "member_max_rq_nm": float(np.max(member_values_array)),
                "ensemble_member_count": len(member_values_array),
                "training_sample_count": int(train_mask.sum()),
                "training_sample_ids": json.dumps(fold_training_ids),
                "held_out_absent_from_fit": held_out_id not in fold_training_ids,
                "prediction_role": "strict_leave_one_out_prediction",
            }
        )
        fold_rows.append(
            {
                "fold_index": held_index + 1,
                "held_out_sample_id": held_out_id,
                "training_sample_count": int(train_mask.sum()),
                "training_sample_ids": json.dumps(fold_training_ids),
                "held_out_absent_from_fit": held_out_id not in fold_training_ids,
                "scaler_fit_scope": "fold_training_rows_only",
                "ridge_fit_scope": "fold_training_rows_only",
            }
        )

    predictions = pd.DataFrame(prediction_rows)
    members = pd.DataFrame(member_rows)
    folds = pd.DataFrame(fold_rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT / "predictions.csv", index=False)
    members.to_csv(OUTPUT / "ensemble_member_predictions.csv", index=False)
    folds.to_csv(OUTPUT / "fold_manifest.csv", index=False)
    return predictions, members, folds


def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    covariance = float(np.mean((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred))))
    denominator = (
        float(np.var(y_true))
        + float(np.var(y_pred))
        + float(np.mean(y_true) - np.mean(y_pred)) ** 2
    )
    return float(2.0 * covariance / denominator)


def calculate_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    y_true = predictions["ground_truth_T4_rq_nm"].to_numpy(float)
    y_pred = predictions["leave_one_out_predicted_rq_nm"].to_numpy(float)
    y_clipped = predictions[
        "leave_one_out_predicted_rq_nm_clipped_nonnegative"
    ].to_numpy(float)
    pearson = pearsonr(y_true, y_pred)
    spearman = spearmanr(y_true, y_pred)
    kendall = kendalltau(y_true, y_pred)

    def basic_metrics(values: np.ndarray) -> dict[str, float]:
        return {
            "MAE_nm": float(mean_absolute_error(y_true, values)),
            "median_absolute_error_nm": float(np.median(np.abs(values - y_true))),
            "RMSE_nm": float(np.sqrt(mean_squared_error(y_true, values))),
            "R2": float(r2_score(y_true, values)),
            "mean_bias_nm": float(np.mean(values - y_true)),
        }

    metrics: dict[str, Any] = {
        "created_at": now(),
        "evaluation_design": "strict 28-fold leave-one-out; each prediction uses the other 27 labeled samples",
        "comparison_target": "T4_second_order_trimmed_mean",
        "sample_count": len(SAMPLE_IDS),
        "historical_sample_count": len(HISTORICAL_IDS),
        "extra_five_sample_count": len(EXTRA_IDS),
        "original_added_training_sample_count": len(ADDED_TRAIN_IDS),
        "original_prospective_test_sample_count": len(PROSPECTIVE_TEST_IDS),
        "raw_model_output": {
            **basic_metrics(y_pred),
            "Pearson_r": float(pearson.statistic),
            "Pearson_p": float(pearson.pvalue),
            "Spearman_rho": float(spearman.statistic),
            "Spearman_p": float(spearman.pvalue),
            "Kendall_tau": float(kendall.statistic),
            "Kendall_p": float(kendall.pvalue),
            "concordance_correlation_coefficient": concordance_correlation_coefficient(
                y_true, y_pred
            ),
        },
        "nonnegative_clipped_output": basic_metrics(y_clipped),
        "negative_raw_prediction_count": int(np.count_nonzero(y_pred < 0.0)),
        "algorithm": {
            "features": "E1_dino_keyframe",
            "per_fold_scaling": "StandardScaler fitted on 27 training rows only",
            "per_member_model": "Ridge(alpha=1.0) fitted on 27 training rows only",
            "ensemble_members": 5,
            "aggregation": "median",
        },
    }
    write_json(OUTPUT / "metrics.json", metrics)
    return metrics


def save_figure(fig: plt.Figure, root: Path, stem: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(root / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(root / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def padded_limits(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    lower = min(0.0, float(np.min(x)), float(np.min(y)))
    upper = max(float(np.max(x)), float(np.max(y)))
    padding = max(0.4, (upper - lower) * 0.08)
    return lower - padding, upper + padding


def annotate_points(ax: plt.Axes, predictions: pd.DataFrame) -> None:
    notable_ids = set(
        predictions.nlargest(5, "absolute_error_nm")["sample_id"].astype(str)
    ) | set(EXTRA_IDS)
    offsets = {
        "6022": (5, 9),
        "6081": (-5, 9),
        "6094": (5, 9),
        "6095": (-5, -11),
        "6099": (5, -11),
        "6342": (7, -11),
        "6358": (7, -11),
        "6382": (7, -11),
        "6389": (7, 9),
        "6390": (7, -11),
    }
    for row in predictions[predictions["sample_id"].isin(notable_ids)].to_dict("records"):
        x = float(row["ground_truth_T4_rq_nm"])
        y = float(row["leave_one_out_predicted_rq_nm"])
        horizontal, vertical = offsets.get(
            str(row["sample_id"]),
            (5, 9 if row["residual_predicted_minus_true_nm"] >= 0 else -11),
        )
        ax.annotate(
            str(row["display_sample_id"]),
            (x, y),
            xytext=(horizontal, vertical),
            textcoords="offset points",
            ha="left" if horizontal > 0 else "right",
            va="bottom" if vertical > 0 else "top",
            fontsize=6.0,
            color="#303030",
            annotation_clip=True,
        )


def make_scatter_figure(predictions: pd.DataFrame, metrics: dict[str, Any]) -> None:
    y_true = predictions["ground_truth_T4_rq_nm"].to_numpy(float)
    y_pred = predictions["leave_one_out_predicted_rq_nm"].to_numpy(float)
    low, high = padded_limits(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7.2, 6.7))
    lower_errors = y_pred - predictions["member_q10_rq_nm"].to_numpy(float)
    upper_errors = predictions["member_q90_rq_nm"].to_numpy(float) - y_pred
    categories = [
        ("frozen_historical", "#4C78A8", "o", 48, "Historical cohort (n=23)"),
        (
            "added_extra_five_training",
            "#59A14F",
            "D",
            95,
            "Original added train (N6358, N6382)",
        ),
        (
            "prospective_extra_five_prediction",
            "#F28E2B",
            "s",
            88,
            "Original prospective test (N6342, N6389, N6390)",
        ),
    ]
    for source, color, marker, size, label in categories:
        mask = predictions["sample_source"].eq(source).to_numpy()
        ax.errorbar(
            y_true[mask],
            y_pred[mask],
            yerr=np.vstack([lower_errors[mask], upper_errors[mask]]),
            fmt="none",
            ecolor=color,
            alpha=0.38,
            elinewidth=0.9,
            capsize=2.0,
            zorder=1,
        )
        ax.scatter(
            y_true[mask],
            y_pred[mask],
            s=size,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.94,
            label=label,
            zorder=3,
        )
    ax.plot([low, high], [low, high], color="#222222", linestyle="--", linewidth=1.1)
    regression = np.polyfit(y_true, y_pred, deg=1)
    regression_x = np.linspace(low, high, 200)
    ax.plot(
        regression_x,
        np.polyval(regression, regression_x),
        color="#E45756",
        linewidth=1.1,
        alpha=0.8,
        label="Linear trend",
    )
    annotate_points(ax, predictions)
    raw = metrics["raw_model_output"]
    metric_text = (
        f"LOO n = 28\n"
        f"MAE = {raw['MAE_nm']:.2f} nm\n"
        f"RMSE = {raw['RMSE_nm']:.2f} nm\n"
        f"$R^2$ = {raw['R2']:.2f}\n"
        f"Pearson $r$ = {raw['Pearson_r']:.2f}\n"
        f"Spearman $\\rho$ = {raw['Spearman_rho']:.2f}"
    )
    ax.text(
        0.035,
        0.965,
        metric_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.9},
    )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Ground-truth T4 Rq (nm)")
    ax.set_ylabel("Leave-one-out predicted Rq (nm)")
    ax.set_title("Strict leave-one-out prediction across all 28 labeled samples")
    ax.grid(color="#E8E8E8", linewidth=0.6)
    ax.legend(loc="lower right", fontsize=7.2, frameon=True)
    fig.tight_layout()
    save_figure(fig, MAIN_FIGURES, "Figure2_leave_one_out_prediction_scatter")


def make_diagnostic_figure(predictions: pd.DataFrame, metrics: dict[str, Any]) -> None:
    ordered = predictions.sort_values("residual_predicted_minus_true_nm").reset_index(drop=True)
    color_by_source = {
        "frozen_historical": "#4C78A8",
        "added_extra_five_training": "#59A14F",
        "prospective_extra_five_prediction": "#F28E2B",
    }
    colors = [color_by_source[source] for source in ordered["sample_source"].tolist()]
    residuals = ordered["residual_predicted_minus_true_nm"].to_numpy(float)
    y_positions = np.arange(len(ordered))

    fig = plt.figure(figsize=(12.8, 6.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.15], wspace=0.35)
    ax_scatter = fig.add_subplot(grid[0, 0])
    ax_residual = fig.add_subplot(grid[0, 1])

    y_true = predictions["ground_truth_T4_rq_nm"].to_numpy(float)
    y_pred = predictions["leave_one_out_predicted_rq_nm"].to_numpy(float)
    low, high = padded_limits(y_true, y_pred)
    for source, color, marker, label, size in [
        ("frozen_historical", "#4C78A8", "o", "Historical (n=23)", 45),
        ("added_extra_five_training", "#59A14F", "D", "Added train (n=2)", 76),
        (
            "prospective_extra_five_prediction",
            "#F28E2B",
            "s",
            "Prospective test (n=3)",
            70,
        ),
    ]:
        subset = predictions[predictions["sample_source"].eq(source)]
        ax_scatter.scatter(
            subset["ground_truth_T4_rq_nm"],
            subset["leave_one_out_predicted_rq_nm"],
            s=size,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            label=label,
            zorder=3,
        )
    ax_scatter.plot([low, high], [low, high], "k--", linewidth=1.0)
    ax_scatter.set_xlim(low, high)
    ax_scatter.set_ylim(low, high)
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.set_xlabel("Ground-truth T4 Rq (nm)")
    ax_scatter.set_ylabel("LOO predicted Rq (nm)")
    ax_scatter.set_title("A  Prediction versus ground truth", loc="left", fontweight="bold")
    ax_scatter.grid(color="#E8E8E8", linewidth=0.6)
    ax_scatter.legend(fontsize=7.5, loc="lower right")
    raw = metrics["raw_model_output"]
    ax_scatter.text(
        0.04,
        0.96,
        f"MAE {raw['MAE_nm']:.2f} nm\n$R^2$ {raw['R2']:.2f}\n$r$ {raw['Pearson_r']:.2f}",
        transform=ax_scatter.transAxes,
        va="top",
        fontsize=8.4,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D0D0D0"},
    )

    ax_residual.hlines(
        y_positions,
        np.minimum(0.0, residuals),
        np.maximum(0.0, residuals),
        color=colors,
        alpha=0.65,
        linewidth=1.2,
    )
    ax_residual.scatter(residuals, y_positions, color=colors, s=34, zorder=3)
    ax_residual.axvline(0.0, color="#222222", linestyle="--", linewidth=1.0)
    ax_residual.set_yticks(y_positions, ordered["display_sample_id"].tolist(), fontsize=7.2)
    ax_residual.set_xlabel("Residual: predicted − ground truth (nm)")
    ax_residual.set_title("B  Per-sample leave-one-out residuals", loc="left", fontweight="bold")
    ax_residual.grid(axis="x", color="#E8E8E8", linewidth=0.6)
    ax_residual.set_ylim(-0.8, len(ordered) - 0.2)

    fig.suptitle(
        "Strict 28-fold leave-one-out diagnostics (27 training samples per fold)",
        fontsize=13,
        y=1.01,
    )
    save_figure(fig, SUPP_FIGURES, "SuppFigure10_leave_one_out_diagnostics")


def write_provenance(
    predictions: pd.DataFrame,
    members: pd.DataFrame,
    folds: pd.DataFrame,
    ensemble: dict[str, Any],
) -> None:
    embedding_path = PKG / "models/encoder/combined_training_and_test_embeddings.npz"
    target_path = PKG / "models/quantitative_model/training_targets.csv"
    ensemble_path = (
        FREEZE
        / "models/quantitative_model/full_cohort_deployment/ensemble_definition.json"
    )
    provenance = {
        "created_at": now(),
        "evaluation_design": "strict leave-one-out across all 28 labeled samples",
        "fold_count": len(folds),
        "samples_per_fold_fit": 27,
        "sample_ids": SAMPLE_IDS,
        "historical_sample_ids": HISTORICAL_IDS,
        "extra_five_sample_ids": EXTRA_IDS,
        "original_added_training_sample_ids": ADDED_TRAIN_IDS,
        "original_prospective_test_sample_ids": PROSPECTIVE_TEST_IDS,
        "comparison_target": TARGET_COLUMNS[0],
        "fit_algorithm": "StandardScaler().fit(X_fold_train); Ridge(alpha=1.0).fit(scaled_X_fold_train, y_fold_train)",
        "aggregation": ensemble["aggregation"],
        "ensemble_members": ensemble["members"],
        "leakage_checks": {
            "all_held_out_samples_absent_from_fold_fit": bool(
                predictions["held_out_absent_from_fit"].all()
            ),
            "all_folds_have_27_training_samples": bool(
                folds["training_sample_count"].eq(27).all()
            ),
            "all_members_have_27_training_samples": bool(
                members["training_sample_count"].eq(27).all()
            ),
        },
        "input_files": [
            {"path": str(embedding_path.relative_to(REPO)), "sha256": sha256_file(embedding_path)},
            {"path": str(target_path.relative_to(REPO)), "sha256": sha256_file(target_path)},
            {
                "path": str(
                    (PKG / "ground_truth_afm/sample_targets.csv").relative_to(REPO)
                ),
                "sha256": sha256_file(PKG / "ground_truth_afm/sample_targets.csv"),
            },
            {"path": str(ensemble_path.relative_to(REPO)), "sha256": sha256_file(ensemble_path)},
        ],
        "outputs": {
            "predictions": str((OUTPUT / "predictions.csv").relative_to(REPO)),
            "member_predictions": str(
                (OUTPUT / "ensemble_member_predictions.csv").relative_to(REPO)
            ),
            "fold_manifest": str((OUTPUT / "fold_manifest.csv").relative_to(REPO)),
            "metrics": str((OUTPUT / "metrics.json").relative_to(REPO)),
            "scatter_figure": str(
                (
                    MAIN_FIGURES
                    / "Figure2_leave_one_out_prediction_scatter.png"
                ).relative_to(REPO)
            ),
        },
    }
    write_json(OUTPUT / "run_provenance.json", provenance)


def update_result_report(predictions: pd.DataFrame, metrics: dict[str, Any]) -> None:
    report_path = PKG / "report/result_summary.md"
    original = report_path.read_text(encoding="utf-8")
    marker = "\n## Strict leave-one-out evaluation"
    if marker in original:
        original = original.split(marker, 1)[0].rstrip() + "\n"
    raw = metrics["raw_model_output"]
    largest_errors = predictions.nlargest(5, "absolute_error_nm")
    lines = [
        "",
        "## Strict leave-one-out evaluation",
        "",
        "Each of all 28 labeled samples (the historical 23 plus all five extra samples) was predicted by refitting the unchanged five-member ensemble on the other 27 samples. The held-out sample was excluded from both `StandardScaler` and `Ridge` fitting in its fold.",
        "",
        "This is a post-hoc all-labeled-sample analysis. It is separate from the original three-sample prospective evaluation because N6342, N6389, and N6390 contribute labels to the training folds for other held-out samples.",
        "",
        "| Metric | Raw LOO ensemble output |",
        "|---|---:|",
        f"| MAE | {raw['MAE_nm']:.4f} nm |",
        f"| Median absolute error | {raw['median_absolute_error_nm']:.4f} nm |",
        f"| RMSE | {raw['RMSE_nm']:.4f} nm |",
        f"| R² | {raw['R2']:.4f} |",
        f"| Pearson r | {raw['Pearson_r']:.4f} |",
        f"| Spearman ρ | {raw['Spearman_rho']:.4f} |",
        "",
        "Largest absolute LOO errors:",
        "",
        "| Sample | Ground truth T4 | LOO prediction | Absolute error |",
        "|---|---:|---:|---:|",
    ]
    for row in largest_errors.to_dict("records"):
        lines.append(
            f"| {row['display_sample_id']} | {row['ground_truth_T4_rq_nm']:.4f} nm | "
            f"{row['leave_one_out_predicted_rq_nm']:.4f} nm | "
            f"{row['absolute_error_nm']:.4f} nm |"
        )
    lines += [
        "",
        "Paper figures: `figures/main/Figure2_leave_one_out_prediction_scatter.*` and `figures/supplementary/SuppFigure10_leave_one_out_diagnostics.*`.",
        "",
        "The complete per-sample table is `predictions/leave_one_out_28/predictions.csv`.",
    ]
    report_path.write_text(original.rstrip() + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def write_leave_one_out_report(predictions: pd.DataFrame, metrics: dict[str, Any]) -> None:
    raw = metrics["raw_model_output"]
    lines = [
        "# Strict leave-one-out prediction report",
        "",
        "All 28 labeled samples are evaluated with strict leave-one-out prediction. For each row, the displayed sample is absent from both `StandardScaler` fitting and all five `Ridge(alpha=1.0)` member fits; the other 27 samples form that fold's training set. The five member outputs are aggregated by their unchanged median rule.",
        "",
        "This is a post-hoc all-labeled-sample analysis and is kept separate from the original three-sample prospective test. In these LOO folds, N6342, N6389, and N6390 are allowed to train predictions for other samples, while each remains excluded from its own fold.",
        "",
        "The table and figures use raw model outputs to preserve the frozen algorithm. Negative Rq predictions are flagged in the CSV; a separate nonnegative-clipped column is supplied but is not used for the primary metrics.",
        "",
        "## Cohort-level metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| n | {metrics['sample_count']} |",
        f"| MAE | {raw['MAE_nm']:.4f} nm |",
        f"| Median absolute error | {raw['median_absolute_error_nm']:.4f} nm |",
        f"| RMSE | {raw['RMSE_nm']:.4f} nm |",
        f"| Mean bias | {raw['mean_bias_nm']:.4f} nm |",
        f"| R² | {raw['R2']:.4f} |",
        f"| Pearson r | {raw['Pearson_r']:.4f} (p={raw['Pearson_p']:.4f}) |",
        f"| Spearman ρ | {raw['Spearman_rho']:.4f} (p={raw['Spearman_p']:.4f}) |",
        f"| Kendall τ | {raw['Kendall_tau']:.4f} (p={raw['Kendall_p']:.4f}) |",
        "",
        "## Per-sample predictions",
        "",
        "| Sample | Source | Ground truth T4 | LOO prediction | Residual | Absolute error | Member q10–q90 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in predictions.to_dict("records"):
        source = {
            "frozen_historical": "Historical",
            "added_extra_five_training": "Original added train",
            "prospective_extra_five_prediction": "Original prospective test",
        }[row["sample_source"]]
        lines.append(
            f"| {row['display_sample_id']} | {source} | "
            f"{row['ground_truth_T4_rq_nm']:.4f} | "
            f"{row['leave_one_out_predicted_rq_nm']:.4f} | "
            f"{row['residual_predicted_minus_true_nm']:+.4f} | "
            f"{row['absolute_error_nm']:.4f} | "
            f"{row['member_q10_rq_nm']:.4f}–{row['member_q90_rq_nm']:.4f} |"
        )
    lines += [
        "",
        "## Figures and machine-readable outputs",
        "",
        "- `figures/main/Figure2_leave_one_out_prediction_scatter.*`",
        "- `figures/supplementary/SuppFigure10_leave_one_out_diagnostics.*`",
        "- `predictions/leave_one_out_28/predictions.csv`",
        "- `predictions/leave_one_out_28/ensemble_member_predictions.csv`",
        "- `predictions/leave_one_out_28/fold_manifest.csv`",
        "- `predictions/leave_one_out_28/metrics.json`",
    ]
    path = PKG / "report/leave_one_out_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_manifest() -> None:
    manifest = PKG / "provenance/MANIFEST.sha256"
    rows = []
    for path in sorted(PKG.rglob("*")):
        if path.is_file() and path != manifest:
            rows.append(f"{sha256_file(path)}  {path.relative_to(PKG)}")
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    embeddings, targets, ensemble = load_inputs()
    predictions, members, folds = run_leave_one_out(embeddings, targets, ensemble)
    metrics = calculate_metrics(predictions)
    make_scatter_figure(predictions, metrics)
    make_diagnostic_figure(predictions, metrics)
    write_provenance(predictions, members, folds, ensemble)
    update_result_report(predictions, metrics)
    write_leave_one_out_report(predictions, metrics)
    refresh_manifest()
    print(
        json.dumps(
            {
                "status": "ok",
                "sample_count": len(predictions),
                "training_sample_count_per_fold": 27,
                "metrics": metrics["raw_model_output"],
                "scatter_figure": str(
                    MAIN_FIGURES / "Figure2_leave_one_out_prediction_scatter.png"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
