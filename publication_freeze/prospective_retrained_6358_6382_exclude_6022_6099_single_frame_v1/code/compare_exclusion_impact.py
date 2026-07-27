#!/usr/bin/env python3
"""Compare the 6022/6099 exclusion experiment with the original experiment."""

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


THIS = Path(__file__).resolve()
NEW = THIS.parents[1]
REPO = next(
    parent
    for parent in THIS.parents
    if (parent / "publication_freeze" / "rheed_afm_single_frame_v1_2026-07-18").is_dir()
)
OLD = REPO / "publication_freeze/prospective_retrained_6358_6382_single_frame_v1"
OUTPUT = NEW / "comparison"
FIGURES = NEW / "figures/comparison"
EXCLUDED_IDS = ["6022", "6099"]
EXTRA_DISPLAY_IDS = ["N6342", "N6358", "N6382", "N6389", "N6390"]
TEST_IDS = ["N6342", "N6389", "N6390"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pearson = pearsonr(y_true, y_pred)
    spearman = spearmanr(y_true, y_pred)
    kendall = kendalltau(y_true, y_pred)
    true_mean = float(np.mean(y_true))
    pred_mean = float(np.mean(y_pred))
    covariance = float(np.mean((y_true - true_mean) * (y_pred - pred_mean)))
    ccc_denominator = float(np.var(y_true) + np.var(y_pred) + (true_mean - pred_mean) ** 2)
    return {
        "MAE_nm": float(mean_absolute_error(y_true, y_pred)),
        "median_absolute_error_nm": float(np.median(np.abs(y_pred - y_true))),
        "RMSE_nm": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mean_bias_nm": float(np.mean(y_pred - y_true)),
        "R2": float(r2_score(y_true, y_pred)),
        "Pearson_r": float(pearson.statistic),
        "Spearman_rho": float(spearman.statistic),
        "Kendall_tau": float(kendall.statistic),
        "CCC": float(2.0 * covariance / ccc_denominator),
    }


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=500, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def load_targets() -> pd.DataFrame:
    targets = pd.read_csv(
        NEW / "ground_truth_afm/sample_targets.csv",
        dtype={"sample_id": str},
    )
    return targets.set_index("sample_id")


def compare_predictions(
    targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old_baseline = pd.read_csv(
        OLD / "predictions/frozen_23_baseline/predictions.csv",
        dtype={"sample_id": str},
    ).set_index("sample_id")
    new_baseline = pd.read_csv(
        NEW / "predictions/reduced_21_baseline/predictions.csv",
        dtype={"sample_id": str},
    ).set_index("sample_id")
    baseline_rows = []
    for sample_id in EXTRA_DISPLAY_IDS:
        true = float(targets.loc[sample_id, "T4_second_order_trimmed_mean"])
        old_value = float(old_baseline.loc[sample_id, "predicted_rq_nm"])
        new_value = float(new_baseline.loc[sample_id, "predicted_rq_nm"])
        baseline_rows.append(
            {
                "sample_id": sample_id,
                "ground_truth_T4_rq_nm": true,
                "original_frozen23_prediction_nm": old_value,
                "excluded_reduced21_prediction_nm": new_value,
                "original_absolute_error_nm": abs(old_value - true),
                "excluded_absolute_error_nm": abs(new_value - true),
                "absolute_error_change_excluded_minus_original_nm": abs(
                    new_value - true
                )
                - abs(old_value - true),
            }
        )
    baseline = pd.DataFrame(baseline_rows)

    old_retrained = pd.read_csv(
        OLD / "predictions/retrained_25/predictions.csv",
        dtype={"sample_id": str},
    ).set_index("sample_id")
    new_retrained = pd.read_csv(
        NEW / "predictions/retrained_23/predictions.csv",
        dtype={"sample_id": str},
    ).set_index("sample_id")
    retrained_rows = []
    for sample_id in TEST_IDS:
        true = float(targets.loc[sample_id, "T4_second_order_trimmed_mean"])
        old_value = float(old_retrained.loc[sample_id, "predicted_rq_nm"])
        new_value = float(new_retrained.loc[sample_id, "predicted_rq_nm"])
        retrained_rows.append(
            {
                "sample_id": sample_id,
                "ground_truth_T4_rq_nm": true,
                "original_retrained25_prediction_nm": old_value,
                "excluded_retrained23_prediction_nm": new_value,
                "original_absolute_error_nm": abs(old_value - true),
                "excluded_absolute_error_nm": abs(new_value - true),
                "absolute_error_change_excluded_minus_original_nm": abs(
                    new_value - true
                )
                - abs(old_value - true),
            }
        )
    retrained = pd.DataFrame(retrained_rows)

    old_loo = pd.read_csv(
        OLD / "predictions/leave_one_out_28/predictions.csv",
        dtype={"sample_id": str},
    )
    new_loo = pd.read_csv(
        NEW / "predictions/leave_one_out_26/predictions.csv",
        dtype={"sample_id": str},
    )
    old_loo_common = old_loo[old_loo["sample_id"].isin(new_loo["sample_id"])].copy()
    old_loo_index = old_loo_common.set_index("sample_id")
    loo_rows = []
    for row in new_loo.to_dict("records"):
        sample_id = row["sample_id"]
        old_row = old_loo_index.loc[sample_id]
        true = float(row["ground_truth_T4_rq_nm"])
        old_value = float(old_row["leave_one_out_predicted_rq_nm"])
        new_value = float(row["leave_one_out_predicted_rq_nm"])
        loo_rows.append(
            {
                "sample_id": sample_id,
                "display_sample_id": row["display_sample_id"],
                "ground_truth_T4_rq_nm": true,
                "original_28fold_prediction_nm": old_value,
                "excluded_26fold_prediction_nm": new_value,
                "original_absolute_error_nm": abs(old_value - true),
                "excluded_absolute_error_nm": abs(new_value - true),
                "absolute_error_change_excluded_minus_original_nm": abs(
                    new_value - true
                )
                - abs(old_value - true),
            }
        )
    loo = pd.DataFrame(loo_rows)

    old_hoo = pd.read_csv(
        OLD / "predictions/held_one_out_afm_28/retrieval_results.csv",
        dtype={"sample_id": str, "source_sample_id": str},
    )
    new_hoo = pd.read_csv(
        NEW / "predictions/held_one_out_afm_26/retrieval_results.csv",
        dtype={"sample_id": str, "source_sample_id": str},
    )
    old_hoo_index = old_hoo.set_index("sample_id")
    hoo_rows = []
    for row in new_hoo.to_dict("records"):
        sample_id = row["sample_id"]
        old_row = old_hoo_index.loc[sample_id]
        true = float(row["ground_truth_selected_rq_nm"])
        old_value = float(old_row["rendered_physical_rq_nm"])
        new_value = float(row["rendered_physical_rq_nm"])
        hoo_rows.append(
            {
                "sample_id": sample_id,
                "display_sample_id": row["display_sample_id"],
                "selected_ground_truth_rq_nm": true,
                "original_rendered_rq_nm": old_value,
                "excluded_rendered_rq_nm": new_value,
                "original_absolute_error_nm": abs(old_value - true),
                "excluded_absolute_error_nm": abs(new_value - true),
                "absolute_error_change_excluded_minus_original_nm": abs(
                    new_value - true
                )
                - abs(old_value - true),
                "original_source_sample_id": str(old_row["source_sample_id"]),
                "excluded_source_sample_id": str(row["source_sample_id"]),
                "retrieved_source_changed": str(old_row["source_sample_id"])
                != str(row["source_sample_id"])
                or str(old_row["source_afm_file_id"])
                != str(row["source_afm_file_id"]),
            }
        )
    hoo = pd.DataFrame(hoo_rows)
    return baseline, retrained, loo, hoo


def build_metric_comparison(
    baseline: pd.DataFrame,
    retrained: pd.DataFrame,
    loo: pd.DataFrame,
    hoo: pd.DataFrame,
) -> pd.DataFrame:
    definitions = [
        (
            "Historical baseline on extra five",
            baseline,
            "ground_truth_T4_rq_nm",
            "original_frozen23_prediction_nm",
            "excluded_reduced21_prediction_nm",
        ),
        (
            "Retrained prospective test",
            retrained,
            "ground_truth_T4_rq_nm",
            "original_retrained25_prediction_nm",
            "excluded_retrained23_prediction_nm",
        ),
        (
            "LOO common retained cohort",
            loo,
            "ground_truth_T4_rq_nm",
            "original_28fold_prediction_nm",
            "excluded_26fold_prediction_nm",
        ),
        (
            "Held-one-out AFM displayed Rq common cohort",
            hoo,
            "selected_ground_truth_rq_nm",
            "original_rendered_rq_nm",
            "excluded_rendered_rq_nm",
        ),
    ]
    rows = []
    for label, frame, true_col, old_col, new_col in definitions:
        y_true = frame[true_col].to_numpy(float)
        old_metrics = metric_dict(y_true, frame[old_col].to_numpy(float))
        new_metrics = metric_dict(y_true, frame[new_col].to_numpy(float))
        rows.append(
            {
                "experiment": label,
                "common_sample_count": len(frame),
                **{
                    f"original_{key}": value
                    for key, value in old_metrics.items()
                },
                **{
                    f"excluded_{key}": value
                    for key, value in new_metrics.items()
                },
                "MAE_change_excluded_minus_original_nm": new_metrics["MAE_nm"]
                - old_metrics["MAE_nm"],
                "RMSE_change_excluded_minus_original_nm": new_metrics[
                    "RMSE_nm"
                ]
                - old_metrics["RMSE_nm"],
                "R2_change_excluded_minus_original": new_metrics["R2"]
                - old_metrics["R2"],
                "Pearson_r_change_excluded_minus_original": new_metrics["Pearson_r"]
                - old_metrics["Pearson_r"],
                "Spearman_rho_change_excluded_minus_original": new_metrics["Spearman_rho"]
                - old_metrics["Spearman_rho"],
                "Kendall_tau_change_excluded_minus_original": new_metrics["Kendall_tau"]
                - old_metrics["Kendall_tau"],
                "CCC_change_excluded_minus_original": new_metrics["CCC"]
                - old_metrics["CCC"],
            }
        )
    return pd.DataFrame(rows)


def build_full_cohort_comparison() -> pd.DataFrame:
    """Describe the raw old-28/new-26 summaries, which are not a fair paired test."""
    definitions = [
        (
            "LOO full available cohort",
            OLD / "predictions/leave_one_out_28/predictions.csv",
            NEW / "predictions/leave_one_out_26/predictions.csv",
            "ground_truth_T4_rq_nm",
            "leave_one_out_predicted_rq_nm",
        ),
        (
            "Held-one-out AFM full available cohort",
            OLD / "predictions/held_one_out_afm_28/retrieval_results.csv",
            NEW / "predictions/held_one_out_afm_26/retrieval_results.csv",
            "ground_truth_selected_rq_nm",
            "rendered_physical_rq_nm",
        ),
    ]
    rows = []
    for label, old_path, new_path, true_col, prediction_col in definitions:
        old_frame = pd.read_csv(old_path)
        new_frame = pd.read_csv(new_path)
        old_metrics = metric_dict(
            old_frame[true_col].to_numpy(float),
            old_frame[prediction_col].to_numpy(float),
        )
        new_metrics = metric_dict(
            new_frame[true_col].to_numpy(float),
            new_frame[prediction_col].to_numpy(float),
        )
        rows.append(
            {
                "experiment": label,
                "original_sample_count": len(old_frame),
                "excluded_sample_count": len(new_frame),
                **{f"original_{key}": value for key, value in old_metrics.items()},
                **{f"excluded_{key}": value for key, value in new_metrics.items()},
                "MAE_change_excluded_minus_original_nm": new_metrics["MAE_nm"]
                - old_metrics["MAE_nm"],
                "RMSE_change_excluded_minus_original_nm": new_metrics["RMSE_nm"]
                - old_metrics["RMSE_nm"],
            }
        )
    return pd.DataFrame(rows)


def compare_retrieval_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    old_prospective = pd.read_csv(
        OLD / "predictions/retrained_25/retrieval/retrieval_results.csv",
        dtype={"sample_id": str, "source_sample_id": str},
    ).set_index("sample_id")
    new_prospective = pd.read_csv(
        NEW / "predictions/retrained_23/retrieval/retrieval_results.csv",
        dtype={"sample_id": str, "source_sample_id": str},
    ).set_index("sample_id")
    rows = []
    for sample_id in TEST_IDS:
        old_row = old_prospective.loc[sample_id]
        new_row = new_prospective.loc[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "original_source_sample_id": old_row["source_sample_id"],
                "original_source_afm_file_id": old_row["source_afm_file_id"],
                "excluded_source_sample_id": new_row["source_sample_id"],
                "excluded_source_afm_file_id": new_row["source_afm_file_id"],
                "source_changed": (
                    old_row["source_sample_id"] != new_row["source_sample_id"]
                    or old_row["source_afm_file_id"]
                    != new_row["source_afm_file_id"]
                ),
            }
        )
    prospective = pd.DataFrame(rows)

    old_baseline = pd.read_csv(
        OLD
        / "predictions/frozen_23_baseline/retrieval/retrieval_results.csv",
        dtype={"sample_id": str, "source_sample_id": str},
    ).set_index("sample_id")
    new_baseline = pd.read_csv(
        NEW
        / "predictions/reduced_21_baseline/retrieval/retrieval_results.csv",
        dtype={"sample_id": str, "source_sample_id": str},
    ).set_index("sample_id")
    baseline_rows = []
    for sample_id in EXTRA_DISPLAY_IDS:
        old_row = old_baseline.loc[sample_id]
        new_row = new_baseline.loc[sample_id]
        baseline_rows.append(
            {
                "sample_id": sample_id,
                "original_source_sample_id": old_row["source_sample_id"],
                "original_source_afm_file_id": old_row["source_afm_file_id"],
                "excluded_source_sample_id": new_row["source_sample_id"],
                "excluded_source_afm_file_id": new_row["source_afm_file_id"],
                "source_changed": (
                    old_row["source_sample_id"] != new_row["source_sample_id"]
                    or old_row["source_afm_file_id"]
                    != new_row["source_afm_file_id"]
                ),
            }
        )
    return prospective, pd.DataFrame(baseline_rows)


def compare_models() -> pd.DataFrame:
    rows = []
    for old_path in sorted(
        (OLD / "models/quantitative_model").glob("model_*.npz")
    ):
        new_path = NEW / "models/quantitative_model" / old_path.name
        old_model = np.load(old_path, allow_pickle=False)
        new_model = np.load(new_path, allow_pickle=False)
        rows.append(
            {
                "member_name": old_path.stem,
                "coefficient_L2_change": float(
                    np.linalg.norm(new_model["coef"] - old_model["coef"])
                ),
                "coefficient_relative_L2_change": float(
                    np.linalg.norm(new_model["coef"] - old_model["coef"])
                    / max(np.linalg.norm(old_model["coef"]), 1e-12)
                ),
                "intercept_change_nm": float(
                    new_model["intercept"] - old_model["intercept"]
                ),
                "feature_mean_L2_change": float(
                    np.linalg.norm(
                        new_model["feature_mean"] - old_model["feature_mean"]
                    )
                ),
                "original_training_count": len(old_model["training_sample_ids"]),
                "excluded_training_count": len(new_model["training_sample_ids"]),
            }
        )
    return pd.DataFrame(rows)


def make_impact_summary_figure(
    retrained: pd.DataFrame,
    loo: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.5))
    old_color = "#4C78A8"
    new_color = "#E45756"
    x = np.arange(len(retrained))

    ax = axes[0, 0]
    for index, row in retrained.iterrows():
        ax.plot(
            [index - 0.14, index + 0.14],
            [
                row["original_retrained25_prediction_nm"],
                row["excluded_retrained23_prediction_nm"],
            ],
            color="#B0B0B0",
            linewidth=1,
        )
    ax.scatter(
        x - 0.14,
        retrained["original_retrained25_prediction_nm"],
        s=70,
        color=old_color,
        label="Original retrained 25",
    )
    ax.scatter(
        x + 0.14,
        retrained["excluded_retrained23_prediction_nm"],
        s=70,
        color=new_color,
        label="Excluded retrained 23",
    )
    ax.scatter(
        x,
        retrained["ground_truth_T4_rq_nm"],
        s=75,
        marker="D",
        color="#222222",
        label="Ground truth",
    )
    ax.set_xticks(x, retrained["sample_id"])
    ax.set_ylabel("Rq (nm)")
    ax.set_title("A  Prospective predictions", loc="left", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", color="#E8E8E8")

    ax = axes[0, 1]
    width = 0.34
    ax.bar(
        x - width / 2,
        retrained["original_absolute_error_nm"],
        width,
        color=old_color,
        label="Original",
    )
    ax.bar(
        x + width / 2,
        retrained["excluded_absolute_error_nm"],
        width,
        color=new_color,
        label="After exclusion",
    )
    ax.set_xticks(x, retrained["sample_id"])
    ax.set_ylabel("Absolute error (nm)")
    ax.set_title(
        "B  Prospective absolute errors", loc="left", fontweight="bold"
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", color="#E8E8E8")

    ax = axes[1, 0]
    metric_labels = [
        "Historical\nbaseline (n=5)",
        "Prospective\nretrained (n=3)",
        "LOO\ncommon 26",
        "AFM HOO\ncommon 26",
    ]
    metric_x = np.arange(len(metrics))
    ax.bar(
        metric_x - width / 2,
        metrics["original_MAE_nm"],
        width,
        color=old_color,
        label="Original",
    )
    ax.bar(
        metric_x + width / 2,
        metrics["excluded_MAE_nm"],
        width,
        color=new_color,
        label="After exclusion",
    )
    ax.set_xticks(metric_x, metric_labels)
    ax.set_ylabel("MAE (nm)")
    ax.set_title(
        "C  Fair common-cohort MAE comparison",
        loc="left",
        fontweight="bold",
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", color="#E8E8E8")

    ax = axes[1, 1]
    ordered = loo.sort_values(
        "absolute_error_change_excluded_minus_original_nm"
    ).reset_index(drop=True)
    changes = ordered[
        "absolute_error_change_excluded_minus_original_nm"
    ].to_numpy(float)
    colors = np.where(changes < 0, "#59A14F", "#E45756")
    ax.bar(
        np.arange(len(ordered)),
        changes,
        color=colors,
        width=0.78,
    )
    ax.axhline(0, color="#222222", linewidth=0.9)
    ax.set_xticks(
        np.arange(len(ordered)),
        ordered["display_sample_id"],
        rotation=90,
        fontsize=7,
    )
    ax.set_ylabel("Absolute-error change (nm)\nafter exclusion − original")
    improved = int(np.count_nonzero(changes < 0))
    ax.set_title(
        f"D  LOO impact on retained samples ({improved}/26 improved)",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="y", color="#E8E8E8")

    fig.suptitle(
        "Impact of excluding samples 6022 and 6099",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, "Figure4_exclusion_impact_summary")


def make_loo_before_after_figure(
    loo: pd.DataFrame, metrics: pd.DataFrame
) -> None:
    true = loo["ground_truth_T4_rq_nm"].to_numpy(float)
    old = loo["original_28fold_prediction_nm"].to_numpy(float)
    new = loo["excluded_26fold_prediction_nm"].to_numpy(float)
    low = min(0.0, true.min(), old.min(), new.min()) - 0.5
    high = max(true.max(), old.max(), new.max()) + 0.5
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharex=True, sharey=True)
    for ax, values, title, metric_row, color in [
        (
            axes[0],
            old,
            "Original model on common retained 26",
            metrics.iloc[2],
            "#4C78A8",
        ),
        (
            axes[1],
            new,
            "After excluding 6022 and 6099",
            metrics.iloc[2],
            "#E45756",
        ),
    ]:
        ax.scatter(true, values, s=46, color=color, alpha=0.85)
        ax.plot([low, high], [low, high], "k--", linewidth=1)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#E8E8E8")
        ax.set_xlabel("Ground-truth T4 Rq (nm)")
        ax.set_title(title)
    axes[0].set_ylabel("Leave-one-out predicted Rq (nm)")
    axes[0].text(
        0.04,
        0.96,
        f"MAE {metrics.iloc[2]['original_MAE_nm']:.2f} nm\n"
        f"RMSE {metrics.iloc[2]['original_RMSE_nm']:.2f} nm",
        transform=axes[0].transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0"},
    )
    axes[1].text(
        0.04,
        0.96,
        f"MAE {metrics.iloc[2]['excluded_MAE_nm']:.2f} nm\n"
        f"RMSE {metrics.iloc[2]['excluded_RMSE_nm']:.2f} nm",
        transform=axes[1].transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#D0D0D0"},
    )
    fig.suptitle(
        "Leave-one-out comparison on the identical retained 26-sample cohort",
        fontsize=14,
    )
    fig.tight_layout()
    save_figure(fig, "Figure5_leave_one_out_common26_before_after")


def write_report(
    baseline: pd.DataFrame,
    retrained: pd.DataFrame,
    loo: pd.DataFrame,
    hoo: pd.DataFrame,
    metrics: pd.DataFrame,
    full_cohort_metrics: pd.DataFrame,
    prospective_sources: pd.DataFrame,
    baseline_sources: pd.DataFrame,
    model_changes: pd.DataFrame,
) -> None:
    metric_index = metrics.set_index("experiment")
    prospective_metric = metric_index.loc["Retrained prospective test"]
    loo_metric = metric_index.loc["LOO common retained cohort"]
    hoo_metric = metric_index.loc[
        "Held-one-out AFM displayed Rq common cohort"
    ]
    loo_improved = int(
        (
            loo["absolute_error_change_excluded_minus_original_nm"] < 0
        ).sum()
    )
    lines = [
        "# Impact of excluding samples 6022 and 6099",
        "",
        "The algorithms, feature definitions, ensemble members, AFM preprocessing, and A3 retrieval ranking are unchanged. Only the two sample groups 6022 and 6099 are removed.",
        "",
        "## Data-count changes",
        "",
        "| Experiment | Original | After exclusion |",
        "|---|---:|---:|",
        "| Historical baseline training | 23 | 21 |",
        "| Prospective retraining | 25 | 23 |",
        "| All-labeled LOO cohort | 28 | 26 |",
        "| LOO training rows per fold | 27 | 25 |",
        "| AFM held-one-out source groups per fold | 27 | 25 |",
        "",
        "## Main findings",
        "",
        f"- Three-sample prospective MAE changed from {prospective_metric['original_MAE_nm']:.4f} to {prospective_metric['excluded_MAE_nm']:.4f} nm (Δ {prospective_metric['MAE_change_excluded_minus_original_nm']:+.4f} nm).",
        f"- On the identical retained 26-sample LOO cohort, MAE changed from {loo_metric['original_MAE_nm']:.4f} to {loo_metric['excluded_MAE_nm']:.4f} nm (Δ {loo_metric['MAE_change_excluded_minus_original_nm']:+.4f} nm); {loo_improved}/26 individual samples improved.",
        f"- On the identical retained 26-sample AFM held-one-out cohort, displayed-map Rq MAE changed from {hoo_metric['original_MAE_nm']:.4f} to {hoo_metric['excluded_MAE_nm']:.4f} nm (Δ {hoo_metric['MAE_change_excluded_minus_original_nm']:+.4f} nm).",
        "- The lower full-cohort LOO and AFM-HOO MAEs are partly caused by removing two unusually high-error samples. The common-26 comparison is the fair test of model changes on retained data.",
        "",
        "## Prospective prediction changes",
        "",
        "| Sample | Ground truth | Original retrained 25 | Excluded retrained 23 | Original abs. error | New abs. error | Error change |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in retrained.to_dict("records"):
        lines.append(
            f"| {row['sample_id']} | {row['ground_truth_T4_rq_nm']:.4f} | "
            f"{row['original_retrained25_prediction_nm']:.4f} | "
            f"{row['excluded_retrained23_prediction_nm']:.4f} | "
            f"{row['original_absolute_error_nm']:.4f} | "
            f"{row['excluded_absolute_error_nm']:.4f} | "
            f"{row['absolute_error_change_excluded_minus_original_nm']:+.4f} |"
        )
    lines += [
        "",
        "## Aggregate fair comparisons",
        "",
        "| Experiment | n | Original MAE | New MAE | ΔMAE | Original RMSE | New RMSE | ΔRMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.to_dict("records"):
        lines.append(
            f"| {row['experiment']} | {row['common_sample_count']} | "
            f"{row['original_MAE_nm']:.4f} | {row['excluded_MAE_nm']:.4f} | "
            f"{row['MAE_change_excluded_minus_original_nm']:+.4f} | "
            f"{row['original_RMSE_nm']:.4f} | {row['excluded_RMSE_nm']:.4f} | "
            f"{row['RMSE_change_excluded_minus_original_nm']:+.4f} |"
        )
    lines += [
        "",
        "## Detailed common-cohort metrics",
        "",
        "| Experiment | Version | Median AE | Bias | R² | Pearson r | Spearman ρ | Kendall τ | CCC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics[
        metrics["experiment"].isin(
            [
                "LOO common retained cohort",
                "Held-one-out AFM displayed Rq common cohort",
            ]
        )
    ].to_dict("records"):
        for prefix, version in [("original", "Original"), ("excluded", "After exclusion")]:
            lines.append(
                f"| {row['experiment']} | {version} | "
                f"{row[f'{prefix}_median_absolute_error_nm']:.4f} | "
                f"{row[f'{prefix}_mean_bias_nm']:+.4f} | "
                f"{row[f'{prefix}_R2']:.4f} | "
                f"{row[f'{prefix}_Pearson_r']:.4f} | "
                f"{row[f'{prefix}_Spearman_rho']:.4f} | "
                f"{row[f'{prefix}_Kendall_tau']:.4f} | "
                f"{row[f'{prefix}_CCC']:.4f} |"
            )
    lines += [
        "",
        "## Raw full-cohort summaries (different n; descriptive only)",
        "",
        "| Experiment | Original n | New n | Original MAE | New MAE | ΔMAE | Original RMSE | New RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in full_cohort_metrics.to_dict("records"):
        lines.append(
            f"| {row['experiment']} | {row['original_sample_count']} | "
            f"{row['excluded_sample_count']} | {row['original_MAE_nm']:.4f} | "
            f"{row['excluded_MAE_nm']:.4f} | "
            f"{row['MAE_change_excluded_minus_original_nm']:+.4f} | "
            f"{row['original_RMSE_nm']:.4f} | {row['excluded_RMSE_nm']:.4f} |"
        )
    lines += [
        "",
        "## Retrieval-source changes",
        "",
        f"Prospective retrained retrieval source changed for {int(prospective_sources['source_changed'].sum())}/3 samples.",
        f"Reduced historical-baseline retrieval source changed for {int(baseline_sources['source_changed'].sum())}/5 samples.",
        f"AFM held-one-out retrieval source changed for {int(hoo['retrieved_source_changed'].sum())}/26 retained samples.",
        "",
        "## Model-parameter sensitivity",
        "",
        f"All {len(model_changes)} ensemble members were refit on 23 rather than 25 rows. Full coefficient/intercept changes are in `comparison/model_parameter_changes.csv`.",
        "",
        "## Figures",
        "",
        "- `figures/comparison/Figure4_exclusion_impact_summary.*`",
        "- `figures/comparison/Figure5_leave_one_out_common26_before_after.*`",
    ]
    path = NEW / "report/exclusion_impact_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_readme_and_master_report(metrics: pd.DataFrame) -> None:
    metric_index = metrics.set_index("experiment")
    prospective = metric_index.loc["Retrained prospective test"]
    loo = metric_index.loc["LOO common retained cohort"]
    readme = NEW / "README.md"
    readme_text = readme.read_text(encoding="utf-8")
    marker = "\n## Exclusion impact comparison"
    if marker in readme_text:
        readme_text = readme_text.split(marker, 1)[0].rstrip() + "\n"
    readme_section = [
        "",
        "## Exclusion impact comparison",
        "",
        "This sensitivity experiment removes samples 6022 and 6099 from every fit and AFM bank without changing algorithms. The complete before/after analysis is `report/exclusion_impact_summary.md`, with comparison figures in `figures/comparison/`.",
    ]
    readme.write_text(
        readme_text.rstrip() + "\n" + "\n".join(readme_section) + "\n",
        encoding="utf-8",
    )

    master = NEW / "report/result_summary.md"
    master_text = master.read_text(encoding="utf-8")
    master_marker = "\n## Comparison with the original experiment"
    if master_marker in master_text:
        master_text = master_text.split(master_marker, 1)[0].rstrip() + "\n"
    master_section = [
        "",
        "## Comparison with the original experiment",
        "",
        f"Prospective three-sample MAE changed from {prospective['original_MAE_nm']:.4f} to {prospective['excluded_MAE_nm']:.4f} nm. On the fair common-26 LOO cohort, MAE changed from {loo['original_MAE_nm']:.4f} to {loo['excluded_MAE_nm']:.4f} nm.",
        "",
        "See `report/exclusion_impact_summary.md` for the full per-sample and retrieval comparison.",
    ]
    master.write_text(
        master_text.rstrip() + "\n" + "\n".join(master_section) + "\n",
        encoding="utf-8",
    )


def refresh_manifest() -> None:
    manifest = NEW / "provenance/MANIFEST.sha256"
    rows = []
    for path in sorted(NEW.rglob("*")):
        if path.is_file() and path != manifest:
            rows.append(f"{sha256_file(path)}  {path.relative_to(NEW)}")
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    targets = load_targets()
    baseline, retrained, loo, hoo = compare_predictions(targets)
    metrics = build_metric_comparison(baseline, retrained, loo, hoo)
    full_cohort_metrics = build_full_cohort_comparison()
    prospective_sources, baseline_sources = compare_retrieval_sources()
    model_changes = compare_models()
    baseline.to_csv(OUTPUT / "baseline_prediction_changes.csv", index=False)
    retrained.to_csv(OUTPUT / "prospective_prediction_changes.csv", index=False)
    loo.to_csv(OUTPUT / "leave_one_out_common26_changes.csv", index=False)
    hoo.to_csv(OUTPUT / "held_one_out_afm_common26_changes.csv", index=False)
    metrics.to_csv(OUTPUT / "aggregate_metric_changes.csv", index=False)
    full_cohort_metrics.to_csv(
        OUTPUT / "full_cohort_metric_changes.csv", index=False
    )
    prospective_sources.to_csv(
        OUTPUT / "prospective_retrieval_source_changes.csv", index=False
    )
    baseline_sources.to_csv(
        OUTPUT / "baseline_retrieval_source_changes.csv", index=False
    )
    model_changes.to_csv(OUTPUT / "model_parameter_changes.csv", index=False)
    make_impact_summary_figure(retrained, loo, metrics)
    make_loo_before_after_figure(loo, metrics)
    write_report(
        baseline,
        retrained,
        loo,
        hoo,
        metrics,
        full_cohort_metrics,
        prospective_sources,
        baseline_sources,
        model_changes,
    )
    update_readme_and_master_report(metrics)
    provenance = {
        "created_at": now(),
        "original_experiment": str(OLD.relative_to(REPO)),
        "exclusion_experiment": str(NEW.relative_to(REPO)),
        "excluded_sample_ids": EXCLUDED_IDS,
        "algorithm_changed": False,
        "comparison_outputs": [
            str(path.relative_to(NEW))
            for path in sorted(OUTPUT.glob("*.csv"))
        ],
    }
    (OUTPUT / "comparison_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    refresh_manifest()
    print(
        json.dumps(
            {
                "status": "ok",
                "prospective_MAE_original": float(
                    metrics.iloc[1]["original_MAE_nm"]
                ),
                "prospective_MAE_excluded": float(
                    metrics.iloc[1]["excluded_MAE_nm"]
                ),
                "loo_common26_MAE_original": float(
                    metrics.iloc[2]["original_MAE_nm"]
                ),
                "loo_common26_MAE_excluded": float(
                    metrics.iloc[2]["excluded_MAE_nm"]
                ),
                "report": str(NEW / "report/exclusion_impact_summary.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
