from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    prediction = pd.read_csv(
        args.predictions, dtype={"growth_run_id": str}
    )
    baseline = pd.read_csv(
        args.baseline, dtype={"growth_run_id": str}
    )
    baseline = baseline.loc[baseline["target"] == "Rq_nm"].set_index(
        "growth_run_id"
    )
    figure_dir = args.output / "figures"

    ordered = prediction.sort_values("true_target").reset_index(drop=True)
    x = np.arange(len(ordered))
    old = baseline.loc[ordered["growth_run_id"], "predicted_target"].to_numpy()
    figure, axis = plt.subplots(figsize=(11.2, 4.6), constrained_layout=True)
    axis.plot(
        x,
        ordered["true_target"],
        "-o",
        color="#D55E00",
        lw=1.8,
        ms=4,
        label="measured sample-median Sq",
    )
    axis.plot(
        x,
        old,
        "--",
        color="#888888",
        lw=1.2,
        label="M15b strict LOO",
    )
    scatter = axis.scatter(
        x,
        ordered["predicted_target"],
        c=100 * ordered["confidence"],
        cmap="viridis",
        vmin=0,
        vmax=100,
        s=44,
        edgecolor="black",
        linewidth=0.45,
        label="M16 strict LOO",
        zorder=4,
    )
    axis.plot(
        x,
        ordered["predicted_target"],
        color="#0072B2",
        lw=1.0,
        alpha=0.75,
    )
    axis.vlines(
        x,
        ordered["interval_lower"],
        ordered["interval_upper"],
        color="#0072B2",
        alpha=0.20,
        lw=1,
    )
    axis.set_xticks(x)
    axis.set_xticklabels(
        ordered["growth_run_id"], rotation=58, ha="right", fontsize=7
    )
    axis.set_ylabel("Areal roughness Sq (nm)")
    axis.set_xlabel("Held growth, ordered by measured Sq")
    axis.set_title(
        "Endpoint-aware temporal/streak ensemble expands both smooth and "
        "rough RHEED regimes"
    )
    axis.legend(frameon=False, ncol=3, loc="upper left")
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.01)
    colorbar.set_label("Error-related confidence index (0–100)")
    _save(figure, figure_dir / "Fig1_m16_all28_ordered_sq")

    figure, axes = plt.subplots(
        1, 2, figsize=(9.2, 4.0), constrained_layout=True
    )
    axes[0].scatter(
        prediction["true_target"],
        prediction["predicted_target"],
        c=100 * prediction["confidence"],
        cmap="viridis",
        vmin=0,
        vmax=100,
        s=55,
        edgecolor="black",
        linewidth=0.5,
    )
    maximum = float(
        max(prediction["true_target"].max(), prediction["predicted_target"].max())
    )
    axes[0].plot([0, maximum], [0, maximum], "k--", lw=1)
    axes[0].set_xlabel("Measured Sq (nm)")
    axes[0].set_ylabel("Strict LOO predicted Sq (nm)")
    axes[0].set_title(
        "MAE 1.07 nm; Pearson r=0.74; Spearman ρ=0.60"
    )
    for sample in ("6101", "N6342", "6095", "6099", "6081"):
        row = prediction.loc[prediction["growth_run_id"] == sample].iloc[0]
        axes[0].annotate(
            sample,
            (row["true_target"], row["predicted_target"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axes[1].scatter(
        100 * prediction["confidence"],
        prediction["absolute_error"],
        c=np.where(prediction["streak_gate"], "#009E73", "#0072B2"),
        s=48,
        edgecolor="black",
        linewidth=0.45,
    )
    relation = spearmanr(
        prediction["confidence"], prediction["absolute_error"]
    )
    axes[1].set_xlabel("Confidence index (0–100)")
    axes[1].set_ylabel("Absolute Sq prediction error (nm)")
    axes[1].set_title(
        f"Confidence vs |error|: ρ={relation.statistic:.2f}, "
        f"p={relation.pvalue:.3f}"
    )
    axes[1].text(
        0.02,
        0.98,
        "green: local-streak expert active",
        transform=axes[1].transAxes,
        va="top",
        fontsize=7,
    )
    _save(figure, figure_dir / "Fig2_m16_sq_and_confidence")

    figure, axis = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    colors = np.where(
        prediction["rough_consensus_gate"],
        "#D55E00",
        np.where(prediction["streak_gate"], "#009E73", "#777777"),
    )
    axis.scatter(
        prediction["streak_feature"],
        prediction["true_target"],
        c=colors,
        s=55,
        edgecolor="black",
        linewidth=0.45,
    )
    for sample in (
        "6101",
        "N6342",
        "N6358",
        "6095",
        "6099",
        "6081",
    ):
        row = prediction.loc[prediction["growth_run_id"] == sample].iloc[0]
        axis.annotate(
            sample,
            (row["streak_feature"], row["true_target"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlabel("Local diffraction-peak horizontal aspect (75th percentile)")
    axis.set_ylabel("Measured sample-median Sq (nm)")
    axis.set_title(
        "Target-blind RHEED endpoint routes: streak-supported smooth vs "
        "temporal-consensus rough"
    )
    axis.text(
        0.02,
        0.98,
        "green: smooth streak gate  |  orange: rough temporal consensus",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
    )
    _save(figure, figure_dir / "Fig3_m16_physical_endpoint_routes")

    metrics = pd.read_csv(args.metrics)
    metric_long = metrics.set_index("model")
    labels = ["overall MAE", "smooth (<1.2 nm) MAE", "rough (≥5 nm) MAE"]
    old_values = metric_long.loc[
        "M15b_current", ["mae_nm", "smooth_mae_nm", "rough_mae_nm"]
    ].to_numpy(float)
    new_values = metric_long.loc[
        "M16_endpoint_streak_dual_resolution",
        ["mae_nm", "smooth_mae_nm", "rough_mae_nm"],
    ].to_numpy(float)
    x = np.arange(3)
    figure, axis = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    axis.bar(x - 0.18, old_values, width=0.36, color="#999999", label="M15b")
    axis.bar(
        x + 0.18,
        new_values,
        width=0.36,
        color="#0072B2",
        label="M16",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Strict LOO MAE (nm)")
    axis.set_title("Endpoint improvement without sacrificing full-cohort stability")
    axis.legend(frameon=False)
    _save(figure, figure_dir / "Fig4_m16_endpoint_ablation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
