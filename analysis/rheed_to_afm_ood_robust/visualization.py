from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "baseline": "#6B7280",
    "density": "#2563EB",
    "residual": "#D97706",
    "temporal": "#059669",
    "multiview": "#7C3AED",
    "nested": "#DC2626",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, root: Path, stem: str) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf"):
        path = root / f"{stem}.{suffix}"
        fig.savefig(path, facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_ood_audit(audit: pd.DataFrame, root: Path) -> list[str]:
    _style()
    frame = audit.sort_values("rheed_only_ood_score")
    colors = [
        COLORS["nested"]
        if rank <= 4
        else COLORS["baseline"]
        for rank in frame["rheed_only_ood_rank"]
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    ax.barh(
        frame["growth_run_id"],
        frame["rheed_only_ood_score"],
        color=colors,
        alpha=0.90,
    )
    ax.set_xlabel("RHEED-only OOD score (mean percentile rank)")
    ax.set_ylabel("Growth run")
    ax.set_xlim(0, 1.05)
    ax.set_title(
        "Target-blind RHEED support audit (AFM targets not used)"
    )
    ax.grid(axis="x", alpha=0.20)
    return _save(fig, root, "fig01_rheed_only_ood_audit")


def plot_exclusion_sensitivity(
    metrics: pd.DataFrame, root: Path
) -> list[str]:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    for target, color in (("Rq_nm", "#2563EB"), ("FSMI_nm", "#059669")):
        rows = metrics.loc[metrics["target"] == target].sort_values(
            "excluded_growth_count"
        )
        axes[0].plot(
            rows["excluded_growth_count"],
            rows["mae"],
            marker="o",
            color=color,
            label=target.replace("_", " "),
        )
        axes[1].plot(
            rows["excluded_growth_count"],
            rows["pearson_r"],
            marker="o",
            color=color,
            label=target.replace("_", " "),
        )
    axes[0].set_ylabel("Held-one-out MAE (nm)")
    axes[1].set_ylabel("Held-one-out Pearson r")
    for ax in axes:
        ax.set_xlabel("Top RHEED-only OOD samples excluded")
        ax.set_xticks([0, 2, 3, 4])
        ax.axvline(0, color="#111827", lw=0.8, alpha=0.4)
        ax.grid(alpha=0.20)
    axes[0].legend(frameon=False)
    fig.suptitle(
        "Sensitivity analysis: exclusion does not improve the frozen M12a head"
    )
    fig.tight_layout()
    return _save(fig, root, "fig02_exclusion_sensitivity")


def plot_method_summary(
    metrics: pd.DataFrame, root: Path
) -> list[str]:
    _style()
    rq = metrics.loc[metrics["target"] == "Rq_nm"].copy()
    # M14i is a target-specific assembly and is exactly M14g for Rq, so
    # plotting both would place two indistinguishable points on top of one
    # another.  Keep the component model and highlight it as the selected Rq
    # head instead.
    rq = rq.loc[rq["method"] != "M14i_target_specific_robust"]
    labels = {
        "M12a_frozen_alpha1": "M12a frozen",
        "M14a_regularized_alpha10": "M14a regularized",
        "M14b_rheed_density_weighted": "M14b density-weighted",
        "M14c_residual_self_paced": "M14c self-paced",
        "M14d_r3d_causal_temporal": "M14d temporal R3D",
        "M14e_multiview_curated20_r3d80": "M14e 20/80 blend",
        "M14f_multiview_curated40_r3d60": "M14f 40/60 blend",
        "M14g_multiview_curated60_r3d40": "M14g 60/40 blend (selected)",
        "M14h_nested_support_aware_selector": "M14h nested selector",
    }
    offsets = {
        "M14d_r3d_causal_temporal": (5, -12),
        "M14e_multiview_curated20_r3d80": (5, 7),
        "M14f_multiview_curated40_r3d60": (5, -13),
        "M14g_multiview_curated60_r3d40": (5, 7),
    }
    selected = rq["method"] == "M14g_multiview_curated60_r3d40"
    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    ax.scatter(
        rq.loc[~selected, "mae"],
        rq.loc[~selected, "pearson_r"],
        c="#2563EB",
        s=45,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.scatter(
        rq.loc[selected, "mae"],
        rq.loc[selected, "pearson_r"],
        c="#D97706",
        marker="*",
        s=120,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    for row in rq.itertuples():
        label = labels.get(str(row.method), str(row.method))
        ax.annotate(
            label,
            (row.mae, row.pearson_r),
            xytext=offsets.get(str(row.method), (5, 3)),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Rq held-one-out MAE (nm; lower is better)")
    ax.set_ylabel("Rq held-one-out Pearson r (higher is better)")
    ax.set_ylim(
        float(rq["pearson_r"].min()) - 0.01,
        float(rq["pearson_r"].max()) + 0.035,
    )
    ax.grid(alpha=0.20)
    ax.set_title("Robust small-sample condition-head ablation")
    return _save(fig, root, "fig03_method_ablation")


def plot_predictions(
    predictions: pd.DataFrame,
    methods: list[str],
    root: Path,
) -> list[str]:
    _style()
    targets = ["Rq_nm", "FSMI_nm"]
    fig, axes = plt.subplots(
        len(methods),
        len(targets),
        figsize=(8.4, 2.8 * len(methods)),
        squeeze=False,
    )
    for row_index, method in enumerate(methods):
        for column_index, target in enumerate(targets):
            ax = axes[row_index, column_index]
            rows = predictions.loc[
                (predictions["method"] == method)
                & (predictions["target"] == target)
            ]
            lower = min(
                float(rows["true_target"].min()),
                float(rows["predicted_target"].min()),
            )
            upper = max(
                float(rows["true_target"].max()),
                float(rows["predicted_target"].max()),
            )
            ax.plot(
                [lower, upper],
                [lower, upper],
                color="#111827",
                lw=0.9,
                ls="--",
            )
            scatter = ax.scatter(
                rows["true_target"],
                rows["predicted_target"],
                c=rows["confidence"],
                cmap="viridis",
                vmin=0,
                vmax=1,
                s=38,
                edgecolor="white",
                linewidth=0.5,
            )
            failures = rows.nlargest(3, "absolute_error")
            for failure in failures.itertuples():
                ax.annotate(
                    failure.growth_run_id,
                    (failure.true_target, failure.predicted_target),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                )
            ax.set_xlabel(f"Measured {target.replace('_', ' ')}")
            ax.set_ylabel(f"Predicted {target.replace('_', ' ')}")
            ax.set_title(method.replace("_", " "), fontsize=8)
            ax.grid(alpha=0.18)
            if column_index == len(targets) - 1:
                fig.colorbar(
                    scatter,
                    ax=ax,
                    fraction=0.046,
                    pad=0.04,
                    label="Confidence",
                )
    fig.tight_layout()
    return _save(fig, root, "fig04_held_one_out_predictions")


def plot_confidence_and_coverage(
    predictions: pd.DataFrame,
    risk_coverage: pd.DataFrame,
    method: str,
    root: Path,
) -> list[str]:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
    for target, color in (("Rq_nm", "#2563EB"), ("FSMI_nm", "#059669")):
        rows = predictions.loc[
            (predictions["method"] == method)
            & (predictions["target"] == target)
        ]
        axes[0].scatter(
            rows["predicted_absolute_error"],
            rows["absolute_error"],
            s=34,
            alpha=0.85,
            color=color,
            label=target.replace("_", " "),
        )
        coverage = risk_coverage.loc[
            (risk_coverage["method"] == method)
            & (risk_coverage["target"] == target)
        ].sort_values("coverage")
        axes[1].plot(
            100 * coverage["coverage"],
            coverage["mae"],
            marker="o",
            color=color,
            label=target.replace("_", " "),
        )
    axes[0].set_xlabel("Predicted absolute error (nm)")
    axes[0].set_ylabel("Realized absolute error (nm)")
    axes[0].set_title("Nested uncertainty calibration")
    axes[1].set_xlabel("Retained highest-confidence samples (%)")
    axes[1].set_ylabel("Selective MAE (nm)")
    axes[1].set_title("Risk-coverage behavior")
    for ax in axes:
        ax.grid(alpha=0.20)
        ax.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, root, "fig05_confidence_risk_coverage")


def plot_training_weights(
    weights: pd.DataFrame, root: Path
) -> list[str]:
    _style()
    frame = weights.sort_values("density_sample_weight")
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.barh(
        y - 0.18,
        frame["density_sample_weight"],
        height=0.34,
        color=COLORS["density"],
        label="RHEED density weight (target-blind)",
    )
    ax.barh(
        y + 0.18,
        frame["residual_self_paced_weight"],
        height=0.34,
        color=COLORS["residual"],
        label="Residual self-paced weight",
    )
    ax.set_yticks(y, frame["growth_run_id"])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Training weight")
    ax.set_ylabel("Growth run")
    ax.set_title("Fold-independent weight audit for the full cohort")
    ax.grid(axis="x", alpha=0.20)
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, root, "fig06_training_sample_weights")
