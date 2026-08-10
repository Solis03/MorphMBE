from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from analysis.rheed_video_afm_story.common import repo_path

METHOD_LABELS = {
    "unconditional_train_mean": "Unconditional\ntrain mean",
    "nearest_rheed_retrieval": "Nearest-RHEED\nretrieval",
    "rheed_conditional_cvae": "RHEED-conditioned\nCVAE",
}

METHOD_COLORS = {
    "unconditional_train_mean": "#9AA0A6",
    "nearest_rheed_retrieval": "#D97706",
    "rheed_conditional_cvae": "#2563EB",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(figure: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base_path.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_training_curves(history_path: str | Path, figure_dir: str | Path) -> None:
    _style()
    history = pd.read_csv(repo_path(history_path))
    validated = history.dropna(subset=["val_selection_score"])
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    axes[0].plot(
        history["epoch"],
        history["train_reconstruction"],
        label="Train reconstruction",
        color="#2563EB",
    )
    axes[0].plot(
        validated["epoch"],
        validated["val_reconstruction"],
        "o-",
        label="Validation reconstruction",
        color="#DC2626",
        markersize=3,
    )
    axes[0].set(xlabel="Epoch", ylabel="Physics-aware loss", title="CVAE optimization")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        validated["epoch"],
        validated["val_prior_composite"],
        "o-",
        color="#7C3AED",
        label="Generated-shape score",
        markersize=3,
    )
    axes[1].plot(
        validated["epoch"],
        validated["val_rq_mae_nm"],
        "s-",
        color="#059669",
        label="Rq MAE (nm)",
        markersize=3,
    )
    best = validated.loc[validated["val_selection_score"].idxmin()]
    axes[1].axvline(best["epoch"], color="black", linestyle="--", linewidth=1)
    axes[1].annotate(
        f"selected epoch {int(best['epoch'])}",
        (best["epoch"], best["val_prior_composite"]),
        xytext=(6, 8),
        textcoords="offset points",
    )
    axes[1].set(
        xlabel="Epoch",
        ylabel="Validation metric",
        title="Prior generation (validation only)",
    )
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    figure.suptitle("Training and validation history", fontweight="bold")
    figure.tight_layout()
    _save(figure, repo_path(figure_dir) / "Fig6_training_validation_curves")


def plot_temporal_ablation(ablation: pd.DataFrame, figure_dir: str | Path) -> None:
    _style()
    ordered = ablation.sort_values("val_selection_score")
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    labels = [
        value.replace("dino_vits14__", "DINOv2: ")
        .replace("r3d_18__", "R3D-18: ")
        .replace("__raw_luminance", "")
        for value in ordered["embedding_id"]
    ]
    axes[0].barh(labels, ordered["val_condition_mae_z"], color="#4F46E5")
    axes[0].set(
        xlabel="Standardized descriptor MAE",
        title="RHEED-to-morphology validation",
    )
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].barh(labels, ordered["val_rq_mae_nm"], color="#059669")
    axes[1].set(xlabel="Rq MAE (nm)", title="Roughness validation")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.2)
    figure.suptitle(
        "Temporal-window ablation (model selection uses validation groups only)",
        fontweight="bold",
    )
    figure.tight_layout()
    _save(figure, repo_path(figure_dir) / "Fig8_temporal_window_ablation")


def plot_metric_comparison(metrics: pd.DataFrame, figure_dir: str | Path) -> None:
    _style()
    requested = [
        ("rq_absolute_error_nm", "Rq MAE (nm)", False),
        ("normalized_psd_log_distance", "PSD log distance", False),
        ("condition_descriptor_mae_z", "Descriptor MAE (z)", False),
        ("ssim", "SSIM", True),
    ]
    methods = [
        "unconditional_train_mean",
        "nearest_rheed_retrieval",
        "rheed_conditional_cvae",
    ]
    figure, axes = plt.subplots(1, 4, figsize=(12.0, 3.25))
    for axis, (column, label, higher) in zip(axes, requested, strict=False):
        values = [
            float(metrics.loc[metrics["method"] == method, column].median())
            for method in methods
        ]
        axis.bar(
            range(len(methods)),
            values,
            color=[METHOD_COLORS[method] for method in methods],
        )
        axis.set_xticks(range(len(methods)), [METHOD_LABELS[m] for m in methods])
        axis.tick_params(axis="x", labelrotation=25)
        axis.set_ylabel(label)
        axis.set_title("Higher is better" if higher else "Lower is better")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Held-out baseline versus generative model", fontweight="bold")
    figure.tight_layout()
    _save(figure, repo_path(figure_dir) / "Fig2_baseline_vs_final_metrics")


def render_metric_table(metric_summary: pd.DataFrame, figure_dir: str | Path) -> None:
    _style()
    metrics = [
        ("rq_absolute_error_nm", "Rq error (nm) ↓"),
        ("normalized_psd_log_distance", "PSD distance ↓"),
        ("condition_descriptor_mae_z", "Descriptor MAE ↓"),
        ("ssim", "SSIM ↑"),
        ("diversity_ratio", "Diversity ratio"),
    ]
    methods = [
        "unconditional_train_mean",
        "nearest_rheed_retrieval",
        "rheed_conditional_cvae",
    ]
    cells: list[list[str]] = []
    for method in methods:
        row = [METHOD_LABELS[method]]
        for metric, _ in metrics:
            match = metric_summary.loc[
                (metric_summary["method"] == method)
                & (metric_summary["metric"] == metric)
            ]
            if match.empty:
                row.append("NA")
            else:
                values = match.iloc[0]
                row.append(
                    f"{values['median']:.3f}\n"
                    f"[{values['ci95_low']:.3f}, {values['ci95_high']:.3f}]"
                )
        cells.append(row)
    figure, axis = plt.subplots(figsize=(12.0, 3.0))
    axis.axis("off")
    table = axis.table(
        cellText=cells,
        colLabels=["Method"] + [label for _, label in metrics],
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.21, 0.158, 0.158, 0.158, 0.158, 0.158],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.85)
    for (row, column), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E5E7EB")
            cell.set_text_props(weight="bold")
        elif column == 0:
            cell.set_facecolor("#F8FAFC")
            cell.set_text_props(weight="bold")
    axis.set_title(
        "Group-level medians with 95% bootstrap intervals",
        fontweight="bold",
        pad=10,
    )
    _save(figure, repo_path(figure_dir) / "baseline_vs_final_metric_table")


def plot_descriptor_correlations(
    predictions: pd.DataFrame, figure_dir: str | Path
) -> None:
    _style()
    selected = [
        ("log_rq_nm", "Rq (nm)", True),
        ("unit_psd_high_fraction", "High-frequency PSD fraction", False),
        ("log_unit_autocorr_length_nm", "Correlation length (nm)", True),
        ("unit_skewness", "Height skewness", False),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.4), layout="constrained")
    for axis, (descriptor, label, exponentiate) in zip(
        axes.ravel(), selected, strict=False
    ):
        rows = predictions.loc[predictions["descriptor"] == descriptor]
        true = rows["true"].to_numpy(float)
        predicted = rows["predicted"].to_numpy(float)
        if exponentiate:
            true = np.exp(true)
            predicted = np.exp(predicted)
        if descriptor == "unit_psd_high_fraction":
            true = 1e5 * true
            predicted = 1e5 * predicted
            label = "high-frequency PSD (×10⁵)"
        lo = float(min(true.min(), predicted.min()))
        hi = float(max(true.max(), predicted.max()))
        axis.scatter(true, predicted, color="#2563EB", edgecolor="white", s=45)
        for x, y, sample_id in zip(
            true, predicted, rows["growth_run_id"].astype(str), strict=False
        ):
            axis.annotate(
                sample_id, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=6
            )
        axis.plot([lo, hi], [lo, hi], "--", color="#6B7280", linewidth=1)
        axis.set_xlabel(f"Measured {label}", fontsize=8)
        axis.set_ylabel(f"RHEED-predicted {label}", fontsize=8)
        axis.grid(alpha=0.2)
    figure.suptitle("Held-out morphology-descriptor prediction", fontweight="bold")
    _save(figure, repo_path(figure_dir) / "Fig7_descriptor_correlations")


def _afm_axis(
    axis: plt.Axes,
    unit_map: np.ndarray,
    rq_nm: float,
    title: str,
    *,
    vlim: float,
) -> None:
    physical = unit_map * float(rq_nm)
    image = axis.imshow(
        physical,
        cmap="viridis",
        origin="lower",
        vmin=-vlim,
        vmax=vlim,
    )
    axis.set_title(title, fontsize=8)
    axis.set_xticks([0, physical.shape[1] - 1], ["0", "1"])
    axis.set_yticks([0, physical.shape[0] - 1], ["0", "1"])
    axis.set_xlabel("µm", labelpad=-8)
    axis.set_ylabel("µm", labelpad=-9)
    return image


def plot_rheed_generated_ground_truth(
    evaluation: dict[str, Any],
    phase1_manifest: pd.DataFrame,
    figure_dir: str | Path,
) -> None:
    _style()
    representatives = evaluation["representatives"]
    groups = sorted(representatives)
    rows = len(groups)
    figure, axes = plt.subplots(rows, 4, figsize=(10.5, 2.25 * rows), squeeze=False)
    manifest = phase1_manifest.copy()
    manifest["sample_id"] = manifest["sample_id"].astype(str)
    for row_index, group in enumerate(groups):
        payload = representatives[group]
        preview = repo_path(
            f"reports/rheed_video_afm_story/phase1/clip_previews/{group}.png"
        )
        if preview.exists():
            axes[row_index, 0].imshow(plt.imread(preview))
        axes[row_index, 0].set_title(f"RHEED window\nsample {group}", fontsize=8)
        axes[row_index, 0].axis("off")
        values = [
            np.abs(payload["real_medoid"] * payload["real_rq"]),
            np.abs(payload["generated_medoid"] * payload["generated_rq"]),
            np.abs(payload["retrieved"] * payload["retrieved_rq"]),
        ]
        vlim = float(max(np.percentile(value, 99) for value in values))
        _afm_axis(
            axes[row_index, 1],
            payload["generated_medoid"],
            payload["generated_rq"],
            f"Generated CVAE\nRq={payload['generated_rq']:.2f} nm",
            vlim=vlim,
        )
        _afm_axis(
            axes[row_index, 2],
            payload["real_medoid"],
            payload["real_rq"],
            f"Measured AFM\nRq={payload['real_rq']:.2f} nm",
            vlim=vlim,
        )
        _afm_axis(
            axes[row_index, 3],
            payload["retrieved"],
            payload["retrieved_rq"],
            f"Retrieval ({payload['retrieved_group']})\nRq={payload['retrieved_rq']:.2f} nm",
            vlim=vlim,
        )
    figure.suptitle(
        "RHEED-conditioned AFM generation on held-out growth groups",
        fontweight="bold",
        y=1.002,
    )
    figure.tight_layout()
    _save(figure, repo_path(figure_dir) / "Fig3_rheed_generated_ground_truth")


def plot_real_vs_generated_ensembles(
    evaluation: dict[str, Any], figure_dir: str | Path
) -> None:
    _style()
    representatives = evaluation["representatives"]
    groups = sorted(representatives)
    figure, axes = plt.subplots(
        len(groups), 5, figsize=(11.0, 2.15 * len(groups)), squeeze=False
    )
    for row_index, group in enumerate(groups):
        payload = representatives[group]
        samples = payload["generated_samples"][:4]
        values = [np.abs(payload["real_medoid"] * payload["real_rq"])] + [
            np.abs(sample * payload["generated_rq"]) for sample in samples
        ]
        vlim = float(max(np.percentile(value, 99) for value in values))
        _afm_axis(
            axes[row_index, 0],
            payload["real_medoid"],
            payload["real_rq"],
            f"{group}: measured",
            vlim=vlim,
        )
        for column, sample in enumerate(samples, 1):
            _afm_axis(
                axes[row_index, column],
                sample,
                payload["generated_rq"],
                f"generated draw {column}",
                vlim=vlim,
            )
    figure.suptitle(
        "Measured representative and uncurated conditional samples",
        fontweight="bold",
        y=1.002,
    )
    figure.tight_layout()
    _save(figure, repo_path(figure_dir) / "Fig4_real_vs_generated_ensembles")


def plot_failure_cases(
    evaluation: dict[str, Any], figure_dir: str | Path, count: int = 3
) -> None:
    _style()
    metrics = evaluation["metrics"]
    failures = (
        metrics.loc[metrics["method"] == "rheed_conditional_cvae"]
        .sort_values("composite_score", ascending=False)
        .head(count)
    )
    representatives = evaluation["representatives"]
    figure, axes = plt.subplots(
        len(failures), 3, figsize=(8.2, 2.5 * len(failures)), squeeze=False
    )
    for row_index, row in enumerate(failures.itertuples(index=False)):
        group = str(row.growth_run_id)
        payload = representatives[group]
        values = [
            np.abs(payload["real_medoid"] * payload["real_rq"]),
            np.abs(payload["generated_medoid"] * payload["generated_rq"]),
            np.abs(
                payload["real_medoid"] * payload["real_rq"]
                - payload["generated_medoid"] * payload["generated_rq"]
            ),
        ]
        vlim = float(max(np.percentile(value, 99) for value in values[:2]))
        _afm_axis(
            axes[row_index, 0],
            payload["real_medoid"],
            payload["real_rq"],
            f"{group}: measured",
            vlim=vlim,
        )
        _afm_axis(
            axes[row_index, 1],
            payload["generated_medoid"],
            payload["generated_rq"],
            f"generated\nPSD d={row.normalized_psd_log_distance:.2f}",
            vlim=vlim,
        )
        difference = (
            payload["generated_medoid"] * payload["generated_rq"]
            - payload["real_medoid"] * payload["real_rq"]
        )
        diff_limit = float(max(np.percentile(np.abs(difference), 99), 1e-6))
        axes[row_index, 2].imshow(
            difference,
            cmap="coolwarm",
            origin="lower",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-diff_limit, vmax=diff_limit),
        )
        axes[row_index, 2].set_title(
            f"height residual (nm)\nRq error={row.rq_absolute_error_nm:.2f} nm",
            fontsize=8,
        )
        axes[row_index, 2].axis("off")
    figure.suptitle(
        "Predefined failure cases: highest held-out composite error",
        fontweight="bold",
    )
    figure.tight_layout()
    _save(figure, repo_path(figure_dir) / "Fig5_failure_cases")


def plot_condition_control(
    condition_control: pd.DataFrame, figure_dir: str | Path
) -> None:
    _style()
    ordered = condition_control.sort_values("growth_run_id")
    positions = np.arange(len(ordered))
    width = 0.38
    figure, axis = plt.subplots(figsize=(7.2, 3.5))
    axis.bar(
        positions - width / 2,
        ordered["correct_condition_descriptor_mae_z"],
        width,
        label="Correct RHEED condition",
        color="#2563EB",
    )
    axis.bar(
        positions + width / 2,
        ordered["permuted_condition_descriptor_mae_z"],
        width,
        label="Permuted condition",
        color="#DC2626",
    )
    axis.set_xticks(positions, ordered["growth_run_id"].astype(str))
    axis.set(
        xlabel="Held-out growth group",
        ylabel="Generated descriptor MAE (z)",
        title="Condition-swap negative control",
    )
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    _save(figure, repo_path(figure_dir) / "Fig9_condition_swap_control")


def make_all_figures(
    *,
    evaluation: dict[str, Any],
    ablation: pd.DataFrame,
    training_history_path: str | Path,
    phase1_manifest: pd.DataFrame,
    figure_dir: str | Path,
) -> None:
    plot_training_curves(training_history_path, figure_dir)
    plot_temporal_ablation(ablation, figure_dir)
    plot_metric_comparison(evaluation["metrics"], figure_dir)
    render_metric_table(evaluation["metric_summary"], figure_dir)
    plot_descriptor_correlations(evaluation["descriptor_predictions"], figure_dir)
    plot_rheed_generated_ground_truth(evaluation, phase1_manifest, figure_dir)
    plot_real_vs_generated_ensembles(evaluation, figure_dir)
    plot_failure_cases(evaluation, figure_dir)
    plot_condition_control(evaluation["condition_control"], figure_dir)
