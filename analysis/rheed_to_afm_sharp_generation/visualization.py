from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.rheed_video_afm_story.afm_descriptors import radial_psd
from analysis.rheed_video_afm_story.common import repo_path


LABELS = {
    "M0_mean_condition_calibrated_spectral": (
        "Mean-condition generator"
    ),
    "M0b_mean_condition_calibrated_refiner": (
        "Mean-condition adversarial refiner"
    ),
    "M1_cvae_blur_baseline": "Previous CVAE",
    "M2_spectral_oracle_condition": "Spectral (oracle condition)",
    "M2_spectral_rheed_condition": "Spectral (RHEED)",
    "M2b_calibrated_spectral_oracle_condition": (
        "Calibrated spectral (oracle)"
    ),
    "M2b_calibrated_spectral_rheed_condition": (
        "Calibrated spectral (RHEED)"
    ),
    "M3_refiner_oracle_condition": "Adversarial refiner (oracle)",
    "M3_refiner_rheed_condition": "Adversarial refiner (RHEED)",
    "M3b_calibrated_refiner_oracle_condition": (
        "Calibrated adversarial refiner (oracle)"
    ),
    "M3b_calibrated_refiner_rheed_condition": (
        "Calibrated adversarial refiner (RHEED)"
    ),
}


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _comparison_panel(evaluation: dict, figure_dir: Path) -> None:
    panels = evaluation["panels"]
    groups = sorted(panels)
    preferred = [
        "M1_cvae_blur_baseline",
        "M2_spectral_rheed_condition",
        "M2b_calibrated_spectral_rheed_condition",
        "M3b_calibrated_refiner_rheed_condition",
    ]
    methods = [
        method
        for method in preferred
        if method in panels[groups[0]]["methods"]
    ]
    columns = ["Measured AFM", *[LABELS[method] for method in methods]]
    figure, axes = plt.subplots(
        len(groups),
        len(columns),
        figsize=(3.0 * len(columns), 2.8 * len(groups)),
        squeeze=False,
    )
    for row_index, group in enumerate(groups):
        payload = panels[group]
        arrays = [
            payload["real_medoid"],
            *[payload["methods"][method]["medoid"] for method in methods],
        ]
        limit = float(
            max(
                2.5,
                np.percentile(
                    np.abs(np.concatenate([array.ravel() for array in arrays])),
                    99,
                ),
            )
        )
        for column_index, (title, array) in enumerate(zip(columns, arrays)):
            axis = axes[row_index, column_index]
            axis.imshow(
                array,
                cmap="viridis",
                vmin=-limit,
                vmax=limit,
                origin="lower",
                interpolation="nearest",
            )
            if row_index == 0:
                axis.set_title(title, fontsize=11)
            if column_index == 0:
                axis.set_ylabel(f"Group {group}\n1 μm", fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        "Validation-only AFM morphology: sharp non-retrieval generators",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "All panels show unit-Rq height. Samples are fixed-seed medoids; "
        "no validation example was hand-picked.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.97))
    _save(figure, figure_dir / "Fig1_real_cvae_spectral_refiner")


def _metric_summary(evaluation: dict, figure_dir: Path) -> None:
    summary = evaluation["summary"].copy()
    summary["label"] = summary["method"].map(LABELS).fillna(summary["method"])
    metrics = [
        ("median_composite_score", "Morphology composite ↓"),
        ("median_normalized_psd_log_distance", "PSD log distance ↓"),
        ("median_sharpness_ratio", "Gradient / real (target 1)"),
        ("median_laplacian_rms_relative_error", "Laplacian relative error ↓"),
        ("texture_gate_pass_fraction", "AFM texture gate fraction ↑"),
    ]
    figure, axes = plt.subplots(1, len(metrics), figsize=(17, 4.4))
    colors = plt.cm.Set2(np.linspace(0, 1, len(summary)))
    for axis, (column, title) in zip(axes, metrics):
        values = summary[column].to_numpy(float)
        axis.bar(np.arange(len(summary)), values, color=colors)
        axis.set_title(title, fontsize=10)
        axis.set_xticks(np.arange(len(summary)))
        axis.set_xticklabels(summary["label"], rotation=55, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        if column == "median_sharpness_ratio":
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
    figure.suptitle(
        "Validation morphology and sharpness metrics", fontsize=14, fontweight="bold"
    )
    figure.tight_layout()
    _save(figure, figure_dir / "Fig2_texture_metric_summary")


def _fft_panel(evaluation: dict, figure_dir: Path) -> None:
    panels = evaluation["panels"]
    group = sorted(panels)[0]
    payload = panels[group]
    methods = [
        method
        for method in (
            "M1_cvae_blur_baseline",
            "M2_spectral_rheed_condition",
            "M2b_calibrated_spectral_rheed_condition",
            "M3b_calibrated_refiner_rheed_condition",
        )
        if method in payload["methods"]
    ]
    arrays = [payload["real_medoid"]] + [
        payload["methods"][method]["medoid"] for method in methods
    ]
    labels = ["Measured AFM"] + [LABELS[method] for method in methods]
    figure, axes = plt.subplots(2, len(arrays), figsize=(3.1 * len(arrays), 6.2))
    for index, (array, label) in enumerate(zip(arrays, labels)):
        axes[0, index].imshow(array, cmap="viridis", origin="lower")
        axes[0, index].set_title(label, fontsize=10)
        spectrum = np.log1p(
            np.abs(np.fft.fftshift(np.fft.fft2(array))) ** 2
        )
        axes[1, index].imshow(spectrum, cmap="magma", origin="lower")
        axes[1, index].set_title("log FFT power", fontsize=10)
        for axis in axes[:, index]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        f"Spatial and frequency-domain review (validation group {group})",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    _save(figure, figure_dir / "Fig3_spatial_frequency_comparison")


def _condition_control(control: pd.DataFrame, figure_dir: Path) -> None:
    methods = list(control["method"].drop_duplicates())
    figure, axes = plt.subplots(
        1, len(methods), figsize=(5.0 * len(methods), 4.2), squeeze=False
    )
    for axis, method in zip(axes[0], methods):
        rows = control.loc[control["method"] == method]
        colors = [
            "#2a9d8f" if value > 0 else "#e76f51"
            for value in rows["wrong_minus_correct"]
        ]
        axis.bar(rows["growth_run_id"], rows["wrong_minus_correct"], color=colors)
        axis.axhline(0, color="black", linewidth=1)
        axis.set_title(LABELS.get(method, method), fontsize=10)
        axis.set_xlabel("Validation growth group")
        axis.set_ylabel("Wrong − correct descriptor error (z)")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Condition-permutation negative control (positive is required)",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout()
    _save(figure, figure_dir / "Fig4_condition_permutation_control")


def _training_curves(history_path: Path, figure_dir: Path) -> None:
    history = pd.read_csv(history_path)
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    axes[0].plot(history["step"], history["generator_loss"], label="Generator")
    axes[0].plot(
        history["step"], history["discriminator_loss"], label="Discriminator"
    )
    axes[0].set_title("Adversarial losses")
    axes[0].legend()
    axes[1].plot(history["step"], history["condition_loss"], label="Condition")
    axes[1].plot(history["step"], history["gradient_loss"], label="Gradient")
    axes[1].set_title("Scientific regularizers")
    axes[1].legend()
    validation = history.dropna(subset=["val_selection_score"])
    axes[2].plot(
        validation["step"],
        validation["val_fft_log_mae"],
        marker="o",
        label="FFT",
    )
    axes[2].plot(
        validation["step"],
        validation["val_gradient_relative_error"],
        marker="o",
        label="Gradient",
    )
    axes[2].plot(
        validation["step"],
        validation["val_selection_score"],
        marker="o",
        label="Selection",
    )
    axes[2].set_title("Validation-only early stopping")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("Training step")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    _save(figure, figure_dir / "Fig5_adversarial_training_curves")


def _all_draws(evaluation: dict, figure_dir: Path) -> None:
    panels = evaluation["panels"]
    groups = sorted(panels)
    method = (
        "M3b_calibrated_refiner_rheed_condition"
        if "M3b_calibrated_refiner_rheed_condition"
        in panels[groups[0]]["methods"]
        else "M2b_calibrated_spectral_rheed_condition"
    )
    draw_count = min(
        8, min(len(panels[group]["methods"][method]["samples"]) for group in groups)
    )
    figure, axes = plt.subplots(
        len(groups),
        draw_count + 1,
        figsize=(2.0 * (draw_count + 1), 2.0 * len(groups)),
        squeeze=False,
    )
    for row_index, group in enumerate(groups):
        arrays = [
            panels[group]["real_medoid"],
            *panels[group]["methods"][method]["samples"][:draw_count],
        ]
        for column_index, array in enumerate(arrays):
            axes[row_index, column_index].imshow(
                array, cmap="viridis", origin="lower"
            )
            axes[row_index, column_index].set_xticks([])
            axes[row_index, column_index].set_yticks([])
            if row_index == 0:
                axes[row_index, column_index].set_title(
                    "Measured" if column_index == 0 else f"Draw {column_index}",
                    fontsize=9,
                )
            if column_index == 0:
                axes[row_index, column_index].set_ylabel(group)
    figure.suptitle(
        f"All fixed-seed validation draws: {LABELS[method]}",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    _save(figure, figure_dir / "Fig6_all_generated_draws")


def _rheed_crop(row: pd.Series) -> np.ndarray:
    paths = json.loads(str(row["clip_frame_paths"]))
    offset = int(row["keyframe_offset_in_clip_x"])
    image = plt.imread(repo_path(paths[offset]))
    y0 = int(row["roi_y"])
    x0 = int(row["roi_x"])
    crop = image[
        y0 : y0 + int(row["roi_height"]),
        x0 : x0 + int(row["roi_width"]),
    ]
    if crop.ndim == 3:
        crop = np.max(crop[..., :3], axis=2)
    lower, upper = np.percentile(crop, [1, 99.7])
    return np.clip(
        (crop.astype(float) - lower) / max(float(upper - lower), 1e-8),
        0,
        1,
    )


def _rheed_to_afm_panel(
    evaluation: dict,
    phase1_manifest_path: str | Path,
    figure_dir: Path,
) -> None:
    phase1 = pd.read_csv(repo_path(phase1_manifest_path))
    phase1["growth_run_id"] = phase1["growth_run_id"].astype(str)
    lookup = phase1.set_index("growth_run_id")
    panels = evaluation["panels"]
    groups = sorted(panels)
    method = (
        "M3b_calibrated_refiner_rheed_condition"
        if "M3b_calibrated_refiner_rheed_condition"
        in panels[groups[0]]["methods"]
        else "M2b_calibrated_spectral_rheed_condition"
    )
    figure, axes = plt.subplots(
        len(groups), 3, figsize=(10.4, 3.25 * len(groups)), squeeze=False
    )
    for row_index, group in enumerate(groups):
        payload = panels[group]
        rheed = _rheed_crop(lookup.loc[group])
        generated_payload = payload["methods"][method]
        generated = (
            generated_payload["medoid"] * float(generated_payload["rq"])
        )
        measured = payload["real_medoid"] * float(payload["true_rq"])
        limit = float(
            max(
                np.percentile(np.abs(measured), 99),
                np.percentile(np.abs(generated), 99),
                1e-6,
            )
        )
        axes[row_index, 0].imshow(
            rheed, cmap="gray", origin="upper", aspect="auto"
        )
        axes[row_index, 1].imshow(
            generated,
            cmap="viridis",
            vmin=-limit,
            vmax=limit,
            origin="lower",
            interpolation="nearest",
        )
        axes[row_index, 2].imshow(
            measured,
            cmap="viridis",
            vmin=-limit,
            vmax=limit,
            origin="lower",
            interpolation="nearest",
        )
        if row_index == 0:
            axes[row_index, 0].set_title("RHEED key frame (ROI)")
            axes[row_index, 1].set_title("Generated AFM")
            axes[row_index, 2].set_title("Measured AFM")
        axes[row_index, 0].set_ylabel(f"Group {group}")
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 1].text(
            0.98,
            0.02,
            f"shared scale ±{limit:.1f} nm",
            transform=axes[row_index, 1].transAxes,
            ha="right",
            va="bottom",
            color="white",
            fontsize=8,
            bbox={
                "facecolor": "black",
                "alpha": 0.45,
                "edgecolor": "none",
                "pad": 2,
            },
        )
    figure.suptitle(
        "RHEED-conditioned non-retrieval AFM generation",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "RHEED ROIs use robust display contrast only. AFM panels use a shared "
        "physical-height scale within each row.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.97))
    _save(figure, figure_dir / "Fig8_rheed_generated_measured_afm")


def _failure_case_panel(evaluation: dict, figure_dir: Path) -> None:
    metrics = evaluation["per_group"]
    method = (
        "M3b_calibrated_refiner_rheed_condition"
        if "M3b_calibrated_refiner_rheed_condition"
        in set(metrics["method"])
        else "M2b_calibrated_spectral_rheed_condition"
    )
    candidates = metrics.loc[metrics["method"] == method].sort_values(
        [
            "condition_descriptor_mae_z",
            "normalized_psd_log_distance",
        ],
        ascending=False,
    )
    row = candidates.iloc[0]
    group = str(row["growth_run_id"])
    payload = evaluation["panels"][group]
    generated = payload["methods"][method]["medoid"]
    measured = payload["real_medoid"]
    limit = float(
        max(
            np.percentile(np.abs(measured), 99),
            np.percentile(np.abs(generated), 99),
        )
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))
    axes[0].imshow(
        measured,
        cmap="viridis",
        vmin=-limit,
        vmax=limit,
        origin="lower",
    )
    axes[1].imshow(
        generated,
        cmap="viridis",
        vmin=-limit,
        vmax=limit,
        origin="lower",
    )
    frequency_real, power_real = radial_psd(measured)
    frequency_generated, power_generated = radial_psd(generated)
    power_real = power_real / max(float(power_real.sum()), 1e-12)
    power_generated = power_generated / max(
        float(power_generated.sum()), 1e-12
    )
    axes[2].plot(
        frequency_real,
        power_real,
        marker="o",
        markersize=3,
        label="Measured",
    )
    axes[2].plot(
        frequency_generated,
        power_generated,
        marker="o",
        markersize=3,
        label="Generated",
    )
    axes[2].set_yscale("log")
    axes[2].set(
        xlabel="radial spatial frequency (pixel⁻¹ index)",
        ylabel="normalized PSD",
        title="Radial power spectrum",
    )
    axes[2].legend()
    axes[2].grid(alpha=0.2)
    axes[0].set_title("Measured unit-Rq AFM")
    axes[1].set_title("Generated unit-Rq AFM")
    for axis in axes[:2]:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"Automatically selected failure case: group {group}",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        f"Descriptor MAE = {float(row['condition_descriptor_mae_z']):.2f} z; "
        f"PSD distance = {float(row['normalized_psd_log_distance']):.2f}; "
        f"sharpness ratio = {float(row['sharpness_ratio']):.2f}. "
        "The generator under-resolves the largest connected islands.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    _save(figure, figure_dir / "Fig9_automatic_failure_case")


def make_sharp_generation_figures(
    *,
    evaluation: dict,
    condition_control: pd.DataFrame,
    training_history_path: Path | None,
    figure_dir: str | Path,
    phase1_manifest_path: str | Path | None = None,
) -> None:
    target = Path(figure_dir)
    target.mkdir(parents=True, exist_ok=True)
    _comparison_panel(evaluation, target)
    _metric_summary(evaluation, target)
    _fft_panel(evaluation, target)
    _condition_control(condition_control, target)
    if training_history_path is not None:
        _training_curves(Path(training_history_path), target)
    _all_draws(evaluation, target)
    if phase1_manifest_path is not None:
        _rheed_to_afm_panel(evaluation, phase1_manifest_path, target)
    _failure_case_panel(evaluation, target)
