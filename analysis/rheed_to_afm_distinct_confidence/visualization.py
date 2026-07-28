from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.rheed_video_afm_story.common import repo_path

from .run import PRIOR_METHOD, SELECTED_METHOD


COLORS = {
    "prior": "#7A5195",
    "raw": "#2F4B7C",
    "selected": "#00A087",
    "failure": "#E64B35",
    "uncertainty": "#F39B7F",
}


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=260, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


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
    lower, upper = np.percentile(crop, [1.0, 99.7])
    return np.clip(
        (crop.astype(float) - lower) / max(float(upper - lower), 1e-8),
        0,
        1,
    )


def _panel_physical(payload: dict[str, Any], method: str) -> tuple[np.ndarray, np.ndarray]:
    generated = (
        np.asarray(payload["methods"][method]["medoid"], dtype=float)
        * float(payload["methods"][method]["rq"])
    )
    measured = (
        np.asarray(payload["real_medoid"], dtype=float)
        * float(payload["true_rq"])
    )
    return generated, measured


def _expanded_atlas(
    *,
    panels: dict[str, dict[str, Any]],
    confidence: pd.DataFrame,
    phase1: pd.DataFrame,
    figure_dir: Path,
) -> None:
    confidence_lookup = confidence.set_index("growth_run_id")
    groups = sorted(
        panels,
        key=lambda group: float(panels[group]["true_rq"]),
    )
    lookup = phase1.set_index("growth_run_id")
    for page, start in enumerate(range(0, len(groups), 6), start=1):
        selected = groups[start : start + 6]
        physical_arrays = []
        for group in selected:
            generated, measured = _panel_physical(
                panels[group], SELECTED_METHOD
            )
            physical_arrays.extend([generated, measured])
        limit = max(
            1e-6,
            float(
                np.percentile(
                    np.abs(np.concatenate([array.ravel() for array in physical_arrays])),
                    99.2,
                )
            ),
        )
        figure, axes = plt.subplots(
            len(selected),
            3,
            figsize=(10.8, 2.65 * len(selected)),
            squeeze=False,
        )
        for row_index, group in enumerate(selected):
            payload = panels[group]
            generated, measured = _panel_physical(
                payload, SELECTED_METHOD
            )
            row = confidence_lookup.loc[group]
            axes[row_index, 0].imshow(
                _rheed_crop(lookup.loc[group]),
                cmap="gray",
                origin="upper",
                aspect="auto",
            )
            for axis, array in zip(
                axes[row_index, 1:], [generated, measured]
            ):
                axis.imshow(
                    array,
                    cmap="viridis",
                    vmin=-limit,
                    vmax=limit,
                    origin="lower",
                    interpolation="nearest",
                )
            if row_index == 0:
                axes[row_index, 0].set_title("RHEED key-frame ROI")
                axes[row_index, 1].set_title("Generated AFM (nm)")
                axes[row_index, 2].set_title("Measured AFM (nm)")
            axes[row_index, 0].set_ylabel(
                f"{group}  [{row['evidence_split']}]\n"
                f"confidence {row['relative_confidence_index']:.0f}/100",
                fontsize=9,
            )
            axes[row_index, 1].text(
                0.02,
                0.04,
                f"Rq {row['predicted_rq_nm']:.2f} nm\n"
                f"90% PI [{row['rq_interval_lower_nm']:.2f}, "
                f"{row['rq_interval_upper_nm']:.2f}]",
                transform=axes[row_index, 1].transAxes,
                fontsize=7.8,
                color="white",
                va="bottom",
                bbox={"facecolor": "black", "alpha": 0.58, "pad": 2},
            )
            axes[row_index, 2].text(
                0.02,
                0.04,
                f"Rq {row['true_rq_nm']:.2f} nm",
                transform=axes[row_index, 2].transAxes,
                fontsize=7.8,
                color="white",
                va="bottom",
                bbox={"facecolor": "black", "alpha": 0.58, "pad": 2},
            )
            for axis in axes[row_index]:
                axis.set_xticks([])
                axis.set_yticks([])
        figure.suptitle(
            "RHEED-conditioned AFM generation across measured roughness\n"
            f"fixed ordering, shared ±{limit:.1f} nm color scale",
            fontsize=14,
            fontweight="bold",
        )
        figure.text(
            0.5,
            0.004,
            "Training rows are strict leave-one-growth-group-out predictions; "
            "validation rows use the frozen training fit. Confidence is a "
            "conservative interval-width index, not a probability.",
            ha="center",
            fontsize=8,
        )
        figure.tight_layout(rect=(0, 0.025, 1, 0.96))
        _save(
            figure,
            figure_dir / f"Fig1{chr(96 + page)}_expanded_roughness_atlas",
        )


def _validation_differentiation(
    *,
    validation_panels: dict[str, dict[str, Any]],
    phase1: pd.DataFrame,
    confidence: pd.DataFrame,
    figure_dir: Path,
) -> None:
    groups = sorted(validation_panels)
    lookup = phase1.set_index("growth_run_id")
    confidence_lookup = confidence.set_index("growth_run_id")
    arrays: list[np.ndarray] = []
    rows: dict[str, list[np.ndarray]] = {}
    for group in groups:
        payload = validation_panels[group]
        prior, measured = _panel_physical(payload, PRIOR_METHOD)
        selected, _ = _panel_physical(payload, SELECTED_METHOD)
        rows[group] = [prior, selected, measured]
        arrays.extend(rows[group])
    limit = max(
        float(np.percentile(np.abs(np.concatenate([a.ravel() for a in arrays])), 99.0)),
        1e-6,
    )
    figure, axes = plt.subplots(
        len(groups), 4, figsize=(13.2, 3.0 * len(groups)), squeeze=False
    )
    for row_index, group in enumerate(groups):
        axes[row_index, 0].imshow(
            _rheed_crop(lookup.loc[group]),
            cmap="gray",
            origin="upper",
            aspect="auto",
        )
        for column, array in enumerate(rows[group], start=1):
            axes[row_index, column].imshow(
                array,
                cmap="viridis",
                vmin=-limit,
                vmax=limit,
                origin="lower",
                interpolation="nearest",
            )
        if row_index == 0:
            for axis, title in zip(
                axes[row_index],
                [
                    "RHEED ROI",
                    "Prior M2b",
                    "M5 hybrid generator",
                    "Measured AFM",
                ],
            ):
                axis.set_title(title)
        row = confidence_lookup.loc[group]
        axes[row_index, 0].set_ylabel(
            f"{group}\nconfidence {row['relative_confidence_index']:.0f}/100"
        )
        axes[row_index, 2].text(
            0.02,
            0.04,
            f"pred Rq {row['predicted_rq_nm']:.2f} nm",
            transform=axes[row_index, 2].transAxes,
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
        )
        axes[row_index, 3].text(
            0.02,
            0.04,
            f"true Rq {row['true_rq_nm']:.2f} nm",
            transform=axes[row_index, 3].transAxes,
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2},
        )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        "Validation condition differentiation at a shared physical height scale",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        f"All AFM panels share ±{limit:.1f} nm. No row is shown in unit-Rq "
        "scale; amplitude differences remain visible.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    _save(figure, figure_dir / "Fig2_validation_condition_differentiation")


def _metric_comparison(
    *,
    new_crossfit: pd.DataFrame,
    prior_crossfit_path: str | Path,
    figure_dir: Path,
) -> pd.DataFrame:
    prior = pd.read_csv(repo_path(prior_crossfit_path))
    prior = prior.loc[
        prior["method"] == "M2b_calibrated_spectral_rheed_condition"
    ].copy()
    prior["method"] = "Prior M2b spectral"
    selected = new_crossfit.loc[
        new_crossfit["method"] == SELECTED_METHOD
    ].copy()
    selected["method"] = "M5 hybrid + confidence"
    combined = pd.concat([prior, selected], ignore_index=True, sort=False)
    metrics = [
        ("rq_absolute_error_nm", "Rq absolute error (nm) ↓"),
        ("normalized_psd_log_distance", "PSD log distance ↓"),
        ("sharpness_ratio", "Sharpness / real (target 1)"),
        ("max_training_ssim", "Max training SSIM ↓"),
    ]
    figure, axes = plt.subplots(1, len(metrics), figsize=(14.5, 4.0))
    labels = ["Prior M2b spectral", "M5 hybrid + confidence"]
    for axis, (column, title) in zip(axes, metrics):
        values = [
            combined.loc[combined["method"] == label, column].median()
            for label in labels
        ]
        axis.bar(
            np.arange(2),
            values,
            color=[COLORS["prior"], COLORS["selected"]],
            width=0.65,
        )
        if column == "sharpness_ratio":
            axis.axhline(1.0, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(np.arange(2))
        axis.set_xticklabels(["Prior M2b", "M5"], rotation=20)
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Strict leave-one-growth-group-out generative comparison",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    _save(figure, figure_dir / "Fig3_crossfit_metric_comparison")
    return combined


def _confidence_figure(
    audit: pd.DataFrame, validation: pd.DataFrame, figure_dir: Path
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    scatter = axes[0].scatter(
        audit["relative_confidence_index"],
        audit["point_error_z"],
        c=audit["interval_width_z"],
        cmap="magma_r",
        s=58,
        edgecolor="black",
        linewidth=0.4,
    )
    for _, row in audit.iterrows():
        if (
            row["point_error_z"] >= audit["point_error_z"].quantile(0.8)
            or row["relative_confidence_index"]
            <= audit["relative_confidence_index"].quantile(0.2)
        ):
            axes[0].annotate(
                str(row["growth_run_id"]),
                (row["relative_confidence_index"], row["point_error_z"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
    axes[0].set_xlabel("Relative confidence index (not probability)")
    axes[0].set_ylabel("Realized descriptor error (z)")
    axes[0].set_title("Self-awareness audit")
    figure.colorbar(scatter, ax=axes[0], label="90% interval width (z)")
    axes[1].bar(
        audit["growth_run_id"],
        audit["component_coverage"],
        color=COLORS["selected"],
    )
    axes[1].axhline(0.90, color="black", linestyle="--", label="Nominal 90%")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=75, labelsize=7)
    axes[1].set_ylabel("Descriptor-component coverage")
    axes[1].set_title("Strict cross-fitted coverage")
    axes[1].legend(fontsize=8)
    x = np.arange(len(validation))
    axes[2].errorbar(
        x,
        validation["predicted_rq_nm"],
        yerr=np.vstack(
            [
                validation["predicted_rq_nm"]
                - validation["rq_interval_lower_nm"],
                validation["rq_interval_upper_nm"]
                - validation["predicted_rq_nm"],
            ]
        ),
        fmt="o",
        color=COLORS["selected"],
        ecolor=COLORS["uncertainty"],
        capsize=4,
        label="Prediction ± 90% PI",
    )
    axes[2].scatter(
        x,
        validation["true_rq_nm"],
        marker="x",
        s=70,
        color="black",
        label="Measured",
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(validation["growth_run_id"])
    axes[2].set_ylabel("Rq (nm)")
    axes[2].set_title("Validation roughness uncertainty")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    rho = audit[["interval_width_z", "point_error_z"]].corr(
        method="spearman"
    ).iloc[0, 1]
    figure.suptitle(
        f"Group-calibrated uncertainty: width–error Spearman ρ = {rho:.2f}",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    _save(figure, figure_dir / "Fig4_confidence_calibration")


def _ablation_and_learning(
    cap_ablation: pd.DataFrame,
    learning: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    axis2 = axes[0].twinx()
    axes[0].plot(
        cap_ablation["variance_cap"],
        cap_ablation["median_rq_mae_nm"],
        "o-",
        color=COLORS["failure"],
        label="Median Rq MAE",
    )
    axis2.plot(
        cap_ablation["variance_cap"],
        cap_ablation["condition_sensitivity_z"],
        "s-",
        color=COLORS["selected"],
        label="Condition sensitivity",
    )
    selected = cap_ablation.loc[cap_ablation["selected"]].iloc[0]
    axes[0].axvline(
        selected["variance_cap"], color="black", linestyle="--", linewidth=1
    )
    axes[0].set_xlabel("Nested variance-factor cap")
    axes[0].set_ylabel("Median Rq MAE (nm)", color=COLORS["failure"])
    axis2.set_ylabel("Pairwise condition sensitivity (z)", color=COLORS["selected"])
    axes[0].set_title("Sensitivity–error Pareto ablation")
    axes[1].fill_between(
        learning["training_group_count"],
        learning["descriptor_mae_z_q25"],
        learning["descriptor_mae_z_q75"],
        color=COLORS["selected"],
        alpha=0.2,
        label="IQR",
    )
    axes[1].plot(
        learning["training_group_count"],
        learning["descriptor_mae_z_median"],
        "o-",
        color=COLORS["selected"],
        linewidth=2,
        label="Median",
    )
    axes[1].set_xlabel("Training growth groups")
    axes[1].set_ylabel("Held-group descriptor MAE (z)")
    axes[1].set_title("Learning curve: more groups reduce error")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    _save(figure, figure_dir / "Fig5_ablation_and_learning_curve")


def _descriptor_correlations(
    predictions: pd.DataFrame, figure_dir: Path
) -> None:
    specs = [
        ("log_rq_nm", "log Rq"),
        ("log_unit_autocorr_length_nm", "log correlation length"),
        ("unit_psd_slope", "PSD slope"),
        ("unit_skewness", "Height skewness"),
    ]
    figure, axes = plt.subplots(1, len(specs), figsize=(14.0, 3.6))
    for axis, (column, label) in zip(axes, specs):
        true = predictions[f"true_raw__{column}"].to_numpy(float)
        predicted = predictions[
            f"selected_predicted_raw__{column}"
        ].to_numpy(float)
        axis.scatter(
            true,
            predicted,
            color=COLORS["selected"],
            edgecolor="black",
            linewidth=0.4,
        )
        lower = min(float(true.min()), float(predicted.min()))
        upper = max(float(true.max()), float(predicted.max()))
        axis.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        rho = pd.Series(true).corr(pd.Series(predicted), method="spearman")
        axis.set_title(f"{label}\nSpearman ρ = {rho:.2f}", fontsize=10)
        axis.set_xlabel("Measured")
        axis.set_ylabel("Cross-fitted prediction")
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Strict held-growth-group morphology descriptor correlations",
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout()
    _save(figure, figure_dir / "Fig6_descriptor_correlations")


def _failure_cases(
    *,
    panels: dict[str, dict[str, Any]],
    audit: pd.DataFrame,
    phase1: pd.DataFrame,
    figure_dir: Path,
) -> None:
    ranked = audit.copy()
    ranked["risk_rank"] = (
        ranked["point_error_z"].rank(pct=True)
        + (100.0 - ranked["relative_confidence_index"]).rank(pct=True)
    )
    choices = [
        str(ranked.sort_values("risk_rank", ascending=False).iloc[0]["growth_run_id"]),
        str(ranked.sort_values("point_error_z", ascending=False).iloc[0]["growth_run_id"]),
        str(ranked.sort_values("relative_confidence_index", ascending=False).iloc[0]["growth_run_id"]),
        str(ranked.sort_values("point_error_z").iloc[0]["growth_run_id"]),
    ]
    groups = list(dict.fromkeys(choices))
    for group in ranked.sort_values("risk_rank", ascending=False)[
        "growth_run_id"
    ].astype(str):
        if len(groups) >= 4:
            break
        if group not in groups:
            groups.append(group)
    lookup = phase1.set_index("growth_run_id")
    audit_lookup = audit.set_index("growth_run_id")
    figure, axes = plt.subplots(
        len(groups), 3, figsize=(10.5, 2.8 * len(groups)), squeeze=False
    )
    for row_index, group in enumerate(groups):
        generated, measured = _panel_physical(
            panels[group], SELECTED_METHOD
        )
        limit = max(
            float(
                np.percentile(
                    np.abs(np.concatenate([generated.ravel(), measured.ravel()])),
                    99.0,
                )
            ),
            1e-6,
        )
        axes[row_index, 0].imshow(
            _rheed_crop(lookup.loc[group]), cmap="gray", aspect="auto"
        )
        axes[row_index, 1].imshow(
            generated,
            cmap="viridis",
            vmin=-limit,
            vmax=limit,
            origin="lower",
        )
        axes[row_index, 2].imshow(
            measured,
            cmap="viridis",
            vmin=-limit,
            vmax=limit,
            origin="lower",
        )
        row = audit_lookup.loc[group]
        axes[row_index, 0].set_ylabel(
            f"{group}\nconf {row['relative_confidence_index']:.0f}/100\n"
            f"error {row['point_error_z']:.2f} z",
            fontsize=8.5,
        )
        if row_index == 0:
            axes[row_index, 0].set_title("RHEED ROI")
            axes[row_index, 1].set_title("Generated AFM")
            axes[row_index, 2].set_title("Measured AFM")
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        "Automatically selected confidence-aware successes and failures",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.01,
        "Cases are selected by precomputed cross-fitted confidence/error ranks, "
        "not manual visual preference.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    _save(figure, figure_dir / "Fig7_confidence_aware_failure_cases")


def make_figures(
    *,
    crossfit: dict[str, Any],
    validation: dict[str, Any],
    uncertainty: dict[str, pd.DataFrame],
    learning_curve: pd.DataFrame,
    phase1_manifest_path: str | Path,
    prior_crossfit_metrics: str | Path,
    figure_dir: str | Path,
) -> pd.DataFrame:
    target = Path(figure_dir)
    target.mkdir(parents=True, exist_ok=True)
    phase1 = pd.read_csv(repo_path(phase1_manifest_path), dtype=str)
    phase1["growth_run_id"] = phase1["growth_run_id"].astype(str)
    audit = uncertainty["audit"].copy()
    audit["growth_run_id"] = audit["growth_run_id"].astype(str)
    audit["evidence_split"] = "strict LOO"
    validation_uncertainty = uncertainty["validation"].copy()
    validation_uncertainty["growth_run_id"] = validation_uncertainty[
        "growth_run_id"
    ].astype(str)
    validation_uncertainty["evidence_split"] = "validation"
    confidence = pd.concat(
        [audit, validation_uncertainty], ignore_index=True, sort=False
    )
    all_panels = {**crossfit["panels"], **validation["panels"]}
    _expanded_atlas(
        panels=all_panels,
        confidence=confidence,
        phase1=phase1,
        figure_dir=target,
    )
    _validation_differentiation(
        validation_panels=validation["panels"],
        phase1=phase1,
        confidence=validation_uncertainty,
        figure_dir=target,
    )
    comparison = _metric_comparison(
        new_crossfit=crossfit["per_group"],
        prior_crossfit_path=prior_crossfit_metrics,
        figure_dir=target,
    )
    _confidence_figure(audit, validation_uncertainty, target)
    _ablation_and_learning(
        crossfit["cap_ablation"], learning_curve, target
    )
    _descriptor_correlations(crossfit["condition_predictions"], target)
    _failure_cases(
        panels=crossfit["panels"],
        audit=audit,
        phase1=phase1,
        figure_dir=target,
    )
    return comparison
