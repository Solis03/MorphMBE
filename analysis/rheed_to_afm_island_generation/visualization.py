from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from skimage import measure

from analysis.rheed_to_afm_distinct_confidence.run import _load_tables
from analysis.rheed_to_afm_distinct_confidence.visualization import _rheed_crop
from analysis.rheed_to_afm_sharp_generation.spectral import load_unit_map
from analysis.rheed_video_afm_story.common import repo_path, write_csv, write_json

from .islands import extract_island_features
from .train_guided_diffusion import _load_config


M5 = "M5_cloudlike_spectral_hybrid"
M6C = "M6c_island_structure_plus_spectral_prior"
M7 = "M7_structure_guided_residual_diffusion"
M10_CROSSFIT = "M10_dense_island_spectral_pareto"
M10_VALIDATION = "M10_dense_island_spectral_pareto"

COLORS = {
    M5: "#7A5195",
    M6C: "#2F4B7C",
    M7: "#E69F00",
    M10_CROSSFIT: "#009E73",
    "real": "#333333",
    "failure": "#D55E00",
}


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _npz_arrays(path: Path) -> tuple[list[np.ndarray], float]:
    payload = np.load(path, allow_pickle=False)
    return (
        [
            np.asarray(array, dtype=np.float32)
            for array in payload["generated_unit_shapes"]
        ],
        float(payload["predicted_rq_nm"]),
    )


def _medoid(arrays: list[np.ndarray]) -> np.ndarray:
    if len(arrays) == 1:
        return arrays[0]
    distances = np.zeros((len(arrays), len(arrays)), dtype=float)
    for first in range(len(arrays)):
        for second in range(first + 1, len(arrays)):
            value = float(
                np.mean(np.abs(arrays[first] - arrays[second]))
            )
            distances[first, second] = value
            distances[second, first] = value
    return arrays[int(np.argmin(distances.mean(axis=1)))]


def _real_group(rows: pd.DataFrame, resolution: int) -> tuple[np.ndarray, float]:
    arrays = [load_unit_map(row, resolution) for _, row in rows.iterrows()]
    return _medoid(arrays), float(rows["rq_nm"].median())


def _add_scale_bar(axis: plt.Axes, pixels: int, *, label: bool = True) -> None:
    length = int(round(0.2 * pixels))
    x0 = int(round(0.06 * pixels))
    y0 = int(round(0.92 * pixels))
    axis.plot(
        [x0, x0 + length],
        [y0, y0],
        color="white",
        linewidth=3.0,
        solid_capstyle="butt",
    )
    if label:
        axis.text(
            x0 + length / 2,
            y0 - 0.035 * pixels,
            "200 nm",
            ha="center",
            va="bottom",
            color="white",
            fontsize=7,
            weight="bold",
        )


def _imshow_height(
    axis: plt.Axes,
    image: np.ndarray,
    *,
    low: float,
    high: float,
    title: str,
    scale_bar: bool = True,
) -> Any:
    shown = axis.imshow(
        image,
        cmap="viridis",
        vmin=low,
        vmax=high,
        origin="upper",
    )
    axis.set_title(title, fontsize=9)
    axis.set_axis_off()
    if scale_bar:
        _add_scale_bar(axis, image.shape[1])
    return shown


def _phase_lookup(phase1: pd.DataFrame) -> pd.DataFrame:
    return (
        phase1.sort_values(["growth_run_id", "sample_id"])
        .drop_duplicates("growth_run_id")
        .assign(growth_run_id=lambda x: x["growth_run_id"].astype(str))
        .set_index("growth_run_id")
    )


def _confidence_tables(
    parent_report: Path, morphology_report: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cross = pd.read_csv(
        parent_report / "uncertainty_training_audit.csv",
        dtype={"growth_run_id": str},
    )
    validation = pd.read_csv(
        parent_report / "uncertainty_validation_predictions.csv",
        dtype={"growth_run_id": str},
    )
    morphology_cross = pd.read_csv(
        morphology_report / "morphology_confidence_crossfit.csv",
        dtype={"growth_run_id": str},
    )
    morphology_validation = pd.read_csv(
        morphology_report / "morphology_confidence_validation.csv",
        dtype={"growth_run_id": str},
    )
    cross = cross.merge(morphology_cross, on="growth_run_id")
    validation = validation.merge(
        morphology_validation, on="growth_run_id"
    )
    return cross, validation


def _validation_panel(
    *,
    descriptors: pd.DataFrame,
    phase: pd.DataFrame,
    confidence: pd.DataFrame,
    m6_root: Path,
    m8_root: Path,
    output: Path,
    resolution: int,
) -> None:
    groups = sorted(
        descriptors.loc[descriptors["split"] == "val", "growth_run_id"]
        .astype(str)
        .unique()
    )
    confidence_lookup = confidence.set_index("growth_run_id")
    figure, axes = plt.subplots(
        len(groups), 4, figsize=(12.0, 3.05 * len(groups)), squeeze=False
    )
    phase_lookup = _phase_lookup(phase)
    last_image = None
    for row_index, group in enumerate(groups):
        rows = descriptors.loc[
            descriptors["growth_run_id"].astype(str) == group
        ]
        real_unit, true_rq = _real_group(rows, resolution)
        m5_arrays, predicted_rq = _npz_arrays(
            m6_root / M5 / f"{group}.npz"
        )
        m8_arrays, _ = _npz_arrays(
            m8_root / "generated_maps" / M10_VALIDATION / f"{group}.npz"
        )
        m5 = _medoid(m5_arrays) * predicted_rq
        m8 = _medoid(m8_arrays) * predicted_rq
        real = real_unit * true_rq
        low, high = np.percentile(
            np.concatenate([m5.ravel(), m8.ravel(), real.ravel()]),
            [1.0, 99.0],
        )
        axes[row_index, 0].imshow(
            _rheed_crop(phase_lookup.loc[group]),
            cmap="gray",
            aspect="auto",
        )
        axes[row_index, 0].set_axis_off()
        confidence_value = float(
            confidence_lookup.loc[group, "relative_confidence_index"]
        )
        morphology_confidence = float(
            confidence_lookup.loc[group, "morphology_confidence_index"]
        )
        predicted_island_error = float(
            confidence_lookup.loc[group, "predicted_island_error_z"]
        )
        axes[row_index, 0].set_title(
            f"RHEED {group}\ncondition conf {confidence_value:.0f}/100; "
            f"morphology conf {morphology_confidence:.0f}/100\n"
            f"expected island error {predicted_island_error:.2f}z",
            fontsize=9,
        )
        _imshow_height(
            axes[row_index, 1],
            m5,
            low=low,
            high=high,
            title=f"M5 cloud field\nRq={predicted_rq:.2f} nm",
        )
        _imshow_height(
            axes[row_index, 2],
            m8,
            low=low,
            high=high,
            title=f"M10 dense islands\nRq={predicted_rq:.2f} nm",
        )
        last_image = _imshow_height(
            axes[row_index, 3],
            real,
            low=low,
            high=high,
            title=(
                f"Measured AFM\nRq={true_rq:.2f} nm; "
                f"shared z=[{low:.1f}, {high:.1f}] nm"
            ),
        )
    figure.suptitle(
        "RHEED-conditioned AFM generation on the pre-existing validation cohort",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.004,
        "M10 uses no measured AFM at inference; confidence is a calibrated "
        "relative index, not a probability. Each row uses a shared height scale.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.965))
    _save(figure, output / "Fig1_validation_rheed_m5_m10_real")


def _atlas(
    *,
    descriptors: pd.DataFrame,
    phase: pd.DataFrame,
    cross_confidence: pd.DataFrame,
    validation_confidence: pd.DataFrame,
    cross_root: Path,
    validation_root: Path,
    output: Path,
    resolution: int,
) -> None:
    phase_lookup = _phase_lookup(phase)
    confidence = pd.concat(
        [cross_confidence, validation_confidence], ignore_index=True
    ).set_index("growth_run_id")
    records = []
    for group, rows in descriptors.loc[
        descriptors["split"].isin(["train", "val"])
    ].groupby(descriptors["growth_run_id"].astype(str)):
        split = str(rows.iloc[0]["split"])
        real_unit, true_rq = _real_group(rows, resolution)
        if split == "train":
            path = cross_root / "generated_maps" / M10_CROSSFIT / f"{group}.npz"
        else:
            path = (
                validation_root
                / "generated_maps"
                / M10_VALIDATION
                / f"{group}.npz"
            )
        arrays, predicted_rq = _npz_arrays(path)
        records.append(
            {
                "group": group,
                "split": split,
                "real": real_unit * true_rq,
                "generated": _medoid(arrays) * predicted_rq,
                "true_rq": true_rq,
                "predicted_rq": predicted_rq,
            }
        )
    records.sort(key=lambda value: float(value["true_rq"]))
    for page, start in enumerate(range(0, len(records), 6), start=1):
        selected = records[start : start + 6]
        figure, axes = plt.subplots(
            len(selected), 3, figsize=(10.6, 2.65 * len(selected)), squeeze=False
        )
        for row_index, record in enumerate(selected):
            group = str(record["group"])
            axes[row_index, 0].imshow(
                _rheed_crop(phase_lookup.loc[group]),
                cmap="gray",
                aspect="auto",
            )
            axes[row_index, 0].set_axis_off()
            value = float(
                confidence.loc[group, "relative_confidence_index"]
            )
            morphology_value = float(
                confidence.loc[group, "morphology_confidence_index"]
            )
            axes[row_index, 0].set_title(
                f"RHEED {group} ({record['split']})\ncondition conf "
                f"{value:.0f}; morphology conf {morphology_value:.0f}/100",
                fontsize=8.5,
            )
            low, high = np.percentile(
                np.concatenate(
                    [
                        record["generated"].ravel(),
                        record["real"].ravel(),
                    ]
                ),
                [1.0, 99.0],
            )
            shown = _imshow_height(
                axes[row_index, 1],
                record["generated"],
                low=low,
                high=high,
                title=f"M10 generated\nRq={record['predicted_rq']:.2f} nm",
            )
            _imshow_height(
                axes[row_index, 2],
                record["real"],
                low=low,
                high=high,
                title=(
                    f"Measured AFM\nRq={record['true_rq']:.2f} nm; "
                    f"shared z=[{low:.1f}, {high:.1f}] nm"
                ),
            )
        figure.suptitle(
            "Expanded M10 morphology atlas sorted by measured roughness "
            f"(page {page})",
            fontsize=12.5,
        )
        figure.text(
            0.5,
            0.004,
            "Train rows are strict held-growth LOO predictions; validation "
            "rows use the frozen all-training fit. No historical test rows.",
            ha="center",
            fontsize=8,
        )
        figure.tight_layout(rect=(0, 0.02, 1, 0.965))
        _save(
            figure,
            output / f"Fig2{chr(96 + page)}_expanded_m10_atlas",
        )


def _metric_figure(
    standard: pd.DataFrame,
    island: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    metrics = [
        (
            standard,
            "median_composite_score",
            "Composite error",
            "lower is better",
        ),
        (
            standard,
            "median_condition_descriptor_mae_z",
            "RHEED-condition MAE",
            "z units",
        ),
        (
            standard,
            "median_normalized_psd_log_distance",
            "PSD log distance",
            "lower is better",
        ),
        (
            island,
            "median_island_feature_mae_z",
            "Island-feature MAE",
            "z units",
        ),
        (
            island,
            "median_afm_prior_mahalanobis",
            "AFM-support distance",
            "lower is better",
        ),
        (
            standard,
            "texture_gate_pass_fraction",
            "AFM texture pass",
            "fraction",
        ),
    ]
    methods = [M5, M10_CROSSFIT]
    labels = ["M5 cloud", "M10 dense islands"]
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.0))
    table_records = []
    for axis, (frame, column, title, subtitle) in zip(
        axes.ravel(), metrics
    ):
        lookup = frame.set_index("method")
        values = [float(lookup.loc[method, column]) for method in methods]
        bars = axis.bar(
            np.arange(len(methods)),
            values,
            color=[COLORS[method] for method in methods],
        )
        axis.set_xticks(np.arange(len(methods)), labels, rotation=18, ha="right")
        axis.set_title(title, fontsize=10.5)
        axis.set_ylabel(subtitle, fontsize=8.5)
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        for method, value in zip(methods, values):
            table_records.append(
                {"method": method, "metric": column, "value": value}
            )
    figure.suptitle(
        "Strict 15-growth leave-one-group-out comparison", fontsize=13
    )
    figure.tight_layout()
    _save(figure, output / "Fig3_crossfit_baseline_vs_dense_islands")
    return pd.DataFrame(table_records)


def _contour(axis: plt.Axes, unit: np.ndarray) -> None:
    smooth = unit
    threshold = float(np.quantile(smooth, 0.70))
    for contour in measure.find_contours(smooth, threshold):
        axis.plot(contour[:, 1], contour[:, 0], color="cyan", linewidth=0.65)


def _topology_figure(
    *,
    descriptors: pd.DataFrame,
    m6_root: Path,
    validation_root: Path,
    output: Path,
    resolution: int,
) -> None:
    rows = descriptors.loc[descriptors["split"] == "val"]
    group_rq = rows.groupby(rows["growth_run_id"].astype(str))["rq_nm"].median()
    group = str((group_rq - group_rq.median()).abs().idxmin())
    held = rows.loc[rows["growth_run_id"].astype(str) == group]
    real_unit, _ = _real_group(held, resolution)
    m5 = _medoid(_npz_arrays(m6_root / M5 / f"{group}.npz")[0])
    m8 = _medoid(
        _npz_arrays(
            validation_root
            / "generated_maps"
            / M10_VALIDATION
            / f"{group}.npz"
        )[0]
    )
    images = [m5, m8, real_unit]
    labels = ["M5 cloud", "M10 dense islands", "Measured AFM"]
    features = [extract_island_features(image) for image in images]
    figure, axes = plt.subplots(2, 3, figsize=(11.8, 7.2))
    low, high = np.percentile(np.concatenate([x.ravel() for x in images]), [1, 99])
    for axis, image, label in zip(axes[0], images, labels):
        axis.imshow(image, cmap="viridis", vmin=low, vmax=high)
        _contour(axis, image)
        _add_scale_bar(axis, resolution)
        axis.set_axis_off()
        axis.set_title(label)
    metric_specs = [
        ("log_component_count_q70", "log island count"),
        ("log_median_area_q70", "log median island area (px)"),
        ("boundary_gradient_ratio_q70", "boundary gradient ratio"),
    ]
    for axis, (column, title) in zip(axes[1], metric_specs):
        values = [float(feature[column]) for feature in features]
        bars = axis.bar(
            np.arange(3),
            values,
            color=[COLORS[M5], COLORS[M10_CROSSFIT], COLORS["real"]],
        )
        axis.set_xticks(np.arange(3), labels, rotation=18, ha="right")
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle(
        f"Explicit island topology audit (validation growth {group}, "
        "cyan = q70 boundaries)",
        fontsize=12.5,
    )
    figure.tight_layout()
    _save(figure, output / "Fig4_island_boundary_and_topology_audit")


def _ablation_figure(
    *,
    m6_report: Path,
    diffusion_root: Path,
    output: Path,
) -> None:
    blend = pd.read_csv(
        m6_report / "crossfit" / "blend_ablation_summary.csv"
    )
    strength_records = []
    for strength in (0.25, 0.45, 0.70, 1.00):
        root = diffusion_root / f"validation_strength_{strength:.2f}"
        standard = pd.read_csv(root / "standard" / "method_summary.csv")
        island = pd.read_csv(root / "island_summary.csv")
        row_s = standard.loc[standard["method"] == M7].iloc[0]
        row_i = island.loc[island["method"] == M7].iloc[0]
        strength_records.append(
            {
                "strength": strength,
                "psd": row_s["median_normalized_psd_log_distance"],
                "sharpness": row_s["median_sharpness_ratio"],
                "texture": row_s["texture_gate_pass_fraction"],
                "island": row_i["island_feature_mae_z"],
            }
        )
    strength = pd.DataFrame(strength_records)
    curves = pd.read_csv(diffusion_root / "training_curves.csv")
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    axes[0].plot(
        blend["blend_weight"],
        blend["median_island_feature_mae_z"],
        "o-",
        color=COLORS[M6C],
        label="island MAE",
    )
    twin = axes[0].twinx()
    twin.plot(
        blend["blend_weight"],
        blend["median_afm_prior_mahalanobis"],
        "s--",
        color=COLORS[M10_CROSSFIT],
        label="AFM support distance",
    )
    axes[0].set_xlabel("Island-structure blend weight")
    axes[0].set_ylabel("Island MAE (z)")
    twin.set_ylabel("AFM-support distance")
    axes[0].set_title("M6 blend ablation")
    axes[1].plot(strength["strength"], strength["psd"], "o-", label="PSD")
    axes[1].plot(
        strength["strength"],
        strength["island"],
        "s-",
        label="island MAE",
    )
    axes[1].axvline(0.25, color=COLORS[M10_CROSSFIT], linestyle=":", label="selected")
    axes[1].set_xlabel("Diffusion strength")
    axes[1].set_ylabel("Error")
    axes[1].set_title("M7 strength ablation")
    axes[1].legend(fontsize=8)
    axes[2].plot(
        curves["step"],
        curves["training_loss"],
        color=COLORS[M7],
        alpha=0.8,
        label="training",
    )
    axes[2].plot(
        curves["step"],
        curves["validation_loss"],
        color=COLORS[M10_CROSSFIT],
        label="AFM-prior validation",
    )
    best = curves.loc[curves["validation_loss"].idxmin()]
    axes[2].scatter(
        [best["step"]],
        [best["validation_loss"]],
        color=COLORS["failure"],
        zorder=5,
        label=f"selected step {int(best['step'])}",
    )
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("Noise-prediction MSE")
    axes[2].set_title("Residual DDPM learning curve")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("Scientifically targeted ablations", fontsize=13)
    figure.tight_layout()
    _save(figure, output / "Fig5_ablation_and_diffusion_training")


def _confidence_figure(
    *,
    cross_standard: pd.DataFrame,
    cross_island: pd.DataFrame,
    cross_confidence: pd.DataFrame,
    validation_standard: pd.DataFrame,
    validation_confidence: pd.DataFrame,
    output: Path,
) -> dict[str, float]:
    cross_s = cross_standard.loc[
        cross_standard["method"] == M10_CROSSFIT
    ][["growth_run_id", "condition_descriptor_mae_z"]]
    cross_i = cross_island.loc[
        cross_island["method"] == M10_CROSSFIT
    ][["growth_run_id", "island_feature_mae_z"]]
    audit = (
        cross_confidence[
            [
                "growth_run_id",
                "relative_confidence_index",
                "morphology_confidence_index",
                "predicted_island_error_z",
                "island_error_90_upper_z",
            ]
        ]
        .merge(cross_s, on="growth_run_id")
        .merge(cross_i, on="growth_run_id")
    )
    rho_condition = spearmanr(
        audit["relative_confidence_index"],
        audit["condition_descriptor_mae_z"],
    )
    rho_island = spearmanr(
        audit["morphology_confidence_index"],
        audit["island_feature_mae_z"],
    )
    val_s = validation_standard.loc[
        validation_standard["method"] == M10_VALIDATION
    ][["growth_run_id", "condition_descriptor_mae_z"]]
    val_i = validation_standard.loc[
        validation_standard["method"] == M10_VALIDATION
    ][["growth_run_id"]].merge(
        validation_confidence[
            [
                "growth_run_id",
                "morphology_confidence_index",
                "predicted_island_error_z",
                "island_error_90_upper_z",
                "realized_island_error_z",
            ]
        ],
        on="growth_run_id",
    )
    val = validation_confidence[
        ["growth_run_id", "relative_confidence_index"]
    ].merge(val_s, on="growth_run_id")
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    axes[0].scatter(
        audit["relative_confidence_index"],
        audit["condition_descriptor_mae_z"],
        c=COLORS[M10_CROSSFIT],
        s=48,
        alpha=0.85,
    )
    axes[0].set_title(f"Condition error: Spearman ρ={rho_condition.statistic:.2f}")
    axes[0].set_ylabel("Condition descriptor MAE (z)")
    axes[1].scatter(
        audit["morphology_confidence_index"],
        audit["island_feature_mae_z"],
        c=COLORS[M7],
        s=48,
        alpha=0.85,
    )
    axes[1].set_title(
        f"Morphology confidence: Spearman ρ={rho_island.statistic:.2f}"
    )
    axes[1].set_ylabel("Island-feature MAE (z)")
    axes[2].scatter(
        val_i["morphology_confidence_index"],
        val_i["realized_island_error_z"],
        c=COLORS[M10_CROSSFIT],
        s=65,
    )
    for _, row in val_i.iterrows():
        axes[2].annotate(
            str(row["growth_run_id"]),
            (
                row["morphology_confidence_index"],
                row["realized_island_error_z"],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axes[2].set_title("Validation morphology confidence")
    axes[2].set_ylabel("Island-feature MAE (z)")
    axes[0].set_xlabel("Condition confidence index (not probability)")
    axes[1].set_xlabel("Morphology confidence index (not probability)")
    axes[2].set_xlabel("Morphology confidence index (not probability)")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Confidence audit: high confidence should correspond to lower error",
        fontsize=12.5,
    )
    figure.tight_layout()
    _save(figure, output / "Fig6_confidence_vs_m10_error")
    return {
        "confidence_vs_condition_error_spearman": float(
            rho_condition.statistic
        ),
        "confidence_vs_condition_error_pvalue": float(rho_condition.pvalue),
        "morphology_confidence_vs_island_error_spearman": float(
            rho_island.statistic
        ),
        "morphology_confidence_vs_island_error_pvalue": float(
            rho_island.pvalue
        ),
    }


def _failure_figure(
    *,
    descriptors: pd.DataFrame,
    phase: pd.DataFrame,
    cross_standard: pd.DataFrame,
    cross_island: pd.DataFrame,
    confidence: pd.DataFrame,
    cross_root: Path,
    output: Path,
    resolution: int,
) -> None:
    metrics = (
        cross_standard.loc[
            cross_standard["method"] == M10_CROSSFIT,
            ["growth_run_id", "condition_descriptor_mae_z"],
        ]
        .merge(
            cross_island.loc[
                cross_island["method"] == M10_CROSSFIT,
                ["growth_run_id", "island_feature_mae_z"],
            ],
            on="growth_run_id",
        )
        .merge(
            confidence[
                [
                    "growth_run_id",
                    "relative_confidence_index",
                    "morphology_confidence_index",
                    "predicted_island_error_z",
                ]
            ],
            on="growth_run_id",
        )
    )
    metrics["combined_error_rank"] = (
        metrics["condition_descriptor_mae_z"].rank(pct=True)
        + metrics["island_feature_mae_z"].rank(pct=True)
    )
    high_success = str(
        metrics.loc[
            metrics["morphology_confidence_index"]
            >= metrics["morphology_confidence_index"].median()
        ]
        .sort_values("combined_error_rank")
        .iloc[0]["growth_run_id"]
    )
    low_failure = str(
        metrics.loc[
            metrics["morphology_confidence_index"]
            <= metrics["morphology_confidence_index"].median()
        ]
        .sort_values("combined_error_rank", ascending=False)
        .iloc[0]["growth_run_id"]
    )
    worst = str(
        metrics.sort_values("combined_error_rank", ascending=False).iloc[0][
            "growth_run_id"
        ]
    )
    best = str(metrics.sort_values("combined_error_rank").iloc[0]["growth_run_id"])
    groups = list(dict.fromkeys([high_success, best, low_failure, worst]))
    phase_lookup = _phase_lookup(phase)
    metric_lookup = metrics.set_index("growth_run_id")
    figure, axes = plt.subplots(
        len(groups), 3, figsize=(10.5, 2.8 * len(groups)), squeeze=False
    )
    for row_index, group in enumerate(groups):
        rows = descriptors.loc[
            descriptors["growth_run_id"].astype(str) == group
        ]
        real_unit, true_rq = _real_group(rows, resolution)
        arrays, predicted_rq = _npz_arrays(
            cross_root
            / "generated_maps"
            / M10_CROSSFIT
            / f"{group}.npz"
        )
        generated = _medoid(arrays) * predicted_rq
        real = real_unit * true_rq
        low, high = np.percentile(
            np.concatenate([generated.ravel(), real.ravel()]), [1, 99]
        )
        row = metric_lookup.loc[group]
        axes[row_index, 0].imshow(
            _rheed_crop(phase_lookup.loc[group]), cmap="gray", aspect="auto"
        )
        axes[row_index, 0].set_axis_off()
        axes[row_index, 0].set_title(
            f"RHEED {group}\ncondition conf "
            f"{row['relative_confidence_index']:.0f}; morphology conf "
            f"{row['morphology_confidence_index']:.0f}/100\n"
            f"expected island error {row['predicted_island_error_z']:.2f}z",
            fontsize=9,
        )
        _imshow_height(
            axes[row_index, 1],
            generated,
            low=low,
            high=high,
            title=(
                f"M10 generated\ncondition err "
                f"{row['condition_descriptor_mae_z']:.2f}z, "
                f"island err {row['island_feature_mae_z']:.2f}z"
            ),
        )
        _imshow_height(
            axes[row_index, 2],
            real,
            low=low,
            high=high,
            title=f"Measured AFM\nRq={true_rq:.2f} nm",
        )
    figure.suptitle(
        "Predefined confidence-aware success and failure audit", fontsize=12.5
    )
    figure.text(
        0.5,
        0.006,
        "Cases are selected algorithmically from strict cross-fitted error and "
        "confidence ranks; no visual cherry-picking.",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.025, 1, 0.96))
    _save(figure, output / "Fig7_confidence_aware_successes_and_failures")


def _correlation_figure(
    *,
    cross_standard: pd.DataFrame,
    cross_island: pd.DataFrame,
    output: Path,
) -> None:
    standard = cross_standard.loc[
        cross_standard["method"] == M10_CROSSFIT
    ]
    island = cross_island.loc[cross_island["method"] == M10_CROSSFIT]
    specs = [
        (
            standard,
            "true_rq_nm",
            "generated_rq_nm",
            "Rq (nm)",
        ),
        (
            island,
            "true__log_component_count_q70",
            "generated__log_component_count_q70",
            "log island count (q70)",
        ),
        (
            island,
            "true__log_median_area_q70",
            "generated__log_median_area_q70",
            "log median island area (px)",
        ),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))
    for axis, (frame, true_column, generated_column, label) in zip(
        axes, specs
    ):
        x = frame[true_column].to_numpy(float)
        y = frame[generated_column].to_numpy(float)
        rho = spearmanr(x, y)
        axis.scatter(x, y, c=COLORS[M10_CROSSFIT], s=50, alpha=0.85)
        low = min(float(x.min()), float(y.min()))
        high = max(float(x.max()), float(y.max()))
        axis.plot([low, high], [low, high], "--", color="0.4", linewidth=1)
        axis.set_xlabel(f"Measured {label}")
        axis.set_ylabel(f"Generated {label}")
        axis.set_title(f"Spearman ρ={rho.statistic:.2f}")
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Strict cross-fitted physical and island-descriptor correlations",
        fontsize=12.5,
    )
    figure.tight_layout()
    _save(figure, output / "Fig8_descriptor_correlations")


def make_figures(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    tables = _load_tables(config)
    descriptors = tables["descriptors"].copy()
    phase = tables["phase1"].copy()
    output = repo_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    parent_report = repo_path(args.parent_report)
    morphology_confidence_report = repo_path(
        args.morphology_confidence_report
    )
    cross_confidence, validation_confidence = _confidence_tables(
        parent_report, morphology_confidence_report
    )
    cross_report = repo_path(args.cross_report)
    cross_root = repo_path(args.cross_root)
    validation_root = repo_path(args.validation_root)
    validation_report = repo_path(args.validation_report)
    m6_root = repo_path(args.m6_validation_root)
    m6_report = repo_path(args.m6_report)
    diffusion_root = repo_path(args.diffusion_root)
    cross_standard = pd.read_csv(
        cross_report / "standard_per_group.csv",
        dtype={"growth_run_id": str},
    )
    cross_island = pd.read_csv(
        cross_report / "island_per_group.csv",
        dtype={"growth_run_id": str},
    )
    standard_summary = pd.read_csv(cross_report / "standard_summary.csv")
    island_summary = pd.read_csv(cross_report / "island_summary.csv")
    validation_standard = pd.read_csv(
        validation_report / "standard" / "per_group_metrics.csv",
        dtype={"growth_run_id": str},
    )
    _validation_panel(
        descriptors=descriptors,
        phase=phase,
        confidence=validation_confidence,
        m6_root=m6_root,
        m8_root=validation_root,
        output=output,
        resolution=int(config["resolution"]),
    )
    _atlas(
        descriptors=descriptors,
        phase=phase,
        cross_confidence=cross_confidence,
        validation_confidence=validation_confidence,
        cross_root=cross_root,
        validation_root=validation_root,
        output=output,
        resolution=int(config["resolution"]),
    )
    metric_table = _metric_figure(standard_summary, island_summary, output)
    write_csv(metric_table, output.parent / "baseline_vs_final_metrics_long.csv")
    _topology_figure(
        descriptors=descriptors,
        m6_root=m6_root,
        validation_root=validation_root,
        output=output,
        resolution=int(config["resolution"]),
    )
    _ablation_figure(
        m6_report=m6_report,
        diffusion_root=diffusion_root,
        output=output,
    )
    confidence_audit = _confidence_figure(
        cross_standard=cross_standard,
        cross_island=cross_island,
        cross_confidence=cross_confidence,
        validation_standard=validation_standard,
        validation_confidence=validation_confidence,
        output=output,
    )
    _failure_figure(
        descriptors=descriptors,
        phase=phase,
        cross_standard=cross_standard,
        cross_island=cross_island,
        confidence=cross_confidence,
        cross_root=cross_root,
        output=output,
        resolution=int(config["resolution"]),
    )
    _correlation_figure(
        cross_standard=cross_standard,
        cross_island=cross_island,
        output=output,
    )
    write_json(
        {
            **confidence_audit,
            "figure_count_png": len(list(output.glob("*.png"))),
            "figure_count_pdf": len(list(output.glob("*.pdf"))),
            "historical_test_used": False,
        },
        output.parent / "visualization_manifest.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_to_afm_island_generation_v2.json",
    )
    parser.add_argument("--cross-report", required=True)
    parser.add_argument("--cross-root", required=True)
    parser.add_argument("--validation-root", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--m6-validation-root", required=True)
    parser.add_argument("--m6-report", required=True)
    parser.add_argument("--diffusion-root", required=True)
    parser.add_argument("--parent-report", required=True)
    parser.add_argument("--morphology-confidence-report", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    make_figures(build_parser().parse_args())


if __name__ == "__main__":
    main()
