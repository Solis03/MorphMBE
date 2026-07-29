from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr, spearmanr

from analysis.rheed_to_afm_distinct_confidence.run import _load_tables
from analysis.rheed_video_afm_story.common import repo_path, write_json

from .run import M10, load_config


SELECTED_COLOR = "#0072B2"
BASELINE_COLOR = "#999999"
REAL_COLOR = "#D55E00"


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.dpi": 130,
            "savefig.dpi": 360,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _generated(
    output: Path, *, split: str, method: str, group: str
) -> tuple[np.ndarray, float]:
    payload = np.load(
        output
        / split
        / "generated_maps"
        / method
        / f"{group}.npz",
        allow_pickle=False,
    )
    unit = np.asarray(payload["generated_unit_shapes"][0], dtype=float)
    rq = float(payload["predicted_rq_nm"])
    return unit * rq, rq


def _phase_row(phase1: pd.DataFrame, group: str) -> pd.Series:
    row = phase1.loc[
        phase1["growth_run_id"].astype(str) == str(group)
    ]
    if len(row) != 1:
        raise ValueError(f"phase-1 row is not unique for {group}")
    return row.iloc[0]


def _rheed_keyframe(phase1: pd.DataFrame, group: str) -> np.ndarray:
    row = _phase_row(phase1, group)
    paths = json.loads(str(row["clip_frame_paths"]))
    offset = int(row.get("keyframe_offset_in_clip_x", len(paths) // 2))
    if not paths:
        cache = np.load(
            repo_path(str(row["clip_cache_path"])),
            allow_pickle=False,
        )
        frames = np.asarray(cache["frames_uint8"], dtype=float)
        if frames.ndim != 3 or not len(frames):
            raise ValueError(f"cached RHEED clip is empty for {group}")
        return frames[int(np.clip(offset, 0, len(frames) - 1))]
    path = repo_path(paths[int(np.clip(offset, 0, len(paths) - 1))])
    return np.asarray(Image.open(path).convert("L"), dtype=float)


def _real_afm(phase1: pd.DataFrame, group: str) -> np.ndarray:
    row = _phase_row(phase1, group)
    return np.asarray(
        np.load(repo_path(str(row["representative_afm_height_array"]))),
        dtype=float,
    )


def _real_afm_label(phase1: pd.DataFrame, group: str) -> dict[str, float]:
    """Return scan-level Sq and separately declared sample-level statistics."""

    row = _phase_row(phase1, group)
    array = _real_afm(phase1, group)
    centered = array - float(np.nanmean(array))
    displayed_sq = float(np.sqrt(np.nanmean(np.square(centered))))
    return {
        "displayed_scan_sq_nm": displayed_sq,
        "sample_median_sq_nm": float(row["primary_rq_nm_median"]),
        "sample_sq_iqr_nm": float(row.get("primary_rq_nm_iqr", np.nan)),
    }


def _scale_bar(axis: plt.Axes, *, pixels: int, length_nm: float = 250) -> None:
    width = 0.25 * pixels
    y = 0.91 * pixels
    x0 = 0.67 * pixels
    axis.plot([x0, x0 + width], [y, y], color="white", lw=3)
    axis.text(
        x0 + width / 2,
        y - 0.04 * pixels,
        f"{length_nm:.0f} nm",
        color="white",
        ha="center",
        va="bottom",
        fontsize=7,
        path_effects=[],
    )


def _surface_panel(
    axis: plt.Axes,
    array: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    title: str,
) -> mpl.image.AxesImage:
    image = axis.imshow(
        array,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axis.set_title(title, fontsize=8.2, linespacing=1.15)
    axis.set_xticks([])
    axis.set_yticks([])
    _scale_bar(axis, pixels=array.shape[1])
    return image


def _comparison_figure(
    *,
    groups: list[str],
    split: str,
    output: Path,
    phase1: pd.DataFrame,
    rq_predictions: pd.DataFrame,
    fsmi_predictions: pd.DataFrame,
    confidence: pd.DataFrame,
    method: str,
    title: str,
) -> plt.Figure:
    rq = rq_predictions.set_index("growth_run_id")
    fsmi = fsmi_predictions.set_index("growth_run_id")
    conf = confidence.set_index("growth_run_id")
    figure, axes = plt.subplots(
        len(groups),
        3,
        figsize=(10.6, 2.9 * len(groups)),
        constrained_layout=True,
        squeeze=False,
    )
    figure.suptitle(title, fontsize=12, fontweight="bold")
    for row_index, group in enumerate(groups):
        rheed = _rheed_keyframe(phase1, group)
        generated, _ = _generated(
            output, split=split, method=method, group=group
        )
        real = _real_afm(phase1, group)
        real_label = _real_afm_label(phase1, group)
        combined = np.concatenate([generated.ravel(), real.ravel()])
        vmin, vmax = np.quantile(combined, [0.01, 0.99])

        axes[row_index, 0].imshow(rheed, cmap="gray")
        axes[row_index, 0].set_xticks([])
        axes[row_index, 0].set_yticks([])
        axes[row_index, 0].set_title(
            f"{group}  RHEED key frame", fontsize=8.5
        )
        image = _surface_panel(
            axes[row_index, 1],
            generated,
            vmin=float(vmin),
            vmax=float(vmax),
            title=(
                f"generated AFM\n"
                f"predicted Sq={rq.loc[group, 'predicted_target']:.2f} nm; "
                f"FSMI={fsmi.loc[group, 'predicted_target']:.2f} nm\n"
                f"C={conf.loc[group, 'joint_confidence_index']:.0f}/100"
            ),
        )
        _surface_panel(
            axes[row_index, 2],
            real,
            vmin=float(vmin),
            vmax=float(vmax),
            title=(
                f"measured AFM\n"
                f"displayed scan Sq="
                f"{real_label['displayed_scan_sq_nm']:.2f} nm\n"
                f"sample median Sq="
                f"{real_label['sample_median_sq_nm']:.2f} ± "
                f"{real_label['sample_sq_iqr_nm']:.2f} nm (IQR)"
            ),
        )
        colorbar = figure.colorbar(
            image,
            ax=[axes[row_index, 1], axes[row_index, 2]],
            fraction=0.035,
            pad=0.015,
        )
        colorbar.set_label("height (nm)")
    return figure


def plot_validation(
    *,
    figure_dir: Path,
    output: Path,
    phase1: pd.DataFrame,
    rq: pd.DataFrame,
    fsmi: pd.DataFrame,
    confidence: pd.DataFrame,
    method: str,
) -> None:
    groups = list(rq["growth_run_id"].astype(str))
    figure = _comparison_figure(
        groups=groups,
        split="validation",
        output=output,
        phase1=phase1,
        rq_predictions=rq,
        fsmi_predictions=fsmi,
        confidence=confidence,
        method=method,
        title=(
            "Pre-existing validation: RHEED → generated AFM → measured AFM"
        ),
    )
    _save(figure, figure_dir / "Fig1_validation_rheed_generated_real")


def plot_crossfit_atlas(
    *,
    figure_dir: Path,
    output: Path,
    phase1: pd.DataFrame,
    rq: pd.DataFrame,
    fsmi: pd.DataFrame,
    confidence: pd.DataFrame,
    method: str,
) -> None:
    ordered = rq.sort_values("true_target")
    groups = list(ordered["growth_run_id"].astype(str))
    for page, start in enumerate(range(0, len(groups), 5), start=1):
        subset = groups[start : start + 5]
        figure = _comparison_figure(
            groups=subset,
            split="crossfit",
            output=output,
            phase1=phase1,
            rq_predictions=rq,
            fsmi_predictions=fsmi,
            confidence=confidence,
            method=method,
            title=(
                "Strict held-one-growth predictions ordered by measured Sq "
                f"(page {page}/3)"
            ),
        )
        _save(
            figure,
            figure_dir / f"Fig2{chr(96 + page)}_held_one_atlas",
        )


def plot_target_scatter(
    *,
    figure_dir: Path,
    rq: pd.DataFrame,
    fsmi: pd.DataFrame,
    confidence: pd.DataFrame,
) -> None:
    conf = confidence.set_index("growth_run_id")
    figure, axes = plt.subplots(
        1, 2, figsize=(8.2, 3.65), constrained_layout=True
    )
    for axis, table, label in (
        (axes[0], rq, "Sq (nm)"),
        (axes[1], fsmi, "FSMI (nm)"),
    ):
        table = table.copy()
        c = table["growth_run_id"].map(
            conf["joint_confidence_index"]
        )
        lower = (
            table["predicted_target"] - table["interval_lower"]
        ).to_numpy(float)
        upper = (
            table["interval_upper"] - table["predicted_target"]
        ).to_numpy(float)
        scatter = axis.scatter(
            table["true_target"],
            table["predicted_target"],
            c=c,
            cmap="viridis",
            vmin=0,
            vmax=100,
            s=45,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        axis.errorbar(
            table["true_target"],
            table["predicted_target"],
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor="#777777",
            alpha=0.55,
            lw=0.8,
            zorder=1,
        )
        lo = float(
            min(table["true_target"].min(), table["interval_lower"].min())
        )
        hi = float(
            max(table["true_target"].max(), table["interval_upper"].max())
        )
        axis.plot([lo, hi], [lo, hi], "--", color="black", lw=1)
        axis.set_xlim(max(0.0, lo - 0.2), hi + 0.2)
        axis.set_ylim(max(0.0, lo - 0.2), hi + 0.2)
        axis.set_xlabel(f"measured {label}")
        axis.set_ylabel(f"predicted {label}")
        rho = spearmanr(
            table["true_target"], table["predicted_target"]
        )
        r = pearsonr(
            table["true_target"], table["predicted_target"]
        )
        axis.set_title(
            f"{label}: Pearson r={r.statistic:.2f}, "
            f"Spearman ρ={rho.statistic:.2f}"
        )
        for _, row in table.iterrows():
            if (
                row["absolute_error"]
                >= table["absolute_error"].quantile(0.85)
            ):
                axis.annotate(
                    str(row["growth_run_id"]),
                    (row["true_target"], row["predicted_target"]),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=7,
                )
    colorbar = figure.colorbar(scatter, ax=axes, shrink=0.82)
    colorbar.set_label("joint confidence index (0–100)")
    _save(figure, figure_dir / "Fig3_rq_fsmi_prediction_scatter")


def plot_dynamic_range(
    *,
    figure_dir: Path,
    rq: pd.DataFrame,
    cross_standard: pd.DataFrame,
    method: str,
) -> None:
    selected = rq.sort_values("true_target").copy()
    baseline = (
        cross_standard.loc[cross_standard["method"] == M10]
        .set_index("growth_run_id")["generated_rq_nm"]
    )
    positions = np.arange(len(selected))
    figure, axis = plt.subplots(
        figsize=(8.2, 3.8), constrained_layout=True
    )
    axis.plot(
        positions,
        selected["true_target"],
        "o-",
        color=REAL_COLOR,
        lw=1.8,
        label="measured sample median Sq",
    )
    axis.plot(
        positions,
        selected["growth_run_id"].map(baseline),
        "o--",
        color=BASELINE_COLOR,
        lw=1.4,
        label="M10 predicted Sq",
    )
    axis.plot(
        positions,
        selected["predicted_target"],
        "o-",
        color=SELECTED_COLOR,
        lw=1.8,
        label=f"{method.split('_')[0]} predicted Sq",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(
        selected["growth_run_id"], rotation=45, ha="right"
    )
    axis.set_ylabel("Sq (nm)")
    axis.set_xlabel("held-out growth group (ordered by measured Sq)")
    axis.set_title(
        "Dynamic-range recovery: the new amplitude head no longer collapses "
        "all predictions near 3 nm"
    )
    axis.legend(frameon=False, ncol=3, loc="upper left")
    _save(figure, figure_dir / "Fig4_rq_dynamic_range_recovery")


def plot_confidence(
    *,
    figure_dir: Path,
    confidence: pd.DataFrame,
) -> None:
    rho = spearmanr(
        confidence["joint_confidence_index"],
        confidence["realized_joint_error_index"],
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(8.2, 3.55), constrained_layout=True
    )
    axes[0].scatter(
        confidence["joint_confidence_index"],
        confidence["realized_joint_error_index"],
        c=confidence["realized_rq_absolute_error_nm"],
        cmap="magma",
        s=46,
        edgecolor="black",
        linewidth=0.4,
    )
    for _, row in confidence.iterrows():
        if (
            row["realized_joint_error_index"]
            >= confidence["realized_joint_error_index"].quantile(0.8)
        ):
            axes[0].annotate(
                str(row["growth_run_id"]),
                (
                    row["joint_confidence_index"],
                    row["realized_joint_error_index"],
                ),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
    axes[0].set_xlabel("joint confidence index (0–100)")
    axes[0].set_ylabel("realized joint error rank")
    axes[0].set_title(
        f"Confidence is error-related: Spearman ρ={rho.statistic:.2f}"
    )

    ordered = confidence.sort_values("joint_confidence_index")
    positions = np.arange(len(ordered))
    axes[1].bar(
        positions - 0.18,
        ordered["predicted_fsmi_absolute_error_nm"],
        width=0.36,
        color=SELECTED_COLOR,
        label="expected FSMI error",
    )
    axes[1].bar(
        positions + 0.18,
        ordered["realized_fsmi_absolute_error_nm"],
        width=0.36,
        color=REAL_COLOR,
        label="realized FSMI error",
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(
        ordered["growth_run_id"], rotation=70, fontsize=7
    )
    axes[1].set_ylabel("FSMI absolute error (nm)")
    axes[1].set_title("Expected and realized FSMI errors")
    axes[1].legend(frameon=False, fontsize=8)
    _save(figure, figure_dir / "Fig5_confidence_calibration")


def plot_ablation(
    *,
    figure_dir: Path,
    standard: pd.DataFrame,
    island: pd.DataFrame,
    surface: pd.DataFrame,
) -> None:
    table = (
        standard[
            [
                "method",
                "median_rq_absolute_error_nm",
                "median_normalized_psd_log_distance",
            ]
        ]
        .merge(
            island[
                [
                    "method",
                    "median_island_feature_mae_z",
                    "median_afm_prior_mahalanobis",
                ]
            ],
            on="method",
        )
        .merge(
            surface[
                [
                    "method",
                    "median_fsmi_absolute_error_nm",
                    "median_generated_island_boundary_contrast",
                ]
            ],
            on="method",
        )
        .set_index("method")
    )
    error_columns = [
        "median_rq_absolute_error_nm",
        "median_normalized_psd_log_distance",
        "median_island_feature_mae_z",
        "median_afm_prior_mahalanobis",
        "median_fsmi_absolute_error_nm",
    ]
    normalized = table[error_columns].div(
        table.loc[M10, error_columns], axis=1
    )
    labels = {
        "median_rq_absolute_error_nm": "Sq error",
        "median_normalized_psd_log_distance": "PSD error",
        "median_island_feature_mae_z": "island error",
        "median_afm_prior_mahalanobis": "AFM-prior distance",
        "median_fsmi_absolute_error_nm": "FSMI error",
    }
    figure, axes = plt.subplots(
        1, 2, figsize=(9.0, 3.7), constrained_layout=True
    )
    x = np.arange(len(error_columns))
    width = 0.19
    colors = [
        BASELINE_COLOR,
        *[
            mpl.colors.to_hex(color)
            for color in mpl.colormaps["tab10"].colors
        ],
    ]
    for index, (method, row) in enumerate(normalized.iterrows()):
        axes[0].bar(
            x + (index - 1.5) * width,
            row,
            width=width,
            color=colors[index],
            label=method.split("_", 1)[0],
        )
    axes[0].axhline(1.0, ls="--", color="black", lw=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [labels[column] for column in error_columns],
        rotation=30,
        ha="right",
    )
    axes[0].set_ylabel("error normalized to M10 (lower is better)")
    axes[0].set_title("Renderer and conditioning ablation")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)

    boundary = table["median_generated_island_boundary_contrast"]
    axes[1].bar(
        np.arange(len(boundary)),
        boundary,
        color=colors,
    )
    axes[1].set_xticks(np.arange(len(boundary)))
    axes[1].set_xticklabels(
        [index.split("_", 1)[0] for index in boundary.index],
        rotation=25,
    )
    axes[1].set_ylabel("generated q70 island boundary contrast")
    axes[1].set_title("SDF contour strength (higher = clearer edge)")
    _save(figure, figure_dir / "Fig6_model_ablation")


def plot_component_correlations(
    *,
    figure_dir: Path,
    group_metrics: pd.DataFrame,
    surface_per_group: pd.DataFrame,
    method: str,
) -> None:
    generated = surface_per_group.loc[
        surface_per_group["method"] == method
    ].set_index("growth_run_id")
    truth = (
        group_metrics.loc[group_metrics["split"] == "train"]
        .set_index("growth_run_id")
        .loc[generated.index]
    )
    pairs = [
        ("sq_nm", "Sq (nm)"),
        ("height_increment_rms_31nm", "31 nm height increment (nm)"),
        ("curvature_relief_31nm", "31 nm curvature relief (nm)"),
        ("bearing_core_equivalent_nm", "bearing equivalent (nm)"),
        ("island_prominence_nm", "island prominence (nm)"),
        (
            "functional_surface_morphology_index_nm",
            "FSMI (nm)",
        ),
    ]
    figure, axes = plt.subplots(
        2, 3, figsize=(8.6, 5.4), constrained_layout=True
    )
    for axis, (column, label) in zip(axes.ravel(), pairs):
        true = truth[column].to_numpy(float)
        predicted = generated[f"generated_{column}"].to_numpy(float)
        axis.scatter(
            true,
            predicted,
            color=SELECTED_COLOR,
            edgecolor="black",
            linewidth=0.35,
            s=34,
        )
        lo = min(true.min(), predicted.min())
        hi = max(true.max(), predicted.max())
        axis.plot([lo, hi], [lo, hi], "--", color="black", lw=0.8)
        axis.set_xlabel(f"measured {label}")
        axis.set_ylabel(f"generated {label}")
        rho = spearmanr(true, predicted).statistic
        axis.set_title(f"{label}\nSpearman ρ={rho:.2f}")
    _save(
        figure, figure_dir / "Fig7_surface_component_correlations"
    )


def plot_failures(
    *,
    figure_dir: Path,
    output: Path,
    phase1: pd.DataFrame,
    rq: pd.DataFrame,
    fsmi: pd.DataFrame,
    confidence: pd.DataFrame,
    method: str,
) -> None:
    ordered = confidence.sort_values(
        ["joint_confidence_index", "realized_joint_error_index"],
        ascending=[True, False],
    )
    groups = list(ordered.head(3)["growth_run_id"].astype(str))
    figure = _comparison_figure(
        groups=groups,
        split="crossfit",
        output=output,
        phase1=phase1,
        rq_predictions=rq,
        fsmi_predictions=fsmi,
        confidence=confidence,
        method=method,
        title=(
            "Lowest-confidence held-one predictions (reported, not hidden)"
        ),
    )
    _save(figure, figure_dir / "Fig8_low_confidence_failures")


def run(config: dict[str, Any]) -> None:
    _style()
    output = repo_path(config["output_root"]) / "development"
    report = repo_path(config["report_root"]) / "development"
    figure_dir = report / "figures"
    tables = _load_tables(config)
    phase1 = tables["phase1"].copy()
    rq = pd.read_csv(
        report / "rq_crossfit_predictions.csv",
        dtype={"growth_run_id": str},
    )
    fsmi = pd.read_csv(
        report / "fsmi_crossfit_predictions.csv",
        dtype={"growth_run_id": str},
    )
    validation_rq = pd.read_csv(
        report / "rq_validation_predictions.csv",
        dtype={"growth_run_id": str},
    )
    validation_fsmi = pd.read_csv(
        report / "fsmi_validation_predictions.csv",
        dtype={"growth_run_id": str},
    )
    confidence = pd.read_csv(
        report / "confidence_crossfit.csv",
        dtype={"growth_run_id": str},
    )
    validation_confidence = pd.read_csv(
        report / "confidence_validation.csv",
        dtype={"growth_run_id": str},
    )
    selected = str(config["selected_method"])
    plot_validation(
        figure_dir=figure_dir,
        output=output,
        phase1=phase1,
        rq=validation_rq,
        fsmi=validation_fsmi,
        confidence=validation_confidence,
        method=selected,
    )
    plot_crossfit_atlas(
        figure_dir=figure_dir,
        output=output,
        phase1=phase1,
        rq=rq,
        fsmi=fsmi,
        confidence=confidence,
        method=selected,
    )
    plot_target_scatter(
        figure_dir=figure_dir,
        rq=rq,
        fsmi=fsmi,
        confidence=confidence,
    )
    cross_standard = pd.read_csv(
        report / "crossfit" / "standard_per_group.csv",
        dtype={"growth_run_id": str},
    )
    plot_dynamic_range(
        figure_dir=figure_dir,
        rq=rq,
        cross_standard=cross_standard,
        method=selected,
    )
    plot_confidence(
        figure_dir=figure_dir, confidence=confidence
    )
    standard_summary = pd.read_csv(
        report / "crossfit" / "standard_summary.csv"
    )
    island_summary = pd.read_csv(
        report / "crossfit" / "island_summary.csv"
    )
    surface_summary = pd.read_csv(
        report / "crossfit" / "functional_surface_summary.csv"
    )
    plot_ablation(
        figure_dir=figure_dir,
        standard=standard_summary,
        island=island_summary,
        surface=surface_summary,
    )
    plot_component_correlations(
        figure_dir=figure_dir,
        group_metrics=pd.read_csv(
            report / "surface_metrics_per_group.csv",
            dtype={"growth_run_id": str},
        ),
        surface_per_group=pd.read_csv(
            report / "crossfit" / "functional_surface_per_group.csv",
            dtype={"growth_run_id": str},
        ),
        method=selected,
    )
    plot_failures(
        figure_dir=figure_dir,
        output=output,
        phase1=phase1,
        rq=rq,
        fsmi=fsmi,
        confidence=confidence,
        method=selected,
    )
    write_json(
        {
            "figure_directory": str(figure_dir),
            "selected_method": selected,
            "png_count": len(list(figure_dir.glob("*.png"))),
            "pdf_count": len(list(figure_dir.glob("*.pdf"))),
            "height_units": "nm",
            "afm_scan_size_nm": float(config["scan_size_nm"]),
            "fixed_order_policy": (
                "Held-one atlas is ordered by measured Sq and the same order "
                "is used within each comparison page."
            ),
        },
        report / "visualization_manifest.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_to_afm_functional_morphology.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
