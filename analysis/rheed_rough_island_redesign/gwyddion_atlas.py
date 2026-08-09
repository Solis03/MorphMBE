"""Presentation atlas with standalone M17, M20, and Gwyddion-style AFM maps."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from analysis.rheed_to_afm_full_cohort_loo.run import load_config
from analysis.rheed_to_afm_functional_morphology.visualization import (
    _generated,
    _real_afm,
    _real_afm_label,
    _rheed_keyframe,
    _save,
)
from analysis.rheed_video_afm_story.common import (
    repo_path,
    sha256_file,
    write_json,
)

M17_METHOD = "M17b_topology_sparse_peak_terrace"
M17_OUTPUT_RELATIVE = Path(
    "outputs/rheed_m17_end_to_end_generation/"
    "20260804_m17_sparse_topology_line3_full27_v1/full27_loo"
)
M17_CONFIG_RELATIVE = Path(
    "configs/rheed_m17_end_to_end_generation_line3_full27_sparse_v1.json"
)
# Exact system gradient distributed by Gwyddion as data/gradients/Gwyddion.net.
GWYDDION_NET_STOPS = (
    (0.0, (0.0, 0.0, 0.0)),
    (0.344671, (0.658824, 0.156863, 0.0588235)),
    (0.687075, (0.953506, 0.759686, 0.363821)),
    (1.0, (1.0, 1.0, 1.0)),
)
DISPLAY_QUANTILES = (0.005, 0.995)


def gwyddion_net_colormap() -> LinearSegmentedColormap:
    """Return the official black-rust-gold-white Gwyddion.net gradient."""

    return LinearSegmentedColormap.from_list(
        "Gwyddion.net",
        list(GWYDDION_NET_STOPS),
        N=256,
    )


def individual_height_limits(array: np.ndarray) -> tuple[float, float]:
    """Robust per-map height limits for presentation without cross-map reuse."""

    finite = np.asarray(array, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("AFM height map has no finite values")
    low, high = np.quantile(finite, DISPLAY_QUANTILES)
    if high <= low:
        center = float(low)
        radius = max(abs(center) * 0.01, 1e-6)
        return center - radius, center + radius
    return float(low), float(high)


def _height_ticks(low: float, high: float) -> list[float]:
    middle = 0.0 if low < 0.0 < high else 0.5 * (low + high)
    return [low, middle, high]


def _outlined_scale_bar(axis: plt.Axes, *, pixels: int) -> None:
    width = 0.25 * pixels
    y = 0.91 * pixels
    x0 = 0.67 * pixels
    axis.plot(
        [x0, x0 + width],
        [y, y],
        color="black",
        lw=5.2,
        solid_capstyle="butt",
    )
    axis.plot(
        [x0, x0 + width],
        [y, y],
        color="white",
        lw=3.0,
        solid_capstyle="butt",
    )
    axis.text(
        x0 + width / 2,
        y - 0.04 * pixels,
        "250 nm",
        color="white",
        ha="center",
        va="bottom",
        fontsize=6.8,
        path_effects=[
            path_effects.Stroke(linewidth=2.2, foreground="black"),
            path_effects.Normal(),
        ],
    )


def _surface_with_height_bar(
    figure: plt.Figure,
    axis: plt.Axes,
    array: np.ndarray,
    *,
    title: str,
) -> tuple[float, float]:
    low, high = individual_height_limits(array)
    image = axis.imshow(
        array,
        cmap=gwyddion_net_colormap(),
        vmin=low,
        vmax=high,
        interpolation="nearest",
    )
    axis.set_title(title, fontsize=7.6, linespacing=1.12)
    axis.set_xticks([])
    axis.set_yticks([])
    _outlined_scale_bar(axis, pixels=array.shape[1])
    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.047,
        pad=0.025,
        aspect=13,
    )
    colorbar.set_ticks(_height_ticks(low, high))
    colorbar.ax.tick_params(labelsize=6.1, length=2)
    colorbar.set_label("height (nm)", fontsize=6.4, labelpad=2.5)
    return low, high


def _load_generated_maps(
    output: Path,
    *,
    method: str,
    groups: list[str],
) -> dict[str, tuple[np.ndarray, float]]:
    return {
        group: _generated(
            output,
            split="crossfit",
            method=method,
            group=group,
        )
        for group in groups
    }


def _sq_comparison_plot(
    predictions: pd.DataFrame,
    output_dir: Path,
) -> str:
    ordered = predictions.sort_values("true_target").reset_index(drop=True)
    ordered.to_csv(
        output_dir / "M20_Sq_measured_vs_predicted_ordered.csv",
        index=False,
    )
    truth = ordered["true_target"].to_numpy(float)
    predicted = ordered["predicted_target"].to_numpy(float)
    lower = ordered["interval_lower"].to_numpy(float)
    upper = ordered["interval_upper"].to_numpy(float)
    residual = predicted - truth
    x = np.arange(len(ordered))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    correlation = float(np.corrcoef(truth, predicted)[0, 1])

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(15.5, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.06},
        constrained_layout=True,
    )
    axes[0].fill_between(
        x,
        lower,
        upper,
        color="#f3c25d",
        alpha=0.22,
        label="M20 prediction interval",
        zorder=0,
    )
    axes[0].vlines(
        x,
        np.minimum(truth, predicted),
        np.maximum(truth, predicted),
        color="#9b9b9b",
        alpha=0.48,
        lw=1.0,
        zorder=1,
    )
    axes[0].plot(
        x,
        truth,
        color="#222222",
        lw=2.2,
        marker="o",
        ms=5.2,
        markerfacecolor="#f3c25d",
        markeredgecolor="#222222",
        label="measured Sq",
        zorder=3,
    )
    axes[0].plot(
        x,
        predicted,
        color="#a8280f",
        lw=2.0,
        marker="D",
        ms=4.7,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="M20 predicted Sq",
        zorder=4,
    )
    axes[0].set_ylabel("Sq (nm)")
    axes[0].grid(axis="y", color="#d9d9d9", lw=0.7)
    axes[0].legend(loc="upper left", frameon=False, ncol=3)
    axes[0].set_title(
        "M20 strict outer-LOO Sq: measured versus predicted\n"
        f"27 growths | MAE {mae:.3f} nm | RMSE {rmse:.3f} nm | Pearson r {correlation:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    for group in ["6062", "6099"]:
        index = int(
            ordered.index[
                ordered["growth_run_id"].astype(str) == group
            ][0]
        )
        axes[0].annotate(
            f"{group}\n{truth[index]:.2f} → {predicted[index]:.2f} nm",
            xy=(index, predicted[index]),
            xytext=(0, 19 if residual[index] >= 0 else -34),
            textcoords="offset points",
            ha="center",
            fontsize=8.3,
            color="#721b0a",
            arrowprops={"arrowstyle": "-", "color": "#a8280f", "lw": 0.8},
        )

    axes[1].axhline(0.0, color="#222222", lw=1.0)
    axes[1].plot(
        x,
        residual,
        color="#a8280f",
        lw=1.4,
        marker="o",
        ms=4.0,
    )
    axes[1].fill_between(
        x,
        0.0,
        residual,
        where=residual >= 0.0,
        color="#f3c25d",
        alpha=0.38,
        interpolate=True,
    )
    axes[1].fill_between(
        x,
        0.0,
        residual,
        where=residual < 0.0,
        color="#a8280f",
        alpha=0.22,
        interpolate=True,
    )
    axes[1].set_ylabel("error\n(pred. − real, nm)")
    axes[1].set_xlabel("held-out growth (ordered by measured Sq)")
    axes[1].grid(axis="y", color="#d9d9d9", lw=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        ordered["growth_run_id"].astype(str),
        rotation=55,
        ha="right",
        fontsize=7.8,
    )
    stem = "M20_Sq_measured_vs_predicted_ordered"
    _save(figure, output_dir / stem)
    return stem


def run(config: dict, *, standalone_root: Path) -> None:
    suffix = str(config["full_run_suffix"])
    m20_output = repo_path(config["output_root"]) / suffix
    report = repo_path(config["report_root"]) / suffix
    m20_method = str(config["selected_method"])
    m17_output = standalone_root / M17_OUTPUT_RELATIVE
    m17_config = standalone_root / M17_CONFIG_RELATIVE
    if not m17_output.is_dir():
        raise FileNotFoundError(f"Standalone M17 output is missing: {m17_output}")
    if not m17_config.is_file():
        raise FileNotFoundError(f"Standalone M17 config is missing: {m17_config}")

    phase1 = pd.read_csv(
        repo_path(config["phase1_manifest"]),
        dtype={"growth_run_id": str},
    )
    predictions = pd.read_csv(
        report / "rq_crossfit_predictions.csv",
        dtype={"growth_run_id": str},
    ).sort_values("true_target")
    prediction_lookup = predictions.set_index("growth_run_id")
    groups = list(predictions["growth_run_id"].astype(str))
    measured = {group: _real_afm(phase1, group) for group in groups}
    m17 = _load_generated_maps(m17_output, method=M17_METHOD, groups=groups)
    m20 = _load_generated_maps(m20_output, method=m20_method, groups=groups)

    figure_dir = (
        report
        / "figures"
        / "gwyddion_individual_height_atlas_M17_standalone_vs_M20"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    range_rows: list[dict[str, float | str]] = []
    stems: list[str] = []
    page_size = 5
    page_count = math.ceil(len(groups) / page_size)
    for page, start in enumerate(range(0, len(groups), page_size), start=1):
        subset = groups[start : start + page_size]
        figure, axes = plt.subplots(
            len(subset),
            4,
            figsize=(16.3, 2.8 * len(subset)),
            constrained_layout=True,
            squeeze=False,
            gridspec_kw={"width_ratios": [1.22, 1.0, 1.0, 1.0]},
        )
        figure.suptitle(
            "Strict outer-LOO AFM atlas: standalone M17 vs M20 "
            f"({page}/{page_count})\n"
            "Gwyddion.net orange palette; every AFM has its own height bar",
            fontsize=11.2,
            fontweight="bold",
        )
        for row_index, group in enumerate(subset):
            row = prediction_lookup.loc[group]
            rheed = _rheed_keyframe(phase1, group)
            label = _real_afm_label(phase1, group)
            axes[row_index, 0].imshow(rheed, cmap="gray")
            axes[row_index, 0].set_title(
                f"{group} held-out RHEED\n"
                f"spot isolation {float(row['rheed_spot_isolation_score']):.3f}",
                fontsize=7.6,
            )
            axes[row_index, 0].set_xticks([])
            axes[row_index, 0].set_yticks([])
            panels = (
                (
                    "measured",
                    measured[group],
                    (
                        "Measured AFM\n"
                        f"displayed Sq {label['displayed_scan_sq_nm']:.2f} nm"
                    ),
                ),
                (
                    "M17_standalone",
                    m17[group][0],
                    (
                        "Standalone M17\n"
                        f"predicted Sq {m17[group][1]:.2f} nm"
                    ),
                ),
                (
                    "M20",
                    m20[group][0],
                    (
                        "M20 connectivity-island\n"
                        f"predicted Sq {m20[group][1]:.2f} nm"
                    ),
                ),
            )
            for column, (source, array, title) in enumerate(panels, start=1):
                low, high = _surface_with_height_bar(
                    figure,
                    axes[row_index, column],
                    array,
                    title=title,
                )
                range_rows.append(
                    {
                        "growth_run_id": group,
                        "source": source,
                        "display_vmin_nm": low,
                        "display_vmax_nm": high,
                    }
                )
        stem = f"Atlas{page}_Gwyddion_individual_height_M17_vs_M20"
        _save(figure, figure_dir / stem)
        stems.append(stem)

    focus = ["6062", "6099"]
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(16.3, 6.15),
        constrained_layout=True,
        squeeze=False,
        gridspec_kw={"width_ratios": [1.22, 1.0, 1.0, 1.0]},
    )
    figure.suptitle(
        "6062 vs 6099: standalone M17 and M20 with individual AFM height bars\n"
        "Gwyddion.net orange palette",
        fontsize=11.4,
        fontweight="bold",
    )
    for row_index, group in enumerate(focus):
        row = prediction_lookup.loc[group]
        label = _real_afm_label(phase1, group)
        axes[row_index, 0].imshow(_rheed_keyframe(phase1, group), cmap="gray")
        axes[row_index, 0].set_title(
            f"{group} RHEED\n"
            f"spot isolation {float(row['rheed_spot_isolation_score']):.3f}",
            fontsize=8.2,
        )
        axes[row_index, 0].set_xticks([])
        axes[row_index, 0].set_yticks([])
        panels = (
            (
                measured[group],
                f"Measured AFM\nSq {label['displayed_scan_sq_nm']:.2f} nm",
            ),
            (
                m17[group][0],
                f"Standalone M17\nSq {m17[group][1]:.2f} nm",
            ),
            (
                m20[group][0],
                f"M20 connectivity-island\nSq {m20[group][1]:.2f} nm",
            ),
        )
        for column, (array, title) in enumerate(panels, start=1):
            _surface_with_height_bar(
                figure,
                axes[row_index, column],
                array,
                title=title,
            )
    focus_stem = "Focus_6062_6099_Gwyddion_individual_height_M17_vs_M20"
    _save(figure, figure_dir / focus_stem)
    pd.DataFrame(range_rows).to_csv(
        figure_dir / "individual_height_ranges.csv",
        index=False,
    )
    sq_stem = _sq_comparison_plot(predictions, figure_dir)

    m17_files = {
        group: (
            m17_output
            / "crossfit"
            / "generated_maps"
            / M17_METHOD
            / f"{group}.npz"
        )
        for group in groups
    }
    write_json(
        {
            "growth_count": len(groups),
            "page_count": page_count,
            "m17_source": {
                "standalone_root": str(standalone_root),
                "config": str(m17_config),
                "config_sha256": sha256_file(m17_config),
                "output": str(m17_output),
                "method": M17_METHOD,
                "generated_map_sha256": {
                    group: sha256_file(path)
                    for group, path in m17_files.items()
                },
            },
            "m20_method": m20_method,
            "m19_displayed": False,
            "palette": {
                "name": "Gwyddion.net",
                "stops": GWYDDION_NET_STOPS,
                "source": (
                    "https://sourceforge.net/p/gwyddion/code/HEAD/tree/"
                    "trunk/gwyddion/data/gradients/Gwyddion.net"
                ),
            },
            "individual_height_bar_per_afm": True,
            "display_height_quantiles": DISPLAY_QUANTILES,
            "atlas_stems": stems,
            "focus_stem": focus_stem,
            "sq_comparison_stem": sq_stem,
        },
        figure_dir / "atlas_manifest.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--standalone-root", type=Path, required=True)
    args = parser.parse_args()
    run(
        load_config(Path(args.config)),
        standalone_root=args.standalone_root.resolve(),
    )


if __name__ == "__main__":
    main()
