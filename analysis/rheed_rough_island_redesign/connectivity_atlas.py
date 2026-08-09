"""Cross-sample physical-scale atlas for M19 versus M20."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.rheed_to_afm_full_cohort_loo.run import load_config
from analysis.rheed_to_afm_functional_morphology.visualization import (
    _generated,
    _real_afm,
    _real_afm_label,
    _rheed_keyframe,
    _save,
    _scale_bar,
)
from analysis.rheed_video_afm_story.common import repo_path, write_json


def _surface(
    axis: plt.Axes,
    array: np.ndarray,
    *,
    title: str,
    limit_nm: float,
) -> plt.AxesImage:
    image = axis.imshow(
        array,
        cmap="viridis",
        vmin=-limit_nm,
        vmax=limit_nm,
        interpolation="nearest",
    )
    axis.set_title(title, fontsize=7.7, linespacing=1.1)
    axis.set_xticks([])
    axis.set_yticks([])
    _scale_bar(axis, pixels=array.shape[1])
    return image


def _global_limit(arrays: list[np.ndarray]) -> float:
    absolute = np.concatenate(
        [np.abs(np.asarray(array, dtype=np.float32)).ravel() for array in arrays]
    )
    return float(np.quantile(absolute, 0.985))


def run(
    config: dict,
    m19_config: dict,
    *,
    selected_method: str,
) -> None:
    suffix = str(config["full_run_suffix"])
    output = repo_path(config["output_root"]) / suffix
    report = repo_path(config["report_root"]) / suffix
    m19_output = (
        repo_path(m19_config["output_root"])
        / str(m19_config["full_run_suffix"])
    )
    m19_method = str(m19_config["selected_method"])
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
    old = {
        group: _generated(
            m19_output,
            split="crossfit",
            method=m19_method,
            group=group,
        )
        for group in groups
    }
    new = {
        group: _generated(
            output,
            split="crossfit",
            method=selected_method,
            group=group,
        )
        for group in groups
    }
    limit = _global_limit(
        [
            *(measured[group] for group in groups),
            *(old[group][0] for group in groups),
            *(new[group][0] for group in groups),
        ]
    )
    figure_dir = (
        report
        / "figures"
        / f"global_physical_atlas_{selected_method}"
    )
    stems: list[str] = []
    page_size = 5
    page_count = math.ceil(len(groups) / page_size)
    for page, start in enumerate(range(0, len(groups), page_size), start=1):
        subset = groups[start : start + page_size]
        figure, axes = plt.subplots(
            len(subset),
            4,
            figsize=(12.5, 2.75 * len(subset)),
            constrained_layout=True,
            squeeze=False,
        )
        figure.suptitle(
            "Strict outer-LOO atlas with one shared physical height scale: "
            f"M19 vs M20 ({page}/{page_count})",
            fontsize=11.5,
            fontweight="bold",
        )
        last_image = None
        for row_index, group in enumerate(subset):
            row = prediction_lookup.loc[group]
            rheed = _rheed_keyframe(phase1, group)
            label = _real_afm_label(phase1, group)
            old_array, old_sq = old[group]
            new_array, new_sq = new[group]
            axes[row_index, 0].imshow(rheed, cmap="gray")
            axes[row_index, 0].set_title(
                f"{group} held-out RHEED\n"
                f"isolation {float(row['rheed_spot_isolation_score']):.3f}",
                fontsize=7.7,
            )
            axes[row_index, 0].set_xticks([])
            axes[row_index, 0].set_yticks([])
            _surface(
                axes[row_index, 1],
                measured[group],
                title=(
                    "measured AFM\n"
                    f"Sq {label['displayed_scan_sq_nm']:.2f} nm"
                ),
                limit_nm=limit,
            )
            _surface(
                axes[row_index, 2],
                old_array,
                title=f"M19 old\nSq {old_sq:.2f} nm",
                limit_nm=limit,
            )
            last_image = _surface(
                axes[row_index, 3],
                new_array,
                title=f"M20 connectivity-coupled\nSq {new_sq:.2f} nm",
                limit_nm=limit,
            )
        assert last_image is not None
        figure.colorbar(
            last_image,
            ax=axes[:, 1:].ravel().tolist(),
            shrink=0.72,
            label="height (nm), shared across all 27 growths",
        )
        stem = f"Atlas{page}_global_nm_M19_vs_M20"
        _save(figure, figure_dir / stem)
        stems.append(stem)

    focus = ["6062", "6099"]
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(12.5, 6.1),
        constrained_layout=True,
        squeeze=False,
    )
    figure.suptitle(
        "6062 vs 6099: shared physical height scale",
        fontsize=12,
        fontweight="bold",
    )
    last_image = None
    for row_index, group in enumerate(focus):
        row = prediction_lookup.loc[group]
        axes[row_index, 0].imshow(_rheed_keyframe(phase1, group), cmap="gray")
        axes[row_index, 0].set_title(
            f"{group} RHEED\n"
            f"spot isolation {float(row['rheed_spot_isolation_score']):.3f}",
            fontsize=8.5,
        )
        axes[row_index, 0].set_xticks([])
        axes[row_index, 0].set_yticks([])
        label = _real_afm_label(phase1, group)
        _surface(
            axes[row_index, 1],
            measured[group],
            title=f"measured AFM\nSq {label['displayed_scan_sq_nm']:.2f} nm",
            limit_nm=limit,
        )
        _surface(
            axes[row_index, 2],
            old[group][0],
            title=f"M19 old\nSq {old[group][1]:.2f} nm",
            limit_nm=limit,
        )
        last_image = _surface(
            axes[row_index, 3],
            new[group][0],
            title=f"M20 connectivity-coupled\nSq {new[group][1]:.2f} nm",
            limit_nm=limit,
        )
    assert last_image is not None
    figure.colorbar(
        last_image,
        ax=axes[:, 1:].ravel().tolist(),
        shrink=0.78,
        label="height (nm), identical scale for both samples",
    )
    focus_stem = "Focus_6062_6099_global_nm_M19_vs_M20"
    _save(figure, figure_dir / focus_stem)
    write_json(
        {
            "selected_method": selected_method,
            "m19_method": m19_method,
            "growth_count": len(groups),
            "page_count": page_count,
            "global_symmetric_height_limit_nm": limit,
            "common_color_scale_across_all_growths": True,
            "stems": stems,
            "focus_stem": focus_stem,
        },
        figure_dir / "atlas_manifest.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--m19-config", required=True)
    parser.add_argument("--method")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    run(
        config,
        load_config(Path(args.m19_config)),
        selected_method=(
            str(args.method) if args.method else str(config["selected_method"])
        ),
    )


if __name__ == "__main__":
    main()
