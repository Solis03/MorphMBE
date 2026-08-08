from __future__ import annotations

import argparse
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
    vmin: float,
    vmax: float,
) -> None:
    axis.imshow(
        array,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axis.set_title(title, fontsize=7.8, linespacing=1.12)
    axis.set_xticks([])
    axis.set_yticks([])
    _scale_bar(axis, pixels=array.shape[1])


def run(config: dict, *, baseline_method: str) -> None:
    suffix = str(config["full_run_suffix"])
    output = repo_path(config["output_root"]) / suffix
    report = repo_path(config["report_root"]) / suffix
    selected = str(config["selected_method"])
    phase1 = pd.read_csv(
        repo_path(config["phase1_manifest"]),
        dtype={"growth_run_id": str},
    )
    predictions = pd.read_csv(
        report / "rq_crossfit_predictions.csv",
        dtype={"growth_run_id": str},
    ).sort_values("true_target")
    groups = list(predictions["growth_run_id"].astype(str))
    prediction_lookup = predictions.set_index("growth_run_id")
    figure_dir = report / "figures" / "atlas_compare_m17_m19"
    stems: list[str] = []
    page_size = 5
    page_count = (len(groups) + page_size - 1) // page_size
    for page, start in enumerate(range(0, len(groups), page_size), start=1):
        subset = groups[start : start + page_size]
        figure, axes = plt.subplots(
            len(subset),
            4,
            figsize=(12.4, 2.78 * len(subset)),
            constrained_layout=True,
            squeeze=False,
        )
        figure.suptitle(
            "Strict outer-LOO full-cohort atlas: M17 baseline vs M19 "
            f"separated rough islands ({page}/{page_count})",
            fontsize=11.5,
            fontweight="bold",
        )
        for row_index, group in enumerate(subset):
            row = prediction_lookup.loc[group]
            rheed = _rheed_keyframe(phase1, group)
            measured = _real_afm(phase1, group)
            measured_label = _real_afm_label(phase1, group)
            baseline, baseline_sq = _generated(
                output,
                split="crossfit",
                method=baseline_method,
                group=group,
            )
            generated, predicted_sq = _generated(
                output,
                split="crossfit",
                method=selected,
                group=group,
            )
            combined = np.concatenate(
                [measured.ravel(), baseline.ravel(), generated.ravel()]
            )
            vmin, vmax = np.quantile(combined, [0.01, 0.99])
            axes[row_index, 0].imshow(rheed, cmap="gray")
            axes[row_index, 0].set_title(
                f"{group} held-out RHEED\n"
                f"measured Sq {float(row['true_target']):.2f} nm",
                fontsize=7.8,
            )
            axes[row_index, 0].set_xticks([])
            axes[row_index, 0].set_yticks([])
            _surface(
                axes[row_index, 1],
                measured,
                title=(
                    "measured AFM\n"
                    f"displayed Sq {measured_label['displayed_scan_sq_nm']:.2f} nm"
                ),
                vmin=vmin,
                vmax=vmax,
            )
            _surface(
                axes[row_index, 2],
                baseline,
                title=f"M17 morphology\npredicted Sq {baseline_sq:.2f} nm",
                vmin=vmin,
                vmax=vmax,
            )
            _surface(
                axes[row_index, 3],
                generated,
                title=f"M19 separated islands\npredicted Sq {predicted_sq:.2f} nm",
                vmin=vmin,
                vmax=vmax,
            )
        stem = f"Atlas{page}_full27_M17_vs_M19"
        _save(figure, figure_dir / stem)
        stems.append(stem)
    write_json(
        {
            "selected_method": selected,
            "baseline_method": baseline_method,
            "growth_count": len(groups),
            "page_count": page_count,
            "ordering": "ascending measured Sq",
            "common_color_scale_within_each_growth": True,
            "stems": stems,
        },
        figure_dir / "atlas_manifest.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--baseline-method",
        default="M17b_topology_sparse_peak_terrace",
    )
    args = parser.parse_args()
    run(load_config(Path(args.config)), baseline_method=args.baseline_method)


if __name__ == "__main__":
    main()
