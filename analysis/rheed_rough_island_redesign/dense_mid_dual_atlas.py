"""Gwyddion-style M17 and paired M22 full-cohort atlas."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis.rheed_rough_island_redesign.gwyddion_atlas import (
    M17_CONFIG_RELATIVE,
    M17_METHOD,
    M17_OUTPUT_RELATIVE,
    _load_generated_maps,
    _sq_comparison_plot,
    _surface_with_height_bar,
)
from analysis.rheed_to_afm_full_cohort_loo.run import load_config
from analysis.rheed_to_afm_functional_morphology.visualization import (
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


def _paths(config: dict[str, Any]) -> tuple[Path, Path]:
    suffix = str(config["full_run_suffix"])
    return (
        repo_path(config["output_root"]) / suffix,
        repo_path(config["report_root"]) / suffix,
    )


def run(
    inclusive: dict[str, Any],
    excluded: dict[str, Any],
    *,
    standalone_root: Path,
) -> None:
    inclusive_output, inclusive_report = _paths(inclusive)
    excluded_output, excluded_report = _paths(excluded)
    m17_output = standalone_root / M17_OUTPUT_RELATIVE
    m17_config = standalone_root / M17_CONFIG_RELATIVE
    if not m17_output.is_dir() or not m17_config.is_file():
        raise FileNotFoundError("standalone M17 reference is incomplete")

    phase1 = pd.read_csv(
        repo_path(inclusive["phase1_manifest"]),
        dtype={"growth_run_id": str},
    )
    inclusive_predictions = pd.read_csv(
        inclusive_report / "rq_crossfit_predictions.csv",
        dtype={"growth_run_id": str},
    ).sort_values("true_target")
    excluded_predictions = pd.read_csv(
        excluded_report / "rq_crossfit_predictions.csv",
        dtype={"growth_run_id": str},
    ).sort_values("true_target")
    if not inclusive_predictions.reset_index(drop=True).equals(
        excluded_predictions.reset_index(drop=True)
    ):
        raise RuntimeError("paired atlas requires identical Sq predictions")

    groups = list(inclusive_predictions["growth_run_id"].astype(str))
    prediction_lookup = inclusive_predictions.set_index("growth_run_id")
    measured = {group: _real_afm(phase1, group) for group in groups}
    m17 = _load_generated_maps(m17_output, method=M17_METHOD, groups=groups)
    inclusive_maps = _load_generated_maps(
        inclusive_output,
        method=str(inclusive["selected_method"]),
        groups=groups,
    )
    excluded_maps = _load_generated_maps(
        excluded_output,
        method=str(excluded["selected_method"]),
        groups=groups,
    )
    figure_dir = (
        repo_path("reports/rheed_m22_dense_mid/20260809_m22_paired_comparison")
        / "figures"
        / "gwyddion_individual_height_atlas_M17_vs_M22_dual"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)

    def draw(subset: list[str], *, stem: str, heading: str) -> None:
        figure, axes = plt.subplots(
            len(subset),
            5,
            figsize=(20.2, 2.85 * len(subset)),
            constrained_layout=True,
            squeeze=False,
            gridspec_kw={"width_ratios": [1.22, 1.0, 1.0, 1.0, 1.0]},
        )
        figure.suptitle(
            heading
            + "\nGwyddion.net orange palette; every AFM has its own height bar",
            fontsize=11.5,
            fontweight="bold",
        )
        for row_index, group in enumerate(subset):
            prediction = prediction_lookup.loc[group]
            label = _real_afm_label(phase1, group)
            axes[row_index, 0].imshow(
                _rheed_keyframe(phase1, group), cmap="gray"
            )
            axes[row_index, 0].set_title(
                f"{group} held-out RHEED\n"
                f"spot isolation "
                f"{float(prediction['rheed_spot_isolation_score']):.3f}",
                fontsize=7.8,
            )
            axes[row_index, 0].set_xticks([])
            axes[row_index, 0].set_yticks([])
            panels = (
                (
                    measured[group],
                    f"Measured AFM\ndisplayed Sq "
                    f"{label['displayed_scan_sq_nm']:.2f} nm",
                ),
                (
                    m17[group][0],
                    f"Standalone M17\npredicted Sq {m17[group][1]:.2f} nm",
                ),
                (
                    inclusive_maps[group][0],
                    "M22 dense-growth (all morphology growths)\n"
                    f"predicted Sq {inclusive_maps[group][1]:.2f} nm",
                ),
                (
                    excluded_maps[group][0],
                    "M22 dense-growth (morphology excludes 6022/6101)\n"
                    f"predicted Sq {excluded_maps[group][1]:.2f} nm",
                ),
            )
            for column, (array, title) in enumerate(panels, start=1):
                _surface_with_height_bar(
                    figure, axes[row_index, column], array, title=title
                )
        _save(figure, figure_dir / stem)

    page_size = 5
    stems: list[str] = []
    for page, start in enumerate(range(0, len(groups), page_size), start=1):
        subset = groups[start : start + page_size]
        stem = f"Atlas_{page:02d}_of_{math.ceil(len(groups) / page_size):02d}"
        draw(
            subset,
            stem=stem,
            heading=f"Strict outer-LOO AFM atlas: M17 vs paired M22 ({page})",
        )
        stems.append(stem)

    focus = list(
        inclusive_predictions.loc[
            inclusive_predictions["true_target"].between(
                3.5, 6.0, inclusive="both"
            ),
            "growth_run_id",
        ].astype(str)
    )
    focus_stem = "Focus_true_Sq_3p5_to_6p0_M17_vs_M22_dual"
    draw(
        focus,
        stem=focus_stem,
        heading=(
            "Intermediate-state focus: all morphology growths vs exclusion of "
            "6022/6101"
        ),
    )
    sq_stem = _sq_comparison_plot(
        inclusive_predictions,
        figure_dir,
        model_short_label="M22",
        model_panel_label="paired M22 (shared Sq head)",
    )
    write_json(
        {
            "growth_count": len(groups),
            "atlas_page_count": len(stems),
            "atlas_stems": stems,
            "focus_groups": focus,
            "focus_stem": focus_stem,
            "sq_comparison_stem": sq_stem,
            "inclusive_method": str(inclusive["selected_method"]),
            "excluded_method": str(excluded["selected_method"]),
            "paired_sq_predictions_identical": True,
            "m17_source_config": str(m17_config),
            "m17_source_config_sha256": sha256_file(m17_config),
            "individual_height_bar_per_afm": True,
            "palette": "Gwyddion.net",
            "m19_displayed": False,
        },
        figure_dir / "atlas_manifest.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inclusive-config", required=True)
    parser.add_argument("--excluded-config", required=True)
    parser.add_argument("--standalone-root", type=Path, required=True)
    args = parser.parse_args()
    run(
        load_config(Path(args.inclusive_config)),
        load_config(Path(args.excluded_config)),
        standalone_root=args.standalone_root.resolve(),
    )


if __name__ == "__main__":
    main()
