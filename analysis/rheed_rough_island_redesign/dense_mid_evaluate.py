"""Paired M22 evaluation of intermediate AFM tone and low-region topology."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage, stats

from analysis.rheed_rough_island_redesign.gwyddion_atlas import (
    DISPLAY_QUANTILES,
)
from analysis.rheed_to_afm_full_cohort_loo.run import load_config
from analysis.rheed_to_afm_functional_morphology.visualization import (
    _generated,
    _real_afm,
)
from analysis.rheed_video_afm_story.common import repo_path, write_json

M21_METHOD = "M21c_strong_growth_layer"
M21_CONFIG = Path("configs/rheed_m21_layered_mid_islands_full27_v4.json")
M20_METHOD = "M20b_connectivity_coupled_islands"
M20_CONFIG = Path("configs/rheed_m20_spot_connectivity_islands_full27_v2.json")
DARK_DISPLAY_LEVEL = 0.18


def _display_metrics(array: np.ndarray) -> dict[str, float]:
    values = np.asarray(array, dtype=float)
    low, high = np.quantile(values, DISPLAY_QUANTILES)
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    dark = normalized <= DARK_DISPLAY_LEVEL
    labels, _ = ndimage.label(dark, structure=np.ones((3, 3)))
    component_areas = np.bincount(labels.ravel())[1:]
    largest = (
        float(component_areas.max() / values.size)
        if component_areas.size
        else 0.0
    )
    return {
        "dark_fraction": float(np.mean(dark)),
        "largest_dark_component_fraction": largest,
        "display_median": float(np.median(normalized)),
        "display_mean": float(np.mean(normalized)),
        "height_skewness": float(stats.skew(values.ravel())),
    }


def _prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}__{key}": value for key, value in metrics.items()}


def _paths(config: dict[str, Any]) -> tuple[Path, Path]:
    suffix = str(config["full_run_suffix"])
    return (
        repo_path(config["output_root"]) / suffix,
        repo_path(config["report_root"]) / suffix,
    )


def _summary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    subsets = {
        "all27": frame,
        "measured_Sq_3p5_to_6p0_nm": frame.loc[
            frame["true_sq_nm"].between(3.5, 6.0, inclusive="both")
        ],
        "predicted_intermediate_3p0_to_6p0_nm": frame.loc[
            frame["predicted_sq_nm"].between(3.0, 6.0, inclusive="both")
        ],
    }
    rows: list[dict[str, float | int | str]] = []
    for subset, subset_frame in subsets.items():
        for source in ("real", "m21", "inclusive", "exclude_6022_6101"):
            row: dict[str, float | int | str] = {
                "subset": subset,
                "source": source,
                "growth_count": int(len(subset_frame)),
            }
            for metric in (
                "dark_fraction",
                "largest_dark_component_fraction",
                "display_median",
                "display_mean",
                "height_skewness",
            ):
                row[f"mean_{metric}"] = float(
                    subset_frame[f"{source}__{metric}"].mean()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def run(inclusive: dict[str, Any], excluded: dict[str, Any]) -> None:
    inclusive_output, inclusive_report = _paths(inclusive)
    excluded_output, excluded_report = _paths(excluded)
    comparison_report = repo_path(
        "reports/rheed_m22_dense_mid/20260809_m22_paired_comparison"
    )
    comparison_report.mkdir(parents=True, exist_ok=True)

    inclusive_predictions = pd.read_csv(
        inclusive_report / "rq_crossfit_predictions.csv",
        dtype={"growth_run_id": str},
    ).sort_values("growth_run_id").reset_index(drop=True)
    excluded_predictions = pd.read_csv(
        excluded_report / "rq_crossfit_predictions.csv",
        dtype={"growth_run_id": str},
    ).sort_values("growth_run_id").reset_index(drop=True)
    prediction_columns = [
        "growth_run_id",
        "true_target",
        "predicted_target",
        "rheed_spot_isolation_score",
        "outer_target_used_for_training",
    ]
    if not inclusive_predictions[prediction_columns].equals(
        excluded_predictions[prediction_columns]
    ):
        raise RuntimeError("paired M22 Sq predictions are not identical")

    phase1 = pd.read_csv(
        repo_path(inclusive["phase1_manifest"]),
        dtype={"growth_run_id": str},
    )
    m21_config = load_config(M21_CONFIG)
    m21_output, m21_report = _paths(m21_config)
    groups = list(inclusive_predictions["growth_run_id"].astype(str))
    rows: list[dict[str, float | str | bool]] = []
    lookup = inclusive_predictions.set_index("growth_run_id")
    for group in groups:
        real = _real_afm(phase1, group)
        m21, _ = _generated(
            m21_output, split="crossfit", method=M21_METHOD, group=group
        )
        inclusive_map, _ = _generated(
            inclusive_output,
            split="crossfit",
            method=str(inclusive["selected_method"]),
            group=group,
        )
        excluded_map, _ = _generated(
            excluded_output,
            split="crossfit",
            method=str(excluded["selected_method"]),
            group=group,
        )
        prediction = lookup.loc[group]
        rows.append(
            {
                "growth_run_id": group,
                "true_sq_nm": float(prediction["true_target"]),
                "predicted_sq_nm": float(prediction["predicted_target"]),
                "rheed_spot_isolation_score": float(
                    prediction["rheed_spot_isolation_score"]
                ),
                **_prefixed("real", _display_metrics(real)),
                **_prefixed("m21", _display_metrics(m21)),
                **_prefixed("inclusive", _display_metrics(inclusive_map)),
                **_prefixed(
                    "exclude_6022_6101", _display_metrics(excluded_map)
                ),
            }
        )
    comparison = pd.DataFrame(rows).sort_values("true_sq_nm")
    comparison.to_csv(
        comparison_report / "display_tone_per_group.csv", index=False
    )
    summary = _summary_rows(comparison)
    summary.to_csv(
        comparison_report / "display_tone_summary.csv", index=False
    )

    island_frames = []
    for label, report, method in (
        ("m21", m21_report, M21_METHOD),
        ("inclusive", inclusive_report, str(inclusive["selected_method"])),
        (
            "exclude_6022_6101",
            excluded_report,
            str(excluded["selected_method"]),
        ),
    ):
        island = pd.read_csv(
            report / "crossfit" / "island_per_group.csv",
            dtype={"growth_run_id": str},
        )
        island_frames.append(
            island.loc[island["method"] == method].assign(source=label)
        )
    island_comparison = pd.concat(island_frames, ignore_index=True)
    island_comparison.to_csv(
        comparison_report / "island_metrics_selected_methods.csv",
        index=False,
    )

    inclusive_folds = pd.read_csv(inclusive_report / "fold_integrity_audit.csv")
    excluded_folds = pd.read_csv(excluded_report / "fold_integrity_audit.csv")
    forbidden = {"6022", "6101"}
    exclusion_clean = True
    for value in excluded_folds["morphology_fit_growth_run_ids"]:
        # CSV serialization may use a Python-list representation; the direct
        # substring check is deliberately conservative for either encoding.
        if any(group in str(value) for group in forbidden):
            exclusion_clean = False
            break

    m20_config = load_config(M20_CONFIG)
    m20_output, _ = _paths(m20_config)
    protected_rows = []
    for prediction in inclusive_predictions.itertuples(index=False):
        group = str(prediction.growth_run_id)
        if not (
            float(prediction.predicted_target) < 2.2
            or float(prediction.predicted_target) >= 7.6
        ):
            continue
        baseline, _ = _generated(
            m20_output, split="crossfit", method=M20_METHOD, group=group
        )
        selected, _ = _generated(
            inclusive_output,
            split="crossfit",
            method=str(inclusive["selected_method"]),
            group=group,
        )
        protected_rows.append(
            {
                "growth_run_id": group,
                "predicted_sq_nm": float(prediction.predicted_target),
                "array_equal_to_accepted_m20": bool(
                    np.array_equal(baseline, selected)
                ),
                "maximum_absolute_height_difference_nm": float(
                    np.max(np.abs(baseline - selected))
                ),
            }
        )
    protected = pd.DataFrame(protected_rows)
    protected.to_csv(
        comparison_report / "inclusive_protected_regime_audit.csv",
        index=False,
    )

    write_json(
        {
            "growth_count": int(len(comparison)),
            "paired_sq_predictions_identical": True,
            "sq_predictions_use_6022_6101_when_allowed_by_outer_fold": True,
            "inclusive_morphology_fit_growth_count_range": sorted(
                map(int, inclusive_folds["morphology_fit_growth_count"].unique())
            ),
            "excluded_morphology_fit_growth_count_range": sorted(
                map(int, excluded_folds["morphology_fit_growth_count"].unique())
            ),
            "excluded_morphology_training_contains_neither_6022_nor_6101": bool(
                exclusion_clean
            ),
            "all_outer_target_leakage_flags_false": bool(
                inclusive_predictions["outer_target_used_for_training"]
                .eq(False)
                .all()
            ),
            "all_inclusive_held_growth_overlap_flags_false": bool(
                inclusive_folds["held_overlap_with_fit"].eq(False).all()
            ),
            "all_excluded_held_growth_overlap_flags_false": bool(
                excluded_folds["held_overlap_with_fit"].eq(False).all()
            ),
            "all_inclusive_protected_arrays_equal_to_accepted_m20": bool(
                protected["array_equal_to_accepted_m20"].all()
            ),
            "dark_display_level": DARK_DISPLAY_LEVEL,
            "display_quantiles": DISPLAY_QUANTILES,
            "query_measured_afm_used_at_inference": False,
        },
        comparison_report / "paired_evaluation_manifest.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inclusive-config", required=True)
    parser.add_argument("--excluded-config", required=True)
    args = parser.parse_args()
    run(
        load_config(Path(args.inclusive_config)),
        load_config(Path(args.excluded_config)),
    )


if __name__ == "__main__":
    main()
