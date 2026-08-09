from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.rheed_to_afm_full_cohort_loo.run import load_config
from analysis.rheed_to_afm_functional_morphology.visualization import _generated
from analysis.rheed_video_afm_story.common import repo_path, write_json


def _mean_metrics(
    frame: pd.DataFrame, prefix: str
) -> dict[str, float | int | str]:
    return {
        "subset": prefix,
        "growth_count": int(len(frame)),
        "baseline_island_feature_mae_z": float(
            frame["baseline_island_feature_mae_z"].mean()
        ),
        "selected_island_feature_mae_z": float(
            frame["selected_island_feature_mae_z"].mean()
        ),
        "relative_island_mae_improvement": float(
            1.0
            - frame["selected_island_feature_mae_z"].mean()
            / frame["baseline_island_feature_mae_z"].mean()
        ),
        "baseline_flat_fraction_absolute_error": float(
            frame["baseline_flat_fraction_absolute_error"].mean()
        ),
        "selected_flat_fraction_absolute_error": float(
            frame["selected_flat_fraction_absolute_error"].mean()
        ),
    }


def run(config: dict) -> None:
    suffix = str(config["full_run_suffix"])
    output = repo_path(config["output_root"]) / suffix
    report = repo_path(config["report_root"]) / suffix
    selected = str(config["selected_method"])
    baseline = "M20b_connectivity_coupled_islands"

    island = pd.read_csv(
        report / "crossfit" / "island_per_group.csv",
        dtype={"growth_run_id": str},
    )
    predictions = pd.read_csv(
        report / "rq_crossfit_predictions.csv",
        dtype={"growth_run_id": str},
    )[
        [
            "growth_run_id",
            "true_target",
            "predicted_target",
            "rheed_spot_isolation_score",
            "outer_target_used_for_training",
        ]
    ]
    selected_rows = island.loc[island["method"] == selected].copy()
    baseline_rows = island.loc[island["method"] == baseline].copy()
    metric_columns = [
        "growth_run_id",
        "island_feature_mae_z",
        "generated__flat_fraction",
        "true__flat_fraction",
    ]
    comparison = baseline_rows[metric_columns].merge(
        selected_rows[metric_columns],
        on="growth_run_id",
        suffixes=("_baseline", "_selected"),
        validate="one_to_one",
    )
    comparison = comparison.merge(
        predictions,
        on="growth_run_id",
        validate="one_to_one",
    )
    comparison = comparison.rename(
        columns={
            "island_feature_mae_z_baseline": (
                "baseline_island_feature_mae_z"
            ),
            "island_feature_mae_z_selected": (
                "selected_island_feature_mae_z"
            ),
        }
    )
    comparison["baseline_flat_fraction_absolute_error"] = np.abs(
        comparison["generated__flat_fraction_baseline"]
        - comparison["true__flat_fraction_baseline"]
    )
    comparison["selected_flat_fraction_absolute_error"] = np.abs(
        comparison["generated__flat_fraction_selected"]
        - comparison["true__flat_fraction_selected"]
    )
    comparison["predicted_growth_regime"] = pd.cut(
        comparison["predicted_target"],
        bins=[-np.inf, 2.2, 3.6, 7.6, np.inf],
        labels=[
            "smooth",
            "transition",
            "layered_intermediate",
            "rough_tail",
        ],
        right=False,
    ).astype(str)
    comparison.to_csv(
        report / "layered_growth_per_group_audit.csv",
        index=False,
    )

    summary_rows = [_mean_metrics(comparison, "all27")]
    for regime, rows in comparison.groupby(
        "predicted_growth_regime", observed=True
    ):
        summary_rows.append(_mean_metrics(rows, f"predicted_{regime}"))
    measured_mid = comparison.loc[
        comparison["true_target"].between(3.5, 6.0, inclusive="both")
    ]
    summary_rows.append(
        _mean_metrics(measured_mid, "measured_Sq_3p5_to_6p0_nm")
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(report / "layered_growth_summary.csv", index=False)

    protected = comparison.loc[
        (comparison["predicted_target"] < 2.2)
        | (comparison["predicted_target"] >= 7.6)
    ]
    protected_rows: list[dict[str, float | bool | str]] = []
    for row in protected.itertuples(index=False):
        baseline_array, baseline_sq = _generated(
            output,
            split="crossfit",
            method=baseline,
            group=str(row.growth_run_id),
        )
        selected_array, selected_sq = _generated(
            output,
            split="crossfit",
            method=selected,
            group=str(row.growth_run_id),
        )
        protected_rows.append(
            {
                "growth_run_id": str(row.growth_run_id),
                "predicted_sq_nm": float(row.predicted_target),
                "array_equal_to_m20": bool(
                    np.array_equal(baseline_array, selected_array)
                ),
                "maximum_absolute_height_difference_nm": float(
                    np.max(np.abs(baseline_array - selected_array))
                ),
                "baseline_sq_nm": float(baseline_sq),
                "selected_sq_nm": float(selected_sq),
            }
        )
    protected_audit = pd.DataFrame(protected_rows)
    protected_audit.to_csv(
        report / "protected_regime_array_audit.csv",
        index=False,
    )

    fold_audit = pd.read_csv(report / "fold_integrity_audit.csv")
    write_json(
        {
            "selected_method": selected,
            "baseline_method": baseline,
            "growth_count": int(len(comparison)),
            "measured_mid_growth_count": int(len(measured_mid)),
            "measured_mid_relative_island_mae_improvement": float(
                summary.loc[
                    summary["subset"] == "measured_Sq_3p5_to_6p0_nm",
                    "relative_island_mae_improvement",
                ].iloc[0]
            ),
            "protected_growth_count": int(len(protected_audit)),
            "all_protected_arrays_equal_to_m20": bool(
                protected_audit["array_equal_to_m20"].all()
            ),
            "maximum_protected_height_difference_nm": float(
                protected_audit[
                    "maximum_absolute_height_difference_nm"
                ].max()
            ),
            "all_outer_fold_leakage_checks_passed": bool(
                fold_audit["held_overlap_with_fit"].eq(False).all()
                and comparison["outer_target_used_for_training"]
                .eq(False)
                .all()
            ),
            "query_measured_afm_used_at_inference": False,
            "selection_scope": (
                "retrospective method development; prospective confirmation "
                "is still required"
            ),
        },
        report / "layered_growth_audit_manifest.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(Path(args.config)))


if __name__ == "__main__":
    main()
