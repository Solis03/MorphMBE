from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.rheed_to_afm_full_cohort_loo.run import load_config
from analysis.rheed_video_afm_story.common import repo_path, write_csv, write_json


def _expm1_median(frame: pd.DataFrame, column: str) -> float:
    return float(np.expm1(np.median(frame[column].to_numpy(float))))


def _sq_metrics(frame: pd.DataFrame, label: str) -> list[dict]:
    truth = frame["true_target"].to_numpy(float)
    predicted = frame["predicted_target"].to_numpy(float)
    records = []
    for stratum, mask in (
        ("all", np.ones(len(frame), dtype=bool)),
        ("smooth_below_1p6_nm", truth < 1.6),
        ("rough_3_to_10_nm", (truth >= 3.0) & (truth <= 10.0)),
    ):
        residual = predicted[mask] - truth[mask]
        records.append(
            {
                "model": label,
                "stratum": stratum,
                "count": int(np.sum(mask)),
                "mae_nm": float(np.mean(np.abs(residual))),
                "rmse_nm": float(np.sqrt(np.mean(np.square(residual)))),
                "bias_nm": float(np.mean(residual)),
            }
        )
    return records


def run(
    config: dict,
    *,
    baseline_method: str,
    m17_predictions: Path,
) -> None:
    suffix = str(config["full_run_suffix"])
    report = repo_path(config["report_root"]) / suffix
    output = repo_path(config["output_root"]) / suffix
    audit = report / "rough_island_audit"
    selected = str(config["selected_method"])
    rq = pd.read_csv(
        report / "rq_crossfit_predictions.csv",
        dtype={"growth_run_id": str},
    )
    rough_groups = set(
        rq.loc[
            (rq["true_target"] >= 3.0) & (rq["true_target"] <= 10.0),
            "growth_run_id",
        ].astype(str)
    )
    island = pd.read_csv(
        report / "crossfit" / "island_per_group.csv",
        dtype={"growth_run_id": str},
    )
    standard = pd.read_csv(
        report / "crossfit" / "standard_per_group.csv",
        dtype={"growth_run_id": str},
    )
    method_records = []
    for method in (baseline_method, selected):
        island_rows = island.loc[
            (island["method"] == method)
            & island["growth_run_id"].isin(rough_groups)
        ]
        standard_rows = standard.loc[
            (standard["method"] == method)
            & standard["growth_run_id"].isin(rough_groups)
        ]
        method_records.append(
            {
                "method": method,
                "rough_growth_count": len(island_rows),
                "median_island_feature_mae_z": float(
                    island_rows["island_feature_mae_z"].median()
                ),
                "median_generated_q70_count": _expm1_median(
                    island_rows, "generated__log_component_count_q70"
                ),
                "median_true_q70_count": _expm1_median(
                    island_rows, "true__log_component_count_q70"
                ),
                "median_generated_q70_area_px": _expm1_median(
                    island_rows, "generated__log_median_area_q70"
                ),
                "median_true_q70_area_px": _expm1_median(
                    island_rows, "true__log_median_area_q70"
                ),
                "median_generated_q55_area_px": _expm1_median(
                    island_rows, "generated__log_median_area_q55"
                ),
                "median_true_q55_area_px": _expm1_median(
                    island_rows, "true__log_median_area_q55"
                ),
                "median_generated_flat_fraction": float(
                    island_rows["generated__flat_fraction"].median()
                ),
                "median_true_flat_fraction": float(
                    island_rows["true__flat_fraction"].median()
                ),
                "median_psd_log_distance": float(
                    standard_rows["normalized_psd_log_distance"].median()
                ),
                "median_sharpness_ratio": float(
                    standard_rows["sharpness_ratio"].median()
                ),
            }
        )
    method_summary = pd.DataFrame(method_records)
    write_csv(method_summary, audit / "rough_3_to_10_method_summary.csv")

    m17_sq = pd.read_csv(m17_predictions, dtype={"growth_run_id": str})
    sq_summary = pd.DataFrame(
        [
            *_sq_metrics(m17_sq, "M17 Sq"),
            *_sq_metrics(rq, "M19 rough-tail Sq"),
        ]
    )
    write_csv(sq_summary, audit / "sq_stratum_summary.csv")

    rough_start = float(config["selected_renderer"]["rough_start_nm"])
    preservation_records = []
    for _, row in rq.loc[rq["predicted_target"] <= rough_start].iterrows():
        group = str(row["growth_run_id"])
        baseline = np.load(
            output
            / "crossfit"
            / "generated_maps"
            / baseline_method
            / f"{group}.npz",
            allow_pickle=False,
        )["generated_unit_shapes"]
        generated = np.load(
            output
            / "crossfit"
            / "generated_maps"
            / selected
            / f"{group}.npz",
            allow_pickle=False,
        )["generated_unit_shapes"]
        preservation_records.append(
            {
                "growth_run_id": group,
                "predicted_sq_nm": float(row["predicted_target"]),
                "array_exactly_equal_to_m17": bool(
                    np.array_equal(baseline, generated)
                ),
                "maximum_absolute_difference": float(
                    np.max(np.abs(baseline - generated))
                ),
            }
        )
    preservation = pd.DataFrame(preservation_records)
    write_csv(preservation, audit / "smooth_m17_exact_preservation.csv")
    write_json(
        {
            "selected_method": selected,
            "baseline_method": baseline_method,
            "rough_stratum": "measured Sq in [3, 10] nm",
            "rough_growth_count": len(rough_groups),
            "smooth_preservation_threshold_predicted_sq_nm": rough_start,
            "smooth_arrays_all_exact": bool(
                preservation["array_exactly_equal_to_m17"].all()
            ),
            "outer_loo_leakage_audit_passed": bool(
                (~pd.read_csv(report / "fold_integrity_audit.csv")[
                    "held_overlap_with_fit"
                ].astype(bool)).all()
            ),
        },
        audit / "audit_manifest.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--baseline-method",
        default="M17b_topology_sparse_peak_terrace",
    )
    parser.add_argument("--m17-predictions", type=Path, required=True)
    args = parser.parse_args()
    run(
        load_config(Path(args.config)),
        baseline_method=args.baseline_method,
        m17_predictions=args.m17_predictions,
    )


if __name__ == "__main__":
    main()
