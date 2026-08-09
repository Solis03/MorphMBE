"""Quantitative audit for the M20 spot-connectivity Sq/morphology redesign."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.rheed_to_afm_full_cohort_loo.run import load_config
from analysis.rheed_video_afm_story.common import repo_path, write_json


def _sq_metrics(
    frame: pd.DataFrame, prediction: str, *, label: str, mask: np.ndarray
) -> dict[str, float | int | str]:
    residual = (
        frame.loc[mask, prediction].to_numpy(float)
        - frame.loc[mask, "true_target"].to_numpy(float)
    )
    return {
        "model": label,
        "stratum": "rough_3_to_10_nm" if not np.all(mask) else "all",
        "growth_count": int(np.sum(mask)),
        "mae_nm": float(np.mean(np.abs(residual))),
        "rmse_nm": float(np.sqrt(np.mean(np.square(residual)))),
        "bias_nm": float(np.mean(residual)),
    }


def _physical_quantiles(path: Path) -> dict[str, float]:
    payload = np.load(path, allow_pickle=False)
    unit = np.asarray(payload["generated_unit_shapes"][0], dtype=float)
    predicted_sq = float(payload["predicted_rq_nm"])
    surface = unit * predicted_sq
    q01, q05, median, q95, q99 = np.quantile(
        surface, [0.01, 0.05, 0.50, 0.95, 0.99]
    )
    return {
        "generated_sq_nm": float(np.std(surface)),
        "q01_nm": float(q01),
        "q05_nm": float(q05),
        "median_nm": float(median),
        "q95_nm": float(q95),
        "q99_nm": float(q99),
        "negative_tail_depth_median_to_q01_nm": float(median - q01),
        "robust_q01_q99_range_nm": float(q99 - q01),
    }


def run(config: dict, m19_config: dict) -> None:
    suffix = str(config["full_run_suffix"])
    output = repo_path(config["output_root"]) / suffix
    report = repo_path(config["report_root"]) / suffix
    audit_dir = report / "connectivity_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    m19_prediction_path = repo_path(
        m19_config["external_target_predictions"]["Rq_nm"]
    )
    m20_prediction_path = repo_path(
        config["external_target_predictions"]["Rq_nm"]
    )
    m19 = pd.read_csv(m19_prediction_path, dtype={"growth_run_id": str})
    m20 = pd.read_csv(m20_prediction_path, dtype={"growth_run_id": str})
    joined = m20.merge(
        m19.loc[:, ["growth_run_id", "predicted_target"]].rename(
            columns={"predicted_target": "m19_sq_nm"}
        ),
        on="growth_run_id",
        validate="one_to_one",
    ).rename(columns={"predicted_target": "m20_sq_nm"})
    truth = joined["true_target"].to_numpy(float)
    rough = (truth >= 3.0) & (truth <= 10.0)
    sq_metrics = pd.DataFrame(
        [
            _sq_metrics(
                joined, "m19_sq_nm", label="M19", mask=np.ones(len(joined), bool)
            ),
            _sq_metrics(joined, "m20_sq_nm", label="M20", mask=np.ones(len(joined), bool)),
            _sq_metrics(joined, "m19_sq_nm", label="M19", mask=rough),
            _sq_metrics(joined, "m20_sq_nm", label="M20", mask=rough),
        ]
    )
    sq_metrics.to_csv(audit_dir / "sq_metrics_m19_vs_m20.csv", index=False)

    standard = pd.read_csv(report / "crossfit" / "standard_per_group.csv")
    island = pd.read_csv(report / "crossfit" / "island_per_group.csv")
    rough_groups = set(joined.loc[rough, "growth_run_id"].astype(str))
    rows: list[dict[str, float | int | str]] = []
    for method in config["publication_ablation_methods"]:
        standard_part = standard.loc[
            (standard["method"] == method)
            & standard["growth_run_id"].astype(str).isin(rough_groups)
        ]
        island_part = island.loc[
            (island["method"] == method)
            & island["growth_run_id"].astype(str).isin(rough_groups)
        ]
        rows.append(
            {
                "method": method,
                "rough_growth_count": len(standard_part),
                "mean_island_feature_mae_z": float(
                    island_part["island_feature_mae_z"].mean()
                ),
                "mean_generated_component_count_q70": float(
                    np.expm1(island_part["generated__log_component_count_q70"]).mean()
                ),
                "mean_real_component_count_q70": float(
                    np.expm1(island_part["true__log_component_count_q70"]).mean()
                ),
                "mean_generated_median_area_q70_px": float(
                    np.expm1(island_part["generated__log_median_area_q70"]).mean()
                ),
                "mean_real_median_area_q70_px": float(
                    np.expm1(island_part["true__log_median_area_q70"]).mean()
                ),
                "mean_generated_flat_fraction": float(
                    island_part["generated__flat_fraction"].mean()
                ),
                "mean_real_flat_fraction": float(
                    island_part["true__flat_fraction"].mean()
                ),
                "mean_normalized_psd_log_distance": float(
                    standard_part["normalized_psd_log_distance"].mean()
                ),
                "mean_sharpness_ratio": float(
                    standard_part["sharpness_ratio"].mean()
                ),
            }
        )
    pd.DataFrame(rows).to_csv(
        audit_dir / "rough_morphology_ablation.csv", index=False
    )

    selected_method = str(config["selected_method"])
    pair_rows: list[dict[str, float | int | str | bool]] = []
    for growth in ["6062", "6099"]:
        row = joined.loc[joined["growth_run_id"].astype(str) == growth].iloc[0]
        map_path = (
            output
            / "crossfit"
            / "generated_maps"
            / selected_method
            / f"{growth}.npz"
        )
        pair_rows.append(
            {
                "growth_run_id": growth,
                "true_sq_nm": float(row["true_target"]),
                "m19_sq_nm": float(row["m19_sq_nm"]),
                "m20_sq_nm": float(row["m20_sq_nm"]),
                "m19_absolute_error_nm": float(
                    abs(row["m19_sq_nm"] - row["true_target"])
                ),
                "m20_absolute_error_nm": float(
                    abs(row["m20_sq_nm"] - row["true_target"])
                ),
                "rheed_spot_isolation_score": float(
                    row["rheed_spot_isolation_score"]
                ),
                "bridged_spot_correction_gate": bool(
                    row["spot_connectivity_gate"]
                ),
                "isolated_spot_uplift_gate": bool(
                    row["isolated_spot_uplift_gate"]
                ),
                **_physical_quantiles(map_path),
            }
        )
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(audit_dir / "focus_6062_6099_physical_audit.csv", index=False)
    by_growth = pair.set_index("growth_run_id")
    checks = {
        "6099_is_more_spot_isolated_than_6062": bool(
            by_growth.loc["6099", "rheed_spot_isolation_score"]
            > by_growth.loc["6062", "rheed_spot_isolation_score"]
        ),
        "6099_predicted_sq_exceeds_6062": bool(
            by_growth.loc["6099", "m20_sq_nm"]
            > by_growth.loc["6062", "m20_sq_nm"]
        ),
        "6099_generated_negative_tail_is_deeper_than_6062": bool(
            by_growth.loc[
                "6099", "negative_tail_depth_median_to_q01_nm"
            ]
            > by_growth.loc[
                "6062", "negative_tail_depth_median_to_q01_nm"
            ]
        ),
        "6099_generated_robust_height_range_exceeds_6062": bool(
            by_growth.loc["6099", "robust_q01_q99_range_nm"]
            > by_growth.loc["6062", "robust_q01_q99_range_nm"]
        ),
        "6062_sq_error_improved_over_m19": bool(
            by_growth.loc["6062", "m20_absolute_error_nm"]
            < by_growth.loc["6062", "m19_absolute_error_nm"]
        ),
        "6099_sq_error_improved_over_m19": bool(
            by_growth.loc["6099", "m20_absolute_error_nm"]
            < by_growth.loc["6099", "m19_absolute_error_nm"]
        ),
    }
    write_json(
        {
            "model": "M20_spot_connectivity_calibrated_sq",
            "selected_morphology_method": selected_method,
            "query_target_used_for_connectivity_decision": False,
            "all_pair_physical_consistency_checks_passed": all(checks.values()),
            "checks": checks,
            "sq_metrics": sq_metrics.to_dict(orient="records"),
        },
        audit_dir / "connectivity_audit_manifest.json",
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Physical consistency checks failed: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--m19-config", required=True)
    args = parser.parse_args()
    run(load_config(Path(args.config)), load_config(Path(args.m19_config)))


if __name__ == "__main__":
    main()
