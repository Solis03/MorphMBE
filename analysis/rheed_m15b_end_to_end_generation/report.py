from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.rheed_to_afm_functional_morphology.visualization import (
    _comparison_figure,
    _save,
)
from analysis.rheed_to_afm_full_cohort_loo.run import (
    load_config,
    load_source_tables,
)
from analysis.rheed_to_afm_full_cohort_loo.visualization import (
    _external_target_confidence,
)
from analysis.rheed_video_afm_story.common import (
    repo_path,
    sha256_file,
    write_csv,
    write_json,
)


M12A = "M12a_edge_preserving_terrace"
M15B = "M15b_auto_r3d_angular_tta"


def _target_metrics(path: Path) -> dict[str, dict[str, float]]:
    rows = pd.read_csv(path).set_index("target")
    return {
        str(target): {
            "mae_nm": float(row["mean_absolute_error"]),
            "rmse_nm": float(row["rmse"]),
            "pearson_r": float(row["pearson_r"]),
            "spearman_rho": float(row["spearman_rho"]),
            "interval_coverage": float(row["interval_coverage"]),
        }
        for target, row in rows.iterrows()
    }


def _selected_method_metrics(path: Path) -> dict[str, float]:
    rows = pd.read_csv(path)
    selected = rows.loc[rows["method"] == M12A]
    if len(selected) != 1:
        raise RuntimeError(f"expected one {M12A} method-summary row")
    row = selected.iloc[0]
    return {
        "generated_rq_mae_nm": float(row["mean_rq_absolute_error_nm"]),
        "generated_fsmi_mae_nm": float(row["mean_fsmi_absolute_error_nm"]),
        "texture_gate_pass_fraction": float(
            row["texture_gate_pass_fraction"]
        ),
        "median_sharpness_ratio": float(row["median_sharpness_ratio"]),
        "median_afm_likeness_percentile": float(
            row["median_afm_likeness_percentile"]
        ),
        "mean_island_feature_mae_z": float(
            row["mean_island_feature_mae_z"]
        ),
    }


def _comparison_table(
    *,
    current_report: Path,
    corrected_m14i_metrics: Path,
) -> pd.DataFrame:
    prior = pd.read_csv(corrected_m14i_metrics)
    if "protocol" in prior.columns:
        prior = prior.loc[prior["protocol"] == "auto→auto (strict LOO)"]
        prior_label = "M14i corrected-target automatic input"
        prior_protocol = "M14i_auto_input_line3"
        mae_column = "mae"
    else:
        baseline_method = "M14b_rheed_density_weighted"
        prior = prior.loc[prior["method"] == baseline_method]
        prior_label = "M14b full-cohort physics baseline"
        prior_protocol = "M14b_full_cohort_physics_baseline"
        mae_column = "mae_nm"
    prior = prior.set_index("target")
    if set(prior.index) != {"Rq_nm", "FSMI_nm"}:
        raise RuntimeError("corrected M14i automatic-input metrics are missing")
    targets = _target_metrics(
        current_report / "target_prediction_summary.csv"
    )
    image = _selected_method_metrics(current_report / "method_summary.csv")
    return pd.DataFrame(
        [
            {
                "protocol": prior_protocol,
                "label": prior_label,
                "sq_mae_nm": float(prior.loc["Rq_nm", mae_column]),
                "sq_pearson_r": float(
                    prior.loc["Rq_nm", "pearson_r"]
                ),
                "fsmi_mae_nm": float(prior.loc["FSMI_nm", mae_column]),
                "fsmi_pearson_r": float(
                    prior.loc["FSMI_nm", "pearson_r"]
                ),
                "generated_sq_mae_nm": np.nan,
                "generated_fsmi_mae_nm": np.nan,
                "texture_gate_pass_fraction": np.nan,
                "median_sharpness_ratio": np.nan,
                "median_afm_likeness_percentile": np.nan,
                "mean_island_feature_mae_z": np.nan,
            },
            {
                "protocol": M15B,
                "label": "M15b corrected-target automatic input",
                "sq_mae_nm": targets["Rq_nm"]["mae_nm"],
                "sq_pearson_r": targets["Rq_nm"]["pearson_r"],
                "fsmi_mae_nm": targets["FSMI_nm"]["mae_nm"],
                "fsmi_pearson_r": targets["FSMI_nm"]["pearson_r"],
                "generated_sq_mae_nm": image[
                    "generated_rq_mae_nm"
                ],
                **{
                    key: value
                    for key, value in image.items()
                    if key != "generated_rq_mae_nm"
                },
            },
        ]
    )


def _integrity_audit(
    *,
    output: Path,
    report: Path,
    strict_predictions: pd.DataFrame,
) -> pd.DataFrame:
    expected = strict_predictions.pivot(
        index="growth_run_id",
        columns="target",
        values="predicted_target",
    )
    folds = pd.read_csv(
        report / "fold_integrity_audit.csv",
        dtype={"held_growth_run_id": str},
    ).set_index("held_growth_run_id")
    records = []
    map_root = (
        output
        / "crossfit"
        / "generated_maps"
        / M12A
    )
    for group in expected.index:
        path = map_root / f"{group}.npz"
        if not path.exists():
            raise RuntimeError(f"generated map is missing for {group}")
        payload = np.load(path, allow_pickle=False)
        shapes = np.asarray(payload["generated_unit_shapes"], dtype=float)
        record = {
            "growth_run_id": group,
            "map_exists": True,
            "draw_count": int(shapes.shape[0]),
            "height": int(shapes.shape[1]),
            "width": int(shapes.shape[2]),
            "unit_rq_mean": float(
                np.mean(np.std(shapes, axis=(1, 2)))
            ),
            "rq_matches_strict_m15b": bool(
                np.isclose(
                    float(payload["predicted_rq_nm"]),
                    float(expected.loc[group, "Rq_nm"]),
                    rtol=1e-7,
                    atol=1e-7,
                )
            ),
            "fsmi_matches_strict_m15b": bool(
                np.isclose(
                    float(payload["predicted_fsmi_nm"]),
                    float(expected.loc[group, "FSMI_nm"]),
                    rtol=1e-7,
                    atol=1e-7,
                )
            ),
            "retrieval_at_inference": bool(
                payload["retrieval_at_inference"]
            ),
            "measured_afm_patch_used_at_inference": bool(
                payload["measured_afm_patch_used_at_inference"]
            ),
            "held_overlap_with_generator_fit": bool(
                folds.loc[group, "held_overlap_with_fit"]
            ),
            "generator_fit_growth_count": int(
                folds.loc[group, "fit_growth_count"]
            ),
        }
        records.append(record)
    audit = pd.DataFrame(records)
    required_true = [
        "map_exists",
        "rq_matches_strict_m15b",
        "fsmi_matches_strict_m15b",
    ]
    required_false = [
        "retrieval_at_inference",
        "measured_afm_patch_used_at_inference",
        "held_overlap_with_generator_fit",
    ]
    if not audit[required_true].all().all():
        raise RuntimeError("end-to-end map/target integrity check failed")
    if audit[required_false].any().any():
        raise RuntimeError("retrieval, measured patch or fold leakage detected")
    expected_fit_count = int(len(expected) - 1)
    if not (
        (audit["draw_count"] == 4)
        & (audit["height"] == 128)
        & (audit["width"] == 128)
        & (audit["generator_fit_growth_count"] == expected_fit_count)
    ).all():
        raise RuntimeError("generated ensemble geometry/fold size changed")
    return audit


def _overview_groups(rq: pd.DataFrame) -> list[str]:
    ordered = rq.sort_values("true_target").reset_index(drop=True)
    positions = [
        0,
        int(round((len(ordered) - 1) / 3)),
        int(round(2 * (len(ordered) - 1) / 3)),
        len(ordered) - 1,
    ]
    groups = list(ordered.iloc[positions]["growth_run_id"].astype(str))
    for group in rq.sort_values(
        "absolute_error", ascending=False
    )["growth_run_id"].astype(str):
        if group not in groups:
            groups.append(group)
            break
    return groups


def _overview_figure(
    *,
    config: dict[str, Any],
    output: Path,
    report: Path,
) -> tuple[list[str], Path]:
    phase1 = load_source_tables(config)["phase1"].copy()
    rq = pd.read_csv(
        report / "rq_crossfit_predictions.csv",
        dtype={"growth_run_id": str},
    )
    fsmi = pd.read_csv(
        report / "fsmi_crossfit_predictions.csv",
        dtype={"growth_run_id": str},
    )
    confidence = pd.read_csv(
        report / "confidence_crossfit.csv",
        dtype={"growth_run_id": str},
    )
    confidence = _external_target_confidence(
        path=config["external_confidence_predictions"],
        fallback=confidence,
        method=str(config["external_confidence_method"]),
    )
    groups = _overview_groups(rq)
    figure = _comparison_figure(
        groups=groups,
        split="crossfit",
        output=output,
        phase1=phase1,
        rq_predictions=rq,
        fsmi_predictions=fsmi,
        confidence=confidence,
        method=M12A,
        title=(
            "M15b + M12a end-to-end automatic-video prediction: "
            "four fixed Sq strata plus the largest non-duplicate Sq failure"
        ),
    )
    figure.set_size_inches(11.2, 2.65 * len(groups))
    destination = report / "figures" / "Fig0_m15b_end_to_end_overview"
    _save(figure, destination)
    return groups, destination.with_suffix(".png")


def run(config_path: str | Path) -> None:
    config = load_config(config_path)
    suffix = str(config.get("full_run_suffix", "full23_loo"))
    output = repo_path(config["output_root"]) / suffix
    report = repo_path(config["report_root"]) / suffix
    strict_path = repo_path(config["external_confidence_predictions"])
    strict = pd.read_csv(
        strict_path,
        dtype={"growth_run_id": str},
    )
    comparison = _comparison_table(
        current_report=report,
        corrected_m14i_metrics=repo_path(
            config["corrected_m14i_metrics"]
        ),
    )
    audit = _integrity_audit(
        output=output,
        report=report,
        strict_predictions=strict,
    )
    groups, overview = _overview_figure(
        config=config,
        output=output,
        report=report,
    )
    write_csv(comparison, report / "baseline_vs_m15b_end_to_end.csv")
    write_csv(audit, report / "end_to_end_integrity_audit.csv")

    current = comparison.loc[
        comparison["protocol"] == M15B
    ].iloc[0]
    prior = comparison.loc[
        comparison["protocol"] != M15B
    ].iloc[0]
    cohort_count = int(len(audit))
    manifest = {
        "experiment_id": config["experiment_id"],
        "model_id": (
            "MorphMBE-M15b-AutoR3D-AngularTTA + "
            "M12a-RangeTerrace"
        ),
        "outer_growth_count": int(len(audit)),
        "outer_fit_growth_count": cohort_count - 1,
        "generated_draws_per_growth": 4,
        "generated_resolution": [128, 128],
        "strict_scalar_prediction_sha256": sha256_file(strict_path),
        "all_scalar_targets_match_generated_map_metadata": bool(
            audit[
                ["rq_matches_strict_m15b", "fsmi_matches_strict_m15b"]
            ].all().all()
        ),
        "all_generator_fold_leakage_checks_passed": bool(
            not audit["held_overlap_with_generator_fit"].any()
        ),
        "retrieval_at_inference": bool(
            audit["retrieval_at_inference"].any()
        ),
        "measured_afm_patch_at_inference": bool(
            audit["measured_afm_patch_used_at_inference"].any()
        ),
        "overview_growths": groups,
        "overview_path": str(overview),
        "sq_mae_improvement_vs_same_cohort_physics_baseline_nm": float(
            prior["sq_mae_nm"] - current["sq_mae_nm"]
        ),
        "fsmi_mae_improvement_vs_same_cohort_physics_baseline_nm": float(
            prior["fsmi_mae_nm"] - current["fsmi_mae_nm"]
        ),
        "claim_boundary": (
            f"Strict outer target and generator-fold LOO over "
            f"{cohort_count} growths. "
            "The M12a family was developed on earlier partitions, so this "
            "is retrospective validation rather than a prospective test."
        ),
    }
    write_json(manifest, report / "end_to_end_manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_m15b_end_to_end_generation.json",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
