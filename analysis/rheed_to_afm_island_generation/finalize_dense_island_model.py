from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.rheed_to_afm_distinct_confidence.run import _load_tables
from analysis.rheed_to_afm_generation.data import ConditionScaler
from analysis.rheed_to_afm_sharp_generation.evaluation import (
    evaluate_method_sets,
)
from analysis.rheed_video_afm_story.common import (
    repo_path,
    sha256_file,
    write_csv,
    write_json,
)

from .evaluation import evaluate_island_methods
from .evaluate_topology_renderers import _structure_blend
from .sample_guided_diffusion import _arrays
from .train_guided_diffusion import _load_config


M5 = "M5_cloudlike_spectral_hybrid"
M6B = "M6b_multiscale_laguerre_terraces"
M10 = "M10_dense_island_spectral_pareto"


def _save(
    root: Path,
    *,
    group: str,
    arrays: list[np.ndarray],
    predicted_rq_nm: float,
    weight: float,
) -> None:
    path = root / "generated_maps" / M10 / f"{group}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        generated_unit_shapes=np.stack(arrays).astype(np.float32),
        predicted_rq_nm=np.asarray(predicted_rq_nm),
        growth_run_id=np.asarray(group),
        method=np.asarray(M10),
        island_structure_weight=np.asarray(float(weight)),
        retrieval_at_inference=np.asarray(False),
        measured_afm_patch_used_at_inference=np.asarray(False),
    )


def _aggregate(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = pd.concat(frames, ignore_index=True)
    records: list[dict[str, Any]] = []
    for method, group in values.groupby("method"):
        record: dict[str, Any] = {
            "method": method,
            "growth_group_count": int(group["growth_run_id"].nunique()),
        }
        for column in group.select_dtypes(include=[np.number, bool]):
            record[f"median_{column}"] = float(
                group[column].astype(float).median()
            )
        if "afm_texture_gate_pass" in group:
            record["texture_gate_pass_fraction"] = float(
                group["afm_texture_gate_pass"].astype(float).mean()
            )
        records.append(record)
    return values, pd.DataFrame(records).sort_values("method")


def run(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    tables = _load_tables(config)
    descriptors = tables["descriptors"]
    train_rows = descriptors.loc[descriptors["split"] == "train"].copy()
    validation_rows = descriptors.loc[descriptors["split"] == "val"].copy()
    train_groups = sorted(train_rows["growth_run_id"].astype(str).unique())
    output = repo_path(args.output)
    report = repo_path(args.report)
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    cross_source = repo_path(args.cross_source)
    validation_source = repo_path(args.validation_source)
    cross_standard_frames = []
    cross_island_frames = []
    for fold, held in enumerate(train_groups):
        fit_groups = set(train_groups)
        fit_groups.remove(held)
        fit_rows = train_rows.loc[
            train_rows["growth_run_id"].astype(str).isin(fit_groups)
        ]
        held_rows = train_rows.loc[
            train_rows["growth_run_id"].astype(str) == held
        ]
        m5 = _arrays(cross_source / M5 / f"{held}.npz")
        m6b = _arrays(cross_source / M6B / f"{held}.npz")
        m10 = _structure_blend(m6b, m5, weight=float(args.weight))
        predicted_rq = float(
            np.load(
                cross_source / M5 / f"{held}.npz", allow_pickle=False
            )["predicted_rq_nm"]
        )
        _save(
            output / "crossfit",
            group=held,
            arrays=m10,
            predicted_rq_nm=predicted_rq,
            weight=float(args.weight),
        )
        scaler = ConditionScaler.fit(
            train_rows,
            list(config["condition_columns"]),
            fit_groups,
        )
        methods = {M5: m5, M10: m10}
        standard = evaluate_method_sets(
            split_rows=held_rows,
            train_rows=fit_rows,
            condition_scaler=scaler,
            generated={
                method: {held: arrays}
                for method, arrays in methods.items()
            },
            generated_rq={
                method: {held: predicted_rq} for method in methods
            },
            output_dir=report / "crossfit" / "folds" / held / "standard",
            resolution=int(config["resolution"]),
        )["per_group"]
        standard.insert(0, "cross_validation_fold", fold)
        cross_standard_frames.append(standard)
        island = evaluate_island_methods(
            held_rows=held_rows,
            train_rows=fit_rows,
            generated=methods,
            resolution=int(config["resolution"]),
        )
        island.insert(0, "cross_validation_fold", fold)
        cross_island_frames.append(island)
    cross_standard, cross_standard_summary = _aggregate(
        cross_standard_frames
    )
    cross_island, cross_island_summary = _aggregate(cross_island_frames)
    write_csv(cross_standard, report / "crossfit" / "standard_per_group.csv")
    write_csv(
        cross_standard_summary, report / "crossfit" / "standard_summary.csv"
    )
    write_csv(cross_island, report / "crossfit" / "island_per_group.csv")
    write_csv(
        cross_island_summary, report / "crossfit" / "island_summary.csv"
    )

    all_train_groups = set(train_groups)
    validation_scaler = ConditionScaler.fit(
        train_rows,
        list(config["condition_columns"]),
        all_train_groups,
    )
    validation_methods: dict[str, dict[str, list[np.ndarray]]] = {
        M5: {},
        M10: {},
    }
    validation_rq: dict[str, dict[str, float]] = {M5: {}, M10: {}}
    validation_island_frames = []
    validation_groups = sorted(
        validation_rows["growth_run_id"].astype(str).unique()
    )
    for group in validation_groups:
        m5 = _arrays(validation_source / M5 / f"{group}.npz")
        m6b = _arrays(validation_source / M6B / f"{group}.npz")
        m10 = _structure_blend(m6b, m5, weight=float(args.weight))
        predicted_rq = float(
            np.load(
                validation_source / M5 / f"{group}.npz",
                allow_pickle=False,
            )["predicted_rq_nm"]
        )
        validation_methods[M5][group] = m5
        validation_methods[M10][group] = m10
        validation_rq[M5][group] = predicted_rq
        validation_rq[M10][group] = predicted_rq
        _save(
            output / "validation",
            group=group,
            arrays=m10,
            predicted_rq_nm=predicted_rq,
            weight=float(args.weight),
        )
        held_rows = validation_rows.loc[
            validation_rows["growth_run_id"].astype(str) == group
        ]
        validation_island_frames.append(
            evaluate_island_methods(
                held_rows=held_rows,
                train_rows=train_rows,
                generated={M5: m5, M10: m10},
                resolution=int(config["resolution"]),
            )
        )
    validation_standard = evaluate_method_sets(
        split_rows=validation_rows,
        train_rows=train_rows,
        condition_scaler=validation_scaler,
        generated=validation_methods,
        generated_rq=validation_rq,
        output_dir=report / "validation" / "standard",
        resolution=int(config["resolution"]),
    )
    validation_island, validation_island_summary = _aggregate(
        validation_island_frames
    )
    write_csv(
        validation_island,
        report / "validation" / "island_per_group.csv",
    )
    write_csv(
        validation_island_summary,
        report / "validation" / "island_summary.csv",
    )
    baseline_vs_final = []
    for split, standard_summary, island_summary in (
        ("strict_15_growth_loo", cross_standard_summary, cross_island_summary),
        (
            "preexisting_validation",
            validation_standard["summary"],
            validation_island_summary,
        ),
    ):
        for method in (M5, M10):
            standard_row = standard_summary.loc[
                standard_summary["method"] == method
            ].iloc[0]
            island_row = island_summary.loc[
                island_summary["method"] == method
            ].iloc[0]
            baseline_vs_final.append(
                {
                    "split": split,
                    "method": method,
                    "rq_mae_nm": standard_row[
                        "median_rq_absolute_error_nm"
                    ],
                    "condition_descriptor_mae_z": standard_row[
                        "median_condition_descriptor_mae_z"
                    ],
                    "psd_log_distance": standard_row[
                        "median_normalized_psd_log_distance"
                    ],
                    "composite_error": standard_row[
                        "median_composite_score"
                    ],
                    "sharpness_ratio": standard_row[
                        "median_sharpness_ratio"
                    ],
                    "texture_gate_pass_fraction": standard_row[
                        "texture_gate_pass_fraction"
                    ],
                    "island_feature_mae_z": island_row[
                        "median_island_feature_mae_z"
                    ],
                    "afm_prior_mahalanobis": island_row[
                        "median_afm_prior_mahalanobis"
                    ],
                    "q70_median_area_log_error": island_row[
                        "median_median_area_q70_log_absolute_error"
                    ],
                    "q70_component_count_log_error": island_row[
                        "median_component_count_q70_log_absolute_error"
                    ],
                    "max_training_ssim": standard_row[
                        "median_max_training_ssim"
                    ],
                }
            )
    baseline_vs_final_frame = pd.DataFrame(baseline_vs_final)
    write_csv(baseline_vs_final_frame, report / "baseline_vs_final_metrics.csv")
    manifest = {
        "selected_method": M10,
        "island_structure_weight": float(args.weight),
        "spectral_weight": 1.0 - float(args.weight),
        "training_growth_groups": train_groups,
        "validation_growth_groups": validation_groups,
        "historical_test_used": False,
        "retrieval_at_inference": False,
        "measured_afm_patch_used_at_inference": False,
        "removelist_sha256": sha256_file(
            repo_path(config["removelist_path"])
        ),
        "crossfit_standard_summary": cross_standard_summary.to_dict(
            orient="records"
        ),
        "crossfit_island_summary": cross_island_summary.to_dict(
            orient="records"
        ),
        "validation_standard_summary": validation_standard[
            "summary"
        ].to_dict(orient="records"),
        "validation_island_summary": validation_island_summary.to_dict(
            orient="records"
        ),
        "claim_boundary": (
            "Development evidence from strict training-growth LOO and the "
            "pre-existing validation cohort. The historical test remains "
            "closed; prospective growth groups are required for confirmation."
        ),
    }
    write_json(manifest, report / "best_model_manifest.json")
    print(json.dumps(manifest, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_to_afm_island_generation_v3_dense.json",
    )
    parser.add_argument("--cross-source", required=True)
    parser.add_argument("--validation-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--weight", type=float, default=0.65)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
