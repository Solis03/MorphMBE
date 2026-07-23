#!/usr/bin/env python3
"""Build benchmark v1 Phase 0 registry, protocol, split, schema, and docs artifacts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from rheed2morph.benchmark.constants import (
    AFM_ONLY_UNMATCHED_ID,
    BASE_SEED,
    HISTORICAL_SAMPLE_IDS,
    METADATA_INVENTORY_COLUMNS,
    PROSPECTIVE_PENDING_TRUTH_ID,
    PROSPECTIVE_PILOT_IDS,
    REGISTRY_COLUMNS,
    UNMATCHED_AND_PENDING_IDS,
)
from rheed2morph.benchmark.hashing import (
    canonical_json,
    repo_root,
    sha256_file,
    write_csv_rows,
    write_json,
    write_sha256_manifest,
)
from rheed2morph.benchmark.registry import (
    build_master_registry,
    cohort_summary,
    historical_rows,
    metadata_inventory,
    paired_rows,
    prospective_pilot_rows,
    source_files,
    unmatched_pending_rows,
)
from rheed2morph.benchmark.splits import generate_outer_logo_from_registry, write_split


REGISTRY_DIR = Path("configs/benchmark_v1/registry")
SPLIT_DIR = Path("configs/benchmark_v1/splits")
SCHEMA_DIR = Path("configs/benchmark_v1/schemas")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-timestamp", default="20260723-110844")
    args = parser.parse_args()
    root = repo_root(Path.cwd())
    build_all(root, args.report_timestamp)


def build_all(root: Path, report_timestamp: str) -> None:
    rows = build_master_registry(root)
    summary = cohort_summary(rows, root)
    write_registry_artifacts(root, rows, summary)
    write_protocol(root, summary)
    write_split_artifacts(root)
    write_experiment_matrix(root)
    write_schema_artifacts(root)
    write_docs(root, summary)
    write_phase0_report(root, report_timestamp, summary)


def write_registry_artifacts(root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    write_csv_rows(root / REGISTRY_DIR / "sample_registry_master_v1.csv", rows, REGISTRY_COLUMNS)
    write_csv_rows(root / REGISTRY_DIR / "paired_supervised_cohort_v1.csv", paired_rows(rows), REGISTRY_COLUMNS)
    write_csv_rows(root / REGISTRY_DIR / "historical_development_cohort_v1.csv", historical_rows(rows), REGISTRY_COLUMNS)
    write_csv_rows(root / REGISTRY_DIR / "prospective_pilot_seen_v1.csv", prospective_pilot_rows(rows), REGISTRY_COLUMNS)
    write_csv_rows(root / REGISTRY_DIR / "unmatched_and_pending_v1.csv", unmatched_pending_rows(rows), REGISTRY_COLUMNS)
    write_json(root / REGISTRY_DIR / "cohort_summary_v1.json", summary)
    write_json(root / REGISTRY_DIR / "sample_registry.schema.json", sample_registry_schema())
    write_sha256_manifest(root / REGISTRY_DIR / "source_files_v1.sha256", source_files(), root)
    write_csv_rows(
        root / REGISTRY_DIR / "metadata_feature_inventory_v1.csv",
        metadata_inventory(root),
        METADATA_INVENTORY_COLUMNS,
    )


def write_protocol(root: Path, summary: dict[str, Any]) -> None:
    strata = summary["fixed_historical_strata_quantiles"]
    text = f"""protocol_version: benchmark_v1
created_for_phase: Nature Communications Benchmark Phase 0
base_seed: {BASE_SEED}
scientific_status: historical_model_selection_only; prospective pilot is seen/exploratory

cohort_roles:
  historical_development:
    count: 23
    role: Primary model-development cohort.
    sample_ids: [{", ".join(HISTORICAL_SAMPLE_IDS)}]
    eligibility: model development, primary nested CV, and final historical training only.
  prospective_pilot_seen:
    count: 4
    role: Seen pilot/external diagnostic after historical-only model selection.
    sample_ids: [{", ".join(PROSPECTIVE_PILOT_IDS)}]
    eligibility: pilot evaluation only; never model selection.
  prospective_pending_truth:
    count: 1
    sample_id: {PROSPECTIVE_PENDING_TRUTH_ID}
    role: RHEED prediction-only pending AFM truth; excluded from all metrics.
  afm_only_unmatched:
    count: 1
    sample_id: {AFM_ONLY_UNMATCHED_ID}
    role: AFM-only unmatched sample; excluded from all supervised metrics.
  future_blind_confirmatory:
    status: not yet collected
    requirement: newly grown independent smooth, intermediate, and rough samples.

primary_target:
  historical_definition: T4_second_order_trimmed_mean
  unit: nm
  registry_values: raw untransformed nm values
  target_transforms: model-side only; predictions must be inverse-transformed into nm before scoring
  clipping: prohibited for primary numerical metrics
  display_clipping: raw and clipped display values must be stored separately
  prospective_available_definition: true_rq_nm_median_second_order
  prospective_directly_comparable_to_T4: false
  prospective_note: Pilot truth is retained but not used for model choice or confirmatory claims.

historical_evaluation_design:
  outer_evaluation: Leave-One-Growth-Group-Out
  outer_fold_count: 23
  outer_test_size_growth_groups: 1
  outer_training_size_growth_groups: 22
  inner_model_selection: Leave-One-Growth-Group-Out restricted to the 22 outer-training groups
  primary_inner_loop_objective: minimize mean MAE in nm
  tie_breaking_order:
    - lower mean inner MAE
    - lower model complexity
    - fewer fitted parameters
    - lexicographically smaller configuration ID
  operations_inside_outer_training_fold_only:
    - feature scaling
    - PCA
    - feature selection
    - metadata imputation
    - hyperparameter optimization
    - target transformation fitting
    - calibration
    - conformal calibration
    - OOD threshold selection
    - ensemble selection
    - temporal frame-selection tuning
  prospective_samples_in_inner_or_outer_cv: prohibited

metrics:
  primary: MAE_nm
  secondary:
    - RMSE_nm
    - R2
    - Spearman_rho
    - Kendall_tau
    - Median_absolute_error_nm
    - Pairwise_concordance
    - Mean_signed_error_nm
    - Low_range_MAE_nm
    - Mid_range_MAE_nm
    - High_range_MAE_nm
  fixed_historical_strata:
    low: target_rq_nm <= {strata["low_max_nm"]:.15g}
    mid: {strata["low_max_nm"]:.15g} < target_rq_nm <= {strata["mid_max_nm"]:.15g}
    high: target_rq_nm > {strata["mid_max_nm"]:.15g}
    definition: quantiles are fixed once from the historical target distribution only
  prospective_pilot_reporting:
    - report every individual prediction
    - report signed and absolute error per sample
    - report aggregate MAE and RMSE cautiously
    - do not make strong claims from R2 or correlation on N=4
    - label all results pilot/exploratory, not confirmatory

statistical_uncertainty_registered_not_executed:
  bootstrap_unit: growth_group
  bootstrap_resamples: 10000
  permutation_unit: growth_group
  permutations: 10000
  confidence_interval: 95 percent
  seed_derivation: base seed + experiment ID + outer fold ID + model configuration ID
  unrecorded_random_seeds: prohibited

prospective_pilot_access_control:
  normal_benchmark_runner_loads_pilot_truth: false
  required_flag: --allow-seen-pilot-evaluation
  required_inputs:
    - selected candidate model ID
    - historical model-selection receipt
    - benchmark protocol hash
    - clean Git working tree
  receipt_fields:
    - timestamp
    - git_commit
    - candidate_model_id
    - historical_selection_metric
    - protocol_hash
    - registry_hash
    - pilot_sample_ids
    - command
    - environment_hash
  repeated_pilot_model_choice: prohibited

future_confirmatory_cohort:
  must_use_new_independent_growths: true
  must_cover_morphology_range: smooth, intermediate, rough
  model_frozen_before_afm: true
  samples_selected_before_truth_known: true
  append_only_registry_versioning: true
  benchmark_v1_overwrite: prohibited

leakage_rules:
  explicitly_prohibited:
    - frame-level random train/test splitting
    - scan-level AFM splitting across sample folds
    - multiple scans from the same sample appearing in train and test
    - using prospective truth for preprocessing
    - selecting representative samples by model error
    - tuning clipping thresholds on test data
    - fitting PCA or StandardScaler before the outer split
    - choosing the best frame count based on prospective results
    - using post-growth metadata as input
    - reporting training-fit performance as independent validation

determinism:
  python: set PYTHONHASHSEED when executing stochastic experiments
  numpy: use deterministic generators seeded from the derived seed
  scikit_learn: pass explicit random_state where available
  pytorch: set manual_seed and record deterministic backend settings
  cuda: record availability and settings; do not claim bitwise GPU determinism unless verified
"""
    (root / "configs/benchmark_v1").mkdir(parents=True, exist_ok=True)
    (root / "configs/benchmark_v1/protocol_v1.yaml").write_text(text, encoding="utf-8")


def write_split_artifacts(root: Path) -> None:
    registry_path = root / REGISTRY_DIR / "sample_registry_master_v1.csv"
    split_path = root / SPLIT_DIR / "historical_outer_logo_v1.json"
    split = generate_outer_logo_from_registry(registry_path)
    write_split(split_path, split)
    fingerprint = sha256_file(split_path)
    (root / SPLIT_DIR / "split_fingerprint_v1.sha256").write_text(
        f"{fingerprint}  configs/benchmark_v1/splits/historical_outer_logo_v1.json\n",
        encoding="utf-8",
    )


def write_experiment_matrix(root: Path) -> None:
    rows = [
        ("B00_fold_median", "fold_median", "Historical fold-median baseline.", "target_only_training_fold", "T4_second_order_trimmed_mean", "historical_nested_logo_v1", "MAE_nm", "confirmatory"),
        ("B01_metadata_only", "metadata_only", "How much pre/during-growth metadata explains roughness?", "allowed_metadata_only", "T4_second_order_trimmed_mean", "historical_nested_logo_v1", "MAE_nm", "exploratory"),
        ("B10_handcrafted_ridge", "handcrafted_rheed", "Ridge on handcrafted RHEED features.", "handcrafted_rheed_features", "T4_second_order_trimmed_mean", "historical_nested_logo_v1", "MAE_nm", "confirmatory"),
        ("B11_handcrafted_knn", "handcrafted_rheed", "KNN on handcrafted RHEED features.", "handcrafted_rheed_features", "T4_second_order_trimmed_mean", "historical_nested_logo_v1", "MAE_nm", "confirmatory"),
        ("B12_handcrafted_svr", "handcrafted_rheed", "SVR on handcrafted RHEED features.", "handcrafted_rheed_features", "T4_second_order_trimmed_mean", "historical_nested_logo_v1", "MAE_nm", "confirmatory"),
        ("B20_dino384_ridge", "dino384", "Ridge on DINO 384 embeddings.", "dino384_single_frame", "T4_second_order_trimmed_mean", "historical_nested_logo_v1", "MAE_nm", "confirmatory"),
        ("B21_dino384_knn", "dino384", "KNN on DINO 384 embeddings.", "dino384_single_frame", "T4_second_order_trimmed_mean", "historical_nested_logo_v1", "MAE_nm", "confirmatory"),
        ("B22_dino384_gpr", "dino384", "Gaussian Process Regression on DINO 384 embeddings.", "dino384_single_frame", "T4_second_order_trimmed_mean", "historical_nested_logo_v1", "MAE_nm", "confirmatory"),
        ("B23_dino384_svr", "dino384", "SVR on DINO 384 embeddings.", "dino384_single_frame", "T4_second_order_trimmed_mean", "historical_nested_logo_v1", "MAE_nm", "confirmatory"),
        ("B30_positive_target_variants", "target_transform", "Positive/log-Rq target variants with inverse-transform scoring.", "varies", "T4_second_order_trimmed_mean", "historical_nested_logo_v1", "MAE_nm", "exploratory"),
        ("B40_dino_dimensionality_reduction", "dimensionality_reduction", "DINO dimensionality-reduction variants.", "dino384_single_frame", "T4_second_order_trimmed_mean", "historical_nested_logo_v1", "MAE_nm", "exploratory"),
        ("B50_temporal_1_4_8_16_frames", "temporal", "Temporal frame count comparison.", "rheed_temporal", "T4_second_order_trimmed_mean", "historical_nested_logo_v1", "MAE_nm", "exploratory"),
        ("B60_afm_repeatability_and_label_uncertainty", "label_uncertainty", "AFM repeatability and target uncertainty.", "afm_scan_grouped", "T4_second_order_trimmed_mean", "historical_nested_logo_v1", "MAE_nm", "exploratory"),
        ("B70_conformal_prediction", "uncertainty", "Conformal prediction intervals.", "selected_candidate_features", "T4_second_order_trimmed_mean", "historical_nested_logo_v1", "MAE_nm", "exploratory"),
        ("B71_embedding_distance_ood", "ood", "Embedding-distance OOD detection.", "selected_embeddings", "T4_second_order_trimmed_mean", "historical_nested_logo_v1", "MAE_nm", "exploratory"),
        ("B80_occlusion_and_gradcam", "interpretability", "Occlusion and Grad-CAM diagnostics.", "selected_image_model", "T4_second_order_trimmed_mean", "historical_nested_logo_v1", "MAE_nm", "exploratory"),
        ("B81_physical_feature_correlation", "physical_correlation", "Physical RHEED feature correlations.", "handcrafted_rheed_features", "T4_second_order_trimmed_mean", "historical_nested_logo_v1", "MAE_nm", "exploratory"),
        ("B90_target_roughness_control_demo", "control_requirements", "Advisory target-roughness-control software requirements.", "allowed_control_metadata", "predicted_Rq_nm", "no_model_training_phase0", "not_applicable", "exploratory"),
    ]
    fieldnames = [
        "experiment_id",
        "family",
        "scientific_question",
        "input_modality",
        "target",
        "evaluation_protocol",
        "primary_metric",
        "status",
        "confirmatory_or_exploratory",
        "planned_output",
        "notes",
    ]
    out = root / "configs/benchmark_v1/experiment_matrix_v1.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "experiment_id": row[0],
                    "family": row[1],
                    "scientific_question": row[2],
                    "input_modality": row[3],
                    "target": row[4],
                    "evaluation_protocol": row[5],
                    "primary_metric": row[6],
                    "status": "PLANNED",
                    "confirmatory_or_exploratory": row[7],
                    "planned_output": f"outputs/benchmark_v1/runs/<run_id>/{row[0]}",
                    "notes": "No model run, prediction, metric value, or optimization performed in Phase 0.",
                }
            )


def write_schema_artifacts(root: Path) -> None:
    schemas = build_schemas()
    examples = build_examples()
    for name, schema in schemas.items():
        write_json(root / SCHEMA_DIR / f"{name}.schema.json", schema)
        write_json(root / SCHEMA_DIR / f"{name}.example.json", examples[name])


def sample_registry_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "benchmark_v1_sample_registry_row",
        "type": "object",
        "required": REGISTRY_COLUMNS,
        "additionalProperties": False,
        "properties": {column: {"type": "string"} for column in REGISTRY_COLUMNS},
        "cohort_role_enum": [
            "historical_development",
            "prospective_pilot_seen",
            "prospective_pending_truth",
            "afm_only_unmatched",
        ],
    }


def build_schemas() -> dict[str, dict[str, Any]]:
    return {
        "run_manifest": {
            "type": "object",
            "required": [
                "run_id",
                "experiment_id",
                "timestamp",
                "git_commit",
                "git_dirty",
                "protocol_version",
                "protocol_hash",
                "registry_hash",
                "split_hash",
                "config_hash",
                "environment_hash",
                "random_seed",
                "outer_fold_id",
                "inner_selection_method",
                "model_family",
                "feature_family",
                "target_transform",
                "sample_ids_train",
                "sample_ids_validation",
                "sample_ids_test",
                "input_source_hashes",
                "output_paths",
                "runtime_seconds",
                "status",
                "failure_reason",
            ],
            "properties": {
                "run_id": {"type": "string"},
                "experiment_id": {"type": "string"},
                "timestamp": {"type": "string"},
                "git_commit": {"type": "string"},
                "git_dirty": {"type": "boolean"},
                "protocol_version": {"type": "string"},
                "protocol_hash": {"type": "string"},
                "registry_hash": {"type": "string"},
                "split_hash": {"type": "string"},
                "config_hash": {"type": "string"},
                "environment_hash": {"type": "string"},
                "random_seed": {"type": "integer"},
                "outer_fold_id": {"type": "string"},
                "inner_selection_method": {"type": "string"},
                "model_family": {"type": "string"},
                "feature_family": {"type": "string"},
                "target_transform": {"type": "string"},
                "sample_ids_train": {"type": "array"},
                "sample_ids_validation": {"type": "array"},
                "sample_ids_test": {"type": "array"},
                "input_source_hashes": {"type": "object"},
                "output_paths": {"type": "array"},
                "runtime_seconds": {"type": "number"},
                "status": {"type": "string"},
                "failure_reason": {"type": "string"},
            },
        },
        "fold_prediction": {
            "type": "object",
            "required": [
                "run_id",
                "outer_fold_id",
                "sample_id",
                "target_rq_nm",
                "prediction_rq_nm_raw",
                "prediction_rq_nm_for_metrics",
                "prediction_rq_nm_clipped_display",
                "display_clip_applied",
            ],
            "properties": {
                "run_id": {"type": "string"},
                "outer_fold_id": {"type": "string"},
                "sample_id": {"type": "string"},
                "target_rq_nm": {"type": "number"},
                "prediction_rq_nm_raw": {"type": "number"},
                "prediction_rq_nm_for_metrics": {"type": "number"},
                "prediction_rq_nm_clipped_display": {"type": "number"},
                "display_clip_applied": {"type": "boolean"},
            },
        },
        "metrics": {
            "type": "object",
            "required": ["run_id", "scope", "MAE_nm", "RMSE_nm", "R2", "n_samples"],
            "properties": {
                "run_id": {"type": "string"},
                "scope": {"type": "string"},
                "MAE_nm": {"type": "number"},
                "RMSE_nm": {"type": "number"},
                "R2": {"type": "number"},
                "n_samples": {"type": "integer"},
            },
        },
        "model_selection_receipt": {
            "type": "object",
            "required": [
                "timestamp",
                "candidate_model_id",
                "historical_selection_metric",
                "historical_selection_metric_name",
                "protocol_hash",
                "registry_hash",
                "split_hash",
                "selected_without_pilot_truth",
            ],
            "properties": {
                "timestamp": {"type": "string"},
                "candidate_model_id": {"type": "string"},
                "historical_selection_metric": {"type": "number"},
                "historical_selection_metric_name": {"type": "string"},
                "protocol_hash": {"type": "string"},
                "registry_hash": {"type": "string"},
                "split_hash": {"type": "string"},
                "selected_without_pilot_truth": {"type": "boolean"},
            },
        },
        "pilot_evaluation_receipt": {
            "type": "object",
            "required": [
                "timestamp",
                "git_commit",
                "candidate_model_id",
                "historical_selection_metric",
                "protocol_hash",
                "registry_hash",
                "pilot_sample_ids",
                "command",
                "environment_hash",
            ],
            "properties": {
                "timestamp": {"type": "string"},
                "git_commit": {"type": "string"},
                "candidate_model_id": {"type": "string"},
                "historical_selection_metric": {"type": "number"},
                "protocol_hash": {"type": "string"},
                "registry_hash": {"type": "string"},
                "pilot_sample_ids": {"type": "array"},
                "command": {"type": "string"},
                "environment_hash": {"type": "string"},
            },
        },
    }


def build_examples() -> dict[str, dict[str, Any]]:
    return {
        "run_manifest": {
            "run_id": "run_example",
            "experiment_id": "B00_fold_median",
            "timestamp": "2026-07-23T00:00:00Z",
            "git_commit": "0a2414a06c4155123dc61cd8a95ded638fb725dc",
            "git_dirty": False,
            "protocol_version": "benchmark_v1",
            "protocol_hash": "sha256-example",
            "registry_hash": "sha256-example",
            "split_hash": "sha256-example",
            "config_hash": "sha256-example",
            "environment_hash": "sha256-example",
            "random_seed": BASE_SEED,
            "outer_fold_id": "outer_logo_6022",
            "inner_selection_method": "historical_inner_logo_mean_MAE_nm",
            "model_family": "fold_median",
            "feature_family": "none",
            "target_transform": "none",
            "sample_ids_train": ["6028"],
            "sample_ids_validation": [],
            "sample_ids_test": ["6022"],
            "input_source_hashes": {},
            "output_paths": ["outputs/benchmark_v1/runs/run_example"],
            "runtime_seconds": 0.0,
            "status": "DRY_RUN",
            "failure_reason": "",
        },
        "fold_prediction": {
            "run_id": "run_example",
            "outer_fold_id": "outer_logo_6022",
            "sample_id": "6022",
            "target_rq_nm": 1.4471265822649002,
            "prediction_rq_nm_raw": -0.5,
            "prediction_rq_nm_for_metrics": -0.5,
            "prediction_rq_nm_clipped_display": 0.0,
            "display_clip_applied": True,
        },
        "metrics": {
            "run_id": "run_example",
            "scope": "historical_outer_logo",
            "MAE_nm": 0.0,
            "RMSE_nm": 0.0,
            "R2": 0.0,
            "n_samples": 23,
        },
        "model_selection_receipt": {
            "timestamp": "2026-07-23T00:00:00Z",
            "candidate_model_id": "example_candidate",
            "historical_selection_metric": 0.0,
            "historical_selection_metric_name": "mean_inner_MAE_nm",
            "protocol_hash": "sha256-example",
            "registry_hash": "sha256-example",
            "split_hash": "sha256-example",
            "selected_without_pilot_truth": True,
        },
        "pilot_evaluation_receipt": {
            "timestamp": "2026-07-23T00:00:00Z",
            "git_commit": "0a2414a06c4155123dc61cd8a95ded638fb725dc",
            "candidate_model_id": "example_candidate",
            "historical_selection_metric": 0.0,
            "protocol_hash": "sha256-example",
            "registry_hash": "sha256-example",
            "pilot_sample_ids": PROSPECTIVE_PILOT_IDS,
            "command": "benchmark_dry_run.py --allow-seen-pilot-evaluation ...",
            "environment_hash": "sha256-example",
        },
    }


def write_docs(root: Path, summary: dict[str, Any]) -> None:
    docs = {
        "benchmark_v1_protocol.md": protocol_doc(summary),
        "benchmark_v1_data_registry.md": data_registry_doc(summary),
        "benchmark_v1_experiment_tracking.md": tracking_doc(),
        "benchmark_v1_claim_boundaries.md": claim_boundaries_doc(),
        "target_roughness_control_requirements.md": control_requirements_doc(),
    }
    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name, text in docs.items():
        (docs_dir / name).write_text(text, encoding="utf-8")


def protocol_doc(summary: dict[str, Any]) -> str:
    return f"""# Benchmark v1 Protocol

Benchmark v1 freezes the scientific evaluation rules before any new computational
experiments are run. Model selection is restricted to the 23 historical growth
groups because those are the only paired samples whose target definition is the
historical primary target, `T4_second_order_trimmed_mean`.

The four current prospective matched samples, `{", ".join(PROSPECTIVE_PILOT_IDS)}`,
are labeled `prospective_pilot_seen`. Their AFM truth has already influenced
discussion, so they are not a blind confirmatory set. They may be used only after
a candidate is selected with historical data alone, and only with pilot labeling.

Nested leave-one-growth-group-out uses 23 outer folds. Each fold holds out one
historical growth group and trains on the other 22. Hyperparameters, scaling,
PCA, feature selection, imputation, calibration, target-transform fitting,
conformal calibration, OOD thresholds, ensemble choices, and temporal frame-count
tuning must occur inside the outer-training data. The inner loop is another
leave-one-growth-group-out over those 22 training groups.

Preprocessing must occur inside folds because any statistic fit before the split
can encode the held-out growth group. That includes StandardScaler, PCA, feature
selection, metadata imputation, calibration, target transformations, and any OOD
thresholds.

Future confirmatory samples must be newly grown, selected before AFM truth is
known, and appended in a new registry version. Benchmark v1 is never overwritten.

All model families compare on the same sample IDs, target definition, split file,
metric names, strata boundaries, and run-recording schema. The fixed historical
strata thresholds are {summary["fixed_historical_strata_quantiles"]["low_max_nm"]:.6g}
nm and {summary["fixed_historical_strata_quantiles"]["mid_max_nm"]:.6g} nm.
"""


def data_registry_doc(summary: dict[str, Any]) -> str:
    return f"""# Benchmark v1 Data Registry

The master registry has {summary["master_registry_rows"]} records:
23 `historical_development`, 4 `prospective_pilot_seen`, 1
`prospective_pending_truth`, and 1 `afm_only_unmatched`.

The paired supervised cohort has {summary["paired_supervised_rows"]} samples:
the 23 historical samples plus the four seen prospective pilot samples. The
historical development cohort is the only cohort eligible for model development
and primary nested CV.

Historical targets come from the frozen target table and use
`T4_second_order_trimmed_mean` in nm. Prospective pilot AFM truth comes from the
frozen prospective AFM table as `true_rq_nm_median_second_order`; this is not the
same aggregation as T4, so pilot aggregate metrics are exploratory and not
directly confirmatory.

`N6390` remains present as `prospective_pending_truth`: it has RHEED and a frozen
legacy prediction record but no AFM truth, and it is excluded from all metrics.

`N6324` remains present as `afm_only_unmatched`: it has AFM truth but no matched
RHEED prediction sample in the frozen prospective prediction package, and it is
excluded from supervised metrics.

The canonical removelist is retained at
`{summary["canonical_removelist_path"]}` with hash
`{summary["canonical_removelist_sha256"]}`. Removed samples are not silently
repaired or reintroduced into the historical development cohort.

AFM scans are grouped by sample. No AFM scan is treated as an independent growth
sample. RHEED grouping is by growth run, and each historical growth run appears
once as an outer held-out group.
"""


def tracking_doc() -> str:
    return """# Benchmark v1 Experiment Tracking

Future runs write full outputs under `outputs/benchmark_v1/runs/<run_id>/` and
compact finalized records under `reports/benchmark_v1/run_records/`.

Run IDs are deterministic from the experiment ID, config hash, Git commit,
protocol hash, and split hash. Existing run directories must not be overwritten;
a dry-run fails if the intended run ID already exists.

Every run manifest records protocol, registry, split, config, environment, input
source hashes, train/validation/test sample IDs, runtime, status, and failure
reason. Failed runs still write a manifest so missing outputs are auditable.

Protocol hashes identify the frozen scientific rules. Registry hashes identify
the sample table. Config hashes identify model-side settings. Environment hashes
identify the Python package state and hardware context.
"""


def claim_boundaries_doc() -> str:
    return """# Benchmark v1 Claim Boundaries

Currently supported:

- strict retrospective evidence on 23 historical growth groups;
- frozen legacy prospective pilot evidence;
- current known failure modes.

Not yet supported:

- blind prospective validation of a new model;
- robust target roughness control;
- reliable high-Rq extrapolation;
- replacement of AFM;
- exact AFM image reconstruction;
- cross-material generalization;
- production deployment.
"""


def control_requirements_doc() -> str:
    return """# Target Roughness Control Requirements

No controller is implemented in Phase 0.

No structured, software-controllable growth variables were discovered in the
machine-readable benchmark metadata. Filename text contains ambiguous process
tokens such as ramp-down temperature labels and GaSb duration labels, but these
are not curated variables and are not allowed as model inputs.

Candidate variables requiring expert curation before any control software:

- ramp-down temperature: unit appears to be C in some filenames; no trusted
  structured range is available;
- deposition or process-stage duration: unit appears to be minutes in some
  filenames; no trusted structured range is available;
- material/process stage labels such as GaSb or AlSb: categorical context only
  until curated.

Future control software must define which variables are available to software,
operator-approved units, observed historical ranges, safety limits, and any hard
constraints. Expert approval is required before using those variables for
recommendations.

The future interface must accept a target Rq in nm, output predicted Rq in nm,
provide recommended actions only in advisory mode by default, require human
confirmation, reject out-of-distribution states, record uncertainty, and log the
full experiment context. Autonomous mode requires a separate safety review.
"""


def write_phase0_report(root: Path, report_timestamp: str, summary: dict[str, Any]) -> None:
    report_dir = root / "reports/benchmark_phase0" / report_timestamp
    report_dir.mkdir(parents=True, exist_ok=True)
    report = f"""# Benchmark Phase 0 Report

Status: artifacts generated; final validation results are recorded separately in
`phase0_summary.json` and validation command logs.

Master registry rows: {summary["master_registry_rows"]}
Paired supervised rows: {summary["paired_supervised_rows"]}
Historical development rows: {summary["historical_development_rows"]}
Prospective pilot rows: {summary["prospective_pilot_seen_rows"]}
Pending truth rows: {summary["prospective_pending_truth_rows"]}
AFM-only unmatched rows: {summary["afm_only_unmatched_rows"]}

Primary target: `T4_second_order_trimmed_mean` in nm for historical development.
Prospective pilot target available: `true_rq_nm_median_second_order`, recorded as
not directly comparable to T4.
"""
    (report_dir / "phase0_report.md").write_text(report, encoding="utf-8")
    write_json(report_dir / "phase0_summary.json", summary)


if __name__ == "__main__":
    main()
