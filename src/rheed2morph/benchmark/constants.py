"""Constants for the benchmark v1 protocol freeze."""

from __future__ import annotations

from pathlib import Path


BASELINE_TAG = "natcomms-publication-baseline-v0-20260723-000924"
BASELINE_COMMIT = "0a2414a06c4155123dc61cd8a95ded638fb725dc"
PROTOCOL_VERSION = "benchmark_v1"
BASE_SEED = 20260723

HISTORICAL_SAMPLE_IDS = [
    "6022",
    "6028",
    "6029",
    "6033",
    "6047",
    "6048",
    "6056",
    "6057",
    "6062",
    "6063",
    "6070",
    "6072",
    "6078",
    "6080",
    "6081",
    "6082",
    "6084",
    "6085",
    "6090",
    "6094",
    "6095",
    "6099",
    "6101",
]
PROSPECTIVE_PILOT_IDS = ["N6342", "N6358", "N6382", "N6389"]
PROSPECTIVE_PENDING_TRUTH_ID = "N6390"
AFM_ONLY_UNMATCHED_ID = "N6324"
UNMATCHED_AND_PENDING_IDS = [PROSPECTIVE_PENDING_TRUTH_ID, AFM_ONLY_UNMATCHED_ID]

RETROSPECTIVE_FREEZE = Path("publication_freeze/rheed_afm_single_frame_v1_2026-07-18")
PROSPECTIVE_FREEZE = Path("publication_freeze/prospective_unseen_single_frame_v1")

HISTORICAL_INDEX = RETROSPECTIVE_FREEZE / "data_snapshot/canonical_sample_index.csv"
HISTORICAL_INDEX_FULL = RETROSPECTIVE_FREEZE / "data_snapshot/canonical_sample_index_full.csv"
HISTORICAL_TARGETS = RETROSPECTIVE_FREEZE / "data_snapshot/sample_targets.csv"
HISTORICAL_REMOVELIST = RETROSPECTIVE_FREEZE / "data_snapshot/removelist.txt"
PROSPECTIVE_PREDICTIONS = PROSPECTIVE_FREEZE / "predictions/full_cohort_single_frame_v1/predictions.csv"
PROSPECTIVE_KEYFRAMES = PROSPECTIVE_FREEZE / "manifests/unseen_keyframe_manifest.csv"
PROSPECTIVE_TRUTH = PROSPECTIVE_FREEZE / "ground_truth_afm/manifests/afm_extra_five_sample_level_ground_truth.csv"
PROSPECTIVE_SCAN_MANIFEST = PROSPECTIVE_FREEZE / "ground_truth_afm/manifests/afm_extra_five_second_order_scan_manifest.csv"
PROSPECTIVE_JOIN = PROSPECTIVE_FREEZE / "ground_truth_afm/manifests/full_cohort_prediction_vs_afm_truth_join.csv"
PROSPECTIVE_MISMATCH = PROSPECTIVE_FREEZE / "ground_truth_afm/manifests/sample_id_mismatch_report.json"

REGISTRY_COLUMNS = [
    "sample_id",
    "growth_group_id",
    "cohort_role",
    "paired_status",
    "has_rheed",
    "has_afm",
    "has_target",
    "target_rq_nm",
    "target_definition",
    "target_source_path",
    "target_source_row",
    "rheed_source_path",
    "afm_source_path",
    "afm_scan_count",
    "historical_or_prospective",
    "truth_visibility",
    "eligible_for_model_development",
    "eligible_for_primary_nested_cv",
    "eligible_for_pilot_evaluation",
    "eligible_for_confirmatory_test",
    "eligible_for_final_training",
    "exclusion_reason",
    "sample_id_status",
    "source_manifest_hash",
    "notes",
]

METADATA_INVENTORY_COLUMNS = [
    "feature_name",
    "source_path",
    "source_column",
    "dtype",
    "unit",
    "sample_coverage_count",
    "missing_count",
    "missing_fraction",
    "constant_or_variable",
    "available_before_growth",
    "available_during_growth",
    "available_after_growth",
    "controllable_by_operator",
    "control_role",
    "leakage_risk",
    "allowed_metadata_only_baseline",
    "allowed_target_control_model",
    "notes",
]

