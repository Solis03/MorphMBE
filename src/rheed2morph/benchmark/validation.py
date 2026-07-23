"""Scientific guard validation for benchmark v1 artifacts."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .constants import (
    AFM_ONLY_UNMATCHED_ID,
    HISTORICAL_REMOVELIST,
    HISTORICAL_SAMPLE_IDS,
    PROSPECTIVE_PENDING_TRUTH_ID,
    PROSPECTIVE_PILOT_IDS,
)
from .hashing import read_csv_rows, sha256_file
from .protocol import validate_protocol_text
from .schemas import schema_errors
from .splits import load_split


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def validate_registry_files(root: Path) -> list[str]:
    errors: list[str] = []
    registry_path = root / "configs/benchmark_v1/registry/sample_registry_master_v1.csv"
    paired_path = root / "configs/benchmark_v1/registry/paired_supervised_cohort_v1.csv"
    historical_path = root / "configs/benchmark_v1/registry/historical_development_cohort_v1.csv"
    pilot_path = root / "configs/benchmark_v1/registry/prospective_pilot_seen_v1.csv"
    unmatched_path = root / "configs/benchmark_v1/registry/unmatched_and_pending_v1.csv"
    registry = read_csv_rows(registry_path)
    paired = read_csv_rows(paired_path)
    historical = read_csv_rows(historical_path)
    pilot = read_csv_rows(pilot_path)
    unmatched = read_csv_rows(unmatched_path)
    if len(registry) != 29:
        errors.append(f"master registry row count {len(registry)} != 29")
    if len(paired) != 27:
        errors.append(f"paired cohort row count {len(paired)} != 27")
    if len(historical) != 23:
        errors.append(f"historical cohort row count {len(historical)} != 23")
    if [row["sample_id"] for row in historical] != HISTORICAL_SAMPLE_IDS:
        errors.append("historical sample IDs do not match canonical order")
    if [row["sample_id"] for row in pilot] != PROSPECTIVE_PILOT_IDS:
        errors.append("prospective pilot IDs do not match canonical set")
    if [row["sample_id"] for row in unmatched] != [PROSPECTIVE_PENDING_TRUTH_ID, AFM_ONLY_UNMATCHED_ID]:
        errors.append("unmatched/pending IDs do not match canonical order")
    ids = [row["sample_id"] for row in registry]
    duplicates = [sample_id for sample_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate sample IDs: {duplicates}")
    historical_groups = [row["growth_group_id"] for row in historical]
    duplicate_groups = [group for group, count in Counter(historical_groups).items() if count > 1]
    if duplicate_groups:
        errors.append(f"duplicate historical growth groups: {duplicate_groups}")
    for row in historical:
        try:
            target = float(row["target_rq_nm"])
        except ValueError:
            errors.append(f"{row['sample_id']}: nonnumeric target")
            continue
        if not math.isfinite(target) or target <= 0:
            errors.append(f"{row['sample_id']}: target is not finite positive")
    by_id = {row["sample_id"]: row for row in registry}
    n6390 = by_id.get(PROSPECTIVE_PENDING_TRUTH_ID)
    if not n6390:
        errors.append("N6390 missing")
    else:
        expected = {
            "has_rheed": True,
            "has_afm": False,
            "has_target": False,
            "cohort_role": "prospective_pending_truth",
        }
        errors.extend(_expect_flags(n6390, expected, PROSPECTIVE_PENDING_TRUTH_ID))
    n6324 = by_id.get(AFM_ONLY_UNMATCHED_ID)
    if not n6324:
        errors.append("N6324 missing")
    else:
        expected = {
            "has_rheed": False,
            "has_afm": True,
            "cohort_role": "afm_only_unmatched",
        }
        errors.extend(_expect_flags(n6324, expected, AFM_ONLY_UNMATCHED_ID))
    for row in registry:
        if row["cohort_role"] == "historical_development":
            errors.extend(
                _expect_flags(
                    row,
                    {
                        "eligible_for_model_development": True,
                        "eligible_for_primary_nested_cv": True,
                        "eligible_for_pilot_evaluation": False,
                        "eligible_for_confirmatory_test": False,
                    },
                    row["sample_id"],
                )
            )
        if row["cohort_role"] == "prospective_pilot_seen":
            errors.extend(
                _expect_flags(
                    row,
                    {
                        "eligible_for_model_development": False,
                        "eligible_for_primary_nested_cv": False,
                        "eligible_for_pilot_evaluation": True,
                        "eligible_for_confirmatory_test": False,
                    },
                    row["sample_id"],
                )
            )
        if row["cohort_role"] in {"prospective_pending_truth", "afm_only_unmatched"}:
            errors.extend(
                _expect_flags(
                    row,
                    {
                        "eligible_for_model_development": False,
                        "eligible_for_primary_nested_cv": False,
                        "eligible_for_pilot_evaluation": False,
                        "eligible_for_confirmatory_test": False,
                        "eligible_for_final_training": False,
                    },
                    row["sample_id"],
                )
            )
    summary = json.loads((root / "configs/benchmark_v1/registry/cohort_summary_v1.json").read_text(encoding="utf-8"))
    if summary.get("canonical_removelist_path") != HISTORICAL_REMOVELIST.as_posix():
        errors.append("canonical removelist is not referenced in cohort summary")
    return errors


def validate_split_file(root: Path) -> list[str]:
    errors: list[str] = []
    split = load_split(root / "configs/benchmark_v1/splits/historical_outer_logo_v1.json")
    folds = split.get("folds", [])
    if len(folds) != 23:
        errors.append(f"outer fold count {len(folds)} != 23")
    test_seen: list[str] = []
    for fold in folds:
        train = set(fold["train_sample_ids"])
        test = set(fold["test_sample_ids"])
        if train & test:
            errors.append(f"{fold['fold_id']}: train/test overlap")
        if len(test) != 1:
            errors.append(f"{fold['fold_id']}: test size is not one")
        test_seen.extend(fold["test_sample_ids"])
        if any(sample.startswith("N") for sample in train | test):
            errors.append(f"{fold['fold_id']}: prospective sample appears in historical CV")
        outer_group = fold["test_growth_group_id"]
        for inner in fold["inner_model_selection"]["folds"]:
            if outer_group in inner["train_growth_group_ids"] or outer_group == inner["validation_growth_group_id"]:
                errors.append(f"{fold['fold_id']}/{inner['inner_fold_id']}: outer-held-out group in inner split")
    if sorted(test_seen, key=HISTORICAL_SAMPLE_IDS.index) != HISTORICAL_SAMPLE_IDS:
        errors.append("historical samples do not each appear exactly once as outer test")
    recorded = (root / "configs/benchmark_v1/splits/split_fingerprint_v1.sha256").read_text(encoding="utf-8").strip()
    actual = sha256_file(root / "configs/benchmark_v1/splits/historical_outer_logo_v1.json")
    if recorded != f"{actual}  configs/benchmark_v1/splits/historical_outer_logo_v1.json":
        errors.append("split fingerprint does not match split file")
    return errors


def validate_metadata_inventory(root: Path) -> list[str]:
    errors: list[str] = []
    rows = read_csv_rows(root / "configs/benchmark_v1/registry/metadata_feature_inventory_v1.csv")
    for row in rows:
        if row["control_role"] == "post_growth_outcome" and parse_bool(row["allowed_metadata_only_baseline"]):
            errors.append(f"post-growth field allowed as metadata feature: {row['feature_name']}")
    return errors


def validate_protocol_file(root: Path) -> list[str]:
    return validate_protocol_text(root / "configs/benchmark_v1/protocol_v1.yaml")


def validate_schema_examples(root: Path) -> list[str]:
    errors: list[str] = []
    schema_dir = root / "configs/benchmark_v1/schemas"
    for schema_path in sorted(schema_dir.glob("*.schema.json")):
        example_path = schema_path.with_name(schema_path.name.replace(".schema.json", ".example.json"))
        if not example_path.exists():
            errors.append(f"missing schema example: {example_path.name}")
            continue
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        payload = json.loads(example_path.read_text(encoding="utf-8"))
        errors.extend([f"{example_path.name}: {err}" for err in schema_errors(schema, payload)])
    return errors


def validate_all(root: Path) -> list[str]:
    errors = []
    errors.extend(validate_registry_files(root))
    errors.extend(validate_split_file(root))
    errors.extend(validate_metadata_inventory(root))
    errors.extend(validate_protocol_file(root))
    errors.extend(validate_schema_examples(root))
    return errors


def _expect_flags(row: dict[str, str], expected: dict[str, object], label: str) -> list[str]:
    errors = []
    for key, expected_value in expected.items():
        observed = row.get(key)
        if isinstance(expected_value, bool):
            if parse_bool(str(observed)) != expected_value:
                errors.append(f"{label}: {key} is {observed}, expected {expected_value}")
        elif observed != expected_value:
            errors.append(f"{label}: {key} is {observed}, expected {expected_value}")
    return errors

