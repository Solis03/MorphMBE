from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from rheed2morph.benchmark.constants import HISTORICAL_SAMPLE_IDS, PROSPECTIVE_PILOT_IDS
from rheed2morph.benchmark.hashing import canonical_json, read_csv_rows, sha256_file
from rheed2morph.benchmark.schemas import schema_errors
from rheed2morph.benchmark.splits import generate_outer_logo_from_registry
from rheed2morph.benchmark.validation import parse_bool, validate_schema_examples


REPO = Path(__file__).resolve().parents[2]
REGISTRY_DIR = REPO / "configs/benchmark_v1/registry"
SPLIT_DIR = REPO / "configs/benchmark_v1/splits"
SCHEMA_DIR = REPO / "configs/benchmark_v1/schemas"
REPORT_DIR = REPO / "reports/benchmark_phase0/20260723-110844"


@pytest.fixture(scope="module")
def registry() -> list[dict[str, str]]:
    return read_csv_rows(REGISTRY_DIR / "sample_registry_master_v1.csv")


@pytest.fixture(scope="module")
def split() -> dict:
    return json.loads((SPLIT_DIR / "historical_outer_logo_v1.json").read_text(encoding="utf-8"))


def test_master_registry_has_29_rows(registry: list[dict[str, str]]) -> None:
    assert len(registry) == 29


def test_paired_cohort_has_27_rows() -> None:
    assert len(read_csv_rows(REGISTRY_DIR / "paired_supervised_cohort_v1.csv")) == 27


def test_historical_development_cohort_has_23_rows() -> None:
    rows = read_csv_rows(REGISTRY_DIR / "historical_development_cohort_v1.csv")
    assert len(rows) == 23
    assert [row["sample_id"] for row in rows] == HISTORICAL_SAMPLE_IDS


def test_prospective_pilot_exact_ids() -> None:
    rows = read_csv_rows(REGISTRY_DIR / "prospective_pilot_seen_v1.csv")
    assert [row["sample_id"] for row in rows] == PROSPECTIVE_PILOT_IDS


def test_n6390_status(registry: list[dict[str, str]]) -> None:
    row = {item["sample_id"]: item for item in registry}["N6390"]
    assert parse_bool(row["has_rheed"]) is True
    assert parse_bool(row["has_afm"]) is False
    assert parse_bool(row["has_target"]) is False
    assert row["cohort_role"] == "prospective_pending_truth"


def test_n6324_status(registry: list[dict[str, str]]) -> None:
    row = {item["sample_id"]: item for item in registry}["N6324"]
    assert parse_bool(row["has_rheed"]) is False
    assert parse_bool(row["has_afm"]) is True
    assert row["cohort_role"] == "afm_only_unmatched"


def test_historical_targets_are_finite_positive(registry: list[dict[str, str]]) -> None:
    for row in registry:
        if row["cohort_role"] == "historical_development":
            assert float(row["target_rq_nm"]) > 0.0


def test_no_duplicate_sample_ids(registry: list[dict[str, str]]) -> None:
    counts = Counter(row["sample_id"] for row in registry)
    assert [sample_id for sample_id, count in counts.items() if count > 1] == []


def test_no_duplicate_historical_growth_groups(registry: list[dict[str, str]]) -> None:
    groups = [row["growth_group_id"] for row in registry if row["cohort_role"] == "historical_development"]
    counts = Counter(groups)
    assert [group for group, count in counts.items() if count > 1] == []


def test_exactly_23_outer_folds(split: dict) -> None:
    assert split["outer_fold_count"] == 23
    assert len(split["folds"]) == 23


def test_every_historical_sample_once_as_outer_test(split: dict) -> None:
    seen = [fold["test_sample_ids"][0] for fold in split["folds"]]
    assert seen == HISTORICAL_SAMPLE_IDS


def test_outer_test_absent_from_outer_training(split: dict) -> None:
    for fold in split["folds"]:
        assert set(fold["test_sample_ids"]).isdisjoint(fold["train_sample_ids"])


def test_no_prospective_sample_in_historical_cv(split: dict) -> None:
    for fold in split["folds"]:
        all_samples = set(fold["test_sample_ids"]) | set(fold["train_sample_ids"])
        assert all(not sample.startswith("N") for sample in all_samples)


def test_inner_folds_never_contain_outer_heldout_group(split: dict) -> None:
    for fold in split["folds"]:
        outer_group = fold["test_growth_group_id"]
        for inner in fold["inner_model_selection"]["folds"]:
            assert inner["validation_growth_group_id"] != outer_group
            assert outer_group not in inner["train_growth_group_ids"]


def test_split_generation_is_deterministic() -> None:
    registry_path = REGISTRY_DIR / "sample_registry_master_v1.csv"
    first = generate_outer_logo_from_registry(registry_path)
    second = generate_outer_logo_from_registry(registry_path)
    assert canonical_json(first) == canonical_json(second)


def test_registry_hashes_are_deterministic() -> None:
    path = REGISTRY_DIR / "sample_registry_master_v1.csv"
    assert sha256_file(path) == sha256_file(path)


def test_protocol_hashes_are_deterministic() -> None:
    path = REPO / "configs/benchmark_v1/protocol_v1.yaml"
    assert sha256_file(path) == sha256_file(path)


def test_default_dry_run_refuses_pilot_truth_access() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/benchmark_v1/benchmark_dry_run.py"),
            "--attempt-pilot-truth-access",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "refused_pilot_truth_access" in result.stdout


def test_pilot_truth_access_requires_flag_and_receipt_inputs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/benchmark_v1/benchmark_dry_run.py"),
            "--attempt-pilot-truth-access",
            "--allow-seen-pilot-evaluation",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert any("--candidate-model-id" in err for err in payload["errors"])
    assert any("--historical-selection-receipt" in err for err in payload["errors"])


def test_no_immutable_publication_file_changed() -> None:
    before = (REPORT_DIR / "canonical_hashes_before.sha256").read_text(encoding="utf-8").splitlines()
    paths = []
    for package in [
        REPO / "publication_freeze/rheed_afm_single_frame_v1_2026-07-18",
        REPO / "publication_freeze/prospective_unseen_single_frame_v1",
    ]:
        paths.extend(p for p in package.rglob("*") if p.is_file())
    current_lines = [
        f"{sha256_file(path)}  {path.relative_to(REPO).as_posix()}"
        for path in sorted(paths, key=lambda item: item.relative_to(REPO).as_posix())
    ]
    assert current_lines == before


def test_canonical_removelist_referenced_and_retained() -> None:
    summary = json.loads((REGISTRY_DIR / "cohort_summary_v1.json").read_text(encoding="utf-8"))
    removelist = REPO / summary["canonical_removelist_path"]
    assert removelist.exists()
    assert sha256_file(removelist) == summary["canonical_removelist_sha256"]


def test_no_afm_scan_treated_as_independent_growth_sample(registry: list[dict[str, str]]) -> None:
    historical = [row for row in registry if row["cohort_role"] == "historical_development"]
    assert len(historical) == len({row["growth_group_id"] for row in historical})
    assert any(int(row["afm_scan_count"]) > 1 for row in historical)


def test_no_post_growth_field_allowed_as_predictive_metadata() -> None:
    rows = read_csv_rows(REGISTRY_DIR / "metadata_feature_inventory_v1.csv")
    leaking = [
        row["feature_name"]
        for row in rows
        if row["control_role"] == "post_growth_outcome" and parse_bool(row["allowed_metadata_only_baseline"])
    ]
    assert leaking == []


def test_raw_negative_prediction_not_replaced_by_clipped_value() -> None:
    example = json.loads((SCHEMA_DIR / "fold_prediction.example.json").read_text(encoding="utf-8"))
    assert example["prediction_rq_nm_raw"] < 0
    assert example["prediction_rq_nm_for_metrics"] == example["prediction_rq_nm_raw"]
    assert example["prediction_rq_nm_clipped_display"] == 0.0


def test_all_schema_examples_validate() -> None:
    assert validate_schema_examples(REPO) == []
    for schema_path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        example_path = schema_path.with_name(schema_path.name.replace(".schema.json", ".example.json"))
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        example = json.loads(example_path.read_text(encoding="utf-8"))
        assert schema_errors(schema, example) == []
