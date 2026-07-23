"""Group-aware split generation for benchmark v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .constants import HISTORICAL_SAMPLE_IDS, PROTOCOL_VERSION
from .hashing import read_csv_rows, sha256_file, write_json


def generate_outer_logo_from_registry(registry_path: Path) -> dict[str, Any]:
    rows = read_csv_rows(registry_path)
    historical = [row for row in rows if row["cohort_role"] == "historical_development"]
    historical = sorted(historical, key=lambda row: HISTORICAL_SAMPLE_IDS.index(row["sample_id"]))
    sample_ids = [row["sample_id"] for row in historical]
    group_ids = [row["growth_group_id"] for row in historical]
    folds: list[dict[str, Any]] = []
    for idx, row in enumerate(historical, start=1):
        test_sample = row["sample_id"]
        test_group = row["growth_group_id"]
        train_rows = [other for other in historical if other["growth_group_id"] != test_group]
        inner = generate_inner_logo(train_rows)
        folds.append(
            {
                "fold_id": f"outer_logo_{test_group}",
                "outer_fold_index": idx,
                "test_growth_group_id": test_group,
                "test_sample_ids": [test_sample],
                "train_growth_group_ids": [item["growth_group_id"] for item in train_rows],
                "train_sample_ids": [item["sample_id"] for item in train_rows],
                "inner_model_selection": {
                    "method": "Leave-One-Growth-Group-Out",
                    "fold_count": len(inner),
                    "folds": inner,
                },
            }
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "split_id": "historical_outer_logo_v1",
        "method": "Leave-One-Growth-Group-Out",
        "source_registry_path": registry_path.as_posix(),
        "historical_sample_ids": sample_ids,
        "historical_growth_group_ids": group_ids,
        "prospective_samples_included": [],
        "outer_fold_count": len(folds),
        "folds": folds,
    }


def generate_inner_logo(train_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    for idx, row in enumerate(train_rows, start=1):
        val_group = row["growth_group_id"]
        inner_train = [other for other in train_rows if other["growth_group_id"] != val_group]
        folds.append(
            {
                "inner_fold_id": f"inner_logo_{val_group}",
                "inner_fold_index": idx,
                "validation_growth_group_id": val_group,
                "validation_sample_ids": [row["sample_id"]],
                "train_growth_group_ids": [item["growth_group_id"] for item in inner_train],
                "train_sample_ids": [item["sample_id"] for item in inner_train],
            }
        )
    return folds


def write_split(path: Path, split: dict[str, Any]) -> None:
    write_json(path, split)


def split_fingerprint(split_path: Path) -> str:
    return sha256_file(split_path)


def load_split(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

