"""Experiment tracking helpers for dry-runs and future benchmark runs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .hashing import sha256_text


def git_commit(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def git_dirty(root: Path) -> bool:
    result = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return bool(result.stdout.strip())


def deterministic_run_id(
    experiment_id: str,
    config_hash: str,
    commit: str,
    protocol_hash: str,
    split_hash: str,
) -> str:
    payload = "|".join([experiment_id, config_hash, commit, protocol_hash, split_hash])
    return f"run_{sha256_text(payload)[:24]}"


def derived_seed(base_seed: int, experiment_id: str, outer_fold_id: str, config_id: str) -> int:
    digest = sha256_text(f"{base_seed}|{experiment_id}|{outer_fold_id}|{config_id}")
    return int(digest[:8], 16)


def mock_run_manifest(
    *,
    run_id: str,
    experiment_id: str,
    commit: str,
    dirty: bool,
    protocol_hash_value: str,
    registry_hash: str,
    split_hash: str,
    environment_hash: str,
    fold: dict[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "timestamp": "DRY_RUN_NO_OUTPUT",
        "git_commit": commit,
        "git_dirty": dirty,
        "protocol_version": "benchmark_v1",
        "protocol_hash": protocol_hash_value,
        "registry_hash": registry_hash,
        "split_hash": split_hash,
        "config_hash": "dry_run_config_hash",
        "environment_hash": environment_hash,
        "random_seed": random_seed,
        "outer_fold_id": fold["fold_id"],
        "inner_selection_method": "historical_inner_logo_mean_MAE_nm",
        "model_family": "DRY_RUN_NO_MODEL",
        "feature_family": "DRY_RUN_NO_FEATURES",
        "target_transform": "none",
        "sample_ids_train": fold["train_sample_ids"],
        "sample_ids_validation": [],
        "sample_ids_test": fold["test_sample_ids"],
        "input_source_hashes": {},
        "output_paths": [],
        "runtime_seconds": 0.0,
        "status": "DRY_RUN",
        "failure_reason": "",
    }


def pilot_access_errors(
    *,
    allow_seen_pilot_evaluation: bool,
    candidate_model_id: str | None,
    historical_selection_receipt: str | None,
    protocol_hash_value: str | None,
    registry_hash: str | None,
    environment_hash: str | None,
    root: Path,
) -> list[str]:
    errors: list[str] = []
    if not allow_seen_pilot_evaluation:
        errors.append("pilot evaluation requires --allow-seen-pilot-evaluation")
    if not candidate_model_id:
        errors.append("pilot evaluation requires --candidate-model-id")
    if not historical_selection_receipt:
        errors.append("pilot evaluation requires --historical-selection-receipt")
    if not protocol_hash_value:
        errors.append("pilot evaluation requires --protocol-hash")
    if not registry_hash:
        errors.append("pilot evaluation requires --registry-hash")
    if not environment_hash:
        errors.append("pilot evaluation requires --environment-hash")
    if git_dirty(root):
        errors.append("pilot evaluation requires a clean Git working tree")
    return errors


def manifest_json(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"

