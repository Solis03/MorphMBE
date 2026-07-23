#!/usr/bin/env python3
"""Benchmark v1 dry-run without loading tensors, models, or pilot truth by default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rheed2morph.benchmark.constants import BASE_SEED
from rheed2morph.benchmark.hashing import repo_root, sha256_file
from rheed2morph.benchmark.protocol import validate_protocol_text
from rheed2morph.benchmark.schemas import schema_errors
from rheed2morph.benchmark.splits import load_split
from rheed2morph.benchmark.tracking import (
    derived_seed,
    deterministic_run_id,
    git_commit,
    git_dirty,
    mock_run_manifest,
    pilot_access_errors,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", default="outer_logo_6022")
    parser.add_argument("--experiment-id", default="B00_fold_median")
    parser.add_argument("--config-hash", default="dry_run_config_hash")
    parser.add_argument("--environment-hash", default="dry_run_environment_hash")
    parser.add_argument("--report-path")
    parser.add_argument("--attempt-pilot-truth-access", action="store_true")
    parser.add_argument("--allow-seen-pilot-evaluation", action="store_true")
    parser.add_argument("--candidate-model-id")
    parser.add_argument("--historical-selection-receipt")
    parser.add_argument("--protocol-hash")
    parser.add_argument("--registry-hash")
    args = parser.parse_args()
    root = repo_root(Path.cwd())
    if args.attempt_pilot_truth_access:
        errors = pilot_access_errors(
            allow_seen_pilot_evaluation=args.allow_seen_pilot_evaluation,
            candidate_model_id=args.candidate_model_id,
            historical_selection_receipt=args.historical_selection_receipt,
            protocol_hash_value=args.protocol_hash,
            registry_hash=args.registry_hash,
            environment_hash=args.environment_hash,
            root=root,
        )
        if errors:
            payload = {"status": "refused_pilot_truth_access", "errors": errors}
            emit(payload, args.report_path, root)
            raise SystemExit(2)
    protocol_path = root / "configs/benchmark_v1/protocol_v1.yaml"
    registry_path = root / "configs/benchmark_v1/registry/sample_registry_master_v1.csv"
    split_path = root / "configs/benchmark_v1/splits/historical_outer_logo_v1.json"
    protocol_errors = validate_protocol_text(protocol_path)
    if protocol_errors:
        payload = {"status": "failed", "errors": protocol_errors}
        emit(payload, args.report_path, root)
        raise SystemExit(1)
    split = load_split(split_path)
    fold = next((item for item in split["folds"] if item["fold_id"] == args.fold_id), None)
    if fold is None:
        raise SystemExit(f"unknown fold: {args.fold_id}")
    protocol_hash_value = sha256_file(protocol_path)
    registry_hash = sha256_file(registry_path)
    split_hash = sha256_file(split_path)
    commit = git_commit(root)
    run_id = deterministic_run_id(args.experiment_id, args.config_hash, commit, protocol_hash_value, split_hash)
    run_dir = root / "outputs/benchmark_v1/runs" / run_id
    if run_dir.exists():
        payload = {"status": "failed_existing_run_id", "run_id": run_id, "run_dir": run_dir.as_posix()}
        emit(payload, args.report_path, root)
        raise SystemExit(1)
    seed = derived_seed(BASE_SEED, args.experiment_id, fold["fold_id"], args.config_hash)
    manifest = mock_run_manifest(
        run_id=run_id,
        experiment_id=args.experiment_id,
        commit=commit,
        dirty=git_dirty(root),
        protocol_hash_value=protocol_hash_value,
        registry_hash=registry_hash,
        split_hash=split_hash,
        environment_hash=args.environment_hash,
        fold=fold,
        random_seed=seed,
    )
    schema = json.loads((root / "configs/benchmark_v1/schemas/run_manifest.schema.json").read_text(encoding="utf-8"))
    errors = schema_errors(schema, manifest)
    payload = {
        "status": "ok" if not errors else "failed",
        "fold_id": fold["fold_id"],
        "train_sample_ids": fold["train_sample_ids"],
        "test_sample_ids": fold["test_sample_ids"],
        "run_id": run_id,
        "manifest_valid": not errors,
        "manifest_errors": errors,
        "protocol_hash": protocol_hash_value,
        "registry_hash": registry_hash,
        "split_hash": split_hash,
        "loaded_image_tensors": False,
        "loaded_dino": False,
        "fit_scaler": False,
        "fit_model": False,
        "generated_predictions": False,
        "wrote_checkpoint": False,
    }
    emit(payload, args.report_path, root)
    if errors:
        raise SystemExit(1)


def emit(payload: dict, report_path: str | None, root: Path) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if report_path:
        out = root / report_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
