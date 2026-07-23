#!/usr/bin/env python3
"""Create BENCHMARK_LOCK_V1 after validation artifacts exist."""

from __future__ import annotations

import argparse
import getpass
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from rheed2morph.benchmark.constants import BASELINE_COMMIT, BASELINE_TAG
from rheed2morph.benchmark.hashing import repo_root, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-snapshot", required=True)
    args = parser.parse_args()
    root = repo_root(Path.cwd())
    lock_path = root / "configs/benchmark_v1/BENCHMARK_LOCK_V1.json"
    payload = {
        "baseline_tag": BASELINE_TAG,
        "baseline_commit": BASELINE_COMMIT,
        "current_git_commit_before_phase0_commits": BASELINE_COMMIT,
        "protocol_version": "benchmark_v1",
        "protocol_hash": sha256_file(root / "configs/benchmark_v1/protocol_v1.yaml"),
        "master_registry_hash": sha256_file(root / "configs/benchmark_v1/registry/sample_registry_master_v1.csv"),
        "historical_registry_hash": sha256_file(root / "configs/benchmark_v1/registry/historical_development_cohort_v1.csv"),
        "prospective_pilot_registry_hash": sha256_file(root / "configs/benchmark_v1/registry/prospective_pilot_seen_v1.csv"),
        "split_hash": sha256_file(root / "configs/benchmark_v1/splits/historical_outer_logo_v1.json"),
        "experiment_matrix_hash": sha256_file(root / "configs/benchmark_v1/experiment_matrix_v1.csv"),
        "environment_hash": sha256_file(root / args.environment_snapshot),
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "created_by": getpass.getuser(),
        "git_branch": git(root, "branch", "--show-current"),
        "scientific_status": "benchmark_v1_protocol_freeze_historical_model_selection_only",
        "future_confirmatory_required": True,
    }
    write_json(lock_path, payload)
    digest = sha256_file(lock_path)
    (root / "configs/benchmark_v1/BENCHMARK_LOCK_V1.sha256").write_text(
        f"{digest}  configs/benchmark_v1/BENCHMARK_LOCK_V1.json\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "lock_hash": digest}, indent=2, sort_keys=True))


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, check=False, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else ""


if __name__ == "__main__":
    main()
