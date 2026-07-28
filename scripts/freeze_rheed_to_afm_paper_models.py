#!/usr/bin/env python3
"""Build and validate the two-model RHEED-to-AFM paper freeze."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


REPO = Path(__file__).resolve().parents[1]
FREEZE_ROOT = (
    REPO
    / "paper_freeze"
    / "rheed_to_afm_dual_generative_models_v1_20260728"
)
M12_NAME = "MorphMBE-M12a-Strict15-RangeTerrace-v1"
M14_NAME = "MorphMBE-M14i-Full23-OODAware-v1"


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO, text=True
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True))


def _write_csv(
    path: Path, rows: list[dict[str, Any]], columns: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tracked_files(
    commit: str, prefixes: list[str], exact_paths: list[str]
) -> list[str]:
    paths = _git("ls-tree", "-r", "--name-only", commit).splitlines()
    selected = []
    for path in paths:
        if Path(path).name.startswith("."):
            continue
        if path in exact_paths or any(
            path == prefix or path.startswith(prefix.rstrip("/") + "/")
            for prefix in prefixes
        ):
            selected.append(path)
    return sorted(set(selected))


def _role(path: str) -> str:
    if path.startswith("analysis/"):
        return "experiment_code"
    if path.startswith("configs/"):
        return "parameters"
    if "/figures/" in path:
        return "result_figure"
    if path.endswith("_report.md"):
        return "scientific_report"
    if path.startswith("reports/"):
        return "result_table_or_manifest"
    if path == "removelist.txt":
        return "canonical_exclusion_policy"
    return "supporting_artifact"


def _artifact_rows(
    *,
    commit: str,
    prefixes: list[str],
    exact_paths: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for relative in _tracked_files(commit, prefixes, exact_paths):
        path = REPO / relative
        if not path.is_file():
            raise RuntimeError(f"artifact missing from worktree: {relative}")
        expected_blob = _git("rev-parse", f"{commit}:{relative}")
        current_blob = _git("hash-object", relative)
        if current_blob != expected_blob:
            raise RuntimeError(
                f"artifact differs from frozen source commit: {relative}"
            )
        rows.append(
            {
                "path": relative,
                "role": _role(relative),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "source_commit": commit,
                "git_blob_oid": expected_blob,
            }
        )
    return rows


def _copy_input_snapshots(root: Path) -> list[dict[str, Any]]:
    sources = [
        "removelist.txt",
        "outputs/rheed_video_afm_story/phase1/modeling_manifest.csv",
        "outputs/rheed_video_afm_story/phase3a/afm_descriptors.csv",
        "outputs/rheed_video_afm_story/phase3a/group_outer_splits.csv",
        "outputs/rheed_video_afm_story/phase4a/rheed_physics_features.csv",
        "outputs/rheed_video_afm_story/phase2a/embedding_registry.csv",
    ]
    rows = []
    target_root = root / "01_SHARED_DATA_PROVENANCE" / "input_snapshots"
    target_root.mkdir(parents=True, exist_ok=True)
    for relative in sources:
        source = REPO / relative
        if not source.is_file():
            raise RuntimeError(f"required derived input missing: {relative}")
        target = target_root / f"{relative.replace('/', '__')}.gz"
        with target.open("wb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_handle,
                mtime=0,
            ) as compressed:
                compressed.write(source.read_bytes())
        rows.append(
            {
                "source_path": relative,
                "snapshot_path": str(target.relative_to(REPO)),
                "source_sha256": _sha256(source),
                "snapshot_sha256": _sha256(target),
                "compression": "gzip",
                "uncompressed_size_bytes": source.stat().st_size,
                "raw_data": False,
            }
        )
    _write_csv(
        root / "01_SHARED_DATA_PROVENANCE" / "input_snapshot_manifest.csv",
        rows,
        [
            "source_path",
            "snapshot_path",
            "source_sha256",
            "snapshot_sha256",
            "compression",
            "uncompressed_size_bytes",
            "raw_data",
        ],
    )
    return rows


def _split_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    m12_rq = _read_csv(
        REPO
        / "reports/rheed_to_afm_functional_morphology/"
        "20260727_m12_range_terrace_v1/development/"
        "rq_crossfit_predictions.csv"
    )
    m12_val = _read_csv(
        REPO
        / "reports/rheed_to_afm_functional_morphology/"
        "20260727_m12_range_terrace_v1/development/"
        "rq_validation_predictions.csv"
    )
    m14_cohort = _read_csv(
        REPO
        / "reports/rheed_to_afm_ood_robust/"
        "20260728_m14_ood_robust_multiview_v3_final/cohort_manifest.csv"
    )
    primary = {row["growth_run_id"] for row in m12_rq}
    validation = {row["growth_run_id"] for row in m12_val}
    all_groups = {row["growth_run_id"] for row in m14_cohort}
    closed_test = all_groups - primary - validation
    if (len(primary), len(validation), len(closed_test)) != (15, 3, 5):
        raise RuntimeError("M12 split cardinality is not 15/3/5")
    provenance = {
        row["growth_run_id"]: row.get(
            "source_split_provenance", row.get("source_split", "")
        )
        for row in m14_cohort
    }
    m12_rows = []
    for group in sorted(all_groups):
        if group in primary:
            role = "strict15_outer_loo"
            protocol = "fit 14 growths, predict held growth"
            included = True
            fit_count = 14
        elif group in validation:
            role = "preexisting_validation"
            protocol = "fit all 15 development growths, predict validation"
            included = False
            fit_count = 15
        else:
            role = "historical_test_closed"
            protocol = "not used by M12 development or selection"
            included = False
            fit_count = 0
        m12_rows.append(
            {
                "growth_run_id": group,
                "paper_role": role,
                "evaluation_protocol": protocol,
                "fit_growth_count": fit_count,
                "included_in_primary_reported_metrics": included,
                "legacy_source_split_provenance": provenance[group],
            }
        )
    m14_rows = [
        {
            "growth_run_id": group,
            "paper_role": "full23_retrospective_outer_loo",
            "evaluation_protocol": "fit 22 growths, predict held growth",
            "fit_growth_count": 22,
            "included_in_primary_reported_metrics": True,
            "legacy_source_split_provenance": provenance[group],
        }
        for group in sorted(all_groups)
    ]
    return m12_rows, m14_rows


def _metric_rows_m14() -> list[dict[str, str]]:
    source = _read_csv(
        REPO
        / "reports/rheed_to_afm_ood_robust/"
        "20260728_m14_ood_robust_multiview_v3_final/"
        "robust_method_metrics.csv"
    )
    keep = {"M12a_frozen_alpha1", "M14i_target_specific_robust"}
    return [row for row in source if row["method"] in keep]


def _checksums(root: Path) -> None:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "checksums.sha256":
            continue
        rows.append(f"{_sha256(path)}  {path.relative_to(root)}")
    _write_text(root / "00_FREEZE_CONTROL" / "checksums.sha256", "\n".join(rows))


def build() -> None:
    if FREEZE_ROOT.exists():
        raise RuntimeError(
            f"freeze already exists and is immutable: {FREEZE_ROOT}"
        )
    m12_commit = _git("rev-parse", "dafc94c")
    m14_commit = _git("rev-parse", "e8ca301")
    m12_prefixes = [
        "analysis/rheed_to_afm_functional_morphology",
        "analysis/rheed_to_afm_generation",
        "analysis/rheed_to_afm_island_generation",
        "analysis/rheed_to_afm_island_diffusion",
        "analysis/rheed_to_afm_distinct_confidence",
        "reports/rheed_to_afm_functional_morphology/"
        "20260727_m12_range_terrace_v1/development",
    ]
    m12_exact = [
        "configs/rheed_to_afm_functional_morphology_m12.json",
        "reports/rheed_to_afm_functional_morphology_report.md",
        "reports/rheed_to_afm_functional_morphology_literature_review.md",
        "removelist.txt",
    ]
    m14_prefixes = [
        "analysis/rheed_to_afm_ood_robust",
        "analysis/rheed_to_afm_full_cohort_loo",
        "analysis/rheed_to_afm_functional_morphology",
        "analysis/rheed_to_afm_generation",
        "analysis/rheed_to_afm_island_generation",
        "analysis/rheed_to_afm_distinct_confidence",
        "reports/rheed_to_afm_ood_robust/"
        "20260728_m14_ood_robust_multiview_v3_final",
        "reports/rheed_to_afm_ood_robust_generation/"
        "20260728_m14_target_specific_m12a_generator_v1/full23_loo",
    ]
    m14_exact = [
        "configs/rheed_to_afm_ood_robust_v3_final.json",
        "configs/rheed_to_afm_ood_robust_generation.json",
        "reports/rheed_to_afm_ood_robust_report.md",
        "reports/rheed_to_afm_ood_robust_literature_review.md",
        "removelist.txt",
    ]
    m12_artifacts = _artifact_rows(
        commit=m12_commit,
        prefixes=m12_prefixes,
        exact_paths=m12_exact,
    )
    m14_artifacts = _artifact_rows(
        commit=m14_commit,
        prefixes=m14_prefixes,
        exact_paths=m14_exact,
    )
    m12_rows, m14_rows = _split_rows()
    shared_inputs = _copy_input_snapshots(FREEZE_ROOT)
    m12_root = FREEZE_ROOT / "02_MODEL_M12A_STRICT15"
    m14_root = FREEZE_ROOT / "03_MODEL_M14I_FULL23"
    columns = [
        "growth_run_id",
        "paper_role",
        "evaluation_protocol",
        "fit_growth_count",
        "included_in_primary_reported_metrics",
        "legacy_source_split_provenance",
    ]
    _write_csv(m12_root / "cohort_and_split.csv", m12_rows, columns)
    _write_csv(m14_root / "cohort_and_split.csv", m14_rows, columns)
    artifact_columns = [
        "path",
        "role",
        "sha256",
        "size_bytes",
        "source_commit",
        "git_blob_oid",
    ]
    _write_csv(
        m12_root / "artifact_manifest.csv",
        m12_artifacts,
        artifact_columns,
    )
    _write_csv(
        m14_root / "artifact_manifest.csv",
        m14_artifacts,
        artifact_columns,
    )
    shutil.copy2(
        REPO / "configs/rheed_to_afm_functional_morphology_m12.json",
        m12_root / "parameters.json",
    )
    shutil.copy2(
        REPO
        / "reports/rheed_to_afm_functional_morphology/"
        "20260727_m12_range_terrace_v1/development/"
        "target_prediction_summary.csv",
        m12_root / "frozen_target_metrics.csv",
    )
    shutil.copy2(
        REPO / "configs/rheed_to_afm_ood_robust_v3_final.json",
        m14_root / "target_head_parameters.json",
    )
    shutil.copy2(
        REPO / "configs/rheed_to_afm_ood_robust_generation.json",
        m14_root / "generator_parameters.json",
    )
    _write_csv(
        m14_root / "frozen_baseline_vs_final_metrics.csv",
        _metric_rows_m14(),
        list(_metric_rows_m14()[0]),
    )
    registry = [
        {
            "paper_model_id": "MODEL_A",
            "canonical_name": M12_NAME,
            "short_name": "M12a-Strict15",
            "primary_protocol": "strict LOO over 15 development growths",
            "outer_fit_count": 14,
            "primary_growth_count": 15,
            "source_commit": m12_commit,
            "status": "frozen development model",
        },
        {
            "paper_model_id": "MODEL_B",
            "canonical_name": M14_NAME,
            "short_name": "M14i-Full23",
            "primary_protocol": "retrospective LOO over all 23 growths",
            "outer_fit_count": 22,
            "primary_growth_count": 23,
            "source_commit": m14_commit,
            "status": "frozen retrospective robustness model",
        },
    ]
    _write_csv(
        FREEZE_ROOT / "00_FREEZE_CONTROL" / "MODEL_NAME_REGISTRY.csv",
        registry,
        list(registry[0]),
    )
    manifest = {
        "freeze_id": "rheed_to_afm_dual_generative_models_v1_20260728",
        "created_date": "2026-07-28",
        "raw_data_copied": False,
        "canonical_removelist_sha256": (
            "8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b"
        ),
        "shared_derived_input_snapshot_count": len(shared_inputs),
        "models": [
            {
                "paper_model_id": "MODEL_A",
                "canonical_name": M12_NAME,
                "source_commit": m12_commit,
                "primary_growth_count": 15,
                "preexisting_validation_growth_count": 3,
                "closed_historical_test_growth_count": 5,
                "closed_historical_test_scan_count": 24,
                "artifact_count": len(m12_artifacts),
                "retrieval_at_inference": False,
                "measured_afm_patch_used_at_inference": False,
                "claim_boundary": (
                    "Development evidence; the 15-growth feature family was "
                    "developed on this cohort and is not a prospective test."
                ),
            },
            {
                "paper_model_id": "MODEL_B",
                "canonical_name": M14_NAME,
                "source_commit": m14_commit,
                "primary_growth_count": 23,
                "outer_fit_growth_count": 22,
                "artifact_count": len(m14_artifacts),
                "retrieval_at_inference": False,
                "measured_afm_patch_used_at_inference": False,
                "claim_boundary": (
                    "Retrospective full-cohort LOO; all 23 growths informed "
                    "method development and a future prospective cohort is required."
                ),
            },
        ],
    }
    _write_json(
        FREEZE_ROOT / "00_FREEZE_CONTROL" / "FREEZE_MANIFEST.json",
        manifest,
    )
    _write_text(
        FREEZE_ROOT / "00_FREEZE_CONTROL" / "FROZEN_DO_NOT_EDIT.md",
        """# Frozen - do not edit

This directory is an immutable paper freeze. Do not overwrite files in place.
Any future change must create a new freeze ID and preserve this directory.
The source Git commits and per-artifact Git blob IDs are authoritative.
""",
    )
    _write_text(
        FREEZE_ROOT / "README.md",
        f"""# Dual generative RHEED-to-AFM paper model freeze

This freeze separates the two models intended to support the first paper
draft. They answer different scientific questions and must not be pooled into
one evaluation table without an explicit protocol column.

## MODEL_A - {M12_NAME}

- Short name: **M12a-Strict15**
- Purpose: strongest development-cohort morphology generation result.
- Primary evaluation: strict leave-one-growth-out over 15 development
  growths; each point fits 14 and predicts one.
- Separate evidence: three pre-existing validation growths, fit from the 15
  development growths.
- Closed evidence: five historical-test growths / 24 AFM scans were not used
  for M12 development, selection or primary evaluation.
- Generator: stochastic edge-preserving Laguerre island/terrace generator.
- Claim boundary: development evidence, not a prospective untouched test.

## MODEL_B - {M14_NAME}

- Short name: **M14i-Full23**
- Purpose: broader retrospective robustness, OOD and confidence evaluation.
- Primary evaluation: all 23 growths held out once; each point fits 22.
- Rq head: M14g curated/R3D 60:40 multiview blend.
- FSMI head: M14b target-blind RHEED-density weighted regression.
- Image generator: frozen M12a edge-preserving island/terrace generator.
- Claim boundary: retrospective method-development evidence, not a future
  untouched test.

## Frozen material

Each model directory contains the exact cohort roles, copied parameter files,
frozen headline metrics and a manifest of every code, result, report and
figure artifact. Each manifest records SHA-256, Git blob ID and source commit.
The shared directory snapshots only derived manifests and feature tables; no
raw RHEED video, raw AFM file or height array is copied.

Run `python3 scripts/freeze_rheed_to_afm_paper_models.py --validate` from the
repository root to verify the freeze.
""",
    )
    _write_text(
        FREEZE_ROOT / "REPRODUCE.md",
        """# Reproduction

## M12a-Strict15

```bash
PYTHONPATH=. .venv/bin/python \
  -m analysis.rheed_to_afm_functional_morphology.run \
  --config configs/rheed_to_afm_functional_morphology_m12.json
```

## M14i-Full23 target head and generator

```bash
PYTHONPATH=. .venv/bin/python \
  -m analysis.rheed_to_afm_ood_robust.run \
  --config configs/rheed_to_afm_ood_robust_v3_final.json

PYTHONPATH=. .venv/bin/python \
  -m analysis.rheed_to_afm_full_cohort_loo.run \
  --config configs/rheed_to_afm_ood_robust_generation.json \
  --mode full

PYTHONPATH=. .venv/bin/python \
  -m analysis.rheed_to_afm_full_cohort_loo.visualization \
  --config configs/rheed_to_afm_ood_robust_generation.json
```

For exact reconstruction, check out the source commit recorded for the model
in `00_FREEZE_CONTROL/MODEL_NAME_REGISTRY.csv`. Do not regenerate publication
numbers from a later mutable worktree without first validating every artifact
hash.
""",
    )
    _checksums(FREEZE_ROOT)
    validate()


def validate() -> None:
    if not FREEZE_ROOT.is_dir():
        raise RuntimeError(f"freeze does not exist: {FREEZE_ROOT}")
    checksum_path = (
        FREEZE_ROOT / "00_FREEZE_CONTROL" / "checksums.sha256"
    )
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        path = FREEZE_ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"freeze checksum mismatch: {relative}")
    input_manifest = (
        FREEZE_ROOT
        / "01_SHARED_DATA_PROVENANCE"
        / "input_snapshot_manifest.csv"
    )
    for row in _read_csv(input_manifest):
        source = REPO / row["source_path"]
        snapshot = REPO / row["snapshot_path"]
        if _sha256(source) != row["source_sha256"]:
            raise RuntimeError(
                f"derived source input changed: {row['source_path']}"
            )
        if _sha256(snapshot) != row["snapshot_sha256"]:
            raise RuntimeError(
                f"compressed input snapshot changed: {row['snapshot_path']}"
            )
        content = gzip.decompress(snapshot.read_bytes())
        if (
            hashlib.sha256(content).hexdigest()
            != row["source_sha256"]
            or len(content) != int(row["uncompressed_size_bytes"])
        ):
            raise RuntimeError(
                f"input snapshot is not byte-exact: {row['snapshot_path']}"
            )
    for relative in (
        "02_MODEL_M12A_STRICT15/artifact_manifest.csv",
        "03_MODEL_M14I_FULL23/artifact_manifest.csv",
    ):
        for row in _read_csv(FREEZE_ROOT / relative):
            path = REPO / row["path"]
            if not path.is_file() or _sha256(path) != row["sha256"]:
                raise RuntimeError(
                    f"external artifact checksum mismatch: {row['path']}"
                )
            blob = _git(
                "rev-parse",
                f"{row['source_commit']}:{row['path']}",
            )
            if blob != row["git_blob_oid"]:
                raise RuntimeError(
                    f"Git blob mismatch: {row['path']}"
                )
    print(f"validated freeze: {FREEZE_ROOT.relative_to(REPO)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate", action="store_true", help="validate an existing freeze"
    )
    args = parser.parse_args()
    if args.validate:
        validate()
    else:
        build()


if __name__ == "__main__":
    main()
