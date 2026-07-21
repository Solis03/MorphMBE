#!/usr/bin/env python3
"""Validate prospective unseen keyframe selections and source integrity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))

from keyframe_selector.common import EXPECTED_SAMPLE_IDS, atomic_write_json, ensure_package_dirs, load_config, load_json, package_root, repo_root_from, sha256_file
from keyframe_selector.decoder import extract_frame
from keyframe_selector.manifests import metadata_path, write_consolidated_manifests
from keyframe_selector.provenance import refresh_provenance, verify_frozen_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true", help="Exit nonzero if any of the five samples is not completed.")
    args = parser.parse_args()
    repo_root = repo_root_from(THIS)
    pkg_root = package_root(repo_root)
    ensure_package_dirs(pkg_root)
    report = validate(repo_root, pkg_root)
    atomic_write_json(pkg_root / "provenance" / "selection_validation_report.json", report)
    write_consolidated_manifests(repo_root, pkg_root)
    refresh_provenance(repo_root, pkg_root)
    print(json.dumps({"status": report["status"], "error_count": len(report["errors"]), "warning_count": len(report["warnings"])}, indent=2))
    if report["errors"] or (args.require_complete and report["incomplete_samples"]):
        raise SystemExit(1)


def validate(repo_root: Path, pkg_root: Path) -> dict[str, Any]:
    cfg = load_config(repo_root)
    errors: list[str] = []
    warnings: list[str] = []
    incomplete: list[str] = []
    seen: set[str] = set()
    for path in sorted((pkg_root / "metadata" / "samples").glob("*.json")):
        seen.add(path.stem)
    if seen != set(EXPECTED_SAMPLE_IDS):
        errors.append(f"expected sample metadata exactly {EXPECTED_SAMPLE_IDS}, found {sorted(seen)}")
    for sample_id in EXPECTED_SAMPLE_IDS:
        path = metadata_path(pkg_root, sample_id)
        if not path.exists():
            incomplete.append(sample_id)
            continue
        payload = load_json(path)
        status = payload.get("sample", {}).get("selection_status")
        if status != "completed":
            incomplete.append(sample_id)
            continue
        selection = payload.get("selection") or {}
        if not isinstance(selection.get("roi"), dict):
            incomplete.append(sample_id)
        validate_completed_sample(repo_root, pkg_root, sample_id, payload, errors, warnings, cfg.display_transforms.get(sample_id))
    for copied in list(pkg_root.rglob("*.mpg")) + list(pkg_root.rglob("*.MPG")) + list(pkg_root.rglob("*.avi")) + list(pkg_root.rglob("*.AVI")) + list(pkg_root.rglob("*.imm")) + list(pkg_root.rglob("*.IMM")):
        errors.append(f"full or raw video-like file present inside package: {copied.relative_to(pkg_root)}")
    frozen = verify_frozen_manifest(repo_root)
    if frozen["status"] != "ok":
        errors.append(f"frozen package manifest check failed: {frozen['status']}")
    return {
        "schema_version": "unseen-keyframe-selection-validation-v1",
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "warnings": warnings,
        "incomplete_samples": incomplete,
        "frozen_directory_verification": frozen,
        "checks": [
            "exactly five expected sample IDs",
            "only MPG source references in completed metadata",
            "no IMM or AVI final metadata source path",
            "frame index within bounds",
            "context window consistency",
            "exported keyframe exists and hash matches",
            "ROI metadata exists in source_frame_pixels and ROI crop hash matches",
            "saved keyframe matches deterministic re-extraction",
            "source video size and mtime unchanged from saved metadata",
            "no full video copied into package",
            "frozen manifest unchanged",
            "no prediction outputs checked or generated",
            "no AFM labels used by this package",
        ],
    }


def validate_completed_sample(
    repo_root: Path,
    pkg_root: Path,
    sample_id: str,
    payload: dict[str, Any],
    errors: list[str],
    warnings: list[str],
    display_transform: str | None,
) -> None:
    source = payload.get("source_video") or {}
    selection = payload.get("selection") or {}
    source_rel = str(source.get("repo_relative_path", ""))
    if not source_rel.lower().endswith(".mpg"):
        errors.append(f"{sample_id}: source is not MPG: {source_rel}")
    if ".imm" in source_rel.lower() or ".avi" in source_rel.lower():
        errors.append(f"{sample_id}: forbidden source extension appears in metadata: {source_rel}")
    source_path = repo_root / source_rel
    if not source_path.exists():
        errors.append(f"{sample_id}: source video missing: {source_rel}")
        return
    stat = source_path.stat()
    if source.get("size_bytes") and int(source["size_bytes"]) != stat.st_size:
        errors.append(f"{sample_id}: source size changed")
    if source.get("mtime_ns") and int(source["mtime_ns"]) != stat.st_mtime_ns:
        errors.append(f"{sample_id}: source mtime changed")
    frame_index = int(selection.get("selected_frame_index_0based", -1))
    frame_count = int(source.get("frame_count") or 0)
    if frame_index < 0 or (frame_count and frame_index >= frame_count):
        errors.append(f"{sample_id}: selected frame index out of bounds")
    raw_rel = str(selection.get("raw_keyframe_png", ""))
    raw_path = repo_root / raw_rel
    roi_rel = str(selection.get("roi_keyframe_png", ""))
    roi_path = repo_root / roi_rel
    if not raw_path.is_file():
        errors.append(f"{sample_id}: raw keyframe PNG missing: {raw_rel}")
    else:
        expected_hash = (payload.get("validation") or {}).get("raw_keyframe_sha256")
        actual_hash = sha256_file(raw_path)
        if expected_hash != actual_hash:
            errors.append(f"{sample_id}: raw keyframe hash mismatch")
        verify_path = pkg_root / "cache" / "frames" / f"{sample_id}_validation_reextract_{frame_index:06d}.png"
        try:
            extract_frame(source_path, frame_index, verify_path, display_transform=display_transform)
            if sha256_file(verify_path) != actual_hash:
                errors.append(f"{sample_id}: saved raw keyframe differs from deterministic re-extraction")
        except Exception as exc:
            errors.append(f"{sample_id}: could not re-extract selected frame: {exc}")
    roi = selection.get("roi")
    if not isinstance(roi, dict):
        errors.append(f"{sample_id}: ROI metadata missing")
    else:
        if roi.get("coordinate_space") != "source_frame_pixels":
            errors.append(f"{sample_id}: ROI coordinate_space is not source_frame_pixels")
        try:
            x = int(roi["x"])
            y = int(roi["y"])
            width = int(roi["width"])
            height = int(roi["height"])
            source_width = int(roi["source_width"])
            source_height = int(roi["source_height"])
            if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > source_width or y + height > source_height:
                errors.append(f"{sample_id}: ROI is outside source bounds or empty")
        except Exception as exc:
            errors.append(f"{sample_id}: invalid ROI metadata: {exc}")
    if not roi_path.is_file():
        errors.append(f"{sample_id}: ROI keyframe crop PNG missing: {roi_rel}")
    else:
        expected_roi_hash = (payload.get("validation") or {}).get("roi_keyframe_sha256")
        actual_roi_hash = sha256_file(roi_path)
        if expected_roi_hash != actual_roi_hash:
            errors.append(f"{sample_id}: ROI keyframe crop hash mismatch")
    before = int(selection.get("effective_frames_before", -1))
    after = int(selection.get("effective_frames_after", -1))
    total = int(selection.get("context_total_frames", -1))
    if before < 0 or after < 0 or total != before + 1 + after:
        errors.append(f"{sample_id}: inconsistent context frame counts")
    start = int(selection.get("context_start_frame_0based", -1))
    end = int(selection.get("context_end_frame_0based", -1))
    stride = int(selection.get("frame_stride", 0))
    if stride <= 0 or start > frame_index or end < frame_index:
        errors.append(f"{sample_id}: invalid context bounds or stride")
    if selection.get("model_ready_keyframe") not in (None, ""):
        warnings.append(f"{sample_id}: model_ready_keyframe exists; this task did not verify model-ready conversion")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
