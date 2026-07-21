#!/usr/bin/env python3
"""Finalize prospective unseen keyframe selections only after all five are complete."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))

from keyframe_selector.common import EXPECTED_SAMPLE_IDS, ensure_package_dirs, load_json, package_root, repo_root_from
from keyframe_selector.manifests import metadata_path, write_consolidated_manifests, write_selection_session
from keyframe_selector.provenance import refresh_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    repo_root = repo_root_from(THIS)
    pkg_root = package_root(repo_root)
    ensure_package_dirs(pkg_root)
    blockers: list[str] = []
    for sample_id in EXPECTED_SAMPLE_IDS:
        path = metadata_path(pkg_root, sample_id)
        if not path.exists():
            blockers.append(f"{sample_id}: missing metadata")
            continue
        payload = load_json(path)
        if payload.get("sample", {}).get("selection_status") != "completed":
            blockers.append(f"{sample_id}: status is {payload.get('sample', {}).get('selection_status')}")
            continue
        source = payload.get("source_video") or {}
        selection = payload.get("selection") or {}
        raw = repo_root / str(selection.get("raw_keyframe_png", ""))
        roi_crop = repo_root / str(selection.get("roi_keyframe_png", ""))
        if str(source.get("extension", "")).lower() != ".mpg":
            blockers.append(f"{sample_id}: source is not MPG")
        if not raw.is_file():
            blockers.append(f"{sample_id}: raw keyframe PNG missing")
        if not roi_crop.is_file():
            blockers.append(f"{sample_id}: ROI keyframe crop PNG missing")
        roi = selection.get("roi")
        if not isinstance(roi, dict) or roi.get("coordinate_space") != "source_frame_pixels":
            blockers.append(f"{sample_id}: ROI metadata missing or not source_frame_pixels")
        for key in ["selected_frame_index_0based", "selected_timestamp_sec", "requested_frames_before", "requested_frames_after", "roi_xyxy"]:
            if selection.get(key) is None:
                blockers.append(f"{sample_id}: missing selection.{key}")
    if blockers:
        for blocker in blockers:
            print(f"BLOCKER: {blocker}", file=sys.stderr)
        raise SystemExit("finalization blocked until all five manual selections are complete")
    write_selection_session(repo_root, pkg_root)
    write_consolidated_manifests(repo_root, pkg_root)
    refresh_provenance(repo_root, pkg_root)
    print(f"finalized manifests under {pkg_root / 'manifests'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
