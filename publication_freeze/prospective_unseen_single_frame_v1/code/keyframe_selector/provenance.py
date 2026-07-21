"""Provenance helpers for the prospective unseen package."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import TOOL_VERSION
from .common import PACKAGE_REL, relpath, sha256_file, write_csv
from .decoder import ffmpeg_version, ffprobe_version


SOURCE_FILES = [
    ("code/discover_unseen_videos.py", "new non-interactive MPG discovery command", "new"),
    ("code/launch_keyframe_selector.py", "new GUI launcher and cache clear command", "new"),
    ("code/finalize_keyframe_selections.py", "new completion-gated finalizer", "new"),
    ("code/validate_keyframe_selections.py", "new validation command", "new"),
    ("code/keyframe_selector/common.py", "shared config, IO, hashing, CSV helpers", "new"),
    ("code/keyframe_selector/decoder.py", "ffprobe/ffmpeg decoder abstraction", "new"),
    ("code/keyframe_selector/gui.py", "PySide6 MPG-in-place keyframe and ROI selector", "minimal repair/wrapper"),
    ("code/keyframe_selector/manifests.py", "metadata, ROI crop, and manifest writer", "new"),
    ("code/keyframe_selector/provenance.py", "package provenance writer", "new"),
]


ORIGINAL_GUI_FILES = [
    ("src/rheed2morph/rheed/manual_roi_qt.py", "most recent functional PySide6 reviewer; GUI pattern and resume/save behavior source", "wrapped"),
    ("src/rheed2morph/rheed/manual_roi.py", "atomic JSON, ROI validation, source-pixel metadata helper source", "wrapped"),
    ("tools/manual_rheed_roi_reviewer.py", "most recent local launcher for PySide6 reviewer", "wrapped"),
]


def git_info(repo_root: Path) -> dict[str, Any]:
    def git(args: list[str]) -> str:
        result = subprocess.run(["git", *args], cwd=repo_root, check=False, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_branch": git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_status_short": git(["status", "--short"]),
        "python_version": sys.version,
        "platform": platform.platform(),
        "tool_version": TOOL_VERSION,
        "gui_framework": "PySide6",
        "video_decoding_backend": "ffprobe metadata + ffmpeg select frame extraction",
        "ffprobe_version": _safe_version(ffprobe_version),
        "ffmpeg_version": _safe_version(ffmpeg_version),
    }


def _safe_version(func) -> str:
    try:
        return str(func())
    except Exception as exc:
        return f"unavailable: {exc}"


def write_git_info(repo_root: Path, package_root: Path) -> None:
    path = package_root / "provenance" / "GIT_INFO.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(git_info(repo_root), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_source_map(repo_root: Path, package_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for packaged, role, copied_or_wrapped in SOURCE_FILES:
        path = package_root / packaged
        rows.append(
            {
                "packaged_relative_path": f"{PACKAGE_REL.as_posix()}/{packaged}",
                "original_source_path": "",
                "role": role,
                "copied_or_wrapped": copied_or_wrapped,
                "sha256": sha256_file(path) if path.exists() else "",
                "notes": "prospective unseen package file",
            }
        )
    for original, role, copied_or_wrapped in ORIGINAL_GUI_FILES:
        path = repo_root / original
        rows.append(
            {
                "packaged_relative_path": "not copied byte-for-byte",
                "original_source_path": original,
                "role": role,
                "copied_or_wrapped": copied_or_wrapped,
                "sha256": sha256_file(path) if path.exists() else "",
                "notes": "audited existing repository source",
            }
        )
    write_csv(
        package_root / "provenance" / "SOURCE_MAP.csv",
        rows,
        ["packaged_relative_path", "original_source_path", "role", "copied_or_wrapped", "sha256", "notes"],
    )


def write_package_manifest(package_root: Path) -> None:
    rows: list[str] = []
    manifest = package_root / "provenance" / "MANIFEST.sha256"
    for path in sorted(package_root.rglob("*")):
        if not path.is_file():
            continue
        if path == manifest:
            continue
        rows.append(f"{sha256_file(path)}  {path.relative_to(package_root).as_posix()}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")


def verify_frozen_manifest(repo_root: Path) -> dict[str, Any]:
    freeze_root = repo_root / "publication_freeze" / "rheed_afm_single_frame_v1_2026-07-18"
    manifest = freeze_root / "provenance" / "MANIFEST.sha256"
    if not manifest.exists():
        return {"status": "missing_manifest", "checked_files": 0, "mismatches": []}
    mismatches = []
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, _, rel = line.partition("  ")
        path = freeze_root / rel
        checked += 1
        if not path.exists():
            mismatches.append({"path": rel, "problem": "missing"})
        else:
            actual = sha256_file(path)
            if actual != expected:
                mismatches.append({"path": rel, "problem": "hash_mismatch", "expected": expected, "actual": actual})
    return {
        "status": "ok" if not mismatches else "modified",
        "checked_files": checked,
        "mismatches": mismatches,
        "freeze_manifest_relpath": relpath(manifest, repo_root),
    }


def refresh_provenance(repo_root: Path, package_root: Path) -> None:
    write_git_info(repo_root, package_root)
    write_source_map(repo_root, package_root)
    write_package_manifest(package_root)
