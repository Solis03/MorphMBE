"""Metadata and manifest writers for prospective unseen keyframe selections."""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from . import TOOL_VERSION
from .common import (
    EXPECTED_SAMPLE_IDS,
    atomic_write_json,
    file_fingerprint,
    load_config,
    load_json,
    relpath,
    sample_numeric,
    utc_now,
    write_csv,
)
from .decoder import extract_frame, frame_sha, probe_video


DISCOVERY_FIELDS = [
    "sample_id",
    "source_video_relpath",
    "filename",
    "size_bytes",
    "codec",
    "width",
    "height",
    "fps",
    "duration_sec",
    "frame_count",
    "decode_test_pass",
    "candidate_count_for_sample",
    "notes",
]

MANIFEST_FIELDS = [
    "sample_id",
    "sample_id_numeric",
    "selection_status",
    "source_video_relpath",
    "source_video_filename",
    "raw_keyframe_relpath",
    "roi_keyframe_relpath",
    "model_ready_keyframe_relpath",
    "roi_xyxy",
    "selected_frame_index_0based",
    "selected_frame_number_1based",
    "selected_timestamp_sec",
    "requested_frames_before",
    "requested_frames_after",
    "effective_frames_before",
    "effective_frames_after",
    "frame_stride",
    "context_start_frame_0based",
    "context_end_frame_0based",
    "context_total_frames",
    "fps",
    "frame_count",
    "width",
    "height",
    "codec",
    "metadata_json_relpath",
    "raw_keyframe_sha256",
    "roi_keyframe_sha256",
    "notes",
]

CONTEXT_FIELDS = [
    "sample_id",
    "source_video_relpath",
    "role",
    "relative_offset",
    "source_frame_index_0based",
    "estimated_timestamp_sec",
    "frame_stride",
]


def metadata_path(package_root: Path, sample_id: str) -> Path:
    return package_root / "metadata" / "samples" / f"{sample_id}.json"


def discover_mpgs(repo_root: Path, package_root: Path, hash_mode: str = "fast") -> list[dict[str, Any]]:
    cfg = load_config(repo_root)
    rows: list[dict[str, Any]] = []
    package_root.joinpath("cache", "frames").mkdir(parents=True, exist_ok=True)
    for sample_id in cfg.sample_ids:
        sample_dir = cfg.data_root / sample_id
        mpgs = sorted(path for path in sample_dir.iterdir() if path.is_file() and path.suffix.lower() == ".mpg") if sample_dir.is_dir() else []
        candidate_count = len(mpgs)
        if candidate_count == 0:
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_video_relpath": "",
                    "filename": "",
                    "size_bytes": "",
                    "codec": "",
                    "width": "",
                    "height": "",
                    "fps": "",
                    "duration_sec": "",
                    "frame_count": "",
                    "decode_test_pass": False,
                    "candidate_count_for_sample": 0,
                    "notes": "BLOCKER: zero MPG files found",
                }
            )
            write_pending_metadata(repo_root, package_root, sample_id, [], hash_mode)
            continue
        sample_candidates: list[dict[str, Any]] = []
        for video in mpgs:
            stat = video.stat()
            notes = []
            meta: dict[str, Any]
            try:
                meta = probe_video(video)
            except Exception as exc:
                meta = {"codec": "", "width": 0, "height": 0, "fps": 0.0, "duration_sec": 0.0, "frame_count": 0, "frame_count_method": "ffprobe failed"}
                notes.append(f"ffprobe failed: {exc}")
            decode_path = package_root / "cache" / "frames" / f"{sample_id}_decode_test_frame0.png"
            display_transform = cfg.display_transforms.get(sample_id)
            decode_pass = False
            try:
                extract_frame(video, 0, decode_path, display_transform=display_transform)
                decode_pass = True
                notes.append("decoded frame 0 with ffmpeg")
            except Exception as exc:
                notes.append(f"decode test failed: {exc}")
            fingerprint, fingerprint_note = file_fingerprint(video, hash_mode)
            candidate = {
                "repo_relative_path": relpath(video, repo_root),
                "filename": video.name,
                "extension": video.suffix.lower(),
                "size_bytes": stat.st_size,
                "mtime_utc": _mtime_utc(stat.st_mtime),
                "mtime_ns": stat.st_mtime_ns,
                "sha256": fingerprint,
                "sha256_note": fingerprint_note,
                **meta,
                "decode_test_pass": decode_pass,
                "display_transform": display_transform or "none",
                "display_width": transformed_dimensions(meta["width"], meta["height"], display_transform)[0],
                "display_height": transformed_dimensions(meta["width"], meta["height"], display_transform)[1],
                "notes": "; ".join(notes),
            }
            sample_candidates.append(candidate)
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_video_relpath": candidate["repo_relative_path"],
                    "filename": video.name,
                    "size_bytes": stat.st_size,
                    "codec": meta["codec"],
                    "width": meta["width"],
                    "height": meta["height"],
                    "fps": meta["fps"],
                    "duration_sec": meta["duration_sec"],
                    "frame_count": meta["frame_count"],
                    "decode_test_pass": decode_pass,
                    "candidate_count_for_sample": candidate_count,
                    "notes": "; ".join(notes) or ("requires GUI video choice" if candidate_count > 1 else "single MPG candidate"),
                }
            )
        write_pending_metadata(repo_root, package_root, sample_id, sample_candidates, hash_mode)
    write_csv(package_root / "manifests" / "discovered_mpg_files.csv", rows, DISCOVERY_FIELDS)
    write_selection_session(repo_root, package_root)
    write_consolidated_manifests(repo_root, package_root)
    return rows


def _mtime_utc(timestamp: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def write_pending_metadata(
    repo_root: Path,
    package_root: Path,
    sample_id: str,
    candidates: list[dict[str, Any]],
    hash_mode: str,
) -> None:
    path = metadata_path(package_root, sample_id)
    existing = load_json(path) if path.exists() else {}
    existing_status = existing.get("sample", {}).get("selection_status")
    keep_completed = existing_status == "completed"
    selected_source = existing.get("source_video")
    if not keep_completed:
        selected_source = candidates[0] if len(candidates) == 1 else None
    elif isinstance(selected_source, dict):
        matched = next((item for item in candidates if item.get("repo_relative_path") == selected_source.get("repo_relative_path")), None)
        if matched:
            selected_source = {**selected_source, **matched}
    payload = {
        "schema_version": "unseen-keyframe-selection-v1",
        "freeze_reference": {
            "freeze_id": "rheed_afm_single_frame_v1_2026-07-18",
            "freeze_path": "publication_freeze/rheed_afm_single_frame_v1_2026-07-18",
        },
        "sample": {
            "sample_id": sample_id,
            "sample_id_numeric": sample_numeric(sample_id),
            "selection_status": existing_status if keep_completed else "pending_manual_selection",
        },
        "source_video_candidates": candidates,
        "source_video": selected_source,
        "selection": existing.get("selection") if keep_completed else None,
        "operator": existing.get("operator") if keep_completed else None,
        "validation": existing.get("validation") if keep_completed else {"source_video_untouched": True},
        "hash_mode": hash_mode,
    }
    atomic_write_json(path, payload)


def write_selection_session(repo_root: Path, package_root: Path) -> None:
    cfg = load_config(repo_root)
    statuses = {}
    for sample_id in cfg.sample_ids:
        path = metadata_path(package_root, sample_id)
        statuses[sample_id] = load_json(path).get("sample", {}).get("selection_status") if path.exists() else "missing"
    atomic_write_json(
        package_root / "metadata" / "selection_session.json",
        {
            "schema_version": "unseen-keyframe-selection-session-v1",
            "updated_at_utc": utc_now(),
            "sample_ids": cfg.sample_ids,
            "status_by_sample": statuses,
            "data_root": relpath(cfg.data_root, repo_root),
            "freeze_reference": {"freeze_id": cfg.freeze_id, "freeze_path": relpath(cfg.freeze_path, repo_root)},
            "tool_version": TOOL_VERSION,
        },
    )


def save_selection(
    repo_root: Path,
    package_root: Path,
    sample_id: str,
    source_video_relpath: str,
    frame_index: int,
    frames_before: int,
    frames_after: int,
    frame_stride: int,
    notes: str,
    displayed_frame_path: Path,
    roi: dict[str, Any],
) -> dict[str, Any]:
    cfg = load_config(repo_root)
    display_transform = cfg.display_transforms.get(sample_id)
    source_video = repo_root / source_video_relpath
    current_meta = probe_video(source_video)
    if current_meta["frame_count"] and frame_index >= current_meta["frame_count"]:
        raise ValueError(f"frame index {frame_index} is outside frame_count {current_meta['frame_count']}")
    raw_path = package_root / "keyframes" / "raw" / f"{sample_id}_frame_{frame_index:06d}.png"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(displayed_frame_path, raw_path)
    raw_hash = frame_sha(raw_path)
    display_width, display_height = transformed_dimensions(current_meta["width"], current_meta["height"], display_transform)
    roi_payload = normalize_roi(roi, display_width, display_height, frame_index)
    roi_path = package_root / "keyframes" / "roi" / f"{sample_id}_frame_{frame_index:06d}_roi.png"
    roi_hash = save_roi_crop(raw_path, roi_path, roi_payload)
    reextract_path = package_root / "cache" / "frames" / f"{sample_id}_verify_frame_{frame_index:06d}.png"
    extract_frame(source_video, frame_index, reextract_path, display_transform=display_transform)
    displayed_equals_saved = frame_sha(displayed_frame_path) == raw_hash
    deterministic_match = frame_sha(reextract_path) == raw_hash
    stat = source_video.stat()
    effective_before = max(0, min(frames_before, frame_index))
    max_after = max(0, (current_meta["frame_count"] - 1 - frame_index) if current_meta["frame_count"] else frames_after)
    effective_after = max(0, min(frames_after, max_after))
    context_start = frame_index - effective_before * frame_stride
    context_end = frame_index + effective_after * frame_stride
    selected_ts = frame_index / current_meta["fps"] if current_meta["fps"] else None
    existing = load_json(metadata_path(package_root, sample_id))
    candidates = existing.get("source_video_candidates", [])
    matched_candidate = next((item for item in candidates if item.get("repo_relative_path") == source_video_relpath), None)
    source_payload = dict(matched_candidate or {})
    source_payload.update(
        {
            "repo_relative_path": source_video_relpath,
            "filename": source_video.name,
            "extension": source_video.suffix.lower(),
            "size_bytes": stat.st_size,
            "mtime_utc": _mtime_utc(stat.st_mtime),
            "mtime_ns": stat.st_mtime_ns,
            **current_meta,
            "display_transform": display_transform or "none",
            "display_width": display_width,
            "display_height": display_height,
        }
    )
    payload = {
        "schema_version": "unseen-keyframe-selection-v1",
        "freeze_reference": {
            "freeze_id": "rheed_afm_single_frame_v1_2026-07-18",
            "freeze_path": "publication_freeze/rheed_afm_single_frame_v1_2026-07-18",
        },
        "sample": {
            "sample_id": sample_id,
            "sample_id_numeric": sample_numeric(sample_id),
            "selection_status": "completed",
        },
        "source_video_candidates": candidates,
        "source_video": source_payload,
        "selection": {
            "selected_frame_index_0based": int(frame_index),
            "selected_frame_number_1based": int(frame_index) + 1,
            "selected_timestamp_sec": selected_ts,
            "selected_pts": None,
            "requested_frames_before": int(frames_before),
            "requested_frames_after": int(frames_after),
            "effective_frames_before": int(effective_before),
            "effective_frames_after": int(effective_after),
            "frame_stride": int(frame_stride),
            "context_start_frame_0based": int(context_start),
            "context_end_frame_0based": int(context_end),
            "context_total_frames": int(effective_before + 1 + effective_after),
            "raw_keyframe_png": relpath(raw_path, repo_root),
            "roi_keyframe_png": relpath(roi_path, repo_root),
            "model_ready_keyframe": None,
            "display_transform": display_transform or "none",
            "roi_xyxy": [
                int(roi_payload["x"]),
                int(roi_payload["y"]),
                int(roi_payload["x"]) + int(roi_payload["width"]),
                int(roi_payload["y"]) + int(roi_payload["height"]),
            ],
            "roi": roi_payload,
            "notes": notes,
        },
        "operator": {
            "selected_at_utc": utc_now(),
            "tool_version": TOOL_VERSION,
            "git_commit": _git_commit(repo_root),
            "hostname": platform.node(),
        },
        "validation": {
            "source_video_untouched": True,
            "displayed_frame_equals_saved_frame": bool(displayed_equals_saved),
            "deterministic_reextract_equals_saved_frame": bool(deterministic_match),
            "raw_keyframe_sha256": raw_hash,
            "roi_keyframe_sha256": roi_hash,
            "roi_inside_source_frame": True,
            "model_ready_transform_verified": False,
        },
    }
    atomic_write_json(metadata_path(package_root, sample_id), payload)
    save_preview(raw_path, package_root / "previews" / f"{sample_id}_keyframe_preview.png", sample_id, frame_index)
    write_selection_session(repo_root, package_root)
    write_consolidated_manifests(repo_root, package_root)
    return payload


def transformed_dimensions(width: int, height: int, display_transform: str | None) -> tuple[int, int]:
    if display_transform == "rotate_clockwise_90":
        return int(height), int(width)
    return int(width), int(height)


def normalize_roi(roi: dict[str, Any], source_width: int, source_height: int, frame_index: int) -> dict[str, Any]:
    x = int(roi["x"])
    y = int(roi["y"])
    width = int(roi["width"])
    height = int(roi["height"])
    if width <= 0 or height <= 0:
        raise ValueError("ROI width and height must be greater than 0.")
    if x < 0 or y < 0 or x + width > source_width or y + height > source_height:
        raise ValueError("ROI must be inside the source frame.")
    return {
        "reference_frame_index": int(frame_index),
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "source_width": int(source_width),
        "source_height": int(source_height),
        "coordinate_space": "source_frame_pixels",
        "x_normalized": x / source_width if source_width else None,
        "y_normalized": y / source_height if source_height else None,
        "width_normalized": width / source_width if source_width else None,
        "height_normalized": height / source_height if source_height else None,
    }


def save_roi_crop(raw_path: Path, roi_path: Path, roi: dict[str, Any]) -> str:
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(raw_path) as image:
        rgb = image.convert("RGB")
        expected = (int(roi["source_width"]), int(roi["source_height"]))
        if rgb.size != expected:
            raise ValueError(f"Raw keyframe size {rgb.size} does not match ROI source size {expected}")
        crop = rgb.crop((int(roi["x"]), int(roi["y"]), int(roi["x"]) + int(roi["width"]), int(roi["y"]) + int(roi["height"])))
        crop.save(roi_path, format="PNG", compress_level=0)
    return frame_sha(roi_path)


def _git_commit(repo_root: Path) -> str:
    import subprocess

    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=False, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def save_preview(raw_path: Path, output_path: Path, sample_id: str, frame_index: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(raw_path) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((640, 420), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (rgb.width, rgb.height + 28), "white")
        canvas.paste(rgb, (0, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), f"{sample_id} frame {frame_index}", fill="black")
        canvas.save(output_path, format="PNG")


def write_consolidated_manifests(repo_root: Path, package_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    for sample_id in EXPECTED_SAMPLE_IDS:
        path = metadata_path(package_root, sample_id)
        if not path.exists():
            rows.append({"sample_id": sample_id, "sample_id_numeric": sample_numeric(sample_id), "selection_status": "missing", "metadata_json_relpath": relpath(path, repo_root)})
            continue
        payload = load_json(path)
        row = manifest_row(repo_root, path, payload)
        rows.append(row)
        if row.get("selection_status") == "completed":
            context_rows.extend(context_rows_for(payload))
    write_csv(package_root / "manifests" / "unseen_keyframe_manifest.csv", rows, MANIFEST_FIELDS)
    atomic_write_json(package_root / "manifests" / "unseen_keyframe_manifest.json", {"rows": rows})
    write_csv(package_root / "manifests" / "unseen_context_frame_index.csv", context_rows, CONTEXT_FIELDS)


def manifest_row(repo_root: Path, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    sample = payload.get("sample") or {}
    source = payload.get("source_video") or {}
    selection = payload.get("selection") or {}
    validation = payload.get("validation") or {}
    selection_status = sample.get("selection_status")
    if selection_status == "completed" and not isinstance(selection.get("roi"), dict):
        selection_status = "needs_roi_review"
    return {
        "sample_id": sample.get("sample_id"),
        "sample_id_numeric": sample.get("sample_id_numeric"),
        "selection_status": selection_status,
        "source_video_relpath": source.get("repo_relative_path", ""),
        "source_video_filename": source.get("filename", ""),
        "raw_keyframe_relpath": selection.get("raw_keyframe_png", ""),
        "roi_keyframe_relpath": selection.get("roi_keyframe_png", ""),
        "model_ready_keyframe_relpath": selection.get("model_ready_keyframe", ""),
        "roi_xyxy": selection.get("roi_xyxy", ""),
        "selected_frame_index_0based": selection.get("selected_frame_index_0based", ""),
        "selected_frame_number_1based": selection.get("selected_frame_number_1based", ""),
        "selected_timestamp_sec": selection.get("selected_timestamp_sec", ""),
        "requested_frames_before": selection.get("requested_frames_before", ""),
        "requested_frames_after": selection.get("requested_frames_after", ""),
        "effective_frames_before": selection.get("effective_frames_before", ""),
        "effective_frames_after": selection.get("effective_frames_after", ""),
        "frame_stride": selection.get("frame_stride", ""),
        "context_start_frame_0based": selection.get("context_start_frame_0based", ""),
        "context_end_frame_0based": selection.get("context_end_frame_0based", ""),
        "context_total_frames": selection.get("context_total_frames", ""),
        "fps": source.get("fps", ""),
        "frame_count": source.get("frame_count", ""),
        "width": source.get("width", ""),
        "height": source.get("height", ""),
        "codec": source.get("codec", ""),
        "metadata_json_relpath": relpath(path, repo_root),
        "raw_keyframe_sha256": validation.get("raw_keyframe_sha256", ""),
        "roi_keyframe_sha256": validation.get("roi_keyframe_sha256", ""),
        "notes": selection.get("notes", ""),
    }


def context_rows_for(payload: dict[str, Any]) -> list[dict[str, Any]]:
    sample_id = payload["sample"]["sample_id"]
    source_rel = payload["source_video"]["repo_relative_path"]
    selection = payload["selection"]
    fps = float(payload["source_video"].get("fps") or 0)
    stride = int(selection.get("frame_stride") or 1)
    key = int(selection["selected_frame_index_0based"])
    before = int(selection["effective_frames_before"])
    after = int(selection["effective_frames_after"])
    rows = []
    for offset in range(-before, after + 1):
        frame_index = key + offset * stride
        rows.append(
            {
                "sample_id": sample_id,
                "source_video_relpath": source_rel,
                "role": "keyframe" if offset == 0 else ("before" if offset < 0 else "after"),
                "relative_offset": f"{offset:+d}" if offset else "0",
                "source_frame_index_0based": frame_index,
                "estimated_timestamp_sec": frame_index / fps if fps else "",
                "frame_stride": stride,
            }
        )
    return rows
