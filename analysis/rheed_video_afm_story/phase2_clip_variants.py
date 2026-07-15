from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .clip_cache import frame_path, luminance_uint8, resize_and_pad
from .common import display_path, repo_path


def selected_video_metadata(row: pd.Series) -> dict[str, Any]:
    path = repo_path(row["metadata_path"])
    metadata = json.loads(path.read_text(encoding="utf-8"))
    video = metadata.get("videos", {}).get(str(row["video_id"]), {})
    return video


def fps_for_row(row: pd.Series) -> tuple[float | None, str]:
    video = selected_video_metadata(row)
    fps = video.get("video_info", {}).get("fps")
    if fps is None:
        return None, "missing"
    try:
        fps_value = float(fps)
    except (TypeError, ValueError):
        return None, "invalid_metadata"
    if not np.isfinite(fps_value) or fps_value <= 0:
        return None, "invalid_metadata"
    return fps_value, "metadata.video_info.fps"


def extracted_frame_count(row: pd.Series) -> int | None:
    video = selected_video_metadata(row)
    value = video.get("video_info", {}).get("extracted_frame_count")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clip_timing_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        fps, source = fps_for_row(row)
        count = extracted_frame_count(row)
        key = int(row["keyframe_index"])
        start = int(row["clip_start_index"])
        end = int(row["clip_end_index"])
        frame_count = end - start + 1
        duration = (frame_count - 1) / fps if fps else np.nan
        preceding = key
        following = (count - 1 - key) if count is not None else np.nan
        fixed_available = []
        flags = []
        if fps is None:
            flags.append("fps_missing")
        else:
            for n in (8, 16):
                need_start = key - int(round(fps * 1.0))
                if need_start >= 0 and count is not None and key < count and fps >= n - 1:
                    fixed_available.append(f"fixed_time_{n}_1s")
        if key - start < frame_count / 3:
            key_pos = "front"
        elif key - start > 2 * frame_count / 3:
            key_pos = "back"
        else:
            key_pos = "central"
        rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "video_id": row["video_id"],
                "fps": fps if fps is not None else np.nan,
                "fps_source": source,
                "keyframe_index": key,
                "original_clip_start": start,
                "original_clip_end": end,
                "original_clip_frame_count": frame_count,
                "original_clip_duration_seconds": duration,
                "keyframe_offset": key - start,
                "keyframe_position": key_pos,
                "preceding_frames_available": preceding,
                "following_frames_available": following,
                "fixed_time_variants_available": ";".join(fixed_available),
                "timing_flags": ";".join(flags),
            }
        )
    return pd.DataFrame(rows)


def variant_indices(row: pd.Series, variant: str, spec: dict[str, Any], fps: float | None, frame_count: int | None) -> tuple[list[int], list[str]]:
    key = int(row["keyframe_index"])
    start = int(row["clip_start_index"])
    end = int(row["clip_end_index"])
    kind = spec["kind"]
    n = int(spec["frame_count"])
    flags: list[str] = []
    if kind == "keyframe":
        return [key], flags
    if kind == "selected":
        return list(range(start, end + 1)), flags
    if kind == "centered":
        left = n // 2 - 1 if n % 2 == 0 else n // 2
        s = key - left
        e = s + n - 1
        if s < start or e > end:
            flags.append("outside_selected_clip")
        return list(range(s, e + 1)), flags
    if kind == "causal":
        s = key - n + 1
        e = key
        if s < start or e > end:
            flags.append("outside_selected_clip")
        return list(range(s, e + 1)), flags
    if kind == "fixed_time_causal":
        if fps is None or frame_count is None:
            return [], ["fps_or_frame_count_missing"]
        seconds = float(spec["seconds"])
        s = key - int(round(fps * seconds))
        e = key
        if s < 0 or e >= frame_count:
            return [], ["video_boundary"]
        indices = np.linspace(s, e, n).round().astype(int).tolist()
        if len(indices) != len(set(indices)):
            return [], ["duplicate_indices_after_sampling"]
        return indices, flags
    raise ValueError(f"Unknown clip variant kind: {kind}")


def _read_variant_frames(row: pd.Series, indices: list[int], output_size: int) -> np.ndarray:
    frames = []
    for index in indices:
        img = Image.open(frame_path(row["frames_dir"], index))
        gray = luminance_uint8(img)
        x, y, w, h = (int(row["roi_x"]), int(row["roi_y"]), int(row["roi_width"]), int(row["roi_height"]))
        crop = gray[y : y + h, x : x + w]
        padded, _, _ = resize_and_pad(crop, output_size)
        frames.append(padded)
    return np.stack(frames, axis=0)


def save_contact_sheet(frames: np.ndarray, indices: list[int], row: pd.Series, variant: str, path: Path) -> None:
    tile = 110
    cols = min(8, len(frames))
    rows = int(np.ceil(len(frames) / cols))
    canvas = Image.new("RGB", (cols * tile, rows * (tile + 18) + 34), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 8), f"{row['sample_id']} | {variant} | {row['video_id']}", fill="black")
    for i, frame in enumerate(frames):
        x = (i % cols) * tile
        y = 30 + (i // cols) * (tile + 18)
        im = Image.fromarray(frame).resize((tile, tile)).convert("RGB")
        canvas.paste(im, (x, y))
        draw.rectangle([x, y, x + tile - 1, y + tile - 1], outline="black")
        draw.text((x + 3, y + tile + 1), str(indices[i]), fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def build_clip_variants(manifest: pd.DataFrame, config: dict[str, Any], output_root: Path, report_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    timing = clip_timing_audit(manifest)
    cache_root = output_root / "clip_variants"
    preview_root = report_root / "clip_variant_previews"
    rows: list[dict[str, Any]] = []
    output_size = int(config.get("image_size", 224))
    for _, row in manifest.iterrows():
        fps, _ = fps_for_row(row)
        total_frames = extracted_frame_count(row)
        for variant, spec in config["clip_variants"].items():
            indices, flags = variant_indices(row, variant, spec, fps, total_frames)
            available = bool(indices) and len(indices) == int(spec["frame_count"]) and len(indices) == len(set(indices))
            for index in indices:
                if not frame_path(row["frames_dir"], index).exists():
                    available = False
                    flags.append(f"missing_frame:{index}")
            cache_path = cache_root / variant / f"{row['sample_id']}.npz"
            preview_path = preview_root / variant / f"{row['sample_id']}.png"
            preprocessing_params = {}
            if available:
                frames = _read_variant_frames(row, indices, output_size)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    frames_uint8=frames.astype(np.uint8),
                    frame_indices=np.asarray(indices, dtype=np.int32),
                    sample_id=str(row["sample_id"]),
                    video_id=str(row["video_id"]),
                    clip_variant=variant,
                )
                save_contact_sheet(frames, indices, row, variant, preview_path)
                finite = frames.astype(float).ravel()
                p01, p99 = np.percentile(finite, [1, 99])
                preprocessing_params = {
                    "raw_luminance": {"input_range": "uint8_0_255"},
                    "clip_robust_contrast": {"p01": float(p01), "p99": float(p99), "scope": "joint_clip"},
                }
            rows.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "growth_run_id": str(row["growth_run_id"]),
                    "video_id": row["video_id"],
                    "clip_variant": variant,
                    "available": available,
                    "frame_indices": json.dumps(indices),
                    "frame_count": len(indices),
                    "cache_path": display_path(cache_path) if available else "",
                    "contact_sheet_path": display_path(preview_path) if available else "",
                    "preprocessing_params": json.dumps(preprocessing_params, sort_keys=True),
                    "variant_flags": ";".join(sorted(set(flags))),
                }
            )
    return pd.DataFrame(rows), timing
