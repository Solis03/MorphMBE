from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .common import display_path, repo_path


def luminance_uint8(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def resize_and_pad(
    crop: np.ndarray, output_size: int
) -> tuple[np.ndarray, float, tuple[int, int, int, int]]:
    height, width = crop.shape
    scale = output_size / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = np.asarray(
        Image.fromarray(crop).resize((new_width, new_height), Image.Resampling.BILINEAR)
    )
    canvas = np.zeros((output_size, output_size), dtype=np.uint8)
    pad_left = (output_size - new_width) // 2
    pad_top = (output_size - new_height) // 2
    canvas[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = resized
    pad_right = output_size - new_width - pad_left
    pad_bottom = output_size - new_height - pad_top
    return canvas, float(scale), (pad_top, pad_bottom, pad_left, pad_right)


def frame_path(frames_dir: str | Path, index: int) -> Path:
    return repo_path(frames_dir) / f"{index}.png"


def build_clip_cache(
    manifest_df: pd.DataFrame, output_root: Path, report_root: Path, output_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for row in manifest_df.to_dict("records"):
        sample_id = str(row["sample_id"])
        indices = list(
            range(int(row["clip_start_index"]), int(row["clip_end_index"]) + 1)
        )
        crops: list[np.ndarray] = []
        cached_frames: list[np.ndarray] = []
        resize_scales: list[float] = []
        paddings: list[tuple[int, int, int, int]] = []
        crop_verified = True
        for index in indices:
            path = frame_path(row["frames_dir"], index)
            image = Image.open(path)
            gray = luminance_uint8(image)
            x, y, w, h = (
                int(row["roi_x"]),
                int(row["roi_y"]),
                int(row["roi_width"]),
                int(row["roi_height"]),
            )
            crop = gray[y : y + h, x : x + w]
            crop_verified = crop_verified and np.array_equal(
                crop, gray[y : y + h, x : x + w]
            )
            padded, scale, padding = resize_and_pad(crop, output_size)
            crops.append(crop)
            cached_frames.append(padded)
            resize_scales.append(scale)
            paddings.append(padding)
        frames = np.stack(cached_frames, axis=0)
        keyframe_offset = int(row["keyframe_index"]) - int(row["clip_start_index"])
        npz_path = output_root / "clip_cache" / f"{sample_id}.npz"
        np.savez_compressed(
            npz_path,
            frames_uint8=frames,
            frame_indices=np.asarray(indices, dtype=np.int32),
            keyframe_index=int(row["keyframe_index"]),
            keyframe_offset=keyframe_offset,
            roi_xywh=np.asarray(
                [row["roi_x"], row["roi_y"], row["roi_width"], row["roi_height"]],
                dtype=np.int32,
            ),
            original_roi_shape=np.asarray(
                [row["roi_height"], row["roi_width"]], dtype=np.int32
            ),
            padding=np.asarray(paddings, dtype=np.int32),
            resize_scale=np.asarray(resize_scales, dtype=np.float32),
            sample_id=sample_id,
            video_id=str(row["video_id"]),
        )
        preview_path = report_root / "clip_previews" / f"{sample_id}.png"
        save_clip_preview(frames, indices, keyframe_offset, row, preview_path)
        quality = compute_quality_metrics(
            frames,
            float(row["roi_width"] * row["roi_height"])
            / float(row["source_width"] * row["source_height"]),
        )
        quality.update(
            {
                "sample_id": sample_id,
                "video_id": row["video_id"],
                "quality_flags": ";".join(quality_flags(quality)),
            }
        )
        quality_rows.append(quality)
        cache_rows.append(
            {
                "sample_id": sample_id,
                "clip_cache_path": display_path(npz_path),
                "clip_preview_path": display_path(preview_path),
                "clip_frame_paths": json.dumps(
                    [display_path(frame_path(row["frames_dir"], i)) for i in indices]
                ),
                "clip_frame_indices": json.dumps(indices),
                "keyframe_offset_in_clip": keyframe_offset,
                "roi_area_fraction": float(row["roi_width"] * row["roi_height"])
                / float(row["source_width"] * row["source_height"]),
                "crop_verified": crop_verified,
                "resize_scale": float(np.mean(resize_scales)),
                "padding": json.dumps(paddings[0]),
            }
        )
    return pd.DataFrame(cache_rows), pd.DataFrame(quality_rows)


def save_clip_preview(
    frames: np.ndarray,
    indices: list[int],
    keyframe_offset: int,
    row: dict[str, Any],
    path: Path,
) -> None:
    tile = 130
    label_h = 20
    cols = 4
    rows = 4
    canvas = Image.new("RGB", (cols * tile, rows * (tile + label_h) + 42), "white")
    draw = ImageDraw.Draw(canvas)
    for i, frame in enumerate(frames):
        image = (
            Image.fromarray(frame)
            .resize((tile, tile), Image.Resampling.BILINEAR)
            .convert("RGB")
        )
        x = (i % cols) * tile
        y = 34 + (i // cols) * (tile + label_h)
        canvas.paste(image, (x, y))
        outline = "red" if i == keyframe_offset else "black"
        width = 4 if i == keyframe_offset else 1
        draw.rectangle([x, y, x + tile - 1, y + tile - 1], outline=outline, width=width)
        draw.text(
            (x + 4, y + tile + 2),
            f"{indices[i]}{' key' if i == keyframe_offset else ''}",
            fill="black",
        )
    title = f"{row['sample_id']} | {row['video_id']} | ROI {row['roi_width']}x{row['roi_height']} | clip {row['clip_start_index']}-{row['clip_end_index']}"
    draw.text((6, 8), title, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _safe_percentile(frames: np.ndarray, q: float) -> float:
    return float(np.percentile(frames.astype(np.float32), q))


def compute_quality_metrics(
    frames: np.ndarray, roi_area_fraction: float
) -> dict[str, float]:
    arr = frames.astype(np.float32)
    frame_means = arr.mean(axis=(1, 2))
    gy, gx = np.gradient(arr, axis=(1, 2))
    lap = np.gradient(gx, axis=2) + np.gradient(gy, axis=1)
    diffs = np.abs(np.diff(arr, axis=0))
    return {
        "roi_area_fraction": float(roi_area_fraction),
        "mean_intensity": float(arr.mean()),
        "intensity_std": float(arr.std()),
        "p01": _safe_percentile(frames, 1),
        "p05": _safe_percentile(frames, 5),
        "p50": _safe_percentile(frames, 50),
        "p95": _safe_percentile(frames, 95),
        "p99": _safe_percentile(frames, 99),
        "zero_dark_pixel_fraction": float(np.mean(arr <= 5)),
        "saturated_pixel_fraction": float(np.mean(arr >= 250)),
        "frame_to_frame_absdiff_mean": float(diffs.mean()) if diffs.size else 0.0,
        "frame_to_frame_absdiff_std": float(diffs.std()) if diffs.size else 0.0,
        "temporal_intensity_drift": float(frame_means[-1] - frame_means[0]),
        "temporal_sharpness_variation": float(np.var(lap.var(axis=(1, 2)))),
        "horizontal_gradient_energy": float(np.mean(gx**2)),
        "vertical_gradient_energy": float(np.mean(gy**2)),
        "temporal_variance": float(np.mean(np.var(arr, axis=0))),
    }


def quality_flags(metrics: dict[str, float]) -> list[str]:
    flags: list[str] = []
    if metrics["mean_intensity"] < 5 or metrics["p95"] < 15:
        flags.append("very_dark")
    if metrics["saturated_pixel_fraction"] > 0.01:
        flags.append("high_saturation")
    if metrics["roi_area_fraction"] < 0.02:
        flags.append("very_small_roi")
    if abs(metrics["temporal_intensity_drift"]) > 30:
        flags.append("abnormal_brightness_drift")
    if metrics["frame_to_frame_absdiff_mean"] < 0.5:
        flags.append("near_static_clip")
    if metrics["frame_to_frame_absdiff_mean"] > 45:
        flags.append("large_frame_discontinuity")
    return flags
