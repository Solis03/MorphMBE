"""Select representative RHEED frames for human inspection."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import numpy as np

from rheed2morph.rheed.frame_quality import (
    FLAG_KEYS,
    NUMERIC_FEATURE_KEYS,
    SCORE_KEYS,
    active_flags,
    enhance_for_display,
    extract_frame_quality_features,
    finite_float,
    frame_to_gray_float32,
    normalize_for_display,
    resize_image,
    score_frame_quality_rows,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


try:  # pragma: no cover - availability depends on the local environment.
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None

try:  # pragma: no cover - availability depends on the local environment.
    import imageio.v3 as iio  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    iio = None


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VIDEO_ROOT = REPO_ROOT / "data" / "rheed_roi_shadow_right_v2_main_raw_crop_videos_256"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "rheed_frame_selection_mvp"
CRITICAL_FLAGS = {"almost_black", "almost_white", "over_saturated", "very_low_dynamic_range", "strong_shadow"}


@dataclass(frozen=True)
class VideoSource:
    path: Path
    sample_id: str
    sample_dir: Path
    matched_requested_glob: bool
    discovery_reason: str


@dataclass
class ProcessedVideoResult:
    sample_id: str
    video_path: Path
    sample_dir: Path
    frame_selection_dir: Path
    scanned_frame_count: int
    sampled_frame_count: int
    candidate_count: int
    low_confidence: bool
    failure: str = ""
    candidate_grid_path: Path | None = None
    overview_frame_path: Path | None = None


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}.")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def git_status_short() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError as exc:
        return f"<git status unavailable: {exc}>"
    text = result.stdout.strip()
    return text if text else "<clean>"


def run_command_output(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(command, cwd=REPO_ROOT, check=False, text=True, capture_output=True)
    except OSError as exc:
        return f"unavailable: {exc}"
    output = (result.stdout or result.stderr).strip()
    return output if output else "<no output>"


def infer_sample_id_from_video(path: Path) -> str:
    stem = path.stem
    for marker in ["_raw_crop", "-raw_crop", " raw_crop", "raw_crop"]:
        index = stem.lower().find(marker)
        if index > 0:
            stem = stem[:index]
            break
    stem = stem.replace("_256x256", "").replace("-256x256", "")
    cleaned = "".join(char for char in stem.strip() if char not in "\\/:*?\"<>|")
    return cleaned or path.stem


def infer_video_source(video_path: Path, video_root: Path, out_root: Path, matched: bool, reason: str) -> VideoSource:
    resolved_video = video_path.resolve()
    try:
        relative = resolved_video.relative_to(video_root.resolve())
        parts = relative.parts
    except ValueError:
        parts = ()
    if len(parts) >= 2 and parts[0].lower() != "videos":
        sample_id = parts[0]
        sample_dir = out_root / sample_id
    elif len(parts) >= 3 and parts[0].lower() == "videos":
        sample_id = infer_sample_id_from_video(resolved_video)
        sample_dir = out_root / sample_id
    elif resolved_video.parent.name.lower() == "videos" and resolved_video.parent.parent != video_root:
        sample_id = resolved_video.parent.parent.name
        sample_dir = out_root / sample_id
    else:
        sample_id = infer_sample_id_from_video(resolved_video)
        sample_dir = out_root / sample_id
    return VideoSource(
        path=resolved_video,
        sample_id=sample_id,
        sample_dir=sample_dir.resolve(),
        matched_requested_glob=matched,
        discovery_reason=reason,
    )


def discover_videos(
    video_root: Path,
    out_root: Path,
    *,
    video_glob: str,
    include_all_mp4: bool,
) -> tuple[list[VideoSource], list[dict[str, Any]], bool]:
    if not video_root.is_dir():
        raise FileNotFoundError(f"Video root does not exist or is not a directory: {video_root}")
    all_mp4 = sorted(path for path in video_root.rglob("*.mp4") if path.is_file() and not path.name.startswith("._"))
    matched = [path for path in all_mp4 if fnmatch.fnmatch(path.name, video_glob)]
    used_fallback = False
    if include_all_mp4:
        selected = all_mp4
        reason = "include_all_mp4"
    elif matched:
        selected = matched
        reason = "matched_video_glob"
    else:
        selected = all_mp4
        reason = "fallback_all_mp4"
        used_fallback = True

    sources = [
        infer_video_source(path, video_root, out_root, fnmatch.fnmatch(path.name, video_glob), reason)
        for path in selected
    ]
    selected_paths = {source.path for source in sources}
    inventory: list[dict[str, Any]] = []
    for path in all_mp4:
        source = infer_video_source(path, video_root, out_root, fnmatch.fnmatch(path.name, video_glob), reason)
        inventory.append(
            {
                "video_path": display_path(path),
                "sample_id": source.sample_id,
                "sample_dir": display_path(source.sample_dir),
                "matched_video_glob": int(source.matched_requested_glob),
                "selected_for_processing": int(path.resolve() in selected_paths),
                "discovery_reason": reason if path.resolve() in selected_paths else "not_selected",
            }
        )
    return sources, inventory, used_fallback


def require_video_backend() -> str:
    if cv2 is not None:
        return "cv2"
    if iio is not None:
        return "imageio"
    raise RuntimeError(
        "No video backend is available. Install opencv-python or imageio/imageio-ffmpeg to read RHEED videos."
    )


def iter_video_frames(
    video_path: Path,
    *,
    sample_every_n_frames: int,
    max_frames_per_video: int,
) -> Iterable[tuple[int, float, np.ndarray]]:
    backend = require_video_backend()
    sample_every_n_frames = max(1, int(sample_every_n_frames))
    max_frames_per_video = max(1, int(max_frames_per_video))
    yielded = 0
    if backend == "cv2":
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"cv2 could not open video: {video_path}")
        fps = finite_float(capture.get(cv2.CAP_PROP_FPS), 0.0)
        frame_idx = 0
        try:
            while yielded < max_frames_per_video:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_idx % sample_every_n_frames == 0:
                    timestamp_sec = frame_idx / fps if fps > 0 else float(yielded)
                    yield frame_idx, timestamp_sec, frame_to_gray_float32(frame)
                    yielded += 1
                frame_idx += 1
        finally:
            capture.release()
        return

    assert iio is not None
    fps = 0.0
    try:
        metadata = iio.immeta(video_path)
        fps = finite_float(metadata.get("fps", 0.0), 0.0)
    except Exception:
        fps = 0.0
    for frame_idx, frame in enumerate(iio.imiter(video_path)):
        if yielded >= max_frames_per_video:
            break
        if frame_idx % sample_every_n_frames != 0:
            continue
        timestamp_sec = frame_idx / fps if fps > 0 else float(yielded)
        yield frame_idx, timestamp_sec, frame_to_gray_float32(frame)
        yielded += 1


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, bool):
                    encoded[key] = int(value)
                elif isinstance(value, float):
                    encoded[key] = f"{value:.8g}"
                else:
                    encoded[key] = value
            writer.writerow(encoded)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.asarray(image), cmap="gray", vmin=0.0, vmax=1.0)


def frame_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_small = resize_image(np.asarray(a, dtype=np.float32), size=64).astype(np.float32, copy=False)
    b_small = resize_image(np.asarray(b, dtype=np.float32), size=64).astype(np.float32, copy=False)
    a_norm = (a_small - float(np.mean(a_small))) / max(float(np.std(a_small)), 1e-6)
    b_norm = (b_small - float(np.mean(b_small))) / max(float(np.std(b_small)), 1e-6)
    return finite_float(np.sqrt(np.mean((a_norm - b_norm) ** 2)) / 2.0)


def row_passes_critical_filters(row: dict[str, Any]) -> bool:
    return not any(bool(row.get(flag, False)) for flag in CRITICAL_FLAGS)


def select_candidate_rows(
    scored_rows: Sequence[dict[str, Any]],
    frame_images: dict[int, np.ndarray] | None,
    *,
    num_candidates: int,
    min_frame_gap: int,
    min_ssim_distance: float,
) -> list[dict[str, Any]]:
    ordered = sorted(scored_rows, key=lambda row: finite_float(row.get("quality_score", 0.0)), reverse=True)
    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()

    def compatible(row: dict[str, Any], *, enforce_gap: bool, enforce_distance: bool) -> bool:
        frame_idx = int(row.get("frame_idx", -1))
        if frame_idx in selected_indices:
            return False
        if enforce_gap:
            for existing in selected:
                if abs(frame_idx - int(existing.get("frame_idx", -1))) < min_frame_gap:
                    return False
        if enforce_distance and frame_images is not None and frame_idx in frame_images:
            for existing in selected:
                existing_idx = int(existing.get("frame_idx", -1))
                if existing_idx in frame_images and frame_distance(frame_images[frame_idx], frame_images[existing_idx]) < min_ssim_distance:
                    return False
        return True

    def add_rows(candidates: Sequence[dict[str, Any]], *, enforce_gap: bool, enforce_distance: bool, fill: bool) -> None:
        for row in candidates:
            if len(selected) >= num_candidates:
                return
            if not compatible(row, enforce_gap=enforce_gap, enforce_distance=enforce_distance):
                continue
            output = dict(row)
            output["low_confidence_candidate"] = bool(fill or not row_passes_critical_filters(row) or row.get("quality_score", 0.0) < 0.35)
            output["selection_note"] = "quality_ranked" if not output["low_confidence_candidate"] else "fill_or_flagged"
            selected.append(output)
            selected_indices.add(int(output.get("frame_idx", -1)))

    eligible = [row for row in ordered if row_passes_critical_filters(row)]
    add_rows(eligible, enforce_gap=True, enforce_distance=True, fill=False)
    add_rows(ordered, enforce_gap=True, enforce_distance=False, fill=True)
    add_rows(ordered, enforce_gap=False, enforce_distance=False, fill=True)

    ranked: list[dict[str, Any]] = []
    for rank, row in enumerate(selected[:num_candidates], start=1):
        output = dict(row)
        output["candidate_rank"] = rank
        output["flags"] = ";".join(active_flags(output))
        ranked.append(output)
    return ranked


def candidate_filename(row: dict[str, Any]) -> str:
    rank = int(row["candidate_rank"])
    frame_idx = int(row["frame_idx"])
    timestamp = finite_float(row.get("timestamp_sec", 0.0))
    score = finite_float(row.get("quality_score", 0.0))
    return f"rank{rank:02d}_frame{frame_idx:06d}_t{timestamp:07.2f}s_score{score:.3f}.png"


def write_manual_template(
    path: Path,
    *,
    sample_id: str,
    source_video: Path,
    overwrite_manual: bool = False,
) -> bool:
    if path.exists() and not overwrite_manual:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    source = display_path(source_video)
    content = f"""# Manual RHEED frame selection for sample: {sample_id}
# Source video: {source}
# Instructions:
#   After inspecting candidate_frames_grid.png, uncomment or add the frame(s) you want to use.
#   One selected frame per line.
#   Accepted formats:
#     rank01
#     frame_idx=123
#     rank01_frame000123_t004.10s_score0.873.png
#
# Recommended:
#   Choose 1 primary frame if one frame is clearly best.
#   Choose 2-3 frames only if several frames show genuinely different clear RHEED patterns.
#   Do not select frames that are shadowed, saturated, blurry, or missing diffraction features.
#
# selected frames below:
# rank01
"""
    path.write_text(content, encoding="utf-8")
    return True


def write_sample_readme(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# RHEED frame candidate selection

This folder contains automatically selected candidate frames for human inspection. The scores are diagnostics, not ground truth.

## Files

- `frame_quality_scores.csv`: every sampled frame with transparent feature values and component scores.
- `candidate_frames.csv`: selected candidate frames in rank order.
- `candidate_frames_grid.png`: raw-normalized candidates for visual inspection.
- `candidate_frames_grid_raw_and_equalized.png`: raw-normalized and contrast-enhanced views side by side.
- `frame_quality_timeseries.png`: score components over time with candidate markers.
- `candidates/`: individual candidate PNG files.
- `manual_selected_frames.txt`: edit this file after reviewing the grids.

## How to choose frames

Prefer frames that are bright enough, unsaturated, sharp, unshadowed, and visibly show diffraction streaks, spots, or texture. Choose one primary frame when possible; use two or three only when genuinely different clear patterns are useful.

## Manual curation

Uncomment or add one selection per line in `manual_selected_frames.txt`, for example `rank01` or `frame_idx=123`. A later manifest builder validates those selections against `candidate_frames.csv`.

This is a human-in-the-loop curation step. It does not train or validate a RHEED-to-AFM prediction model.
""",
        encoding="utf-8",
    )


def _grid_shape(count: int) -> tuple[int, int]:
    cols = min(4, max(1, math.ceil(math.sqrt(count))))
    rows = max(1, math.ceil(count / cols))
    return rows, cols


def write_candidate_grids(
    frame_selection_dir: Path,
    *,
    sample_id: str,
    source_video: Path,
    scored_count: int,
    candidates: Sequence[dict[str, Any]],
    frame_images: dict[int, np.ndarray],
    low_confidence: bool,
) -> tuple[Path, Path]:
    grid_path = frame_selection_dir / "candidate_frames_grid.png"
    pair_grid_path = frame_selection_dir / "candidate_frames_grid_raw_and_equalized.png"
    if not candidates:
        return grid_path, pair_grid_path

    rows, cols = _grid_shape(len(candidates))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.2), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, row in zip(axes.ravel(), candidates):
        frame_idx = int(row["frame_idx"])
        image = frame_images[frame_idx]
        axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        flags = row.get("flags", "")
        flag_text = f"\n{flags}" if flags else ""
        axis.set_title(
            f"rank{int(row['candidate_rank']):02d} frame {frame_idx}\n"
            f"t={finite_float(row.get('timestamp_sec', 0.0)):.2f}s score={finite_float(row.get('quality_score', 0.0)):.3f}"
            f"{flag_text}",
            fontsize=8,
        )
    confidence = " LOW CONFIDENCE" if low_confidence else ""
    fig.suptitle(
        f"{sample_id} | {source_video.name} | scanned {scored_count} frames | selected {len(candidates)}{confidence}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    frame_selection_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(grid_path, dpi=160)
    plt.close(fig)

    pair_rows = len(candidates)
    fig, axes = plt.subplots(pair_rows, 2, figsize=(7.0, max(2.6, pair_rows * 2.3)), squeeze=False)
    for row_index, candidate in enumerate(candidates):
        frame_idx = int(candidate["frame_idx"])
        raw = frame_images[frame_idx]
        enhanced = enhance_for_display(raw)
        for col, image, label in [(0, raw, "raw-normalized"), (1, enhanced, "enhanced")]:
            axis = axes[row_index, col]
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axis.axis("off")
            axis.set_title(
                f"rank{int(candidate['candidate_rank']):02d} frame {frame_idx} {label}\n"
                f"score={finite_float(candidate.get('quality_score', 0.0)):.3f}",
                fontsize=8,
            )
    fig.suptitle(f"{sample_id} raw/equalized candidate comparison{confidence}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(pair_grid_path, dpi=150)
    plt.close(fig)
    return grid_path, pair_grid_path


def write_timeseries(
    frame_selection_dir: Path,
    *,
    scored_rows: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
) -> Path:
    path = frame_selection_dir / "frame_quality_timeseries.png"
    if not scored_rows:
        return path
    x = np.asarray([int(row["frame_idx"]) for row in scored_rows], dtype=float)
    fig, axis = plt.subplots(figsize=(12, 5))
    for key, label in [
        ("quality_score", "quality"),
        ("brightness_score", "brightness"),
        ("sharpness_score", "sharpness"),
        ("shadow_penalty", "shadow penalty"),
        ("saturation_penalty", "saturation penalty"),
    ]:
        axis.plot(x, [finite_float(row.get(key, 0.0)) for row in scored_rows], label=label, linewidth=1.2)
    candidate_x = [int(row["frame_idx"]) for row in candidates]
    candidate_y = [finite_float(row.get("quality_score", 0.0)) for row in candidates]
    axis.scatter(candidate_x, candidate_y, marker="o", s=48, color="black", label="selected candidates", zorder=5)
    axis.set_xlabel("frame index")
    axis.set_ylabel("score / penalty")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def process_video(source: VideoSource, args: argparse.Namespace) -> ProcessedVideoResult:
    frame_selection_dir = source.sample_dir / "frame_selection"
    candidates_dir = frame_selection_dir / "candidates"
    source.sample_dir.mkdir(parents=True, exist_ok=True)
    frame_selection_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    for stale_png in candidates_dir.glob("rank*_frame*.png"):
        stale_png.unlink()

    scored_input: list[dict[str, Any]] = []
    frame_images: dict[int, np.ndarray] = {}
    scanned_frame_count = 0
    for sample_order, (frame_idx, timestamp_sec, gray) in enumerate(
        iter_video_frames(
            source.path,
            sample_every_n_frames=args.sample_every_n_frames,
            max_frames_per_video=args.max_frames_per_video,
        )
    ):
        scanned_frame_count += 1
        features = extract_frame_quality_features(gray)
        features.update(
            {
                "sample_id": source.sample_id,
                "video_path": display_path(source.path),
                "frame_idx": int(frame_idx),
                "sample_order": int(sample_order),
                "timestamp_sec": finite_float(timestamp_sec),
            }
        )
        scored_input.append(features)
        frame_images[int(frame_idx)] = resize_image(normalize_for_display(gray), size=args.display_size)

    if not scored_input:
        raise RuntimeError(f"No frames were decoded from {source.path}")

    scored_rows = score_frame_quality_rows(scored_input)
    candidates = select_candidate_rows(
        scored_rows,
        frame_images,
        num_candidates=args.num_candidates,
        min_frame_gap=args.min_frame_gap,
        min_ssim_distance=args.min_ssim_distance,
    )
    low_confidence = (
        len(candidates) < args.num_candidates
        or sum(row_passes_critical_filters(row) for row in scored_rows) < args.num_candidates
        or np.median([finite_float(row.get("quality_score", 0.0)) for row in candidates]) < 0.35
    )

    (frame_selection_dir / "source_video.txt").write_text(f"{display_path(source.path)}\n", encoding="utf-8")
    quality_fieldnames = [
        "sample_id",
        "video_path",
        "frame_idx",
        "sample_order",
        "timestamp_sec",
        *NUMERIC_FEATURE_KEYS,
        *SCORE_KEYS,
        *FLAG_KEYS,
    ]
    write_csv(frame_selection_dir / "frame_quality_scores.csv", scored_rows, quality_fieldnames)

    for candidate in candidates:
        frame_idx = int(candidate["frame_idx"])
        png_path = candidates_dir / candidate_filename(candidate)
        save_image(png_path, frame_images[frame_idx])
        candidate["candidate_png_path"] = display_path(png_path)

    candidate_fieldnames = [
        "sample_id",
        "video_path",
        "candidate_rank",
        "frame_idx",
        "timestamp_sec",
        "quality_score",
        "brightness_score",
        "dynamic_range_score",
        "sharpness_score",
        "pattern_visibility_score",
        "contrast_score",
        "shadow_penalty",
        "saturation_penalty",
        "blur_penalty",
        "low_dynamic_range_penalty",
        "flags",
        "low_confidence_candidate",
        "selection_note",
        "candidate_png_path",
    ]
    write_csv(frame_selection_dir / "candidate_frames.csv", candidates, candidate_fieldnames)
    grid_path, _ = write_candidate_grids(
        frame_selection_dir,
        sample_id=source.sample_id,
        source_video=source.path,
        scored_count=len(scored_rows),
        candidates=candidates,
        frame_images=frame_images,
        low_confidence=low_confidence,
    )
    write_timeseries(frame_selection_dir, scored_rows=scored_rows, candidates=candidates)
    write_manual_template(
        frame_selection_dir / "manual_selected_frames.txt",
        sample_id=source.sample_id,
        source_video=source.path,
        overwrite_manual=args.overwrite_manual,
    )
    write_sample_readme(frame_selection_dir / "README_frame_selection.md")

    if args.save_all_scored_thumbnails:
        thumb_dir = frame_selection_dir / "scored_thumbnails"
        thumb_dir.mkdir(parents=True, exist_ok=True)
        for row in scored_rows:
            frame_idx = int(row["frame_idx"])
            save_image(thumb_dir / f"frame{frame_idx:06d}_score{finite_float(row.get('quality_score', 0.0)):.3f}.png", frame_images[frame_idx])

    overview_path = Path(candidates[0]["candidate_png_path"]) if candidates else None
    if overview_path is not None and not overview_path.is_absolute():
        overview_path = REPO_ROOT / overview_path
    return ProcessedVideoResult(
        sample_id=source.sample_id,
        video_path=source.path,
        sample_dir=source.sample_dir,
        frame_selection_dir=frame_selection_dir,
        scanned_frame_count=scanned_frame_count,
        sampled_frame_count=len(scored_rows),
        candidate_count=len(candidates),
        low_confidence=low_confidence,
        candidate_grid_path=grid_path,
        overview_frame_path=overview_path,
    )


def write_global_overview(report_dir: Path, results: Sequence[ProcessedVideoResult]) -> Path:
    path = report_dir / "global_candidate_overview_grid.png"
    good_results = [result for result in results if not result.failure and result.overview_frame_path and result.overview_frame_path.is_file()]
    if not good_results:
        return path
    rows, cols = _grid_shape(len(good_results))
    cols = min(8, max(cols, 1))
    rows = math.ceil(len(good_results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.3), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, result in zip(axes.ravel(), good_results):
        image = plt.imread(result.overview_frame_path)
        axis.imshow(image, cmap="gray")
        axis.set_title(f"{result.sample_id}\n{'LOW' if result.low_confidence else 'OK'}", fontsize=7)
        axis.axis("off")
    fig.suptitle("Rank-1 candidate overview across processed RHEED videos", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    report_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_histograms(report_dir: Path, results: Sequence[ProcessedVideoResult]) -> Path:
    path = report_dir / "sample_quality_histograms.png"
    quality_scores: list[float] = []
    shadow_penalties: list[float] = []
    saturation_penalties: list[float] = []
    blur_penalties: list[float] = []
    for result in results:
        csv_path = result.frame_selection_dir / "candidate_frames.csv"
        if result.failure or not csv_path.is_file():
            continue
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                quality_scores.append(finite_float(row.get("quality_score", 0.0)))
                shadow_penalties.append(finite_float(row.get("shadow_penalty", 0.0)))
                saturation_penalties.append(finite_float(row.get("saturation_penalty", 0.0)))
                blur_penalties.append(finite_float(row.get("blur_penalty", 0.0)))
    if not quality_scores:
        return path
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), squeeze=False)
    for axis, values, title in [
        (axes[0, 0], quality_scores, "candidate quality score"),
        (axes[0, 1], shadow_penalties, "shadow penalty"),
        (axes[1, 0], saturation_penalties, "saturation penalty"),
        (axes[1, 1], blur_penalties, "blur penalty"),
    ]:
        axis.hist(values, bins=20, color="#4c78a8", alpha=0.85)
        axis.set_title(title)
        axis.set_xlim(0.0, 1.0)
        axis.grid(alpha=0.20)
    fig.tight_layout()
    report_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_global_tables(
    report_dir: Path,
    *,
    inventory_rows: Sequence[dict[str, Any]],
    results: Sequence[ProcessedVideoResult],
) -> tuple[Path, Path, Path]:
    inventory_path = report_dir / "video_inventory.csv"
    write_csv(
        inventory_path,
        inventory_rows,
        ["video_path", "sample_id", "sample_dir", "matched_video_glob", "selected_for_processing", "discovery_reason"],
    )
    summary_rows = [
        {
            "sample_id": result.sample_id,
            "video_path": display_path(result.video_path),
            "sample_dir": display_path(result.sample_dir),
            "frame_selection_dir": display_path(result.frame_selection_dir),
            "scanned_frame_count": result.scanned_frame_count,
            "sampled_frame_count": result.sampled_frame_count,
            "candidate_count": result.candidate_count,
            "low_confidence": int(result.low_confidence),
            "candidate_grid_path": display_path(result.candidate_grid_path) if result.candidate_grid_path else "",
            "failure": result.failure,
        }
        for result in results
    ]
    summary_path = report_dir / "frame_selection_summary.csv"
    write_csv(
        summary_path,
        summary_rows,
        [
            "sample_id",
            "video_path",
            "sample_dir",
            "frame_selection_dir",
            "scanned_frame_count",
            "sampled_frame_count",
            "candidate_count",
            "low_confidence",
            "candidate_grid_path",
            "failure",
        ],
    )
    failed_rows = [
        {
            "sample_id": result.sample_id,
            "video_path": display_path(result.video_path),
            "failure": result.failure,
        }
        for result in results
        if result.failure
    ]
    failed_path = report_dir / "failed_videos.csv"
    write_csv(failed_path, failed_rows, ["sample_id", "video_path", "failure"])
    return inventory_path, summary_path, failed_path


def collect_environment() -> dict[str, str]:
    env = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cv2": getattr(cv2, "__version__", "not available") if cv2 is not None else "not available",
        "imageio": "not available",
        "numpy": np.__version__,
        "scipy": "not available",
        "skimage": "not used",
    }
    if iio is not None:
        try:
            import imageio  # type: ignore

            env["imageio"] = getattr(imageio, "__version__", "available")
        except ModuleNotFoundError:
            env["imageio"] = "available"
    try:
        import scipy  # type: ignore

        env["scipy"] = getattr(scipy, "__version__", "available")
    except ModuleNotFoundError:
        pass
    return env


def write_report(
    report_dir: Path,
    *,
    args: argparse.Namespace,
    git_before: str,
    git_after: str,
    env: dict[str, str],
    inventory_rows: Sequence[dict[str, Any]],
    used_fallback: bool,
    results: Sequence[ProcessedVideoResult],
    global_overview_path: Path,
    histogram_path: Path,
    command: str,
) -> Path:
    report_path = report_dir / "codex_report.md"
    successes = [result for result in results if not result.failure]
    failures = [result for result in results if result.failure]
    low_confidence = [result for result in successes if result.low_confidence]
    examples = [display_path(result.candidate_grid_path) for result in successes[:5] if result.candidate_grid_path]
    common_failure_modes = "none" if not failures else "; ".join(sorted({result.failure for result in failures})[:5])
    raw_crop_count = sum(1 for row in inventory_rows if int(row["matched_video_glob"]) == 1)
    mp4_count = len(inventory_rows)
    report_dir.mkdir(parents=True, exist_ok=True)
    text = f"""# RHEED frame selection MVP report

Generated: {datetime.now(UTC).isoformat(timespec="seconds")}

## Scope

This task only selects candidate RHEED frames for human verification. It does not train or validate a RHEED-to-AFM prediction model.

## Git Status

Before run:

```text
{git_before}
```

After run:

```text
{git_after}
```

## Files Created Or Modified

- Per-sample `frame_selection/` folders under `{display_path(resolve_path(args.out_root))}`
- `{display_path(report_dir / "video_inventory.csv")}`
- `{display_path(report_dir / "frame_selection_summary.csv")}`
- `{display_path(report_dir / "failed_videos.csv")}`
- `{display_path(global_overview_path)}`
- `{display_path(histogram_path)}`
- `{display_path(report_path)}`

## Exact Command

```bash
{command}
```

## Environment

| package | version |
| --- | --- |
| python | {env["python"]} |
| platform | {env["platform"]} |
| cv2 | {env["cv2"]} |
| imageio | {env["imageio"]} |
| numpy | {env["numpy"]} |
| scipy | {env["scipy"]} |
| skimage | {env["skimage"]} |

## Input Data Inventory

- Video root: `{display_path(resolve_path(args.video_root))}`
- MP4 files found: {mp4_count}
- Files matching `{args.video_glob}`: {raw_crop_count}
- Used all MP4 fallback: {used_fallback}
- Videos processed successfully: {len(successes)}
- Failures: {len(failures)}
- Example video paths:
{chr(10).join(f"  - `{row['video_path']}`" for row in list(inventory_rows)[:5])}

## Sample Folder Behavior

Sample IDs are inferred from the first directory under the video root when videos already live inside sample folders, or from the video filename with the raw-crop suffix removed for shared video directories. Outputs are written to `<out-root>/<sample_id>/frame_selection/`. Source videos are never moved or modified.

## Frame Scoring Summary

Features used: brightness percentiles, dynamic range, edge and center intensity ratios, dark obstruction proxies, Laplacian variance, Tenengrad gradients, local contrast, entropy, FFT low/mid/high frequency power, FFT anisotropy, projection prominence, and projection entropy.

Quality score formula:

```text
quality_score =
  0.18 * brightness_score
  + 0.20 * dynamic_range_score
  + 0.22 * sharpness_score
  + 0.22 * pattern_visibility_score
  + 0.18 * contrast_score
  - 0.35 * shadow_penalty
  - 0.25 * saturation_penalty
  - 0.20 * blur_penalty
  - 0.20 * low_dynamic_range_penalty
```

Scores are clipped to `[0, 1]`, with component normalization performed within each video. Critical rejection flags are `almost_black`, `almost_white`, `over_saturated`, `very_low_dynamic_range`, and `strong_shadow`. The requested candidate count is {args.num_candidates} per video, with `min_frame_gap={args.min_frame_gap}` and `min_ssim_distance={args.min_ssim_distance}`.

## Quality Summary

- Low-confidence videos: {len(low_confidence)}
- Common failure modes: {common_failure_modes}
- Shadow, saturation, and blur statistics are summarized in `{display_path(histogram_path)}`.

## Output Examples

{chr(10).join(f"- `{path}`" for path in examples) if examples else "- No successful candidate grids were produced."}

Global overview grid: `{display_path(global_overview_path)}`

## Manual Selection Workflow

Open each sample's `frame_selection/candidate_frames_grid.png` and `candidate_frames_grid_raw_and_equalized.png`, then edit `manual_selected_frames.txt`. Uncomment `rank01` or add lines such as `frame_idx=123` for selected frame(s).

Build the manifest for future experiments with:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.manual_frame_selection --root {display_path(resolve_path(args.out_root))} --out {display_path(resolve_path(args.out_root) / "manual_selected_frame_manifest.csv")}
```

## Known Limitations

- The scoring is transparent image-quality triage, not physics-aware RHEED interpretation.
- Component normalization is per video, so scores are best compared within a sample.
- Similarity suppression uses lightweight image-distance checks for diversity.
- Manual review remains required before future experiments consume selected frames.

## Next Recommended Command

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.manual_frame_selection --root {display_path(resolve_path(args.out_root))} --out {display_path(resolve_path(args.out_root) / "manual_selected_frame_manifest.csv")}
```
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_VIDEO_ROOT))
    parser.add_argument("--video-glob", default="*raw_crop*.mp4")
    parser.add_argument("--include-all-mp4", type=str_to_bool, default=False)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--sample-every-n-frames", type=int, default=1)
    parser.add_argument("--max-frames-per-video", type=int, default=1200)
    parser.add_argument("--display-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-frame-gap", type=int, default=5)
    parser.add_argument("--min-ssim-distance", type=float, default=0.03)
    parser.add_argument("--strict", type=str_to_bool, default=False)
    parser.add_argument("--overwrite", type=str_to_bool, default=False)
    parser.add_argument("--overwrite-manual", type=str_to_bool, default=False)
    parser.add_argument("--save-all-scored-thumbnails", type=str_to_bool, default=False)
    parser.add_argument("--make-timeseries", type=str_to_bool, default=True)
    parser.add_argument("--make-contact-sheet", type=str_to_bool, default=True)
    parser.add_argument("--manual-template", type=str_to_bool, default=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-video-limit", type=int, default=3)
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)
    video_root = resolve_path(args.video_root)
    out_root = resolve_path(args.out_root)
    report_root = resolve_path(args.report_root)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_dir = report_root / timestamp
    git_before = git_status_short()
    env = collect_environment()
    command_args = sys.argv[1:] if argv is None else list(argv)
    command = (
        "PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.select_representative_frames "
        + shlex.join(command_args)
    )

    sources, inventory_rows, used_fallback = discover_videos(
        video_root,
        out_root,
        video_glob=args.video_glob,
        include_all_mp4=args.include_all_mp4,
    )
    if args.debug:
        sources = sources[: max(1, args.debug_video_limit)]
        print(f"[debug] Processing first {len(sources)} videos.")
    print(f"Discovered {len(inventory_rows)} MP4 files; processing {len(sources)} videos.")
    if used_fallback:
        print(f"No files matched {args.video_glob!r}; falling back to all MP4 files.")

    results: list[ProcessedVideoResult] = []
    for index, source in enumerate(sources, start=1):
        print(f"[{index}/{len(sources)}] {source.sample_id}: {display_path(source.path)}")
        try:
            result = process_video(source, args)
        except Exception as exc:
            if args.strict:
                raise
            result = ProcessedVideoResult(
                sample_id=source.sample_id,
                video_path=source.path,
                sample_dir=source.sample_dir,
                frame_selection_dir=source.sample_dir / "frame_selection",
                scanned_frame_count=0,
                sampled_frame_count=0,
                candidate_count=0,
                low_confidence=True,
                failure=f"{type(exc).__name__}: {exc}",
            )
            print(f"  failed: {result.failure}")
        results.append(result)

    inventory_path, summary_path, failed_path = write_global_tables(
        report_dir,
        inventory_rows=inventory_rows,
        results=results,
    )
    global_overview_path = write_global_overview(report_dir, results)
    histogram_path = write_histograms(report_dir, results)
    git_after_before_report = git_status_short()
    report_path = write_report(
        report_dir,
        args=args,
        git_before=git_before,
        git_after=git_after_before_report,
        env=env,
        inventory_rows=inventory_rows,
        used_fallback=used_fallback,
        results=results,
        global_overview_path=global_overview_path,
        histogram_path=histogram_path,
        command=command,
    )
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Git Status After Report Write\n\n```text\n")
        handle.write(git_status_short())
        handle.write("\n```\n")
    print(f"Wrote inventory: {display_path(inventory_path)}")
    print(f"Wrote summary: {display_path(summary_path)}")
    print(f"Wrote failures: {display_path(failed_path)}")
    print(f"Wrote report: {display_path(report_path)}")
    failures = sum(1 for result in results if result.failure)
    return 1 if failures and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
