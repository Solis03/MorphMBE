"""Prepare full-frame RHEED PNG libraries for manual keyframe selection.

This module exports every decodable frame from MP4/MOV videos under ``data/pair``
without model preprocessing. The generated metadata keeps manual keyframe
selection fields separate from reproducible extraction fields so reruns can
refresh video information without erasing human annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)
VIDEO_SUFFIXES = {".mp4", ".mov"}
REPORT_FIELDS = [
    "sample_id",
    "video_id",
    "source_video",
    "output_frames_dir",
    "status",
    "reported_frame_count",
    "extracted_frame_count",
    "fps",
    "width",
    "height",
    "duration_seconds",
    "image_format",
    "color_mode",
    "bit_depth",
    "source_size_bytes",
    "error_message",
]


@dataclass(frozen=True)
class SourceStat:
    """Stable source-file attributes used for idempotent extraction checks."""

    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class VideoSource:
    """An MP4 discovered under one sample directory."""

    sample_id: str
    sample_dir: Path
    source_path: Path
    video_id: str


@dataclass
class ReportRow:
    """One row in the extraction report."""

    sample_id: str
    video_id: str
    source_video: str
    output_frames_dir: str
    status: str
    reported_frame_count: int | None = None
    extracted_frame_count: int | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    image_format: str = "png"
    color_mode: str | None = None
    bit_depth: int | None = None
    source_size_bytes: int | None = None
    error_message: str = ""

    def as_csv_row(self) -> dict[str, Any]:
        """Return a CSV-ready row with newlines removed from error text."""

        row = self.__dict__.copy()
        row["error_message"] = str(row.get("error_message") or "").replace("\r", " ").replace("\n", " ")
        return row


@dataclass
class ExtractionSummary:
    """Batch summary printed at the end of extraction."""

    sample_count: int
    video_count: int
    success_count: int
    skipped_count: int
    failed_count: int
    incomplete_count: int
    total_png_count: int
    output_size_bytes: int
    failed_sources: list[str]
    report_path: Path | None


def repo_relative(path: Path, repo_root: Path) -> str:
    """Return a POSIX path relative to the repository root when possible."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def finite_float(value: Any) -> float | None:
    """Convert finite numeric metadata to ``float`` and normalize missing values."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def optional_int(value: Any) -> int | None:
    """Convert finite positive metadata to ``int`` and normalize unknown values."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed) or parsed < 0:
        return None
    return int(parsed)


def file_stat(path: Path) -> SourceStat:
    """Read source-file size and mtime without mutating the file."""

    stat = path.stat()
    return SourceStat(size_bytes=int(stat.st_size), mtime_ns=int(stat.st_mtime_ns))


def bit_depth_for_dtype(dtype: np.dtype[Any]) -> int:
    """Return the PNG bit depth implied by an integer frame dtype."""

    dtype = np.dtype(dtype)
    if dtype == np.uint8:
        return 8
    if dtype == np.uint16:
        return 16
    return int(dtype.itemsize * 8)


def to_rgb_array(frame: np.ndarray) -> np.ndarray:
    """Normalize decoder output to an RGB array without scientific preprocessing."""

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"Decoded frame is not a color image: shape={array.shape}")
    rgb = np.ascontiguousarray(array[:, :, :3])
    if not np.issubdtype(rgb.dtype, np.integer):
        raise ValueError(f"Decoded frame dtype is not integer pixel data: dtype={rgb.dtype}")
    return rgb


def write_png_rgb(path: Path, rgb: np.ndarray, png_compression: int) -> None:
    """Write an RGB PNG without palette conversion or lossy operations."""

    if rgb.dtype == np.uint8:
        Image.fromarray(rgb, mode="RGB").save(path, format="PNG", compress_level=png_compression)
        return
    imageio.imwrite(path, rgb, format="PNG", compress_level=png_compression)


def discover_videos(input_root: Path, sample_id: str | None = None, limit: int | None = None) -> list[VideoSource]:
    """Discover MP4/MOV files under top-level sample directories in deterministic order."""

    root = input_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing input root: {input_root}")

    sample_dirs = [path for path in sorted(root.iterdir(), key=lambda p: p.name) if path.is_dir()]
    if sample_id is not None:
        sample_dirs = [path for path in sample_dirs if path.name == sample_id]
        if not sample_dirs:
            raise ValueError(f"Sample id not found under {input_root}: {sample_id}")

    discovered: list[VideoSource] = []
    for sample_dir in sample_dirs:
        video_files = [
            path
            for path in sorted(sample_dir.rglob("*"), key=lambda p: p.relative_to(sample_dir).as_posix())
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ]
        stem_counts: dict[str, int] = {}
        for path in video_files:
            stem_counts[path.stem] = stem_counts.get(path.stem, 0) + 1
        for path in video_files:
            video_id = path.stem or "video"
            if stem_counts[path.stem] > 1:
                import hashlib

                rel = path.relative_to(sample_dir).as_posix()
                digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]
                video_id = f"{video_id}__{digest}"
            discovered.append(
                VideoSource(
                    sample_id=sample_dir.name,
                    sample_dir=sample_dir,
                    source_path=path,
                    video_id=video_id,
                )
            )

    discovered.sort(key=lambda item: (item.sample_id, item.source_path.as_posix(), item.video_id))
    if limit is not None:
        discovered = discovered[: max(0, limit)]
    return discovered


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty object when the file is absent."""

    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a standard JSON object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False)
    tmp_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def existing_selection(existing_metadata: dict[str, Any], video_id: str, reset_selection: bool) -> dict[str, Any]:
    """Preserve or reset manual selection fields for one video."""

    if reset_selection:
        return {"keyframe_index": None, "clip_frame_count": None}
    existing_video = existing_metadata.get("videos", {}).get(video_id, {})
    selection = existing_video.get("selection", {}) if isinstance(existing_video, dict) else {}
    return {
        "keyframe_index": selection.get("keyframe_index"),
        "clip_frame_count": selection.get("clip_frame_count"),
    }


def build_sample_metadata(
    sample_id: str,
    source_sample_dir: Path,
    repo_root: Path,
    existing_metadata: dict[str, Any],
    reset_selection: bool,
) -> dict[str, Any]:
    """Create the stable sample-level metadata shell."""

    notes = "" if reset_selection else str(existing_metadata.get("notes", ""))
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "source_sample_dir": repo_relative(source_sample_dir, repo_root),
        "videos": {},
        "notes": notes,
    }


def build_video_metadata(
    video: VideoSource,
    frames_dir_rel: str,
    repo_root: Path,
    stat_before: SourceStat,
    reported_frame_count: int | None,
    extracted_frame_count: int | None,
    width: int | None,
    height: int | None,
    fps: float | None,
    duration_seconds: float | None,
    codec: str | None,
    pixel_format: str | None,
    png_compression: int,
    color_mode: str | None,
    bit_depth: int | None,
    completed: bool,
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Build one video's metadata object."""

    return {
        "video_id": video.video_id,
        "source_video": repo_relative(video.source_path, repo_root),
        "frames_dir": frames_dir_rel,
        "frame_index_base": 0,
        "frame_filename_pattern": "{index}.png",
        "video_info": {
            "codec": codec,
            "pixel_format": pixel_format,
            "fps": fps,
            "reported_frame_count": reported_frame_count,
            "extracted_frame_count": extracted_frame_count,
            "width": width,
            "height": height,
            "duration_seconds": duration_seconds,
            "source_size_bytes": stat_before.size_bytes,
            "source_mtime_ns": stat_before.mtime_ns,
        },
        "extraction": {
            "completed": completed,
            "image_format": "png",
            "lossless_output": True,
            "png_compression": png_compression,
            "color_mode": color_mode,
            "bit_depth": bit_depth,
            "preprocessing_applied": [],
        },
        "selection": selection,
    }


def complete_png_indices(frames_dir: Path, expected_count: int) -> bool:
    """Return true when exactly ``0.png`` through ``N-1.png`` are present."""

    if expected_count < 0 or not frames_dir.is_dir():
        return False
    png_files = {path.name for path in frames_dir.glob("*.png")}
    expected = {f"{index}.png" for index in range(expected_count)}
    return png_files == expected


def can_skip(video_metadata: dict[str, Any], frames_dir: Path, stat_before: SourceStat) -> bool:
    """Return true when existing extraction is complete for the unchanged source."""

    try:
        info = video_metadata["video_info"]
        extraction = video_metadata["extraction"]
        expected_count = int(info["extracted_frame_count"])
    except (KeyError, TypeError, ValueError):
        return False
    if not extraction.get("completed"):
        return False
    if int(info.get("source_size_bytes", -1)) != stat_before.size_bytes:
        return False
    if int(info.get("source_mtime_ns", -1)) != stat_before.mtime_ns:
        return False
    return complete_png_indices(frames_dir, expected_count)


def validate_png_frames(frames_dir: Path, expected_count: int, width: int, height: int) -> None:
    """Validate saved PNG names, dimensions, and truecolor readability."""

    if expected_count <= 0:
        raise ValueError("No frames were extracted.")
    if not complete_png_indices(frames_dir, expected_count):
        raise ValueError(f"PNG index set is incomplete in {frames_dir}")
    for index in range(expected_count):
        png_path = frames_dir / f"{index}.png"
        with Image.open(png_path) as image:
            image.load()
            if image.format != "PNG":
                raise ValueError(f"Not a PNG file: {png_path}")
            if image.size != (width, height):
                raise ValueError(f"Unexpected PNG size for {png_path}: {image.size}, expected {(width, height)}")
            if image.mode in {"L", "LA", "P", "1"}:
                raise ValueError(f"PNG is not stored as a color image: {png_path} mode={image.mode}")
            if image.mode != "RGB":
                raise ValueError(f"PNG mode is {image.mode}, expected RGB: {png_path}")


def compare_sampled_pixels(video_path: Path, frames_dir: Path, extracted_count: int) -> None:
    """Compare a small deterministic set of decoded frames with saved PNGs."""

    sample_indices = sorted({0, extracted_count // 2, extracted_count - 1})
    remaining = set(sample_indices)
    reader = imageio.get_reader(str(video_path), "ffmpeg")
    try:
        for index, frame in enumerate(reader):
            if index not in remaining:
                continue
            decoded = to_rgb_array(frame)
            with Image.open(frames_dir / f"{index}.png") as image:
                saved = np.asarray(image)
            if decoded.shape != saved.shape or not np.array_equal(decoded, saved):
                raise ValueError(f"PNG pixel mismatch against decoded frame {index} for {video_path}")
            remaining.remove(index)
            if not remaining:
                break
        if remaining:
            raise ValueError(f"Could not re-decode sampled frame indices {sorted(remaining)} for {video_path}")
    finally:
        reader.close()


def replace_directory_atomically(source_dir: Path, target_dir: Path) -> None:
    """Replace a directory after successful validation, restoring on failure."""

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    backup_dir: Path | None = None
    if target_dir.exists():
        backup_dir = target_dir.with_name(f".{target_dir.name}.backup.{os.getpid()}")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        os.replace(target_dir, backup_dir)
    try:
        os.replace(source_dir, target_dir)
    except Exception:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        if backup_dir is not None and backup_dir.exists():
            os.replace(backup_dir, target_dir)
        raise
    if backup_dir is not None and backup_dir.exists():
        shutil.rmtree(backup_dir)


def reader_metadata(path: Path) -> tuple[dict[str, Any], Any]:
    """Open a reader and return best-effort metadata with the live reader."""

    reader = imageio.get_reader(str(path), "ffmpeg")
    try:
        metadata = dict(reader.get_meta_data())
    except Exception:
        metadata = {}
    return metadata, reader


def extract_video_to_temp(
    video: VideoSource,
    temp_frames_dir: Path,
    png_compression: int,
) -> tuple[dict[str, Any], int, int, int, str, int]:
    """Decode every frame from one MP4 and write sequential PNGs to a temp dir."""

    temp_frames_dir.mkdir(parents=True, exist_ok=True)
    metadata, reader = reader_metadata(video.source_path)
    width: int | None = None
    height: int | None = None
    color_mode = "RGB"
    bit_depth: int | None = None
    extracted_count = 0
    try:
        for frame in reader:
            rgb = to_rgb_array(frame)
            if width is None or height is None:
                height = int(rgb.shape[0])
                width = int(rgb.shape[1])
                bit_depth = bit_depth_for_dtype(rgb.dtype)
            elif (height, width) != (int(rgb.shape[0]), int(rgb.shape[1])):
                raise ValueError(
                    f"Frame {extracted_count} size changed from {(width, height)} "
                    f"to {(int(rgb.shape[1]), int(rgb.shape[0]))}"
                )
            write_png_rgb(temp_frames_dir / f"{extracted_count}.png", rgb, png_compression)
            extracted_count += 1
    finally:
        reader.close()
    if width is None or height is None or bit_depth is None:
        raise ValueError(f"No frames decoded from {video.source_path}")
    return metadata, extracted_count, width, height, color_mode, bit_depth


def video_info_from_metadata(metadata: dict[str, Any]) -> tuple[int | None, float | None, float | None, str | None, str | None]:
    """Extract reported frame count, fps, duration, codec, and pixel format."""

    reported_frame_count = optional_int(metadata.get("nframes"))
    fps = finite_float(metadata.get("fps"))
    duration_seconds = finite_float(metadata.get("duration"))
    codec = metadata.get("codec") or metadata.get("video_codec")
    pixel_format = metadata.get("pix_fmt") or metadata.get("source_pix_fmt")
    return (
        reported_frame_count,
        fps,
        duration_seconds,
        str(codec) if codec is not None else None,
        str(pixel_format) if pixel_format is not None else None,
    )


def output_tree_size(path: Path) -> int:
    """Return total bytes under an output tree."""

    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def process_video(
    video: VideoSource,
    output_root: Path,
    repo_root: Path,
    png_compression: int,
    overwrite: bool,
    reset_selection: bool,
) -> ReportRow:
    """Process one MP4, updating sample metadata and returning a report row."""

    sample_out = output_root / video.sample_id
    metadata_path = sample_out / "metadata.json"
    existing_metadata = load_json(metadata_path)
    sample_metadata = build_sample_metadata(video.sample_id, video.sample_dir, repo_root, existing_metadata, reset_selection)
    existing_videos = existing_metadata.get("videos", {})
    if isinstance(existing_videos, dict):
        sample_metadata["videos"].update(existing_videos)
    selection = existing_selection(existing_metadata, video.video_id, reset_selection)

    video_out = sample_out / "videos" / video.video_id
    frames_dir = video_out / "frames"
    frames_dir_rel = Path("videos") / video.video_id / "frames"
    source_rel = repo_relative(video.source_path, repo_root)
    frames_rel = repo_relative(frames_dir, repo_root)
    stat_before = file_stat(video.source_path)

    if not overwrite and isinstance(sample_metadata["videos"].get(video.video_id), dict):
        existing_video = sample_metadata["videos"][video.video_id]
        if can_skip(existing_video, frames_dir, stat_before):
            LOGGER.info("Skipping unchanged completed video: %s", source_rel)
            info = existing_video.get("video_info", {})
            extraction = existing_video.get("extraction", {})
            return ReportRow(
                sample_id=video.sample_id,
                video_id=video.video_id,
                source_video=source_rel,
                output_frames_dir=frames_rel,
                status="skipped",
                reported_frame_count=info.get("reported_frame_count"),
                extracted_frame_count=info.get("extracted_frame_count"),
                fps=info.get("fps"),
                width=info.get("width"),
                height=info.get("height"),
                duration_seconds=info.get("duration_seconds"),
                color_mode=extraction.get("color_mode"),
                bit_depth=extraction.get("bit_depth"),
                source_size_bytes=stat_before.size_bytes,
            )

    temp_parent = video_out
    temp_frames_dir: Path | None = None
    try:
        temp_parent.mkdir(parents=True, exist_ok=True)
        temp_frames_dir = Path(tempfile.mkdtemp(prefix=".frames.", dir=temp_parent))
        metadata, extracted_count, width, height, color_mode, bit_depth = extract_video_to_temp(
            video=video,
            temp_frames_dir=temp_frames_dir,
            png_compression=png_compression,
        )
        reported_frame_count, fps, duration_seconds, codec, pixel_format = video_info_from_metadata(metadata)
        validate_png_frames(temp_frames_dir, extracted_count, width, height)
        compare_sampled_pixels(video.source_path, temp_frames_dir, extracted_count)
        stat_after = file_stat(video.source_path)
        if stat_after != stat_before:
            raise ValueError(f"Source MP4 changed during extraction: {source_rel}")

        replace_directory_atomically(temp_frames_dir, frames_dir)
        temp_frames_dir = None
        sample_metadata["videos"][video.video_id] = build_video_metadata(
            video=video,
            frames_dir_rel=frames_dir_rel.as_posix(),
            repo_root=repo_root,
            stat_before=stat_before,
            reported_frame_count=reported_frame_count,
            extracted_frame_count=extracted_count,
            width=width,
            height=height,
            fps=fps,
            duration_seconds=duration_seconds,
            codec=codec,
            pixel_format=pixel_format,
            png_compression=png_compression,
            color_mode=color_mode,
            bit_depth=bit_depth,
            completed=True,
            selection=selection,
        )
        atomic_write_json(metadata_path, sample_metadata)
        LOGGER.info("Extracted %s frame(s): %s", extracted_count, source_rel)
        return ReportRow(
            sample_id=video.sample_id,
            video_id=video.video_id,
            source_video=source_rel,
            output_frames_dir=frames_rel,
            status="success",
            reported_frame_count=reported_frame_count,
            extracted_frame_count=extracted_count,
            fps=fps,
            width=width,
            height=height,
            duration_seconds=duration_seconds,
            color_mode=color_mode,
            bit_depth=bit_depth,
            source_size_bytes=stat_before.size_bytes,
        )
    except Exception as exc:
        LOGGER.error("Failed to extract %s: %s", source_rel, exc)
        if temp_frames_dir is not None:
            shutil.rmtree(temp_frames_dir, ignore_errors=True)
        status = "incomplete" if frames_dir.exists() else "failed"
        sample_metadata["videos"][video.video_id] = build_video_metadata(
            video=video,
            frames_dir_rel=frames_dir_rel.as_posix(),
            repo_root=repo_root,
            stat_before=stat_before,
            reported_frame_count=None,
            extracted_frame_count=None,
            width=None,
            height=None,
            fps=None,
            duration_seconds=None,
            codec=None,
            pixel_format=None,
            png_compression=png_compression,
            color_mode=None,
            bit_depth=None,
            completed=False,
            selection=selection,
        )
        atomic_write_json(metadata_path, sample_metadata)
        return ReportRow(
            sample_id=video.sample_id,
            video_id=video.video_id,
            source_video=source_rel,
            output_frames_dir=frames_rel,
            status=status,
            source_size_bytes=stat_before.size_bytes,
            error_message=str(exc),
        )


def write_report(output_root: Path, rows: Sequence[ReportRow]) -> Path:
    """Write a deterministic CSV extraction report."""

    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "extraction_report.csv"
    rows_sorted = sorted(rows, key=lambda row: (row.sample_id, row.video_id, row.source_video))
    tmp_path = report_path.with_name(f".{report_path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow(row.as_csv_row())
    os.replace(tmp_path, report_path)
    return report_path


def print_dry_run(videos: Sequence[VideoSource], output_root: Path, repo_root: Path) -> None:
    """Print discovered samples and input-to-output mappings without writing files."""

    sample_ids = sorted({video.sample_id for video in videos})
    print(f"Discovered samples with MP4/MOV: {len(sample_ids)}")
    for sample_id in sample_ids:
        print(f"sample {sample_id}")
        for video in [item for item in videos if item.sample_id == sample_id]:
            frames_dir = output_root / video.sample_id / "videos" / video.video_id / "frames"
            print(
                "  process "
                f"{repo_relative(video.source_path, repo_root)} -> {repo_relative(frames_dir, repo_root)}"
            )


def summarize(videos: Sequence[VideoSource], rows: Sequence[ReportRow], output_root: Path, report_path: Path | None) -> ExtractionSummary:
    """Build aggregate counts for CLI output and tests."""

    status_counts = {status: 0 for status in ("success", "skipped", "failed", "incomplete")}
    total_png_count = 0
    failed_sources: list[str] = []
    for row in rows:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1
        if row.extracted_frame_count is not None and row.status in {"success", "skipped"}:
            total_png_count += int(row.extracted_frame_count)
        if row.status in {"failed", "incomplete"}:
            failed_sources.append(row.source_video)
    return ExtractionSummary(
        sample_count=len({video.sample_id for video in videos}),
        video_count=len(videos),
        success_count=status_counts["success"],
        skipped_count=status_counts["skipped"],
        failed_count=status_counts["failed"],
        incomplete_count=status_counts["incomplete"],
        total_png_count=total_png_count,
        output_size_bytes=output_tree_size(output_root),
        failed_sources=sorted(failed_sources),
        report_path=report_path,
    )


def print_summary(summary: ExtractionSummary) -> None:
    """Print the required final batch summary."""

    print(f"samples: {summary.sample_count}")
    print(f"videos: {summary.video_count}")
    print(f"success: {summary.success_count}")
    print(f"skipped: {summary.skipped_count}")
    print(f"failed: {summary.failed_count}")
    print(f"incomplete: {summary.incomplete_count}")
    print(f"total_png_count: {summary.total_png_count}")
    print(f"output_size_bytes: {summary.output_size_bytes}")
    if summary.report_path is not None:
        print(f"report: {summary.report_path}")
    if summary.failed_sources:
        print("failed_or_incomplete_sources:")
        for source in summary.failed_sources:
            print(f"  {source}")


def run_extraction(
    input_root: Path,
    output_root: Path,
    repo_root: Path,
    dry_run: bool = False,
    overwrite: bool = False,
    reset_selection: bool = False,
    sample_id: str | None = None,
    limit: int | None = None,
    png_compression: int = 3,
) -> ExtractionSummary:
    """Run the RHEED keyframe PNG preparation workflow."""

    if not 0 <= png_compression <= 9:
        raise ValueError("--png-compression must be between 0 and 9")
    videos = discover_videos(input_root=input_root, sample_id=sample_id, limit=limit)
    if dry_run:
        print_dry_run(videos, output_root, repo_root)
        return summarize(videos, [], output_root, report_path=None)

    rows: list[ReportRow] = []
    for video in videos:
        rows.append(
            process_video(
                video=video,
                output_root=output_root,
                repo_root=repo_root,
                png_compression=png_compression,
                overwrite=overwrite,
                reset_selection=reset_selection,
            )
        )
    report_path = write_report(output_root, rows)
    return summarize(videos, rows, output_root, report_path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/pair"), help="Input paired sample root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/rheed_keyframe_selection"),
        help="Output root for PNG frames and metadata.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print discovered work without writing outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate PNG frames even when extraction is complete.")
    parser.add_argument(
        "--reset-selection",
        action="store_true",
        help="Reset manual selection fields to null. By default they are preserved.",
    )
    parser.add_argument("--sample-id", default=None, help="Only process one sample id.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N discovered MP4 files.")
    parser.add_argument(
        "--png-compression",
        type=int,
        default=3,
        help="PNG compression level 0-9. This affects file size and speed, not pixels.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root for relative metadata paths.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main(argv: Iterable[str] | None = None) -> ExtractionSummary:
    """CLI entrypoint."""

    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    repo_root = args.repo_root.expanduser().resolve()
    summary = run_extraction(
        input_root=args.input_root,
        output_root=args.output_root,
        repo_root=repo_root,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        reset_selection=args.reset_selection,
        sample_id=args.sample_id,
        limit=args.limit,
        png_compression=args.png_compression,
    )
    print_summary(summary)
    return summary


if __name__ == "__main__":
    main()
