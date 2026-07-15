"""Manual RHEED keyframe and ROI selection helpers.

The GUI in ``tools/manual_rheed_roi_reviewer.py`` uses this module for all
metadata, coordinate, sorting, validation, and preview operations. Keeping these
parts independent from Tkinter makes the reviewer easier to test and keeps the
saved ROI coordinates tied to source-frame pixels, not display pixels.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class DisplayTransform:
    """Mapping between source-frame pixels and a fitted display rectangle."""

    source_width: int
    source_height: int
    display_x: float
    display_y: float
    display_width: float
    display_height: float

    @property
    def scale_x(self) -> float:
        return self.display_width / self.source_width

    @property
    def scale_y(self) -> float:
        return self.display_height / self.source_height

    def display_to_source_point(self, display_x: float, display_y: float) -> tuple[int, int]:
        """Map one display point to clamped source-frame pixel coordinates."""

        x = round((display_x - self.display_x) / self.scale_x)
        y = round((display_y - self.display_y) / self.scale_y)
        return (
            max(0, min(self.source_width, int(x))),
            max(0, min(self.source_height, int(y))),
        )

    def source_to_display_rect(self, roi: "ROI") -> tuple[float, float, float, float]:
        """Map a source-frame ROI to display rectangle coordinates."""

        return (
            self.display_x + roi.x * self.scale_x,
            self.display_y + roi.y * self.scale_y,
            self.display_x + (roi.x + roi.width) * self.scale_x,
            self.display_y + (roi.y + roi.height) * self.scale_y,
        )


@dataclass(frozen=True)
class ROI:
    """A rectangular region in source-frame pixel coordinates."""

    x: int
    y: int
    width: int
    height: int
    source_width: int
    source_height: int
    reference_frame_index: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @classmethod
    def from_points(
        cls,
        first: tuple[int, int],
        second: tuple[int, int],
        source_width: int,
        source_height: int,
        reference_frame_index: int,
    ) -> "ROI":
        """Build a normalized ROI from two source-frame points."""

        x0, y0 = first
        x1, y1 = second
        left = max(0, min(source_width, min(x0, x1)))
        top = max(0, min(source_height, min(y0, y1)))
        right = max(0, min(source_width, max(x0, x1)))
        bottom = max(0, min(source_height, max(y0, y1)))
        return cls(
            x=int(left),
            y=int(top),
            width=int(right - left),
            height=int(bottom - top),
            source_width=int(source_width),
            source_height=int(source_height),
            reference_frame_index=int(reference_frame_index),
        )

    @classmethod
    def from_metadata(cls, payload: dict[str, Any]) -> "ROI":
        """Restore ROI from metadata."""

        return cls(
            x=int(payload["x"]),
            y=int(payload["y"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            source_width=int(payload["source_width"]),
            source_height=int(payload["source_height"]),
            reference_frame_index=int(payload["reference_frame_index"]),
        )

    def to_metadata(self) -> dict[str, Any]:
        """Serialize ROI using source-frame pixel and normalized coordinates."""

        return {
            "reference_frame_index": int(self.reference_frame_index),
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
            "source_width": int(self.source_width),
            "source_height": int(self.source_height),
            "coordinate_space": "source_frame_pixels",
            "x_normalized": self.x / self.source_width,
            "y_normalized": self.y / self.source_height,
            "width_normalized": self.width / self.source_width,
            "height_normalized": self.height / self.source_height,
        }


@dataclass(frozen=True)
class VideoRecord:
    """One reviewable video entry."""

    sample_id: str
    video_id: str
    metadata_path: Path
    sample_dir: Path
    frames_dir: Path
    video_payload: dict[str, Any]


def numeric_frame_index(path: Path) -> int:
    """Return the integer frame index encoded in a PNG filename."""

    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Frame filename is not a decimal index: {path.name}") from exc


def sorted_frame_paths(frames_dir: Path) -> list[Path]:
    """Return PNG frames sorted by numeric index, not lexicographic order."""

    frames = [path for path in frames_dir.glob("*.png") if path.is_file()]
    return sorted(frames, key=numeric_frame_index)


def frame_path_for_index(frames_dir: Path, index: int) -> Path:
    """Return the expected frame path for one 0-based index."""

    return frames_dir / f"{int(index)}.png"


def load_json(path: Path) -> dict[str, Any]:
    """Load a metadata JSON object."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write metadata JSON while preserving standard formatting."""

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


def discover_video_records(root: Path) -> list[VideoRecord]:
    """Discover all videos under a keyframe-selection root."""

    root = root.expanduser().resolve()
    records: list[VideoRecord] = []
    for metadata_path in sorted(root.glob("*/metadata.json"), key=lambda p: p.parent.name):
        metadata = load_json(metadata_path)
        sample_id = str(metadata.get("sample_id", metadata_path.parent.name))
        videos = metadata.get("videos", {})
        if not isinstance(videos, dict):
            continue
        for video_id in sorted(videos):
            payload = videos[video_id]
            if not isinstance(payload, dict):
                continue
            frames_dir = metadata_path.parent / str(payload.get("frames_dir", ""))
            records.append(
                VideoRecord(
                    sample_id=sample_id,
                    video_id=str(video_id),
                    metadata_path=metadata_path,
                    sample_dir=metadata_path.parent,
                    frames_dir=frames_dir,
                    video_payload=payload,
                )
            )
    return records


def selection_from_record(record: VideoRecord) -> dict[str, Any]:
    """Return a copy of the current selection object for one video."""

    selection = record.video_payload.get("selection", {})
    return dict(selection) if isinstance(selection, dict) else {}


def roi_from_selection(selection: dict[str, Any]) -> ROI | None:
    """Restore a saved ROI, if present and complete."""

    roi_payload = selection.get("roi")
    if not isinstance(roi_payload, dict):
        return None
    return ROI.from_metadata(roi_payload)


def video_is_complete(record: VideoRecord) -> bool:
    """Return true when keyframe, clip count, and ROI are present and valid."""

    selection = selection_from_record(record)
    try:
        keyframe_index = int(selection["keyframe_index"])
        clip_frame_count = int(selection["clip_frame_count"])
        roi = roi_from_selection(selection)
        if roi is None:
            return False
        validate_selection(
            frames_dir=record.frames_dir,
            keyframe_index=keyframe_index,
            clip_frame_count=clip_frame_count,
            roi=roi,
        )
    except Exception:
        return False
    return True


def select_start_index(records: list[VideoRecord], start_from_first: bool, sample_id: str | None, video_id: str | None) -> int:
    """Select the initial video index for the reviewer."""

    if not records:
        return 0
    for index, record in enumerate(records):
        if sample_id is not None and record.sample_id != sample_id:
            continue
        if video_id is not None and record.video_id != video_id:
            continue
        return index
    if start_from_first:
        return 0
    for index, record in enumerate(records):
        if not video_is_complete(record):
            return index
    return 0


def filtered_records(records: Iterable[VideoRecord], sample_id: str | None, video_id: str | None) -> list[VideoRecord]:
    """Apply optional sample/video filters."""

    output = []
    for record in records:
        if sample_id is not None and record.sample_id != sample_id:
            continue
        if video_id is not None and record.video_id != video_id:
            continue
        output.append(record)
    return output


def fit_display_transform(source_width: int, source_height: int, box_width: int, box_height: int) -> DisplayTransform:
    """Fit a source image into a display box while preserving aspect ratio."""

    if source_width <= 0 or source_height <= 0:
        raise ValueError("Source dimensions must be positive.")
    if box_width <= 0 or box_height <= 0:
        raise ValueError("Display box dimensions must be positive.")
    scale = min(box_width / source_width, box_height / source_height)
    display_width = source_width * scale
    display_height = source_height * scale
    return DisplayTransform(
        source_width=source_width,
        source_height=source_height,
        display_x=(box_width - display_width) / 2.0,
        display_y=(box_height - display_height) / 2.0,
        display_width=display_width,
        display_height=display_height,
    )


def validate_roi(roi: ROI) -> None:
    """Validate that an ROI is non-empty and inside the source frame."""

    if roi.width <= 0 or roi.height <= 0:
        raise ValueError("ROI width and height must be greater than 0.")
    if roi.source_width <= 0 or roi.source_height <= 0:
        raise ValueError("Source dimensions must be positive.")
    if roi.x < 0 or roi.y < 0:
        raise ValueError("ROI x/y must be non-negative.")
    if roi.x2 > roi.source_width or roi.y2 > roi.source_height:
        raise ValueError("ROI must be fully inside the source frame.")


def validate_selection(frames_dir: Path, keyframe_index: int, clip_frame_count: int, roi: ROI) -> None:
    """Validate keyframe, clip count, and ROI before metadata save."""

    if keyframe_index < 0:
        raise ValueError("keyframe_index must be non-negative.")
    if not frame_path_for_index(frames_dir, keyframe_index).is_file():
        raise ValueError(f"Keyframe PNG does not exist: {frame_path_for_index(frames_dir, keyframe_index)}")
    if clip_frame_count <= 0:
        raise ValueError("clip_frame_count must be a positive integer.")
    validate_roi(roi)


def clip_indices(keyframe_index: int, clip_frame_count: int, min_index: int, max_index: int) -> list[int]:
    """Return actual in-bounds clip frame indices centered around keyframe when possible."""

    if clip_frame_count <= 0:
        return []
    before = (clip_frame_count - 1) // 2
    after = clip_frame_count - 1 - before
    start = keyframe_index - before
    end = keyframe_index + after
    if start < min_index:
        end += min_index - start
        start = min_index
    if end > max_index:
        start -= end - max_index
        end = max_index
    start = max(min_index, start)
    end = min(max_index, end)
    return list(range(start, end + 1))


def update_video_selection(
    metadata_path: Path,
    video_id: str,
    keyframe_index: int,
    clip_frame_count: int,
    roi: ROI,
) -> dict[str, Any]:
    """Atomically update one video's selection without touching other videos."""

    metadata = load_json(metadata_path)
    videos = metadata.get("videos")
    if not isinstance(videos, dict) or video_id not in videos:
        raise ValueError(f"Video id {video_id!r} not found in {metadata_path}")
    video_payload = videos[video_id]
    if not isinstance(video_payload, dict):
        raise ValueError(f"Video payload for {video_id!r} is invalid.")
    selection = dict(video_payload.get("selection", {})) if isinstance(video_payload.get("selection"), dict) else {}
    selection["keyframe_index"] = int(keyframe_index)
    selection["clip_frame_count"] = int(clip_frame_count)
    selection["roi"] = roi.to_metadata()
    video_payload["selection"] = selection
    atomic_write_json(metadata_path, metadata)
    return metadata


def crop_roi_pixels(frame_path: Path, roi: ROI) -> Image.Image:
    """Crop an ROI from a PNG using exact source-pixel coordinates."""

    validate_roi(roi)
    with Image.open(frame_path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (roi.source_width, roi.source_height):
            raise ValueError(f"Frame size {rgb.size} does not match ROI source size {(roi.source_width, roi.source_height)}")
        return rgb.crop((roi.x, roi.y, roi.x2, roi.y2))


def save_roi_preview(
    record: VideoRecord,
    keyframe_index: int,
    clip_frame_count: int,
    roi: ROI,
    output_path: Path | None = None,
    thumb_height: int = 160,
) -> Path:
    """Save a QC-only contact sheet of actual in-bounds ROI crops."""

    frames = sorted_frame_paths(record.frames_dir)
    if not frames:
        raise ValueError(f"No PNG frames found in {record.frames_dir}")
    indices = clip_indices(keyframe_index, clip_frame_count, numeric_frame_index(frames[0]), numeric_frame_index(frames[-1]))
    if not indices:
        raise ValueError("No clip indices to preview.")
    crops: list[tuple[int, Image.Image]] = []
    for index in indices:
        crop = crop_roi_pixels(frame_path_for_index(record.frames_dir, index), roi)
        scale = thumb_height / max(1, crop.height)
        thumb = crop.resize((max(1, int(round(crop.width * scale))), thumb_height), Image.Resampling.NEAREST)
        crops.append((index, thumb))
    label_height = 24
    padding = 8
    width = padding + sum(thumb.width + padding for _, thumb in crops)
    height = thumb_height + label_height + padding * 2
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    x = padding
    for index, thumb in crops:
        sheet.paste(thumb, (x, padding + label_height))
        draw.text((x, padding), str(index), fill="black")
        x += thumb.width + padding
    target = output_path or (record.sample_dir / "videos" / record.video_id / "roi_preview.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, format="PNG")
    return target
