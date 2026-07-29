"""Automatic RHEED aperture ROI and rotation-phase keyframe selection.

The selector is designed around the acquisition physics described by the
operator:

* the sample rotates at approximately constant angular speed;
* a bright diffraction feature follows a repeatable, left-opening trajectory;
* the desired phase is the right-most trajectory vertex, where horizontal
  motion changes sign while the feature continues upward;
* any sufficiently clear occurrence of that phase is acceptable.

The module never writes to source videos.  It supports both decoded video
streams and already-extracted lossless PNG frame directories.  ROI estimation
uses a temporal aperture mask plus a multi-frame diffraction-activity map.
Keyframe selection then tracks the dominant high-pass diffraction response and
scores either every frame (quality baseline) or only physical trajectory
vertices.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Iterable, Iterator, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from scipy import ndimage
from scipy.signal import find_peaks, peak_prominences
from skimage import measure, morphology
from skimage.filters import threshold_otsu

from .orientation import rotate_frame_clockwise


SUPPORTED_VIDEO_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
}

KEYFRAME_METHODS = (
    "quality_only",
    "vertex_clarity",
    "physics_vertex",
    "front_visibility",
    "compact_physics",
    "compact_visibility",
    "supervised_phase_ranker",
    "deep_visibility_ranker",
)

ROI_METHODS = (
    "aperture_inscribed",
    "activity_safe",
    "calibrated_safe",
)

SUPERVISED_PHASE_FEATURES = (
    "spot_x",
    "spot_y",
    "clarity",
    "sharpness",
    "spot_energy",
    "mean_intensity",
    "absolute_contrast",
    "prominence",
    "pre_dx",
    "post_dx",
    "upward_dy",
    "q_spot_x",
    "q_spot_y",
    "q_clarity",
    "q_sharpness",
    "q_spot_energy",
    "q_mean_intensity",
    "q_absolute_contrast",
    "q_prominence",
    "q_pre_dx",
    "q_post_dx",
    "q_upward_dy",
    "direction_consistent",
    "tracker_front",
    "cross_tracker_distance",
    "cross_tracker_agreement",
    "cross_tracker_direction_support",
)


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in source-frame pixels."""

    x: int
    y: int
    width: int
    height: int
    source_width: int
    source_height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def clipped(self) -> "Rect":
        x0 = int(np.clip(self.x, 0, max(self.source_width - 1, 0)))
        y0 = int(np.clip(self.y, 0, max(self.source_height - 1, 0)))
        x1 = int(np.clip(self.x2, x0 + 1, self.source_width))
        y1 = int(np.clip(self.y2, y0 + 1, self.source_height))
        return Rect(
            x=x0,
            y=y0,
            width=x1 - x0,
            height=y1 - y0,
            source_width=self.source_width,
            source_height=self.source_height,
        )

    def as_slices(self) -> tuple[slice, slice]:
        clipped = self.clipped()
        return (
            slice(clipped.y, clipped.y2),
            slice(clipped.x, clipped.x2),
        )


@dataclass(frozen=True)
class ROIPrediction:
    method: str
    rect: Rect
    aperture_area_fraction: float
    safe_pixel_fraction: float
    activity_coverage: float
    confidence: float
    analysis_scale: float
    circular_edge_intrusion_fraction: float | None = None


@dataclass(frozen=True)
class KeyframePrediction:
    method: str
    frame_index: int
    score: float
    confidence: float
    candidate_count: int
    estimated_period_frames: float | None
    spot_x: float
    spot_y: float
    clarity: float
    vertex_prominence: float
    direction_consistent: bool
    visibility_rank: float | None = None
    eligible_candidate_count: int | None = None
    selection_margin: float | None = None


@dataclass(frozen=True)
class AutomaticSelection:
    source: str
    frame_count: int
    roi: ROIPrediction
    keyframes: dict[str, KeyframePrediction]
    tracking_roi: ROIPrediction | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(frozen=True)
class ApertureAnalysis:
    source_width: int
    source_height: int
    scale: float
    reference: np.ndarray
    aperture_mask: np.ndarray
    safe_mask: np.ndarray
    activity: np.ndarray


def _rgb_uint8(frame: np.ndarray) -> np.ndarray:
    values = np.asarray(frame)
    if values.ndim == 2:
        values = np.repeat(values[..., None], 3, axis=2)
    if values.ndim != 3 or values.shape[2] < 3:
        raise ValueError(f"Expected RGB-like frame, got {values.shape}")
    values = values[..., :3]
    if values.dtype != np.uint8:
        values = values.astype(np.float32)
        if float(np.nanmax(values)) <= 1.5:
            values = values * 255.0
        values = np.clip(
            np.nan_to_num(values, nan=0.0, posinf=255.0, neginf=0.0),
            0,
            255,
        ).astype(np.uint8)
    return np.ascontiguousarray(values)


def _gray_float(frame: np.ndarray) -> np.ndarray:
    values = _rgb_uint8(frame).astype(np.float32) / 255.0
    # The diffraction signal is effectively monochrome.  Averaging avoids
    # relying on the RGB/BGR convention of a particular decoder.
    return values.mean(axis=2, dtype=np.float32)


def _brightness_float(frame: np.ndarray) -> np.ndarray:
    # The recorded phosphor screen is strongly blue.  Max-channel brightness
    # is more reliable than luminance for separating it from a black eyepiece.
    return _rgb_uint8(frame).astype(np.float32).max(axis=2) / 255.0


def _resize_float(
    image: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    source = np.asarray(image, dtype=np.float32)
    mode = "F"
    return np.asarray(
        Image.fromarray(source, mode=mode).resize(
            (int(width), int(height)),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    )


def iter_video_frames(
    path: str | Path,
    *,
    rotation_clockwise_degrees: int = 0,
) -> Iterator[tuple[int, np.ndarray]]:
    """Decode common video formats without modifying the source."""

    source = Path(path)
    if source.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        raise ValueError(
            f"Unsupported video suffix {source.suffix!r}; supported: "
            f"{sorted(SUPPORTED_VIDEO_SUFFIXES)}"
        )
    reader = imageio.get_reader(str(source), "ffmpeg")
    try:
        for index, frame in enumerate(reader):
            yield index, rotate_frame_clockwise(
                _rgb_uint8(frame),
                rotation_clockwise_degrees,
            )
    finally:
        reader.close()


def sorted_png_paths(path: str | Path) -> list[Path]:
    root = Path(path)
    paths = [item for item in root.glob("*.png") if item.stem.isdigit()]
    return sorted(paths, key=lambda item: int(item.stem))


def iter_png_frames(
    path: str | Path,
    *,
    rotation_clockwise_degrees: int = 0,
) -> Iterator[tuple[int, np.ndarray]]:
    for frame_path in sorted_png_paths(path):
        with Image.open(frame_path) as image:
            yield int(frame_path.stem), rotate_frame_clockwise(
                np.asarray(image.convert("RGB"), dtype=np.uint8),
                rotation_clockwise_degrees,
            )


def _source_factory(
    source: str | Path,
    *,
    rotation_clockwise_degrees: int = 0,
) -> tuple[Callable[[], Iterator[tuple[int, np.ndarray]]], int | None, str]:
    path = Path(source)
    if path.is_dir():
        paths = sorted_png_paths(path)
        if not paths:
            raise FileNotFoundError(f"No numeric PNG frames in {path}")
        return (
            lambda: iter_png_frames(
                path,
                rotation_clockwise_degrees=rotation_clockwise_degrees,
            ),
            len(paths),
            str(path),
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    return (
        lambda: iter_video_frames(
            path,
            rotation_clockwise_degrees=rotation_clockwise_degrees,
        ),
        None,
        str(path),
    )


def _reservoir_sample(
    frames: Iterable[tuple[int, np.ndarray]],
    *,
    maximum: int,
    seed: int = 17,
) -> tuple[list[np.ndarray], int]:
    """Deterministically sample a stream while counting every decoded frame."""

    rng = np.random.default_rng(seed)
    selected: list[tuple[int, np.ndarray]] = []
    count = 0
    for index, frame in frames:
        count += 1
        if len(selected) < maximum:
            selected.append((index, frame.copy()))
            continue
        position = int(rng.integers(0, count))
        if position < maximum:
            selected[position] = (index, frame.copy())
    selected.sort(key=lambda item: item[0])
    return [frame for _, frame in selected], count


def sample_frames(
    source: str | Path,
    *,
    maximum: int = 48,
    rotation_clockwise_degrees: int = 0,
) -> tuple[list[np.ndarray], int]:
    factory, known_count, _ = _source_factory(
        source,
        rotation_clockwise_degrees=rotation_clockwise_degrees,
    )
    if known_count is not None:
        paths = sorted_png_paths(source)
        indices = np.linspace(
            0, len(paths) - 1, min(maximum, len(paths))
        ).round().astype(int)
        frames = []
        for position in indices:
            with Image.open(paths[int(position)]) as image:
                frames.append(
                    rotate_frame_clockwise(
                        np.asarray(image.convert("RGB"), dtype=np.uint8),
                        rotation_clockwise_degrees,
                    )
                )
        return frames, known_count
    return _reservoir_sample(factory(), maximum=maximum)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels = measure.label(np.asarray(mask, dtype=bool))
    regions = measure.regionprops(labels)
    if not regions:
        return np.zeros_like(mask, dtype=bool)
    largest = max(regions, key=lambda region: region.area)
    return labels == largest.label


def analyze_aperture(
    frames: Sequence[np.ndarray],
    *,
    max_side: int = 320,
    safe_margin_fraction: float = 0.012,
) -> ApertureAnalysis:
    """Estimate the illuminated eyepiece interior and diffraction activity."""

    if not frames:
        raise ValueError("At least one frame is required for ROI estimation")
    first = _rgb_uint8(frames[0])
    source_height, source_width = first.shape[:2]
    if source_height <= 0 or source_width <= 0:
        raise ValueError("Invalid source frame dimensions")
    scale = min(1.0, float(max_side) / max(source_height, source_width))
    width = max(48, int(round(source_width * scale)))
    height = max(48, int(round(source_height * scale)))

    brightness = []
    for frame in frames:
        rgb = _rgb_uint8(frame)
        if rgb.shape[:2] != (source_height, source_width):
            raise ValueError("Video frame dimensions changed within one source")
        brightness.append(
            _resize_float(
                _brightness_float(rgb),
                width=width,
                height=height,
            )
        )
    stack = np.stack(brightness).astype(np.float32)
    reference = np.median(stack, axis=0)
    try:
        otsu = float(threshold_otsu(reference))
    except ValueError:
        otsu = float(np.median(reference))
    threshold = max(0.004, 0.52 * otsu)
    aperture = reference > threshold
    radius = max(2, int(round(min(height, width) * 0.018)))
    aperture = morphology.closing(aperture, morphology.disk(radius))
    aperture = ndimage.binary_fill_holes(aperture)
    aperture = morphology.remove_small_objects(
        aperture,
        max_size=max(64, int(aperture.size * 0.015)),
    )
    aperture = _largest_component(aperture)
    aperture = morphology.closing(
        aperture,
        morphology.disk(max(2, int(round(min(height, width) * 0.025)))),
    )
    aperture = ndimage.binary_fill_holes(aperture)
    aperture = _largest_component(aperture)
    if float(aperture.mean()) < 0.03:
        raise RuntimeError("Unable to identify a sufficiently large aperture")

    distance = ndimage.distance_transform_edt(aperture)
    safe_margin = max(2.0, min(height, width) * safe_margin_fraction)
    safe = distance > safe_margin
    if float(safe.mean()) < 0.02:
        safe = aperture.copy()

    high_pass_frames = []
    for image in stack:
        values = image[safe]
        low, high = np.percentile(values, [5.0, 99.7])
        normalized = np.clip(
            (image - low) / max(float(high - low), 1e-4),
            0.0,
            1.0,
        )
        small = ndimage.gaussian_filter(normalized, 0.7)
        large = ndimage.gaussian_filter(
            normalized,
            max(2.5, min(height, width) * 0.018),
        )
        response = np.maximum(small - large, 0.0)
        response[~safe] = 0.0
        cutoff = float(np.percentile(response[safe], 97.5))
        high_pass_frames.append(np.maximum(response - cutoff, 0.0))
    activity = np.quantile(np.stack(high_pass_frames), 0.90, axis=0)
    maximum = float(activity.max())
    if maximum > 0:
        activity = activity / maximum
    return ApertureAnalysis(
        source_width=source_width,
        source_height=source_height,
        scale=scale,
        reference=reference.astype(np.float32),
        aperture_mask=aperture.astype(bool),
        safe_mask=safe.astype(bool),
        activity=np.asarray(activity, dtype=np.float32),
    )


def _integral(values: np.ndarray) -> np.ndarray:
    return np.pad(
        np.asarray(values),
        ((1, 0), (1, 0)),
        mode="constant",
    ).cumsum(axis=0).cumsum(axis=1)


def _rect_sums(integral: np.ndarray, height: int, width: int) -> np.ndarray:
    return (
        integral[height:, width:]
        - integral[:-height, width:]
        - integral[height:, :-width]
        + integral[:-height, :-width]
    )


def _search_safe_rectangle(
    analysis: ApertureAnalysis,
    *,
    method: str,
    aspect_ratio: float,
    calibrated_scale: float,
) -> tuple[int, int, int, int, float, float]:
    safe = analysis.safe_mask
    activity = analysis.activity
    height, width = safe.shape
    unsafe_integral = _integral((~safe).astype(np.int32))
    activity_integral = _integral(activity.astype(np.float64))
    maximum_width = min(width - 2, int((height - 2) / aspect_ratio))
    minimum_width = max(12, int(round(maximum_width * 0.38)))
    activity_total = max(float(activity.sum()), 1e-8)

    candidates: list[tuple[float, int, int, int, int, float]] = []
    for rect_width in range(maximum_width, minimum_width - 1, -1):
        rect_height = max(8, int(round(rect_width * aspect_ratio)))
        if rect_height >= height:
            continue
        unsafe = _rect_sums(
            unsafe_integral,
            rect_height,
            rect_width,
        )
        tolerance = max(1.0, rect_width * rect_height * 0.002)
        valid = unsafe <= tolerance
        if not np.any(valid):
            continue
        activity_sum = _rect_sums(
            activity_integral,
            rect_height,
            rect_width,
        )
        coverage = activity_sum / activity_total
        area_fraction = rect_width * rect_height / max(float(safe.sum()), 1.0)
        if method == "aperture_inscribed":
            target_width = maximum_width
            size_penalty = abs(rect_width - target_width) / maximum_width
            scores = coverage - 0.20 * size_penalty
        elif method == "calibrated_safe":
            target_width = calibrated_scale * maximum_width
            size_penalty = abs(rect_width - target_width) / maximum_width
            scores = coverage - 0.55 * size_penalty - 0.03 * area_fraction
        elif method == "activity_safe":
            mean_activity = activity_sum / max(
                float(rect_width * rect_height), 1.0
            )
            density_scale = max(
                float(np.percentile(mean_activity[valid], 95)), 1e-8
            )
            scores = (
                coverage
                - 0.16 * area_fraction
                + 0.05 * mean_activity / density_scale
            )
        else:
            raise ValueError(f"Unknown ROI method: {method}")
        scores = np.where(valid, scores, -np.inf)
        position = int(np.argmax(scores))
        y0, x0 = np.unravel_index(position, scores.shape)
        score = float(scores[y0, x0])
        coverage_value = float(coverage[y0, x0])
        candidates.append(
            (
                score,
                x0,
                y0,
                rect_width,
                rect_height,
                coverage_value,
            )
        )
        if method == "aperture_inscribed":
            # The first feasible width is the largest safe rectangle.
            break
    if not candidates:
        y, x = np.where(safe)
        if not len(x):
            raise RuntimeError("No safe aperture pixels available")
        return (
            int(x.min()),
            int(y.min()),
            int(x.max() - x.min() + 1),
            int(y.max() - y.min() + 1),
            0.0,
            0.0,
        )
    _, x0, y0, rect_width, rect_height, coverage = max(
        candidates, key=lambda item: item[0]
    )
    sub = safe[y0 : y0 + rect_height, x0 : x0 + rect_width]
    return (
        x0,
        y0,
        rect_width,
        rect_height,
        float(sub.mean()),
        coverage,
    )


def predict_roi(
    frames: Sequence[np.ndarray],
    *,
    method: str = "calibrated_safe",
    aspect_ratio: float = 1.54,
    calibrated_scale: float = 0.90,
    analysis: ApertureAnalysis | None = None,
    lattice_calibration: Mapping[str, Any] | str | Path | None = None,
) -> tuple[ROIPrediction, ApertureAnalysis]:
    """Predict a border-safe ROI from sampled full-video frames."""

    if method not in (*ROI_METHODS, "full_lattice"):
        raise ValueError(f"Unknown ROI method {method!r}")
    aperture = analysis or analyze_aperture(frames)
    if method == "full_lattice":
        if lattice_calibration is None:
            raise ValueError(
                "full_lattice ROI requires a lattice calibration bundle"
            )
        from rheed2morph.rheed.lattice_roi import (
            predict_full_lattice_roi,
        )

        return (
            predict_full_lattice_roi(aperture, lattice_calibration),
            aperture,
        )
    x, y, width, height, safe_fraction, coverage = _search_safe_rectangle(
        aperture,
        method=method,
        aspect_ratio=float(aspect_ratio),
        calibrated_scale=float(calibrated_scale),
    )
    inverse = 1.0 / aperture.scale
    rect = Rect(
        x=int(round(x * inverse)),
        y=int(round(y * inverse)),
        width=max(1, int(round(width * inverse))),
        height=max(1, int(round(height * inverse))),
        source_width=aperture.source_width,
        source_height=aperture.source_height,
    ).clipped()
    aperture_fraction = float(aperture.aperture_mask.mean())
    confidence = float(
        np.clip(
            0.55 * safe_fraction
            + 0.30 * min(coverage / 0.80, 1.0)
            + 0.15 * min(aperture_fraction / 0.20, 1.0),
            0.0,
            1.0,
        )
    )
    return (
        ROIPrediction(
            method=method,
            rect=rect,
            aperture_area_fraction=aperture_fraction,
            safe_pixel_fraction=float(safe_fraction),
            activity_coverage=float(coverage),
            confidence=confidence,
            analysis_scale=aperture.scale,
        ),
        aperture,
    )


def _spot_features(frame: np.ndarray, roi: Rect) -> dict[str, float]:
    y_slice, x_slice = roi.as_slices()
    gray = _gray_float(frame)[y_slice, x_slice]
    if gray.size == 0:
        raise ValueError("Predicted ROI is empty")
    work = _resize_float(gray, width=96, height=160)
    low, high = np.percentile(work, [5.0, 99.7])
    normalized = np.clip(
        (work - low) / max(float(high - low), 1e-4),
        0.0,
        1.0,
    )
    response = (
        ndimage.gaussian_filter(normalized, 0.8)
        - ndimage.gaussian_filter(normalized, 5.0)
    )
    response[:8] = 0.0
    response[-8:] = 0.0
    response[:, :5] = 0.0
    response[:, -5:] = 0.0
    positive = np.maximum(response, 0.0)
    compact_threshold = float(np.percentile(positive, 99.3))
    compact_weights = (
        np.maximum(positive - compact_threshold, 0.0) ** 2
    )
    yy, xx = np.indices(positive.shape)
    if float(compact_weights.sum()) < 1e-8:
        y, x = np.unravel_index(int(np.argmax(response)), response.shape)
        compact_x = float(x)
        compact_y = float(y)
    else:
        compact_total = float(compact_weights.sum())
        compact_x = float(
            (xx * compact_weights).sum() / compact_total
        )
        compact_y = float(
            (yy * compact_weights).sum() / compact_total
        )

    # Track the horizontal front of the complete diffraction family rather
    # than only the single brightest pixel.  The latter can jump between
    # vertically separated streaks and create false trajectory vertices.
    aggregate_threshold = float(np.percentile(positive, 95.0))
    aggregate_weights = np.maximum(
        positive - aggregate_threshold, 0.0
    )
    aggregate_total = float(aggregate_weights.sum())
    if aggregate_total < 1e-8:
        front_x = compact_x
        aggregate_y = compact_y
    else:
        columns = aggregate_weights.sum(axis=0)
        cumulative = np.cumsum(columns)
        front_x = float(
            np.searchsorted(cumulative, 0.85 * cumulative[-1])
        )
        rows = aggregate_weights.sum(axis=1)
        aggregate_y = float(
            (np.arange(len(rows), dtype=float) * rows).sum()
            / aggregate_total
        )
    raw_narrow = ndimage.gaussian_filter(work, 0.8)
    raw_wide = ndimage.gaussian_filter(work, 5.0)
    raw_response = np.maximum(raw_narrow - raw_wide, 0.0)
    gy, gx = np.gradient(normalized)
    return {
        "spot_x": front_x,
        "spot_y": aggregate_y,
        "compact_spot_x": compact_x,
        "compact_spot_y": compact_y,
        "clarity": float(
            np.percentile(response, 99.8) / (np.std(response) + 1e-6)
        ),
        "sharpness": float(np.mean(np.hypot(gx, gy))),
        "spot_energy": float(np.percentile(response, 99.5)),
        "absolute_contrast": float(np.percentile(raw_response, 99.5)),
        "mean_intensity": float(work.mean()),
    }


def extract_spot_trajectory(
    frames: Iterable[tuple[int, np.ndarray]],
    roi: Rect,
) -> list[dict[str, float | int]]:
    records = []
    for frame_index, frame in frames:
        feature = _spot_features(frame, roi)
        records.append({"frame_index": int(frame_index), **feature})
    if len(records) < 3:
        raise ValueError("At least three decoded frames are required")
    return records


def extract_multi_roi_trajectories(
    frames: Iterable[tuple[int, np.ndarray]],
    rois: dict[str, Rect],
) -> dict[str, list[dict[str, float | int]]]:
    """Extract several ROI trajectories in one source-decoding pass."""

    if not rois:
        raise ValueError("At least one ROI is required")
    records: dict[str, list[dict[str, float | int]]] = {
        name: [] for name in rois
    }
    for frame_index, frame in frames:
        for name, roi in rois.items():
            feature = _spot_features(frame, roi)
            records[name].append(
                {"frame_index": int(frame_index), **feature}
            )
    if min(map(len, records.values())) < 3:
        raise ValueError("At least three decoded frames are required")
    return records


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    if len(values) <= 1:
        return np.ones(len(values), dtype=np.float64)
    return order.astype(np.float64) / float(len(values) - 1)


def _estimate_period(peaks: np.ndarray) -> float | None:
    if len(peaks) < 3:
        return None
    differences = np.diff(np.asarray(peaks, dtype=float))
    differences = differences[(differences >= 10.0) & (differences <= 120.0)]
    if len(differences) < 2:
        return None
    center = float(np.median(differences))
    retained = differences[np.abs(differences - center) <= max(5.0, 0.35 * center)]
    if len(retained):
        center = float(np.median(retained))
    return center


def _select_keyframes_core(
    trajectory: Sequence[dict[str, float | int]],
) -> tuple[dict[str, KeyframePrediction], list[dict[str, Any]]]:
    """Return predictions for one choice of trajectory coordinates."""

    if len(trajectory) < 3:
        raise ValueError("Trajectory is too short")
    frame_indices = np.asarray(
        [int(row["frame_index"]) for row in trajectory], dtype=int
    )
    x_raw = np.asarray([float(row["spot_x"]) for row in trajectory])
    y_raw = np.asarray([float(row["spot_y"]) for row in trajectory])
    clarity = np.asarray([float(row["clarity"]) for row in trajectory])
    sharpness = np.asarray([float(row["sharpness"]) for row in trajectory])
    energy = np.asarray([float(row["spot_energy"]) for row in trajectory])
    mean_intensity = np.asarray(
        [float(row["mean_intensity"]) for row in trajectory]
    )
    absolute_contrast = np.asarray(
        [
            float(row.get("absolute_contrast", row["spot_energy"]))
            for row in trajectory
        ]
    )
    x = ndimage.gaussian_filter1d(
        ndimage.median_filter(x_raw, size=3, mode="nearest"),
        sigma=1.5,
        mode="nearest",
    )
    y = ndimage.gaussian_filter1d(
        ndimage.median_filter(y_raw, size=3, mode="nearest"),
        sigma=1.5,
        mode="nearest",
    )

    minimum_distance = max(10, min(18, len(trajectory) // 12))
    prominence_floor = max(0.8, float(np.std(x) * 0.10))
    peak_positions, _ = find_peaks(
        x,
        distance=minimum_distance,
        prominence=prominence_floor,
    )
    if not len(peak_positions):
        peak_positions = np.asarray([int(np.argmax(x))], dtype=int)
    prominences = peak_prominences(x, peak_positions)[0]
    period = _estimate_period(frame_indices[peak_positions])

    q_clarity = _rank(clarity)
    q_sharpness = _rank(sharpness)
    q_energy = _rank(energy)
    quality_all = 0.55 * q_clarity + 0.25 * q_sharpness + 0.20 * q_energy
    quality_position = int(np.argmax(quality_all))

    candidate_rows: list[dict[str, Any]] = []
    for candidate_index, position in enumerate(peak_positions):
        left = max(0, position - 4)
        right = min(len(x) - 1, position + 4)
        local = slice(max(0, position - 4), min(len(x), position + 5))
        pre_dx = float(x[position] - x[left])
        post_dx = float(x[position] - x[right])
        upward_dy = float(y[left] - y[right])
        direction = bool(pre_dx > 0.0 and post_dx > 0.0 and upward_dy > 0.0)
        candidate_rows.append(
            {
                "position": int(position),
                "frame_index": int(frame_indices[position]),
                "spot_x": float(x[position]),
                "spot_y": float(y[position]),
                "clarity": float(np.median(clarity[local])),
                "sharpness": float(np.median(sharpness[local])),
                "spot_energy": float(np.median(energy[local])),
                "mean_intensity": float(np.median(mean_intensity[local])),
                "absolute_contrast": float(
                    np.median(absolute_contrast[local])
                ),
                "prominence": float(prominences[candidate_index]),
                "pre_dx": pre_dx,
                "post_dx": post_dx,
                "upward_dy": upward_dy,
                "direction_consistent": direction,
            }
        )

    for name in (
        "spot_x",
        "spot_y",
        "clarity",
        "sharpness",
        "spot_energy",
        "mean_intensity",
        "absolute_contrast",
        "prominence",
        "pre_dx",
        "post_dx",
        "upward_dy",
    ):
        values = np.asarray([float(row[name]) for row in candidate_rows])
        ranks = _rank(values)
        for row, rank in zip(candidate_rows, ranks):
            row[f"q_{name}"] = float(rank)
    for row in candidate_rows:
        row["vertex_clarity_score"] = float(
            0.48 * row["q_clarity"]
            + 0.24 * row["q_sharpness"]
            + 0.16 * row["q_spot_energy"]
            + 0.07 * row["q_spot_x"]
            + 0.05 * row["q_prominence"]
        )
        row["physics_vertex_score"] = float(
            0.30 * row["q_clarity"]
            + 0.10 * row["q_sharpness"]
            + 0.12 * row["q_spot_energy"]
            + 0.18 * row["q_spot_x"]
            + 0.14 * row["q_prominence"]
            + 0.06 * row["q_pre_dx"]
            + 0.06 * row["q_post_dx"]
            + 0.04 * row["q_upward_dy"]
            + 0.10 * float(row["direction_consistent"])
        )
        row["front_visibility_score"] = float(
            0.16 * row["q_clarity"]
            + 0.07 * row["q_sharpness"]
            + 0.07 * row["q_spot_energy"]
            + 0.20 * row["q_mean_intensity"]
            + 0.16 * row["q_absolute_contrast"]
            + 0.12 * row["q_spot_x"]
            + 0.08 * row["q_prominence"]
            + 0.03 * row["q_pre_dx"]
            + 0.03 * row["q_post_dx"]
            + 0.03 * row["q_upward_dy"]
            + 0.12 * float(row["direction_consistent"])
        )

    predictions: dict[str, KeyframePrediction] = {}
    quality_median = float(np.median(quality_all))
    quality_mad = float(
        1.4826 * np.median(np.abs(quality_all - quality_median))
    )
    quality_confidence = float(
        1.0
        / (
            1.0
            + np.exp(
                -(
                    float(quality_all[quality_position]) - quality_median
                )
                / max(quality_mad, 0.05)
            )
        )
    )
    predictions["quality_only"] = KeyframePrediction(
        method="quality_only",
        frame_index=int(frame_indices[quality_position]),
        score=float(quality_all[quality_position]),
        confidence=quality_confidence,
        candidate_count=len(candidate_rows),
        estimated_period_frames=period,
        spot_x=float(x[quality_position]),
        spot_y=float(y[quality_position]),
        clarity=float(clarity[quality_position]),
        vertex_prominence=0.0,
        direction_consistent=False,
    )

    for method, score_name in (
        ("vertex_clarity", "vertex_clarity_score"),
        ("physics_vertex", "physics_vertex_score"),
        ("front_visibility", "front_visibility_score"),
    ):
        eligible = candidate_rows
        if method == "front_visibility" and len(candidate_rows) >= 4:
            intensity_floor = float(
                np.percentile(
                    [row["mean_intensity"] for row in candidate_rows],
                    35.0,
                )
            )
            contrast_floor = float(
                np.percentile(
                    [row["absolute_contrast"] for row in candidate_rows],
                    20.0,
                )
            )
            eligible = [
                row
                for row in candidate_rows
                if row["mean_intensity"] >= intensity_floor
                and row["absolute_contrast"] >= contrast_floor
            ]
            if not eligible:
                eligible = candidate_rows
        ordered = sorted(
            eligible,
            key=lambda row: (
                float(row[score_name]),
                float(row["clarity"]),
                -int(row["frame_index"]),
            ),
            reverse=True,
        )
        selected = ordered[0]
        margin = (
            float(selected[score_name]) - float(ordered[1][score_name])
            if len(ordered) > 1
            else float(selected[score_name])
        )
        support = min(len(candidate_rows) / 5.0, 1.0)
        direction_bonus = (
            1.0 if bool(selected["direction_consistent"]) else 0.65
        )
        confidence = float(
            np.clip(
                (0.45 + 0.55 * min(max(margin, 0.0) / 0.20, 1.0))
                * (0.75 + 0.25 * support)
                * direction_bonus,
                0.0,
                1.0,
            )
        )
        predictions[method] = KeyframePrediction(
            method=method,
            frame_index=int(selected["frame_index"]),
            score=float(selected[score_name]),
            confidence=confidence,
            candidate_count=len(candidate_rows),
            estimated_period_frames=period,
            spot_x=float(selected["spot_x"]),
            spot_y=float(selected["spot_y"]),
            clarity=float(selected["clarity"]),
            vertex_prominence=float(selected["prominence"]),
            direction_consistent=bool(selected["direction_consistent"]),
        )
    return predictions, candidate_rows


def select_keyframes(
    trajectory: Sequence[dict[str, float | int]],
) -> tuple[dict[str, KeyframePrediction], list[dict[str, Any]]]:
    """Return front-tracker and compact bright-spot model predictions."""

    predictions, candidate_rows = _select_keyframes_core(trajectory)
    if all(
        "compact_spot_x" in row and "compact_spot_y" in row
        for row in trajectory
    ):
        compact_trajectory = [
            {
                **row,
                "spot_x": row["compact_spot_x"],
                "spot_y": row["compact_spot_y"],
            }
            for row in trajectory
        ]
        compact_predictions, compact_candidates = _select_keyframes_core(
            compact_trajectory
        )
        predictions["compact_physics"] = replace(
            compact_predictions["physics_vertex"],
            method="compact_physics",
        )
        predictions["compact_visibility"] = replace(
            compact_predictions["front_visibility"],
            method="compact_visibility",
        )
        for row in compact_candidates:
            row["coordinate_model"] = "compact_bright_spot"
    for row in candidate_rows:
        row["coordinate_model"] = "diffraction_front"
    return predictions, candidate_rows


def _supervised_candidate_rows(
    trajectory: Sequence[dict[str, float | int]],
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    front_predictions, front_candidates = _select_keyframes_core(trajectory)
    if not all(
        "compact_spot_x" in row and "compact_spot_y" in row
        for row in trajectory
    ):
        raise ValueError(
            "The supervised ranker requires V2 front and compact coordinates"
        )
    compact_trajectory = [
        {
            **row,
            "spot_x": row["compact_spot_x"],
            "spot_y": row["compact_spot_y"],
        }
        for row in trajectory
    ]
    compact_predictions, compact_candidates = _select_keyframes_core(
        compact_trajectory
    )
    by_tracker = {
        "front": front_candidates,
        "compact": compact_candidates,
    }
    rows: list[dict[str, Any]] = []
    for tracker, candidates in by_tracker.items():
        other = by_tracker[
            "compact" if tracker == "front" else "front"
        ]
        for candidate in candidates:
            frame_index = int(candidate["frame_index"])
            nearest = min(
                other,
                key=lambda item: abs(
                    int(item["frame_index"]) - frame_index
                ),
            )
            distance = abs(int(nearest["frame_index"]) - frame_index)
            rows.append(
                {
                    **candidate,
                    "tracker": tracker,
                    "tracker_front": float(tracker == "front"),
                    "cross_tracker_distance": float(min(distance, 60)),
                    "cross_tracker_agreement": float(
                        np.exp(-distance / 3.0)
                    ),
                    "cross_tracker_direction_support": float(
                        bool(nearest["direction_consistent"])
                        and distance <= 4
                    ),
                }
            )
    periods = {
        "front": front_predictions[
            "physics_vertex"
        ].estimated_period_frames,
        "compact": compact_predictions[
            "physics_vertex"
        ].estimated_period_frames,
    }
    return rows, periods


def predict_keyframe_with_ranker(
    trajectory: Sequence[dict[str, float | int]],
    ranker_path: str | Path,
) -> KeyframePrediction:
    """Rank physical vertex candidates using the fitted human-phase model."""

    try:
        import joblib
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "The supervised phase ranker requires joblib and pandas"
        ) from exc
    bundle = joblib.load(Path(ranker_path))
    features = tuple(bundle["features"])
    if features != SUPERVISED_PHASE_FEATURES:
        raise ValueError(
            "Ranker feature schema does not match this selector version"
        )
    candidates, periods = _supervised_candidate_rows(trajectory)
    table = pd.DataFrame(candidates)
    scores = np.asarray(
        bundle["model"].predict(table[list(features)]), dtype=float
    )
    position = int(np.argmax(scores))
    selected = candidates[position]
    calibrated = float(
        bundle["calibrator"].predict([scores[position]])[0]
    )
    return KeyframePrediction(
        method="supervised_phase_ranker",
        frame_index=int(selected["frame_index"]),
        score=float(scores[position]),
        confidence=float(np.clip(calibrated, 0.0, 1.0)),
        candidate_count=len(candidates),
        estimated_period_frames=periods[str(selected["tracker"])],
        spot_x=float(selected["spot_x"]),
        spot_y=float(selected["spot_y"]),
        clarity=float(selected["clarity"]),
        vertex_prominence=float(selected["prominence"]),
        direction_consistent=bool(selected["direction_consistent"]),
    )


def predict_keyframe_with_deep_visibility(
    trajectory: Sequence[dict[str, float | int]],
    candidate_frames: dict[int, np.ndarray],
    roi: Rect,
    ranker_path: str | Path,
    *,
    foundation_cache_dir: str | Path | None = None,
    device: str | None = None,
) -> KeyframePrediction:
    """Rank physical vertices using DINOv2 and explicit spot visibility."""

    from rheed2morph.rheed.spot_visibility import (
        score_deep_visibility_candidates,
    )

    candidates, periods = _supervised_candidate_rows(trajectory)
    result = score_deep_visibility_candidates(
        candidates,
        candidate_frames,
        roi,
        ranker_path,
        foundation_cache_dir=foundation_cache_dir,
        device=device,
    )
    selected = result["selected_candidate"]
    return KeyframePrediction(
        method="deep_visibility_ranker",
        frame_index=int(selected["frame_index"]),
        score=float(result["score"]),
        confidence=float(result["confidence"]),
        candidate_count=int(result["candidate_count"]),
        estimated_period_frames=periods[str(selected["tracker"])],
        spot_x=float(selected["spot_x"]),
        spot_y=float(selected["spot_y"]),
        clarity=float(selected["clarity"]),
        vertex_prominence=float(selected["prominence"]),
        direction_consistent=bool(selected["direction_consistent"]),
        visibility_rank=float(result["visibility_rank"]),
        eligible_candidate_count=int(result["eligible_candidate_count"]),
        selection_margin=float(result["selection_margin"]),
    )


def select_from_source(
    source: str | Path,
    *,
    roi_method: str = "calibrated_safe",
    aspect_ratio: float = 1.54,
    calibrated_scale: float = 0.90,
    roi_sample_count: int = 48,
    phase_ranker_path: str | Path | None = None,
    deep_visibility_ranker_path: str | Path | None = None,
    foundation_cache_dir: str | Path | None = None,
    deep_device: str | None = None,
    full_lattice_calibration_path: (
        Mapping[str, Any] | str | Path | None
    ) = None,
) -> tuple[AutomaticSelection, list[dict[str, Any]], ApertureAnalysis]:
    """Run keyframe selection and optional complete-lattice ROI refinement.

    The tracking ROI remains the frozen ``calibrated_safe`` geometry used to
    train the V5 keyframe model.  A larger four-boundary ROI can be predicted
    afterwards for visualization/export without changing keyframe scores.
    """

    if roi_method == "full_lattice":
        raise ValueError(
            "full_lattice is an export ROI, not a tracking ROI; pass its "
            "bundle with full_lattice_calibration_path"
        )
    sampled, counted = sample_frames(source, maximum=roi_sample_count)
    roi, analysis = predict_roi(
        sampled,
        method=roi_method,
        aspect_ratio=aspect_ratio,
        calibrated_scale=calibrated_scale,
        lattice_calibration=(
            full_lattice_calibration_path
            if roi_method == "full_lattice"
            else None
        ),
    )
    factory, known_count, display_source = _source_factory(source)
    trajectory = extract_spot_trajectory(factory(), roi.rect)
    predictions, candidates = select_keyframes(trajectory)
    if phase_ranker_path is not None:
        predictions["supervised_phase_ranker"] = (
            predict_keyframe_with_ranker(trajectory, phase_ranker_path)
        )
    if deep_visibility_ranker_path is not None:
        deep_candidates, _ = _supervised_candidate_rows(trajectory)
        required = {
            int(candidate["frame_index"]) for candidate in deep_candidates
        }
        candidate_frames: dict[int, np.ndarray] = {}
        for frame_index, frame in factory():
            if frame_index in required:
                candidate_frames[int(frame_index)] = frame
            if len(candidate_frames) == len(required):
                break
        if len(candidate_frames) != len(required):
            missing = sorted(required - set(candidate_frames))
            raise IndexError(
                f"Could not decode {len(missing)} candidate frames"
            )
        predictions["deep_visibility_ranker"] = (
            predict_keyframe_with_deep_visibility(
                trajectory,
                candidate_frames,
                roi.rect,
                deep_visibility_ranker_path,
                foundation_cache_dir=foundation_cache_dir,
                device=deep_device,
            )
        )
    frame_count = known_count or counted or len(trajectory)
    output_roi = roi
    if (
        full_lattice_calibration_path is not None
        and roi_method != "full_lattice"
    ):
        output_roi, _ = predict_roi(
            sampled,
            method="full_lattice",
            analysis=analysis,
            lattice_calibration=full_lattice_calibration_path,
        )
    selection = AutomaticSelection(
        source=display_source,
        frame_count=int(frame_count),
        roi=output_roi,
        keyframes=predictions,
        tracking_roi=roi if output_roi.method != roi.method else None,
    )
    return selection, trajectory, analysis


def load_frame(source: str | Path, index: int) -> np.ndarray:
    path = Path(source)
    if path.is_dir():
        frame_path = path / f"{int(index)}.png"
        with Image.open(frame_path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    for frame_index, frame in iter_video_frames(path):
        if frame_index == int(index):
            return frame
    raise IndexError(f"Frame {index} not found in {source}")


def save_selection_artifacts(
    selection: AutomaticSelection,
    *,
    output_dir: str | Path,
    selected_method: str = "physics_vertex",
) -> dict[str, Path]:
    """Save JSON, full-frame overlay and cropped selected frame."""

    if selected_method not in selection.keyframes:
        raise KeyError(selected_method)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "automatic_selection.json"
    result_path.write_text(
        json.dumps(selection.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prediction = selection.keyframes[selected_method]
    frame = load_frame(selection.source, prediction.frame_index)
    image = Image.fromarray(frame).convert("RGB")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    rect = selection.roi.rect
    draw.rectangle(
        [rect.x, rect.y, rect.x2 - 1, rect.y2 - 1],
        outline=(255, 80, 40),
        width=max(2, int(round(min(image.size) / 250))),
    )
    draw.text(
        (max(4, rect.x), max(4, rect.y - 20)),
        (
            f"{selected_method} frame={prediction.frame_index} "
            f"confidence={prediction.confidence:.2f}"
        ),
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    overlay_path = output / "selected_frame_with_roi.png"
    overlay.save(overlay_path)

    crop = image.crop((rect.x, rect.y, rect.x2, rect.y2))
    crop = ImageEnhance.Contrast(crop).enhance(1.8)
    crop_path = output / "selected_roi.png"
    crop.save(crop_path)
    return {
        "json": result_path,
        "overlay": overlay_path,
        "crop": crop_path,
    }
