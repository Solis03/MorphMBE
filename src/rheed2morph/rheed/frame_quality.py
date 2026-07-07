"""Transparent frame-quality features for cropped RHEED videos."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


try:  # pragma: no cover - availability depends on the local environment.
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None

try:  # pragma: no cover - availability depends on the local environment.
    from scipy import ndimage  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    ndimage = None


EPS = 1e-8

NUMERIC_FEATURE_KEYS = [
    "height",
    "width",
    "mean_intensity",
    "std_intensity",
    "p01",
    "p05",
    "p50",
    "p95",
    "p99",
    "dynamic_range_p99_p01",
    "bright_pixel_fraction",
    "saturated_pixel_fraction",
    "dark_pixel_fraction",
    "left_edge_mean",
    "right_edge_mean",
    "top_edge_mean",
    "bottom_edge_mean",
    "center_mean",
    "edge_to_center_ratio",
    "dark_edge_fraction",
    "largest_dark_component_fraction",
    "vertical_shadow_score",
    "horizontal_shadow_score",
    "shadow_penalty",
    "laplacian_variance",
    "tenengrad_score",
    "gradient_mean",
    "gradient_std",
    "local_contrast_score",
    "entropy",
    "fft_low_frequency_power",
    "fft_mid_frequency_power",
    "fft_high_frequency_power",
    "fft_anisotropy_proxy",
    "horizontal_projection_peak_prominence",
    "vertical_projection_peak_prominence",
    "projection_entropy",
    "approx_streak_spot_visibility_score",
    "hough_line_count",
]

FLAG_KEYS = [
    "almost_black",
    "almost_white",
    "over_saturated",
    "very_low_dynamic_range",
    "strong_shadow",
    "too_blurry",
    "possible_bad_frame",
]

SCORE_KEYS = [
    "brightness_score",
    "dynamic_range_score",
    "sharpness_score",
    "pattern_visibility_score",
    "contrast_score",
    "saturation_penalty",
    "blur_penalty",
    "low_dynamic_range_penalty",
    "quality_score",
]


def clip01(value: float | np.ndarray) -> float | np.ndarray:
    return np.clip(value, 0.0, 1.0)


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(result):
        return float(default)
    return result


def frame_to_gray_float32(frame: np.ndarray) -> np.ndarray:
    """Convert a video frame with unknown channel convention to gray float32 in [0, 1]."""
    array = np.asarray(frame)
    if array.ndim == 2:
        gray = array.astype(np.float32, copy=False)
    elif array.ndim == 3:
        if array.shape[2] == 1:
            gray = array[..., 0].astype(np.float32, copy=False)
        else:
            # RHEED videos are effectively monochrome; channel averaging is robust to
            # RGB/BGR uncertainty and avoids treating codec color order as signal.
            gray = array[..., :3].astype(np.float32, copy=False).mean(axis=2)
    else:
        raise ValueError(f"Expected a 2D or 3D frame, got shape {array.shape}.")

    gray = np.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    max_value = finite_float(np.max(gray), 0.0) if gray.size else 0.0
    min_value = finite_float(np.min(gray), 0.0) if gray.size else 0.0
    if max_value > 1.5:
        if max_value <= 255.0 and min_value >= 0.0:
            gray = gray / 255.0
        else:
            gray = (gray - min_value) / max(max_value - min_value, EPS)
    return np.clip(gray, 0.0, 1.0).astype(np.float32, copy=False)


def resize_image(image: np.ndarray, size: int | None = None, max_side: int | None = None) -> np.ndarray:
    if size is None and max_side is None:
        return np.asarray(image)
    array = np.asarray(image)
    height, width = array.shape[:2]
    if height <= 0 or width <= 0:
        return array
    if size is not None:
        new_width = int(size)
        new_height = int(size)
    else:
        assert max_side is not None
        scale = min(1.0, float(max_side) / float(max(height, width)))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
    if (new_height, new_width) == (height, width):
        return array.copy()
    if cv2 is not None:
        interpolation = cv2.INTER_AREA if new_height < height or new_width < width else cv2.INTER_LINEAR
        return cv2.resize(array, (new_width, new_height), interpolation=interpolation)
    y_idx = np.linspace(0, height - 1, new_height).astype(int)
    x_idx = np.linspace(0, width - 1, new_width).astype(int)
    return array[np.ix_(y_idx, x_idx)]


def normalize_for_display(gray: np.ndarray, p01: float | None = None, p99: float | None = None) -> np.ndarray:
    frame = frame_to_gray_float32(gray)
    low = finite_float(np.percentile(frame, 1) if p01 is None else p01, 0.0)
    high = finite_float(np.percentile(frame, 99) if p99 is None else p99, 1.0)
    if high <= low + EPS:
        low = finite_float(np.min(frame), 0.0)
        high = finite_float(np.max(frame), 1.0)
    display = (frame - low) / max(high - low, EPS)
    return np.clip(display, 0.0, 1.0).astype(np.float32, copy=False)


def enhance_for_display(gray: np.ndarray) -> np.ndarray:
    display = normalize_for_display(gray)
    if cv2 is not None:
        image_u8 = np.asarray(np.clip(display * 255.0, 0.0, 255.0), dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image_u8).astype(np.float32) / 255.0
    hist, bins = np.histogram(display.ravel(), bins=256, range=(0.0, 1.0), density=False)
    cdf = hist.cumsum().astype(np.float64)
    if cdf[-1] <= 0:
        return display
    cdf /= cdf[-1]
    equalized = np.interp(display.ravel(), bins[:-1], cdf).reshape(display.shape)
    return np.clip(equalized, 0.0, 1.0).astype(np.float32, copy=False)


def _largest_dark_component_fraction(dark_mask: np.ndarray) -> float:
    mask = np.asarray(dark_mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return 0.0
    if cv2 is not None:
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        if stats.shape[0] <= 1:
            return 0.0
        return finite_float(stats[1:, cv2.CC_STAT_AREA].max() / mask.size)
    if ndimage is not None:
        labels, count = ndimage.label(mask)
        if count <= 0:
            return 0.0
        sizes = np.bincount(labels.ravel())
        return finite_float(sizes[1:].max() / mask.size) if sizes.size > 1 else 0.0
    # Fallback proxy: contiguous dark columns/rows catch most obstruction cases.
    row_fraction = np.max(mask.mean(axis=1))
    col_fraction = np.max(mask.mean(axis=0))
    return finite_float(max(row_fraction, col_fraction) * float(mask.mean()))


def _entropy(values: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(values.ravel(), bins=bins, range=(0.0, 1.0), density=False)
    total = hist.sum()
    if total <= 0:
        return 0.0
    probs = hist.astype(np.float64) / float(total)
    probs = probs[probs > 0]
    return finite_float(-(probs * np.log2(probs)).sum())


def _projection_prominence(projection: np.ndarray) -> float:
    values = np.asarray(projection, dtype=np.float64)
    if values.size == 0:
        return 0.0
    centered = values - np.median(values)
    spread = np.percentile(values, 90) - np.percentile(values, 10)
    return finite_float(np.max(centered) / max(spread, EPS))


def _projection_entropy(horizontal: np.ndarray, vertical: np.ndarray) -> float:
    pieces: list[float] = []
    for projection in (horizontal, vertical):
        values = np.asarray(projection, dtype=np.float64)
        values = values - values.min()
        total = values.sum()
        if total <= 0:
            continue
        probs = values / total
        probs = probs[probs > 0]
        pieces.append(finite_float(-(probs * np.log2(probs)).sum() / max(math.log2(values.size), EPS)))
    if not pieces:
        return 0.0
    return finite_float(np.mean(pieces))


def _frequency_features(gray: np.ndarray) -> dict[str, float]:
    work = resize_image(gray, max_side=128).astype(np.float32, copy=False)
    work = work - float(np.mean(work))
    if work.size == 0 or float(np.std(work)) <= EPS:
        return {
            "fft_low_frequency_power": 0.0,
            "fft_mid_frequency_power": 0.0,
            "fft_high_frequency_power": 0.0,
            "fft_anisotropy_proxy": 0.0,
        }
    spectrum = np.fft.fftshift(np.fft.fft2(work))
    power = np.abs(spectrum) ** 2
    height, width = work.shape
    yy, xx = np.indices((height, width))
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = np.sqrt(((yy - cy) / max(height, 1)) ** 2 + ((xx - cx) / max(width, 1)) ** 2)
    radius = radius / max(float(radius.max()), EPS)
    total = float(power.sum()) + EPS
    low = finite_float(power[(radius > 0.02) & (radius <= 0.18)].sum() / total)
    mid = finite_float(power[(radius > 0.18) & (radius <= 0.45)].sum() / total)
    high = finite_float(power[radius > 0.45].sum() / total)

    row_band = power[max(0, int(cy) - 2) : min(height, int(cy) + 3), :].sum()
    col_band = power[:, max(0, int(cx) - 2) : min(width, int(cx) + 3)].sum()
    anisotropy = abs(float(row_band - col_band)) / max(float(row_band + col_band), EPS)
    return {
        "fft_low_frequency_power": low,
        "fft_mid_frequency_power": mid,
        "fft_high_frequency_power": high,
        "fft_anisotropy_proxy": finite_float(anisotropy),
    }


def _gradient_features(gray: np.ndarray) -> dict[str, float]:
    if cv2 is not None:
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    else:
        grad_y, grad_x = np.gradient(gray.astype(np.float32, copy=False))
        laplacian = (
            -4.0 * gray
            + np.roll(gray, 1, axis=0)
            + np.roll(gray, -1, axis=0)
            + np.roll(gray, 1, axis=1)
            + np.roll(gray, -1, axis=1)
        )
    gradient_magnitude = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    return {
        "laplacian_variance": finite_float(np.var(laplacian)),
        "tenengrad_score": finite_float(np.mean(gradient_magnitude * gradient_magnitude)),
        "gradient_mean": finite_float(np.mean(gradient_magnitude)),
        "gradient_std": finite_float(np.std(gradient_magnitude)),
    }


def _local_contrast(gray: np.ndarray, block_count: int = 8) -> float:
    height, width = gray.shape
    y_edges = np.linspace(0, height, min(block_count, height) + 1, dtype=int)
    x_edges = np.linspace(0, width, min(block_count, width) + 1, dtype=int)
    values: list[float] = []
    for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
        for x0, x1 in zip(x_edges[:-1], x_edges[1:]):
            block = gray[y0:y1, x0:x1]
            if block.size:
                values.append(float(np.std(block)))
    return finite_float(np.mean(values) if values else 0.0)


def extract_frame_quality_features(frame: np.ndarray) -> dict[str, Any]:
    """Extract finite, interpretable quality features from one frame."""
    gray = frame_to_gray_float32(frame)
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError(f"Expected a non-empty grayscale frame, got shape {gray.shape}.")

    height, width = gray.shape
    p01, p05, p50, p95, p99 = [finite_float(value) for value in np.percentile(gray, [1, 5, 50, 95, 99])]
    mean_intensity = finite_float(np.mean(gray))
    std_intensity = finite_float(np.std(gray))
    dynamic_range = finite_float(p99 - p01)
    bright_fraction = finite_float(np.mean(gray >= 0.90))
    saturated_fraction = finite_float(np.mean(gray >= 0.985))
    dark_fraction = finite_float(np.mean(gray <= 0.05))

    edge_w = max(1, int(round(width * 0.10)))
    edge_h = max(1, int(round(height * 0.10)))
    center_y0 = max(0, int(round(height * 0.25)))
    center_y1 = min(height, int(round(height * 0.75)))
    center_x0 = max(0, int(round(width * 0.25)))
    center_x1 = min(width, int(round(width * 0.75)))
    center = gray[center_y0:center_y1, center_x0:center_x1]
    center_mean = finite_float(np.mean(center) if center.size else mean_intensity)
    left_edge_mean = finite_float(np.mean(gray[:, :edge_w]))
    right_edge_mean = finite_float(np.mean(gray[:, width - edge_w :]))
    top_edge_mean = finite_float(np.mean(gray[:edge_h, :]))
    bottom_edge_mean = finite_float(np.mean(gray[height - edge_h :, :]))
    edge_values = np.concatenate(
        [
            gray[:, :edge_w].ravel(),
            gray[:, width - edge_w :].ravel(),
            gray[:edge_h, :].ravel(),
            gray[height - edge_h :, :].ravel(),
        ]
    )
    edge_mean = finite_float(np.mean(edge_values))
    edge_to_center_ratio = finite_float(edge_mean / max(center_mean, EPS))
    dark_edge_fraction = finite_float(np.mean(edge_values <= max(0.05, p05 + 0.02)))
    dark_mask = gray <= max(0.05, p05 + 0.02)
    largest_dark_component_fraction = _largest_dark_component_fraction(dark_mask)

    column_means = gray.mean(axis=0)
    row_means = gray.mean(axis=1)
    column_deficit = clip01((center_mean - np.percentile(column_means, 10)) / max(center_mean, EPS))
    row_deficit = clip01((center_mean - np.percentile(row_means, 10)) / max(center_mean, EPS))
    dark_columns = finite_float(np.mean(column_means < max(0.05, center_mean * 0.55)))
    dark_rows = finite_float(np.mean(row_means < max(0.05, center_mean * 0.55)))
    vertical_shadow_score = finite_float(max(column_deficit, dark_columns))
    horizontal_shadow_score = finite_float(max(row_deficit, dark_rows))
    edge_deficit = finite_float(clip01((0.85 - edge_to_center_ratio) / 0.85))
    shadow_penalty = finite_float(
        clip01(
            0.30 * edge_deficit
            + 0.25 * vertical_shadow_score
            + 0.20 * horizontal_shadow_score
            + 0.15 * dark_edge_fraction
            + 0.10 * clip01(largest_dark_component_fraction / 0.35)
        )
    )

    gradient = _gradient_features(gray)
    local_contrast_score = _local_contrast(gray)
    entropy = _entropy(gray)
    frequencies = _frequency_features(gray)

    horizontal_projection = gray.mean(axis=1)
    vertical_projection = gray.mean(axis=0)
    horizontal_prominence = _projection_prominence(horizontal_projection)
    vertical_prominence = _projection_prominence(vertical_projection)
    projection_entropy = _projection_entropy(horizontal_projection, vertical_projection)
    approx_pattern_score = finite_float(
        clip01(
            0.35 * clip01(horizontal_prominence / 4.0)
            + 0.20 * clip01(vertical_prominence / 4.0)
            + 0.20 * frequencies["fft_anisotropy_proxy"]
            + 0.15 * clip01(frequencies["fft_mid_frequency_power"] / 0.50)
            + 0.10 * projection_entropy
        )
    )

    almost_black = mean_intensity < 0.025 or p99 < 0.08
    almost_white = mean_intensity > 0.94 or p05 > 0.90
    over_saturated = saturated_fraction > 0.20 or (p99 >= 0.995 and bright_fraction > 0.35)
    very_low_dynamic_range = dynamic_range < 0.045 or std_intensity < 0.012
    strong_shadow = shadow_penalty > 0.55 or largest_dark_component_fraction > 0.45
    too_blurry = (
        gradient["laplacian_variance"] < 1e-6
        and gradient["tenengrad_score"] < 1e-6
        and local_contrast_score < 0.005
    )
    possible_bad_frame = almost_black or almost_white or over_saturated or very_low_dynamic_range or strong_shadow or too_blurry

    features: dict[str, Any] = {
        "height": float(height),
        "width": float(width),
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "p01": p01,
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "dynamic_range_p99_p01": dynamic_range,
        "bright_pixel_fraction": bright_fraction,
        "saturated_pixel_fraction": saturated_fraction,
        "dark_pixel_fraction": dark_fraction,
        "left_edge_mean": left_edge_mean,
        "right_edge_mean": right_edge_mean,
        "top_edge_mean": top_edge_mean,
        "bottom_edge_mean": bottom_edge_mean,
        "center_mean": center_mean,
        "edge_to_center_ratio": edge_to_center_ratio,
        "dark_edge_fraction": dark_edge_fraction,
        "largest_dark_component_fraction": largest_dark_component_fraction,
        "vertical_shadow_score": vertical_shadow_score,
        "horizontal_shadow_score": horizontal_shadow_score,
        "shadow_penalty": shadow_penalty,
        "local_contrast_score": local_contrast_score,
        "entropy": entropy,
        "horizontal_projection_peak_prominence": horizontal_prominence,
        "vertical_projection_peak_prominence": vertical_prominence,
        "projection_entropy": projection_entropy,
        "approx_streak_spot_visibility_score": approx_pattern_score,
        "hough_line_count": 0.0,
        "almost_black": bool(almost_black),
        "almost_white": bool(almost_white),
        "over_saturated": bool(over_saturated),
        "very_low_dynamic_range": bool(very_low_dynamic_range),
        "strong_shadow": bool(strong_shadow),
        "too_blurry": bool(too_blurry),
        "possible_bad_frame": bool(possible_bad_frame),
    }
    features.update(gradient)
    features.update(frequencies)

    for key in NUMERIC_FEATURE_KEYS:
        features[key] = finite_float(features.get(key, 0.0))
    return features


def _array(rows: Sequence[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([finite_float(row.get(key, 0.0)) for row in rows], dtype=np.float64)


def _robust_scale(values: np.ndarray, *, high_is_good: bool = True) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        return values
    low = float(np.percentile(values, 10))
    high = float(np.percentile(values, 90))
    if high <= low + EPS:
        scaled = np.full(values.shape, 0.5 if float(np.mean(values)) > 0 else 0.0, dtype=np.float64)
    else:
        scaled = (values - low) / (high - low)
    scaled = np.clip(scaled, 0.0, 1.0)
    return scaled if high_is_good else 1.0 - scaled


def score_frame_quality_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add transparent per-video component scores and a combined [0, 1] score."""
    if not rows:
        return []

    mean = _array(rows, "mean_intensity")
    p95 = _array(rows, "p95")
    p99 = _array(rows, "p99")
    dynamic_range = _array(rows, "dynamic_range_p99_p01")
    std = _array(rows, "std_intensity")
    local_contrast = _array(rows, "local_contrast_score")
    entropy = _array(rows, "entropy")
    lap = np.log1p(_array(rows, "laplacian_variance") * 100.0)
    tenengrad = np.log1p(_array(rows, "tenengrad_score") * 100.0)
    gradient_std = np.log1p(_array(rows, "gradient_std") * 100.0)
    pattern = _array(rows, "approx_streak_spot_visibility_score")
    horizontal_prom = np.log1p(_array(rows, "horizontal_projection_peak_prominence"))
    vertical_prom = np.log1p(_array(rows, "vertical_projection_peak_prominence"))
    fft_mid = _array(rows, "fft_mid_frequency_power")
    shadow = _array(rows, "shadow_penalty")
    saturated_fraction = _array(rows, "saturated_pixel_fraction")
    bright_fraction = _array(rows, "bright_pixel_fraction")

    brightness_low = clip01((mean - 0.03) / 0.12)
    brightness_high = clip01((0.92 - mean) / 0.22)
    highlight_headroom = clip01((0.995 - p99) / 0.18)
    brightness_score = np.sqrt(np.asarray(brightness_low) * np.asarray(brightness_high))
    brightness_score = 0.80 * brightness_score + 0.20 * np.asarray(highlight_headroom)
    brightness_score = np.asarray(clip01(brightness_score), dtype=np.float64)

    dynamic_range_score = 0.60 * _robust_scale(dynamic_range) + 0.40 * np.asarray(clip01(dynamic_range / 0.30))
    sharpness_score = (
        0.40 * _robust_scale(lap)
        + 0.35 * _robust_scale(tenengrad)
        + 0.15 * _robust_scale(gradient_std)
        + 0.10 * np.asarray(clip01(_array(rows, "gradient_mean") / 0.20))
    )
    pattern_visibility_score = (
        0.45 * _robust_scale(pattern)
        + 0.20 * _robust_scale(horizontal_prom)
        + 0.15 * _robust_scale(vertical_prom)
        + 0.20 * _robust_scale(fft_mid)
    )
    contrast_score = (
        0.35 * _robust_scale(std)
        + 0.35 * _robust_scale(local_contrast)
        + 0.30 * np.asarray(clip01(entropy / 5.5))
    )
    saturation_penalty = np.asarray(
        clip01(np.maximum(saturated_fraction / 0.10, np.maximum(bright_fraction - 0.30, 0.0) / 0.40))
    )
    blur_penalty = np.asarray(clip01(1.0 - sharpness_score))
    low_dynamic_range_penalty = np.asarray(clip01((0.12 - dynamic_range) / 0.12))

    quality = (
        0.18 * brightness_score
        + 0.20 * dynamic_range_score
        + 0.22 * sharpness_score
        + 0.22 * pattern_visibility_score
        + 0.18 * contrast_score
        - 0.35 * shadow
        - 0.25 * saturation_penalty
        - 0.20 * blur_penalty
        - 0.20 * low_dynamic_range_penalty
    )
    quality = np.asarray(clip01(quality), dtype=np.float64)

    scored: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        output = dict(row)
        output["brightness_score"] = finite_float(brightness_score[index])
        output["dynamic_range_score"] = finite_float(dynamic_range_score[index])
        output["sharpness_score"] = finite_float(sharpness_score[index])
        output["pattern_visibility_score"] = finite_float(pattern_visibility_score[index])
        output["contrast_score"] = finite_float(contrast_score[index])
        output["saturation_penalty"] = finite_float(saturation_penalty[index])
        output["blur_penalty"] = finite_float(blur_penalty[index])
        output["low_dynamic_range_penalty"] = finite_float(low_dynamic_range_penalty[index])
        output["quality_score"] = finite_float(quality[index])
        output["possible_bad_frame"] = bool(output.get("possible_bad_frame", False)) or output["quality_score"] < 0.20
        for key in SCORE_KEYS:
            output[key] = finite_float(output.get(key, 0.0))
        scored.append(output)
    return scored


def active_flags(row: dict[str, Any]) -> list[str]:
    return [key for key in FLAG_KEYS if bool(row.get(key, False))]
