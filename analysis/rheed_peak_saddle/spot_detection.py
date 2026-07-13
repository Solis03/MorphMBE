"""Continuous-scale spot detection for synthetic peak-saddle validation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from analysis.rheed_peak_saddle.preprocessing import estimate_smooth_background


@dataclass(frozen=True)
class SpotEstimate:
    spot_id: int
    center_x: float
    center_y: float
    peak_intensity: float
    sigma_x: float
    sigma_y: float
    equivalent_width: float
    eccentricity: float
    local_background: float
    fit_residual: float
    saturation_flag: int
    edge_or_crop_flag: int
    detection_confidence: float


def _mad(values: np.ndarray) -> float:
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def _moment_width(
    image: np.ndarray,
    background: float,
    x: int,
    y: int,
    *,
    radius: int = 9,
) -> tuple[float, float, float, float]:
    height, width = image.shape
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    patch = image[y0:y1, x0:x1].astype(float) - background
    patch = np.maximum(patch, 0.0)
    if float(patch.sum()) <= 1e-9:
        return 99.0, 99.0, 0.0, 1.0
    yy, xx = np.indices(patch.shape, dtype=float)
    xx += x0
    yy += y0
    weight = patch / float(patch.sum())
    cx = float(np.sum(xx * weight))
    cy = float(np.sum(yy * weight))
    var_x = float(np.sum(((xx - cx) ** 2) * weight))
    var_y = float(np.sum(((yy - cy) ** 2) * weight))
    sigma_x = math.sqrt(max(var_x, 1e-6))
    sigma_y = math.sqrt(max(var_y, 1e-6))
    model = float(np.max(patch)) * np.exp(-0.5 * (((xx - cx) / max(sigma_x, 1e-6)) ** 2 + ((yy - cy) / max(sigma_y, 1e-6)) ** 2))
    fit_residual = float(np.mean(np.abs(patch - model)) / max(float(np.max(patch)), 1e-6))
    return sigma_x, sigma_y, float(np.max(patch)), fit_residual


def _local_background(image: np.ndarray, x: int, y: int, *, inner: int = 11, outer: int = 18) -> float:
    height, width = image.shape
    y0 = max(0, y - outer)
    y1 = min(height, y + outer + 1)
    x0 = max(0, x - outer)
    x1 = min(width, x + outer + 1)
    yy, xx = np.indices((y1 - y0, x1 - x0), dtype=float)
    yy += y0
    xx += x0
    rr = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    mask = (rr >= inner) & (rr <= outer)
    values = image[y0:y1, x0:x1][mask]
    if values.size == 0:
        return float(np.percentile(image, 10.0))
    return float(np.median(values))


def detect_spots(
    image: np.ndarray,
    *,
    max_spots: int = 64,
    min_prominence: float | None = None,
    min_distance: float = 9.0,
    max_equivalent_width: float = 10.5,
    saturation_quantile: float = 99.97,
) -> tuple[SpotEstimate, ...]:
    """Detect compact bright RHEED spots without global component thresholding."""
    values = np.asarray(image, dtype=float)
    if values.ndim != 2:
        raise ValueError("image must be 2D")
    background = estimate_smooth_background(values, sigma=20.0)
    corrected = values - background
    smooth = ndimage.gaussian_filter(corrected, sigma=1.0)
    local_max = smooth == ndimage.maximum_filter(smooth, size=7, mode="nearest")
    finite = np.isfinite(smooth)
    robust_sigma = 1.4826 * _mad(smooth[finite])
    threshold = float(np.median(smooth[finite]) + 3.0 * robust_sigma)
    threshold = max(threshold, float(np.percentile(smooth[finite], 92.0)) * 0.45)
    if min_prominence is not None:
        threshold = max(threshold, float(min_prominence))
    candidate_y, candidate_x = np.nonzero(local_max & finite & (smooth > threshold))
    scored: list[tuple[float, int, int]] = []
    for y, x in zip(candidate_y, candidate_x):
        scored.append((float(smooth[y, x]), int(y), int(x)))
    scored.sort(reverse=True)

    saturation_level = float(np.percentile(values[finite], saturation_quantile))
    spots: list[SpotEstimate] = []
    for score, y, x in scored:
        if any(math.hypot(spot.center_x - x, spot.center_y - y) < min_distance for spot in spots):
            continue
        local_bg = _local_background(values, x, y)
        sigma_x, sigma_y, prominence, fit_residual = _moment_width(values, local_bg, x, y)
        equivalent_width = math.sqrt(max(sigma_x * sigma_y, 1e-6))
        eccentricity = 1.0 - min(sigma_x, sigma_y) / max(max(sigma_x, sigma_y), 1e-6)
        edge_flag = int(x < 2.2 * sigma_x or y < 2.2 * sigma_y or x > values.shape[1] - 2.2 * sigma_x or y > values.shape[0] - 2.2 * sigma_y)
        saturation_flag = int(values[y, x] >= saturation_level and saturation_level > np.percentile(values[finite], 99.0))
        if equivalent_width > max_equivalent_width:
            continue
        if prominence < max(threshold * 0.75, 0.025):
            continue
        confidence = float(np.clip((prominence / max(prominence + 3.0 * robust_sigma, 1e-6)) * (1.0 - 0.35 * edge_flag), 0.0, 1.0))
        spots.append(
            SpotEstimate(
                spot_id=len(spots),
                center_x=float(x),
                center_y=float(y),
                peak_intensity=float(values[y, x]),
                sigma_x=float(sigma_x),
                sigma_y=float(sigma_y),
                equivalent_width=float(equivalent_width),
                eccentricity=float(eccentricity),
                local_background=float(local_bg),
                fit_residual=float(fit_residual),
                saturation_flag=saturation_flag,
                edge_or_crop_flag=edge_flag,
                detection_confidence=confidence,
            )
        )
        if len(spots) >= max_spots:
            break
    return tuple(sorted(spots, key=lambda spot: (spot.center_y, spot.center_x)))

