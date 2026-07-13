"""RHEED frame-quality v2 features with hard artifact rejection.

The v2 scorer separates validity from information quality. It is intentionally
heuristic and inspectable: every hard rejection is tied to a named flag that is
written to CSV for manual review.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from rheed2morph.rheed.frame_quality import (
    EPS,
    clip01,
    enhance_for_display,
    finite_float,
    frame_to_gray_float32,
    normalize_for_display,
    resize_image,
)


try:  # pragma: no cover - availability depends on the local environment.
    import cv2  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None

try:  # pragma: no cover - availability depends on the local environment.
    from scipy import ndimage  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    ndimage = None


FEATURE_KEYS_V2 = [
    "height",
    "width",
    "p01",
    "p05",
    "p50",
    "p95",
    "p99",
    "mean_intensity",
    "std_intensity",
    "dynamic_range_p99_p01",
    "saturated_pixel_fraction",
    "bright_pixel_fraction",
    "dark_pixel_fraction",
    "extreme_pixel_fraction",
    "graylevel_entropy",
    "occupied_histogram_bins",
    "largest_extreme_component_fraction",
    "largest_rectangular_component_fraction",
    "rectangular_component_score",
    "block_edge_score",
    "row_transition_score",
    "column_transition_score",
    "left_shadow_ratio",
    "right_shadow_ratio",
    "top_shadow_ratio",
    "bottom_shadow_ratio",
    "shadow_score",
    "local_contrast_after_bgsub",
    "dog_response_mean",
    "dog_response_p95",
    "log_response_mean",
    "log_response_p95",
    "horizontal_projection_peak_score",
    "vertical_projection_peak_score",
    "projection_peak_score",
    "fft_low_frequency_power",
    "fft_mid_frequency_power",
    "fft_high_frequency_power",
    "fft_anisotropy",
    "plausible_spot_streak_component_count",
    "plausible_spot_streak_score",
    "pattern_visibility_score",
    "non_shadow_score",
    "artifact_penalty",
]

FLAG_KEYS_V2 = [
    "almost_black",
    "almost_white",
    "over_saturated",
    "very_low_dynamic_range",
    "strong_shadow",
    "binary_artifact",
    "blocky_artifact",
    "too_few_gray_levels",
    "extreme_pixel_fraction_too_high",
    "largest_rectangular_component_too_large",
    "no_plausible_rheed_pattern",
    "isolated_quality_spike",
]

CRITICAL_REJECT_FLAGS_V2 = {
    "almost_black",
    "almost_white",
    "over_saturated",
    "very_low_dynamic_range",
    "strong_shadow",
    "binary_artifact",
    "blocky_artifact",
    "too_few_gray_levels",
    "extreme_pixel_fraction_too_high",
    "largest_rectangular_component_too_large",
    "no_plausible_rheed_pattern",
    "isolated_quality_spike",
}

SCORE_KEYS_V2 = [
    "validity_score",
    "information_score",
    "temporal_consistency_score",
    "final_score",
]


def active_reject_flags_v2(row: dict[str, Any]) -> list[str]:
    return [key for key in FLAG_KEYS_V2 if bool(row.get(key, False))]


def critical_reject_flags_v2(row: dict[str, Any]) -> list[str]:
    return [key for key in active_reject_flags_v2(row) if key in CRITICAL_REJECT_FLAGS_V2]


def passes_hard_reject_v2(row: dict[str, Any]) -> bool:
    return not critical_reject_flags_v2(row)


def _histogram_stats(gray: np.ndarray) -> tuple[float, float]:
    values = np.asarray(gray, dtype=np.float32)
    hist, _ = np.histogram(values.ravel(), bins=64, range=(0.0, 1.0), density=False)
    total = int(hist.sum())
    if total <= 0:
        return 0.0, 0.0
    probs = hist.astype(np.float64) / float(total)
    occupied = int(np.count_nonzero(hist))
    probs = probs[probs > 0]
    entropy = finite_float(-(probs * np.log2(probs)).sum())
    return entropy, float(occupied)


def _gaussian_filter(image: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    if cv2 is not None:
        ksize = max(3, int(round(sigma * 6)) | 1)
        return cv2.GaussianBlur(values, (ksize, ksize), sigmaX=sigma, sigmaY=sigma).astype(np.float32, copy=False)
    if ndimage is not None:
        return ndimage.gaussian_filter(values, sigma=sigma).astype(np.float32, copy=False)
    radius = max(1, int(round(2 * sigma)))
    padded = np.pad(values, radius, mode="reflect")
    output = np.zeros_like(values, dtype=np.float32)
    size = radius * 2 + 1
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            output[y, x] = finite_float(np.mean(padded[y : y + size, x : x + size]))
    return output


def _dog_log_features(gray: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    work = resize_image(gray, max_side=192).astype(np.float32, copy=False)
    narrow = _gaussian_filter(work, sigma=1.0)
    wide = _gaussian_filter(work, sigma=4.0)
    dog = narrow - wide
    if cv2 is not None:
        log = cv2.Laplacian(_gaussian_filter(work, sigma=1.2), cv2.CV_32F, ksize=3)
    else:
        log = (
            -4.0 * work
            + np.roll(work, 1, axis=0)
            + np.roll(work, -1, axis=0)
            + np.roll(work, 1, axis=1)
            + np.roll(work, -1, axis=1)
        )
    bgsub = work - _gaussian_filter(work, sigma=max(8.0, min(work.shape) / 18.0))
    local_contrast = finite_float(np.std(bgsub))
    return bgsub, {
        "local_contrast_after_bgsub": local_contrast,
        "dog_response_mean": finite_float(np.mean(np.abs(dog))),
        "dog_response_p95": finite_float(np.percentile(np.abs(dog), 95)),
        "log_response_mean": finite_float(np.mean(np.abs(log))),
        "log_response_p95": finite_float(np.percentile(np.abs(log), 95)),
    }


def _component_stats(mask: np.ndarray) -> tuple[float, float, float, int]:
    binary = np.asarray(mask, dtype=bool)
    if binary.size == 0 or not binary.any():
        return 0.0, 0.0, 0.0, 0
    height, width = binary.shape
    image_area = float(height * width)
    if cv2 is not None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        if count <= 1:
            return 0.0, 0.0, 0.0, 0
        largest_area = 0.0
        largest_rect = 0.0
        largest_rect_score = 0.0
        kept = 0
        for label_id in range(1, count):
            area = float(stats[label_id, cv2.CC_STAT_AREA])
            bbox_area = float(max(1, stats[label_id, cv2.CC_STAT_WIDTH] * stats[label_id, cv2.CC_STAT_HEIGHT]))
            area_frac = area / image_area
            rect_frac = bbox_area / image_area
            fill = area / bbox_area
            largest_area = max(largest_area, area_frac)
            largest_rect = max(largest_rect, rect_frac)
            largest_rect_score = max(largest_rect_score, area_frac * fill)
            if area >= max(4.0, image_area * 0.00025):
                kept += 1
        return finite_float(largest_area), finite_float(largest_rect), finite_float(largest_rect_score), kept
    if ndimage is not None:
        labels, count = ndimage.label(binary)
        largest_area = 0.0
        largest_rect = 0.0
        largest_rect_score = 0.0
        kept = 0
        for label_id in range(1, count + 1):
            y, x = np.where(labels == label_id)
            if x.size == 0:
                continue
            area = float(x.size)
            bbox_area = float((x.max() - x.min() + 1) * (y.max() - y.min() + 1))
            area_frac = area / image_area
            rect_frac = bbox_area / image_area
            fill = area / max(bbox_area, 1.0)
            largest_area = max(largest_area, area_frac)
            largest_rect = max(largest_rect, rect_frac)
            largest_rect_score = max(largest_rect_score, area_frac * fill)
            if area >= max(4.0, image_area * 0.00025):
                kept += 1
        return finite_float(largest_area), finite_float(largest_rect), finite_float(largest_rect_score), kept
    row_frac = finite_float(np.max(binary.mean(axis=1)))
    col_frac = finite_float(np.max(binary.mean(axis=0)))
    area_frac = finite_float(binary.mean())
    proxy = max(row_frac, col_frac) * area_frac
    return area_frac, proxy, proxy, int(area_frac > 0)


def _block_transition_score(gray: np.ndarray) -> tuple[float, float, float]:
    work = resize_image(gray, max_side=128).astype(np.float32, copy=False)
    row_diff = np.abs(np.diff(work, axis=0))
    col_diff = np.abs(np.diff(work, axis=1))
    threshold = max(0.18, finite_float(np.percentile(np.abs(work - np.median(work)), 95), 0.0) * 0.75)
    row_score = finite_float(np.mean(row_diff > threshold)) if row_diff.size else 0.0
    col_score = finite_float(np.mean(col_diff > threshold)) if col_diff.size else 0.0
    return finite_float(max(row_score, col_score)), row_score, col_score


def _shadow_ratios(gray: np.ndarray) -> tuple[float, float, float, float, float]:
    height, width = gray.shape
    edge_w = max(1, int(round(width * 0.12)))
    edge_h = max(1, int(round(height * 0.12)))
    center = gray[
        max(0, int(height * 0.25)) : max(1, int(height * 0.75)),
        max(0, int(width * 0.25)) : max(1, int(width * 0.75)),
    ]
    center_mean = finite_float(center.mean() if center.size else gray.mean(), 0.0)
    denom = max(center_mean, 0.04)
    ratios = [
        finite_float(1.0 - np.mean(gray[:, :edge_w]) / denom),
        finite_float(1.0 - np.mean(gray[:, width - edge_w :]) / denom),
        finite_float(1.0 - np.mean(gray[:edge_h, :]) / denom),
        finite_float(1.0 - np.mean(gray[height - edge_h :, :]) / denom),
    ]
    ratios = [finite_float(clip01(value)) for value in ratios]
    return ratios[0], ratios[1], ratios[2], ratios[3], finite_float(max(ratios))


def _projection_peak_score(projection: np.ndarray) -> float:
    values = np.asarray(projection, dtype=np.float64)
    if values.size < 3:
        return 0.0
    smoothed = values
    if ndimage is not None:
        smoothed = ndimage.gaussian_filter1d(values, sigma=1.0)
    centered = smoothed - np.median(smoothed)
    spread = np.percentile(smoothed, 95) - np.percentile(smoothed, 5)
    prominence = finite_float(np.max(centered) / max(spread, EPS))
    local_maxima = 0
    threshold = np.percentile(smoothed, 80)
    for idx in range(1, smoothed.size - 1):
        if smoothed[idx] >= smoothed[idx - 1] and smoothed[idx] >= smoothed[idx + 1] and smoothed[idx] > threshold:
            local_maxima += 1
    return finite_float(clip01(0.65 * clip01(prominence / 1.5) + 0.35 * clip01(local_maxima / 8.0)))


def _frequency_features(gray: np.ndarray) -> dict[str, float]:
    work = resize_image(gray, max_side=128).astype(np.float32, copy=False)
    work = work - finite_float(work.mean(), 0.0)
    if work.size == 0 or finite_float(work.std(), 0.0) <= EPS:
        return {
            "fft_low_frequency_power": 0.0,
            "fft_mid_frequency_power": 0.0,
            "fft_high_frequency_power": 0.0,
            "fft_anisotropy": 0.0,
        }
    spectrum = np.fft.fftshift(np.fft.fft2(work))
    power = np.abs(spectrum) ** 2
    height, width = work.shape
    yy, xx = np.indices(work.shape)
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = np.sqrt(((yy - cy) / max(height, 1)) ** 2 + ((xx - cx) / max(width, 1)) ** 2)
    radius = radius / max(finite_float(radius.max()), EPS)
    total = finite_float(power.sum(), 0.0) + EPS
    row_band = power[max(0, int(cy) - 2) : min(height, int(cy) + 3), :].sum()
    col_band = power[:, max(0, int(cx) - 2) : min(width, int(cx) + 3)].sum()
    return {
        "fft_low_frequency_power": finite_float(power[(radius > 0.02) & (radius <= 0.18)].sum() / total),
        "fft_mid_frequency_power": finite_float(power[(radius > 0.18) & (radius <= 0.45)].sum() / total),
        "fft_high_frequency_power": finite_float(power[radius > 0.45].sum() / total),
        "fft_anisotropy": finite_float(abs(row_band - col_band) / max(row_band + col_band, EPS)),
    }


def _pattern_component_count(gray: np.ndarray, bgsub: np.ndarray, dog_response_p95: float) -> int:
    work = resize_image(gray, size=bgsub.shape[0] if bgsub.shape[0] == bgsub.shape[1] else None, max_side=None)
    if work.shape != bgsub.shape:
        work = resize_image(gray, size=min(bgsub.shape))
    score = np.maximum(bgsub, 0.0)
    score = score + 0.5 * np.abs(bgsub)
    threshold = max(
        finite_float(np.percentile(score, 88), 0.0),
        finite_float(np.median(score) + 1.2 * np.std(score), 0.0),
        dog_response_p95 * 0.25,
    )
    mask = score > threshold
    largest_area, largest_rect, _rect_score, count = _component_stats(mask)
    _ = largest_area, largest_rect
    return int(count)


def extract_frame_quality_features_v2(frame: np.ndarray) -> dict[str, Any]:
    gray = frame_to_gray_float32(frame)
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError(f"Expected a non-empty grayscale frame, got shape {gray.shape}.")

    height, width = gray.shape
    p01, p05, p50, p95, p99 = [finite_float(value) for value in np.percentile(gray, [1, 5, 50, 95, 99])]
    mean_intensity = finite_float(gray.mean())
    std_intensity = finite_float(gray.std())
    dynamic_range = finite_float(p99 - p01)
    saturated_fraction = finite_float(np.mean(gray >= 0.985))
    bright_fraction = finite_float(np.mean(gray >= 0.90))
    dark_fraction = finite_float(np.mean(gray <= 0.05))
    extreme_mask = (gray < 0.02) | (gray > 0.98)
    extreme_fraction = finite_float(extreme_mask.mean())
    entropy, occupied_bins = _histogram_stats(gray)
    largest_extreme, largest_rect, rectangular_score, _ = _component_stats(extreme_mask)
    block_edge, row_transition, column_transition = _block_transition_score(gray)
    left_shadow, right_shadow, top_shadow, bottom_shadow, shadow_score = _shadow_ratios(gray)
    bgsub, response_features = _dog_log_features(gray)
    freq = _frequency_features(gray)

    horizontal_projection_score = _projection_peak_score(gray.mean(axis=1))
    vertical_projection_score = _projection_peak_score(gray.mean(axis=0))
    projection_peak_score = finite_float(max(horizontal_projection_score, vertical_projection_score))
    component_count = _pattern_component_count(gray, bgsub, response_features["dog_response_p95"])

    local_contrast_score = finite_float(clip01(response_features["local_contrast_after_bgsub"] / 0.08))
    dog_score = finite_float(clip01(response_features["dog_response_p95"] / 0.08))
    log_score = finite_float(clip01(response_features["log_response_p95"] / 0.18))
    fft_score = finite_float(
        clip01(0.60 * clip01(freq["fft_mid_frequency_power"] / 0.45) + 0.40 * freq["fft_anisotropy"])
    )
    projection_score = projection_peak_score
    component_score = finite_float(clip01(component_count / 8.0))
    plausible_spot_streak_score = finite_float(
        clip01(0.40 * component_score + 0.25 * projection_score + 0.20 * dog_score + 0.15 * fft_score)
    )
    pattern_visibility_score = finite_float(
        clip01(0.35 * local_contrast_score + 0.25 * dog_score + 0.20 * log_score + 0.20 * projection_score)
    )
    non_shadow_score = finite_float(1.0 - shadow_score)

    almost_black = mean_intensity < 0.025 or p99 < 0.075
    almost_white = mean_intensity > 0.965 or p05 > 0.93
    over_saturated = saturated_fraction > 0.35 or (saturated_fraction > 0.18 and p95 > 0.99)
    very_low_dynamic_range = dynamic_range < 0.040 or std_intensity < 0.010
    too_few_gray_levels = occupied_bins <= 6 and dynamic_range > 0.10
    extreme_pixel_fraction_too_high = extreme_fraction > 0.65
    largest_rectangular_component_too_large = largest_rect > 0.32 or rectangular_score > 0.20
    binary_artifact = (
        (occupied_bins <= 8 and extreme_fraction > 0.28 and dynamic_range > 0.35)
        or (entropy < 1.40 and extreme_fraction > 0.22 and largest_extreme > 0.08)
        or (largest_rect > 0.18 and extreme_fraction > 0.22 and occupied_bins <= 16)
    )
    blocky_artifact = (
        (largest_rect > 0.16 and rectangular_score > 0.08 and extreme_fraction > 0.12)
        or (block_edge > 0.10 and occupied_bins <= 18 and dynamic_range > 0.35)
        or (largest_rectangular_component_too_large and extreme_fraction > 0.08)
    )
    strong_shadow = shadow_score > 0.70 and dark_fraction > 0.12
    has_plausible_pattern = (
        pattern_visibility_score >= 0.12
        and plausible_spot_streak_score >= 0.08
        and local_contrast_score >= 0.04
    ) or (
        projection_peak_score >= 0.35
        and local_contrast_score >= 0.04
        and not (binary_artifact or blocky_artifact)
    )
    no_plausible_rheed_pattern = not has_plausible_pattern

    artifact_penalty = finite_float(
        clip01(
            0.25 * clip01(extreme_fraction / 0.45)
            + 0.25 * clip01(rectangular_score / 0.16)
            + 0.20 * clip01(block_edge / 0.10)
            + 0.15 * float(binary_artifact)
            + 0.15 * float(blocky_artifact)
        )
    )
    validity_score = finite_float(
        clip01(
            1.0
            - 0.24 * float(almost_black or almost_white)
            - 0.20 * float(over_saturated)
            - 0.18 * float(very_low_dynamic_range)
            - 0.25 * float(binary_artifact)
            - 0.25 * float(blocky_artifact)
            - 0.16 * float(too_few_gray_levels)
            - 0.18 * float(extreme_pixel_fraction_too_high)
            - 0.20 * float(largest_rectangular_component_too_large)
            - 0.18 * shadow_score
            - 0.16 * float(no_plausible_rheed_pattern)
        )
    )
    information_score = finite_float(
        clip01(
            0.30 * pattern_visibility_score
            + 0.25 * plausible_spot_streak_score
            + 0.18 * local_contrast_score
            + 0.12 * projection_peak_score
            + 0.10 * fft_score
            + 0.05 * non_shadow_score
        )
    )

    features: dict[str, Any] = {
        "height": float(height),
        "width": float(width),
        "p01": p01,
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "dynamic_range_p99_p01": dynamic_range,
        "saturated_pixel_fraction": saturated_fraction,
        "bright_pixel_fraction": bright_fraction,
        "dark_pixel_fraction": dark_fraction,
        "extreme_pixel_fraction": extreme_fraction,
        "graylevel_entropy": entropy,
        "occupied_histogram_bins": occupied_bins,
        "largest_extreme_component_fraction": largest_extreme,
        "largest_rectangular_component_fraction": largest_rect,
        "rectangular_component_score": rectangular_score,
        "block_edge_score": block_edge,
        "row_transition_score": row_transition,
        "column_transition_score": column_transition,
        "left_shadow_ratio": left_shadow,
        "right_shadow_ratio": right_shadow,
        "top_shadow_ratio": top_shadow,
        "bottom_shadow_ratio": bottom_shadow,
        "shadow_score": shadow_score,
        "horizontal_projection_peak_score": horizontal_projection_score,
        "vertical_projection_peak_score": vertical_projection_score,
        "projection_peak_score": projection_peak_score,
        "plausible_spot_streak_component_count": float(component_count),
        "plausible_spot_streak_score": plausible_spot_streak_score,
        "pattern_visibility_score": pattern_visibility_score,
        "non_shadow_score": non_shadow_score,
        "artifact_penalty": artifact_penalty,
        "almost_black": bool(almost_black),
        "almost_white": bool(almost_white),
        "over_saturated": bool(over_saturated),
        "very_low_dynamic_range": bool(very_low_dynamic_range),
        "strong_shadow": bool(strong_shadow),
        "binary_artifact": bool(binary_artifact),
        "blocky_artifact": bool(blocky_artifact),
        "too_few_gray_levels": bool(too_few_gray_levels),
        "extreme_pixel_fraction_too_high": bool(extreme_pixel_fraction_too_high),
        "largest_rectangular_component_too_large": bool(largest_rectangular_component_too_large),
        "no_plausible_rheed_pattern": bool(no_plausible_rheed_pattern),
        "isolated_quality_spike": False,
        "validity_score": validity_score,
        "information_score": information_score,
        "temporal_consistency_score": 1.0,
        "final_score": 0.0,
    }
    features.update(response_features)
    features.update(freq)
    for key in FEATURE_KEYS_V2 + SCORE_KEYS_V2:
        features[key] = finite_float(features.get(key, 0.0))
    return features


def normalized_pattern_image(gray: np.ndarray) -> np.ndarray:
    work = resize_image(frame_to_gray_float32(gray), size=96)
    bgsub = work - _gaussian_filter(work, sigma=8.0)
    low = finite_float(np.percentile(bgsub, 2), 0.0)
    high = finite_float(np.percentile(bgsub, 98), 1.0)
    if high <= low + EPS:
        high = low + 1.0
    norm = np.clip((bgsub - low) / (high - low), 0.0, 1.0)
    norm = (norm - finite_float(norm.mean(), 0.0)) / max(finite_float(norm.std(), 0.0), 1e-6)
    return np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def frame_similarity_v2(a: np.ndarray, b: np.ndarray) -> float:
    aa = normalized_pattern_image(a)
    bb = normalized_pattern_image(b)
    distance = finite_float(np.sqrt(np.mean((aa - bb) ** 2)) / 2.0)
    return finite_float(np.exp(-2.2 * distance))


def add_temporal_consistency_scores_v2(
    rows: Sequence[dict[str, Any]],
    frame_images: dict[int, np.ndarray],
    *,
    enabled: bool = True,
    window: int = 2,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row.get("sample_order", row.get("frame_idx", 0))))
    if not enabled or len(ordered) == 1:
        for row in ordered:
            row["temporal_consistency_score"] = 1.0
            row["isolated_quality_spike"] = False
            row["final_score"] = compute_final_score_v2(row)
        return ordered

    validity = np.asarray([finite_float(row.get("validity_score", 0.0)) for row in ordered], dtype=np.float64)
    information = np.asarray([finite_float(row.get("information_score", 0.0)) for row in ordered], dtype=np.float64)
    for index, row in enumerate(ordered):
        frame_idx = int(row.get("frame_idx", -1))
        sims: list[float] = []
        for neighbor_index in (index - 1, index + 1):
            if 0 <= neighbor_index < len(ordered):
                neighbor_idx = int(ordered[neighbor_index].get("frame_idx", -1))
                if frame_idx in frame_images and neighbor_idx in frame_images:
                    sims.append(frame_similarity_v2(frame_images[frame_idx], frame_images[neighbor_idx]))
        local_start = max(0, index - window)
        local_end = min(len(ordered), index + window + 1)
        local_validity = finite_float(np.median(validity[local_start:local_end]), 0.0)
        local_information = finite_float(np.median(information[local_start:local_end]), 0.0)
        similarity = finite_float(np.mean(sims) if sims else 0.5)
        local_score = finite_float(clip01(0.45 * similarity + 0.30 * local_validity + 0.25 * local_information))
        spike = (
            finite_float(row.get("information_score", 0.0)) > max(0.40, local_information + 0.22)
            and (similarity < 0.22 or local_validity < 0.35)
        )
        if spike:
            local_score = min(local_score, 0.15)
            row["isolated_quality_spike"] = True
        else:
            row["isolated_quality_spike"] = False
        row["temporal_consistency_score"] = finite_float(local_score)
        row["final_score"] = compute_final_score_v2(row)
    return ordered


def compute_final_score_v2(row: dict[str, Any]) -> float:
    final = finite_float(row.get("validity_score", 0.0)) * (
        0.25 * finite_float(row.get("pattern_visibility_score", 0.0))
        + 0.20 * finite_float(row.get("plausible_spot_streak_score", 0.0))
        + 0.15 * finite_float(row.get("local_contrast_after_bgsub", 0.0)) / 0.08
        + 0.15 * finite_float(row.get("projection_peak_score", 0.0))
        + 0.10 * (
            0.60 * clip01(finite_float(row.get("fft_mid_frequency_power", 0.0)) / 0.45)
            + 0.40 * finite_float(row.get("fft_anisotropy", 0.0))
        )
        + 0.10 * finite_float(row.get("temporal_consistency_score", 1.0))
        + 0.05 * finite_float(row.get("non_shadow_score", 0.0))
    ) - finite_float(row.get("artifact_penalty", 0.0))
    return finite_float(clip01(final))


def assign_sample_status_v2(
    *,
    accepted_rows: Sequence[dict[str, Any]],
    min_accepted_for_good: int,
    min_accepted_for_usable: int,
    min_accepted_for_low_confidence: int,
) -> tuple[str, str, bool]:
    count = len(accepted_rows)
    if count <= 0:
        return "EXCLUDE", "no frames passed the v2 hard validity gate", True
    mean_final = finite_float(np.mean([finite_float(row.get("final_score", 0.0)) for row in accepted_rows]), 0.0)
    mean_validity = finite_float(np.mean([finite_float(row.get("validity_score", 0.0)) for row in accepted_rows]), 0.0)
    if count >= min_accepted_for_good and mean_final >= 0.30 and mean_validity >= 0.55:
        return "GOOD", f"{count} accepted frames with acceptable mean v2 quality", False
    if count >= min_accepted_for_usable and mean_validity >= 0.45:
        return "USABLE", f"{count} accepted frames passed v2 hard rejection", False
    if count >= min_accepted_for_low_confidence:
        return "LOW_CONFIDENCE", f"only {count} accepted frame(s) or low mean quality", True
    return "EXCLUDE", f"{count} accepted frames is below the low-confidence minimum", True


def status_rank_v2(status: str) -> int:
    ranks = {"EXCLUDE": 0, "LOW_CONFIDENCE": 1, "USABLE": 2, "GOOD": 3}
    return ranks.get(status, 0)
