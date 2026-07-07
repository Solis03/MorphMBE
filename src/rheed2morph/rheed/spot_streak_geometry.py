"""Spot, streak, and elongated-bar geometry features for RHEED masks."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from rheed2morph.rheed.frame_quality import finite_float


try:  # pragma: no cover - availability depends on the local environment.
    from scipy import ndimage  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    ndimage = None


COMPONENT_TYPES = [
    "round_spot",
    "elongated_spot",
    "horizontal_bar",
    "vertical_streak",
    "diffuse_blob",
    "artifact_candidate",
]

FRAME_SHAPE_FEATURE_NAMES = [
    "total_component_count",
    "round_spot_count",
    "elongated_spot_count",
    "horizontal_bar_count",
    "vertical_streak_count",
    "diffuse_blob_count",
    "artifact_candidate_count",
    "component_density",
    "mean_area",
    "median_area",
    "area_iqr",
    "mean_aspect_ratio",
    "median_aspect_ratio",
    "mean_eccentricity",
    "median_eccentricity",
    "orientation_entropy",
    "dominant_orientation_deg",
    "centroid_x_mean",
    "centroid_x_std",
    "centroid_y_mean",
    "centroid_y_std",
    "right_side_component_fraction",
    "center_component_fraction",
    "nearest_neighbor_distance_mean",
    "nearest_neighbor_distance_std",
    "horizontal_spacing_peak",
    "vertical_spacing_peak",
    "projection_peak_count_x",
    "projection_peak_count_y",
    "projection_peak_prominence_x",
    "projection_peak_prominence_y",
    "fft_low_power",
    "fft_mid_power",
    "fft_high_power",
    "fft_anisotropy",
    "ring_likeness_proxy",
    "spot_to_streak_ratio",
    "elongated_to_round_ratio",
    "bar_like_score",
    "mask_confidence",
    "artifact_fraction",
    "snr_score",
    "pairwise_distance_hist_0",
    "pairwise_distance_hist_1",
    "pairwise_distance_hist_2",
    "pairwise_distance_hist_3",
    "pairwise_distance_hist_4",
    "pairwise_distance_hist_5",
]


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    binary = np.asarray(mask) > 0
    if ndimage is not None:
        return ndimage.label(binary)
    labels = np.zeros(binary.shape, dtype=np.int32)
    current = 0
    height, width = binary.shape
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or labels[y, x] != 0:
                continue
            current += 1
            stack = [(y, x)]
            labels[y, x] = current
            while stack:
                cy, cx = stack.pop()
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if binary[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return labels, current


def _orientation_entropy(orientations: np.ndarray) -> float:
    if orientations.size == 0:
        return 0.0
    hist, _ = np.histogram(orientations, bins=12, range=(-90.0, 90.0), density=False)
    total = hist.sum()
    if total <= 0:
        return 0.0
    probs = hist.astype(np.float64) / float(total)
    probs = probs[probs > 0]
    return finite_float(-(probs * np.log2(probs)).sum() / math.log2(12))


def _dominant_orientation(orientations: np.ndarray) -> float:
    if orientations.size == 0:
        return 0.0
    hist, edges = np.histogram(orientations, bins=18, range=(-90.0, 90.0), density=False)
    idx = int(np.argmax(hist))
    return finite_float((edges[idx] + edges[idx + 1]) / 2.0)


def _projection_peaks(projection: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(projection, dtype=np.float64)
    if values.size < 3:
        return 0.0, 0.0, 0.0
    threshold = np.percentile(values, 80)
    peaks = []
    for index in range(1, values.size - 1):
        if values[index] >= values[index - 1] and values[index] >= values[index + 1] and values[index] > threshold:
            peaks.append(index)
    prominence = finite_float((values.max() - np.median(values)) / max(np.percentile(values, 95) - np.percentile(values, 5), 1e-8))
    if len(peaks) >= 2:
        spacing = finite_float(np.median(np.diff(peaks)))
    else:
        spacing = 0.0
    return float(len(peaks)), spacing, prominence


def _frequency_features(image: np.ndarray) -> dict[str, float]:
    work = np.asarray(image, dtype=np.float32)
    work = work - finite_float(work.mean())
    if work.size == 0 or finite_float(work.std()) <= 1e-8:
        return {"fft_low_power": 0.0, "fft_mid_power": 0.0, "fft_high_power": 0.0, "fft_anisotropy": 0.0}
    power = np.abs(np.fft.fftshift(np.fft.fft2(work))) ** 2
    height, width = work.shape
    yy, xx = np.indices(work.shape)
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = np.sqrt(((yy - cy) / max(height, 1)) ** 2 + ((xx - cx) / max(width, 1)) ** 2)
    radius = radius / max(finite_float(radius.max()), 1e-8)
    total = finite_float(power.sum(), 0.0) + 1e-8
    row_band = power[max(0, int(cy) - 2) : min(height, int(cy) + 3), :].sum()
    col_band = power[:, max(0, int(cx) - 2) : min(width, int(cx) + 3)].sum()
    return {
        "fft_low_power": finite_float(power[(radius > 0.02) & (radius <= 0.18)].sum() / total),
        "fft_mid_power": finite_float(power[(radius > 0.18) & (radius <= 0.45)].sum() / total),
        "fft_high_power": finite_float(power[radius > 0.45].sum() / total),
        "fft_anisotropy": finite_float(abs(row_band - col_band) / max(row_band + col_band, 1e-8)),
    }


def _component_from_pixels(
    label_id: int,
    y: np.ndarray,
    x: np.ndarray,
    enhanced: np.ndarray,
    artifact_mask: np.ndarray,
    min_area: int,
) -> dict[str, Any] | None:
    area = int(x.size)
    if area < min_area:
        return None
    height, width = enhanced.shape
    x0, x1 = int(x.min()), int(x.max()) + 1
    y0, y1 = int(y.min()), int(y.max()) + 1
    bbox_w = max(1, x1 - x0)
    bbox_h = max(1, y1 - y0)
    centroid_x = finite_float(x.mean() / max(width - 1, 1))
    centroid_y = finite_float(y.mean() / max(height - 1, 1))
    coords = np.stack([x.astype(np.float64), y.astype(np.float64)], axis=1)
    if coords.shape[0] >= 3:
        cov = np.cov(coords, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        major_var = float(eigvals[-1])
        minor_var = float(eigvals[0])
        major_axis_length = 4.0 * math.sqrt(major_var)
        minor_axis_length = 4.0 * math.sqrt(minor_var)
        vec = eigvecs[:, -1]
        orientation = math.degrees(math.atan2(vec[1], vec[0]))
        if orientation > 90:
            orientation -= 180
        if orientation < -90:
            orientation += 180
        eccentricity = math.sqrt(max(0.0, 1.0 - minor_var / max(major_var, 1e-8)))
    else:
        major_axis_length = float(max(bbox_w, bbox_h))
        minor_axis_length = float(min(bbox_w, bbox_h))
        orientation = 0.0
        eccentricity = 0.0
    aspect_ratio = finite_float(bbox_w / max(bbox_h, 1))
    solidity = finite_float(area / max(bbox_w * bbox_h, 1))
    component_values = enhanced[y, x]
    pad = 4
    yy0 = max(0, y0 - pad)
    yy1 = min(height, y1 + pad)
    xx0 = max(0, x0 - pad)
    xx1 = min(width, x1 + pad)
    local_patch = enhanced[yy0:yy1, xx0:xx1]
    local_background = finite_float(np.median(local_patch), 0.0)
    mean_intensity = finite_float(component_values.mean(), 0.0)
    relative_intensity = finite_float((mean_intensity + 1.05) / max(local_background + 1.05, 1e-6))
    near_border = x0 <= 1 or y0 <= 1 or x1 >= width - 1 or y1 >= height - 1
    artifact_fraction = finite_float(artifact_mask[y, x].mean() if artifact_mask.size else 0.0)
    component_type = classify_component(
        area=area,
        aspect_ratio=aspect_ratio,
        eccentricity=eccentricity,
        orientation=orientation,
        solidity=solidity,
        relative_intensity=relative_intensity,
        near_border=near_border,
        artifact_fraction=artifact_fraction,
        image_area=height * width,
    )
    return {
        "component_id": label_id,
        "component_type": component_type,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "area": area,
        "bbox_width": bbox_w,
        "bbox_height": bbox_h,
        "aspect_ratio": aspect_ratio,
        "eccentricity": finite_float(eccentricity),
        "orientation": finite_float(orientation),
        "major_axis_length": finite_float(major_axis_length),
        "minor_axis_length": finite_float(minor_axis_length),
        "solidity": solidity,
        "mean_enhanced_intensity": mean_intensity,
        "local_background": local_background,
        "relative_intensity": relative_intensity,
        "fwhm_major_proxy": finite_float(major_axis_length),
        "fwhm_minor_proxy": finite_float(minor_axis_length),
        "artifact_fraction": artifact_fraction,
        "near_border": bool(near_border),
    }


def classify_component(
    *,
    area: int,
    aspect_ratio: float,
    eccentricity: float,
    orientation: float,
    solidity: float,
    relative_intensity: float,
    near_border: bool,
    artifact_fraction: float,
    image_area: int,
) -> str:
    if near_border or artifact_fraction > 0.35 or relative_intensity > 1.95:
        return "artifact_candidate"
    if area > image_area * 0.045 and relative_intensity < 1.10:
        return "diffuse_blob"
    if aspect_ratio >= 2.8 and abs(orientation) <= 25:
        return "horizontal_bar"
    if aspect_ratio <= 0.40 and abs(orientation) >= 55:
        return "vertical_streak"
    if eccentricity < 0.55 and 0.65 <= aspect_ratio <= 1.55:
        return "round_spot"
    if eccentricity >= 0.55 or aspect_ratio >= 1.55 or aspect_ratio <= 0.65:
        return "elongated_spot"
    if solidity < 0.30:
        return "diffuse_blob"
    return "round_spot"


def extract_components_and_frame_features(
    *,
    soft_mask: np.ndarray,
    enhanced_image: np.ndarray,
    artifact_mask: np.ndarray | None = None,
    min_area: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    mask_values = np.asarray(soft_mask, dtype=np.float32)
    threshold = max(0.35, finite_float(np.percentile(mask_values, 88), 0.35))
    binary = mask_values >= threshold
    if ndimage is not None:
        binary = ndimage.binary_opening(binary, iterations=1)
        binary = ndimage.binary_closing(binary, iterations=1)
    labels, count = _label_mask(binary)
    enhanced = np.asarray(enhanced_image, dtype=np.float32)
    artifacts = np.zeros_like(enhanced, dtype=np.float32) if artifact_mask is None else np.asarray(artifact_mask, dtype=np.float32)
    components: list[dict[str, Any]] = []
    for label_id in range(1, count + 1):
        y, x = np.nonzero(labels == label_id)
        component = _component_from_pixels(label_id, y, x, enhanced, artifacts, min_area)
        if component is not None:
            components.append(component)
    features = summarize_frame_geometry(components, soft_mask=mask_values, enhanced_image=enhanced, artifact_mask=artifacts)
    return components, features


def _pairwise_distance_features(centroids: np.ndarray) -> tuple[np.ndarray, float, float]:
    if centroids.shape[0] < 2:
        return np.zeros(6, dtype=np.float32), 0.0, 0.0
    diffs = centroids[:, None, :] - centroids[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=2))
    distances = distances[np.triu_indices(centroids.shape[0], k=1)]
    hist, _ = np.histogram(distances, bins=6, range=(0.0, math.sqrt(2.0)), density=False)
    hist = hist.astype(np.float32)
    hist = hist / max(float(hist.sum()), 1.0)
    nearest = []
    full = np.sqrt(np.sum(diffs * diffs, axis=2))
    full[full == 0] = np.inf
    nearest = np.min(full, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    return hist, finite_float(nearest.mean() if nearest.size else 0.0), finite_float(nearest.std() if nearest.size else 0.0)


def summarize_frame_geometry(
    components: Sequence[dict[str, Any]],
    *,
    soft_mask: np.ndarray,
    enhanced_image: np.ndarray,
    artifact_mask: np.ndarray,
) -> dict[str, float]:
    height, width = soft_mask.shape
    features = {name: 0.0 for name in FRAME_SHAPE_FEATURE_NAMES}
    total = len(components)
    features["total_component_count"] = float(total)
    for component_type in COMPONENT_TYPES:
        features[f"{component_type}_count"] = float(sum(1 for item in components if item["component_type"] == component_type))
    features["component_density"] = finite_float(total / max((height * width) / 10000.0, 1e-8))
    features["mask_confidence"] = finite_float(np.clip(soft_mask.mean() * 8.0 + np.percentile(soft_mask, 98), 0.0, 1.0))
    features["artifact_fraction"] = finite_float(artifact_mask.mean() if artifact_mask.size else 0.0)
    features["snr_score"] = finite_float(np.clip(np.std(enhanced_image) * 2.5, 0.0, 1.0))
    features.update(_frequency_features(enhanced_image))

    projection_x = soft_mask.sum(axis=0)
    projection_y = soft_mask.sum(axis=1)
    px_count, x_spacing, px_prom = _projection_peaks(projection_x)
    py_count, y_spacing, py_prom = _projection_peaks(projection_y)
    features["projection_peak_count_x"] = px_count
    features["projection_peak_count_y"] = py_count
    features["horizontal_spacing_peak"] = finite_float(x_spacing / max(width, 1))
    features["vertical_spacing_peak"] = finite_float(y_spacing / max(height, 1))
    features["projection_peak_prominence_x"] = px_prom
    features["projection_peak_prominence_y"] = py_prom
    features["ring_likeness_proxy"] = finite_float(min(px_count, py_count) / max(max(px_count, py_count), 1.0))

    if not components:
        return {key: finite_float(value) for key, value in features.items()}

    areas = np.asarray([item["area"] for item in components], dtype=np.float64)
    aspects = np.asarray([item["aspect_ratio"] for item in components], dtype=np.float64)
    eccentricities = np.asarray([item["eccentricity"] for item in components], dtype=np.float64)
    orientations = np.asarray([item["orientation"] for item in components], dtype=np.float64)
    centroids = np.asarray([[item["centroid_x"], item["centroid_y"]] for item in components], dtype=np.float64)
    features["mean_area"] = finite_float(areas.mean())
    features["median_area"] = finite_float(np.median(areas))
    features["area_iqr"] = finite_float(np.percentile(areas, 75) - np.percentile(areas, 25))
    features["mean_aspect_ratio"] = finite_float(aspects.mean())
    features["median_aspect_ratio"] = finite_float(np.median(aspects))
    features["mean_eccentricity"] = finite_float(eccentricities.mean())
    features["median_eccentricity"] = finite_float(np.median(eccentricities))
    features["orientation_entropy"] = _orientation_entropy(orientations)
    features["dominant_orientation_deg"] = _dominant_orientation(orientations)
    features["centroid_x_mean"] = finite_float(centroids[:, 0].mean())
    features["centroid_x_std"] = finite_float(centroids[:, 0].std())
    features["centroid_y_mean"] = finite_float(centroids[:, 1].mean())
    features["centroid_y_std"] = finite_float(centroids[:, 1].std())
    features["right_side_component_fraction"] = finite_float(np.mean(centroids[:, 0] > 0.66))
    features["center_component_fraction"] = finite_float(np.mean((centroids[:, 0] > 0.33) & (centroids[:, 0] < 0.66)))
    hist, nearest_mean, nearest_std = _pairwise_distance_features(centroids)
    for index, value in enumerate(hist):
        features[f"pairwise_distance_hist_{index}"] = finite_float(value)
    features["nearest_neighbor_distance_mean"] = nearest_mean
    features["nearest_neighbor_distance_std"] = nearest_std
    round_count = features["round_spot_count"]
    elongated_count = features["elongated_spot_count"]
    horizontal_count = features["horizontal_bar_count"]
    vertical_count = features["vertical_streak_count"]
    streak_count = horizontal_count + vertical_count + elongated_count
    features["spot_to_streak_ratio"] = finite_float((round_count + elongated_count) / max(horizontal_count + vertical_count, 1.0))
    features["elongated_to_round_ratio"] = finite_float((elongated_count + horizontal_count + vertical_count) / max(round_count, 1.0))
    features["bar_like_score"] = finite_float(
        np.clip(
            0.35 * horizontal_count / max(total, 1)
            + 0.25 * elongated_count / max(total, 1)
            + 0.20 * np.clip(features["mean_aspect_ratio"] / 4.0, 0.0, 1.0)
            + 0.20 * features["mean_eccentricity"],
            0.0,
            1.0,
        )
    )
    return {key: finite_float(value) for key, value in features.items()}


def component_rows_for_csv(sample_id: str, frame_idx: int, components: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for component in components:
        row = {"sample_id": sample_id, "frame_idx": frame_idx}
        row.update(component)
        rows.append(row)
    return rows


def colorize_component_overlay(base: np.ndarray, components: Sequence[dict[str, Any]], soft_mask: np.ndarray) -> np.ndarray:
    gray = np.asarray(base, dtype=np.float32)
    gray = (gray - gray.min()) / max(float(gray.max() - gray.min()), 1e-8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    colors = {
        "round_spot": np.asarray([0.10, 0.65, 1.00], dtype=np.float32),
        "elongated_spot": np.asarray([0.25, 0.85, 0.25], dtype=np.float32),
        "horizontal_bar": np.asarray([1.00, 0.65, 0.05], dtype=np.float32),
        "vertical_streak": np.asarray([0.85, 0.30, 1.00], dtype=np.float32),
        "diffuse_blob": np.asarray([0.80, 0.80, 0.20], dtype=np.float32),
        "artifact_candidate": np.asarray([1.00, 0.10, 0.10], dtype=np.float32),
    }
    if ndimage is not None:
        labels, count = ndimage.label(soft_mask > max(0.35, np.percentile(soft_mask, 88)))
    else:
        labels, count = _label_mask(soft_mask > max(0.35, np.percentile(soft_mask, 88)))
    for component in components:
        label_id = int(component["component_id"])
        if label_id > count:
            continue
        mask = labels == label_id
        color = colors.get(component["component_type"], np.asarray([1.0, 1.0, 1.0], dtype=np.float32))
        rgb[mask] = 0.35 * rgb[mask] + 0.65 * color
    return np.clip(rgb, 0.0, 1.0)

