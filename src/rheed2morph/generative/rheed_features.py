"""Deterministic handcrafted RHEED features."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _entropy(values: np.ndarray) -> float:
    arr = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(np.sum(arr))
    if total <= 1e-12:
        return 0.0
    probs = arr / total
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log(probs)) / np.log(max(len(arr), 2)))


def _peak_location_width(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or float(np.max(arr)) <= 1e-12:
        return 0.0, 0.0
    index = int(np.argmax(arr))
    threshold = float(np.max(arr)) * 0.5
    mask = arr >= threshold
    positions = np.where(mask)[0]
    width = float(positions[-1] - positions[0] + 1) / float(arr.size) if positions.size else 0.0
    return float(index) / float(max(arr.size - 1, 1)), width


def _laplacian_sharpness(frame: np.ndarray) -> float:
    lap = -4.0 * frame.copy()
    lap[1:, :] += frame[:-1, :]
    lap[:-1, :] += frame[1:, :]
    lap[:, 1:] += frame[:, :-1]
    lap[:, :-1] += frame[:, 1:]
    return float(np.var(lap))


def _fft_features(frame: np.ndarray) -> dict[str, float]:
    centered = frame.astype(np.float64) - float(np.mean(frame))
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    h, w = power.shape
    yy, xx = np.indices(power.shape)
    rr = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2)
    max_r = float(rr.max())
    low = float(np.mean(np.log1p(power[rr <= max_r / 3.0])))
    mid = float(np.mean(np.log1p(power[(rr > max_r / 3.0) & (rr <= 2.0 * max_r / 3.0)])))
    high = float(np.mean(np.log1p(power[rr > 2.0 * max_r / 3.0])))
    horizontal = float(np.sum(power[h // 2 - 1 : h // 2 + 2, :]))
    vertical = float(np.sum(power[:, w // 2 - 1 : w // 2 + 2]))
    anisotropy = abs(horizontal - vertical) / max(horizontal + vertical, 1e-12)
    return {
        "fft_low_power": low,
        "fft_mid_power": mid,
        "fft_high_power": high,
        "fft_anisotropy": float(anisotropy),
    }


def _frame_features(frame: np.ndarray) -> dict[str, float]:
    image = np.asarray(frame, dtype=np.float32)
    gy, gx = np.gradient(image)
    grad = np.sqrt(gx * gx + gy * gy)
    horizontal_projection = image.mean(axis=0)
    vertical_projection = image.mean(axis=1)
    h_peak, h_width = _peak_location_width(horizontal_projection)
    v_peak, v_width = _peak_location_width(vertical_projection)
    total = float(np.sum(np.maximum(image, 0.0)))
    if total <= 1e-12:
        com_x = 0.5
        com_y = 0.5
    else:
        yy, xx = np.indices(image.shape)
        com_x = float(np.sum(xx * image) / total) / float(max(image.shape[1] - 1, 1))
        com_y = float(np.sum(yy * image) / total) / float(max(image.shape[0] - 1, 1))
    output = {
        "mean_intensity": float(np.mean(image)),
        "std_intensity": float(np.std(image)),
        "p01_intensity": float(np.percentile(image, 1.0)),
        "p05_intensity": float(np.percentile(image, 5.0)),
        "p50_intensity": float(np.percentile(image, 50.0)),
        "p95_intensity": float(np.percentile(image, 95.0)),
        "p99_intensity": float(np.percentile(image, 99.0)),
        "saturated_fraction": float(np.mean(image >= 0.99)),
        "dark_fraction": float(np.mean(image <= 0.01)),
        "laplacian_sharpness": _laplacian_sharpness(image),
        "gradient_mean": float(np.mean(grad)),
        "gradient_std": float(np.std(grad)),
        "horizontal_projection_entropy": _entropy(horizontal_projection),
        "vertical_projection_entropy": _entropy(vertical_projection),
        "horizontal_peak_location": h_peak,
        "horizontal_peak_width": h_width,
        "vertical_peak_location": v_peak,
        "vertical_peak_width": v_width,
        "intensity_center_of_mass_x": com_x,
        "intensity_center_of_mass_y": com_y,
    }
    output.update(_fft_features(image))
    return output


def compute_rheed_features(video_tensor: np.ndarray) -> dict[str, float]:
    """Compute finite deterministic features from [T, 1, H, W] RHEED tensors."""

    array = np.asarray(video_tensor, dtype=np.float32)
    if array.ndim != 4 or array.shape[1] != 1:
        raise ValueError(f"Expected RHEED tensor [T,1,H,W], got {array.shape}")
    frames = array[:, 0]
    per_frame = [_frame_features(frame) for frame in frames]
    names = list(per_frame[0])
    features: dict[str, float] = {}
    for name in names:
        values = np.asarray([row[name] for row in per_frame], dtype=np.float64)
        features[f"{name}_mean"] = float(np.mean(values))
        features[f"{name}_std"] = float(np.std(values))
    if frames.shape[0] >= 2:
        features["temporal_mean_abs_frame_difference"] = float(np.mean(np.abs(np.diff(frames, axis=0))))
    else:
        features["temporal_mean_abs_frame_difference"] = 0.0
    frame_means = np.asarray([row["mean_intensity"] for row in per_frame], dtype=np.float64)
    sharpness = np.asarray([row["laplacian_sharpness"] for row in per_frame], dtype=np.float64)
    features["temporal_std_frame_mean_intensity"] = float(np.std(frame_means))
    features["temporal_std_frame_sharpness"] = float(np.std(sharpness))
    return {key: _finite(value) for key, value in features.items()}


def impute_feature_rows(rows: list[dict[str, Any]], feature_columns: list[str], train_mask: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, float], dict[str, float]]:
    matrix = np.asarray([[float(row.get(col, "nan")) for col in feature_columns] for row in rows], dtype=np.float64)
    counts = {col: int(np.sum(~np.isfinite(matrix[:, index]))) for index, col in enumerate(feature_columns)}
    train = matrix[train_mask] if np.any(train_mask) else matrix
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    imputed = matrix.copy()
    for index in range(imputed.shape[1]):
        mask = ~np.isfinite(imputed[:, index])
        imputed[mask, index] = medians[index]
    train_imputed = imputed[train_mask] if np.any(train_mask) else imputed
    means = np.mean(train_imputed, axis=0)
    stds = np.std(train_imputed, axis=0)
    stds = np.where(stds > 1e-8, stds, 1.0)
    updated = []
    for row_index, row in enumerate(rows):
        new_row = dict(row)
        for col_index, col in enumerate(feature_columns):
            new_row[col] = f"{float(imputed[row_index, col_index]):.10g}"
        updated.append(new_row)
    return (
        updated,
        counts,
        {col: float(means[index]) for index, col in enumerate(feature_columns)},
        {col: float(stds[index]) for index, col in enumerate(feature_columns)},
    )
