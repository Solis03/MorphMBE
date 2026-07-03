"""AFM morphology descriptor extraction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from rheed2morph.generative.common import replace_nonfinite


DESCRIPTOR_NAMES = [
    "height_mean",
    "height_std",
    "rq",
    "ra",
    "peak_to_valley",
    "p01",
    "p05",
    "p50",
    "p95",
    "p99",
    "robust_range",
    "skewness",
    "kurtosis",
    "mean_abs_gradient",
    "gradient_std",
    "gradient_orientation_entropy",
    "gradient_anisotropy",
    "psd_low_power",
    "psd_mid_power",
    "psd_high_power",
    "psd_slope",
    "autocorrelation_length_px",
    "island_coverage",
    "island_count",
    "island_mean_area_px",
]


def _safe_float(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return output if math.isfinite(output) else float("nan")


def _gradient_descriptors(array: np.ndarray) -> dict[str, float]:
    gy, gx = np.gradient(array.astype(np.float64))
    magnitude = np.sqrt(gx * gx + gy * gy)
    mean_abs_gradient = float(np.mean(magnitude))
    gradient_std = float(np.std(magnitude))
    angles = np.arctan2(gy, gx).ravel()
    weights = magnitude.ravel()
    if float(np.sum(weights)) <= 1e-12:
        orientation_entropy = 0.0
    else:
        hist, _ = np.histogram(angles, bins=18, range=(-np.pi, np.pi), weights=weights)
        probs = hist.astype(np.float64) / max(float(np.sum(hist)), 1e-12)
        probs = probs[probs > 0.0]
        orientation_entropy = float(-np.sum(probs * np.log(probs)) / np.log(18.0))
    cov = np.cov(np.stack([gx.ravel(), gy.ravel()], axis=0))
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals[-1] <= 1e-12:
        anisotropy = 0.0
    else:
        anisotropy = float((eigvals[-1] - eigvals[0]) / eigvals[-1])
    return {
        "mean_abs_gradient": mean_abs_gradient,
        "gradient_std": gradient_std,
        "gradient_orientation_entropy": orientation_entropy,
        "gradient_anisotropy": anisotropy,
    }


def _radial_profile(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = values.shape
    yy, xx = np.indices(values.shape)
    radius = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2).astype(np.int32)
    max_radius = int(radius.max())
    sums = np.bincount(radius.ravel(), weights=values.ravel(), minlength=max_radius + 1)
    counts = np.bincount(radius.ravel(), minlength=max_radius + 1)
    profile = sums / np.maximum(counts, 1)
    radii = np.arange(profile.shape[0], dtype=np.float64)
    return radii, profile.astype(np.float64)


def _psd_descriptors(array: np.ndarray) -> dict[str, float]:
    centered = array.astype(np.float64) - float(np.mean(array))
    spectrum = np.fft.fftshift(np.fft.fft2(centered))
    power = np.abs(spectrum) ** 2
    radii, profile = _radial_profile(power)
    if profile.shape[0] < 8:
        return {
            "psd_low_power": float("nan"),
            "psd_mid_power": float("nan"),
            "psd_high_power": float("nan"),
            "psd_slope": float("nan"),
        }
    profile = profile[1:]
    radii = radii[1:]
    n = profile.shape[0]
    low = float(np.mean(np.log1p(profile[: max(1, n // 3)])))
    mid = float(np.mean(np.log1p(profile[max(1, n // 3) : max(2, 2 * n // 3)])))
    high = float(np.mean(np.log1p(profile[max(2, 2 * n // 3) :])))
    mask = (radii > 0) & (profile > 0)
    if int(np.sum(mask)) >= 3:
        slope = float(np.polyfit(np.log(radii[mask]), np.log(profile[mask]), deg=1)[0])
    else:
        slope = float("nan")
    return {
        "psd_low_power": low,
        "psd_mid_power": mid,
        "psd_high_power": high,
        "psd_slope": slope,
    }


def _autocorrelation_length(array: np.ndarray) -> float:
    centered = array.astype(np.float64) - float(np.mean(array))
    variance = float(np.var(centered))
    if variance <= 1e-12:
        return 0.0
    spectrum = np.fft.fft2(centered)
    corr = np.fft.fftshift(np.fft.ifft2(np.abs(spectrum) ** 2).real)
    corr /= max(float(np.max(corr)), 1e-12)
    radii, profile = _radial_profile(corr)
    target = 1.0 / math.e
    for radius, value in zip(radii[1:], profile[1:]):
        if value <= target:
            return float(radius)
    return float(radii[-1])


def _island_descriptors(array: np.ndarray) -> dict[str, float]:
    threshold = float(np.percentile(array, 75.0))
    mask = array > threshold
    coverage = float(np.mean(mask))
    try:
        from scipy import ndimage

        labels, count = ndimage.label(mask)
        if count == 0:
            mean_area = 0.0
        else:
            areas = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
            mean_area = float(np.mean(areas))
        island_count = float(count)
    except Exception:
        island_count = float("nan")
        mean_area = float("nan")
    return {
        "island_coverage": coverage,
        "island_count": island_count,
        "island_mean_area_px": mean_area,
    }


def compute_afm_descriptors(height_map: np.ndarray) -> dict[str, float]:
    """Compute finite scalar AFM morphology descriptors for a 2D height map."""

    array = replace_nonfinite(np.asarray(height_map, dtype=np.float32))
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D AFM height map, got shape {array.shape}")
    values = array.astype(np.float64)
    mean = float(np.mean(values))
    std = float(np.std(values))
    centered = values - mean
    safe_std = std if std > 1e-12 else 1.0
    p01, p05, p50, p95, p99 = np.percentile(values, [1.0, 5.0, 50.0, 95.0, 99.0])
    descriptors: dict[str, float] = {
        "height_mean": mean,
        "height_std": std,
        "rq": std,
        "ra": float(np.mean(np.abs(centered))),
        "peak_to_valley": float(np.max(values) - np.min(values)),
        "p01": float(p01),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "p99": float(p99),
        "robust_range": float(p99 - p01),
        "skewness": float(np.mean((centered / safe_std) ** 3)),
        "kurtosis": float(np.mean((centered / safe_std) ** 4)),
    }
    for builder in (_gradient_descriptors, _psd_descriptors, lambda arr: {"autocorrelation_length_px": _autocorrelation_length(arr)}, _island_descriptors):
        try:
            descriptors.update(builder(values))
        except Exception:
            for name in DESCRIPTOR_NAMES:
                descriptors.setdefault(name, float("nan"))
    return {name: _safe_float(descriptors.get(name, float("nan"))) for name in DESCRIPTOR_NAMES}


def descriptor_matrix(rows: list[dict[str, float]], names: list[str] | None = None) -> np.ndarray:
    selected = names or DESCRIPTOR_NAMES
    matrix = np.asarray([[row.get(name, float("nan")) for name in selected] for row in rows], dtype=np.float32)
    return matrix
