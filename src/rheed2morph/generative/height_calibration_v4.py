"""Height-scale calibration helpers for AFM prior v4."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from rheed2morph.generative.afm_prior_v2_utils import compute_afm_descriptors_v2
from rheed2morph.generative.condition_control_v3_utils import finite_float


AMPLITUDE_DESCRIPTOR_WEIGHTS = {
    "rq": 1.0,
    "height_std": 1.0,
    "ra": 0.8,
    "robust_range": 0.7,
    "peak_to_valley": 0.3,
}


@dataclass(frozen=True)
class HeightCalibrationResult:
    scale_nm_per_unit: float
    offset_nm: float
    clamped: bool
    unclamped_scale_nm_per_unit: float
    mode: str
    descriptors_before: dict[str, float]
    descriptors_after: dict[str, float]
    target_descriptors: dict[str, float]
    calibration_error: float


def compute_height_descriptors(image: np.ndarray) -> dict[str, float]:
    """Compute finite AFM descriptors used by height calibration."""

    return compute_afm_descriptors_v2(np.asarray(image, dtype=np.float32))


def _target_descriptors(target: Mapping[str, Any], names: Sequence[str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for name in names:
        value = finite_float(target.get(name, float("nan")))
        if math.isfinite(value):
            output[name] = value
    return output


def _mode_weights(mode: str) -> dict[str, float]:
    if mode == "none":
        return {}
    if mode == "rq_only":
        return {"rq": 1.0}
    if mode == "ra_only":
        return {"ra": 1.0}
    if mode == "range_only":
        return {"robust_range": 1.0}
    if mode == "train_median_scale":
        return {}
    if mode == "weighted_rq_ra_range":
        return {"rq": 1.0, "ra": 0.8, "robust_range": 0.7}
    raise ValueError(f"Unsupported calibration mode: {mode}")


def _scale_bounds_tuple(scale_bounds: Mapping[str, Any] | Sequence[float] | None) -> tuple[float, float] | None:
    if scale_bounds is None:
        return None
    if isinstance(scale_bounds, Mapping):
        low = finite_float(scale_bounds.get("scale_low", scale_bounds.get("p01", float("nan"))))
        high = finite_float(scale_bounds.get("scale_high", scale_bounds.get("p99", float("nan"))))
    else:
        values = list(scale_bounds)
        if len(values) < 2:
            return None
        low = finite_float(values[0])
        high = finite_float(values[1])
    if not math.isfinite(low) or not math.isfinite(high) or high <= 0:
        return None
    low = max(low, 1e-8)
    high = max(high, low)
    return low, high


def fit_affine_height_scale(
    normalized_image: np.ndarray,
    target_descriptors: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
    scale_bounds: Mapping[str, Any] | Sequence[float] | None = None,
    allow_extrapolation: bool = False,
    default_scale: float | None = None,
) -> dict[str, float | bool]:
    """Fit positive scale for amplitude descriptors with a closed-form least-squares solution."""

    descriptors = compute_height_descriptors(normalized_image)
    active_weights = dict(weights or AMPLITUDE_DESCRIPTOR_WEIGHTS)
    numerator = 0.0
    denominator = 0.0
    used = 0
    for name, weight in active_weights.items():
        generated = finite_float(descriptors.get(name, float("nan")))
        target = finite_float(target_descriptors.get(name, float("nan")))
        if not math.isfinite(generated) or not math.isfinite(target) or generated <= 1e-8 or target < 0:
            continue
        w = max(float(weight), 0.0)
        numerator += w * generated * target
        denominator += w * generated * generated
        used += 1
    if denominator > 1e-12:
        scale = numerator / denominator
    elif default_scale is not None and math.isfinite(float(default_scale)):
        scale = float(default_scale)
    else:
        scale = 1.0
    if not math.isfinite(scale) or scale <= 0:
        scale = float(default_scale) if default_scale is not None and math.isfinite(float(default_scale)) else 1.0
    unclamped = float(scale)
    clamped = False
    bounds = _scale_bounds_tuple(scale_bounds)
    if bounds is not None and not allow_extrapolation:
        low, high = bounds
        scale = float(np.clip(scale, low, high))
        clamped = not math.isclose(scale, unclamped, rel_tol=1e-8, abs_tol=1e-8)
    return {
        "scale_nm_per_unit": float(scale),
        "unclamped_scale_nm_per_unit": unclamped,
        "clamped": bool(clamped),
        "used_descriptor_count": int(used),
    }


def apply_height_calibration(normalized_image: np.ndarray, scale: float, offset: float = 0.0) -> np.ndarray:
    image = np.asarray(normalized_image, dtype=np.float32)
    return (float(offset) + float(scale) * image).astype(np.float32)


def descriptor_error(
    generated: Mapping[str, Any],
    target: Mapping[str, Any],
    names: Sequence[str],
    scales: Mapping[str, float] | None = None,
) -> float:
    values: list[float] = []
    for name in names:
        gen = finite_float(generated.get(name, float("nan")))
        req = finite_float(target.get(name, float("nan")))
        if not math.isfinite(gen) or not math.isfinite(req):
            continue
        denom = abs(float(scales.get(name, 1.0))) if scales else max(abs(req), 1.0)
        values.append(abs(gen - req) / max(denom, 1e-6))
    return float(np.mean(values)) if values else float("inf")


def calibration_mode_weights(mode: str) -> dict[str, float]:
    return _mode_weights(mode)


def calibrate_generated_afm(
    normalized_image: np.ndarray,
    target_condition: Mapping[str, Any],
    schema: Mapping[str, Any] | None = None,
    calibration_mode: str = "weighted_rq_ra_range",
    scale_bounds: Mapping[str, Any] | Sequence[float] | None = None,
    allow_extrapolation: bool = False,
) -> tuple[np.ndarray, HeightCalibrationResult]:
    before = compute_height_descriptors(normalized_image)
    if calibration_mode == "none":
        scale = 1.0
        offset = 0.0
        clamped = False
        unclamped = 1.0
    elif calibration_mode == "train_median_scale":
        bounds = _scale_bounds_tuple(scale_bounds)
        if isinstance(scale_bounds, Mapping):
            scale = finite_float(scale_bounds.get("scale_median", float("nan")))
        else:
            scale = float("nan")
        if not math.isfinite(scale):
            scale = float(np.mean(bounds)) if bounds is not None else 1.0
        offset = finite_float(target_condition.get("p50", target_condition.get("height_mean", 0.0)), 0.0)
        clamped = False
        unclamped = scale
    else:
        weights = _mode_weights(calibration_mode)
        target = _target_descriptors(target_condition, list(weights))
        fit = fit_affine_height_scale(
            normalized_image,
            target,
            weights=weights,
            scale_bounds=scale_bounds,
            allow_extrapolation=allow_extrapolation,
            default_scale=finite_float(scale_bounds.get("scale_median", 1.0), 1.0) if isinstance(scale_bounds, Mapping) else 1.0,
        )
        scale = float(fit["scale_nm_per_unit"])
        unclamped = float(fit["unclamped_scale_nm_per_unit"])
        clamped = bool(fit["clamped"])
        offset = finite_float(target_condition.get("p50", target_condition.get("height_mean", 0.0)), 0.0)
    calibrated = apply_height_calibration(normalized_image, scale, offset)
    after = compute_height_descriptors(calibrated)
    target_names = list((schema or {}).get("descriptor_columns", [])) if schema is not None else list(AMPLITUDE_DESCRIPTOR_WEIGHTS)
    if not target_names:
        target_names = list(AMPLITUDE_DESCRIPTOR_WEIGHTS)
    targets = _target_descriptors(target_condition, target_names)
    error_names = [name for name in ("rq", "ra", "robust_range") if name in targets]
    error = descriptor_error(after, targets, error_names or list(targets))
    result = HeightCalibrationResult(
        scale_nm_per_unit=float(scale),
        offset_nm=float(offset),
        clamped=bool(clamped),
        unclamped_scale_nm_per_unit=float(unclamped),
        mode=calibration_mode,
        descriptors_before=before,
        descriptors_after=after,
        target_descriptors=targets,
        calibration_error=error,
    )
    return calibrated, result


def evaluate_calibrated_descriptors(
    calibrated_image_nm: np.ndarray,
    target_descriptors: Mapping[str, Any],
    descriptor_names: Sequence[str] = ("rq", "ra", "robust_range"),
) -> dict[str, float]:
    generated = compute_height_descriptors(calibrated_image_nm)
    output: dict[str, float] = {"descriptor_error": descriptor_error(generated, target_descriptors, descriptor_names)}
    for name in descriptor_names:
        output[f"generated_{name}"] = finite_float(generated.get(name, float("nan")))
        output[f"target_{name}"] = finite_float(target_descriptors.get(name, float("nan")))
    return output


def summarize_scale_values(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray([float(value) for value in values if math.isfinite(float(value)) and float(value) > 0], dtype=np.float64)
    if arr.size == 0:
        return {"scale_low": 1.0, "scale_high": 1.0, "scale_median": 1.0, "scale_mean": 1.0}
    return {
        "scale_low": float(np.percentile(arr, 1.0)),
        "scale_high": float(np.percentile(arr, 99.0)),
        "scale_median": float(np.median(arr)),
        "scale_mean": float(np.mean(arr)),
        "scale_std": float(np.std(arr)),
        "scale_min": float(np.min(arr)),
        "scale_max": float(np.max(arr)),
    }
