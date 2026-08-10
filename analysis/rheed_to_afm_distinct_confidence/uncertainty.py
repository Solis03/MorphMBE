from __future__ import annotations

import numpy as np


def jackknife_plus_interval(
    query_predictions_z: np.ndarray,
    calibration_predictions_z: np.ndarray,
    calibration_truth_z: np.ndarray,
    *,
    alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Component-wise finite-sample Jackknife+ interval.

    Each row must come from the model that excluded the corresponding
    calibration growth group.
    """

    query = np.asarray(query_predictions_z, dtype=float)
    predicted = np.asarray(calibration_predictions_z, dtype=float)
    truth = np.asarray(calibration_truth_z, dtype=float)
    if query.shape != predicted.shape or predicted.shape != truth.shape:
        raise ValueError("query, calibration prediction, and truth shapes differ")
    residual = np.abs(truth - predicted)
    lower_candidates = query - residual
    upper_candidates = query + residual
    lower = np.quantile(lower_candidates, alpha / 2.0, axis=0, method="lower")
    upper = np.quantile(upper_candidates, 1.0 - alpha / 2.0, axis=0, method="higher")
    return lower.astype(np.float32), upper.astype(np.float32)


def relative_confidence_index(
    widths: np.ndarray,
    *,
    reference_widths: np.ndarray,
) -> np.ndarray:
    """Map interval width to a conservative 0--100 index, not probability.

    The exponential term gives an absolute penalty for intervals that span
    several training standard deviations. A small rank adjustment preserves
    ordering information without stretching nearly identical wide intervals
    across the misleading full 0--100 range.
    """

    values = np.asarray(widths, dtype=float)
    reference = np.asarray(reference_widths, dtype=float)
    low, high = np.quantile(reference, [0.05, 0.95])
    if high - low < 1e-8:
        relative = np.full_like(values, 0.5, dtype=float)
    else:
        relative = np.clip((high - values) / (high - low), 0.0, 1.0)
    absolute = 100.0 * np.exp(-values / 3.0)
    confidence = absolute * (0.75 + 0.25 * relative)
    return np.clip(confidence, 0.0, 100.0).astype(np.float32)


def interval_width_z(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))
