"""Continuous image representations for peak-saddle adhesion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class RheedChannels:
    linear: np.ndarray
    background: np.ndarray
    log_corrected: np.ndarray
    horizontal_ridge: np.ndarray


def estimate_smooth_background(image: np.ndarray, *, sigma: float = 18.0) -> np.ndarray:
    """Estimate only broad screen/background structure."""
    values = np.asarray(image, dtype=float)
    smooth = ndimage.gaussian_filter(values, sigma=sigma)
    floor = float(np.nanpercentile(values, 2.0))
    return np.maximum(smooth, floor)


def horizontal_ridge_response(image: np.ndarray, *, sigma_y: float = 1.6, sigma_x: float = 5.0) -> np.ndarray:
    """Return a scale-aware horizontal ridge-support channel."""
    values = np.asarray(image, dtype=float)
    smoothed = ndimage.gaussian_filter(values, sigma=(sigma_y, sigma_x))
    vertical_second = -ndimage.gaussian_filter(smoothed, sigma=(sigma_y, sigma_x), order=(2, 0))
    return np.maximum(vertical_second, 0.0)


def make_channels(image: np.ndarray, *, background_sigma: float = 18.0, epsilon: float = 1e-6) -> RheedChannels:
    """Build the required continuous channels without per-image equalization."""
    linear = np.asarray(image, dtype=np.float32)
    background = estimate_smooth_background(linear, sigma=background_sigma).astype(np.float32)
    log_corrected = (np.log(np.maximum(linear, 0.0) + epsilon) - np.log(np.maximum(background, 0.0) + epsilon)).astype(np.float32)
    ridge = horizontal_ridge_response(log_corrected).astype(np.float32)
    return RheedChannels(linear=linear, background=background, log_corrected=log_corrected, horizontal_ridge=ridge)

