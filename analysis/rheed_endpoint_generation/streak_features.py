"""Target-blind diffraction-spot elongation features for smooth-surface sensing.

The generic RHEED feature table thresholds the complete illuminated field.
That is useful for overall morphology, but broad phosphor-screen illumination
can dominate the connected components and hide whether the *local diffraction
maxima* are spots or horizontal streaks.  This module removes the broad
background, finds local maxima, and measures the intensity-weighted second
moment of each maximum.  No AFM values enter the calculation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max


FEATURE_NAMES = (
    "local_peak_count",
    "local_peak_aspect",
    "local_peak_horizontal_alignment",
    "local_peak_horizontal_aspect",
    "local_peak_horizontal_aspect_q75",
    "local_peak_strong_streak_fraction",
)
PRIMARY_STREAK_FEATURE = "local_peak_horizontal_aspect_q75"


def _peak_shape_features(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"RHEED frame must be 2-D, got {image.shape}")
    image /= 255.0 if image.max(initial=0.0) > 1.5 else 1.0
    # Difference of Gaussians rejects the broad screen illumination and
    # retains diffraction maxima at the spatial scale of spots/streaks.
    residual = ndimage.gaussian_filter(image, 1.0) - ndimage.gaussian_filter(
        image, 10.0
    )
    border = min(10, max(1, min(residual.shape) // 8))
    residual[:border] = 0.0
    residual[-border:] = 0.0
    residual[:, :border] = 0.0
    residual[:, -border:] = 0.0
    peaks = peak_local_max(
        residual,
        min_distance=5,
        threshold_abs=float(np.quantile(residual, 0.88)),
        num_peaks=24,
    )
    measurements: list[tuple[float, float, float, float]] = []
    for y, x in peaks:
        patch = residual[
            max(0, y - 5) : min(residual.shape[0], y + 6),
            max(0, x - 12) : min(residual.shape[1], x + 13),
        ]
        yy, xx = np.indices(patch.shape)
        weight = np.clip(patch - np.quantile(patch, 0.30), 0.0, None)
        total = float(weight.sum())
        if total <= 1e-12:
            continue
        center_x = float(np.sum(weight * xx) / total)
        center_y = float(np.sum(weight * yy) / total)
        covariance = np.asarray(
            [
                [
                    np.sum(weight * np.square(xx - center_x)) / total,
                    np.sum(weight * (xx - center_x) * (yy - center_y))
                    / total,
                ],
                [
                    np.sum(weight * (xx - center_x) * (yy - center_y))
                    / total,
                    np.sum(weight * np.square(yy - center_y)) / total,
                ],
            ]
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        aspect = float(
            np.sqrt(
                (max(eigenvalues[1], 0.0) + 1e-4)
                / (max(eigenvalues[0], 0.0) + 1e-4)
            )
        )
        horizontal_alignment = float(abs(eigenvectors[0, 1]))
        measurements.append(
            (
                float(residual[y, x]),
                aspect,
                horizontal_alignment,
                aspect * horizontal_alignment,
            )
        )
    if not measurements:
        return np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    values = np.asarray(measurements, dtype=np.float64)
    weights = np.clip(values[:, 0], 0.0, None) + 1e-6
    horizontal_aspect = values[:, 3]
    return np.asarray(
        [
            len(values),
            np.average(values[:, 1], weights=weights),
            np.average(values[:, 2], weights=weights),
            np.average(horizontal_aspect, weights=weights),
            np.quantile(horizontal_aspect, 0.75),
            np.mean(horizontal_aspect > 2.0),
        ],
        dtype=np.float64,
    )


def extract_streak_features(
    frames: np.ndarray,
    *,
    causal_frame_count: int = 8,
) -> dict[str, float]:
    """Return median local-peak shape descriptors over a causal clip."""

    clip = np.asarray(frames)
    if clip.ndim != 3:
        raise ValueError(f"RHEED clip must be [T,H,W], got {clip.shape}")
    if len(clip) < causal_frame_count:
        raise ValueError(
            f"need at least {causal_frame_count} frames, got {len(clip)}"
        )
    per_frame = np.stack(
        [_peak_shape_features(frame) for frame in clip[:causal_frame_count]]
    )
    aggregate = np.median(per_frame, axis=0)
    return {
        name: float(value) for name, value in zip(FEATURE_NAMES, aggregate)
    }


def extract_streak_features_from_npz(path: str | Path) -> dict[str, float]:
    with np.load(Path(path), allow_pickle=False) as payload:
        return extract_streak_features(payload["frames_uint8"])
