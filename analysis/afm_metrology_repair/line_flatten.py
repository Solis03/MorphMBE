from __future__ import annotations

import numpy as np


def sq_nm(height_nm: np.ndarray) -> float:
    """Return areal RMS height Sq in nm after removal of the areal mean."""

    values = np.asarray(height_nm, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    centered = finite - float(np.mean(finite))
    return float(np.sqrt(np.mean(np.square(centered))))


def _design_matrix(width: int, order: int) -> np.ndarray:
    if order not in (0, 1, 2, 3):
        raise ValueError(f"line-flatten order must be 0, 1, 2 or 3, got {order}")
    x = np.linspace(-1.0, 1.0, int(width), dtype=np.float64)
    return np.vander(x, N=order + 1, increasing=True)


def line_flatten(
    height_nm: np.ndarray,
    order: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit and subtract one polynomial independently from every scan line.

    The fast-scan direction is assumed to be the second array axis, matching
    NanoScope image row storage and this repository's decoded ZSensor arrays.
    Finite-only least squares is used for rows containing invalid pixels.
    """

    height = np.asarray(height_nm, dtype=np.float64)
    if height.ndim != 2:
        raise ValueError(f"expected a 2D AFM height map, got {height.shape}")
    design = _design_matrix(height.shape[1], int(order))
    background = np.full_like(height, np.nan, dtype=np.float64)
    finite = np.isfinite(height)

    if finite.all():
        coefficients = np.linalg.lstsq(design, height.T, rcond=None)[0]
        background = (design @ coefficients).T
    else:
        minimum = int(order) + 1
        for row_index, row in enumerate(height):
            valid = finite[row_index]
            if int(valid.sum()) < minimum:
                continue
            coefficients = np.linalg.lstsq(
                design[valid],
                row[valid],
                rcond=None,
            )[0]
            background[row_index] = design @ coefficients

    corrected = height - background
    return corrected.astype(np.float32), background.astype(np.float32)
