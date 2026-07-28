from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.special import expit

from analysis.rheed_video_afm_story.rq_disentanglement import project_unit_rq_np


def signed_distance_island_relief(
    structure: np.ndarray,
    *,
    boundary_width_px: float,
    levels: tuple[float, ...] = (0.48, 0.65, 0.78, 0.88),
    level_weights: tuple[float, ...] = (0.32, 0.31, 0.23, 0.14),
) -> np.ndarray:
    """Convert a continuous capture-zone field into nested AFM island relief.

    Each quantile level becomes a continuous signed-distance contour instead
    of a hard threshold.  Summing nested levels produces island shoulders and
    hilltops while retaining differentiable boundaries and novel stochastic
    geometry.
    """

    if len(levels) != len(level_weights):
        raise ValueError("levels and level_weights must have equal length")
    source = ndimage.gaussian_filter(
        np.asarray(structure, dtype=float), sigma=0.65, mode="wrap"
    )
    relief = np.zeros_like(source)
    width = max(float(boundary_width_px), 0.15)
    for level, weight in zip(levels, level_weights):
        mask = source >= float(np.quantile(source, level))
        inside = ndimage.distance_transform_edt(mask)
        outside = ndimage.distance_transform_edt(~mask)
        relief += float(weight) * expit((inside - outside) / width)
    local_relief = source - ndimage.gaussian_filter(
        source, sigma=3.0, mode="wrap"
    )
    relief += 0.16 * np.tanh(1.4 * local_relief)
    return project_unit_rq_np(relief).astype(np.float32)


def amplitude_conditioned_blend(
    structure: np.ndarray,
    spectral: np.ndarray,
    *,
    structure_weight: float,
) -> np.ndarray:
    return project_unit_rq_np(
        float(structure_weight) * np.asarray(structure, dtype=float)
        + (1.0 - float(structure_weight))
        * np.asarray(spectral, dtype=float)
    ).astype(np.float32)


def sdf_contour_blend(
    structure: np.ndarray,
    prior: np.ndarray,
    *,
    boundary_width_px: float,
    relief_weight: float,
    texture_weight: float,
    base_structure_weight: float,
) -> np.ndarray:
    relief = signed_distance_island_relief(
        structure, boundary_width_px=boundary_width_px
    )
    base = np.asarray(prior, dtype=float)
    fine = base - ndimage.gaussian_filter(
        base, sigma=1.4, mode="wrap"
    )
    amplitude_base = amplitude_conditioned_blend(
        structure, prior, structure_weight=base_structure_weight
    )
    remainder = 1.0 - float(relief_weight) - float(texture_weight)
    if remainder < 0:
        raise ValueError("relief and texture weights must sum to <= 1")
    return project_unit_rq_np(
        float(relief_weight) * relief
        + remainder * amplitude_base
        + float(texture_weight) * fine
    ).astype(np.float32)


def edge_preserving_terrace_blend(
    structure: np.ndarray,
    prior: np.ndarray,
    *,
    boundary_width_px: float,
    structure_weight: float,
    plateau_weight: float,
    relief_weight: float,
    spectral_weight: float,
    texture_weight: float,
) -> np.ndarray:
    """Create AFM-like terraces and grooves from novel capture zones.

    The structure is procedurally generated. The spectral input contributes a
    learned AFM population prior, not a measured patch or nearest neighbour.
    """

    source = ndimage.gaussian_filter(
        np.asarray(structure, dtype=float), sigma=0.35, mode="wrap"
    )
    center = float(np.median(source))
    robust_scale = max(
        float(np.quantile(source, 0.75) - np.quantile(source, 0.25))
        / 1.349,
        1e-6,
    )
    plateau = np.tanh((source - center) / (1.20 * robust_scale))
    plateau = ndimage.gaussian_filter(
        plateau, sigma=0.35, mode="wrap"
    )
    relief = signed_distance_island_relief(
        source, boundary_width_px=boundary_width_px
    )
    spectral = ndimage.gaussian_filter(
        np.asarray(prior, dtype=float), sigma=0.65, mode="wrap"
    )
    fine = np.asarray(prior, dtype=float) - ndimage.gaussian_filter(
        np.asarray(prior, dtype=float), sigma=1.35, mode="wrap"
    )
    weights = np.asarray(
        [
            structure_weight,
            plateau_weight,
            relief_weight,
            spectral_weight,
            texture_weight,
        ],
        dtype=float,
    )
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError(
            "terrace renderer weights must be nonnegative and sum to 1"
        )
    return project_unit_rq_np(
        weights[0] * source
        + weights[1] * plateau
        + weights[2] * relief
        + weights[3] * spectral
        + weights[4] * fine
    ).astype(np.float32)


def render_ensemble(
    structure: list[np.ndarray],
    spectral: list[np.ndarray],
    *,
    mode: str,
    structure_weight: float = 0.65,
    boundary_width_px: float = 0.55,
    relief_weight: float = 0.65,
    texture_weight: float = 0.08,
    base_structure_weight: float = 0.65,
    plateau_weight: float = 0.15,
    spectral_weight: float = 0.12,
) -> list[np.ndarray]:
    result = []
    for index in range(max(len(structure), len(spectral))):
        island = structure[index % len(structure)]
        prior = spectral[index % len(spectral)]
        if mode == "amplitude_conditioned":
            rendered = amplitude_conditioned_blend(
                island, prior, structure_weight=structure_weight
            )
        elif mode == "sdf":
            rendered = sdf_contour_blend(
                island,
                prior,
                boundary_width_px=boundary_width_px,
                relief_weight=relief_weight,
                texture_weight=texture_weight,
                base_structure_weight=base_structure_weight,
            )
        elif mode == "terrace":
            rendered = edge_preserving_terrace_blend(
                island,
                prior,
                boundary_width_px=boundary_width_px,
                structure_weight=structure_weight,
                plateau_weight=plateau_weight,
                relief_weight=relief_weight,
                spectral_weight=spectral_weight,
                texture_weight=texture_weight,
            )
        else:
            raise ValueError(f"unknown renderer mode: {mode}")
        result.append(rendered)
    return result
