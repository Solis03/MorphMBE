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


def smooth_microtexture_blend(
    structure: np.ndarray,
    prior: np.ndarray,
    *,
    low_frequency_weight: float = 0.10,
) -> np.ndarray:
    """Render dense low-amplitude morphology without coarse island collapse.

    Extremely smooth GaSb surfaces still contain spatial texture, but their
    topography is dominated by densely packed small features rather than
    isolated high islands separated by connected deep valleys.  Band-pass
    components from the learned spectral prior and the generated capture-zone
    field preserve that texture while suppressing the coarse terrace scale.
    """

    island = np.asarray(structure, dtype=float)
    spectral = np.asarray(prior, dtype=float)
    low = ndimage.gaussian_filter(island, sigma=3.5, mode="wrap")
    spectral_mid = spectral - ndimage.gaussian_filter(
        spectral, sigma=2.2, mode="wrap"
    )
    spectral_fine = spectral - ndimage.gaussian_filter(
        spectral, sigma=0.8, mode="wrap"
    )
    island_micro = island - ndimage.gaussian_filter(
        island, sigma=1.6, mode="wrap"
    )
    texture = (
        float(low_frequency_weight) * low
        + 0.45 * spectral_mid
        + 0.25 * spectral_fine
        + (0.30 - float(low_frequency_weight)) * island_micro
    )
    center = float(np.median(texture))
    robust_scale = max(
        float(np.quantile(texture, 0.75) - np.quantile(texture, 0.25))
        / 1.349,
        1e-6,
    )
    # A soft robust clip removes the large connected extrema visible in the
    # previous smooth-regime renderings without flattening the small texture.
    texture = np.tanh((texture - center) / (2.5 * robust_scale))
    return project_unit_rq_np(texture).astype(np.float32)


def smooth_microisland_blend(
    structure: np.ndarray,
    prior: np.ndarray,
    *,
    spectral_weight: float = 0.72,
    spectral_sigma_px: float = 1.05,
    structure_sigma_px: float = 1.45,
    microisland_weight: float = 0.18,
    microisland_spacing_px: int = 5,
    microisland_sigma_px: float = 1.55,
) -> np.ndarray:
    """Render dense, rounded low-Sq morphology from generated fields.

    The first smooth-regime renderer amplified band-pass residuals and could
    therefore look pixel-like even though its height amplitude was correct.
    Here the learned population prior remains the dominant field, but both
    conditional fields are low-pass filtered at approximately the scale of
    the smallest resolved AFM islands.  Sparse local maxima from that same
    generated prior add rounded hilltops.  No measured AFM patch, nearest
    neighbour, or held-out AFM is used at inference.
    """

    if not 0.0 <= float(spectral_weight) <= 1.0:
        raise ValueError("spectral weight must lie in [0, 1]")
    if not 0.0 <= float(microisland_weight) <= 1.0:
        raise ValueError("microisland weight must lie in [0, 1]")
    spacing = max(int(microisland_spacing_px), 3)
    if spacing % 2 == 0:
        spacing += 1

    spectral = np.asarray(prior, dtype=float)
    island = np.asarray(structure, dtype=float)
    spectral_base = ndimage.gaussian_filter(
        spectral, sigma=float(spectral_sigma_px), mode="wrap"
    )
    structure_base = ndimage.gaussian_filter(
        island, sigma=float(structure_sigma_px), mode="wrap"
    )
    base = (
        float(spectral_weight) * spectral_base
        + (1.0 - float(spectral_weight)) * structure_base
    )

    peak_source = ndimage.gaussian_filter(
        spectral, sigma=0.80, mode="wrap"
    )
    maxima = peak_source == ndimage.maximum_filter(
        peak_source, size=spacing, mode="wrap"
    )
    maxima &= peak_source >= float(np.quantile(peak_source, 0.62))
    impulses = np.zeros_like(peak_source)
    baseline = float(np.quantile(peak_source, 0.35))
    impulses[maxima] = np.maximum(peak_source[maxima] - baseline, 0.0)
    microislands = ndimage.gaussian_filter(
        impulses, sigma=float(microisland_sigma_px), mode="wrap"
    )
    if float(np.std(microislands)) > 1e-8:
        microislands = project_unit_rq_np(microislands)

    texture = (
        (1.0 - float(microisland_weight)) * base
        + float(microisland_weight) * microislands
    )
    center = float(np.median(texture))
    robust_scale = max(
        float(np.quantile(texture, 0.75) - np.quantile(texture, 0.25))
        / 1.349,
        1e-6,
    )
    texture = np.tanh((texture - center) / (3.0 * robust_scale))
    return project_unit_rq_np(texture).astype(np.float32)


def regime_adaptive_terrace_blend(
    structure: np.ndarray,
    prior: np.ndarray,
    *,
    conditioning_sq_nm: float,
    smooth_full_below_nm: float,
    terrace_full_above_nm: float,
    boundary_width_px: float,
    structure_weight: float,
    plateau_weight: float,
    relief_weight: float,
    spectral_weight: float,
    texture_weight: float,
    smooth_low_frequency_weight: float = 0.10,
) -> np.ndarray:
    """Interpolate between dense microtexture and the M12a terrace renderer."""

    if terrace_full_above_nm <= smooth_full_below_nm:
        raise ValueError("terrace threshold must exceed smooth threshold")
    smooth = smooth_microtexture_blend(
        structure,
        prior,
        low_frequency_weight=smooth_low_frequency_weight,
    )
    terrace = edge_preserving_terrace_blend(
        structure,
        prior,
        boundary_width_px=boundary_width_px,
        structure_weight=structure_weight,
        plateau_weight=plateau_weight,
        relief_weight=relief_weight,
        spectral_weight=spectral_weight,
        texture_weight=texture_weight,
    )
    terrace_fraction = float(
        np.clip(
            (
                float(conditioning_sq_nm)
                - float(smooth_full_below_nm)
            )
            / (
                float(terrace_full_above_nm)
                - float(smooth_full_below_nm)
            ),
            0.0,
            1.0,
        )
    )
    return project_unit_rq_np(
        (1.0 - terrace_fraction) * smooth + terrace_fraction * terrace
    ).astype(np.float32)


def regime_adaptive_microisland_terrace_blend(
    structure: np.ndarray,
    prior: np.ndarray,
    *,
    conditioning_sq_nm: float,
    smooth_full_below_nm: float,
    terrace_full_above_nm: float,
    boundary_width_px: float,
    structure_weight: float,
    plateau_weight: float,
    relief_weight: float,
    spectral_weight: float,
    texture_weight: float,
    smooth_spectral_weight: float = 0.72,
    smooth_spectral_sigma_px: float = 1.05,
    smooth_structure_sigma_px: float = 1.45,
    smooth_microisland_weight: float = 0.18,
    smooth_microisland_spacing_px: int = 5,
    smooth_microisland_sigma_px: float = 1.55,
) -> np.ndarray:
    """Interpolate a dense micro-island regime into the frozen terrace prior."""

    if terrace_full_above_nm <= smooth_full_below_nm:
        raise ValueError("terrace threshold must exceed smooth threshold")
    smooth = smooth_microisland_blend(
        structure,
        prior,
        spectral_weight=smooth_spectral_weight,
        spectral_sigma_px=smooth_spectral_sigma_px,
        structure_sigma_px=smooth_structure_sigma_px,
        microisland_weight=smooth_microisland_weight,
        microisland_spacing_px=smooth_microisland_spacing_px,
        microisland_sigma_px=smooth_microisland_sigma_px,
    )
    terrace = edge_preserving_terrace_blend(
        structure,
        prior,
        boundary_width_px=boundary_width_px,
        structure_weight=structure_weight,
        plateau_weight=plateau_weight,
        relief_weight=relief_weight,
        spectral_weight=spectral_weight,
        texture_weight=texture_weight,
    )
    terrace_fraction = float(
        np.clip(
            (
                float(conditioning_sq_nm)
                - float(smooth_full_below_nm)
            )
            / (
                float(terrace_full_above_nm)
                - float(smooth_full_below_nm)
            ),
            0.0,
            1.0,
        )
    )
    return project_unit_rq_np(
        (1.0 - terrace_fraction) * smooth + terrace_fraction * terrace
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
    conditioning_sq_nm: float | None = None,
    smooth_full_below_nm: float = 0.80,
    terrace_full_above_nm: float = 1.60,
    smooth_low_frequency_weight: float = 0.10,
    smooth_spectral_weight: float = 0.72,
    smooth_spectral_sigma_px: float = 1.05,
    smooth_structure_sigma_px: float = 1.45,
    smooth_microisland_weight: float = 0.18,
    smooth_microisland_spacing_px: int = 5,
    smooth_microisland_sigma_px: float = 1.55,
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
        elif mode == "regime_adaptive_terrace":
            if conditioning_sq_nm is None:
                raise ValueError(
                    "regime-adaptive rendering requires conditioning Sq"
                )
            rendered = regime_adaptive_terrace_blend(
                island,
                prior,
                conditioning_sq_nm=conditioning_sq_nm,
                smooth_full_below_nm=smooth_full_below_nm,
                terrace_full_above_nm=terrace_full_above_nm,
                boundary_width_px=boundary_width_px,
                structure_weight=structure_weight,
                plateau_weight=plateau_weight,
                relief_weight=relief_weight,
                spectral_weight=spectral_weight,
                texture_weight=texture_weight,
                smooth_low_frequency_weight=(
                    smooth_low_frequency_weight
                ),
            )
        elif mode == "regime_adaptive_microisland_terrace":
            if conditioning_sq_nm is None:
                raise ValueError(
                    "regime-adaptive rendering requires conditioning Sq"
                )
            rendered = regime_adaptive_microisland_terrace_blend(
                island,
                prior,
                conditioning_sq_nm=conditioning_sq_nm,
                smooth_full_below_nm=smooth_full_below_nm,
                terrace_full_above_nm=terrace_full_above_nm,
                boundary_width_px=boundary_width_px,
                structure_weight=structure_weight,
                plateau_weight=plateau_weight,
                relief_weight=relief_weight,
                spectral_weight=spectral_weight,
                texture_weight=texture_weight,
                smooth_spectral_weight=smooth_spectral_weight,
                smooth_spectral_sigma_px=smooth_spectral_sigma_px,
                smooth_structure_sigma_px=smooth_structure_sigma_px,
                smooth_microisland_weight=smooth_microisland_weight,
                smooth_microisland_spacing_px=(
                    smooth_microisland_spacing_px
                ),
                smooth_microisland_sigma_px=smooth_microisland_sigma_px,
            )
        else:
            raise ValueError(f"unknown renderer mode: {mode}")
        result.append(rendered)
    return result
