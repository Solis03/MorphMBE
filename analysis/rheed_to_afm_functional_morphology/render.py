from __future__ import annotations

import numpy as np
from scipy import ndimage, stats
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


def _top_k_local_maxima(
    field: np.ndarray,
    *,
    count: int,
    spacing_px: int,
    minimum_quantile: float,
) -> np.ndarray:
    """Return a deterministic sparse peak impulse map.

    Candidate locations are local maxima of the generated spectral field.
    Selecting the strongest ``count`` candidates keeps the renderer
    stochastic through its generated inputs while making the number of
    visually dominant hilltops explicit and auditable.
    """

    spacing = max(int(spacing_px), 3)
    if spacing % 2 == 0:
        spacing += 1
    source = np.asarray(field, dtype=float)
    maxima = source == ndimage.maximum_filter(
        source, size=spacing, mode="wrap"
    )
    maxima &= source >= float(np.quantile(source, minimum_quantile))
    coordinates = np.argwhere(maxima)
    impulses = np.zeros_like(source)
    if count <= 0 or not len(coordinates):
        return impulses
    strengths = source[coordinates[:, 0], coordinates[:, 1]]
    order = np.argsort(strengths, kind="stable")[::-1][: int(count)]
    selected = coordinates[order]
    baseline = float(np.median(source))
    impulses[selected[:, 0], selected[:, 1]] = np.maximum(
        source[selected[:, 0], selected[:, 1]] - baseline,
        0.0,
    )
    return impulses


def topology_conditioned_sparse_microisland_blend(
    structure: np.ndarray,
    prior: np.ndarray,
    *,
    island_target: dict[str, float] | None,
    spectral_weight: float = 0.74,
    spectral_sigma_px: float = 0.90,
    structure_sigma_px: float = 1.35,
    fine_texture_weight: float = 0.10,
    sparse_peak_weight: float = 0.075,
    shoulder_peak_weight: float = 0.0,
    peak_count_scale: float = 0.30,
    peak_count_min: int = 4,
    peak_count_max: int = 24,
    peak_spacing_px: int = 9,
    peak_sigma_px: float = 1.15,
    peak_minimum_quantile: float = 0.70,
    shoulder_count_scale: float = 0.85,
    shoulder_count_min: int = 20,
    shoulder_count_max: int = 48,
    shoulder_spacing_px: int = 7,
    shoulder_sigma_px: float = 2.4,
    shoulder_minimum_quantile: float = 0.58,
    winsor_quantile: float = 0.997,
) -> np.ndarray:
    """Render fine smooth-surface texture with a few persistent hilltops.

    M16b used a fixed dense local-maximum field and a final tanh operation.
    That combination made many moderate peaks similarly bright.  Here the
    q82 connected-component count predicted from RHEED in the enclosing
    leave-one-growth fold controls the number of persistent peaks.  Fine
    spectral residuals remain present, so reducing bright peaks does not turn
    the AFM into a featureless plane.  No sample identifier or held AFM enters
    this function.
    """

    for name, value in {
        "spectral_weight": spectral_weight,
        "fine_texture_weight": fine_texture_weight,
        "sparse_peak_weight": sparse_peak_weight,
        "shoulder_peak_weight": shoulder_peak_weight,
    }.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if (
        float(fine_texture_weight)
        + float(sparse_peak_weight)
        + float(shoulder_peak_weight)
        >= 1.0
    ):
        raise ValueError(
            "fine-texture, sparse-peak and shoulder weights must sum to < 1"
        )
    if not 0.5 < float(winsor_quantile) < 1.0:
        raise ValueError("winsor quantile must lie in (0.5, 1)")

    spectral = np.asarray(prior, dtype=float)
    island = np.asarray(structure, dtype=float)
    spectral_base = project_unit_rq_np(
        ndimage.gaussian_filter(
            spectral, sigma=float(spectral_sigma_px), mode="wrap"
        )
    )
    structure_base = project_unit_rq_np(
        ndimage.gaussian_filter(
            island, sigma=float(structure_sigma_px), mode="wrap"
        )
    )
    base = project_unit_rq_np(
        float(spectral_weight) * spectral_base
        + (1.0 - float(spectral_weight)) * structure_base
    )
    fine = spectral - ndimage.gaussian_filter(
        spectral, sigma=0.72, mode="wrap"
    )
    if float(np.std(fine)) > 1e-8:
        fine = project_unit_rq_np(fine)

    predicted_count = 32.0
    if island_target is not None and "log_component_count_q82" in island_target:
        predicted_count = float(
            np.expm1(float(island_target["log_component_count_q82"]))
        )
    peak_count = int(
        np.clip(
            np.rint(predicted_count * float(peak_count_scale)),
            int(peak_count_min),
            int(peak_count_max),
        )
    )
    peak_source = ndimage.gaussian_filter(spectral, sigma=0.75, mode="wrap")
    impulses = _top_k_local_maxima(
        peak_source,
        count=peak_count,
        spacing_px=int(peak_spacing_px),
        minimum_quantile=float(peak_minimum_quantile),
    )
    sparse_peaks = ndimage.gaussian_filter(
        impulses, sigma=float(peak_sigma_px), mode="wrap"
    )
    if float(np.std(sparse_peaks)) > 1e-8:
        sparse_peaks = project_unit_rq_np(sparse_peaks)

    shoulder_count = int(
        np.clip(
            np.rint(predicted_count * float(shoulder_count_scale)),
            int(shoulder_count_min),
            int(shoulder_count_max),
        )
    )
    shoulder_impulses = _top_k_local_maxima(
        peak_source,
        count=shoulder_count,
        spacing_px=int(shoulder_spacing_px),
        minimum_quantile=float(shoulder_minimum_quantile),
    )
    shoulders = ndimage.gaussian_filter(
        shoulder_impulses,
        sigma=float(shoulder_sigma_px),
        mode="wrap",
    )
    if float(np.std(shoulders)) > 1e-8:
        shoulders = project_unit_rq_np(shoulders)

    base_weight = (
        1.0
        - float(fine_texture_weight)
        - float(sparse_peak_weight)
        - float(shoulder_peak_weight)
    )
    texture = (
        base_weight * base
        + float(fine_texture_weight) * fine
        + float(sparse_peak_weight) * sparse_peaks
        + float(shoulder_peak_weight) * shoulders
    )
    # Preserve naturally sparse tails.  M16b's tanh compressed them into a
    # broad, nearly uniform population of medium-height yellow plateaus.
    lower = float(np.quantile(texture, 1.0 - float(winsor_quantile)))
    upper = float(np.quantile(texture, float(winsor_quantile)))
    texture = np.clip(texture, lower, upper)
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


def regime_adaptive_topology_sparse_terrace_blend(
    structure: np.ndarray,
    prior: np.ndarray,
    *,
    conditioning_sq_nm: float,
    island_target: dict[str, float] | None,
    smooth_full_below_nm: float,
    terrace_full_above_nm: float,
    boundary_width_px: float,
    structure_weight: float,
    plateau_weight: float,
    relief_weight: float,
    spectral_weight: float,
    texture_weight: float,
    smooth_spectral_weight: float = 0.74,
    smooth_spectral_sigma_px: float = 0.90,
    smooth_structure_sigma_px: float = 1.35,
    smooth_fine_texture_weight: float = 0.10,
    smooth_sparse_peak_weight: float = 0.075,
    smooth_shoulder_peak_weight: float = 0.0,
    smooth_peak_count_scale: float = 0.30,
    smooth_peak_count_min: int = 4,
    smooth_peak_count_max: int = 24,
    smooth_peak_spacing_px: int = 9,
    smooth_peak_sigma_px: float = 1.15,
    smooth_peak_minimum_quantile: float = 0.70,
    smooth_shoulder_count_scale: float = 0.85,
    smooth_shoulder_count_min: int = 20,
    smooth_shoulder_count_max: int = 48,
    smooth_shoulder_spacing_px: int = 7,
    smooth_shoulder_sigma_px: float = 2.4,
    smooth_shoulder_minimum_quantile: float = 0.58,
    smooth_winsor_quantile: float = 0.997,
) -> np.ndarray:
    """Interpolate topology-conditioned smooth texture into M12a terraces."""

    if terrace_full_above_nm <= smooth_full_below_nm:
        raise ValueError("terrace threshold must exceed smooth threshold")
    smooth = topology_conditioned_sparse_microisland_blend(
        structure,
        prior,
        island_target=island_target,
        spectral_weight=smooth_spectral_weight,
        spectral_sigma_px=smooth_spectral_sigma_px,
        structure_sigma_px=smooth_structure_sigma_px,
        fine_texture_weight=smooth_fine_texture_weight,
        sparse_peak_weight=smooth_sparse_peak_weight,
        shoulder_peak_weight=smooth_shoulder_peak_weight,
        peak_count_scale=smooth_peak_count_scale,
        peak_count_min=smooth_peak_count_min,
        peak_count_max=smooth_peak_count_max,
        peak_spacing_px=smooth_peak_spacing_px,
        peak_sigma_px=smooth_peak_sigma_px,
        peak_minimum_quantile=smooth_peak_minimum_quantile,
        shoulder_count_scale=smooth_shoulder_count_scale,
        shoulder_count_min=smooth_shoulder_count_min,
        shoulder_count_max=smooth_shoulder_count_max,
        shoulder_spacing_px=smooth_shoulder_spacing_px,
        shoulder_sigma_px=smooth_shoulder_sigma_px,
        shoulder_minimum_quantile=smooth_shoulder_minimum_quantile,
        winsor_quantile=smooth_winsor_quantile,
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
            (float(conditioning_sq_nm) - float(smooth_full_below_nm))
            / (float(terrace_full_above_nm) - float(smooth_full_below_nm)),
            0.0,
            1.0,
        )
    )
    return project_unit_rq_np(
        (1.0 - terrace_fraction) * smooth + terrace_fraction * terrace
    ).astype(np.float32)


def regime_adaptive_separated_island_blend(
    structure: np.ndarray,
    prior: np.ndarray,
    baseline_structure: np.ndarray,
    *,
    conditioning_sq_nm: float,
    island_target: dict[str, float] | None,
    rough_start_nm: float = 2.20,
    rough_full_nm: float = 3.60,
    rough_structure_weight: float = 0.90,
    rough_texture_weight: float = 0.10,
    rough_texture_sigma_px: float = 2.4,
    rough_tip_sigma_px: float = 0.35,
    rough_isolation_score: float = 0.50,
    rough_isolation_strength: float = 1.0,
    smooth_full_below_nm: float = 0.80,
    terrace_full_above_nm: float = 1.60,
    boundary_width_px: float = 0.75,
    structure_weight: float = 0.50,
    plateau_weight: float = 0.14,
    relief_weight: float = 0.18,
    spectral_weight: float = 0.13,
    texture_weight: float = 0.05,
    smooth_spectral_weight: float = 0.74,
    smooth_spectral_sigma_px: float = 0.90,
    smooth_structure_sigma_px: float = 1.35,
    smooth_fine_texture_weight: float = 0.10,
    smooth_sparse_peak_weight: float = 0.075,
    smooth_shoulder_peak_weight: float = 0.0,
    smooth_peak_count_scale: float = 0.30,
    smooth_peak_count_min: int = 4,
    smooth_peak_count_max: int = 24,
    smooth_peak_spacing_px: int = 9,
    smooth_peak_sigma_px: float = 1.15,
    smooth_peak_minimum_quantile: float = 0.70,
    smooth_shoulder_count_scale: float = 0.85,
    smooth_shoulder_count_min: int = 20,
    smooth_shoulder_count_max: int = 48,
    smooth_shoulder_spacing_px: int = 7,
    smooth_shoulder_sigma_px: float = 2.4,
    smooth_shoulder_minimum_quantile: float = 0.58,
    smooth_winsor_quantile: float = 0.997,
) -> np.ndarray:
    """Keep M17 below the rough regime, then reveal finite domed islands.

    Only high-pass AFM-prior texture is admitted to the rough branch.  Broad
    prior modes would recreate the low, flat basins this branch is designed to
    remove.  The M17 baseline receives its original Laguerre structure, so the
    already-strong smooth-surface behavior is unchanged below ``rough_start``.
    """

    if rough_full_nm <= rough_start_nm:
        raise ValueError("rough full threshold must exceed rough start")
    if not 0.0 <= rough_structure_weight <= 1.0:
        raise ValueError("rough structure weight must be in [0, 1]")
    if rough_texture_weight < 0.0:
        raise ValueError("rough texture weight must be nonnegative")

    baseline = regime_adaptive_topology_sparse_terrace_blend(
        baseline_structure,
        prior,
        conditioning_sq_nm=conditioning_sq_nm,
        island_target=island_target,
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
        smooth_fine_texture_weight=smooth_fine_texture_weight,
        smooth_sparse_peak_weight=smooth_sparse_peak_weight,
        smooth_shoulder_peak_weight=smooth_shoulder_peak_weight,
        smooth_peak_count_scale=smooth_peak_count_scale,
        smooth_peak_count_min=smooth_peak_count_min,
        smooth_peak_count_max=smooth_peak_count_max,
        smooth_peak_spacing_px=smooth_peak_spacing_px,
        smooth_peak_sigma_px=smooth_peak_sigma_px,
        smooth_peak_minimum_quantile=smooth_peak_minimum_quantile,
        smooth_shoulder_count_scale=smooth_shoulder_count_scale,
        smooth_shoulder_count_min=smooth_shoulder_count_min,
        smooth_shoulder_count_max=smooth_shoulder_count_max,
        smooth_shoulder_spacing_px=smooth_shoulder_spacing_px,
        smooth_shoulder_sigma_px=smooth_shoulder_sigma_px,
        smooth_shoulder_minimum_quantile=smooth_shoulder_minimum_quantile,
        smooth_winsor_quantile=smooth_winsor_quantile,
    )
    if float(conditioning_sq_nm) <= float(rough_start_nm):
        return baseline
    rounded = ndimage.gaussian_filter(
        project_unit_rq_np(structure),
        sigma=max(float(rough_tip_sigma_px), 0.0),
        mode="wrap",
    )
    isolation = float(np.clip(rough_isolation_score, 0.0, 1.0))
    isolation_strength = max(float(rough_isolation_strength), 0.0)
    median = float(np.median(rounded))
    centered = rounded - median
    # Preserve the requested ordering in the rendered morphology itself:
    # isolated RHEED spots deepen the connected substrate, whereas bridged
    # or streak-like spots compress the negative tail before Sq scaling.
    valley_scale = 1.0 + isolation_strength * 0.60 * (isolation - 0.50)
    rounded = median + np.where(
        centered < 0.0,
        valley_scale * centered,
        (1.0 + isolation_strength * 0.08 * (isolation - 0.50)) * centered,
    )
    rounded = project_unit_rq_np(rounded)
    highpass = np.asarray(prior, dtype=np.float64) - ndimage.gaussian_filter(
        np.asarray(prior, dtype=np.float64),
        sigma=max(float(rough_texture_sigma_px), 0.5),
        mode="wrap",
    )
    highpass = project_unit_rq_np(highpass)
    rough = project_unit_rq_np(
        float(rough_structure_weight) * rounded
        + float(rough_texture_weight) * highpass
    )
    fraction = float(
        np.clip(
            (float(conditioning_sq_nm) - float(rough_start_nm))
            / (float(rough_full_nm) - float(rough_start_nm)),
            0.0,
            1.0,
        )
    )
    # Smoothstep avoids a visible model-regime seam around 3 nm.
    fraction = fraction * fraction * (3.0 - 2.0 * fraction)
    return project_unit_rq_np(
        (1.0 - fraction) * baseline + fraction * rough
    ).astype(np.float32)


def regime_adaptive_discrete_smooth_island_blend(
    smooth_structure: np.ndarray,
    rough_structure: np.ndarray,
    prior: np.ndarray,
    baseline_structure: np.ndarray,
    *,
    conditioning_sq_nm: float,
    island_target: dict[str, float] | None,
    smooth_island_full_below_nm: float = 3.30,
    m22_full_above_nm: float = 3.80,
    smooth_structure_weight: float = 0.94,
    smooth_texture_weight: float = 0.06,
    smooth_texture_sigma_px: float = 1.6,
    smooth_tip_sigma_px: float = 0.25,
    smooth_marginal_skew_shape: float = -1.0,
    **m22_parameters: float,
) -> np.ndarray:
    """Use explicit micro-islands at low Sq and frozen M22 at high Sq.

    Smooth AFM scans have a compact, mildly left-skewed height marginal: only
    a small fraction of pixels occupy the deepest channels.  A rank-preserving
    skew-normal marginal regularizer reproduces that property without moving
    island boundaries or borrowing a measured AFM patch.  The mapping is
    disabled for the frozen M22 rough branch.
    """

    if m22_full_above_nm <= smooth_island_full_below_nm:
        raise ValueError("M22 full threshold must exceed smooth-island full threshold")
    if not 0.0 <= smooth_structure_weight <= 1.0:
        raise ValueError("smooth structure weight must be in [0, 1]")
    if smooth_texture_weight < 0.0:
        raise ValueError("smooth texture weight must be nonnegative")
    smooth = ndimage.gaussian_filter(
        project_unit_rq_np(smooth_structure),
        sigma=max(float(smooth_tip_sigma_px), 0.0),
        mode="wrap",
    )
    highpass = np.asarray(prior, dtype=np.float64) - ndimage.gaussian_filter(
        np.asarray(prior, dtype=np.float64),
        sigma=max(float(smooth_texture_sigma_px), 0.5),
        mode="wrap",
    )
    highpass = project_unit_rq_np(highpass)
    smooth = project_unit_rq_np(
        float(smooth_structure_weight) * smooth
        + float(smooth_texture_weight) * highpass
    )
    flat = np.asarray(smooth, dtype=np.float64).ravel()
    order = np.argsort(flat, kind="stable")
    probabilities = (np.arange(flat.size, dtype=float) + 0.5) / flat.size
    remapped = stats.skewnorm.ppf(
        probabilities, float(smooth_marginal_skew_shape)
    )
    smooth_flat = np.empty_like(remapped)
    smooth_flat[order] = remapped
    smooth = project_unit_rq_np(smooth_flat.reshape(smooth.shape))
    m22 = regime_adaptive_separated_island_blend(
        rough_structure,
        prior,
        baseline_structure,
        conditioning_sq_nm=conditioning_sq_nm,
        island_target=island_target,
        **m22_parameters,
    )
    fraction = float(
        np.clip(
            (float(conditioning_sq_nm) - float(smooth_island_full_below_nm))
            / (float(m22_full_above_nm) - float(smooth_island_full_below_nm)),
            0.0,
            1.0,
        )
    )
    fraction = fraction * fraction * (3.0 - 2.0 * fraction)
    if fraction >= 1.0:
        return m22.astype(np.float32, copy=False)
    return project_unit_rq_np(
        (1.0 - fraction) * smooth + fraction * m22
    ).astype(np.float32)


def render_ensemble(
    structure: list[np.ndarray],
    spectral: list[np.ndarray],
    *,
    mode: str,
    baseline_structure: list[np.ndarray] | None = None,
    rough_structure: list[np.ndarray] | None = None,
    structure_weight: float = 0.65,
    boundary_width_px: float = 0.55,
    relief_weight: float = 0.65,
    texture_weight: float = 0.08,
    base_structure_weight: float = 0.65,
    plateau_weight: float = 0.15,
    spectral_weight: float = 0.12,
    conditioning_sq_nm: float | None = None,
    island_target: dict[str, float] | None = None,
    smooth_full_below_nm: float = 0.80,
    terrace_full_above_nm: float = 1.60,
    smooth_low_frequency_weight: float = 0.10,
    smooth_spectral_weight: float = 0.72,
    smooth_spectral_sigma_px: float = 1.05,
    smooth_structure_sigma_px: float = 1.45,
    smooth_microisland_weight: float = 0.18,
    smooth_microisland_spacing_px: int = 5,
    smooth_microisland_sigma_px: float = 1.55,
    smooth_fine_texture_weight: float = 0.10,
    smooth_sparse_peak_weight: float = 0.075,
    smooth_shoulder_peak_weight: float = 0.0,
    smooth_peak_count_scale: float = 0.30,
    smooth_peak_count_min: int = 4,
    smooth_peak_count_max: int = 24,
    smooth_peak_spacing_px: int = 9,
    smooth_peak_sigma_px: float = 1.15,
    smooth_peak_minimum_quantile: float = 0.70,
    smooth_shoulder_count_scale: float = 0.85,
    smooth_shoulder_count_min: int = 20,
    smooth_shoulder_count_max: int = 48,
    smooth_shoulder_spacing_px: int = 7,
    smooth_shoulder_sigma_px: float = 2.4,
    smooth_shoulder_minimum_quantile: float = 0.58,
    smooth_winsor_quantile: float = 0.997,
    rough_start_nm: float = 2.20,
    rough_full_nm: float = 3.60,
    rough_structure_weight: float = 0.90,
    rough_texture_weight: float = 0.10,
    rough_texture_sigma_px: float = 2.4,
    rough_tip_sigma_px: float = 0.35,
    rough_isolation_score: float = 0.50,
    rough_isolation_strength: float = 1.0,
    smooth_island_full_below_nm: float = 3.30,
    m22_full_above_nm: float = 3.80,
    smooth_marginal_skew_shape: float = -1.0,
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
        elif mode == "regime_adaptive_topology_sparse_terrace":
            if conditioning_sq_nm is None:
                raise ValueError(
                    "regime-adaptive rendering requires conditioning Sq"
                )
            rendered = regime_adaptive_topology_sparse_terrace_blend(
                island,
                prior,
                conditioning_sq_nm=conditioning_sq_nm,
                island_target=island_target,
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
                smooth_fine_texture_weight=smooth_fine_texture_weight,
                smooth_sparse_peak_weight=smooth_sparse_peak_weight,
                smooth_shoulder_peak_weight=smooth_shoulder_peak_weight,
                smooth_peak_count_scale=smooth_peak_count_scale,
                smooth_peak_count_min=smooth_peak_count_min,
                smooth_peak_count_max=smooth_peak_count_max,
                smooth_peak_spacing_px=smooth_peak_spacing_px,
                smooth_peak_sigma_px=smooth_peak_sigma_px,
                smooth_peak_minimum_quantile=smooth_peak_minimum_quantile,
                smooth_shoulder_count_scale=smooth_shoulder_count_scale,
                smooth_shoulder_count_min=smooth_shoulder_count_min,
                smooth_shoulder_count_max=smooth_shoulder_count_max,
                smooth_shoulder_spacing_px=smooth_shoulder_spacing_px,
                smooth_shoulder_sigma_px=smooth_shoulder_sigma_px,
                smooth_shoulder_minimum_quantile=(
                    smooth_shoulder_minimum_quantile
                ),
                smooth_winsor_quantile=smooth_winsor_quantile,
            )
        elif mode == "regime_adaptive_separated_islands":
            if conditioning_sq_nm is None:
                raise ValueError(
                    "regime-adaptive rendering requires conditioning Sq"
                )
            if baseline_structure is None:
                raise ValueError(
                    "separated-island rendering requires baseline structure"
                )
            rendered = regime_adaptive_separated_island_blend(
                island,
                prior,
                baseline_structure[index % len(baseline_structure)],
                conditioning_sq_nm=conditioning_sq_nm,
                island_target=island_target,
                rough_start_nm=rough_start_nm,
                rough_full_nm=rough_full_nm,
                rough_structure_weight=rough_structure_weight,
                rough_texture_weight=rough_texture_weight,
                rough_texture_sigma_px=rough_texture_sigma_px,
                rough_tip_sigma_px=rough_tip_sigma_px,
                rough_isolation_score=rough_isolation_score,
                rough_isolation_strength=rough_isolation_strength,
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
                smooth_fine_texture_weight=smooth_fine_texture_weight,
                smooth_sparse_peak_weight=smooth_sparse_peak_weight,
                smooth_shoulder_peak_weight=smooth_shoulder_peak_weight,
                smooth_peak_count_scale=smooth_peak_count_scale,
                smooth_peak_count_min=smooth_peak_count_min,
                smooth_peak_count_max=smooth_peak_count_max,
                smooth_peak_spacing_px=smooth_peak_spacing_px,
                smooth_peak_sigma_px=smooth_peak_sigma_px,
                smooth_peak_minimum_quantile=(
                    smooth_peak_minimum_quantile
                ),
                smooth_shoulder_count_scale=smooth_shoulder_count_scale,
                smooth_shoulder_count_min=smooth_shoulder_count_min,
                smooth_shoulder_count_max=smooth_shoulder_count_max,
                smooth_shoulder_spacing_px=smooth_shoulder_spacing_px,
                smooth_shoulder_sigma_px=smooth_shoulder_sigma_px,
                smooth_shoulder_minimum_quantile=(
                    smooth_shoulder_minimum_quantile
                ),
                smooth_winsor_quantile=smooth_winsor_quantile,
            )
        elif mode == "regime_adaptive_discrete_smooth_islands":
            if conditioning_sq_nm is None:
                raise ValueError(
                    "regime-adaptive rendering requires conditioning Sq"
                )
            if baseline_structure is None or rough_structure is None:
                raise ValueError(
                    "discrete smooth-island rendering requires baseline and rough structures"
                )
            rendered = regime_adaptive_discrete_smooth_island_blend(
                island,
                rough_structure[index % len(rough_structure)],
                prior,
                baseline_structure[index % len(baseline_structure)],
                conditioning_sq_nm=conditioning_sq_nm,
                island_target=island_target,
                smooth_island_full_below_nm=smooth_island_full_below_nm,
                m22_full_above_nm=m22_full_above_nm,
                smooth_structure_weight=structure_weight,
                smooth_texture_weight=texture_weight,
                smooth_texture_sigma_px=rough_texture_sigma_px,
                smooth_tip_sigma_px=rough_tip_sigma_px,
                smooth_marginal_skew_shape=smooth_marginal_skew_shape,
                rough_start_nm=rough_start_nm,
                rough_full_nm=rough_full_nm,
                rough_structure_weight=rough_structure_weight,
                rough_texture_weight=rough_texture_weight,
                rough_texture_sigma_px=rough_texture_sigma_px,
                rough_tip_sigma_px=rough_tip_sigma_px,
                rough_isolation_score=rough_isolation_score,
                rough_isolation_strength=rough_isolation_strength,
                smooth_full_below_nm=smooth_full_below_nm,
                terrace_full_above_nm=terrace_full_above_nm,
                boundary_width_px=boundary_width_px,
                structure_weight=0.50,
                plateau_weight=plateau_weight,
                relief_weight=relief_weight,
                spectral_weight=spectral_weight,
                texture_weight=0.05,
                smooth_spectral_weight=smooth_spectral_weight,
                smooth_spectral_sigma_px=smooth_spectral_sigma_px,
                smooth_structure_sigma_px=smooth_structure_sigma_px,
                smooth_fine_texture_weight=smooth_fine_texture_weight,
                smooth_sparse_peak_weight=smooth_sparse_peak_weight,
                smooth_shoulder_peak_weight=smooth_shoulder_peak_weight,
                smooth_peak_count_scale=smooth_peak_count_scale,
                smooth_peak_count_min=smooth_peak_count_min,
                smooth_peak_count_max=smooth_peak_count_max,
                smooth_peak_spacing_px=smooth_peak_spacing_px,
                smooth_peak_sigma_px=smooth_peak_sigma_px,
                smooth_peak_minimum_quantile=smooth_peak_minimum_quantile,
                smooth_shoulder_count_scale=smooth_shoulder_count_scale,
                smooth_shoulder_count_min=smooth_shoulder_count_min,
                smooth_shoulder_count_max=smooth_shoulder_count_max,
                smooth_shoulder_spacing_px=smooth_shoulder_spacing_px,
                smooth_shoulder_sigma_px=smooth_shoulder_sigma_px,
                smooth_shoulder_minimum_quantile=smooth_shoulder_minimum_quantile,
                smooth_winsor_quantile=smooth_winsor_quantile,
            )
        else:
            raise ValueError(f"unknown renderer mode: {mode}")
        result.append(rendered)
    return result
