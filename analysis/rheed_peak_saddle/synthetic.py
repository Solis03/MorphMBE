"""Deterministic synthetic grayscale RHEED renderer for Stage 1 validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage


BRIDGE_STRENGTH_GRID = tuple(round(i * 0.05, 2) for i in range(21))
DEVELOPMENT_SEEDS = tuple(2026071300 + i for i in range(len(BRIDGE_STRENGTH_GRID) * 3 + 8))
HOLDOUT_SEEDS = tuple(2026072300 + i for i in range(len(BRIDGE_STRENGTH_GRID) * 3 + 8))
DEVELOPMENT_V2_SEEDS = tuple(2026075100 + i for i in range(len(BRIDGE_STRENGTH_GRID) * 4 + 14))
HOLDOUT_V2_SEEDS = tuple(2026077100 + i for i in range(len(BRIDGE_STRENGTH_GRID) * 4 + 14))


@dataclass(frozen=True)
class SyntheticSpotTruth:
    spot_id: int
    row_id: int
    site_index: int
    center_x: float
    center_y: float
    sigma_x: float
    sigma_y: float
    amplitude: float
    profile_family: str
    missing: int = 0
    edge_or_crop_flag: int = 0
    saturation_flag: int = 0


@dataclass(frozen=True)
class SyntheticPairTruth:
    pair_id: str
    spot_i: int
    spot_j: int
    row_id: int
    site_i: int
    site_j: int
    true_bridge_strength: float
    bridge_width: float
    orientation: str = "horizontal"
    adversarial_type: str = ""
    valid_expected: int = 1
    ineligible_reason: str = ""


@dataclass(frozen=True)
class SyntheticRheed:
    image_id: str
    split: str
    image: np.ndarray
    display_image: np.ndarray
    background_map: np.ndarray
    bridge_map: np.ndarray
    valid_region_mask: np.ndarray
    spots: tuple[SyntheticSpotTruth, ...]
    pairs: tuple[SyntheticPairTruth, ...]
    nuisance: dict[str, float | str | int]


def _as_float32(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float32)


def _rotated_coordinates(
    xx: np.ndarray,
    yy: np.ndarray,
    center_x: float,
    center_y: float,
    angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta = math.radians(angle_degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = xx - center_x
    dy = yy - center_y
    along = cos_t * dx + sin_t * dy
    across = -sin_t * dx + cos_t * dy
    return along, across


def _spot_profile(
    xx: np.ndarray,
    yy: np.ndarray,
    spot: SyntheticSpotTruth,
    angle_degrees: float,
) -> np.ndarray:
    along, across = _rotated_coordinates(xx, yy, spot.center_x, spot.center_y, angle_degrees)
    radius2 = (along / max(spot.sigma_x, 1e-6)) ** 2 + (across / max(spot.sigma_y, 1e-6)) ** 2
    if spot.profile_family == "moffat":
        beta = 2.3
        return spot.amplitude * np.power(1.0 + radius2 / beta, -beta)
    return spot.amplitude * np.exp(-0.5 * radius2)


def _line_masks(
    shape: tuple[int, int],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    half_width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    yy, xx = np.indices(shape, dtype=float)
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        empty = np.zeros(shape, dtype=bool)
        return empty, empty, empty
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    rel_x = xx - x0
    rel_y = yy - y0
    along = rel_x * ux + rel_y * uy
    across = rel_x * px + rel_y * py
    segment = (along >= 0.0) & (along <= length)
    return segment & (np.abs(across) <= half_width), segment, across


def _background(
    shape: tuple[int, int],
    *,
    halo_strength: float,
    gradient_strength: float,
    gradient_axis: str,
    direct_beam_strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = shape
    yy, xx = np.indices(shape, dtype=float)
    screen = 0.035 + 0.025 * np.exp(
        -0.5 * (((xx - width * 0.48) / (width * 0.75)) ** 2 + ((yy - height * 0.5) / (height * 0.9)) ** 2)
    )
    halo = halo_strength * np.exp(
        -0.5 * (((xx - width * 0.52) / (width * 0.34)) ** 2 + ((yy - height * 0.52) / (height * 0.42)) ** 2)
    )
    if gradient_axis == "vertical":
        gradient = gradient_strength * yy / max(height - 1, 1)
    else:
        gradient = gradient_strength * xx / max(width - 1, 1)
    beam_x = width * (0.42 + 0.10 * rng.random())
    beam_y = height * (0.45 + 0.12 * rng.random())
    direct_beam = direct_beam_strength * np.exp(
        -0.5 * (((xx - beam_x) / (width * 0.12)) ** 2 + ((yy - beam_y) / (height * 0.18)) ** 2)
    )
    return screen + halo + gradient + direct_beam


def render_synthetic_rheed(
    *,
    split: str,
    seed: int,
    image_id: str,
    bridge_strength: float,
    image_shape: tuple[int, int] = (160, 240),
    row_count: int = 2,
    spots_per_row: int = 6,
    spacing: float = 30.0,
    spacing_jitter: float = 1.4,
    width_scale: float = 1.0,
    profile_mix: str = "mixed",
    halo_strength: float = 0.04,
    gradient_strength: float = 0.02,
    gradient_axis: str = "horizontal",
    direct_beam_strength: float = 0.0,
    exposure: float = 1.0,
    additive_offset: float = 0.0,
    display_gamma: float = 1.0,
    read_noise_sigma: float = 0.002,
    poisson_noise_scale: float = 0.0,
    blur_sigma: float = 0.0,
    rotation_degrees: float = 0.0,
    row_tilt_degrees: float = 0.0,
    row_curvature: float = 0.0,
    translation_x: float = 0.0,
    translation_y: float = 0.0,
    crop_left: float = 0.0,
    missing_spot_rate: float = 0.0,
    unequal_intensity: float = 0.25,
    bridge_width: float = 4.0,
    vertical_bridge: bool = False,
    isolated_on_halo: bool = False,
    saturated_spots: bool = False,
    partial_crop: bool = False,
    missing_site_indices: tuple[tuple[int, int], ...] | None = None,
    duplicate_spot_artifact: bool = False,
    false_halo_peak: bool = False,
) -> SyntheticRheed:
    """Render a synthetic RHEED image plus full spot/pair ground truth."""
    rng = np.random.default_rng(seed)
    height, width = image_shape
    yy, xx = np.indices(image_shape, dtype=float)
    background = _background(
        image_shape,
        halo_strength=halo_strength + (0.10 if isolated_on_halo else 0.0),
        gradient_strength=gradient_strength,
        gradient_axis=gradient_axis,
        direct_beam_strength=direct_beam_strength,
        rng=rng,
    )
    signal = np.zeros(image_shape, dtype=float)
    bridge_map = np.zeros(image_shape, dtype=float)
    valid_region = np.ones(image_shape, dtype=bool)
    if crop_left > 0:
        valid_region[:, : int(round(crop_left))] = False

    row_gap = 34.0 if row_count > 1 else 0.0
    base_y = height * 0.50 - row_gap * (row_count - 1) / 2.0
    start_x = width * 0.50 - spacing * (spots_per_row - 1) / 2.0 + translation_x
    spots: list[SyntheticSpotTruth] = []
    rendered_ids: list[int] = []
    global_angle = row_tilt_degrees + rotation_degrees
    forced_missing = set(missing_site_indices or ())

    for row_id in range(row_count):
        row_y = base_y + row_id * row_gap + translation_y
        for col in range(spots_per_row):
            missing = ((row_id, col) in forced_missing) or (rng.random() < missing_spot_rate and 0 < col < spots_per_row - 1)
            jitter = rng.normal(0.0, spacing_jitter)
            cx = start_x + col * spacing + jitter
            centered_col = col - (spots_per_row - 1) / 2.0
            cy = row_y + math.tan(math.radians(row_tilt_degrees)) * centered_col * spacing + row_curvature * centered_col**2
            if partial_crop and row_id == 0 and col == 0:
                cx = 3.2
            sigma_x = width_scale * rng.uniform(4.2, 6.5)
            sigma_y = width_scale * rng.uniform(2.4, 4.1)
            amp = rng.uniform(0.82, 1.18) * (1.0 - unequal_intensity * (col % 2))
            family = "gaussian"
            if profile_mix == "moffat":
                family = "moffat"
            elif profile_mix == "mixed":
                family = "moffat" if (row_id + col + seed) % 3 == 0 else "gaussian"
            edge_flag = int(cx < 2.5 * sigma_x or cy < 2.5 * sigma_y or cx > width - 2.5 * sigma_x or cy > height - 2.5 * sigma_y)
            sat_flag = int(saturated_spots and row_id == 0 and col in (1, 2))
            spot = SyntheticSpotTruth(
                spot_id=len(spots),
                row_id=row_id,
                site_index=col,
                center_x=float(cx),
                center_y=float(cy),
                sigma_x=float(sigma_x),
                sigma_y=float(sigma_y),
                amplitude=float(amp * (1.7 if sat_flag else 1.0)),
                profile_family=family,
                missing=int(missing),
                edge_or_crop_flag=edge_flag,
                saturation_flag=sat_flag,
            )
            spots.append(spot)
            if not missing:
                signal += _spot_profile(xx, yy, spot, global_angle)
                if duplicate_spot_artifact and row_id == 0 and col == max(1, spots_per_row // 2):
                    dup = SyntheticSpotTruth(
                        spot_id=-1,
                        row_id=row_id,
                        site_index=col,
                        center_x=float(cx + 0.85 * sigma_x),
                        center_y=float(cy + 0.35 * sigma_y),
                        sigma_x=float(max(1.8, sigma_x * 0.55)),
                        sigma_y=float(max(1.2, sigma_y * 0.55)),
                        amplitude=float(amp * 0.45),
                        profile_family="gaussian",
                    )
                    signal += _spot_profile(xx, yy, dup, global_angle)
                rendered_ids.append(spot.spot_id)

    if false_halo_peak:
        peak = SyntheticSpotTruth(
            spot_id=-2,
            row_id=0,
            site_index=-1,
            center_x=float(width * 0.50),
            center_y=float(height * 0.50 + (row_gap * 0.5 if row_count > 1 else 11.0)),
            sigma_x=7.5,
            sigma_y=5.5,
            amplitude=0.42,
            profile_family="moffat",
        )
        signal += _spot_profile(xx, yy, peak, 0.0)

    pairs: list[SyntheticPairTruth] = []
    spots_by_row: dict[int, list[SyntheticSpotTruth]] = {}
    all_spots_by_row: dict[int, list[SyntheticSpotTruth]] = {}
    for spot in spots:
        all_spots_by_row.setdefault(spot.row_id, []).append(spot)
        if not spot.missing:
            spots_by_row.setdefault(spot.row_id, []).append(spot)
    for row_id, row_spots in all_spots_by_row.items():
        row_spots = sorted(row_spots, key=lambda spot: spot.site_index)
        for left, right in zip(row_spots[:-1], row_spots[1:]):
            gap = math.hypot(right.center_x - left.center_x, right.center_y - left.center_y)
            expected_gap = spacing * 1.35
            ineligible_reasons: list[str] = []
            if left.missing or right.missing:
                ineligible_reasons.append("missing_endpoint")
            if left.edge_or_crop_flag or right.edge_or_crop_flag:
                ineligible_reasons.append("edge_or_partial_crop")
            if gap > expected_gap:
                ineligible_reasons.append("implausible_true_spacing")
            valid_expected = int(not ineligible_reasons)
            strength = 0.0 if vertical_bridge else float(bridge_strength)
            pair = SyntheticPairTruth(
                pair_id=f"{image_id}_r{row_id}_{left.spot_id}_{right.spot_id}",
                spot_i=left.spot_id,
                spot_j=right.spot_id,
                row_id=row_id,
                site_i=left.site_index,
                site_j=right.site_index,
                true_bridge_strength=strength,
                bridge_width=float(bridge_width),
                orientation="horizontal",
                adversarial_type=(
                    "vertical_bridge"
                    if vertical_bridge
                    else "isolated_halo"
                    if isolated_on_halo and strength == 0
                    else "partial_crop"
                    if left.edge_or_crop_flag or right.edge_or_crop_flag
                    else "saturated"
                    if left.saturation_flag or right.saturation_flag
                    else ""
                ),
                valid_expected=valid_expected,
                ineligible_reason=";".join(ineligible_reasons),
            )
            pairs.append(pair)
            if strength > 0 and valid_expected:
                mask, _, _ = _line_masks(
                    image_shape,
                    left.center_x,
                    left.center_y,
                    right.center_x,
                    right.center_y,
                    max(float(bridge_width), 1.0),
                )
                _, _, across = _line_masks(
                    image_shape,
                    left.center_x,
                    left.center_y,
                    right.center_x,
                    right.center_y,
                    max(float(bridge_width) * 2.5, 1.0),
                )
                core_radius = 1.5 * max(left.sigma_y, right.sigma_y)
                yy_f = yy
                xx_f = xx
                core_left = (xx_f - left.center_x) ** 2 + (yy_f - left.center_y) ** 2 <= core_radius**2
                core_right = (xx_f - right.center_x) ** 2 + (yy_f - right.center_y) ** 2 <= core_radius**2
                bridge = 0.82 * strength * min(left.amplitude, right.amplitude) * np.exp(-0.5 * (across / max(bridge_width, 1.0)) ** 2)
                bridge *= mask & ~(core_left | core_right)
                bridge_map = np.maximum(bridge_map, bridge)

    if vertical_bridge and len(spots_by_row) >= 2:
        upper = sorted(spots_by_row[min(spots_by_row)], key=lambda spot: spot.center_x)[spots_per_row // 2]
        lower = sorted(spots_by_row[max(spots_by_row)], key=lambda spot: spot.center_x)[spots_per_row // 2]
        mask, _, across = _line_masks(image_shape, upper.center_x, upper.center_y, lower.center_x, lower.center_y, bridge_width)
        vertical = 0.85 * min(upper.amplitude, lower.amplitude) * np.exp(-0.5 * (across / max(bridge_width, 1.0)) ** 2)
        bridge_map = np.maximum(bridge_map, vertical * mask)

    signal += bridge_map
    linear = exposure * (background + signal) + additive_offset
    if poisson_noise_scale > 0:
        linear += rng.normal(0.0, poisson_noise_scale * np.sqrt(np.clip(linear, 0.0, None) + 1e-6), size=image_shape)
    if read_noise_sigma > 0:
        linear += rng.normal(0.0, read_noise_sigma, size=image_shape)
    if blur_sigma > 0:
        linear = ndimage.gaussian_filter(linear, blur_sigma)
    saturation_level = 1.25 if saturated_spots else 1.8
    linear = np.clip(linear, 0.0, saturation_level)

    display = np.clip(linear - np.nanmin(linear), 0.0, None)
    display /= max(float(np.nanmax(display)), 1e-8)
    display = np.power(display, max(float(display_gamma), 1e-6))

    return SyntheticRheed(
        image_id=image_id,
        split=split,
        image=_as_float32(linear),
        display_image=_as_float32(display),
        background_map=_as_float32(exposure * background + additive_offset),
        bridge_map=_as_float32(exposure * bridge_map),
        valid_region_mask=valid_region,
        spots=tuple(spots),
        pairs=tuple(pairs),
        nuisance={
            "seed": int(seed),
            "bridge_strength": float(bridge_strength),
            "row_count": int(row_count),
            "spots_per_row": int(spots_per_row),
            "spacing": float(spacing),
            "width_scale": float(width_scale),
            "profile_mix": profile_mix,
            "halo_strength": float(halo_strength),
            "gradient_strength": float(gradient_strength),
            "gradient_axis": gradient_axis,
            "direct_beam_strength": float(direct_beam_strength),
            "exposure": float(exposure),
            "additive_offset": float(additive_offset),
            "display_gamma": float(display_gamma),
            "read_noise_sigma": float(read_noise_sigma),
            "poisson_noise_scale": float(poisson_noise_scale),
            "blur_sigma": float(blur_sigma),
            "rotation_degrees": float(rotation_degrees),
            "row_tilt_degrees": float(row_tilt_degrees),
            "row_curvature": float(row_curvature),
            "translation_x": float(translation_x),
            "translation_y": float(translation_y),
            "crop_left": float(crop_left),
            "missing_spot_rate": float(missing_spot_rate),
            "unequal_intensity": float(unequal_intensity),
            "bridge_width": float(bridge_width),
            "vertical_bridge": int(vertical_bridge),
            "isolated_on_halo": int(isolated_on_halo),
            "saturated_spots": int(saturated_spots),
            "partial_crop": int(partial_crop),
            "duplicate_spot_artifact": int(duplicate_spot_artifact),
            "false_halo_peak": int(false_halo_peak),
            "missing_site_indices": ";".join(f"{row}:{col}" for row, col in sorted(forced_missing)),
        },
    )


def _sweep_kwargs(split: str, index: int, strength: float) -> dict[str, float | int | str | bool]:
    holdout = split == "holdout"
    variant = index % 3
    return {
        "row_count": 3 if holdout and variant == 2 else 2,
        "spots_per_row": 7 if holdout and variant != 0 else 6,
        "spacing": 28.0 + 2.5 * variant + (1.5 if holdout else 0.0),
        "spacing_jitter": 1.1 if not holdout else 1.8,
        "width_scale": 0.9 + 0.12 * variant + (0.08 if holdout else 0.0),
        "profile_mix": "moffat" if holdout and variant == 1 else "mixed",
        "halo_strength": 0.02 + 0.035 * variant + (0.025 if holdout else 0.0),
        "gradient_strength": 0.012 + 0.008 * variant,
        "gradient_axis": "vertical" if (holdout and variant == 2) else "horizontal",
        "direct_beam_strength": 0.08 if variant == 2 else 0.02,
        "exposure": [0.75, 1.0, 1.25][variant],
        "additive_offset": [0.00, 0.025, 0.05][variant],
        "display_gamma": [0.96, 1.00, 1.04][variant],
        "read_noise_sigma": 0.0015 + 0.001 * variant,
        "poisson_noise_scale": 0.003 if variant == 2 else 0.001,
        "blur_sigma": 0.35 if variant == 1 else (0.55 if holdout and variant == 2 else 0.0),
        "rotation_degrees": [-1.2, 0.4, 1.6][variant] + (0.5 if holdout else 0.0),
        "row_tilt_degrees": [-2.0, 0.0, 2.2][variant],
        "row_curvature": 0.04 if holdout and variant == 2 else 0.015 * variant,
        "translation_x": [-1.5, 0.0, 1.5][variant],
        "translation_y": [0.8, -0.5, 1.2][variant],
        "missing_spot_rate": 0.03 if variant == 2 else 0.0,
        "unequal_intensity": 0.18 + 0.08 * variant,
        "bridge_width": 3.2 + 0.5 * variant,
        "bridge_strength": strength,
    }


def make_synthetic_split(split: str) -> tuple[SyntheticRheed, ...]:
    """Create the deterministic development or locked holdout synthetic set."""
    if split not in {"development", "holdout"}:
        raise ValueError("split must be 'development' or 'holdout'")
    seeds = DEVELOPMENT_SEEDS if split == "development" else HOLDOUT_SEEDS
    examples: list[SyntheticRheed] = []
    seed_index = 0
    for strength_index, strength in enumerate(BRIDGE_STRENGTH_GRID):
        for variant in range(3):
            seed = seeds[seed_index]
            seed_index += 1
            kwargs = _sweep_kwargs(split, variant, strength)
            examples.append(
                render_synthetic_rheed(
                    split=split,
                    seed=seed,
                    image_id=f"{split}_sweep_{strength_index:02d}_{variant}",
                    **kwargs,
                )
            )
    adversarial_specs = (
        {"bridge_strength": 0.0, "isolated_on_halo": True, "halo_strength": 0.18, "direct_beam_strength": 0.0},
        {"bridge_strength": 0.0, "gradient_strength": 0.11, "gradient_axis": "horizontal"},
        {"bridge_strength": 0.0, "vertical_bridge": True, "row_count": 2, "spots_per_row": 5},
        {"bridge_strength": 0.0, "direct_beam_strength": 0.42, "halo_strength": 0.08},
        {"bridge_strength": 0.75, "unequal_intensity": 0.55},
        {"bridge_strength": 0.65, "partial_crop": True, "crop_left": 8.0},
        {"bridge_strength": 0.65, "saturated_spots": True, "blur_sigma": 0.15},
        {"bridge_strength": 0.0, "halo_strength": 0.22, "blur_sigma": 0.45, "profile_mix": "moffat"},
    )
    for adv_index, spec in enumerate(adversarial_specs):
        seed = seeds[seed_index]
        seed_index += 1
        base = {
            "row_count": 2 if split == "development" else 3,
            "spots_per_row": 6,
            "spacing": 30.0 if split == "development" else 32.0,
            "spacing_jitter": 1.2 if split == "development" else 2.0,
            "width_scale": 1.0 if split == "development" else 1.12,
            "profile_mix": "mixed",
            "halo_strength": 0.04,
            "gradient_strength": 0.02,
            "direct_beam_strength": 0.02,
            "exposure": 1.0,
            "additive_offset": 0.02,
            "display_gamma": 1.0,
            "read_noise_sigma": 0.002,
            "poisson_noise_scale": 0.001,
            "blur_sigma": 0.0,
            "rotation_degrees": 0.7 if split == "holdout" else 0.0,
            "row_tilt_degrees": -1.0 if adv_index % 2 else 1.0,
            "row_curvature": 0.03 if split == "holdout" else 0.0,
            "translation_x": 0.0,
            "translation_y": 0.0,
            "missing_spot_rate": 0.0,
            "unequal_intensity": 0.22,
            "bridge_width": 3.5,
        }
        base.update(spec)
        examples.append(
            render_synthetic_rheed(
                split=split,
                seed=seed,
                image_id=f"{split}_adversarial_{adv_index:02d}",
                **base,
            )
        )
    return tuple(examples)


def make_nuisance_invariance_set(split: str) -> tuple[SyntheticRheed, ...]:
    """Small paired set with identical morphology and changed exposure/gamma/offset."""
    seeds = (2026073301, 2026073302, 2026073303) if split == "development" else (2026074301, 2026074302, 2026074303)
    examples: list[SyntheticRheed] = []
    strengths = (0.15, 0.50, 0.85)
    nuisance_variants = (
        {"exposure": 1.0, "additive_offset": 0.02, "display_gamma": 1.0},
        {"exposure": 0.70, "additive_offset": 0.00, "display_gamma": 0.96},
        {"exposure": 1.35, "additive_offset": 0.06, "display_gamma": 1.04},
    )
    for strength_index, strength in enumerate(strengths):
        for variant_index, nuisance in enumerate(nuisance_variants):
            examples.append(
                render_synthetic_rheed(
                    split=split,
                    seed=seeds[strength_index],
                    image_id=f"{split}_nuisance_{strength_index}_{variant_index}",
                    bridge_strength=strength,
                    row_count=2 if split == "development" else 3,
                    spots_per_row=6,
                    spacing=30.0 if split == "development" else 31.5,
                    spacing_jitter=0.0,
                    width_scale=1.0,
                    profile_mix="mixed",
                    halo_strength=0.05,
                    gradient_strength=0.02,
                    direct_beam_strength=0.04,
                    read_noise_sigma=0.0,
                    poisson_noise_scale=0.0,
                    blur_sigma=0.0,
                    rotation_degrees=0.0,
                    row_tilt_degrees=0.8,
                    row_curvature=0.0,
                    translation_x=0.0,
                    translation_y=0.0,
                    missing_spot_rate=0.0,
                    unequal_intensity=0.22,
                    bridge_width=3.5,
                    **nuisance,
                )
            )
    return tuple(examples)


def _sweep_kwargs_v2(split: str, variant: int, strength: float) -> dict[str, float | int | str | bool | tuple[tuple[int, int], ...]]:
    holdout = split == "holdout_v2"
    kwargs: dict[str, float | int | str | bool | tuple[tuple[int, int], ...]] = {
        "row_count": 2 + int((variant == 3) or (holdout and variant == 2)),
        "spots_per_row": 7 if variant in (1, 3) else 6,
        "spacing": 29.0 + 1.2 * variant + (1.1 if holdout else 0.0),
        "spacing_jitter": 0.9 + 0.25 * variant + (0.25 if holdout else 0.0),
        "width_scale": 0.95 + 0.08 * variant + (0.05 if holdout else 0.0),
        "profile_mix": "moffat" if holdout and variant == 2 else "mixed",
        "halo_strength": 0.025 + 0.025 * variant + (0.018 if holdout else 0.0),
        "gradient_strength": 0.015 + 0.006 * variant,
        "gradient_axis": "vertical" if variant == 3 else "horizontal",
        "direct_beam_strength": 0.06 if variant == 2 else 0.02,
        "exposure": (0.82, 1.00, 1.18, 1.30)[variant],
        "additive_offset": (0.00, 0.02, 0.045, 0.06)[variant],
        "display_gamma": (0.97, 1.0, 1.03, 1.05)[variant],
        "read_noise_sigma": 0.0012 + 0.0006 * variant,
        "poisson_noise_scale": 0.001 + 0.001 * (variant >= 2),
        "blur_sigma": 0.10 * variant,
        "rotation_degrees": (-1.4, -0.4, 0.8, 1.7)[variant] + (0.35 if holdout else 0.0),
        "row_tilt_degrees": (-1.6, 0.4, 1.5, -2.0)[variant],
        "row_curvature": 0.018 * variant + (0.015 if holdout and variant == 3 else 0.0),
        "translation_x": (-1.0, 0.8, -0.5, 1.6)[variant],
        "translation_y": (0.5, -0.4, 1.0, -0.8)[variant],
        "missing_spot_rate": 0.0,
        "unequal_intensity": 0.18 + 0.07 * variant,
        "bridge_width": 1.9 + 0.18 * variant,
    }
    if variant == 1:
        kwargs["missing_site_indices"] = ((0, 3),)
    if variant == 3:
        kwargs["missing_site_indices"] = ((0, 2), (0, 3))
    return kwargs


def make_synthetic_split_v2(split: str) -> tuple[SyntheticRheed, ...]:
    """Create Stage 1B development or fresh locked holdout-v2 examples."""
    if split not in {"development_v2", "holdout_v2"}:
        raise ValueError("split must be 'development_v2' or 'holdout_v2'")
    seeds = DEVELOPMENT_V2_SEEDS if split == "development_v2" else HOLDOUT_V2_SEEDS
    examples: list[SyntheticRheed] = []
    seed_index = 0
    for strength_index, strength in enumerate(BRIDGE_STRENGTH_GRID):
        for variant in range(4):
            kwargs = _sweep_kwargs_v2(split, variant, strength)
            examples.append(
                render_synthetic_rheed(
                    split=split,
                    seed=seeds[seed_index],
                    image_id=f"{split}_sweep_{strength_index:02d}_{variant}",
                    bridge_strength=strength,
                    **kwargs,
                )
            )
            seed_index += 1
    challenge_specs = (
        ("missing_one_site", {"bridge_strength": 0.45, "missing_site_indices": ((0, 3),)}),
        ("missing_two_sites", {"bridge_strength": 0.45, "missing_site_indices": ((0, 2), (0, 3))}),
        ("duplicate_local_maximum", {"bridge_strength": 0.65, "duplicate_spot_artifact": True}),
        ("false_halo_maximum", {"bridge_strength": 0.0, "false_halo_peak": True, "halo_strength": 0.16}),
        ("direct_beam_broad_maximum", {"bridge_strength": 0.0, "direct_beam_strength": 0.48, "halo_strength": 0.08}),
        ("nonuniform_spacing", {"bridge_strength": 0.55, "spacing_jitter": 3.0}),
        ("nearby_rows", {"bridge_strength": 0.55, "row_count": 3, "spots_per_row": 6}),
        ("mild_curvature", {"bridge_strength": 0.55, "row_curvature": 0.08}),
        ("unequal_amplitudes", {"bridge_strength": 0.75, "unequal_intensity": 0.55}),
        ("saturated_endpoint", {"bridge_strength": 0.65, "saturated_spots": True}),
        ("partial_crop_endpoint", {"bridge_strength": 0.65, "partial_crop": True, "crop_left": 8.0}),
        ("edge_of_frame_row", {"bridge_strength": 0.50, "translation_y": -42.0}),
        ("valid_near_missing_site", {"bridge_strength": 0.70, "missing_site_indices": ((0, 4),)}),
        ("vertical_bridge_negative", {"bridge_strength": 0.0, "vertical_bridge": True, "row_count": 2}),
    )
    for challenge_index, (name, spec) in enumerate(challenge_specs):
        base: dict[str, float | int | str | bool | tuple[tuple[int, int], ...]] = {
            "row_count": 2 + int(split == "holdout_v2" and challenge_index % 4 == 0),
            "spots_per_row": 7 if challenge_index % 3 == 0 else 6,
            "spacing": 30.0 + (1.0 if split == "holdout_v2" else 0.0),
            "spacing_jitter": 1.0 + (0.4 if split == "holdout_v2" else 0.0),
            "width_scale": 1.0 + (0.08 if split == "holdout_v2" else 0.0),
            "profile_mix": "mixed",
            "halo_strength": 0.05,
            "gradient_strength": 0.025,
            "direct_beam_strength": 0.03,
            "exposure": 1.0,
            "additive_offset": 0.02,
            "display_gamma": 1.0,
            "read_noise_sigma": 0.0015,
            "poisson_noise_scale": 0.001,
            "blur_sigma": 0.1,
            "rotation_degrees": 0.4 + (0.4 if split == "holdout_v2" else 0.0),
            "row_tilt_degrees": -1.0 if challenge_index % 2 else 1.0,
            "row_curvature": 0.02,
            "translation_x": 0.0,
            "translation_y": 0.0,
            "missing_spot_rate": 0.0,
            "unequal_intensity": 0.22,
            "bridge_width": 3.5,
        }
        base.update(spec)
        examples.append(
            render_synthetic_rheed(
                split=split,
                seed=seeds[seed_index],
                image_id=f"{split}_challenge_{name}",
                **base,
            )
        )
        seed_index += 1
    return tuple(examples)


def iter_true_spots(example: SyntheticRheed, *, require_rendered: bool = True) -> Iterable[SyntheticSpotTruth]:
    for spot in example.spots:
        if require_rendered and spot.missing:
            continue
        yield spot
