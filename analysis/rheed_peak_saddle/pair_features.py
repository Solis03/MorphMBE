"""Pair-level peak-saddle adhesion measurement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from analysis.rheed_peak_saddle.merge_tree import maximum_bottleneck_saddle
from analysis.rheed_peak_saddle.row_grouping import AdjacentPairCandidate
from analysis.rheed_peak_saddle.spot_detection import SpotEstimate


PAIR_FEATURE_NAMES = (
    "raw_peak_saddle_adhesion",
    "raw_peak_saddle_adhesion_unclipped",
    "isolation_persistence",
    "direct_corridor_valley_ratio",
    "corridor_mean_ratio",
    "ridge_energy_ratio",
    "bridge_width_ratio",
    "spot_spacing_over_width",
    "pair_measurement_confidence",
)


@dataclass(frozen=True)
class PairMasks:
    corridor_mask: np.ndarray
    bridge_body_mask: np.ndarray
    seed_i_mask: np.ndarray
    seed_j_mask: np.ndarray
    background_mask: np.ndarray


@dataclass(frozen=True)
class PairFeature:
    image_id: str
    pair_id: str
    spot_i: int
    spot_j: int
    row_label: int
    peak_i: float
    peak_j: float
    saddle_intensity: float
    background_intensity: float
    background_method: str
    raw_peak_saddle_adhesion: float
    raw_peak_saddle_adhesion_unclipped: float
    isolation_persistence: float
    direct_corridor_valley_ratio: float
    corridor_mean_ratio: float
    ridge_energy_ratio: float
    bridge_width_ratio: float
    spot_spacing_over_width: float
    pair_measurement_confidence: float
    valid: int
    invalid_reason: str


def pair_masks(
    shape: tuple[int, int],
    left: SpotEstimate,
    right: SpotEstimate,
    *,
    corridor_half_width: float | None = None,
    core_radius: float | None = None,
    background_offset: float | None = None,
) -> PairMasks:
    height, width = shape
    yy, xx = np.indices(shape, dtype=float)
    dx = right.center_x - left.center_x
    dy = right.center_y - left.center_y
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        empty = np.zeros(shape, dtype=bool)
        return PairMasks(empty, empty, empty, empty, empty)
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    rel_x = xx - left.center_x
    rel_y = yy - left.center_y
    along = rel_x * ux + rel_y * uy
    across = rel_x * px + rel_y * py
    width_est = max(left.equivalent_width, right.equivalent_width, 1.5)
    half_width = float(corridor_half_width if corridor_half_width is not None else max(3.0, 1.35 * width_est))
    core = float(core_radius if core_radius is not None else max(3.0, 0.95 * width_est))
    offset = float(background_offset if background_offset is not None else max(8.0, 3.0 * width_est))
    segment = (along >= 0.0) & (along <= length)
    corridor = segment & (np.abs(across) <= half_width)
    seed_i = ((xx - left.center_x) ** 2 + (yy - left.center_y) ** 2 <= core**2) & corridor
    seed_j = ((xx - right.center_x) ** 2 + (yy - right.center_y) ** 2 <= core**2) & corridor
    for seed, spot in ((seed_i, left), (seed_j, right)):
        if not np.any(seed):
            cx = int(round(spot.center_x))
            cy = int(round(spot.center_y))
            if 0 <= cy < height and 0 <= cx < width and corridor[cy, cx]:
                seed[cy, cx] = True
    bridge_body = corridor & ~seed_i & ~seed_j
    bg = segment & (
        ((np.abs(across - offset) <= half_width) | (np.abs(across + offset) <= half_width))
    )
    bg &= along >= core
    bg &= along <= length - core
    return PairMasks(corridor, bridge_body, seed_i, seed_j, bg)


def _robust_peak(image: np.ndarray, mask: np.ndarray) -> float:
    values = image[mask]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, 98.0))


def _robust_background(image: np.ndarray, masks: PairMasks) -> tuple[float, str]:
    values = image[masks.background_mask]
    method = "symmetric_offset_corridors"
    if values.size < 8:
        values = image[masks.bridge_body_mask]
        method = "bridge_body_low_percentile_fallback"
    if values.size == 0:
        return float("nan"), "no_background_pixels"
    if method.endswith("fallback"):
        return float(np.percentile(values, 15.0)), method
    return float(np.median(values)), method


def measure_pair_adhesion(
    image: np.ndarray,
    spots: Sequence[SpotEstimate],
    pair: AdjacentPairCandidate,
    *,
    image_id: str,
    ridge_channel: np.ndarray | None = None,
    epsilon: float = 1e-6,
) -> tuple[PairFeature, PairMasks]:
    """Measure peak-saddle adhesion from continuous grayscale intensity."""
    values = np.asarray(image, dtype=float)
    left = spots[pair.spot_i]
    right = spots[pair.spot_j]
    invalid_reasons: list[str] = []
    if left.edge_or_crop_flag or right.edge_or_crop_flag:
        invalid_reasons.append("edge_or_crop_spot")
    if pair.spacing_over_width < 2.5:
        invalid_reasons.append("spacing_too_small")
    if pair.spacing_over_width > 12.0:
        invalid_reasons.append("spacing_too_large")
    masks = pair_masks(values.shape, left, right)
    if int(masks.seed_i_mask.sum()) < 2 or int(masks.seed_j_mask.sum()) < 2:
        invalid_reasons.append("missing_seed_pixels")
    if int(masks.bridge_body_mask.sum()) < 4:
        invalid_reasons.append("empty_bridge_corridor")
    background, background_method = _robust_background(values, masks)
    peak_i = _robust_peak(values, masks.seed_i_mask)
    peak_j = _robust_peak(values, masks.seed_j_mask)
    corrected = values - background
    saddle = maximum_bottleneck_saddle(corrected, masks.seed_i_mask, masks.seed_j_mask, masks.corridor_mask)
    if not saddle.connected:
        invalid_reasons.append("corridor_not_connected")
    denominator = min(peak_i - background, peak_j - background)
    if not math.isfinite(denominator) or denominator <= epsilon:
        invalid_reasons.append("nonpositive_peak_denominator")
    if background_method == "no_background_pixels":
        invalid_reasons.append("insufficient_background_pixels")
    if invalid_reasons:
        adhesion_unclipped = float("nan")
        adhesion = float("nan")
        isolation = float("nan")
        direct_ratio = float("nan")
        mean_ratio = float("nan")
        ridge_ratio = float("nan")
        width_ratio = float("nan")
        confidence = 0.0
        valid = 0
    else:
        adhesion_unclipped = float(saddle.saddle_intensity / (denominator + epsilon))
        adhesion = float(np.clip(adhesion_unclipped, 0.0, 1.0))
        isolation = float(1.0 - adhesion)
        bridge_values = corrected[masks.bridge_body_mask]
        direct_ratio = float(np.percentile(bridge_values, 10.0) / (denominator + epsilon)) if bridge_values.size else float("nan")
        mean_ratio = float(np.mean(bridge_values) / (denominator + epsilon)) if bridge_values.size else float("nan")
        if ridge_channel is not None and np.any(masks.bridge_body_mask):
            ridge_values = np.asarray(ridge_channel, dtype=float)[masks.bridge_body_mask]
            ridge_ratio = float(np.mean(np.maximum(ridge_values, 0.0)) / (np.mean(np.maximum(corrected[masks.seed_i_mask | masks.seed_j_mask], 0.0)) + epsilon))
        else:
            ridge_ratio = float("nan")
        threshold = background + 0.5 * max(saddle.saddle_intensity, 0.0)
        width_ratio = float(np.mean(values[masks.bridge_body_mask] >= threshold)) if np.any(masks.bridge_body_mask) else float("nan")
        confidence = float(
            np.clip(
                pair.pair_selection_confidence
                * min(left.detection_confidence, right.detection_confidence)
                * (1.0 - 0.15 * max(left.saturation_flag, right.saturation_flag)),
                0.0,
                1.0,
            )
        )
        valid = 1
    feature = PairFeature(
        image_id=image_id,
        pair_id=pair.pair_id,
        spot_i=pair.spot_i,
        spot_j=pair.spot_j,
        row_label=pair.row_label,
        peak_i=float(peak_i),
        peak_j=float(peak_j),
        saddle_intensity=float(saddle.saddle_intensity),
        background_intensity=float(background),
        background_method=background_method,
        raw_peak_saddle_adhesion=float(adhesion),
        raw_peak_saddle_adhesion_unclipped=float(adhesion_unclipped),
        isolation_persistence=float(isolation),
        direct_corridor_valley_ratio=float(direct_ratio),
        corridor_mean_ratio=float(mean_ratio),
        ridge_energy_ratio=float(ridge_ratio),
        bridge_width_ratio=float(width_ratio),
        spot_spacing_over_width=float(pair.spacing_over_width),
        pair_measurement_confidence=float(confidence),
        valid=valid,
        invalid_reason=";".join(invalid_reasons),
    )
    return feature, masks
