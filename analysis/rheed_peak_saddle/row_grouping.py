"""Robust row grouping and adjacent-pair formation for RHEED spot centers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from analysis.rheed_peak_saddle.spot_detection import SpotEstimate


@dataclass(frozen=True)
class RowGroupingResult:
    dominant_angle_degrees: float
    row_labels: tuple[int, ...]
    row_consistency: float
    rotated_x: tuple[float, ...]
    rotated_y: tuple[float, ...]


@dataclass(frozen=True)
class AdjacentPairCandidate:
    pair_id: str
    spot_i: int
    spot_j: int
    row_label: int
    spacing: float
    spacing_over_width: float
    pair_selection_confidence: float


@dataclass(frozen=True)
class LatticeSpotAssignment:
    detection_index: int
    row_label: int
    lattice_index: int
    local_u: float
    local_v: float
    row_spacing: float
    lattice_fit_residual: float
    lattice_assignment_confidence: float
    duplicate_candidate: int = 0


@dataclass(frozen=True)
class LatticeRowResult:
    assignments: tuple[LatticeSpotAssignment, ...]
    row_spacing_by_label: dict[int, float]
    missing_site_count_by_label: dict[int, int]


def estimate_dominant_row_angle(spots: Sequence[SpotEstimate]) -> float:
    """Estimate the near-horizontal row angle from pairwise displacement voting."""
    if len(spots) < 2:
        return 0.0
    angles: list[float] = []
    weights: list[float] = []
    for i, left in enumerate(spots):
        for right in spots[i + 1 :]:
            dx = right.center_x - left.center_x
            dy = right.center_y - left.center_y
            dist = math.hypot(dx, dy)
            if dist < 12.0 or dist > 95.0:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            while angle <= -90.0:
                angle += 180.0
            while angle > 90.0:
                angle -= 180.0
            if abs(angle) <= 28.0:
                angles.append(angle)
                weights.append(1.0 / max(abs(dy) + 1.0, 1.0))
    if not angles:
        return 0.0
    order = np.argsort(angles)
    sorted_angles = np.asarray(angles, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    cdf = np.cumsum(sorted_weights) / max(float(np.sum(sorted_weights)), 1e-8)
    return float(sorted_angles[int(np.searchsorted(cdf, 0.5))])


def group_spot_rows(spots: Sequence[SpotEstimate]) -> RowGroupingResult:
    angle = estimate_dominant_row_angle(spots)
    theta = math.radians(angle)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    xs = np.asarray([spot.center_x for spot in spots], dtype=float)
    ys = np.asarray([spot.center_y for spot in spots], dtype=float)
    rx = cos_t * xs + sin_t * ys
    ry = -sin_t * xs + cos_t * ys
    if len(spots) == 0:
        return RowGroupingResult(angle, tuple(), 0.0, tuple(), tuple())
    widths = np.asarray([spot.equivalent_width for spot in spots], dtype=float)
    row_tolerance = max(8.0, 2.4 * float(np.median(widths)) if widths.size else 8.0)
    order = np.argsort(ry)
    labels = np.full(len(spots), -1, dtype=int)
    current_label = 0
    current_values: list[float] = []
    last_center = float(ry[order[0]])
    for index in order:
        value = float(ry[index])
        if current_values and abs(value - last_center) > row_tolerance:
            current_label += 1
            current_values = []
        labels[index] = current_label
        current_values.append(value)
        last_center = float(np.median(current_values))
    residuals = []
    for label in sorted(set(labels.tolist())):
        row_values = ry[labels == label]
        if row_values.size:
            residuals.extend(np.abs(row_values - np.median(row_values)).tolist())
    consistency = float(np.exp(-np.median(residuals) / max(row_tolerance, 1e-6))) if residuals else 1.0
    return RowGroupingResult(
        dominant_angle_degrees=angle,
        row_labels=tuple(int(label) for label in labels),
        row_consistency=consistency,
        rotated_x=tuple(float(value) for value in rx),
        rotated_y=tuple(float(value) for value in ry),
    )


def form_adjacent_pairs(
    spots: Sequence[SpotEstimate],
    grouping: RowGroupingResult,
    *,
    image_id: str = "image",
) -> tuple[AdjacentPairCandidate, ...]:
    """Form only geometrically adjacent neighbors within each row."""
    if len(spots) != len(grouping.row_labels):
        raise ValueError("spots and row labels must have the same length")
    pairs: list[AdjacentPairCandidate] = []
    labels = np.asarray(grouping.row_labels, dtype=int)
    rx = np.asarray(grouping.rotated_x, dtype=float)
    widths = np.asarray([spot.equivalent_width for spot in spots], dtype=float)
    row_spacings: list[float] = []
    row_pairs: list[tuple[int, int, int, float]] = []
    for row_label in sorted(set(labels.tolist())):
        indices = [int(i) for i in np.where(labels == row_label)[0]]
        indices.sort(key=lambda index: rx[index])
        for left, right in zip(indices[:-1], indices[1:]):
            spacing = abs(float(rx[right] - rx[left]))
            if spacing <= 1e-6:
                continue
            row_spacings.append(spacing)
            row_pairs.append((row_label, left, right, spacing))
    if not row_pairs:
        return tuple()
    median_spacing = float(np.median(row_spacings))
    for row_label, left, right, spacing in row_pairs:
        width = max(float((widths[left] + widths[right]) / 2.0), 1e-6)
        spacing_over_width = spacing / width
        plausible = 2.5 <= spacing_over_width <= 9.5 and spacing <= 1.65 * median_spacing
        if not plausible:
            continue
        spacing_score = math.exp(-abs(spacing - median_spacing) / max(median_spacing, 1e-6))
        confidence = float(np.clip(0.55 + 0.45 * spacing_score, 0.0, 1.0))
        pairs.append(
            AdjacentPairCandidate(
                pair_id=f"{image_id}_p{len(pairs):03d}",
                spot_i=left,
                spot_j=right,
                row_label=int(row_label),
                spacing=float(spacing),
                spacing_over_width=float(spacing_over_width),
                pair_selection_confidence=confidence,
            )
        )
    return tuple(pairs)


def estimate_nominal_lattice_spacing(spots: Sequence[SpotEstimate], grouping: RowGroupingResult) -> float:
    """Estimate the fundamental row lattice spacing, preferring the smallest strong period."""
    if len(spots) < 2:
        return float("nan")
    labels = np.asarray(grouping.row_labels, dtype=int)
    rx = np.asarray(grouping.rotated_x, dtype=float)
    widths = np.asarray([spot.equivalent_width for spot in spots], dtype=float)
    median_width = float(np.median(widths)) if widths.size else 5.0
    diffs: list[float] = []
    for row_label in sorted(set(labels.tolist())):
        indices = [int(i) for i in np.where(labels == row_label)[0]]
        if len(indices) < 2:
            continue
        indices.sort(key=lambda index: rx[index])
        nearest = [float(abs(rx[right] - rx[left])) for left, right in zip(indices[:-1], indices[1:])]
        for diff in nearest:
            if 5.2 * median_width <= diff <= 12.0 * median_width:
                diffs.append(diff)
        for i, left in enumerate(indices):
            for right in indices[i + 1 :]:
                diff = float(abs(rx[right] - rx[left]))
                if 5.2 * median_width <= diff <= 12.0 * median_width:
                    diffs.append(diff)
    if not diffs:
        return float("nan")
    values = np.asarray(sorted(diffs), dtype=float)
    candidates = np.unique(np.round(values, 1))
    best_candidate = float(np.median(values))
    best_score = -float("inf")
    for candidate in candidates:
        if candidate <= 1e-6:
            continue
        score = 0.0
        for diff in values:
            multiple = max(1, int(round(diff / candidate)))
            residual = abs(diff - multiple * candidate)
            score += math.exp(-residual / max(0.12 * candidate, 1e-6))
        if candidate < 6.0 * median_width:
            score *= 0.55
        if score > best_score:
            best_score = score
            best_candidate = float(candidate)
    return best_candidate


def assign_lattice_indices(
    spots: Sequence[SpotEstimate],
    grouping: RowGroupingResult,
    *,
    nominal_spacing: float | None = None,
) -> LatticeRowResult:
    """Assign integer site indices along each row while allowing missing sites."""
    if len(spots) != len(grouping.row_labels):
        raise ValueError("spots and row labels must have the same length")
    spacing = float(nominal_spacing) if nominal_spacing and math.isfinite(float(nominal_spacing)) else estimate_nominal_lattice_spacing(spots, grouping)
    if not math.isfinite(spacing) or spacing <= 1e-6:
        spacing = 30.0
    labels = np.asarray(grouping.row_labels, dtype=int)
    rx = np.asarray(grouping.rotated_x, dtype=float)
    ry = np.asarray(grouping.rotated_y, dtype=float)
    assignments: list[LatticeSpotAssignment] = []
    row_spacing_by_label: dict[int, float] = {}
    missing_site_count_by_label: dict[int, int] = {}
    for row_label in sorted(set(labels.tolist())):
        indices = [int(i) for i in np.where(labels == row_label)[0]]
        if not indices:
            continue
        indices.sort(key=lambda index: rx[index])
        row_diffs = [float(abs(rx[right] - rx[left])) for left, right in zip(indices[:-1], indices[1:])]
        plausible = [diff for diff in row_diffs if 0.55 * spacing <= diff <= 1.45 * spacing]
        row_spacing = float(np.median(plausible)) if plausible else spacing
        row_spacing_by_label[int(row_label)] = row_spacing
        lattice_indices: dict[int, int] = {indices[0]: 0}
        current_index = 0
        for left, right in zip(indices[:-1], indices[1:]):
            diff = float(rx[right] - rx[left])
            step = max(1, int(round(abs(diff) / max(row_spacing, 1e-6))))
            if abs(diff) < 0.55 * row_spacing:
                step = 0
            current_index += step
            lattice_indices[right] = current_index
        missing_site_count_by_label[int(row_label)] = max(0, (max(lattice_indices.values()) - min(lattice_indices.values()) + 1) - len(set(lattice_indices.values())))
        seen: dict[int, list[int]] = {}
        for index, lattice_index in lattice_indices.items():
            seen.setdefault(lattice_index, []).append(index)
        origin_candidates = [rx[index] - lattice_indices[index] * row_spacing for index in indices]
        origin = float(np.median(origin_candidates)) if origin_candidates else float(rx[indices[0]])
        keep_for_index: dict[int, int] = {}
        for lattice_index, det_indices in seen.items():
            keep_for_index[lattice_index] = min(
                det_indices,
                key=lambda idx: (abs(rx[idx] - (origin + lattice_index * row_spacing)), -spots[idx].detection_confidence),
            )
        for index in indices:
            lattice_index = lattice_indices[index]
            fitted_u = origin + lattice_index * row_spacing
            residual = float(abs(rx[index] - fitted_u))
            duplicate = int(keep_for_index.get(lattice_index) != index)
            confidence = float(np.clip(math.exp(-residual / max(0.35 * row_spacing, 1e-6)) * (1.0 - 0.6 * duplicate), 0.0, 1.0))
            assignments.append(
                LatticeSpotAssignment(
                    detection_index=index,
                    row_label=int(row_label),
                    lattice_index=int(lattice_index),
                    local_u=float(rx[index]),
                    local_v=float(ry[index]),
                    row_spacing=float(row_spacing),
                    lattice_fit_residual=residual,
                    lattice_assignment_confidence=confidence,
                    duplicate_candidate=duplicate,
                )
            )
    return LatticeRowResult(tuple(assignments), row_spacing_by_label, missing_site_count_by_label)


def form_lattice_adjacent_pairs(
    spots: Sequence[SpotEstimate],
    grouping: RowGroupingResult,
    lattice: LatticeRowResult,
    *,
    image_id: str = "image",
    min_confidence: float = 0.45,
) -> tuple[AdjacentPairCandidate, ...]:
    assignment_by_detection = {assignment.detection_index: assignment for assignment in lattice.assignments}
    pairs: list[AdjacentPairCandidate] = []
    labels = sorted(set(assignment.row_label for assignment in lattice.assignments))
    for row_label in labels:
        row_assignments = [assignment for assignment in lattice.assignments if assignment.row_label == row_label and not assignment.duplicate_candidate]
        row_assignments.sort(key=lambda assignment: assignment.lattice_index)
        for left, right in zip(row_assignments[:-1], row_assignments[1:]):
            if right.lattice_index - left.lattice_index != 1:
                continue
            if left.lattice_assignment_confidence < min_confidence or right.lattice_assignment_confidence < min_confidence:
                continue
            spacing = abs(right.local_u - left.local_u)
            row_spacing = max((left.row_spacing + right.row_spacing) / 2.0, 1e-6)
            if not (0.55 * row_spacing <= spacing <= 1.45 * row_spacing):
                continue
            width = max((spots[left.detection_index].equivalent_width + spots[right.detection_index].equivalent_width) / 2.0, 1e-6)
            confidence = float(
                np.clip(
                    min(left.lattice_assignment_confidence, right.lattice_assignment_confidence)
                    * math.exp(-abs(spacing - row_spacing) / max(row_spacing, 1e-6)),
                    0.0,
                    1.0,
                )
            )
            pairs.append(
                AdjacentPairCandidate(
                    pair_id=f"{image_id}_lp{len(pairs):03d}",
                    spot_i=left.detection_index,
                    spot_j=right.detection_index,
                    row_label=int(row_label),
                    spacing=float(spacing),
                    spacing_over_width=float(spacing / width),
                    pair_selection_confidence=confidence,
                )
            )
    return tuple(pairs)
