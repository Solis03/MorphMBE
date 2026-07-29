"""Full-lattice RHEED ROI prediction.

The original automatic ROI maximizes activity inside a fixed-aspect-ratio
rectangle.  That is useful for tracking but can crop a sparse diffraction
family at the right, top or bottom.  This module instead predicts four
independent boundaries relative to the detected eyepiece/screen geometry,
then moves the left boundary inside the circular aperture arc.

Calibration bundles contain only group-level boundary statistics fitted from
training videos.  They never contain raw frames or held-video annotations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

from rheed2morph.rheed.automatic_roi_keyframe import (
    ApertureAnalysis,
    ROIPrediction,
    Rect,
)


BOUNDARY_NAMES = ("left", "right", "top", "bottom")


def aperture_bounds(
    analysis: ApertureAnalysis,
) -> tuple[int, int, int, int]:
    """Return the detected aperture bounding box at analysis resolution."""

    ys, xs = np.where(analysis.aperture_mask)
    if not len(xs):
        raise RuntimeError("Aperture mask is empty")
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def normalized_roi_bounds(
    rect: Rect,
    analysis: ApertureAnalysis,
) -> dict[str, float]:
    """Express a source-space rectangle relative to the aperture box."""

    x0, y0, x1, y1 = aperture_bounds(analysis)
    inverse = 1.0 / analysis.scale
    aperture_x0 = x0 * inverse
    aperture_y0 = y0 * inverse
    aperture_width = max((x1 - x0) * inverse, 1e-8)
    aperture_height = max((y1 - y0) * inverse, 1e-8)
    return {
        "left": (rect.x - aperture_x0) / aperture_width,
        "right": (rect.x2 - aperture_x0) / aperture_width,
        "top": (rect.y - aperture_y0) / aperture_height,
        "bottom": (rect.y2 - aperture_y0) / aperture_height,
    }


def orientation_group(analysis: ApertureAnalysis) -> str:
    return (
        "portrait"
        if analysis.source_height > analysis.source_width
        else "landscape"
    )


def _load_bundle(
    bundle_or_path: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    if isinstance(bundle_or_path, Mapping):
        return dict(bundle_or_path)
    return joblib.load(Path(bundle_or_path))


def _safe_fraction(
    mask: np.ndarray,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> float:
    crop = mask[y0:y1, x0:x1]
    return float(crop.mean()) if crop.size else 0.0


def circular_arc_intrusion_fraction(
    analysis: ApertureAnalysis,
    rect: Rect,
) -> float:
    """Fraction of ROI rows where the circular aperture enters from the left.

    The intended vertical screen/shadow transition on the right is allowed.
    A row counts as an eyepiece-arc intrusion only when the ROI starts outside
    the aperture and enters the illuminated aperture later in the same row.
    """

    scale = analysis.scale
    height, width = analysis.aperture_mask.shape
    x0 = int(np.clip(np.floor(rect.x * scale), 0, width - 1))
    x1 = int(np.clip(np.ceil(rect.x2 * scale), x0 + 1, width))
    y0 = int(np.clip(np.floor(rect.y * scale), 0, height - 1))
    y1 = int(np.clip(np.ceil(rect.y2 * scale), y0 + 1, height))
    crop = analysis.aperture_mask[y0:y1, x0:x1]
    if not crop.size:
        return 1.0
    return float(np.mean((~crop[:, 0]) & crop.any(axis=1)))


def predict_full_lattice_roi(
    analysis: ApertureAnalysis,
    bundle_or_path: Mapping[str, Any] | str | Path,
) -> ROIPrediction:
    """Predict a four-boundary ROI that contains the complete point family."""

    bundle = _load_bundle(bundle_or_path)
    if int(bundle.get("schema_version", -1)) != 1:
        raise ValueError("Expected a schema-version 1 lattice ROI bundle")
    group = orientation_group(analysis)
    groups = bundle["groups"]
    calibration = groups.get(group, groups["global"])
    for name in BOUNDARY_NAMES:
        if name not in calibration:
            raise KeyError(f"Missing calibrated ROI boundary {name!r}")

    aperture_x0, aperture_y0, aperture_x1, aperture_y1 = aperture_bounds(
        analysis
    )
    aperture_width = aperture_x1 - aperture_x0
    aperture_height = aperture_y1 - aperture_y0
    left = aperture_x0 + float(calibration["left"]) * aperture_width
    right = aperture_x0 + float(calibration["right"]) * aperture_width
    top = aperture_y0 + float(calibration["top"]) * aperture_height
    bottom = aperture_y0 + float(calibration["bottom"]) * aperture_height
    left -= float(
        bundle.get("left_padding_aperture_fraction", 0.0)
    ) * aperture_width
    right += float(
        bundle.get("right_padding_aperture_fraction", 0.0)
    ) * aperture_width
    top -= float(
        bundle.get("top_padding_aperture_fraction", 0.0)
    ) * aperture_height
    bottom += float(
        bundle.get("bottom_padding_aperture_fraction", 0.0)
    ) * aperture_height

    mask_height, mask_width = analysis.aperture_mask.shape
    top_index = int(np.clip(np.floor(top), 0, mask_height - 1))
    bottom_index = int(
        np.clip(np.ceil(bottom), top_index + 1, mask_height)
    )

    # The circular eyepiece edge is the first illuminated pixel in each row.
    # Keeping the ROI left of the spots but to the right of the maximum arc
    # location prevents the characteristic black circular wedge.
    first_inside = []
    last_inside = []
    for row in range(top_index, bottom_index):
        columns = np.flatnonzero(analysis.aperture_mask[row])
        if len(columns):
            first_inside.append(int(columns[0]))
            last_inside.append(int(columns[-1]) + 1)
    if not first_inside:
        raise RuntimeError("No aperture rows overlap the proposed lattice ROI")
    arc_margin = float(bundle.get("arc_margin_analysis_pixels", 2.0))
    left = max(left, max(first_inside) + arc_margin)

    # Always include the stable right light/shadow transition.  The learned
    # right-side padding may extend slightly past it, which is intentional.
    right_boundary = float(np.quantile(last_inside, 0.90))
    right = max(
        right,
        right_boundary
        + float(bundle.get("right_boundary_margin_analysis_pixels", 1.0)),
    )

    minimum_width = float(bundle.get("minimum_width_fraction", 0.40))
    if right - left < minimum_width * aperture_width:
        left = right - minimum_width * aperture_width
        left = max(left, max(first_inside) + arc_margin)

    x0 = int(np.clip(np.floor(left), 0, mask_width - 1))
    x1 = int(np.clip(np.ceil(right), x0 + 1, mask_width))
    y0 = top_index
    y1 = bottom_index
    inverse = 1.0 / analysis.scale
    rect = Rect(
        x=int(round(x0 * inverse)),
        y=int(round(y0 * inverse)),
        width=max(1, int(round((x1 - x0) * inverse))),
        height=max(1, int(round((y1 - y0) * inverse))),
        source_width=analysis.source_width,
        source_height=analysis.source_height,
    ).clipped()

    safe_fraction = _safe_fraction(
        analysis.safe_mask,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
    )
    activity_total = max(float(analysis.activity.sum()), 1e-8)
    activity_coverage = float(
        analysis.activity[y0:y1, x0:x1].sum() / activity_total
    )
    arc_intrusion = circular_arc_intrusion_fraction(analysis, rect)
    aperture_fraction = float(analysis.aperture_mask.mean())
    confidence = float(
        np.clip(
            0.45 * (1.0 - arc_intrusion)
            + 0.25 * min(activity_coverage / 0.75, 1.0)
            + 0.15 * min(safe_fraction / 0.75, 1.0)
            + 0.15 * min(aperture_fraction / 0.20, 1.0),
            0.0,
            1.0,
        )
    )
    return ROIPrediction(
        method="full_lattice",
        rect=rect,
        aperture_area_fraction=aperture_fraction,
        safe_pixel_fraction=safe_fraction,
        activity_coverage=activity_coverage,
        confidence=confidence,
        analysis_scale=analysis.scale,
        circular_edge_intrusion_fraction=arc_intrusion,
    )
