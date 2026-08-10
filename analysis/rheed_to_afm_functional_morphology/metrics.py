from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure

FSMI_COMPONENTS = [
    "sq_nm",
    "height_increment_rms_31nm",
    "curvature_relief_31nm",
    "bearing_core_equivalent_nm",
    "island_prominence_nm",
]


def _interior_regions(
    mask: np.ndarray,
    intensity: np.ndarray,
    *,
    minimum_area: int = 5,
) -> list[measure._regionprops.RegionProperties]:
    labels = measure.label(mask, connectivity=2)
    height, width = mask.shape
    return [
        region
        for region in measure.regionprops(labels, intensity_image=intensity)
        if region.area >= int(minimum_area)
        and not (
            region.bbox[0] == 0
            or region.bbox[1] == 0
            or region.bbox[2] == height
            or region.bbox[3] == width
        )
    ]


def extract_surface_metrics(
    array: np.ndarray,
    *,
    scan_size_nm: float = 1000.0,
    analysis_scale_nm: float = 31.25,
) -> dict[str, float]:
    """Extract bandwidth-explicit height, slope, bearing and island metrics.

    ``functional_surface_morphology_index_nm`` (FSMI) is the root-mean-square
    of five height-equivalent terms.  It is an experimental research
    descriptor, not an ISO or SEMI standardized parameter.  Its ingredients
    are chosen to retain the information that a scalar Sq/Rq discards:

    * areal RMS height, Sq;
    * RMS height change over a declared 31.25 nm lateral separation;
    * a second-difference curvature relief at the same separation;
    * one quarter of the 10--90% material-ratio/bearing height span; and
    * median q70 island prominence.

    All terms have units of nanometres, so no cohort-fitted scaling or
    target-dependent weights are required.
    """

    height = np.asarray(array, dtype=np.float64)
    if height.ndim != 2 or min(height.shape) < 16:
        raise ValueError("surface metric input must be a 2D AFM height map")
    height = height - float(np.mean(height))
    pixel_nm = float(scan_size_nm) / max(height.shape[0] - 1, 1)
    step = int(
        np.clip(
            np.round(float(analysis_scale_nm) / pixel_nm),
            1,
            min(height.shape) // 5,
        )
    )
    realized_scale_nm = float(step * pixel_nm)

    dx = height[:, step:] - height[:, :-step]
    dy = height[step:, :] - height[:-step, :]
    increments = np.concatenate([dx.ravel(), dy.ravel()])
    increment_rms = float(np.sqrt(np.mean(np.square(increments))))

    d2x = height[:, 2 * step :] - 2.0 * height[:, step:-step] + height[:, : -2 * step]
    d2y = height[2 * step :, :] - 2.0 * height[step:-step, :] + height[: -2 * step, :]
    second_differences = np.concatenate([d2x.ravel(), d2y.ravel()])
    curvature_relief = float(0.5 * np.sqrt(np.mean(np.square(second_differences))))

    height[step:-step, step:-step]
    gx = (height[step:-step, 2 * step :] - height[step:-step, : -2 * step]) / (
        2.0 * realized_scale_nm
    )
    gy = (height[2 * step :, step:-step] - height[: -2 * step, step:-step]) / (
        2.0 * realized_scale_nm
    )
    gradient = np.hypot(gx, gy)
    sdq = float(np.sqrt(np.mean(np.square(gradient))))
    sdr_percent = float(100.0 * np.mean(np.sqrt(1.0 + np.square(gradient)) - 1.0))

    q01, q10, q50, q70, q90, q99 = np.quantile(
        height, [0.01, 0.10, 0.50, 0.70, 0.90, 0.99]
    )
    bearing_core_span = float(q90 - q10)
    bearing_peak_span = float(q99 - q90)
    bearing_valley_span = float(q10 - q01)
    bearing_equivalent = float(0.25 * bearing_core_span)

    smooth = ndimage.gaussian_filter(height, sigma=0.8, mode="reflect")
    island_mask = smooth >= float(np.quantile(smooth, 0.70))
    regions = _interior_regions(island_mask, smooth)
    prominences = np.asarray(
        [max(float(region.intensity_mean - q50), 0.0) for region in regions],
        dtype=float,
    )
    areas_nm2 = np.asarray(
        [float(region.area) * pixel_nm**2 for region in regions],
        dtype=float,
    )
    island_prominence = float(np.median(prominences) if len(prominences) else 0.0)
    island_area = float(np.median(areas_nm2) if len(areas_nm2) else 0.0)

    boundary = island_mask ^ ndimage.binary_erosion(island_mask)
    pixel_gradient = np.hypot(*np.gradient(smooth))
    boundary_contrast = float(
        np.mean(pixel_gradient[boundary]) / max(float(np.mean(pixel_gradient)), 1e-12)
    )

    sq = float(np.sqrt(np.mean(np.square(height))))
    sa = float(np.mean(np.abs(height)))
    terms = np.asarray(
        [
            sq,
            increment_rms,
            curvature_relief,
            bearing_equivalent,
            island_prominence,
        ],
        dtype=float,
    )
    fsmi = float(np.sqrt(np.mean(np.square(terms))))
    return {
        "sq_nm": sq,
        "sa_nm": sa,
        "analysis_scale_nm": realized_scale_nm,
        "height_increment_rms_31nm": increment_rms,
        "scale_dependent_rms_slope_31nm": (
            increment_rms / max(realized_scale_nm, 1e-12)
        ),
        "curvature_relief_31nm": curvature_relief,
        "scale_dependent_rms_curvature_31nm_inv": (
            2.0 * curvature_relief / max(realized_scale_nm**2, 1e-12)
        ),
        "sdq_31nm": sdq,
        "sdr_31nm_percent": sdr_percent,
        "bearing_core_span_p90_p10_nm": bearing_core_span,
        "bearing_peak_span_p99_p90_nm": bearing_peak_span,
        "bearing_valley_span_p10_p01_nm": bearing_valley_span,
        "bearing_core_equivalent_nm": bearing_equivalent,
        "island_prominence_nm": island_prominence,
        "island_median_area_nm2": island_area,
        "island_count_per_um2": float(len(regions)),
        "island_boundary_contrast": boundary_contrast,
        "functional_surface_morphology_index_nm": fsmi,
    }


def scan_metric_table(
    rows: pd.DataFrame,
    *,
    scan_size_nm: float = 1000.0,
    analysis_scale_nm: float = 31.25,
) -> pd.DataFrame:
    records: list[dict[str, float | str]] = []
    for _, row in rows.iterrows():
        metrics = extract_surface_metrics(
            np.load(Path(str(row["plane_corrected_array_path"]))),
            scan_size_nm=scan_size_nm,
            analysis_scale_nm=analysis_scale_nm,
        )
        records.append(
            {
                "sample_id": str(row["sample_id"]),
                "growth_run_id": str(row["growth_run_id"]),
                "split": str(row["split"]),
                **metrics,
            }
        )
    return pd.DataFrame(records)


def group_metric_table(scan_metrics: pd.DataFrame) -> pd.DataFrame:
    result = (
        scan_metrics.groupby(["split", "growth_run_id"])
        .median(numeric_only=True)
        .reset_index()
    )
    result["growth_run_id"] = result["growth_run_id"].astype(str)
    return result.sort_values(["split", "growth_run_id"]).reset_index(drop=True)
