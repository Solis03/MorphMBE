"""Target-independent RHEED connectivity and isolation features."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import numpy as np
from scipy import ndimage, spatial
from skimage import exposure, feature, filters, measure, morphology, segmentation

from analysis.rheed_roughness.run import display_path, safe_float
from analysis.rheed_roughness.run import compute_morphology_scores
from analysis.rheed_single_frame.data import ExperimentPaths, write_csv_rows
from analysis.rheed_single_frame.preprocessing import PreprocessedImage
from analysis.rheed_single_frame.removelist import RemovelistAudit, assert_no_removed_samples
from rheed2morph.rheed.shape_preprocessing import preprocess_frame_for_shape
from rheed2morph.rheed.spot_streak_geometry import extract_components_and_frame_features


matplotlib.use("Agg")
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore", category=FutureWarning, module=r".*skimage.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


PHYSICAL_INTERPRETABLE_FEATURES = [
    "horizontal_connectivity_score",
    "horizontal_closing_gain",
    "horizontal_run_length",
    "isolation_score",
    "isolated_component_fraction",
    "horizontal_neighbor_fraction",
    "largest_component_fraction",
]

FEATURE_GROUP_PREFIXES = {
    "physics": (
        "horizontal_",
        "isolation_",
        "isolated_",
        "component_",
        "blob_",
        "compact_",
        "largest_component",
        "nearest_",
        "vertical_",
        "graph_",
        "skeleton_",
        "closing_",
        "structure_",
        "fft_",
        "hog_",
        "edge_",
        "line_",
        "existing_",
        "morphology_index",
        "raw_spottiness",
        "raw_streakiness",
    ),
    "nuisance": (
        "mean_intensity",
        "median_intensity",
        "intensity_std",
        "dynamic_range",
        "saturation_fraction",
        "underexposure_fraction",
        "background_gradient",
        "sharpness",
        "image_",
        "valid_roi_fraction",
        "pattern_centroid",
        "bright_pixel_centroid",
    ),
}


@dataclass(frozen=True)
class FeatureResult:
    sample_id: str
    features: dict[str, Any]
    threshold_rows: tuple[dict[str, Any], ...]
    component_rows: tuple[dict[str, Any], ...]
    overlay_path: Path
    closing_path: Path
    skeleton_path: Path


def _finite_stats(values: Sequence[float], prefix: str) -> dict[str, float]:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_q90": 0.0,
            f"{prefix}_std": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_q90": float(np.percentile(arr, 90)),
        f"{prefix}_std": float(np.std(arr)),
    }


def horizontal_run_lengths(mask: np.ndarray) -> list[int]:
    runs: list[int] = []
    for row in np.asarray(mask, dtype=bool):
        padded = np.concatenate([[False], row, [False]])
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        stops = np.flatnonzero(padded[:-1] & ~padded[1:])
        runs.extend((stops - starts).astype(int).tolist())
    return runs


def _binary_thresholds(image: np.ndarray, valid_mask: np.ndarray, percentiles: Sequence[float], block_size: int) -> list[tuple[str, np.ndarray, float]]:
    valid_values = image[valid_mask]
    if valid_values.size == 0:
        return []
    thresholds: list[tuple[str, np.ndarray, float]] = []
    try:
        otsu = float(filters.threshold_otsu(valid_values))
    except Exception:
        otsu = float(np.percentile(valid_values, 85))
    thresholds.append(("otsu", image > otsu, otsu))
    for pct in percentiles:
        threshold = float(np.percentile(valid_values, pct))
        thresholds.append((f"p{int(pct)}", image > threshold, threshold))
    block = int(block_size)
    if block % 2 == 0:
        block += 1
    block = max(7, block)
    try:
        local = filters.threshold_local(image, block_size=block, method="gaussian")
        thresholds.append(("adaptive_local", image > local, float(np.nanmedian(local[valid_mask]))))
    except Exception:
        pass
    cleaned: list[tuple[str, np.ndarray, float]] = []
    for name, mask, threshold in thresholds:
        binary = np.asarray(mask & valid_mask, dtype=bool)
        binary = morphology.remove_small_objects(binary, max_size=3)
        binary = morphology.remove_small_holes(binary, max_size=3)
        cleaned.append((name, binary, threshold))
    return cleaned


def _component_table(mask: np.ndarray, image: np.ndarray, sample_id: str, threshold_name: str) -> list[dict[str, Any]]:
    labels = measure.label(mask, connectivity=2)
    rows: list[dict[str, Any]] = []
    for region in measure.regionprops(labels, intensity_image=image):
        if region.area < 4:
            continue
        minr, minc, maxr, maxc = region.bbox
        width = max(1, maxc - minc)
        height = max(1, maxr - minr)
        perimeter = max(float(region.perimeter), 1e-8)
        circularity = float(4.0 * math.pi * region.area / (perimeter * perimeter))
        intensity_mean = float(region.intensity_mean) if hasattr(region, "intensity_mean") else float(region.mean_intensity)
        rows.append(
            {
                "sample_id": sample_id,
                "threshold": threshold_name,
                "component_id": int(region.label),
                "area_px": float(region.area),
                "centroid_y": float(region.centroid[0]),
                "centroid_x": float(region.centroid[1]),
                "bbox_width_px": float(width),
                "bbox_height_px": float(height),
                "aspect_ratio": float(width / max(height, 1)),
                "equivalent_diameter_px": float(region.equivalent_diameter_area),
                "eccentricity": float(region.eccentricity),
                "solidity": float(region.solidity),
                "circularity": circularity,
                "mean_intensity": intensity_mean,
                "orientation_rad": float(region.orientation),
            }
        )
    return rows


def _largest_component_area(mask: np.ndarray) -> float:
    labels = measure.label(mask, connectivity=2)
    counts = np.bincount(labels.ravel())
    if counts.size <= 1:
        return 0.0
    return float(counts[1:].max())


def _graph_features(component_rows: Sequence[dict[str, Any]], shape: tuple[int, int]) -> dict[str, float]:
    height, width = shape
    comps = [row for row in component_rows if float(row.get("area_px", 0)) >= 4]
    n = len(comps)
    if n == 0:
        return {
            "component_count": 0.0,
            "blob_count": 0.0,
            "isolated_component_fraction": 0.0,
            "compact_component_fraction": 0.0,
            "horizontal_neighbor_fraction": 0.0,
            "horizontal_connectivity_graph_density": 0.0,
            "largest_graph_component_fraction": 0.0,
            "graph_component_count": 0.0,
            "nearest_neighbor_distance": 0.0,
            "horizontal_nearest_neighbor_gap": 0.0,
            "vertical_nearest_neighbor_gap": 0.0,
            "horizontal_gap_normalized_by_diameter": 0.0,
            "fraction_without_horizontally_aligned_neighbor": 0.0,
        }
    centroids = np.asarray([[row["centroid_x"], row["centroid_y"]] for row in comps], dtype=np.float64)
    diam = np.asarray([max(float(row.get("equivalent_diameter_px", 1.0)), 1.0) for row in comps], dtype=np.float64)
    compact = [
        float(row.get("circularity", 0.0)) > 0.45 and float(row.get("solidity", 0.0)) > 0.55 and float(row.get("aspect_ratio", 1.0)) < 1.6
        for row in comps
    ]
    if n == 1:
        return {
            "component_count": 1.0,
            "blob_count": 1.0,
            "isolated_component_fraction": 1.0,
            "compact_component_fraction": float(np.mean(compact)),
            "horizontal_neighbor_fraction": 0.0,
            "horizontal_connectivity_graph_density": 0.0,
            "largest_graph_component_fraction": 1.0,
            "graph_component_count": 1.0,
            "nearest_neighbor_distance": 0.0,
            "horizontal_nearest_neighbor_gap": float(width),
            "vertical_nearest_neighbor_gap": float(height),
            "horizontal_gap_normalized_by_diameter": float(width / max(float(np.median(diam)), 1.0)),
            "fraction_without_horizontally_aligned_neighbor": 1.0,
        }
    dist = spatial.distance.squareform(spatial.distance.pdist(centroids))
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    dx = np.abs(centroids[:, None, 0] - centroids[None, :, 0])
    dy = np.abs(centroids[:, None, 1] - centroids[None, :, 1])
    med_diam = max(float(np.median(diam)), 1.0)
    y_tol = max(0.045 * height, 1.5 * med_diam)
    x_max = max(0.22 * width, 6.0 * med_diam)
    x_min = 0.25 * med_diam
    horizontal_edges = (dy <= y_tol) & (dx >= x_min) & (dx <= x_max)
    np.fill_diagonal(horizontal_edges, False)
    horizontal_neighbor = horizontal_edges.any(axis=1)
    edge_count = int(np.triu(horizontal_edges, 1).sum())
    density = edge_count / max(n * (n - 1) / 2.0, 1.0)
    labels, graph_count = ndimage.label(np.eye(n, dtype=bool))  # placeholder for fallback shape
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in zip(*np.triu_indices(n, 1)):
        if horizontal_edges[i, j]:
            union(int(i), int(j))
    roots = [find(i) for i in range(n)]
    _, counts = np.unique(roots, return_counts=True)
    horizontal_gap = np.where(horizontal_edges, dx, np.inf)
    vertical_gap = np.where(horizontal_edges, dy, np.inf)
    finite_hgap = horizontal_gap[np.isfinite(horizontal_gap)]
    finite_vgap = vertical_gap[np.isfinite(vertical_gap)]
    return {
        "component_count": float(n),
        "blob_count": float(sum(compact)),
        "isolated_component_fraction": float(1.0 - np.mean(horizontal_neighbor)),
        "compact_component_fraction": float(np.mean(compact)),
        "horizontal_neighbor_fraction": float(np.mean(horizontal_neighbor)),
        "horizontal_connectivity_graph_density": float(density),
        "largest_graph_component_fraction": float(counts.max() / n),
        "graph_component_count": float(counts.size),
        "nearest_neighbor_distance": float(np.median(nearest[np.isfinite(nearest)])) if np.isfinite(nearest).any() else 0.0,
        "horizontal_nearest_neighbor_gap": float(np.median(finite_hgap)) if finite_hgap.size else float(width),
        "vertical_nearest_neighbor_gap": float(np.median(finite_vgap)) if finite_vgap.size else float(height),
        "horizontal_gap_normalized_by_diameter": float((np.median(finite_hgap) if finite_hgap.size else width) / med_diam),
        "fraction_without_horizontally_aligned_neighbor": float(1.0 - np.mean(horizontal_neighbor)),
    }


def _threshold_features(mask: np.ndarray, image: np.ndarray, valid_mask: np.ndarray, sample_id: str, threshold_name: str) -> tuple[dict[str, float], list[dict[str, Any]]]:
    height, width = mask.shape
    valid_area = max(float(valid_mask.sum()), 1.0)
    component_rows = _component_table(mask, image, sample_id, threshold_name)
    graph = _graph_features(component_rows, mask.shape)
    runs = horizontal_run_lengths(mask)
    run_stats = _finite_stats([run / max(width, 1) for run in runs], "horizontal_run_length")
    largest_before = _largest_component_area(mask)
    h_gains = []
    v_gains = []
    gaps_closed = []
    for length in (5, 11, 21):
        h_closed = morphology.closing(mask, footprint=morphology.footprint_rectangle((1, length)))
        v_closed = morphology.closing(mask, footprint=morphology.footprint_rectangle((length, 1)))
        h_largest = _largest_component_area(h_closed)
        v_largest = _largest_component_area(v_closed)
        h_gains.append((h_largest - largest_before) / valid_area)
        v_gains.append((v_largest - largest_before) / valid_area)
        gaps_closed.append(float(np.logical_and(h_closed, ~mask).sum() / valid_area))
    elongated = [
        row
        for row in component_rows
        if float(row.get("aspect_ratio", 1.0)) >= 1.7 and abs(float(row.get("orientation_rad", 0.0))) > math.radians(45)
    ]
    skeleton = morphology.skeletonize(mask)
    horizontal_links = np.logical_and(skeleton[:, 1:], skeleton[:, :-1]).sum()
    vertical_links = np.logical_and(skeleton[1:, :], skeleton[:-1, :]).sum()
    total_links = max(float(horizontal_links + vertical_links), 1.0)
    neighbors = ndimage.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant", cval=0) - skeleton.astype(np.uint8)
    branch_count = float(np.logical_and(skeleton, neighbors >= 3).sum())
    areas = [float(row["area_px"]) / valid_area for row in component_rows]
    diam = [float(row["equivalent_diameter_px"]) / max(width, height) for row in component_rows]
    ecc = [float(row["eccentricity"]) for row in component_rows]
    aspect = [float(row["aspect_ratio"]) for row in component_rows]
    solidity = [float(row["solidity"]) for row in component_rows]
    circ = [float(row["circularity"]) for row in component_rows]
    features = {
        "bright_pixel_fraction": float(mask.sum() / valid_area),
        "component_count_per_valid_area": graph["component_count"] / valid_area,
        "component_count": graph["component_count"],
        "blob_count": graph["blob_count"],
        "isolated_component_fraction": graph["isolated_component_fraction"],
        "compact_component_fraction": graph["compact_component_fraction"],
        "largest_component_fraction": largest_before / valid_area,
        "component_area_median": float(np.median(areas)) if areas else 0.0,
        "component_area_q90": float(np.percentile(areas, 90)) if areas else 0.0,
        "equivalent_diameter_median": float(np.median(diam)) if diam else 0.0,
        "eccentricity_median": float(np.median(ecc)) if ecc else 0.0,
        "aspect_ratio_median": float(np.median(aspect)) if aspect else 0.0,
        "solidity_median": float(np.median(solidity)) if solidity else 0.0,
        "circularity_median": float(np.median(circ)) if circ else 0.0,
        "horizontal_closing_gain": float(np.median(h_gains)),
        "vertical_closing_gain": float(np.median(v_gains)),
        "horizontal_to_vertical_closing_gain": float((np.median(h_gains) + 1e-8) / (np.median(v_gains) + 1e-8)),
        "fraction_bright_pixels_in_horizontal_components": float(sum(float(row["area_px"]) for row in elongated) / max(float(mask.sum()), 1.0)),
        "horizontal_skeleton_fraction": float(horizontal_links / total_links),
        "horizontal_branch_count": branch_count,
        "average_width_horizontal_structures": float(mask.sum() / max(float(skeleton.sum()), 1.0)),
        "gaps_closed_by_horizontal_structuring_elements": float(np.median(gaps_closed)),
        **run_stats,
        **graph,
    }
    return features, component_rows


def _texture_features(image: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    work = np.asarray(image, dtype=np.float32)
    gy, gx = np.gradient(work)
    tensor_xx = filters.gaussian(gx * gx, sigma=2)
    tensor_yy = filters.gaussian(gy * gy, sigma=2)
    tensor_xy = filters.gaussian(gx * gy, sigma=2)
    trace = tensor_xx + tensor_yy
    det_term = np.sqrt((tensor_xx - tensor_yy) ** 2 + 4 * tensor_xy * tensor_xy)
    anisotropy = det_term / np.maximum(trace, 1e-8)
    orientation = 0.5 * np.arctan2(2 * tensor_xy, tensor_xx - tensor_yy)
    fft = np.abs(np.fft.fftshift(np.fft.fft2(work - float(np.mean(work[valid_mask]))))) ** 2
    h, w = work.shape
    cy, cx = h // 2, w // 2
    row_band = fft[max(0, cy - 3) : min(h, cy + 4), :].sum()
    col_band = fft[:, max(0, cx - 3) : min(w, cx + 4)].sum()
    total_fft = max(float(fft.sum()), 1e-8)
    edges = feature.canny(work, sigma=1.0, mask=valid_mask)
    hog_vector = feature.hog(work, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(1, 1), feature_vector=True)
    hog_bins = np.reshape(hog_vector, (-1, 9)).mean(axis=0) if hog_vector.size else np.zeros(9)
    return {
        "structure_tensor_anisotropy": float(np.median(anisotropy[valid_mask])) if valid_mask.any() else 0.0,
        "dominant_orientation_deg": float(np.degrees(np.median(orientation[valid_mask]))) if valid_mask.any() else 0.0,
        "horizontal_gradient_energy": float(np.mean(gx[valid_mask] ** 2)) if valid_mask.any() else 0.0,
        "vertical_gradient_energy": float(np.mean(gy[valid_mask] ** 2)) if valid_mask.any() else 0.0,
        "horizontal_vs_vertical_gradient_energy": float((np.mean(gx[valid_mask] ** 2) + 1e-8) / (np.mean(gy[valid_mask] ** 2) + 1e-8))
        if valid_mask.any()
        else 0.0,
        "horizontal_vs_vertical_fft_power": float((row_band + 1e-8) / (col_band + 1e-8)),
        "fft_low_frequency_power": float(fft[h // 2 - h // 16 : h // 2 + h // 16 + 1, w // 2 - w // 16 : w // 2 + w // 16 + 1].sum() / total_fft),
        "fft_mid_high_frequency_power": float(1.0 - fft[h // 2 - h // 8 : h // 2 + h // 8 + 1, w // 2 - w // 8 : w // 2 + w // 8 + 1].sum() / total_fft),
        "edge_density": float(edges[valid_mask].mean()) if valid_mask.any() else 0.0,
        **{f"hog_orientation_bin_{idx}": float(value) for idx, value in enumerate(hog_bins)},
    }


def _nuisance_features(item: PreprocessedImage) -> dict[str, float]:
    img = item.gray_padded
    mask = item.valid_mask
    values = img[mask]
    yy, xx = np.indices(img.shape)
    weights = np.clip(item.normalized - np.percentile(item.normalized[mask], 70), 0.0, None) * mask
    total = float(weights.sum())
    if total > 1e-8:
        cx = float((xx * weights).sum() / total / max(img.shape[1] - 1, 1))
        cy = float((yy * weights).sum() / total / max(img.shape[0] - 1, 1))
    else:
        cx = cy = 0.5
    return {
        "mean_intensity": float(np.mean(values)) if values.size else 0.0,
        "median_intensity": float(np.median(values)) if values.size else 0.0,
        "intensity_std": float(np.std(values)) if values.size else 0.0,
        "dynamic_range": float(np.percentile(values, 99) - np.percentile(values, 1)) if values.size else 0.0,
        "saturation_fraction": float(np.mean(values >= 0.98)) if values.size else 0.0,
        "underexposure_fraction": float(np.mean(values <= 0.02)) if values.size else 0.0,
        "background_gradient": safe_float(item.audit_row.get("background_gradient")),
        "sharpness": safe_float(item.audit_row.get("sharpness")),
        "image_height": float(item.audit_row["original_height"]),
        "image_width": float(item.audit_row["original_width"]),
        "valid_roi_fraction": float(mask.mean()),
        "pattern_centroid_x": cx,
        "pattern_centroid_y": cy,
        "bright_pixel_centroid_x": cx,
        "bright_pixel_centroid_y": cy,
    }


def _aggregate_threshold_features(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    non_features = {"sample_id", "threshold_name", "threshold_value"}
    keys = sorted({key for row in rows for key in row if key not in non_features})
    out: dict[str, float] = {}
    for key in keys:
        values = np.asarray([safe_float(row.get(key), math.nan) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            out[key] = float(np.median(values))
            out[f"{key}_threshold_std"] = float(np.std(values))
        else:
            out[key] = 0.0
            out[f"{key}_threshold_std"] = 0.0
    return out


def _existing_morphology_features(item: PreprocessedImage) -> dict[str, float]:
    processed = preprocess_frame_for_shape(item.gray_padded, image_size=item.gray_padded.shape[0])
    _, geometry = extract_components_and_frame_features(
        soft_mask=processed.channels["soft_spot_streak_mask"],
        enhanced_image=processed.channels["log_bgsub"],
        artifact_mask=processed.artifact_mask,
    )
    scores = compute_morphology_scores(geometry)
    return {
        "existing_morphology_index": scores["morphology_index"],
        "morphology_index": scores["morphology_index"],
        "raw_spottiness": scores["raw_spottiness"],
        "raw_streakiness": scores["raw_streakiness"],
        "existing_detector_confidence": safe_float(geometry.get("mask_confidence")),
    }


def _composite_scores(features: dict[str, float]) -> dict[str, float]:
    h = np.mean(
        [
            np.clip(features.get("horizontal_neighbor_fraction", 0.0), 0, 1),
            np.clip(features.get("largest_graph_component_fraction", 0.0), 0, 1),
            np.clip(features.get("horizontal_skeleton_fraction", 0.0), 0, 1),
            np.clip(features.get("fraction_bright_pixels_in_horizontal_components", 0.0), 0, 1),
            np.clip(features.get("horizontal_closing_gain", 0.0) * 20.0, 0, 1),
            np.clip(features.get("horizontal_run_length_q90", 0.0) * 4.0, 0, 1),
        ]
    )
    compact = np.clip(features.get("compact_component_fraction", 0.0), 0, 1)
    isolated_compact = compact * np.clip(features.get("isolated_component_fraction", 0.0), 0, 1)
    fragmented = np.clip(features.get("graph_component_count", 0.0) / max(features.get("component_count", 1.0), 1.0), 0, 1)
    fragmented *= np.clip(features.get("component_count", 0.0) / 6.0, 0, 1)
    small_largest = np.clip(1.0 - features.get("largest_component_fraction", 0.0) * 8.0, 0, 1)
    no_aligned_compact = compact * np.clip(features.get("fraction_without_horizontally_aligned_neighbor", 0.0), 0, 1)
    iso = np.mean([isolated_compact, compact, small_largest, fragmented, no_aligned_compact])
    return {
        "horizontal_connectivity_score": float(h),
        "isolation_score": float(iso),
        "horizontal_run_length": float(features.get("horizontal_run_length_q90", 0.0)),
    }


def _overlay_image(image: np.ndarray, mask: np.ndarray, component_rows: Sequence[dict[str, Any]]) -> np.ndarray:
    base = np.repeat(np.clip(image[..., None], 0, 1), 3, axis=2)
    boundaries = segmentation.find_boundaries(measure.label(mask), mode="outer")
    base[boundaries] = [1.0, 0.2, 0.05]
    for row in component_rows:
        y = int(round(float(row["centroid_y"])))
        x = int(round(float(row["centroid_x"])))
        y0, y1 = max(0, y - 2), min(base.shape[0], y + 3)
        x0, x1 = max(0, x - 2), min(base.shape[1], x + 3)
        base[y0:y1, x0:x1] = [0.1, 0.85, 1.0]
    return base


def _save_overlay_figures(item: PreprocessedImage, result_rows: Sequence[dict[str, float]], component_rows: Sequence[dict[str, Any]], paths: ExperimentPaths) -> tuple[Path, Path, Path]:
    best = max(result_rows, key=lambda row: row.get("horizontal_neighbor_fraction", 0.0) + row.get("bright_pixel_fraction", 0.0))
    threshold = best["threshold_name"]
    binary = best["_mask"]
    overlay = _overlay_image(item.normalized, binary, [row for row in component_rows if row["threshold"] == threshold])
    h_closed = morphology.closing(binary, footprint=morphology.footprint_rectangle((1, 11)))
    skeleton = morphology.skeletonize(binary)
    out_dir = paths.reports_dir / "feature_overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / f"{item.sample_id}_components.png"
    closing_path = out_dir / f"{item.sample_id}_horizontal_closing.png"
    skeleton_path = out_dir / f"{item.sample_id}_skeleton.png"
    plt.imsave(overlay_path, overlay)
    plt.imsave(closing_path, h_closed, cmap="gray")
    skel_rgb = _overlay_image(item.normalized, skeleton, [])
    plt.imsave(skeleton_path, skel_rgb)
    return overlay_path, closing_path, skeleton_path


def extract_features_for_image(item: PreprocessedImage, paths: ExperimentPaths, config: dict[str, Any]) -> FeatureResult:
    thresholds = _binary_thresholds(
        item.normalized,
        item.valid_mask,
        config.get("rheed", {}).get("threshold_percentiles", [75, 85, 90, 95]),
        int(config.get("rheed", {}).get("adaptive_threshold_block_size", 31)),
    )
    threshold_rows: list[dict[str, Any]] = []
    all_components: list[dict[str, Any]] = []
    for threshold_name, binary, threshold_value in thresholds:
        feats, comps = _threshold_features(binary, item.normalized, item.valid_mask, item.sample_id, threshold_name)
        row = {"sample_id": item.sample_id, "threshold_name": threshold_name, "threshold_value": threshold_value, **feats, "_mask": binary}
        threshold_rows.append(row)
        all_components.extend(comps)
    if not threshold_rows:
        raise ValueError(f"No threshold masks could be generated for sample {item.sample_id}")
    overlay_path, closing_path, skeleton_path = _save_overlay_figures(item, threshold_rows, all_components, paths)
    public_threshold_rows = [{key: value for key, value in row.items() if key != "_mask"} for row in threshold_rows]
    agg = _aggregate_threshold_features(public_threshold_rows)
    texture = _texture_features(item.normalized, item.valid_mask)
    nuisance = _nuisance_features(item)
    existing = _existing_morphology_features(item)
    features = {
        "sample_id": item.sample_id,
        "manual_rheed_path": display_path(item.manual_rheed_path, paths.repo_root),
        **agg,
        **texture,
        **nuisance,
        **existing,
        "component_overlay_path": display_path(overlay_path, paths.repo_root),
        "horizontal_closing_overlay_path": display_path(closing_path, paths.repo_root),
        "skeleton_overlay_path": display_path(skeleton_path, paths.repo_root),
    }
    features.update(_composite_scores(features))
    return FeatureResult(
        sample_id=item.sample_id,
        features=features,
        threshold_rows=tuple(public_threshold_rows),
        component_rows=tuple(all_components),
        overlay_path=overlay_path,
        closing_path=closing_path,
        skeleton_path=skeleton_path,
    )


def extract_feature_table(
    images: Sequence[PreprocessedImage],
    paths: ExperimentPaths,
    config: dict[str, Any],
    removelist: RemovelistAudit,
) -> list[FeatureResult]:
    assert_no_removed_samples((item.sample_id for item in images), removelist.sample_ids, context="feature extraction")
    results = [extract_features_for_image(item, paths, config) for item in images]
    assert_no_removed_samples((result.sample_id for result in results), removelist.sample_ids, context="feature table writing")
    write_csv_rows(paths.outputs_dir / "physics_features.csv", [result.features for result in results])
    write_csv_rows(paths.outputs_dir / "threshold_feature_details.csv", [row for result in results for row in result.threshold_rows])
    write_csv_rows(paths.outputs_dir / "component_features.csv", [row for result in results for row in result.component_rows])
    return results


def physics_feature_names(rows: Sequence[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    names = []
    for key in rows[0]:
        if key in {"sample_id", "manual_rheed_path", "component_overlay_path", "horizontal_closing_overlay_path", "skeleton_overlay_path"}:
            continue
        if any(key.startswith(prefix) or key == prefix for prefix in FEATURE_GROUP_PREFIXES["physics"]):
            names.append(key)
    return sorted(names)


def nuisance_feature_names(rows: Sequence[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    names = []
    for key in rows[0]:
        if any(key.startswith(prefix) or key == prefix for prefix in FEATURE_GROUP_PREFIXES["nuisance"]):
            names.append(key)
    return sorted(names)
