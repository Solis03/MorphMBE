from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import feature, filters, measure, morphology

from .common import repo_path, write_csv


def _as_float(frame: np.ndarray) -> np.ndarray:
    arr = frame.astype(np.float32) / 255.0
    p1, p99 = np.percentile(arr, [1, 99])
    return np.clip((arr - p1) / max(p99 - p1, 1e-6), 0, 1)


def _runs(mask: np.ndarray, axis: int) -> tuple[float, float]:
    longest, spans = 0, []
    arr = mask if axis == 1 else mask.T
    for row in arr:
        padded = np.r_[False, row, False]
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        lengths = edges[1::2] - edges[::2]
        if lengths.size:
            longest = max(longest, int(lengths.max()))
            spans.append(float(lengths.max() / max(mask.shape[axis], 1)))
    return float(longest / max(mask.shape[axis], 1)), float(np.mean(spans) if spans else 0.0)


def _component_features(arr: np.ndarray, threshold: float) -> dict[str, float]:
    mask = arr >= np.percentile(arr, threshold)
    lab = measure.label(mask)
    props = measure.regionprops(lab, intensity_image=arr)
    areas = np.asarray([p.area for p in props], dtype=float)
    ecc = np.asarray([p.eccentricity for p in props], dtype=float) if props else np.array([])
    sol = np.asarray([p.solidity for p in props], dtype=float) if props else np.array([])
    circ = []
    for p in props:
        circ.append(4 * np.pi * p.area / max(p.perimeter**2, 1e-6))
    circ = np.asarray(circ, dtype=float)
    return {
        f"component_count_p{int(threshold)}": float(len(props)),
        f"component_area_median_p{int(threshold)}": float(np.median(areas) / arr.size) if areas.size else 0.0,
        f"component_area_q90_p{int(threshold)}": float(np.percentile(areas, 90) / arr.size) if areas.size else 0.0,
        f"component_eccentricity_mean_p{int(threshold)}": float(np.mean(ecc)) if ecc.size else 0.0,
        f"component_solidity_mean_p{int(threshold)}": float(np.mean(sol)) if sol.size else 0.0,
        f"component_circularity_mean_p{int(threshold)}": float(np.mean(circ)) if circ.size else 0.0,
        f"round_component_fraction_p{int(threshold)}": float(np.mean((circ > 0.45) & (ecc < 0.75))) if circ.size else 0.0,
        f"largest_component_area_fraction_p{int(threshold)}": float(areas.max() / arr.size) if areas.size else 0.0,
    }


def frame_physics_features(frame: np.ndarray) -> dict[str, float]:
    arr = _as_float(frame)
    gx = filters.sobel_h(arr)
    gy = filters.sobel_v(arr)
    grad = np.hypot(gx, gy)
    # Structure tensor coherence.
    a00, a01, a11 = feature.structure_tensor(arr, sigma=2)
    l1, l2 = feature.structure_tensor_eigenvalues((a00, a01, a11))
    coherence = (l1 - l2) / (l1 + l2 + 1e-6)
    orientation = np.mod(np.arctan2(gy, gx), np.pi)
    hist, _ = np.histogram(orientation, bins=18, range=(0, np.pi), weights=grad)
    prob = hist / (hist.sum() + 1e-12)
    orient_entropy = float(-np.sum(prob * np.log(prob + 1e-12)) / np.log(18))
    line_responses = []
    blob_responses = []
    for sigma in (1, 2, 4):
        line_responses.append(np.abs(filters.sato(arr, sigmas=[sigma], black_ridges=False)))
        blob_responses.append(np.abs(ndimage.gaussian_laplace(arr, sigma=sigma)))
    line = np.max(np.stack(line_responses), axis=0)
    blob = np.max(np.stack(blob_responses), axis=0)
    bright = arr >= np.percentile(arr, 95)
    hrun, hcont = _runs(bright, axis=1)
    vrun, vcont = _runs(bright, axis=0)
    skel = morphology.skeletonize(bright)
    neigh = ndimage.convolve(skel.astype(int), np.ones((3, 3), dtype=int), mode="constant")
    skel_len = float(skel.mean())
    branch_density = float(((skel) & (neigh >= 5)).mean())
    endpoint_density = float(((skel) & (neigh == 2)).mean())
    fft = np.abs(np.fft.fftshift(np.fft.fft2(arr - arr.mean()))) ** 2
    cy, cx = np.array(fft.shape) // 2
    horiz = float(fft[cy - 2 : cy + 3, :].sum())
    vert = float(fft[:, cx - 2 : cx + 3].sum())
    row_proj = arr.mean(axis=1)
    col_proj = arr.mean(axis=0)
    center = arr[arr.shape[0] // 4 : 3 * arr.shape[0] // 4, arr.shape[1] // 4 : 3 * arr.shape[1] // 4]
    border = np.concatenate([arr[:20].ravel(), arr[-20:].ravel(), arr[:, :20].ravel(), arr[:, -20:].ravel()])
    out = {
        "line_response_q50": float(np.percentile(line, 50)),
        "line_response_q90": float(np.percentile(line, 90)),
        "line_response_q97": float(np.percentile(line, 97)),
        "line_connected_fraction": float((line > np.percentile(line, 95)).mean()),
        "structure_tensor_coherence": float(np.nanmean(coherence)),
        "dominant_orientation": float(np.angle(np.sum(np.exp(1j * 2 * orientation) * grad)) / 2),
        "orientation_entropy": orient_entropy,
        "longest_bright_horizontal_run": hrun,
        "longest_bright_vertical_run": vrun,
        "horizontal_run_continuity": hcont,
        "vertical_run_continuity": vcont,
        "fft_horizontal_vertical_anisotropy": float((horiz - vert) / (horiz + vert + 1e-9)),
        "row_projection_sharpness": float(np.std(row_proj) / (np.mean(row_proj) + 1e-6)),
        "column_projection_sharpness": float(np.std(col_proj) / (np.mean(col_proj) + 1e-6)),
        "blob_response_energy": float(np.mean(blob**2)),
        "blob_response_q95": float(np.percentile(blob, 95)),
        "local_maximum_density": float(morphology.local_maxima(arr).mean()),
        "spot_peak_width_proxy": float(np.sqrt(max(np.sum(bright), 1)) / max(arr.shape)),
        "skeleton_total_length_fraction": skel_len,
        "skeleton_branch_point_density": branch_density,
        "skeleton_endpoint_density": endpoint_density,
        "connected_bright_span_fraction": float(max(hrun, vrun)),
        "horizontal_percolation_fraction": float(np.any(bright, axis=1).mean()),
        "vertical_percolation_fraction": float(np.any(bright, axis=0).mean()),
        "dark_background_median": float(np.median(arr[arr <= np.percentile(arr, 25)])),
        "low_percentile_intensity": float(np.percentile(arr, 5)),
        "high_percentile_intensity": float(np.percentile(arr, 95)),
        "diffuse_to_peak_intensity_ratio": float(np.percentile(arr, 50) / max(np.percentile(arr, 99), 1e-6)),
        "saturation_fraction": float((frame >= 250).mean()),
        "background_spatial_smoothness": float(1.0 / (1.0 + np.std(filters.sobel(arr)))),
        "center_to_border_ratio": float(center.mean() / max(border.mean(), 1e-6)),
    }
    for thr in (90, 95, 97):
        out.update(_component_features(arr, thr))
    out["component_merge_rate_p97_to_p90"] = out["component_count_p90"] - out["component_count_p97"]
    return out


def aggregate_temporal(features: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(features[0])
    out: dict[str, float] = {}
    for key in keys:
        vals = np.asarray([f[key] for f in features], dtype=float)
        out[f"{key}_median"] = float(np.nanmedian(vals))
        out[f"{key}_iqr"] = float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25))
        out[f"{key}_std"] = float(np.nanstd(vals))
        out[f"{key}_first_last_diff"] = float(vals[-1] - vals[0])
        out[f"{key}_slope"] = float(np.polyfit(np.arange(len(vals)), vals, 1)[0]) if len(vals) > 1 else 0.0
    out["temporal_stability"] = float(1.0 / (1.0 + np.nanmedian([out[k] for k in out if k.endswith("_std")])))
    return out


def summarize_categories(row: pd.Series) -> dict[str, float]:
    spot_cols = [c for c in row.index if any(s in c for s in ("blob", "local_maximum", "component_circularity", "round_component", "spot_peak")) and c.endswith("_median")]
    streak_cols = [c for c in row.index if any(s in c for s in ("line_response", "coherence", "horizontal_run", "vertical_run", "row_projection", "fft_horizontal")) and c.endswith("_median")]
    conn_cols = [c for c in row.index if any(s in c for s in ("skeleton", "percolation", "connected_bright", "merge_rate", "largest_component")) and c.endswith("_median")]
    diff_cols = [c for c in row.index if any(s in c for s in ("background", "diffuse", "saturation", "center_to_border", "percentile_intensity")) and c.endswith("_median")]
    return {
        "spot_summary_raw": float(np.nanmean(np.asarray(row[spot_cols], dtype=float))) if spot_cols else 0.0,
        "streak_summary_raw": float(np.nanmean(np.asarray(row[streak_cols], dtype=float))) if streak_cols else 0.0,
        "connection_summary_raw": float(np.nanmean(np.asarray(row[conn_cols], dtype=float))) if conn_cols else 0.0,
        "diffuse_summary_raw": float(np.nanmean(np.asarray(row[diff_cols], dtype=float))) if diff_cols else 0.0,
    }


def extract_rheed_physics_features(manifest: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    clip_root = repo_path(config["rheed_clip_root"])
    for _, mrow in manifest.iterrows():
        sid = str(mrow["sample_id"])
        row: dict[str, Any] = {"sample_id": sid, "growth_run_id": str(mrow["growth_run_id"]), "video_stage": str(mrow.get("video_stage", "unknown"))}
        for variant in config["rheed_clip_variants"]:
            path = clip_root / variant / f"{sid}.npz"
            z = np.load(path)
            frames = z["frames_uint8"]
            frame_features = [frame_physics_features(frame) for frame in frames]
            agg = aggregate_temporal(frame_features)
            for key, value in agg.items():
                row[f"{variant}__{key}"] = value
            if variant == "selected_16":
                row["temporal_brightness_drift"] = float(frames[-1].mean() - frames[0].mean()) / 255.0
        row.update(summarize_categories(pd.Series(row)))
        rows.append(row)
    out = pd.DataFrame(rows)
    write_csv(out, repo_path(config["output_root"]) / "rheed_physics_features.csv")
    return out


def feature_source_is_target_blind() -> bool:
    return True
