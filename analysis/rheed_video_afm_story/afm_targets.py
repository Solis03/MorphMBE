from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import display_path, finite_median, median_abs_deviation, repo_path


def _scan_id_from_path(path: Path) -> str:
    name = path.name
    suffix = "_plane_corrected.npy"
    return name[: -len(suffix)] if name.endswith(suffix) else path.stem


def _load_descriptor_rq(config: dict[str, Any]) -> dict[str, float]:
    table_path = repo_path(config["afm_descriptor_table_path"])
    if not table_path.exists():
        return {}
    df = pd.read_csv(table_path)
    result: dict[str, float] = {}
    for row in df.to_dict("records"):
        result[str(row["afm_path"])] = float(row["Rq"])
    return result


def _scan_size_value(metadata: dict[str, Any]) -> tuple[float | None, float | None]:
    value = metadata.get("scan_size_um")
    if isinstance(value, list) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None, None


def collect_afm_scans(config: dict[str, Any], selected_sample_ids: set[str]) -> pd.DataFrame:
    descriptor_rq = _load_descriptor_rq(config)
    rows: list[dict[str, Any]] = []
    root = repo_path(config["afm_source_roots"]["plane_corrected"])
    for height_path in sorted(root.rglob("*_plane_corrected.npy")):
        try:
            sample_id = height_path.relative_to(root).parts[0]
        except IndexError:
            continue
        if sample_id not in selected_sample_ids:
            continue
        metadata_path = height_path.with_name(height_path.name.replace(".npy", "_metadata.json"))
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        height = np.asarray(np.load(height_path), dtype=float)
        finite = height[np.isfinite(height)]
        rq_recomputed = float(np.sqrt(np.mean((finite - np.mean(finite)) ** 2))) if finite.size else np.nan
        scan_x, scan_y = _scan_size_value(metadata)
        resolution = metadata.get("resolution") or list(height.shape)
        descriptor_value = descriptor_rq.get(display_path(height_path))
        rq_existing_source = "descriptor_table" if descriptor_value is not None else "metadata_height_std"
        rq_existing = descriptor_value if descriptor_value is not None else metadata.get("height_std_nm")
        rq_existing = float(rq_existing) if rq_existing not in (None, "") else np.nan
        abs_diff = abs(rq_recomputed - rq_existing) if np.isfinite(rq_existing) else np.nan
        rel_diff = abs_diff / rq_existing if np.isfinite(rq_existing) and rq_existing != 0 else np.nan
        unit = metadata.get("height_unit_exported") or metadata.get("height_unit_original") or "unknown"
        flags: list[str] = []
        if unit != "nm":
            flags.append(f"height_unit_not_nm:{unit}")
        if finite.size == 0:
            flags.append("no_finite_pixels")
        if rq_existing_source == "descriptor_table" and np.isfinite(abs_diff) and abs_diff > 1e-5:
            flags.append("rq_existing_mismatch_gt_1e-5nm")
        rows.append(
            {
                "sample_id": str(sample_id),
                "afm_file_id": _scan_id_from_path(height_path),
                "afm_path": display_path(height_path),
                "height_array_path": display_path(height_path),
                "metadata_path": display_path(metadata_path),
                "raw_afm_file": metadata.get("raw_afm_file") or metadata.get("raw_file") or "",
                "scan_size_um": scan_x if scan_x == scan_y else np.nan,
                "scan_size_x_um": scan_x,
                "scan_size_y_um": scan_y,
                "resolution_x": int(resolution[1]) if len(resolution) >= 2 else height.shape[1],
                "resolution_y": int(resolution[0]) if len(resolution) >= 2 else height.shape[0],
                "height_unit": unit,
                "channel": metadata.get("primary_channel", ""),
                "rq_existing_nm": rq_existing,
                "rq_existing_source": rq_existing_source,
                "rq_recomputed_nm": rq_recomputed,
                "rq_absolute_difference_nm": abs_diff,
                "rq_relative_difference": rel_diff,
                "used_for_primary_target": False,
                "is_representative": False,
                "quality_flags": ";".join(flags),
            }
        )
    return pd.DataFrame(rows)


def build_sample_targets(
    scan_df: pd.DataFrame,
    primary_size_um: float,
    tolerance_um: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scan_df.empty:
        return pd.DataFrame(), scan_df
    scan_df = scan_df.copy()
    scan_df["is_primary_size"] = (
        (scan_df["scan_size_x_um"] - primary_size_um).abs() <= tolerance_um
    ) & ((scan_df["scan_size_y_um"] - primary_size_um).abs() <= tolerance_um)
    target_rows: list[dict[str, Any]] = []
    for sample_id, group in scan_df.groupby("sample_id", sort=True):
        valid = group[np.isfinite(group["rq_recomputed_nm"]) & (group["rq_recomputed_nm"] > 0)]
        primary = valid[valid["is_primary_size"]]
        best_size = np.nan
        best = pd.DataFrame()
        if not valid.empty:
            size_counts = valid.groupby(["scan_size_x_um", "scan_size_y_um"]).size().reset_index(name="n")
            size_counts["distance_from_1um"] = (
                (size_counts["scan_size_x_um"] - primary_size_um).abs()
                + (size_counts["scan_size_y_um"] - primary_size_um).abs()
            )
            choice = size_counts.sort_values(["distance_from_1um", "scan_size_x_um", "scan_size_y_um"]).iloc[0]
            best = valid[
                (valid["scan_size_x_um"] == choice["scan_size_x_um"])
                & (valid["scan_size_y_um"] == choice["scan_size_y_um"])
            ]
            best_size = float(choice["scan_size_x_um"]) if choice["scan_size_x_um"] == choice["scan_size_y_um"] else np.nan

        def summarize(rows: pd.DataFrame) -> dict[str, Any]:
            values = rows["rq_recomputed_nm"].to_numpy(dtype=float)
            med = finite_median(values)
            iqr = float(np.percentile(values, 75) - np.percentile(values, 25)) if len(values) else np.nan
            mad = median_abs_deviation(values)
            rep = rows.iloc[(rows["rq_recomputed_nm"] - med).abs().to_numpy().argmin()] if len(rows) else None
            return {
                "count": int(len(rows)),
                "median": med,
                "iqr": iqr,
                "mad": mad,
                "min": float(np.min(values)) if len(values) else np.nan,
                "max": float(np.max(values)) if len(values) else np.nan,
                "rep": rep,
            }

        primary_summary = summarize(primary)
        best_summary = summarize(best)
        rep = primary_summary["rep"] if primary_summary["rep"] is not None else best_summary["rep"]
        if rep is not None:
            scan_df.loc[scan_df["afm_path"] == rep["afm_path"], "is_representative"] = True
        if len(primary):
            scan_df.loc[primary.index, "used_for_primary_target"] = True
        target_rows.append(
            {
                "sample_id": str(sample_id),
                "primary_afm_available": bool(len(primary)),
                "primary_afm_scan_size_um": primary_size_um if len(primary) else np.nan,
                "primary_afm_scan_count": primary_summary["count"],
                "primary_rq_nm_median": primary_summary["median"],
                "primary_rq_nm_iqr": primary_summary["iqr"],
                "primary_rq_nm_mad": primary_summary["mad"],
                "primary_rq_nm_min": primary_summary["min"],
                "primary_rq_nm_max": primary_summary["max"],
                "representative_afm_path": rep["afm_path"] if rep is not None else "",
                "representative_afm_height_array": rep["height_array_path"] if rep is not None else "",
                "representative_afm_scan_id": rep["afm_file_id"] if rep is not None else "",
                "best_available_afm_scan_size_um": best_size,
                "best_available_rq_nm_median": best_summary["median"],
                "best_available_scan_count": best_summary["count"],
                "cohort_primary_1um": bool(len(primary)),
                "cohort_exploratory_best_available": bool(len(best)),
                "afm_scan_count_all": int(len(valid)),
            }
        )
    return pd.DataFrame(target_rows), scan_df
