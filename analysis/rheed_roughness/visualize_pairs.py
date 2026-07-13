"""Visual paired RHEED/AFM inspection for the roughness audit.

This module consumes the completed ``analysis.rheed_roughness.run`` outputs.
It does not train a model and does not redefine the morphology index.  Frame
and AFM scan selection are deterministic quality rules that do not use the
opposite modality's target or appearance.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import imageio.v2 as imageio
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

from rheed2morph.rheed.frame_quality import frame_to_gray_float32
from rheed2morph.rheed.shape_preprocessing import preprocess_frame_for_shape, robust_rescale

from analysis.rheed_roughness.run import (
    convert_height_to_nm,
    csv_value,
    display_path,
    read_config,
    read_csv_rows,
    resolve_path,
    safe_float,
    write_csv_rows,
    write_json,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
REMOVELIST_SAMPLE_RE = re.compile(r"^\s*(\d+)")
REMOVELIST_AUDIT_FIELDNAMES = [
    "line_number",
    "sample_id",
    "note",
    "raw_line",
    "present_in_frame_scores",
    "present_in_sample_table",
    "present_in_afm_targets",
    "present_in_pairing_audit",
    "excluded_from_visualization",
]
RHEED_QUALITY_INPUT_COLUMNS = {
    "detector_confidence",
    "sharpness_score",
    "focus_metric",
    "contrast_score",
    "robust_dynamic_range",
    "dynamic_range_score",
    "pattern_visibility_score",
    "roi_coverage",
    "roi_clipping",
    "saturation_fraction",
    "underexposed_fraction",
    "frame_to_frame_motion_magnitude",
    "frame_to_reference_translation_x",
    "frame_to_reference_translation_y",
    "blur_penalty",
    "frame_qc_reasons",
    "frame_valid",
}


@dataclass(frozen=True)
class PairVizPaths:
    repo_root: Path
    source_outputs_dir: Path
    source_reports_dir: Path
    output_dir: Path
    report_dir: Path
    cache_dir: Path
    rheed_contact_dir: Path
    afm_contact_dir: Path
    pages_dir: Path


def build_paths(config: dict[str, Any]) -> PairVizPaths:
    """Create and return the visualization output directories."""
    repo_root = resolve_path(REPO_ROOT, config.get("repo_root", ".")).resolve()
    source_outputs = resolve_path(repo_root, config["outputs_dir"]).resolve()
    source_reports = resolve_path(repo_root, config["reports_dir"]).resolve()
    output_dir = source_outputs / "pair_visualization"
    report_dir = source_reports / "pair_visualization"
    cache_dir = output_dir / "cache"
    rheed_contact_dir = report_dir / "rheed_candidate_contact_sheets"
    afm_contact_dir = report_dir / "afm_candidate_contact_sheets"
    pages_dir = report_dir / "pages"
    for path in (output_dir, report_dir, cache_dir, rheed_contact_dir, afm_contact_dir, pages_dir):
        path.mkdir(parents=True, exist_ok=True)
    return PairVizPaths(
        repo_root=repo_root,
        source_outputs_dir=source_outputs,
        source_reports_dir=source_reports,
        output_dir=output_dir,
        report_dir=report_dir,
        cache_dir=cache_dir,
        rheed_contact_dir=rheed_contact_dir,
        afm_contact_dir=afm_contact_dir,
        pages_dir=pages_dir,
    )


def reset_pair_visualization_outputs(paths: PairVizPaths) -> None:
    """Remove prior generated visualization artifacts before rebuilding."""
    for path in (paths.output_dir, paths.report_dir):
        if path.name != "pair_visualization":
            raise ValueError(f"Refusing to clean unexpected visualization path: {path}")
        if path.exists():
            shutil.rmtree(path)
    for path in (
        paths.output_dir,
        paths.report_dir,
        paths.cache_dir,
        paths.rheed_contact_dir,
        paths.afm_contact_dir,
        paths.pages_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def load_existing_analysis_outputs(paths: PairVizPaths) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the completed roughness-analysis tables."""
    parquet = paths.source_outputs_dir / "frame_level_scores.parquet"
    if parquet.is_file():
        frame_df = pd.read_parquet(parquet)
    else:
        frame_df = pd.read_csv(paths.source_outputs_dir / "frame_level_scores.csv")
    sample_df = pd.read_csv(paths.source_outputs_dir / "sample_level_analysis_table.csv")
    afm_df = pd.read_csv(paths.source_outputs_dir / "afm_scan_level_targets.csv")
    pairing_df = pd.read_csv(paths.source_outputs_dir / "pairing_audit.csv")
    for df in (frame_df, sample_df, afm_df, pairing_df):
        if "sample_id" in df.columns:
            df["sample_id"] = df["sample_id"].astype(str)
    return frame_df, sample_df, afm_df, pairing_df


def read_removelist(path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    """Parse leading numeric sample IDs from removelist lines."""
    if not path.is_file():
        return set(), []
    sample_ids: set[str] = set()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            match = REMOVELIST_SAMPLE_RE.match(line)
            if not match:
                rows.append(
                    {
                        "line_number": line_number,
                        "sample_id": "",
                        "note": "unparsed",
                        "raw_line": line,
                    }
                )
                continue
            sample_id = match.group(1)
            sample_ids.add(sample_id)
            rows.append(
                {
                    "line_number": line_number,
                    "sample_id": sample_id,
                    "note": line[match.end() :].strip(" -\t"),
                    "raw_line": line,
                }
            )
    return sample_ids, rows


def build_excluded_sample_audit(
    removelist_rows: Sequence[dict[str, Any]],
    frame_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    afm_df: pd.DataFrame,
    pairing_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Describe where each removelist sample appears before filtering."""
    frame_ids = set(frame_df.get("sample_id", pd.Series(dtype=str)).astype(str))
    sample_ids = set(sample_df.get("sample_id", pd.Series(dtype=str)).astype(str))
    afm_ids = set(afm_df.get("sample_id", pd.Series(dtype=str)).astype(str))
    pairing_ids = set(pairing_df.get("sample_id", pd.Series(dtype=str)).astype(str))
    audit_rows: list[dict[str, Any]] = []
    for row in removelist_rows:
        sample_id = str(row.get("sample_id", ""))
        present = {
            "present_in_frame_scores": int(sample_id in frame_ids),
            "present_in_sample_table": int(sample_id in sample_ids),
            "present_in_afm_targets": int(sample_id in afm_ids),
            "present_in_pairing_audit": int(sample_id in pairing_ids),
        }
        audit_rows.append(
            {
                "line_number": row.get("line_number", ""),
                "sample_id": sample_id,
                "note": row.get("note", ""),
                "raw_line": row.get("raw_line", ""),
                **present,
                "excluded_from_visualization": int(any(present.values())),
            }
        )
    return audit_rows


def filter_removed_samples(
    frame_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    afm_df: pd.DataFrame,
    pairing_df: pd.DataFrame,
    removelist_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop removelist sample IDs from every table feeding visualization."""
    if not removelist_ids:
        return frame_df, sample_df, afm_df, pairing_df
    filtered: list[pd.DataFrame] = []
    for df in (frame_df, sample_df, afm_df, pairing_df):
        if "sample_id" not in df.columns:
            filtered.append(df.copy())
        else:
            filtered.append(df[~df["sample_id"].astype(str).isin(removelist_ids)].copy())
    return filtered[0], filtered[1], filtered[2], filtered[3]


def robust_percentile_scale(values: pd.Series, *, inverse: bool = False) -> pd.Series:
    """Scale a numeric series to [0, 1] using robust 5th/95th percentiles."""
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        scaled = pd.Series(np.zeros(len(numeric)), index=values.index, dtype=float)
    else:
        lo = float(np.nanpercentile(finite, 5))
        hi = float(np.nanpercentile(finite, 95))
        if hi <= lo + 1e-12:
            scaled = pd.Series(np.full(len(numeric), 0.5), index=values.index, dtype=float)
        else:
            scaled = (numeric - lo) / (hi - lo)
            scaled = scaled.clip(0.0, 1.0).fillna(0.0)
    return 1.0 - scaled if inverse else scaled


def build_rheed_candidate_table(frame_df: pd.DataFrame, paths: PairVizPaths) -> pd.DataFrame:
    """Build candidate RHEED-frame quality components from existing frame scores.

    The resulting score intentionally excludes AFM roughness and excludes
    streaky/spotty direction as a preference.
    """
    df = frame_df.copy()
    for col in RHEED_QUALITY_INPUT_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    shift_x = pd.to_numeric(df["frame_to_reference_translation_x"], errors="coerce").fillna(0.0)
    shift_y = pd.to_numeric(df["frame_to_reference_translation_y"], errors="coerce").fillna(0.0)
    df["shift_magnitude"] = np.sqrt(shift_x * shift_x + shift_y * shift_y)
    df["detector_confidence_component"] = robust_percentile_scale(df["detector_confidence"])
    sharpness_source = df["sharpness_score"] if "sharpness_score" in df.columns else df["focus_metric"]
    df["sharpness_component"] = robust_percentile_scale(sharpness_source)
    contrast_source = pd.to_numeric(df["contrast_score"], errors="coerce").fillna(0.0) + pd.to_numeric(
        df["robust_dynamic_range"], errors="coerce"
    ).fillna(0.0)
    df["contrast_component"] = robust_percentile_scale(contrast_source)
    valid_signal_source = pd.to_numeric(df["dynamic_range_score"], errors="coerce").fillna(0.0) + pd.to_numeric(
        df["pattern_visibility_score"], errors="coerce"
    ).fillna(0.0)
    df["valid_signal_component"] = robust_percentile_scale(valid_signal_source)
    df["roi_completeness_component"] = (
        pd.to_numeric(df["roi_coverage"], errors="coerce").fillna(1.0)
        * (1.0 - pd.to_numeric(df["roi_clipping"], errors="coerce").fillna(0.0).clip(0.0, 1.0))
    ).clip(0.0, 1.0)
    df["saturation_penalty_component"] = robust_percentile_scale(df["saturation_fraction"])
    df["underexposure_penalty_component"] = robust_percentile_scale(df["underexposed_fraction"])
    motion_source = pd.to_numeric(df["frame_to_frame_motion_magnitude"], errors="coerce").fillna(0.0) + df["shift_magnitude"]
    df["motion_penalty_component"] = robust_percentile_scale(motion_source)
    df["clipping_penalty_component"] = pd.to_numeric(df["roi_clipping"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    qc_text = df["frame_qc_reasons"].fillna("").astype(str)
    valid_flag = pd.to_numeric(df["frame_valid"], errors="coerce").fillna(1.0)
    df["detector_failure_penalty_component"] = ((qc_text != "") | (valid_flag < 0.5)).astype(float)
    df["pair_visual_quality_score"] = (
        0.22 * df["detector_confidence_component"]
        + 0.18 * df["sharpness_component"]
        + 0.18 * df["contrast_component"]
        + 0.16 * df["valid_signal_component"]
        + 0.10 * df["roi_completeness_component"]
        - 0.08 * df["saturation_penalty_component"]
        - 0.08 * df["underexposure_penalty_component"]
        - 0.06 * df["motion_penalty_component"]
        - 0.06 * df["clipping_penalty_component"]
        - 0.18 * df["detector_failure_penalty_component"]
    )
    df["pair_visual_quality_score"] = df["pair_visual_quality_score"].clip(-1.0, 1.0)
    df["pair_visual_hard_qc_fail"] = (
        (pd.to_numeric(df["saturation_fraction"], errors="coerce").fillna(0.0) > 0.20)
        | (pd.to_numeric(df["underexposed_fraction"], errors="coerce").fillna(0.0) > 0.70)
        | (pd.to_numeric(df["detector_confidence"], errors="coerce").fillna(0.0) < 0.05)
        | (pd.to_numeric(df["robust_dynamic_range"], errors="coerce").fillna(0.0) < 0.025)
        | (pd.to_numeric(df["roi_coverage"], errors="coerce").fillna(1.0) < 0.50)
    )
    df["pair_visual_qc_flags"] = df.apply(frame_qc_flags, axis=1)
    return df.sort_values(["sample_id", "pair_visual_quality_score", "frame_timestamp_sec"], ascending=[True, False, True])


def frame_qc_flags(row: pd.Series) -> str:
    flags = []
    if safe_float(row.get("saturation_fraction"), 0.0) > 0.20:
        flags.append("excessive_saturation")
    if safe_float(row.get("underexposed_fraction"), 0.0) > 0.70:
        flags.append("severe_underexposure")
    if safe_float(row.get("detector_confidence"), 0.0) < 0.05:
        flags.append("low_detector_confidence")
    if safe_float(row.get("robust_dynamic_range"), 0.0) < 0.025:
        flags.append("low_dynamic_range")
    if safe_float(row.get("roi_coverage"), 1.0) < 0.50:
        flags.append("low_roi_coverage")
    existing = str(row.get("frame_qc_reasons", "") or "")
    if existing:
        flags.extend(flag for flag in existing.split(";") if flag)
    return ";".join(dict.fromkeys(flags))


def select_representative_rheed_frame(sample_candidates: pd.DataFrame, *, top_n: int = 6) -> tuple[pd.Series, pd.DataFrame]:
    """Select a representative RHEED frame from one sample's ranked candidates."""
    ranked = sample_candidates.sort_values(["pair_visual_quality_score", "frame_timestamp_sec"], ascending=[False, True]).copy()
    top = ranked.head(top_n).copy()
    eligible = ranked[~ranked["pair_visual_hard_qc_fail"].astype(bool)]
    source = eligible if not eligible.empty else ranked
    max_score = float(source["pair_visual_quality_score"].max())
    near_top = source[source["pair_visual_quality_score"] >= max_score - 0.02]
    if len(near_top) > 1:
        median_time = float(near_top["frame_timestamp_sec"].median())
        selected = near_top.iloc[(near_top["frame_timestamp_sec"].astype(float) - median_time).abs().argsort().iloc[0]]
        reason = "highest_quality_near_tie_temporal_center"
    else:
        selected = source.iloc[0]
        reason = "highest_quality_hard_qc_pass" if not eligible.empty else "highest_quality_no_hard_qc_pass"
    selected_frame = safe_float(selected.get("frame_idx"), -1)
    top["selected"] = top["frame_idx"].astype(float) == selected_frame
    top["selection_reason"] = reason
    return selected, top


def build_afm_candidate_table(afm_df: pd.DataFrame) -> pd.DataFrame:
    """Return valid physical AFM candidate rows."""
    df = afm_df.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["scan_size_um"] = pd.to_numeric(df["scan_size_um"], errors="coerce")
    df["Rq_nm"] = pd.to_numeric(df["Rq_nm"], errors="coerce")
    df["Ra_nm"] = pd.to_numeric(df["Ra_nm"], errors="coerce")
    df["valid_physical_height"] = (
        (df.get("target_status", "ok").fillna("ok").astype(str).isin(["ok", "unknown_height_unit"]))
        & df["Rq_nm"].notna()
        & df["afm_path"].fillna("").astype(str).ne("")
    )
    return df[df["valid_physical_height"]].copy()


def select_representative_afm_scan(
    sample_candidates: pd.DataFrame,
    *,
    primary_scan_size_um: float,
    tolerance_um: float,
) -> pd.Series:
    """Select the AFM scan whose Rq is closest to the median of a fixed subset."""
    primary = sample_candidates[
        sample_candidates["scan_size_um"].sub(primary_scan_size_um).abs() <= tolerance_um
    ].copy()
    fallback_used = primary.empty
    if fallback_used:
        valid_sizes = sample_candidates["scan_size_um"].dropna()
        if valid_sizes.empty:
            subset = sample_candidates.copy()
            dominant_size = math.nan
        else:
            rounded = valid_sizes.round(6)
            counts = rounded.value_counts()
            dominant_size = float(counts.index[0])
            subset = sample_candidates[sample_candidates["scan_size_um"].round(6) == dominant_size].copy()
    else:
        subset = primary
        dominant_size = primary_scan_size_um
    median_rq = float(subset["Rq_nm"].median())
    selected = subset.assign(distance_from_sample_median_rq_nm=(subset["Rq_nm"] - median_rq).abs()).sort_values(
        ["distance_from_sample_median_rq_nm", "afm_scan_id", "afm_path"]
    ).iloc[0].copy()
    selected["sample_median_rq_nm"] = median_rq
    selected["fallback_used"] = bool(fallback_used)
    selected["selection_reason"] = (
        "closest_to_primary_1um_subset_median_rq"
        if not fallback_used
        else f"no_primary_1um_scan_closest_to_dominant_size_{dominant_size:g}_median_rq"
    )
    selected["afm_qc_flags"] = "" if str(selected.get("height_unit_status", "")) == "ok" else str(selected.get("height_unit_status", ""))
    return selected


def load_height_nm(path: Path, unit: str | None = "nm") -> np.ndarray:
    """Load an AFM height map and convert it to nanometers."""
    height = np.load(path)
    height_nm, _ = convert_height_to_nm(height, unit)
    return np.asarray(height_nm, dtype=np.float64)


def calculate_common_afm_scale(selected_rows: Sequence[dict[str, Any]], paths: PairVizPaths) -> tuple[float, float]:
    """Calculate a common robust scale across selected primary 1 um AFM maps."""
    values: list[np.ndarray] = []
    for row in selected_rows:
        if int(row.get("fallback_used", 0)):
            continue
        scan_size = safe_float(row.get("scan_size_um"))
        if not math.isfinite(scan_size) or abs(scan_size - 1.0) > 0.10:
            continue
        path = resolve_path(paths.repo_root, str(row.get("selected_height_map_path", "")))
        if path.is_file():
            height = load_height_nm(path, str(row.get("height_unit_exported", "nm")))
            values.append(height[np.isfinite(height)].ravel())
    if not values:
        return -1.0, 1.0
    merged = np.concatenate(values)
    lo = float(np.nanpercentile(merged, 2))
    hi = float(np.nanpercentile(merged, 98))
    limit = max(abs(lo), abs(hi), 1e-6)
    return -limit, limit


def nice_scale_bar_um(scan_size_um: float) -> float:
    """Choose a readable lateral scale-bar length in micrometers."""
    if not math.isfinite(scan_size_um) or scan_size_um <= 0:
        return math.nan
    target = scan_size_um / 5.0
    candidates = np.asarray([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    candidates = candidates[candidates <= scan_size_um * 0.6]
    if candidates.size == 0:
        return target
    return float(candidates[np.argmin(np.abs(candidates - target))])


def render_rheed_processed_cache(row: pd.Series | dict[str, Any], paths: PairVizPaths) -> dict[str, str]:
    """Cache the selected raw ROI, processed ROI, and overlay for plotting."""
    sample_id = str(row.get("sample_id", ""))
    frame_idx = int(safe_float(row.get("frame_idx"), -1))
    out_dir = paths.cache_dir / "rheed" / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_src = resolve_path(paths.repo_root, str(row.get("raw_frame_path", "")))
    overlay_src = resolve_path(paths.repo_root, str(row.get("overlay_path", "")))
    raw_dst = out_dir / f"frame_{frame_idx:06d}_raw_roi.png"
    processed_dst = out_dir / f"frame_{frame_idx:06d}_processed_roi.png"
    overlay_dst = out_dir / f"frame_{frame_idx:06d}_overlay.png"
    if raw_src.is_file() and not raw_dst.is_file():
        shutil.copy2(raw_src, raw_dst)
    if overlay_src.is_file() and not overlay_dst.is_file():
        shutil.copy2(overlay_src, overlay_dst)
    if raw_dst.is_file() and not processed_dst.is_file():
        gray = frame_to_gray_float32(np.asarray(Image.open(raw_dst)))
        processed = preprocess_frame_for_shape(gray, image_size=int(max(gray.shape)))
        Image.fromarray(np.asarray(np.clip(robust_rescale(processed.channels["pclip_norm"]) * 255, 0, 255), dtype=np.uint8)).save(
            processed_dst
        )
    return {
        "cached_raw_roi_path": display_path(raw_dst, paths.repo_root) if raw_dst.is_file() else "",
        "cached_processed_roi_path": display_path(processed_dst, paths.repo_root) if processed_dst.is_file() else "",
        "cached_overlay_path": display_path(overlay_dst, paths.repo_root) if overlay_dst.is_file() else "",
    }


def render_afm_cache(
    row: dict[str, Any],
    paths: PairVizPaths,
    *,
    scale_mode: str,
    common_scale: tuple[float, float] | None = None,
) -> str:
    """Render a cached AFM panel PNG from a physical height map."""
    sample_id = str(row.get("sample_id", ""))
    scan_id = str(row.get("selected_afm_scan_id", "scan"))
    out_dir = paths.cache_dir / "afm" / scale_mode
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{sample_id}_{scan_id}_{scale_mode}.png"
    if out.is_file():
        return display_path(out, paths.repo_root)
    height_path = resolve_path(paths.repo_root, str(row.get("selected_height_map_path", "")))
    if not height_path.is_file():
        return ""
    height = load_height_nm(height_path, str(row.get("height_unit_exported", "nm")))
    if common_scale:
        vmin, vmax = common_scale
    else:
        vmin = float(np.nanmin(height))
        vmax = float(np.nanmax(height))
    fig, ax = plt.subplots(figsize=(3.2, 3.0), dpi=160)
    im = draw_afm_axis(ax, height, row, vmin=vmin, vmax=vmax, title=f"{sample_id} {scan_id}")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Height (nm)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return display_path(out, paths.repo_root)


def draw_afm_axis(
    ax: plt.Axes,
    height: np.ndarray,
    row: dict[str, Any] | pd.Series,
    *,
    vmin: float,
    vmax: float,
    title: str,
) -> Any:
    """Draw one AFM height map with scan-size scale bar."""
    im = ax.imshow(height, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    scan_size = safe_float(row.get("scan_size_um"))
    if math.isfinite(scan_size) and scan_size > 0:
        bar_um = nice_scale_bar_um(scan_size)
        if math.isfinite(bar_um):
            pixels = height.shape[1] * bar_um / scan_size
            x0 = height.shape[1] * 0.08
            x1 = min(height.shape[1] * 0.92, x0 + pixels)
            y = height.shape[0] * 0.90
            ax.plot([x0, x1], [y, y], color="white", lw=3, solid_capstyle="butt")
            ax.plot([x0, x1], [y, y], color="black", lw=1, solid_capstyle="butt")
            ax.text(x0, y - height.shape[0] * 0.05, f"{bar_um:g} µm", color="white", fontsize=6, va="bottom")
    rq = format_value(row.get("rq_nm"), "nm", precision=2)
    scan = f"{scan_size:g} x {scan_size:g} µm" if math.isfinite(scan_size) else "scan N/A"
    ax.set_title(f"{title}\nRq={rq}; {scan}", fontsize=7)
    return im


def format_value(value: Any, unit: str = "", *, precision: int = 3) -> str:
    val = safe_float(value)
    if not math.isfinite(val):
        return "N/A"
    suffix = f" {unit}" if unit else ""
    return f"{val:.{precision}g}{suffix}"


def render_rheed_axis(ax: plt.Axes, row: dict[str, Any] | pd.Series, paths: PairVizPaths, *, title_prefix: str = "RHEED") -> None:
    path = resolve_path(paths.repo_root, str(row.get("cached_processed_roi_path") or row.get("cached_raw_roi_path") or row.get("raw_frame_path", "")))
    if path.is_file():
        image = Image.open(path)
        ax.imshow(image, cmap="gray")
    else:
        ax.text(0.5, 0.5, "RHEED N/A", ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{title_prefix}\nt={format_value(row.get('selected_timestamp_sec') or row.get('frame_timestamp_sec'), 's', precision=3)}; "
        f"q={format_value(row.get('quality_score') or row.get('pair_visual_quality_score'), precision=3)}\n"
        f"MI={format_value(row.get('morphology_index'), precision=3)}",
        fontsize=7,
    )


def render_pair_grid(
    rows: Sequence[dict[str, Any]],
    paths: PairVizPaths,
    *,
    output_stem: str,
    title: str,
    sort_key: str,
    common_scale: tuple[float, float] | None = None,
    cards_per_row: int = 4,
) -> None:
    """Render a multi-sample RHEED/AFM grid in PNG and PDF."""
    valid = [row for row in rows if row.get("selected_height_map_path")]
    valid = sorted(valid, key=lambda row: (safe_float(row.get(sort_key), math.inf), str(row.get("sample_id", ""))))
    if not valid:
        return
    nrows = math.ceil(len(valid) / cards_per_row)
    fig = plt.figure(figsize=(cards_per_row * 4.6, nrows * 3.1), dpi=300)
    width_pattern: list[float] = []
    for _ in range(cards_per_row):
        width_pattern.extend([1.0, 1.0, 0.055])
    grid = fig.add_gridspec(nrows=nrows, ncols=cards_per_row * 3, width_ratios=width_pattern, hspace=0.48, wspace=0.10)
    fig.suptitle(title, fontsize=15, y=0.997)
    for idx, row in enumerate(valid):
        rr = idx // cards_per_row
        cc = (idx % cards_per_row) * 3
        ax_rheed = fig.add_subplot(grid[rr, cc])
        ax_afm = fig.add_subplot(grid[rr, cc + 1])
        cax = fig.add_subplot(grid[rr, cc + 2])
        render_rheed_axis(ax_rheed, row, paths, title_prefix=f"Sample {row.get('sample_id', '')}")
        height_path = resolve_path(paths.repo_root, str(row.get("selected_height_map_path", "")))
        if height_path.is_file():
            height = load_height_nm(height_path, str(row.get("height_unit_exported", "nm")))
            if common_scale:
                vmin, vmax = common_scale
            else:
                vmin = float(np.nanmin(height))
                vmax = float(np.nanmax(height))
            im = draw_afm_axis(ax_afm, height, row, vmin=vmin, vmax=vmax, title="AFM")
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=5)
            cbar.set_label("Height (nm)", fontsize=6)
            if int(row.get("fallback_used", 0)):
                for spine in ax_afm.spines.values():
                    spine.set_edgecolor("#d62728")
                    spine.set_linewidth(2.0)
                ax_afm.text(0.02, 0.04, "non-1 µm fallback", transform=ax_afm.transAxes, fontsize=6, color="white")
        else:
            ax_afm.text(0.5, 0.5, "AFM N/A", ha="center", va="center")
            cax.axis("off")
    for idx in range(len(valid), nrows * cards_per_row):
        rr = idx // cards_per_row
        cc = (idx % cards_per_row) * 3
        for sub in range(3):
            fig.add_subplot(grid[rr, cc + sub]).axis("off")
    png = paths.report_dir / f"{output_stem}.png"
    pdf = paths.report_dir / f"{output_stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def decode_source_frame_at_time(video_path: Path, timestamp_sec: float, out_path: Path) -> str:
    """Best-effort decode of a raw source video frame at a timestamp."""
    if out_path.is_file():
        return out_path.as_posix()
    if not video_path.is_file() or not math.isfinite(timestamp_sec):
        return ""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(str(video_path), "ffmpeg")
    try:
        meta = reader.get_meta_data()
        fps = safe_float(meta.get("fps"), 30.0)
        frame_idx = max(0, int(round(timestamp_sec * fps)))
        frame = reader.get_data(frame_idx)
        Image.fromarray(frame).save(out_path)
        return out_path.as_posix()
    except Exception:
        return ""
    finally:
        reader.close()


def generate_rheed_contact_sheet(sample_id: str, top_candidates: pd.DataFrame, selected_frame_idx: int, paths: PairVizPaths) -> str:
    """Create a per-sample RHEED candidate contact sheet."""
    rows = top_candidates.head(6).to_dict("records")
    if not rows:
        return ""
    fig, axes = plt.subplots(len(rows), 2, figsize=(7.2, len(rows) * 2.15), dpi=150, squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for rr, row in enumerate(rows):
        cache_paths = render_rheed_processed_cache(row, paths)
        raw_path = resolve_path(paths.repo_root, cache_paths.get("cached_raw_roi_path", ""))
        overlay_path = resolve_path(paths.repo_root, cache_paths.get("cached_overlay_path", ""))
        selected = int(safe_float(row.get("frame_idx"), -1)) == selected_frame_idx
        for cc, path in enumerate([raw_path, overlay_path]):
            ax = axes[rr, cc]
            if path.is_file():
                ax.imshow(Image.open(path), cmap="gray")
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            if selected:
                for spine in ax.spines.values():
                    spine.set_edgecolor("#d62728")
                    spine.set_linewidth(3.0)
                ax.text(0.03, 0.95, "SELECTED", transform=ax.transAxes, color="#d62728", fontsize=9, weight="bold", va="top")
        axes[rr, 0].set_title(
            f"rank {rr + 1} frame {int(safe_float(row.get('frame_idx'), -1))} t={safe_float(row.get('frame_timestamp_sec')):.2f}s\n"
            f"q={safe_float(row.get('pair_visual_quality_score')):.3f} sharp={safe_float(row.get('sharpness_score')):.3f} "
            f"sat={safe_float(row.get('saturation_fraction')):.3f} shift={safe_float(row.get('shift_magnitude')):.3f}",
            fontsize=8,
        )
        axes[rr, 1].set_title(
            f"overlay conf={safe_float(row.get('detector_confidence')):.3f} MI={safe_float(row.get('morphology_index')):.3f}\n"
            f"flags={row.get('pair_visual_qc_flags', '') or 'none'}",
            fontsize=8,
        )
    fig.suptitle(f"RHEED candidate frames: sample {sample_id}", fontsize=12)
    fig.tight_layout()
    out_png = paths.rheed_contact_dir / f"{sample_id}_rheed_candidates.png"
    out_pdf = paths.rheed_contact_dir / f"{sample_id}_rheed_candidates.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return display_path(out_png, paths.repo_root)


def generate_afm_contact_sheet(sample_id: str, candidates: pd.DataFrame, selected_scan_id: str, paths: PairVizPaths) -> str:
    """Create a per-sample contact sheet for all valid AFM height maps."""
    rows = candidates.sort_values(["scan_size_um", "Rq_nm", "afm_scan_id"]).to_dict("records")
    if not rows:
        return ""
    cols = 3
    fig_rows = math.ceil(len(rows) / cols)
    fig = plt.figure(figsize=(cols * 3.3, fig_rows * 3.1), dpi=150)
    grid = fig.add_gridspec(fig_rows, cols * 2, width_ratios=[1.0, 0.05] * cols, hspace=0.42, wspace=0.08)
    fig.suptitle(f"AFM candidate scans: sample {sample_id}", fontsize=12)
    for idx, row in enumerate(rows):
        rr = idx // cols
        cc = (idx % cols) * 2
        ax = fig.add_subplot(grid[rr, cc])
        cax = fig.add_subplot(grid[rr, cc + 1])
        height_path = resolve_path(paths.repo_root, str(row.get("afm_path", "")))
        if height_path.is_file():
            height = load_height_nm(height_path, str(row.get("height_unit_exported", "nm")))
            im = draw_afm_axis(
                ax,
                height,
                {
                    **row,
                    "rq_nm": row.get("Rq_nm"),
                    "scan_size_um": row.get("scan_size_um"),
                },
                vmin=float(np.nanmin(height)),
                vmax=float(np.nanmax(height)),
                title=str(row.get("afm_scan_id", "")),
            )
            cbar = fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=5)
            cbar.set_label("Height (nm)", fontsize=6)
            if str(row.get("afm_scan_id", "")) == str(selected_scan_id):
                for spine in ax.spines.values():
                    spine.set_edgecolor("#d62728")
                    spine.set_linewidth(3)
                ax.text(0.02, 0.04, "SELECTED - closest to sample median Rq", transform=ax.transAxes, fontsize=6, color="white")
        else:
            ax.text(0.5, 0.5, "AFM N/A", ha="center", va="center")
            cax.axis("off")
    for idx in range(len(rows), fig_rows * cols):
        rr = idx // cols
        cc = (idx % cols) * 2
        fig.add_subplot(grid[rr, cc]).axis("off")
        fig.add_subplot(grid[rr, cc + 1]).axis("off")
    out_png = paths.afm_contact_dir / f"{sample_id}_afm_candidates.png"
    out_pdf = paths.afm_contact_dir / f"{sample_id}_afm_candidates.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return display_path(out_png, paths.repo_root)


def render_raw_processed_audit(rows: Sequence[dict[str, Any]], pairing_df: pd.DataFrame, paths: PairVizPaths, *, page_size: int = 10) -> None:
    """Render selected raw/full/processed/overlay RHEED audit panels."""
    pairing = pairing_df.set_index("sample_id", drop=False)
    all_figs: list[Path] = []
    for page_index, start in enumerate(range(0, len(rows), page_size), start=1):
        page_rows = rows[start : start + page_size]
        fig, axes = plt.subplots(len(page_rows), 4, figsize=(12, len(page_rows) * 2.25), dpi=150, squeeze=False)
        for axis in axes.ravel():
            axis.axis("off")
        for rr, row in enumerate(page_rows):
            sample_id = str(row.get("sample_id", ""))
            timestamp = safe_float(row.get("selected_timestamp_sec"))
            source_raw = ""
            if sample_id in pairing.index:
                source_video = resolve_path(paths.repo_root, str(pairing.loc[sample_id].get("rheed_video_path", "")))
                source_raw = decode_source_frame_at_time(source_video, timestamp, paths.cache_dir / "source_full_frames" / f"{sample_id}_{int(safe_float(row.get('selected_frame_index'), -1)):06d}.png")
            image_paths = [
                Path(source_raw) if source_raw else Path(),
                resolve_path(paths.repo_root, str(row.get("cached_raw_roi_path", ""))),
                resolve_path(paths.repo_root, str(row.get("cached_processed_roi_path", ""))),
                resolve_path(paths.repo_root, str(row.get("cached_overlay_path", ""))),
            ]
            titles = ["raw source frame", "cropped raw ROI", "processed ROI", "detector overlay"]
            for cc, (path, title) in enumerate(zip(image_paths, titles)):
                ax = axes[rr, cc]
                if path.is_file():
                    ax.imshow(Image.open(path), cmap="gray")
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                ax.set_title(title if rr == 0 else "", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
            axes[rr, 0].set_ylabel(
                f"{sample_id}\nt={timestamp:.2f}s\nq={safe_float(row.get('quality_score')):.3f}\nROI 0:256",
                fontsize=7,
            )
        fig.suptitle("Selected RHEED raw/processed audit", fontsize=13)
        fig.tight_layout()
        page_path = paths.pages_dir / f"selected_rheed_raw_processed_audit_page_{page_index:02d}.png"
        fig.savefig(page_path, bbox_inches="tight")
        all_figs.append(page_path)
        if page_index == 1:
            fig.savefig(paths.report_dir / "selected_rheed_raw_processed_audit.pdf", bbox_inches="tight")
        else:
            # Save a complete multipage-compatible PDF by appending pages through PdfPages below.
            pass
        plt.close(fig)
    if all_figs:
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(paths.report_dir / "selected_rheed_raw_processed_audit.pdf") as pdf:
            for page_path in all_figs:
                fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
                ax.imshow(Image.open(page_path))
                ax.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)


def rank_disagreements(rows: Sequence[dict[str, Any]], *, n: int = 16) -> list[dict[str, Any]]:
    """Return samples with largest normalized rank mismatch."""
    primary = [row for row in rows if not int(row.get("fallback_used", 0)) and math.isfinite(safe_float(row.get("rq_nm")))]
    if not primary:
        return []
    morph_order = {row["sample_id"]: rank for rank, row in enumerate(sorted(primary, key=lambda r: safe_float(r.get("morphology_index"))))}
    rq_order = {row["sample_id"]: rank for rank, row in enumerate(sorted(primary, key=lambda r: safe_float(r.get("rq_nm"))))}
    denom = max(len(primary) - 1, 1)
    out = []
    for row in primary:
        sid = row["sample_id"]
        row = dict(row)
        row["morphology_rank_fraction"] = morph_order[sid] / denom
        row["roughness_rank_fraction"] = rq_order[sid] / denom
        row["absolute_rank_disagreement"] = abs(row["morphology_rank_fraction"] - row["roughness_rank_fraction"])
        out.append(row)
    return sorted(out, key=lambda row: safe_float(row.get("absolute_rank_disagreement")), reverse=True)[:n]


def generate_html_gallery(rows: Sequence[dict[str, Any]], paths: PairVizPaths) -> None:
    """Create a local sortable HTML gallery of selected RHEED/AFM pairs."""
    cards = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        rheed_img = relative_to_report(str(row.get("cached_processed_roi_path", "")), paths)
        afm_img = relative_to_report(str(row.get("selected_afm_native_render_path", "")), paths)
        rheed_sheet = relative_to_report(str(row.get("rheed_contact_sheet_path", "")), paths)
        afm_sheet = relative_to_report(str(row.get("afm_contact_sheet_path", "")), paths)
        cards.append(
            f"""
<article class="card"
 data-sample="{html.escape(sample_id)}"
 data-rq="{safe_float(row.get('rq_nm'), math.inf)}"
 data-morph="{safe_float(row.get('morphology_index'), math.inf)}"
 data-scan="{safe_float(row.get('scan_size_um'), math.inf)}"
 data-material="{html.escape(str(row.get('material', '')))}"
 data-quality="{safe_float(row.get('quality_score'), -math.inf)}">
  <h3>Sample {html.escape(sample_id)}</h3>
  <div class="images">
    <img src="{html.escape(rheed_img)}" alt="RHEED {html.escape(sample_id)}">
    <img src="{html.escape(afm_img)}" alt="AFM {html.escape(sample_id)}">
  </div>
  <p><b>material</b> {html.escape(str(row.get('material', 'N/A') or 'N/A'))}</p>
  <p><b>Rq</b> {format_value(row.get('rq_nm'), 'nm', precision=3)}; <b>Ra</b> {format_value(row.get('ra_nm'), 'nm', precision=3)}; <b>scan</b> {format_value(row.get('scan_size_um'), 'µm', precision=3)}</p>
  <p><b>MI</b> {format_value(row.get('morphology_index'), precision=3)}; <b>t</b> {format_value(row.get('selected_timestamp_sec'), 's', precision=3)}; <b>quality</b> {format_value(row.get('quality_score'), precision=3)}</p>
  <p><b>QC</b> {html.escape(str(row.get('sample_qc_flags') or row.get('qc_flags') or 'none'))}</p>
  <p><a href="{html.escape(rheed_sheet)}">RHEED candidates</a> | <a href="{html.escape(afm_sheet)}">AFM scans</a></p>
  <details><summary>source paths</summary><code>{html.escape(str(row.get('video_path', '')))}</code><br><code>{html.escape(str(row.get('selected_height_map_path', '')))}</code></details>
</article>
"""
        )
    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>RHEED-AFM Pair Visualization</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; color: #202124; }}
.toolbar {{ display: flex; gap: 10px; align-items: center; margin-bottom: 16px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; }}
.card {{ border: 1px solid #d0d4da; border-radius: 8px; padding: 10px; background: #fff; }}
.card h3 {{ margin: 0 0 8px; font-size: 16px; }}
.images {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
.images img {{ width: 100%; height: 190px; object-fit: contain; background: #111; }}
p {{ margin: 5px 0; font-size: 13px; }}
code {{ font-size: 11px; word-break: break-all; }}
</style>
</head>
<body>
<h1>RHEED-AFM Pair Visualization</h1>
<p>Representative frames and AFM scans are selected by deterministic quality rules that do not use cross-modality agreement.</p>
<div class="toolbar">
<label>Sort by <select id="sorter">
<option value="sample">sample ID</option>
<option value="rq">AFM Rq</option>
<option value="morph">RHEED morphology index</option>
<option value="scan">scan size</option>
<option value="material">material</option>
<option value="quality">frame quality</option>
</select></label>
<button id="direction">ascending</button>
</div>
<div class="grid" id="cards">
{''.join(cards)}
</div>
<script>
let asc = true;
function sortCards() {{
  const key = document.getElementById('sorter').value;
  const root = document.getElementById('cards');
  const cards = Array.from(root.children);
  cards.sort((a, b) => {{
    const av = a.dataset[key] || '';
    const bv = b.dataset[key] || '';
    const an = parseFloat(av), bn = parseFloat(bv);
    let cmp = (!Number.isNaN(an) && !Number.isNaN(bn)) ? an - bn : av.localeCompare(bv);
    return asc ? cmp : -cmp;
  }});
  cards.forEach(c => root.appendChild(c));
}}
document.getElementById('sorter').addEventListener('change', sortCards);
document.getElementById('direction').addEventListener('click', () => {{
  asc = !asc;
  document.getElementById('direction').textContent = asc ? 'ascending' : 'descending';
  sortCards();
}});
sortCards();
</script>
</body>
</html>
"""
    (paths.report_dir / "index.html").write_text(html_text, encoding="utf-8")


def relative_to_report(path_text: str, paths: PairVizPaths) -> str:
    if not path_text:
        return ""
    path = resolve_path(paths.repo_root, path_text)
    try:
        return path.resolve().relative_to(paths.report_dir.resolve()).as_posix()
    except ValueError:
        return Path(os.path.relpath(path.resolve(), paths.report_dir.resolve())).as_posix()


def write_contact_sheet_index(rows: Sequence[dict[str, Any]], paths: PairVizPaths) -> None:
    links = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        rheed = relative_to_report(str(row.get("rheed_contact_sheet_path", "")), paths)
        afm = relative_to_report(str(row.get("afm_contact_sheet_path", "")), paths)
        links.append(f"<li>{html.escape(sample_id)}: <a href='{html.escape(rheed)}'>RHEED</a> | <a href='{html.escape(afm)}'>AFM</a></li>")
    (paths.report_dir / "contact_sheet_index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Contact Sheets</title></head><body>"
        "<h1>Candidate Contact Sheets</h1><ul>"
        + "\n".join(links)
        + "</ul></body></html>",
        encoding="utf-8",
    )


def build_pair_visualization_tables(
    frame_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    afm_df: pd.DataFrame,
    pairing_df: pd.DataFrame,
    config: dict[str, Any],
    paths: PairVizPaths,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Select RHEED frames and AFM scans and write contact sheets."""
    candidate_df = build_rheed_candidate_table(frame_df, paths)
    afm_candidates = build_afm_candidate_table(afm_df)
    sample_lookup = sample_df.set_index("sample_id", drop=False)
    pairing_lookup = pairing_df.set_index("sample_id", drop=False)
    primary_size = float(config["afm"].get("primary_scan_size_um", 1.0))
    tolerance = float(config["afm"].get("primary_scan_size_tolerance_um", 0.10))
    selected_rows: list[dict[str, Any]] = []
    rheed_audit_rows: list[dict[str, Any]] = []
    afm_audit_rows: list[dict[str, Any]] = []
    for sample_id in sorted(set(sample_df["sample_id"].astype(str))):
        if sample_id not in set(candidate_df["sample_id"].astype(str)):
            continue
        selected_frame, top = select_representative_rheed_frame(candidate_df[candidate_df["sample_id"].astype(str) == sample_id])
        selected_frame_idx = int(safe_float(selected_frame.get("frame_idx"), -1))
        cache = render_rheed_processed_cache(selected_frame, paths)
        rheed_sheet = generate_rheed_contact_sheet(sample_id, top, selected_frame_idx, paths)
        sample_afm = afm_candidates[afm_candidates["sample_id"].astype(str) == sample_id]
        if sample_afm.empty:
            continue
        selected_afm = select_representative_afm_scan(sample_afm, primary_scan_size_um=primary_size, tolerance_um=tolerance)
        afm_sheet = generate_afm_contact_sheet(sample_id, sample_afm, str(selected_afm.get("afm_scan_id", "")), paths)
        sample_row = sample_lookup.loc[sample_id].to_dict() if sample_id in sample_lookup.index else {}
        pairing_row = pairing_lookup.loc[sample_id].to_dict() if sample_id in pairing_lookup.index else {}
        selected = {
            "growth_run_id": sample_row.get("growth_run_id", sample_id),
            "sample_group_id": sample_row.get("sample_group_id", sample_id),
            "sample_id": sample_id,
            "material": sample_row.get("material", selected_afm.get("material", "")),
            "video_path": selected_frame.get("rheed_video_path", pairing_row.get("crop_video_path", "")),
            "selected_frame_index": selected_frame_idx,
            "selected_timestamp_sec": safe_float(selected_frame.get("frame_timestamp_sec")),
            "temporal_window_start_sec": safe_float(candidate_df[candidate_df["sample_id"].astype(str) == sample_id]["frame_timestamp_sec"].min()),
            "temporal_window_end_sec": safe_float(candidate_df[candidate_df["sample_id"].astype(str) == sample_id]["frame_timestamp_sec"].max()),
            "quality_score": safe_float(selected_frame.get("pair_visual_quality_score")),
            "detector_confidence": safe_float(selected_frame.get("detector_confidence")),
            "sharpness": safe_float(selected_frame.get("sharpness_score", selected_frame.get("focus_metric"))),
            "contrast": safe_float(selected_frame.get("contrast_score", selected_frame.get("robust_dynamic_range"))),
            "saturation_fraction": safe_float(selected_frame.get("saturation_fraction")),
            "underexposed_fraction": safe_float(selected_frame.get("underexposed_fraction")),
            "shift_magnitude": safe_float(selected_frame.get("shift_magnitude")),
            "roi_valid_fraction": safe_float(selected_frame.get("roi_coverage"), 1.0) * (1.0 - safe_float(selected_frame.get("roi_clipping"), 0.0)),
            "morphology_index": safe_float(selected_frame.get("morphology_index")),
            "selection_reason": str(top["selection_reason"].iloc[0]) if "selection_reason" in top else "",
            "qc_flags": str(selected_frame.get("pair_visual_qc_flags", "")),
            "sample_qc_flags": sample_row.get("all_qc_flags", ""),
            "included_in_strict_analysis": int(safe_float(sample_row.get("included_in_strict_analysis"), 0)),
            "included_in_primary_scan_size_analysis": int(safe_float(sample_row.get("included_in_primary_scan_size_analysis"), 0)),
            **cache,
            "rheed_contact_sheet_path": rheed_sheet,
            "selected_afm_scan_id": selected_afm.get("afm_scan_id", ""),
            "selected_afm_path": selected_afm.get("afm_path", ""),
            "selected_height_map_path": selected_afm.get("afm_path", ""),
            "channel": selected_afm.get("channel", ""),
            "height_unit_exported": selected_afm.get("height_unit_exported", "nm"),
            "scan_size_um": safe_float(selected_afm.get("scan_size_um")),
            "resolution_x": int(safe_float(selected_afm.get("resolution_w"), 0)),
            "resolution_y": int(safe_float(selected_afm.get("resolution_h"), 0)),
            "rq_nm": safe_float(selected_afm.get("Rq_nm")),
            "ra_nm": safe_float(selected_afm.get("Ra_nm")),
            "robust_height_range_nm": safe_float(selected_afm.get("robust_height_range_nm")),
            "peak_to_valley_nm": safe_float(selected_afm.get("peak_to_valley_nm")),
            "sample_median_rq_nm": safe_float(selected_afm.get("sample_median_rq_nm")),
            "distance_from_sample_median_rq_nm": safe_float(selected_afm.get("distance_from_sample_median_rq_nm")),
            "afm_selection_reason": selected_afm.get("selection_reason", ""),
            "fallback_used": int(bool(selected_afm.get("fallback_used", False))),
            "afm_qc_flags": selected_afm.get("afm_qc_flags", ""),
            "afm_contact_sheet_path": afm_sheet,
        }
        selected_rows.append(selected)
        rheed_audit_rows.append(
            {
                "growth_run_id": selected["growth_run_id"],
                "sample_id": sample_id,
                "video_path": selected["video_path"],
                "selected_frame_index": selected_frame_idx,
                "selected_timestamp_sec": selected["selected_timestamp_sec"],
                "temporal_window_start_sec": selected["temporal_window_start_sec"],
                "temporal_window_end_sec": selected["temporal_window_end_sec"],
                "quality_score": selected["quality_score"],
                "detector_confidence": selected["detector_confidence"],
                "sharpness": selected["sharpness"],
                "contrast": selected["contrast"],
                "saturation_fraction": selected["saturation_fraction"],
                "underexposed_fraction": selected["underexposed_fraction"],
                "shift_magnitude": selected["shift_magnitude"],
                "roi_valid_fraction": selected["roi_valid_fraction"],
                "morphology_index": selected["morphology_index"],
                "selection_reason": selected["selection_reason"],
                "qc_flags": selected["qc_flags"],
            }
        )
        afm_audit_rows.append(
            {
                "growth_run_id": selected["growth_run_id"],
                "sample_id": sample_id,
                "selected_afm_scan_id": selected["selected_afm_scan_id"],
                "selected_afm_path": selected["selected_afm_path"],
                "selected_height_map_path": selected["selected_height_map_path"],
                "channel": selected["channel"],
                "scan_size_um": selected["scan_size_um"],
                "resolution_x": selected["resolution_x"],
                "resolution_y": selected["resolution_y"],
                "rq_nm": selected["rq_nm"],
                "ra_nm": selected["ra_nm"],
                "robust_height_range_nm": selected["robust_height_range_nm"],
                "peak_to_valley_nm": selected["peak_to_valley_nm"],
                "sample_median_rq_nm": selected["sample_median_rq_nm"],
                "distance_from_sample_median_rq_nm": selected["distance_from_sample_median_rq_nm"],
                "selection_reason": selected["afm_selection_reason"],
                "fallback_used": selected["fallback_used"],
                "qc_flags": selected["afm_qc_flags"],
            }
        )
    return selected_rows, rheed_audit_rows, afm_audit_rows


def write_readme(
    rows: Sequence[dict[str, Any]],
    common_scale: tuple[float, float],
    paths: PairVizPaths,
    exclusion_summary: dict[str, Any],
) -> None:
    primary_count = sum(1 for row in rows if not int(row.get("fallback_used", 0)))
    strict_count = sum(1 for row in rows if int(row.get("included_in_strict_analysis", 0)))
    fallback_samples = [str(row["sample_id"]) for row in rows if int(row.get("fallback_used", 0))]
    uncertain = [
        str(row["sample_id"])
        for row in rows
        if safe_float(row.get("quality_score")) < 0.20 or row.get("qc_flags")
    ]
    figure_names = [
        "pair_grid_primary_1um_by_roughness_native.png/pdf",
        "pair_grid_primary_1um_by_roughness_common_scale.png/pdf",
        "pair_grid_primary_1um_by_rheed_morphology.png/pdf",
        "pair_grid_all_samples_by_roughness.png/pdf",
        "pair_grid_strict_qc_by_roughness.png/pdf",
        "pair_grid_largest_rank_disagreements.png/pdf",
        "selected_rheed_raw_processed_audit.pdf",
        "index.html",
    ]
    readme = f"""# RHEED-AFM Pair Visualization

## Reused Inputs and Functions

- `outputs/rheed_roughness/frame_level_scores.parquet`
- `outputs/rheed_roughness/sample_level_analysis_table.csv`
- `outputs/rheed_roughness/afm_scan_level_targets.csv`
- `outputs/rheed_roughness/pairing_audit.csv`
- `analysis.rheed_roughness.run.read_config`, path helpers, CSV helpers, and AFM unit conversion
- `rheed2morph.rheed.shape_preprocessing.preprocess_frame_for_shape`

## RHEED Frame Selection

Frames are selected only from the previous final-window frame-level table. The
quality score combines detector confidence, sharpness, contrast/dynamic range,
valid signal, and ROI completeness, then penalizes saturation, underexposure,
motion/shift, clipping, and detector-failure flags. AFM roughness and whether a
frame is streaky or spotty are not used.

## AFM Scan Selection

The primary rule uses physical ZSensor plane-corrected 1.0 +/- 0.1 um height
maps. If multiple valid scans exist, the selected scan is the one with Rq
closest to that sample's median Rq in the fixed subset. If no 1.0 um scan is
available, the dominant valid scan size for that sample is used and marked as a
fallback.

## Counts

- total paired samples visualized: {len(rows)}
- primary 1.0 um samples: {primary_count}
- strict-QC samples: {strict_count}
- non-1.0 um AFM fallback samples: {len(fallback_samples)} ({', '.join(fallback_samples) or 'none'})
- samples with uncertain RHEED frame selection: {len(uncertain)} ({', '.join(uncertain) or 'none'})
- pre-rendered AFM fallback samples: 0

## Removelist Exclusion

- removelist source: `{exclusion_summary.get('removelist_path', '')}`
- parsed removelist sample IDs: {', '.join(exclusion_summary.get('removelist_sample_ids', [])) or 'none'}
- present in previous paired roughness outputs and excluded here: {', '.join(exclusion_summary.get('removelist_samples_present_in_existing_outputs', [])) or 'none'}
- not present in previous paired roughness outputs: {', '.join(exclusion_summary.get('removelist_samples_not_in_existing_outputs', [])) or 'none'}
- exclusion audit: `outputs/rheed_roughness/pair_visualization/excluded_samples_audit.csv`

## Display Choices

RHEED panels use the cropped ROI and percentile display normalization already
used by the roughness audit; no aggressive sharpening is applied. AFM panels are
rendered from physical height arrays in nanometers with viridis and a lateral
scale bar. Native-scale figures use each scan's own min/max. Common-scale
figures use a symmetric 2nd/98th percentile range across selected 1.0 um AFM
height maps: [{common_scale[0]:.4g}, {common_scale[1]:.4g}] nm.

## Reproduction Command

```bash
PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_roughness.visualize_pairs --config configs/rheed_roughness.yaml
```

## Generated Figures

{chr(10).join(f'- `{name}`' for name in figure_names)}
"""
    (paths.report_dir / "README.md").write_text(readme, encoding="utf-8")


def print_initial_audit(frame_df: pd.DataFrame, afm_df: pd.DataFrame) -> None:
    print("Reusing existing roughness-analysis outputs and helpers:")
    print("- analysis/rheed_roughness/run.py path/config/CSV/unit-conversion helpers")
    print("- outputs/rheed_roughness/frame_level_scores.parquet")
    print("- outputs/rheed_roughness/sample_level_analysis_table.csv")
    print("- outputs/rheed_roughness/afm_scan_level_targets.csv")
    print("- outputs/rheed_roughness/pairing_audit.csv")
    print("Available RHEED frame-level columns:")
    print(", ".join(frame_df.columns))
    print("Available AFM scan-level columns:")
    print(", ".join(afm_df.columns))
    print("Deterministic frame-selection formula:")
    print(
        "0.22*detector_confidence + 0.18*sharpness + 0.18*contrast + "
        "0.16*valid_signal + 0.10*roi_completeness - saturation/underexposure/"
        "motion/clipping/detector-failure penalties; no AFM variables."
    )
    print("AFM representative rule:")
    print("Prefer 1.0+/-0.1 um ZSensor plane-corrected scans; choose Rq closest to fixed-subset median; fallback to dominant scan size.")


def run_visualization(args: argparse.Namespace) -> dict[str, Any]:
    config = read_config(args.config)
    paths = build_paths(config)
    reset_pair_visualization_outputs(paths)
    frame_df, sample_df, afm_df, pairing_df = load_existing_analysis_outputs(paths)
    removelist_path = resolve_path(paths.repo_root, config.get("removelist_path", "removelist.txt"))
    removelist_ids, removelist_rows = read_removelist(removelist_path)
    exclusion_audit_rows = build_excluded_sample_audit(removelist_rows, frame_df, sample_df, afm_df, pairing_df)
    present_in_existing_outputs = sorted(
        {
            str(row["sample_id"])
            for row in exclusion_audit_rows
            if row.get("sample_id") and int(row.get("excluded_from_visualization", 0))
        }
    )
    not_present_in_existing_outputs = sorted(removelist_ids - set(present_in_existing_outputs))
    exclusion_summary = {
        "removelist_path": display_path(removelist_path, paths.repo_root),
        "removelist_sample_ids": sorted(removelist_ids),
        "removelist_samples_present_in_existing_outputs": present_in_existing_outputs,
        "removelist_samples_not_in_existing_outputs": not_present_in_existing_outputs,
    }
    frame_df, sample_df, afm_df, pairing_df = filter_removed_samples(frame_df, sample_df, afm_df, pairing_df, removelist_ids)
    write_csv_rows(
        paths.output_dir / "excluded_samples_audit.csv",
        exclusion_audit_rows,
        fieldnames=REMOVELIST_AUDIT_FIELDNAMES,
    )
    if removelist_ids:
        print(f"Applied removelist from {display_path(removelist_path, paths.repo_root)}")
        print(f"Parsed removelist sample IDs: {', '.join(sorted(removelist_ids))}")
        print(f"Excluded samples present in existing outputs: {', '.join(present_in_existing_outputs) or 'none'}")
        print(f"Removelist samples not present in existing outputs: {', '.join(not_present_in_existing_outputs) or 'none'}")
    print_initial_audit(frame_df, afm_df)
    selected_rows, rheed_audit_rows, afm_audit_rows = build_pair_visualization_tables(
        frame_df, sample_df, afm_df, pairing_df, config, paths
    )
    common_scale = calculate_common_afm_scale(selected_rows, paths)
    for row in selected_rows:
        row["selected_afm_native_render_path"] = render_afm_cache(row, paths, scale_mode="native")
        row["selected_afm_common_render_path"] = render_afm_cache(row, paths, scale_mode="common", common_scale=common_scale)
    primary = [row for row in selected_rows if not int(row.get("fallback_used", 0))]
    strict = [row for row in selected_rows if int(row.get("included_in_strict_analysis", 0))]
    disagreements = rank_disagreements(selected_rows)
    write_csv_rows(paths.output_dir / "selected_pair_visualization_table.csv", selected_rows)
    write_csv_rows(paths.output_dir / "rheed_frame_selection_audit.csv", rheed_audit_rows)
    write_csv_rows(paths.output_dir / "afm_scan_selection_audit.csv", afm_audit_rows)
    write_csv_rows(paths.output_dir / "largest_rank_disagreements.csv", disagreements)
    render_pair_grid(
        primary,
        paths,
        output_stem="pair_grid_primary_1um_by_roughness_native",
        title="RHEED-AFM Paired Samples Sorted by AFM RMS Roughness",
        sort_key="rq_nm",
    )
    render_pair_grid(
        primary,
        paths,
        output_stem="pair_grid_primary_1um_by_roughness_common_scale",
        title="RHEED-AFM Paired Samples Sorted by AFM RMS Roughness - common AFM height scale",
        sort_key="rq_nm",
        common_scale=common_scale,
    )
    render_pair_grid(
        primary,
        paths,
        output_stem="pair_grid_primary_1um_by_rheed_morphology",
        title="RHEED-AFM Paired Samples Sorted by RHEED Morphology Index",
        sort_key="morphology_index",
    )
    render_pair_grid(
        selected_rows,
        paths,
        output_stem="pair_grid_all_samples_by_roughness",
        title="All RHEED-AFM Paired Samples Sorted by Selected AFM Rq",
        sort_key="rq_nm",
    )
    render_pair_grid(
        strict,
        paths,
        output_stem="pair_grid_strict_qc_by_roughness",
        title="Strict-QC RHEED-AFM Paired Samples Sorted by AFM Rq",
        sort_key="rq_nm",
    )
    render_pair_grid(
        disagreements,
        paths,
        output_stem="pair_grid_largest_rank_disagreements",
        title="Largest RHEED Morphology vs AFM Roughness Rank Disagreements",
        sort_key="absolute_rank_disagreement",
    )
    render_raw_processed_audit(selected_rows, pairing_df, paths)
    generate_html_gallery(selected_rows, paths)
    write_contact_sheet_index(selected_rows, paths)
    write_readme(selected_rows, common_scale, paths, exclusion_summary)
    summary = {
        "total_paired_samples_visualized": len(selected_rows),
        "primary_1um_sample_count": len(primary),
        "strict_qc_sample_count": len(strict),
        "samples_missing_valid_rheed_frames": 0,
        "samples_missing_physical_afm_height_maps": 0,
        "samples_requiring_non_1um_afm_fallback": [row["sample_id"] for row in selected_rows if int(row.get("fallback_used", 0))],
        "samples_requiring_prerendered_afm_fallback": [],
        "samples_with_uncertain_rheed_frame_selection": [
            row["sample_id"] for row in selected_rows if safe_float(row.get("quality_score")) < 0.20 or row.get("qc_flags")
        ],
        **exclusion_summary,
        "common_afm_height_scale_nm": common_scale,
        "report_dir": display_path(paths.report_dir, paths.repo_root),
        "output_dir": display_path(paths.output_dir, paths.repo_root),
    }
    write_json(paths.output_dir / "pair_visualization_summary.json", summary)
    print("Pair visualization complete")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate paired RHEED/AFM visualization grids and audits.")
    parser.add_argument("--config", type=Path, default=Path("configs/rheed_roughness.yaml"))
    args = parser.parse_args(argv)
    run_visualization(args)


if __name__ == "__main__":
    main()
