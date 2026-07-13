"""Run a data-first audit of RHEED streak/spot scores versus AFM roughness.

The analysis deliberately reuses the repository's existing RHEED shape
preprocessing and spot/streak component geometry code.  The primary scientific
quantity here is association at the independent sample/growth-run level, not
frame-level model optimization.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image
from scipy import ndimage, stats
from sklearn.feature_extraction import DictVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rheed2morph.rheed.frame_quality import (
    extract_frame_quality_features,
    finite_float,
    frame_to_gray_float32,
    score_frame_quality_rows,
)
from rheed2morph.rheed.shape_preprocessing import (
    preprocess_frame_for_shape,
    robust_rescale,
)
from rheed2morph.rheed.spot_streak_geometry import (
    FRAME_SHAPE_FEATURE_NAMES,
    colorize_component_overlay,
    extract_components_and_frame_features,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RE = re.compile(r"N?(\d{4})")
ROUGHNESS_KEYWORDS = {
    "rms_roughness",
    "roughness_rms",
    "rq",
    "sq",
    "ra",
    "roughness",
    "z_min",
    "z_max",
    "height_min",
    "height_max",
    "colorbar_min",
    "colorbar_max",
    "height_range",
    "peak_to_valley",
    "scan_size",
    "scan_size_x",
    "scan_size_y",
    "height_unit",
    "z_unit",
    "channel",
}


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    outputs_dir: Path
    reports_dir: Path
    assets_dir: Path
    figures_dir: Path
    manual_review_dir: Path


def resolve_path(repo_root: Path, raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path


def display_path(path: Path, repo_root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def read_config(path: Path) -> dict[str, Any]:
    """Read the JSON-compatible YAML config without requiring PyYAML."""
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} is parsed as JSON-compatible YAML in this environment. "
            "Use JSON syntax or install a YAML parser."
        ) from exc


def make_paths(config: dict[str, Any], *, smoke: bool = False) -> Paths:
    repo_root = resolve_path(REPO_ROOT, config.get("repo_root", ".")).resolve()
    outputs = resolve_path(repo_root, config["outputs_dir"])
    reports = resolve_path(repo_root, config["reports_dir"])
    if smoke:
        outputs = outputs.with_name(outputs.name + "_smoke")
        reports = reports.with_name(reports.name + "_smoke")
    assets = reports / "assets"
    figures = reports / "figures"
    manual = reports / "manual_review"
    for path in (outputs, reports, assets, figures, manual):
        path.mkdir(parents=True, exist_ok=True)
    return Paths(repo_root, outputs, reports, assets, figures, manual)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.10g}"
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, list | tuple):
        return ";".join(str(item) for item in value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer | np.floating):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_parquet_or_note(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write parquet when pyarrow is available; otherwise write a reproducible note.

    The CSV counterpart is always written by the caller.  We avoid fabricating a
    parquet file when the local environment lacks a parquet engine.
    """
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
    except Exception as exc:  # pragma: no cover - depends on optional pyarrow.
        note = path.with_suffix(path.suffix + ".unavailable.txt")
        note.write_text(
            "Parquet export was skipped because no pandas parquet engine was available.\n"
            f"Original error: {type(exc).__name__}: {exc}\n"
            "The CSV file with the same stem is the authoritative fallback.\n",
            encoding="utf-8",
        )


def sample_number(value: str | Path) -> str:
    match = SAMPLE_RE.search(str(value))
    return match.group(1) if match else ""


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def discover_crop_videos(root: Path) -> tuple[dict[str, Path], list[dict[str, str]]]:
    videos: dict[str, Path] = {}
    issues: list[dict[str, str]] = []
    if not root.is_dir():
        return videos, [{"sample_id": "", "issue": "missing_crop_video_root", "path": root.as_posix()}]
    for sample_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        sid = sample_number(sample_dir.name)
        if not sid:
            continue
        candidates = sorted((sample_dir / "videos").glob("*raw_crop*.*"))
        candidates = [path for path in candidates if path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}]
        if not candidates:
            issues.append({"sample_id": sid, "issue": "missing_raw_crop_video", "path": sample_dir.as_posix()})
            continue
        if sid in videos:
            issues.append({"sample_id": sid, "issue": "duplicate_crop_video_folder", "path": sample_dir.as_posix()})
            continue
        videos[sid] = candidates[0]
        if len(candidates) > 1:
            issues.append(
                {
                    "sample_id": sid,
                    "issue": "multiple_crop_videos_kept_first",
                    "path": candidates[0].as_posix(),
                    "extra": ";".join(path.as_posix() for path in candidates[1:]),
                }
            )
    return videos, issues


def validate_unique_pairing(rows: Sequence[dict[str, Any]], key_fields: Sequence[str], value_field: str) -> list[dict[str, Any]]:
    """Return duplicate/ambiguous pairings instead of silently choosing one."""
    values_by_key: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in rows:
        key = tuple(row.get(field, "") for field in key_fields)
        value = str(row.get(value_field, ""))
        if value:
            values_by_key[key].add(value)
    issues = []
    for key, values in sorted(values_by_key.items()):
        if len(values) > 1:
            issue = {field: key[index] for index, field in enumerate(key_fields)}
            issue["ambiguous_field"] = value_field
            issue["candidate_count"] = len(values)
            issue["candidates"] = ";".join(sorted(values))
            issues.append(issue)
    return issues


def flatten_json_keys(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_json_keys(child, child_prefix)
    elif isinstance(value, list):
        yield prefix, f"list[{len(value)}]"
        for index, child in enumerate(value[:3]):
            yield from flatten_json_keys(child, f"{prefix}[{index}]")
    else:
        yield prefix, value


def discover_json_schema(metadata_roots: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key_counter: Counter[str] = Counter()
    roughness_hits: Counter[str] = Counter()
    examples: dict[str, Any] = {}
    file_count = 0
    for root in metadata_roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*metadata.json")):
            file_count += 1
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for key, value in flatten_json_keys(payload):
                key_counter[key] += 1
                lowered = key.lower().replace(".", "_")
                if any(token in lowered for token in ROUGHNESS_KEYWORDS):
                    roughness_hits[key] += 1
                    examples.setdefault(key, value)
    rows = [
        {
            "json_key": key,
            "count": count,
            "roughness_candidate": int(key in roughness_hits),
            "example": str(examples.get(key, ""))[:200],
        }
        for key, count in key_counter.most_common()
    ]
    summary = {
        "metadata_file_count": file_count,
        "unique_json_key_count": len(key_counter),
        "roughness_candidate_keys": dict(roughness_hits.most_common()),
    }
    return rows, summary


def convert_height_to_nm(values: np.ndarray, unit: str | None) -> tuple[np.ndarray, str]:
    """Convert a physical AFM height map to nanometers and report unit status."""
    unit_text = str(unit or "").strip().lower().replace("µ", "u")
    if unit_text in {"nm", "nanometer", "nanometers"}:
        return np.asarray(values, dtype=np.float64), "ok"
    if unit_text in {"um", "micron", "microns", "micrometer", "micrometers"}:
        return np.asarray(values, dtype=np.float64) * 1000.0, "ok"
    if unit_text in {"m", "meter", "meters"}:
        return np.asarray(values, dtype=np.float64) * 1e9, "ok"
    if unit_text in {"a", "angstrom", "angstroms"}:
        return np.asarray(values, dtype=np.float64) * 0.1, "ok"
    return np.asarray(values, dtype=np.float64), "unknown_height_unit"


def metadata_for_height_path(path: Path) -> Path | None:
    candidates = []
    if path.name.endswith("_plane_corrected.npy"):
        base = path.name.removesuffix("_plane_corrected.npy")
        candidates.append(path.with_name(f"{base}_plane_corrected_metadata.json"))
        candidates.append(path.with_name(f"{base}_metadata.json"))
    else:
        candidates.append(path.with_name(f"{path.stem}_metadata.json"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def scan_id_from_path(path: Path) -> str:
    name = path.stem
    return name.removesuffix("_plane_corrected").removesuffix("_height")


def load_descriptor_lookup(repo_root: Path) -> dict[str, dict[str, str]]:
    path = repo_root / "data" / "afm_descriptor_reconstruction" / "afm_descriptors.csv"
    if not path.is_file():
        return {}
    lookup = {}
    for row in read_csv_rows(path):
        lookup[display_path(resolve_path(repo_root, row.get("afm_path", "")), repo_root)] = row
    return lookup


def radial_psd_slope(height_nm: np.ndarray) -> float:
    work = np.nan_to_num(height_nm - np.nanmean(height_nm), nan=0.0)
    if work.size == 0 or float(np.std(work)) <= 1e-12:
        return math.nan
    power = np.abs(np.fft.fftshift(np.fft.fft2(work))) ** 2
    yy, xx = np.indices(work.shape)
    cy = (work.shape[0] - 1) / 2.0
    cx = (work.shape[1] - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).ravel()
    values = power.ravel()
    mask = (radius > 2) & np.isfinite(values) & (values > 0)
    if mask.sum() < 20:
        return math.nan
    x = np.log10(radius[mask])
    y = np.log10(values[mask])
    return safe_float(np.polyfit(x, y, 1)[0])


def autocorr_length_1d(values: np.ndarray) -> float:
    centered = np.nan_to_num(values - np.nanmean(values), nan=0.0)
    if centered.size < 4 or float(np.std(centered)) <= 1e-12:
        return math.nan
    corr = np.correlate(centered, centered, mode="full")[centered.size - 1 :]
    corr = corr / max(float(corr[0]), 1e-12)
    below = np.flatnonzero(corr < math.exp(-1))
    return float(below[0]) if below.size else float(centered.size)


def compute_afm_descriptors(height_nm: np.ndarray) -> dict[str, float]:
    values = np.asarray(height_nm, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: math.nan for key in ["Rq_nm", "Ra_nm", "robust_height_range_nm", "peak_to_valley_nm"]}
    centered = values - float(np.mean(values))
    rq = float(np.sqrt(np.mean(centered * centered)))
    ra = float(np.mean(np.abs(centered)))
    p01, p05, p25, p50, p75, p95, p99 = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    std = max(float(np.std(centered)), 1e-12)
    skew = float(np.mean((centered / std) ** 3))
    kurt = float(np.mean((centered / std) ** 4))
    threshold = p75 + 0.5 * (p75 - p25)
    coverage = float(np.mean(values > threshold))
    return {
        "Rq_nm": rq,
        "Ra_nm": ra,
        "robust_height_range_nm": float(p95 - p05),
        "peak_to_valley_nm": float(np.max(values) - np.min(values)),
        "p01_nm": float(p01),
        "p05_nm": float(p05),
        "p25_nm": float(p25),
        "p50_nm": float(p50),
        "p75_nm": float(p75),
        "p95_nm": float(p95),
        "p99_nm": float(p99),
        "skewness": skew,
        "kurtosis": kurt,
        "coverage_fraction": coverage,
    }


def extract_afm_targets(
    repo_root: Path,
    candidate_rows: Sequence[dict[str, str]],
    config: dict[str, Any],
    paths: Paths,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    descriptor_lookup = load_descriptor_lookup(repo_root)
    seen_paths: set[str] = set()
    output_rows: list[dict[str, Any]] = []
    prefer_plane = bool(config["afm"].get("prefer_plane_corrected", True))
    for row in candidate_rows:
        if row.get("is_physical_height_map", "").lower() != "true":
            continue
        if prefer_plane and row.get("is_plane_corrected", "").lower() != "true":
            continue
        afm_path = resolve_path(repo_root, row["afm_path"])
        rel = display_path(afm_path, repo_root)
        if rel in seen_paths or not afm_path.is_file():
            continue
        seen_paths.add(rel)
        metadata_path = metadata_for_height_path(afm_path)
        metadata: dict[str, Any] = {}
        if metadata_path and metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                metadata = {}
        unit = metadata.get("height_unit_exported") or metadata.get("height_unit_original") or config["afm"].get("height_unit_default", "nm")
        try:
            array = np.load(afm_path)
        except Exception as exc:
            output_rows.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "afm_path": rel,
                    "target_status": "load_failed",
                    "target_error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        height_nm, unit_status = convert_height_to_nm(np.asarray(array), str(unit))
        descriptors = compute_afm_descriptors(height_nm)
        if height_nm.ndim == 2:
            descriptors["autocorrelation_length_x_px"] = autocorr_length_1d(np.nanmean(height_nm, axis=0))
            descriptors["autocorrelation_length_y_px"] = autocorr_length_1d(np.nanmean(height_nm, axis=1))
            descriptors["radial_psd_slope"] = radial_psd_slope(height_nm)
        else:
            descriptors["autocorrelation_length_x_px"] = math.nan
            descriptors["autocorrelation_length_y_px"] = math.nan
            descriptors["radial_psd_slope"] = math.nan
        metadata_rq = safe_float(metadata.get("height_std_nm"))
        descriptor_row = descriptor_lookup.get(rel, {})
        descriptor_rq = safe_float(descriptor_row.get("Rq"))
        scan_size = safe_float(row.get("scan_size_um"))
        resolution_h = int(safe_float(row.get("resolution_h"), 0) or 0)
        resolution_w = int(safe_float(row.get("resolution_w"), 0) or 0)
        scan_id = scan_id_from_path(afm_path)
        output_rows.append(
            {
                "sample_id": row.get("sample_id", ""),
                "growth_run_id": row.get("group_id", row.get("sample_id", "")),
                "sample_group_id": row.get("group_id", row.get("sample_id", "")),
                "material": row.get("material", ""),
                "afm_scan_id": scan_id,
                "afm_path": rel,
                "metadata_path": display_path(metadata_path, repo_root) if metadata_path else "",
                "scan_size_um": scan_size,
                "afm_resolution": f"{resolution_h}x{resolution_w}" if resolution_h and resolution_w else "unknown",
                "resolution_h": resolution_h,
                "resolution_w": resolution_w,
                "channel": row.get("channel_name", metadata.get("primary_channel", "")),
                "height_unit_original": metadata.get("height_unit_original", unit),
                "height_unit_exported": metadata.get("height_unit_exported", unit),
                "height_unit_status": unit_status,
                "metadata_Rq_nm": metadata_rq,
                "descriptor_Rq_nm": descriptor_rq,
                "metadata_minus_recomputed_Rq_nm": metadata_rq - descriptors["Rq_nm"]
                if math.isfinite(metadata_rq)
                else math.nan,
                "descriptor_minus_recomputed_Rq_nm": descriptor_rq - descriptors["Rq_nm"]
                if math.isfinite(descriptor_rq)
                else math.nan,
                "target_status": "ok" if unit_status == "ok" else unit_status,
                "preprocessing": "plane_corrected_height_map",
                **descriptors,
            }
        )

    summary = {
        "afm_scan_level_target_count": len(output_rows),
        "afm_samples_with_targets": len({row["sample_id"] for row in output_rows if row.get("target_status") in {"ok", "unknown_height_unit"}}),
        "height_unit_status_counts": dict(Counter(row.get("height_unit_status", "") for row in output_rows)),
        "scan_size_counts": dict(Counter(round(safe_float(row.get("scan_size_um")), 3) for row in output_rows if math.isfinite(safe_float(row.get("scan_size_um"))))),
        "resolution_counts": dict(Counter(row.get("afm_resolution", "") for row in output_rows)),
    }
    return output_rows, summary


def aggregate_numeric(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if array.size == 0:
        return {"median": math.nan, "iqr": math.nan, "mad": math.nan, "min": math.nan, "max": math.nan, "n": 0}
    median = float(np.median(array))
    return {
        "median": median,
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "mad": float(np.median(np.abs(array - median))),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "n": int(array.size),
    }


def aggregate_afm_by_sample(afm_rows: Sequence[dict[str, Any]], config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    primary_size = float(config["afm"]["primary_scan_size_um"])
    tol = float(config["afm"]["primary_scan_size_tolerance_um"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in afm_rows:
        if row.get("target_status") not in {"ok", "unknown_height_unit"}:
            continue
        grouped[str(row["sample_id"])].append(row)
    sample_targets: dict[str, dict[str, Any]] = {}
    for sample_id, rows in grouped.items():
        primary_rows = [
            row
            for row in rows
            if math.isfinite(safe_float(row.get("scan_size_um"))) and abs(safe_float(row.get("scan_size_um")) - primary_size) <= tol
        ]
        all_rq = aggregate_numeric([safe_float(row.get("Rq_nm")) for row in rows])
        primary_rq = aggregate_numeric([safe_float(row.get("Rq_nm")) for row in primary_rows])
        primary_ra = aggregate_numeric([safe_float(row.get("Ra_nm")) for row in primary_rows])
        all_ra = aggregate_numeric([safe_float(row.get("Ra_nm")) for row in rows])
        robust = aggregate_numeric([safe_float(row.get("robust_height_range_nm")) for row in primary_rows or rows])
        ptv = aggregate_numeric([safe_float(row.get("peak_to_valley_nm")) for row in primary_rows or rows])
        scan_sizes = sorted({round(safe_float(row.get("scan_size_um")), 6) for row in rows if math.isfinite(safe_float(row.get("scan_size_um")))})
        representative = min(primary_rows or rows, key=lambda row: abs(safe_float(row.get("Rq_nm")) - (primary_rq["median"] if primary_rows else all_rq["median"])))
        sample_targets[sample_id] = {
            "sample_id": sample_id,
            "number_of_afm_scans": len(rows),
            "number_of_primary_scan_size_afm_scans": len(primary_rows),
            "afm_Rq_median_nm": primary_rq["median"] if primary_rows else math.nan,
            "afm_Rq_iqr_nm": primary_rq["iqr"] if primary_rows else math.nan,
            "afm_Rq_mad_nm": primary_rq["mad"] if primary_rows else math.nan,
            "afm_Rq_min_nm": primary_rq["min"] if primary_rows else math.nan,
            "afm_Rq_max_nm": primary_rq["max"] if primary_rows else math.nan,
            "afm_Ra_median_nm": primary_ra["median"] if primary_rows else math.nan,
            "afm_Ra_iqr_nm": primary_ra["iqr"] if primary_rows else math.nan,
            "afm_Rq_all_scan_sizes_median_nm": all_rq["median"],
            "afm_Ra_all_scan_sizes_median_nm": all_ra["median"],
            "robust_height_range_median_nm": robust["median"],
            "peak_to_valley_median_nm": ptv["median"],
            "afm_scan_sizes_um": ";".join(f"{size:.6g}" for size in scan_sizes),
            "primary_scan_size_um": primary_size,
            "has_primary_scan_size": bool(primary_rows),
            "representative_afm_scan_id": representative.get("afm_scan_id", ""),
            "representative_afm_path": representative.get("afm_path", ""),
            "representative_metadata_path": representative.get("metadata_path", ""),
            "afm_resolution": representative.get("afm_resolution", ""),
            "height_unit_statuses": ";".join(sorted({str(row.get("height_unit_status", "")) for row in rows})),
        }
    summary = {
        "sample_level_target_count": len(sample_targets),
        "samples_with_primary_scan_size": sum(1 for row in sample_targets.values() if row["has_primary_scan_size"]),
        "primary_scan_size_um": primary_size,
        "primary_scan_size_tolerance_um": tol,
    }
    return sample_targets, summary


def compute_morphology_scores(features: dict[str, Any], epsilon: float = 1e-8) -> dict[str, float]:
    """Compute the frozen spotty-to-streaky index from existing component features."""
    round_count = finite_float(features.get("round_spot_count", 0.0))
    elongated_count = finite_float(features.get("elongated_spot_count", 0.0))
    diffuse_count = finite_float(features.get("diffuse_blob_count", 0.0))
    horizontal_count = finite_float(features.get("horizontal_bar_count", 0.0))
    vertical_count = finite_float(features.get("vertical_streak_count", 0.0))
    total = max(finite_float(features.get("total_component_count", 0.0)), 1.0)
    bar_like = finite_float(features.get("bar_like_score", 0.0))
    spottiness = round_count + 0.60 * elongated_count + 0.25 * diffuse_count
    streakiness = horizontal_count + vertical_count + 0.60 * elongated_count + bar_like * total
    morphology_index = spottiness / max(spottiness + streakiness + epsilon, epsilon)
    return {
        "raw_spottiness": float(spottiness),
        "raw_streakiness": float(streakiness),
        "morphology_index": float(np.clip(morphology_index, 0.0, 1.0)),
    }


def video_metadata(path: Path) -> dict[str, float]:
    reader = imageio.get_reader(str(path), "ffmpeg")
    try:
        meta = reader.get_meta_data()
    finally:
        reader.close()
    fps = safe_float(meta.get("fps"), 30.0)
    duration = safe_float(meta.get("duration"), math.nan)
    nframes = safe_float(meta.get("nframes"), math.nan)
    if not math.isfinite(nframes) or nframes > 1e8:
        nframes = duration * fps if math.isfinite(duration) and math.isfinite(fps) else math.nan
    width, height = meta.get("source_size") or meta.get("size") or (math.nan, math.nan)
    return {
        "fps": fps,
        "duration_sec": duration,
        "frame_count": float(nframes),
        "width": safe_float(width),
        "height": safe_float(height),
    }


def final_window_indices(meta: dict[str, float], *, fraction: float, max_frames: int) -> list[int]:
    nframes = int(meta.get("frame_count", 0)) if math.isfinite(meta.get("frame_count", math.nan)) else 0
    if nframes <= 0:
        return []
    start = max(0, int(math.floor(nframes * max(0.0, 1.0 - fraction))))
    stop = max(start, nframes - 1)
    count = max(1, min(max_frames, stop - start + 1))
    return sorted({int(round(value)) for value in np.linspace(start, stop, count)})


def read_video_frames(path: Path, indices: Sequence[int]) -> list[tuple[int, np.ndarray]]:
    frames: list[tuple[int, np.ndarray]] = []
    reader = imageio.get_reader(str(path), "ffmpeg")
    try:
        for index in indices:
            try:
                frame = reader.get_data(int(index))
            except Exception:
                continue
            frames.append((int(index), frame))
    finally:
        reader.close()
    return frames


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.asarray(np.clip(robust_rescale(arr) * 255.0, 0, 255), dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(path)
    else:
        arr = np.asarray(np.clip(arr * 255.0 if arr.dtype.kind == "f" else arr, 0, 255), dtype=np.uint8)
        Image.fromarray(arr).save(path)


def centroid_from_frame(gray: np.ndarray) -> tuple[float, float]:
    values = np.asarray(gray, dtype=np.float64)
    weights = np.clip(values - np.percentile(values, 60), 0.0, None)
    total = float(weights.sum())
    if total <= 1e-12:
        return math.nan, math.nan
    yy, xx = np.indices(values.shape)
    return float((xx * weights).sum() / total / max(values.shape[1] - 1, 1)), float((yy * weights).sum() / total / max(values.shape[0] - 1, 1))


def process_rheed_video(
    sample_id: str,
    video_path: Path,
    config: dict[str, Any],
    paths: Paths,
    *,
    smoke: bool = False,
    use_cache: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache_path = paths.outputs_dir / "cache" / "frame_scores" / f"{sample_id}.json"
    if use_cache and cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return payload["frame_rows"], payload["video_summary"]

    rheed_cfg = config["rheed"]
    max_frames = int(rheed_cfg["smoke_frames_per_video"] if smoke else rheed_cfg["max_primary_frames_per_video"])
    meta = video_metadata(video_path)
    indices = final_window_indices(meta, fraction=float(rheed_cfg["primary_window_fraction"]), max_frames=max_frames)
    frame_records = read_video_frames(video_path, indices)
    quality_rows_raw: list[dict[str, Any]] = []
    raw_frames: list[tuple[int, np.ndarray, np.ndarray]] = []
    for frame_idx, frame in frame_records:
        gray = frame_to_gray_float32(frame)
        quality = extract_frame_quality_features(gray)
        quality.update({"frame_idx": frame_idx})
        quality_rows_raw.append(quality)
        raw_frames.append((frame_idx, frame, gray))
    scored_quality_rows = score_frame_quality_rows(quality_rows_raw)
    quality_by_frame = {int(row["frame_idx"]): row for row in scored_quality_rows}

    frame_rows: list[dict[str, Any]] = []
    prev_centroid: tuple[float, float] | None = None
    reference_centroid: tuple[float, float] | None = None
    for frame_idx, frame, gray in raw_frames:
        processed = preprocess_frame_for_shape(gray, image_size=int(rheed_cfg["image_size"]))
        components, geometry = extract_components_and_frame_features(
            soft_mask=processed.channels["soft_spot_streak_mask"],
            enhanced_image=processed.channels["log_bgsub"],
            artifact_mask=processed.artifact_mask,
        )
        scores = compute_morphology_scores(geometry, epsilon=float(rheed_cfg.get("epsilon", 1e-8)))
        quality = quality_by_frame.get(frame_idx, {})
        centroid = centroid_from_frame(processed.channels["pclip_norm"])
        if reference_centroid is None:
            reference_centroid = centroid
        shift_x = centroid[0] - reference_centroid[0] if all(math.isfinite(v) for v in centroid + reference_centroid) else math.nan
        shift_y = centroid[1] - reference_centroid[1] if all(math.isfinite(v) for v in centroid + reference_centroid) else math.nan
        motion = (
            math.hypot(centroid[0] - prev_centroid[0], centroid[1] - prev_centroid[1])
            if prev_centroid is not None and all(math.isfinite(v) for v in centroid + prev_centroid)
            else math.nan
        )
        prev_centroid = centroid

        frame_dir = paths.assets_dir / "rheed_frames" / sample_id
        raw_path = frame_dir / f"frame_{frame_idx:06d}_raw.png"
        overlay_path = frame_dir / f"frame_{frame_idx:06d}_overlay.png"
        if not use_cache or not raw_path.is_file():
            save_image(raw_path, processed.raw_gray)
            overlay = colorize_component_overlay(
                processed.channels["pclip_norm"],
                components,
                processed.channels["soft_spot_streak_mask"],
            )
            save_image(overlay_path, overlay)

        timestamp = frame_idx / meta["fps"] if math.isfinite(meta.get("fps", math.nan)) and meta["fps"] > 0 else math.nan
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "growth_run_id": sample_id,
            "sample_group_id": sample_id,
            "rheed_video_id": video_path.stem,
            "rheed_video_path": display_path(video_path, paths.repo_root),
            "frame_idx": frame_idx,
            "frame_timestamp_sec": timestamp,
            "primary_window": 1,
            "roi_x0": 0,
            "roi_y0": 0,
            "roi_x1": int(processed.raw_gray.shape[1]),
            "roi_y1": int(processed.raw_gray.shape[0]),
            "roi_coverage": 1.0,
            "roi_clipping": 0.0,
            "raw_frame_path": display_path(raw_path, paths.repo_root),
            "overlay_path": display_path(overlay_path, paths.repo_root),
            "raw_streakiness": scores["raw_streakiness"],
            "raw_spottiness": scores["raw_spottiness"],
            "morphology_index": scores["morphology_index"],
            "detector_confidence": finite_float(geometry.get("mask_confidence", 0.0)),
            "pattern_centroid_x": centroid[0],
            "pattern_centroid_y": centroid[1],
            "direct_beam_centroid_x": centroid[0],
            "direct_beam_centroid_y": centroid[1],
            "frame_to_reference_translation_x": shift_x,
            "frame_to_reference_translation_y": shift_y,
            "frame_to_frame_motion_magnitude": motion,
            "image_resolution": f"{int(meta['height'])}x{int(meta['width'])}",
            "video_fps": meta["fps"],
            "video_duration_sec": meta["duration_sec"],
        }
        row.update({f"geom_{key}": finite_float(geometry.get(key, 0.0)) for key in FRAME_SHAPE_FEATURE_NAMES})
        row.update({key: quality.get(key, "") for key in quality})
        row.update(
            {
                "mean_intensity": finite_float(quality.get("mean_intensity", processed.audit_features["raw_mean"])),
                "median_intensity": finite_float(quality.get("p50", 0.0)),
                "intensity_std": finite_float(quality.get("std_intensity", processed.audit_features["raw_std"])),
                "robust_dynamic_range": finite_float(quality.get("dynamic_range_p99_p01", 0.0)),
                "saturation_fraction": finite_float(quality.get("saturated_pixel_fraction", 0.0)),
                "underexposed_fraction": finite_float(quality.get("dark_pixel_fraction", 0.0)),
                "focus_metric": finite_float(quality.get("laplacian_variance", 0.0)),
                "background_intensity": finite_float(quality.get("edge_to_center_ratio", 0.0)),
                "background_gradient": finite_float(
                    abs(finite_float(quality.get("left_edge_mean", 0.0)) - finite_float(quality.get("right_edge_mean", 0.0)))
                ),
                "camera_tool_id": "unknown",
                "batch_id": "unknown",
            }
        )
        frame_rows.append(row)

    video_summary = aggregate_video_scores(sample_id, video_path, frame_rows, meta, config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(json_ready({"frame_rows": frame_rows, "video_summary": video_summary}), indent=2), encoding="utf-8")
    return frame_rows, video_summary


def agg_values(rows: Sequence[dict[str, Any]], key: str, *, valid_only: bool = False) -> np.ndarray:
    values = []
    for row in rows:
        if valid_only and not bool(row.get("frame_valid", True)):
            continue
        value = safe_float(row.get(key))
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def median_or_nan(rows: Sequence[dict[str, Any]], key: str) -> float:
    arr = agg_values(rows, key)
    return float(np.median(arr)) if arr.size else math.nan


def iqr_or_nan(rows: Sequence[dict[str, Any]], key: str) -> float:
    arr = agg_values(rows, key)
    return float(np.percentile(arr, 75) - np.percentile(arr, 25)) if arr.size else math.nan


def mad_or_nan(rows: Sequence[dict[str, Any]], key: str) -> float:
    arr = agg_values(rows, key)
    if arr.size == 0:
        return math.nan
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def trimmed_mean(values: np.ndarray, trim_fraction: float = 0.10) -> float:
    if values.size == 0:
        return math.nan
    ordered = np.sort(values)
    trim = int(ordered.size * trim_fraction)
    if trim and ordered.size > 2 * trim:
        ordered = ordered[trim:-trim]
    return float(np.mean(ordered))


def aggregate_video_scores(
    sample_id: str,
    video_path: Path,
    frame_rows: Sequence[dict[str, Any]],
    meta: dict[str, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    qc = config["qc"]
    valid_rows = []
    for row in frame_rows:
        reasons = []
        if finite_float(row.get("saturation_fraction", 0.0)) > float(qc["max_median_saturation_fraction"]) * 2:
            reasons.append("excessive_saturation")
        if finite_float(row.get("underexposed_fraction", 0.0)) > float(qc["max_median_underexposed_fraction"]) * 1.25:
            reasons.append("severe_underexposure")
        if finite_float(row.get("robust_dynamic_range", 0.0)) < float(qc["min_median_dynamic_range"]):
            reasons.append("insufficient_signal")
        if finite_float(row.get("detector_confidence", 0.0)) < float(qc["min_median_detector_confidence"]) * 0.5:
            reasons.append("detector_failure")
        row["frame_qc_reasons"] = ";".join(reasons)
        row["frame_valid"] = int(not reasons)
        if not reasons:
            valid_rows.append(row)
    source = valid_rows if valid_rows else list(frame_rows)
    morph = agg_values(source, "morphology_index")
    summary = {
        "sample_id": sample_id,
        "growth_run_id": sample_id,
        "sample_group_id": sample_id,
        "rheed_video_id": video_path.stem,
        "rheed_video_path": display_path(video_path),
        "video_frame_count": meta.get("frame_count", math.nan),
        "video_duration_sec": meta.get("duration_sec", math.nan),
        "video_fps": meta.get("fps", math.nan),
        "video_resolution": f"{int(meta.get('height', 0))}x{int(meta.get('width', 0))}",
        "number_of_rheed_videos": 1,
        "number_of_analyzed_frames": len(frame_rows),
        "number_of_valid_rheed_frames": len(valid_rows),
        "valid_frame_fraction": len(valid_rows) / max(len(frame_rows), 1),
        "median_morphology_index": float(np.median(morph)) if morph.size else math.nan,
        "trimmed_mean_morphology_index": trimmed_mean(morph),
        "morphology_index_iqr": iqr_or_nan(source, "morphology_index"),
        "morphology_index_mad": mad_or_nan(source, "morphology_index"),
        "morphology_index_p10": float(np.percentile(morph, 10)) if morph.size else math.nan,
        "morphology_index_p90": float(np.percentile(morph, 90)) if morph.size else math.nan,
        "median_streakiness": median_or_nan(source, "raw_streakiness"),
        "median_spottiness": median_or_nan(source, "raw_spottiness"),
        "temporal_instability": mad_or_nan(source, "morphology_index"),
        "cycle_to_cycle_variability": math.nan,
        "rotation_period_sec": math.nan,
        "rotation_period_source": "not_available",
        "mean_intensity_median": median_or_nan(source, "mean_intensity"),
        "saturation_fraction_median": median_or_nan(source, "saturation_fraction"),
        "underexposed_fraction_median": median_or_nan(source, "underexposed_fraction"),
        "robust_dynamic_range_median": median_or_nan(source, "robust_dynamic_range"),
        "focus_metric_median": median_or_nan(source, "focus_metric"),
        "pattern_centroid_x_median": median_or_nan(source, "pattern_centroid_x"),
        "pattern_centroid_y_median": median_or_nan(source, "pattern_centroid_y"),
        "frame_shift_rms": frame_shift_rms(source),
        "detector_confidence_median": median_or_nan(source, "detector_confidence"),
        "representative_raw_frame_path": representative_frame_path(source, "raw_frame_path"),
        "representative_overlay_path": representative_frame_path(source, "overlay_path"),
        "aggregation_rule": f"final_{config['rheed']['primary_window_fraction']:.0%}_up_to_{config['rheed']['max_primary_frames_per_video']}_frames",
        "contributing_frame_indices": ";".join(str(row.get("frame_idx", "")) for row in source),
    }
    return summary


def frame_shift_rms(rows: Sequence[dict[str, Any]]) -> float:
    shifts = []
    for row in rows:
        x = safe_float(row.get("frame_to_reference_translation_x"))
        y = safe_float(row.get("frame_to_reference_translation_y"))
        if math.isfinite(x) and math.isfinite(y):
            shifts.append(x * x + y * y)
    return float(math.sqrt(np.mean(shifts))) if shifts else math.nan


def representative_frame_path(rows: Sequence[dict[str, Any]], key: str) -> str:
    if not rows:
        return ""
    ordered = sorted(rows, key=lambda row: abs(safe_float(row.get("morphology_index"), 0.5) - median_or_nan(rows, "morphology_index")))
    return str(ordered[0].get(key, ""))


def run_rheed_extraction(
    sample_rows: Sequence[dict[str, str]],
    crop_videos: dict[str, Path],
    config: dict[str, Any],
    paths: Paths,
    *,
    smoke: bool = False,
    use_cache: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frame_rows: list[dict[str, Any]] = []
    video_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for manifest_row in sample_rows:
        sample_id = manifest_row["sample_id"]
        video_path = crop_videos.get(sample_id)
        if video_path is None:
            failures.append({"sample_id": sample_id, "failure": "missing_crop_video"})
            continue
        try:
            rows, summary = process_rheed_video(sample_id, video_path, config, paths, smoke=smoke, use_cache=use_cache)
        except Exception as exc:
            failures.append({"sample_id": sample_id, "failure": f"{type(exc).__name__}: {exc}", "video_path": video_path.as_posix()})
            continue
        frame_rows.extend(rows)
        video_rows.append(summary)
    return frame_rows, video_rows, failures


def build_pairing_audit(
    repo_root: Path,
    representative_rows: Sequence[dict[str, str]],
    candidate_rows: Sequence[dict[str, str]],
    crop_videos: dict[str, Path],
    crop_issues: Sequence[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        if row.get("is_physical_height_map", "").lower() == "true" and row.get("is_plane_corrected", "").lower() == "true":
            candidate_by_sample[row.get("sample_id", "")].append(row)
    audit_rows = []
    for row in representative_rows:
        sample_id = row["sample_id"]
        afm_candidates = candidate_by_sample.get(sample_id, [])
        metadata_path = metadata_for_height_path(resolve_path(repo_root, row["afm_path"]))
        audit_rows.append(
            {
                "growth_run_id": row.get("group_id", sample_id),
                "sample_group_id": row.get("group_id", sample_id),
                "sample_id": sample_id,
                "rheed_video_id": Path(row.get("rheed_path", "")).stem,
                "rheed_video_path": row.get("rheed_path", ""),
                "crop_video_path": display_path(crop_videos[sample_id], repo_root) if sample_id in crop_videos else "",
                "afm_scan_id": scan_id_from_path(resolve_path(repo_root, row.get("afm_path", ""))),
                "afm_path": row.get("afm_path", ""),
                "metadata_path": display_path(metadata_path, repo_root) if metadata_path else "",
                "material": row.get("material", ""),
                "substrate": "",
                "growth_batch": "",
                "scan_size_um": row.get("afm_scan_size_um", ""),
                "afm_resolution": "",
                "pairing_status": "paired" if sample_id in crop_videos and afm_candidates else "unmatched",
                "pairing_notes": row.get("selection_reason", ""),
                "all_plane_corrected_afm_scan_count": len(afm_candidates),
            }
        )
    duplicate_afm = validate_unique_pairing(audit_rows, ["sample_id"], "afm_path")
    duplicate_rheed = validate_unique_pairing(audit_rows, ["sample_id"], "rheed_video_path")
    summary = {
        "representative_pair_count": len(audit_rows),
        "paired_count": sum(1 for row in audit_rows if row["pairing_status"] == "paired"),
        "unmatched_count": sum(1 for row in audit_rows if row["pairing_status"] != "paired"),
        "crop_video_issue_count": len(crop_issues),
        "duplicate_afm_pairing_count": len(duplicate_afm),
        "duplicate_rheed_pairing_count": len(duplicate_rheed),
        "crop_video_issues": crop_issues,
        "duplicate_afm_pairings": duplicate_afm,
        "duplicate_rheed_pairings": duplicate_rheed,
    }
    return audit_rows, summary


def qc_membership(row: dict[str, Any], config: dict[str, Any]) -> tuple[bool, list[dict[str, Any]]]:
    qc = config["qc"]
    issues: list[dict[str, Any]] = []
    checks = [
        (
            "insufficient_valid_frames",
            safe_float(row.get("number_of_valid_rheed_frames"), 0.0),
            float(qc["min_valid_frames"]),
            "min",
        ),
        (
            "insufficient_valid_frame_fraction",
            safe_float(row.get("valid_frame_fraction"), 0.0),
            float(qc["min_valid_fraction"]),
            "min",
        ),
        (
            "excessive_saturation",
            safe_float(row.get("saturation_fraction_median"), 0.0),
            float(qc["max_median_saturation_fraction"]),
            "max",
        ),
        (
            "severe_underexposure",
            safe_float(row.get("underexposed_fraction_median"), 0.0),
            float(qc["max_median_underexposed_fraction"]),
            "max",
        ),
        (
            "insufficient_signal",
            safe_float(row.get("robust_dynamic_range_median"), 0.0),
            float(qc["min_median_dynamic_range"]),
            "min",
        ),
        (
            "detector_failure",
            safe_float(row.get("detector_confidence_median"), 0.0),
            float(qc["min_median_detector_confidence"]),
            "min",
        ),
    ]
    for reason, value, threshold, mode in checks:
        fail = value < threshold if mode == "min" else value > threshold
        if fail:
            issues.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "growth_run_id": row.get("growth_run_id", ""),
                    "level": "sample",
                    "exclusion_reason": reason,
                    "measured_value": value,
                    "threshold": threshold,
                    "included_in_inclusive_analysis": 1,
                    "included_in_strict_analysis": 0,
                }
            )
    if "unknown_height_unit" in str(row.get("height_unit_statuses", "")):
        issues.append(
            {
                "sample_id": row.get("sample_id", ""),
                "growth_run_id": row.get("growth_run_id", ""),
                "level": "sample",
                "exclusion_reason": "unknown_height_unit",
                "measured_value": row.get("height_unit_statuses", ""),
                "threshold": "known physical height unit",
                "included_in_inclusive_analysis": 1,
                "included_in_strict_analysis": 0,
            }
        )
    return not issues, issues


def build_sample_analysis_table(
    representative_rows: Sequence[dict[str, str]],
    video_rows: Sequence[dict[str, Any]],
    afm_sample_targets: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rep_by_sample = {row["sample_id"]: row for row in representative_rows}
    video_by_sample = {row["sample_id"]: row for row in video_rows}
    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    for sample_id in sorted(set(rep_by_sample) | set(video_by_sample) | set(afm_sample_targets)):
        rep = rep_by_sample.get(sample_id, {})
        video = video_by_sample.get(sample_id, {})
        afm = afm_sample_targets.get(sample_id, {})
        has_primary_target = math.isfinite(safe_float(afm.get("afm_Rq_median_nm")))
        has_any_target = math.isfinite(safe_float(afm.get("afm_Rq_all_scan_sizes_median_nm")))
        inclusive = bool(video) and bool(afm) and has_any_target
        row = {
            "sample_id": sample_id,
            "growth_run_id": rep.get("group_id", sample_id),
            "sample_group_id": rep.get("group_id", sample_id),
            "material": rep.get("material", ""),
            "substrate": "",
            "growth_batch": "",
            "camera_tool": video.get("camera_tool_id", "unknown"),
            **video,
            **afm,
            "included_in_inclusive_analysis": int(inclusive),
            "included_in_primary_scan_size_analysis": int(inclusive and has_primary_target),
        }
        strict_ok, issues = qc_membership(row, config)
        row["included_in_strict_analysis"] = int(inclusive and strict_ok)
        row["all_qc_flags"] = ";".join(issue["exclusion_reason"] for issue in issues)
        ledger.extend(issues)
        if inclusive and not issues:
            ledger.append(
                {
                    "sample_id": sample_id,
                    "growth_run_id": row.get("growth_run_id", sample_id),
                    "level": "sample",
                    "exclusion_reason": "",
                    "measured_value": "",
                    "threshold": "",
                    "included_in_inclusive_analysis": 1,
                    "included_in_strict_analysis": 1,
                }
            )
        rows.append(row)
    return rows, ledger


def finite_pairs(rows: Sequence[dict[str, Any]], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    clean = []
    x_values = []
    y_values = []
    for row in rows:
        x = safe_float(row.get(x_key))
        y = safe_float(row.get(y_key))
        if math.isfinite(x) and math.isfinite(y):
            clean.append(row)
            x_values.append(x)
            y_values.append(y)
    return np.asarray(x_values, dtype=np.float64), np.asarray(y_values, dtype=np.float64), clean


def safe_spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 3 or np.unique(x_arr).size < 2 or np.unique(y_arr).size < 2:
        return math.nan, math.nan
    result = stats.spearmanr(x_arr, y_arr)
    return safe_float(result.statistic), safe_float(result.pvalue)


def safe_kendall(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 3 or np.unique(x_arr).size < 2 or np.unique(y_arr).size < 2:
        return math.nan, math.nan
    result = stats.kendalltau(x_arr, y_arr)
    return safe_float(result.statistic), safe_float(result.pvalue)


def bootstrap_spearman(x: Sequence[float], y: Sequence[float], *, resamples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 4:
        return math.nan, math.nan
    stats_out = []
    for _ in range(resamples):
        idx = rng.integers(0, x_arr.size, size=x_arr.size)
        value, _ = safe_spearman(x_arr[idx], y_arr[idx])
        if math.isfinite(value):
            stats_out.append(value)
    if not stats_out:
        return math.nan, math.nan
    return float(np.percentile(stats_out, 2.5)), float(np.percentile(stats_out, 97.5))


def permutation_p_value(x: Sequence[float], y: Sequence[float], observed: float, *, resamples: int, seed: int) -> float:
    if not math.isfinite(observed):
        return math.nan
    rng = np.random.default_rng(seed)
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.size < 4:
        return math.nan
    count = 0
    total = 0
    for _ in range(resamples):
        shuffled = rng.permutation(y_arr)
        stat, _ = safe_spearman(x_arr, shuffled)
        if math.isfinite(stat):
            count += int(abs(stat) >= abs(observed))
            total += 1
    return float((count + 1) / (total + 1)) if total else math.nan


def correlation_analysis(
    sample_rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    paths: Paths,
    *,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    resamples = int(config["statistics"]["smoke_resamples"] if smoke else config["statistics"]["bootstrap_resamples"])
    permutations = int(config["statistics"]["smoke_resamples"] if smoke else config["statistics"]["permutation_resamples"])
    seed = int(config["random_seed"])
    subsets = [
        ("primary_1um_inclusive", [row for row in sample_rows if int(row.get("included_in_inclusive_analysis", 0)) and bool(row.get("has_primary_scan_size"))], "afm_Rq_median_nm"),
        ("primary_1um_strict_qc", [row for row in sample_rows if int(row.get("included_in_strict_analysis", 0)) and bool(row.get("has_primary_scan_size"))], "afm_Rq_median_nm"),
        ("all_scan_sizes_inclusive", [row for row in sample_rows if int(row.get("included_in_inclusive_analysis", 0))], "afm_Rq_all_scan_sizes_median_nm"),
    ]
    predictors = [
        ("median_morphology_index", "higher means more spotty"),
        ("median_spottiness", "raw spot component score"),
        ("median_streakiness", "raw streak/bar component score"),
    ]
    results: list[dict[str, Any]] = []
    for subset_name, rows, target in subsets:
        for y_key, scale in [(target, "raw_nm"), (target, "log10_nm")]:
            working_rows = []
            for row in rows:
                y = safe_float(row.get(target))
                if scale == "log10_nm":
                    if y <= 0 or not math.isfinite(y):
                        continue
                    copy = dict(row)
                    copy["_log_target"] = math.log10(y)
                    working_rows.append(copy)
                    y_key = "_log_target"
                else:
                    working_rows.append(row)
            for predictor, note in predictors:
                x, y, clean = finite_pairs(working_rows, predictor, y_key)
                spearman, spearman_p = safe_spearman(x, y)
                kendall, kendall_p = safe_kendall(x, y)
                ci_low, ci_high = bootstrap_spearman(x, y, resamples=resamples, seed=seed + len(results))
                perm_p = permutation_p_value(x, y, spearman, resamples=permutations, seed=seed + 1000 + len(results))
                results.append(
                    {
                        "analysis_subset": subset_name,
                        "target": target,
                        "target_scale": scale,
                        "predictor": predictor,
                        "predictor_note": note,
                        "n_samples": len(clean),
                        "spearman_rho": spearman,
                        "spearman_p_asymptotic": spearman_p,
                        "spearman_bootstrap_ci_low": ci_low,
                        "spearman_bootstrap_ci_high": ci_high,
                        "spearman_permutation_p": perm_p,
                        "kendall_tau": kendall,
                        "kendall_p_asymptotic": kendall_p,
                        "bootstrap_resamples": resamples,
                        "permutation_resamples": permutations,
                    }
                )
    return results


def feature_matrix(rows: Sequence[dict[str, Any]], feature_keys: Sequence[str], categorical_keys: Sequence[str] = ()) -> tuple[np.ndarray, list[str]]:
    dicts: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key in feature_keys:
            value = safe_float(row.get(key))
            item[key] = value if math.isfinite(value) else 0.0
        for key in categorical_keys:
            value = str(row.get(key, "") or "unknown")
            item[f"{key}={value}"] = 1.0
        dicts.append(item)
    vec = DictVectorizer(sparse=False)
    if not dicts:
        return np.zeros((0, 0)), []
    return vec.fit_transform(dicts), list(vec.feature_names_)


def grouped_cv_predictions(
    rows: Sequence[dict[str, Any]],
    target_key: str,
    feature_keys: Sequence[str],
    categorical_keys: Sequence[str] = (),
    group_key: str = "growth_run_id",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    y = np.asarray([safe_float(row.get(target_key)) for row in rows], dtype=np.float64)
    valid_mask = np.isfinite(y)
    clean_rows = [row for row, ok in zip(rows, valid_mask) if ok]
    y = y[valid_mask]
    if y.size < 3:
        return y, np.full_like(y, np.nan), []
    groups = np.asarray([str(row.get(group_key, row.get("sample_id", ""))) for row in clean_rows])
    X, names = feature_matrix(clean_rows, feature_keys, categorical_keys)
    preds = np.full(y.shape, np.nan, dtype=np.float64)
    if X.shape[1] == 0 or len(np.unique(groups)) < 3:
        preds[:] = np.mean(y)
        return y, preds, names
    splitter = LeaveOneGroupOut()
    for train, test in splitter.split(X, y, groups):
        if np.unique(y[train]).size < 2:
            preds[test] = np.mean(y[train])
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X[train], y[train])
        preds[test] = model.predict(X[test])
    return y, preds, names


def prediction_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() < 3:
        return {"mae": math.nan, "rmse": math.nan, "r2": math.nan, "spearman": math.nan}
    yv = y[mask]
    pv = pred[mask]
    sp, _ = safe_spearman(yv, pv)
    return {
        "mae": float(mean_absolute_error(yv, pv)),
        "rmse": float(math.sqrt(mean_squared_error(yv, pv))),
        "r2": float(r2_score(yv, pv)),
        "spearman": sp,
    }


def regression_and_cv_analysis(sample_rows: Sequence[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [row for row in sample_rows if int(row.get("included_in_inclusive_analysis", 0)) and bool(row.get("has_primary_scan_size"))]
    for row in rows:
        rq = safe_float(row.get("afm_Rq_median_nm"))
        row["log10_afm_Rq_median_nm"] = math.log10(rq) if rq > 0 and math.isfinite(rq) else math.nan
    nuisance = [
        "mean_intensity_median",
        "saturation_fraction_median",
        "underexposed_fraction_median",
        "focus_metric_median",
        "pattern_centroid_x_median",
        "pattern_centroid_y_median",
        "frame_shift_rms",
        "detector_confidence_median",
    ]
    morphology = ["median_morphology_index"]
    metadata_numeric = ["number_of_primary_scan_size_afm_scans"]
    metadata_cat = ["material", "afm_resolution"]
    model_specs = [
        ("A_nuisance_only", nuisance, []),
        ("B_rheed_morphology_only", morphology, []),
        ("C_nuisance_plus_rheed", nuisance + morphology, []),
        ("D_process_metadata_only", metadata_numeric, metadata_cat),
        ("E_metadata_plus_rheed", metadata_numeric + morphology, metadata_cat),
    ]
    cv_rows: list[dict[str, Any]] = []
    stored_metrics: dict[str, dict[str, float]] = {}
    for name, num, cat in model_specs:
        y, pred, names = grouped_cv_predictions(rows, "log10_afm_Rq_median_nm", num, cat)
        metrics = prediction_metrics(y, pred)
        stored_metrics[name] = metrics
        cv_rows.append(
            {
                "model": name,
                "target": "log10_afm_Rq_median_nm",
                "n_samples": int(np.isfinite(y).sum()),
                "feature_count": len(names),
                "grouping": "leave_one_growth_run_out",
                "out_of_fold_mae": metrics["mae"],
                "out_of_fold_rmse": metrics["rmse"],
                "out_of_fold_r2": metrics["r2"],
                "out_of_fold_spearman": metrics["spearman"],
                "delta_mae_vs_nuisance": metrics["mae"] - stored_metrics.get("A_nuisance_only", {}).get("mae", math.nan),
                "delta_r2_vs_nuisance": metrics["r2"] - stored_metrics.get("A_nuisance_only", {}).get("r2", math.nan),
            }
        )
    # Score predictability from nuisance variables, a negative-control audit.
    y_score, pred_score, score_feature_names = grouped_cv_predictions(rows, "median_morphology_index", nuisance, [])
    score_metrics = prediction_metrics(y_score, pred_score)
    cv_rows.append(
        {
            "model": "negative_control_score_from_nuisance",
            "target": "median_morphology_index",
            "n_samples": int(np.isfinite(y_score).sum()),
            "feature_count": len(score_feature_names),
            "grouping": "leave_one_growth_run_out",
            "out_of_fold_mae": score_metrics["mae"],
            "out_of_fold_rmse": score_metrics["rmse"],
            "out_of_fold_r2": score_metrics["r2"],
            "out_of_fold_spearman": score_metrics["spearman"],
            "delta_mae_vs_nuisance": math.nan,
            "delta_r2_vs_nuisance": math.nan,
        }
    )

    regression_rows: list[dict[str, Any]] = []
    x, y, clean = finite_pairs(rows, "median_morphology_index", "log10_afm_Rq_median_nm")
    if len(clean) >= 4:
        base_spearman, base_p = safe_spearman(x, y)
        regression_rows.append(
            {
                "model": "B_rheed_morphology_only",
                "target": "log10_afm_Rq_median_nm",
                "n_samples": len(clean),
                "association_type": "spearman",
                "morphology_effect": base_spearman,
                "p_value": base_p,
                "covariates": "",
            }
        )
        X_nuisance, names = feature_matrix(clean, nuisance, [])
        if X_nuisance.shape[1] > 0 and len(clean) > X_nuisance.shape[1] + 2:
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X_nuisance)
            y_res = y - LinearRegression().fit(Xs, y).predict(Xs)
            x_res = x - LinearRegression().fit(Xs, x).predict(Xs)
            partial, partial_p = safe_spearman(x_res, y_res)
            regression_rows.append(
                {
                    "model": "C_partial_nuisance_adjusted",
                    "target": "log10_afm_Rq_median_nm",
                    "n_samples": len(clean),
                    "association_type": "residual_spearman",
                    "morphology_effect": partial,
                    "p_value": partial_p,
                    "covariates": ";".join(names),
                }
            )
    return regression_rows, cv_rows


def material_stratified_results(sample_rows: Sequence[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    min_n = int(config["statistics"]["min_stratum_samples"])
    rows = [row for row in sample_rows if int(row.get("included_in_inclusive_analysis", 0)) and bool(row.get("has_primary_scan_size"))]
    output: list[dict[str, Any]] = []
    for material in sorted({str(row.get("material", "unknown")) for row in rows}):
        subset = [row for row in rows if str(row.get("material", "unknown")) == material]
        x, y, clean = finite_pairs(subset, "median_morphology_index", "afm_Rq_median_nm")
        rho, p = safe_spearman(x, y)
        output.append(
            {
                "stratum_type": "material",
                "stratum": material,
                "n_samples": len(clean),
                "analysis_status": "ok" if len(clean) >= min_n else "too_few_samples",
                "spearman_rho": rho,
                "spearman_p": p,
            }
        )
    return output


def apply_named_perturbation(gray: np.ndarray, name: str, rng: np.random.Generator) -> np.ndarray:
    image = frame_to_gray_float32(gray)
    if name.startswith("brightness_"):
        scale = float(name.split("_", 1)[1])
        return np.clip(image * scale, 0.0, 1.0)
    if name.startswith("contrast_"):
        scale = float(name.split("_", 1)[1])
        return np.clip((image - 0.5) * scale + 0.5, 0.0, 1.0)
    if name.startswith("gamma_"):
        gamma = float(name.split("_", 1)[1])
        return np.clip(image, 0.0, 1.0) ** gamma
    if name.startswith("translate_x_"):
        shift = float(name.rsplit("_", 1)[1])
        return np.asarray(ndimage.shift(image, shift=(0, shift), mode="nearest", order=1), dtype=np.float32)
    if name.startswith("translate_y_"):
        shift = float(name.rsplit("_", 1)[1])
        return np.asarray(ndimage.shift(image, shift=(shift, 0), mode="nearest", order=1), dtype=np.float32)
    if name.startswith("rotate_"):
        angle = float(name.split("_", 1)[1])
        return np.asarray(ndimage.rotate(image, angle=angle, reshape=False, mode="nearest", order=1), dtype=np.float32)
    if name.startswith("blur_"):
        sigma = float(name.split("_", 1)[1])
        return np.asarray(ndimage.gaussian_filter(image, sigma=sigma), dtype=np.float32)
    if name.startswith("noise_"):
        sigma = float(name.split("_", 1)[1])
        return np.clip(image + rng.normal(0.0, sigma, size=image.shape), 0.0, 1.0)
    if name.startswith("saturate_"):
        level = float(name.split("_", 1)[1])
        return np.clip(image, 0.0, level) / max(level, 1e-8)
    if name == "lowfreq_gradient":
        yy, xx = np.indices(image.shape)
        grad = (xx / max(image.shape[1] - 1, 1) - 0.5) * 0.25 + (yy / max(image.shape[0] - 1, 1) - 0.5) * 0.15
        return np.clip(image + grad, 0.0, 1.0)
    if name == "crop_jitter":
        shifted = ndimage.shift(image, shift=(5, -5), mode="nearest", order=1)
        return np.asarray(shifted, dtype=np.float32)
    if name == "scale_0.9":
        zoomed = ndimage.zoom(image, 0.9, order=1)
        pad_y = (image.shape[0] - zoomed.shape[0]) // 2
        pad_x = (image.shape[1] - zoomed.shape[1]) // 2
        canvas = np.zeros_like(image)
        canvas[pad_y : pad_y + zoomed.shape[0], pad_x : pad_x + zoomed.shape[1]] = zoomed
        return canvas
    return image


def detector_score_for_gray(gray: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    processed = preprocess_frame_for_shape(gray, image_size=int(config["rheed"]["image_size"]))
    _, geometry = extract_components_and_frame_features(
        soft_mask=processed.channels["soft_spot_streak_mask"],
        enhanced_image=processed.channels["log_bgsub"],
        artifact_mask=processed.artifact_mask,
    )
    scores = compute_morphology_scores(geometry, epsilon=float(config["rheed"].get("epsilon", 1e-8)))
    scores["detector_confidence"] = finite_float(geometry.get("mask_confidence", 0.0))
    return scores


def perturbation_names(config: dict[str, Any]) -> list[str]:
    p = config["perturbations"]
    names = []
    names.extend(f"brightness_{value:g}" for value in p["brightness_scales"])
    names.extend(f"contrast_{value:g}" for value in p["contrast_scales"])
    names.extend(f"gamma_{value:g}" for value in p["gammas"])
    names.extend(f"translate_x_{value:g}" for value in p["translations_px"])
    names.extend(f"translate_y_{value:g}" for value in p["translations_px"])
    names.extend(f"rotate_{value:g}" for value in p["rotations_deg"])
    names.extend(f"blur_{value:g}" for value in p["blur_sigmas"])
    names.extend(f"noise_{value:g}" for value in p["noise_sigmas"])
    names.extend(f"saturate_{value:g}" for value in p["saturation_levels"])
    names.extend(["lowfreq_gradient", "crop_jitter", "scale_0.9"])
    return names


def run_perturbation_audit(frame_rows: Sequence[dict[str, Any]], config: dict[str, Any], paths: Paths, *, smoke: bool = False) -> list[dict[str, Any]]:
    seed = int(config["random_seed"])
    rng = np.random.default_rng(seed)
    max_frames = min(int(config["perturbations"]["max_frames"]), len(frame_rows))
    if smoke:
        max_frames = min(12, max_frames)
    sorted_rows = sorted(frame_rows, key=lambda row: safe_float(row.get("morphology_index"), 0.0))
    if max_frames <= 0:
        return []
    selected_idx = sorted({int(round(v)) for v in np.linspace(0, len(sorted_rows) - 1, max_frames)})
    selected = [sorted_rows[index] for index in selected_idx]
    output: list[dict[str, Any]] = []
    examples_dir = paths.assets_dir / "perturbation_examples"
    names = perturbation_names(config)
    for row in selected:
        raw_path = resolve_path(paths.repo_root, str(row.get("raw_frame_path", "")))
        if not raw_path.is_file():
            continue
        gray = frame_to_gray_float32(np.asarray(Image.open(raw_path)))
        original = detector_score_for_gray(gray, config)
        natural_sd = np.std([safe_float(item.get("morphology_index")) for item in frame_rows]) or 1.0
        for name in names:
            perturbed = apply_named_perturbation(gray, name, rng)
            score = detector_score_for_gray(perturbed, config)
            delta = score["morphology_index"] - original["morphology_index"]
            output.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "frame_idx": row.get("frame_idx", ""),
                    "perturbation": name,
                    "original_morphology_index": original["morphology_index"],
                    "perturbed_morphology_index": score["morphology_index"],
                    "absolute_score_change": abs(delta),
                    "signed_score_change": delta,
                    "score_change_dataset_sd": abs(delta) / max(float(natural_sd), 1e-8),
                    "original_spottiness": original["raw_spottiness"],
                    "perturbed_spottiness": score["raw_spottiness"],
                    "original_streakiness": original["raw_streakiness"],
                    "perturbed_streakiness": score["raw_streakiness"],
                    "detector_failure": int(score["detector_confidence"] < float(config["qc"]["min_median_detector_confidence"])),
                }
            )
        if len(list(examples_dir.glob("*.png"))) < 12:
            save_image(examples_dir / f"{row.get('sample_id')}_{row.get('frame_idx')}_original.png", gray)
            save_image(examples_dir / f"{row.get('sample_id')}_{row.get('frame_idx')}_brightness_0p5.png", apply_named_perturbation(gray, "brightness_0.5", rng))
            save_image(examples_dir / f"{row.get('sample_id')}_{row.get('frame_idx')}_translate_x_8.png", apply_named_perturbation(gray, "translate_x_8", rng))
    return output


def summarize_perturbations(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["perturbation"])].append(row)
    summary = []
    for name, group in sorted(grouped.items()):
        original = [safe_float(row.get("original_morphology_index")) for row in group]
        perturbed = [safe_float(row.get("perturbed_morphology_index")) for row in group]
        rho, _ = safe_spearman(original, perturbed)
        changes = [safe_float(row.get("absolute_score_change")) for row in group if math.isfinite(safe_float(row.get("absolute_score_change")))]
        summary.append(
            {
                "perturbation": name,
                "n_frames": len(group),
                "median_absolute_score_change": float(np.median(changes)) if changes else math.nan,
                "p90_absolute_score_change": float(np.percentile(changes, 90)) if changes else math.nan,
                "rank_preservation_spearman": rho,
                "detector_failure_rate": float(np.mean([safe_float(row.get("detector_failure"), 0.0) for row in group])) if group else math.nan,
            }
        )
    return sorted(summary, key=lambda row: safe_float(row.get("median_absolute_score_change"), -1), reverse=True)


def negative_controls(
    frame_rows: Sequence[dict[str, Any]],
    sample_rows: Sequence[dict[str, Any]],
    perturbation_summary: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    nuisance_model = next((row for row in sample_rows if False), None)
    _ = nuisance_model
    bg_score_rows = [row for row in perturbation_summary if str(row.get("perturbation")) in {"translate_x_8", "translate_y_8", "crop_jitter"}]
    morph, rq, clean = finite_pairs(
        [row for row in sample_rows if int(row.get("included_in_inclusive_analysis", 0)) and bool(row.get("has_primary_scan_size"))],
        "median_morphology_index",
        "afm_Rq_median_nm",
    )
    real_rho, _ = safe_spearman(morph, rq)
    return [
        {
            "control": "permuted_target",
            "metric": "implemented_in_correlation_results",
            "value": "spearman_permutation_p",
            "interpretation": "sample-level AFM labels are shuffled in the reported permutation test",
        },
        {
            "control": "misaligned_roi_proxy",
            "metric": "median_shift_or_crop_score_change",
            "value": float(np.median([safe_float(row.get("median_absolute_score_change")) for row in bg_score_rows])) if bg_score_rows else math.nan,
            "interpretation": "large values indicate ROI position sensitivity",
        },
        {
            "control": "nuisance_only_prediction",
            "metric": "implemented_in_cv_model_comparison",
            "value": "A_nuisance_only",
            "interpretation": "compare against models adding morphology",
        },
        {
            "control": "score_predictability_audit",
            "metric": "implemented_in_cv_model_comparison",
            "value": "negative_control_score_from_nuisance",
            "interpretation": "high score predictability from nuisance variables is a warning",
        },
        {
            "control": "observed_primary_association_reference",
            "metric": "spearman_rho",
            "value": real_rho,
            "interpretation": f"computed on {len(clean)} primary samples before permutation",
        },
    ]


def render_afm_image(sample_id: str, afm_rel_path: str, paths: Paths) -> str:
    if not afm_rel_path:
        return ""
    afm_path = resolve_path(paths.repo_root, afm_rel_path)
    if not afm_path.is_file():
        return ""
    out = paths.assets_dir / "afm" / f"{sample_id}_{scan_id_from_path(afm_path)}.png"
    if out.is_file():
        return display_path(out, paths.repo_root)
    try:
        height = np.load(afm_path)
    except Exception:
        return ""
    fig, axis = plt.subplots(figsize=(3.2, 3.0), dpi=150)
    im = axis.imshow(height, cmap="viridis")
    axis.set_title(f"{sample_id} AFM", fontsize=9)
    axis.axis("off")
    cbar = fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("nm", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return display_path(out, paths.repo_root)


def make_data_overview_figures(
    sample_rows: Sequence[dict[str, Any]],
    afm_rows: Sequence[dict[str, Any]],
    pairing_summary: dict[str, Any],
    paths: Paths,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=150)
    counts = {
        "paired": pairing_summary.get("paired_count", 0),
        "inclusive": sum(int(row.get("included_in_inclusive_analysis", 0)) for row in sample_rows),
        "strict": sum(int(row.get("included_in_strict_analysis", 0)) for row in sample_rows),
    }
    axes[0].bar(counts.keys(), counts.values(), color=["#4c78a8", "#59a14f", "#e15759"])
    axes[0].set_title("Sample flow")
    axes[0].set_ylabel("samples")
    scan_sizes = [safe_float(row.get("scan_size_um")) for row in afm_rows if math.isfinite(safe_float(row.get("scan_size_um")))]
    axes[1].hist(scan_sizes, bins=20, color="#76b7b2", edgecolor="white")
    axes[1].set_title("AFM scan sizes")
    axes[1].set_xlabel("um")
    rq = [safe_float(row.get("Rq_nm")) for row in afm_rows if math.isfinite(safe_float(row.get("Rq_nm")))]
    axes[2].hist(rq, bins=24, color="#f28e2b", edgecolor="white")
    axes[2].set_title("AFM Rq distribution")
    axes[2].set_xlabel("nm")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "data_overview.png")
    plt.close(fig)


def write_scatter_svg(
    path: Path,
    rows: Sequence[dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    title: str,
    x_label: str,
    y_label: str,
    increasing: bool = True,
) -> None:
    x, y, clean = finite_pairs(rows, x_key, y_key)
    width, height = 760, 520
    margin = 72
    if x.size == 0:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'><text x='20' y='30'>No data</text></svg>", encoding="utf-8")
        return
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    pad_x = (x_max - x_min) * 0.06
    pad_y = (y_max - y_min) * 0.08
    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    def sx(value: float) -> float:
        return margin + (value - x_min) / (x_max - x_min) * (width - 2 * margin)

    def sy(value: float) -> float:
        return height - margin - (value - y_min) / (y_max - y_min) * (height - 2 * margin)

    pieces = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2:.1f}' y='28' text-anchor='middle' font-size='18' font-family='sans-serif'>{html.escape(title)}</text>",
        f"<line x1='{margin}' y1='{height-margin}' x2='{width-margin}' y2='{height-margin}' stroke='#333'/>",
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height-margin}' stroke='#333'/>",
        f"<text x='{width/2:.1f}' y='{height-18}' text-anchor='middle' font-size='13' font-family='sans-serif'>{html.escape(x_label)}</text>",
        f"<text x='20' y='{height/2:.1f}' transform='rotate(-90 20 {height/2:.1f})' text-anchor='middle' font-size='13' font-family='sans-serif'>{html.escape(y_label)}</text>",
    ]
    # Isotonic monotonic trend, shown only within observed data.
    if x.size >= 4 and np.unique(x).size >= 3:
        order = np.argsort(x)
        iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
        try:
            y_fit = iso.fit_transform(x[order], y[order])
            trend_points = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(x[order], y_fit))
            pieces.append(f"<polyline points='{trend_points}' fill='none' stroke='#d62728' stroke-width='3' opacity='0.85'/>")
        except Exception:
            pass
    for row, xv, yv in zip(clean, x, y):
        sample_id = str(row.get("sample_id", ""))
        material = str(row.get("material", ""))
        title_text = f"{sample_id} {material} {x_key}={xv:.4g} {y_key}={yv:.4g}"
        link = str(row.get("representative_overlay_path", ""))
        circle = (
            f"<circle cx='{sx(xv):.1f}' cy='{sy(yv):.1f}' r='5.5' fill='#1f77b4' "
            "stroke='white' stroke-width='1.2' opacity='0.88'>"
            f"<title>{html.escape(title_text)}</title></circle>"
        )
        if link:
            circle = f"<a href='../{html.escape(link)}'>{circle}</a>"
        pieces.append(circle)
    pieces.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(pieces), encoding="utf-8")


def write_temporal_plots(video_rows: Sequence[dict[str, Any]], frame_rows: Sequence[dict[str, Any]], paths: Paths) -> None:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        by_sample[str(row["sample_id"])].append(row)
    out_dir = paths.figures_dir / "temporal"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sample_id, rows in by_sample.items():
        rows = sorted(rows, key=lambda row: safe_float(row.get("frame_timestamp_sec"), 0.0))
        if not rows:
            continue
        t = [safe_float(row.get("frame_timestamp_sec")) for row in rows]
        fig, axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True, dpi=140)
        axes[0].plot(t, [safe_float(row.get("morphology_index")) for row in rows], marker="o", label="morphology")
        axes[0].plot(t, [safe_float(row.get("raw_spottiness")) for row in rows], alpha=0.6, label="spot")
        axes[0].plot(t, [safe_float(row.get("raw_streakiness")) for row in rows], alpha=0.6, label="streak")
        axes[0].legend(fontsize=7)
        axes[1].plot(t, [safe_float(row.get("mean_intensity")) for row in rows], marker=".", label="mean")
        axes[1].plot(t, [safe_float(row.get("saturation_fraction")) for row in rows], marker=".", label="saturation")
        axes[1].legend(fontsize=7)
        axes[2].plot(t, [safe_float(row.get("pattern_centroid_x")) for row in rows], label="x")
        axes[2].plot(t, [safe_float(row.get("pattern_centroid_y")) for row in rows], label="y")
        axes[2].legend(fontsize=7)
        axes[3].plot(t, [safe_float(row.get("focus_metric")) for row in rows], label="focus")
        axes[3].plot(t, [safe_float(row.get("frame_to_frame_motion_magnitude")) for row in rows], label="motion")
        axes[3].legend(fontsize=7)
        axes[3].set_xlabel("timestamp sec")
        for axis in axes:
            axis.grid(alpha=0.25)
        fig.suptitle(f"{sample_id} temporal diagnostics", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / f"{sample_id}_temporal.png")
        plt.close(fig)


def write_confound_figures(sample_rows: Sequence[dict[str, Any]], paths: Paths) -> None:
    rows = [row for row in sample_rows if int(row.get("included_in_inclusive_analysis", 0))]
    keys = [
        "median_morphology_index",
        "mean_intensity_median",
        "saturation_fraction_median",
        "underexposed_fraction_median",
        "focus_metric_median",
        "pattern_centroid_x_median",
        "pattern_centroid_y_median",
        "frame_shift_rms",
        "afm_Rq_median_nm",
    ]
    matrix = []
    labels = []
    for key_a in keys:
        row_vals = []
        for key_b in keys:
            a, b, _ = finite_pairs(rows, key_a, key_b)
            rho, _ = safe_spearman(a, b)
            row_vals.append(rho if math.isfinite(rho) else 0.0)
        matrix.append(row_vals)
        labels.append(key_a.replace("_median", "").replace("median_", ""))
    fig, axis = plt.subplots(figsize=(8, 7), dpi=150)
    im = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axis.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=axis, label="Spearman rho")
    axis.set_title("Nuisance and target rank-correlation heatmap")
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "confound_correlation_heatmap.png")
    plt.close(fig)


def write_perturbation_figure(summary_rows: Sequence[dict[str, Any]], paths: Paths) -> None:
    rows = sorted(summary_rows, key=lambda row: safe_float(row.get("median_absolute_score_change"), -1), reverse=True)[:20]
    if not rows:
        return
    fig, axis = plt.subplots(figsize=(9, max(4, 0.34 * len(rows))), dpi=150)
    names = [str(row["perturbation"]) for row in rows]
    values = [safe_float(row.get("median_absolute_score_change")) for row in rows]
    axis.barh(range(len(rows)), values, color="#4c78a8")
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels(names, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("median absolute morphology-index change")
    axis.set_title("Perturbation sensitivity ranking")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "perturbation_sensitivity.png")
    plt.close(fig)


def write_material_forest(rows: Sequence[dict[str, Any]], paths: Paths) -> None:
    valid = [row for row in rows if math.isfinite(safe_float(row.get("spearman_rho")))]
    if not valid:
        return
    fig, axis = plt.subplots(figsize=(7, max(3.0, 0.35 * len(valid))), dpi=150)
    y = np.arange(len(valid))
    axis.scatter([safe_float(row["spearman_rho"]) for row in valid], y, color="#59a14f")
    axis.axvline(0, color="#333", lw=1)
    axis.set_yticks(y)
    axis.set_yticklabels([f"{row['stratum']} (n={row['n_samples']})" for row in valid], fontsize=8)
    axis.set_xlabel("Spearman rho")
    axis.set_title("Material-stratified morphology vs Rq")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "material_forest.png")
    plt.close(fig)


def html_img(src: str, alt: str, width: str = "100%") -> str:
    return f"<img src='{html.escape(src)}' alt='{html.escape(alt)}' style='width:{width};height:auto'>"


def relative_to_report(path_text: str, paths: Paths) -> str:
    if not path_text:
        return ""
    path = resolve_path(paths.repo_root, path_text)
    try:
        return path.resolve().relative_to(paths.reports_dir.resolve()).as_posix()
    except ValueError:
        try:
            return Path(os.path.relpath(path.resolve(), paths.reports_dir.resolve())).as_posix()
        except Exception:
            return path_text


def make_gallery_cards(rows: Sequence[dict[str, Any]], paths: Paths, *, sort_key: str, limit: int | None = None) -> str:
    ordered = sorted(rows, key=lambda row: safe_float(row.get(sort_key), math.inf))
    if limit is not None:
        ordered = ordered[:limit]
    cards = []
    for row in ordered:
        sample_id = str(row.get("sample_id", ""))
        overlay = relative_to_report(str(row.get("representative_overlay_path", "")), paths)
        raw = relative_to_report(str(row.get("representative_raw_frame_path", "")), paths)
        afm = relative_to_report(str(row.get("representative_afm_render_path", "")), paths)
        card = [
            "<article class='card'>",
            f"<h3>{html.escape(sample_id)}</h3>",
            "<div class='thumbs'>",
        ]
        if raw:
            card.append(html_img(raw, f"{sample_id} raw"))
        if overlay:
            card.append(html_img(overlay, f"{sample_id} overlay"))
        if afm:
            card.append(html_img(afm, f"{sample_id} afm"))
        card.extend(
            [
                "</div>",
                "<p>",
                f"morph={safe_float(row.get('median_morphology_index')):.3g}, "
                f"spot={safe_float(row.get('median_spottiness')):.3g}, "
                f"streak={safe_float(row.get('median_streakiness')):.3g}, "
                f"Rq={safe_float(row.get('afm_Rq_median_nm')):.3g} nm, "
                f"flags={html.escape(str(row.get('all_qc_flags', '')))}",
                "</p>",
                "</article>",
            ]
        )
        cards.append("\n".join(card))
    return "\n".join(cards)


def make_quadrant_gallery(sample_rows: Sequence[dict[str, Any]], paths: Paths) -> str:
    rows = [row for row in sample_rows if int(row.get("included_in_inclusive_analysis", 0))]
    morph = np.asarray([safe_float(row.get("median_morphology_index")) for row in rows])
    rq = np.asarray([safe_float(row.get("afm_Rq_median_nm")) for row in rows])
    if len(rows) < 4:
        return "<p>Not enough paired samples for quadrant gallery.</p>"
    m_med = float(np.nanmedian(morph))
    r_med = float(np.nanmedian(rq))
    groups = [
        ("spotty and rough", [row for row in rows if safe_float(row.get("median_morphology_index")) >= m_med and safe_float(row.get("afm_Rq_median_nm")) >= r_med]),
        ("streaky and smooth", [row for row in rows if safe_float(row.get("median_morphology_index")) < m_med and safe_float(row.get("afm_Rq_median_nm")) < r_med]),
        ("spotty but smooth", [row for row in rows if safe_float(row.get("median_morphology_index")) >= m_med and safe_float(row.get("afm_Rq_median_nm")) < r_med]),
        ("streaky but rough", [row for row in rows if safe_float(row.get("median_morphology_index")) < m_med and safe_float(row.get("afm_Rq_median_nm")) >= r_med]),
    ]
    sections = []
    for title, group in groups:
        sections.append(f"<h3>{html.escape(title)} (n={len(group)})</h3>")
        sections.append("<div class='gallery'>")
        sections.append(make_gallery_cards(group, paths, sort_key="sample_id"))
        sections.append("</div>")
    return "\n".join(sections)


def write_html_report(
    sample_rows: Sequence[dict[str, Any]],
    correlation_rows: Sequence[dict[str, Any]],
    cv_rows: Sequence[dict[str, Any]],
    perturbation_summary: Sequence[dict[str, Any]],
    material_rows: Sequence[dict[str, Any]],
    audit_summary: dict[str, Any],
    paths: Paths,
) -> None:
    top_corr = next(
        (
            row
            for row in correlation_rows
            if row["analysis_subset"] == "primary_1um_inclusive"
            and row["predictor"] == "median_morphology_index"
            and row["target_scale"] == "raw_nm"
        ),
        {},
    )
    for row in sample_rows:
        row["representative_afm_render_path"] = render_afm_image(str(row.get("sample_id", "")), str(row.get("representative_afm_path", "")), paths)
    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; color: #202124; }
    h1, h2, h3 { letter-spacing: 0; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top; }
    th { background: #f3f4f6; position: sticky; top: 0; }
    .metric { display: inline-block; margin: 6px 12px 6px 0; padding: 8px 10px; border: 1px solid #ddd; border-radius: 6px; background: #fafafa; }
    .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
    .card { border: 1px solid #d8d8d8; border-radius: 8px; padding: 10px; background: white; }
    .card h3 { margin: 0 0 6px 0; font-size: 15px; }
    .card p { font-size: 12px; margin: 8px 0 0 0; }
    .thumbs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; align-items: start; }
    .figure { max-width: 900px; margin: 12px 0 22px; }
    code { background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }
    """
    corr_table = table_html(correlation_rows[:30])
    cv_table = table_html(cv_rows)
    pert_table = table_html(perturbation_summary[:20])
    material_table = table_html(material_rows)
    score_gallery = make_gallery_cards(
        [row for row in sample_rows if int(row.get("included_in_inclusive_analysis", 0))],
        paths,
        sort_key="median_morphology_index",
    )
    rough_gallery = make_gallery_cards(
        [row for row in sample_rows if int(row.get("included_in_inclusive_analysis", 0))],
        paths,
        sort_key="afm_Rq_median_nm",
    )
    html_text = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>RHEED Roughness Audit</title>
<style>{css}</style>
</head>
<body>
<h1>RHEED Streak/Spot Score versus AFM Roughness</h1>
<p>This static report reuses the existing RHEED shape preprocessing and spot/streak geometry implementation. It reports association and diagnostics, not causal or optimized prediction claims.</p>
<div class="metric">paired samples: {audit_summary.get('paired_count', '')}</div>
<div class="metric">inclusive samples: {sum(int(row.get('included_in_inclusive_analysis', 0)) for row in sample_rows)}</div>
<div class="metric">strict-QC samples: {sum(int(row.get('included_in_strict_analysis', 0)) for row in sample_rows)}</div>
<div class="metric">primary morphology Spearman rho: {safe_float(top_corr.get('spearman_rho')):.3g}</div>
<div class="metric">permutation p: {safe_float(top_corr.get('spearman_permutation_p')):.3g}</div>

<h2>Data Overview</h2>
<div class="figure">{html_img('figures/data_overview.png', 'data overview')}</div>

<h2>Correlation Diagnostics</h2>
<div class="figure">{html_img('figures/correlation_morphology_vs_rq.svg', 'morphology scatter')}</div>
<div class="figure">{html_img('figures/correlation_spottiness_vs_rq.svg', 'spottiness scatter')}</div>
<div class="figure">{html_img('figures/correlation_streakiness_vs_rq.svg', 'streakiness scatter')}</div>
{corr_table}

<h2>Score-Sorted RHEED Gallery</h2>
<div class="gallery">{score_gallery}</div>

<h2>AFM-Roughness-Sorted Paired Gallery</h2>
<div class="gallery">{rough_gallery}</div>

<h2>Agreement and Counterexamples</h2>
{make_quadrant_gallery(sample_rows, paths)}

<h2>Temporal Diagnostics</h2>
<p>Per-sample traces are under <code>figures/temporal/</code>.</p>

<h2>Confound Diagnostics</h2>
<div class="figure">{html_img('figures/confound_correlation_heatmap.png', 'confound heatmap')}</div>

<h2>Group-Aware Prediction</h2>
{cv_table}

<h2>Material and Scan-Size Heterogeneity</h2>
<div class="figure">{html_img('figures/material_forest.png', 'material forest')}</div>
{material_table}

<h2>Perturbation Sensitivity</h2>
<div class="figure">{html_img('figures/perturbation_sensitivity.png', 'perturbation sensitivity')}</div>
{pert_table}

<h2>Manual Review</h2>
<p>Blinded review materials are in <code>manual_review/</code>. Human ratings were not fabricated; the validation-result table records the pending annotation status.</p>
</body>
</html>
"""
    (paths.reports_dir / "index.html").write_text(html_text, encoding="utf-8")


def table_html(rows: Sequence[dict[str, Any]], max_rows: int | None = None) -> str:
    if not rows:
        return "<p>No rows.</p>"
    rows = list(rows[:max_rows] if max_rows else rows)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    parts = ["<table>", "<thead><tr>"]
    parts.extend(f"<th>{html.escape(key)}</th>" for key in keys)
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for key in keys:
            value = row.get(key, "")
            if isinstance(value, float):
                text = "" if math.isnan(value) else f"{value:.4g}"
            else:
                text = str(value)
            parts.append(f"<td>{html.escape(text)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def write_manual_review_materials(frame_rows: Sequence[dict[str, Any]], sample_rows: Sequence[dict[str, Any]], config: dict[str, Any], paths: Paths) -> list[dict[str, Any]]:
    target = int(config["human_review"]["target_frames"])
    per_sample = int(config["human_review"]["frames_per_sample"])
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        by_sample[str(row.get("sample_id", ""))].append(row)
    selected = []
    for sample_id, rows in sorted(by_sample.items()):
        ordered = sorted(rows, key=lambda row: safe_float(row.get("frame_timestamp_sec"), 0.0))
        if not ordered:
            continue
        idxs = sorted({int(round(v)) for v in np.linspace(0, len(ordered) - 1, min(per_sample, len(ordered)))})
        selected.extend(ordered[index] for index in idxs)
    selected = sorted(selected, key=lambda row: (safe_float(row.get("morphology_index"), 0.0), str(row.get("sample_id", ""))))[:target]
    rng = np.random.default_rng(int(config["random_seed"]))
    rng.shuffle(selected)
    manifest = []
    template = []
    key_rows = []
    review_dir = paths.manual_review_dir
    contact_dir = review_dir / "review_contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(selected, start=1):
        review_id = f"HR{idx:04d}"
        src = resolve_path(paths.repo_root, str(row.get("raw_frame_path", "")))
        dest = review_dir / "frames" / f"{review_id}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file() and not dest.is_file():
            Image.open(src).save(dest)
        manifest.append(
            {
                "review_id": review_id,
                "image_path": display_path(dest, paths.repo_root),
                "presentation_order": idx,
            }
        )
        template.append(
            {
                "review_id": review_id,
                "ordinal_rating_1_to_5": "",
                "low_signal": "",
                "overexposed": "",
                "underexposed": "",
                "saturated": "",
                "off_center": "",
                "motion_blur": "",
                "roi_failure": "",
                "ring_like": "",
                "diffuse_amorphous": "",
                "multiple_overlapping_patterns": "",
                "uncertain": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "review_id": review_id,
                "sample_id": row.get("sample_id", ""),
                "frame_idx": row.get("frame_idx", ""),
                "morphology_index": row.get("morphology_index", ""),
                "raw_spottiness": row.get("raw_spottiness", ""),
                "raw_streakiness": row.get("raw_streakiness", ""),
                "afm_Rq_median_nm": next((s.get("afm_Rq_median_nm", "") for s in sample_rows if s.get("sample_id") == row.get("sample_id")), ""),
                "source_raw_frame_path": row.get("raw_frame_path", ""),
            }
        )
    write_csv_rows(review_dir / "blinded_review_manifest.csv", manifest)
    write_csv_rows(review_dir / "annotation_template.csv", template)
    write_csv_rows(review_dir / "unblind_key.csv", key_rows)
    instructions = """# Blinded RHEED Morphology Review

Rate each anonymous raw RHEED frame using:

1. strongly streaky
2. streaky-dominant
3. mixed streaks and spots
4. spotty-dominant
5. strongly spotty

Set artifact flags independently from the morphology rating. Do not use file
names or unblind keys while rating.
"""
    (review_dir / "instructions.md").write_text(instructions, encoding="utf-8")
    write_review_html(manifest, review_dir)
    write_contact_sheets(manifest, paths, contact_dir)
    return [
        {
            "analysis": "human_validation",
            "status": "pending_annotations",
            "n_review_frames": len(manifest),
            "spearman_human_vs_algorithm": math.nan,
            "weighted_cohens_kappa": math.nan,
            "rater_agreement": math.nan,
            "notes": "Blinded review set generated; fill annotation_template.csv and rerun human validation analysis.",
        }
    ]


def write_review_html(manifest: Sequence[dict[str, Any]], review_dir: Path) -> None:
    cards = []
    for row in manifest:
        img = Path(str(row["image_path"])).name
        cards.append(
            f"<article><h3>{html.escape(str(row['review_id']))}</h3>"
            f"<img src='frames/{html.escape(img)}' style='width:220px;height:220px;object-fit:contain;background:#111'>"
            "</article>"
        )
    text = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Blinded RHEED Review</title>"
        "<style>body{font-family:sans-serif;margin:24px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:14px}article{border:1px solid #ddd;border-radius:8px;padding:10px}</style>"
        "</head><body><h1>Blinded RHEED Review</h1><p>Use the annotation template. Algorithm scores and sample identities are hidden here.</p><div class='grid'>"
        + "\n".join(cards)
        + "</div></body></html>"
    )
    (review_dir / "review.html").write_text(text, encoding="utf-8")


def write_contact_sheets(manifest: Sequence[dict[str, Any]], paths: Paths, contact_dir: Path) -> None:
    per_sheet = 20
    for sheet_idx, start in enumerate(range(0, len(manifest), per_sheet), start=1):
        rows = manifest[start : start + per_sheet]
        cols = 5
        fig_rows = math.ceil(len(rows) / cols)
        fig, axes = plt.subplots(fig_rows, cols, figsize=(cols * 2.2, fig_rows * 2.5), dpi=140, squeeze=False)
        for axis in axes.ravel():
            axis.axis("off")
        for axis, row in zip(axes.ravel(), rows):
            image = Image.open(resolve_path(paths.repo_root, row["image_path"]))
            axis.imshow(image, cmap="gray")
            axis.set_title(str(row["review_id"]), fontsize=8)
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(contact_dir / f"contact_sheet_{sheet_idx:02d}.png")
        plt.close(fig)


def write_text_reports(
    paths: Paths,
    config: dict[str, Any],
    audit_summary: dict[str, Any],
    afm_summary: dict[str, Any],
    correlation_rows: Sequence[dict[str, Any]],
    cv_rows: Sequence[dict[str, Any]],
    perturbation_summary: Sequence[dict[str, Any]],
    human_rows: Sequence[dict[str, Any]],
) -> None:
    primary = next(
        (
            row
            for row in correlation_rows
            if row["analysis_subset"] == "primary_1um_inclusive"
            and row["predictor"] == "median_morphology_index"
            and row["target_scale"] == "raw_nm"
        ),
        {},
    )
    strict = next(
        (
            row
            for row in correlation_rows
            if row["analysis_subset"] == "primary_1um_strict_qc"
            and row["predictor"] == "median_morphology_index"
            and row["target_scale"] == "raw_nm"
        ),
        {},
    )
    best_pert = perturbation_summary[0] if perturbation_summary else {}
    score_pred = next((row for row in cv_rows if row["model"] == "negative_control_score_from_nuisance"), {})
    results = f"""# RHEED Roughness Results

## Direct Answer

The primary inclusive 1.0 um analysis found Spearman rho =
{safe_float(primary.get('spearman_rho')):.4g} for the frozen morphology index
versus AFM Rq, with bootstrap CI
[{safe_float(primary.get('spearman_bootstrap_ci_low')):.4g},
{safe_float(primary.get('spearman_bootstrap_ci_high')):.4g}] and sample-level
permutation p = {safe_float(primary.get('spearman_permutation_p')):.4g}
(n = {primary.get('n_samples', '')}). The strict-QC counterpart found rho =
{safe_float(strict.get('spearman_rho')):.4g} (n = {strict.get('n_samples', '')}).

This should be read as an association audit. It is not evidence of causation,
and it is material/setup specific unless replicated outside this dataset.

## Required Questions

1. Visual measurement of streaky-to-spotty axis: the existing component
   geometry detector was reused. Inspect `index.html` galleries sorted by score;
   the report shows both agreement cases and counterexamples.
2. Human ratings: blinded review materials were generated, but ratings were not
   fabricated. `human_validation_results.csv` is pending annotation.
3. Sensitivity: the largest median perturbation effect was
   `{best_pert.get('perturbation', '')}` with median absolute score change
   {safe_float(best_pert.get('median_absolute_score_change')):.4g}.
4. AFM association: see the primary rho and permutation p above.
5. Confound adjustment: see `regression_results.csv`; nuisance residualization
   and group-aware models are reported separately from raw correlation.
6. Out-of-fold prediction: see `cv_model_comparison.csv`; all metrics are
   leave-one-growth-run-out.
7. Materials/batches: see `material_stratified_results.csv`; small strata are
   explicitly labeled too few samples.
8. Counterexamples: see the agreement/disagreement quadrants in `index.html`.
9. Color-bar/peak-to-valley: `sample_level_analysis_table.csv` separates Rq,
   Ra, robust range, and peak-to-valley span.
10. Use recommendation: treat the score as an exploratory morphology diagnostic
    unless the strict-QC and confound-adjusted results are strong enough for the
    specific material/setup in question.

## Nuisance Warning

The score-predictability audit from nuisance variables produced out-of-fold R2 =
{safe_float(score_pred.get('out_of_fold_r2')):.4g} and Spearman =
{safe_float(score_pred.get('out_of_fold_spearman')):.4g}. High values here
would mean the score is partly an acquisition artifact proxy.
"""
    (paths.reports_dir / "results.md").write_text(textwrap.dedent(results).strip() + "\n", encoding="utf-8")

    methods = f"""# RHEED Roughness Methods

## Reused Implementation

- RHEED preprocessing: `rheed2morph.rheed.shape_preprocessing.preprocess_frame_for_shape`
- Frame quality: `rheed2morph.rheed.frame_quality.extract_frame_quality_features`
- Spot/streak geometry: `rheed2morph.rheed.spot_streak_geometry.extract_components_and_frame_features`

## Reproduction Command

Run from the repository root:

```bash
PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_roughness.run --config configs/rheed_roughness.yaml
```

## Frozen RHEED Score

The pre-specified score is:

- spottiness = {config['rheed']['morphology_score_definition']['spottiness']}
- streakiness = {config['rheed']['morphology_score_definition']['streakiness']}
- morphology_index = {config['rheed']['morphology_score_definition']['morphology_index']}

The formula was fixed before reading AFM correlations.

## AFM Target

Plane-corrected ZSensor height maps were converted to nanometers and Rq was
recomputed directly from the physical height map. Metadata and descriptor Rq
values are retained for validation.

Primary target: sample-level median Rq for scans within
{config['afm']['primary_scan_size_um']} +/- {config['afm']['primary_scan_size_tolerance_um']} um.

## Statistics

The independent unit is `sample_id`/`growth_run_id`. Primary association uses
Spearman and Kendall rank correlations, sample-level bootstrap confidence
intervals, and sample-level permutation tests. Predictive metrics use
leave-one-growth-run-out predictions only.
"""
    (paths.reports_dir / "methods.md").write_text(textwrap.dedent(methods).strip() + "\n", encoding="utf-8")

    limitations = f"""# RHEED Roughness Limitations

- Material labels are inferred from existing manifests and filenames; several
  labels are scan-size or sample-token-like values rather than clean chemistry.
- Substrate, azimuth, rotation period, exposure, and camera metadata are mostly
  unavailable, so confound adjustment is limited to measurable image proxies.
- Human validation cannot be completed until blinded ratings are entered.
- The RHEED score operates on pre-cropped videos; raw full-frame ROI failures
  are not fully represented.
- No samples were removed because they weakened the expected relationship.
- Peak-to-valley/color-bar span is reported separately and is not interpreted as
  RMS roughness.

Data audit summary: {json.dumps(json_ready(audit_summary), sort_keys=True)[:1200]}

AFM audit summary: {json.dumps(json_ready(afm_summary), sort_keys=True)[:1200]}
"""
    (paths.reports_dir / "limitations.md").write_text(textwrap.dedent(limitations).strip() + "\n", encoding="utf-8")


def write_data_audit(
    paths: Paths,
    config: dict[str, Any],
    schema_rows: Sequence[dict[str, Any]],
    schema_summary: dict[str, Any],
    pairing_summary: dict[str, Any],
    candidate_rows: Sequence[dict[str, str]],
    representative_rows: Sequence[dict[str, str]],
    video_rows: Sequence[dict[str, Any]],
    afm_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    material_counts = Counter(row.get("material", "") for row in representative_rows)
    scan_counts = Counter(round(safe_float(row.get("scan_size_um")), 3) for row in afm_rows if math.isfinite(safe_float(row.get("scan_size_um"))))
    resolution_counts = Counter(row.get("afm_resolution", "") for row in afm_rows)
    video_res_counts = Counter(row.get("video_resolution", "") for row in video_rows)
    summary = {
        "modules_reused": {
            "RHEED video reading": "imageio.v2.get_reader over pre-cropped crop videos",
            "ROI extraction": "existing data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 crop-video dataset",
            "video stabilization": "not available; centroid shift proxies are computed",
            "streak_spot_feature_extraction": "rheed2morph.rheed.spot_streak_geometry.extract_components_and_frame_features",
            "rheed_preprocessing": "rheed2morph.rheed.shape_preprocessing.preprocess_frame_for_shape",
            "frame_quality": "rheed2morph.rheed.frame_quality.extract_frame_quality_features",
            "AFM processing": "plane-corrected height maps from data/plane_corrected_afm; Rq/Ra recomputed",
            "metadata_parsing": "recursive metadata JSON schema discovery in processed and plane-corrected AFM roots",
            "sample_pairing": "existing manifest_all_size_representative_one_to_one.csv and afm_candidate_table_complete.csv",
        },
        "data_roots": config["data_roots"],
        "counts": {
            "growth_runs": len({row.get("group_id", row.get("sample_id", "")) for row in representative_rows}),
            "sample_groups": len({row.get("sample_id", "") for row in representative_rows}),
            "representative_pairs": len(representative_rows),
            "candidate_table_rows": len(candidate_rows),
            "afm_scan_level_targets": len(afm_rows),
            "rheed_videos_processed": len(video_rows),
            "metadata_json_files": schema_summary.get("metadata_file_count", 0),
            **pairing_summary,
        },
        "distributions": {
            "materials": dict(material_counts),
            "afm_scan_sizes_um": dict(scan_counts),
            "afm_resolutions": dict(resolution_counts),
            "height_units": dict(Counter(row.get("height_unit_exported", "") for row in afm_rows)),
            "video_resolutions": dict(video_res_counts),
            "video_durations_sec": {
                "min": float(np.nanmin([safe_float(row.get("video_duration_sec")) for row in video_rows])) if video_rows else math.nan,
                "median": float(np.nanmedian([safe_float(row.get("video_duration_sec")) for row in video_rows])) if video_rows else math.nan,
                "max": float(np.nanmax([safe_float(row.get("video_duration_sec")) for row in video_rows])) if video_rows else math.nan,
            },
        },
        "metadata_schema_summary": schema_summary,
        "substrate_rotation": {
            "used": "unknown",
            "period_available": False,
            "notes": "No explicit rotation-period metadata field was found; final-window fallback was used.",
        },
    }
    write_json(paths.reports_dir / "data_audit.json", summary)
    write_csv_rows(paths.outputs_dir / "metadata_json_schema_keys.csv", schema_rows)
    md = [
        "# RHEED Roughness Data Audit",
        "",
        "## Reused Files and Functions",
    ]
    for name, value in summary["modules_reused"].items():
        md.append(f"- {name}: `{value}`")
    md.extend(
        [
            "",
            "## Counts",
            "",
            table_markdown(summary["counts"]),
            "",
            "## Distributions",
            "",
            f"- materials: `{dict(material_counts)}`",
            f"- AFM scan sizes: `{dict(scan_counts)}`",
            f"- AFM resolutions: `{dict(resolution_counts)}`",
            f"- video resolutions: `{dict(video_res_counts)}`",
            "",
            "## Metadata Roughness Candidates",
            "",
        ]
    )
    for key, count in list(schema_summary.get("roughness_candidate_keys", {}).items())[:40]:
        md.append(f"- `{key}`: {count}")
    md.extend(
        [
            "",
            "## Pairing Notes",
            "",
            f"- Paired rows: {pairing_summary.get('paired_count', 0)}",
            f"- Unmatched rows: {pairing_summary.get('unmatched_count', 0)}",
            f"- Crop-video issues: {pairing_summary.get('crop_video_issue_count', 0)}",
            "- Ambiguous pairings are reported in `data_audit.json`; no AFM-outcome-based filtering is applied.",
        ]
    )
    (paths.reports_dir / "data_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def table_markdown(mapping: dict[str, Any]) -> str:
    rows = ["| key | value |", "| --- | ---: |"]
    for key, value in mapping.items():
        if isinstance(value, dict | list):
            text = json.dumps(json_ready(value))[:160]
        else:
            text = str(value)
        rows.append(f"| `{key}` | {text} |")
    return "\n".join(rows)


def write_all_outputs(
    paths: Paths,
    pairing_rows: Sequence[dict[str, Any]],
    frame_rows: Sequence[dict[str, Any]],
    video_rows: Sequence[dict[str, Any]],
    afm_rows: Sequence[dict[str, Any]],
    sample_rows: Sequence[dict[str, Any]],
    ledger_rows: Sequence[dict[str, Any]],
    correlation_rows: Sequence[dict[str, Any]],
    regression_rows: Sequence[dict[str, Any]],
    cv_rows: Sequence[dict[str, Any]],
    material_rows: Sequence[dict[str, Any]],
    perturbation_rows: Sequence[dict[str, Any]],
    perturbation_summary: Sequence[dict[str, Any]],
    human_rows: Sequence[dict[str, Any]],
    negative_control_rows: Sequence[dict[str, Any]],
) -> None:
    write_csv_rows(paths.outputs_dir / "pairing_audit.csv", pairing_rows)
    write_csv_rows(paths.outputs_dir / "frame_level_scores.csv", frame_rows)
    write_parquet_or_note(paths.outputs_dir / "frame_level_scores.parquet", frame_rows)
    write_csv_rows(paths.outputs_dir / "video_level_scores.csv", video_rows)
    write_csv_rows(paths.outputs_dir / "afm_scan_level_targets.csv", afm_rows)
    write_csv_rows(paths.outputs_dir / "sample_level_analysis_table.csv", sample_rows)
    write_csv_rows(paths.outputs_dir / "qc_exclusion_ledger.csv", ledger_rows)
    write_csv_rows(paths.outputs_dir / "correlation_results.csv", correlation_rows)
    write_csv_rows(paths.outputs_dir / "regression_results.csv", regression_rows)
    write_csv_rows(paths.outputs_dir / "cv_model_comparison.csv", cv_rows)
    write_csv_rows(paths.outputs_dir / "material_stratified_results.csv", material_rows)
    write_csv_rows(paths.outputs_dir / "perturbation_results.csv", perturbation_rows)
    write_csv_rows(paths.outputs_dir / "perturbation_summary.csv", perturbation_summary)
    write_csv_rows(paths.outputs_dir / "human_validation_results.csv", human_rows)
    write_csv_rows(paths.outputs_dir / "negative_control_results.csv", negative_control_rows)


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    config = read_config(args.config)
    paths = make_paths(config, smoke=args.smoke)
    np.random.seed(int(config["random_seed"]))
    manifest_cfg = config["manifests"]
    data_roots = config["data_roots"]
    candidate_rows = read_csv_rows(resolve_path(paths.repo_root, manifest_cfg["candidate_table"]))
    representative_rows = read_csv_rows(resolve_path(paths.repo_root, manifest_cfg["representative_one_to_one"]))
    if args.limit_samples:
        keep = {str(row["sample_id"]) for row in representative_rows[: args.limit_samples]}
        representative_rows = [row for row in representative_rows if row["sample_id"] in keep]
        candidate_rows = [row for row in candidate_rows if row.get("sample_id") in keep]
    crop_videos, crop_issues = discover_crop_videos(resolve_path(paths.repo_root, data_roots["crop_video_root"]))
    pairing_rows, pairing_summary = build_pairing_audit(paths.repo_root, representative_rows, candidate_rows, crop_videos, crop_issues)
    schema_rows, schema_summary = discover_json_schema(
        [
            resolve_path(paths.repo_root, data_roots["processed_afm_root"]),
            resolve_path(paths.repo_root, data_roots["plane_corrected_afm_root"]),
        ]
    )
    afm_rows, afm_summary = extract_afm_targets(paths.repo_root, candidate_rows, config, paths)
    afm_sample_targets, afm_sample_summary = aggregate_afm_by_sample(afm_rows, config)
    frame_rows, video_rows, rheed_failures = run_rheed_extraction(
        representative_rows,
        crop_videos,
        config,
        paths,
        smoke=args.smoke,
        use_cache=not args.no_cache,
    )
    if rheed_failures:
        write_csv_rows(paths.outputs_dir / "rheed_extraction_failures.csv", rheed_failures)
    sample_rows, ledger_rows = build_sample_analysis_table(representative_rows, video_rows, afm_sample_targets, config)
    data_audit = write_data_audit(
        paths,
        config,
        schema_rows,
        schema_summary,
        pairing_summary,
        candidate_rows,
        representative_rows,
        video_rows,
        afm_rows,
    )
    correlation_rows = correlation_analysis(sample_rows, config, paths, smoke=args.smoke)
    regression_rows, cv_rows = regression_and_cv_analysis(sample_rows, config)
    material_rows = material_stratified_results(sample_rows, config)
    perturbation_rows = run_perturbation_audit(frame_rows, config, paths, smoke=args.smoke)
    perturbation_summary = summarize_perturbations(perturbation_rows)
    human_rows = write_manual_review_materials(frame_rows, sample_rows, config, paths)
    negative_control_rows = negative_controls(frame_rows, sample_rows, perturbation_summary)

    # Figures depend on the sample table, including AFM render paths populated in the HTML writer.
    make_data_overview_figures(sample_rows, afm_rows, pairing_summary, paths)
    scatter_rows = [row for row in sample_rows if int(row.get("included_in_inclusive_analysis", 0)) and bool(row.get("has_primary_scan_size"))]
    write_scatter_svg(
        paths.figures_dir / "correlation_morphology_vs_rq.svg",
        scatter_rows,
        "median_morphology_index",
        "afm_Rq_median_nm",
        title="Morphology index vs AFM Rq",
        x_label="median morphology index",
        y_label="AFM Rq median (nm)",
        increasing=True,
    )
    write_scatter_svg(
        paths.figures_dir / "correlation_spottiness_vs_rq.svg",
        scatter_rows,
        "median_spottiness",
        "afm_Rq_median_nm",
        title="Spottiness vs AFM Rq",
        x_label="median spottiness",
        y_label="AFM Rq median (nm)",
        increasing=True,
    )
    write_scatter_svg(
        paths.figures_dir / "correlation_streakiness_vs_rq.svg",
        scatter_rows,
        "median_streakiness",
        "afm_Rq_median_nm",
        title="Streakiness vs AFM Rq",
        x_label="median streakiness",
        y_label="AFM Rq median (nm)",
        increasing=False,
    )
    write_temporal_plots(video_rows, frame_rows, paths)
    write_confound_figures(sample_rows, paths)
    write_perturbation_figure(perturbation_summary, paths)
    write_material_forest(material_rows, paths)
    write_html_report(sample_rows, correlation_rows, cv_rows, perturbation_summary, material_rows, data_audit["counts"], paths)
    write_text_reports(paths, config, data_audit, {**afm_summary, **afm_sample_summary}, correlation_rows, cv_rows, perturbation_summary, human_rows)
    write_all_outputs(
        paths,
        pairing_rows,
        frame_rows,
        video_rows,
        afm_rows,
        sample_rows,
        ledger_rows,
        correlation_rows,
        regression_rows,
        cv_rows,
        material_rows,
        perturbation_rows,
        perturbation_summary,
        human_rows,
        negative_control_rows,
    )
    manifest = {
        "outputs_dir": display_path(paths.outputs_dir, paths.repo_root),
        "reports_dir": display_path(paths.reports_dir, paths.repo_root),
        "frame_rows": len(frame_rows),
        "video_rows": len(video_rows),
        "afm_scan_rows": len(afm_rows),
        "sample_rows": len(sample_rows),
        "rheed_failures": rheed_failures,
        "primary_result": next(
            (
                row
                for row in correlation_rows
                if row["analysis_subset"] == "primary_1um_inclusive"
                and row["predictor"] == "median_morphology_index"
                and row["target_scale"] == "raw_nm"
            ),
            {},
        ),
    }
    write_json(paths.outputs_dir / "run_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run RHEED streak/spot versus AFM roughness audit.")
    parser.add_argument("--config", type=Path, default=Path("configs/rheed_roughness.yaml"))
    parser.add_argument("--smoke", action="store_true", help="Run a small smoke path with fewer frames/resamples.")
    parser.add_argument("--limit-samples", type=int, default=0, help="Limit representative samples for debugging.")
    parser.add_argument("--no-cache", action="store_true", help="Recompute frame-level features even if cache exists.")
    args = parser.parse_args(argv)
    manifest = run_pipeline(args)
    print("RHEED roughness analysis complete")
    print(json.dumps(json_ready(manifest), indent=2))


if __name__ == "__main__":
    main()
