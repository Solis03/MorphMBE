"""Manual RHEED screenshot to AFM height-map visualization.

This module intentionally does not decode RHEED videos or rank video frames.
It uses only manually selected image files whose basename starts with
``select`` and pairs them to existing physical AFM height arrays by exact
normalized sample ID.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import numpy as np
from PIL import Image

from analysis.rheed_roughness.run import (
    convert_height_to_nm,
    csv_value,
    display_path,
    read_config,
    read_csv_rows,
    resolve_path,
    safe_float,
    write_csv_rows,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RE = re.compile(r"N?(\d{4})", re.IGNORECASE)
VALID_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
IGNORED_SUFFIXES = {".mov", ".mp4", ".avi", ".mkv", ".m4v", ".mts", ".m2ts"}
IGNORED_NAME_PARTS = {"zone.identifier"}
MANUAL_AUDIT_FIELDS = [
    "manual_folder",
    "normalized_manual_folder",
    "manual_rheed_path",
    "manual_rheed_filename",
    "manual_candidate_count",
    "all_manual_candidates",
    "sample_id",
    "sample_group_id",
    "growth_run_id",
    "match_status",
    "match_method",
    "possible_matches",
    "warnings",
    "included_in_figure",
]
AFM_AUDIT_FIELDS = [
    "sample_id",
    "sample_group_id",
    "growth_run_id",
    "selected_afm_scan_id",
    "selected_afm_path",
    "selected_height_map_path",
    "channel",
    "scan_size_um",
    "scan_size_x_um",
    "scan_size_y_um",
    "resolution_x",
    "resolution_y",
    "rq_nm",
    "rq_source",
    "ra_nm",
    "robust_height_range_nm",
    "peak_to_valley_nm",
    "native_display_min_nm",
    "native_display_max_nm",
    "common_display_min_nm",
    "common_display_max_nm",
    "number_of_candidate_scans",
    "sample_median_rq_nm",
    "distance_from_median_rq_nm",
    "selection_reason",
    "qc_flags",
]
MANIFEST_FIELDS = [
    "sample_id",
    "sample_group_id",
    "growth_run_id",
    "manual_rheed_folder",
    "manual_rheed_path",
    "manual_rheed_filename",
    "manual_selection_status",
    "selected_afm_scan_id",
    "selected_afm_path",
    "selected_height_map_path",
    "rq_nm",
    "rq_source",
    "ra_nm",
    "scan_size_um",
    "afm_resolution",
    "included_in_native_figure",
    "included_in_common_scale_figure",
    "skip_reason",
    "warnings",
]


@dataclass(frozen=True)
class ManualVizPaths:
    repo_root: Path
    manual_root: Path
    output_dir: Path
    report_dir: Path
    cache_dir: Path
    pages_dir: Path
    gallery_assets_dir: Path
    plane_corrected_afm_root: Path
    descriptor_csv: Path
    source_outputs_dir: Path


@dataclass(frozen=True)
class ManualSelection:
    manual_folder: Path
    normalized_manual_folder: str
    sample_id: str
    selected_path: Path | None
    candidates: tuple[Path, ...]
    warnings: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class AFMCandidate:
    sample_id: str
    sample_group_id: str
    growth_run_id: str
    material: str
    afm_scan_id: str
    afm_path: Path
    selected_height_map_path: Path
    channel: str
    height_unit_exported: str
    scan_size_um: float
    scan_size_x_um: float
    scan_size_y_um: float
    resolution_x: int
    resolution_y: int
    rq_nm: float
    rq_source: str
    rq_recomputed_nm: float
    ra_nm: float
    robust_height_range_nm: float
    peak_to_valley_nm: float
    qc_flags: str


@dataclass(frozen=True)
class SelectedPair:
    sample_id: str
    sample_group_id: str
    growth_run_id: str
    material: str
    manual_folder: Path
    manual_rheed_path: Path
    manual_rheed_filename: str
    all_manual_candidates: tuple[Path, ...]
    manual_warnings: tuple[str, ...]
    afm: AFMCandidate
    number_of_candidate_scans: int
    sample_median_rq_nm: float
    distance_from_median_rq_nm: float
    selection_reason: str
    native_display_min_nm: float
    native_display_max_nm: float
    common_display_min_nm: float | None = None
    common_display_max_nm: float | None = None
    cached_manual_rheed_path: Path | None = None
    cached_afm_native_path: Path | None = None
    cached_afm_common_path: Path | None = None


def build_paths(config: dict[str, Any], manual_selection_root: Path) -> ManualVizPaths:
    """Resolve all output and input paths for manual pair visualization."""
    repo_root = resolve_path(REPO_ROOT, config.get("repo_root", ".")).resolve()
    outputs = resolve_path(repo_root, config["outputs_dir"]).resolve()
    reports = resolve_path(repo_root, config["reports_dir"]).resolve()
    output_dir = outputs / "manual_pair_visualization"
    report_dir = reports / "manual_pair_visualization"
    cache_dir = output_dir / "cache"
    pages_dir = report_dir / "pages"
    gallery_assets = report_dir / "assets"
    plane_root = resolve_path(repo_root, config.get("data_roots", {}).get("plane_corrected_afm_root", "data/plane_corrected_afm"))
    descriptor_csv = repo_root / "data" / "afm_descriptor_reconstruction" / "afm_descriptors.csv"
    for path in (output_dir, report_dir, cache_dir, pages_dir, gallery_assets):
        path.mkdir(parents=True, exist_ok=True)
    return ManualVizPaths(
        repo_root=repo_root,
        manual_root=resolve_path(repo_root, manual_selection_root).resolve(),
        output_dir=output_dir,
        report_dir=report_dir,
        cache_dir=cache_dir,
        pages_dir=pages_dir,
        gallery_assets_dir=gallery_assets,
        plane_corrected_afm_root=plane_root.resolve(),
        descriptor_csv=descriptor_csv,
        source_outputs_dir=outputs,
    )


def normalize_manual_sample_id(value: str | Path) -> str:
    """Return the exact four-digit sample ID embedded in a path component."""
    match = SAMPLE_RE.search(str(value))
    return match.group(1) if match else ""


def is_ignored_file(path: Path) -> bool:
    """Return true for hidden, metadata, video, or temporary files."""
    name = path.name
    lowered = name.lower()
    if name.startswith(".") or name.startswith("._") or lowered.endswith("~"):
        return True
    if any(part in lowered for part in IGNORED_NAME_PARTS):
        return True
    if path.suffix.lower() in IGNORED_SUFFIXES:
        return True
    if "temp" in lowered and path.suffix.lower() == ".bmp":
        return True
    return False


def valid_select_image(path: Path) -> bool:
    """Return true only for supported image files whose basename starts select."""
    if is_ignored_file(path):
        return False
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES and path.stem.lower().startswith("select")


def manual_selection_priority(path: Path) -> tuple[int, str]:
    """Deterministic priority for multiple manual select images."""
    stem = path.stem.lower()
    if stem == "select":
        rank = 0
    elif stem.startswith("select_final"):
        rank = 1
    elif stem.startswith("select_best"):
        rank = 2
    else:
        rank = 3
    return rank, path.name.lower()


def sample_folder_for_path(path: Path, manual_root: Path) -> Path | None:
    """Return the first descendant folder under manual_root containing a sample ID."""
    try:
        rel_parts = path.resolve().relative_to(manual_root.resolve()).parts
    except ValueError:
        return None
    cur = manual_root
    for part in rel_parts[:-1]:
        cur = cur / part
        if normalize_manual_sample_id(part):
            return cur
    return None


def discover_manual_rheed_images(manual_root: Path) -> list[ManualSelection]:
    """Discover manual sample folders and select approved select* RHEED images."""
    sample_dirs: dict[str, Path] = {}
    if not manual_root.is_dir():
        return []
    for child in sorted(manual_root.iterdir()):
        if child.is_dir() and normalize_manual_sample_id(child.name):
            sample_dirs[display_path(child, REPO_ROOT)] = child
    for path in sorted(manual_root.rglob("*")):
        sid = normalize_manual_sample_id(path.parent)
        folder = sample_folder_for_path(path, manual_root)
        if folder is not None:
            sample_dirs.setdefault(display_path(folder, REPO_ROOT), folder)
    grouped: dict[Path, list[Path]] = defaultdict(list)
    for path in sorted(manual_root.rglob("*")):
        if valid_select_image(path):
            folder = sample_folder_for_path(path, manual_root)
            if folder is not None:
                grouped[folder].append(path)
    selections: list[ManualSelection] = []
    for folder in sorted(sample_dirs.values(), key=lambda p: display_path(p, REPO_ROOT)):
        sid = normalize_manual_sample_id(folder.name)
        candidates = tuple(sorted(grouped.get(folder, []), key=lambda p: p.name.lower()))
        warnings: list[str] = []
        if not candidates:
            selections.append(
                ManualSelection(
                    manual_folder=folder,
                    normalized_manual_folder=sid,
                    sample_id=sid,
                    selected_path=None,
                    candidates=(),
                    warnings=(),
                    status="missing_manual_selection",
                )
            )
            continue
        selected = sorted(candidates, key=manual_selection_priority)[0]
        if len(candidates) > 1:
            warnings.append("multiple_manual_selections")
        selections.append(
            ManualSelection(
                manual_folder=folder,
                normalized_manual_folder=sid,
                sample_id=sid,
                selected_path=selected,
                candidates=candidates,
                warnings=tuple(warnings),
                status="ok",
            )
        )
    return selections


def rel(path: Path | None, repo_root: Path) -> str:
    return display_path(path, repo_root) if path is not None else ""


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def metadata_path_for_height(path: Path) -> Path | None:
    base = path.name.removesuffix("_plane_corrected.npy")
    for candidate in (path.with_name(f"{base}_plane_corrected_metadata.json"), path.with_name(f"{base}_metadata.json")):
        if candidate.is_file():
            return candidate
    return None


def scan_id_from_height_path(path: Path) -> str:
    return path.stem.removesuffix("_plane_corrected").removesuffix("_height")


def parse_scan_size_pair(metadata: dict[str, Any], fallback: float = math.nan) -> tuple[float, float, float]:
    """Return mean, x, and y scan size in um."""
    value = metadata.get("scan_size_um")
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x = normalize_scan_size_value(value[0])
        y = normalize_scan_size_value(value[1])
        if math.isfinite(x) and math.isfinite(y):
            return float((x + y) / 2.0), x, y
    if isinstance(value, (int, float)):
        size = normalize_scan_size_value(value)
        if math.isfinite(size):
            return size, size, size
    if math.isfinite(fallback):
        return fallback, fallback, fallback
    return math.nan, math.nan, math.nan


def scan_size_from_filename_pair(metadata: dict[str, Any]) -> tuple[float, float, float]:
    """Return filename-parsed scan size metadata when present."""
    value = metadata.get("scan_size_from_filename_um")
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x = normalize_scan_size_value(value[0])
        y = normalize_scan_size_value(value[1])
        if math.isfinite(x) and math.isfinite(y):
            return float((x + y) / 2.0), x, y
    if isinstance(value, (int, float)):
        size = normalize_scan_size_value(value)
        if math.isfinite(size):
            return size, size, size
    return math.nan, math.nan, math.nan


def normalize_scan_size_value(value: Any) -> float:
    size = safe_float(value)
    if not math.isfinite(size) or size <= 0:
        return math.nan
    if 50.0 <= size <= 20_000.0:
        return size / 1000.0
    return size


def infer_material(path: Path, metadata: dict[str, Any]) -> str:
    for key in ("raw_file", "raw_afm_file", "relative_path"):
        value = metadata.get(key)
        if isinstance(value, str):
            match = re.search(r"\b([A-Z][a-z]Sb)\b", value)
            if match:
                return match.group(1)
    match = re.search(r"\b([A-Z][a-z]Sb)\b", path.as_posix())
    return match.group(1) if match else ""


def descriptor_lookup(path: Path, repo_root: Path) -> dict[str, dict[str, str]]:
    """Load AFM descriptor rows keyed by repo-relative AFM path."""
    if not path.is_file():
        return {}
    rows = read_csv_rows(path)
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        afm_path = resolve_path(repo_root, row.get("afm_path", ""))
        lookup[display_path(afm_path, repo_root)] = row
    return lookup


def load_height_nm(path: Path, unit: str | None) -> tuple[np.ndarray, str]:
    values = np.load(path)
    height, status = convert_height_to_nm(values, unit)
    return np.asarray(height, dtype=np.float64), status


def recompute_height_stats(path: Path, unit: str | None) -> dict[str, float]:
    height, _ = load_height_nm(path, unit)
    finite = height[np.isfinite(height)]
    if finite.size == 0:
        return {
            "rq": math.nan,
            "ra": math.nan,
            "robust_height_range": math.nan,
            "peak_to_valley": math.nan,
            "p01": math.nan,
            "p99": math.nan,
        }
    centered = finite - float(np.mean(finite))
    p01, p05, p95, p99 = np.percentile(finite, [1, 5, 95, 99])
    return {
        "rq": float(np.sqrt(np.mean(centered * centered))),
        "ra": float(np.mean(np.abs(centered))),
        "robust_height_range": float(p95 - p05),
        "peak_to_valley": float(np.max(finite) - np.min(finite)),
        "p01": float(p01),
        "p99": float(p99),
    }


def load_afm_candidates(paths: ManualVizPaths) -> list[AFMCandidate]:
    """Load valid physical AFM candidates from plane-corrected arrays."""
    desc = descriptor_lookup(paths.descriptor_csv, paths.repo_root)
    candidates: list[AFMCandidate] = []
    for afm_path in sorted(paths.plane_corrected_afm_root.glob("*/*/*_plane_corrected.npy")):
        metadata_path = metadata_path_for_height(afm_path)
        metadata = load_json(metadata_path)
        sample_id = str(metadata.get("sample_id") or normalize_manual_sample_id(afm_path))
        if not sample_id:
            continue
        rel_path = display_path(afm_path, paths.repo_root)
        desc_row = desc.get(rel_path, {})
        unit = str(metadata.get("height_unit_exported") or metadata.get("height_unit_original") or "nm")
        resolution = metadata.get("resolution")
        if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
            resolution_y = int(safe_float(resolution[0], 0))
            resolution_x = int(safe_float(resolution[1], 0))
        else:
            try:
                arr = np.load(afm_path, mmap_mode="r")
                resolution_y, resolution_x = int(arr.shape[-2]), int(arr.shape[-1])
            except Exception:
                resolution_y, resolution_x = 0, 0
        scan_size, scan_x, scan_y = parse_scan_size_pair(metadata)
        filename_scan_size, filename_scan_x, filename_scan_y = scan_size_from_filename_pair(metadata)
        channel = str(metadata.get("primary_channel") or "ZSensor")
        rq = safe_float(desc_row.get("Rq"), math.nan)
        ra = safe_float(desc_row.get("Ra"), math.nan)
        robust_range = safe_float(desc_row.get("p95"), math.nan) - safe_float(desc_row.get("p05"), math.nan)
        peak_to_valley = safe_float(desc_row.get("peak_to_valley"), math.nan)
        rq_source = "loaded_from_descriptor_table" if math.isfinite(rq) else "recomputed_from_height_map"
        recomputed_rq = math.nan
        qc_flags: list[str] = []
        if not math.isfinite(rq) or not math.isfinite(ra) or not math.isfinite(robust_range) or not math.isfinite(peak_to_valley):
            stats = recompute_height_stats(afm_path, unit)
            rq = rq if math.isfinite(rq) else stats["rq"]
            ra = ra if math.isfinite(ra) else stats["ra"]
            robust_range = robust_range if math.isfinite(robust_range) else stats["robust_height_range"]
            peak_to_valley = peak_to_valley if math.isfinite(peak_to_valley) else stats["peak_to_valley"]
            recomputed_rq = stats["rq"]
        elif metadata_path is not None:
            stats = recompute_height_stats(afm_path, unit)
            recomputed_rq = stats["rq"]
            if math.isfinite(recomputed_rq) and abs(recomputed_rq - rq) > max(0.2, 0.2 * abs(rq)):
                qc_flags.append("rq_descriptor_recompute_disagreement")
        _, unit_status = convert_height_to_nm(np.asarray([0.0]), unit)
        if unit_status != "ok":
            qc_flags.append(unit_status)
        if not math.isfinite(scan_size) or scan_size <= 0:
            qc_flags.append("invalid_scan_size")
        elif (
            math.isfinite(filename_scan_size)
            and abs(filename_scan_x - scan_x) > 0.05
            and abs(filename_scan_y - scan_y) > 0.05
        ):
            qc_flags.append(
                f"scan_size_metadata_filename_disagreement:{scan_x:g}x{scan_y:g}_metadata_vs_{filename_scan_x:g}x{filename_scan_y:g}_filename"
            )
        if not math.isfinite(rq):
            qc_flags.append("missing_rq")
        candidates.append(
            AFMCandidate(
                sample_id=sample_id,
                sample_group_id=sample_id,
                growth_run_id=sample_id,
                material=infer_material(afm_path, metadata),
                afm_scan_id=str(metadata.get("afm_file_id") or scan_id_from_height_path(afm_path)),
                afm_path=afm_path,
                selected_height_map_path=afm_path,
                channel=channel,
                height_unit_exported=unit,
                scan_size_um=scan_size,
                scan_size_x_um=scan_x,
                scan_size_y_um=scan_y,
                resolution_x=resolution_x,
                resolution_y=resolution_y,
                rq_nm=rq,
                rq_source=rq_source,
                rq_recomputed_nm=recomputed_rq,
                ra_nm=ra,
                robust_height_range_nm=robust_range,
                peak_to_valley_nm=peak_to_valley,
                qc_flags=";".join(qc_flags),
            )
        )
    return candidates


def valid_physical_afm(candidate: AFMCandidate) -> bool:
    if not candidate.selected_height_map_path.is_file():
        return False
    if "unknown_height_unit" in candidate.qc_flags:
        return False
    if "invalid_scan_size" in candidate.qc_flags:
        return False
    return math.isfinite(candidate.rq_nm)


def select_representative_afm_scan(
    candidates: Sequence[AFMCandidate],
    *,
    primary_scan_size_um: float,
    tolerance_um: float,
) -> tuple[AFMCandidate | None, float, float, str]:
    """Select a representative scan without using RHEED appearance."""
    valid = [candidate for candidate in candidates if valid_physical_afm(candidate)]
    if not valid:
        return None, math.nan, math.nan, "missing_physical_afm_height_map"
    primary = [c for c in valid if abs(c.scan_size_um - primary_scan_size_um) <= tolerance_um]
    if primary:
        subset = primary
        reason = "closest_to_primary_1um_subset_median_rq"
    else:
        sizes = [round(c.scan_size_um, 6) for c in valid if math.isfinite(c.scan_size_um)]
        if not sizes:
            return None, math.nan, math.nan, "invalid_scan_size"
        counts = Counter(sizes)
        dominant_size = sorted(counts, key=lambda s: (-counts[s], s))[0]
        subset = [c for c in valid if round(c.scan_size_um, 6) == dominant_size]
        reason = f"no_primary_1um_scan_closest_to_dominant_size_{dominant_size:g}_median_rq"
    median_rq = float(np.median([c.rq_nm for c in subset]))
    selected = sorted(
        subset,
        key=lambda c: (abs(c.rq_nm - median_rq), c.afm_scan_id.lower(), display_path(c.afm_path, REPO_ROOT)),
    )[0]
    return selected, median_rq, abs(selected.rq_nm - median_rq), reason


def match_manual_folder_to_sample(
    selection: ManualSelection,
    afm_by_sample: dict[str, list[AFMCandidate]],
) -> tuple[str, str, list[str]]:
    """Match manual folder to AFM candidates using exact normalized sample ID."""
    sid = selection.sample_id
    if not sid:
        return "ambiguous_sample_match", "no_normalized_manual_sample_id", []
    possible = sorted(afm_by_sample.get(sid, []), key=lambda c: c.afm_scan_id)
    if not possible:
        return "missing_afm_pair", "exact_normalized_sample_id", []
    return "matched", "exact_normalized_sample_id", [sid]


def robust_display_limits(path: Path, unit: str) -> tuple[float, float]:
    """Return native robust 1st/99th percentile display limits."""
    height, _ = load_height_nm(path, unit)
    values = height[np.isfinite(height)]
    if values.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(values, [1, 99])
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def common_scale_for_pairs(pairs: Sequence[SelectedPair], primary_size: float, tolerance: float) -> tuple[float, float] | None:
    """Return pooled 1st/99th percentile scale for the homogeneous 1 um subset."""
    chunks: list[np.ndarray] = []
    for pair in pairs:
        if abs(pair.afm.scan_size_um - primary_size) > tolerance:
            continue
        height, _ = load_height_nm(pair.afm.selected_height_map_path, pair.afm.height_unit_exported)
        values = height[np.isfinite(height)]
        if values.size:
            chunks.append(values.ravel())
    if not chunks:
        return None
    pooled = np.concatenate(chunks)
    lo, hi = np.percentile(pooled, [1, 99])
    if hi <= lo:
        return float(np.nanmin(pooled)), float(np.nanmax(pooled) + 1.0)
    return float(lo), float(hi)


def build_selected_pairs(
    selections: Sequence[ManualSelection],
    afm_candidates: Sequence[AFMCandidate],
    *,
    primary_scan_size_um: float,
    tolerance_um: float,
) -> tuple[list[SelectedPair], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build selected pairs and audit rows."""
    afm_by_sample: dict[str, list[AFMCandidate]] = defaultdict(list)
    for candidate in afm_candidates:
        afm_by_sample[candidate.sample_id].append(candidate)
    pairs: list[SelectedPair] = []
    manual_rows: list[dict[str, Any]] = []
    afm_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for selection in selections:
        warnings = list(selection.warnings)
        match_status, match_method, possible_matches = match_manual_folder_to_sample(selection, afm_by_sample)
        selected_pair: SelectedPair | None = None
        selected_afm: AFMCandidate | None = None
        skip_reason = ""
        if selection.status != "ok":
            match_status = selection.status
            skip_reason = selection.status
        elif match_status != "matched":
            skip_reason = match_status
        else:
            sample_candidates = afm_by_sample.get(selection.sample_id, [])
            selected_afm, median_rq, distance, reason = select_representative_afm_scan(
                sample_candidates,
                primary_scan_size_um=primary_scan_size_um,
                tolerance_um=tolerance_um,
            )
            if selected_afm is None:
                skip_reason = reason
                match_status = reason
            else:
                native_min, native_max = robust_display_limits(selected_afm.selected_height_map_path, selected_afm.height_unit_exported)
                selected_pair = SelectedPair(
                    sample_id=selection.sample_id,
                    sample_group_id=selection.sample_id,
                    growth_run_id=selection.sample_id,
                    material=selected_afm.material,
                    manual_folder=selection.manual_folder,
                    manual_rheed_path=selection.selected_path or Path(),
                    manual_rheed_filename=(selection.selected_path.name if selection.selected_path else ""),
                    all_manual_candidates=selection.candidates,
                    manual_warnings=selection.warnings,
                    afm=selected_afm,
                    number_of_candidate_scans=len([c for c in sample_candidates if valid_physical_afm(c)]),
                    sample_median_rq_nm=median_rq,
                    distance_from_median_rq_nm=distance,
                    selection_reason=reason,
                    native_display_min_nm=native_min,
                    native_display_max_nm=native_max,
                )
                pairs.append(selected_pair)
        manual_rows.append(
            {
                "manual_folder": rel(selection.manual_folder, REPO_ROOT),
                "normalized_manual_folder": selection.normalized_manual_folder,
                "manual_rheed_path": rel(selection.selected_path, REPO_ROOT),
                "manual_rheed_filename": selection.selected_path.name if selection.selected_path else "",
                "manual_candidate_count": len(selection.candidates),
                "all_manual_candidates": [rel(path, REPO_ROOT) for path in selection.candidates],
                "sample_id": selection.sample_id,
                "sample_group_id": selection.sample_id,
                "growth_run_id": selection.sample_id,
                "match_status": match_status,
                "match_method": match_method,
                "possible_matches": possible_matches,
                "warnings": warnings,
                "included_in_figure": int(selected_pair is not None),
            }
        )
        if selected_pair is not None:
            afm = selected_pair.afm
            afm_rows.append(
                {
                    "sample_id": selected_pair.sample_id,
                    "sample_group_id": selected_pair.sample_group_id,
                    "growth_run_id": selected_pair.growth_run_id,
                    "selected_afm_scan_id": afm.afm_scan_id,
                    "selected_afm_path": rel(afm.afm_path, REPO_ROOT),
                    "selected_height_map_path": rel(afm.selected_height_map_path, REPO_ROOT),
                    "channel": afm.channel,
                    "scan_size_um": afm.scan_size_um,
                    "scan_size_x_um": afm.scan_size_x_um,
                    "scan_size_y_um": afm.scan_size_y_um,
                    "resolution_x": afm.resolution_x,
                    "resolution_y": afm.resolution_y,
                    "rq_nm": afm.rq_nm,
                    "rq_source": afm.rq_source,
                    "ra_nm": afm.ra_nm,
                    "robust_height_range_nm": afm.robust_height_range_nm,
                    "peak_to_valley_nm": afm.peak_to_valley_nm,
                    "native_display_min_nm": selected_pair.native_display_min_nm,
                    "native_display_max_nm": selected_pair.native_display_max_nm,
                    "common_display_min_nm": "",
                    "common_display_max_nm": "",
                    "number_of_candidate_scans": selected_pair.number_of_candidate_scans,
                    "sample_median_rq_nm": selected_pair.sample_median_rq_nm,
                    "distance_from_median_rq_nm": selected_pair.distance_from_median_rq_nm,
                    "selection_reason": selected_pair.selection_reason,
                    "qc_flags": afm.qc_flags,
                }
            )
        manifest_rows.append(
            {
                "sample_id": selection.sample_id,
                "sample_group_id": selection.sample_id,
                "growth_run_id": selection.sample_id,
                "manual_rheed_folder": rel(selection.manual_folder, REPO_ROOT),
                "manual_rheed_path": rel(selection.selected_path, REPO_ROOT),
                "manual_rheed_filename": selection.selected_path.name if selection.selected_path else "",
                "manual_selection_status": selection.status,
                "selected_afm_scan_id": selected_afm.afm_scan_id if selected_afm else "",
                "selected_afm_path": rel(selected_afm.afm_path, REPO_ROOT) if selected_afm else "",
                "selected_height_map_path": rel(selected_afm.selected_height_map_path, REPO_ROOT) if selected_afm else "",
                "rq_nm": selected_afm.rq_nm if selected_afm else "",
                "rq_source": selected_afm.rq_source if selected_afm else "",
                "ra_nm": selected_afm.ra_nm if selected_afm else "",
                "scan_size_um": selected_afm.scan_size_um if selected_afm else "",
                "afm_resolution": f"{selected_afm.resolution_x}x{selected_afm.resolution_y}" if selected_afm else "",
                "included_in_native_figure": int(selected_pair is not None),
                "included_in_common_scale_figure": 0,
                "skip_reason": skip_reason,
                "warnings": warnings,
            }
        )
    return pairs, manual_rows, afm_rows, manifest_rows


def scan_label(pair: SelectedPair) -> str:
    x = pair.afm.scan_size_x_um
    y = pair.afm.scan_size_y_um
    if math.isfinite(x) and math.isfinite(y):
        return f"{x:.3g} x {y:.3g} um"
    return f"{pair.afm.scan_size_um:.3g} x {pair.afm.scan_size_um:.3g} um"


def nice_scale_bar_um(scan_size_x_um: float) -> float:
    if not math.isfinite(scan_size_x_um) or scan_size_x_um <= 0:
        return math.nan
    target = scan_size_x_um / 5.0
    candidates = np.asarray([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    candidates = candidates[candidates <= scan_size_x_um * 0.6]
    if candidates.size == 0:
        return target
    return float(candidates[np.argmin(np.abs(candidates - target))])


def rheed_display_image(path: Path) -> tuple[np.ndarray, str | None]:
    """Load the manual RHEED image with minimal display transformation."""
    image = Image.open(path)
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr, "gray"
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[:, :, :3]
        diff = np.mean(np.std(rgb.astype(float), axis=2))
        if diff < 2.0:
            gray = np.asarray(Image.fromarray(rgb).convert("L"))
            return gray, "gray"
        return rgb, None
    return arr, None


def render_manual_rheed_panel(ax: plt.Axes, pair: SelectedPair) -> None:
    image, cmap = rheed_display_image(pair.manual_rheed_path)
    ax.imshow(image, cmap=cmap, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Sample {pair.sample_id}\nRHEED\nManual selection\n{pair.manual_rheed_filename}",
        fontsize=7,
    )


def render_afm_height_panel(
    ax: plt.Axes,
    pair: SelectedPair,
    *,
    vmin: float,
    vmax: float,
    title: str = "AFM",
) -> Any:
    height, _ = load_height_nm(pair.afm.selected_height_map_path, pair.afm.height_unit_exported)
    sx = pair.afm.scan_size_x_um
    sy = pair.afm.scan_size_y_um
    if math.isfinite(sx) and math.isfinite(sy) and sx > 0 and sy > 0:
        extent = [0, sx, sy, 0]
    else:
        extent = None
    im = ax.imshow(height, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest", extent=extent, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if extent is not None:
        bar = nice_scale_bar_um(sx)
        if math.isfinite(bar):
            x0 = sx * 0.08
            x1 = min(sx * 0.92, x0 + bar)
            y = sy * 0.90
            ax.plot([x0, x1], [y, y], color="white", lw=3, solid_capstyle="butt")
            ax.plot([x0, x1], [y, y], color="black", lw=1, solid_capstyle="butt")
            ax.text(x0, y - sy * 0.05, f"{bar:g} um", color="white", fontsize=6, va="bottom")
    ax.set_title(
        f"{title}\nRq = {pair.afm.rq_nm:.3f} nm\nScan: {scan_label(pair)}",
        fontsize=7,
    )
    return im


def pair_sort_value(pair: SelectedPair, sort_key: str) -> tuple[Any, str]:
    if sort_key == "sample_id":
        return pair.sample_id, pair.sample_id
    if sort_key == "material":
        return pair.material, pair.sample_id
    if sort_key == "scan_size":
        return pair.afm.scan_size_um, pair.sample_id
    return pair.afm.rq_nm, pair.sample_id


def render_pair_grid(
    pairs: Sequence[SelectedPair],
    paths: ManualVizPaths,
    *,
    output_stem: str,
    title: str,
    sort_key: str,
    common_scale: tuple[float, float] | None = None,
    cards_per_row: int = 4,
) -> None:
    """Render a publication-scale grid with manual RHEED and AFM panels."""
    rows = sorted(pairs, key=lambda pair: pair_sort_value(pair, sort_key))
    if not rows:
        return
    nrows = math.ceil(len(rows) / cards_per_row)
    fig = plt.figure(figsize=(cards_per_row * 4.8, nrows * 3.25), dpi=300)
    width_pattern: list[float] = []
    for _ in range(cards_per_row):
        width_pattern.extend([1.05, 1.0, 0.06])
    grid = fig.add_gridspec(nrows=nrows, ncols=cards_per_row * 3, width_ratios=width_pattern, hspace=0.58, wspace=0.12)
    fig.suptitle(title, fontsize=15, y=0.997)
    for idx, pair in enumerate(rows):
        rr = idx // cards_per_row
        cc = (idx % cards_per_row) * 3
        ax_rheed = fig.add_subplot(grid[rr, cc])
        ax_afm = fig.add_subplot(grid[rr, cc + 1])
        cax = fig.add_subplot(grid[rr, cc + 2])
        render_manual_rheed_panel(ax_rheed, pair)
        if common_scale is None:
            vmin, vmax = pair.native_display_min_nm, pair.native_display_max_nm
            title_text = "AFM"
        else:
            vmin, vmax = common_scale
            title_text = "AFM common scale"
        im = render_afm_height_panel(ax_afm, pair, vmin=vmin, vmax=vmax, title=title_text)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=5)
        cbar.set_label("Height (nm)", fontsize=6)
    for idx in range(len(rows), nrows * cards_per_row):
        rr = idx // cards_per_row
        cc = (idx % cards_per_row) * 3
        for sub in range(3):
            fig.add_subplot(grid[rr, cc + sub]).axis("off")
    png = paths.report_dir / f"{output_stem}.png"
    pdf = paths.report_dir / f"{output_stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def render_paginated_pairs(
    pairs: Sequence[SelectedPair],
    paths: ManualVizPaths,
    *,
    title: str,
    per_page: int = 16,
    cards_per_row: int = 4,
) -> None:
    rows = sorted(pairs, key=lambda pair: (pair.afm.rq_nm, pair.sample_id))
    if len(rows) <= per_page:
        return
    for page_idx in range(0, len(rows), per_page):
        page = rows[page_idx : page_idx + per_page]
        stem = f"pages/manual_pairs_page_{page_idx // per_page + 1:02d}"
        render_pair_grid(
            page,
            paths,
            output_stem=stem,
            title=f"{title} - page {page_idx // per_page + 1}",
            sort_key="rq_nm",
            cards_per_row=cards_per_row,
        )


def cache_manual_rheed(pair: SelectedPair, paths: ManualVizPaths) -> Path:
    out_dir = paths.gallery_assets_dir / "rheed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{pair.sample_id}_{pair.manual_rheed_path.name}"
    if not out.is_file():
        shutil.copy2(pair.manual_rheed_path, out)
    return out


def render_cached_afm(pair: SelectedPair, paths: ManualVizPaths, *, scale: str, common_scale: tuple[float, float] | None = None) -> Path:
    out_dir = paths.gallery_assets_dir / "afm"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{pair.sample_id}_{pair.afm.afm_scan_id}_{scale}.png"
    if out.is_file():
        return out
    if common_scale is None:
        vmin, vmax = pair.native_display_min_nm, pair.native_display_max_nm
    else:
        vmin, vmax = common_scale
    fig, ax = plt.subplots(figsize=(3.1, 3.0), dpi=180)
    im = render_afm_height_panel(ax, pair, vmin=vmin, vmax=vmax, title="AFM")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Height (nm)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def with_cached_assets(pairs: Sequence[SelectedPair], paths: ManualVizPaths, common_scale: tuple[float, float] | None) -> list[SelectedPair]:
    """Return copies of selected pairs with cached gallery asset paths."""
    out: list[SelectedPair] = []
    for pair in pairs:
        cached_rheed = cache_manual_rheed(pair, paths)
        cached_afm = render_cached_afm(pair, paths, scale="native")
        cached_common = (
            render_cached_afm(pair, paths, scale="common", common_scale=common_scale)
            if common_scale is not None and is_common_scale_pair(pair)
            else None
        )
        out.append(
            SelectedPair(
                **{
                    **pair.__dict__,
                    "cached_manual_rheed_path": cached_rheed,
                    "cached_afm_native_path": cached_afm,
                    "cached_afm_common_path": cached_common,
                }
            )
        )
    return out


def is_common_scale_pair(pair: SelectedPair, primary_size: float = 1.0, tolerance: float = 0.10) -> bool:
    return abs(pair.afm.scan_size_um - primary_size) <= tolerance


def relative_to_report(path: Path | None, report_dir: Path) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(report_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def generate_html_gallery(pairs: Sequence[SelectedPair], paths: ManualVizPaths) -> None:
    """Write a local, static HTML gallery."""
    cards: list[str] = []
    for pair in pairs:
        rheed_src = html.escape(relative_to_report(pair.cached_manual_rheed_path, paths.report_dir))
        afm_src = html.escape(relative_to_report(pair.cached_afm_native_path, paths.report_dir))
        cards.append(
            f"""
<article class="card" data-sample="{html.escape(pair.sample_id)}" data-rq="{pair.afm.rq_nm:.12g}" data-scan="{pair.afm.scan_size_um:.12g}" data-material="{html.escape(pair.material or 'unknown')}">
  <h2>Sample {html.escape(pair.sample_id)}</h2>
  <div class="images">
    <figure><img src="{rheed_src}" alt="Manual RHEED screenshot for sample {html.escape(pair.sample_id)}"><figcaption>RHEED manual selection</figcaption></figure>
    <figure><img src="{afm_src}" alt="AFM height map for sample {html.escape(pair.sample_id)}"><figcaption>AFM physical height</figcaption></figure>
  </div>
  <dl>
    <dt>RHEED file</dt><dd>{html.escape(pair.manual_rheed_filename)}</dd>
    <dt>Rq</dt><dd>{pair.afm.rq_nm:.3f} nm</dd>
    <dt>Ra</dt><dd>{pair.afm.ra_nm:.3f} nm</dd>
    <dt>Scan size</dt><dd>{html.escape(scan_label(pair))}</dd>
    <dt>AFM scan ID</dt><dd>{html.escape(pair.afm.afm_scan_id)}</dd>
    <dt>Material</dt><dd>{html.escape(pair.material or 'unknown')}</dd>
    <dt>AFM QC flags</dt><dd>{html.escape(pair.afm.qc_flags or 'none')}</dd>
  </dl>
  <details>
    <summary>Source paths</summary>
    <code>{html.escape(rel(pair.manual_rheed_path, paths.repo_root))}</code><br>
    <code>{html.escape(rel(pair.afm.selected_height_map_path, paths.repo_root))}</code>
  </details>
</article>
"""
        )
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Manual RHEED-AFM Pair Gallery</title>
<style>
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f8; color: #1d1f21; }}
header {{ position: sticky; top: 0; z-index: 2; background: #fff; border-bottom: 1px solid #d9dde2; padding: 14px 18px; display: flex; gap: 16px; align-items: center; }}
h1 {{ font-size: 18px; margin: 0; }}
select {{ font-size: 14px; padding: 5px 8px; }}
main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 14px; padding: 14px; }}
.card {{ background: #fff; border: 1px solid #d9dde2; border-radius: 8px; padding: 12px; }}
.card h2 {{ margin: 0 0 8px; font-size: 16px; }}
.images {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
figure {{ margin: 0; }}
img {{ width: 100%; height: 220px; object-fit: contain; background: #111; }}
figcaption {{ font-size: 12px; color: #555; margin-top: 4px; }}
dl {{ display: grid; grid-template-columns: 110px 1fr; gap: 3px 8px; font-size: 13px; }}
dt {{ font-weight: 600; }}
dd {{ margin: 0; overflow-wrap: anywhere; }}
details {{ font-size: 12px; color: #444; }}
code {{ overflow-wrap: anywhere; }}
</style>
</head>
<body>
<header>
  <h1>Manual RHEED-AFM Pair Gallery</h1>
  <label>Sort <select id="sorter">
    <option value="rq">Rq</option>
    <option value="sample">Sample ID</option>
    <option value="scan">Scan size</option>
    <option value="material">Material</option>
  </select></label>
</header>
<main id="cards">
{''.join(cards)}
</main>
<script>
const cards = document.getElementById('cards');
document.getElementById('sorter').addEventListener('change', event => {{
  const mode = event.target.value;
  const items = Array.from(cards.children);
  items.sort((a, b) => {{
    if (mode === 'sample') return a.dataset.sample.localeCompare(b.dataset.sample, undefined, {{numeric: true}});
    if (mode === 'material') return a.dataset.material.localeCompare(b.dataset.material) || a.dataset.sample.localeCompare(b.dataset.sample, undefined, {{numeric: true}});
    if (mode === 'scan') return Number(a.dataset.scan) - Number(b.dataset.scan) || a.dataset.sample.localeCompare(b.dataset.sample, undefined, {{numeric: true}});
    return Number(a.dataset.rq) - Number(b.dataset.rq) || a.dataset.sample.localeCompare(b.dataset.sample, undefined, {{numeric: true}});
  }});
  items.forEach(item => cards.appendChild(item));
}});
</script>
</body>
</html>
"""
    (paths.report_dir / "index.html").write_text(doc, encoding="utf-8")


def write_skipped_samples(manifest_rows: Sequence[dict[str, Any]], paths: ManualVizPaths) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        reason = str(row.get("skip_reason") or "")
        if reason:
            grouped[reason].append(row)
    lines = ["# Skipped Manual RHEED-AFM Samples", ""]
    if not grouped:
        lines.append("No samples were skipped.")
    for reason in sorted(grouped):
        lines.extend([f"## {reason}", ""])
        for row in grouped[reason]:
            lines.append(
                f"- sample {row.get('sample_id', '')}: {row.get('manual_rheed_folder', '')} "
                f"{row.get('manual_rheed_filename', '')}".strip()
            )
        lines.append("")
    (paths.report_dir / "skipped_samples.md").write_text("\n".join(lines), encoding="utf-8")


def write_readme(
    paths: ManualVizPaths,
    *,
    selections: Sequence[ManualSelection],
    pairs: Sequence[SelectedPair],
    manifest_rows: Sequence[dict[str, Any]],
    common_scale: tuple[float, float] | None,
    command: str,
) -> None:
    skipped = [row for row in manifest_rows if row.get("skip_reason")]
    reasons = Counter(str(row.get("skip_reason")) for row in skipped)
    lines = [
        "# Manual RHEED-AFM Pair Visualization",
        "",
        "## Folder structure",
        "",
        "The manual selection root was inspected recursively. In this checkout, sample folders are direct children of `data/manual_selection` and contain `RHEED` and `AFM` subfolders.",
        "",
        "## Manual RHEED rule",
        "",
        "Only image files with a basename starting with `select` are used. Supported extensions are `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, and `.webp`. Videos, hidden files, metadata files, and temporary files are ignored.",
        "",
        "Multiple selections are resolved deterministically by exact stem `select`, then `select_final*`, then `select_best*`, then lexicographic filename order. The warning is recorded as `multiple_manual_selections`.",
        "",
        "## Sample matching",
        "",
        "Manual folders are matched by exact normalized four-digit sample ID to AFM sample IDs from plane-corrected AFM metadata. Broad fuzzy matching is not used.",
        "",
        "## AFM representative scan",
        "",
        "Valid physical plane-corrected height maps are selected without using RHEED appearance. The rule prefers 1.0 um scans and chooses the scan whose Rq is closest to the sample median Rq within that subset. If no 1.0 um scan exists, the dominant valid scan size is used and the median-closeness rule is applied within that size.",
        "",
        "## Rq definition",
        "",
        "The displayed roughness is RMS roughness, `Rq = sqrt(mean((z - mean(z))^2))`, in nanometers. Descriptor-table Rq is used when available; otherwise Rq is recomputed from the physical height map.",
        "",
        "## Rendering",
        "",
        "RHEED panels show the manual screenshot with minimal display transformation and preserved aspect ratio. AFM panels are rendered from physical height arrays in nanometers with viridis, equal spatial aspect, a physical color bar, and a lateral scale bar.",
        "",
        "Native AFM scale uses per-scan 1st to 99th percentile height limits. Common scale uses pooled 1st to 99th percentile limits over selected 1.0 um scans.",
        "",
        "## Counts",
        "",
        f"- sample folders inspected: {len(selections)}",
        f"- included samples: {len(pairs)}",
        f"- skipped samples: {len(skipped)}",
    ]
    for reason, count in sorted(reasons.items()):
        lines.append(f"- skipped `{reason}`: {count}")
    if common_scale is not None:
        lines.append(f"- common AFM scale: {common_scale[0]:.4g} to {common_scale[1]:.4g} nm")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `reports/rheed_roughness/manual_pair_visualization/manual_rheed_afm_pairs_by_roughness_native.png`",
            "- `reports/rheed_roughness/manual_pair_visualization/manual_rheed_afm_pairs_by_roughness_native.pdf`",
            "- `reports/rheed_roughness/manual_pair_visualization/manual_rheed_afm_pairs_by_roughness_common_scale.png`",
            "- `reports/rheed_roughness/manual_pair_visualization/manual_rheed_afm_pairs_by_roughness_common_scale.pdf`",
            "- `reports/rheed_roughness/manual_pair_visualization/manual_rheed_afm_pairs_by_sample_id.png`",
            "- `reports/rheed_roughness/manual_pair_visualization/manual_rheed_afm_pairs_by_sample_id.pdf`",
            "- `reports/rheed_roughness/manual_pair_visualization/index.html`",
            "- `reports/rheed_roughness/manual_pair_visualization/skipped_samples.md`",
            "- `outputs/rheed_roughness/manual_pair_visualization/manual_selection_audit.csv`",
            "- `outputs/rheed_roughness/manual_pair_visualization/afm_selection_audit.csv`",
            "- `outputs/rheed_roughness/manual_pair_visualization/manual_pair_figure_manifest.csv`",
            "",
            "## Reproduction",
            "",
            f"```bash\n{command}\n```",
            "",
            "The optional morphology-index sorted figure was not generated unless a reliable existing morphology table was available.",
        ]
    )
    (paths.report_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def update_common_inclusion(
    manifest_rows: list[dict[str, Any]],
    afm_rows: list[dict[str, Any]],
    pairs: Sequence[SelectedPair],
    common_scale: tuple[float, float] | None,
    *,
    primary_size: float,
    tolerance: float,
) -> None:
    common_ids = {pair.sample_id for pair in pairs if common_scale is not None and abs(pair.afm.scan_size_um - primary_size) <= tolerance}
    for row in manifest_rows:
        row["included_in_common_scale_figure"] = int(str(row.get("sample_id")) in common_ids)
    for row in afm_rows:
        if str(row.get("sample_id")) in common_ids and common_scale is not None:
            row["common_display_min_nm"] = common_scale[0]
            row["common_display_max_nm"] = common_scale[1]


def write_audits(
    paths: ManualVizPaths,
    manual_rows: Sequence[dict[str, Any]],
    afm_rows: Sequence[dict[str, Any]],
    manifest_rows: Sequence[dict[str, Any]],
) -> None:
    write_csv_rows(paths.output_dir / "manual_selection_audit.csv", manual_rows, MANUAL_AUDIT_FIELDS)
    write_csv_rows(paths.output_dir / "afm_selection_audit.csv", afm_rows, AFM_AUDIT_FIELDS)
    write_csv_rows(paths.output_dir / "manual_pair_figure_manifest.csv", manifest_rows, MANIFEST_FIELDS)


def print_preimplementation_summary(selections: Sequence[ManualSelection], afm_candidates: Sequence[AFMCandidate], paths: ManualVizPaths) -> None:
    valid = [s for s in selections if s.selected_path is not None]
    print("Pre-implementation inspection summary:")
    print(f"1. Manual selection root: {display_path(paths.manual_root, paths.repo_root)}")
    print("   Structure: sample folders with RHEED/AFM children were detected.")
    print(f"2. Valid select* images found: {len(valid)}")
    for selection in valid:
        print(f"   - {selection.sample_id}: {display_path(selection.selected_path, paths.repo_root)}")
    print("3. Sample-ID matching strategy: exact four-digit normalized folder sample ID to AFM metadata sample_id; no broad fuzzy matching.")
    print("4. AFM columns available: sample_id, afm_path, afm_scan_id, channel, height_unit_exported, scan_size_um, resolution, Rq, Ra, robust height range, peak-to-valley.")
    print("5. AFM representative rule: prefer valid 1.0 um scans, then Rq closest to subset median; otherwise dominant valid scan size with the same median-closeness rule.")
    print("6. Planned layout: four sample cards per row, raw manual RHEED image plus physical AFM height map, per-AFM color bar, native and common height-scale figures.")
    print(f"AFM physical height candidates loaded: {len(afm_candidates)}")


def run(config_path: Path, manual_selection_root: Path, *, cards_per_row: int = 4) -> dict[str, Any]:
    config = read_config(config_path)
    paths = build_paths(config, manual_selection_root)
    selections = discover_manual_rheed_images(paths.manual_root)
    afm_candidates = load_afm_candidates(paths)
    print_preimplementation_summary(selections, afm_candidates, paths)
    primary_size = float(config.get("afm", {}).get("primary_scan_size_um", 1.0))
    tolerance = float(config.get("afm", {}).get("primary_scan_size_tolerance_um", 0.10))
    pairs, manual_rows, afm_rows, manifest_rows = build_selected_pairs(
        selections,
        afm_candidates,
        primary_scan_size_um=primary_size,
        tolerance_um=tolerance,
    )
    common_scale = common_scale_for_pairs(pairs, primary_size, tolerance)
    update_common_inclusion(manifest_rows, afm_rows, pairs, common_scale, primary_size=primary_size, tolerance=tolerance)
    cached_pairs = with_cached_assets(pairs, paths, common_scale)
    write_audits(paths, manual_rows, afm_rows, manifest_rows)
    title = "Manually Selected RHEED Frames and Corresponding AFM Topography\nSorted by AFM RMS Roughness"
    render_pair_grid(
        cached_pairs,
        paths,
        output_stem="manual_rheed_afm_pairs_by_roughness_native",
        title=title,
        sort_key="rq_nm",
        cards_per_row=cards_per_row,
    )
    if common_scale is not None:
        common_pairs = [pair for pair in cached_pairs if abs(pair.afm.scan_size_um - primary_size) <= tolerance]
        render_pair_grid(
            common_pairs,
            paths,
            output_stem="manual_rheed_afm_pairs_by_roughness_common_scale",
            title=title + "\nCommon AFM height scale for 1.0 um subset",
            sort_key="rq_nm",
            common_scale=common_scale,
            cards_per_row=cards_per_row,
        )
    render_pair_grid(
        cached_pairs,
        paths,
        output_stem="manual_rheed_afm_pairs_by_sample_id",
        title="Manually Selected RHEED Frames and Corresponding AFM Topography\nSorted by Sample ID",
        sort_key="sample_id",
        cards_per_row=cards_per_row,
    )
    render_paginated_pairs(cached_pairs, paths, title=title, cards_per_row=cards_per_row)
    generate_html_gallery(cached_pairs, paths)
    write_skipped_samples(manifest_rows, paths)
    command = f"PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_roughness.visualize_manual_pairs --config {config_path} --manual-selection-root {manual_selection_root}"
    write_readme(paths, selections=selections, pairs=cached_pairs, manifest_rows=manifest_rows, common_scale=common_scale, command=command)
    folders_with_select = sum(1 for selection in selections if selection.selected_path is not None)
    missing_select = sum(1 for selection in selections if selection.status == "missing_manual_selection")
    ambiguous = sum(1 for row in manual_rows if row.get("match_status") == "ambiguous_sample_match")
    multiple = sum(1 for selection in selections if "multiple_manual_selections" in selection.warnings)
    matched = sum(1 for row in manual_rows if row.get("match_status") == "matched")
    common_count = sum(1 for row in manifest_rows if int(row.get("included_in_common_scale_figure", 0)))
    unavailable_afm = sum(1 for row in manifest_rows if str(row.get("skip_reason")) in {"missing_afm_pair", "missing_physical_afm_height_map", "invalid_height_unit", "invalid_scan_size"})
    summary = {
        "total_sample_folders_inspected": len(selections),
        "folders_containing_select_image": folders_with_select,
        "folders_skipped_missing_select_image": missing_select,
        "successfully_matched_paired_samples": matched,
        "ambiguous_matches": ambiguous,
        "samples_with_multiple_manual_screenshots": multiple,
        "samples_with_physical_afm_height_maps": len(cached_pairs),
        "samples_included_native_scale_figure": len(cached_pairs),
        "samples_included_common_scale_1um_figure": common_count,
        "samples_skipped_valid_afm_unavailable": unavailable_afm,
    }
    print("Manual pair visualization summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    print(f"Reports written to: {display_path(paths.report_dir, paths.repo_root)}")
    print(f"Audits written to: {display_path(paths.output_dir, paths.repo_root)}")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rheed_roughness.yaml"))
    parser.add_argument("--manual-selection-root", type=Path, default=Path("data/manual_selection"))
    parser.add_argument("--cards-per-row", type=int, default=4)
    args = parser.parse_args(argv)
    run(args.config, args.manual_selection_root, cards_per_row=args.cards_per_row)


if __name__ == "__main__":
    main()
