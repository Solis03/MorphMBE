#!/usr/bin/env python3
"""Scan processed AFM folders into a complete candidate table."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "data" / "processed_afm"
DEFAULT_PLANE_CORRECTED_ROOT = REPO_ROOT / "data" / "plane_corrected_afm"
DEFAULT_PAIR_ROOT = REPO_ROOT / "data" / "pair"
ALLOWED_SUFFIXES = {".npy", ".png", ".tif", ".tiff", ".csv", ".txt"}
VIDEO_SUFFIXES = {".mov", ".mp4", ".avi", ".mkv", ".m4v", ".mts", ".m2ts"}
MATERIAL_RE = re.compile(r"\b([A-Z][a-z]Sb)\b")
N_TAG_RE = re.compile(r"n\d{4}", re.IGNORECASE)


@dataclass(frozen=True)
class AFMCandidateRow:
    sample_id: str
    group_id: str
    material: str
    afm_path: Path
    metadata_path: Path | None
    original_source_path: str
    rheed_path: Path | None
    scan_size_um: float | None
    scan_size_source: str
    resolution_h: int | None
    resolution_w: int | None
    channel_name: str
    is_plane_corrected: bool
    is_rendered_image: bool
    is_physical_height_map: bool


def infer_material(sample_id: str, path: Path, metadata: dict[str, Any]) -> str:
    for key in ("raw_file", "raw_afm_file"):
        value = metadata.get(key)
        if isinstance(value, str):
            match = MATERIAL_RE.search(Path(value).stem)
            if match is not None:
                return match.group(1)
    match = MATERIAL_RE.search(path.stem)
    if match is not None:
        return match.group(1)
    stripped = N_TAG_RE.sub("", path.stem).strip(" _-")
    if stripped:
        token = stripped.split("_")[0].split()[0]
        if any(char.isalpha() for char in token):
            return token
    return "unknown"


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def resolve_existing_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.exists():
        return expanded.resolve()
    repo_relative = REPO_ROOT / expanded
    if repo_relative.exists():
        return repo_relative.resolve()
    return expanded.resolve()


def require_video_backend() -> Any:
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing video dependency `imageio-ffmpeg`.") from exc
    return imageio_ffmpeg


def visible_video_files(sample_root: Path) -> list[Path]:
    rheed_root = sample_root / "RHEED"
    if not rheed_root.is_dir():
        return []
    return [
        path
        for path in sorted(rheed_root.iterdir())
        if path.is_file() and not path.name.startswith("._") and path.suffix.lower() in VIDEO_SUFFIXES
    ]


def choose_canonical_rheed_path(sample_root: Path) -> tuple[Path | None, str]:
    imageio_ffmpeg = require_video_backend()
    candidates = []
    for path in visible_video_files(sample_root):
        try:
            frame_count, duration_seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
        except Exception:
            continue
        if frame_count <= 0 or not np.isfinite(duration_seconds) or duration_seconds <= 0:
            continue
        candidates.append(
            (
                1 if "main" in path.stem.lower() else 0,
                float(duration_seconds),
                int(frame_count),
                path.name.lower(),
                path.resolve(),
            )
        )
    if not candidates:
        return None, "no decodable video"
    _, duration_seconds, _, _, selected = sorted(candidates, reverse=True)[0]
    if "main" in selected.stem.lower():
        return selected, "contains_main"
    return selected, f"longest_decodable:{duration_seconds:.3f}s"


def parse_afm_scan_size_um(path_text: str) -> float | None:
    lowered = Path(path_text).stem.lower()
    lowered = re.sub(r"(?<=\d)p(?=\d)", ".", lowered)
    lowered = lowered.replace("-", " ").replace("_", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    patterns = [
        re.compile(r"(?P<a>\d+(?:\.\d+)?)\s*(?P<ua>um|nm)\s*x\s*(?P<b>\d+(?:\.\d+)?)\s*(?P<ub>um|nm)?"),
        re.compile(r"(?P<a>\d+(?:\.\d+)?)\s*x\s*(?P<b>\d+(?:\.\d+)?)\s*(?P<u>um|nm)"),
        re.compile(r"(?P<a>\d+(?:\.\d+)?)\s*x\s*(?P<b>\d+(?:\.\d+)?)"),
        re.compile(r"(?P<a>\d+(?:\.\d+)?)\s*(?P<u>um|nm)"),
    ]
    for pattern in patterns:
        match = pattern.search(lowered)
        if match is None:
            continue
        groups = match.groupdict()
        if "b" in groups and groups.get("b") is not None:
            def convert(value: float, unit: str | None) -> float | None:
                if unit == "um":
                    return value
                if unit == "nm":
                    return value / 1000.0
                if 0.05 <= value <= 20.0:
                    return value
                if 50.0 <= value <= 20_000.0:
                    return value / 1000.0
                return None
            first = convert(float(groups["a"]), groups.get("ua") or groups.get("u"))
            second = convert(float(groups["b"]), groups.get("ub") or groups.get("u") or groups.get("ua"))
            if first is not None and second is not None:
                return float((first + second) / 2.0)
        else:
            value = float(groups["a"])
            unit = groups.get("u")
            if unit == "um":
                return value
            if unit == "nm":
                return value / 1000.0
    return None


def metadata_scan_size_um(metadata_path: Path) -> float | None:
    def normalize(value: float) -> float | None:
        if not np.isfinite(value):
            return None
        if 0.01 <= value <= 20.0:
            return float(value)
        if 50.0 <= value <= 20_000.0:
            return float(value) / 1000.0
        return float(value) if value > 0 else None

    payload = load_metadata(metadata_path)
    value = payload.get("scan_size_um")
    if isinstance(value, list | tuple) and len(value) >= 2:
        x_val = normalize(float(value[0]))
        y_val = normalize(float(value[1]))
        if x_val is not None and y_val is not None:
            return float((x_val + y_val) / 2.0)
    if isinstance(value, int | float):
        return normalize(float(value))
    channels = payload.get("channels")
    if isinstance(channels, dict):
        for channel_payload in channels.values():
            if not isinstance(channel_payload, dict):
                continue
            value = channel_payload.get("scan_size_um")
            if isinstance(value, list | tuple) and len(value) >= 2:
                x_val = normalize(float(value[0]))
                y_val = normalize(float(value[1]))
                if x_val is not None and y_val is not None:
                    return float((x_val + y_val) / 2.0)
    return None


def resolution_from_file(path: Path) -> tuple[int, int] | None:
    try:
        if path.suffix.lower() == ".npy":
            array = np.load(path, mmap_mode="r")
            if array.ndim >= 2:
                return int(array.shape[-2]), int(array.shape[-1])
            return None
        if path.suffix.lower() in {".png", ".tif", ".tiff"}:
            with Image.open(path) as image:
                return int(image.height), int(image.width)
        if path.suffix.lower() in {".csv", ".txt"}:
            delimiter = "," if path.suffix.lower() == ".csv" else None
            array = np.loadtxt(path, delimiter=delimiter)
            if array.ndim >= 2:
                return int(array.shape[-2]), int(array.shape[-1])
    except Exception:
        return None
    return None


def load_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def channel_name_for_file(path: Path, metadata: dict[str, Any]) -> str:
    primary = metadata.get("primary_channel")
    if isinstance(primary, str) and primary:
        return primary
    if "zsensor" in path.stem.lower():
        return "ZSensor"
    return "unknown"


def is_rendered_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".tif", ".tiff"} and "render" in path.stem.lower()


def is_physical_height_map(path: Path) -> bool:
    suffix = path.suffix.lower()
    stem = path.stem.lower()
    if suffix == ".npy" and "fitted_plane" not in stem:
        return True
    if suffix in {".csv", ".txt"} and any(token in stem for token in ("height", "zsensor", "topography")):
        return True
    return False


def scan_root(
    root: Path,
    pair_root: Path,
    plane_corrected: bool,
) -> list[AFMCandidateRow]:
    rows: list[AFMCandidateRow] = []
    rheed_by_sample: dict[str, Path | None] = {}
    for sample_root in sorted(path for path in root.iterdir() if path.is_dir()):
        rheed_path, _ = choose_canonical_rheed_path(resolve_existing_path(pair_root / sample_root.name))
        rheed_by_sample[sample_root.name] = rheed_path
        for scan_dir in sorted(path for path in sample_root.iterdir() if path.is_dir()):
            metadata_files = sorted(scan_dir.glob("*metadata.json"))
            metadata_path = metadata_files[0] if metadata_files else None
            metadata = load_metadata(metadata_path) if metadata_path is not None else {}
            sample_id = str(metadata.get("sample_id") or sample_root.name)
            group_id = sample_id
            material = infer_material(sample_id, scan_dir, metadata)
            original_source = str(metadata.get("raw_file") or metadata.get("raw_afm_file") or "")
            scan_size_um = None
            scan_size_source = "missing"
            if metadata_path is not None:
                scan_size_um = metadata_scan_size_um(metadata_path)
                if scan_size_um is not None:
                    scan_size_source = f"metadata:{display_path(metadata_path)}"
            for file_path in sorted(path for path in scan_dir.iterdir() if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES):
                if file_path.suffix.lower() == ".txt" and "inspection" in file_path.name.lower():
                    continue
                if file_path.suffix.lower() == ".csv" and file_path.name.lower().endswith("summary.csv"):
                    continue
                file_scan_size = scan_size_um
                file_scan_source = scan_size_source
                if file_scan_size is None:
                    filename_scan = parse_afm_scan_size_um(file_path.name)
                    if filename_scan is not None:
                        file_scan_size = filename_scan
                        file_scan_source = f"filename:{file_path.name}"
                resolution = resolution_from_file(file_path)
                rows.append(
                    AFMCandidateRow(
                        sample_id=sample_id,
                        group_id=group_id,
                        material=material,
                        afm_path=file_path.resolve(),
                        metadata_path=metadata_path.resolve() if metadata_path is not None else None,
                        original_source_path=original_source,
                        rheed_path=rheed_by_sample[sample_root.name],
                        scan_size_um=file_scan_size,
                        scan_size_source=file_scan_source,
                        resolution_h=None if resolution is None else resolution[0],
                        resolution_w=None if resolution is None else resolution[1],
                        channel_name=channel_name_for_file(file_path, metadata),
                        is_plane_corrected=plane_corrected,
                        is_rendered_image=is_rendered_image(file_path),
                        is_physical_height_map=is_physical_height_map(file_path),
                    )
                )
    return rows


def build_complete_candidate_rows(
    processed_root: Path,
    plane_corrected_root: Path,
    pair_root: Path,
) -> list[AFMCandidateRow]:
    rows = []
    if processed_root.is_dir():
        rows.extend(scan_root(processed_root, pair_root, plane_corrected=False))
    if plane_corrected_root.is_dir():
        rows.extend(scan_root(plane_corrected_root, pair_root, plane_corrected=True))
    return rows


def candidate_rows_to_dicts(rows: Sequence[AFMCandidateRow]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "sample_id": row.sample_id,
                "group_id": row.group_id,
                "material": row.material,
                "afm_path": display_path(row.afm_path),
                "original_source_path": row.original_source_path,
                "rheed_path": "" if row.rheed_path is None else display_path(row.rheed_path),
                "scan_size_um": "" if row.scan_size_um is None else f"{row.scan_size_um:.6f}",
                "scan_size_source": row.scan_size_source,
                "resolution_h": "" if row.resolution_h is None else row.resolution_h,
                "resolution_w": "" if row.resolution_w is None else row.resolution_w,
                "channel_name": row.channel_name,
                "is_plane_corrected": str(row.is_plane_corrected).lower(),
                "is_rendered_image": str(row.is_rendered_image).lower(),
                "is_physical_height_map": str(row.is_physical_height_map).lower(),
                "metadata_path": "" if row.metadata_path is None else display_path(row.metadata_path),
            }
        )
    return out


def coverage_rows(rows: Sequence[AFMCandidateRow]) -> list[dict[str, Any]]:
    size_counts = Counter()
    size_groups: dict[str, set[str]] = defaultdict(set)
    sizes_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        size_label = "nan" if row.scan_size_um is None else f"{row.scan_size_um:.3f}"
        size_counts[size_label] += 1
        size_groups[size_label].add(row.group_id)
        sizes_by_group[row.group_id].add(size_label)
    multi_size_samples = sum(len({size for size in sizes if size != 'nan'}) > 1 for sizes in sizes_by_group.values())
    only_1um = sum(sizes == {"1.000"} or sizes == {"1.016"} for sizes in sizes_by_group.values())
    with_half_or_five = sum(any(size in {"0.500", "5.000"} for size in sizes) for sizes in sizes_by_group.values())
    out = []
    for size_label in sorted(size_counts):
        out.append(
            {
                "scan_size_um": size_label,
                "afm_file_count": size_counts[size_label],
                "unique_group_count": len(size_groups[size_label]),
                "sample_count_with_multiple_scan_sizes": multi_size_samples,
                "sample_count_only_1um": only_1um,
                "sample_count_with_0p5um_or_5um": with_half_or_five,
            }
        )
    return out
