#!/usr/bin/env python3
"""Build a manifest for all valid plane-corrected AFM ZSensor height maps."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "plane_corrected_afm"
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT / "data" / "afm_descriptor_reconstruction_large" / "large_afm_manifest.csv"
)
DEFAULT_SUMMARY_JSON = DEFAULT_OUTPUT_CSV.with_name("large_afm_manifest_summary.json")
MAX_BAD_PIXEL_FRACTION = 0.20


def display_path(path: Path | None) -> str:
    if path is None:
        return ""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_input_dir(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.exists():
        return expanded.resolve()
    data_relative = REPO_ROOT / "data" / expanded
    if data_relative.exists():
        return data_relative.resolve()
    return expanded.resolve()


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        x = number(value.get("x") or value.get("width") or value.get("scan_size_x"))
        y = number(value.get("y") or value.get("height") or value.get("scan_size_y"))
        return (x, y) if x is not None and y is not None else None
    if isinstance(value, list | tuple) and len(value) >= 2:
        x, y = number(value[0]), number(value[1])
        return (x, y) if x is not None and y is not None else None
    scalar = number(value)
    return (scalar, scalar) if scalar is not None else None


def nested_items(value: Any) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            items.append((str(key), child))
            items.extend(nested_items(child))
    elif isinstance(value, list):
        for child in value:
            items.extend(nested_items(child))
    return items


def convert_to_um(values: tuple[float, float], key: str) -> tuple[float, float] | None:
    key = key.lower()
    if "nm" in key or "nanometer" in key:
        return values[0] / 1000.0, values[1] / 1000.0
    if "um" in key or "micron" in key or "micrometer" in key:
        # Some legacy metadata in this dataset stores nanometer-scale scans in
        # scan_size_um, e.g. 500 for a 500 nm scan. Treat very large values as
        # likely nm rather than accepting implausible hundreds-of-um AFM scans.
        if values[0] > 20.0 or values[1] > 20.0:
            return values[0] / 1000.0, values[1] / 1000.0
        return values
    if 0.01 <= values[0] <= 100.0 and 0.01 <= values[1] <= 100.0:
        return values
    if 10.0 <= values[0] <= 100_000.0 and 10.0 <= values[1] <= 100_000.0:
        return values[0] / 1000.0, values[1] / 1000.0
    return None


def scan_size_from_metadata(metadata: dict[str, Any]) -> tuple[float, float] | None:
    wanted = {
        "scansize",
        "scansizeum",
        "scansizenm",
        "physicalsize",
        "physicalsizeum",
        "physicalsizenm",
        "width",
        "height",
    }
    seen: dict[str, Any] = {}
    for key, value in nested_items(metadata):
        normalized = re.sub(r"[^a-z0-9]", "", key.lower())
        if normalized in wanted:
            seen[normalized] = value
            values = pair(value)
            if values is not None:
                converted = convert_to_um(values, key)
                if converted is not None:
                    return converted

    width = number(seen.get("width"))
    height = number(seen.get("height"))
    if width is not None and height is not None:
        return convert_to_um((width, height), "width height")
    return None


def scan_size_from_filename(path: Path) -> tuple[float, float] | None:
    name = path.stem.lower()
    if re.search(r"(^|[_-])1\s*um([_-]|$)", name):
        return 1.0, 1.0
    if re.search(r"(^|[_-])5\s*um([_-]|$)", name):
        return 5.0, 5.0
    if re.search(r"(^|[_-])500[_-]?nm([_-]|$)", name) or "0_5um" in name:
        return 0.5, 0.5
    if re.search(r"(^|[_-])200[_-]?nm([_-]|$)", name):
        return 0.2, 0.2
    return None


def source_channel(metadata: dict[str, Any]) -> str:
    primary = metadata.get("primary_channel") or metadata.get("source_channel")
    if primary:
        return str(primary)
    channels = metadata.get("channels")
    if isinstance(channels, dict) and "ZSensor" in channels:
        return "ZSensor"
    available = metadata.get("available_channels")
    if isinstance(available, list) and "ZSensor" in available:
        return "ZSensor"
    return "unknown"


def metadata_path_for(array_path: Path) -> Path | None:
    base = array_path.name.removesuffix("_plane_corrected.npy")
    path = array_path.with_name(f"{base}_plane_corrected_metadata.json")
    return path if path.exists() else None


def png_path_for(array_path: Path) -> Path:
    base = array_path.name.removesuffix("_plane_corrected.npy")
    return array_path.with_name(f"{base}_plane_corrected_render.png")


def validate_array(path: Path) -> tuple[tuple[int, int] | None, str | None]:
    try:
        array = np.load(path, mmap_mode="r")
    except Exception as exc:  # noqa: BLE001
        return None, f"unreadable array: {exc}"
    if array.ndim != 2:
        return None, f"not 2D: shape={array.shape}"
    if not np.issubdtype(array.dtype, np.number):
        return None, f"not numeric: dtype={array.dtype}"
    bad_fraction = float(np.mean(~np.isfinite(array)))
    if bad_fraction > MAX_BAD_PIXEL_FRACTION:
        return None, f"too many NaN/Inf pixels: {bad_fraction:.1%}"
    return (int(array.shape[0]), int(array.shape[1])), None


def is_1um(x_um: float, y_um: float) -> bool:
    return 0.98 <= x_um <= 1.02 and 0.98 <= y_um <= 1.02


def build_manifest(input_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scanned = 0
    skipped = Counter()
    for array_path in sorted(input_dir.rglob("*_plane_corrected.npy")):
        scanned += 1
        shape, array_note = validate_array(array_path)
        if shape is None:
            skipped[array_note or "invalid array"] += 1
            continue
        metadata_path = metadata_path_for(array_path)
        metadata = load_json(metadata_path)
        channel = source_channel(metadata)
        if channel != "ZSensor":
            skipped[f"non-ZSensor channel: {channel}"] += 1
            continue
        scan_size = scan_size_from_metadata(metadata) or scan_size_from_filename(array_path)
        if scan_size is None:
            skipped["missing scan size"] += 1
            continue
        scan_x, scan_y = scan_size
        if scan_x <= 0 or scan_y <= 0:
            skipped["non-positive scan size"] += 1
            continue
        height_px, width_px = shape
        sample_id = str(metadata.get("sample_id") or array_path.relative_to(input_dir).parts[0])
        rows.append(
            {
                "row_id": len(rows) + 1,
                "sample_id": sample_id,
                "afm_path": display_path(array_path),
                "metadata_path": display_path(metadata_path),
                "height_png_path": display_path(png_path_for(array_path) if png_path_for(array_path).exists() else None),
                "scan_size_x_um": scan_x,
                "scan_size_y_um": scan_y,
                "original_array_shape": f"{height_px}x{width_px}",
                "original_height_pixels": height_px,
                "original_width_pixels": width_px,
                "original_height_unit": metadata.get("height_unit_exported") or metadata.get("height_unit_original") or "",
                "source_channel": channel,
                "area_um2": scan_x * scan_y,
                "pixel_size_x_nm": scan_x * 1000.0 / width_px,
                "pixel_size_y_nm": scan_y * 1000.0 / height_px,
                "aspect_ratio": scan_x / scan_y,
                "is_1um_scan": is_1um(scan_x, scan_y),
            }
        )

    scan_sizes = Counter(f"{row['scan_size_x_um']:.3g}x{row['scan_size_y_um']:.3g}" for row in rows)
    resolutions = Counter(row["original_array_shape"] for row in rows)
    one_um = sum(bool(row["is_1um_scan"]) for row in rows)
    summary = {
        "input_dir": display_path(input_dir),
        "total_afm_files_scanned": scanned,
        "valid_files_included": len(rows),
        "unique_sample_id_count": len({row["sample_id"] for row in rows}),
        "one_um_scan_count": one_um,
        "non_one_um_scan_count": len(rows) - one_um,
        "scan_size_distribution": dict(sorted(scan_sizes.items())),
        "resolution_distribution": dict(sorted(resolutions.items())),
        "skipped": dict(skipped),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "row_id",
        "sample_id",
        "afm_path",
        "metadata_path",
        "height_png_path",
        "scan_size_x_um",
        "scan_size_y_um",
        "original_array_shape",
        "original_height_pixels",
        "original_width_pixels",
        "original_height_unit",
        "source_channel",
        "area_um2",
        "pixel_size_x_nm",
        "pixel_size_y_nm",
        "aspect_ratio",
        "is_1um_scan",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)
    args = parser.parse_args()
    input_dir = resolve_input_dir(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    rows, summary = build_manifest(input_dir)
    output_csv = args.output_csv.expanduser().resolve()
    summary_json = args.summary_json.expanduser().resolve()
    write_csv(output_csv, rows)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Large AFM manifest summary")
    print(f"  total AFM files scanned: {summary['total_afm_files_scanned']}")
    print(f"  valid files included: {summary['valid_files_included']}")
    print(f"  unique sample_id: {summary['unique_sample_id_count']}")
    print(f"  1um scans: {summary['one_um_scan_count']}")
    print(f"  non-1um scans: {summary['non_one_um_scan_count']}")
    print(f"  scan size distribution: {summary['scan_size_distribution']}")
    print(f"  resolution distribution: {summary['resolution_distribution']}")
    print(f"  wrote: {display_path(output_csv)}")
    print(f"  wrote: {display_path(summary_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
