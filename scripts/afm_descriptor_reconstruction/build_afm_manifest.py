#!/usr/bin/env python3
"""Build a manifest of valid 1 um x 1 um plane-corrected AFM height maps."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "plane_corrected_afm"
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT / "data" / "afm_descriptor_reconstruction" / "afm_1um_manifest.csv"
)
DEFAULT_SUMMARY_JSON = DEFAULT_OUTPUT_CSV.with_name("afm_1um_manifest_summary.json")
MIN_SCAN_SIZE_UM = 0.98
MAX_SCAN_SIZE_UM = 1.02


@dataclass(frozen=True)
class ManifestRow:
    row_id: int
    sample_id: str
    afm_path: Path
    metadata_path: Path | None
    height_png_path: Path | None
    scan_size_x_um: float | None
    scan_size_y_um: float | None
    array_shape: tuple[int, int]
    source_channel: str
    notes: str


def display_path(path: Path | None) -> str:
    """Return a stable repo-relative path when possible."""
    if path is None:
        return ""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_input_dir(path: Path) -> Path:
    """Resolve user input, accepting either data/plane_corrected_afm or plane_corrected_afm."""
    expanded = path.expanduser()
    if expanded.exists():
        return expanded.resolve()
    data_relative = REPO_ROOT / "data" / expanded
    if data_relative.exists():
        return data_relative.resolve()
    return expanded.resolve()


def normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def numeric(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def pair_from_value(value: Any) -> tuple[float, float] | None:
    """Extract an x/y numeric pair from common list or dict metadata shapes."""
    if isinstance(value, dict):
        x_value = (
            value.get("x")
            or value.get("X")
            or value.get("width")
            or value.get("Width")
            or value.get("scan_size_x")
        )
        y_value = (
            value.get("y")
            or value.get("Y")
            or value.get("height")
            or value.get("Height")
            or value.get("scan_size_y")
        )
        x_num = numeric(x_value)
        y_num = numeric(y_value)
        if x_num is not None and y_num is not None:
            return x_num, y_num
    if isinstance(value, list | tuple):
        values = [numeric(item) for item in value[:2]]
        if len(values) == 2 and values[0] is not None and values[1] is not None:
            return values[0], values[1]
    value_num = numeric(value)
    if value_num is not None:
        return value_num, value_num
    return None


def iter_metadata_items(value: Any) -> list[tuple[str, Any]]:
    """Flatten nested metadata while preserving the original leaf keys."""
    items: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            items.append((str(key), child))
            items.extend(iter_metadata_items(child))
    elif isinstance(value, list):
        for child in value:
            items.extend(iter_metadata_items(child))
    return items


def unit_from_mapping(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    unit = value.get("unit") or value.get("units") or value.get("physical_unit")
    return str(unit).lower() if unit is not None else ""


def convert_pair_to_um(pair: tuple[float, float], unit_hint: str) -> tuple[float, float] | None:
    """Convert a scan-size pair to micrometers using key/unit hints."""
    unit_hint = unit_hint.lower()
    if "nm" in unit_hint or "nanometer" in unit_hint:
        return pair[0] / 1000.0, pair[1] / 1000.0
    if "um" in unit_hint or "micron" in unit_hint or "micrometer" in unit_hint:
        return pair

    # When units are not explicit, only accept values that are clearly um or nm.
    if 0.1 <= pair[0] <= 20.0 and 0.1 <= pair[1] <= 20.0:
        return pair
    if 100.0 <= pair[0] <= 20_000.0 and 100.0 <= pair[1] <= 20_000.0:
        return pair[0] / 1000.0, pair[1] / 1000.0
    return None


def scan_size_from_metadata(metadata: dict[str, Any]) -> tuple[float, float, str] | None:
    """Find scan size using common metadata names, preferring explicit units."""
    priority_keys = (
        "scansizeum",
        "scansizemicrometer",
        "scansizemicrometers",
        "scansizenm",
        "scansizenanometer",
        "scansizenanometers",
        "scansize",
        "physicalsizeum",
        "physicalsizenm",
        "physicalsize",
    )
    all_items = iter_metadata_items(metadata)
    by_key: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for key, value in all_items:
        by_key[normalized_key(key)].append((key, value))

    for wanted in priority_keys:
        for key, value in by_key.get(wanted, []):
            pair = pair_from_value(value)
            if pair is None:
                continue
            unit_hint = f"{key} {unit_from_mapping(value)}"
            converted = convert_pair_to_um(pair, unit_hint)
            if converted is not None:
                return converted[0], converted[1], f"metadata:{key}"

    width_values = by_key.get("width", []) + by_key.get("scanwidth", [])
    height_values = by_key.get("height", []) + by_key.get("scanheight", [])
    for width_key, width_value in width_values:
        for height_key, height_value in height_values:
            width = numeric(width_value)
            height = numeric(height_value)
            if width is None or height is None:
                continue
            converted = convert_pair_to_um((width, height), f"{width_key} {height_key}")
            if converted is not None:
                return converted[0], converted[1], f"metadata:{width_key}/{height_key}"

    return None


def scan_size_from_filename(path: Path) -> tuple[float, float, str] | None:
    """Fallback parser for names such as N6048_1um_026 or 500_nm."""
    name = path.stem.lower()
    if re.search(r"(^|[_-])1\s*um([_-]|$)", name):
        return 1.0, 1.0, "filename:1um"
    if re.search(r"(^|[_-])1000\s*nm([_-]|$)", name):
        return 1.0, 1.0, "filename:1000nm"
    return None


def within_target_scan_size(x_um: float | None, y_um: float | None) -> bool:
    return (
        x_um is not None
        and y_um is not None
        and MIN_SCAN_SIZE_UM <= x_um <= MAX_SCAN_SIZE_UM
        and MIN_SCAN_SIZE_UM <= y_um <= MAX_SCAN_SIZE_UM
    )


def load_metadata(path: Path | None) -> tuple[dict[str, Any], str | None]:
    if path is None or not path.exists():
        return {}, "metadata missing"
    try:
        with path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"metadata unreadable: {exc}"
    if not isinstance(metadata, dict):
        return {}, "metadata is not a JSON object"
    return metadata, None


def matching_metadata_path(array_path: Path) -> Path | None:
    if not array_path.name.endswith("_plane_corrected.npy"):
        return None
    base_name = array_path.name.removesuffix("_plane_corrected.npy")
    metadata_path = array_path.with_name(f"{base_name}_plane_corrected_metadata.json")
    return metadata_path if metadata_path.exists() else None


def matching_png_path(array_path: Path) -> Path | None:
    base_name = array_path.name.removesuffix("_plane_corrected.npy")
    render_path = array_path.with_name(f"{base_name}_plane_corrected_render.png")
    if render_path.exists():
        return render_path
    png_matches = sorted(array_path.parent.glob("*.png"))
    return png_matches[0] if png_matches else None


def sample_id_for_path(path: Path, input_dir: Path, metadata: dict[str, Any]) -> str:
    sample_id = metadata.get("sample_id")
    if sample_id not in ("", None):
        return str(sample_id)
    try:
        return path.relative_to(input_dir).parts[0]
    except (ValueError, IndexError):
        return path.parent.parent.name


def source_channel_from_metadata(metadata: dict[str, Any]) -> str:
    channel = metadata.get("primary_channel") or metadata.get("source_channel")
    if channel not in ("", None):
        return str(channel)
    channels = metadata.get("channels")
    if isinstance(channels, dict) and "ZSensor" in channels:
        return "ZSensor"
    available = metadata.get("available_channels")
    if isinstance(available, list) and "ZSensor" in available:
        return "ZSensor"
    return "unknown"


def validate_array(path: Path) -> tuple[tuple[int, int] | None, str | None]:
    try:
        array = np.load(path, mmap_mode="r")
    except Exception as exc:  # noqa: BLE001 - keep manifest building robust.
        return None, f"array unreadable: {exc}"
    if array.ndim != 2:
        return None, f"array is not 2D: shape={array.shape}"
    if not np.issubdtype(array.dtype, np.number):
        return None, f"array is not numeric: dtype={array.dtype}"
    return (int(array.shape[0]), int(array.shape[1])), None


def is_plane_corrected_candidate(path: Path) -> bool:
    return path.suffix == ".npy" and path.name.endswith("_plane_corrected.npy")


def build_manifest(input_dir: Path) -> tuple[list[ManifestRow], dict[str, Any]]:
    all_sample_ids = sorted([path.name for path in input_dir.iterdir() if path.is_dir()])
    npy_files = sorted(input_dir.rglob("*.npy"))
    rows: list[ManifestRow] = []
    shape_counts: Counter[str] = Counter()

    for array_path in npy_files:
        if not is_plane_corrected_candidate(array_path):
            continue

        notes: list[str] = []
        metadata_path = matching_metadata_path(array_path)
        metadata, metadata_note = load_metadata(metadata_path)
        if metadata_note:
            notes.append(metadata_note)

        scan_size = scan_size_from_metadata(metadata) if metadata else None
        if scan_size is None:
            scan_size = scan_size_from_filename(array_path)
        if scan_size is None:
            continue

        scan_size_x_um, scan_size_y_um, scan_size_source = scan_size
        if not within_target_scan_size(scan_size_x_um, scan_size_y_um):
            continue

        array_shape, array_note = validate_array(array_path)
        if array_note is not None or array_shape is None:
            continue

        source_channel = source_channel_from_metadata(metadata)
        if source_channel != "ZSensor":
            # With no metadata, filename fallback cannot prove channel provenance.
            if metadata:
                continue
            notes.append("source channel unknown; accepted by filename fallback")

        notes.append(scan_size_source)
        shape_counts[f"{array_shape[0]}x{array_shape[1]}"] += 1
        rows.append(
            ManifestRow(
                row_id=len(rows) + 1,
                sample_id=sample_id_for_path(array_path, input_dir, metadata),
                afm_path=array_path,
                metadata_path=metadata_path,
                height_png_path=matching_png_path(array_path),
                scan_size_x_um=scan_size_x_um,
                scan_size_y_um=scan_size_y_um,
                array_shape=array_shape,
                source_channel=source_channel,
                notes="; ".join(notes),
            )
        )

    valid_counts_by_sample = Counter(row.sample_id for row in rows)
    samples_with_zero = [sample_id for sample_id in all_sample_ids if valid_counts_by_sample[sample_id] == 0]
    samples_with_multiple = {
        sample_id: count
        for sample_id, count in sorted(valid_counts_by_sample.items())
        if count > 1
    }
    summary = {
        "input_dir": display_path(input_dir),
        "total_files_scanned": len(npy_files),
        "valid_1um_afm_files_found": len(rows),
        "number_of_unique_sample_ids": len(valid_counts_by_sample),
        "samples_with_zero_valid_1um_files": samples_with_zero,
        "samples_with_multiple_valid_1um_files": samples_with_multiple,
        "distribution_of_array_shapes": dict(sorted(shape_counts.items())),
    }
    return rows, summary


def write_manifest_csv(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_id",
                "sample_id",
                "afm_path",
                "metadata_path",
                "height_png_path",
                "scan_size_x_um",
                "scan_size_y_um",
                "array_shape",
                "source_channel",
                "notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "row_id": row.row_id,
                    "sample_id": row.sample_id,
                    "afm_path": display_path(row.afm_path),
                    "metadata_path": display_path(row.metadata_path),
                    "height_png_path": display_path(row.height_png_path),
                    "scan_size_x_um": f"{row.scan_size_x_um:.6g}"
                    if row.scan_size_x_um is not None
                    else "",
                    "scan_size_y_um": f"{row.scan_size_y_um:.6g}"
                    if row.scan_size_y_um is not None
                    else "",
                    "array_shape": f"{row.array_shape[0]}x{row.array_shape[1]}",
                    "source_channel": row.source_channel,
                    "notes": row.notes,
                }
            )


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def print_summary(summary: dict[str, Any], output_csv: Path, summary_json: Path, dry_run: bool) -> None:
    print("AFM 1um manifest summary")
    print(f"  input_dir: {summary['input_dir']}")
    print(f"  total .npy files scanned: {summary['total_files_scanned']}")
    print(f"  valid 1um AFM files found: {summary['valid_1um_afm_files_found']}")
    print(f"  unique sample_ids: {summary['number_of_unique_sample_ids']}")
    print(
        "  samples with zero valid 1um files: "
        f"{len(summary['samples_with_zero_valid_1um_files'])}"
    )
    print(
        "  samples with multiple valid 1um files: "
        f"{len(summary['samples_with_multiple_valid_1um_files'])}"
    )
    print(f"  array shape distribution: {summary['distribution_of_array_shapes']}")
    if dry_run:
        print("  dry run: no files written")
    else:
        print(f"  wrote CSV: {display_path(output_csv)}")
        print(f"  wrote JSON summary: {display_path(summary_json)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a CSV manifest for valid 1um x 1um plane-corrected AFM arrays."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Input plane_corrected_afm directory. Relative plane_corrected_afm resolves under data/.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Output manifest CSV path.",
    )
    parser.add_argument(
        "--summary_json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Scan and print the summary without writing CSV or JSON files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_dir = resolve_input_dir(args.input_dir)
    output_csv = args.output_csv.expanduser().resolve()
    summary_json = args.summary_json.expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    rows, summary = build_manifest(input_dir)
    if not args.dry_run:
        write_manifest_csv(output_csv, rows)
        write_summary_json(summary_json, summary)
    print_summary(summary, output_csv, summary_json, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
