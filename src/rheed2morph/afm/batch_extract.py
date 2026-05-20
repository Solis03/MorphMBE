#!/usr/bin/env python3
"""
Batch extract AFM raw files organized as pair/{sample_id}/AFM/{raw_afm_file}.

Example:
    python scripts/batch_extract_afm_by_sample.py \
      --pair_root data/pair \
      --output_root data/processed_afm

The first output level is the sample/growth id. The second level is the AFM
file id, for example:

    data/processed_afm/6022/N6022_Ctr_000/N6022_Ctr_000_height.npy
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rheed2morph.afm.inspect import extract_one_afm_file, make_safe_id


AFM_SUMMARY_COLUMNS = [
    "sample_id",
    "afm_file_id",
    "raw_afm_file",
    "relative_path",
    "status",
    "primary_channel",
    "secondary_channel",
    "available_channels",
    "resolution_h",
    "resolution_w",
    "scan_size_x_um",
    "scan_size_y_um",
    "height_unit_original",
    "height_unit_exported",
    "height_min_nm",
    "height_max_nm",
    "height_mean_nm",
    "height_std_nm",
    "output_dir",
    "error_message",
]

SAMPLE_SUMMARY_COLUMNS = [
    "sample_id",
    "afm_dir",
    "num_raw_afm_files_found",
    "num_success",
    "num_failed",
    "num_zsensor",
    "num_height_fallback",
    "scan_sizes_um",
    "resolutions",
    "height_std_nm_mean",
    "height_std_nm_min",
    "height_std_nm_max",
    "representative_afm_file_id",
]

# Raw Bruker/Nanoscope files in this dataset often have numeric suffixes such
# as ".000". Image exports are intentionally skipped.
IGNORED_IMAGE_SUFFIXES = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}


def is_raw_afm_candidate(path: Path) -> bool:
    """Return True for files that should be attempted as raw AFM files."""
    if not path.is_file():
        return False
    if path.name.startswith("."):
        return False
    return path.suffix.lower() not in IGNORED_IMAGE_SUFFIXES


def find_samples(pair_root: Path) -> list[Path]:
    """Return pair/{sample_id} directories sorted by sample id."""
    return sorted([p for p in pair_root.iterdir() if p.is_dir()], key=lambda p: p.name)


def find_raw_afm_files(afm_dir: Path) -> list[Path]:
    """Return direct child files in AFM folder, excluding rendered/exported images."""
    if not afm_dir.is_dir():
        return []
    return sorted([p for p in afm_dir.iterdir() if is_raw_afm_candidate(p)], key=lambda p: p.name)


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def display_path(path: Path) -> str:
    """Return a portable path string for reports."""
    if not path.is_absolute():
        return str(path)
    return os.path.relpath(path, Path.cwd())


def unique_join(values: list[str]) -> str:
    return ";".join(sorted({value for value in values if value}))


def number_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sample_has_scan_size(
    rows: list[dict[str, Any]],
    target_x_um: float,
    target_y_um: float,
    tolerance: float = 1e-6,
) -> bool:
    """Return True if the sample has at least one successful scan matching target size."""
    for row in rows:
        if row.get("status") != "success":
            continue
        x_value = number_or_none(row.get("scan_size_x_um"))
        y_value = number_or_none(row.get("scan_size_y_um"))
        if x_value is None or y_value is None:
            continue
        if abs(x_value - target_x_um) <= tolerance and abs(y_value - target_y_um) <= tolerance:
            return True
    return False


def choose_representative(rows: list[dict[str, Any]]) -> str:
    successes = [row for row in rows if row["status"] == "success"]
    zsensor_rows = sorted(
        [row for row in successes if row["primary_channel"] == "ZSensor"],
        key=lambda row: row["afm_file_id"],
    )
    if zsensor_rows:
        return zsensor_rows[0]["afm_file_id"]
    if successes:
        return sorted(successes, key=lambda row: row["afm_file_id"])[0]["afm_file_id"]
    return ""


def build_sample_summary(
    sample_dir: Path,
    afm_dir: Path,
    rows: list[dict[str, Any]],
    num_raw_found: int,
) -> dict[str, Any]:
    success_rows = [row for row in rows if row["status"] == "success"]
    failed_rows = [row for row in rows if row["status"] == "failed"]
    height_std_values = [
        value
        for value in (number_or_none(row["height_std_nm"]) for row in success_rows)
        if value is not None
    ]
    scan_sizes = unique_join(
        [
            f"{row['scan_size_x_um']}x{row['scan_size_y_um']}"
            for row in success_rows
            if row["scan_size_x_um"] != "" and row["scan_size_y_um"] != ""
        ]
    )
    resolutions = unique_join(
        [
            f"{row['resolution_h']}x{row['resolution_w']}"
            for row in success_rows
            if row["resolution_h"] != "" and row["resolution_w"] != ""
        ]
    )

    return {
        "sample_id": sample_dir.name,
        "afm_dir": display_path(afm_dir),
        "num_raw_afm_files_found": num_raw_found,
        "num_success": len(success_rows),
        "num_failed": len(failed_rows),
        "num_zsensor": sum(row["primary_channel"] == "ZSensor" for row in success_rows),
        "num_height_fallback": sum(row["primary_channel"] == "Height" for row in success_rows),
        "scan_sizes_um": scan_sizes,
        "resolutions": resolutions,
        "height_std_nm_mean": (
            sum(height_std_values) / len(height_std_values) if height_std_values else ""
        ),
        "height_std_nm_min": min(height_std_values) if height_std_values else "",
        "height_std_nm_max": max(height_std_values) if height_std_values else "",
        "representative_afm_file_id": choose_representative(success_rows),
    }


def distribution(rows: list[dict[str, Any]], x_key: str, y_key: str) -> Counter[str]:
    values = []
    for row in rows:
        if row["status"] != "success":
            continue
        x_value = row.get(x_key, "")
        y_value = row.get(y_key, "")
        if x_value != "" and y_value != "":
            values.append(f"{x_value}x{y_value}")
    return Counter(values)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch extract AFM raw files from pair/{sample_id}/AFM folders."
    )
    parser.add_argument(
        "--pair_root",
        required=True,
        type=Path,
        help="Root folder containing pair/{sample_id}/AFM.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        type=Path,
        help="Root folder for processed AFM outputs.",
    )
    args = parser.parse_args()

    pair_root = args.pair_root.expanduser()
    output_root = args.output_root.expanduser()

    if not pair_root.is_dir():
        raise SystemExit(f"pair_root does not exist or is not a directory: {pair_root}")

    all_afm_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    rows_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for sample_dir in find_samples(pair_root):
        sample_id = sample_dir.name
        afm_dir = sample_dir / "AFM"
        raw_files = find_raw_afm_files(afm_dir)

        for raw_file in raw_files:
            afm_file_id = make_safe_id(raw_file.name)
            relative_path = str(raw_file.relative_to(pair_root))
            file_output_dir = output_root / sample_id / afm_file_id
            print(f"[{sample_id}] extracting {raw_file.name}")
            row = extract_one_afm_file(
                raw_file,
                file_output_dir,
                sample_id=sample_id,
                afm_file_id=afm_file_id,
                relative_path=relative_path,
            )
            all_afm_rows.append(row)
            rows_by_sample[sample_id].append(row)

        sample_rows.append(
            build_sample_summary(
                sample_dir=sample_dir,
                afm_dir=afm_dir,
                rows=rows_by_sample[sample_id],
                num_raw_found=len(raw_files),
            )
        )

    afm_summary_path = output_root / "afm_summary.csv"
    sample_summary_path = output_root / "sample_summary.csv"
    write_csv(afm_summary_path, AFM_SUMMARY_COLUMNS, all_afm_rows)
    write_csv(sample_summary_path, SAMPLE_SUMMARY_COLUMNS, sample_rows)

    success_rows = [row for row in all_afm_rows if row["status"] == "success"]
    failed_rows = [row for row in all_afm_rows if row["status"] == "failed"]
    scan_size_dist = distribution(all_afm_rows, "scan_size_x_um", "scan_size_y_um")
    resolution_dist = distribution(all_afm_rows, "resolution_h", "resolution_w")
    sample_ids = sorted(rows_by_sample)
    samples_missing_1um = [
        sample_id
        for sample_id in sample_ids
        if not sample_has_scan_size(rows_by_sample[sample_id], target_x_um=1.0, target_y_um=1.0)
    ]
    samples_with_1um = len(sample_ids) - len(samples_missing_1um)

    print()
    print("AFM batch extraction summary")
    print("----------------------------")
    print(f"Samples scanned: {len(sample_rows)}")
    print(f"Raw AFM files found: {len(all_afm_rows)}")
    print(f"Successfully processed: {len(success_rows)}")
    print(f"Failed: {len(failed_rows)}")
    print(f"Using ZSensor: {sum(row['primary_channel'] == 'ZSensor' for row in success_rows)}")
    print(f"Using Height fallback: {sum(row['primary_channel'] == 'Height' for row in success_rows)}")
    print(f"Scan size distribution: {dict(scan_size_dist)}")
    print(f"Resolution distribution: {dict(resolution_dist)}")
    print(f"Samples with >=1 1.0x1.0 scan: {samples_with_1um}/{len(sample_ids)}")
    if samples_missing_1um:
        print(f"Samples missing 1.0x1.0 scan: {', '.join(samples_missing_1um)}")
    else:
        print("Samples missing 1.0x1.0 scan: none")
    print(f"AFM summary CSV: {afm_summary_path}")
    print(f"Sample summary CSV: {sample_summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
