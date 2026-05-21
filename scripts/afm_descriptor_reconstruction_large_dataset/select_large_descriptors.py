#!/usr/bin/env python3
"""Select standardized descriptors for the large AFM reconstruction experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESCRIPTOR_CSV = (
    REPO_ROOT / "data" / "afm_descriptor_reconstruction_large" / "large_afm_descriptors.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "afm_descriptor_reconstruction_large" / "selected_descriptors"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "afm_descriptor_reconstruction_large"
ID_COLUMNS = {
    "row_id",
    "sample_id",
    "afm_path",
    "network_input_path",
    "descriptor_json_path",
    "warnings",
}
PROTECTED_COLUMNS = {
    "scan_size_x_um",
    "scan_size_y_um",
    "area_um2",
    "log_area_um2",
    "pixel_size_x_nm",
    "pixel_size_y_nm",
    "mean_pixel_size_nm",
    "aspect_ratio",
    "is_1um_scan",
}
MAX_MISSING_FRACTION = 0.20
NEAR_ZERO_VARIANCE = 1e-10
CORR_THRESHOLD = 0.98


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_existing_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.exists():
        return expanded.resolve()
    repo_relative = REPO_ROOT / expanded
    if repo_relative.exists():
        return repo_relative.resolve()
    return expanded.resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric_values(rows: list[dict[str, str]], column: str) -> tuple[np.ndarray, bool]:
    values = []
    saw_finite = False
    for row in rows:
        raw = row.get(column, "")
        if raw == "":
            values.append(np.nan)
            continue
        try:
            value = float(raw)
        except ValueError:
            return np.asarray([]), False
        values.append(value)
        saw_finite = saw_finite or np.isfinite(value)
    return np.asarray(values, dtype=float), saw_finite


def select(rows: list[dict[str, str]]) -> tuple[list[str], np.ndarray, dict[str, list[str]], int]:
    dropped: dict[str, list[str]] = defaultdict(list)
    values_by_column: dict[str, np.ndarray] = {}
    numeric_columns: list[str] = []
    for column in rows[0]:
        if column in ID_COLUMNS:
            dropped[column].append("identifier/non-feature")
            continue
        values, is_numeric = numeric_values(rows, column)
        if not is_numeric:
            dropped[column].append("non-numeric")
            continue
        numeric_columns.append(column)
        values_by_column[column] = values

    filtered: list[str] = []
    for column in numeric_columns:
        values = values_by_column[column]
        missing = float(np.mean(~np.isfinite(values)))
        finite = values[np.isfinite(values)]
        if missing > MAX_MISSING_FRACTION:
            dropped[column].append(f"too many missing values ({missing:.1%})")
            continue
        if finite.size == 0:
            dropped[column].append("all values missing")
            continue
        variance = float(np.var(finite))
        if variance == 0:
            dropped[column].append("zero variance")
            continue
        if variance < NEAR_ZERO_VARIANCE:
            dropped[column].append(f"near-zero variance ({variance:.3g})")
            continue
        filtered.append(column)

    matrix = np.column_stack([values_by_column[column] for column in filtered]).astype(float)
    for index, column in enumerate(filtered):
        col = matrix[:, index]
        col[~np.isfinite(col)] = float(np.nanmedian(col))
        matrix[:, index] = col
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    standardized = (matrix - mean) / std

    corr = np.corrcoef(standardized, rowvar=False)
    corr = np.nan_to_num(corr)
    keep: list[int] = []
    dropped_indices: set[int] = set()
    for index, column in enumerate(filtered):
        if index in dropped_indices:
            continue
        keep.append(index)
        for other in range(index + 1, len(filtered)):
            if other in dropped_indices:
                continue
            if abs(float(corr[index, other])) > CORR_THRESHOLD:
                if filtered[other] in PROTECTED_COLUMNS:
                    continue
                dropped_indices.add(other)
                dropped[filtered[other]].append(
                    f"correlated with {column} (r={corr[index, other]:.3f})"
                )
    selected_columns = [filtered[index] for index in keep]
    selected_matrix = standardized[:, keep]
    return selected_columns, selected_matrix, dropped, len(numeric_columns)


def write_selected_csv(path: Path, rows: list[dict[str, str]], columns: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "sample_id", "afm_path"] + columns)
        writer.writeheader()
        for row_index, row in enumerate(rows):
            out: dict[str, Any] = {
                "row_id": row["row_id"],
                "sample_id": row["sample_id"],
                "afm_path": row["afm_path"],
            }
            for col_index, column in enumerate(columns):
                out[column] = f"{matrix[row_index, col_index]:.10g}"
            writer.writerow(out)


def write_heatmap(path: Path, columns: list[str], matrix: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr)
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(columns)), max(7, 0.33 * len(columns))), dpi=170)
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=90, fontsize=5)
    ax.set_yticklabels(columns, fontsize=5)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor_csv", type=Path, default=DEFAULT_DESCRIPTOR_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    descriptor_csv = resolve_existing_path(args.descriptor_csv)
    rows = read_csv(descriptor_csv)
    columns, matrix, dropped, original_count = select(rows)
    output_dir = args.output_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    table_path = output_dir / "selected_descriptors.csv"
    columns_path = output_dir / "selected_descriptor_columns.json"
    matrix_path = output_dir / "selected_descriptor_matrix.npy"
    report_path = output_dir / "descriptor_selection_report.md"
    heatmap_path = report_dir / "selected_descriptor_correlation_heatmap.png"
    write_selected_csv(table_path, rows, columns, matrix)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(matrix_path, matrix)
    columns_path.write_text(json.dumps(columns, indent=2) + "\n", encoding="utf-8")
    dropped_text = "\n".join(f"- `{col}`: {'; '.join(reason)}" for col, reason in sorted(dropped.items()))
    report_path.write_text(
        "# Large AFM Descriptor Selection Report\n\n"
        f"- Original numeric candidates: {original_count}\n"
        f"- Selected descriptors: {len(columns)}\n"
        f"- Correlation threshold: {CORR_THRESHOLD}\n\n"
        "Scan-size and pixel-scale features were protected from correlation-based dropping unless invalid.\n\n"
        "## Dropped Columns\n\n"
        f"{dropped_text}\n\n"
        "## Selected Columns\n\n"
        + "\n".join(f"- `{column}`" for column in columns)
        + "\n",
        encoding="utf-8",
    )
    write_heatmap(heatmap_path, columns, matrix)
    print("Large descriptor selection summary")
    print(f"  original numeric descriptors: {original_count}")
    print(f"  selected descriptor count: {len(columns)}")
    print(f"  selected descriptors: {', '.join(columns)}")
    print(f"  wrote: {display_path(table_path)}")
    print(f"  heatmap: {display_path(heatmap_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
