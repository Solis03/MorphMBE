#!/usr/bin/env python3
"""Select a compact, standardized AFM descriptor matrix."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESCRIPTOR_CSV = REPO_ROOT / "data" / "afm_descriptor_reconstruction" / "afm_descriptors.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "afm_descriptor_reconstruction" / "selected_descriptors"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "afm_descriptor_reconstruction"

IDENTIFIER_COLUMNS = {
    "row_id",
    "sample_id",
    "afm_path",
    "descriptor_json_path",
    "network_input_path",
    "array_shape",
    "warnings",
}
MAX_MISSING_FRACTION = 0.20
NEAR_ZERO_VARIANCE_THRESHOLD = 1e-10
CORRELATION_THRESHOLD = 0.95


def display_path(path: Path | None) -> str:
    if path is None:
        return ""
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


def parse_numeric_column(rows: list[dict[str, str]], column: str) -> tuple[np.ndarray, bool]:
    values: list[float] = []
    saw_numeric = False
    for row in rows:
        raw_value = row.get(column, "")
        if raw_value in ("", None):
            values.append(np.nan)
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return np.asarray([], dtype=float), False
        values.append(value)
        if np.isfinite(value):
            saw_numeric = True
    return np.asarray(values, dtype=float), saw_numeric


def collect_numeric_candidates(
    rows: list[dict[str, str]],
) -> tuple[list[str], dict[str, np.ndarray], dict[str, list[str]]]:
    columns = list(rows[0].keys()) if rows else []
    numeric_columns: list[str] = []
    values_by_column: dict[str, np.ndarray] = {}
    dropped: dict[str, list[str]] = defaultdict(list)

    for column in columns:
        if column in IDENTIFIER_COLUMNS:
            dropped[column].append("identifier field")
            continue
        values, is_numeric = parse_numeric_column(rows, column)
        if not is_numeric:
            dropped[column].append("non-numeric")
            continue
        numeric_columns.append(column)
        values_by_column[column] = values
    return numeric_columns, values_by_column, dropped


def filter_columns(
    numeric_columns: list[str],
    values_by_column: dict[str, np.ndarray],
    dropped: dict[str, list[str]],
) -> list[str]:
    kept: list[str] = []
    for column in numeric_columns:
        values = values_by_column[column]
        missing_fraction = float(np.mean(~np.isfinite(values)))
        finite_values = values[np.isfinite(values)]
        if missing_fraction > MAX_MISSING_FRACTION:
            dropped[column].append(f"too many missing values ({missing_fraction:.1%})")
            continue
        if finite_values.size == 0:
            dropped[column].append("all values missing")
            continue
        variance = float(np.var(finite_values))
        if variance == 0.0:
            dropped[column].append("zero variance")
            continue
        if variance < NEAR_ZERO_VARIANCE_THRESHOLD:
            dropped[column].append(f"near-zero variance ({variance:.3g})")
            continue
        kept.append(column)
    return kept


def impute_matrix(columns: list[str], values_by_column: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    matrix = np.column_stack([values_by_column[column] for column in columns]).astype(float)
    medians: dict[str, float] = {}
    for index, column in enumerate(columns):
        values = matrix[:, index]
        median = float(np.nanmedian(values))
        values[~np.isfinite(values)] = median
        matrix[:, index] = values
        medians[column] = median
    return matrix, medians


def remove_correlated_columns(
    columns: list[str],
    standardized: np.ndarray,
    dropped: dict[str, list[str]],
) -> list[str]:
    if len(columns) <= 1:
        return columns

    corr = np.corrcoef(standardized, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    selected_indices: list[int] = []
    dropped_indices: set[int] = set()

    for index, column in enumerate(columns):
        if index in dropped_indices:
            continue
        selected_indices.append(index)
        for other_index in range(index + 1, len(columns)):
            if other_index in dropped_indices:
                continue
            if abs(float(corr[index, other_index])) > CORRELATION_THRESHOLD:
                dropped_indices.add(other_index)
                dropped[columns[other_index]].append(
                    f"highly correlated with {column} (r={corr[index, other_index]:.3f})"
                )
    return [columns[index] for index in selected_indices]


def write_selected_csv(
    path: Path,
    source_rows: list[dict[str, str]],
    selected_columns: list[str],
    selected_matrix: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["row_id", "sample_id", "afm_path"] + selected_columns
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, source_row in enumerate(source_rows):
            output_row: dict[str, Any] = {
                "row_id": source_row.get("row_id", ""),
                "sample_id": source_row.get("sample_id", ""),
                "afm_path": source_row.get("afm_path", ""),
            }
            for col_index, column in enumerate(selected_columns):
                output_row[column] = f"{selected_matrix[row_index, col_index]:.10g}"
            writer.writerow(output_row)


def write_report(
    path: Path,
    original_descriptor_count: int,
    selected_columns: list[str],
    dropped: dict[str, list[str]],
    output_paths: dict[str, Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dropped_lines = []
    for column in sorted(dropped):
        reasons = "; ".join(dropped[column])
        dropped_lines.append(f"- `{column}`: {reasons}")
    if not dropped_lines:
        dropped_lines.append("- None")

    selected_lines = [f"- `{column}`" for column in selected_columns]
    path.write_text(
        "# AFM Descriptor Selection Report\n\n"
        "This report summarizes descriptor selection for the AFM descriptor-to-image "
        "reconstruction baseline.\n\n"
        f"- Original descriptor candidates: {original_descriptor_count}\n"
        f"- Selected descriptors: {len(selected_columns)}\n"
        f"- Missing-value threshold: {MAX_MISSING_FRACTION:.0%}\n"
        f"- Near-zero variance threshold: {NEAR_ZERO_VARIANCE_THRESHOLD:g}\n"
        f"- Redundancy threshold: absolute Pearson correlation > {CORRELATION_THRESHOLD:.2f}\n\n"
        "With roughly 37 unique samples, this analysis is exploratory and not "
        "statistically conclusive. The selected features should be treated as a "
        "compact baseline for reconstruction experiments, not as final morphology "
        "markers.\n\n"
        "## Dropped Columns\n\n"
        + "\n".join(dropped_lines)
        + "\n\n## Selected Columns\n\n"
        + "\n".join(selected_lines)
        + "\n\n## Outputs\n\n"
        + "\n".join(f"- `{name}`: `{display_path(path_value)}`" for name, path_value in output_paths.items())
        + "\n",
        encoding="utf-8",
    )


def write_heatmap(path: Path, selected_columns: list[str], selected_matrix: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if selected_matrix.shape[1] <= 1:
        corr = np.eye(selected_matrix.shape[1])
    else:
        corr = np.corrcoef(selected_matrix, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig_width = max(7.0, 0.36 * len(selected_columns))
    fig_height = max(6.5, 0.34 * len(selected_columns))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=170)
    image = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(np.arange(len(selected_columns)))
    ax.set_yticks(np.arange(len(selected_columns)))
    ax.set_xticklabels(selected_columns, rotation=90, fontsize=6)
    ax.set_yticklabels(selected_columns, fontsize=6)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def select_descriptors(rows: list[dict[str, str]]) -> tuple[list[str], np.ndarray, StandardScaler, dict[str, list[str]], int]:
    numeric_columns, values_by_column, dropped = collect_numeric_candidates(rows)
    original_descriptor_count = len(numeric_columns)
    filtered_columns = filter_columns(numeric_columns, values_by_column, dropped)
    imputed_matrix, medians = impute_matrix(filtered_columns, values_by_column)

    scaler = StandardScaler()
    standardized = scaler.fit_transform(imputed_matrix)
    scaler.feature_names_in_ = np.asarray(filtered_columns, dtype=object)
    scaler.imputation_medians_ = medians

    selected_columns = remove_correlated_columns(filtered_columns, standardized, dropped)
    selected_indices = [filtered_columns.index(column) for column in selected_columns]
    selected_matrix = standardized[:, selected_indices]
    return selected_columns, selected_matrix, scaler, dropped, original_descriptor_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select compact standardized descriptors for AFM reconstruction."
    )
    parser.add_argument("--descriptor_csv", type=Path, default=DEFAULT_DESCRIPTOR_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    descriptor_csv = resolve_existing_path(args.descriptor_csv)
    output_dir = args.output_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()

    if not descriptor_csv.is_file():
        raise SystemExit(f"Descriptor CSV does not exist: {descriptor_csv}")

    rows = read_csv(descriptor_csv)
    if not rows:
        raise SystemExit(f"Descriptor CSV has no rows: {descriptor_csv}")

    selected_columns, selected_matrix, scaler, dropped, original_count = select_descriptors(rows)
    if not selected_columns:
        raise SystemExit("No descriptors were selected.")

    matrix_path = output_dir / "selected_descriptor_matrix.npy"
    table_path = output_dir / "selected_descriptor_table.csv"
    columns_path = output_dir / "selected_descriptor_columns.json"
    scaler_path = output_dir / "selected_descriptor_scaler.joblib"
    report_path = output_dir / "descriptor_selection_report.md"
    heatmap_path = report_dir / "selected_descriptor_correlation_heatmap.png"

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(matrix_path, selected_matrix)
    write_selected_csv(table_path, rows, selected_columns, selected_matrix)
    columns_path.write_text(json.dumps(selected_columns, indent=2) + "\n", encoding="utf-8")
    joblib.dump(scaler, scaler_path)
    write_heatmap(heatmap_path, selected_columns, selected_matrix)
    write_report(
        report_path,
        original_count,
        selected_columns,
        dropped,
        {
            "selected matrix": matrix_path,
            "selected table": table_path,
            "selected columns": columns_path,
            "scaler": scaler_path,
            "correlation heatmap": heatmap_path,
        },
    )

    print("AFM descriptor selection summary")
    print(f"  original numeric descriptor candidates: {original_count}")
    print(f"  selected descriptor count: {len(selected_columns)}")
    print(f"  selected descriptors: {', '.join(selected_columns)}")
    print(f"  selected matrix: {display_path(matrix_path)}")
    print(f"  selected table: {display_path(table_path)}")
    print(f"  selected columns JSON: {display_path(columns_path)}")
    print(f"  scaler: {display_path(scaler_path)}")
    print(f"  report: {display_path(report_path)}")
    print(f"  heatmap: {display_path(heatmap_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
