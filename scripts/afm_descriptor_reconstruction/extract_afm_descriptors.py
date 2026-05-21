#!/usr/bin/env python3
"""Extract compact descriptors from plane-corrected 1um AFM height maps."""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "afm_descriptor_reconstruction" / "afm_1um_manifest.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data" / "afm_descriptor_reconstruction" / "afm_descriptors.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "afm_descriptor_reconstruction" / "descriptors"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "afm_descriptor_reconstruction"

ID_COLUMNS = [
    "row_id",
    "sample_id",
    "afm_path",
    "descriptor_json_path",
    "network_input_path",
    "array_shape",
]

DESCRIPTOR_COLUMNS = [
    "mean_height",
    "median_height",
    "std_height",
    "min_height",
    "max_height",
    "peak_to_valley",
    "p01",
    "p05",
    "p25",
    "p75",
    "p95",
    "p99",
    "iqr",
    "Ra",
    "Rq",
    "Rsk",
    "Rku",
    "grad_mean",
    "grad_std",
    "grad_p95",
    "orientation_entropy",
    "low_freq_power",
    "mid_freq_power",
    "high_freq_power",
    "high_low_power_ratio",
    "radial_psd_slope",
    "autocorr_length_x",
    "autocorr_length_y",
    "anisotropy_ratio",
    "coverage_fraction",
    "connected_component_count",
    "component_density",
    "mean_component_area",
    "median_component_area",
    "max_component_area",
    "mean_component_height",
    "median_component_height",
]


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


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_height(path: Path) -> tuple[np.ndarray | None, list[str]]:
    warnings: list[str] = []
    try:
        array = np.load(path)
    except Exception as exc:  # noqa: BLE001 - keep batch extraction robust.
        return None, [f"failed to load array: {exc}"]

    if array.ndim != 2:
        return None, [f"array is not 2D: shape={array.shape}"]
    if not np.issubdtype(array.dtype, np.number):
        return None, [f"array is not numeric: dtype={array.dtype}"]

    height = np.asarray(array, dtype=float)
    finite_mask = np.isfinite(height)
    if not np.any(finite_mask):
        return None, ["array has no finite pixels"]
    if not np.all(finite_mask):
        fill_value = float(np.median(height[finite_mask]))
        height = height.copy()
        height[~finite_mask] = fill_value
        warnings.append(
            f"replaced {int(np.size(height) - np.count_nonzero(finite_mask))} NaN/Inf pixels with finite median"
        )
    return height, warnings


def save_network_input(height: np.ndarray, path: Path) -> tuple[float, float, str | None]:
    low, high = np.percentile(height, [1.0, 99.0])
    warning = None
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(height))
        high = float(np.max(height))
        warning = "robust normalization fell back to min/max"
    if high <= low:
        normalized = np.zeros_like(height, dtype=np.float32)
        warning = "network input is all zeros because height range is zero"
    else:
        clipped = np.clip(height, low, high)
        normalized = ((clipped - low) / (high - low) * 2.0 - 1.0).astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, normalized)
    return float(low), float(high), warning


def height_statistics(height: np.ndarray) -> dict[str, float]:
    percentiles = np.percentile(height, [1, 5, 25, 75, 95, 99])
    mean = float(np.mean(height))
    median = float(np.median(height))
    std = float(np.std(height))
    min_height = float(np.min(height))
    max_height = float(np.max(height))
    return {
        "mean_height": mean,
        "median_height": median,
        "std_height": std,
        "min_height": min_height,
        "max_height": max_height,
        "peak_to_valley": max_height - min_height,
        "p01": float(percentiles[0]),
        "p05": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p75": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "p99": float(percentiles[5]),
        "iqr": float(percentiles[3] - percentiles[2]),
    }


def roughness_descriptors(height: np.ndarray, mean: float, std: float) -> dict[str, float]:
    centered = height - mean
    rq = float(np.sqrt(np.mean(centered**2)))
    if std > 0:
        normalized = centered / std
        rsk = float(np.mean(normalized**3))
        rku = float(np.mean(normalized**4))
    else:
        rsk = 0.0
        rku = 0.0
    return {
        "Ra": float(np.mean(np.abs(centered))),
        "Rq": rq,
        "Rsk": rsk,
        "Rku": rku,
    }


def gradient_descriptors(height: np.ndarray) -> dict[str, float]:
    grad_y, grad_x = np.gradient(height)
    magnitude = np.hypot(grad_x, grad_y)
    angles = np.mod(np.arctan2(grad_y, grad_x), np.pi)
    weights = magnitude.ravel()
    histogram, _ = np.histogram(angles.ravel(), bins=18, range=(0.0, np.pi), weights=weights)
    total = float(np.sum(histogram))
    if total > 0:
        probabilities = histogram / total
        probabilities = probabilities[probabilities > 0]
        entropy = float(-np.sum(probabilities * np.log2(probabilities)) / np.log2(18))
    else:
        entropy = 0.0
    return {
        "grad_mean": float(np.mean(magnitude)),
        "grad_std": float(np.std(magnitude)),
        "grad_p95": float(np.percentile(magnitude, 95)),
        "orientation_entropy": entropy,
    }


def frequency_descriptors(height: np.ndarray) -> dict[str, float]:
    centered = height - float(np.mean(height))
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    rows, cols = height.shape
    fy = np.fft.fftshift(np.fft.fftfreq(rows))
    fx = np.fft.fftshift(np.fft.fftfreq(cols))
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    nonzero = radius > 0

    low = float(np.mean(power[(radius > 0) & (radius <= 0.10)])) if np.any((radius > 0) & (radius <= 0.10)) else 0.0
    mid = float(np.mean(power[(radius > 0.10) & (radius <= 0.25)])) if np.any((radius > 0.10) & (radius <= 0.25)) else 0.0
    high = float(np.mean(power[radius > 0.25])) if np.any(radius > 0.25) else 0.0
    ratio = high / low if low > 0 else 0.0

    radial_psd_slope = 0.0
    if np.any(nonzero):
        r = radius[nonzero].ravel()
        p = power[nonzero].ravel()
        bins = np.linspace(float(np.min(r)), float(np.max(r)), 32)
        bin_centers: list[float] = []
        bin_power: list[float] = []
        for start, end in zip(bins[:-1], bins[1:]):
            mask = (r >= start) & (r < end)
            if np.any(mask):
                value = float(np.mean(p[mask]))
                if value > 0:
                    bin_centers.append((start + end) / 2.0)
                    bin_power.append(value)
        if len(bin_centers) >= 3:
            radial_psd_slope = float(
                np.polyfit(np.log(np.asarray(bin_centers)), np.log(np.asarray(bin_power)), 1)[0]
            )

    return {
        "low_freq_power": low,
        "mid_freq_power": mid,
        "high_freq_power": high,
        "high_low_power_ratio": ratio,
        "radial_psd_slope": radial_psd_slope,
    }


def first_correlation_length(profile: np.ndarray) -> float:
    if profile.size == 0 or profile[0] <= 0:
        return 0.0
    normalized = profile / profile[0]
    threshold = 1.0 / np.e
    below = np.flatnonzero(normalized <= threshold)
    return float(below[0]) if below.size else float(profile.size - 1)


def autocorrelation_descriptors(height: np.ndarray) -> dict[str, float]:
    centered = height - float(np.mean(height))
    power = np.abs(np.fft.fft2(centered)) ** 2
    autocorr = np.fft.fftshift(np.fft.ifft2(power).real)
    center_y, center_x = autocorr.shape[0] // 2, autocorr.shape[1] // 2
    x_profile = autocorr[center_y, center_x:]
    y_profile = autocorr[center_y:, center_x]
    length_x = first_correlation_length(x_profile)
    length_y = first_correlation_length(y_profile)
    smaller = min(length_x, length_y)
    larger = max(length_x, length_y)
    anisotropy = larger / smaller if smaller > 0 else 0.0
    return {
        "autocorr_length_x": length_x,
        "autocorr_length_y": length_y,
        "anisotropy_ratio": anisotropy,
    }


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    rows, cols = mask.shape
    for row in range(rows):
        for col in range(cols):
            if not mask[row, col] or visited[row, col]:
                continue
            component: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            while queue:
                y, x = queue.popleft()
                component.append((y, x))
                for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (
                        0 <= next_y < rows
                        and 0 <= next_x < cols
                        and mask[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        queue.append((next_y, next_x))
            components.append(component)
    return components


def segmentation_descriptors(height: np.ndarray, median: float, std: float) -> dict[str, float]:
    threshold = median + 0.5 * std
    mask = height > threshold
    coverage = float(np.mean(mask))
    components = connected_components(mask)
    areas = np.asarray([len(component) for component in components], dtype=float)
    component_heights = []
    for component in components:
        ys, xs = zip(*component)
        component_heights.append(float(np.mean(height[np.asarray(ys), np.asarray(xs)])))
    heights = np.asarray(component_heights, dtype=float)
    pixel_count = float(height.size)
    return {
        "coverage_fraction": coverage,
        "connected_component_count": float(len(components)),
        "component_density": float(len(components) / pixel_count),
        "mean_component_area": float(np.mean(areas)) if areas.size else 0.0,
        "median_component_area": float(np.median(areas)) if areas.size else 0.0,
        "max_component_area": float(np.max(areas)) if areas.size else 0.0,
        "mean_component_height": float(np.mean(heights)) if heights.size else 0.0,
        "median_component_height": float(np.median(heights)) if heights.size else 0.0,
    }


def extract_descriptors(height: np.ndarray) -> dict[str, float]:
    stats = height_statistics(height)
    descriptors: dict[str, float] = {}
    descriptors.update(stats)
    descriptors.update(roughness_descriptors(height, stats["mean_height"], stats["std_height"]))
    descriptors.update(gradient_descriptors(height))
    descriptors.update(frequency_descriptors(height))
    descriptors.update(autocorrelation_descriptors(height))
    descriptors.update(segmentation_descriptors(height, stats["median_height"], stats["std_height"]))
    return descriptors


def output_stem(row: dict[str, str], afm_path: Path) -> str:
    sample_id = row.get("sample_id") or afm_path.parent.parent.name
    base = afm_path.name.removesuffix("_plane_corrected.npy")
    return f"{sample_id}_{base}"


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


def write_descriptor_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ID_COLUMNS + DESCRIPTOR_COLUMNS + ["warnings"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in writer.fieldnames})


def write_descriptor_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values_by_column: dict[str, np.ndarray] = {}
    for column in DESCRIPTOR_COLUMNS:
        values = np.asarray([float(row[column]) for row in rows], dtype=float)
        values_by_column[column] = values
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["descriptor", "count", "mean", "std", "min", "p25", "median", "p75", "max"],
        )
        writer.writeheader()
        for column, values in values_by_column.items():
            writer.writerow(
                {
                    "descriptor": column,
                    "count": values.size,
                    "mean": format_value(float(np.mean(values))),
                    "std": format_value(float(np.std(values))),
                    "min": format_value(float(np.min(values))),
                    "p25": format_value(float(np.percentile(values, 25))),
                    "median": format_value(float(np.median(values))),
                    "p75": format_value(float(np.percentile(values, 75))),
                    "max": format_value(float(np.max(values))),
                }
            )


def write_correlation_heatmap(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray([[float(row[column]) for column in DESCRIPTOR_COLUMNS] for row in rows], dtype=float)
    if matrix.shape[0] < 2:
        corr = np.eye(len(DESCRIPTOR_COLUMNS))
    else:
        corr = np.corrcoef(matrix, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 10), dpi=160)
    image = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(np.arange(len(DESCRIPTOR_COLUMNS)))
    ax.set_yticks(np.arange(len(DESCRIPTOR_COLUMNS)))
    ax.set_xticklabels(DESCRIPTOR_COLUMNS, rotation=90, fontsize=5)
    ax.set_yticklabels(DESCRIPTOR_COLUMNS, fontsize=5)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_readme(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# AFM Descriptor Reconstruction Reports\n\n"
        "These reports summarize compact descriptors extracted from plane-corrected "
        "ZSensor AFM height maps. The input arrays are already plane-corrected; no "
        "additional fitted-plane subtraction is applied during descriptor extraction.\n\n"
        "This is a small-data descriptor baseline. The goal is to test how much "
        "morphology information can be retained in compact physical features before "
        "moving to heavier image reconstruction models.\n\n"
        "The descriptor table contains height statistics, roughness metrics, slope "
        "features, simple frequency features, spatial correlation estimates, and "
        "threshold-mask component measurements. Network input arrays are separately "
        "robust-clipped and normalized to [-1, 1].\n",
        encoding="utf-8",
    )


def process_manifest(
    manifest_rows: list[dict[str, str]],
    output_dir: Path,
    network_input_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    descriptor_rows: list[dict[str, Any]] = []
    all_warnings: list[str] = []

    for manifest_row in manifest_rows:
        afm_path = resolve_existing_path(Path(manifest_row["afm_path"]))
        stem = output_stem(manifest_row, afm_path)
        descriptor_json_path = output_dir / f"{stem}_descriptors.json"
        network_input_path = network_input_dir / f"{stem}_network_input.npy"

        height, warnings = load_height(afm_path)
        if height is None:
            message = f"row {manifest_row.get('row_id', '')}: {display_path(afm_path)}: {'; '.join(warnings)}"
            all_warnings.append(message)
            continue

        low, high, network_warning = save_network_input(height, network_input_path)
        if network_warning:
            warnings.append(network_warning)

        descriptors = extract_descriptors(height)
        row: dict[str, Any] = {
            "row_id": manifest_row.get("row_id", ""),
            "sample_id": manifest_row.get("sample_id", ""),
            "afm_path": display_path(afm_path),
            "descriptor_json_path": display_path(descriptor_json_path),
            "network_input_path": display_path(network_input_path),
            "array_shape": f"{height.shape[0]}x{height.shape[1]}",
            "warnings": "; ".join(warnings),
        }
        row.update(descriptors)
        descriptor_rows.append(row)

        payload = {
            "row_id": row["row_id"],
            "sample_id": row["sample_id"],
            "afm_path": row["afm_path"],
            "array_shape": row["array_shape"],
            "network_input_path": row["network_input_path"],
            "network_input_normalization": {
                "method": "clip to 1st/99th percentile then scale to [-1, 1]",
                "clip_low": low,
                "clip_high": high,
            },
            "descriptors": descriptors,
            "warnings": warnings,
        }
        write_descriptor_json(descriptor_json_path, payload)

        for warning in warnings:
            all_warnings.append(f"row {row['row_id']}: {warning}")

    return descriptor_rows, all_warnings


def print_run_summary(
    rows: list[dict[str, Any]],
    warnings: list[str],
    output_csv: Path,
    output_dir: Path,
    network_input_dir: Path,
    report_dir: Path,
) -> None:
    print("AFM descriptor extraction summary")
    print(f"  descriptor rows: {len(rows)}")
    print(f"  descriptor columns: {', '.join(DESCRIPTOR_COLUMNS)}")
    print(f"  warnings: {len(warnings)}")
    for warning in warnings[:20]:
        print(f"    - {warning}")
    if len(warnings) > 20:
        print(f"    - ... {len(warnings) - 20} more")
    print(f"  descriptor CSV: {display_path(output_csv)}")
    print(f"  descriptor JSON dir: {display_path(output_dir)}")
    print(f"  network inputs dir: {display_path(network_input_dir)}")
    print(f"  reports dir: {display_path(report_dir)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract descriptors from manifest-listed plane-corrected AFM height maps."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--network_input_dir",
        type=Path,
        default=None,
        help="Directory for robust-clipped [-1, 1] network input arrays.",
    )
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = resolve_existing_path(args.manifest)
    output_csv = args.output_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    network_input_dir = (
        args.network_input_dir.expanduser().resolve()
        if args.network_input_dir is not None
        else output_csv.parent / "network_inputs"
    )
    report_dir = args.report_dir.expanduser().resolve()

    if not manifest_path.is_file():
        raise SystemExit(f"Manifest does not exist: {manifest_path}")

    manifest_rows = read_manifest(manifest_path)
    descriptor_rows, warnings = process_manifest(manifest_rows, output_dir, network_input_dir)
    if not descriptor_rows:
        raise SystemExit("No descriptors were extracted.")

    write_descriptor_csv(output_csv, descriptor_rows)
    write_summary_csv(report_dir / "descriptor_summary.csv", descriptor_rows)
    write_correlation_heatmap(report_dir / "descriptor_correlation_heatmap.png", descriptor_rows)
    write_readme(report_dir / "README.md")
    print_run_summary(descriptor_rows, warnings, output_csv, output_dir, network_input_dir, report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
