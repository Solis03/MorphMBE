#!/usr/bin/env python3
"""Extract descriptors and normalized network inputs for the large AFM dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT / "data" / "afm_descriptor_reconstruction_large" / "large_afm_manifest.csv"
)
DEFAULT_OUTPUT_CSV = DEFAULT_MANIFEST.with_name("large_afm_descriptors.csv")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "afm_descriptor_reconstruction_large" / "descriptors"
DEFAULT_NETWORK_INPUT_DIR = (
    REPO_ROOT / "data" / "afm_descriptor_reconstruction_large" / "network_inputs"
)
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "afm_descriptor_reconstruction_large"


BASE_DESCRIPTOR_COLUMNS = [
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
SCAN_DESCRIPTOR_COLUMNS = [
    "scan_size_x_um",
    "scan_size_y_um",
    "area_um2",
    "log_area_um2",
    "pixel_size_x_nm",
    "pixel_size_y_nm",
    "mean_pixel_size_nm",
    "aspect_ratio",
    "original_height_pixels",
    "original_width_pixels",
    "is_1um_scan",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resize_bilinear(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape == (size, size):
        return image.astype(np.float32, copy=False)
    rows, cols = image.shape
    y_new = np.linspace(0, rows - 1, size)
    x_new = np.linspace(0, cols - 1, size)
    resized_rows = np.vstack([np.interp(x_new, np.arange(cols), row) for row in image])
    return np.vstack([np.interp(y_new, np.arange(rows), resized_rows[:, col]) for col in range(size)]).T.astype(
        np.float32
    )


def load_height(path: Path) -> tuple[np.ndarray | None, list[str]]:
    warnings: list[str] = []
    try:
        height = np.asarray(np.load(path), dtype=float)
    except Exception as exc:  # noqa: BLE001
        return None, [f"failed to load: {exc}"]
    if height.ndim != 2:
        return None, [f"not 2D: shape={height.shape}"]
    finite = np.isfinite(height)
    if not np.any(finite):
        return None, ["no finite pixels"]
    if not np.all(finite):
        fill = float(np.median(height[finite]))
        height = height.copy()
        height[~finite] = fill
        warnings.append(f"filled {int(np.size(height) - np.count_nonzero(finite))} NaN/Inf pixels")
    return height, warnings


def save_network_input(height: np.ndarray, path: Path, image_size: int) -> tuple[float, float, str | None]:
    low, high = np.percentile(height, [1.0, 99.0])
    warning = None
    if high <= low or not np.isfinite(low + high):
        low, high = float(np.min(height)), float(np.max(height))
        warning = "normalization used min/max fallback"
    if high <= low:
        normalized = np.zeros_like(height, dtype=np.float32)
        warning = "constant image normalized to zeros"
    else:
        normalized = ((np.clip(height, low, high) - low) / (high - low) * 2.0 - 1.0).astype(np.float32)
    resized = resize_bilinear(normalized, image_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, resized)
    return float(low), float(high), warning


def height_stats(height: np.ndarray) -> dict[str, float]:
    p = np.percentile(height, [1, 5, 25, 75, 95, 99])
    mean = float(np.mean(height))
    median = float(np.median(height))
    std = float(np.std(height))
    centered = height - mean
    norm = centered / std if std > 0 else np.zeros_like(centered)
    return {
        "mean_height": mean,
        "median_height": median,
        "std_height": std,
        "min_height": float(np.min(height)),
        "max_height": float(np.max(height)),
        "peak_to_valley": float(np.max(height) - np.min(height)),
        "p01": float(p[0]),
        "p05": float(p[1]),
        "p25": float(p[2]),
        "p75": float(p[3]),
        "p95": float(p[4]),
        "p99": float(p[5]),
        "iqr": float(p[3] - p[2]),
        "Ra": float(np.mean(np.abs(centered))),
        "Rq": float(np.sqrt(np.mean(centered**2))),
        "Rsk": float(np.mean(norm**3)) if std > 0 else 0.0,
        "Rku": float(np.mean(norm**4)) if std > 0 else 0.0,
    }


def gradient_desc(height: np.ndarray, pixel_x_nm: float, pixel_y_nm: float) -> dict[str, float]:
    gy, gx = np.gradient(height, pixel_y_nm, pixel_x_nm)
    mag = np.hypot(gx, gy)
    angles = np.mod(np.arctan2(gy, gx), np.pi)
    hist, _ = np.histogram(angles.ravel(), bins=18, range=(0.0, np.pi), weights=mag.ravel())
    total = float(np.sum(hist))
    if total > 0:
        probs = hist / total
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)) / np.log2(18))
    else:
        entropy = 0.0
    return {
        "grad_mean": float(np.mean(mag)),
        "grad_std": float(np.std(mag)),
        "grad_p95": float(np.percentile(mag, 95)),
        "orientation_entropy": entropy,
    }


def frequency_desc(height: np.ndarray, pixel_x_nm: float, pixel_y_nm: float) -> dict[str, float]:
    centered = height - float(np.mean(height))
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    rows, cols = height.shape
    fy = np.fft.fftshift(np.fft.fftfreq(rows, d=pixel_y_nm / 1000.0))
    fx = np.fft.fftshift(np.fft.fftfreq(cols, d=pixel_x_nm / 1000.0))
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    nonzero = radius > 0
    if not np.any(nonzero):
        return {key: 0.0 for key in ["low_freq_power", "mid_freq_power", "high_freq_power", "high_low_power_ratio", "radial_psd_slope"]}
    max_r = float(np.max(radius[nonzero]))
    low_mask = (radius > 0) & (radius <= 0.2 * max_r)
    mid_mask = (radius > 0.2 * max_r) & (radius <= 0.5 * max_r)
    high_mask = radius > 0.5 * max_r
    low = float(np.mean(power[low_mask])) if np.any(low_mask) else 0.0
    mid = float(np.mean(power[mid_mask])) if np.any(mid_mask) else 0.0
    high = float(np.mean(power[high_mask])) if np.any(high_mask) else 0.0
    r = radius[nonzero].ravel()
    p = power[nonzero].ravel()
    bins = np.linspace(float(np.min(r)), float(np.max(r)), 32)
    centers, values = [], []
    for start, end in zip(bins[:-1], bins[1:]):
        mask = (r >= start) & (r < end)
        if np.any(mask):
            value = float(np.mean(p[mask]))
            if value > 0:
                centers.append((start + end) / 2.0)
                values.append(value)
    slope = float(np.polyfit(np.log(centers), np.log(values), 1)[0]) if len(centers) >= 3 else 0.0
    return {
        "low_freq_power": low,
        "mid_freq_power": mid,
        "high_freq_power": high,
        "high_low_power_ratio": high / low if low > 0 else 0.0,
        "radial_psd_slope": slope,
    }


def first_corr_length(profile: np.ndarray) -> float:
    if profile.size == 0 or profile[0] <= 0:
        return 0.0
    normalized = profile / profile[0]
    below = np.flatnonzero(normalized <= 1.0 / np.e)
    return float(below[0]) if below.size else float(profile.size - 1)


def autocorr_desc(height: np.ndarray, pixel_x_nm: float, pixel_y_nm: float) -> dict[str, float]:
    centered = height - float(np.mean(height))
    ac = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(centered)) ** 2).real)
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    lx = first_corr_length(ac[cy, cx:]) * pixel_x_nm
    ly = first_corr_length(ac[cy:, cx]) * pixel_y_nm
    smaller, larger = min(lx, ly), max(lx, ly)
    return {
        "autocorr_length_x": float(lx),
        "autocorr_length_y": float(ly),
        "anisotropy_ratio": float(larger / smaller) if smaller > 0 else 0.0,
    }


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    visited = np.zeros(mask.shape, dtype=bool)
    result: list[list[tuple[int, int]]] = []
    rows, cols = mask.shape
    for y in range(rows):
        for x in range(cols):
            if visited[y, x] or not mask[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            comp: list[tuple[int, int]] = []
            while queue:
                yy, xx = queue.popleft()
                comp.append((yy, xx))
                for ny, nx in ((yy - 1, xx), (yy + 1, xx), (yy, xx - 1), (yy, xx + 1)):
                    if 0 <= ny < rows and 0 <= nx < cols and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            result.append(comp)
    return result


def segment_desc(height: np.ndarray, median: float, std: float, pixel_area_nm2: float) -> dict[str, float]:
    mask = height > median + 0.5 * std
    comps = components(mask)
    areas = np.asarray([len(comp) * pixel_area_nm2 for comp in comps], dtype=float)
    comp_heights = []
    for comp in comps:
        ys, xs = zip(*comp)
        comp_heights.append(float(np.mean(height[np.asarray(ys), np.asarray(xs)])))
    heights = np.asarray(comp_heights, dtype=float)
    total_area = height.size * pixel_area_nm2
    return {
        "coverage_fraction": float(np.mean(mask)),
        "connected_component_count": float(len(comps)),
        "component_density": float(len(comps) / total_area) if total_area > 0 else 0.0,
        "mean_component_area": float(np.mean(areas)) if areas.size else 0.0,
        "median_component_area": float(np.median(areas)) if areas.size else 0.0,
        "max_component_area": float(np.max(areas)) if areas.size else 0.0,
        "mean_component_height": float(np.mean(heights)) if heights.size else 0.0,
        "median_component_height": float(np.median(heights)) if heights.size else 0.0,
    }


def output_stem(row: dict[str, str]) -> str:
    afm_path = Path(row["afm_path"])
    return f"{row['sample_id']}_{afm_path.name.removesuffix('_plane_corrected.npy')}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "row_id",
        "sample_id",
        "afm_path",
        "network_input_path",
        "descriptor_json_path",
    ] + BASE_DESCRIPTOR_COLUMNS + SCAN_DESCRIPTOR_COLUMNS + ["warnings"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--network_input_dir", type=Path, default=DEFAULT_NETWORK_INPUT_DIR)
    parser.add_argument("--report_dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--image_size", type=int, default=64)
    args = parser.parse_args()
    manifest = resolve_existing_path(args.manifest)
    rows = read_csv(manifest)
    output_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for row in rows:
        afm_path = resolve_existing_path(Path(row["afm_path"]))
        height, row_warnings = load_height(afm_path)
        if height is None:
            warnings.extend([f"row {row['row_id']}: {warning}" for warning in row_warnings])
            continue
        stem = output_stem(row)
        network_path = args.network_input_dir.expanduser().resolve() / f"{stem}_network_input.npy"
        descriptor_json_path = args.output_dir.expanduser().resolve() / f"{stem}_descriptors.json"
        _, _, norm_warning = save_network_input(height, network_path, args.image_size)
        if norm_warning:
            row_warnings.append(norm_warning)
        scan_x = float(row["scan_size_x_um"])
        scan_y = float(row["scan_size_y_um"])
        px_x = float(row["pixel_size_x_nm"])
        px_y = float(row["pixel_size_y_nm"])
        desc = height_stats(height)
        desc.update(gradient_desc(height, px_x, px_y))
        desc.update(frequency_desc(height, px_x, px_y))
        desc.update(autocorr_desc(height, px_x, px_y))
        desc.update(segment_desc(height, desc["median_height"], desc["std_height"], px_x * px_y))
        scan_desc = {
            "scan_size_x_um": scan_x,
            "scan_size_y_um": scan_y,
            "area_um2": float(row["area_um2"]),
            "log_area_um2": float(np.log(float(row["area_um2"]))),
            "pixel_size_x_nm": px_x,
            "pixel_size_y_nm": px_y,
            "mean_pixel_size_nm": (px_x + px_y) / 2.0,
            "aspect_ratio": float(row["aspect_ratio"]),
            "original_height_pixels": float(row["original_height_pixels"]),
            "original_width_pixels": float(row["original_width_pixels"]),
            "is_1um_scan": 1.0 if row["is_1um_scan"].lower() == "true" else 0.0,
        }
        out = {
            "row_id": row["row_id"],
            "sample_id": row["sample_id"],
            "afm_path": row["afm_path"],
            "network_input_path": display_path(network_path),
            "descriptor_json_path": display_path(descriptor_json_path),
            **desc,
            **scan_desc,
            "warnings": "; ".join(row_warnings),
        }
        descriptor_json_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor_json_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        output_rows.append(out)
        warnings.extend([f"row {row['row_id']}: {warning}" for warning in row_warnings])

    write_csv(args.output_csv.expanduser().resolve(), output_rows)
    report_dir = args.report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "descriptor_extraction_notes.md").write_text(
        "# Large AFM Descriptor Extraction Notes\n\n"
        "Descriptors were extracted from already plane-corrected ZSensor height maps. "
        "No fitted-plane subtraction was applied. Gradient, frequency, autocorrelation, "
        "and component-area descriptors use physical pixel sizes from metadata when available. "
        "Images were robust-clipped and resized to 64x64 network inputs, while scan size "
        "and pixel scale remain explicit descriptor features.\n",
        encoding="utf-8",
    )
    print("Large AFM descriptor extraction summary")
    print(f"  descriptor rows: {len(output_rows)}")
    print(f"  warnings: {len(warnings)}")
    for warning in warnings[:20]:
        print(f"    - {warning}")
    print(f"  wrote: {display_path(args.output_csv.expanduser().resolve())}")
    print(f"  network inputs: {display_path(args.network_input_dir.expanduser().resolve())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
