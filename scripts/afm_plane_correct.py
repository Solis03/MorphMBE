#!/usr/bin/env python3
"""Subtract fitted first-order planes from processed AFM height maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


MIN_VALID_PIXELS = 3
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = REPO_ROOT / "data" / "processed_afm"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "plane_corrected_afm"


def normalized_image_coordinates(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return x, y coordinate grids normalized to [-1, 1]."""
    rows, cols = shape
    x_values = np.linspace(-1.0, 1.0, cols)
    y_values = np.linspace(-1.0, 1.0, rows)
    return np.meshgrid(x_values, y_values)


def fit_plane(height: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Fit z = a*x + b*y + c to finite pixels in a 2D height map."""
    array = np.asarray(height, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D height map, got shape {array.shape}.")

    finite_mask = np.isfinite(array)
    valid_count = int(np.count_nonzero(finite_mask))
    if valid_count < MIN_VALID_PIXELS:
        raise ValueError(f"Need at least {MIN_VALID_PIXELS} finite pixels, found {valid_count}.")

    x_grid, y_grid = normalized_image_coordinates(array.shape)
    design = np.column_stack(
        [
            x_grid[finite_mask],
            y_grid[finite_mask],
            np.ones(valid_count),
        ]
    )
    values = array[finite_mask]

    coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    if rank < 3:
        raise ValueError("Valid pixels do not span enough coordinates to fit a plane.")

    a, b, c = (float(value) for value in coefficients)
    plane = a * x_grid + b * y_grid + c
    return plane, {"a": a, "b": b, "c": c}


def subtract_fitted_plane(height: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return height minus fitted plane, the fitted plane, and coefficients."""
    array = np.asarray(height, dtype=float)
    plane, coefficients = fit_plane(array)
    corrected = array - plane
    return corrected, plane, coefficients


def save_render(height: np.ndarray, output_path: Path, title: str) -> None:
    """Save a grayscale PNG render with a labeled colorbar."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    array = np.asarray(height, dtype=float)
    masked = np.ma.masked_invalid(array)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    image = ax.imshow(masked, cmap="gray", origin="upper")
    ax.set_title(title)
    ax.set_axis_off()
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("height after plane correction (nm)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected object metadata in {metadata_path}.")
    return metadata


def write_metadata(metadata: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def display_path(path: Path) -> str:
    resolved_path = path.resolve()
    for root in (Path.cwd().resolve(), REPO_ROOT):
        try:
            return str(resolved_path.relative_to(root))
        except ValueError:
            continue
    return str(path)


def output_paths(
    input_height_path: Path,
    input_root: Path,
    output_root: Path,
) -> tuple[str, Path, Path, Path, Path, Path]:
    relative_dir = input_height_path.parent.relative_to(input_root)
    output_dir = output_root / relative_dir
    if not input_height_path.name.endswith("_height.npy"):
        raise ValueError(f"Unexpected height filename: {input_height_path}")

    base_name = input_height_path.name.removesuffix("_height.npy")
    metadata_path = input_height_path.with_name(f"{base_name}_metadata.json")
    corrected_path = output_dir / f"{base_name}_plane_corrected.npy"
    render_path = output_dir / f"{base_name}_plane_corrected_render.png"
    plane_path = output_dir / f"{base_name}_fitted_plane.npy"
    corrected_metadata_path = output_dir / f"{base_name}_plane_corrected_metadata.json"
    return base_name, metadata_path, corrected_path, render_path, plane_path, corrected_metadata_path


def process_height_file(input_height_path: Path, input_root: Path, output_root: Path, overwrite: bool) -> str:
    (
        base_name,
        metadata_path,
        corrected_path,
        render_path,
        plane_path,
        corrected_metadata_path,
    ) = output_paths(input_height_path, input_root, output_root)

    if corrected_path.exists() and not overwrite:
        return f"skipped existing: {display_path(corrected_path)}"

    height = np.load(input_height_path)
    corrected, fitted_plane, coefficients = subtract_fitted_plane(height)

    corrected_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(corrected_path, corrected)
    np.save(plane_path, fitted_plane)
    save_render(corrected, render_path, f"{base_name} plane corrected")

    metadata = load_metadata(metadata_path)
    metadata["plane_correction"] = {
        "enabled": True,
        "model": "z = a*x + b*y + c",
        "method": "least_squares",
        "coefficients": coefficients,
        "input_height_file": display_path(input_height_path),
        "output_height_file": display_path(corrected_path),
    }
    write_metadata(metadata, corrected_metadata_path)

    return f"wrote: {display_path(corrected_path)}"


def find_height_files(input_root: Path, limit: int | None) -> list[Path]:
    height_files = sorted(input_root.rglob("*_height.npy"))
    if limit is not None:
        return height_files[:limit]
    return height_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Subtract fitted first-order planes from processed AFM height maps."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root containing processed AFM *_height.npy files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root for plane-corrected AFM outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing corrected outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of height maps to process.",
    )
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative.")
    if not input_root.exists():
        parser.error(f"Input root does not exist: {input_root}")

    height_files = find_height_files(input_root, args.limit)
    if not height_files:
        print(f"No *_height.npy files found under {display_path(input_root)}.")
        return 0

    num_written = 0
    num_skipped = 0
    num_failed = 0
    for height_path in height_files:
        try:
            message = process_height_file(height_path, input_root, output_root, args.overwrite)
        except ValueError as exc:
            num_failed += 1
            print(f"warning: skipped {display_path(height_path)}: {exc}")
            continue

        print(message)
        if message.startswith("wrote:"):
            num_written += 1
        else:
            num_skipped += 1

    print(
        "Plane correction complete: "
        f"{num_written} written, {num_skipped} skipped, {num_failed} failed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
