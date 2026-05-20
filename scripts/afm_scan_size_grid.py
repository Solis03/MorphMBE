#!/usr/bin/env python3
"""Render grid overview figures for AFM scans with a selected scan size."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "data" / "processed_afm"
DEFAULT_PLANE_CORRECTED_ROOT = REPO_ROOT / "data" / "plane_corrected_afm"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "figures" / "afm_scan_size_grids"


@dataclass(frozen=True)
class AFMRenderItem:
    sample_id: str
    afm_file_id: str
    array_path: Path
    metadata_path: Path
    scan_size_um: tuple[float, float]


def display_path(path: Path) -> str:
    resolved_path = path.resolve()
    for root in (Path.cwd().resolve(), REPO_ROOT):
        try:
            return str(resolved_path.relative_to(root))
        except ValueError:
            continue
    return str(path)


def load_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected object metadata in {path}.")
    return metadata


def scan_size_matches(
    metadata: dict[str, Any],
    target_size_um: float,
    tolerance: float,
) -> tuple[float, float] | None:
    scan_size = metadata.get("scan_size_um")
    if not isinstance(scan_size, list | tuple) or len(scan_size) != 2:
        return None
    try:
        size_x = float(scan_size[0])
        size_y = float(scan_size[1])
    except (TypeError, ValueError):
        return None

    if abs(size_x - target_size_um) <= tolerance and abs(size_y - target_size_um) <= tolerance:
        return size_x, size_y
    return None


def collect_items(
    root: Path,
    metadata_suffix: str,
    array_suffix: str,
    target_size_um: float,
    tolerance: float,
    limit: int | None,
) -> list[AFMRenderItem]:
    items: list[AFMRenderItem] = []
    seen_sample_ids: set[str] = set()
    for metadata_path in sorted(root.rglob(f"*{metadata_suffix}")):
        metadata = load_metadata(metadata_path)
        scan_size_um = scan_size_matches(metadata, target_size_um, tolerance)
        if scan_size_um is None:
            continue

        base_name = metadata_path.name.removesuffix(metadata_suffix)
        sample_id = str(metadata.get("sample_id") or metadata_path.parent.parent.name)
        if sample_id in seen_sample_ids:
            continue

        array_path = metadata_path.with_name(f"{base_name}{array_suffix}")
        if not array_path.exists():
            print(f"warning: missing array for {display_path(metadata_path)}: {array_path.name}")
            continue

        items.append(
            AFMRenderItem(
                sample_id=sample_id,
                afm_file_id=str(metadata.get("afm_file_id") or base_name),
                array_path=array_path,
                metadata_path=metadata_path,
                scan_size_um=scan_size_um,
            )
        )
        seen_sample_ids.add(sample_id)
        if limit is not None and len(items) >= limit:
            break
    return items


def robust_limits(array: np.ndarray) -> tuple[float | None, float | None]:
    finite_values = array[np.isfinite(array)]
    if finite_values.size == 0:
        return None, None
    low, high = np.nanpercentile(finite_values, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        return None, None
    return float(low), float(high)


def save_grid(
    items: list[AFMRenderItem],
    output_path: Path,
    title: str,
    columns: int,
    dpi: int,
) -> None:
    if not items:
        print(f"warning: no matching scans for {title}; no figure written.")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = max(1, columns)
    rows = math.ceil(len(items) / columns)
    figure_width = columns * 3.0
    figure_height = rows * 3.1 + 0.6

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(rows, columns, figsize=(figure_width, figure_height), dpi=dpi)
    axes_array = np.atleast_1d(axes).reshape(rows, columns)
    fig.suptitle(title, fontsize=14)

    for index, (item, ax) in enumerate(zip(items, axes_array.flat), start=1):
        height = np.asarray(np.load(item.array_path), dtype=float)
        masked_height = np.ma.masked_invalid(height)
        vmin, vmax = robust_limits(height)
        image = ax.imshow(masked_height, cmap="gray", origin="upper", vmin=vmin, vmax=vmax)
        ax.set_axis_off()
        ax.set_title(
            f"{index:02d} | sample {item.sample_id}\n{item.afm_file_id}",
            fontsize=7,
            pad=3,
        )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
        colorbar.set_label("height (nm)", fontsize=6)
        colorbar.ax.tick_params(labelsize=5, length=2)

    for ax in axes_array.flat[len(items) :]:
        ax.set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {display_path(output_path)} ({len(items)} subplots)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create large grid figures for AFM scans matching a scan size."
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Root containing processed AFM outputs.",
    )
    parser.add_argument(
        "--plane-corrected-root",
        type=Path,
        default=DEFAULT_PLANE_CORRECTED_ROOT,
        help="Root containing plane-corrected AFM outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for overview PNG figures.",
    )
    parser.add_argument(
        "--scan-size",
        type=float,
        default=1.0,
        help="Square scan size in micrometers to include.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for matching scan_size_um.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=6,
        help="Number of subplot columns in each overview figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output PNG resolution.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit per overview figure for quick testing.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.scan_size <= 0:
        raise SystemExit("--scan-size must be positive.")
    if args.tolerance < 0:
        raise SystemExit("--tolerance must be non-negative.")
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be non-negative.")

    processed_root = args.processed_root.expanduser().resolve()
    plane_corrected_root = args.plane_corrected_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not processed_root.is_dir():
        raise SystemExit(f"Processed AFM root does not exist: {processed_root}")
    if not plane_corrected_root.is_dir():
        raise SystemExit(f"Plane-corrected AFM root does not exist: {plane_corrected_root}")

    processed_items = collect_items(
        processed_root,
        metadata_suffix="_metadata.json",
        array_suffix="_height.npy",
        target_size_um=args.scan_size,
        tolerance=args.tolerance,
        limit=args.limit,
    )
    plane_corrected_items = collect_items(
        plane_corrected_root,
        metadata_suffix="_plane_corrected_metadata.json",
        array_suffix="_plane_corrected.npy",
        target_size_um=args.scan_size,
        tolerance=args.tolerance,
        limit=args.limit,
    )

    size_label = f"{args.scan_size:g}um"
    save_grid(
        processed_items,
        output_dir / f"processed_afm_scan_size_{size_label}_grid.png",
        f"Processed AFM scans, scan_size_um = [{args.scan_size:g}, {args.scan_size:g}]",
        columns=args.columns,
        dpi=args.dpi,
    )
    save_grid(
        plane_corrected_items,
        output_dir / f"plane_corrected_afm_scan_size_{size_label}_grid.png",
        f"Plane-corrected AFM scans, scan_size_um = [{args.scan_size:g}, {args.scan_size:g}]",
        columns=args.columns,
        dpi=args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
