#!/usr/bin/env python3
"""
Inspect one AFM raw file, identify useful morphology channels, and export
ZSensor/Height as numpy height map + rendered image + metadata.

Example:
    python scripts/inspect_afm_raw.py \
      --input pair/6022/AFM/N6022\\ Ctr.000 \
      --output_dir data/processed_afm

Notes:
    - Direct Bruker/Nanoscope parsing is attempted with pySPM when available.
    - Gwyddion .gwy files are attempted with gwyfile when available.
    - Exported TIFF/TXT files can be read by the fallback helper.
    - If direct parsing fails, the script still inspects the Nanoscope header
      where possible and prints clear next steps instead of crashing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


HEIGHT_CHANNEL_PREFERENCES = ("ZSensor", "Height")
HEIGHT_LIKE_NAMES = (
    "height",
    "zsensor",
    "z sensor",
    "topography",
    "sensor",
)


@dataclass
class ChannelInfo:
    name: str
    shape: list[int] | None = None
    unit: str = "unknown"
    scan_size_um: list[float] | None = None
    stats_original: dict[str, float] | None = None
    stats_nm: dict[str, float] | None = None
    description: str | None = None
    source: str | None = None


@dataclass
class AFMReadResult:
    raw_format: str = "unknown"
    channels: dict[str, ChannelInfo] = field(default_factory=dict)
    arrays: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def import_required_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Missing required dependency: numpy. Install with: "
            "python3 -m pip install numpy matplotlib"
        ) from exc
    return np


def make_safe_id(name: str) -> str:
    """Create a filesystem-safe id from a file or folder name."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def parse_sample_id(path: Path) -> str:
    """Create a filesystem-safe sample id from the raw filename."""
    return make_safe_id(path.name)


def display_path(path: Path) -> str:
    """Return a portable path string for reports and metadata."""
    if not path.is_absolute():
        return str(path)
    return os.path.relpath(path, Path.cwd())


def infer_scan_size_from_filename(name: str) -> list[float] | None:
    """Infer scan size from filename tokens such as 1um, 5um, 500nm, 0.5um."""
    match = re.search(r"(?i)(\d+(?:\.\d+)?)\s*(um|µm|μm|nm)", name)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "nm":
        value /= 1000.0
    return [value, value]


def normalize_unit(unit: str | None) -> str:
    if not unit:
        return "unknown"
    unit = unit.strip().replace("μ", "u").replace("µ", "u")
    if unit in {"~m", "um", "uM"}:
        return "um"
    return unit


def parse_scan_size_text(text: str) -> list[float] | None:
    """Parse Nanoscope scan size text into [x_um, y_um]."""
    numbers = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]
    if not numbers:
        return None

    unit_match = re.search(r"(?i)(nm|~m|um|µm|μm|m)\s*$", text.strip())
    unit = normalize_unit(unit_match.group(1) if unit_match else "unknown")

    if len(numbers) == 1:
        x_value = y_value = numbers[0]
    else:
        x_value, y_value = numbers[:2]

    if unit == "nm":
        return [x_value / 1000.0, y_value / 1000.0]
    if unit == "um":
        return [x_value, y_value]
    if unit == "m":
        return [x_value * 1_000_000.0, y_value * 1_000_000.0]
    return None


def numeric_stats(array: Any) -> dict[str, float]:
    np = import_required_numpy()
    values = np.asarray(array, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": math.nan, "max": math.nan, "mean": math.nan, "std": math.nan}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def channel_is_height_like(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in HEIGHT_LIKE_NAMES)


def convert_height_to_nm(array: Any, unit: str | None) -> tuple[Any, str, str]:
    """Convert height map to nm when units are known or clearly meter-scale."""
    np = import_required_numpy()
    unit_norm = normalize_unit(unit)
    values = np.asarray(array, dtype=float)

    if unit_norm == "nm":
        return values, "nm", "unit already nm"
    if unit_norm == "m":
        return values * 1_000_000_000.0, "nm", "converted m to nm"
    if unit_norm in {"um", "micron", "micrometer", "micrometre"}:
        return values * 1000.0, "nm", "converted um to nm"

    finite = values[np.isfinite(values)]
    if finite.size:
        max_abs = float(np.max(np.abs(finite)))
        # AFM height data stored in meters is typically around 1e-12 to 1e-6.
        # Treat that range as obvious SI meters when the metadata unit is absent.
        if 0 < max_abs < 1e-5:
            return values * 1_000_000_000.0, "nm", (
                "unit unknown; values appear meter-scale, converted to nm"
            )

    return values, "unknown", "unit unknown; exported values were not converted"


def inspect_nanoscope_header(path: Path) -> AFMReadResult:
    """Extract channel metadata from a Bruker/Nanoscope text header."""
    result = AFMReadResult(raw_format="Nanoscope/Bruker header")
    text = path.read_bytes().decode("latin-1", errors="ignore")
    lines = text.splitlines()

    current: dict[str, Any] | None = None
    global_scan_size_um: list[float] | None = None
    sensitivity_units: dict[str, str] = {}

    def finish_current() -> None:
        nonlocal current
        if not current or "name" not in current:
            current = None
            return

        x = current.get("valid_x") or current.get("samps")
        y = current.get("valid_y") or current.get("samps")
        shape = [int(y), int(x)] if x and y else None
        name = str(current["name"])
        result.channels[name] = ChannelInfo(
            name=name,
            shape=shape,
            unit=current.get("unit", "unknown"),
            scan_size_um=current.get("scan_size_um") or global_scan_size_um,
            description=current.get("description"),
            source="nanoscope_header",
        )
        current = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith("\\"):
            continue

        sens_match = re.match(
            r"\\@Sens\.\s*([^:]+):\s*\S+\s+[-+\d.Ee]+\s+([A-Za-zµμ~()/*]+)",
            line,
        )
        if sens_match:
            sensitivity_name = "Sens. " + sens_match.group(1).strip()
            sensitivity_unit = normalize_unit(sens_match.group(2).split("/")[0])
            sensitivity_units[sensitivity_name] = sensitivity_unit
            continue

        if line.startswith("\\Data offset:"):
            finish_current()
            current = {}
            continue

        scan_match = re.match(r"\\Scan Size:\s*(.+)$", line)
        if scan_match:
            scan_size = parse_scan_size_text(scan_match.group(1))
            if current is not None:
                current["scan_size_um"] = scan_size
            elif scan_size:
                global_scan_size_um = scan_size
            continue

        if current is None:
            continue

        for key, field_name in (
            ("\\Samps/line:", "samps"),
            ("\\Valid data len X:", "valid_x"),
            ("\\Valid data len Y:", "valid_y"),
        ):
            if line.startswith(key):
                value_match = re.search(r"\d+", line)
                if value_match:
                    current[field_name] = int(value_match.group(0))

        image_match = re.match(r'\\@\d+:Image Data:\s*S\s*\[([^\]]*)\]\s*"([^"]*)"', line)
        if image_match:
            bracket_name = image_match.group(1).strip()
            quoted_name = image_match.group(2).strip()
            current["name"] = bracket_name or quoted_name
            current["description"] = quoted_name or bracket_name
            continue

        zscale_match = re.match(
            r"\\@\d+:Z scale:.*\[([^\]]+)\].*\)\s*[-+\d.Ee]+\s*([A-Za-zµμ~]+)?",
            line,
        )
        if zscale_match:
            sensitivity_name = zscale_match.group(1).strip()
            displayed_unit = normalize_unit(zscale_match.group(2) or "unknown")
            current["unit"] = sensitivity_units.get(sensitivity_name, displayed_unit)

    finish_current()
    return result


def read_with_pyspm(path: Path) -> AFMReadResult:
    """Read Bruker/Nanoscope files with pySPM, if installed."""
    try:
        import pySPM
    except ImportError as exc:
        raise RuntimeError(
            "pySPM is not installed. Suggested install: python3 -m pip install pySPM"
        ) from exc

    np = import_required_numpy()
    result = AFMReadResult(raw_format="Nanoscope/Bruker via pySPM")

    bruker = pySPM.Bruker(str(path))
    channel_specs: list[tuple[str, str]] = []
    for layer in getattr(bruker, "layers", []):
        for key in (b"@2:Image Data", b"@3:Image Data"):
            raw_values = layer.get(key)
            if not raw_values:
                continue
            raw_value = raw_values[0].decode("latin1", errors="ignore")
            match = re.match(r'S\s*\[([^\]]*)\]\s*"([^"]*)"', raw_value.strip())
            if match:
                bracket_name = match.group(1).strip()
                quoted_name = match.group(2).strip()
                channel_name = bracket_name or quoted_name
                pyspm_name = quoted_name or bracket_name
                channel_specs.append((channel_name, pyspm_name))

    if not channel_specs:
        raise RuntimeError("pySPM did not expose any image channels.")

    for name, pyspm_name in channel_specs:
        try:
            image = bruker.get_channel(pyspm_name)
        except Exception as exc:  # Keep inspecting other channels.
            result.notes.append(
                f"Could not read channel {name!r} ({pyspm_name!r}) with pySPM: {exc}"
            )
            result.channels[name] = ChannelInfo(name=name, source="pySPM")
            continue

        pixels = getattr(image, "pixels", None)
        if pixels is None:
            pixels = getattr(image, "data", None)
        if pixels is None:
            pixels = image

        array = np.asarray(pixels, dtype=float)
        unit = normalize_unit(getattr(image, "unit", None) or getattr(image, "units", None))

        scan_size_um = None
        size = getattr(image, "size", None)
        if isinstance(size, dict):
            x_size = size.get("real", {}).get("x") or size.get("x")
            y_size = size.get("real", {}).get("y") or size.get("y")
            if x_size and y_size:
                scan_size_um = [float(x_size), float(y_size)]

        result.arrays[name] = array
        result.channels[name] = ChannelInfo(
            name=name,
            shape=list(array.shape),
            unit=unit,
            scan_size_um=scan_size_um,
            stats_original=numeric_stats(array) if channel_is_height_like(name) else None,
            description=pyspm_name,
            source="pySPM",
        )

    return result


def walk_gwy_tree(obj: Any, prefix: str = ""):
    """Yield leaf-like objects from a gwyfile object without assuming one schema."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from walk_gwy_tree(value, f"{prefix}/{key}")
    elif hasattr(obj, "items"):
        for key, value in obj.items():
            yield from walk_gwy_tree(value, f"{prefix}/{key}")
    else:
        yield prefix, obj


def read_with_gwyfile(path: Path) -> AFMReadResult:
    """Read a Gwyddion .gwy export with gwyfile, if installed."""
    try:
        import gwyfile
    except ImportError as exc:
        raise RuntimeError(
            "gwyfile is not installed. Suggested install: python3 -m pip install gwyfile"
        ) from exc

    np = import_required_numpy()
    result = AFMReadResult(raw_format="Gwyddion .gwy via gwyfile")
    gwy_object = gwyfile.load(str(path))

    titles: dict[str, str] = {}
    data_fields: list[tuple[str, Any]] = []
    for key, value in walk_gwy_tree(gwy_object):
        if key.endswith("/title") and isinstance(value, str):
            titles[key.rsplit("/", 2)[0]] = value
        elif hasattr(value, "data") and hasattr(value, "xres") and hasattr(value, "yres"):
            data_fields.append((key, value))

    for key, field_obj in data_fields:
        title_prefix = key.rsplit("/", 1)[0]
        name = titles.get(title_prefix) or Path(key).name or f"channel_{len(result.channels)}"
        array = np.asarray(field_obj.data, dtype=float).reshape(
            int(field_obj.yres), int(field_obj.xres)
        )
        unit = normalize_unit(str(getattr(field_obj, "si_unit_z", "unknown")))
        scan_size_um = None
        if getattr(field_obj, "xreal", None) and getattr(field_obj, "yreal", None):
            scan_size_um = [float(field_obj.xreal) * 1_000_000, float(field_obj.yreal) * 1_000_000]

        result.arrays[name] = array
        result.channels[name] = ChannelInfo(
            name=name,
            shape=list(array.shape),
            unit=unit,
            scan_size_um=scan_size_um,
            stats_original=numeric_stats(array) if channel_is_height_like(name) else None,
            source="gwyfile",
        )

    return result


def read_exported_tiff_or_txt(path: Path) -> AFMReadResult:
    """Fallback reader for height maps exported from Gwyddion as TIFF or TXT."""
    np = import_required_numpy()
    suffix = path.suffix.lower()
    result = AFMReadResult(raw_format=f"exported {suffix}")

    if suffix in {".txt", ".csv", ".dat"}:
        delimiter = "," if suffix == ".csv" else None
        array = np.loadtxt(path, delimiter=delimiter)
    elif suffix in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError as exc:
            raise RuntimeError(
                "tifffile is not installed. Suggested install: python3 -m pip install tifffile"
            ) from exc
        array = tifffile.imread(path)
    else:
        raise RuntimeError(
            "Fallback reader only supports .tif, .tiff, .txt, .csv, and .dat exports."
        )

    if array.ndim > 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise RuntimeError(f"Expected a 2D exported height map, got shape {array.shape}.")

    result.arrays["Height"] = np.asarray(array, dtype=float)
    result.channels["Height"] = ChannelInfo(
        name="Height",
        shape=list(array.shape),
        unit="unknown",
        stats_original=numeric_stats(array),
        source="exported_tiff_or_txt",
    )
    result.notes.append("Read exported file; physical scan size and height unit may be absent.")
    return result


def merge_header_metadata(primary: AFMReadResult, header: AFMReadResult) -> AFMReadResult:
    """Fill missing channel metadata from header inspection."""
    if not header.channels:
        return primary
    for name, info in header.channels.items():
        if name not in primary.channels:
            primary.channels[name] = info
            continue
        target = primary.channels[name]
        target.shape = target.shape or info.shape
        target.scan_size_um = target.scan_size_um or info.scan_size_um
        if target.unit == "unknown":
            target.unit = info.unit
    return primary


def inspect_channels(path: Path) -> AFMReadResult:
    """Try direct readers, then collect header-level channel metadata."""
    header_result = inspect_nanoscope_header(path)
    errors: list[str] = []

    if path.suffix.lower() == ".gwy":
        readers = (read_with_gwyfile,)
    elif path.suffix.lower() in {".tif", ".tiff", ".txt", ".csv", ".dat"}:
        readers = (read_exported_tiff_or_txt,)
    else:
        readers = (read_with_pyspm,)

    for reader in readers:
        try:
            parsed = reader(path)
            parsed = merge_header_metadata(parsed, header_result)
            if parsed.arrays:
                parsed.notes.append(f"Array data read with {reader.__name__}.")
                return parsed
            errors.append(f"{reader.__name__} did not return array data.")
        except Exception as exc:
            errors.append(f"{reader.__name__}: {exc}")

    header_result.notes.append("direct Python parsing failed")
    header_result.notes.extend(errors)
    header_result.notes.append(
        "Install dependencies with: python3 -m pip install numpy matplotlib pySPM gwyfile tifffile imageio"
    )
    header_result.notes.append(
        "Fallback: use Gwyddion to export the desired channel as .gwy, .tiff, or .txt, then rerun this script."
    )
    return header_result


def choose_primary_channel(result: AFMReadResult) -> str:
    available = result.channels.keys()
    lowered_to_original = {name.lower(): name for name in available}

    for preferred in HEIGHT_CHANNEL_PREFERENCES:
        if preferred.lower() in lowered_to_original:
            return lowered_to_original[preferred.lower()]

    raise RuntimeError(
        "Neither ZSensor nor Height was found. Available channels: "
        + ", ".join(result.channels.keys())
    )


def low_frequency_gradient_note(array_nm: Any) -> str | None:
    np = import_required_numpy()
    array = np.asarray(array_nm, dtype=float)
    if array.ndim != 2:
        return None
    std = float(np.nanstd(array))
    if std == 0 or not np.isfinite(std):
        return None
    row_delta = float(np.nanmedian(array[-1, :]) - np.nanmedian(array[0, :]))
    col_delta = float(np.nanmedian(array[:, -1]) - np.nanmedian(array[:, 0]))
    if max(abs(row_delta), abs(col_delta)) > 2.0 * std:
        return (
            "possible low-frequency gradient/tilt detected; no leveling was applied"
        )
    return None


def render_height_maps(array_nm: Any, output_base: Path) -> tuple[Path, Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Missing required dependency: matplotlib. Install with: "
            "python3 -m pip install matplotlib"
        ) from exc

    np = import_required_numpy()
    array = np.asarray(array_nm, dtype=float)

    no_colorbar_path = output_base.with_name(output_base.name + "_render_nocolorbar.png")
    colorbar_path = output_base.with_name(output_base.name + "_render.png")

    plt.imsave(no_colorbar_path, array, cmap="gray")

    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    image = ax.imshow(array, cmap="gray", origin="upper")
    ax.set_axis_off()
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("height (nm)")
    fig.tight_layout()
    fig.savefig(colorbar_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    return colorbar_path, no_colorbar_path


def save_outputs(
    path: Path,
    output_dir: Path,
    result: AFMReadResult,
    primary_channel: str,
    sample_id: str | None = None,
    afm_file_id: str | None = None,
    relative_path: str | None = None,
) -> dict[str, Path]:
    np = import_required_numpy()
    afm_file_id = afm_file_id or parse_sample_id(path)
    sample_id = sample_id or afm_file_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / afm_file_id

    channel_info = result.channels[primary_channel]
    original_array = result.arrays[primary_channel]
    height_nm, exported_unit, conversion_note = convert_height_to_nm(
        original_array, channel_info.unit
    )
    height_nm = np.asarray(height_nm, dtype=float)
    stats_nm = numeric_stats(height_nm)
    channel_info.stats_nm = stats_nm

    gradient_note = low_frequency_gradient_note(height_nm)
    notes = list(result.notes)
    notes.append(conversion_note)
    if gradient_note:
        notes.append(gradient_note)

    npy_path = output_base.with_name(output_base.name + "_height.npy")
    np.save(npy_path, height_nm)
    render_path, render_no_colorbar_path = render_height_maps(height_nm, output_base)

    filename_scan_size = infer_scan_size_from_filename(path.name)
    scan_size_um = channel_info.scan_size_um or filename_scan_size

    metadata = {
        "sample_id": sample_id,
        "afm_file_id": afm_file_id,
        "raw_file": display_path(path),
        "raw_afm_file": display_path(path),
        "relative_path": relative_path,
        "output_dir": display_path(output_dir),
        "format": result.raw_format,
        "available_channels": list(result.channels.keys()),
        "primary_channel": primary_channel,
        "secondary_channel": "Height" if "Height" in result.channels else None,
        "resolution": list(height_nm.shape),
        "scan_size_um": scan_size_um,
        "scan_size_from_filename_um": filename_scan_size,
        "height_unit_original": channel_info.unit,
        "height_unit_exported": exported_unit,
        "height_min_nm": stats_nm["min"],
        "height_max_nm": stats_nm["max"],
        "height_mean_nm": stats_nm["mean"],
        "height_std_nm": stats_nm["std"],
        "channels": {
            name: {
                "shape": info.shape,
                "unit": info.unit,
                "scan_size_um": info.scan_size_um,
                "stats_original": info.stats_original,
                "stats_nm": info.stats_nm,
                "description": info.description,
                "source": info.source,
            }
            for name, info in result.channels.items()
        },
        "notes": "; ".join(notes),
    }

    metadata_path = output_base.with_name(output_base.name + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    inspection_path = output_base.with_name(output_base.name + "_inspection.txt")
    inspection_path.write_text(make_summary(path, result, primary_channel, metadata), encoding="utf-8")

    return {
        "height": npy_path,
        "render": render_path,
        "render_nocolorbar": render_no_colorbar_path,
        "metadata": metadata_path,
        "inspection": inspection_path,
    }


def make_summary(
    path: Path,
    result: AFMReadResult,
    primary_channel: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    lines = [
        "AFM inspection summary",
        "----------------------",
        f"Raw file: {display_path(path)}",
        f"Format: {result.raw_format}",
        "Available channels: " + (", ".join(result.channels.keys()) or "none found"),
    ]

    if "ZSensor" in result.channels:
        lines.append("Contains ZSensor channel: yes")
    else:
        lines.append("Contains ZSensor channel: no")

    if "Height" in result.channels:
        lines.append("Contains Height channel: yes")
    else:
        lines.append("Contains Height channel: no")

    for name, info in result.channels.items():
        shape = "unknown" if not info.shape else f"{info.shape[0]} x {info.shape[1]}"
        scan_size = "unknown" if not info.scan_size_um else f"{info.scan_size_um[0]} x {info.scan_size_um[1]} um"
        lines.append(f"- {name}: shape={shape}, unit={info.unit}, scan_size={scan_size}")
        if info.stats_original:
            stats = info.stats_original
            lines.append(
                f"  original stats: min={stats['min']:.6g}, max={stats['max']:.6g}, "
                f"mean={stats['mean']:.6g}, std={stats['std']:.6g}"
            )

    filename_scan = infer_scan_size_from_filename(path.name)
    lines.append(f"Filename scan size: {filename_scan or 'not found'}")

    if primary_channel and metadata:
        lines.extend(
            [
                f"Primary channel used: {primary_channel}",
                f"Resolution: {metadata['resolution'][0]} x {metadata['resolution'][1]}",
                f"Scan size: {metadata['scan_size_um'] or 'unknown'}",
                f"Exported height unit: {metadata['height_unit_exported']}",
                (
                    "Height range: "
                    f"{metadata['height_min_nm']:.6g} to {metadata['height_max_nm']:.6g} nm"
                ),
            ]
        )

    if result.notes:
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in result.notes)

    return "\n".join(lines) + "\n"


def build_summary_row(
    *,
    status: str,
    input_path: Path,
    output_dir: Path,
    sample_id: str,
    afm_file_id: str,
    relative_path: str | None,
    result: AFMReadResult | None = None,
    metadata: dict[str, Any] | None = None,
    error_message: str = "",
) -> dict[str, Any]:
    channels = list(result.channels.keys()) if result else []
    resolution = metadata.get("resolution") if metadata else None
    scan_size_um = metadata.get("scan_size_um") if metadata else None
    return {
        "sample_id": sample_id,
        "afm_file_id": afm_file_id,
        "raw_afm_file": display_path(input_path),
        "relative_path": relative_path,
        "status": status,
        "primary_channel": metadata.get("primary_channel") if metadata else "",
        "secondary_channel": metadata.get("secondary_channel") if metadata else "",
        "available_channels": ";".join(channels),
        "resolution_h": resolution[0] if resolution else "",
        "resolution_w": resolution[1] if resolution else "",
        "scan_size_x_um": scan_size_um[0] if scan_size_um else "",
        "scan_size_y_um": scan_size_um[1] if scan_size_um else "",
        "height_unit_original": metadata.get("height_unit_original") if metadata else "",
        "height_unit_exported": metadata.get("height_unit_exported") if metadata else "",
        "height_min_nm": metadata.get("height_min_nm") if metadata else "",
        "height_max_nm": metadata.get("height_max_nm") if metadata else "",
        "height_mean_nm": metadata.get("height_mean_nm") if metadata else "",
        "height_std_nm": metadata.get("height_std_nm") if metadata else "",
        "output_dir": display_path(output_dir),
        "error_message": error_message,
    }


def write_failed_inspection(
    path: Path,
    output_dir: Path,
    result: AFMReadResult,
    error: str,
    afm_file_id: str | None = None,
) -> Path:
    afm_file_id = afm_file_id or parse_sample_id(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    inspection_path = output_dir / f"{afm_file_id}_inspection.txt"
    text = make_summary(path, result) + f"\nError: {error}\n"
    inspection_path.write_text(text, encoding="utf-8")
    return inspection_path


def extract_one_afm_file(
    input_path: Path,
    output_dir: Path,
    sample_id: str | None = None,
    afm_file_id: str | None = None,
    relative_path: str | None = None,
) -> dict[str, Any]:
    """Inspect/extract one AFM file and return one CSV-ready summary row."""
    raw_path = input_path.expanduser()
    afm_file_id = afm_file_id or parse_sample_id(raw_path)
    sample_id = sample_id or afm_file_id

    if not raw_path.is_file():
        return build_summary_row(
            status="failed",
            input_path=raw_path,
            output_dir=output_dir,
            sample_id=sample_id,
            afm_file_id=afm_file_id,
            relative_path=relative_path,
            error_message=f"Input file does not exist: {display_path(raw_path)}",
        )

    result = inspect_channels(raw_path)

    try:
        primary = choose_primary_channel(result)
        if primary not in result.arrays:
            raise RuntimeError(
                f"Channel {primary!r} was identified in metadata but array data could not be read."
            )
        saved = save_outputs(
            raw_path,
            output_dir,
            result,
            primary,
            sample_id=sample_id,
            afm_file_id=afm_file_id,
            relative_path=relative_path,
        )
        metadata = json.loads(saved["metadata"].read_text(encoding="utf-8"))
        return build_summary_row(
            status="success",
            input_path=raw_path,
            output_dir=output_dir,
            sample_id=sample_id,
            afm_file_id=afm_file_id,
            relative_path=relative_path,
            result=result,
            metadata=metadata,
        )
    except Exception as exc:
        write_failed_inspection(raw_path, output_dir, result, str(exc), afm_file_id)
        return build_summary_row(
            status="failed",
            input_path=raw_path,
            output_dir=output_dir,
            sample_id=sample_id,
            afm_file_id=afm_file_id,
            relative_path=relative_path,
            result=result,
            error_message=str(exc),
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect and export one AFM raw height map."
    )
    parser.add_argument("--input", required=True, type=Path, help="AFM raw file path.")
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory where processed outputs will be written.",
    )
    args = parser.parse_args()

    raw_path = args.input.expanduser()
    afm_file_id = parse_sample_id(raw_path)
    output_dir = args.output_dir.expanduser() / afm_file_id
    row = extract_one_afm_file(
        raw_path,
        output_dir,
        sample_id=afm_file_id,
        afm_file_id=afm_file_id,
        relative_path=str(raw_path),
    )

    inspection_path = output_dir / f"{afm_file_id}_inspection.txt"
    if inspection_path.exists():
        print(inspection_path.read_text(encoding="utf-8"))

    if row["status"] == "success":
        print("Saved:")
        for suffix in ("height.npy", "render.png", "render_nocolorbar.png", "metadata.json", "inspection.txt"):
            print(f"- {output_dir / f'{afm_file_id}_{suffix}'}")
        return 0

    print(f"Error: {row['error_message']}", file=sys.stderr)
    print("Direct Python parsing failed or no usable height array was found.", file=sys.stderr)
    print(f"Inspection report written to: {inspection_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
