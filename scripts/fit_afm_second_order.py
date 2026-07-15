#!/usr/bin/env python3
"""Subtract second-order fitted backgrounds from raw physical AFM height maps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "afm_second_order"
CLIPPING_SIGMA = 3.5
MAX_ROBUST_ITERATIONS = 5
MODEL_TERMS = {
    "y2": ("constant", "x", "y", "y2"),
    "full2d": ("constant", "x", "y", "x2", "xy", "y2"),
}
MANIFEST_FIELDS = [
    "source_path",
    "source_relative_path",
    "output_path",
    "background_path",
    "metadata_path",
    "status",
    "error_message",
    "model",
    "robust",
    "input_shape",
    "output_shape",
    "input_dtype",
    "output_dtype",
    "finite_pixel_count",
    "fit_pixel_count",
    "fit_fraction",
    "robust_iterations",
    "source_sha256_before",
    "source_sha256_after",
    "output_sha256",
    "coefficients",
    "rank",
    "condition_number",
    "rq_before",
    "rq_after",
    "ra_before",
    "ra_after",
    "peak_to_valley_before",
    "peak_to_valley_after",
    "vertical_edge_center_bow_before",
    "vertical_edge_center_bow_after",
]


@dataclass(frozen=True)
class ProcessOptions:
    model: str = "y2"
    robust: bool = True
    save_background: bool = True
    verbose: bool = False


def display_path(path: Path | None) -> str:
    if path is None:
        return ""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def matching_metadata_path(height_path: Path) -> Path:
    if height_path.name.endswith("_height.npy"):
        base = height_path.name.removesuffix("_height.npy")
        return height_path.with_name(f"{base}_metadata.json")
    return height_path.with_suffix(".json")


def is_raw_afm_height_file(path: Path) -> bool:
    if path.name != f"{path.stem}.npy":
        return False
    if not path.name.endswith("_height.npy"):
        return False
    lowered_parts = {part.lower() for part in path.parts}
    excluded = {
        "afm_second_order",
        "plane_corrected_afm",
        "network_inputs",
        "mlp_decoder",
        "pca_decoder",
        "descriptors",
        "rheed",
        "cache",
        "__pycache__",
    }
    if lowered_parts & excluded:
        return False
    metadata = load_json(matching_metadata_path(path))
    channel = metadata.get("primary_channel") or metadata.get("source_channel")
    return str(channel) == "ZSensor"


def discover_inputs(input_dir: Path, limit: int | None = None) -> list[Path]:
    files = [path for path in sorted(input_dir.rglob("*.npy")) if is_raw_afm_height_file(path)]
    return files if limit is None else files[:limit]


def normalized_coordinates(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = shape
    x_values = np.linspace(-1.0, 1.0, cols, dtype=np.float64)
    y_values = np.linspace(-1.0, 1.0, rows, dtype=np.float64)
    return np.meshgrid(x_values, y_values)


def design_matrix(shape: tuple[int, int], model: str) -> np.ndarray:
    x_grid, y_grid = normalized_coordinates(shape)
    if model == "y2":
        columns = [np.ones(shape, dtype=np.float64), x_grid, y_grid, y_grid * y_grid]
    elif model == "full2d":
        columns = [
            np.ones(shape, dtype=np.float64),
            x_grid,
            y_grid,
            x_grid * x_grid,
            x_grid * y_grid,
            y_grid * y_grid,
        ]
    else:
        raise ValueError(f"Unsupported model: {model}")
    return np.column_stack([column.ravel() for column in columns])


def fit_background(
    z2d: np.ndarray, mask: np.ndarray, model: str
) -> tuple[np.ndarray, np.ndarray, int, float]:
    terms = MODEL_TERMS[model]
    if int(np.count_nonzero(mask)) < len(terms):
        raise ValueError("insufficient_finite_pixels")
    design = design_matrix(z2d.shape, model)
    fit_design = design[mask.ravel()]
    values = z2d.ravel()[mask.ravel()].astype(np.float64, copy=False)
    coefficients, _, rank, _ = np.linalg.lstsq(fit_design, values, rcond=None)
    if int(rank) < len(terms):
        raise ValueError("rank_deficient")
    condition_number = float(np.linalg.cond(fit_design))
    background = (design @ coefficients).reshape(z2d.shape)
    return background, coefficients.astype(np.float64), int(rank), condition_number


def robust_fit(
    z2d: np.ndarray, model: str, robust: bool
) -> tuple[np.ndarray, np.ndarray, int, float, np.ndarray, int]:
    finite_mask = np.isfinite(z2d)
    terms = MODEL_TERMS[model]
    finite_count = int(np.count_nonzero(finite_mask))
    if finite_count < len(terms):
        raise ValueError("insufficient_finite_pixels")

    fit_mask = finite_mask.copy()
    iterations = 0
    if robust:
        for _ in range(MAX_ROBUST_ITERATIONS):
            background, _, _, _ = fit_background(z2d, fit_mask, model)
            residual = z2d - background
            current = residual[fit_mask]
            median = float(np.median(current))
            mad = float(np.median(np.abs(current - median)))
            if mad <= 1e-12:
                break
            sigma = 1.4826 * mad
            new_mask = finite_mask & (np.abs(residual - median) <= CLIPPING_SIGMA * sigma)
            min_pixels = max(20 * len(terms), int(math.ceil(0.10 * finite_count)))
            if int(np.count_nonzero(new_mask)) < min_pixels:
                break
            iterations += 1
            if np.array_equal(new_mask, fit_mask):
                break
            fit_mask = new_mask

    background, coefficients, rank, condition_number = fit_background(z2d, fit_mask, model)
    return background, coefficients, rank, condition_number, fit_mask, iterations


def squeeze_to_2d(array: np.ndarray) -> tuple[np.ndarray | None, Callable[[np.ndarray], np.ndarray] | None]:
    if array.ndim == 2:
        return array, lambda value: value
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0, :, :], lambda value: value[np.newaxis, :, :]
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[:, :, 0], lambda value: value[:, :, np.newaxis]
    squeezed = np.squeeze(array)
    if squeezed.ndim == 2 and array.ndim == 3 and 1 in array.shape:
        return squeezed, lambda value: np.reshape(value, array.shape)
    return None, None


def output_dtype_for(input_dtype: np.dtype) -> np.dtype:
    if input_dtype == np.dtype("float64"):
        return np.dtype("float64")
    return np.dtype("float32")


def roughness_rq(array: np.ndarray) -> float:
    finite = array[np.isfinite(array)].astype(np.float64, copy=False)
    if finite.size == 0:
        return float("nan")
    centered = finite - float(np.mean(finite))
    return float(np.sqrt(np.mean(centered * centered)))


def roughness_ra(array: np.ndarray) -> float:
    finite = array[np.isfinite(array)].astype(np.float64, copy=False)
    if finite.size == 0:
        return float("nan")
    centered = finite - float(np.mean(finite))
    return float(np.mean(np.abs(centered)))


def peak_to_valley(array: np.ndarray) -> float:
    finite = array[np.isfinite(array)].astype(np.float64, copy=False)
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite) - np.min(finite))


def finite_median(array: np.ndarray) -> float:
    finite = array[np.isfinite(array)].astype(np.float64, copy=False)
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def vertical_edge_center_bow(array: np.ndarray) -> float:
    z2d, _ = squeeze_to_2d(np.asarray(array))
    if z2d is None:
        return float("nan")
    rows = z2d.shape[0]
    edge_rows = max(1, int(math.ceil(rows * 0.10)))
    center_start = max(0, int(math.floor(rows * 0.40)))
    center_end = min(rows, int(math.ceil(rows * 0.60)))
    top = finite_median(z2d[:edge_rows, :])
    bottom = finite_median(z2d[-edge_rows:, :])
    center = finite_median(z2d[center_start:center_end, :])
    return float((top + bottom) / 2.0 - center)


def coefficients_dict(model: str, coefficients: np.ndarray) -> dict[str, float]:
    return {term: float(value) for term, value in zip(MODEL_TERMS[model], coefficients, strict=True)}


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def atomic_write_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            tmp_name = handle.name
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp_name, path)
    except FileExistsError:
        raise
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink()
            except FileNotFoundError:
                pass


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False, encoding="utf-8"
        ) as handle:
            tmp_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp_name, path)
    except FileExistsError:
        raise
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink()
            except FileNotFoundError:
                pass


def replace_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False, encoding="utf-8"
        ) as handle:
            tmp_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        tmp_name = ""
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink()
            except FileNotFoundError:
                pass


def planned_paths(source_path: Path, input_dir: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    rel = source_path.relative_to(input_dir)
    output_path = output_dir / rel
    background_path = output_dir / "_backgrounds" / rel
    metadata_path = output_dir / "_metadata" / rel.with_suffix(".json")
    return output_path, background_path, metadata_path


def base_row(
    source_path: Path,
    input_dir: Path,
    output_dir: Path,
    options: ProcessOptions,
    status: str,
    error_message: str = "",
) -> dict[str, Any]:
    output_path, background_path, metadata_path = planned_paths(source_path, input_dir, output_dir)
    return {
        "source_path": display_path(source_path),
        "source_relative_path": str(source_path.relative_to(input_dir)),
        "output_path": display_path(output_path),
        "background_path": display_path(background_path) if options.save_background else "",
        "metadata_path": display_path(metadata_path),
        "status": status,
        "error_message": error_message,
        "model": options.model,
        "robust": bool(options.robust),
        "input_shape": "",
        "output_shape": "",
        "input_dtype": "",
        "output_dtype": "",
        "finite_pixel_count": "",
        "fit_pixel_count": "",
        "fit_fraction": "",
        "robust_iterations": "",
        "source_sha256_before": "",
        "source_sha256_after": "",
        "output_sha256": "",
        "coefficients": "",
        "rank": "",
        "condition_number": "",
        "rq_before": "",
        "rq_after": "",
        "ra_before": "",
        "ra_after": "",
        "peak_to_valley_before": "",
        "peak_to_valley_after": "",
        "vertical_edge_center_bow_before": "",
        "vertical_edge_center_bow_after": "",
    }


def process_file(
    source_path: Path, input_dir: Path, output_dir: Path, options: ProcessOptions
) -> dict[str, Any]:
    row = base_row(source_path, input_dir, output_dir, options, "failed")
    output_path, background_path, metadata_path = planned_paths(source_path, input_dir, output_dir)
    try:
        source_sha_before = sha256_file(source_path)
        row["source_sha256_before"] = source_sha_before
        if output_path.exists() or metadata_path.exists() or (options.save_background and background_path.exists()):
            row["status"] = "exists_skipped"
            row["source_sha256_after"] = sha256_file(source_path)
            return row

        original = np.load(source_path, allow_pickle=False)
        row["input_shape"] = json.dumps(list(original.shape))
        row["input_dtype"] = str(original.dtype)
        if not np.issubdtype(original.dtype, np.number):
            row["status"] = "unsupported_dtype"
            row["error_message"] = f"non-numeric dtype: {original.dtype}"
            row["source_sha256_after"] = sha256_file(source_path)
            return row

        z2d_view, restore_shape = squeeze_to_2d(original)
        if z2d_view is None or restore_shape is None:
            row["status"] = "unsupported_shape"
            row["error_message"] = f"unsupported shape: {original.shape}"
            row["source_sha256_after"] = sha256_file(source_path)
            return row

        z2d = np.asarray(z2d_view, dtype=np.float64)
        finite_mask = np.isfinite(z2d)
        finite_count = int(np.count_nonzero(finite_mask))
        row["finite_pixel_count"] = finite_count
        if finite_count < len(MODEL_TERMS[options.model]):
            row["status"] = "insufficient_finite_pixels"
            row["error_message"] = "not enough finite pixels for selected model"
            row["source_sha256_after"] = sha256_file(source_path)
            return row

        background2d, coefficients, rank, condition_number, fit_mask, iterations = robust_fit(
            z2d, options.model, options.robust
        )
        corrected2d = np.array(z2d, copy=True)
        corrected2d[finite_mask] = z2d[finite_mask] - background2d[finite_mask]
        corrected2d[~finite_mask] = z2d[~finite_mask]
        if not np.all(np.isfinite(corrected2d[finite_mask])):
            raise ValueError("corrected finite input pixels are not finite")

        restored_corrected = restore_shape(corrected2d)
        restored_background = restore_shape(background2d)
        if restored_corrected.shape != original.shape:
            raise ValueError(f"output shape mismatch: {restored_corrected.shape} != {original.shape}")

        output_dtype = output_dtype_for(original.dtype)
        corrected_out = restored_corrected.astype(output_dtype, copy=False)
        background_out = restored_background.astype(output_dtype, copy=False)
        row["output_shape"] = json.dumps(list(corrected_out.shape))
        row["output_dtype"] = str(corrected_out.dtype)
        row["fit_pixel_count"] = int(np.count_nonzero(fit_mask))
        row["fit_fraction"] = float(np.count_nonzero(fit_mask) / finite_count)
        row["robust_iterations"] = iterations
        row["coefficients"] = json.dumps(coefficients_dict(options.model, coefficients), sort_keys=True)
        row["rank"] = rank
        row["condition_number"] = condition_number
        row["rq_before"] = roughness_rq(z2d)
        row["rq_after"] = roughness_rq(corrected2d)
        row["ra_before"] = roughness_ra(z2d)
        row["ra_after"] = roughness_ra(corrected2d)
        row["peak_to_valley_before"] = peak_to_valley(z2d)
        row["peak_to_valley_after"] = peak_to_valley(corrected2d)
        row["vertical_edge_center_bow_before"] = vertical_edge_center_bow(z2d)
        row["vertical_edge_center_bow_after"] = vertical_edge_center_bow(corrected2d)

        atomic_write_array(output_path, corrected_out)
        if options.save_background:
            atomic_write_array(background_path, background_out)

        output_sha = sha256_file(output_path)
        row["output_sha256"] = output_sha
        source_sha_after = sha256_file(source_path)
        row["source_sha256_after"] = source_sha_after
        if source_sha_before != source_sha_after:
            raise ValueError("source hash changed during processing")

        formula = (
            "c0 + c1*x + c2*y + c3*y^2"
            if options.model == "y2"
            else "c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2"
        )
        metadata = {
            "source_path": display_path(source_path),
            "output_path": display_path(output_path),
            "processing": "second_order_background_subtraction",
            "model": options.model,
            "formula": formula,
            "coordinate_convention": {
                "array_axis_0": "y / vertical / rows",
                "array_axis_1": "x / horizontal / columns",
                "coordinate_range": "[-1, 1]",
            },
            "robust": bool(options.robust),
            "robust_iterations": iterations,
            "clipping_sigma": CLIPPING_SIGMA,
            "finite_pixel_count": finite_count,
            "final_fit_pixel_count": int(np.count_nonzero(fit_mask)),
            "final_fit_fraction": float(np.count_nonzero(fit_mask) / finite_count),
            "coefficients": coefficients_dict(options.model, coefficients),
            "rank": rank,
            "condition_number": condition_number,
            "input_shape": list(original.shape),
            "input_dtype": str(original.dtype),
            "output_dtype": str(corrected_out.dtype),
            "height_units": "unchanged_from_input",
            "rq_before": row["rq_before"],
            "rq_after": row["rq_after"],
            "ra_before": row["ra_before"],
            "ra_after": row["ra_after"],
            "peak_to_valley_before": row["peak_to_valley_before"],
            "peak_to_valley_after": row["peak_to_valley_after"],
            "vertical_edge_center_bow_before": row["vertical_edge_center_bow_before"],
            "vertical_edge_center_bow_after": row["vertical_edge_center_bow_after"],
            "source_sha256": source_sha_after,
            "output_sha256": output_sha,
            "dtype_conversion": str(original.dtype) != str(corrected_out.dtype),
            "notes": "No normalization, smoothing, resizing, line correction, or clipping was applied.",
        }
        atomic_write_text(metadata_path, json.dumps(json_safe(metadata), indent=2, sort_keys=True) + "\n")
        row["status"] = "success"
        return row
    except FileExistsError:
        row["status"] = "exists_skipped"
        row["source_sha256_after"] = sha256_file(source_path)
        return row
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["error_message"] = str(exc)
        try:
            row["source_sha256_after"] = sha256_file(source_path)
        except OSError:
            pass
        return row


def write_manifest(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = output_dir / "processing_manifest.csv"
    jsonl_path = output_dir / "processing_manifest.jsonl"
    csv_text_handle = tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8", delete=False)
    tmp_path = Path(csv_text_handle.name)
    try:
        with csv_text_handle as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
        replace_text(csv_path, tmp_path.read_text(encoding="utf-8"))
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass

    lines = []
    for row in rows:
        json_row = dict(row)
        if isinstance(json_row.get("coefficients"), str) and json_row["coefficients"]:
            json_row["coefficients"] = json.loads(json_row["coefficients"])
        lines.append(json.dumps(json_safe(json_row), sort_keys=True))
    replace_text(jsonl_path, "\n".join(lines) + ("\n" if lines else ""))


def metric_values(rows: list[dict[str, Any]], before_key: str, after_key: str) -> list[tuple[float, float]]:
    values = []
    for row in rows:
        if row.get("status") != "success":
            continue
        try:
            before = float(row[before_key])
            after = float(row[after_key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(before) and math.isfinite(after):
            values.append((before, after))
    return values


def median_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def write_qc_summary(
    output_dir: Path,
    input_dir: Path,
    model: str,
    rows: list[dict[str, Any]],
    qc_note: str = "",
) -> None:
    counts = Counter(str(row.get("status")) for row in rows)
    errors = Counter(str(row.get("error_message") or row.get("status")) for row in rows if row.get("status") != "success")
    rq_pairs = metric_values(rows, "rq_before", "rq_after")
    bow_pairs = metric_values(rows, "vertical_edge_center_bow_before", "vertical_edge_center_bow_after")
    rq_relative = [after / before - 1.0 for before, after in rq_pairs if before != 0.0]
    bow_abs_change = [abs(after) - abs(before) for before, after in bow_pairs]
    hash_ok = all(
        row.get("source_sha256_before") == row.get("source_sha256_after")
        for row in rows
        if row.get("source_sha256_before") and row.get("source_sha256_after")
    )
    lines = [
        f"input_root: {display_path(input_dir)}",
        f"output_root: {display_path(output_dir)}",
        f"model: {model}",
        f"total_inputs: {len(rows)}",
        f"success_count: {counts.get('success', 0)}",
        f"skipped_count: {sum(count for status, count in counts.items() if status.endswith('skipped'))}",
        f"error_count: {len(rows) - counts.get('success', 0) - sum(count for status, count in counts.items() if status.endswith('skipped'))}",
        f"median_rq_relative_change: {median_or_nan(rq_relative)}",
        f"median_vertical_bow_abs_change: {median_or_nan(bow_abs_change)}",
        f"source_hash_all_unchanged: {hash_ok}",
        f"errors_by_type: {dict(sorted(errors.items()))}",
    ]
    if qc_note:
        lines.append(f"qc_note: {qc_note}")
    replace_text(output_dir / "_qc" / "qc_summary.txt", "\n".join(lines) + "\n")


def finite_percentile_limits(array: np.ndarray) -> tuple[float, float] | None:
    finite = array[np.isfinite(array)].astype(np.float64, copy=False)
    if finite.size == 0:
        return None
    low, high = np.percentile(finite, [2.0, 98.0])
    if low == high:
        pad = 1.0 if low == 0 else abs(low) * 0.01
        return float(low - pad), float(high + pad)
    return float(low), float(high)


def make_qc_grid(
    input_dir: Path,
    output_dir: Path,
    rows: list[dict[str, Any]],
    model: str,
    qc_count: int,
) -> str:
    if qc_count <= 0:
        return "qc grid disabled"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        return f"matplotlib unavailable; skipped qc grid: {exc}"

    success_rows = [row for row in rows if row.get("status") == "success"][:qc_count]
    if not success_rows:
        return "no successful samples for qc grid"

    fig, axes = plt.subplots(len(success_rows), 4, figsize=(14, 3.0 * len(success_rows)), dpi=140)
    if len(success_rows) == 1:
        axes = np.asarray([axes])
    for row_axes, row in zip(axes, success_rows, strict=True):
        source = (REPO_ROOT / row["source_path"]).resolve() if not Path(row["source_path"]).is_absolute() else Path(row["source_path"])
        output = (REPO_ROOT / row["output_path"]).resolve() if not Path(row["output_path"]).is_absolute() else Path(row["output_path"])
        background = (REPO_ROOT / row["background_path"]).resolve() if not Path(row["background_path"]).is_absolute() else Path(row["background_path"])
        original = np.squeeze(np.load(source, allow_pickle=False)).astype(np.float64)
        corrected = np.squeeze(np.load(output, allow_pickle=False)).astype(np.float64)
        bg = np.squeeze(np.load(background, allow_pickle=False)).astype(np.float64)
        shared_limits = finite_percentile_limits(np.concatenate([original.ravel(), corrected.ravel()]))
        bg_limits = finite_percentile_limits(bg)
        title = (
            f"{row['source_relative_path']}\n"
            f"{model}, Rq {float(row['rq_before']):.4g}->{float(row['rq_after']):.4g}, "
            f"bow {float(row['vertical_edge_center_bow_before']):.4g}->{float(row['vertical_edge_center_bow_after']):.4g}"
        )
        row_axes[0].imshow(original, cmap="viridis", origin="upper", vmin=shared_limits[0], vmax=shared_limits[1])
        row_axes[0].set_title(title, fontsize=7)
        row_axes[1].imshow(bg, cmap="magma", origin="upper", vmin=bg_limits[0], vmax=bg_limits[1])
        row_axes[1].set_title("fitted background", fontsize=8)
        row_axes[2].imshow(corrected, cmap="viridis", origin="upper", vmin=shared_limits[0], vmax=shared_limits[1])
        row_axes[2].set_title("corrected", fontsize=8)
        row_profile = np.arange(original.shape[0])
        row_axes[3].plot(row_profile, np.nanmedian(original, axis=1), label="original", linewidth=1)
        row_axes[3].plot(row_profile, np.nanmedian(corrected, axis=1), label="corrected", linewidth=1)
        row_axes[3].set_title("row median", fontsize=8)
        row_axes[3].legend(fontsize=6)
        for ax in row_axes[:3]:
            ax.set_axis_off()
    fig.tight_layout()
    path = output_dir / "_qc" / "qc_grid.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return f"qc grid written: {display_path(path)}"


def print_dry_run(input_dir: Path, output_dir: Path, files: list[Path]) -> None:
    excluded = [
        "data/afm_second_order",
        "data/plane_corrected_afm",
        "data/afm_descriptor_reconstruction/network_inputs",
        "data/afm_descriptor_reconstruction_large/network_inputs",
        "RHEED directories",
        "latent/reconstruction/cache directories",
    ]
    print(f"Input root: {display_path(input_dir)}")
    print("Input selector: *_height.npy with matching metadata primary_channel/source_channel == ZSensor")
    print(f"AFM file count: {len(files)}")
    print("First 10 input files:")
    for path in files[:10]:
        print(f"  {display_path(path)}")
    print("Excluded directories/classes:")
    for item in excluded:
        print(f"  {item}")
    print(f"Output root: {display_path(output_dir)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Root containing raw physical AFM NPY height maps.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Root for second-order outputs.")
    parser.add_argument("--model", choices=sorted(MODEL_TERMS), default="y2")
    parser.add_argument("--robust", dest="robust", action="store_true", default=True)
    parser.add_argument("--no-robust", dest="robust", action="store_false")
    parser.add_argument("--save-background", dest="save_background", action="store_true", default=True)
    parser.add_argument("--no-save-background", dest="save_background", action="store_false")
    parser.add_argument("--qc-count", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.qc_count < 0:
        parser.error("--qc-count must be non-negative")
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(f"input directory does not exist: {input_dir}")
    files = discover_inputs(input_dir, args.limit)

    if args.dry_run:
        print_dry_run(input_dir, output_dir, files)
        return 0
    if args.workers != 1:
        print(f"warning: --workers={args.workers} requested; processing serially to preserve manifest order.")
    if not files:
        print_dry_run(input_dir, output_dir, files)
        return 1

    options = ProcessOptions(model=args.model, robust=args.robust, save_background=args.save_background, verbose=args.verbose)
    rows: list[dict[str, Any]] = []
    for path in files:
        row = process_file(path, input_dir, output_dir, options)
        rows.append(row)
        if args.verbose:
            message = row["status"]
            if row.get("error_message"):
                message += f": {row['error_message']}"
            print(f"{message}: {row['source_relative_path']}")

    write_manifest(output_dir, rows)
    qc_note = make_qc_grid(input_dir, output_dir, rows, args.model, args.qc_count)
    write_qc_summary(output_dir, input_dir, args.model, rows, qc_note)
    counts = Counter(str(row.get("status")) for row in rows)
    print(
        "Second-order AFM fitting complete: "
        f"{counts.get('success', 0)} success, "
        f"{sum(count for status, count in counts.items() if status.endswith('skipped'))} skipped, "
        f"{len(rows) - counts.get('success', 0) - sum(count for status, count in counts.items() if status.endswith('skipped'))} failed."
    )
    print(qc_note)
    print(f"Manifest: {display_path(output_dir / 'processing_manifest.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
