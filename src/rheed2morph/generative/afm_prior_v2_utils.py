"""Shared utilities for the AFM prior v2 workflow."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from rheed2morph.generative.afm_descriptors import DESCRIPTOR_NAMES, compute_afm_descriptors
from rheed2morph.generative.common import (
    REPO_ROOT,
    display_path,
    load_height_array,
    read_csv_rows,
    replace_nonfinite,
    resolve_repo_path,
    robust_normalize_to_unit,
    write_csv_rows,
)


V2_EXTRA_DESCRIPTOR_NAMES = [
    "height_min",
    "height_max",
    "slope_p50",
    "slope_p95",
    "slope_p99",
    "psd_peak_frequency",
    "island_mean_height",
]
V2_DESCRIPTOR_NAMES = list(DESCRIPTOR_NAMES) + V2_EXTRA_DESCRIPTOR_NAMES

META_COLUMNS = {
    "row_id",
    "parent_row_id",
    "sample_id",
    "group_id",
    "split",
    "source_kind",
    "network_input_path",
    "descriptor_height_path",
    "source_path",
    "patch_id",
    "is_patch",
    "metrics_scope",
}


def bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected a boolean-like value, got {value!r}")


def stable_id(*parts: Any, prefix: str = "afm") -> str:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def format_float(value: float) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.10g}"


def scan_target_um(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace(" ", "")
    if text in {"", "all", "*", "none"}:
        return None
    match = re.search(r"([0-9]+(?:[._][0-9]+)?)", text)
    if not match:
        return None
    number = float(match.group(1).replace("_", "."))
    if "nm" in text and "um" not in text:
        return number / 1000.0
    return number


def parse_scan_um_from_text(text: str) -> float | None:
    compact = text.lower().replace(" ", "_")
    if re.search(r"(?<![0-9])0_5_?um", compact):
        return 0.5
    patterns = [
        r"(?:^|[^0-9])([0-9]+(?:\.[0-9]+)?)_?um(?![a-z])",
        r"(?:^|[^0-9])([0-9]+(?:\.[0-9]+)?)_?nm(?![a-z])",
    ]
    for pattern in patterns:
        match = re.search(pattern, compact)
        if match:
            number = float(match.group(1))
            return number / 1000.0 if "nm" in match.group(0) else number
    return None


def scan_um_from_row(row: dict[str, str]) -> float | None:
    for key in ("scan_size_um", "afm_scan_size_um", "scan_size_x_um", "scan_size_y_um", "area_um2"):
        value = row.get(key, "").strip()
        if not value:
            continue
        try:
            parsed = float(value)
        except ValueError:
            parsed = scan_target_um(value)
        if parsed is None or not math.isfinite(float(parsed)):
            continue
        if key == "area_um2" and parsed > 0:
            parsed = math.sqrt(parsed)
        if parsed > 50.0:
            parsed = parsed / 1000.0
        return float(parsed)
    return None


def matches_scan(scan_um: float | None, target: float | None) -> bool:
    if target is None:
        return True
    if scan_um is None:
        return False
    tolerance = max(0.12, target * 0.12)
    return abs(float(scan_um) - target) <= tolerance


def compute_afm_descriptors_v2(height_map: np.ndarray) -> dict[str, float]:
    array = replace_nonfinite(np.asarray(height_map, dtype=np.float32))
    base = compute_afm_descriptors(array)
    values = array.astype(np.float64)
    gy, gx = np.gradient(values)
    slope = np.sqrt(gx * gx + gy * gy).ravel()
    centered = values - float(np.mean(values))
    spectrum = np.fft.fftshift(np.fft.fft2(centered))
    power = np.abs(spectrum) ** 2
    yy, xx = np.indices(power.shape)
    radius = np.sqrt((yy - power.shape[0] // 2) ** 2 + (xx - power.shape[1] // 2) ** 2)
    flat_power = power.ravel()
    flat_radius = radius.ravel()
    mask = flat_radius > 0
    if np.any(mask):
        peak_radius = float(flat_radius[mask][int(np.argmax(flat_power[mask]))])
        peak_frequency = peak_radius / max(float(min(power.shape)), 1.0)
    else:
        peak_frequency = 0.0
    threshold = float(np.percentile(values, 75.0))
    high_values = values[values > threshold]
    extra = {
        "height_min": float(np.min(values)),
        "height_max": float(np.max(values)),
        "slope_p50": float(np.percentile(slope, 50.0)),
        "slope_p95": float(np.percentile(slope, 95.0)),
        "slope_p99": float(np.percentile(slope, 99.0)),
        "psd_peak_frequency": peak_frequency,
        "island_mean_height": float(np.mean(high_values)) if high_values.size else 0.0,
    }
    merged = {**base, **extra}
    finite: dict[str, float] = {}
    for name in V2_DESCRIPTOR_NAMES:
        value = float(merged.get(name, float("nan")))
        finite[name] = value if math.isfinite(value) else float("nan")
    return finite


def descriptor_matrix(rows: Sequence[dict[str, Any]], columns: Sequence[str] = V2_DESCRIPTOR_NAMES) -> np.ndarray:
    return np.asarray([[float(row.get(col, "nan")) for col in columns] for row in rows], dtype=np.float32)


def standardize_descriptor_rows(
    rows: list[dict[str, Any]],
    descriptor_columns: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix = descriptor_matrix(rows, descriptor_columns).astype(np.float64)
    train_mask = np.asarray([row.get("split") == "train" for row in rows], dtype=bool)
    train = matrix[train_mask] if np.any(train_mask) else matrix
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    nan_counts = {name: int(np.sum(~np.isfinite(matrix[:, i]))) for i, name in enumerate(descriptor_columns)}
    imputed = matrix.copy()
    for index in range(imputed.shape[1]):
        bad = ~np.isfinite(imputed[:, index])
        imputed[bad, index] = medians[index]
    train_imputed = imputed[train_mask] if np.any(train_mask) else imputed
    means = np.mean(train_imputed, axis=0)
    stds = np.std(train_imputed, axis=0)
    stds = np.where(stds > 1e-8, stds, 1.0)
    updated: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        out = dict(row)
        for col_index, name in enumerate(descriptor_columns):
            out[name] = format_float(float(imputed[row_index, col_index]))
        updated.append(out)
    scaler = {
        "descriptor_columns": list(descriptor_columns),
        "train_median": {name: float(medians[i]) for i, name in enumerate(descriptor_columns)},
        "train_mean": {name: float(means[i]) for i, name in enumerate(descriptor_columns)},
        "train_std": {name: float(stds[i]) for i, name in enumerate(descriptor_columns)},
        "nan_imputation_counts": nan_counts,
    }
    return updated, scaler


def split_groups(groups: Sequence[str], seed: int, by_group: bool = True) -> dict[str, str]:
    unique = sorted(set(str(group) for group in groups)) if by_group else [str(group) for group in groups]
    rng = np.random.default_rng(seed)
    shuffled = list(unique)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n <= 1:
        return {group: "train" for group in shuffled}
    n_train = max(1, int(round(n * 0.70)))
    n_val = max(1, int(round(n * 0.15))) if n >= 3 else 0
    if n_train + n_val >= n:
        n_train = max(1, n - 1)
        n_val = 0 if n == 2 else 1
    mapping: dict[str, str] = {}
    for index, group in enumerate(shuffled):
        if index < n_train:
            mapping[group] = "train"
        elif index < n_train + n_val:
            mapping[group] = "val"
        else:
            mapping[group] = "test"
    return mapping


def source_kind_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return "png_fallback"
    if name == "network_input.npy":
        return "network_input"
    return "physical_height"


def source_priority(path: Path) -> int:
    text = path.as_posix().lower()
    name = path.name.lower()
    if "processed_zsensor_nm" in name:
        return 0
    if "raw_zsensor" in name:
        return 1
    if "plane_corrected" in name:
        return 2
    if "zsensor" in name:
        return 3
    if name.endswith("_height.npy") or "height" in name:
        return 4
    if name == "network_input.npy":
        return 8
    if text.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        return 20
    return 10


@dataclass(frozen=True)
class AFMCandidate:
    path: Path
    sample_id: str
    group_id: str
    afm_file_id: str
    scan_size_um: float | None
    source_kind: str
    metadata_path: str
    metadata_source: str

    @property
    def logical_key(self) -> str:
        scan = "unknown" if self.scan_size_um is None else f"{self.scan_size_um:.3f}"
        return f"{self.sample_id}|{self.afm_file_id}|{scan}"


def _infer_sample_from_path(path: Path) -> str:
    for part in path.parts:
        match = re.fullmatch(r"N?([0-9]{4})", part.replace(" - Copy", ""))
        if match:
            return match.group(1)
    for part in reversed(path.parts):
        match = re.search(r"([0-9]{4})", part)
        if match:
            return match.group(1)
    return path.parent.parent.name if path.parent.parent.name else path.parent.name


def _metadata_for_path(path: Path, metadata_by_path: dict[str, dict[str, str]], metadata_by_dir: dict[str, dict[str, str]]) -> dict[str, str]:
    resolved = path.resolve().as_posix()
    if resolved in metadata_by_path:
        return metadata_by_path[resolved]
    for parent in [path.parent, path.parent.parent]:
        key = parent.resolve().as_posix()
        if key in metadata_by_dir:
            return metadata_by_dir[key]
    return {}


def _add_path_hint(path: Path, row: dict[str, str], metadata_by_path: dict[str, dict[str, str]], metadata_by_dir: dict[str, dict[str, str]]) -> None:
    try:
        resolved = resolve_repo_path(path)
    except Exception:
        return
    if not resolved.exists():
        return
    metadata_by_path[resolved.as_posix()] = row
    metadata_by_dir[resolved.parent.as_posix()] = row


def load_afm_metadata_hints(extra_manifest: Path | None = None) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], list[str]]:
    metadata_by_path: dict[str, dict[str, str]] = {}
    metadata_by_dir: dict[str, dict[str, str]] = {}
    scanned: list[str] = []
    hint_paths = [
        REPO_ROOT / "data" / "processed_afm" / "afm_summary.csv",
        REPO_ROOT / "data" / "afm_descriptor_reconstruction_large" / "large_afm_manifest.csv",
        REPO_ROOT / "data" / "manifests" / "afm_candidate_table_complete.csv",
        REPO_ROOT / "data" / "manifests" / "manifest_all_size_representative_one_to_one.csv",
        REPO_ROOT / "data" / "manifests" / "manifest_1um_one_to_one.csv",
    ]
    if extra_manifest is not None:
        hint_paths.insert(0, resolve_repo_path(extra_manifest))
    for csv_path in hint_paths:
        if not csv_path.is_file() or display_path(csv_path) in scanned:
            continue
        scanned.append(display_path(csv_path))
        for row in read_csv_rows(csv_path):
            enriched = dict(row)
            if "group_id" not in enriched and enriched.get("sample_id"):
                enriched["group_id"] = enriched["sample_id"]
            if "scan_size_um" not in enriched:
                scan_um = scan_um_from_row(enriched)
                if scan_um is not None:
                    enriched["scan_size_um"] = str(scan_um)
            for key in ("afm_path", "network_input_path", "raw_afm_file", "height_png_path"):
                value = enriched.get(key, "").strip()
                if value:
                    _add_path_hint(Path(value), enriched, metadata_by_path, metadata_by_dir)
            output_dir = enriched.get("output_dir", "").strip()
            file_id = enriched.get("afm_file_id", "").strip()
            if output_dir and file_id:
                directory = resolve_repo_path(Path(output_dir))
                for suffix in ("_height.npy", "_render_nocolorbar.png", "_render.png", "_metadata.json"):
                    _add_path_hint(directory / f"{file_id}{suffix}", enriched, metadata_by_path, metadata_by_dir)
                plane = REPO_ROOT / "data" / "plane_corrected_afm" / str(enriched.get("sample_id", "")) / file_id / f"{file_id}_plane_corrected.npy"
                _add_path_hint(plane, enriched, metadata_by_path, metadata_by_dir)
    return metadata_by_path, metadata_by_dir, scanned


def _candidate_globs(root: Path) -> Iterable[Path]:
    patterns = [
        "**/processed_zsensor_nm.npy",
        "**/raw_zsensor.npy",
        "**/*plane_corrected.npy",
        "**/*zsensor*.npy",
        "**/*height*.npy",
        "**/network_input.npy",
        "**/*render_nocolorbar.png",
    ]
    seen: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path in seen or not path.is_file():
                continue
            if path.name.lower().endswith("_fitted_plane.npy"):
                continue
            seen.add(path)
            yield path


def discover_afm_candidates(
    afm_root: Path | None,
    manifest: Path | None,
    include_unpaired_afm: bool,
    scan_filter: str,
) -> tuple[list[AFMCandidate], dict[str, Any]]:
    metadata_by_path, metadata_by_dir, scanned_manifests = load_afm_metadata_hints(manifest)
    roots = [resolve_repo_path(afm_root)] if afm_root is not None else [REPO_ROOT / "data", REPO_ROOT / "reports"]
    manifest_paths: set[Path] = set()
    if manifest is not None:
        manifest_path = resolve_repo_path(manifest)
        manifest_dir = manifest_path.parent
        for row in read_csv_rows(manifest_path):
            for key in ("afm_path", "network_input_path", "raw_afm_file", "height_png_path"):
                value = row.get(key, "").strip()
                if value:
                    path = resolve_repo_path(Path(value), manifest_dir)
                    if path.exists():
                        manifest_paths.add(path)
            output_dir = row.get("output_dir", "").strip()
            file_id = row.get("afm_file_id", "").strip()
            if output_dir and file_id:
                directory = resolve_repo_path(Path(output_dir), manifest_dir)
                for suffix in ("_height.npy", "_plane_corrected.npy", "_render_nocolorbar.png"):
                    candidate = directory / f"{file_id}{suffix}"
                    if candidate.exists():
                        manifest_paths.add(candidate)
    paths: set[Path] = set(manifest_paths)
    if include_unpaired_afm or manifest is None:
        for root in roots:
            if root.exists():
                paths.update(path.resolve() for path in _candidate_globs(root))
    target = scan_target_um(scan_filter)
    candidates: list[AFMCandidate] = []
    all_scan_values: list[float] = []
    for path in sorted(paths):
        metadata = _metadata_for_path(path, metadata_by_path, metadata_by_dir)
        scan_um = scan_um_from_row(metadata)
        if scan_um is None:
            scan_um = parse_scan_um_from_text(path.as_posix())
        if scan_um is not None and math.isfinite(scan_um):
            all_scan_values.append(float(scan_um))
        if not matches_scan(scan_um, target):
            continue
        source_kind = source_kind_for_path(path)
        if source_kind == "png_fallback":
            physical_peer = list(path.parent.glob("*height*.npy")) + list(path.parent.glob("*plane_corrected.npy"))
            if physical_peer:
                continue
        sample_id = metadata.get("sample_id", "").strip() or _infer_sample_from_path(path)
        group_id = metadata.get("group_id", "").strip() or sample_id
        afm_file_id = metadata.get("afm_file_id", "").strip() or path.parent.name
        candidates.append(
            AFMCandidate(
                path=path.resolve(),
                sample_id=sample_id,
                group_id=group_id,
                afm_file_id=afm_file_id,
                scan_size_um=scan_um,
                source_kind=source_kind,
                metadata_path=metadata.get("metadata_path", ""),
                metadata_source=metadata.get("scan_size_source", "metadata_or_path"),
            )
        )
    best_by_key: dict[str, AFMCandidate] = {}
    duplicates = 0
    for candidate in candidates:
        key = candidate.logical_key
        current = best_by_key.get(key)
        if current is None or source_priority(candidate.path) < source_priority(current.path):
            if current is not None:
                duplicates += 1
            best_by_key[key] = candidate
        else:
            duplicates += 1
    deduped = sorted(best_by_key.values(), key=lambda item: (item.sample_id, item.afm_file_id, item.path.as_posix()))
    scan_counts: dict[str, int] = {}
    for value in all_scan_values:
        key = f"{value:.3f}"
        scan_counts[key] = scan_counts.get(key, 0) + 1
    diagnostics = {
        "scanned_manifests": scanned_manifests,
        "candidate_path_count_before_filter": len(paths),
        "candidate_count_after_scan_filter": len(candidates),
        "deduplicated_candidate_count": len(deduped),
        "deduplicated_duplicate_count": duplicates,
        "scan_size_counts_raw_candidates": scan_counts,
        "search_roots": [display_path(root) for root in roots],
        "searched_patterns": [
            "**/processed_zsensor_nm.npy",
            "**/raw_zsensor.npy",
            "**/*plane_corrected.npy",
            "**/*zsensor*.npy",
            "**/*height*.npy",
            "**/network_input.npy",
            "**/*render_nocolorbar.png",
        ],
    }
    return deduped, diagnostics


def crop_array(array: np.ndarray, patch_size: int, mode: str, patch_index: int, patches_per_image: int, seed: int) -> np.ndarray:
    h, w = array.shape
    if patch_size <= 0 or (h <= patch_size and w <= patch_size):
        return array
    patch_h = min(patch_size, h)
    patch_w = min(patch_size, w)
    if mode == "random":
        rng = np.random.default_rng(seed + patch_index * 9973)
        y0 = int(rng.integers(0, max(h - patch_h + 1, 1)))
        x0 = int(rng.integers(0, max(w - patch_w + 1, 1)))
    elif mode == "deterministic":
        grid = int(math.ceil(math.sqrt(max(patches_per_image, 1))))
        ys = np.linspace(0, max(h - patch_h, 0), grid, dtype=int)
        xs = np.linspace(0, max(w - patch_w, 0), grid, dtype=int)
        coords = [(int(y), int(x)) for y in ys for x in xs]
        y0, x0 = coords[patch_index % len(coords)]
    else:
        y0 = max((h - patch_h) // 2, 0)
        x0 = max((w - patch_w) // 2, 0)
    return array[y0 : y0 + patch_h, x0 : x0 + patch_w]


@dataclass(frozen=True)
class V2IndexRecord:
    row_id: str
    sample_id: str
    group_id: str
    split: str
    network_input_path: Path
    descriptor_height_path: Path | None
    parent_row_id: str
    source_kind: str


def load_v2_index(path: Path, split: str | None = None, limit: int | None = None) -> list[V2IndexRecord]:
    rows = read_csv_rows(resolve_repo_path(path))
    records: list[V2IndexRecord] = []
    for row in rows:
        if split is not None and row.get("split") != split:
            continue
        descriptor_path = row.get("descriptor_height_path", "")
        records.append(
            V2IndexRecord(
                row_id=row["row_id"],
                sample_id=row.get("sample_id", row["row_id"]),
                group_id=row.get("group_id", row.get("sample_id", row["row_id"])),
                split=row.get("split", ""),
                network_input_path=resolve_repo_path(Path(row["network_input_path"])),
                descriptor_height_path=resolve_repo_path(Path(descriptor_path)) if descriptor_path else None,
                parent_row_id=row.get("parent_row_id", row["row_id"]),
                source_kind=row.get("source_kind", ""),
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


class AFMPriorV2Dataset(Dataset[tuple[torch.Tensor, dict[str, str]]]):
    def __init__(self, records: Sequence[V2IndexRecord]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, str]]:
        record = self.records[index]
        image = load_height_array(record.network_input_path)
        tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))[None]
        return tensor, {
            "row_id": record.row_id,
            "sample_id": record.sample_id,
            "group_id": record.group_id,
            "split": record.split,
            "parent_row_id": record.parent_row_id,
        }


def condition_columns_from_schema(schema: dict[str, Any]) -> list[str]:
    return list(schema.get("condition_columns", [])) or [f"cond_{name}" for name in schema["descriptor_columns"]]


def build_condition_matrix_v2(
    condition_rows: Sequence[dict[str, str]],
    selected_row_ids: Sequence[str] | np.ndarray,
    schema: dict[str, Any],
) -> np.ndarray:
    by_row = {str(row["row_id"]): row for row in condition_rows}
    condition_cols = condition_columns_from_schema(schema)
    proto_count = int(schema.get("prototype_count", 0))
    matrix: list[list[float]] = []
    for row_id_value in selected_row_ids:
        row = by_row[str(row_id_value)]
        values = [float(row[col]) for col in condition_cols]
        if proto_count > 0:
            one_hot = [0.0] * proto_count
            proto = row.get("prototype_id", "")
            if proto != "":
                index = int(float(proto))
                if 0 <= index < proto_count:
                    one_hot[index] = 1.0
            values.extend(one_hot)
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float32)


def resize_tensor_to(image: torch.Tensor, image_size: int) -> torch.Tensor:
    if tuple(image.shape[-2:]) == (image_size, image_size):
        return image
    return F.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=False)


def write_training_curves(path: Path, history: Sequence[dict[str, float]], keys: Sequence[str]) -> None:
    if not history:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    epochs = [row["epoch"] for row in history]
    for key in keys:
        values = [row.get(key) for row in history]
        if any(value is not None and math.isfinite(float(value)) for value in values):
            ax.plot(epochs, values, label=key)
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_descriptor_plots(out_dir: Path, descriptor_rows: Sequence[dict[str, Any]], descriptor_columns: Sequence[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [row for row in descriptor_rows if row.get("split") == "train"]
    val = [row for row in descriptor_rows if row.get("split") == "val"]
    selected = list(descriptor_columns[: min(12, len(descriptor_columns))])
    fig, axes = plt.subplots(3, 4, figsize=(12, 8), dpi=150, squeeze=False)
    for axis, name in zip(axes.ravel(), selected):
        train_values = [float(row[name]) for row in train if row.get(name, "") != ""]
        val_values = [float(row[name]) for row in val if row.get(name, "") != ""]
        axis.hist(train_values, bins=20, alpha=0.65, label="train")
        if val_values:
            axis.hist(val_values, bins=20, alpha=0.65, label="val")
        axis.set_title(name, fontsize=8)
        axis.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "descriptor_histograms_train_val.png")
    plt.close(fig)
    matrix = descriptor_matrix(list(descriptor_rows), descriptor_columns)
    if matrix.shape[0] >= 2:
        corr = np.corrcoef(np.nan_to_num(matrix, nan=0.0), rowvar=False)
    else:
        corr = np.zeros((len(descriptor_columns), len(descriptor_columns)), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(descriptor_columns)))
    ax.set_yticks(range(len(descriptor_columns)))
    ax.set_xticklabels(descriptor_columns, rotation=90, fontsize=5)
    ax.set_yticklabels(descriptor_columns, fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "descriptor_correlation_matrix.png")
    plt.close(fig)


def write_npz_latents(path: Path, latents: np.ndarray, raw: np.ndarray, metas: Sequence[dict[str, str]], split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        latents=latents.astype(np.float32),
        latents_raw=raw.astype(np.float32),
        row_ids=np.asarray([meta["row_id"] for meta in metas]),
        sample_ids=np.asarray([meta["sample_id"] for meta in metas]),
        group_ids=np.asarray([meta["group_id"] for meta in metas]),
        parent_row_ids=np.asarray([meta.get("parent_row_id", meta["row_id"]) for meta in metas]),
        splits=np.asarray([split] * len(metas)),
    )


def read_metric_rows(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(resolve_repo_path(path)) if resolve_repo_path(path).is_file() else []


def write_rows_if_any(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    if rows:
        write_csv_rows(path, rows, fieldnames)
