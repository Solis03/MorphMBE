"""Prepare AFM tensors, descriptors, splits, and prototype labels."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from rheed2morph.generative.afm_descriptors import DESCRIPTOR_NAMES, compute_afm_descriptors
from rheed2morph.generative.common import (
    REPO_ROOT,
    display_path,
    load_height_array,
    read_csv_rows,
    robust_normalize_to_unit,
    resolve_repo_path,
    set_seed,
    write_csv_rows,
    write_json,
)


PATH_COLUMNS = (
    "afm_height_path",
    "processed_afm_path",
    "afm_path",
    "afm_file",
    "network_input_path",
    "raw_afm_file",
)
NETWORK_COLUMNS = ("network_input_path",)
SCAN_COLUMNS = ("scan_size", "afm_scan_size_um", "scan_size_um", "scan_size_x_um", "scan_size_y_um")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare AFM latent diffusion MVP data.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--afm-root", type=Path, default=None)
    parser.add_argument("--scan-size-filter", type=str, default="1um")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _stable_row_id(source: str, index: int) -> str:
    digest = hashlib.sha1(f"{source}|{index}".encode("utf-8")).hexdigest()[:10]
    return f"afm_{index:05d}_{digest}"


def _parse_scan_target(value: str | None) -> float | None:
    if not value:
        return None
    text = str(value).strip().lower().replace(" ", "")
    match = re.search(r"([0-9]*\.?[0-9]+)", text)
    if not match:
        return None
    number = float(match.group(1))
    if "nm" in text and "um" not in text:
        return number / 1000.0
    return number


def _row_scan_um(row: dict[str, str]) -> float | None:
    for column in SCAN_COLUMNS:
        value = row.get(column, "")
        if value == "":
            continue
        parsed = _parse_scan_target(value)
        if parsed is not None:
            if parsed > 50.0:
                return parsed / 1000.0
            return parsed
    return None


def _matches_scan_filter(row: dict[str, str], target: float | None) -> bool:
    if target is None:
        return True
    value = _row_scan_um(row)
    if value is None:
        joined = " ".join(str(v).lower() for v in row.values())
        return f"{target:g}um" in joined or f"{target:g}_um" in joined
    tolerance = max(0.12, target * 0.12)
    return abs(value - target) <= tolerance


def discover_manifest() -> Path:
    preferred = [
        REPO_ROOT / "data" / "manifests" / "manifest_1um_one_to_one.csv",
        REPO_ROOT / "data" / "manifests" / "manifest_all_size_representative_one_to_one.csv",
        REPO_ROOT / "data" / "afm_descriptor_reconstruction" / "afm_1um_manifest.csv",
        REPO_ROOT / "data" / "processed_afm" / "afm_summary.csv",
    ]
    for path in preferred:
        if path.is_file():
            return path
    candidates = sorted((REPO_ROOT / "data").rglob("*manifest*.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No AFM manifest CSV found. Pass --manifest or --afm-root.")


def _path_from_row(row: dict[str, str], column: str, manifest_dir: Path) -> Path | None:
    value = row.get(column, "").strip()
    if not value:
        return None
    path = resolve_repo_path(Path(value), manifest_dir)
    return path if path.exists() else None


def _output_dir_height(row: dict[str, str], manifest_dir: Path) -> Path | None:
    output_dir = row.get("output_dir", "").strip()
    file_id = row.get("afm_file_id", "").strip()
    if not output_dir or not file_id:
        return None
    directory = resolve_repo_path(Path(output_dir), manifest_dir)
    candidates = [
        directory / f"{file_id}_height.npy",
        directory / f"{file_id}_processed_zsensor_nm.npy",
        directory / "processed_zsensor_nm.npy",
        directory / "network_input.npy",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _output_dir_png(row: dict[str, str], manifest_dir: Path) -> Path | None:
    output_dir = row.get("output_dir", "").strip()
    if not output_dir:
        return None
    directory = resolve_repo_path(Path(output_dir), manifest_dir)
    candidates = sorted(directory.glob("*render_nocolorbar.png")) + sorted(directory.glob("*render.png"))
    return candidates[0].resolve() if candidates else None


def _best_existing_path(row: dict[str, str], manifest_dir: Path, afm_root: Path | None, columns: tuple[str, ...]) -> Path | None:
    for column in columns:
        path = _path_from_row(row, column, manifest_dir)
        if path is not None:
            return path
        if afm_root is not None and row.get(column, "").strip():
            rooted = resolve_repo_path(afm_root / row[column].strip(), manifest_dir)
            if rooted.exists():
                return rooted
    return None


def _infer_sample_id(row: dict[str, str], path: Path | None) -> tuple[str, str]:
    for column in ("sample_id", "sample", "growth_id"):
        value = row.get(column, "").strip()
        if value:
            return value, f"column:{column}"
    if path is not None:
        for part in reversed(path.parts):
            match = re.search(r"([0-9]{4})", part)
            if match:
                return match.group(1), "path_numeric_token"
        return path.parent.name, "path_parent"
    return "unknown", "unknown"


def _infer_group_id(row: dict[str, str], sample_id: str, path: Path | None) -> tuple[str, str]:
    for column in ("group_id", "sample_group", "growth_id"):
        value = row.get(column, "").strip()
        if value:
            return value, f"column:{column}"
    if sample_id != "unknown":
        return sample_id, "sample_id"
    if path is not None:
        return path.parent.name, "path_parent"
    return "unknown", "unknown"


def _split_groups(groups: list[str], seed: int) -> dict[str, str]:
    unique = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n = len(unique)
    if n <= 1:
        return {group: "train" for group in unique}
    n_train = max(1, int(round(n * 0.70)))
    n_val = max(1, int(round(n * 0.15))) if n >= 3 else 0
    if n_train + n_val >= n:
        n_train = max(1, n - 1)
        n_val = 0 if n == 2 else 1
    split_for_group: dict[str, str] = {}
    for index, group in enumerate(unique):
        if index < n_train:
            split_for_group[group] = "train"
        elif index < n_train + n_val:
            split_for_group[group] = "val"
        else:
            split_for_group[group] = "test"
    return split_for_group


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.10g}"


def _standardize_descriptors(rows: list[dict[str, Any]], train_mask: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix = np.asarray([[float(row[name]) for name in DESCRIPTOR_NAMES] for row in rows], dtype=np.float64)
    nan_counts = {name: int(np.sum(~np.isfinite(matrix[:, i]))) for i, name in enumerate(DESCRIPTOR_NAMES)}
    train = matrix[train_mask]
    if train.shape[0] == 0:
        train = matrix
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    imputed = matrix.copy()
    for col_index in range(imputed.shape[1]):
        mask = ~np.isfinite(imputed[:, col_index])
        imputed[mask, col_index] = medians[col_index]
    train_imputed = imputed[train_mask] if np.any(train_mask) else imputed
    means = np.mean(train_imputed, axis=0)
    stds = np.std(train_imputed, axis=0)
    stds = np.where(stds > 1e-8, stds, 1.0)
    updated: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        new_row = dict(row)
        for col_index, name in enumerate(DESCRIPTOR_NAMES):
            new_row[name] = _format_float(float(imputed[row_index, col_index]))
        updated.append(new_row)
    scaler = {
        "descriptor_columns": DESCRIPTOR_NAMES,
        "train_median": {name: float(medians[i]) for i, name in enumerate(DESCRIPTOR_NAMES)},
        "train_mean": {name: float(means[i]) for i, name in enumerate(DESCRIPTOR_NAMES)},
        "train_std": {name: float(stds[i]) for i, name in enumerate(DESCRIPTOR_NAMES)},
        "nan_imputation_counts": nan_counts,
    }
    return updated, scaler


def _write_prototypes(out_dir: Path, descriptor_rows: list[dict[str, Any]], scaler: dict[str, Any]) -> dict[str, Any]:
    train_rows = [row for row in descriptor_rows if row.get("split") == "train"]
    train_groups = sorted({str(row["group_id"]) for row in train_rows})
    k = min(6, max(2, len(train_groups) // 4))
    if len(train_rows) < k or k < 2:
        return {"prototype_status": "skipped_small_dataset", "prototype_k": 0, "cluster_counts": {}}
    try:
        from sklearn.cluster import KMeans
    except Exception as exc:
        return {"prototype_status": f"skipped_sklearn_unavailable:{exc}", "prototype_k": 0, "cluster_counts": {}}
    means = np.asarray([scaler["train_mean"][name] for name in DESCRIPTOR_NAMES], dtype=np.float32)
    stds = np.asarray([scaler["train_std"][name] for name in DESCRIPTOR_NAMES], dtype=np.float32)
    matrix = np.asarray([[float(row[name]) for name in DESCRIPTOR_NAMES] for row in descriptor_rows], dtype=np.float32)
    standardized = (matrix - means[None]) / stds[None]
    train_mask = np.asarray([row.get("split") == "train" for row in descriptor_rows], dtype=bool)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(standardized[train_mask])
    labels = model.predict(standardized)
    rows: list[dict[str, Any]] = []
    for row, label in zip(descriptor_rows, labels):
        rows.append(
            {
                "row_id": row["row_id"],
                "sample_id": row["sample_id"],
                "group_id": row["group_id"],
                "split": row["split"],
                "prototype_id": int(label),
            }
        )
    write_csv_rows(out_dir / "prototype_labels.csv", rows, ["row_id", "sample_id", "group_id", "split", "prototype_id"])
    counts = {str(label): int(np.sum(labels == label)) for label in sorted(set(labels.tolist()))}
    return {"prototype_status": "ok", "prototype_k": int(k), "cluster_counts": counts}


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir = out_dir / "standardized_tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve_repo_path(args.manifest) if args.manifest else discover_manifest()
    afm_root = resolve_repo_path(args.afm_root) if args.afm_root else None
    manifest_dir = manifest_path.parent
    target_scan = _parse_scan_target(args.scan_size_filter)
    source_rows = read_csv_rows(manifest_path)
    data_rows: list[dict[str, Any]] = []
    descriptor_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    group_rules: dict[str, int] = {}
    source_counts = {"physical_height": 0, "network_input": 0, "png_fallback": 0}

    for source_index, row in enumerate(source_rows, start=1):
        if not _matches_scan_filter(row, target_scan):
            continue
        physical_path = _output_dir_height(row, manifest_dir)
        if physical_path is None:
            physical_path = _best_existing_path(row, manifest_dir, afm_root, PATH_COLUMNS)
        network_source = _best_existing_path(row, manifest_dir, afm_root, NETWORK_COLUMNS)
        png_fallback = None if physical_path is not None or network_source is not None else _output_dir_png(row, manifest_dir)
        source_path = network_source or physical_path or png_fallback
        if source_path is None:
            skipped.append({"source_index": str(source_index), "reason": "no_loadable_afm_path"})
            continue
        try:
            source_array = load_height_array(source_path)
            descriptor_array = load_height_array(physical_path or source_path)
        except Exception as exc:
            skipped.append({"source_index": str(source_index), "path": str(source_path), "reason": f"load_failed:{exc}"})
            continue
        sample_id, sample_rule = _infer_sample_id(row, source_path)
        group_id, group_rule = _infer_group_id(row, sample_id, source_path)
        group_rules[group_rule] = group_rules.get(group_rule, 0) + 1
        row_id = row.get("row_id", "").strip() or _stable_row_id(source_path.as_posix(), source_index)
        tensor = robust_normalize_to_unit(source_array, int(args.image_size))
        tensor_path = tensor_dir / f"{row_id}.npy"
        np.save(tensor_path, tensor.astype(np.float32))
        if network_source is not None:
            source_kind = "network_input"
        elif physical_path is not None:
            source_kind = "physical_height"
        else:
            source_kind = "png_fallback"
        source_counts[source_kind] += 1
        descriptor_values = compute_afm_descriptors(descriptor_array)
        scan_value = _row_scan_um(row)
        base = {
            "row_id": row_id,
            "sample_id": sample_id,
            "group_id": group_id,
            "split": "",
            "network_input_path": display_path(tensor_path),
            "source_network_input_path": display_path(network_source) if network_source else "",
            "descriptor_height_path": display_path(physical_path or source_path),
            "source_path": display_path(source_path),
            "source_kind": source_kind,
            "scan_size_um": _format_float(float(scan_value)) if scan_value is not None else "",
            "sample_id_rule": sample_rule,
            "group_id_rule": group_rule,
            "height_shape": "x".join(map(str, descriptor_array.shape)),
        }
        data_rows.append(base)
        descriptor_row = {key: base[key] for key in ("row_id", "sample_id", "group_id", "split", "source_kind")}
        descriptor_row.update({name: _format_float(value) for name, value in descriptor_values.items()})
        descriptor_rows.append(descriptor_row)
        if args.limit is not None and len(data_rows) >= int(args.limit):
            break

    if not data_rows:
        raise RuntimeError(
            f"No AFM samples were indexed from {manifest_path}. "
            "Pass --manifest/--afm-root or relax --scan-size-filter."
        )
    split_for_group = _split_groups([str(row["group_id"]) for row in data_rows], int(args.seed))
    for row in data_rows:
        row["split"] = split_for_group[str(row["group_id"])]
    for row in descriptor_rows:
        row["split"] = split_for_group[str(row["group_id"])]
    train_mask = np.asarray([row.get("split") == "train" for row in descriptor_rows], dtype=bool)
    descriptor_rows, scaler = _standardize_descriptors(descriptor_rows, train_mask)
    prototype_info = _write_prototypes(out_dir, descriptor_rows, scaler)
    split_rows = [{"group_id": group, "split": split} for group, split in sorted(split_for_group.items())]
    write_csv_rows(out_dir / "data_index.csv", data_rows)
    write_csv_rows(out_dir / "splits.csv", split_rows, ["group_id", "split"])
    write_csv_rows(out_dir / "afm_descriptors.csv", descriptor_rows)
    write_json(out_dir / "descriptor_scaler.json", scaler)
    split_sample_counts = {split: sum(1 for row in data_rows if row["split"] == split) for split in ("train", "val", "test")}
    split_group_counts = {split: sum(1 for value in split_for_group.values() if value == split) for split in ("train", "val", "test")}
    inventory = {
        "manifest_path": display_path(manifest_path),
        "afm_root": display_path(afm_root) if afm_root else "",
        "scan_size_filter": args.scan_size_filter,
        "image_size": int(args.image_size),
        "sample_count": len(data_rows),
        "group_count": len(split_for_group),
        "split_sample_counts": split_sample_counts,
        "split_group_counts": split_group_counts,
        "group_id_rules": group_rules,
        "source_counts": source_counts,
        "skipped_count": len(skipped),
        "skipped_examples": skipped[:20],
        **prototype_info,
    }
    write_json(out_dir / "data_inventory.json", inventory)
    return inventory


def main() -> None:
    args = build_parser().parse_args()
    inventory = prepare_dataset(args)
    print(f"Wrote AFM latent dataset to {display_path(resolve_repo_path(args.out))}")
    print(f"Indexed {inventory['sample_count']} AFM files across {inventory['group_count']} groups.")


if __name__ == "__main__":
    main()
