from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from skimage.transform import resize

from .afm_descriptors import describe_map
from .common import display_path, read_id_list, repo_path, save_parquet, sha256_file, sha256_object, write_csv
from .rq_disentanglement import ra_np, rq_np, unit_shape


def load_phase3_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = pd.read_csv(repo_path(config["phase1_afm_audit_path"]), dtype={"sample_id": str})
    phase1 = pd.read_csv(repo_path(config["phase1_manifest_path"]), dtype={"sample_id": str})
    return audit, phase1


def downsample_height(arr: np.ndarray, resolution: int) -> np.ndarray:
    if arr.shape == (resolution, resolution):
        return arr.astype(np.float32)
    out = resize(
        arr.astype(np.float32),
        (resolution, resolution),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )
    return out.astype(np.float32)


def build_afm_manifest(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_root = repo_path(config["output_root"])
    arrays_root = output_root / "normalized_arrays"
    arrays_root.mkdir(parents=True, exist_ok=True)
    removelist = read_id_list(config["removelist_path"])
    audit, phase1 = load_phase3_inputs(config)
    paired_primary = set(phase1.query("usable_for_modeling and cohort_primary_1um")["sample_id"].astype(str))
    paired_exploratory = set(phase1.query("usable_for_modeling")["sample_id"].astype(str))
    representatives = set(phase1["representative_afm_height_array"].dropna().astype(str))

    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    tol = float(config["scan_size_tolerance_um"])
    config_hash = sha256_object(config)
    removelist_hash = sha256_file(config["removelist_path"])
    for raw in audit.to_dict("records"):
        sid = str(raw["sample_id"])
        excluded = sid in removelist
        unit_ok = str(raw.get("height_unit", "")) == config["height_unit"]
        sx = float(raw.get("scan_size_x_um", np.nan))
        sy = float(raw.get("scan_size_y_um", np.nan))
        primary_size = abs(sx - float(config["primary_scan_size_um"])) <= tol and abs(sy - float(config["primary_scan_size_um"])) <= tol
        array_path = str(raw["height_array_path"])
        source_exists = repo_path(array_path).exists()
        quality_flags = [] if pd.isna(raw.get("quality_flags")) else [str(raw.get("quality_flags"))]
        quality_pass = (not excluded) and unit_ok and primary_size and source_exists
        if not source_exists:
            quality_flags.append("missing_plane_corrected_array")
        if excluded:
            quality_flags.append("canonical_removelist_excluded_before_discovery")
        if not unit_ok:
            quality_flags.append("height_unit_not_nm")
        if not primary_size:
            quality_flags.append("not_primary_1um")
        audit_rows.append(
            {
                "sample_id": sid,
                "source_afm_file": raw.get("raw_afm_file", ""),
                "plane_corrected_array_path": array_path,
                "excluded_by_removelist": excluded,
                "height_unit_ok_nm": unit_ok,
                "primary_1um": primary_size,
                "source_exists": source_exists,
                "quality_pass": quality_pass,
                "quality_flags": ";".join([x for x in quality_flags if x and x != "nan"]),
            }
        )
        if not quality_pass:
            continue
        arr = np.load(repo_path(array_path)).astype(np.float32)
        finite = np.isfinite(arr)
        if finite.mean() < 1.0:
            arr = np.where(finite, arr, float(np.nanmean(arr[finite]))).astype(np.float32)
        z0, s256, q = unit_shape(arr, epsilon=float(config["epsilon"]))
        if arr.shape != (int(raw["resolution_y"]), int(raw["resolution_x"])):
            quality_flags.append("decode_or_shape_mismatch")
        s_paths = {}
        z_paths = {}
        resize_records = {}
        for res in sorted(set(config["input_resolutions"].values())):
            resized = downsample_height(arr, int(res))
            rz0, rs, rq_resized = unit_shape(resized, epsilon=float(config["epsilon"]))
            s_path = arrays_root / f"{sid}__{raw['afm_file_id']}__unit_shape_{res}.npy"
            z_path = arrays_root / f"{sid}__{raw['afm_file_id']}__centered_{res}.npy"
            np.save(s_path, rs.astype(np.float32))
            np.save(z_path, rz0.astype(np.float32))
            s_paths[str(res)] = display_path(s_path)
            z_paths[str(res)] = display_path(z_path)
            resize_records[str(res)] = {
                "rq_after_resize_nm": rq_resized,
                "ra_after_resize_nm": ra_np(rz0),
                "shape_mean": float(rs.mean()),
                "shape_rq": rq_np(rs),
            }
        physical_desc = describe_map(arr, "physical", scan_size_um=float(config["primary_scan_size_um"]))
        unit_desc = describe_map(s256, "unit", scan_size_um=float(config["primary_scan_size_um"]))
        rows.append(
            {
                "sample_id": sid,
                "growth_run_id": str(sid),
                "source_afm_file": raw.get("raw_afm_file", ""),
                "afm_file_id": raw.get("afm_file_id", ""),
                "plane_corrected_array_path": array_path,
                "scan_size_x_um": sx,
                "scan_size_y_um": sy,
                "resolution_x": int(raw["resolution_x"]),
                "resolution_y": int(raw["resolution_y"]),
                "height_unit": "nm",
                "rq_nm": q,
                "ra_nm": ra_np(z0),
                "robust_height_range_nm": float(np.percentile(z0, 99) - np.percentile(z0, 1)),
                "height_skewness": physical_desc["physical_skewness"],
                "height_kurtosis": physical_desc["physical_kurtosis"],
                "paired_primary": sid in paired_primary,
                "paired_exploratory": sid in paired_exploratory,
                "unpaired_support": sid not in paired_exploratory,
                "representative_for_sample": array_path in representatives or bool(raw.get("is_representative", False)),
                "quality_pass": True,
                "quality_flags": ";".join([x for x in quality_flags if x and x != "nan"]),
                "removelist_hash": removelist_hash,
                "source_array_hash": sha256_file(array_path),
                "config_hash": config_hash,
                "unit_shape_paths": json.dumps(s_paths, sort_keys=True),
                "centered_map_paths": json.dumps(z_paths, sort_keys=True),
                "resize_audit": json.dumps(resize_records, sort_keys=True),
                **physical_desc,
                **unit_desc,
            }
        )
    manifest = pd.DataFrame(rows).sort_values(["sample_id", "afm_file_id"]).reset_index(drop=True)
    decoder_audit = pd.DataFrame(audit_rows).sort_values(["sample_id", "source_afm_file"]).reset_index(drop=True)
    descriptors = manifest[
        [
            "sample_id",
            "growth_run_id",
            "afm_file_id",
            "plane_corrected_array_path",
            "rq_nm",
            "ra_nm",
            "robust_height_range_nm",
        ]
        + [c for c in manifest.columns if c.startswith("physical_") or c.startswith("unit_")]
    ].copy()
    write_csv(manifest, output_root / "afm_decoder_manifest.csv")
    save_parquet(manifest, output_root / "afm_decoder_manifest.parquet")
    write_csv(decoder_audit, output_root / "afm_decoder_audit.csv")
    write_csv(descriptors, output_root / "afm_descriptors.csv")
    return manifest, decoder_audit, descriptors


def load_unit_shapes(manifest: pd.DataFrame, resolution: int) -> np.ndarray:
    arrays = []
    for paths in manifest["unit_shape_paths"]:
        path = json.loads(paths)[str(resolution)]
        arrays.append(np.load(repo_path(path)).astype(np.float32))
    return np.stack(arrays)


def load_centered_maps(manifest: pd.DataFrame, resolution: int) -> np.ndarray:
    arrays = []
    for paths in manifest["centered_map_paths"]:
        path = json.loads(paths)[str(resolution)]
        arrays.append(np.load(repo_path(path)).astype(np.float32))
    return np.stack(arrays)
