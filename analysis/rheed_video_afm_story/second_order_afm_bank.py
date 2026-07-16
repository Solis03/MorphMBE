from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .common import display_path, repo_path, save_parquet, sha256_file, write_csv, write_json
from .rq_disentanglement import ra_np, unit_shape


def build_second_order_bank(modeling_manifest: pd.DataFrame, descriptors: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out = repo_path(config["variant_output_root"]) / "afm_bank"
    maps = out / "second_order_unit_shape_maps"
    patch_root = out / "second_order_patch_bank"
    maps.mkdir(parents=True, exist_ok=True)
    patch_root.mkdir(parents=True, exist_ok=True)
    excluded = set(config["excluded_samples"])
    primary = modeling_manifest.query("usable_for_modeling and cohort_primary_1um").copy()
    primary["sample_id"] = primary["sample_id"].astype(str)
    primary = primary[~primary["sample_id"].isin(excluded)].sort_values("sample_id").reset_index(drop=True)
    desc = descriptors.set_index(["sample_id", "scan_id"])
    rows: list[dict[str, Any]] = []
    decoder_rows: list[dict[str, Any]] = []
    patch_registry: dict[str, Any] = {"variant_id": "afm_second_order_y2_v1", "patch_bank_root": display_path(patch_root), "samples": []}
    primary_ids = set(primary["sample_id"].astype(str))
    primary_scans = descriptors[
        descriptors["sample_id"].astype(str).isin(primary_ids)
        & (descriptors["scan_size_x_um"].sub(float(config["primary_scan_size_um"])).abs() <= float(config["scan_size_tolerance_um"]))
        & (descriptors["scan_size_y_um"].sub(float(config["primary_scan_size_um"])).abs() <= float(config["scan_size_tolerance_um"]))
    ].copy()
    rep_scan_by_sample = {
        str(row["sample_id"]): str(row["representative_afm_scan_id"]) for row in primary.to_dict("records")
    }
    for drow in primary_scans.to_dict("records"):
        sid = str(drow["sample_id"])
        scan_id = str(drow["scan_id"])
        arr = np.load(repo_path(drow["second_order_afm_path"]), allow_pickle=False).astype(np.float32)
        centered, unit, rq = unit_shape(arr)
        base = f"{sid}__{scan_id}"
        centered_path = maps / f"{base}_centered_256.npy"
        unit_path = maps / f"{base}_unit_shape_256.npy"
        np.save(centered_path, centered.astype(np.float32))
        np.save(unit_path, unit.astype(np.float32))
        decoder_rows.append(
            {
                "sample_id": sid,
                "growth_run_id": str(drow["growth_run_id"]),
                "source_afm_file": drow["raw_afm_path"],
                "afm_file_id": scan_id,
                "plane_corrected_array_path": drow["second_order_afm_path"],
                "second_order_afm_path": drow["second_order_afm_path"],
                "scan_size_x_um": float(drow["scan_size_x_um"]),
                "scan_size_y_um": float(drow["scan_size_y_um"]),
                "resolution_x": int(drow["resolution_x"]),
                "resolution_y": int(drow["resolution_y"]),
                "height_unit": "nm",
                "rq_nm": float(rq),
                "ra_nm": ra_np(centered),
                "robust_height_range_nm": float(drow["robust_height_range_nm"]),
                "height_skewness": float(drow["height_skewness"]),
                "height_kurtosis": float(drow["height_kurtosis"]),
                "paired_primary": True,
                "paired_exploratory": True,
                "unpaired_support": False,
                "representative_for_sample": scan_id == rep_scan_by_sample.get(sid),
                "quality_pass": True,
                "quality_flags": "",
                "source_array_hash": sha256_file(drow["second_order_afm_path"]),
                "unit_shape_paths": json.dumps({"256": display_path(unit_path)}, sort_keys=True),
                "centered_map_paths": json.dumps({"256": display_path(centered_path)}, sort_keys=True),
                "resize_audit": json.dumps({}, sort_keys=True),
                **{k: v for k, v in drow.items() if str(k).startswith("physical_") or str(k).startswith("unit_")},
            }
        )
    for row in primary.to_dict("records"):
        sid = str(row["sample_id"])
        scan_id = str(row["representative_afm_scan_id"])
        d = desc.loc[(sid, scan_id)]
        arr = np.load(repo_path(d["second_order_afm_path"]), allow_pickle=False).astype(np.float32)
        centered, unit, rq = unit_shape(arr)
        base = f"{sid}__{scan_id}"
        physical_path = maps / f"{base}_physical_centered.npy"
        unit_path = maps / f"{base}_unit_shape.npy"
        np.save(physical_path, centered.astype(np.float32))
        np.save(unit_path, unit.astype(np.float32))
        bank_row = {
            "sample_id": sid,
            "growth_run_id": str(row["growth_run_id"]),
            "scan_id": scan_id,
            "source_afm": d["second_order_afm_path"],
            "second_order_afm_path": d["second_order_afm_path"],
            "physical_map_path": d["second_order_afm_path"],
            "physical_centered_map_path": display_path(physical_path),
            "unit_shape_map_path": display_path(unit_path),
            "rq_nm": float(rq),
            "ra_nm": ra_np(centered),
            "robust_height_range_nm": float(d["robust_height_range_nm"]),
            "psd_low_fraction": float(d["psd_low_fraction"]),
            "psd_mid_fraction": float(d["psd_mid_fraction"]),
            "psd_high_fraction": float(d["psd_high_fraction"]),
            "psd_slope": float(d["psd_slope"]),
            "correlation_length_nm": float(d["correlation_length_nm"]),
            "anisotropy": float(d["anisotropy"]),
            "height_skewness": float(d["height_skewness"]),
            "height_kurtosis": float(d["height_kurtosis"]),
            "representative_for_sample": True,
            "source_hash": sha256_file(d["second_order_afm_path"]),
            "variant_id": "afm_second_order_y2_v1",
            "prototype_id": "second_order_physical_bank",
            "prototype_purity": 1.0,
            "mixed_morphology": False,
            "decoder_reconstruction_path": "",
            "high_frequency_residual_path": "",
            "residual_rms_amplitude": np.nan,
            "residual_psd_total": np.nan,
            "residual_low_band_energy": np.nan,
            "residual_mid_band_energy": np.nan,
            "residual_high_band_energy": np.nan,
        }
        rows.append(bank_row)
        patch_registry["samples"].append(
            {
                "sample_id": sid,
                "scan_id": scan_id,
                "unit_shape_map_path": display_path(unit_path),
                "source_hash": bank_row["source_hash"],
            }
        )
    bank = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    decoder = pd.DataFrame(decoder_rows).sort_values(["sample_id", "afm_file_id"]).reset_index(drop=True)
    write_csv(bank, out / "second_order_afm_morphology_bank.csv")
    save_parquet(bank, out / "second_order_afm_morphology_bank.parquet")
    write_csv(decoder, out / "second_order_afm_decoder_manifest.csv")
    save_parquet(decoder, out / "second_order_afm_decoder_manifest.parquet")
    write_json(patch_registry, out / "second_order_patch_bank_registry.json")
    return bank
