from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

from .afm_descriptors import describe_map
from .common import median_abs_deviation, repo_path, save_parquet, write_csv, write_json
from .rq_disentanglement import ra_np, rq_np, unit_shape


def descriptor_row(mapping_row: pd.Series) -> dict[str, Any]:
    arr = np.load(repo_path(mapping_row["second_order_afm_path"]), allow_pickle=False).astype(np.float32)
    if not np.isfinite(arr).all():
        finite = np.isfinite(arr)
        arr = np.where(finite, arr, float(np.nanmean(arr[finite])) if finite.any() else 0.0).astype(np.float32)
    centered, unit, rq = unit_shape(arr)
    physical = describe_map(arr, "physical", scan_size_um=float(mapping_row["scan_size_x_um"]))
    unit_desc = describe_map(unit, "unit", scan_size_um=float(mapping_row["scan_size_x_um"]))
    return {
        "sample_id": str(mapping_row["sample_id"]),
        "growth_run_id": str(mapping_row["growth_run_id"]),
        "scan_id": str(mapping_row["scan_id"]),
        "second_order_afm_path": mapping_row["second_order_afm_path"],
        "raw_afm_path": mapping_row["raw_afm_path"],
        "first_order_afm_path": mapping_row["first_order_afm_path"],
        "scan_size_x_um": float(mapping_row["scan_size_x_um"]),
        "scan_size_y_um": float(mapping_row["scan_size_y_um"]),
        "resolution_x": int(mapping_row["resolution_x"]),
        "resolution_y": int(mapping_row["resolution_y"]),
        "height_unit": mapping_row["height_unit"],
        "rq_nm": rq,
        "ra_nm": ra_np(centered),
        "robust_height_range_nm": float(np.percentile(centered, 99) - np.percentile(centered, 1)),
        "p99_minus_p1_nm": float(np.percentile(centered, 99) - np.percentile(centered, 1)),
        "p95_minus_p5_nm": float(np.percentile(centered, 95) - np.percentile(centered, 5)),
        "height_skewness": physical["physical_skewness"],
        "height_kurtosis": physical["physical_kurtosis"],
        "mean_absolute_gradient": physical["physical_mean_abs_gradient"],
        "rms_gradient": physical["physical_rms_gradient"],
        "psd_low_fraction": unit_desc["unit_psd_low_fraction"],
        "psd_mid_fraction": unit_desc["unit_psd_mid_fraction"],
        "psd_high_fraction": unit_desc["unit_psd_high_fraction"],
        "psd_slope": unit_desc["unit_psd_slope"],
        "correlation_length_nm": unit_desc["unit_autocorr_length_nm"],
        "directional_correlation_length_x_nm": unit_desc["unit_corr_length_x_nm"],
        "directional_correlation_length_y_nm": unit_desc["unit_corr_length_y_nm"],
        "anisotropy": unit_desc["unit_anisotropy_ratio"],
        "source_hash": mapping_row["second_order_sha256"],
        **physical,
        **unit_desc,
    }


def rebuild_targets(mapping: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    out = repo_path(config["variant_output_root"])
    target_root = out / "targets"
    comp_root = out / "comparison"
    target_root.mkdir(parents=True, exist_ok=True)
    comp_root.mkdir(parents=True, exist_ok=True)
    phase1_path = repo_path(config["phase1_manifest_path"])
    if phase1_path.suffix == ".parquet":
        try:
            phase1 = pd.read_parquet(phase1_path)
        except ImportError:
            phase1 = pd.read_csv(phase1_path.with_suffix(".csv"), dtype={"sample_id": str, "growth_run_id": str})
        phase1["sample_id"] = phase1["sample_id"].astype(str)
        phase1["growth_run_id"] = phase1["growth_run_id"].astype(str)
    else:
        phase1 = pd.read_csv(phase1_path, dtype={"sample_id": str, "growth_run_id": str})
    phase1_audit = pd.read_csv(repo_path(config["phase1_afm_audit_path"]), dtype={"sample_id": str})

    ok = mapping[
        mapping["mapping_status"].eq("ok")
        & mapping["output_exists"].astype(bool)
        & mapping["height_unit"].eq("nm")
        & ~mapping["second_order_afm_path"].str.contains("_backgrounds", regex=False)
    ].copy()
    descriptors = pd.DataFrame([descriptor_row(row) for _, row in ok.iterrows()]).sort_values(["sample_id", "scan_id"]).reset_index(drop=True)

    tol = float(config["scan_size_tolerance_um"])
    primary_scan = descriptors[
        (descriptors["scan_size_x_um"].sub(float(config["primary_scan_size_um"])).abs() <= tol)
        & (descriptors["scan_size_y_um"].sub(float(config["primary_scan_size_um"])).abs() <= tol)
    ].copy()

    first_scan = phase1_audit.rename(
        columns={
            "afm_file_id": "scan_id",
            "rq_recomputed_nm": "first_order_rq_nm",
            "height_array_path": "first_order_afm_path",
        }
    )
    scan_compare = primary_scan.merge(
        first_scan[["sample_id", "scan_id", "first_order_rq_nm", "first_order_afm_path", "is_representative"]],
        on=["sample_id", "scan_id"],
        how="left",
        suffixes=("", "_phase1"),
    )
    scan_compare["second_order_rq_nm"] = scan_compare["rq_nm"]
    scan_compare["rq_difference_nm"] = scan_compare["second_order_rq_nm"] - scan_compare["first_order_rq_nm"]
    scan_compare["rq_relative_change"] = scan_compare["rq_difference_nm"] / scan_compare["first_order_rq_nm"].replace(0, np.nan)
    write_csv(scan_compare, comp_root / "first_vs_second_order_scan_targets.csv")
    write_csv(descriptors, target_root / "second_order_afm_descriptors.csv")
    write_csv(primary_scan, target_root / "second_order_afm_scan_audit.csv")

    first_targets = phase1.set_index("sample_id")
    rows: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    for sid, group in primary_scan.groupby("sample_id", sort=True):
        rqs = group["rq_nm"].to_numpy(float)
        med = float(np.median(rqs))
        rep = group.assign(_dist=(group["rq_nm"] - med).abs()).sort_values(["_dist", "scan_id"]).iloc[0]
        first = first_targets.loc[str(sid)] if str(sid) in first_targets.index else pd.Series(dtype=object)
        first_ra = np.nan
        first_rep_scan = str(first.get("representative_afm_scan_id", ""))
        if first_rep_scan:
            first_match = phase1_audit[(phase1_audit["sample_id"].astype(str) == str(sid)) & (phase1_audit["afm_file_id"].astype(str) == first_rep_scan)]
            if len(first_match):
                arr = np.load(repo_path(first_match.iloc[0]["height_array_path"]), allow_pickle=False)
                first_ra = ra_np(arr)
        row = {
            "sample_id": str(sid),
            "growth_run_id": str(first.get("growth_run_id", sid)),
            "first_order_rq_nm": float(first.get("primary_rq_nm_median", np.nan)),
            "second_order_rq_nm": med,
            "rq_difference_nm": med - float(first.get("primary_rq_nm_median", np.nan)),
            "rq_relative_change": (med - float(first.get("primary_rq_nm_median", np.nan))) / max(float(first.get("primary_rq_nm_median", np.nan)), 1e-12),
            "first_order_ra_nm": first_ra,
            "second_order_ra_nm": float(rep["ra_nm"]),
            "first_order_representative_scan": first_rep_scan,
            "second_order_representative_scan": str(rep["scan_id"]),
            "representative_scan_changed": str(first_rep_scan) != str(rep["scan_id"]),
            "second_order_scan_count": int(len(group)),
            "second_order_rq_iqr": float(np.percentile(rqs, 75) - np.percentile(rqs, 25)),
            "second_order_rq_mad": median_abs_deviation(rqs),
            "second_order_rq_min": float(np.min(rqs)),
            "second_order_rq_max": float(np.max(rqs)),
            "second_order_ground_truth_afm_path": rep["second_order_afm_path"],
        }
        rows.append(row)
        representative_rows.append({**row, "second_order_afm_path": rep["second_order_afm_path"], "scan_id": rep["scan_id"]})
    sample_targets = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    representatives = pd.DataFrame(representative_rows).sort_values("sample_id").reset_index(drop=True)

    modeling = phase1.copy()
    target_map = sample_targets.set_index("sample_id")
    for idx, row in modeling.iterrows():
        sid = str(row["sample_id"])
        if sid not in target_map.index:
            continue
        target = target_map.loc[sid]
        modeling.at[idx, "primary_afm_available"] = True
        modeling.at[idx, "primary_afm_scan_size_um"] = float(config["primary_scan_size_um"])
        modeling.at[idx, "primary_afm_scan_count"] = int(target["second_order_scan_count"])
        modeling.at[idx, "primary_rq_nm_median"] = float(target["second_order_rq_nm"])
        modeling.at[idx, "primary_rq_nm_iqr"] = float(target["second_order_rq_iqr"])
        modeling.at[idx, "primary_rq_nm_mad"] = float(target["second_order_rq_mad"])
        modeling.at[idx, "primary_rq_nm_min"] = float(target["second_order_rq_min"])
        modeling.at[idx, "primary_rq_nm_max"] = float(target["second_order_rq_max"])
        modeling.at[idx, "representative_afm_path"] = target["second_order_ground_truth_afm_path"]
        modeling.at[idx, "representative_afm_height_array"] = target["second_order_ground_truth_afm_path"]
        modeling.at[idx, "representative_afm_scan_id"] = target["second_order_representative_scan"]
        modeling.at[idx, "afm_target_variant"] = "second_order_y2"
    write_csv(modeling, target_root / "second_order_modeling_manifest.csv")
    save_parquet(modeling, target_root / "second_order_modeling_manifest.parquet")
    write_csv(sample_targets, target_root / "second_order_sample_targets.csv")
    write_csv(representatives, target_root / "second_order_representative_afm.csv")

    sample_compare = sample_targets.copy()
    sample_compare["first_order_rank"] = sample_compare["first_order_rq_nm"].rank(method="min")
    sample_compare["second_order_rank"] = sample_compare["second_order_rq_nm"].rank(method="min")
    sample_compare["rq_rank_change"] = sample_compare["second_order_rank"] - sample_compare["first_order_rank"]
    write_csv(sample_compare, comp_root / "first_vs_second_order_sample_targets.csv")
    desc_compare = scan_compare.copy()
    write_csv(desc_compare, comp_root / "first_vs_second_order_descriptor_changes.csv")
    write_csv(sample_compare[["sample_id", "first_order_rank", "second_order_rank", "rq_rank_change"]], comp_root / "target_rank_change.csv")

    primary_ids = (
        phase1.query("usable_for_modeling and cohort_primary_1um")["sample_id"].astype(str).tolist()
    )
    primary_ids = [sid for sid in primary_ids if sid not in set(config["excluded_samples"])]
    primary_targets = sample_compare[sample_compare["sample_id"].isin(primary_ids)]
    summary = {
        "valid_second_order_output_count": int(len(mapping[mapping["output_exists"].astype(bool)])),
        "valid_primary_1um_scan_count": int(len(primary_scan[primary_scan["sample_id"].isin(primary_ids)])),
        "primary_growth_group_count": int(len(primary_ids)),
        "mapping_complete": bool(mapping["mapping_status"].eq("ok").all()),
        "primary_first_order_rq_distribution": primary_targets["first_order_rq_nm"].describe().to_dict(),
        "primary_second_order_rq_distribution": primary_targets["second_order_rq_nm"].describe().to_dict(),
        "per_sample_rq_change_min": float(primary_targets["rq_difference_nm"].min()),
        "per_sample_rq_change_max": float(primary_targets["rq_difference_nm"].max()),
        "pearson_first_second_rq": float(pearsonr(primary_targets["first_order_rq_nm"], primary_targets["second_order_rq_nm"]).statistic),
        "spearman_first_second_rq": float(spearmanr(primary_targets["first_order_rq_nm"], primary_targets["second_order_rq_nm"]).statistic),
        "kendall_first_second_rq": float(kendalltau(primary_targets["first_order_rq_nm"], primary_targets["second_order_rq_nm"]).statistic),
        "rank_reorder_count": int((primary_targets["rq_rank_change"] != 0).sum()),
        "representative_scan_changed_count": int(primary_targets["representative_scan_changed"].sum()),
    }
    write_json(summary, comp_root / "target_comparison_summary.json")
    return modeling, sample_targets, descriptors, summary
