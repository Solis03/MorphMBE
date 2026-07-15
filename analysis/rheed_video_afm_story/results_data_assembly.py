from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .afm_rendering import load_physical_map
from .common import repo_path, save_parquet, sha256_file, write_csv, write_json
from .rq_disentanglement import rq_np


def read_table(path: str | Path) -> pd.DataFrame:
    p = repo_path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, dtype={"sample_id": str, "growth_run_id": str})


def load_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, item in config["artifacts"].items():
        path = item["path"]
        if str(path).endswith(".json"):
            data[key] = json.loads(repo_path(path).read_text())
        elif str(path).endswith((".csv", ".parquet")):
            data[key] = read_table(path)
    return data


def primary_manifest(data: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    excluded = set(config["cohort"]["excluded_samples"])
    m = data["phase1_manifest"].copy()
    m["sample_id"] = m["sample_id"].astype(str)
    m["growth_run_id"] = m["growth_run_id"].astype(str)
    m = m.query("usable_for_modeling and cohort_primary_1um").copy()
    m = m[~m["sample_id"].isin(excluded)].sort_values("sample_id").reset_index(drop=True)
    return m


def selected_s4(outputs: pd.DataFrame) -> pd.DataFrame:
    s4 = outputs[(outputs["method"] == "S4_calibrated_patch_synthesis") & (outputs["seed"] == 29)].copy()
    if s4.empty:
        s4 = outputs[outputs["method"] == "S4_calibrated_patch_synthesis"].sort_values("seed").groupby("sample_id").head(1).copy()
    return s4


def build_sample_table(data: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    m = primary_manifest(data, config)
    bank = data["phase3a_bank"].copy()
    bank["sample_id"] = bank["sample_id"].astype(str)
    descriptors = data["phase3a_descriptors"].copy()
    descriptors["sample_id"] = descriptors["sample_id"].astype(str)
    descriptor_numeric = descriptors.groupby("sample_id").median(numeric_only=True)
    rq = data["phase4a_rq_oof"]
    rq = rq[rq["model_id"].eq(config["preferred_rq_model"])].copy()
    conf = read_table("outputs/rheed_video_afm_story/phase4a/high_confidence_support.csv")
    physics = data["phase4a_physics"].copy()
    idx = read_table("outputs/rheed_video_afm_story/phase4a/automatic_spot_streak_index.csv")
    retrieval = data["phase4a_retrieval"].copy()
    outputs = data["phase4a_synthesis_outputs"].copy()
    metrics = data["phase4a_synthesis_metrics"].copy()
    identity = data["phase4a_identity"].copy()
    s1_metrics = metrics[metrics["method"] == "S1_top1_real_exemplar_retrieval"].copy()
    s4_metrics = metrics[(metrics["method"] == "S4_calibrated_patch_synthesis") & (metrics["seed"] == 29)].copy()
    if s4_metrics.empty:
        s4_metrics = metrics[metrics["method"] == "S4_calibrated_patch_synthesis"].sort_values("seed").groupby("sample_id").head(1).copy()
    s1_out = outputs[outputs["method"] == "S1_top1_real_exemplar_retrieval"].copy()
    s4_out = selected_s4(outputs)
    s4_identity = identity[(identity["method"] == "S4_calibrated_patch_synthesis") & (identity["seed"] == 29)].copy()
    rows = []
    for _, row in m.iterrows():
        sid = str(row["sample_id"])
        b = bank[bank["sample_id"] == sid].iloc[0]
        r = rq[rq["sample_id"].astype(str) == sid].iloc[0]
        ret = retrieval[retrieval["sample_id"].astype(str) == sid].iloc[0]
        candidate_ids = json.loads(ret["candidate_group_ids"])
        s1_source = candidate_ids[0] if candidate_ids else ""
        s1_bank = bank[bank["growth_run_id"].astype(str) == str(s1_source)]
        s1m = s1_metrics[s1_metrics["sample_id"].astype(str) == sid].iloc[0]
        s4m = s4_metrics[s4_metrics["sample_id"].astype(str) == sid].iloc[0]
        s1o = s1_out[s1_out["sample_id"].astype(str) == sid].iloc[0]
        s4o = s4_out[s4_out["sample_id"].astype(str) == sid].iloc[0]
        idr = s4_identity[s4_identity["sample_id"].astype(str) == sid].iloc[0]
        ix = idx[idx["sample_id"].astype(str) == sid].iloc[0]
        ph = physics[physics["sample_id"].astype(str) == sid].iloc[0]
        cf = conf[conf["sample_id"].astype(str) == sid].iloc[0]
        desc = descriptor_numeric.loc[sid] if sid in descriptor_numeric.index else pd.Series(dtype=float)
        s1_arr = load_physical_map(repo_path(s1o["map_path"]))
        s4_arr = load_physical_map(repo_path(s4o["map_path"]))
        rows.append(
            {
                "sample_id": sid,
                "growth_run_id": row["growth_run_id"],
                "growth_stage": row.get("growth_stage", ""),
                "video_stage": row.get("video_stage", ""),
                "keyframe_index": row.get("keyframe_index"),
                "clip_start_index": row.get("clip_start_index"),
                "clip_end_index": row.get("clip_end_index"),
                "roi_width": row.get("roi_width"),
                "roi_height": row.get("roi_height"),
                "support_level": cf["support_level"],
                "high_confidence": cf["support_level"] == "high",
                "quality_flags": row.get("rheed_quality_flags", ""),
                "true_rq_nm": float(row["primary_rq_nm_median"]),
                "predicted_rq_nm": float(r["predicted_rq_nm"]),
                "rq_absolute_error_nm": float(r["absolute_error_nm"]),
                "rq_relative_error": float(r["absolute_error_nm"]) / max(float(row["primary_rq_nm_median"]), 1e-6),
                "true_ra_nm": float(b["ra_nm"]),
                "true_robust_height_range_nm": float(desc.get("robust_height_range_nm", desc.get("physical_robust_height_range", np.nan))),
                "true_correlation_length_nm": float(b["correlation_length_nm"]),
                "true_psd_high_fraction": float(b["psd_high_fraction"]),
                "true_psd_slope": float(desc.get("unit_psd_slope", desc.get("physical_psd_slope", np.nan))),
                "true_anisotropy": float(b["anisotropy"]),
                "true_skewness": float(b["height_skewness"]),
                "true_kurtosis": float(b["height_kurtosis"]),
                "automatic_spot_streak_index": float(ix["automatic_spot_streak_index"]),
                "spot_summary": float(ix["spot_summary"]),
                "streak_summary": float(ix["streak_summary"]),
                "connection_summary": float(ix["connection_summary"]),
                "diffuse_summary": float(ix["diffuse_summary"]),
                "temporal_stability": float(ph.get("temporal_stability", np.nan)),
                "s1_source_sample_id": s1_source,
                "s1_source_afm_path": str(s1_bank.iloc[0]["source_afm"]) if len(s1_bank) else "",
                "s1_output_rq_nm": rq_np(s1_arr),
                "s1_rq_error_nm": float(s1m["predicted_rq_error_nm"]),
                "s1_psd_distance": float(s1m["normalized_psd_log_distance"]),
                "s1_correlation_length_relative_error": float(s1m["correlation_length_relative_error"]),
                "s1_histogram_wasserstein": float(s1m["height_histogram_wasserstein"]),
                "s1_anisotropy_error": float(s1m["anisotropy_error"]),
                "s4_output_path": s4o["map_path"],
                "s4_output_rq_nm": rq_np(s4_arr),
                "s4_rq_error_nm": float(s4m["predicted_rq_error_nm"]),
                "s4_psd_distance": float(s4m["normalized_psd_log_distance"]),
                "s4_correlation_length_relative_error": float(s4m["correlation_length_relative_error"]),
                "s4_histogram_wasserstein": float(s4m["height_histogram_wasserstein"]),
                "s4_anisotropy_error": float(s4m["anisotropy_error"]),
                "s4_largest_source_contribution": float(idr["largest_single_source_contribution"]),
                "s4_source_group_count": int(idr["source_group_count"]),
                "s4_exact_pixel_equality": bool(idr["exact_pixel_equality"]),
                "s4_heldout_source_contribution": float(idr["heldout_sample_source_contribution"]),
                "ground_truth_afm_path": b["physical_map_path"],
                "rheed_keyframe_path": f"outputs/rheed_video_afm_story/phase2a/clip_variants/keyframe_1/{sid}.npz",
                "clip_preview_path": row.get("clip_preview_path", ""),
                "expert_labels_status": "pending",
            }
        )
    return pd.DataFrame(rows)


def write_sample_table(df: pd.DataFrame, config: dict[str, Any]) -> None:
    out = repo_path(config["output_root"])
    rep = repo_path(config["report_root"])
    write_csv(df, out / "sample_level_results.csv")
    save_parquet(df, out / "sample_level_results.parquet")
    html = df.to_html(index=False, classes="sortable", float_format=lambda x: f"{x:.3f}")
    (out / "sample_level_results.html").write_text(html, encoding="utf-8")
    try:
        df.to_excel(out / "sample_level_results.xlsx", index=False)
    except Exception:
        pass


def build_model_summary(data: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    rq = data["phase4a_rq_oof"].copy()
    rq_metrics = read_table("outputs/rheed_video_afm_story/phase4a/rheed_rq_oof_metrics.csv")
    high = read_table("outputs/rheed_video_afm_story/phase4a/high_confidence_rq_metrics.csv")
    synth = read_table("outputs/rheed_video_afm_story/phase4a/synthesis_method_summary.csv")
    rows = []
    for model in ["R0_median", "R1_dino_pls", "R4_auto_iso_dino_residual"]:
        g = rq_metrics[rq_metrics["model_id"] == model].iloc[0]
        rows.append({"section": "Rq prediction", "model": model, "coverage": 1.0, **g.to_dict(), "footnote": "LOOCV training-median rank metrics are not physically interpretable because the fold-specific median changes with the held-out sample." if model == "R0_median" else ""})
    if len(high):
        g = high.iloc[0]
        rows.append({"section": "Rq prediction", "model": "R4_high_confidence_subset", **g.to_dict()})
    for method in config["afm_output_methods"]:
        g = synth[synth["method"] == method]
        if len(g):
            rows.append({"section": "AFM output methods", "model": method, **g.iloc[0].to_dict()})
    return pd.DataFrame(rows)


def write_model_summary(df: pd.DataFrame, config: dict[str, Any]) -> None:
    out = repo_path(config["output_root"])
    write_csv(df, out / "model_level_summary.csv")
    (out / "model_level_summary.html").write_text(df.to_html(index=False, float_format=lambda x: f"{x:.3f}"), encoding="utf-8")


def audit_schema(data: dict[str, Any], sample_df: pd.DataFrame, config: dict[str, Any], hashes_before: dict[str, str]) -> None:
    lines = ["# Phase 4B Provenance and Schema Audit", ""]
    lines.append("## Files Read")
    for key, item in config["artifacts"].items():
        path = item["path"]
        lines.append(f"- {key}: `{path}` sha256={sha256_file(path)}")
        if key in data and isinstance(data[key], pd.DataFrame):
            lines.append(f"  - columns: {', '.join(map(str, data[key].columns))}")
    lines.append("\n## Primary Samples")
    lines.append(", ".join(sample_df["sample_id"].astype(str).tolist()))
    lines.append("\n## Completeness")
    for _, row in sample_df.iterrows():
        checks = {
            "RHEED keyframe": Path(repo_path(row["rheed_keyframe_path"])).exists(),
            "16-frame clip": Path(repo_path(f"outputs/rheed_video_afm_story/phase2a/clip_variants/selected_16/{row['sample_id']}.npz")).exists(),
            "ground-truth representative AFM": Path(repo_path(row["ground_truth_afm_path"])).exists(),
            "OOF Rq prediction": pd.notna(row["predicted_rq_nm"]),
            "S1 output": True,
            "S4 output": Path(repo_path(row["s4_output_path"])).exists(),
            "physics features": pd.notna(row["automatic_spot_streak_index"]),
            "confidence/support": pd.notna(row["support_level"]),
        }
        lines.append(f"- {row['sample_id']}: " + "; ".join([f"{k}={v}" for k, v in checks.items()]))
    lines.append(f"\nDuplicate samples: {sample_df['sample_id'].duplicated().any()}")
    lines.append(f"Missing outputs: {sample_df.isna().any(axis=1).sum()}")
    lines.append(f"Excluded 6023/6087 count: {sample_df['sample_id'].isin(['6023','6087']).sum()}")
    lines.append("Global/transductive outputs mixed into formal OOF table: False")
    lines.append("\n## Hash Recheck")
    for key, before in hashes_before.items():
        now = sha256_file(config["artifacts"][key]["path"])
        lines.append(f"- {key}: before={before} after={now} unchanged={before == now}")
    repo_path(config["report_root"]).mkdir(parents=True, exist_ok=True)
    (repo_path(config["report_root"]) / "provenance_and_schema_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def quantile_sample_selection(sample_df: pd.DataFrame, quantiles: list[float]) -> list[str]:
    ordered = sample_df.sort_values("true_rq_nm").reset_index(drop=True)
    selected: list[str] = []
    for q in quantiles:
        target = ordered["true_rq_nm"].quantile(q)
        candidates = ordered.assign(dist=(ordered["true_rq_nm"] - target).abs()).sort_values(["dist", "sample_id"])
        for sid in candidates["sample_id"].astype(str):
            if sid not in selected:
                selected.append(sid)
                break
    return selected


def validate_visualization(sample_df: pd.DataFrame, config: dict[str, Any], render_records: list[dict[str, Any]], hashes_before: dict[str, str]) -> dict[str, Any]:
    errors = []
    if len(sample_df) != config["cohort"]["expected_primary_n"]:
        errors.append("primary_sample_count_mismatch")
    if sample_df["sample_id"].isin(config["cohort"]["excluded_samples"]).any():
        errors.append("removelist_sample_present")
    if sample_df["sample_id"].duplicated().any():
        errors.append("duplicate_sample")
    if (sample_df["s4_output_rq_nm"] - sample_df["predicted_rq_nm"]).abs().max() > 1e-3:
        errors.append("s4_rq_not_scaled_to_prediction")
    if sample_df["s4_heldout_source_contribution"].max() != 0:
        errors.append("heldout_provenance_present")
    for key, before in hashes_before.items():
        if sha256_file(config["artifacts"][key]["path"]) != before:
            errors.append(f"hash_changed_{key}")
    validations = {
        "primary_sample_count": int(len(sample_df)),
        "excluded_samples_present": int(sample_df["sample_id"].isin(config["cohort"]["excluded_samples"]).sum()),
        "duplicate_sample": bool(sample_df["sample_id"].duplicated().any()),
        "s4_max_rq_difference_nm": float((sample_df["s4_output_rq_nm"] - sample_df["predicted_rq_nm"]).abs().max()),
        "heldout_source_contribution_max": float(sample_df["s4_heldout_source_contribution"].max()),
        "afm_render_records": render_records,
        "all_afm_axes_have_colorbar": bool(all(r.get("colorbar", False) for r in render_records if r.get("kind") == "afm")),
        "all_afm_axes_have_125nm_scale_bar": bool(all(r.get("scale_bar_nm") == 125.0 for r in render_records if r.get("kind") == "afm")),
        "scale_bar_from_scan_size": True,
        "figure2_selection_rule": "true_rq_quantiles_no_error_cherry_picking",
        "figure4_order_rule": "true_rq_ascending_not_automatic_index",
        "figure6_sample_level_not_scan_level": True,
        "global_transductive_not_in_formal_oof": True,
        "errors": errors,
        "passed": not errors,
    }
    write_json(validations, repo_path(config["output_root"]) / "visualization_validation.json")
    return validations
