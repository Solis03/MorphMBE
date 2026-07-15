from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .afm_targets import build_sample_targets, collect_afm_scans
from .baseline_features import keyframe_feature_vector, temporal_feature_vector
from .baseline_rq import run_oof_baselines
from .clip_cache import build_clip_cache, frame_path
from .common import (
    display_path,
    ensure_dirs,
    infer_material,
    load_config,
    parse_stage,
    read_id_list,
    repo_path,
    save_parquet,
    sha256_file,
    sha256_object,
    write_csv,
    write_json,
)
from .visualize_phase1 import baseline_figures, summary_figures


def load_selected_manifest(config: dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(repo_path(config["source_manifest_path"]), dtype={"sample_id": str})
    int_cols = [
        "keyframe_index",
        "clip_frame_count",
        "roi_x",
        "roi_y",
        "roi_width",
        "roi_height",
        "source_width",
        "source_height",
        "clip_start_index",
        "clip_end_index",
        "actual_clip_frame_count",
    ]
    for col in int_cols:
        df[col] = df[col].astype(int)
    return df


def validate_selected(df: pd.DataFrame, discarded_ids: set[str], removelist_ids: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if len(df) != 27 or df["sample_id"].nunique() != 27:
        errors.append(f"selected manifest expected 27 unique samples, found rows={len(df)} unique={df['sample_id'].nunique()}")
    duplicated = df["sample_id"][df["sample_id"].duplicated()].tolist()
    if duplicated:
        errors.append(f"duplicate sample_id values: {duplicated}")
    videos_per_sample = df.groupby("sample_id")["video_id"].nunique()
    if not (videos_per_sample == 1).all():
        errors.append("one or more samples have multiple selected videos")
    discarded_overlap = sorted(set(df["sample_id"]) & discarded_ids)
    if discarded_overlap:
        errors.append(f"discarded samples present in selected manifest: {discarded_overlap}")
    removelist_overlap = sorted(set(df["sample_id"]) & removelist_ids)
    if removelist_overlap:
        warnings.append(f"selected samples in canonical removelist and excluded from modeling: {removelist_overlap}")
    for row in df.to_dict("records"):
        indices = list(range(row["clip_start_index"], row["clip_end_index"] + 1))
        if len(indices) != int(row["actual_clip_frame_count"]):
            errors.append(f"{row['sample_id']}: actual_clip_frame_count mismatch")
        if len(indices) != 16:
            errors.append(f"{row['sample_id']}: clip is not 16 frames")
        if row["keyframe_index"] not in indices:
            errors.append(f"{row['sample_id']}: keyframe outside clip")
        if row["roi_x"] < 0 or row["roi_y"] < 0 or row["roi_width"] <= 0 or row["roi_height"] <= 0:
            errors.append(f"{row['sample_id']}: invalid ROI dimensions")
        if row["roi_x"] + row["roi_width"] > row["source_width"] or row["roi_y"] + row["roi_height"] > row["source_height"]:
            errors.append(f"{row['sample_id']}: ROI outside source bounds")
        for index in indices:
            path = frame_path(row["frames_dir"], index)
            if not path.exists():
                errors.append(f"{row['sample_id']}: missing frame {display_path(path)}")
        keyframe = frame_path(row["frames_dir"], row["keyframe_index"])
        if keyframe.exists():
            with Image.open(keyframe) as image:
                if image.size != (row["source_width"], row["source_height"]):
                    errors.append(f"{row['sample_id']}: keyframe shape mismatch {image.size}")
    return {"errors": errors, "warnings": warnings, "removelist_overlap": removelist_overlap}


def build_manifest(config: dict[str, Any], selected_df: pd.DataFrame, target_df: pd.DataFrame, scan_df: pd.DataFrame, cache_df: pd.DataFrame, quality_df: pd.DataFrame, removelist_ids: set[str]) -> pd.DataFrame:
    df = selected_df.copy()
    df["growth_run_id"] = df["sample_id"]
    df["sample_group_id"] = df["sample_id"]
    df["source_video"] = df["source_video"].map(display_path)
    df["frames_dir"] = df["frames_dir"].map(display_path)
    df["metadata_path"] = df["metadata_path"].map(display_path)
    df["keyframe_offset_in_clip"] = df["keyframe_index"] - df["clip_start_index"]
    df["growth_stage"] = df["video_id"].map(parse_stage)
    df["video_stage"] = df["video_id"].map(parse_stage)
    afm_ids_by_sample = scan_df.groupby("sample_id")["afm_file_id"].apply(list).to_dict() if not scan_df.empty else {}
    df["material"] = [infer_material(sid, afm_ids_by_sample.get(sid, [])) for sid in df["sample_id"]]
    df["substrate"] = "unknown"
    df["camera_or_tool_id"] = "unknown"
    df["fps"] = np.nan
    df = df.merge(target_df, on="sample_id", how="left")
    df = df.merge(cache_df, on="sample_id", how="left")
    q_cols = ["sample_id", "quality_flags"]
    df = df.merge(quality_df[q_cols].rename(columns={"quality_flags": "rheed_quality_flags"}), on="sample_id", how="left")
    df["excluded_by_removelist"] = df["sample_id"].isin(removelist_ids)
    df["exclusion_reason"] = np.where(df["excluded_by_removelist"], "canonical_removelist_conflict", "")
    df["usable_for_modeling"] = ~df["excluded_by_removelist"]
    df["rheed_quality_pass"] = True
    df["manifest_source_hash"] = sha256_file(config["source_manifest_path"])
    df["removelist_hash"] = sha256_file(config["removelist_path"])
    df["config_hash"] = sha256_object(config)
    bool_cols = ["primary_afm_available", "cohort_primary_1um", "cohort_exploratory_best_available"]
    for col in bool_cols:
        df[col] = df[col].fillna(False).astype(bool)
    return df


def feature_matrices(manifest: pd.DataFrame) -> dict[str, tuple[list[str], np.ndarray]]:
    key_rows: list[np.ndarray] = []
    temporal_rows: list[np.ndarray] = []
    key_names: list[str] | None = None
    temporal_names: list[str] | None = None
    for row in manifest.to_dict("records"):
        data = np.load(repo_path(row["clip_cache_path"]))
        frames = data["frames_uint8"]
        key_offset = int(data["keyframe_offset"])
        names, values = keyframe_feature_vector(frames, key_offset)
        key_names = names if key_names is None else key_names
        key_rows.append(values)
        names, values = temporal_feature_vector(frames)
        temporal_names = names if temporal_names is None else temporal_names
        temporal_rows.append(values)
    return {
        "B1_keyframe": (key_names or [], np.vstack(key_rows)),
        "B2_temporal": (temporal_names or [], np.vstack(temporal_rows)),
    }


def write_audit_report(path: Path, selected_df: pd.DataFrame, manifest: pd.DataFrame, scan_df: pd.DataFrame, validation: dict[str, Any], config: dict[str, Any]) -> None:
    primary = manifest[manifest["cohort_primary_1um"] & manifest["usable_for_modeling"]]
    valid = manifest[manifest["cohort_exploratory_best_available"] & manifest["usable_for_modeling"]]
    scan_counts = scan_df.groupby("sample_id").size() if not scan_df.empty else pd.Series(dtype=int)
    lines = [
        "# Phase 1 Repository Audit",
        "",
        f"- Selected manifest: `{config['source_manifest_path']}`",
        f"- Selected manifest schema: `{list(selected_df.columns)}`",
        f"- Selected sample count: {selected_df['sample_id'].nunique()}",
        f"- Selected sample IDs: {', '.join(selected_df['sample_id'].tolist())}",
        f"- Canonical removelist: `{config['removelist_path']}`",
        f"- Removelist hash: `{sha256_file(config['removelist_path'])}`",
        f"- Discarded overlap in selected manifest: {sorted(set(selected_df['sample_id']) & read_id_list(config['discarded_list_path']))}",
        f"- Removelist conflict in selected manifest: {validation['removelist_overlap']}",
        f"- Modeling-eligible selected samples after removelist: {int(manifest['usable_for_modeling'].sum())}",
        f"- Selected samples with any valid AFM: {int(scan_df['sample_id'].nunique()) if not scan_df.empty else 0}",
        f"- Modeling-eligible samples with valid AFM: {int(valid.shape[0])}",
        f"- Samples with primary 1 x 1 um AFM: {int(primary.shape[0])}",
        f"- AFM scan count per sample range: {int(scan_counts.min()) if len(scan_counts) else 0} to {int(scan_counts.max()) if len(scan_counts) else 0}",
        "- AFM target source: recomputed Rq from `data/plane_corrected_afm/*/*_plane_corrected.npy`; descriptor/metadata values are retained in `afm_scan_audit.csv` for comparison.",
        f"- AFM unit conflicts: {int(scan_df['quality_flags'].fillna('').str.contains('height_unit_not_nm').sum()) if not scan_df.empty else 0}",
        f"- Rq existing/recomputed conflicts > 1e-5 nm: {int(scan_df['quality_flags'].fillna('').str.contains('rq_existing_mismatch').sum()) if not scan_df.empty else 0}",
        f"- Pairing conflicts: {validation['errors']}",
        "",
        "## Per-Sample AFM Scan Counts",
        "",
        scan_counts.astype(int).to_string() if len(scan_counts) else "No AFM scans found.",
        "",
        "## Warnings",
        "",
        "\n".join(f"- {w}" for w in validation["warnings"]) or "- None",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_phase_report(path: Path, manifest: pd.DataFrame, metrics: pd.DataFrame, neighbor: pd.DataFrame, validation: dict[str, Any], outputs: dict[str, Any]) -> None:
    primary = manifest[manifest["cohort_primary_1um"] & manifest["usable_for_modeling"]].copy()
    rq = primary["primary_rq_nm_median"].dropna()
    lines = [
        "# Phase 1 RHEED Video to AFM Report",
        "",
        "## Cohort",
        "",
        f"- Frozen selected samples: {manifest['sample_id'].nunique()}",
        f"- Modeling-eligible samples after removelist: {int(manifest['usable_for_modeling'].sum())}",
        f"- Primary 1 x 1 um cohort: {len(primary)}",
        f"- Exploratory best-available cohort: {int((manifest['cohort_exploratory_best_available'] & manifest['usable_for_modeling']).sum())}",
        f"- Removelist conflicts excluded from modeling: {validation['removelist_overlap']}",
        f"- Rq range/median: {rq.min():.3f} to {rq.max():.3f} nm, median {rq.median():.3f} nm" if len(rq) else "- Rq range/median: unavailable",
        "",
        "## Baseline Metrics",
        "",
        markdown_table(metrics) if not metrics.empty else "Baseline skipped.",
        "",
        "## Leakage Audit",
        "",
        f"- KNN neighbor rows: {len(neighbor)}",
        f"- KNN neighbor leakage-free: {bool(neighbor['leakage_free'].all()) if not neighbor.empty else True}",
        "",
        "## Key Risks",
        "",
        f"- Selected sample(s) {', '.join(validation['removelist_overlap'])} are present in the canonical removelist and were excluded from modeling.",
        "- Growth/video stage and material are inferred from local metadata/file names where explicit fields are absent.",
        "- The primary cohort is small; OOF metrics should be treated as screening signals, not definitive model evidence.",
        "- Best-available exploratory scans are reported separately and not mixed into the primary baseline.",
        "",
        "## Outputs",
        "",
        "\n".join(f"- `{v}`" for v in outputs.values() if isinstance(v, str)),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for record in df.to_dict("records"):
        values = []
        for col in cols:
            value = record[col]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    output_root, report_root = ensure_dirs(config)
    selected = load_selected_manifest(config)
    discarded_ids = read_id_list(config["discarded_list_path"])
    removelist_ids = read_id_list(config["removelist_path"])
    validation = validate_selected(selected, discarded_ids, removelist_ids)
    if validation["errors"]:
        raise RuntimeError("; ".join(validation["errors"]))
    selected_ids = set(selected["sample_id"].astype(str))
    scan_df = collect_afm_scans(config, selected_ids)
    unit_conflicts = scan_df["quality_flags"].fillna("").str.contains("height_unit_not_nm").sum() if not scan_df.empty else 0
    rq_conflicts = scan_df["quality_flags"].fillna("").str.contains("rq_existing_mismatch").sum() if not scan_df.empty else 0
    if unit_conflicts:
        write_csv(scan_df, output_root / "afm_scan_audit.csv")
        raise RuntimeError("AFM height unit conflict found; refusing target modeling.")
    if rq_conflicts:
        write_csv(scan_df, output_root / "afm_scan_audit.csv")
        raise RuntimeError("Existing and recomputed AFM Rq differ materially; refusing baseline.")
    target_df, scan_df = build_sample_targets(
        scan_df,
        float(config["primary_afm_scan_size_um"]),
        float(config["primary_afm_scan_size_tolerance_um"]),
    )
    cache_df, quality_df = build_clip_cache(selected, output_root, report_root, int(config["rheed_output_size"]))
    manifest = build_manifest(config, selected, target_df, scan_df, cache_df, quality_df, removelist_ids)
    write_csv(scan_df, output_root / "afm_scan_audit.csv")
    write_csv(quality_df, output_root / "rheed_quality_metrics.csv")
    manifest_csv = output_root / "modeling_manifest.csv"
    write_csv(manifest, manifest_csv)
    manifest_parquet = save_parquet(manifest, output_root / "modeling_manifest.parquet")
    write_audit_report(report_root / "repo_audit.md", selected, manifest, scan_df, validation, config)
    summary_figures(manifest, quality_df, scan_df, report_root)
    primary = manifest[manifest["usable_for_modeling"] & manifest["cohort_primary_1um"]].copy().reset_index(drop=True)
    pred_df = pd.DataFrame()
    metrics_df = pd.DataFrame()
    neighbor_df = pd.DataFrame()
    if len(primary) >= 5:
        features = feature_matrices(primary)
        pred_df, metrics_df, neighbor_df = run_oof_baselines(primary, features, config)
        write_csv(pred_df, output_root / "oof_predictions.csv")
        write_csv(metrics_df, output_root / "baseline_metrics.csv")
        write_csv(neighbor_df, output_root / "baseline_neighbor_audit.csv")
        baseline_figures(pred_df, metrics_df, neighbor_df, primary, report_root)
    outputs = {
        "modeling_manifest_csv": display_path(manifest_csv),
        "modeling_manifest_parquet": manifest_parquet,
        "afm_scan_audit": display_path(output_root / "afm_scan_audit.csv"),
        "rheed_quality_metrics": display_path(output_root / "rheed_quality_metrics.csv"),
        "oof_predictions": display_path(output_root / "oof_predictions.csv"),
        "baseline_metrics": display_path(output_root / "baseline_metrics.csv"),
        "baseline_neighbor_audit": display_path(output_root / "baseline_neighbor_audit.csv"),
        "repo_audit": display_path(report_root / "repo_audit.md"),
        "phase1_report": display_path(report_root / "phase1_report.md"),
    }
    write_phase_report(report_root / "phase1_report.md", manifest, metrics_df, neighbor_df, validation, outputs)
    summary = {
        "selected_sample_ids": selected["sample_id"].tolist(),
        "selected_schema": list(selected.columns),
        "removelist_path": config["removelist_path"],
        "removelist_hash": sha256_file(config["removelist_path"]),
        "removelist_conflicts": validation["removelist_overlap"],
        "modeling_eligible_samples": int(manifest["usable_for_modeling"].sum()),
        "primary_1um_cohort_n": int((manifest["usable_for_modeling"] & manifest["cohort_primary_1um"]).sum()),
        "exploratory_best_available_cohort_n": int((manifest["usable_for_modeling"] & manifest["cohort_exploratory_best_available"]).sum()),
        "rq_primary_min_nm": float(primary["primary_rq_nm_median"].min()) if not primary.empty else None,
        "rq_primary_max_nm": float(primary["primary_rq_nm_median"].max()) if not primary.empty else None,
        "rq_primary_median_nm": float(primary["primary_rq_nm_median"].median()) if not primary.empty else None,
        "clip_cache_verified_all": bool(manifest["crop_verified"].all()),
        "stage_distribution": manifest[manifest["usable_for_modeling"]]["video_stage"].value_counts().to_dict(),
        "material_distribution": manifest[manifest["usable_for_modeling"]]["material"].value_counts().to_dict(),
        "baseline_metrics": metrics_df.to_dict("records"),
        "knn_neighbor_leakage_free": bool(neighbor_df["leakage_free"].all()) if not neighbor_df.empty else True,
        "outputs": outputs,
    }
    write_json(summary, output_root / "phase1_summary.json")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build phase-1 RHEED-video-to-AFM modeling outputs.")
    parser.add_argument("--config", default="configs/rheed_video_afm_story_phase1.yaml")
    args = parser.parse_args()
    summary = run(args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
