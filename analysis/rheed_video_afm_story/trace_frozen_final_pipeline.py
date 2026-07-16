#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.rheed_video_afm_story.rq_disentanglement import physical_from_q, project_unit_rq_np, unit_shape  # noqa: E402


UNKNOWN = "UNKNOWN - not recoverable from current frozen artifact"

DESCRIPTOR_COLS = [
    "rq_nm",
    "ra_nm",
    "robust_height_range_nm",
    "psd_low_fraction",
    "psd_mid_fraction",
    "psd_high_fraction",
    "psd_slope",
    "correlation_length_nm",
    "anisotropy",
    "height_skewness",
    "height_kurtosis",
]


@dataclass
class Paths:
    out: Path
    freeze: Path
    phase6a: Path
    phase7a: Path
    phase7b: Path
    phase2a: Path
    canonical_index: Path
    embedding_npz: Path
    phase7_config: Path


def repo(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: serialize_cell(row.get(k, "")) for k in fieldnames})


def serialize_cell(value: Any) -> str:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.12g}"
    return "" if value is None else str(value)


def md_table(rows: list[dict[str, Any]], fieldnames: list[str] | None = None, max_rows: int | None = None) -> str:
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row})
    shown = rows if max_rows is None else rows[:max_rows]
    def esc(x: Any) -> str:
        return serialize_cell(x).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(fieldnames) + " |", "| " + " | ".join(["---"] * len(fieldnames)) + " |"]
    for row in shown:
        lines.append("| " + " | ".join(esc(row.get(k, "")) for k in fieldnames) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append(f"\nShowing {max_rows} of {len(rows)} rows.")
    return "\n".join(lines) + "\n"


def write_table_pair(base: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None, max_md_rows: int | None = None) -> None:
    write_csv_rows(base.with_suffix(".csv"), rows, fieldnames)
    write_text(base.with_suffix(".md"), md_table(rows, fieldnames, max_md_rows))


def arr_stats(arr: np.ndarray) -> dict[str, Any]:
    a = np.asarray(arr)
    finite = np.asarray(a, dtype=float)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "min": float(np.nanmin(finite)) if finite.size else None,
        "max": float(np.nanmax(finite)) if finite.size else None,
        "mean": float(np.nanmean(finite)) if finite.size else None,
        "std": float(np.nanstd(finite)) if finite.size else None,
    }


def trace_row(
    node_id: str,
    sample_id: str,
    track: str,
    operation: str,
    tensor_name: str,
    value: np.ndarray | list[Any] | None,
    source_file: str,
    source_function: str,
    notes: str = "",
    device: str = "cpu",
) -> dict[str, Any]:
    if value is None:
        stats = {"shape": UNKNOWN, "dtype": UNKNOWN, "min": "", "max": "", "mean": "", "std": ""}
    else:
        stats = arr_stats(np.asarray(value))
    return {
        "node_id": node_id,
        "sample_id": sample_id,
        "track": track,
        "operation": operation,
        "tensor_name": tensor_name,
        "shape": stats["shape"],
        "dtype": stats["dtype"],
        "device": device,
        "min": stats["min"],
        "max": stats["max"],
        "mean": stats["mean"],
        "std": stats["std"],
        "source_file": source_file,
        "source_function": source_function,
        "notes": notes,
    }


def load_paths() -> Paths:
    latest = (ROOT / "paper_freeze/LATEST_FREEZE.txt").read_text(encoding="utf-8").strip()
    freeze = repo(latest)
    phase6a = ROOT / "outputs/rheed_video_afm_story/variants/afm_second_order_y2_v1/phase6a_exhaustive_discovery"
    phase7a = ROOT / "outputs/rheed_video_afm_story/variants/afm_second_order_y2_v1/phase7a_reconstruction_first"
    phase7b = ROOT / "outputs/rheed_video_afm_story/variants/afm_second_order_y2_v1/phase7b_fixed_method_atlases"
    return Paths(
        out=ROOT / "reports/final_algorithm_audit",
        freeze=freeze,
        phase6a=phase6a,
        phase7a=phase7a,
        phase7b=phase7b,
        phase2a=ROOT / "outputs/rheed_video_afm_story/phase2a",
        canonical_index=phase6a / "canonical_index/canonical_sample_index.csv",
        embedding_npz=ROOT / "outputs/rheed_video_afm_story/phase2a/embeddings/dino_vits14__keyframe_1__raw_luminance.npz",
        phase7_config=ROOT / "configs/rheed_video_afm_story_phase7a.yaml",
    )


def load_artifacts(paths: Paths) -> dict[str, Any]:
    ensemble = read_json(paths.freeze / "12_FULL_COHORT_DEPLOYMENT/quantitative_model/ensemble_definition.json")
    model_registry = read_json(paths.freeze / "12_FULL_COHORT_DEPLOYMENT/quantitative_model/model_registry.json")
    strict_visual = read_json(paths.freeze / "05_MODELS/strict_visual/registry.json")
    retrieval_config = read_json(paths.freeze / "12_FULL_COHORT_DEPLOYMENT/visual_model/retrieval_config.json")
    encoder = read_json(paths.freeze / "12_FULL_COHORT_DEPLOYMENT/encoder/encoder_identifier.json")
    conditions = pd.read_csv(paths.phase7a / "condition_vectors/phase7_condition_vectors.csv", dtype={"sample_id": str})
    target_table = pd.read_csv(paths.phase6a / "target_variants/target_variant_table.csv", dtype={"sample_id": str})
    strict_pred = pd.read_csv(paths.phase6a / "finalists/ensemble_oof_predictions.csv", dtype={"sample_id": str})
    desc_pred = pd.read_csv(paths.phase6a / "all_descriptor_predictions.csv", dtype={"sample_id": str})
    proto_pred = pd.read_csv(paths.phase6a / "prototype_predictions.csv", dtype={"sample_id": str})
    canonical = pd.read_csv(paths.canonical_index, dtype={"sample_id": str, "growth_run_id": str})
    scans = pd.read_csv(ROOT / "outputs/rheed_video_afm_story/variants/afm_second_order_y2_v1/targets/second_order_afm_descriptors.csv", dtype={"sample_id": str, "scan_id": str})
    bank = pd.read_csv(paths.freeze / "12_FULL_COHORT_DEPLOYMENT/visual_model/afm_bank_manifest.csv", dtype={"sample_id": str, "scan_id": str})
    phase7b_summary = pd.read_csv(paths.phase7b / "fixed_method_family_summary.csv") if (paths.phase7b / "fixed_method_family_summary.csv").exists() else pd.DataFrame()
    phase7b_audit = pd.read_csv(paths.phase7b / "method_audit.csv", dtype={"sample_id": str}) if (paths.phase7b / "method_audit.csv").exists() else pd.DataFrame()
    strict_visual_metrics = pd.read_csv(paths.phase7a / "strict_visual_metrics.csv", dtype={"sample_id": str})
    return {
        "ensemble": ensemble,
        "model_registry": model_registry,
        "strict_visual": strict_visual,
        "retrieval_config": retrieval_config,
        "encoder": encoder,
        "conditions": conditions,
        "target_table": target_table,
        "strict_pred": strict_pred,
        "desc_pred": desc_pred,
        "proto_pred": proto_pred,
        "canonical": canonical,
        "scans": scans,
        "bank": bank,
        "phase7b_summary": phase7b_summary,
        "phase7b_audit": phase7b_audit,
        "strict_visual_metrics": strict_visual_metrics,
    }


def active_ids(canonical: pd.DataFrame) -> list[str]:
    primary = canonical[canonical["is_primary"].astype(str).eq("True")].copy()
    return sorted(primary["sample_id"].astype(str).tolist())


def representative_table(scans: pd.DataFrame, canonical: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    target_by = canonical.set_index("sample_id")["second_order_rq_nm"].astype(float).to_dict()
    rows = []
    for sid, group in scans[scans["sample_id"].isin(ids)].groupby("sample_id", sort=True):
        target = target_by[sid]
        row = group.assign(_d=(group["rq_nm"].astype(float) - target).abs()).sort_values(["_d", "scan_id"]).iloc[0].copy()
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def parse_probs(value: Any) -> list[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [float(x) for x in value]
    try:
        return [float(x) for x in ast.literal_eval(str(value))]
    except Exception:
        return []


def descriptor_vector_from_row(row: pd.Series, prefix: str = "") -> np.ndarray:
    vals = []
    for col in DESCRIPTOR_COLS:
        key = f"{prefix}{col}" if prefix else col
        vals.append(float(row.get(key, np.nan)))
    arr = np.asarray(vals, dtype=float)
    med = np.nanmedian(arr) if np.isfinite(arr).any() else 0.0
    return np.where(np.isfinite(arr), arr, med)


def condition_vector(cond_row: pd.Series, q_key: str = "condition_q50_rq_nm") -> np.ndarray:
    vals = []
    for col in DESCRIPTOR_COLS:
        if col == "rq_nm":
            vals.append(float(cond_row[q_key]))
        else:
            vals.append(float(cond_row.get(f"predicted_{col}", np.nan)))
    arr = np.asarray(vals, dtype=float)
    med = np.nanmedian(arr) if np.isfinite(arr).any() else 0.0
    return np.where(np.isfinite(arr), arr, med)


def rank_a3_sources(representative: pd.DataFrame, sid: str, cond_row: pd.Series, ids: list[str], q_key: str = "condition_q50_rq_nm") -> pd.DataFrame:
    rows = representative[(representative["sample_id"].isin(ids)) & (~representative["sample_id"].eq(sid))].copy()
    cvec = condition_vector(cond_row, q_key=q_key)
    mat = np.vstack([descriptor_vector_from_row(r) for _, r in rows.iterrows()])
    scale = np.maximum(np.nanstd(mat, axis=0), 1e-6)
    desc_score = np.sqrt((((mat - cvec) / scale) ** 2).sum(axis=1))
    rq_penalty = 0.05 * (rows["rq_nm"].astype(float).to_numpy() - float(cond_row[q_key])).astype(float)
    rows["descriptor_z_euclidean"] = desc_score
    rows["rq_penalty_abs"] = np.abs(rq_penalty)
    rows["rank_score"] = desc_score + rows["rq_penalty_abs"]
    return rows.sort_values(["rank_score", "sample_id", "scan_id"]).reset_index(drop=True)


def read_map(path: str | Path) -> np.ndarray:
    arr = np.load(repo(path), allow_pickle=False).astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.all():
        fill = float(np.nanmean(arr[finite])) if finite.any() else 0.0
        arr = np.where(finite, arr, fill).astype(np.float32)
    return arr


def resize_center(arr: np.ndarray, resolution: int = 256) -> np.ndarray:
    if arr.shape != (resolution, resolution):
        from skimage.transform import resize

        arr = resize(arr.astype(float), (resolution, resolution), order=1, mode="reflect", anti_aliasing=True, preserve_range=True).astype(np.float32)
    arr = arr - float(np.mean(arr))
    return arr.astype(np.float32)


def load_embedding_for_sample(paths: Paths, sid: str) -> tuple[np.ndarray, dict[str, Any]]:
    z = np.load(paths.embedding_npz, allow_pickle=False)
    ids = [str(x) for x in z["sample_ids"].tolist()]
    idx = ids.index(sid)
    meta = {k: z[k].tolist() if hasattr(z[k], "tolist") else str(z[k]) for k in z.files if k != "embeddings"}
    return np.asarray(z["embeddings"][idx], dtype=np.float32), meta


def full_cohort_member_predictions(paths: Paths, embedding: np.ndarray, ensemble: dict[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray, float]:
    preds = []
    scaled_stack = []
    for member in ensemble["members"]:
        z = np.load(paths.freeze / "12_FULL_COHORT_DEPLOYMENT/quantitative_model" / f"{member['name']}.npz", allow_pickle=False)
        scaled = (embedding.astype(float) - z["feature_mean"]) / np.maximum(z["feature_scale"], 1e-9)
        pred = float(np.dot(scaled, z["coef"]) + float(z["intercept"]))
        scaled_stack.append(scaled.astype(np.float32))
        preds.append(
            {
                "member": member["name"],
                "trial_id": member["trial_id"],
                "model_family": member["model_family"],
                "feature_set": member["feature_set"],
                "target_variant": member["target_variant"],
                "input_dim": int(len(z["coef"])),
                "prediction_nm": pred,
            }
        )
    pred_vec = np.asarray([p["prediction_nm"] for p in preds], dtype=np.float32)
    return preds, np.vstack(scaled_stack), float(np.median(pred_vec))


def preprocess_raw_for_trace(canonical_row: pd.Series) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, str]:
    frame_path = repo(canonical_row["frames_dir"]) / f"{int(canonical_row['keyframe_index'])}.png"
    if not frame_path.exists():
        return None, None, None, str(frame_path)
    img = Image.open(frame_path)
    raw = np.asarray(img.convert("RGB"), dtype=np.uint8)
    gray = np.asarray(img.convert("RGB"), dtype=np.float32)
    gray = np.clip(np.rint(0.2126 * gray[..., 0] + 0.7152 * gray[..., 1] + 0.0722 * gray[..., 2]), 0, 255).astype(np.uint8)
    x, y, w, h = (int(canonical_row["roi_x"]), int(canonical_row["roi_y"]), int(canonical_row["roi_width"]), int(canonical_row["roi_height"]))
    crop = gray[y : y + h, x : x + w]
    return raw, gray, crop, str(frame_path)


def trace_sample(paths: Paths, artifacts: dict[str, Any], sid: str, label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    canonical = artifacts["canonical"].set_index("sample_id").loc[sid]
    conditions = artifacts["conditions"].set_index("sample_id")
    cond = conditions.loc[sid]
    ids = active_ids(artifacts["canonical"])
    representative = representative_table(artifacts["scans"], artifacts["canonical"], ids)
    strict = artifacts["strict_pred"].set_index("sample_id").loc[sid]

    raw, gray, crop, frame_path = preprocess_raw_for_trace(canonical)
    rows.append(trace_row("A01", sid, label, "read source keyframe PNG", "raw_png_rgb", raw, "clip_cache.py", "frame_path/luminance_uint8", frame_path))
    rows.append(trace_row("A02", sid, label, "convert RGB to luminance", "grayscale_uint8", gray, "clip_cache.py", "luminance_uint8"))
    rows.append(trace_row("A03", sid, label, "manual ROI crop", "roi_crop_uint8", crop, "clip_cache.py", "build_clip_cache/_read_variant_frames", f"xywh={[int(canonical['roi_x']), int(canonical['roi_y']), int(canonical['roi_width']), int(canonical['roi_height'])]}"))

    clip_path = paths.phase2a / "clip_variants/keyframe_1" / f"{sid}.npz"
    clip = np.load(clip_path, allow_pickle=False)
    frame = np.asarray(clip["frames_uint8"][0], dtype=np.uint8)
    rows.append(trace_row("A04", sid, label, "resize ROI and zero-pad to square", "keyframe_1_frames_uint8", clip["frames_uint8"], "phase2_clip_variants.py", "_read_variant_frames/build_clip_variants", str(clip_path)))
    raw_lum = frame.astype(np.float32) / 255.0
    rgb = np.repeat(raw_lum[None, None, :, :], 3, axis=1)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    norm = (rgb - mean) / std
    rows.append(trace_row("A05", sid, label, "scale uint8 to [0,1] and duplicate luminance to RGB", "dino_rgb_float", rgb, "pretrained_embeddings.py", "preprocess_frames"))
    rows.append(trace_row("A06", sid, label, "ImageNet normalization", "dino_normalized_input", norm, "pretrained_embeddings.py", "preprocess_frames", "DINO input shape [T=1,C=3,H=224,W=224]"))
    rows.append(trace_row("A07", sid, label, "DINOv2 ViT-S/14 patch projection", "dino_patch_tokens", np.zeros((1, 256, 384), dtype=np.float32), "pretrained_embeddings.py", "load_dino", "shape recovered from dinov2_vits14 identifier: 224/14=16, 16*16=256 patches"))
    rows.append(trace_row("A08", sid, label, "prepend CLS token and transformer blocks", "dino_token_sequence", np.zeros((1, 257, 384), dtype=np.float32), "pretrained_embeddings.py", "load_dino", "upstream DINO weights are not copied in freeze; tensor values are not re-executed"))
    rows.append(trace_row("A09", sid, label, "DINO CLS/frame embedding", "dino_cls_output", np.zeros((1, 384), dtype=np.float32), "pretrained_embeddings.py", "extract_embeddings", "shape recovered from DINO ViT-S/14; values not serialized separately"))
    embedding, embed_meta = load_embedding_for_sample(paths, sid)
    rows.append(trace_row("A10", sid, label, "temporal aggregation mean/std/delta/slope", "final_rheed_embedding", embedding[None, :], "pretrained_embeddings.py", "temporal_aggregate", "keyframe_1 final dim = 4 * 384 = 1536; std/delta/slope components are zero for T=1"))

    member_preds, scaled_stack, full_pred = full_cohort_member_predictions(paths, embedding, artifacts["ensemble"])
    rows.append(trace_row("Q01", sid, label, "member feature scaling", "feature_scaler_outputs", scaled_stack, "build_final_paper_freeze.py", "fit_full_cohort_quantitative", "one row per ensemble member"))
    rows.append(trace_row("Q02", sid, label, "ridge member scalar outputs", "ensemble_member_outputs", np.asarray([x["prediction_nm"] for x in member_preds], dtype=np.float32), "build_final_paper_freeze.py", "fit_full_cohort_quantitative"))
    rows.append(trace_row("Q03", sid, label, "median aggregation", "full_cohort_predicted_rq_q50", np.asarray([full_pred], dtype=np.float32), "build_final_paper_freeze.py", "fit_full_cohort_quantitative", "full-cohort in-sample for historical samples; not test performance"))
    rows.append(trace_row("Q04", sid, label, "strict OOF fold prediction", "strict_oof_predicted_rq_q50", np.asarray([float(strict["predicted_target_nm"])], dtype=np.float32), "run_phase6a.py", "finalists/ensemble_oof_predictions", "fold coefficients are not serialized in freeze"))
    rows.append(trace_row("Q05", sid, label, "strict interval quantiles", "strict_q10_q50_q90", cond[["condition_q10_rq_nm", "condition_q50_rq_nm", "condition_q90_rq_nm"]].to_numpy(dtype=np.float32), "run_phase7a.py", "build_conditions"))

    desc_vec = condition_vector(cond, q_key="condition_q50_rq_nm")
    rows.append(trace_row("B01", sid, label, "assemble predicted descriptor vector", "predicted_descriptor_vector", desc_vec[None, :], "run_phase7a.py", "condition_for/condition_vector", ",".join(DESCRIPTOR_COLS)))
    probs = np.asarray(parse_probs(cond.get("prototype_probabilities")), dtype=np.float32)
    rows.append(trace_row("B02", sid, label, "load prototype probabilities", "prototype_probabilities", probs[None, :] if probs.size else probs, "run_phase7a.py", "build_conditions"))
    ranked = rank_a3_sources(representative, sid, cond, ids)
    cand_mat = np.vstack([descriptor_vector_from_row(r) for _, r in ranked.iterrows()]).astype(np.float32)
    rows.append(trace_row("B03", sid, label, "strict OOF representative candidate descriptor bank excluding held-out sample", "afm_candidate_matrix", cand_mat, "run_phase7a.py", "source_rows/rank_sources"))
    rows.append(trace_row("B04", sid, label, "A3 distance vector", "a3_distance_vector", ranked["rank_score"].to_numpy(np.float32), "run_phase7a.py", "rank_sources", "sqrt(sum(((candidate-condition)/std(candidate_bank))^2)) + 0.05*abs(candidate_rq-condition_rq)"))
    selected = ranked.iloc[0]
    rows.append(trace_row("B05", sid, label, "select minimum A3 distance with sample_id/scan_id tie-break", "selected_source_index", np.asarray([0], dtype=np.int32), "run_phase7a.py", "rank_sources/synthesize", f"source={selected['sample_id']} scan={selected['scan_id']}"))
    source = resize_center(read_map(selected["second_order_afm_path"]), 256)
    rows.append(trace_row("B06", sid, label, "load selected historical source AFM", "source_afm_centered_nm", source, "run_phase7a.py", "read_map/resize_center", str(selected["second_order_afm_path"])))
    _, unit, _ = unit_shape(read_map(selected["second_order_afm_path"]))
    unit = project_unit_rq_np(resize_center(unit, 256))
    rows.append(trace_row("B07", sid, label, "normalize morphology to unit Rq", "unit_rq_morphology", unit, "rq_disentanglement.py", "unit_shape/project_unit_rq_np"))
    qmaps = []
    for key in ["condition_q10_rq_nm", "condition_q50_rq_nm", "condition_q90_rq_nm"]:
        qmaps.append(physical_from_q(unit, float(cond[key])))
    rows.append(trace_row("B08", sid, label, "rescale one source morphology to q10/q50/q90 amplitudes", "representative_afm_q10_q50_q90", np.stack(qmaps, axis=0), "rq_disentanglement.py", "physical_from_q", "q10/q50/q90 share the same selected source morphology; only amplitude differs"))

    example = {
        "sample_id": sid,
        "label": label,
        "true_t4_rq_nm": float(artifacts["target_table"].set_index("sample_id").loc[sid, "T4_second_order_trimmed_mean"]),
        "strict_oof_predicted_rq_nm": float(strict["predicted_target_nm"]),
        "strict_condition_q10_q50_q90_nm": [float(cond["condition_q10_rq_nm"]), float(cond["condition_q50_rq_nm"]), float(cond["condition_q90_rq_nm"])],
        "full_cohort_in_sample_predicted_rq_nm": full_pred,
        "full_cohort_member_predictions": member_preds,
        "prototype_probabilities": parse_probs(cond.get("prototype_probabilities")),
        "selected_source_sample_id": str(selected["sample_id"]),
        "selected_source_scan_id": str(selected["scan_id"]),
        "selected_source_afm_path": str(selected["second_order_afm_path"]),
        "top5_a3_sources": ranked.head(5)[["sample_id", "scan_id", "rq_nm", "descriptor_z_euclidean", "rq_penalty_abs", "rank_score", "second_order_afm_path"]].to_dict("records"),
        "embedding_metadata": embed_meta,
    }
    return rows, example


def build_ledgers(paths: Paths, artifacts: dict[str, Any], trace_rows: list[dict[str, Any]], examples: list[dict[str, Any]]) -> dict[str, Any]:
    ensemble = artifacts["ensemble"]
    z = np.load(paths.embedding_npz, allow_pickle=False)
    ids = active_ids(artifacts["canonical"])
    bank = artifacts["bank"]
    representative = representative_table(artifacts["scans"], artifacts["canonical"], ids)
    strict_candidate_counts = []
    for sid in ids:
        cond = artifacts["conditions"].set_index("sample_id").loc[sid]
        strict_candidate_counts.append(len(rank_a3_sources(representative, sid, cond, ids)))
    model_rows = []
    for member in ensemble["members"]:
        p = paths.freeze / "12_FULL_COHORT_DEPLOYMENT/quantitative_model" / f"{member['name']}.npz"
        m = np.load(p, allow_pickle=False)
        model_rows.append(
            {
                "parameter_group": "quantitative_member",
                "name": member["name"],
                "value": f"{member['model_family']} alpha=1.0",
                "shape": list(m["coef"].shape),
                "source_file": str(p.relative_to(ROOT)),
                "evidence": "serialized coef/intercept/feature_mean/feature_scale",
            }
        )
    parameter_rows = [
        {
            "parameter_group": "quantitative_ensemble",
            "name": "model_name",
            "value": artifacts["model_registry"]["model_name"],
            "shape": "",
            "source_file": str((paths.freeze / "12_FULL_COHORT_DEPLOYMENT/quantitative_model/model_registry.json").relative_to(ROOT)),
            "evidence": "freeze model registry",
        },
        {
            "parameter_group": "quantitative_ensemble",
            "name": "member_count",
            "value": len(ensemble["members"]),
            "shape": "",
            "source_file": str((paths.freeze / "12_FULL_COHORT_DEPLOYMENT/quantitative_model/ensemble_definition.json").relative_to(ROOT)),
            "evidence": "ensemble_definition.json",
        },
        {
            "parameter_group": "quantitative_ensemble",
            "name": "aggregation",
            "value": ensemble["aggregation"],
            "shape": "",
            "source_file": str((paths.freeze / "12_FULL_COHORT_DEPLOYMENT/quantitative_model/ensemble_definition.json").relative_to(ROOT)),
            "evidence": "median of five member outputs in Rq nm space",
        },
        {
            "parameter_group": "encoder",
            "name": "embedding_npz_shape",
            "value": list(z["embeddings"].shape),
            "shape": list(z["embeddings"].shape),
            "source_file": str(paths.embedding_npz.relative_to(ROOT)),
            "evidence": "cached Phase2A DINO keyframe embeddings",
        },
        {
            "parameter_group": "visual",
            "name": "strict_a3_candidate_count_per_fold",
            "value": sorted(set(strict_candidate_counts)),
            "shape": "",
            "source_file": "analysis/rheed_video_afm_story/run_phase7a.py",
            "evidence": "source_rows excludes heldout sample from 23 representative groups",
        },
        {
            "parameter_group": "visual",
            "name": "deployment_bank_groups_scans",
            "value": {"groups": int(bank["sample_id"].nunique()), "scans": int(len(bank))},
            "shape": "",
            "source_file": str((paths.freeze / "12_FULL_COHORT_DEPLOYMENT/visual_model/afm_bank_manifest.csv").relative_to(ROOT)),
            "evidence": "full-cohort visual bank manifest",
        },
    ] + model_rows

    descriptor_rows = [
        {
            "index": i,
            "descriptor": col,
            "strict_condition_column": "condition_q50_rq_nm" if col == "rq_nm" else f"predicted_{col}",
            "active_in_a3": True,
            "source_file": "analysis/rheed_video_afm_story/run_phase7a.py",
            "source_function": "DESCRIPTOR_COLS/condition_vector/rank_sources",
        }
        for i, col in enumerate(DESCRIPTOR_COLS)
    ]

    tensor_rows = []
    seen = set()
    for row in trace_rows:
        key = (row["node_id"], row["tensor_name"])
        if key in seen:
            continue
        seen.add(key)
        tensor_rows.append(
            {
                "node_id": row["node_id"],
                "tensor_name": row["tensor_name"],
                "shape": row["shape"],
                "dtype": row["dtype"],
                "operation": row["operation"],
                "source_file": row["source_file"],
                "source_function": row["source_function"],
                "notes": row["notes"],
            }
        )

    evidence_rows = [
        {
            "claim_id": "C01",
            "claim": "Final quantitative route is RHEED keyframe -> cached DINO embedding -> five ridge members -> median Rq.",
            "evidence_file": str((paths.freeze / "12_FULL_COHORT_DEPLOYMENT/quantitative_model/ensemble_definition.json").relative_to(ROOT)),
            "evidence_detail": "members all use E1_dino_keyframe; aggregation is median",
            "status": "confirmed",
        },
        {
            "claim_id": "C02",
            "claim": "Strict visual benchmark excludes held-out AFM source from retrieval candidates.",
            "evidence_file": "analysis/rheed_video_afm_story/run_phase7a.py:408-412",
            "evidence_detail": "source_rows removes sid for strict/oracle tracks",
            "status": "confirmed",
        },
        {
            "claim_id": "C03",
            "claim": "Final strict visual method for the architecture diagram is fixed A3 retrieval, not per-sample mixed best.",
            "evidence_file": str((paths.phase7b / "fixed_method_registry.csv").relative_to(ROOT)),
            "evidence_detail": "Phase7B fixed method registry pins retrieval family to A3",
            "status": "confirmed",
        },
        {
            "claim_id": "C04",
            "claim": "Route B is representative retrieval and Rq rescaling, not a pixel decoder.",
            "evidence_file": "analysis/rheed_video_afm_story/run_phase7a.py:486-489",
            "evidence_detail": "A3 uses top_maps[0] retrieved real AFM exemplar",
            "status": "confirmed",
        },
        {
            "claim_id": "C05",
            "claim": "Phase3A AFM autoencoder is not in the final RHEED-to-Rq/A3 pipeline.",
            "evidence_file": str((paths.freeze / "16_CLAIMS_AND_LIMITATIONS/reviewer_question_and_answer.md").relative_to(ROOT)),
            "evidence_detail": "final route uses DINO/Ridge and A3 retrieval; no AE model files are invoked",
            "status": "confirmed",
        },
        {
            "claim_id": "C06",
            "claim": "Freeze unseen inference script does not implement the same strict A3 descriptor route.",
            "evidence_file": str((paths.freeze / "13_UNSEEN_INFERENCE/predict_unseen_batch.py").relative_to(ROOT)),
            "evidence_detail": "script builds deterministic_vector and selects source by abs(rq_nm - pred)",
            "status": "discrepancy",
        },
    ]

    return {
        "parameter_rows": parameter_rows,
        "descriptor_rows": descriptor_rows,
        "tensor_rows": tensor_rows,
        "evidence_rows": evidence_rows,
        "strict_candidate_counts": strict_candidate_counts,
        "examples": examples,
    }


def build_architecture(paths: Paths, artifacts: dict[str, Any], ledgers: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {"node_id": "N01", "label": "Raw RHEED keyframe PNG", "track": "shared", "shape": "H x W x 3 uint8", "role": "input"},
        {"node_id": "N02", "label": "Manual ROI crop and luminance", "track": "shared", "shape": "roi_height x roi_width -> 224 x 224 uint8", "role": "preprocessing"},
        {"node_id": "N03", "label": "DINOv2 ViT-S/14 frozen encoder", "track": "quantitative", "shape": "[1,3,224,224] -> CLS [1,384]", "role": "encoder"},
        {"node_id": "N04", "label": "Phase2A temporal aggregation", "track": "quantitative", "shape": "[1,384] -> [1,1536]", "role": "feature"},
        {"node_id": "N05", "label": "Five full-cohort ridge members", "track": "quantitative", "shape": "5 x scalar Rq", "role": "regression"},
        {"node_id": "N06", "label": "Median Rq ensemble", "track": "quantitative", "shape": "q50 scalar nm", "role": "prediction"},
        {"node_id": "N07", "label": "Strict interval q10/q50/q90", "track": "strict_oof", "shape": "3 scalar Rq values", "role": "uncertainty"},
        {"node_id": "N08", "label": "Predicted descriptor vector", "track": "visual", "shape": "[1,11]", "role": "condition"},
        {"node_id": "N09", "label": "Strict A3 representative AFM bank", "track": "strict_oof", "shape": "22 x 11 candidates per held-out fold", "role": "retrieval bank"},
        {"node_id": "N10", "label": "A3 distance and source selection", "track": "visual", "shape": "22 distances -> one source", "role": "selection"},
        {"node_id": "N11", "label": "Selected historical AFM morphology", "track": "visual", "shape": "256 x 256", "role": "source map"},
        {"node_id": "N12", "label": "Unit-Rq morphology projection", "track": "visual", "shape": "256 x 256", "role": "normalization"},
        {"node_id": "N13", "label": "Representative AFM q10/q50/q90 maps", "track": "visual", "shape": "3 x 256 x 256", "role": "output"},
        {"node_id": "N14", "label": "Full-cohort deployment visual bank", "track": "deployment", "shape": "23 groups / 116 scans", "role": "future-only bank"},
        {"node_id": "N15", "label": "Current freeze unseen script", "track": "deployment caveat", "shape": "deterministic placeholder embedding; Rq-nearest scan", "role": "discrepancy"},
    ]
    edges = [
        {"source": "N01", "target": "N02", "label": "crop"},
        {"source": "N02", "target": "N03", "label": "normalize"},
        {"source": "N03", "target": "N04", "label": "aggregate"},
        {"source": "N04", "target": "N05", "label": "scaled input"},
        {"source": "N05", "target": "N06", "label": "median"},
        {"source": "N06", "target": "N07", "label": "strict residual interval"},
        {"source": "N06", "target": "N08", "label": "Rq condition"},
        {"source": "N08", "target": "N10", "label": "descriptor vector"},
        {"source": "N09", "target": "N10", "label": "candidate descriptors"},
        {"source": "N10", "target": "N11", "label": "min distance source"},
        {"source": "N11", "target": "N12", "label": "unit Rq"},
        {"source": "N07", "target": "N13", "label": "amplitude values"},
        {"source": "N12", "target": "N13", "label": "shared morphology rescaled"},
        {"source": "N14", "target": "N15", "label": "current script caveat"},
    ]
    return nodes, edges


def write_architecture_files(paths: Paths, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    spec = {
        "title": "RHEED-to-Rq and Representative-AFM final architecture",
        "main_figure_rule": "Use fixed A3 retrieval for the visual route; do not draw Phase3A AE or Phase7B non-A3 comparison methods as final components.",
        "strict_oof_rule": "Each strict visual fold uses 22 candidate source groups after excluding the held-out sample.",
        "deployment_rule": "Full-cohort deployment bank has 23 groups and 116 scans and is future-only, not an independent test.",
        "nodes": nodes,
        "edges": edges,
    }
    write_json(paths.out / "architecture_spec.json", spec)
    yaml_lines = ["title: RHEED-to-Rq and Representative-AFM final architecture", "nodes:"]
    for n in nodes:
        yaml_lines.append(f"  - node_id: {n['node_id']}")
        for k in ["label", "track", "shape", "role"]:
            yaml_lines.append(f"    {k}: {json.dumps(n[k], ensure_ascii=False)}")
    yaml_lines.append("edges:")
    for e in edges:
        yaml_lines.append(f"  - source: {e['source']}")
        yaml_lines.append(f"    target: {e['target']}")
        yaml_lines.append(f"    label: {json.dumps(e['label'], ensure_ascii=False)}")
    write_text(paths.out / "architecture_spec.yaml", "\n".join(yaml_lines))
    write_table_pair(paths.out / "architecture_nodes", nodes, ["node_id", "label", "track", "shape", "role"])
    write_table_pair(paths.out / "architecture_edges", edges, ["source", "target", "label"])
    write_text(
        paths.out / "architecture_layout.md",
        """# Architecture Layout

Use a left-to-right layout with two visible lanes after the shared RHEED preprocessing block.

Top lane: quantitative RHEED-to-Rq route from frozen DINO features to five ridge members and median q50 Rq.

Bottom lane: representative-AFM route from predicted Rq/descriptors to fixed A3 candidate ranking, selected historical morphology, unit-Rq normalization, and q10/q50/q90 amplitude rescaling.

Place strict OOF and deployment as separate callouts. The strict OOF callout must say "22 candidate groups per held-out fold; held-out AFM excluded." The deployment callout must say "23 historical groups / 116 scans; future-only; not an independent test." Do not draw a neural AFM pixel decoder in the final pipeline.
""",
    )
    write_text(
        paths.out / "architecture_drawing_prompt_main_figure.md",
        """Draw a publication-grade methods schematic for a RHEED-to-AFM surrogate pipeline.

Main message: the deployable quantitative route predicts AFM roughness Rq from a frozen DINO RHEED embedding; the visual route retrieves a representative historical AFM morphology using fixed A3 descriptor ranking and rescales it to q10/q50/q90 Rq.

Required blocks: Raw RHEED keyframe PNG; manual ROI crop; 224 x 224 luminance image; DINOv2 ViT-S/14 frozen encoder; 1536-D temporal aggregate feature; five ridge regressors; median q50 Rq; strict q10/q90 interval; 11-D descriptor vector; strict A3 AFM bank; A3 distance; selected historical AFM; unit-Rq morphology; q10/q50/q90 representative AFM maps.

Required warnings in small callouts: "Representative retrieval, not pixel reconstruction"; "Final visual method: fixed A3"; "Phase3A AFM autoencoder not used"; "Strict OOF separated from full-cohort future deployment"; "Do not show per-sample mixed best as the final method."
""",
    )
    write_text(
        paths.out / "architecture_drawing_prompt_detailed_supplement.md",
        """Draw a detailed supplementary architecture diagram with exact tensor dimensions.

Show the DINO input tensor [1,3,224,224], ViT-S/14 patch tokens [1,256,384], token sequence [1,257,384], CLS/frame embedding [1,384], and Phase2A temporal aggregate [1,1536]. Show five ridge model inputs [1,1536], five scalar member outputs, median aggregation in nm Rq space, and the strict q10/q50/q90 definitions.

For the visual lane, show the 11 descriptors in order: rq_nm, ra_nm, robust_height_range_nm, psd_low_fraction, psd_mid_fraction, psd_high_fraction, psd_slope, correlation_length_nm, anisotropy, height_skewness, height_kurtosis. Show strict A3 ranking over 22 candidate representative groups per held-out sample, selected source AFM [256,256], unit-Rq morphology [256,256], and three rescaled maps [3,256,256].

Include a deployment caveat panel: the full-cohort visual bank contains 23 groups and 116 scans, but the current frozen unseen script should be treated as a technical smoke script because it uses deterministic placeholder embeddings and Rq-nearest scan selection rather than the full strict A3 descriptor route.
""",
    )

    mmd = ["flowchart LR"]
    for n in nodes:
        safe_label = n["label"].replace('"', "'")
        mmd.append(f'  {n["node_id"]}["{safe_label}<br/>{n["shape"]}"]')
    for e in edges:
        mmd.append(f'  {e["source"]} -->|{e["label"]}| {e["target"]}')
    write_text(paths.out / "architecture_preview_main.mmd", "\n".join(mmd))
    write_text(paths.out / "architecture_preview_detailed.mmd", "\n".join(mmd))

    dot = ["digraph G {", "rankdir=LR;", 'node [shape=box, style="rounded,filled", fillcolor="#F7F7F7"];']
    for n in nodes:
        label = f"{n['node_id']}\\n{n['label']}\\n{n['shape']}".replace('"', "'")
        dot.append(f'{n["node_id"]} [label="{label}"];')
    for e in edges:
        dot.append(f'{e["source"]} -> {e["target"]} [label="{e["label"]}"];')
    dot.append("}")
    for name in ["architecture_preview_main", "architecture_preview_detailed"]:
        dot_path = paths.out / f"{name}.dot"
        write_text(dot_path, "\n".join(dot))
        dot_bin = shutil.which("dot")
        if dot_bin:
            for ext in ["svg", "pdf"]:
                subprocess.run([dot_bin, f"-T{ext}", str(dot_path), "-o", str(paths.out / f"{name}.{ext}")], check=False)


def write_reports(paths: Paths, artifacts: dict[str, Any], ledgers: dict[str, Any], examples: list[dict[str, Any]]) -> dict[str, Any]:
    ensemble = artifacts["ensemble"]
    model_registry = artifacts["model_registry"]
    bank = artifacts["bank"]
    phase7b_summary = artifacts["phase7b_summary"]
    retrieval_row = {}
    if not phase7b_summary.empty:
        family_col = "method_family" if "method_family" in phase7b_summary.columns else "family"
        r = phase7b_summary[phase7b_summary[family_col].astype(str).eq("retrieval")]
        if not r.empty:
            retrieval_row = r.iloc[0].to_dict()

    member_list = ", ".join(f"{m['name']} ({m['trial_id']}, {m['target_variant']})" for m in ensemble["members"])
    validation = {
        "passed": True,
        "checks": [],
        "unrecovered_or_ambiguous": [
            "Strict OOF fold coefficients are not serialized in the freeze; strict Rq predictions are recovered from Phase6A OOF artifact, while full-cohort member coefficients are serialized.",
            "DINO backbone internals are not serialized as tensors in the freeze; patch/token shapes are recovered from the dinov2_vits14 identifier and input size, while final cached feature values are serialized.",
            "Freeze unseen inference script is a technical smoke implementation and does not perform real DINO extraction or full strict A3 descriptor ranking.",
        ],
    }
    def add_check(name: str, passed: bool, detail: str) -> None:
        validation["checks"].append({"name": name, "passed": bool(passed), "detail": detail})
        validation["passed"] = bool(validation["passed"] and passed)

    add_check("ensemble_member_count_is_5", len(ensemble["members"]) == 5, str(len(ensemble["members"])))
    add_check("embedding_dim_is_1536", any(r["tensor_name"] == "final_rheed_embedding" and "1536" in str(r["shape"]) for r in ledgers["tensor_rows"]), "Phase2A temporal aggregate")
    add_check("descriptor_dim_is_11", len(DESCRIPTOR_COLS) == 11, ",".join(DESCRIPTOR_COLS))
    add_check("strict_candidate_count_is_22", set(ledgers["strict_candidate_counts"]) == {22}, str(sorted(set(ledgers["strict_candidate_counts"]))))
    add_check("deployment_bank_is_23_groups_116_scans", bank["sample_id"].nunique() == 23 and len(bank) == 116, f"{bank['sample_id'].nunique()} groups / {len(bank)} scans")
    if not artifacts["phase7b_audit"].empty:
        a3 = artifacts["phase7b_audit"][artifacts["phase7b_audit"]["method_id"].astype(str).eq("A3")]
        add_check("phase7b_a3_heldout_source_zero", (not a3.empty) and float(a3["max_heldout_source_contribution"].max()) == 0.0, "Phase7B method_audit A3")
    else:
        add_check("phase7b_a3_heldout_source_zero", False, "Phase7B audit missing")

    write_json(paths.out / "methodology_validation.json", validation)
    check_rows = validation["checks"] + [{"name": "overall", "passed": validation["passed"], "detail": "; ".join(validation["unrecovered_or_ambiguous"])}]
    write_text(paths.out / "methodology_validation.md", md_table(check_rows, ["name", "passed", "detail"]))

    discrepancy_rows = [
        {
            "issue": "Freeze all_23_sample_atlas is copied from Phase7A per-sample best visual output, not fixed A3.",
            "resolution": "Use Phase7B Fig_fixed_retrieval_all_23_strict_oof_atlas for the fixed final retrieval figure.",
            "severity": "method-clarification",
        },
        {
            "issue": "Phase7A condition_vectors predicted_rq_nm column can reflect descriptor-predicted rq after descriptor loop.",
            "resolution": "Use condition_q50_rq_nm for the Rq conditioning and predicted Rq in fixed atlases.",
            "severity": "column-semantics",
        },
        {
            "issue": "Frozen unseen inference script uses deterministic placeholder vectors and Rq-nearest scan, not actual DINO extraction plus full A3 descriptor ranking.",
            "resolution": "Treat the script as technical smoke code until replaced by a true DINO/A3 implementation.",
            "severity": "deployment-gap",
        },
    ]
    write_table_pair(paths.out / "discrepancy_report", discrepancy_rows, ["issue", "resolution", "severity"])

    readme = f"""# Final Algorithm Audit

This directory is a read-only audit layer over the existing frozen RHEED-to-AFM artifacts. It does not train models, reselect methods, edit raw data, or modify the paper freeze.

Core conclusion: the strict visual result should be described as RHEED-conditioned representative AFM retrieval using fixed A3, not as a neural AFM pixel decoder and not as Phase3A AFM autoencoder performance.

Quantitative model: `{model_registry['model_name']}`.

Visual method for the final strict architecture: `A3` representative AFM retrieval. Phase7B comparison methods are benchmark-only.

Known deployment gap: `13_UNSEEN_INFERENCE/predict_unseen_batch.py` is a technical smoke script and does not yet implement actual DINO feature extraction plus descriptor A3 ranking.
"""
    write_text(paths.out / "README.md", readme)

    cn = f"""# 最终方法学澄清

## 结论

当前可作为论文 strict OOF 证据的主线不是 `AFM -> AFM decoder`，也不是 `RHEED -> neural decoder -> AFM pixels`。最终应表述为两条相连但不同的路线：

1. **RHEED -> Rq 定量预测**：RHEED keyframe PNG 经过手工 ROI、灰度化、224 x 224 resize/pad、ImageNet normalization，进入冻结 DINOv2 ViT-S/14。Phase2A 缓存的单帧 CLS 为 384 维，但实际回归输入是 `mean/std/delta/slope` temporal aggregate，因此是 1536 维。freeze 的 full-cohort 定量模型是 `{model_registry['model_name']}`，由 5 个 ridge member 组成，在 Rq nm 空间取 median。

2. **RHEED-conditioned representative AFM retrieval**：strict visual 路线使用 Phase7A/Phase7B 的固定 `A3` 检索。输入是预测 Rq、预测 AFM descriptors 和 prototype probabilities；candidate bank 是历史 AFM representative maps。strict OOF 时每个 held-out growth group 被排除，所以每折有 22 个 source group。A3 选出一个历史 AFM source morphology，将它投影到 unit-Rq，再用 q10/q50/q90 Rq 重新缩放成三张 representative AFM。

## 必须避免的误述

- 不要说 final visual result 是 Phase3A AFM autoencoder。
- 不要说 final visual result 是 RHEED 直接 decoder 出 AFM pixels。
- 不要把 Phase7A 的 per-sample mixed-best atlas 当成固定方法图。
- main architecture figure 应画 fixed A3 retrieval。
- strict OOF 和 full-cohort future deployment 必须分开。

## q10/q50/q90

strict OOF 中，q50 是 Phase6A strict OOF 的 predicted Rq。q10/q90 来自该 fold training samples 的 absolute-error 分布：`q10 = max(0.001, pred - err90)`，`q90 = max(q10+0.001, pred + err90)`。三张 AFM 图共享同一个 selected source morphology，只改变 Rq amplitude。

## 部署包差异

freeze 的 registry 写的是 A3 full-cohort retrieval，但 `13_UNSEEN_INFERENCE/predict_unseen_batch.py` 当前实际使用 deterministic placeholder embedding，并按 `abs(source_rq - pred)` 选 source scan。这不是 strict OOF 的完整 DINO -> descriptor -> A3 路线，应标为 technical smoke implementation。
"""
    write_text(paths.out / "methodology_report_cn.md", cn)

    en = f"""# Methods Clarification

The frozen study contains two linked routes. The quantitative route predicts AFM roughness from RHEED. The visual route retrieves and rescales a representative historical AFM morphology. The final strict visual result is not an AFM autoencoder test and not a RHEED-to-pixel decoder.

RHEED frames were reduced to a manually selected keyframe ROI, converted to luminance, resized and padded to 224 x 224, replicated to three channels, and normalized with ImageNet statistics before frozen DINOv2 ViT-S/14 feature extraction. The DINO frame CLS dimension is 384. The serialized Phase2A feature used by the final regressors is a temporal aggregate of mean, standard deviation, first-last delta, and temporal slope, yielding a 1536-dimensional vector for each sample.

The frozen full-cohort quantitative model is `{model_registry['model_name']}`. It contains five ridge members: {member_list}. Each member scales the 1536-dimensional RHEED feature with its serialized mean and scale, predicts a scalar roughness value in nm, and the ensemble reports the median in Rq space.

For strict representative-AFM visualization, the final fixed method is A3 retrieval. A condition vector is assembled from q50 Rq and the 11 predicted descriptors. Candidate historical AFM representative maps from the outer-training groups are scored by z-scaled descriptor Euclidean distance plus a 0.05 absolute Rq penalty. The held-out sample's AFM is excluded. The minimum-score source map is projected to unit Rq and rescaled to q10, q50, and q90. These three maps share one morphology and differ only in amplitude.

The current frozen unseen inference script should not be described as a completed production implementation of this full route. It uses deterministic placeholder vectors and Rq-nearest scan retrieval, so it is a technical smoke implementation pending replacement by actual DINO extraction and A3 descriptor ranking.
"""
    write_text(paths.out / "methods_paper_en.md", en)

    tex = r"""\subsection{Final RHEED-to-Rq and representative-AFM pipeline}
The final quantitative model predicts AFM roughness from frozen RHEED features. A manually selected RHEED keyframe ROI is converted to luminance, resized and padded to $224\times224$, replicated to three channels, ImageNet-normalized, and encoded with frozen DINOv2 ViT-S/14. The DINO frame CLS dimension is 384. The serialized Phase2A feature concatenates mean, standard deviation, first--last delta, and temporal slope, producing a 1536-dimensional vector. Five ridge regressors predict scalar Rq values in nm, and the ensemble output is their median.

The final strict visual method is fixed A3 representative retrieval, not a neural AFM pixel decoder. For each strict held-out sample, predicted Rq and 11 predicted AFM descriptors define the retrieval condition. Historical representative AFM maps from the outer-training groups are ranked by z-scaled descriptor Euclidean distance plus a 0.05 absolute Rq penalty. The selected source morphology is normalized to unit Rq and rescaled to q10, q50, and q90. The three rendered AFM maps therefore share morphology and differ only in amplitude.
"""
    write_text(paths.out / "methods_paper_en.tex", tex)

    for ex in examples:
        body = f"""# Worked Example: {ex['sample_id']} ({ex['label']})

True T4 Rq: {ex['true_t4_rq_nm']:.6f} nm

Strict OOF predicted q50 Rq: {ex['strict_oof_predicted_rq_nm']:.6f} nm

Strict q10/q50/q90: {', '.join(f'{x:.6f}' for x in ex['strict_condition_q10_q50_q90_nm'])} nm

Full-cohort in-sample prediction: {ex['full_cohort_in_sample_predicted_rq_nm']:.6f} nm

Selected A3 source: sample {ex['selected_source_sample_id']}, scan {ex['selected_source_scan_id']}.

Source path: `{ex['selected_source_afm_path']}`.

## Ensemble Members

{md_table(ex['full_cohort_member_predictions'], ['member', 'trial_id', 'model_family', 'feature_set', 'target_variant', 'input_dim', 'prediction_nm'])}

## Top A3 Sources

{md_table(ex['top5_a3_sources'], ['sample_id', 'scan_id', 'rq_nm', 'descriptor_z_euclidean', 'rq_penalty_abs', 'rank_score', 'second_order_afm_path'])}
"""
        fname = "worked_example_low_rq.md" if ex["label"] == "low_true_rq" else "worked_example_high_rq.md"
        write_text(paths.out / fname, body)

    caption_main_en = "Main architecture of the frozen RHEED-to-Rq and representative-AFM retrieval pipeline. The visual branch uses fixed A3 retrieval and Rq rescaling of a historical source morphology; it is not a pixel decoder."
    caption_detail_en = "Detailed tensor-level architecture. DINOv2 ViT-S/14 produces 384-dimensional frame embeddings, Phase2A stores a 1536-dimensional temporal aggregate, and A3 ranks 22 strict OOF candidate representative AFM groups with an 11-dimensional descriptor vector."
    write_text(paths.out / "caption_main_en.md", caption_main_en)
    write_text(paths.out / "caption_detailed_en.md", caption_detail_en)
    write_text(paths.out / "caption_main_cn.md", "冻结 RHEED-to-Rq 与 representative-AFM 检索流程主图。visual 分支使用固定 A3 检索和 Rq 重缩放，不是像素 decoder。")
    write_text(paths.out / "caption_detailed_cn.md", "张量级细节图。DINOv2 ViT-S/14 的单帧 embedding 为 384 维，Phase2A 存储 1536 维 temporal aggregate；A3 使用 11 维 descriptor 对 strict OOF 的 22 个 candidate representative AFM group 排序。")

    return validation


def main() -> None:
    paths = load_paths()
    paths.out.mkdir(parents=True, exist_ok=True)
    artifacts = load_artifacts(paths)
    ids = active_ids(artifacts["canonical"])
    t4 = artifacts["target_table"].set_index("sample_id").loc[ids, "T4_second_order_trimmed_mean"].astype(float)
    low_sid = str(t4.idxmin())
    high_sid = str(t4.idxmax())

    low_rows, low_ex = trace_sample(paths, artifacts, low_sid, "low_true_rq")
    high_rows, high_ex = trace_sample(paths, artifacts, high_sid, "high_true_rq")
    trace_rows = low_rows + high_rows
    examples = [low_ex, high_ex]

    write_json(paths.out / "runtime_trace_low_rq.json", low_rows)
    write_json(paths.out / "runtime_trace_high_rq.json", high_rows)
    write_table_pair(paths.out / "runtime_trace", trace_rows, ["node_id", "sample_id", "track", "operation", "tensor_name", "shape", "dtype", "device", "min", "max", "mean", "std", "source_file", "source_function", "notes"], max_md_rows=None)

    ledgers = build_ledgers(paths, artifacts, trace_rows, examples)
    write_table_pair(paths.out / "evidence_ledger", ledgers["evidence_rows"], ["claim_id", "claim", "evidence_file", "evidence_detail", "status"])
    write_table_pair(paths.out / "tensor_shape_ledger", ledgers["tensor_rows"], ["node_id", "tensor_name", "shape", "dtype", "operation", "source_file", "source_function", "notes"])
    write_table_pair(paths.out / "parameter_ledger", ledgers["parameter_rows"], ["parameter_group", "name", "value", "shape", "source_file", "evidence"])
    write_table_pair(paths.out / "descriptor_ledger", ledgers["descriptor_rows"], ["index", "descriptor", "strict_condition_column", "active_in_a3", "source_file", "source_function"])

    nodes, edges = build_architecture(paths, artifacts, ledgers)
    write_architecture_files(paths, nodes, edges)
    validation = write_reports(paths, artifacts, ledgers, examples)

    summary = {
        "output_root": str(paths.out.relative_to(ROOT)),
        "low_true_rq_sample": low_sid,
        "high_true_rq_sample": high_sid,
        "validation_passed": validation["passed"],
        "quantitative_model": artifacts["model_registry"]["model_name"],
        "visual_model": "fixed A3 representative AFM retrieval",
        "created_files": sorted(str(p.relative_to(ROOT)) for p in paths.out.glob("*")),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
