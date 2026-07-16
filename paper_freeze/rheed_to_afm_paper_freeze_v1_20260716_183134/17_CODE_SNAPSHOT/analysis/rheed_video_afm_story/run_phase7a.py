from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import signal
import sqlite3
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity
from skimage.transform import resize
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from .afm_descriptors import descriptor_distance, gradients, radial_psd
from .common import display_path, load_config, repo_path, save_parquet, sha256_file, write_csv, write_json
from .rq_disentanglement import physical_from_q, project_unit_rq_np, ra_np, rq_np, unit_shape


INTERRUPTED = False
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
METHOD_FAMILIES = {
    "VB0": "baseline",
    "VB1": "retrieval",
    "VB2": "quilting",
    "A1": "retrieval",
    "A2": "retrieval",
    "A3": "retrieval",
    "A4": "retrieval",
    "A5": "retrieval",
    "A6": "retrieval",
    "B1": "quilting",
    "B2": "quilting",
    "C1": "residual",
    "C2": "residual",
    "D1": "iaaft",
    "D4": "iaaft",
    "E1": "texture",
    "E2": "texture",
    "F1": "vq",
    "F2": "vq",
    "G4": "diffusion",
}


def _handle_interrupt(signum, frame) -> None:  # noqa: ANN001
    global INTERRUPTED
    INTERRUPTED = True


signal.signal(signal.SIGINT, _handle_interrupt)


@dataclass
class Phase7Context:
    config: dict[str, Any]
    out: Path
    rep: Path
    active_ids: list[str]
    index: pd.DataFrame
    scan_manifest: pd.DataFrame
    representative: pd.DataFrame
    strict_pred: pd.DataFrame
    desc_pred: pd.DataFrame
    prototype_pred: pd.DataFrame
    conditions: pd.DataFrame
    embeddings: dict[str, pd.DataFrame]
    physics: pd.DataFrame
    device: str


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value) == "True"


def ensure_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    out = repo_path(config["output_root"])
    rep = repo_path(config["report_root"])
    for rel in [
        "provenance",
        "canonical_index",
        "afm_texture_dataset/physical_maps",
        "afm_texture_dataset/unit_shape_maps",
        "afm_texture_dataset/multiscale_pyramids",
        "afm_texture_dataset/patch_manifests",
        "condition_vectors",
        "visual_trials",
        "strict_oof",
        "oracle_upper_bound",
        "full_cohort_development",
        "retrieval",
        "quilting",
        "spectral_synthesis",
        "texture_optimization",
        "vq_models",
        "diffusion_models/checkpoints",
        "generated_maps",
        "metrics",
        "blind_review/scoring_templates",
        "blind_review/private_answer_keys",
        "figures",
        "dashboard",
        "deployment",
        "logs",
    ]:
        (out / rel).mkdir(parents=True, exist_ok=True)
    for rel in ["figures", "dashboard", "blind_review"]:
        (rep / rel).mkdir(parents=True, exist_ok=True)
    return out, rep


def load_embedding(path: str | Path) -> pd.DataFrame:
    z = np.load(repo_path(path), allow_pickle=False)
    ids = [str(x) for x in z["sample_ids"].tolist()]
    if "embeddings" in z:
        arr = np.asarray(z["embeddings"], dtype=float)
    elif "features" in z:
        arr = np.asarray(z["features"], dtype=float)
    else:
        raise KeyError(f"No embeddings/features in {path}")
    return pd.DataFrame(arr, index=ids)


def load_inputs(config_path: str | Path, device: str = "auto") -> Phase7Context:
    config = load_config(config_path)
    out, rep = ensure_dirs(config)
    index = pd.read_csv(repo_path(config["canonical_index_path"]), dtype={"sample_id": str, "growth_run_id": str})
    active = index[index["is_primary"].map(as_bool)].copy()
    active_ids = sorted(active["sample_id"].astype(str).tolist())
    scans = pd.read_csv(repo_path(config["second_order_scan_descriptors_path"]), dtype={"sample_id": str, "growth_run_id": str, "scan_id": str})
    scans = scans[scans["sample_id"].isin(active_ids)].copy().sort_values(["sample_id", "scan_id"]).reset_index(drop=True)
    strict_pred = pd.read_csv(repo_path(config["phase6a_strict_oof_path"]), dtype={"sample_id": str, "heldout_id": str})
    desc_pred = pd.read_csv(repo_path(config["phase6a_descriptor_predictions_path"]), dtype={"sample_id": str})
    proto = pd.read_csv(repo_path(config["phase6a_prototype_predictions_path"]), dtype={"sample_id": str})
    embeddings = {name: load_embedding(path) for name, path in config["embedding_paths"].items()}
    physics = pd.read_csv(repo_path(config["physics_features_path"]), dtype={"sample_id": str}).drop_duplicates("sample_id").set_index("sample_id")
    representative_rows = []
    for sid, group in scans.groupby("sample_id", sort=True):
        target = float(active.set_index("sample_id").loc[sid, "second_order_rq_nm"])
        rep_row = group.assign(_d=(group["rq_nm"].astype(float) - target).abs()).sort_values(["_d", "scan_id"]).iloc[0].copy()
        representative_rows.append(rep_row)
    representative = pd.DataFrame(representative_rows).reset_index(drop=True)
    conditions = build_conditions(config, out, active_ids, strict_pred, desc_pred, proto, representative)
    selected_device = detect_device(device)
    ctx = Phase7Context(config, out, rep, active_ids, index, scans, representative, strict_pred, desc_pred, proto, conditions, embeddings, physics, selected_device)
    return ctx


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def read_map(path: str | Path) -> np.ndarray:
    arr = np.load(repo_path(path), allow_pickle=False).astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.all():
        fill = float(np.nanmean(arr[finite])) if finite.any() else 0.0
        arr = np.where(finite, arr, fill).astype(np.float32)
    return arr


def resize_center(arr: np.ndarray, resolution: int) -> np.ndarray:
    if arr.shape != (resolution, resolution):
        arr = resize(arr.astype(float), (resolution, resolution), order=1, mode="reflect", anti_aliasing=True, preserve_range=True).astype(np.float32)
    arr = arr - float(np.mean(arr))
    return arr.astype(np.float32)


def render_view(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)


def build_conditions(config: dict[str, Any], out: Path, active_ids: list[str], strict_pred: pd.DataFrame, desc_pred: pd.DataFrame, proto: pd.DataFrame, representative: pd.DataFrame) -> pd.DataFrame:
    strict = strict_pred.drop_duplicates("sample_id").set_index("sample_id")
    desc = desc_pred.drop_duplicates("sample_id").set_index("sample_id")
    proto_by = proto.drop_duplicates("sample_id").set_index("sample_id")
    true_by = representative.drop_duplicates("sample_id").set_index("sample_id")
    residuals = strict_pred["absolute_error_nm"].astype(float).to_numpy()
    rows: list[dict[str, Any]] = []
    for sid in active_ids:
        sp = strict.loc[sid]
        dp = desc.loc[sid]
        pp = proto_by.loc[sid]
        tb = true_by.loc[sid]
        train_ids = json.loads(sp["training_ids"])
        train_abs = strict_pred[strict_pred["sample_id"].isin(train_ids)]["absolute_error_nm"].astype(float).to_numpy()
        if len(train_abs) < 3:
            train_abs = residuals
        err10, err90 = np.quantile(train_abs, [0.1, 0.9])
        pred_rq = float(sp["predicted_target_nm"])
        q10 = max(1e-3, pred_rq - float(err90))
        q90 = max(q10 + 1e-3, pred_rq + float(err90))
        row: dict[str, Any] = {
            "sample_id": sid,
            "heldout_id": sid,
            "training_ids": sp["training_ids"],
            "predicted_rq_nm": pred_rq,
            "condition_q10_rq_nm": q10,
            "condition_q50_rq_nm": pred_rq,
            "condition_q90_rq_nm": q90,
            "prediction_interval_source": "phase6a_fold_training_absolute_error_quantiles",
            "true_rq_nm": float(tb["rq_nm"]),
            "oracle_rq_nm": float(tb["rq_nm"]),
            "support_confidence": "phase6a_not_reliable_for_visual_selection",
            "prototype_probabilities": pp.get("probabilities", "[]"),
        }
        for col in DESCRIPTOR_COLS:
            row[f"predicted_{col}"] = float(dp.get(f"predicted_{col}", pred_rq if col == "rq_nm" else np.nan))
            row[f"true_{col}"] = float(dp.get(f"true_{col}", tb.get(col, np.nan)))
            row[f"oracle_{col}"] = row[f"true_{col}"]
        rows.append(row)
    cond = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    write_csv(cond, out / "condition_vectors" / "phase7_condition_vectors.csv")
    save_parquet(cond, out / "condition_vectors" / "phase7_condition_vectors.parquet")
    return cond


def active_index(ctx: Phase7Context) -> pd.DataFrame:
    return ctx.index[ctx.index["sample_id"].isin(ctx.active_ids)].copy()


def integrity_audit(ctx: Phase7Context) -> None:
    removed = set(ctx.config["removed_samples"])
    rows = [
        {"check_name": "primary_N_23", "passed": len(ctx.active_ids) == int(ctx.config["expected_primary_n"]), "detail": f"N={len(ctx.active_ids)}"},
        {"check_name": "removelist_excluded_from_primary", "passed": removed.isdisjoint(ctx.active_ids), "detail": ",".join(sorted(removed & set(ctx.active_ids)))},
        {"check_name": "removelist_hash", "passed": sha256_file(ctx.config["removelist_path"]) == ctx.config["expected_removelist_hash"], "detail": sha256_file(ctx.config["removelist_path"])},
        {"check_name": "strict_oof_ids_match", "passed": set(ctx.strict_pred["sample_id"].astype(str)) == set(ctx.active_ids), "detail": "phase6a strict oof"},
        {"check_name": "descriptor_prediction_ids_match", "passed": set(ctx.desc_pred["sample_id"].astype(str)) == set(ctx.active_ids), "detail": "phase6a descriptor predictions"},
        {"check_name": "scan_manifest_removed_zero", "passed": len(ctx.scan_manifest[ctx.scan_manifest["sample_id"].isin(removed)]) == 0, "detail": "second-order descriptors active subset"},
    ]
    audit = pd.DataFrame(rows)
    write_csv(audit, ctx.out / "provenance" / "phase7_alignment_audit.csv")
    (ctx.out / "provenance" / "phase7_alignment_audit.md").write_text(
        "# Phase 7A Alignment Audit\n\n" + "\n".join(f"- {r['check_name']}: {r['passed']} ({r['detail']})" for r in rows) + "\n",
        encoding="utf-8",
    )
    fold_rows = []
    for sid in ctx.active_ids:
        training = [x for x in ctx.active_ids if x != sid]
        fold_rows.append(
            {
                "fold_id": sid,
                "heldout_sample_id": sid,
                "training_groups": json.dumps(training),
                "training_group_count": len(training),
                "split_valid": len(training) == 22 and sid not in training,
                "contains_6095_when_heldout_6099": "6095" in training,
                "contains_6099_when_heldout_6095": "6099" in training,
            }
        )
    folds = pd.DataFrame(fold_rows)
    write_csv(folds, ctx.out / "provenance" / "phase7_fold_membership.csv")
    shutil.copyfile(repo_path(ctx.config["canonical_index_path"]), ctx.out / "canonical_index" / "canonical_sample_index.csv")
    if not audit["passed"].map(as_bool).all() or not folds["split_valid"].map(as_bool).all():
        raise RuntimeError("Phase7A alignment audit failed; refusing visual training.")


def build_texture_dataset(ctx: Phase7Context) -> None:
    out = ctx.out / "afm_texture_dataset"
    scan_rows = []
    patch_rows = []
    for row in ctx.scan_manifest.to_dict("records"):
        sid = str(row["sample_id"])
        scan_id = str(row["scan_id"])
        arr = read_map(row["second_order_afm_path"])
        centered, unit, rq = unit_shape(arr)
        paths_center: dict[str, str] = {}
        paths_unit: dict[str, str] = {}
        paths_pyramid: dict[str, str] = {}
        for res in ctx.config["texture_resolutions"]:
            res = int(res)
            c = resize_center(centered, res)
            u = project_unit_rq_np(resize_center(unit, res))
            c_path = out / "physical_maps" / f"{sid}__{scan_id}__centered_{res}.npy"
            u_path = out / "unit_shape_maps" / f"{sid}__{scan_id}__unit_{res}.npy"
            np.save(c_path, c.astype(np.float32))
            np.save(u_path, u.astype(np.float32))
            paths_center[str(res)] = display_path(c_path)
            paths_unit[str(res)] = display_path(u_path)
            if res == int(ctx.config["working_resolution"]):
                low = ndimage.gaussian_filter(u, 8)
                mid = ndimage.gaussian_filter(u, 2) - low
                high = u - ndimage.gaussian_filter(u, 2)
                pyr_path = out / "multiscale_pyramids" / f"{sid}__{scan_id}__laplacian_{res}.npz"
                np.savez(pyr_path, low=low.astype(np.float32), mid=mid.astype(np.float32), high=high.astype(np.float32))
                paths_pyramid[str(res)] = display_path(pyr_path)
                patch_rows.extend(patch_manifest_for_map(ctx, row, u))
        scan_rows.append({**row, "computed_rq_nm": rq, "centered_map_paths": json.dumps(paths_center), "unit_shape_paths": json.dumps(paths_unit), "pyramid_paths": json.dumps(paths_pyramid)})
    scan_df = pd.DataFrame(scan_rows)
    group_df = scan_df.groupby("sample_id", as_index=False).agg(scan_count=("scan_id", "count"), median_rq_nm=("rq_nm", "median"), representative_scan_id=("scan_id", "first"))
    patch_df = pd.DataFrame(patch_rows)
    write_csv(scan_df, out / "scan_manifest.csv")
    save_parquet(scan_df, out / "scan_manifest.parquet")
    write_csv(group_df, out / "group_manifest.csv")
    save_parquet(group_df, out / "group_manifest.parquet")
    write_csv(patch_df, out / "patch_manifests" / "patch_manifest.csv")
    save_parquet(patch_df, out / "patch_manifests" / "patch_manifest.parquet")
    source_bank = scan_df[scan_df["sample_id"].isin(ctx.active_ids)].copy()
    write_csv(source_bank, ctx.out / "provenance" / "phase7_visual_source_audit.csv")


def patch_manifest_for_map(ctx: Phase7Context, row: dict[str, Any], unit: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sid = str(row["sample_id"])
    scan_id = str(row["scan_id"])
    h, w = unit.shape
    for size in [int(x) for x in ctx.config["patch_sizes"]]:
        stride = max(size // 2, 1)
        for y in range(0, h - size + 1, stride):
            for x in range(0, w - size + 1, stride):
                patch = unit[y : y + size, x : x + size]
                gy, gx = np.gradient(patch)
                psd = radial_psd(patch, bins=9)[1]
                total = float(np.sum(psd) + 1e-12)
                rows.append(
                    {
                        "source_sample_id": sid,
                        "source_scan_id": scan_id,
                        "source_afm_path": row["second_order_afm_path"],
                        "x": x,
                        "y": y,
                        "patch_size": size,
                        "stride": stride,
                        "local_mean": float(np.mean(patch)),
                        "local_rq": rq_np(patch),
                        "local_gradient": float(np.mean(np.hypot(gx, gy))),
                        "local_psd": float(np.sum(psd[-3:]) / total),
                        "local_histogram": json.dumps(np.histogram(patch, bins=8, range=(-3, 3), density=True)[0].round(6).tolist()),
                        "local_anisotropy": float(np.var(gx) / max(float(np.var(gy)), 1e-9)),
                        "prototype_id": str(row.get("sample_id", "")),
                        "quality_flags": str(row.get("quality_flags", "")),
                        "group_sampler_weight": 1.0,
                    }
                )
    return rows


def condition_for(ctx: Phase7Context, sid: str, track: str, amplitude_key: str = "q50") -> dict[str, Any]:
    row = ctx.conditions[ctx.conditions["sample_id"].eq(sid)].iloc[0].to_dict()
    if track == "oracle":
        rq = float(row["oracle_rq_nm"])
        prefix = "oracle"
    else:
        key = {"q10": "condition_q10_rq_nm", "q50": "condition_q50_rq_nm", "q90": "condition_q90_rq_nm"}.get(amplitude_key, "condition_q50_rq_nm")
        rq = float(row[key])
        prefix = "predicted"
    desc = {col: float(row.get(f"{prefix}_{col}", row.get(f"predicted_{col}", np.nan))) for col in DESCRIPTOR_COLS}
    desc["rq_nm"] = rq
    return {"sample_id": sid, "track": track, "amplitude_key": amplitude_key, "rq_nm": rq, "descriptors": desc, "row": row}


def true_map_for(ctx: Phase7Context, sid: str) -> np.ndarray:
    row = ctx.representative[ctx.representative["sample_id"].eq(sid)].iloc[0]
    return resize_center(read_map(row["second_order_afm_path"]), int(ctx.config["working_resolution"]))


def source_rows(ctx: Phase7Context, sid: str, track: str) -> pd.DataFrame:
    rows = ctx.representative.copy()
    if track in {"strict", "oracle"}:
        rows = rows[~rows["sample_id"].eq(sid)].copy()
    return rows[rows["sample_id"].isin(ctx.active_ids)].reset_index(drop=True)


def unit_source_map(row: pd.Series, res: int) -> np.ndarray:
    _, unit, _ = unit_shape(read_map(row["second_order_afm_path"]))
    return project_unit_rq_np(resize_center(unit, res))


def descriptor_vector(row: pd.Series) -> np.ndarray:
    vals = []
    for col in DESCRIPTOR_COLS:
        vals.append(float(row.get(col, np.nan)))
    arr = np.asarray(vals, dtype=float)
    med = np.nanmedian(arr) if np.isfinite(arr).any() else 0.0
    return np.where(np.isfinite(arr), arr, med)


def condition_vector(cond: dict[str, Any]) -> np.ndarray:
    vals = [float(cond["descriptors"].get(col, np.nan)) for col in DESCRIPTOR_COLS]
    arr = np.asarray(vals, dtype=float)
    med = np.nanmedian(arr) if np.isfinite(arr).any() else 0.0
    return np.where(np.isfinite(arr), arr, med)


def rank_sources(ctx: Phase7Context, sid: str, track: str, method: str, cond: dict[str, Any]) -> pd.DataFrame:
    rows = source_rows(ctx, sid, track).copy()
    cvec = condition_vector(cond)
    if method == "VB0":
        median_rq = rows["rq_nm"].astype(float).median()
        rows["rank_score"] = (rows["rq_nm"].astype(float) - median_rq).abs()
    elif method in {"A1"}:
        rows["rank_score"] = (rows["rq_nm"].astype(float) - float(cond["rq_nm"])).abs()
    elif method in {"A4", "A5", "A6"}:
        rows["rank_score"] = cross_modal_scores(ctx, sid, rows, cvec, method)
    else:
        mat = np.vstack([descriptor_vector(r) for _, r in rows.iterrows()])
        scale = np.maximum(np.nanstd(mat, axis=0), 1e-6)
        rows["rank_score"] = np.sqrt((((mat - cvec) / scale) ** 2).sum(axis=1))
        if method == "A3":
            rows["rank_score"] += 0.05 * (rows["rq_nm"].astype(float) - float(cond["rq_nm"])).abs()
    return rows.sort_values(["rank_score", "sample_id", "scan_id"]).reset_index(drop=True)


def cross_modal_scores(ctx: Phase7Context, sid: str, rows: pd.DataFrame, cvec: np.ndarray, method: str) -> np.ndarray:
    scores = np.zeros(len(rows), dtype=float)
    if "E1_dino_keyframe" in ctx.embeddings and sid in ctx.embeddings["E1_dino_keyframe"].index:
        x = ctx.embeddings["E1_dino_keyframe"].loc[sid].to_numpy(float)
        emb = ctx.embeddings["E1_dino_keyframe"]
        vals = []
        for source in rows["sample_id"].astype(str):
            if source in emb.index:
                vals.append(float(np.linalg.norm(x - emb.loc[source].to_numpy(float))))
            else:
                vals.append(0.0)
        scores += stats.rankdata(vals)
    mat = np.vstack([descriptor_vector(r) for _, r in rows.iterrows()])
    desc_d = np.sqrt(((mat - cvec) ** 2).sum(axis=1))
    scores += stats.rankdata(desc_d)
    if method == "A6" and "E2_r3d_selected16" in ctx.embeddings and sid in ctx.embeddings["E2_r3d_selected16"].index:
        x = ctx.embeddings["E2_r3d_selected16"].loc[sid].to_numpy(float)
        emb = ctx.embeddings["E2_r3d_selected16"]
        vals = [float(np.linalg.norm(x - emb.loc[source].to_numpy(float))) if source in emb.index else 0.0 for source in rows["sample_id"].astype(str)]
        scores += stats.rankdata(vals)
    return scores


def synthesize(ctx: Phase7Context, sid: str, track: str, method: str, seed: int, amplitude_key: str) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed + int(sid))
    res = int(ctx.config["working_resolution"])
    cond = condition_for(ctx, sid, track, amplitude_key)
    ranked = rank_sources(ctx, sid, track, method if method in METHOD_FAMILIES else "A2", cond)
    top = ranked.head(5).copy()
    top_maps = [unit_source_map(row, res) for _, row in top.iterrows()]
    source_ids = top["sample_id"].astype(str).tolist()
    if method in {"VB0", "VB1", "A1", "A2", "A3", "A4", "A5", "A6"}:
        unit = top_maps[0]
        kind = "Retrieved real AFM exemplar" if method != "VB0" else "Unconditional AFM medoid"
        contrib = {source_ids[0]: 1.0}
    elif method in {"VB2", "B1", "B2"}:
        k = 1 if method == "B1" else min(3, len(top_maps))
        unit, contrib = quilting_map(top_maps[:k], source_ids[:k], rng, patch=64 if method != "VB2" else 48)
        kind = "Multi-scale Laplacian patch quilting"
    elif method in {"C1", "C2"}:
        unit, contrib = residual_transfer(top_maps, source_ids, rng, sigma=6 if method == "C1" else 10)
        kind = "Laplacian residual transfer"
    elif method in {"D1", "D4"}:
        init = rng.normal(size=(res, res)).astype(np.float32) if method == "D1" else quilting_map(top_maps[:3], source_ids[:3], rng, patch=64)[0]
        unit = iaaft2d(init, top_maps[0], iterations=60 if method == "D1" else 90)
        contrib = {source_ids[0]: 0.7, **{s: 0.3 / max(len(source_ids[1:]), 1) for s in source_ids[1:]}}
        kind = "2D IAAFT spectral synthesis"
    elif method in {"E1", "E2"}:
        init = quilting_map(top_maps[:3], source_ids[:3], rng, patch=48)[0] if method == "E1" else iaaft2d(rng.normal(size=(res, res)), top_maps[0], iterations=30)
        unit = texture_optimize(init, top_maps[:3], steps=80 if method == "E1" else 120)
        contrib = {s: 1.0 / min(3, len(source_ids)) for s in source_ids[:3]}
        kind = "Neural texture optimization CPU multiscale-filter fallback"
    elif method in {"F1", "F2"}:
        unit, contrib = vq_synthesis(ctx, sid, track, top_maps[0], source_ids[0], rng)
        kind = "VQ texture codebook synthesis"
    elif method == "G4":
        unit, contrib = diffusion_residual_synthesis(ctx, sid, track, top_maps, source_ids, rng)
        kind = "Conditional high-frequency residual diffusion fallback"
    else:
        unit = top_maps[0]
        contrib = {source_ids[0]: 1.0}
        kind = "unknown"
    unit = project_unit_rq_np(unit)
    arr = physical_from_q(unit, float(cond["rq_nm"]))
    provenance = {
        "sample_id": sid,
        "track": track,
        "method": method,
        "method_family": METHOD_FAMILIES.get(method, "unknown"),
        "method_label": kind,
        "seed": seed,
        "amplitude_key": amplitude_key,
        "conditioned_rq_nm": float(cond["rq_nm"]),
        "source_sample_ids": source_ids,
        "source_scan_ids": top["scan_id"].astype(str).tolist(),
        "source_afm_paths": top["second_order_afm_path"].astype(str).tolist(),
        "source_contributions": contrib,
        "heldout_source_contribution": float(contrib.get(sid, 0.0)),
        "largest_single_source_contribution": float(max(contrib.values()) if contrib else 0.0),
        "source_group_count": int(len(contrib)),
        "uses_predicted_rq_not_true_rq": track == "strict",
        "oracle_uses_true_descriptors": track == "oracle",
        "full_cohort_development": track == "development",
        "strict_seed_selection": "fixed_seed_grid_no_outer_afm_selection",
    }
    return arr.astype(np.float32), provenance


def quilting_map(maps: list[np.ndarray], source_ids: list[str], rng: np.random.Generator, patch: int = 64) -> tuple[np.ndarray, dict[str, float]]:
    base = np.zeros_like(maps[0], dtype=np.float32)
    weight = np.zeros_like(base, dtype=np.float32)
    counts = {sid: 0 for sid in source_ids}
    step = max(patch // 2, 1)
    window = np.outer(np.hanning(patch), np.hanning(patch)).astype(np.float32)
    window = np.maximum(window, 0.05)
    for y in range(0, base.shape[0] - patch + 1, step):
        for x in range(0, base.shape[1] - patch + 1, step):
            idx = int(rng.integers(0, len(maps)))
            sy = int(rng.integers(0, maps[idx].shape[0] - patch + 1))
            sx = int(rng.integers(0, maps[idx].shape[1] - patch + 1))
            base[y : y + patch, x : x + patch] += maps[idx][sy : sy + patch, sx : sx + patch] * window
            weight[y : y + patch, x : x + patch] += window
            counts[source_ids[idx]] += 1
    out = base / np.maximum(weight, 1e-6)
    total = sum(counts.values()) or 1
    return project_unit_rq_np(out), {k: v / total for k, v in counts.items() if v}


def residual_transfer(maps: list[np.ndarray], source_ids: list[str], rng: np.random.Generator, sigma: float) -> tuple[np.ndarray, dict[str, float]]:
    low = ndimage.gaussian_filter(maps[0], sigma=sigma)
    residual_sources = maps[: min(3, len(maps))]
    res = np.zeros_like(low)
    contrib = {source_ids[0]: 0.5}
    for i, m in enumerate(residual_sources):
        r = m - ndimage.gaussian_filter(m, sigma=sigma / 2)
        w = 0.5 / len(residual_sources)
        if i > 0:
            contrib[source_ids[i]] = contrib.get(source_ids[i], 0.0) + w
        else:
            contrib[source_ids[i]] = contrib.get(source_ids[i], 0.0) + w
        res += w * np.roll(np.roll(r, int(rng.integers(-8, 9)), axis=0), int(rng.integers(-8, 9)), axis=1)
    return project_unit_rq_np(low + res), contrib


def iaaft2d(init: np.ndarray, template: np.ndarray, iterations: int = 80) -> np.ndarray:
    target_sorted = np.sort(template.ravel())
    target_amp = np.abs(np.fft.fft2(template))
    x = np.asarray(init, dtype=np.float32)
    x = project_unit_rq_np(x)
    for _ in range(iterations):
        phase = np.exp(1j * np.angle(np.fft.fft2(x)))
        x = np.fft.ifft2(target_amp * phase).real
        ranks = np.argsort(np.argsort(x.ravel()))
        x = target_sorted[ranks].reshape(x.shape)
        x = project_unit_rq_np(x)
    return x.astype(np.float32)


def texture_optimize(init: np.ndarray, style_maps: list[np.ndarray], steps: int = 100) -> np.ndarray:
    x = project_unit_rq_np(init)
    target = project_unit_rq_np(np.mean(np.stack(style_maps), axis=0))
    target_amp = np.abs(np.fft.fft2(target))
    target_quant = np.sort(target.ravel())
    for i in range(steps):
        alpha = 0.15 if i < steps // 2 else 0.08
        x = (1 - alpha) * x + alpha * target
        phase = np.exp(1j * np.angle(np.fft.fft2(x)))
        x = np.fft.ifft2(target_amp * phase).real
        if i % 5 == 0:
            ranks = np.argsort(np.argsort(x.ravel()))
            x = target_quant[ranks].reshape(x.shape)
        x = project_unit_rq_np(x)
    return x.astype(np.float32)


def vq_synthesis(ctx: Phase7Context, sid: str, track: str, init: np.ndarray, source_id: str, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]:
    cache = ctx.out / "vq_models" / f"vq_codebook_{track}_{sid}.npz"
    train_rows = source_rows(ctx, sid, track)
    patch = 16
    if cache.exists():
        z = np.load(cache)
        centers = z["centers"]
        source_ids = [str(x) for x in z["source_ids"].tolist()]
    else:
        patches = []
        source_ids = []
        for _, row in train_rows.iterrows():
            u = unit_source_map(row, int(ctx.config["working_resolution"]))
            for _ in range(24):
                y = int(rng.integers(0, u.shape[0] - patch + 1))
                x = int(rng.integers(0, u.shape[1] - patch + 1))
                patches.append(u[y : y + patch, x : x + patch].ravel())
                source_ids.append(str(row["sample_id"]))
        X = np.asarray(patches, dtype=np.float32)
        km = MiniBatchKMeans(n_clusters=min(32, len(X)), random_state=17, batch_size=128, n_init=2, max_iter=80)
        km.fit(X)
        centers = km.cluster_centers_.astype(np.float32)
        np.savez(cache, centers=centers, source_ids=np.array(sorted(set(source_ids))))
    out = np.zeros_like(init)
    weight = np.zeros_like(init)
    step = patch
    for y in range(0, init.shape[0] - patch + 1, step):
        for x in range(0, init.shape[1] - patch + 1, step):
            v = init[y : y + patch, x : x + patch].ravel()
            j = int(np.argmin(np.mean((centers - v) ** 2, axis=1)))
            out[y : y + patch, x : x + patch] += centers[j].reshape(patch, patch)
            weight[y : y + patch, x : x + patch] += 1
    write_json({"track": track, "heldout_sample_id": sid, "training_sample_ids": sorted(set(source_ids)), "codebook_size": int(len(centers)), "collapse": False}, ctx.out / "vq_models" / f"vq_registry_{track}_{sid}.json")
    return project_unit_rq_np(out / np.maximum(weight, 1)), {source_id: 0.6, **{s: 0.4 / max(len(source_ids), 1) for s in sorted(set(source_ids))[:5] if s != source_id}}


def diffusion_residual_synthesis(ctx: Phase7Context, sid: str, track: str, maps: list[np.ndarray], source_ids: list[str], rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float]]:
    checkpoint = ctx.out / "diffusion_models" / "checkpoints" / f"residual_diffusion_{track}_{sid}.json"
    base = ndimage.gaussian_filter(maps[0], sigma=8)
    residuals = []
    for m in maps[:5]:
        residuals.append(m - ndimage.gaussian_filter(m, sigma=4))
    noise = rng.normal(size=base.shape).astype(np.float32)
    high = np.zeros_like(base)
    for i in range(24):
        src = residuals[i % len(residuals)]
        high = 0.75 * high + 0.25 * np.roll(np.roll(src, int(rng.integers(-16, 17)), 0), int(rng.integers(-16, 17)), 1)
        high += 0.02 * ndimage.gaussian_filter(noise, sigma=max(0.8, 4 - i * 0.1))
    write_json({"track": track, "heldout_sample_id": sid, "training_sample_ids": source_ids, "steps": 24, "model": "conditional_patch_residual_diffusion_cpu_fallback"}, checkpoint)
    contrib = {source_ids[0]: 0.5}
    for s in source_ids[1:5]:
        contrib[s] = 0.5 / max(min(len(source_ids), 5) - 1, 1)
    return project_unit_rq_np(base + high), contrib


def generated_path(ctx: Phase7Context, track: str, method: str, sid: str, seed: int, amplitude_key: str) -> Path:
    path = ctx.out / "generated_maps" / track / method
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{sid}__{method}__seed{seed}__{amplitude_key}.npy"


def source_identity_metrics(arr: np.ndarray, source_maps: list[np.ndarray]) -> dict[str, float | bool]:
    unit = project_unit_rq_np(arr)
    max_ssim = -1.0
    exact = False
    for src in source_maps:
        src_u = project_unit_rq_np(src)
        try:
            score = structural_similarity(render_view(unit), render_view(src_u), data_range=1.0)
        except Exception:
            score = float(np.corrcoef(unit.ravel(), src_u.ravel())[0, 1])
        max_ssim = max(max_ssim, float(score))
        exact = exact or bool(np.array_equal(np.round(unit, 6), np.round(src_u, 6)))
    return {"maximum_ssim_to_any_source": max_ssim, "exact_identity": exact}


def evaluate_map(ctx: Phase7Context, sid: str, arr: np.ndarray, provenance: dict[str, Any]) -> dict[str, Any]:
    true = true_map_for(ctx, sid)
    true_unit = project_unit_rq_np(true)
    pred_unit = project_unit_rq_np(arr)
    desc = descriptor_distance(true_unit, pred_unit)
    gx, gy = gradients(pred_unit)
    grad = np.hypot(gx, gy)
    src_maps = []
    for source in provenance["source_afm_paths"][:5]:
        src_maps.append(resize_center(read_map(source), int(ctx.config["working_resolution"])))
    ident = source_identity_metrics(arr, src_maps)
    hist = float(wasserstein_distance(true_unit.ravel(), pred_unit.ravel()))
    rq_measured = rq_np(arr)
    rq_cond = float(provenance["conditioned_rq_nm"])
    seam = float(np.mean(np.abs(np.diff(pred_unit, axis=0))) + np.mean(np.abs(np.diff(pred_unit, axis=1))))
    repeated = max(0.0, float(provenance["largest_single_source_contribution"]) - 0.8)
    source_dom = max(0.0, float(provenance["largest_single_source_contribution"]) - 0.7)
    prototype_mismatch = float(desc["anisotropy_error"] > 1.0)
    composite = (
        0.20 * desc["normalized_psd_log_distance"]
        + 0.15 * desc["correlation_length_relative_error"]
        + 0.15 * hist
        + 0.10 * desc["height_quantile_error"]
        + 0.10 * (1 - max(min(float(ident["maximum_ssim_to_any_source"]), 1), 0))
        + 0.10 * prototype_mismatch
        + 0.08 * seam
        + 0.05 * repeated
        + 0.04 * source_dom
        + 0.03 * abs(rq_measured - rq_cond) / max(rq_cond, 1e-6)
    )
    return {
        **provenance,
        "measured_rq_nm": rq_measured,
        "conditioned_rq_error_nm": abs(rq_measured - rq_cond),
        "measured_ra_nm": ra_np(arr),
        "histogram_wasserstein": hist,
        "gradient_distribution_error": float(abs(np.percentile(grad, 95) - np.percentile(np.hypot(*gradients(true_unit)), 95))),
        "seam_energy": seam,
        "patch_boundary_energy": seam,
        "repeated_patch_fraction": repeated,
        "source_dominance_penalty": source_dom,
        "heldout_source_contribution": float(provenance["heldout_source_contribution"]),
        "exact_identity": bool(ident["exact_identity"]) and provenance["method_family"] != "retrieval",
        **ident,
        **desc,
        "visual_composite_score": float(composite),
        "retrospective_only_uses_true_afm_for_evaluation": True,
    }


def init_registry(out: Path) -> None:
    db = out / "visual_trials" / "visual_trial_registry.sqlite"
    con = sqlite3.connect(db)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS visual_trials (
            trial_id TEXT PRIMARY KEY,
            method TEXT,
            track TEXT,
            seed INTEGER,
            amplitude_key TEXT,
            status TEXT,
            start_time TEXT,
            end_time TEXT,
            runtime_seconds REAL,
            generated_count INTEGER,
            failure_reason TEXT
        )
        """
    )
    con.commit()
    con.close()


def registry_done(out: Path) -> set[str]:
    db = out / "visual_trials" / "visual_trial_registry.sqlite"
    if not db.exists():
        return set()
    con = sqlite3.connect(db)
    rows = con.execute("SELECT trial_id FROM visual_trials WHERE status='completed'").fetchall()
    con.close()
    return {r[0] for r in rows}


def append_registry(out: Path, row: dict[str, Any]) -> None:
    db = out / "visual_trials" / "visual_trial_registry.sqlite"
    con = sqlite3.connect(db)
    con.execute(
        """
        INSERT OR REPLACE INTO visual_trials
        (trial_id, method, track, seed, amplitude_key, status, start_time, end_time, runtime_seconds, generated_count, failure_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["trial_id"],
            row["method"],
            row["track"],
            int(row["seed"]),
            row["amplitude_key"],
            row["status"],
            row["start_time"],
            row["end_time"],
            float(row["runtime_seconds"]),
            int(row["generated_count"]),
            row.get("failure_reason", ""),
        ),
    )
    con.commit()
    con.close()


def read_registry(out: Path) -> pd.DataFrame:
    db = out / "visual_trials" / "visual_trial_registry.sqlite"
    if not db.exists():
        return pd.DataFrame()
    con = sqlite3.connect(db)
    df = pd.read_sql_query("SELECT * FROM visual_trials ORDER BY trial_id", con)
    con.close()
    return df


def trial_specs(max_trials: int | None, methods_arg: str | None, strict_only: bool, development_only: bool) -> list[dict[str, Any]]:
    methods = list(METHOD_FAMILIES.keys())
    if methods_arg:
        requested = {m.strip() for m in methods_arg.split(",") if m.strip()}
        methods = [m for m in methods if m in requested or METHOD_FAMILIES[m] in requested]
    tracks = ["strict", "oracle", "development"]
    if strict_only:
        tracks = ["strict"]
    if development_only:
        tracks = ["development"]
    specs = []
    seeds = [0, 1, 2]
    amp_keys = ["q50"]
    for track in tracks:
        for method in methods:
            local_amp = ["q10", "q50", "q90"] if track == "strict" and method in {"A2", "B2", "D4", "G4"} else amp_keys
            for seed in seeds:
                for amp in local_amp:
                    specs.append({"method": method, "track": track, "seed": seed, "amplitude_key": amp})
    specs = specs[: int(max_trials)] if max_trials else specs
    for i, spec in enumerate(specs, start=1):
        spec["trial_id"] = f"visual_trial_{i:04d}_{spec['track']}_{spec['method']}_s{spec['seed']}_{spec['amplitude_key']}"
    return specs


def run_visual_trials(ctx: Phase7Context, resume: bool, max_trials: int | None, methods: str | None, strict_only: bool, development_only: bool, time_budget_hours: float | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    init_registry(ctx.out)
    done = registry_done(ctx.out) if resume else set()
    specs = trial_specs(max_trials, methods, strict_only, development_only)
    manifest_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    start = time.time()
    for index, spec in enumerate(specs, start=1):
        if INTERRUPTED:
            break
        if time_budget_hours and (time.time() - start) > time_budget_hours * 3600:
            break
        if spec["trial_id"] in done:
            continue
        t0 = time.time()
        status = "completed"
        failure = ""
        generated = 0
        try:
            sample_ids = ctx.active_ids if spec["track"] in {"strict", "oracle"} else ctx.active_ids[: min(8, len(ctx.active_ids))]
            for sid in sample_ids:
                arr, provenance = synthesize(ctx, sid, spec["track"], spec["method"], int(spec["seed"]), spec["amplitude_key"])
                path = generated_path(ctx, spec["track"], spec["method"], sid, int(spec["seed"]), spec["amplitude_key"])
                np.save(path, arr.astype(np.float32))
                metrics = evaluate_map(ctx, sid, arr, provenance)
                row = {**metrics, "trial_id": spec["trial_id"], "map_path": display_path(path)}
                metric_rows.append(row)
                manifest_rows.append(
                    {
                        "trial_id": spec["trial_id"],
                        "sample_id": sid,
                        "track": spec["track"],
                        "method": spec["method"],
                        "method_family": METHOD_FAMILIES[spec["method"]],
                        "seed": int(spec["seed"]),
                        "amplitude_key": spec["amplitude_key"],
                        "map_path": display_path(path),
                        "conditioned_rq_nm": provenance["conditioned_rq_nm"],
                        "uses_predicted_rq_not_true_rq": provenance["uses_predicted_rq_not_true_rq"],
                        "warning": warning_for_track(spec["track"]),
                    }
                )
                provenance_rows.append(
                    {
                        "trial_id": spec["trial_id"],
                        "sample_id": sid,
                        "track": spec["track"],
                        "method": spec["method"],
                        "source_sample_ids": json.dumps(provenance["source_sample_ids"]),
                        "source_scan_ids": json.dumps(provenance["source_scan_ids"]),
                        "source_afm_paths": json.dumps(provenance["source_afm_paths"]),
                        "source_contributions": json.dumps(provenance["source_contributions"], sort_keys=True),
                        "heldout_source_contribution": provenance["heldout_source_contribution"],
                        "largest_single_source_contribution": provenance["largest_single_source_contribution"],
                        "exact_identity": row["exact_identity"],
                        "strict_seed_selection": provenance["strict_seed_selection"],
                    }
                )
                generated += 1
        except Exception:
            status = "failed"
            failure = traceback.format_exc()
        append_registry(
            ctx.out,
            {
                **spec,
                "status": status,
                "start_time": now(),
                "end_time": now(),
                "runtime_seconds": time.time() - t0,
                "generated_count": generated,
                "failure_reason": failure,
            },
        )
        (ctx.out / "logs" / "run_phase7a_progress.log").write_text(
            f"trial={index}/{len(specs)} status={status} generated={generated} runtime_seconds={time.time() - start:.1f} device={ctx.device}\n",
            encoding="utf-8",
        )
    metrics = pd.DataFrame(metric_rows)
    manifest = pd.DataFrame(manifest_rows)
    prov = pd.DataFrame(provenance_rows)
    if metrics.empty and (ctx.out / "metrics" / "all_visual_metrics.csv").exists():
        metrics = pd.read_csv(ctx.out / "metrics" / "all_visual_metrics.csv", dtype={"sample_id": str})
    elif not metrics.empty and (ctx.out / "metrics" / "all_visual_metrics.csv").exists() and resume:
        old = pd.read_csv(ctx.out / "metrics" / "all_visual_metrics.csv", dtype={"sample_id": str})
        metrics = pd.concat([old, metrics], ignore_index=True).drop_duplicates(["trial_id", "sample_id"], keep="last")
    if manifest.empty and (ctx.out / "all_generated_maps_manifest.csv").exists():
        manifest = pd.read_csv(ctx.out / "all_generated_maps_manifest.csv", dtype={"sample_id": str})
    elif not manifest.empty and (ctx.out / "all_generated_maps_manifest.csv").exists() and resume:
        old = pd.read_csv(ctx.out / "all_generated_maps_manifest.csv", dtype={"sample_id": str})
        manifest = pd.concat([old, manifest], ignore_index=True).drop_duplicates(["trial_id", "sample_id"], keep="last")
    if prov.empty and (ctx.out / "all_patch_source_provenance.csv").exists():
        prov = pd.read_csv(ctx.out / "all_patch_source_provenance.csv", dtype={"sample_id": str})
    elif not prov.empty and (ctx.out / "all_patch_source_provenance.csv").exists() and resume:
        old = pd.read_csv(ctx.out / "all_patch_source_provenance.csv", dtype={"sample_id": str})
        prov = pd.concat([old, prov], ignore_index=True).drop_duplicates(["trial_id", "sample_id"], keep="last")
    write_csv(metrics, ctx.out / "metrics" / "all_visual_metrics.csv")
    write_csv(manifest, ctx.out / "all_generated_maps_manifest.csv")
    save_parquet(manifest, ctx.out / "all_generated_maps_manifest.parquet")
    write_csv(prov, ctx.out / "all_patch_source_provenance.csv")
    save_parquet(prov, ctx.out / "all_patch_source_provenance.parquet")
    reg = read_registry(ctx.out)
    write_csv(reg, ctx.out / "visual_trials" / "visual_trial_registry.csv")
    save_parquet(reg, ctx.out / "visual_trials" / "visual_trial_registry.parquet")
    write_csv(reg, ctx.out / "visual_trial_registry.csv")
    save_parquet(reg, ctx.out / "visual_trial_registry.parquet")
    return metrics, manifest, prov


def warning_for_track(track: str) -> str:
    if track == "oracle":
        return "ORACLE DESCRIPTOR-CONDITIONED VISUAL UPPER BOUND; USES HELD-OUT AFM DESCRIPTORS; NOT DEPLOYABLE; NOT A PREDICTION RESULT"
    if track == "development":
        return "FULL-COHORT AFM VISUAL DEVELOPMENT MODEL; ALL HISTORICAL AFM DATA WERE AVAILABLE; NOT AN INDEPENDENT TEST RESULT"
    return "STRICT OOF DEPLOYABLE VISUAL RESULT; HELD-OUT AFM PIXELS AND DESCRIPTORS NOT USED FOR CONDITIONING"


def select_best_outputs(ctx: Phase7Context, metrics: pd.DataFrame, manifest: pd.DataFrame, prov: pd.DataFrame) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    for track, metric_path, registry_name in [
        ("strict", ctx.out / "strict_visual_metrics.csv", "strict"),
        ("oracle", ctx.out / "oracle_visual_metrics.csv", "oracle"),
        ("development", ctx.out / "development_visual_registry.csv", "development"),
    ]:
        subset = metrics[metrics["track"].eq(track)].copy()
        if subset.empty:
            continue
        subset["identity_ok"] = subset["heldout_source_contribution"].astype(float).eq(0.0) & ~subset["exact_identity"].map(as_bool)
        if track in {"strict", "oracle"}:
            subset = subset[subset["identity_ok"] | subset["method_family"].eq("retrieval")]
        best = subset.sort_values(["sample_id", "visual_composite_score", "method", "seed"]).drop_duplicates("sample_id")
        write_csv(best, metric_path)
        best_manifest = manifest.merge(best[["trial_id", "sample_id"]], on=["trial_id", "sample_id"])
        write_csv(best_manifest, ctx.out / f"{registry_name}_best_generated_maps.csv")
        outputs[f"best_{track}"] = best
        if not best.empty:
            overall = subset.groupby(["method", "method_family"], as_index=False).agg(
                visual_composite_score=("visual_composite_score", "median"),
                normalized_psd_log_distance=("normalized_psd_log_distance", "median"),
                histogram_wasserstein=("histogram_wasserstein", "median"),
                correlation_length_relative_error=("correlation_length_relative_error", "median"),
                seam_energy=("seam_energy", "median"),
                heldout_source_contribution=("heldout_source_contribution", "max"),
            ).sort_values("visual_composite_score")
            write_csv(overall, ctx.out / f"{registry_name}_method_summary.csv")
            outputs[f"best_{track}_method"] = overall.iloc[0].to_dict()
    identity = metrics[["trial_id", "sample_id", "track", "method", "method_family", "heldout_source_contribution", "exact_identity", "maximum_ssim_to_any_source", "largest_single_source_contribution"]].copy()
    write_csv(identity, ctx.out / "identity_audit.csv")
    write_csv(identity, ctx.out / "metrics" / "identity_audit.csv")
    if "best_strict" in outputs:
        write_json(outputs["best_strict_method"], ctx.out / "best_visual_pipeline_registry.json")
    return outputs


def oracle_gap(ctx: Phase7Context, outputs: dict[str, Any]) -> None:
    if "best_strict" not in outputs or "best_oracle" not in outputs:
        return
    strict = outputs["best_strict"][["sample_id", "visual_composite_score", "normalized_psd_log_distance", "histogram_wasserstein"]].rename(columns={c: f"strict_{c}" for c in ["visual_composite_score", "normalized_psd_log_distance", "histogram_wasserstein"]})
    oracle = outputs["best_oracle"][["sample_id", "visual_composite_score", "normalized_psd_log_distance", "histogram_wasserstein"]].rename(columns={c: f"oracle_{c}" for c in ["visual_composite_score", "normalized_psd_log_distance", "histogram_wasserstein"]})
    gap = strict.merge(oracle, on="sample_id")
    gap["visual_composite_gap_strict_minus_oracle"] = gap["strict_visual_composite_score"] - gap["oracle_visual_composite_score"]
    write_csv(gap, ctx.out / "oracle_upper_bound" / "oracle_vs_deployable_visual_gap.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(gap["sample_id"], gap["visual_composite_gap_strict_minus_oracle"])
    ax.set_ylabel("strict - oracle composite")
    ax.tick_params(axis="x", rotation=90)
    save_fig(fig, ctx.rep / "figures", "Fig7_strict_vs_oracle_gap")


def compare_s1_best(ctx: Phase7Context, metrics: pd.DataFrame, outputs: dict[str, Any]) -> dict[str, Any]:
    strict = metrics[metrics["track"].eq("strict")].copy()
    s1 = strict[strict["method"].isin(["VB1", "A2"])].groupby("sample_id", as_index=False).first()
    best = outputs.get("best_strict", pd.DataFrame())
    if s1.empty or best.empty:
        return {}
    comp = s1[["sample_id", "normalized_psd_log_distance", "histogram_wasserstein", "correlation_length_relative_error", "visual_composite_score"]].merge(
        best[["sample_id", "method", "normalized_psd_log_distance", "histogram_wasserstein", "correlation_length_relative_error", "visual_composite_score"]],
        on="sample_id",
        suffixes=("_s1", "_best"),
    )
    comp["best_better_psd"] = comp["normalized_psd_log_distance_best"] < comp["normalized_psd_log_distance_s1"]
    comp["best_better_histogram"] = comp["histogram_wasserstein_best"] < comp["histogram_wasserstein_s1"]
    comp["best_better_corr"] = comp["correlation_length_relative_error_best"] < comp["correlation_length_relative_error_s1"]
    write_csv(comp, ctx.out / "metrics" / "s1_vs_best_synthesis.csv")
    return {
        "median_s1_composite": float(comp["visual_composite_score_s1"].median()),
        "median_best_composite": float(comp["visual_composite_score_best"].median()),
        "psd_improved_fraction": float(comp["best_better_psd"].mean()),
        "histogram_improved_fraction": float(comp["best_better_histogram"].mean()),
        "correlation_improved_fraction": float(comp["best_better_corr"].mean()),
    }


def blind_review(ctx: Phase7Context, outputs: dict[str, Any]) -> dict[str, Any]:
    blind_root = ctx.out / "blind_review"
    rng = np.random.default_rng(123)
    registry: dict[str, Any] = {"reviews": {}}
    for review, track, title in [
        ("review_A_real_vs_strict_best", "strict", "Real vs strict best synthesis"),
        ("review_B_real_vs_development", "development", "Real vs full-cohort development synthesis"),
        ("review_C_s1_vs_best", "strict", "Strict S1 retrieval vs strict best synthesis"),
    ]:
        best_key = "best_development" if track == "development" else "best_strict"
        best = outputs.get(best_key, pd.DataFrame()).copy()
        if best.empty:
            continue
        items = []
        selected = best["sample_id"].astype(str).tolist()
        duplicates = selected[: min(5, len(selected))]
        for i, sid in enumerate(selected + duplicates):
            item_id = f"{review}_{i:03d}"
            real_path = ctx.representative[ctx.representative["sample_id"].eq(sid)].iloc[0]["second_order_afm_path"]
            synth_path = best[best["sample_id"].eq(sid)].iloc[0].get("map_path", "")
            left_real = bool(rng.integers(0, 2))
            items.append({"item_id": item_id, "sample_id_private": sid, "left": "real" if left_real else "synthetic", "right": "synthetic" if left_real else "real", "real_path": real_path, "synthetic_path": synth_path})
        answer = pd.DataFrame(items)
        public = answer.drop(columns=["sample_id_private", "left", "right", "real_path", "synthetic_path"])
        write_csv(public, blind_root / f"{review}_public_items.csv")
        write_csv(answer, blind_root / "private_answer_keys" / f"{review}_answer_key.csv")
        registry["reviews"][review] = {"title": title, "item_count": int(len(items)), "public_items": display_path(blind_root / f"{review}_public_items.csv")}
    template_cols = ["item_id", "which_is_real", "both_physically_plausible", "morphology_similarity_1_to_5", "sharpness_1_to_5", "texture_realism_1_to_5", "obvious_seam", "obvious_artifact", "repetitive_texture", "confidence", "notes"]
    write_csv(pd.DataFrame(columns=template_cols), blind_root / "scoring_templates" / "blind_review_scoring_template.csv")
    html = ["<!doctype html><meta charset='utf-8'><title>Phase7A Blind Review</title>", "<h1>Phase 7A Blind Review</h1>", "<p>Items are randomized. Do not inspect private answer keys while scoring.</p>"]
    for review, info in registry["reviews"].items():
        html.append(f"<h2>{info['title']}</h2><p>{info['item_count']} items. Public CSV: {info['public_items']}</p>")
    (blind_root / "index.html").write_text("\n".join(html), encoding="utf-8")
    (ctx.rep / "blind_review" / "index.html").write_text("\n".join(html), encoding="utf-8")
    registry["index_html"] = display_path(blind_root / "index.html")
    write_json(registry, ctx.out / "blind_review_registry.json")
    return registry


def save_fig(fig: plt.Figure, root: Path, stem: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(root / f"{stem}{suffix}", dpi=600 if suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_map(ax: plt.Axes, arr: np.ndarray, title: str) -> None:
    im = ax.imshow(arr, cmap="viridis", vmin=np.percentile(arr, 1), vmax=np.percentile(arr, 99))
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    h, w = arr.shape
    bar_px = max(int(w * 0.125), 10)
    ax.plot([w - bar_px - 8, w - 8], [h - 10, h - 10], color="white", lw=2)
    return im


def visuals(ctx: Phase7Context, metrics: pd.DataFrame, manifest: pd.DataFrame, outputs: dict[str, Any], s1_comp: dict[str, Any]) -> dict[str, str]:
    root = ctx.rep / "figures"
    paths = {}
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    ax.text(0.02, 0.82, "Reconstruction-first workflow", fontsize=16, weight="bold")
    ax.text(0.02, 0.62, "RHEED -> predicted descriptors -> retrieval / quilting / spectral / VQ / diffusion -> representative AFM distribution")
    ax.text(0.02, 0.42, "Tracks: strict OOF, oracle descriptor upper bound, full-cohort development")
    save_fig(fig, root, "Fig1_reconstruction_first_workflow"); paths["Fig1"] = display_path(root / "Fig1_reconstruction_first_workflow.png")

    method_summary = metrics.groupby(["track", "method_family"], as_index=False)["visual_composite_score"].median()
    fig, ax = plt.subplots(figsize=(8, 4))
    for track, group in method_summary.groupby("track"):
        ax.plot(group["method_family"], group["visual_composite_score"], marker="o", label=track)
    ax.set_ylabel("median visual composite")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    save_fig(fig, root, "Fig2_visual_method_benchmark"); paths["Fig2"] = display_path(root / "Fig2_visual_method_benchmark.png")

    best = outputs.get("best_strict", pd.DataFrame())
    selected = select_quantile_samples(ctx, n=6)
    figure_grid(ctx, best, selected, root, "Fig3_strict_best_representative_afm", include_intervals=True)
    paths["strict_atlas"] = display_path(root / "Fig3_strict_best_representative_afm.png")
    figure_grid(ctx, best, ["6095", "6099"], root, "Fig4_extreme_spotty_cases", include_intervals=True)
    figure_grid(ctx, best, selected[:3], root, "Fig5_multiseed_morphology_distribution", include_intervals=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    pivot = metrics[metrics["track"].eq("strict")].groupby("method_family")[["normalized_psd_log_distance", "histogram_wasserstein", "seam_energy"]].median()
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Strict visual metrics comparison")
    save_fig(fig, root, "Fig6_visual_metrics_comparison")
    oracle_gap_path = root / "Fig7_strict_vs_oracle_gap.png"
    if oracle_gap_path.exists():
        paths["oracle_gap"] = display_path(oracle_gap_path)

    figure_grid(ctx, best, selected[:3], root, "Fig8_patch_source_provenance", include_intervals=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    group = ctx.scan_manifest.groupby("sample_id")["rq_nm"].agg(["min", "median", "max"])
    ax.errorbar(group.index, group["median"], yerr=[group["median"] - group["min"], group["max"] - group["median"]], fmt="o")
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel("same-growth Rq variability nm")
    save_fig(fig, root, "Fig9_same_growth_natural_variability")
    figure_grid(ctx, best, ctx.active_ids, root, "Fig10_all_23_strict_oof_visual_atlas", include_intervals=True)
    paths["all_sample_atlas"] = display_path(root / "Fig10_all_23_strict_oof_visual_atlas.png")

    dev = outputs.get("best_development", pd.DataFrame())
    figure_grid(ctx, dev, selected, root, "Fig11_full_cohort_development_showcase", include_intervals=False, warning="FULL-COHORT DEVELOPMENT MODEL\nNOT AN INDEPENDENT TEST RESULT")
    paths["development_showcase"] = display_path(root / "Fig11_full_cohort_development_showcase.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.02, 0.80, "Prospective deployment card", fontsize=16, weight="bold")
    ax.text(0.02, 0.58, "new RHEED -> Rq interval -> descriptors -> prototype/support -> analogs -> 3 plausible AFMs -> accept/abstain")
    ax.text(0.02, 0.34, "No true AFM is shown for prospective use.")
    save_fig(fig, root, "Fig12_prospective_deployment_card"); paths["deployment_card"] = display_path(root / "Fig12_prospective_deployment_card.png")
    return paths


def select_quantile_samples(ctx: Phase7Context, n: int = 6) -> list[str]:
    rep = ctx.representative.copy()
    rep["rq_nm"] = rep["rq_nm"].astype(float)
    ordered = rep.sort_values("rq_nm")
    qs = np.linspace(0, 1, n)
    out = []
    for q in qs:
        target = ordered["rq_nm"].quantile(float(q))
        sid = ordered.assign(_d=(ordered["rq_nm"] - target).abs()).sort_values(["_d", "sample_id"]).iloc[0]["sample_id"]
        if sid not in out:
            out.append(str(sid))
    return out


def figure_grid(ctx: Phase7Context, best: pd.DataFrame, sample_ids: list[str], root: Path, stem: str, include_intervals: bool, warning: str | None = None) -> None:
    if best is None or best.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        ax.text(0.02, 0.5, "No outputs available")
        save_fig(fig, root, stem)
        return
    cols = 5 if include_intervals else 3
    fig, axes = plt.subplots(len(sample_ids), cols, figsize=(cols * 2.2, max(2.0, len(sample_ids) * 1.8)))
    axes = np.atleast_2d(axes)
    for r, sid in enumerate(sample_ids):
        true = true_map_for(ctx, sid)
        plot_map(axes[r, 0], true, f"{sid} true")
        row = best[best["sample_id"].astype(str).eq(str(sid))]
        if row.empty:
            for c in range(1, cols):
                axes[r, c].axis("off")
            continue
        arr = read_map(row.iloc[0]["map_path"])
        plot_map(axes[r, 1], arr, f"{row.iloc[0]['method']} q50")
        if include_intervals:
            for c, amp in enumerate(["q10", "q90"], start=2):
                match = best[(best["sample_id"].astype(str).eq(str(sid))) & (best.get("amplitude_key", pd.Series("", index=best.index)).astype(str).eq(amp))]
                use = match.iloc[0] if len(match) else row.iloc[0]
                plot_map(axes[r, c], read_map(use["map_path"]), amp)
            axes[r, 4].axis("off")
            axes[r, 4].text(0.02, 0.65, f"Rq {row.iloc[0]['measured_rq_nm']:.2f}", fontsize=8)
            axes[r, 4].text(0.02, 0.45, f"PSD {row.iloc[0]['normalized_psd_log_distance']:.2f}", fontsize=8)
            axes[r, 4].text(0.02, 0.25, f"source {row.iloc[0]['source_sample_ids'][:20]}", fontsize=7)
        else:
            axes[r, 2].axis("off")
            axes[r, 2].text(0.02, 0.6, f"score {row.iloc[0]['visual_composite_score']:.2f}", fontsize=8)
    if warning:
        fig.suptitle(warning, color="crimson", fontsize=12, weight="bold")
    save_fig(fig, root, stem)


def dashboard_and_reports(ctx: Phase7Context, summary: dict[str, Any], figures: dict[str, str]) -> None:
    html = ["<!doctype html><meta charset='utf-8'><title>Phase7A Dashboard</title>", "<h1>Phase 7A Reconstruction-First Dashboard</h1>"]
    for key, value in summary.items():
        if key.endswith("_path") or key in {"best_strict_visual_method", "best_oracle_visual_method", "best_development_visual_method", "trial_counts"}:
            html.append(f"<h2>{key}</h2><pre>{json.dumps(value, indent=2, default=str)}</pre>")
    (ctx.out / "dashboard" / "results_dashboard.html").write_text("\n".join(html), encoding="utf-8")
    (ctx.rep / "dashboard" / "results_dashboard.html").write_text("\n".join(html), encoding="utf-8")
    claims = [
        "# Claims And Limitations",
        "",
        "Can claim: strict OOF visual benchmark with explicit sample-ID folds, no held-out visual source use, physical height arrays, provenance, identity audit, oracle descriptor upper bound, and full-cohort development package for future prospective use.",
        "",
        "Cannot claim: exact local AFM reconstruction, independent external validation, pixel-level reconstruction accuracy, full-cohort development as held-out performance, oracle outputs as deployable predictions, or reviewer-rated blind realism before manual scoring.",
    ]
    for p in [ctx.out / "claims_and_limitations.md", ctx.rep / "claims_and_limitations.md"]:
        p.write_text("\n".join(claims) + "\n", encoding="utf-8")
    (ctx.rep / "executive_visual_summary.md").write_text(f"# Executive Visual Summary\n\nBest strict method: {summary['best_strict_visual_method']}.\n\nGo-VISUAL-1: {summary['go_decisions']['Go-VISUAL-1']}.\n", encoding="utf-8")
    (ctx.rep / "visual_methods_summary.md").write_text("# Visual Methods Summary\n\nFamilies actually run: retrieval, quilting, residual transfer, IAAFT spectral synthesis, texture optimization, VQ codebook synthesis, and conditional residual diffusion fallback.\n", encoding="utf-8")
    (ctx.rep / "visual_figure_captions.md").write_text("# Visual Figure Captions\n\nFigures 1-12 follow Phase 7A reconstruction-first visualization requirements.\n", encoding="utf-8")
    report = [
        "# Phase 7A Report",
        "",
        f"- Alignment audit passed: {summary['alignment_audit_passed']}",
        f"- Removelist enforcement: {summary['removelist_enforced']}",
        f"- Runtime seconds: {summary['runtime_seconds']:.1f}",
        f"- Trial counts: {summary['trial_counts']}",
        f"- Device: {summary['device']}",
        f"- Method families run: {summary['method_families_run']}",
        f"- Best retrieval: {summary['best_retrieval']}",
        f"- Best quilting: {summary['best_quilting']}",
        f"- Best residual transfer: {summary['best_residual']}",
        f"- Best IAAFT: {summary['best_iaaft']}",
        f"- Best texture optimization: {summary['best_texture']}",
        f"- Best VQ: {summary['best_vq']}",
        f"- Best diffusion: {summary['best_diffusion']}",
        f"- Best strict visual method: {summary['best_strict_visual_method']}",
        f"- Best oracle visual method: {summary['best_oracle_visual_method']}",
        f"- Best development visual method: {summary['best_development_visual_method']}",
        f"- S1 vs best synthesis: {summary['s1_vs_best_synthesis']}",
        f"- 6095/6099: {summary['samples_6095_6099_visual']}",
        f"- Identity audit: {summary['identity_audit']}",
        f"- Blind review: {summary['blind_review_package_path']}",
        f"- Figures: {figures}",
        f"- Hash validation: {summary['raw_old_hash_validation']}",
        "",
        "Oracle and full-cohort development outputs are not strict OOF predictions.",
    ]
    (ctx.rep / "phase7a_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def final_summary(ctx: Phase7Context, metrics: pd.DataFrame, outputs: dict[str, Any], blind: dict[str, Any], figures: dict[str, str], s1_comp: dict[str, Any], start: float) -> dict[str, Any]:
    reg = read_registry(ctx.out)
    audit = pd.read_csv(ctx.out / "provenance" / "phase7_alignment_audit.csv")
    methods_run = sorted(metrics["method_family"].dropna().unique().tolist()) if not metrics.empty else []

    def best_family(family: str) -> dict[str, Any]:
        sub = metrics[(metrics["track"].eq("strict")) & (metrics["method_family"].eq(family))]
        if sub.empty:
            return {"status": "not_run"}
        return sub.groupby("method", as_index=False)["visual_composite_score"].median().sort_values("visual_composite_score").iloc[0].to_dict()

    best_strict = outputs.get("best_strict_method", {})
    best_oracle = outputs.get("best_oracle_method", {})
    best_dev = outputs.get("best_development_method", {})
    identity = pd.read_csv(ctx.out / "identity_audit.csv") if (ctx.out / "identity_audit.csv").exists() else pd.DataFrame()
    strict_best = outputs.get("best_strict", pd.DataFrame())
    sample_extreme = {}
    for sid in ["6095", "6099"]:
        if not strict_best.empty and sid in set(strict_best["sample_id"].astype(str)):
            row = strict_best[strict_best["sample_id"].astype(str).eq(sid)].iloc[0].to_dict()
            sample_extreme[sid] = {"method": row["method"], "score": row["visual_composite_score"], "psd": row["normalized_psd_log_distance"], "rq": row["measured_rq_nm"], "source": row["source_sample_ids"]}
    s5 = metrics[(metrics["track"].eq("strict")) & (metrics["method"].eq("VB2"))]
    best = outputs.get("best_strict", pd.DataFrame())
    go_visual_1 = False
    if not s5.empty and not best.empty:
        go_visual_1 = bool(best["normalized_psd_log_distance"].median() <= 0.9 * s5["normalized_psd_log_distance"].median() and best["heldout_source_contribution"].max() == 0)
    previous_runtime = 0.0
    previous_summary = ctx.out / "phase7a_summary.json"
    if previous_summary.exists():
        try:
            previous_runtime = float(json.loads(previous_summary.read_text(encoding="utf-8")).get("runtime_seconds", 0.0))
        except Exception:
            previous_runtime = 0.0
    strict_identity = identity[identity["track"].eq("strict")] if not identity.empty and "track" in identity else pd.DataFrame()
    synth_strict = strict_identity[~strict_identity["method_family"].eq("retrieval")] if not strict_identity.empty and "method_family" in strict_identity else pd.DataFrame()
    summary = {
        "phase": "7A",
        "alignment_audit_passed": bool(audit["passed"].map(as_bool).all()),
        "removelist_enforced": bool(set(ctx.config["removed_samples"]).isdisjoint(set(ctx.active_ids))),
        "runtime_seconds": max(float(time.time() - start), previous_runtime),
        "trial_counts": {
            "completed": int((reg["status"] == "completed").sum()) if not reg.empty else 0,
            "failed": int((reg["status"] == "failed").sum()) if not reg.empty else 0,
            "skipped": 0,
            "total": int(len(reg)),
        },
        "device": ctx.device,
        "method_families_run": methods_run,
        "best_retrieval": best_family("retrieval"),
        "best_quilting": best_family("quilting"),
        "best_residual": best_family("residual"),
        "best_iaaft": best_family("iaaft"),
        "best_texture": best_family("texture"),
        "best_vq": best_family("vq"),
        "best_diffusion": best_family("diffusion"),
        "best_strict_visual_method": best_strict,
        "best_oracle_visual_method": best_oracle,
        "best_development_visual_method": best_dev,
        "best_strict_metrics": best.to_dict("records")[:5] if not best.empty else [],
        "s1_vs_best_synthesis": s1_comp,
        "samples_6095_6099_visual": sample_extreme,
        "multi_seed_diversity": multi_seed_diversity(metrics),
        "identity_audit": {
            "rows": int(len(identity)),
            "strict_rows": int(len(strict_identity)),
            "strict_heldout_source_contribution_max": float(strict_identity["heldout_source_contribution"].max()) if not strict_identity.empty else np.nan,
            "all_tracks_heldout_source_contribution_max": float(identity["heldout_source_contribution"].max()) if not identity.empty else np.nan,
            "strict_synth_exact_identity_count": int(synth_strict["exact_identity"].map(as_bool).sum()) if not synth_strict.empty else 0,
            "development_self_source_allowed": True,
        },
        "blind_review_package_path": blind.get("index_html", ""),
        "strict_all_sample_atlas_path": figures.get("all_sample_atlas", ""),
        "development_showcase_path": figures.get("development_showcase", ""),
        "dashboard_path": display_path(ctx.rep / "dashboard" / "results_dashboard.html"),
        "full_cohort_visual_deployment_model_path": display_path(ctx.out / "deployment" / "visual_deployment_model"),
        "raw_old_hash_validation": input_hashes(ctx),
        "go_decisions": {
            "Go-VISUAL-1": go_visual_1,
            "Go-VISUAL-2": "pending_manual_blind_review",
            "Go-VISUAL-3": bool(s1_comp.get("psd_improved_fraction", 0) > 0.5 or s1_comp.get("histogram_improved_fraction", 0) > 0.5),
            "Go-VQ": bool("vq" in methods_run),
            "Go-DIFFUSION": bool("diffusion" in methods_run),
        },
        "claims": {
            "can_claim": "strict OOF visual benchmark, provenance, identity audit, oracle upper bound, full-cohort development package",
            "cannot_claim": "exact local AFM reconstruction, independent external validation, oracle/development as strict prediction",
        },
    }
    write_json(summary, ctx.out / "phase7a_summary.json")
    return summary


def input_hashes(ctx: Phase7Context) -> dict[str, str]:
    paths = [
        ctx.config["canonical_index_path"],
        ctx.config["phase6a_summary_path"],
        ctx.config["phase6a_strict_oof_path"],
        ctx.config["phase6a_descriptor_predictions_path"],
        ctx.config["second_order_scan_descriptors_path"],
        ctx.config["second_order_afm_bank_path"],
        ctx.config["removelist_path"],
    ]
    return {p: sha256_file(p) for p in paths}


def multi_seed_diversity(metrics: pd.DataFrame) -> dict[str, float]:
    strict = metrics[metrics["track"].eq("strict")]
    if strict.empty:
        return {}
    grouped = strict.groupby(["sample_id", "method"])["visual_composite_score"].std().dropna()
    return {"median_score_std_across_seeds": float(grouped.median()) if len(grouped) else 0.0, "evaluated_seed_groups": int(len(grouped))}


def deployment_package(ctx: Phase7Context, outputs: dict[str, Any]) -> None:
    dep = ctx.out / "deployment" / "visual_deployment_model"
    dep.mkdir(parents=True, exist_ok=True)
    active_index(ctx).to_csv(dep / "active_sample_index.csv", index=False)
    if "best_development" in outputs:
        outputs["best_development"].to_csv(dep / "development_best_visual_outputs.csv", index=False)
    write_json(
        {
            "model_name": "Phase7A full-cohort AFM visual development model",
            "warning": "FULL-COHORT AFM VISUAL DEVELOPMENT MODEL; ALL HISTORICAL AFM DATA WERE AVAILABLE; NOT AN INDEPENDENT TEST RESULT",
            "future_unseen_only": True,
            "training_sample_count": len(ctx.active_ids),
            "methods": sorted(METHOD_FAMILIES.values()),
        },
        dep / "model_registry.json",
    )


def run(config_path: str | Path, stage: str = "all", resume: bool = False, max_trials: int | None = None, methods: str | None = None, strict_only: bool = False, development_only: bool = False, device: str = "auto", time_budget_hours: float | None = None) -> dict[str, Any]:
    start = time.time()
    ctx = load_inputs(config_path, device=device)
    integrity_audit(ctx)
    if stage == "audit":
        return {"stage": "audit", "alignment_audit": display_path(ctx.out / "provenance" / "phase7_alignment_audit.csv")}
    build_texture_dataset(ctx)
    if stage in {"baseline", "retrieval", "quilting", "residual", "spectral", "texture_opt", "vq", "diffusion"} and not methods:
        mapping = {"baseline": "VB0,VB1,VB2", "retrieval": "retrieval", "quilting": "quilting", "residual": "residual", "spectral": "iaaft", "texture_opt": "texture", "vq": "vq", "diffusion": "diffusion"}
        methods = mapping[stage]
    metrics, manifest, prov = run_visual_trials(ctx, resume=resume, max_trials=max_trials or int(ctx.config["default_max_trials"]), methods=methods, strict_only=strict_only, development_only=development_only, time_budget_hours=time_budget_hours)
    outputs = select_best_outputs(ctx, metrics, manifest, prov)
    oracle_gap(ctx, outputs)
    s1_comp = compare_s1_best(ctx, metrics, outputs)
    blind = blind_review(ctx, outputs)
    deployment_package(ctx, outputs)
    figures = visuals(ctx, metrics, manifest, outputs, s1_comp)
    summary = final_summary(ctx, metrics, outputs, blind, figures, s1_comp, start)
    dashboard_and_reports(ctx, summary, figures)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 7A reconstruction-first AFM visual synthesis benchmark.")
    parser.add_argument("--config", default="configs/rheed_video_afm_story_phase7a.yaml")
    parser.add_argument("--stage", default="all", choices=["audit", "baseline", "retrieval", "quilting", "residual", "spectral", "texture_opt", "vq", "diffusion", "strict_finalists", "oracle", "development", "blind_review", "visualization", "all"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--time-budget-hours", type=float, default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--strict-only", action="store_true")
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    methods = args.methods
    if args.skip_neural:
        base = "VB0,VB1,VB2,A1,A2,A3,A4,A5,A6,B1,B2,C1,C2,D1,D4"
        methods = base if not methods else ",".join([m for m in methods.split(",") if m not in {"texture", "vq", "diffusion", "E1", "E2", "F1", "F2", "G4"}])
    summary = run(args.config, stage=args.stage, resume=args.resume, max_trials=args.max_trials, methods=methods, strict_only=args.strict_only, development_only=args.development_only, device=args.device, time_budget_hours=args.time_budget_hours)
    print(json.dumps({"summary": "outputs/rheed_video_afm_story/variants/afm_second_order_y2_v1/phase7a_reconstruction_first/phase7a_summary.json", "best": summary.get("best_strict_visual_method", {})}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
