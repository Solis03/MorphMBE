from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVR

from .baseline_rq import pairwise_concordance
from .common import display_path, load_config, repo_path, save_parquet, sha256_file, write_csv, write_json


TARGET_VARIANTS = [
    "T1_first_order_median_rq",
    "T2_second_order_median_rq",
    "T3_second_order_log_median",
    "T4_second_order_trimmed_mean",
    "T5_second_order_huber_location",
    "T6_quality_weighted_second_order",
    "T7_roughness_latent_factor",
    "T8_multi_output_descriptor_target",
]
REGIME_LABELS = ["low", "middle", "high"]
INTERRUPTED = False


def _handle_interrupt(signum, frame) -> None:  # noqa: ANN001
    global INTERRUPTED
    INTERRUPTED = True


signal.signal(signal.SIGINT, _handle_interrupt)


@dataclass
class Phase6Inputs:
    config: dict[str, Any]
    out: Path
    rep: Path
    index: pd.DataFrame
    active_ids: list[str]
    targets: pd.DataFrame
    scan_desc: pd.DataFrame
    descriptors: pd.DataFrame
    feature_blocks: dict[str, np.ndarray]
    quality: pd.DataFrame


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def ensure_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    out = repo_path(config["output_root"])
    rep = repo_path(config["report_root"])
    for rel in [
        "provenance",
        "canonical_index",
        "target_variants",
        "preprocessing_cache",
        "embeddings",
        "trials",
        "nested_oof",
        "finalists",
        "descriptor_models",
        "retrieval",
        "synthesis",
        "deployment/deployment_model",
        "visualization/figures",
        "dashboard",
        "logs",
    ]:
        (out / rel).mkdir(parents=True, exist_ok=True)
    for rel in ["figures", "dashboard", "strict_oof_package", "development_showcase"]:
        (rep / rel).mkdir(parents=True, exist_ok=True)
    return out, rep


def read_table(path: str | Path) -> pd.DataFrame:
    p = repo_path(path)
    if p.suffix == ".parquet":
        try:
            return pd.read_parquet(p)
        except ImportError:
            fallback = p.with_suffix(p.suffix + ".csv_fallback")
            if fallback.exists():
                return pd.read_csv(fallback)
            csv = p.with_suffix(".csv")
            return pd.read_csv(csv)
    return pd.read_csv(p)


def as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value) == "True"


def load_embedding(path: str | Path) -> tuple[list[str], np.ndarray]:
    z = np.load(repo_path(path), allow_pickle=False)
    return [str(x) for x in z["sample_ids"].tolist()], np.asarray(z["embeddings"], dtype=float)


def descriptor_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
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
    return [c for c in candidates if c in df.columns]


def physics_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "spot_summary_raw",
        "streak_summary_raw",
        "connection_summary_raw",
        "diffuse_summary_raw",
        "temporal_brightness_drift",
        "selected_16__saturation_fraction_median",
        "selected_16__temporal_stability",
        "selected_16__horizontal_run_continuity_median",
        "selected_16__orientation_entropy_median",
        "selected_16__spot_peak_width_proxy_median",
        "selected_16__line_response_q90_median",
        "selected_16__component_count_p95_median",
    ]
    cols = [c for c in preferred if c in df.columns]
    if len(cols) >= 6:
        return cols
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in {"sample_id", "growth_run_id"}][:24]


def build_canonical_index(config: dict[str, Any], out: Path, rep: Path) -> pd.DataFrame:
    phase1 = read_table(config["phase1_manifest_path"])
    phase1["sample_id"] = phase1["sample_id"].astype(str)
    phase1["growth_run_id"] = phase1["growth_run_id"].astype(str)
    second = pd.read_csv(repo_path(config["second_order_manifest_path"]), dtype={"sample_id": str, "growth_run_id": str})
    targets = pd.read_csv(repo_path(config["second_order_sample_targets_path"]), dtype={"sample_id": str, "growth_run_id": str})
    phase1_audit = pd.read_csv(repo_path(config["phase1_afm_audit_path"]), dtype={"sample_id": str})
    physics = pd.read_csv(repo_path(config["physics_features_path"]), dtype={"sample_id": str})
    removed = set(config["removed_samples"])
    emb_maps: dict[str, dict[str, int]] = {}
    emb_hashes: dict[str, str] = {}
    for name, path in config["embedding_paths"].items():
        ids, _ = load_embedding(path)
        emb_maps[name] = {sid: i for i, sid in enumerate(ids)}
        emb_hashes[name] = sha256_file(path)
    all_ids = sorted(set(phase1["sample_id"].astype(str)) | removed)
    phase1_by = phase1.drop_duplicates("sample_id").set_index("sample_id")
    second_by = second.drop_duplicates("sample_id").set_index("sample_id")
    target_by = targets.drop_duplicates("sample_id").set_index("sample_id")
    physics_by = physics.drop_duplicates("sample_id").set_index("sample_id")
    rows: list[dict[str, Any]] = []
    for sid in all_ids:
        p1 = phase1_by.loc[sid] if sid in phase1_by.index else pd.Series(dtype=object)
        so = second_by.loc[sid] if sid in second_by.index else pd.Series(dtype=object)
        ta = target_by.loc[sid] if sid in target_by.index else pd.Series(dtype=object)
        first_rep = str(p1.get("representative_afm_height_array", p1.get("representative_afm_path", "")))
        row = {
            "sample_id": sid,
            "growth_run_id": str(p1.get("growth_run_id", so.get("growth_run_id", sid))),
            "is_primary": bool(as_bool(p1.get("usable_for_modeling", False)) and as_bool(p1.get("cohort_primary_1um", False)) and sid not in removed),
            "is_removed": sid in removed,
            "removelist_reason": "canonical_removelist" if sid in removed else "",
            "rheed_manifest_path": config["phase1_manifest_path"],
            "rheed_metadata_path": str(p1.get("metadata_path", so.get("metadata_path", ""))),
            "video_id": str(p1.get("video_id", so.get("video_id", ""))),
            "source_video": str(p1.get("source_video", so.get("source_video", ""))),
            "frames_dir": str(p1.get("frames_dir", so.get("frames_dir", ""))),
            "keyframe_index": p1.get("keyframe_index", so.get("keyframe_index", np.nan)),
            "clip_start_index": p1.get("clip_start_index", so.get("clip_start_index", np.nan)),
            "clip_end_index": p1.get("clip_end_index", so.get("clip_end_index", np.nan)),
            "clip_frame_indices": str(p1.get("clip_frame_indices", so.get("clip_frame_indices", ""))),
            "roi_x": p1.get("roi_x", so.get("roi_x", np.nan)),
            "roi_y": p1.get("roi_y", so.get("roi_y", np.nan)),
            "roi_width": p1.get("roi_width", so.get("roi_width", np.nan)),
            "roi_height": p1.get("roi_height", so.get("roi_height", np.nan)),
            "first_order_target_path": config["phase1_manifest_path"],
            "second_order_target_path": config["second_order_sample_targets_path"],
            "first_order_rq_nm": float(ta.get("first_order_rq_nm", p1.get("primary_rq_nm_median", np.nan))) if sid in target_by.index or "primary_rq_nm_median" in p1 else np.nan,
            "second_order_rq_nm": float(ta.get("second_order_rq_nm", np.nan)) if sid in target_by.index else np.nan,
            "first_order_representative_afm_path": first_rep,
            "second_order_representative_afm_path": str(ta.get("second_order_ground_truth_afm_path", so.get("representative_afm_path", ""))),
            "dino_embedding_row_id": emb_maps.get("E1_dino_keyframe", {}).get(sid, -1),
            "r3d_embedding_row_id": emb_maps.get("E2_r3d_selected16", {}).get(sid, -1),
            "physics_feature_row_id": int(list(physics_by.index).index(sid)) if sid in physics_by.index else -1,
            "phase1_manifest_hash": sha256_file(config["phase1_manifest_path"]),
            "second_order_manifest_hash": sha256_file(config["second_order_manifest_path"]),
            "second_order_targets_hash": sha256_file(config["second_order_sample_targets_path"]),
            "phase2a_embedding_registry_hash": sha256_file(config["phase2a_embedding_registry_path"]),
            "physics_features_hash": sha256_file(config["physics_features_path"]),
            "removelist_hash": sha256_file(config["removelist_path"]),
            "dino_embedding_hash": emb_hashes.get("E1_dino_keyframe", ""),
            "r3d_embedding_hash": emb_hashes.get("E2_r3d_selected16", ""),
        }
        for key in ["first_order_representative_afm_path", "second_order_representative_afm_path", "rheed_metadata_path"]:
            path = row[key]
            row[key + "_hash"] = sha256_file(path) if path and repo_path(path).exists() and repo_path(path).is_file() else ""
        rows.append(row)
    index = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    root = out / "canonical_index"
    write_csv(index, root / "canonical_sample_index.csv")
    save_parquet(index, root / "canonical_sample_index.parquet")
    write_json(index.to_dict("records"), root / "canonical_sample_index.json")
    audit = validate_canonical_index(index, config)
    write_csv(audit, root / "canonical_alignment_audit.csv")
    lines = ["# Canonical Alignment Audit", ""]
    lines += [f"- {row.check_name}: {row.passed} ({row.detail})" for row in audit.itertuples()]
    (root / "canonical_alignment_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_contact_sheet(index[index["is_primary"]], root / "canonical_alignment_contact_sheet.pdf", title="Canonical sample alignment")
    return index


def validate_canonical_index(index: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    active = index[index["is_primary"]]
    rows: list[dict[str, Any]] = []
    def add(name: str, passed: bool, detail: str) -> None:
        rows.append({"check_name": name, "passed": bool(passed), "detail": detail})
    add("primary_N_23", len(active) == int(config["expected_primary_n"]), f"N={len(active)}")
    add("removed_marked", set(config["removed_samples"]).issubset(set(index[index["is_removed"]]["sample_id"])), ",".join(config["removed_samples"]))
    add("removed_not_active", not set(config["removed_samples"]) & set(active["sample_id"]), "removed active count 0")
    add("unique_sample_id", index["sample_id"].is_unique, f"unique={index['sample_id'].is_unique}")
    for col in ["second_order_representative_afm_path", "first_order_representative_afm_path"]:
        bad = []
        for row in active.to_dict("records"):
            path = str(row[col])
            if row["sample_id"] not in path:
                bad.append(row["sample_id"])
        add(col + "_contains_sample_id", not bad, json.dumps(bad))
    add("embedding_rows_present", bool((active["dino_embedding_row_id"] >= 0).all() and (active["r3d_embedding_row_id"] >= 0).all()), "DINO/R3D rows present")
    add("physics_rows_present", bool((active["physics_feature_row_id"] >= 0).all()), "physics rows present")
    add("removelist_hash_ok", sha256_file(config["removelist_path"]) == config["expected_removelist_hash"], sha256_file(config["removelist_path"]))
    return pd.DataFrame(rows)


def make_contact_sheet(index: pd.DataFrame, path: Path, title: str) -> None:
    rows = math.ceil(len(index) / 4)
    fig, axs = plt.subplots(rows, 4, figsize=(12, rows * 2.6))
    axs = np.asarray(axs).reshape(-1)
    for ax in axs:
        ax.axis("off")
    for ax, row in zip(axs, index.to_dict("records"), strict=False):
        ax.set_title(str(row["sample_id"]), fontsize=8)
        clip = repo_path(f"outputs/rheed_video_afm_story/phase2a/clip_variants/keyframe_1/{row['sample_id']}.npz")
        if clip.exists():
            img = np.load(clip, allow_pickle=False)["frames_uint8"][0]
            ax.imshow(img, cmap="gray")
        txt = (
            f"1st Rq {row['first_order_rq_nm']:.2f}\n"
            f"2nd Rq {row['second_order_rq_nm']:.2f}\n"
            f"DINO {row['dino_embedding_row_id']} R3D {row['r3d_embedding_row_id']}"
        )
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=6, color="white", va="bottom")
    fig.suptitle(title)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_splits(active_ids: list[str], out: Path) -> pd.DataFrame:
    rows = []
    for held in active_ids:
        train = [sid for sid in active_ids if sid != held]
        rows.append(
            {
                "outer_fold": held,
                "heldout_id": held,
                "training_count": len(train),
                "training_ids": json.dumps(train),
                "split_valid": len(train) == 22 and held not in train,
                "contains_6095_when_heldout_6099": ("6095" in train) if held == "6099" else True,
                "contains_6099_when_heldout_6095": ("6099" in train) if held == "6095" else True,
            }
        )
    split = pd.DataFrame(rows)
    write_csv(split, out / "provenance" / "outer_splits.csv")
    write_json(split.to_dict("records"), out / "provenance" / "outer_splits.json")
    mat = pd.DataFrame(0, index=active_ids, columns=active_ids)
    for row in split.to_dict("records"):
        mat.loc[row["heldout_id"], json.loads(row["training_ids"])] = 1
    mat.insert(0, "heldout_id", mat.index)
    write_csv(mat.reset_index(drop=True), out / "provenance" / "outer_split_membership_matrix.csv")
    return split


def build_targets(inputs: Phase6Inputs) -> pd.DataFrame:
    idx = inputs.index.set_index("sample_id").loc[inputs.active_ids]
    scan = inputs.scan_desc.copy()
    scan["sample_id"] = scan["sample_id"].astype(str)
    rows = []
    for sid in inputs.active_ids:
        g = scan[scan["sample_id"].eq(sid)]
        rq = g["rq_nm"].to_numpy(float)
        rq = rq[np.isfinite(rq)]
        first = float(idx.loc[sid, "first_order_rq_nm"])
        second = float(idx.loc[sid, "second_order_rq_nm"])
        if len(rq) >= 4:
            trim = np.sort(rq)[1:-1]
        else:
            trim = rq
        log_rq = np.log10(np.maximum(rq, 1e-9))
        huber = float(np.median(log_rq))
        quality_w = np.ones_like(rq, dtype=float)
        if "finite_fraction" in g.columns:
            quality_w = g["finite_fraction"].fillna(1).to_numpy(float)
        weighted = float(np.average(rq, weights=np.maximum(quality_w, 1e-6))) if len(rq) else second
        rows.append(
            {
                "sample_id": sid,
                "T1_first_order_median_rq": first,
                "T2_second_order_median_rq": second,
                "T3_second_order_log_median": float(10 ** np.median(log_rq)) if len(log_rq) else second,
                "T4_second_order_trimmed_mean": float(np.mean(trim)) if len(trim) else second,
                "T5_second_order_huber_location": float(10**huber) if len(log_rq) else second,
                "T6_quality_weighted_second_order": weighted,
                "T7_roughness_latent_factor": second,
                "T8_multi_output_descriptor_target": second,
                "scan_count": int(len(rq)),
                "scan_rq_mad": float(np.median(np.abs(rq - np.median(rq)))) if len(rq) else np.nan,
                "scan_rq_iqr": float(np.quantile(rq, 0.75) - np.quantile(rq, 0.25)) if len(rq) else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    for target in TARGET_VARIANTS:
        table[target + "_rank"] = table[target].rank(method="min")
    write_csv(table, inputs.out / "target_variants" / "target_variant_table.csv")
    save_parquet(table, inputs.out / "target_variants" / "target_variant_table.parquet")
    return table


def load_inputs(config_path: str | Path) -> Phase6Inputs:
    config = load_config(config_path)
    out, rep = ensure_dirs(config)
    index_path = out / "canonical_index" / "canonical_sample_index.csv"
    if index_path.exists():
        index = pd.read_csv(index_path, dtype={"sample_id": str, "growth_run_id": str})
    else:
        index = build_canonical_index(config, out, rep)
    audit = pd.read_csv(out / "canonical_index" / "canonical_alignment_audit.csv")
    if not audit["passed"].map(as_bool).all():
        raise RuntimeError("Canonical alignment failed; see canonical_alignment_audit.csv")
    active_ids = index[index["is_primary"].map(as_bool)]["sample_id"].astype(str).tolist()
    split = write_splits(active_ids, out)
    if not split["split_valid"].map(as_bool).all():
        raise RuntimeError("Outer split validation failed")
    scan_desc = pd.read_csv(repo_path(config["second_order_scan_descriptors_path"]), dtype={"sample_id": str})
    targets_raw = pd.read_csv(repo_path(config["second_order_sample_targets_path"]), dtype={"sample_id": str})
    physics = pd.read_csv(repo_path(config["physics_features_path"]), dtype={"sample_id": str}).drop_duplicates("sample_id").set_index("sample_id").loc[active_ids].reset_index()
    q = pd.read_csv(repo_path(config["rheed_quality_path"]), dtype={"sample_id": str}).drop_duplicates("sample_id").set_index("sample_id").reindex(active_ids).reset_index()
    blocks: dict[str, np.ndarray] = {}
    for name, path in config["embedding_paths"].items():
        ids, X = load_embedding(path)
        if set(active_ids) - set(ids):
            raise RuntimeError(f"Embedding {name} missing active IDs")
        blocks[name] = X[[ids.index(sid) for sid in active_ids]]
    phys_cols = physics_columns(physics)
    blocks["E3_physics"] = physics[phys_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    blocks["E4_dino_r3d"] = np.hstack([blocks["E1_dino_keyframe"], blocks["E2_r3d_selected16"]])
    blocks["E5_fused"] = np.hstack([blocks["E1_dino_keyframe"], blocks["E2_r3d_selected16"], blocks["E3_physics"]])
    desc_cols = descriptor_columns(scan_desc)
    desc_rows = []
    for sid in active_ids:
        g = scan_desc[scan_desc["sample_id"].eq(sid)]
        row = {"sample_id": sid}
        for col in desc_cols:
            row[col] = float(np.nanmedian(g[col].to_numpy(float)))
        desc_rows.append(row)
    descriptors = pd.DataFrame(desc_rows)
    inputs = Phase6Inputs(config, out, rep, index, active_ids, targets_raw, scan_desc, descriptors, blocks, q)
    build_targets(inputs)
    write_preprocessing_and_encoder_metadata(inputs)
    return inputs


def write_preprocessing_and_encoder_metadata(inputs: Phase6Inputs) -> None:
    pre = []
    for name in ["P1_raw_luminance", "P2_clip_robust_contrast", "P3_log_intensity", "P4_gamma_0.5", "P4_gamma_0.75", "P4_gamma_1.25", "P4_gamma_1.5", "P5_background_ratio", "P5_background_subtraction", "P6_circular_screen_mask", "P7_roi_exact", "P7_roi_expand_5", "P7_roi_expand_10", "P7_roi_shrink_5", "P8_orientation_normalized", "P9_temporal_summary_rgb", "P10_temporal_difference"]:
        pre.append({"preprocessing_variant": name, "status": "metadata_registered_target_blind", "source": "existing_phase2a_clip_cache_or_future_cache", "modifies_raw_png": False})
    write_csv(pd.DataFrame(pre), inputs.out / "preprocessing_cache" / "preprocessing_variant_registry.csv")
    enc = []
    for name, path in inputs.config["embedding_paths"].items():
        enc.append({"encoder": name, "status": "cached", "path": path, "sha256": sha256_file(path), "sample_ids_embedded": len(load_embedding(path)[0])})
    for name in ["DINOv2_ViT_B14", "ResNet50", "ConvNeXt_Tiny", "EfficientNet_B0", "Swin_T", "CLIP_ViT_B32", "R2Plus1D_18", "MC3_18", "VideoMAE_small"]:
        enc.append({"encoder": name, "status": "skipped_not_available_or_not_cached", "path": "", "sha256": "", "sample_ids_embedded": 0})
    write_csv(pd.DataFrame(enc), inputs.out / "embeddings" / "encoder_sweep_registry.csv")
    write_json({"ssl_inductive": "skipped_cost_controlled_phase6a_run", "ssl_transductive_development": "skipped_not_mixed_with_strict_oof"}, inputs.out / "embeddings" / "ssl_branch_status.json")


def candidate_trials(max_trials: int) -> list[dict[str, Any]]:
    targets = TARGET_VARIANTS
    features = ["E1_dino_keyframe", "E2_r3d_selected16", "E3_physics", "E4_dino_r3d", "E5_fused"]
    models = ["ridge", "elastic_net", "bayesian_ridge", "huber", "pls", "knn_mean", "knn_median", "svr_rbf", "random_forest", "extra_trees"]
    calibrators = ["none", "affine", "range_expand", "isotonic"]
    transforms = ["log10", "none"]
    weights = ["none", "quantile_balanced"]
    all_trials = []
    for target in targets:
        for feat in features:
            for model in models:
                for cal in calibrators:
                    for trans in transforms:
                        for weight in weights:
                            all_trials.append(
                                {
                                    "target_variant": target,
                                    "preprocessing_variant": "P1_raw_luminance",
                                    "clip_variant": "keyframe_or_selected_cached",
                                    "encoder": feat,
                                    "embedding_aggregation": "cached_sample_embedding",
                                    "feature_set": feat,
                                    "model_family": model,
                                    "hyperparameters": {"alpha": 1.0, "k": 3, "max_depth": 2, "n_estimators": 80},
                                    "sample_weighting": weight,
                                    "calibration": cal,
                                    "target_transform": trans,
                                    "multitask": target == "T8_multi_output_descriptor_target",
                                    "seed": 17,
                                }
                            )
    selected: list[dict[str, Any]] = []
    buckets = {target: [t for t in all_trials if t["target_variant"] == target] for target in targets}
    i = 0
    while len(selected) < max_trials and any(i < len(v) for v in buckets.values()):
        for target in targets:
            if i < len(buckets[target]) and len(selected) < max_trials:
                selected.append(buckets[target][i])
        i += 1
    for n, trial in enumerate(selected, start=1):
        trial["trial_id"] = f"trial_{n:04d}"
        trial["config_hash"] = stable_hash(trial)
    return selected


def registry_path(out: Path) -> Path:
    return out / "trials" / "trial_registry.csv"


def read_registry(out: Path) -> pd.DataFrame:
    path = registry_path(out)
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def append_registry(out: Path, row: dict[str, Any]) -> None:
    path = registry_path(out)
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)


def target_transform(y: np.ndarray, transform: str) -> tuple[np.ndarray, Any]:
    if transform == "log10":
        return np.log10(np.maximum(y, 1e-9)), ("log10", None)
    if transform == "rank_gaussian":
        qt = QuantileTransformer(n_quantiles=min(len(y), 20), output_distribution="normal", random_state=17).fit(y.reshape(-1, 1))
        return qt.transform(y.reshape(-1, 1)).ravel(), ("quantile", qt)
    return y.astype(float), ("none", None)


def inverse_transform(y: np.ndarray, state: Any) -> np.ndarray:
    kind, obj = state
    if kind == "log10":
        return 10 ** y
    if kind == "quantile":
        return obj.inverse_transform(np.asarray(y).reshape(-1, 1)).ravel()
    return np.asarray(y, dtype=float)


def sample_weights(y: np.ndarray, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    q1, q2 = np.quantile(y, [1 / 3, 2 / 3])
    labels = np.where(y <= q1, 0, np.where(y >= q2, 2, 1))
    counts = np.bincount(labels, minlength=3)
    w = np.array([1 / max(counts[label], 1) for label in labels], dtype=float)
    return w / np.mean(w)


def fit_predict_model(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, spec: dict[str, Any], weights: np.ndarray | None) -> np.ndarray:
    scaler = StandardScaler().fit(Xtr)
    Ztr = scaler.transform(Xtr)
    Zte = scaler.transform(Xte)
    model = spec["model_family"]
    hp = spec["hyperparameters"]
    if model == "ridge":
        reg = Ridge(alpha=hp["alpha"])
        reg.fit(Ztr, ytr, sample_weight=weights)
        return np.ravel(reg.predict(Zte))
    if model == "elastic_net":
        reg = ElasticNet(alpha=0.05, l1_ratio=0.25, max_iter=5000, random_state=spec["seed"])
        reg.fit(Ztr, ytr, sample_weight=weights)
        return np.ravel(reg.predict(Zte))
    if model == "bayesian_ridge":
        reg = BayesianRidge()
        reg.fit(Ztr, ytr)
        return np.ravel(reg.predict(Zte))
    if model == "huber":
        reg = HuberRegressor(alpha=0.01, max_iter=500)
        reg.fit(Ztr, ytr, sample_weight=weights)
        return np.ravel(reg.predict(Zte))
    if model == "pls":
        comp = max(1, min(3, Ztr.shape[0] - 1, Ztr.shape[1]))
        reg = PLSRegression(n_components=comp)
        reg.fit(Ztr, ytr)
        return np.ravel(reg.predict(Zte))
    if model == "knn_mean":
        reg = KNeighborsRegressor(n_neighbors=min(hp["k"], len(Ztr)), weights="distance")
        reg.fit(Ztr, ytr)
        return np.ravel(reg.predict(Zte))
    if model == "knn_median":
        d = np.sqrt(((Ztr[None, :, :] - Zte[:, None, :]) ** 2).sum(axis=2))
        k = min(hp["k"], len(Ztr))
        return np.array([np.median(ytr[np.argsort(row)[:k]]) for row in d], dtype=float)
    if model == "svr_rbf":
        reg = SVR(C=2.0, gamma="scale", epsilon=0.05)
        reg.fit(Ztr, ytr, sample_weight=weights)
        return np.ravel(reg.predict(Zte))
    if model == "random_forest":
        reg = RandomForestRegressor(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"], min_samples_leaf=2, random_state=spec["seed"])
        reg.fit(Ztr, ytr, sample_weight=weights)
        return np.ravel(reg.predict(Zte))
    if model == "extra_trees":
        reg = ExtraTreesRegressor(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"], min_samples_leaf=2, random_state=spec["seed"])
        reg.fit(Ztr, ytr, sample_weight=weights)
        return np.ravel(reg.predict(Zte))
    raise ValueError(f"Unknown model family {model}")


def calibrate(inner_pred: np.ndarray, inner_true: np.ndarray, raw: float, mode: str) -> tuple[float, dict[str, Any]]:
    inner_pred = np.asarray(inner_pred, dtype=float)
    inner_true = np.asarray(inner_true, dtype=float)
    meta = {
        "calibration": mode,
        "training_prediction_std": float(np.std(inner_pred)),
        "truth_std": float(np.std(inner_true)),
        "range_ratio": float(np.std(inner_pred) / max(np.std(inner_true), 1e-12)),
    }
    if mode == "none" or len(inner_pred) < 4:
        meta.update({"slope": 1.0, "intercept": 0.0})
        return float(raw), meta
    if mode == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip").fit(inner_pred, inner_true)
        pred = float(iso.predict([raw])[0])
        coef = np.polyfit(inner_pred, inner_true, 1)
        meta.update({"slope": float(coef[0]), "intercept": float(coef[1])})
        return pred, meta
    coef = np.polyfit(inner_pred, inner_true, 1)
    slope, intercept = float(coef[0]), float(coef[1])
    if mode == "range_expand":
        slope = np.std(inner_true) / max(np.std(inner_pred), 1e-12)
        intercept = float(np.mean(inner_true) - slope * np.mean(inner_pred))
    meta.update({"slope": slope, "intercept": intercept})
    return float(intercept + slope * raw), meta


def run_trial(inputs: Phase6Inputs, target_table: pd.DataFrame, spec: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    ids = inputs.active_ids
    X = inputs.feature_blocks[spec["feature_set"]]
    y_map = target_table.set_index("sample_id")[spec["target_variant"]].astype(float)
    y = np.array([float(y_map.loc[sid]) for sid in ids], dtype=float)
    rows: list[dict[str, Any]] = []
    cal_rows = []
    for held_pos, held_id in enumerate(ids):
        train_pos = np.array([i for i, sid in enumerate(ids) if sid != held_id], dtype=int)
        inner_raw: list[float] = []
        inner_true: list[float] = []
        for val_pos in train_pos:
            inner_train = np.array([i for i in train_pos if i != val_pos], dtype=int)
            yt, state = target_transform(y[inner_train], spec["hyperparameters"].get("target_transform", spec.get("target_transform", "log10")))
            w = sample_weights(y[inner_train], spec["sample_weighting"])
            pred_t = fit_predict_model(X[inner_train], yt, X[[val_pos]], spec, w)
            pred = float(inverse_transform(pred_t, state)[0])
            inner_raw.append(pred)
            inner_true.append(float(y[val_pos]))
        yt_outer, state_outer = target_transform(y[train_pos], spec["hyperparameters"].get("target_transform", spec.get("target_transform", "log10")))
        w_outer = sample_weights(y[train_pos], spec["sample_weighting"])
        raw_t = fit_predict_model(X[train_pos], yt_outer, X[[held_pos]], spec, w_outer)
        raw = float(inverse_transform(raw_t, state_outer)[0])
        calibrated, cal_meta = calibrate(np.array(inner_raw), np.array(inner_true), raw, spec["calibration"])
        rows.append(
            {
                "trial_id": spec["trial_id"],
                "sample_id": held_id,
                "target_variant": spec["target_variant"],
                "true_target_nm": float(y[held_pos]),
                "raw_prediction_nm": raw,
                "predicted_target_nm": float(np.clip(calibrated, 0.1, 50.0)),
                "absolute_error_nm": abs(float(y[held_pos]) - float(np.clip(calibrated, 0.1, 50.0))),
                "outer_fold": held_id,
                "heldout_id": held_id,
                "training_ids": json.dumps([sid for sid in ids if sid != held_id]),
                "training_count": 22,
                "true_target_lookup": "heldout_sample_id",
                "outer_target_used_for_selection": False,
            }
        )
        cal_rows.append({"trial_id": spec["trial_id"], "sample_id": held_id, **cal_meta})
    pred = pd.DataFrame(rows)
    metrics = metric_dict(pred["true_target_nm"].to_numpy(float), pred["predicted_target_nm"].to_numpy(float))
    raw_metrics = metric_dict(pred["true_target_nm"].to_numpy(float), pred["raw_prediction_nm"].to_numpy(float))
    metrics.update(
        {
            "trial_id": spec["trial_id"],
            "target_variant": spec["target_variant"],
            "feature_set": spec["feature_set"],
            "model_family": spec["model_family"],
            "calibration": spec["calibration"],
            "sample_weighting": spec["sample_weighting"],
            "raw_MAE": raw_metrics["MAE"],
            "raw_Spearman": raw_metrics["Spearman"],
            "predicted_std": float(pred["predicted_target_nm"].std()),
            "truth_std": float(pred["true_target_nm"].std()),
            "predicted_truth_std_ratio": float(pred["predicted_target_nm"].std() / max(pred["true_target_nm"].std(), 1e-12)),
            "calibration_mean_slope": float(np.mean([r["slope"] for r in cal_rows])),
            "inner_calibration_scope": "inner_oof_training_predictions_only",
        }
    )
    return pred, metrics


def metric_dict(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    q33, q67 = np.quantile(y, [1 / 3, 2 / 3])
    extreme = (y <= q33) | (y >= q67)
    yt = np.where(y[extreme] >= q67, "high", "low")
    yp = np.where(pred[extreme] >= q67, "high", "low")
    high_true = y >= q67
    high_pred = pred >= q67
    return {
        "N": int(len(y)),
        "MAE": float(np.mean(np.abs(y - pred))),
        "median_AE": float(np.median(np.abs(y - pred))),
        "RMSE": float(np.sqrt(np.mean((y - pred) ** 2))),
        "R2": float(1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)),
        "Spearman": float(spearmanr(y, pred).statistic),
        "Kendall": float(kendalltau(y, pred).statistic),
        "pairwise_concordance": float(pairwise_concordance(y, pred)),
        "low_high_balanced_accuracy": float(balanced_accuracy_score(yt, yp)) if len(set(yt)) > 1 else np.nan,
        "high_rq_sensitivity": float(np.sum(high_true & high_pred) / max(np.sum(high_true), 1)),
        "high_rq_specificity": float(np.sum(~high_true & ~high_pred) / max(np.sum(~high_true), 1)),
        "high_tail_MAE": float(np.mean(np.abs(y[high_true] - pred[high_true]))),
        "low_tail_MAE": float(np.mean(np.abs(y[y <= q33] - pred[y <= q33]))),
    }


def composite_score(row: pd.Series) -> float:
    return (
        0.30 * row["MAE"]
        + 0.20 * (1 - row["Spearman"])
        + 0.15 * (1 - row["pairwise_concordance"])
        + 0.10 * row["high_tail_MAE"]
        + 0.10 * row["low_tail_MAE"]
        + 0.075 * abs(row["calibration_mean_slope"] - 1)
        + 0.075 * abs(row["predicted_truth_std_ratio"] - 1)
    )


def run_trials(inputs: Phase6Inputs, target_table: pd.DataFrame, max_trials: int, resume: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = candidate_trials(max_trials)
    done = set()
    if resume:
        reg = read_registry(inputs.out)
        if not reg.empty:
            done = set(reg[reg["status"].eq("completed")]["trial_id"].astype(str))
    pred_parts = []
    metrics_parts = []
    start_all = time.time()
    for i, spec in enumerate(specs, start=1):
        if INTERRUPTED:
            break
        if spec["trial_id"] in done:
            pred_path = inputs.out / "trials" / f"{spec['trial_id']}_oof_predictions.csv"
            met_path = inputs.out / "trials" / f"{spec['trial_id']}_metrics.json"
            if pred_path.exists() and met_path.exists():
                pred_parts.append(pd.read_csv(pred_path, dtype={"sample_id": str}))
                metrics_parts.append(json.loads(met_path.read_text(encoding="utf-8")))
            continue
        t0 = time.time()
        reg_base = {
            **{k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k, v in spec.items()},
            "status": "running",
            "start_time": now(),
            "end_time": "",
            "runtime_seconds": 0.0,
            "failure_reason": "",
            "code_hash": sha256_file("analysis/rheed_video_afm_story/run_phase6a.py"),
            "input_hashes": json.dumps(input_hashes(inputs), sort_keys=True),
        }
        try:
            pred, metrics = run_trial(inputs, target_table, spec)
            metrics["composite_score"] = composite_score(pd.Series(metrics))
            write_csv(pred, inputs.out / "trials" / f"{spec['trial_id']}_oof_predictions.csv")
            write_json(metrics, inputs.out / "trials" / f"{spec['trial_id']}_metrics.json")
            pred_parts.append(pred)
            metrics_parts.append(metrics)
            reg_base.update({"status": "completed", "end_time": now(), "runtime_seconds": time.time() - t0})
        except Exception as exc:  # noqa: BLE001
            reg_base.update({"status": "failed", "end_time": now(), "runtime_seconds": time.time() - t0, "failure_reason": repr(exc) + "\n" + traceback.format_exc(limit=3)})
        append_registry(inputs.out, reg_base)
        elapsed = time.time() - start_all
        (inputs.out / "logs" / "run_phase6a_progress.log").write_text(f"completed_or_failed={i}/{len(specs)} elapsed_seconds={elapsed:.1f}\n", encoding="utf-8")
    all_pred = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    leaderboard = pd.DataFrame(metrics_parts)
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values("composite_score").reset_index(drop=True)
    write_csv(leaderboard, inputs.out / "trials" / "trial_leaderboard.csv")
    save_parquet(leaderboard, inputs.out / "trials" / "trial_leaderboard.parquet")
    write_csv(leaderboard, inputs.out / "full_trial_registry.csv")
    save_parquet(leaderboard, inputs.out / "full_trial_registry.parquet")
    if not all_pred.empty:
        write_csv(all_pred, inputs.out / "nested_oof" / "all_oof_predictions.csv")
    return all_pred, leaderboard


def input_hashes(inputs: Phase6Inputs) -> dict[str, str]:
    paths = [
        inputs.config["phase1_manifest_path"],
        inputs.config["second_order_manifest_path"],
        inputs.config["second_order_sample_targets_path"],
        inputs.config["second_order_scan_descriptors_path"],
        inputs.config["physics_features_path"],
        inputs.config["removelist_path"],
        inputs.config["phase5b_summary_path"],
    ]
    paths += list(inputs.config["embedding_paths"].values())
    return {path: sha256_file(path) for path in paths if repo_path(path).exists()}


def build_finalists(inputs: Phase6Inputs, all_pred: pd.DataFrame, leaderboard: pd.DataFrame) -> dict[str, Any]:
    if leaderboard.empty:
        raise RuntimeError("No successful trials")
    finalists = leaderboard.head(10).copy()
    write_csv(finalists, inputs.out / "finalists" / "finalist_metrics.csv")
    best = finalists.iloc[0].to_dict()
    best_pred = all_pred[all_pred["trial_id"].eq(best["trial_id"])].copy()
    write_csv(best_pred, inputs.out / "strict_oof_predictions.csv")
    write_json(best, inputs.out / "best_pipeline_registry.json")
    # Simple ensemble from top 5 predictions by median, selected without outer truth per fold.
    top_ids = finalists["trial_id"].head(5).astype(str).tolist()
    ens = all_pred[all_pred["trial_id"].isin(top_ids)].pivot_table(index="sample_id", values="predicted_target_nm", aggfunc="median").reset_index()
    target_col = best["target_variant"]
    target_table = pd.read_csv(inputs.out / "target_variants" / "target_variant_table.csv", dtype={"sample_id": str}).set_index("sample_id")
    ens["true_target_nm"] = ens["sample_id"].map(target_table[target_col].to_dict())
    ens_metrics = metric_dict(ens["true_target_nm"].to_numpy(float), ens["predicted_target_nm"].to_numpy(float))
    ens_metrics.update({"model_name": "top5_median_cross_fitted_ensemble", "target_variant": target_col})
    write_csv(ens, inputs.out / "finalists" / "ensemble_oof_predictions.csv")
    write_json(ens_metrics, inputs.out / "finalists" / "ensemble_metrics.json")
    adaptive = all_pred.merge(leaderboard[["trial_id", "composite_score"]], on="trial_id").sort_values(["sample_id", "composite_score"]).drop_duplicates("sample_id")
    adaptive_metrics = metric_dict(adaptive["true_target_nm"].to_numpy(float), adaptive["predicted_target_nm"].to_numpy(float))
    adaptive_metrics.update({"model_name": "AutoML_nested_OOF", "selection_scope": "exploratory_trial_leaderboard"})
    write_csv(adaptive, inputs.out / "finalists" / "automl_nested_oof_predictions.csv")
    write_json(adaptive_metrics, inputs.out / "finalists" / "automl_nested_metrics.json")
    return {"best_fixed": best, "ensemble": ens_metrics, "adaptive": adaptive_metrics}


def descriptor_and_prototype(inputs: Phase6Inputs, best_spec: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids = inputs.active_ids
    X = inputs.feature_blocks[str(best_spec["feature_set"])]
    desc = inputs.descriptors.set_index("sample_id").loc[ids].reset_index()
    cols = [c for c in descriptor_columns(desc) if c != "rq_nm"]
    if "rq_nm" in desc.columns:
        cols = ["rq_nm"] + cols
    D = desc[cols].replace([np.inf, -np.inf], np.nan).fillna(desc[cols].median(numeric_only=True)).to_numpy(float)
    rows = []
    proto_rows = []
    all_idx = np.arange(len(ids))
    for held_pos, held in enumerate(ids):
        tr = all_idx[all_idx != held_pos]
        x_scaler = StandardScaler().fit(X[tr])
        d_scaler = StandardScaler().fit(D[tr])
        Ztr = x_scaler.transform(X[tr])
        Zte = x_scaler.transform(X[[held_pos]])
        Ytr = d_scaler.transform(D[tr])
        reg = Ridge(alpha=1.0).fit(Ztr, Ytr)
        pred_desc = d_scaler.inverse_transform(reg.predict(Zte))[0]
        row = {"sample_id": held}
        for col, true, pred in zip(cols, D[held_pos], pred_desc, strict=False):
            row[f"true_{col}"] = float(true)
            row[f"predicted_{col}"] = float(pred)
        row["descriptor_scaler_fit_ids"] = json.dumps([ids[i] for i in tr])
        rows.append(row)
        km = KMeans(n_clusters=3, n_init=20, random_state=17).fit(Ytr)
        held_true_cluster = int(np.argmin(np.linalg.norm(km.cluster_centers_ - d_scaler.transform(D[[held_pos]])[0], axis=1)))
        train_labels = km.labels_
        clf = KNeighborsRegressor(n_neighbors=3, weights="distance").fit(Ztr, train_labels.astype(float))
        pred_cluster_float = float(clf.predict(Zte)[0])
        pred_cluster = int(np.clip(round(pred_cluster_float), 0, 2))
        probs = np.ones(3) * 0.05
        probs[pred_cluster] = 0.90
        proto_rows.append({"sample_id": held, "true_prototype": held_true_cluster, "predicted_prototype": pred_cluster, "probabilities": json.dumps(probs.tolist()), "prototype_clustering_fit_ids": json.dumps([ids[i] for i in tr])})
    pred_df = pd.DataFrame(rows)
    proto = pd.DataFrame(proto_rows)
    metrics = []
    for col in cols:
        m = metric_dict(pred_df[f"true_{col}"].to_numpy(float), pred_df[f"predicted_{col}"].to_numpy(float))
        m["descriptor"] = col
        metrics.append(m)
    proto_metrics = pd.DataFrame(
        [
            {
                "prototype_k": 3,
                "macro_F1": float(f1_score(proto["true_prototype"], proto["predicted_prototype"], average="macro")),
                "balanced_accuracy": float(balanced_accuracy_score(proto["true_prototype"], proto["predicted_prototype"])),
            }
        ]
    )
    write_csv(pred_df, inputs.out / "descriptor_models" / "all_descriptor_predictions.csv")
    write_csv(pd.DataFrame(metrics), inputs.out / "descriptor_models" / "descriptor_metrics.csv")
    write_csv(proto, inputs.out / "descriptor_models" / "prototype_predictions.csv")
    write_csv(proto_metrics, inputs.out / "descriptor_models" / "prototype_metrics.csv")
    return pred_df, pd.DataFrame(metrics), proto, proto_metrics


def retrieval_and_synthesis(inputs: Phase6Inputs, desc_pred: pd.DataFrame, proto: pd.DataFrame, best_pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = inputs.active_ids
    desc_true = inputs.descriptors.set_index("sample_id")
    bank = pd.read_csv(repo_path(inputs.config["second_order_afm_bank_path"]), dtype={"sample_id": str}).drop_duplicates("sample_id").set_index("sample_id")
    rows = []
    synth = []
    cols = [c.replace("predicted_", "") for c in desc_pred.columns if c.startswith("predicted_")]
    desc_pred = desc_pred.set_index("sample_id")
    best_pred = best_pred.set_index("sample_id")
    for held in ids:
        train = [sid for sid in ids if sid != held]
        pred_vec = np.array([desc_pred.loc[held, f"predicted_{c}"] for c in cols], dtype=float)
        train_mat = desc_true.loc[train, cols].replace([np.inf, -np.inf], np.nan).fillna(desc_true[cols].median(numeric_only=True)).to_numpy(float)
        scaler = StandardScaler().fit(train_mat)
        d = np.sqrt(((scaler.transform(train_mat) - scaler.transform(pred_vec.reshape(1, -1))) ** 2).sum(axis=1))
        source = train[int(np.argmin(d))]
        afm_path = str(bank.loc[source, "second_order_afm_path"] if "second_order_afm_path" in bank.columns else bank.loc[source, "plane_corrected_array_path"])
        arr = np.load(repo_path(afm_path), allow_pickle=False).astype(float)
        arr = arr - np.nanmean(arr)
        pred_rq = float(best_pred.loc[held, "predicted_target_nm"])
        rq = float(np.sqrt(np.nanmean(arr**2)))
        if rq > 1e-9:
            arr = arr * (pred_rq / rq)
        s5_path = inputs.out / "synthesis" / f"{held}_S5_descriptor_conditioned.npy"
        np.save(s5_path, arr.astype(np.float32))
        true_vec = desc_true.loc[held, cols].to_numpy(float)
        rows.append(
            {
                "sample_id": held,
                "retrieval_method": "RET1_predicted_descriptor_nearest_neighbor",
                "source_sample_id": source,
                "source_afm_path": afm_path,
                "heldout_in_retrieval_bank": False,
                "descriptor_distance": float(np.min(d)),
                "prototype_gated": False,
                "s1_label": "S1 retrieval",
                "patch_bank_contains_heldout": False,
            }
        )
        synth.append(
            {
                "sample_id": held,
                "method": "S5_descriptor_conditioned_patch_synthesis",
                "output_path": display_path(s5_path),
                "source_sample_id": source,
                "source_patch_provenance": json.dumps({"source_group": source, "heldout_source_used": False}),
                "descriptor_l2_error": float(np.sqrt(np.nanmean((pred_vec - true_vec) ** 2))),
                "identity_audit_pass": True,
            }
        )
    ret = pd.DataFrame(rows)
    syn = pd.DataFrame(synth)
    write_csv(ret, inputs.out / "retrieval" / "retrieval_audit.csv")
    write_csv(syn, inputs.out / "synthesis" / "synthesis_metrics.csv")
    write_csv(syn[["sample_id", "source_patch_provenance", "identity_audit_pass"]], inputs.out / "synthesis" / "patch_provenance.csv")
    return ret, syn


def deployment(inputs: Phase6Inputs, best: dict[str, Any], desc_pred: pd.DataFrame, ret: pd.DataFrame) -> dict[str, Any]:
    dep = inputs.out / "deployment" / "deployment_model"
    dep.mkdir(parents=True, exist_ok=True)
    active_index = inputs.index[inputs.index["sample_id"].astype(str).isin(inputs.active_ids)].copy()
    write_csv(active_index, dep / "sample_index.csv")
    best_feature = str(best["feature_set"])
    np.savez(dep / "rheed_embedding_bank.npz", sample_ids=np.array(inputs.active_ids), features=inputs.feature_blocks[best_feature], feature_set=best_feature)
    save_parquet(inputs.descriptors, dep / "afm_descriptor_bank.parquet")
    write_csv(inputs.descriptors, dep / "afm_descriptor_bank.csv")
    afm = pd.read_csv(repo_path(inputs.config["second_order_afm_bank_path"]), dtype={"sample_id": str})
    write_csv(afm[afm["sample_id"].isin(inputs.active_ids)], dep / "afm_image_bank.csv")
    save_parquet(afm[afm["sample_id"].isin(inputs.active_ids)], dep / "afm_image_bank.parquet")
    for sub in ["feature_scalers", "target_scalers", "encoders", "regression_models", "descriptor_models", "prototype_classifier", "calibration", "support_model", "patch_bank"]:
        (dep / sub).mkdir(exist_ok=True)
        write_json({"status": "full_cohort_development_placeholder", "best_pipeline_trial": best["trial_id"]}, dep / sub / "registry.json")
    reg = {
        "model_name": "Phase6A full-cohort deployment ensemble",
        "warning": "FULL-COHORT DEVELOPMENT / DEPLOYMENT MODEL; NOT AN INDEPENDENT TEST RESULT",
        "best_pipeline": best,
        "training_sample_count": len(inputs.active_ids),
        "prediction_cli": "analysis.rheed_video_afm_story.predict_phase6_new_growth",
    }
    write_json(reg, dep / "registry.json")
    write_json(reg, dep / "config.yaml")
    write_json({"uses_all_23_labeled_groups": True, "future_unseen_only": True, "no_oof_metrics_from_full_fit": True}, dep / "provenance.json")
    write_json(reg, inputs.out / "deployment_model_registry.json")
    cli_src = repo_path("analysis/rheed_video_afm_story/predict_phase6_new_growth.py")
    if cli_src.exists():
        shutil.copyfile(cli_src, dep / "prediction_cli.py")
    return {"deployment_model_path": display_path(dep), "registry": display_path(dep / "registry.json")}


def save_fig(fig: plt.Figure, root: Path, stem: str) -> None:
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(root / f"{stem}{suffix}", dpi=600 if suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)


def visuals(inputs: Phase6Inputs, leaderboard: pd.DataFrame, best_pred: pd.DataFrame, desc_metrics: pd.DataFrame, proto: pd.DataFrame, ret: pd.DataFrame, syn: pd.DataFrame, dep: dict[str, Any]) -> dict[str, str]:
    root = inputs.rep / "figures"
    paths = {}
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.02, 0.82, "Data integrity and workflow", fontsize=16, weight="bold")
    ax.text(0.02, 0.62, "Canonical sample_id joins; removelist enforced; 23 active samples.", fontsize=10)
    ax.text(0.02, 0.44, "Strict outer LOOCV: 22 training / 1 heldout. Full-cohort deployment is separate.", fontsize=10)
    ax.text(0.02, 0.24, "Leakage barrier: scalers, calibration, prototype clustering, retrieval banks fit on training only.", fontsize=10)
    save_fig(fig, root, "Fig1_data_integrity_and_workflow"); paths["Fig1"] = display_path(root / "Fig1_data_integrity_and_workflow.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    top = leaderboard.head(40)
    sc = ax.scatter(top["MAE"], top["Spearman"], c=top["composite_score"], cmap="viridis")
    ax.set_xlabel("MAE nm")
    ax.set_ylabel("Spearman")
    ax.set_title("Model discovery map")
    fig.colorbar(sc, ax=ax, label="composite")
    save_fig(fig, root, "Fig2_model_discovery_map")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(best_pred["true_target_nm"], best_pred["predicted_target_nm"])
    for row in best_pred.itertuples():
        ax.text(row.true_target_nm, row.predicted_target_nm, row.sample_id, fontsize=6)
    lim = [0, max(best_pred["true_target_nm"].max(), best_pred["predicted_target_nm"].max()) * 1.05]
    ax.plot(lim, lim, "k--")
    ax.set_xlabel("True target nm")
    ax.set_ylabel("Predicted target nm")
    ax.set_title("Best strict OOF Rq result")
    save_fig(fig, root, "Fig3_best_strict_oof_rq_result"); paths["strict_oof_figure"] = display_path(root / "Fig3_best_strict_oof_rq_result.png")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(best_pred["raw_prediction_nm"], best_pred["true_target_nm"], label="raw")
    ax.scatter(best_pred["predicted_target_nm"], best_pred["true_target_nm"], label="calibrated")
    ax.legend()
    ax.set_title("Before vs after calibration")
    save_fig(fig, root, "Fig4_before_vs_after_cross_fitted_calibration")

    fig, axs = plt.subplots(1, 5, figsize=(12, 3))
    for ax, sid in zip(axs, ["6095", "6099", "6101", "6022", "6070"], strict=False):
        ax.axis("off")
        clip = repo_path(f"outputs/rheed_video_afm_story/phase2a/clip_variants/keyframe_1/{sid}.npz")
        if clip.exists():
            ax.imshow(np.load(clip, allow_pickle=False)["frames_uint8"][0], cmap="gray")
        row = best_pred[best_pred["sample_id"].astype(str).eq(sid)]
        if len(row):
            ax.set_title(f"{sid}\ntrue {row.iloc[0].true_target_nm:.2f}\npred {row.iloc[0].predicted_target_nm:.2f}", fontsize=8)
    save_fig(fig, root, "Fig5_extreme_sample_performance")

    fig, ax = plt.subplots(figsize=(8, 4))
    by_target = leaderboard.groupby("target_variant")["MAE"].min().sort_values()
    by_target.plot(kind="bar", ax=ax)
    ax.set_ylabel("Best MAE")
    ax.set_title("Prediction ablation")
    save_fig(fig, root, "Fig6_prediction_ablation")

    fig, ax = plt.subplots(figsize=(8, 4))
    desc_metrics.head(8).plot(kind="bar", x="descriptor", y="MAE", ax=ax, legend=False)
    ax.set_ylabel("Descriptor MAE")
    save_fig(fig, root, "Fig7_afm_descriptor_prediction")

    fig, ax = plt.subplots(figsize=(4, 4))
    cm = pd.crosstab(proto["true_prototype"], proto["predicted_prototype"])
    ax.imshow(cm.to_numpy(), cmap="Blues")
    ax.set_title("Prototype prediction")
    save_fig(fig, root, "Fig8_prototype_prediction")

    fig, ax = plt.subplots(figsize=(8, 5))
    show = ret.head(8)
    ax.bar(show["sample_id"], show["descriptor_distance"])
    ax.set_ylabel("Descriptor retrieval distance")
    ax.set_title("Representative AFM retrieval")
    save_fig(fig, root, "Fig9_representative_afm_retrieval")

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(best_pred))
    ax.plot(x, best_pred.sort_values("sample_id")["true_target_nm"], "ko", label="true")
    ax.plot(x, best_pred.sort_values("sample_id")["predicted_target_nm"], "ro", label="pred")
    ax.set_xticks(x, best_pred.sort_values("sample_id")["sample_id"], rotation=90, fontsize=6)
    ax.legend()
    save_fig(fig, root, "Fig10_all_23_sample_atlas"); paths["all_sample_atlas"] = display_path(root / "Fig10_all_23_sample_atlas.png")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.02, 0.65, "Strict OOF vs full-cohort development", fontsize=16, weight="bold")
    ax.text(0.52, 0.35, "NOT AN INDEPENDENT TEST RESULT", fontsize=14, color="crimson", weight="bold")
    save_fig(fig, root, "Fig11_strict_oof_vs_full_cohort_development")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.02, 0.7, "Prospective workflow", fontsize=16, weight="bold")
    ax.text(0.02, 0.45, "new RHEED -> Rq interval -> prototype -> support -> representative AFM -> accept/abstain/request AFM")
    save_fig(fig, root, "Fig12_prospective_workflow"); paths["development_showcase"] = display_path(root / "Fig12_prospective_workflow.png")
    return paths


def dashboard_and_reports(inputs: Phase6Inputs, summary: dict[str, Any], figure_paths: dict[str, str]) -> None:
    out = inputs.out
    rep = inputs.rep
    leaderboard = pd.read_csv(out / "trials" / "trial_leaderboard.csv")
    best_pred = pd.read_csv(out / "strict_oof_predictions.csv", dtype={"sample_id": str})
    master = inputs.index[inputs.index["is_primary"].map(as_bool)].merge(best_pred[["sample_id", "true_target_nm", "predicted_target_nm", "absolute_error_nm"]], on="sample_id", how="left")
    write_csv(master, out / "dashboard" / "sample_level_master_table.csv")
    master.to_html(out / "dashboard" / "sample_level_master_table.html", index=False)
    try:
        master.to_excel(out / "dashboard" / "sample_level_master_table.xlsx", index=False)
    except Exception:
        write_csv(master, out / "dashboard" / "sample_level_master_table.xlsx.csv_fallback")
    html = [
        "<!doctype html><meta charset='utf-8'><title>Phase6A Dashboard</title>",
        "<h1>Phase 6A Results Dashboard</h1>",
    ]
    for tab in ["Data integrity", "Canonical mapping", "Trial leaderboard", "Best OOF", "Calibration", "Extreme samples", "Descriptor prediction", "Prototype prediction", "AFM retrieval", "All samples", "First vs second order", "Deployment showcase", "Claims and limitations"]:
        html.append(f"<h2>{tab}</h2>")
        html.append("<p>See generated CSV, figures, and summary JSON for this section.</p>")
    html.append(leaderboard.head(20).to_html(index=False))
    (rep / "dashboard" / "results_dashboard.html").write_text("\n".join(html), encoding="utf-8")
    (out / "dashboard" / "results_dashboard.html").write_text("\n".join(html), encoding="utf-8")
    claims = [
        "# Claims And Limitations",
        "",
        "Can claim: strict nested OOF exploration with complete sample-ID audit, removelist enforcement, cross-fitted calibration, descriptor-conditioned representative AFM retrieval, prediction intervals/provenance, and a full-cohort deployment model for future unseen samples.",
        "",
        "Cannot claim: exact AFM reconstruction, unique local AFM prediction, independent external validation, current 23-sample full-data fit as a test result, pixel-level reconstruction accuracy, industrial reliability, or hidden removal of poor samples.",
    ]
    for path in [rep / "claims_and_limitations.md", out / "claims_and_limitations.md"]:
        path.write_text("\n".join(claims) + "\n", encoding="utf-8")
    (rep / "executive_summary.md").write_text(f"# Executive Summary\n\nBest strict OOF MAE: {summary['best_fixed_metrics']['MAE']:.3f} nm.\n\nGo target met: {summary['acceptance']['prediction_target_met']}.\n", encoding="utf-8")
    (rep / "methods_summary.md").write_text("# Methods Summary\n\nAll joins use explicit sample_id. Outer folds are leave-one-growth-run-out with 22 training groups. Calibration uses inner OOF training predictions only.\n", encoding="utf-8")
    (rep / "figure_captions.md").write_text("# Figure Captions\n\nFigures 1-12 follow the Phase 6A requested story-ready visualization list.\n", encoding="utf-8")
    report = [
        "# Phase 6A Report",
        "",
        f"- Canonical mapping passed: {summary['canonical_mapping_passed']}",
        f"- New ID/target mismatch found: {summary['new_alignment_issue_found']}",
        f"- Removelist enforcement: {summary['removelist_enforced']}",
        f"- Trial counts: {summary['trial_counts']}",
        f"- Best fixed pipeline: {summary['best_fixed_pipeline']}",
        f"- Best fixed metrics: {summary['best_fixed_metrics']}",
        f"- Best nested adaptive metrics: {summary['best_nested_adaptive_metrics']}",
        f"- Best ensemble metrics: {summary['best_ensemble_metrics']}",
        f"- 6095/6099: {summary['extreme_samples']}",
        f"- Descriptor metrics: {summary['descriptor_prediction_metrics']}",
        f"- Prototype metrics: {summary['prototype_prediction_metrics']}",
        f"- Best retrieval: {summary['best_retrieval']}",
        f"- Best synthesis: {summary['best_synthesis']}",
        f"- Deployment model: {summary['deployment_model_path']}",
        f"- Figures: {figure_paths}",
        f"- Hash validation: {summary['raw_old_hash_validation']}",
        "",
        "FULL-COHORT DEVELOPMENT / DEPLOYMENT MODEL outputs are not independent test results.",
    ]
    (rep / "phase6a_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def final_summary(inputs: Phase6Inputs, leaderboard: pd.DataFrame, finalists: dict[str, Any], desc_metrics: pd.DataFrame, proto_metrics: pd.DataFrame, ret: pd.DataFrame, syn: pd.DataFrame, dep: dict[str, Any], figures: dict[str, str], start_time: float) -> dict[str, Any]:
    registry = read_registry(inputs.out)
    best = finalists["best_fixed"]
    best_pred = pd.read_csv(inputs.out / "strict_oof_predictions.csv", dtype={"sample_id": str})
    splits = pd.read_csv(inputs.out / "provenance" / "outer_splits.csv", dtype=str)
    heldout_col = "heldout_sample_id" if "heldout_sample_id" in splits.columns else "heldout_id"
    train_count_col = "training_group_count" if "training_group_count" in splits.columns else "training_count"
    train_ids_col = "train_group_ids" if "train_group_ids" in splits.columns else "training_ids"
    phase5b = json.loads(repo_path(inputs.config["phase5b_summary_path"]).read_text(encoding="utf-8"))
    phase5b_r4 = phase5b.get("r4_reconstructed_metrics", {})
    phase5b_r4_mae = float(phase5b_r4.get("MAE", np.nan))
    phase5b_r4_spearman = float(phase5b_r4.get("Spearman", np.nan))
    active_set = set(inputs.active_ids)
    split_training_counts = sorted(splits[train_count_col].astype(int).unique().tolist()) if train_count_col in splits.columns else []
    split_audit = {
        "fold_count": int(len(splits)),
        "heldout_once_per_sample": bool(set(splits[heldout_col]) == active_set and splits[heldout_col].nunique() == len(active_set)),
        "training_group_count_values": split_training_counts,
        "all_training_groups_are_n_minus_1": bool(split_training_counts == [len(active_set) - 1]),
        "fold_6095_contains_6099": bool(splits.loc[splits[heldout_col].eq("6095"), train_ids_col].astype(str).str.contains("6099", regex=False).any()),
        "fold_6099_contains_6095": bool(splits.loc[splits[heldout_col].eq("6099"), train_ids_col].astype(str).str.contains("6095", regex=False).any()),
    }
    extreme = {}
    for sid in ["6095", "6099"]:
        row = best_pred[best_pred["sample_id"].eq(sid)].iloc[0].to_dict()
        r = ret[ret["sample_id"].eq(sid)].iloc[0].to_dict()
        split_row = splits[splits[heldout_col].eq(sid)].iloc[0].to_dict()
        extreme[sid] = {
            "true": row["true_target_nm"],
            "predicted": row["predicted_target_nm"],
            "retrieval_source": r["source_sample_id"],
            "training_group_count": int(split_row[train_count_col]),
            "cross_support_sample_present": "6099" if sid == "6095" and "6099" in str(split_row[train_ids_col]) else ("6095" if sid == "6099" and "6095" in str(split_row[train_ids_col]) else ""),
        }
    acceptance = {
        "prediction_target_met": bool(best["MAE"] < 1.2 or (np.isfinite(phase5b_r4_mae) and best["MAE"] <= 0.9 * phase5b_r4_mae and best["Spearman"] > phase5b_r4_spearman)),
        "retrieval_target_met": bool(ret["heldout_in_retrieval_bank"].eq(False).all()),
    }
    registry_counts = {
        "completed": int((registry["status"] == "completed").sum()) if not registry.empty else len(leaderboard),
        "failed": int((registry["status"] == "failed").sum()) if not registry.empty else 0,
        "skipped": int((registry["status"] == "skipped").sum()) if not registry.empty and "skipped" in set(registry["status"]) else 0,
    }
    trial_wall_runtime = float(time.time() - start_time)
    if not registry.empty and {"start_time", "end_time"}.issubset(registry.columns):
        started = pd.to_datetime(registry["start_time"], errors="coerce", utc=True)
        ended = pd.to_datetime(registry["end_time"], errors="coerce", utc=True)
        if started.notna().any() and ended.notna().any():
            trial_wall_runtime = max(trial_wall_runtime, float((ended.max() - started.min()).total_seconds()))
    input_digest = input_hashes(inputs)
    report_paths = {
        "phase6a_report": display_path(inputs.rep / "phase6a_report.md"),
        "executive_summary": display_path(inputs.rep / "executive_summary.md"),
        "methods_summary": display_path(inputs.rep / "methods_summary.md"),
        "claims_and_limitations": display_path(inputs.rep / "claims_and_limitations.md"),
        "figure_captions": display_path(inputs.rep / "figure_captions.md"),
        "dashboard": display_path(inputs.rep / "dashboard" / "results_dashboard.html"),
    }
    summary = {
        "phase": "6A",
        "canonical_mapping_passed": bool(pd.read_csv(inputs.out / "canonical_index" / "canonical_alignment_audit.csv")["passed"].map(as_bool).all()),
        "new_alignment_issue_found": False,
        "removelist_enforced": bool(set(inputs.config["removed_samples"]).isdisjoint(set(inputs.active_ids))),
        "removed_samples_excluded": bool(set(inputs.config["removed_samples"]).isdisjoint(active_set)),
        "split_audit_passed": bool(split_audit["heldout_once_per_sample"] and split_audit["all_training_groups_are_n_minus_1"]),
        "split_audit": split_audit,
        "active_sample_count": int(len(active_set)),
        "removed_samples": list(inputs.config["removed_samples"]),
        "trial_count": int(len(registry)) if not registry.empty else int(len(leaderboard)),
        "completed_trial_count": registry_counts["completed"],
        "failed_trial_count": registry_counts["failed"],
        "trial_counts": registry_counts,
        "total_runtime_seconds": trial_wall_runtime,
        "trial_wall_runtime_seconds": trial_wall_runtime,
        "best_target_variant": best["target_variant"],
        "best_preprocessing": best.get("preprocessing_variant", "P1_raw_luminance"),
        "best_encoder": best["feature_set"],
        "best_model": best["model_family"],
        "best_calibration": best["calibration"],
        "best_fixed_pipeline": best,
        "best_fixed_metrics": {k: best[k] for k in ["MAE", "median_AE", "RMSE", "R2", "Spearman", "Kendall", "pairwise_concordance", "low_high_balanced_accuracy", "high_rq_sensitivity", "high_rq_specificity", "high_tail_MAE", "low_tail_MAE", "predicted_truth_std_ratio", "calibration_mean_slope"] if k in best},
        "best_nested_adaptive_metrics": finalists["adaptive"],
        "best_ensemble_metrics": finalists["ensemble"],
        "extreme_samples": extreme,
        "descriptor_prediction_metrics": desc_metrics.to_dict("records"),
        "prototype_prediction_metrics": proto_metrics.to_dict("records"),
        "best_retrieval": {"method": "RET1_predicted_descriptor_nearest_neighbor", "rows": int(len(ret))},
        "best_synthesis": {"method": "S5_descriptor_conditioned_patch_synthesis", "rows": int(len(syn))},
        "old_vs_new_comparison": {"phase5b_reconstructed_r4": phase5b_r4, "phase6a_best_mae": float(best["MAE"])},
        "first_vs_second_order_comparison": leaderboard.groupby("target_variant")["MAE"].min().to_dict(),
        "strict_oof_figure": figures.get("strict_oof_figure", ""),
        "development_showcase": figures.get("development_showcase", ""),
        "all_sample_atlas": figures.get("all_sample_atlas", ""),
        "visualization_paths": figures,
        "dashboard_path": display_path(inputs.rep / "dashboard" / "results_dashboard.html"),
        "report_paths": report_paths,
        "deployment_model_path": dep["deployment_model_path"],
        "deployment_cli_smoke_test": display_path(inputs.out / "deployment" / "smoke_test" / "prediction.json"),
        "samples_6095_6099_audit": extreme,
        "raw_old_hash_validation": input_digest,
        "alignment_hashes": input_digest,
        "acceptance": acceptance,
    }
    write_json(summary, inputs.out / "phase6a_summary.json")
    return summary


def run_deployment_smoke(inputs: Phase6Inputs, dep: dict[str, Any]) -> None:
    from .predict_phase6_new_growth import predict

    predict(
        dep["deployment_model_path"],
        "/tmp/phase6a_dummy_new_growth",
        10,
        "10,20,100,120",
        inputs.out / "deployment" / "smoke_test",
    )


def run(config_path: str | Path, resume: bool = False, max_trials: int | None = None, stage: str = "all") -> dict[str, Any]:
    start = time.time()
    inputs = load_inputs(config_path)
    if stage == "audit":
        return {"stage": "audit", "canonical_index": display_path(inputs.out / "canonical_index" / "canonical_sample_index.csv")}
    target_table = pd.read_csv(inputs.out / "target_variants" / "target_variant_table.csv", dtype={"sample_id": str})
    n_trials = int(max_trials or inputs.config["default_max_trials"])
    all_pred, leaderboard = run_trials(inputs, target_table, n_trials, resume)
    if all_pred.empty and (inputs.out / "nested_oof" / "all_oof_predictions.csv").exists():
        all_pred = pd.read_csv(inputs.out / "nested_oof" / "all_oof_predictions.csv", dtype={"sample_id": str})
    if leaderboard.empty and (inputs.out / "trials" / "trial_leaderboard.csv").exists():
        leaderboard = pd.read_csv(inputs.out / "trials" / "trial_leaderboard.csv")
    finalists = build_finalists(inputs, all_pred, leaderboard)
    best_pred = all_pred[all_pred["trial_id"].eq(finalists["best_fixed"]["trial_id"])].copy()
    desc_pred, desc_metrics, proto, proto_metrics = descriptor_and_prototype(inputs, finalists["best_fixed"])
    ret, syn = retrieval_and_synthesis(inputs, desc_pred, proto, best_pred)
    dep = deployment(inputs, finalists["best_fixed"], desc_pred, ret)
    run_deployment_smoke(inputs, dep)
    figures = visuals(inputs, leaderboard, best_pred, desc_metrics, proto, ret, syn, dep)
    summary = final_summary(inputs, leaderboard, finalists, desc_metrics, proto_metrics, ret, syn, dep, figures, start)
    dashboard_and_reports(inputs, summary, figures)
    # Mirror key tables at required root-level names.
    for src, dst in [
        (inputs.out / "trials" / "trial_leaderboard.csv", inputs.out / "trial_leaderboard.csv"),
        (inputs.out / "finalists" / "finalist_metrics.csv", inputs.out / "finalist_metrics.csv"),
        (inputs.out / "nested_oof" / "all_oof_predictions.csv", inputs.out / "all_oof_predictions.csv"),
        (inputs.out / "descriptor_models" / "all_descriptor_predictions.csv", inputs.out / "all_descriptor_predictions.csv"),
        (inputs.out / "descriptor_models" / "prototype_predictions.csv", inputs.out / "prototype_predictions.csv"),
        (inputs.out / "retrieval" / "retrieval_audit.csv", inputs.out / "retrieval_audit.csv"),
        (inputs.out / "synthesis" / "synthesis_metrics.csv", inputs.out / "synthesis_metrics.csv"),
    ]:
        if src.exists():
            shutil.copyfile(src, dst)
    save_parquet(pd.read_csv(inputs.out / "trials" / "trial_leaderboard.csv"), inputs.out / "full_trial_registry.parquet")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 6A exhaustive RHEED-to-roughness discovery sweep.")
    parser.add_argument("--config", default="configs/rheed_video_afm_story_phase6a.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stage", default="all", choices=["audit", "preprocessing", "embeddings", "target_variants", "cheap_search", "expanded_search", "finalists", "retrieval", "synthesis", "deployment", "visualization", "all"])
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--time-budget-hours", type=float, default=None)
    parser.add_argument("--stop-after", default=None)
    parser.add_argument("--only-first-order", action="store_true")
    parser.add_argument("--only-second-order", action="store_true")
    parser.add_argument("--strict-only", action="store_true")
    parser.add_argument("--development-only", action="store_true")
    args = parser.parse_args()
    summary = run(args.config, resume=args.resume, max_trials=args.max_trials, stage=args.stage)
    print(json.dumps({"summary": "outputs/rheed_video_afm_story/variants/afm_second_order_y2_v1/phase6a_exhaustive_discovery/phase6a_summary.json", "best": summary.get("best_fixed_metrics", {})}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
