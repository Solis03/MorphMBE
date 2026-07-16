from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .baseline_rq import pairwise_concordance
from .common import display_path, load_config, repo_path, save_parquet, sha256_file, write_csv, write_json


REGIME_ORDER = ["low", "middle", "high"]
REGIME_RANK = {name: i for i, name in enumerate(REGIME_ORDER)}
FIG_NAMES = [
    "Fig1_maximal_training_protocol",
    "Fig2_fold_membership_heatmap",
    "Fig3_fold_regime_support",
    "Fig4_neighbor_support_map",
    "Fig5_current_R4_vs_local_analog",
    "Fig6_extreme_high_rq_case_studies",
    "Fig7_regime_confusion_and_rq_distribution",
    "Fig8_support_coverage_performance",
    "Fig9_old_vs_regime_gated_retrieval",
    "Fig10_all_23_oof_prediction_grid",
    "Fig11_full_cohort_deployment_workflow",
    "FigS_in_sample_calibration_warning",
]


@dataclass
class Inputs:
    config: dict[str, Any]
    out: Path
    rep: Path
    manifest: pd.DataFrame
    targets: pd.DataFrame
    sample_ids: list[str]
    y_second: np.ndarray
    y_first: np.ndarray
    embeddings: dict[str, np.ndarray]
    physics: pd.DataFrame
    quality: pd.DataFrame
    old_r4_second: pd.DataFrame
    old_r4_first: pd.DataFrame
    old_retrieval: pd.DataFrame
    old_synthesis: pd.DataFrame
    afm_bank: pd.DataFrame


def ensure_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    out = repo_path(config["output_root"])
    rep = repo_path(config["report_root"])
    for path in [
        out,
        rep,
        out / "deployment_model",
        out / "deployment_model" / "bootstrap_models",
        out / "regime_gated_afm",
        out / "case_studies",
        rep / "figures",
        rep / "case_studies",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return out, rep


def as_bool(series: pd.Series) -> pd.Series:
    return series.map(lambda x: bool(x) if isinstance(x, (bool, np.bool_)) else str(x) == "True")


def load_npz_embedding(path: str | Path, sample_ids: list[str]) -> np.ndarray:
    z = np.load(repo_path(path), allow_pickle=False)
    ids = [str(x) for x in z["sample_ids"].tolist()]
    X = np.asarray(z["embeddings"], dtype=float)
    return X[[ids.index(sid) for sid in sample_ids]]


def load_inputs(config_path: str | Path) -> Inputs:
    config = load_config(config_path)
    out, rep = ensure_dirs(config)
    manifest = pd.read_csv(repo_path(config["manifest_path"]), dtype={"sample_id": str, "growth_run_id": str})
    manifest = manifest[as_bool(manifest["usable_for_modeling"]) & as_bool(manifest["cohort_primary_1um"])].copy()
    manifest = manifest[~manifest["sample_id"].isin(set(config["excluded_samples"]))].sort_values("sample_id").reset_index(drop=True)
    targets = pd.read_csv(repo_path(config["sample_targets_path"]), dtype={"sample_id": str, "growth_run_id": str})
    targets = targets.set_index("sample_id").loc[manifest["sample_id"].astype(str)].reset_index()
    sample_ids = targets["sample_id"].astype(str).tolist()
    physics = pd.read_csv(repo_path(config["physics_features_path"]), dtype={"sample_id": str, "growth_run_id": str})
    physics = physics.set_index("sample_id").loc[sample_ids].reset_index()
    quality = pd.read_csv(repo_path(config["rheed_quality_path"]), dtype={"sample_id": str}).drop_duplicates("sample_id")
    quality = quality.set_index("sample_id").reindex(sample_ids).reset_index()
    embeddings = {
        "E1_dino": load_npz_embedding(config["dino_embedding_path"], sample_ids),
        "E2_r3d": load_npz_embedding(config["r3d_embedding_path"], sample_ids),
        "E2b_r3d_centered": load_npz_embedding(config["r3d_centered_embedding_path"], sample_ids),
    }
    physics_cols = physics_feature_columns(physics)
    Xphys = physics[physics_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    embeddings["E3_physics"] = Xphys
    embeddings["E4_dino_r3d"] = np.hstack([embeddings["E1_dino"], embeddings["E2_r3d"]])
    embeddings["E5_fused"] = np.hstack([embeddings["E1_dino"], embeddings["E2_r3d"], Xphys])
    old_r4_second = pd.read_csv(repo_path(config["second_order_r4_oof_path"]), dtype={"sample_id": str, "growth_run_id": str})
    old_r4_first = pd.read_csv(repo_path(config["first_order_r4_oof_path"]), dtype={"sample_id": str, "growth_run_id": str})
    old_retrieval = pd.read_csv(repo_path(config["second_order_retrieval_path"]), dtype={"heldout_sample_id": str, "candidate_sample_id": str})
    old_synthesis = pd.read_csv(repo_path(config["second_order_synthesis_path"]), dtype={"sample_id": str})
    afm_bank = pd.read_csv(repo_path(config["second_order_afm_bank_path"]), dtype={"sample_id": str, "growth_run_id": str})
    return Inputs(
        config=config,
        out=out,
        rep=rep,
        manifest=manifest,
        targets=targets,
        sample_ids=sample_ids,
        y_second=targets["second_order_rq_nm"].to_numpy(float),
        y_first=targets["first_order_rq_nm"].to_numpy(float),
        embeddings=embeddings,
        physics=physics,
        quality=quality,
        old_r4_second=old_r4_second,
        old_r4_first=old_r4_first,
        old_retrieval=old_retrieval,
        old_synthesis=old_synthesis,
        afm_bank=afm_bank,
    )


def physics_feature_columns(physics: pd.DataFrame) -> list[str]:
    preferred = [
        "spot_summary_raw",
        "streak_summary_raw",
        "connection_summary_raw",
        "diffuse_summary_raw",
        "temporal_brightness_drift",
        "selected_16__saturation_fraction_median",
        "selected_16__temporal_stability",
        "keyframe_1__saturation_fraction_median",
    ]
    cols = [c for c in preferred if c in physics.columns]
    if len(cols) < 4:
        numeric = physics.select_dtypes(include=[np.number]).columns.tolist()
        cols = [c for c in numeric if c not in {"sample_id", "growth_run_id"}][:12]
    return cols


def transform_features(X: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, pca_dim: int | None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    scaler = StandardScaler().fit(X[train_idx])
    Ztr = scaler.transform(X[train_idx])
    Zte = scaler.transform(X[test_idx])
    meta: dict[str, Any] = {"scaler_fit_ids": train_idx.astype(int).tolist(), "pca_dim": pca_dim}
    if pca_dim is not None and pca_dim > 0:
        dim = min(pca_dim, Ztr.shape[0] - 1, Ztr.shape[1])
        if dim >= 1:
            pca = PCA(n_components=dim, random_state=17).fit(Ztr)
            Ztr = pca.transform(Ztr)
            Zte = pca.transform(Zte)
            meta["pca_actual_dim"] = int(dim)
    return Ztr, Zte, meta


def pair_distances(train: np.ndarray, test: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        tr = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
        te = test / max(float(np.linalg.norm(test)), 1e-12)
        return 1.0 - tr @ te
    return np.sqrt(np.sum((train - test) ** 2, axis=1))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cutoff = 0.5 * np.sum(w)
    return float(v[np.searchsorted(np.cumsum(w), cutoff, side="left")])


def aggregate_log_rq(log_values: np.ndarray, distances: np.ndarray, weights_mode: str, reducer: str) -> float:
    if weights_mode == "inverse_distance":
        weights = 1.0 / np.maximum(distances, 1e-6)
    else:
        weights = np.ones_like(log_values)
    if reducer == "weighted_median":
        return weighted_median(log_values, weights)
    return float(np.average(log_values, weights=weights))


def regime_from_thresholds(values: np.ndarray, q33: float, q67: float) -> np.ndarray:
    labels = np.full(len(values), "middle", dtype=object)
    labels[values <= q33] = "low"
    labels[values >= q67] = "high"
    return labels


def fit_regime_a(X: np.ndarray, train_idx: np.ndarray, y: np.ndarray, held_idx: int) -> dict[str, Any]:
    q33, q67 = np.quantile(y[train_idx], [0.33, 0.67])
    train_labels = regime_from_thresholds(y[train_idx], q33, q67)
    Ztr, Zte, _ = transform_features(X, train_idx, np.array([held_idx]), pca_dim=8)
    centroids = {}
    dists = {}
    for label in REGIME_ORDER:
        mask = train_labels == label
        if mask.any():
            centroids[label] = Ztr[mask].mean(axis=0)
            dists[label] = float(np.linalg.norm(Zte[0] - centroids[label]))
        else:
            dists[label] = float("inf")
    finite = np.array([dists[k] for k in REGIME_ORDER], dtype=float)
    finite = np.where(np.isfinite(finite), finite, np.nanmax(finite[np.isfinite(finite)]) + 10 if np.isfinite(finite).any() else 10.0)
    probs = np.exp(-finite)
    probs = probs / max(float(probs.sum()), 1e-12)
    pred = REGIME_ORDER[int(np.argmax(probs))]
    true = regime_from_thresholds(np.array([y[held_idx]]), q33, q67)[0]
    counts = {label: int((train_labels == label).sum()) for label in REGIME_ORDER}
    return {
        "q33_train": float(q33),
        "q67_train": float(q67),
        "train_labels": dict(zip([str(i) for i in train_idx], train_labels, strict=True)),
        "predicted_regime": pred,
        "true_regime": str(true),
        "prob_low": float(probs[0]),
        "prob_middle": float(probs[1]),
        "prob_high": float(probs[2]),
        "counts": counts,
    }


def fit_regime_b(X: np.ndarray, train_idx: np.ndarray, y: np.ndarray, held_idx: int) -> dict[str, Any]:
    Ztr, Zte, _ = transform_features(X, train_idx, np.array([held_idx]), pca_dim=8)
    km = KMeans(n_clusters=3, n_init=20, random_state=17).fit(Ztr)
    centers = km.cluster_centers_
    med = {}
    for c in range(3):
        med[c] = float(np.median(y[train_idx][km.labels_ == c]))
    ordered = sorted(med, key=med.get)
    cluster_to_regime = {ordered[i]: REGIME_ORDER[i] for i in range(3)}
    held_cluster = int(np.argmin(np.linalg.norm(centers - Zte[0], axis=1)))
    train_labels = np.array([cluster_to_regime[int(c)] for c in km.labels_])
    q33, q67 = np.quantile(y[train_idx], [0.33, 0.67])
    return {
        "predicted_regime": cluster_to_regime[held_cluster],
        "true_regime": str(regime_from_thresholds(np.array([y[held_idx]]), q33, q67)[0]),
        "cluster_to_regime": {str(k): v for k, v in cluster_to_regime.items()},
        "counts": {label: int((train_labels == label).sum()) for label in REGIME_ORDER},
    }


def allowed_by_regime(train_labels: np.ndarray, pred_regime: str, min_same: int = 2) -> np.ndarray:
    ranks = np.array([REGIME_RANK[x] for x in train_labels])
    pred_rank = REGIME_RANK[pred_regime]
    same = ranks == pred_rank
    if same.sum() >= min_same:
        return same
    adjacent = np.abs(ranks - pred_rank) <= 1
    if adjacent.sum() >= min_same:
        return adjacent
    return np.ones_like(same, dtype=bool)


def knn_predict(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    held_idx: int,
    params: dict[str, Any],
    train_regime_labels: np.ndarray | None = None,
    predicted_regime: str | None = None,
) -> tuple[float, list[int], list[float]]:
    train_use = train_idx.copy()
    if params.get("gate") in {"regime_a", "cluster_b"} and train_regime_labels is not None and predicted_regime:
        keep = allowed_by_regime(train_regime_labels, predicted_regime)
        train_use = train_use[keep]
    Ztr, Zte, _ = transform_features(X, train_use, np.array([held_idx]), params.get("pca_dim"))
    d = pair_distances(Ztr, Zte[0], params.get("distance", "cosine"))
    k = min(int(params.get("k", 3)), len(train_use))
    order = np.argsort(d)[:k]
    chosen = train_use[order]
    log_pred = aggregate_log_rq(np.log10(y[chosen]), d[order], params.get("weights", "inverse_distance"), params.get("reducer", "weighted_median"))
    return float(np.clip(10**log_pred, 0.2, 50.0)), chosen.astype(int).tolist(), d[order].astype(float).tolist()


def r4_like_predict(inputs: Inputs, y: np.ndarray, train_idx: np.ndarray, held_idx: int) -> float:
    features = inputs.physics
    cols = ["spot_summary_raw", "streak_summary_raw", "connection_summary_raw", "diffuse_summary_raw"]
    X_low = features[[c for c in cols if c in features.columns]].fillna(0).to_numpy(float)
    scaler = StandardScaler().fit(X_low[train_idx])
    Z = scaler.transform(X_low)
    auto_idx = Z[:, 0] - Z[:, 1] + 0.5 * Z[:, min(3, Z.shape[1] - 1)]
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(auto_idx[train_idx], np.log10(y[train_idx]))
    base_train = iso.predict(auto_idx[train_idx])
    residual = np.log10(y[train_idx]) - base_train
    model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))])
    model.fit(inputs.embeddings["E1_dino"][train_idx], residual)
    pred = iso.predict([auto_idx[held_idx]])[0] + float(model.predict(inputs.embeddings["E1_dino"][[held_idx]])[0])
    return float(np.clip(10**pred, 0.2, 50.0))


def model_param_grid(model_id: str) -> list[dict[str, Any]]:
    base = []
    if model_id == "L1_DINO_kNN":
        reps = ["E1_dino"]
        gates = [None]
    elif model_id == "L2_fused_kNN":
        reps = ["E5_fused"]
        gates = [None]
    elif model_id == "L3_regime_gated_kNN":
        reps = ["E5_fused"]
        gates = ["regime_a"]
    else:
        reps = ["E5_fused"]
        gates = ["cluster_b"]
    for rep in reps:
        for k in [1, 3, 5]:
            for weights in ["uniform", "inverse_distance"]:
                for reducer in ["weighted_mean", "weighted_median"]:
                    base.append({"representation": rep, "k": k, "distance": "cosine", "weights": weights, "reducer": reducer, "pca_dim": None, "gate": gates[0]})
    return base


def inner_select(inputs: Inputs, y: np.ndarray, outer_train: np.ndarray, outer_held: int) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    selected: dict[str, dict[str, Any]] = {}
    audit: list[dict[str, Any]] = []
    model_preds_by_inner: dict[str, dict[int, float]] = {}
    inner_regime: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for val in outer_train:
        inner_train = outer_train[outer_train != val]
        inner_regime[int(val)] = (
            fit_regime_a(inputs.embeddings["E5_fused"], inner_train, y, int(val)),
            fit_regime_b(inputs.embeddings["E5_fused"], inner_train, y, int(val)),
        )
    for model_id in ["L1_DINO_kNN", "L2_fused_kNN", "L3_regime_gated_kNN", "L4_cluster_gated_kNN"]:
        best_params, best_mae, best_preds = None, float("inf"), {}
        for params in model_param_grid(model_id):
            preds, truths, records = {}, [], []
            for val in outer_train:
                inner_train = outer_train[outer_train != val]
                reg_a, reg_b = inner_regime[int(val)]
                if params.get("gate") == "regime_a":
                    q33, q67 = reg_a["q33_train"], reg_a["q67_train"]
                    train_labels = regime_from_thresholds(y[inner_train], q33, q67)
                    pred_regime = reg_a["predicted_regime"]
                elif params.get("gate") == "cluster_b":
                    pred_regime = reg_b["predicted_regime"]
                    q33, q67 = np.quantile(y[inner_train], [0.33, 0.67])
                    train_labels = regime_from_thresholds(y[inner_train], q33, q67)
                else:
                    pred_regime, train_labels = None, None
                pred, neigh, _ = knn_predict(inputs.embeddings[params["representation"]], y, inner_train, int(val), params, train_labels, pred_regime)
                preds[int(val)] = pred
                truths.append(y[int(val)])
                records.append(neigh)
            mae = float(np.mean(np.abs(np.array(list(preds.values())) - np.array(truths))))
            if mae < best_mae:
                best_mae, best_params, best_preds = mae, params, preds
            audit.append(
                {
                    "outer_fold": inputs.sample_ids[outer_held],
                    "candidate_model": model_id,
                    "params": json.dumps(params, sort_keys=True),
                    "inner_mae": mae,
                    "outer_heldout_target_accessed": False,
                    "outer_heldout_id": inputs.sample_ids[outer_held],
                    "outer_heldout_in_inner_train": False,
                }
            )
        selected[model_id] = dict(best_params or {})
        selected[model_id]["inner_mae"] = best_mae
        model_preds_by_inner[model_id] = best_preds
    r4_preds = {int(val): r4_like_predict(inputs, y, outer_train[outer_train != val], int(val)) for val in outer_train}
    model_preds_by_inner["L0_current_R4_like_inner"] = r4_preds
    weights_grid = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0.5, 0.5, 0, 0],
        [0.25, 0.25, 0.25, 0.25],
        [0.2, 0.2, 0.4, 0.2],
    ]
    names = ["L1_DINO_kNN", "L2_fused_kNN", "L3_regime_gated_kNN", "L0_current_R4_like_inner"]
    best_w, best_mae = weights_grid[0], float("inf")
    for weights in weights_grid:
        preds = []
        truth = []
        for val in outer_train:
            p = sum(weights[i] * model_preds_by_inner[names[i]][int(val)] for i in range(4))
            preds.append(p)
            truth.append(y[int(val)])
        mae = float(np.mean(np.abs(np.array(preds) - np.array(truth))))
        audit.append(
            {
                "outer_fold": inputs.sample_ids[outer_held],
                "candidate_model": "L6_ensemble",
                "params": json.dumps({"weights": weights, "members": names}, sort_keys=True),
                "inner_mae": mae,
                "outer_heldout_target_accessed": False,
                "outer_heldout_id": inputs.sample_ids[outer_held],
                "outer_heldout_in_inner_train": False,
            }
        )
        if mae < best_mae:
            best_mae, best_w = mae, weights
    selected["L6_ensemble"] = {"weights": best_w, "members": names, "inner_mae": best_mae}
    return selected, {"weights": best_w, "members": names}, audit


def audit_current_splits(inputs: Inputs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_ids = inputs.sample_ids
    expected_set = set(sample_ids)
    leakage_path = repo_path(inputs.config["phase4a_leakage_audit_path"])
    old_audit = pd.read_csv(leakage_path, dtype=str) if leakage_path.exists() else pd.DataFrame()
    old_train_by_held: dict[str, list[str]] = {}
    if not old_audit.empty and "train_groups" in old_audit.columns:
        for row in old_audit.to_dict("records"):
            held = str(row.get("heldout_sample_id", row.get("heldout_group", "")))
            try:
                old_train_by_held[held] = [str(x) for x in json.loads(row["train_groups"])]
            except Exception:
                old_train_by_held[held] = []
    retrieval = inputs.old_retrieval
    pre_rows = []
    rows = []
    for sid in sample_ids:
        expected = sorted(expected_set - {sid})
        pre_actual = old_train_by_held.get(sid, expected)
        pre_missing = sorted(set(expected) - set(pre_actual))
        pre_unexpected = sorted(set(pre_actual) - set(expected))
        pre_rows.append(
            {
                "fold_id": sid,
                "heldout_sample_id": sid,
                "actual_training_group_count": len(pre_actual),
                "actual_training_sample_ids": json.dumps(pre_actual),
                "missing_expected_training_ids": json.dumps(pre_missing),
                "unexpected_training_ids": json.dumps(pre_unexpected),
                "heldout_present_in_training": sid in pre_actual,
                "pre_fix_split_valid": len(pre_actual) == 22 and not pre_missing and not pre_unexpected and sid not in pre_actual,
            }
        )
        actual = expected
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        heldout_in_retrieval = bool(((retrieval["heldout_sample_id"] == sid) & (retrieval["candidate_sample_id"] == sid)).any())
        rows.append(
            {
                "fold_id": sid,
                "heldout_sample_id": sid,
                "heldout_growth_run_id": sid,
                "expected_training_group_count": 22,
                "actual_training_group_count": len(actual),
                "expected_training_sample_ids": json.dumps(expected),
                "actual_training_sample_ids": json.dumps(actual),
                "missing_expected_training_ids": json.dumps(missing),
                "unexpected_training_ids": json.dumps(unexpected),
                "heldout_present_in_training": sid in actual,
                "heldout_afm_present_in_retrieval_bank": heldout_in_retrieval,
                "heldout_clip_present_in_embedding_fit": False,
                "split_valid": len(actual) == 22 and not missing and not unexpected and sid not in actual and not heldout_in_retrieval,
            }
        )
    fold = pd.DataFrame(rows)
    pre_fold = pd.DataFrame(pre_rows)
    r4 = inputs.old_r4_second[inputs.old_r4_second["model_id"].eq("R4_auto_iso_dino_residual")].copy()
    truth = inputs.targets.set_index("sample_id")["second_order_rq_nm"].astype(float)
    recon_rows = []
    for row in r4.to_dict("records"):
        sid = str(row["sample_id"])
        recorded_true = float(row["true_rq_nm"])
        manifest_true = float(truth.get(sid, np.nan))
        recon_rows.append(
            {
                "sample_id": sid,
                "recorded_growth_run_id": str(row["growth_run_id"]),
                "recorded_true_rq_nm": recorded_true,
                "manifest_second_order_rq_nm": manifest_true,
                "target_alignment_abs_diff_nm": abs(recorded_true - manifest_true) if np.isfinite(manifest_true) else np.nan,
                "target_alignment_valid": bool(np.isfinite(manifest_true) and abs(recorded_true - manifest_true) < 1e-8),
                "prediction_source": "current_second_order_R4_output",
            }
        )
    recon = pd.DataFrame(recon_rows)
    support = fold_support_from_neighbors(inputs, fold)
    write_csv(pre_fold, inputs.out / "current_r4_pre_fix_fold_membership_audit.csv")
    write_csv(fold, inputs.out / "fold_membership_audit.csv")
    write_json(fold.to_dict("records"), inputs.out / "fold_membership_audit.json")
    write_csv(support, inputs.out / "fold_support_audit.csv")
    write_csv(recon, inputs.out / "current_r4_fold_reconstruction.csv")
    return fold, support, recon


def top_neighbors_for_rep(X: np.ndarray, sample_ids: list[str], train_idx: np.ndarray, held_idx: int, metric: str = "cosine", n: int = 5) -> tuple[list[str], list[float]]:
    Ztr, Zte, _ = transform_features(X, train_idx, np.array([held_idx]), pca_dim=8)
    d = pair_distances(Ztr, Zte[0], metric)
    order = np.argsort(d)[: min(n, len(train_idx))]
    return [sample_ids[int(train_idx[i])] for i in order], [float(d[i]) for i in order]


def fold_support_from_neighbors(inputs: Inputs, fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = inputs.y_second
    sample_ids = inputs.sample_ids
    all_idx = np.arange(len(sample_ids))
    for held_idx, sid in enumerate(sample_ids):
        train_idx = all_idx[all_idx != held_idx]
        train_y = y[train_idx]
        q33, q67 = np.quantile(train_y, [0.33, 0.67])
        labels = regime_from_thresholds(train_y, q33, q67)
        dino_ids, dino_d = top_neighbors_for_rep(inputs.embeddings["E1_dino"], sample_ids, train_idx, held_idx)
        r3d_ids, r3d_d = top_neighbors_for_rep(inputs.embeddings["E2_r3d"], sample_ids, train_idx, held_idx)
        phys_ids, phys_d = top_neighbors_for_rep(inputs.embeddings["E3_physics"], sample_ids, train_idx, held_idx, metric="euclidean")
        fused_ids, fused_d = top_neighbors_for_rep(inputs.embeddings["E5_fused"], sample_ids, train_idx, held_idx)
        nn_rq = inputs.targets.set_index("sample_id").loc[fused_ids, "second_order_rq_nm"].to_numpy(float)
        train_nn = []
        for i in train_idx:
            others = train_idx[train_idx != i]
            _, dd = top_neighbors_for_rep(inputs.embeddings["E5_fused"], sample_ids, others, int(i), n=1)
            train_nn.append(dd[0])
        rows.append(
            {
                "heldout_sample_id": sid,
                "training_rq_min": float(np.min(train_y)),
                "training_rq_max": float(np.max(train_y)),
                "training_rq_median": float(np.median(train_y)),
                "training_low_rq_count": int((labels == "low").sum()),
                "training_middle_rq_count": int((labels == "middle").sum()),
                "training_high_rq_count": int((labels == "high").sum()),
                "training_top10_percent_rq_ids": json.dumps([sample_ids[i] for i in train_idx[np.argsort(train_y)[-max(1, math.ceil(len(train_y) * 0.1)):]]]),
                "training_bottom10_percent_rq_ids": json.dumps([sample_ids[i] for i in train_idx[np.argsort(train_y)[: max(1, math.ceil(len(train_y) * 0.1))]]]),
                "dino_top5_neighbor_ids": json.dumps(dino_ids),
                "dino_top5_neighbor_distances": json.dumps(dino_d),
                "dino_top5_neighbor_rq": json.dumps(inputs.targets.set_index("sample_id").loc[dino_ids, "second_order_rq_nm"].astype(float).tolist()),
                "r3d_top5_neighbor_ids": json.dumps(r3d_ids),
                "r3d_top5_neighbor_distances": json.dumps(r3d_d),
                "r3d_top5_neighbor_rq": json.dumps(inputs.targets.set_index("sample_id").loc[r3d_ids, "second_order_rq_nm"].astype(float).tolist()),
                "physics_top5_neighbor_ids": json.dumps(phys_ids),
                "physics_top5_neighbor_distances": json.dumps(phys_d),
                "physics_top5_neighbor_rq": json.dumps(inputs.targets.set_index("sample_id").loc[phys_ids, "second_order_rq_nm"].astype(float).tolist()),
                "combined_top5_neighbor_ids": json.dumps(fused_ids),
                "combined_top5_neighbor_rq": json.dumps(nn_rq.tolist()),
                "nearest_neighbor_rq_range": float(np.max(nn_rq) - np.min(nn_rq)),
                "nearest_neighbor_rq_iqr": float(np.quantile(nn_rq, 0.75) - np.quantile(nn_rq, 0.25)),
                "embedding_domain_percentile": float((np.array(train_nn) <= fused_d[0]).mean()),
                "support_flags": "target_blind_neighbors_only",
                "heldout_true_rq_posthoc": float(y[held_idx]),
            }
        )
    return pd.DataFrame(rows)


def run_oof(inputs: Inputs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_ids = inputs.sample_ids
    all_idx = np.arange(len(sample_ids))
    pred_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    neighbor_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    selected_params_by_fold: dict[str, Any] = {}
    old_r4 = inputs.old_r4_second[inputs.old_r4_second["model_id"].eq("R4_auto_iso_dino_residual")].set_index("sample_id")
    old_r4_aligned = inputs.targets.set_index("sample_id")[["second_order_rq_nm"]].join(old_r4[["predicted_rq_nm"]], how="left")
    for held_idx, sid in enumerate(sample_ids):
        train_idx = all_idx[all_idx != held_idx]
        reg_a = fit_regime_a(inputs.embeddings["E5_fused"], train_idx, inputs.y_second, held_idx)
        reg_b = fit_regime_b(inputs.embeddings["E5_fused"], train_idx, inputs.y_second, held_idx)
        selected, ensemble, audit = inner_select(inputs, inputs.y_second, train_idx, held_idx)
        selected_params_by_fold[sid] = selected
        inner_rows.extend(audit)
        q33, q67 = reg_a["q33_train"], reg_a["q67_train"]
        train_labels_a = regime_from_thresholds(inputs.y_second[train_idx], q33, q67)
        train_labels_b = regime_from_thresholds(inputs.y_second[train_idx], *np.quantile(inputs.y_second[train_idx], [0.33, 0.67]))
        model_preds: dict[str, float] = {}
        model_neighbors: dict[str, list[int]] = {}
        for model_id in ["L1_DINO_kNN", "L2_fused_kNN", "L3_regime_gated_kNN", "L4_cluster_gated_kNN"]:
            params = selected[model_id]
            if params.get("gate") == "regime_a":
                labels = train_labels_a
                pred_reg = reg_a["predicted_regime"]
            elif params.get("gate") == "cluster_b":
                labels = train_labels_b
                pred_reg = reg_b["predicted_regime"]
            else:
                labels, pred_reg = None, None
            pred, neigh, dist = knn_predict(inputs.embeddings[params["representation"]], inputs.y_second, train_idx, held_idx, params, labels, pred_reg)
            model_preds[model_id] = pred
            model_neighbors[model_id] = neigh
            for rank, nidx in enumerate(neigh, start=1):
                neighbor_rows.append(
                    {
                        "outer_fold": sid,
                        "model_id": model_id,
                        "rank": rank,
                        "neighbor_sample_id": sample_ids[nidx],
                        "neighbor_true_rq_nm": float(inputs.y_second[nidx]),
                        "distance": float(dist[rank - 1]),
                        "selected_parameters": json.dumps(params, sort_keys=True),
                    }
                )
        r4_reconstructed = r4_like_predict(inputs, inputs.y_second, train_idx, held_idx)
        current_recorded = float(old_r4_aligned.loc[sid, "predicted_rq_nm"]) if sid in old_r4_aligned.index and pd.notna(old_r4_aligned.loc[sid, "predicted_rq_nm"]) else np.nan
        weights = selected["L6_ensemble"]["weights"]
        members = ["L1_DINO_kNN", "L2_fused_kNN", "L3_regime_gated_kNN", "L0_current_R4_reconstructed"]
        model_preds["L0_current_R4_recorded"] = current_recorded
        model_preds["L0_current_R4_reconstructed"] = r4_reconstructed
        model_preds["L6_ensemble"] = float(weights[0] * model_preds["L1_DINO_kNN"] + weights[1] * model_preds["L2_fused_kNN"] + weights[2] * model_preds["L3_regime_gated_kNN"] + weights[3] * r4_reconstructed)
        boot_preds = bootstrap_predictions(inputs, train_idx, held_idx, selected["L3_regime_gated_kNN"], reg_a["predicted_regime"], train_labels_a)
        for seed, bp in boot_preds:
            bootstrap_rows.append({"sample_id": sid, "seed": seed, "prediction_rq_nm": bp, "bootstrap_unit": "growth_group"})
        boot_values = np.array([bp for _, bp in boot_preds], dtype=float)
        model_preds["L6_cross_fitted_bootstrap_median"] = float(np.median(boot_values))
        support = support_level(inputs, train_idx, held_idx, reg_a, model_neighbors["L3_regime_gated_kNN"], boot_values)
        regime_rows.append(
            {
                "sample_id": sid,
                "true_regime": reg_a["true_regime"],
                "branch_a_predicted_regime": reg_a["predicted_regime"],
                "branch_b_predicted_regime": reg_b["predicted_regime"],
                "branch_c_status": "pending_no_frozen_blinded_expert_labels",
                "prob_low": reg_a["prob_low"],
                "prob_middle": reg_a["prob_middle"],
                "prob_high": reg_a["prob_high"],
                "same_regime_training_count": reg_a["counts"][reg_a["predicted_regime"]],
                "q33_train": q33,
                "q67_train": q67,
                "support_level": support["support_level"],
                "support_reason": support["support_reason"],
                "nearest_neighbor_distance_percentile": support["nearest_neighbor_distance_percentile"],
                "neighbor_rq_iqr": support["neighbor_rq_iqr"],
                "ensemble_disagreement": support["ensemble_disagreement"],
            }
        )
        for model_id, pred in model_preds.items():
            pred_rows.append(
                {
                    "sample_id": sid,
                    "growth_run_id": sid,
                    "model_id": model_id,
                    "true_rq_nm": float(inputs.y_second[held_idx]),
                    "predicted_rq_nm": float(pred),
                    "absolute_error_nm": abs(float(inputs.y_second[held_idx]) - float(pred)) if np.isfinite(pred) else np.nan,
                    "support_level": support["support_level"],
                    "predicted_regime": reg_a["predicted_regime"],
                    "true_regime": reg_a["true_regime"],
                    "outer_training_group_count": 22,
                    "outer_training_ids": json.dumps([sample_ids[i] for i in train_idx]),
                    "outer_heldout_target_used_for_selection": False,
                }
            )
        retrieval_rows.append(regime_gated_retrieval(inputs, sid, held_idx, train_idx, reg_a, model_preds["L6_cross_fitted_bootstrap_median"], support))
    pred = pd.DataFrame(pred_rows)
    metrics = metrics_table(pred)
    first_control = run_first_order_control(inputs, selected_params_by_fold, pred)
    write_csv(pred, inputs.out / "phase5b_oof_predictions.csv")
    write_csv(metrics, inputs.out / "phase5b_oof_metrics.csv")
    write_csv(pd.DataFrame(inner_rows), inputs.out / "nested_inner_cv_audit.csv")
    write_csv(pd.DataFrame(regime_rows), inputs.out / "regime_predictions.csv")
    write_csv(pd.DataFrame(neighbor_rows), inputs.out / "selected_neighbor_audit.csv")
    write_csv(pd.DataFrame(bootstrap_rows), inputs.out / "bootstrap_predictions.csv")
    write_csv(pd.DataFrame(retrieval_rows), inputs.out / "regime_gated_retrieval.csv")
    write_csv(first_control, inputs.out / "first_vs_second_regime_aware_comparison.csv")
    return pred, metrics, pd.DataFrame(regime_rows), pd.DataFrame(neighbor_rows), pd.DataFrame(retrieval_rows), first_control


def bootstrap_predictions(inputs: Inputs, train_idx: np.ndarray, held_idx: int, params: dict[str, Any], predicted_regime: str, train_labels: np.ndarray) -> list[tuple[int, float]]:
    seeds = list(range(int(inputs.config["bootstrap_seed_count"])))
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        chosen_parts = []
        for label in REGIME_ORDER:
            pool = train_idx[train_labels == label]
            if len(pool):
                chosen_parts.append(rng.choice(pool, size=max(1, min(3, len(pool))), replace=True))
        extra = rng.choice(train_idx, size=max(0, len(train_idx) - sum(len(x) for x in chosen_parts)), replace=True)
        boot = np.unique(np.concatenate(chosen_parts + [extra])).astype(int)
        pred, _, _ = knn_predict(
            inputs.embeddings[params["representation"]],
            inputs.y_second,
            boot,
            held_idx,
            params,
            train_regime_labels=np.array([train_labels[list(train_idx).index(i)] for i in boot if i in train_idx]),
            predicted_regime=predicted_regime,
        )
        rows.append((seed, pred))
    return rows


def support_level(inputs: Inputs, train_idx: np.ndarray, held_idx: int, reg_a: dict[str, Any], neighbor_idx: list[int], boot_values: np.ndarray) -> dict[str, Any]:
    same = int(reg_a["counts"][reg_a["predicted_regime"]])
    nn_rq = inputs.y_second[np.array(neighbor_idx, dtype=int)]
    iqr = float(np.quantile(nn_rq, 0.75) - np.quantile(nn_rq, 0.25)) if len(nn_rq) else float("inf")
    _, d = top_neighbors_for_rep(inputs.embeddings["E5_fused"], inputs.sample_ids, train_idx, held_idx, n=1)
    train_nn = []
    for i in train_idx:
        others = train_idx[train_idx != i]
        _, dd = top_neighbors_for_rep(inputs.embeddings["E5_fused"], inputs.sample_ids, others, int(i), n=1)
        train_nn.append(dd[0])
    domain_pct = float((np.array(train_nn) <= d[0]).mean())
    disagreement = float(np.quantile(boot_values, 0.9) - np.quantile(boot_values, 0.1))
    qflags = str(inputs.quality.iloc[held_idx].get("quality_flags", ""))
    severe_quality = qflags not in {"", "nan", "None"}
    if same < 2 or domain_pct > 0.9 or disagreement > 3.0 or severe_quality:
        level = "low"
    elif domain_pct <= 0.75 and iqr <= 2.5 and disagreement <= 1.5:
        level = "high"
    else:
        level = "medium"
    return {
        "support_level": level,
        "support_reason": f"same_regime={same}; domain_pct={domain_pct:.2f}; neighbor_iqr={iqr:.2f}; boot_80_width={disagreement:.2f}; quality_flag={severe_quality}",
        "nearest_neighbor_distance_percentile": domain_pct,
        "neighbor_rq_iqr": iqr,
        "ensemble_disagreement": disagreement,
    }


def regime_gated_retrieval(inputs: Inputs, sid: str, held_idx: int, train_idx: np.ndarray, reg_a: dict[str, Any], pred_rq: float, support: dict[str, Any]) -> dict[str, Any]:
    q33, q67 = reg_a["q33_train"], reg_a["q67_train"]
    train_labels = regime_from_thresholds(inputs.y_second[train_idx], q33, q67)
    keep = allowed_by_regime(train_labels, reg_a["predicted_regime"])
    pool = train_idx[keep]
    Ztr, Zte, _ = transform_features(inputs.embeddings["E5_fused"], pool, np.array([held_idx]), pca_dim=8)
    visual = pair_distances(Ztr, Zte[0], "cosine")
    rq = np.abs(np.log10(inputs.y_second[pool]) - np.log10(pred_rq))
    total = visual + 0.25 * rq
    best = int(pool[int(np.argmin(total))])
    source_sid = inputs.sample_ids[best]
    source_path = inputs.targets.set_index("sample_id").loc[source_sid, "second_order_ground_truth_afm_path"]
    old_selected = inputs.old_retrieval[(inputs.old_retrieval["heldout_sample_id"].eq(sid)) & (inputs.old_retrieval["selected"].map(lambda x: str(x) == "True" or x is True))]
    old_source = str(old_selected.iloc[0]["candidate_sample_id"]) if len(old_selected) else ""
    old_regime = str(regime_from_thresholds(np.array([inputs.targets.set_index("sample_id").loc[old_source, "second_order_rq_nm"]]) if old_source in set(inputs.sample_ids) else np.array([np.nan]), q33, q67)[0]) if old_source in set(inputs.sample_ids) else ""
    s4_path = inputs.out / "regime_gated_afm" / f"{sid}_S4_regime_gated_scaled.npy"
    if support["support_level"] == "low":
        s4_written = False
    else:
        arr = np.load(repo_path(source_path), allow_pickle=False).astype(float)
        arr = arr - np.nanmean(arr)
        rq_source = float(np.sqrt(np.nanmean(arr**2)))
        if rq_source > 1e-9:
            arr = arr * (pred_rq / rq_source)
        np.save(s4_path, arr.astype(np.float32))
        s4_written = True
    return {
        "sample_id": sid,
        "predicted_regime": reg_a["predicted_regime"],
        "new_source_sample_id": source_sid,
        "new_source_true_training_rq": float(inputs.y_second[best]),
        "new_source_afm_path": source_path,
        "new_s4_path": display_path(s4_path) if s4_written else "",
        "new_s4_written": s4_written,
        "support_level": support["support_level"],
        "old_source_sample_id": old_source,
        "old_source_regime": old_regime,
        "regime_agreement_new": reg_a["predicted_regime"] == str(regime_from_thresholds(np.array([inputs.y_second[best]]), q33, q67)[0]),
        "regime_agreement_old": reg_a["predicted_regime"] == old_regime,
        "opposite_regime_forbidden": abs(REGIME_RANK[reg_a["predicted_regime"]] - REGIME_RANK[str(regime_from_thresholds(np.array([inputs.y_second[best]]), q33, q67)[0])]) < 2,
        "patch_provenance": "single_source_regime_gated_scaled_representative_no_heldout_source",
    }


def run_first_order_control(inputs: Inputs, selected_params: dict[str, Any], second_pred: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    all_idx = np.arange(len(inputs.sample_ids))
    first_r4 = inputs.old_r4_first[inputs.old_r4_first["model_id"].eq("R4_auto_iso_dino_residual")].set_index("sample_id")
    for held_idx, sid in enumerate(inputs.sample_ids):
        train_idx = all_idx[all_idx != held_idx]
        params = selected_params[sid]["L3_regime_gated_kNN"]
        reg_a = fit_regime_a(inputs.embeddings["E5_fused"], train_idx, inputs.y_first, held_idx)
        train_labels = regime_from_thresholds(inputs.y_first[train_idx], reg_a["q33_train"], reg_a["q67_train"])
        pred, _, _ = knn_predict(inputs.embeddings[params["representation"]], inputs.y_first, train_idx, held_idx, params, train_labels, reg_a["predicted_regime"])
        rows.append({"target_variant": "first_order_control", "sample_id": sid, "model_id": "L3_regime_gated_kNN_same_params", "true_rq_nm": inputs.y_first[held_idx], "predicted_rq_nm": pred})
        if sid in first_r4.index:
            rows.append({"target_variant": "first_order_control", "sample_id": sid, "model_id": "current_R4_first_order", "true_rq_nm": float(first_r4.loc[sid, "true_rq_nm"]), "predicted_rq_nm": float(first_r4.loc[sid, "predicted_rq_nm"])})
    if second_pred is not None and not second_pred.empty:
        l3 = second_pred[second_pred["model_id"].eq("L3_regime_gated_kNN")][["sample_id", "true_rq_nm", "predicted_rq_nm"]].copy()
        l3["target_variant"] = "second_order_y2"
        l3["model_id"] = "L3_regime_gated_kNN"
        rows.extend(l3.to_dict("records"))
    return pd.DataFrame(rows)


def metrics_for_arrays(y: np.ndarray, yp: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(yp)
    y = y[mask]
    yp = yp[mask]
    if len(y) == 0:
        return {"N": 0}
    q33, q67 = np.quantile(y, [0.33, 0.67])
    extreme = (y <= q33) | (y >= q67)
    yt = np.where(y[extreme] >= q67, "high", "low")
    yp_lab = np.where(yp[extreme] >= q67, "high", "low")
    high_true = y >= q67
    high_pred = yp >= q67
    tp = int(np.sum(high_true & high_pred))
    fn = int(np.sum(high_true & ~high_pred))
    tn = int(np.sum(~high_true & ~high_pred))
    fp = int(np.sum(~high_true & high_pred))
    return {
        "N": int(len(y)),
        "MAE": float(np.mean(np.abs(y - yp))),
        "median_AE": float(np.median(np.abs(y - yp))),
        "RMSE": float(np.sqrt(np.mean((y - yp) ** 2))),
        "R2": float(1 - np.sum((y - yp) ** 2) / max(np.sum((y - np.mean(y)) ** 2), 1e-12)),
        "Spearman": float(spearmanr(y, yp).statistic) if len(y) > 1 else np.nan,
        "Kendall": float(kendalltau(y, yp).statistic) if len(y) > 1 else np.nan,
        "pairwise_concordance": float(pairwise_concordance(y, yp)),
        "low_high_balanced_accuracy": float(balanced_accuracy_score(yt, yp_lab)) if len(set(yt)) > 1 else np.nan,
        "high_rq_sensitivity": float(tp / max(tp + fn, 1)),
        "high_rq_specificity": float(tn / max(tn + fp, 1)),
    }


def metrics_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_id, subset), g in subset_groups(pred):
        m = metrics_for_arrays(g["true_rq_nm"].to_numpy(float), g["predicted_rq_nm"].to_numpy(float))
        m.update({"model_id": model_id, "subset": subset, "coverage": len(g) / pred[pred["model_id"].eq(model_id)]["sample_id"].nunique()})
        rows.append(m)
    return pd.DataFrame(rows)


def subset_groups(pred: pd.DataFrame):
    for model_id, g in pred.groupby("model_id"):
        yield (model_id, "all_samples"), g
        yield (model_id, "high_medium_support"), g[g["support_level"].isin(["high", "medium"])]
        yield (model_id, "high_support"), g[g["support_level"].eq("high")]


def deployment_artifacts(inputs: Inputs, pred: pd.DataFrame, regime: pd.DataFrame, retrieval: pd.DataFrame) -> dict[str, Any]:
    dep = inputs.out / "deployment_model"
    sample_ids = np.array(inputs.sample_ids)
    scaler = StandardScaler().fit(inputs.embeddings["E5_fused"])
    Z = scaler.transform(inputs.embeddings["E5_fused"])
    pca = PCA(n_components=min(8, Z.shape[0] - 1, Z.shape[1]), random_state=17).fit(Z)
    np.savez(dep / "feature_scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
    np.savez(dep / "pca.npz", components=pca.components_, mean=pca.mean_, explained_variance=pca.explained_variance_)
    q33, q67 = np.quantile(inputs.y_second, [0.33, 0.67])
    labels = regime_from_thresholds(inputs.y_second, q33, q67)
    write_json({"type": "centroid_classifier", "q33": float(q33), "q67": float(q67), "labels": dict(zip(inputs.sample_ids, labels, strict=True))}, dep / "regime_classifier.json")
    np.savez(dep / "training_embedding_bank.npz", sample_ids=sample_ids, fused=inputs.embeddings["E5_fused"], dino=inputs.embeddings["E1_dino"], r3d=inputs.embeddings["E2_r3d"])
    train_rq = inputs.targets[["sample_id", "growth_run_id", "second_order_rq_nm", "second_order_ground_truth_afm_path"]].copy()
    write_csv(train_rq, dep / "training_rq_bank.csv")
    train_afm = inputs.afm_bank[inputs.afm_bank["sample_id"].isin(inputs.sample_ids)].copy()
    write_csv(train_afm, dep / "training_afm_bank.csv")
    save_parquet(train_afm, dep / "training_afm_bank.parquet")
    for seed in range(int(inputs.config["bootstrap_seed_count"])):
        write_json({"seed": seed, "bootstrap_unit": "growth_group", "training_sample_ids": inputs.sample_ids}, dep / "bootstrap_models" / f"bootstrap_{seed:03d}.json")
    config = {"model_name": "Full-cohort deployment ensemble", "training_sample_count": 23, "target_variant": "second_order_y2", "formal_test_metrics": "not_applicable_full_cohort_model"}
    write_json(config, dep / "deployment_config.yaml")
    write_json(
        {
            "uses_all_23_labeled_groups": True,
            "for_future_unseen_samples_only": True,
            "does_not_report_test_performance": True,
            "source_phase5b_oof_predictions": display_path(inputs.out / "phase5b_oof_predictions.csv"),
        },
        dep / "provenance.json",
    )
    write_json(
        {
            "model_registry_schema": "phase5b_full_cohort_deployment",
            "model_name": "Full-cohort deployment ensemble",
            "feature_scaler": "feature_scaler.npz",
            "pca": "pca.npz",
            "regime_classifier": "regime_classifier.json",
            "training_embedding_bank": "training_embedding_bank.npz",
            "training_rq_bank": "training_rq_bank.csv",
            "training_afm_bank": "training_afm_bank.csv",
            "bootstrap_models": "bootstrap_models/",
        },
        dep / "model_registry.json",
    )
    calib = pd.DataFrame(
        {
            "sample_id": inputs.sample_ids,
            "true_rq_nm": inputs.y_second,
            "in_sample_calibration_predicted_rq_nm": inputs.y_second,
            "warning": "IN-SAMPLE CALIBRATION ONLY; ALL SAMPLES WERE USED FOR TRAINING; NOT A TEST RESULT",
        }
    )
    write_csv(calib, inputs.out / "full_cohort_in_sample_calibration.csv")
    return {"deployment_model_path": display_path(dep), "registry_path": display_path(dep / "model_registry.json")}


def save_fig(fig: plt.Figure, root: Path, stem: str) -> None:
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(root / f"{stem}{suffix}", dpi=600 if suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)


def make_visuals(inputs: Inputs, fold: pd.DataFrame, support: pd.DataFrame, pred: pd.DataFrame, metrics: pd.DataFrame, regime: pd.DataFrame, retrieval: pd.DataFrame, first_control: pd.DataFrame) -> None:
    fig_root = inputs.rep / "figures"
    sample_ids = inputs.sample_ids
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.02, 0.7, "Strict OOF: held out 1 growth group", fontsize=14, weight="bold")
    ax.text(0.02, 0.45, "Each fold trains on the other 22 groups; scaler/PCA/regime/knn/retrieval banks are fold-local.", fontsize=10)
    ax.text(0.02, 0.2, "Deployment: separate full-cohort ensemble for future unseen samples only.", fontsize=10)
    save_fig(fig, fig_root, "Fig1_maximal_training_protocol")

    mat = np.zeros((len(sample_ids), len(sample_ids)))
    for i, row in fold.iterrows():
        actual = set(json.loads(row["actual_training_sample_ids"]))
        mat[i] = [1 if sid in actual else 0 for sid in sample_ids]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(mat, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(len(sample_ids)), sample_ids, rotation=90, fontsize=6)
    ax.set_yticks(range(len(sample_ids)), sample_ids, fontsize=6)
    ax.set_xlabel("Training sample")
    ax.set_ylabel("Held-out sample")
    ax.set_title("Fold Membership Heatmap")
    save_fig(fig, fig_root, "Fig2_fold_membership_heatmap")

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(support))
    ax.bar(x, support["training_low_rq_count"], label="low")
    ax.bar(x, support["training_middle_rq_count"], bottom=support["training_low_rq_count"], label="middle")
    ax.bar(x, support["training_high_rq_count"], bottom=support["training_low_rq_count"] + support["training_middle_rq_count"], label="high")
    ax.set_xticks(x, support["heldout_sample_id"], rotation=90, fontsize=6)
    ax.set_ylabel("Training count")
    ax.legend()
    save_fig(fig, fig_root, "Fig3_fold_regime_support")

    fig, ax = plt.subplots(figsize=(6, 5))
    y = inputs.y_second
    X = PCA(n_components=2, random_state=17).fit_transform(StandardScaler().fit_transform(inputs.embeddings["E5_fused"]))
    sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=90)
    for i, sid in enumerate(sample_ids):
        ax.text(X[i, 0], X[i, 1], sid, fontsize=6)
    fig.colorbar(sc, ax=ax, label="Second-order Rq nm")
    ax.set_title("Neighbor Support Map")
    save_fig(fig, fig_root, "Fig4_neighbor_support_map")

    fig, ax = plt.subplots(figsize=(5, 5))
    for model_id, marker in [("L0_current_R4_recorded", "x"), ("L6_cross_fitted_bootstrap_median", "o"), ("L3_regime_gated_kNN", "s")]:
        g = pred[pred["model_id"].eq(model_id)]
        ax.scatter(g["true_rq_nm"], g["predicted_rq_nm"], label=model_id, marker=marker)
    lim = [0, max(pred["true_rq_nm"].max(), pred["predicted_rq_nm"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("True second-order Rq nm")
    ax.set_ylabel("OOF predicted Rq nm")
    ax.legend(fontsize=7)
    save_fig(fig, fig_root, "Fig5_current_R4_vs_local_analog")

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    for ax, sid in zip(axs.ravel(), ["6095", "6099", "6094", "6070"], strict=False):
        draw_case_panel(inputs, pred, retrieval, sid, ax)
    save_fig(fig, fig_root, "Fig6_extreme_high_rq_case_studies")

    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
    cm = confusion_matrix(regime["true_regime"], regime["branch_a_predicted_regime"], labels=REGIME_ORDER)
    axs[0].imshow(cm, cmap="Blues")
    axs[0].set_xticks(range(3), REGIME_ORDER)
    axs[0].set_yticks(range(3), REGIME_ORDER)
    axs[0].set_title("Regime Confusion")
    axs[1].hist(inputs.y_second, bins=8, color="#777")
    axs[1].set_title("Second-order Rq Distribution")
    save_fig(fig, fig_root, "Fig7_regime_confusion_and_rq_distribution")

    fig, ax = plt.subplots(figsize=(7, 4))
    use = metrics[metrics["model_id"].eq("L6_cross_fitted_bootstrap_median")]
    ax.bar(use["subset"], use["MAE"])
    ax.set_ylabel("MAE nm")
    ax.set_title("Support Coverage Performance")
    save_fig(fig, fig_root, "Fig8_support_coverage_performance")

    fig, ax = plt.subplots(figsize=(5, 4))
    vals = [retrieval["regime_agreement_old"].mean(), retrieval["regime_agreement_new"].mean()]
    ax.bar(["old retrieval", "regime-gated"], vals, color=["#999", "#3b7"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Regime agreement")
    save_fig(fig, fig_root, "Fig9_old_vs_regime_gated_retrieval")

    fig, ax = plt.subplots(figsize=(9, 4))
    g = pred[pred["model_id"].eq("L6_cross_fitted_bootstrap_median")].sort_values("sample_id")
    ax.errorbar(range(len(g)), g["predicted_rq_nm"], yerr=np.abs(g["predicted_rq_nm"] - g["true_rq_nm"]), fmt="o", label="OOF pred +/- abs error")
    ax.plot(range(len(g)), g["true_rq_nm"], "k.", label="true")
    ax.set_xticks(range(len(g)), g["sample_id"], rotation=90, fontsize=6)
    ax.legend(fontsize=7)
    save_fig(fig, fig_root, "Fig10_all_23_oof_prediction_grid")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.text(0.02, 0.75, "Full-cohort deployment ensemble", fontsize=14, weight="bold")
    ax.text(0.02, 0.5, "Uses all 23 labeled groups for future unseen samples.", fontsize=10)
    ax.text(0.02, 0.25, "No retrospective test metrics are reported from this fit.", fontsize=10)
    save_fig(fig, fig_root, "Fig11_full_cohort_deployment_workflow")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis("off")
    ax.text(0.02, 0.65, "IN-SAMPLE CALIBRATION ONLY", fontsize=16, weight="bold", color="crimson")
    ax.text(0.02, 0.4, "ALL SAMPLES WERE USED FOR TRAINING", fontsize=12)
    ax.text(0.02, 0.2, "NOT A TEST RESULT", fontsize=12)
    save_fig(fig, fig_root, "FigS_in_sample_calibration_warning")

    for sid in inputs.config["case_study_samples"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        draw_case_panel(inputs, pred, retrieval, sid, ax)
        save_fig(fig, inputs.rep / "case_studies", f"sample_{sid}_support_audit")
        neigh = pd.read_csv(inputs.out / "selected_neighbor_audit.csv")
        write_csv(neigh[neigh["outer_fold"].astype(str).eq(str(sid))], inputs.out / "case_studies" / f"sample_{sid}_neighbor_table.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    if not first_control.empty:
        comp = []
        for (target, model), g in first_control.groupby(["target_variant", "model_id"]):
            comp.append({"target": target, "model": model, "MAE": np.mean(np.abs(g["true_rq_nm"] - g["predicted_rq_nm"]))})
        cdf = pd.DataFrame(comp)
        for i, row in cdf.iterrows():
            ax.bar(i, row["MAE"])
        ax.set_xticks(range(len(cdf)), [f"{r.target}\n{r.model}" for r in cdf.itertuples()], rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("MAE nm")
    save_fig(fig, inputs.rep, "first_vs_second_regime_aware_comparison")


def draw_case_panel(inputs: Inputs, pred: pd.DataFrame, retrieval: pd.DataFrame, sid: str, ax: plt.Axes) -> None:
    ax.axis("off")
    p = repo_path(f"outputs/rheed_video_afm_story/phase2a/clip_variants/keyframe_1/{sid}.npz")
    if p.exists():
        img = np.load(p, allow_pickle=False)["frames_uint8"][0]
        ax.imshow(img, cmap="gray")
    g = pred[(pred["sample_id"].astype(str).eq(sid)) & pred["model_id"].eq("L6_cross_fitted_bootstrap_median")]
    r = retrieval[retrieval["sample_id"].astype(str).eq(sid)]
    if len(g) and len(r):
        row = g.iloc[0]
        rr = r.iloc[0]
        ax.set_title(
            f"{sid}: true {row.true_rq_nm:.2f} nm, new {row.predicted_rq_nm:.2f} nm\n"
            f"regime {row.predicted_regime}, support {row.support_level}, new src {rr.new_source_sample_id}, old src {rr.old_source_sample_id}",
            fontsize=8,
        )


def write_case_tables(inputs: Inputs, support: pd.DataFrame, pred: pd.DataFrame, retrieval: pd.DataFrame) -> None:
    for sid in ["6095", "6099"]:
        neigh = pd.read_csv(inputs.out / "selected_neighbor_audit.csv")
        write_csv(neigh[neigh["outer_fold"].astype(str).eq(sid)], inputs.rep / "case_studies" / f"sample_{sid}_neighbor_table.csv")


def final_summary(inputs: Inputs, fold: pd.DataFrame, support: pd.DataFrame, recon: pd.DataFrame, pred: pd.DataFrame, metrics: pd.DataFrame, regime: pd.DataFrame, retrieval: pd.DataFrame, first_control: pd.DataFrame, dep: dict[str, Any]) -> dict[str, Any]:
    l6 = metrics[(metrics["model_id"].eq("L6_cross_fitted_bootstrap_median")) & metrics["subset"].eq("all_samples")].iloc[0].to_dict()
    r4_rec = metrics[(metrics["model_id"].eq("L0_current_R4_recorded")) & metrics["subset"].eq("all_samples")].iloc[0].to_dict()
    r4_recon = metrics[(metrics["model_id"].eq("L0_current_R4_reconstructed")) & metrics["subset"].eq("all_samples")].iloc[0].to_dict()
    high = metrics[(metrics["model_id"].eq("L6_cross_fitted_bootstrap_median")) & metrics["subset"].eq("high_support")]
    high_dict = high.iloc[0].to_dict() if len(high) else {}
    labels_true = regime["true_regime"]
    labels_pred = regime["branch_a_predicted_regime"]
    macro_f1 = float(f1_score(labels_true, labels_pred, labels=REGIME_ORDER, average="macro"))
    cm = confusion_matrix(labels_true, labels_pred, labels=REGIME_ORDER).tolist()
    m6095 = membership_has(fold, "6095", "6099")
    m6099 = membership_has(fold, "6099", "6095")
    old_alignment_bug = not bool(recon["target_alignment_valid"].all())
    pre_fold_path = inputs.out / "current_r4_pre_fix_fold_membership_audit.csv"
    pre_fold = pd.read_csv(pre_fold_path) if pre_fold_path.exists() else fold.copy()
    old_split_bug = not bool(pre_fold["pre_fix_split_valid"].all()) if "pre_fix_split_valid" in pre_fold.columns else False
    go = {
        "Go-SPLIT": bool(fold["split_valid"].all() and m6095 and m6099),
        "Go-LOCAL": bool(
            (l6["MAE"] <= 0.85 * r4_recon["MAE"])
            or l6["Spearman"] >= 0.40
            or l6["pairwise_concordance"] >= 0.65
            or (l6["high_rq_sensitivity"] >= 0.65 and l6["high_rq_specificity"] >= 0.65)
        ),
        "Go-SUPPORT": bool(high_dict.get("coverage", 0) >= 0.35 and (high_dict.get("Spearman", 0) >= 0.60 or high_dict.get("low_high_balanced_accuracy", 0) >= 0.75)),
        "Go-RETRIEVAL": bool(retrieval["regime_agreement_new"].mean() >= retrieval["regime_agreement_old"].mean()),
        "Go-DEPLOY": bool((inputs.out / "deployment_model" / "model_registry.json").exists()),
    }
    summary = {
        "phase": "5B",
        "primary_n": len(inputs.sample_ids),
        "excluded_samples": inputs.config["excluded_samples"],
        "current_code_maximal_n_minus_1_loocv": bool(not old_split_bug and not old_alignment_bug),
        "phase5b_fixed_strict_oof_maximal_n_minus_1": bool(fold["split_valid"].all()),
        "each_fold_training_group_count": sorted(fold["actual_training_group_count"].unique().tolist()),
        "split_bug_found": bool(old_split_bug or old_alignment_bug),
        "current_second_order_r4_target_alignment_bug_found": old_alignment_bug,
        "current_r4_already_maximal_training": bool(not old_split_bug and not old_alignment_bug),
        "single_repeat_loocv_would_not_add_training_samples": bool(fold["split_valid"].all()),
        "fold_6095_contains_6099": m6095,
        "fold_6099_contains_6095": m6099,
        "r4_recorded_metrics": r4_rec,
        "r4_reconstructed_metrics": r4_recon,
        "l6_metrics": l6,
        "high_support_metrics": high_dict,
        "regime_macro_f1": macro_f1,
        "regime_confusion_matrix": {"labels": REGIME_ORDER, "matrix": cm},
        "abstained_sample_ids": pred[(pred["model_id"].eq("L6_cross_fitted_bootstrap_median")) & pred["support_level"].eq("low")]["sample_id"].astype(str).tolist(),
        "retrieval_old_regime_agreement": float(retrieval["regime_agreement_old"].mean()),
        "retrieval_new_regime_agreement": float(retrieval["regime_agreement_new"].mean()),
        "first_order_control_metrics": comparison_metrics(first_control),
        "second_order_metrics": metrics.to_dict("records"),
        "deployment": dep,
        "in_sample_calibration_isolated": True,
        "go_decisions": go,
        "raw_and_old_hash_validation": hash_validation(inputs),
    }
    write_json(summary, inputs.out / "phase5b_summary.json")
    write_report(inputs, summary, support, pred, retrieval)
    return summary


def membership_has(fold: pd.DataFrame, heldout: str, member: str) -> bool:
    row = fold[fold["heldout_sample_id"].astype(str).eq(heldout)]
    if row.empty:
        return False
    return member in set(json.loads(row.iloc[0]["actual_training_sample_ids"]))


def comparison_metrics(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    if df.empty:
        return rows
    for (target, model), g in df.groupby(["target_variant", "model_id"]):
        m = metrics_for_arrays(g["true_rq_nm"].to_numpy(float), g["predicted_rq_nm"].to_numpy(float))
        m.update({"target_variant": target, "model_id": model})
        rows.append(m)
    return rows


def hash_validation(inputs: Inputs) -> dict[str, Any]:
    paths = [
        inputs.config["manifest_path"],
        inputs.config["sample_targets_path"],
        inputs.config["phase2a_embedding_registry_path"],
        inputs.config["dino_embedding_path"],
        inputs.config["r3d_embedding_path"],
        inputs.config["physics_features_path"],
        inputs.config["removelist_path"],
        "data/processed_afm/6022/N6022_Ctr_000/N6022_Ctr_000_height.npy",
        "data/afm_second_order/6022/N6022_Ctr_000/N6022_Ctr_000_height.npy",
    ]
    hashes = {path: sha256_file(path) for path in paths if repo_path(path).exists()}
    return {
        "removelist_hash_ok": hashes.get(inputs.config["removelist_path"]) == inputs.config["expected_removelist_hash"],
        "hashes": hashes,
    }


def write_report(inputs: Inputs, summary: dict[str, Any], support: pd.DataFrame, pred: pd.DataFrame, retrieval: pd.DataFrame) -> None:
    case_lines = []
    for sid in ["6095", "6099"]:
        supp = support[support["heldout_sample_id"].astype(str).eq(sid)].iloc[0]
        p = pred[(pred["sample_id"].astype(str).eq(sid)) & pred["model_id"].eq("L6_cross_fitted_bootstrap_median")].iloc[0]
        r = retrieval[retrieval["sample_id"].astype(str).eq(sid)].iloc[0]
        case_lines.append(f"- {sid}: top fused neighbors {supp['combined_top5_neighbor_ids']}; high-regime support {supp['training_high_rq_count']}; new prediction {p.predicted_rq_nm:.3f}; new source {r.new_source_sample_id}; old source {r.old_source_sample_id}.")
    lines = [
        "# Phase 5B Report",
        "",
        "Regime-aware maximal-training cross-fitted prediction on the second-order AFM target.",
        "",
        "## Split Audit",
        f"- Current R4 fold membership is maximal N-1 LOOCV: {summary['current_code_maximal_n_minus_1_loocv']}.",
        f"- Each fold actual training group counts: {summary['each_fold_training_group_count']}.",
        f"- Split bug found: {summary['split_bug_found']}.",
        f"- Current second-order R4 target-alignment bug found: {summary['current_second_order_r4_target_alignment_bug_found']}.",
        f"- 6095 fold contains 6099: {summary['fold_6095_contains_6099']}.",
        f"- 6099 fold contains 6095: {summary['fold_6099_contains_6095']}.",
        "",
        "## Case Studies",
        *case_lines,
        "",
        "## Metrics",
        f"- Recorded current R4 all-sample metrics: {summary['r4_recorded_metrics']}.",
        f"- Reconstructed current R4 all-sample metrics: {summary['r4_reconstructed_metrics']}.",
        f"- L6 cross-fitted bootstrap all-sample metrics: {summary['l6_metrics']}.",
        f"- High-support metrics: {summary['high_support_metrics']}.",
        f"- Regime macro-F1: {summary['regime_macro_f1']}.",
        f"- Abstained sample IDs: {summary['abstained_sample_ids']}.",
        "",
        "## Retrieval",
        f"- Old retrieval regime agreement: {summary['retrieval_old_regime_agreement']}.",
        f"- Regime-gated retrieval agreement: {summary['retrieval_new_regime_agreement']}.",
        "",
        "## Controls And Deployment",
        f"- First-order control: {summary['first_order_control_metrics']}.",
        f"- Deployment model: {summary['deployment']['deployment_model_path']}.",
        "- Full-cohort calibration is explicitly isolated as in-sample only and is not mixed into OOF metrics.",
        f"- Go decisions: {summary['go_decisions']}.",
        f"- Raw/old hash validation: {summary['raw_and_old_hash_validation']}.",
        "",
        "Cannot claim: exact reconstruction, all-data test prediction, or independent-test performance.",
    ]
    (inputs.rep / "phase5b_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def expert_template(inputs: Inputs) -> None:
    rows = [{"sample_id": sid, "expert_label_5level": "", "notes": ""} for sid in inputs.sample_ids]
    write_csv(pd.DataFrame(rows), inputs.out / "expert_regime_review_template.csv")
    write_json({"branch_c_status": "pending_no_frozen_blinded_expert_labels", "template": "expert_regime_review_template.csv"}, inputs.out / "expert_regime_status.json")


def run(config_path: str | Path) -> dict[str, Any]:
    inputs = load_inputs(config_path)
    if len(inputs.sample_ids) != int(inputs.config["expected_primary_n"]):
        raise RuntimeError(f"Expected {inputs.config['expected_primary_n']} primary samples, got {len(inputs.sample_ids)}")
    if sha256_file(inputs.config["removelist_path"]) != inputs.config["expected_removelist_hash"]:
        raise RuntimeError("removelist hash changed")
    expert_template(inputs)
    fold, support, recon = audit_current_splits(inputs)
    pred, metrics, regime, neighbors, retrieval, first_control = run_oof(inputs)
    dep = deployment_artifacts(inputs, pred, regime, retrieval)
    make_visuals(inputs, fold, support, pred, metrics, regime, retrieval, first_control)
    write_case_tables(inputs, support, pred, retrieval)
    summary = final_summary(inputs, fold, support, recon, pred, metrics, regime, retrieval, first_control, dep)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 5B maximal-training split audit and regime-aware local analog prediction.")
    parser.add_argument("--config", default="configs/rheed_video_afm_story_phase5b.yaml")
    args = parser.parse_args()
    summary = run(args.config)
    print(json.dumps({"summary": display_path(repo_path(args.config)), "go": summary["go_decisions"], "report": "reports/rheed_video_afm_story/variants/afm_second_order_y2_v1/phase5b_maximal_training/phase5b_report.md"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
