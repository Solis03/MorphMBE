from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .baseline_rq import pairwise_concordance
from .common import repo_path, write_csv
from .embedding_regression import load_embedding


def regression_metrics(pred: pd.DataFrame, model_col: str = "model_id") -> pd.DataFrame:
    rows = []
    for model_id, g in pred.groupby(model_col):
        y = g["true_rq_nm"].to_numpy(float)
        yp = g["predicted_rq_nm"].to_numpy(float)
        rows.append(
            {
                "model_id": model_id,
                "N": len(g),
                "MAE": float(mean_absolute_error(y, yp)),
                "median_AE": float(np.median(np.abs(y - yp))),
                "RMSE": float(np.sqrt(np.mean((y - yp) ** 2))),
                "R2": float(r2_score(y, yp)) if len(g) > 2 else np.nan,
                "Spearman": float(spearmanr(y, yp).statistic),
                "Kendall_tau": float(kendalltau(y, yp).statistic),
                "pairwise_concordance": pairwise_concordance(y, yp),
                "low_high_balanced_accuracy": low_high_balanced_accuracy(y, yp),
                "high_rq_sensitivity": high_specificity_sensitivity(y, yp)[0],
                "high_rq_specificity": high_specificity_sensitivity(y, yp)[1],
            }
        )
    return pd.DataFrame(rows).sort_values("MAE")


def low_high_balanced_accuracy(y: np.ndarray, yp: np.ndarray) -> float:
    q33, q67 = np.quantile(y, [0.33, 0.67])
    mask = (y <= q33) | (y >= q67)
    if mask.sum() < 4:
        return np.nan
    yt = np.where(y[mask] >= q67, "high", "low")
    yp_lab = np.where(yp[mask] >= q67, "high", "low")
    return float(balanced_accuracy_score(yt, yp_lab))


def high_specificity_sensitivity(y: np.ndarray, yp: np.ndarray) -> tuple[float, float]:
    thr = np.quantile(y, 0.67)
    yt = y >= thr
    pred = yp >= thr
    tp = np.sum(yt & pred)
    fn = np.sum(yt & ~pred)
    tn = np.sum(~yt & ~pred)
    fp = np.sum(~yt & pred)
    return float(tp / max(tp + fn, 1)), float(tn / max(tn + fp, 1))


def load_dino_embeddings(config: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    registry = pd.read_csv(repo_path(config["phase2a_embedding_registry_path"]))
    row = registry[registry["embedding_id"] == config["dino_embedding_id"]].iloc[0]
    ids, X = load_embedding(row["path"])
    return [str(x) for x in ids], X.astype(float)


def automatic_index_oof(features: pd.DataFrame, groups: np.ndarray, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    summary_cols = ["spot_summary_raw", "streak_summary_raw", "connection_summary_raw", "diffuse_summary_raw"]
    X = features[summary_cols].to_numpy(float)
    for held in np.unique(groups):
        tr = groups != held
        te = groups == held
        scaler = StandardScaler().fit(X[tr])
        Z = scaler.transform(X)
        idx = Z[:, 0] - Z[:, 1] + 0.5 * Z[:, 3]
        for i in np.where(te)[0]:
            rows.append(
                {
                    "sample_id": str(features.iloc[i]["sample_id"]),
                    "growth_run_id": str(features.iloc[i]["growth_run_id"]),
                    "spot_summary": float(Z[i, 0]),
                    "streak_summary": float(Z[i, 1]),
                    "connection_summary": float(Z[i, 2]),
                    "diffuse_summary": float(Z[i, 3]),
                    "automatic_spot_streak_index": float(idx[i]),
                    "scaler_fit_scope": "outer_training_samples_only",
                }
            )
    return pd.DataFrame(rows)


def run_rq_models(manifest: pd.DataFrame, features: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_root = repo_path(config["output_root"])
    samples = manifest["sample_id"].astype(str).to_numpy()
    groups = manifest["growth_run_id"].astype(str).to_numpy()
    y = np.log10(manifest["primary_rq_nm_median"].to_numpy(float))
    y_rq = manifest["primary_rq_nm_median"].to_numpy(float)
    features = features.set_index("sample_id").loc[samples].reset_index()
    idx_oof = automatic_index_oof(features, groups, config).set_index("sample_id").loc[samples].reset_index()
    ids, dino = load_dino_embeddings(config)
    X_dino = dino[[ids.index(sid) for sid in samples]]
    low_cols = ["spot_summary_raw", "streak_summary_raw", "connection_summary_raw", "diffuse_summary_raw", "temporal_brightness_drift", "selected_16__saturation_fraction_median"]
    X_low = features[low_cols].fillna(0).to_numpy(float)
    pred_rows, audit_rows = [], []
    for held in np.unique(groups):
        tr = groups != held
        te = groups == held
        h = int(np.where(te)[0][0])
        # R0
        median_log = float(np.median(y[tr]))
        pred_rows.append(row(samples[h], groups[h], "R0_median", y_rq[h], 10**median_log, "training_fold_median"))
        # R1
        pls = Pipeline([("scaler", StandardScaler()), ("model", PLSRegression(n_components=min(2, tr.sum() - 1)))])
        pls.fit(X_dino[tr], y[tr])
        r1 = float(np.ravel(pls.predict(X_dino[te]))[0])
        pred_rows.append(row(samples[h], groups[h], "R1_dino_pls", y_rq[h], 10**r1, "phase2a_dino_keyframe_pls_reused"))
        # R2
        auto_idx = idx_oof["automatic_spot_streak_index"].to_numpy(float)
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(auto_idx[tr], y[tr])
        r2 = float(iso.predict(auto_idx[te])[0])
        pred_rows.append(row(samples[h], groups[h], "R2_auto_isotonic", y_rq[h], 10**r2, "monotonic_isotonic_training_fold_only"))
        # R3
        ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
        ridge.fit(X_low[tr], y[tr])
        r3 = float(ridge.predict(X_low[te])[0])
        pred_rows.append(row(samples[h], groups[h], "R3_auto_lowdim_ridge", y_rq[h], 10**r3, "low_dim_physics_ridge"))
        # R4
        train_iso = iso.predict(auto_idx[tr])
        residual = y[tr] - train_iso
        resid_model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))])
        resid_model.fit(X_dino[tr], residual)
        r4 = r2 + float(resid_model.predict(X_dino[te])[0])
        pred_rows.append(row(samples[h], groups[h], "R4_auto_iso_dino_residual", y_rq[h], 10**r4, "isotonic_plus_dino_residual_training_fold_only"))
        audit_rows.append({"heldout_sample_id": samples[h], "heldout_group": groups[h], "train_groups": json.dumps(sorted(set(groups[tr]))), "outer_target_used_for_training": False, "global_scaler_used": False})
    pred = pd.DataFrame(pred_rows)
    metrics = regression_metrics(pred)
    write_csv(idx_oof.reset_index(drop=True), output_root / "automatic_spot_streak_index.csv")
    write_csv(pred, output_root / "rheed_rq_oof_predictions.csv")
    write_csv(metrics, output_root / "rheed_rq_oof_metrics.csv")
    write_csv(pd.DataFrame(audit_rows), output_root / "rq_model_leakage_audit.csv")
    return pred, metrics, idx_oof.reset_index(drop=True), pd.DataFrame(audit_rows)


def row(sample_id: str, group: str, model_id: str, true_rq: float, pred_rq: float, notes: str) -> dict[str, Any]:
    pred_rq = float(np.clip(pred_rq, 0.2, 50.0))
    return {
        "sample_id": sample_id,
        "growth_run_id": group,
        "model_id": model_id,
        "true_rq_nm": float(true_rq),
        "predicted_rq_nm": pred_rq,
        "absolute_error_nm": abs(float(true_rq) - pred_rq),
        "model_notes": notes,
        "uses_predicted_rq_for_synthesis": True,
    }


def high_confidence_subset(manifest: pd.DataFrame, rq_pred: pd.DataFrame, idx: pd.DataFrame, features: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    best = rq_pred[rq_pred["model_id"] == "R4_auto_iso_dino_residual"].copy()
    samples = manifest["sample_id"].astype(str).to_numpy()
    groups = manifest["growth_run_id"].astype(str).to_numpy()
    idx_map = idx.set_index("sample_id")["automatic_spot_streak_index"].astype(float)
    stab = features.set_index("sample_id")["selected_16__saturation_fraction_median"].astype(float)
    rows = []
    for sid, group in zip(samples, groups):
        tr = groups != group
        train_ids = samples[tr]
        train_idx = idx_map.loc[train_ids].to_numpy(float)
        lo, hi = np.quantile(train_idx, [config["confidence_thresholds"]["automatic_index_extreme_quantile"], 1 - config["confidence_thresholds"]["automatic_index_extreme_quantile"]])
        val = float(idx_map.loc[sid])
        stable = float(stab.loc[sid]) < 0.02
        high = (val <= lo or val >= hi) and stable
        rows.append({"sample_id": sid, "support_level": "high" if high else "abstain", "automatic_index": val, "target_blind_rule": "index_extreme_and_stable_no_heldout_error", "uses_heldout_error": False})
    support = pd.DataFrame(rows)
    merged = best.merge(support, on="sample_id")
    high = merged[merged["support_level"] == "high"]
    metrics = regression_metrics(high, "model_id") if len(high) >= 3 else pd.DataFrame([{"model_id": "R4_auto_iso_dino_residual", "N": len(high), "MAE": np.nan, "Spearman": np.nan, "low_high_balanced_accuracy": np.nan}])
    metrics["coverage"] = len(high) / len(best)
    metrics["abstained_count"] = int((support["support_level"] == "abstain").sum())
    write_csv(support, repo_path(config["output_root"]) / "high_confidence_support.csv")
    write_csv(metrics, repo_path(config["output_root"]) / "high_confidence_rq_metrics.csv")
    return support, metrics
