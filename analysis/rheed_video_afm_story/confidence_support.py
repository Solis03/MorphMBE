from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors

from .baseline_rq import pairwise_concordance
from .common import repo_path
from .embedding_regression import load_embedding


def best_regression_rows(metrics: pd.DataFrame, n: int = 5) -> list[str]:
    return metrics.sort_values("MAE_nm")["embedding_id"].drop_duplicates().head(n).tolist()


def support_scores(
    manifest: pd.DataFrame,
    embedding_registry: pd.DataFrame,
    regression_pred: pd.DataFrame,
    ranking_pred: pd.DataFrame,
    quality: pd.DataFrame,
    best_embedding_id: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    emb_row = embedding_registry[embedding_registry["embedding_id"] == best_embedding_id].iloc[0]
    ids, X_all = load_embedding(emb_row["path"])
    sample_ids = manifest["sample_id"].astype(str).to_numpy()
    X = X_all[[ids.index(sid) for sid in sample_ids]]
    y = manifest["primary_rq_nm_median"].to_numpy(float)
    groups = manifest["growth_run_id"].astype(str).to_numpy()
    q = quality[["sample_id", "quality_flags"]].copy()
    q["sample_id"] = q["sample_id"].astype(str)
    qmap = q.set_index("sample_id")["quality_flags"].fillna("").to_dict()
    rows = []
    preferred = [
        "resnet18__keyframe_1__raw_luminance",
        "resnet18__centered_8__raw_luminance",
        "dino_vits14__centered_8__raw_luminance",
        "r3d_18__centered_8__raw_luminance",
    ]
    available_embeddings = regression_pred["embedding_id"].drop_duplicates().astype(str).tolist()
    ensemble = [emb_id for emb_id in preferred if emb_id in available_embeddings]
    if not ensemble:
        ensemble = available_embeddings[:4]
    for i, sid in enumerate(sample_ids):
        tr = groups != groups[i]
        train_X = X[tr]
        train_y = y[tr]
        nn = NearestNeighbors(n_neighbors=min(5, len(train_y)), metric="cosine")
        nn.fit(train_X)
        dists, neigh = nn.kneighbors(X[[i]])
        dists = dists[0]
        neigh = neigh[0]
        train_indices = np.where(tr)[0][neigh]
        neighbor_rq = y[train_indices]
        neighbor_iqr = float(np.percentile(neighbor_rq, 75) - np.percentile(neighbor_rq, 25))
        domain_distance = float(np.mean(dists[: min(3, len(dists))]))
        preds = regression_pred[(regression_pred["sample_id"].astype(str) == sid) & (regression_pred["embedding_id"].isin(ensemble))]
        model_disagreement = float(np.std(preds["pred_log_rq"].to_numpy(float))) if len(preds) else np.nan
        raw = regression_pred[(regression_pred["sample_id"].astype(str) == sid) & (regression_pred["preprocessing"] == "raw_luminance")]
        robust = regression_pred[(regression_pred["sample_id"].astype(str) == sid) & (regression_pred["preprocessing"] == "clip_robust_contrast")]
        preprocessing_sensitivity = float(abs(raw["pred_log_rq"].mean() - robust["pred_log_rq"].mean())) if len(raw) and len(robust) else np.nan
        by_variant = regression_pred[regression_pred["sample_id"].astype(str) == sid].groupby("clip_variant")["pred_log_rq"].mean()
        temporal_sensitivity = float(by_variant.std()) if len(by_variant) > 1 else np.nan
        train_domains = []
        train_iqrs = []
        for j in np.where(tr)[0]:
            tr2 = tr.copy()
            tr2[j] = False
            if tr2.sum() < 3:
                continue
            nn2 = NearestNeighbors(n_neighbors=min(5, tr2.sum()), metric="cosine").fit(X[tr2])
            d2, n2 = nn2.kneighbors(X[[j]])
            train_domains.append(float(np.mean(d2[0][: min(3, len(d2[0]))])))
            train_iqrs.append(float(np.percentile(y[np.where(tr2)[0][n2[0]]], 75) - np.percentile(y[np.where(tr2)[0][n2[0]]], 25)))
        domain_q75 = np.quantile(train_domains, 0.75) if train_domains else domain_distance
        iqr_q75 = np.quantile(train_iqrs, 0.75) if train_iqrs else neighbor_iqr
        disagreement_train = regression_pred[regression_pred["sample_id"].astype(str).isin(sample_ids[tr])].groupby("sample_id")["pred_log_rq"].std().dropna()
        dis_q75 = disagreement_train.quantile(0.75) if len(disagreement_train) else model_disagreement
        flags = str(qmap.get(sid, ""))
        bad_quality = any(token in flags for token in ("very_dark", "large_frame_discontinuity", "decode_or_shape_mismatch"))
        penalties = [
            domain_distance > domain_q75,
            neighbor_iqr > iqr_q75,
            np.isfinite(model_disagreement) and model_disagreement > dis_q75,
            bad_quality,
        ]
        support_score = 1.0 - sum(bool(p) for p in penalties) / len(penalties)
        support_level = "high" if support_score >= 0.75 else "medium" if support_score >= 0.5 else "low"
        best_pred = regression_pred[(regression_pred["sample_id"].astype(str) == sid) & (regression_pred["embedding_id"] == best_embedding_id)].sort_values("head").iloc[0]
        rows.append(
            {
                "sample_id": sid,
                "predicted_rq": float(best_pred["pred_rq_nm"]),
                "prediction_interval_low": np.nan,
                "prediction_interval_high": np.nan,
                "embedding_domain_distance": domain_distance,
                "neighbor_ids": json.dumps(sample_ids[train_indices].tolist()),
                "neighbor_distances": json.dumps([float(x) for x in dists.tolist()]),
                "neighbor_rq_values": json.dumps([float(x) for x in neighbor_rq.tolist()]),
                "neighbor_rq_iqr": neighbor_iqr,
                "model_disagreement": model_disagreement,
                "preprocessing_sensitivity": preprocessing_sensitivity,
                "temporal_sensitivity": temporal_sensitivity,
                "quality_flags": flags,
                "support_score": support_score,
                "support_level": support_level,
            }
        )
    return pd.DataFrame(rows)


def conformal_intervals(regression_pred: pd.DataFrame, manifest: pd.DataFrame, best_key: tuple[str, str]) -> pd.DataFrame:
    emb_id, head = best_key
    pred = regression_pred[(regression_pred["embedding_id"] == emb_id) & (regression_pred["head"] == head)].copy()
    residual = np.abs(pred["true_log_rq"].to_numpy(float) - pred["pred_log_rq"].to_numpy(float))
    lows80, highs80, lows90, highs90 = [], [], [], []
    for idx, row in pred.iterrows():
        train_resid = np.delete(residual, list(pred.index).index(idx))
        q80 = float(np.quantile(train_resid, 0.8))
        q90 = float(np.quantile(train_resid, 0.9))
        p = float(row["pred_log_rq"])
        lows80.append(10 ** (p - q80))
        highs80.append(10 ** (p + q80))
        lows90.append(10 ** (p - q90))
        highs90.append(10 ** (p + q90))
    out = pred[["sample_id", "true_rq_nm", "pred_rq_nm", "pred_log_rq"]].copy()
    out["pi80_low_nm"] = lows80
    out["pi80_high_nm"] = highs80
    out["pi90_low_nm"] = lows90
    out["pi90_high_nm"] = highs90
    out["covered_80"] = (out["true_rq_nm"] >= out["pi80_low_nm"]) & (out["true_rq_nm"] <= out["pi80_high_nm"])
    out["covered_90"] = (out["true_rq_nm"] >= out["pi90_low_nm"]) & (out["true_rq_nm"] <= out["pi90_high_nm"])
    return out


def coverage_performance(conf: pd.DataFrame, regression_pred: pd.DataFrame, best_key: tuple[str, str]) -> pd.DataFrame:
    emb_id, head = best_key
    pred = regression_pred[(regression_pred["embedding_id"] == emb_id) & (regression_pred["head"] == head)].merge(conf[["sample_id", "support_score"]], on="sample_id")
    pred = pred.sort_values("support_score", ascending=False)
    rows = []
    for frac in (1.0, 0.8, 0.6, 0.5, 0.4):
        n = max(1, int(round(len(pred) * frac)))
        g = pred.head(n)
        rows.append(
            {
                "coverage_fraction": frac,
                "N": len(g),
                "MAE_nm": float(mean_absolute_error(g["true_rq_nm"], g["pred_rq_nm"])),
                "Spearman": float(spearmanr(g["true_rq_nm"], g["pred_rq_nm"]).statistic) if len(g) > 2 else np.nan,
                "pairwise_concordance": pairwise_concordance(g["true_rq_nm"].to_numpy(float), g["pred_rq_nm"].to_numpy(float)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict[str, float]:
    vals = {"MAE": [], "Spearman": [], "pairwise_concordance": []}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals["MAE"].append(mean_absolute_error(y_true[idx], y_pred[idx]))
        vals["Spearman"].append(spearmanr(y_true[idx], y_pred[idx]).statistic if len(np.unique(y_true[idx])) > 1 else np.nan)
        vals["pairwise_concordance"].append(pairwise_concordance(y_true[idx], y_pred[idx]))
    out = {}
    for key, arr in vals.items():
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        out[f"{key}_ci_low"] = float(np.quantile(a, 0.025)) if a.size else np.nan
        out[f"{key}_ci_high"] = float(np.quantile(a, 0.975)) if a.size else np.nan
    return out


def permutation_summary(y_true: np.ndarray, y_pred: np.ndarray, median_mae: float, rng: np.random.Generator, n_perm: int) -> dict[str, float]:
    obs_s = spearmanr(y_true, y_pred).statistic
    obs_c = pairwise_concordance(y_true, y_pred)
    obs_mae = mean_absolute_error(y_true, y_pred)
    ps, pc, pm = 0, 0, 0
    for _ in range(n_perm):
        yp = rng.permutation(y_pred)
        ps += spearmanr(y_true, yp).statistic >= obs_s
        pc += pairwise_concordance(y_true, yp) >= obs_c
        pm += (median_mae - mean_absolute_error(y_true, yp)) >= (median_mae - obs_mae)
    return {
        "spearman_empirical_p": float((ps + 1) / (n_perm + 1)),
        "concordance_empirical_p": float((pc + 1) / (n_perm + 1)),
        "mae_improvement_empirical_p": float((pm + 1) / (n_perm + 1)),
    }


def retrieval_audit(manifest: pd.DataFrame, embedding_registry: pd.DataFrame, embedding_ids: list[str], output_root) -> pd.DataFrame:
    rows = []
    sample_ids = manifest["sample_id"].astype(str).to_numpy()
    rq = manifest["primary_rq_nm_median"].to_numpy(float)
    stage = manifest["video_stage"].astype(str).to_numpy()
    afm = manifest["representative_afm_path"].astype(str).to_numpy()
    for emb_id in embedding_ids:
        emb_row = embedding_registry[embedding_registry["embedding_id"] == emb_id].iloc[0]
        ids, X_all = load_embedding(emb_row["path"])
        X = X_all[[ids.index(sid) for sid in sample_ids]]
        dmat = cosine_distances(X)
        for i, sid in enumerate(sample_ids):
            order = np.argsort(dmat[i])
            neigh = [j for j in order if j != i][:5]
            rows.append(
                {
                    "sample_id": sid,
                    "embedding_id": emb_id,
                    "neighbor_sample_ids": json.dumps(sample_ids[neigh].tolist()),
                    "neighbor_distances": json.dumps([float(dmat[i, j]) for j in neigh]),
                    "neighbor_rq": json.dumps([float(rq[j]) for j in neigh]),
                    "neighbor_representative_afm_path": json.dumps(afm[neigh].tolist()),
                    "neighbor_stage": json.dumps(stage[neigh].tolist()),
                    "neighbor_rq_dispersion": float(np.percentile(rq[neigh], 75) - np.percentile(rq[neigh], 25)),
                    "stage_agreement": float(np.mean(stage[neigh] == stage[i])),
                    "rank_agreement": float(np.mean((rq[neigh] - rq[i]) * np.arange(len(neigh)) >= 0)),
                }
            )
    return pd.DataFrame(rows)
