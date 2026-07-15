from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import BayesianRidge, HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .baseline_rq import pairwise_concordance
from .common import repo_path


def load_embedding(path: str) -> tuple[list[str], np.ndarray]:
    data = np.load(repo_path(path))
    return [str(x) for x in data["sample_ids"].tolist()], np.asarray(data["embeddings"], dtype=float)


def candidate_heads(config: dict[str, Any], n_train: int, n_features: int, family: str) -> list[tuple[str, Any, dict[str, Any]]]:
    grid = config["regression_grids"]
    out = []
    dims = [d for d in config["pca_dimensions"] if d <= min(n_train - 1, n_features)]
    dims = [None] + dims
    if family == "Ridge":
        for pca_dim in [None, 4, 8]:
            if pca_dim is not None and pca_dim not in dims:
                continue
            for alpha in [0.1, 10.0]:
                out.append(("Ridge", Ridge(alpha=alpha), {"pca_dim": pca_dim, "alpha": alpha}))
    elif family == "BayesianRidge":
        for pca_dim in [4]:
            if pca_dim is not None and pca_dim not in dims:
                continue
            out.append(("BayesianRidge", BayesianRidge(), {"pca_dim": pca_dim}))
    elif family == "PLSRegression":
        for n in [1, 2, 4]:
            if n <= min(n_train - 1, n_features):
                out.append(("PLSRegression", PLSRegression(n_components=n), {"pca_dim": None, "n_components": n}))
    elif family == "HuberRegressor":
        for pca_dim in [4]:
            if pca_dim is not None and pca_dim not in dims:
                continue
            for alpha in [0.001]:
                out.append(("HuberRegressor", HuberRegressor(alpha=alpha, max_iter=1000), {"pca_dim": pca_dim, "alpha": alpha}))
    elif family == "KNN":
        for pca_dim in [4, 8]:
            if pca_dim not in dims:
                continue
            for k in [2, 3, 5]:
                if k < n_train:
                    for weights in ["distance"]:
                        for metric in ["cosine", "euclidean"]:
                            out.append(("KNN", KNeighborsRegressor(n_neighbors=k, weights=weights, metric=metric), {"pca_dim": pca_dim, "k": k, "weights": weights, "metric": metric}))
    return out


def make_pipeline(model: Any, pca_dim: int | None) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if pca_dim is not None:
        steps.append(("pca", PCA(n_components=pca_dim, random_state=17)))
    steps.append(("model", model))
    return Pipeline(steps)


def inner_select(X: np.ndarray, y: np.ndarray, groups: np.ndarray, config: dict[str, Any], family: str) -> tuple[str, Pipeline, dict[str, Any], dict[str, float]]:
    candidates = candidate_heads(config, len(y), X.shape[1], family)
    if not candidates:
        candidates = candidate_heads(config, len(y), X.shape[1], "Ridge")
    best = None
    for name, model, params in candidates:
        try:
            pipe = make_pipeline(model, params.get("pca_dim"))
            pipe.fit(X, y)
            pred = np.ravel(pipe.predict(X))
        except Exception:
            continue
        mae = mean_absolute_error(y, pred)
        rho = spearmanr(y, pred).statistic if len(y) > 2 else np.nan
        score = (mae, -rho if np.isfinite(rho) else 0.0)
        if best is None or score < best[0]:
            best = (score, name, model, params, {"mae_log": float(mae), "spearman": float(rho) if np.isfinite(rho) else np.nan, "selection_mode": "outer_training_fit_only"})
    if best is None:
        name, model, params = candidates[0]
        return name, make_pipeline(model, params.get("pca_dim")), params, {}
    _, name, model, params, info = best
    return name, make_pipeline(model, params.get("pca_dim")), params, info


def prediction_metrics(pred: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in pred.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        yt = g["true_rq_nm"].to_numpy(float)
        yp = g["pred_rq_nm"].to_numpy(float)
        rho = spearmanr(yt, yp).statistic if len(g) > 2 else np.nan
        tau = kendalltau(yt, yp).statistic if len(g) > 2 else np.nan
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "N": len(g),
                "MAE_nm": float(mean_absolute_error(yt, yp)),
                "median_AE_nm": float(np.median(np.abs(yt - yp))),
                "RMSE_nm": float(mean_squared_error(yt, yp) ** 0.5),
                "R2": float(r2_score(yt, yp)) if len(g) > 1 else np.nan,
                "Spearman": float(rho) if np.isfinite(rho) else np.nan,
                "Kendall_tau": float(tau) if np.isfinite(tau) else np.nan,
                "pairwise_concordance": pairwise_concordance(yt, yp),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("MAE_nm")


def run_regression(manifest: pd.DataFrame, embedding_registry: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_rq = manifest["primary_rq_nm_median"].to_numpy(float)
    y = np.log10(y_rq)
    sample_ids = manifest["sample_id"].astype(str).to_numpy()
    groups = manifest["growth_run_id"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for _, emb_row in embedding_registry.iterrows():
        ids, X_all = load_embedding(emb_row["path"])
        order = [ids.index(sid) for sid in sample_ids]
        X = X_all[order]
        for family in ("Ridge", "BayesianRidge", "PLSRegression", "HuberRegressor", "KNN"):
            for held in np.unique(groups):
                te = groups == held
                tr = ~te
                name, pipe, params, info = inner_select(X[tr], y[tr], groups[tr], config, family)
                pipe.fit(X[tr], y[tr])
                pred_log = float(np.ravel(pipe.predict(X[te]))[0])
                test_i = int(np.where(te)[0][0])
                pred_rq = float(10**pred_log)
                rows.append(
                    {
                        "sample_id": sample_ids[test_i],
                        "growth_run_id": held,
                        "embedding_id": emb_row["embedding_id"],
                        "encoder": emb_row["encoder"],
                        "clip_variant": emb_row["clip_variant"],
                        "preprocessing": emb_row["preprocessing"],
                        "head": name,
                        "true_rq_nm": float(y_rq[test_i]),
                        "true_log_rq": float(y[test_i]),
                        "pred_log_rq": pred_log,
                        "pred_rq_nm": pred_rq,
                        "absolute_error_nm": abs(pred_rq - y_rq[test_i]),
                        "selected_hyperparameters": json.dumps(params, sort_keys=True),
                    }
                )
                selections.append(
                    {
                        "heldout_sample_id": sample_ids[test_i],
                        "embedding_id": emb_row["embedding_id"],
                        "head": family,
                        "selected_head": name,
                        "selected_hyperparameters": json.dumps(params, sort_keys=True),
                        "inner_mae_log": info.get("mae_log", np.nan),
                        "inner_spearman": info.get("spearman", np.nan),
                    }
                )
    pred = pd.DataFrame(rows)
    metrics = prediction_metrics(pred, ["embedding_id", "encoder", "clip_variant", "preprocessing", "head"])
    return pred, metrics, pd.DataFrame(selections)


def metadata_matrix(manifest: pd.DataFrame, quality: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    q = quality.copy()
    q["sample_id"] = q["sample_id"].astype(str)
    df = manifest.merge(q, on=["sample_id", "video_id"], how="left", suffixes=("", "_q"))
    num_cols = ["fps", "roi_area_fraction", "mean_intensity", "saturated_pixel_fraction", "temporal_intensity_drift"]
    cat_cols = ["growth_stage", "video_stage", "camera_or_tool_id"]
    numeric = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float)
    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
    cat = enc.fit_transform(df[cat_cols].fillna("unknown").astype(str))
    names = num_cols + [f"cat_{v}" for vals in enc.categories_ for v in vals]
    return names, np.hstack([numeric, cat]).astype(float)


def run_metadata_controls(manifest: pd.DataFrame, quality: pd.DataFrame, best_embedding_row: pd.Series, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_ids = manifest["sample_id"].astype(str).to_numpy()
    groups = manifest["growth_run_id"].astype(str).to_numpy()
    y_rq = manifest["primary_rq_nm_median"].to_numpy(float)
    y = np.log10(y_rq)
    _, Xm = metadata_matrix(manifest, quality)
    ids, Xe_all = load_embedding(best_embedding_row["path"])
    Xe = Xe_all[[ids.index(sid) for sid in sample_ids]]
    feature_sets = {
        "M1_metadata_only": Xm,
        "M2_rheed_embedding_only": Xe,
        "M3_embedding_plus_metadata": np.hstack([Xe, Xm]),
    }
    rows = []
    for name, X in feature_sets.items():
        for held in np.unique(groups):
            te, tr = groups == held, groups != held
            _, pipe, params, _ = inner_select(X[tr], y[tr], groups[tr], config, "Ridge")
            pipe.fit(X[tr], y[tr])
            pred = float(np.ravel(pipe.predict(X[te]))[0])
            i = int(np.where(te)[0][0])
            rows.append({"sample_id": sample_ids[i], "model": name, "true_rq_nm": y_rq[i], "pred_rq_nm": 10**pred, "pred_log_rq": pred})
        if name == "M1_metadata_only":
            pass
    # M4 residual model.
    for held in np.unique(groups):
        te, tr = groups == held, groups != held
        _, meta_pipe, _, _ = inner_select(Xm[tr], y[tr], groups[tr], config, "Ridge")
        meta_pipe.fit(Xm[tr], y[tr])
        train_resid = y[tr] - np.ravel(meta_pipe.predict(Xm[tr]))
        _, res_pipe, _, _ = inner_select(Xe[tr], train_resid, groups[tr], config, "Ridge")
        res_pipe.fit(Xe[tr], train_resid)
        pred = float(np.ravel(meta_pipe.predict(Xm[te]))[0] + np.ravel(res_pipe.predict(Xe[te]))[0])
        i = int(np.where(te)[0][0])
        rows.append({"sample_id": sample_ids[i], "model": "M4_metadata_plus_rheed_residual", "true_rq_nm": y_rq[i], "pred_rq_nm": 10**pred, "pred_log_rq": pred})
    pred = pd.DataFrame(rows)
    metrics = prediction_metrics(pred.rename(columns={"model": "control_model"}), ["control_model"])
    return pred, metrics
