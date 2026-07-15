from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _candidate_models(config: dict[str, Any], n_train: int, n_features: int, family: str | None = None) -> list[tuple[str, Any, dict[str, Any]]]:
    grid = config["baseline_hyperparameter_grids"]
    models: list[tuple[str, Any, dict[str, Any]]] = []
    if family in (None, "Ridge"):
        for alpha in grid["ridge_alpha"]:
            models.append(("Ridge", Ridge(alpha=alpha), {"alpha": alpha}))
    if family in (None, "ElasticNet"):
        for alpha in grid["elasticnet_alpha"]:
            for l1_ratio in grid["elasticnet_l1_ratio"]:
                models.append(("ElasticNet", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, tol=1e-3, random_state=config["random_seed"]), {"alpha": alpha, "l1_ratio": l1_ratio}))
    max_pls = max(1, min(n_train - 1, n_features))
    if family in (None, "PLSRegression"):
        for n_components in grid["pls_n_components"]:
            if n_components <= max_pls:
                models.append(("PLSRegression", PLSRegression(n_components=n_components), {"n_components": n_components}))
    return models


def _inner_select_regressor(X: np.ndarray, y: np.ndarray, config: dict[str, Any], family: str) -> tuple[str, Pipeline, dict[str, Any]]:
    n_train, n_features = X.shape
    candidates = _candidate_models(config, n_train, n_features, family)
    if n_train < 4:
        name, model, params = candidates[0]
        return name, Pipeline([("scaler", StandardScaler()), ("model", model)]), params
    cv = KFold(n_splits=min(5, n_train), shuffle=True, random_state=config["random_seed"])
    best: tuple[float, str, Any, dict[str, Any]] | None = None
    for name, model, params in candidates:
        preds = np.zeros(n_train, dtype=float)
        ok = True
        for train_idx, val_idx in cv.split(X):
            try:
                pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    pipe.fit(X[train_idx], y[train_idx])
                preds[val_idx] = np.ravel(pipe.predict(X[val_idx]))
            except Exception:
                ok = False
                break
        if not ok:
            continue
        score = mean_absolute_error(y, preds)
        if best is None or score < best[0]:
            best = (score, name, model, params)
    if best is None:
        name, model, params = candidates[0]
    else:
        _, name, model, params = best
    return name, Pipeline([("scaler", StandardScaler()), ("model", model)]), params


def _inner_select_knn(X: np.ndarray, y: np.ndarray, config: dict[str, Any]) -> tuple[Pipeline, dict[str, Any]]:
    grid = config["baseline_hyperparameter_grids"]
    n_train = len(y)
    candidates: list[dict[str, Any]] = []
    for k in grid["knn_k"]:
        if k < n_train:
            for weights in grid["knn_weights"]:
                for metric in grid["knn_metric"]:
                    candidates.append({"n_neighbors": k, "weights": weights, "metric": metric})
    if n_train < 4 or not candidates:
        params = {"n_neighbors": min(2, max(1, n_train)), "weights": "distance", "metric": "euclidean"}
        return Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(**params))]), params
    cv = KFold(n_splits=min(5, n_train), shuffle=True, random_state=config["random_seed"])
    best: tuple[float, dict[str, Any]] | None = None
    for params in candidates:
        preds = np.zeros(n_train, dtype=float)
        ok = True
        for train_idx, val_idx in cv.split(X):
            if params["n_neighbors"] >= len(train_idx):
                ok = False
                break
            try:
                pipe = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(**params))])
                pipe.fit(X[train_idx], y[train_idx])
                preds[val_idx] = pipe.predict(X[val_idx])
            except Exception:
                ok = False
                break
        if not ok:
            continue
        score = mean_absolute_error(y, preds)
        if best is None or score < best[0]:
            best = (score, params)
    params = best[1] if best is not None else {"n_neighbors": min(2, max(1, n_train)), "weights": "distance", "metric": "euclidean"}
    return Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(**params))]), params


def run_oof_baselines(
    cohort_df: pd.DataFrame,
    feature_sets: dict[str, tuple[list[str], np.ndarray]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    neighbor_rows: list[dict[str, Any]] = []
    sample_ids = cohort_df["sample_id"].astype(str).to_numpy()
    groups = cohort_df["growth_run_id"].astype(str).to_numpy()
    y_rq = cohort_df["primary_rq_nm_median"].to_numpy(dtype=float)
    y = np.log10(y_rq)
    unique_groups = np.unique(groups)
    for held_group in unique_groups:
        test_idx = np.where(groups == held_group)[0]
        train_idx = np.where(groups != held_group)[0]
        if len(test_idx) != 1:
            raise ValueError("This phase expects one sample per growth_run_id for strict LOGO.")
        test_i = int(test_idx[0])
        median_pred = float(np.median(y[train_idx]))
        rows.append(_prediction_row(sample_ids[test_i], "primary_1um", y_rq[test_i], y[test_i], "B0_training_fold_median", median_pred, held_group, {}, [], []))
        for feature_name, (_, X) in feature_sets.items():
            if feature_name not in {"B1_keyframe", "B2_temporal"}:
                continue
            for family in ("Ridge", "PLSRegression", "ElasticNet"):
                model_name, pipe, params = _inner_select_regressor(X[train_idx], y[train_idx], config, family)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    pipe.fit(X[train_idx], y[train_idx])
                pred = float(np.ravel(pipe.predict(X[[test_i]]))[0])
                rows.append(_prediction_row(sample_ids[test_i], "primary_1um", y_rq[test_i], y[test_i], f"{feature_name}_{model_name}", pred, held_group, params, [], []))
        names, X = feature_sets["B2_temporal"]
        knn_pipe, knn_params = _inner_select_knn(X[train_idx], y[train_idx], config)
        knn_pipe.fit(X[train_idx], y[train_idx])
        pred = float(knn_pipe.predict(X[[test_i]])[0])
        scaler = knn_pipe.named_steps["scaler"]
        model = knn_pipe.named_steps["model"]
        test_scaled = scaler.transform(X[[test_i]])
        train_scaled = scaler.transform(X[train_idx])
        distances, local_indices = model.kneighbors(test_scaled, return_distance=True)
        local_indices = local_indices[0]
        distances = distances[0]
        neighbor_ids = [str(sample_ids[train_idx[j]]) for j in local_indices]
        neighbor_distances = [float(d) for d in distances]
        neighbor_rq = [float(y_rq[train_idx[j]]) for j in local_indices]
        rows.append(_prediction_row(sample_ids[test_i], "primary_1um", y_rq[test_i], y[test_i], "B3_knn_temporal_retrieval", pred, held_group, knn_params, neighbor_ids, neighbor_distances))
        for rank, j in enumerate(local_indices, start=1):
            neighbor_rows.append(
                {
                    "heldout_sample_id": str(sample_ids[test_i]),
                    "outer_fold": str(held_group),
                    "neighbor_rank": rank,
                    "neighbor_sample_id": str(sample_ids[train_idx[j]]),
                    "neighbor_distance": float(distances[rank - 1]),
                    "neighbor_rq_nm": float(y_rq[train_idx[j]]),
                    "selected_hyperparameters": json.dumps(knn_params, sort_keys=True),
                    "leakage_free": str(sample_ids[train_idx[j]]) != str(sample_ids[test_i]),
                }
            )
    pred_df = pd.DataFrame(rows)
    metrics_df = compute_metrics(pred_df)
    return pred_df, metrics_df, pd.DataFrame(neighbor_rows)


def _prediction_row(sample_id: str, cohort: str, true_rq: float, true_log: float, model_name: str, pred_log: float, fold: str, params: dict[str, Any], neighbor_ids: list[str], neighbor_distances: list[float]) -> dict[str, Any]:
    pred_rq = float(10**pred_log)
    return {
        "sample_id": str(sample_id),
        "cohort": cohort,
        "true_rq_nm": float(true_rq),
        "true_log_rq": float(true_log),
        "model_name": model_name,
        "pred_log_rq": pred_log,
        "pred_rq_nm": pred_rq,
        "absolute_error_nm": abs(pred_rq - true_rq),
        "relative_error": abs(pred_rq - true_rq) / true_rq,
        "outer_fold": str(fold),
        "selected_hyperparameters": json.dumps(params, sort_keys=True),
        "neighbor_ids": json.dumps(neighbor_ids),
        "neighbor_distances": json.dumps(neighbor_distances),
    }


def pairwise_concordance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = 0
    correct = 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            true_delta = y_true[i] - y_true[j]
            pred_delta = y_pred[i] - y_pred[j]
            if true_delta == 0:
                continue
            total += 1
            if true_delta * pred_delta > 0:
                correct += 1
            elif pred_delta == 0:
                correct += 0.5
    return float(correct / total) if total else float("nan")


def compute_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, group in pred_df.groupby("model_name", sort=True):
        y_true = group["true_rq_nm"].to_numpy(dtype=float)
        y_pred = group["pred_rq_nm"].to_numpy(dtype=float)
        rho = spearmanr(y_true, y_pred).statistic if len(y_true) > 2 else np.nan
        tau = kendalltau(y_true, y_pred).statistic if len(y_true) > 2 else np.nan
        rows.append(
            {
                "cohort": group["cohort"].iloc[0],
                "model_name": model_name,
                "N": int(len(group)),
                "MAE_nm": float(mean_absolute_error(y_true, y_pred)),
                "median_absolute_error_nm": float(np.median(np.abs(y_true - y_pred))),
                "RMSE_nm": float(mean_squared_error(y_true, y_pred) ** 0.5),
                "R2": float(r2_score(y_true, y_pred)) if len(group) > 1 else np.nan,
                "Spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "Kendall_tau": float(tau) if np.isfinite(tau) else np.nan,
                "pairwise_concordance": pairwise_concordance(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows).sort_values(["MAE_nm", "model_name"])
