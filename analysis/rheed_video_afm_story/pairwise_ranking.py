from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .baseline_rq import pairwise_concordance
from .common import repo_path
from .embedding_regression import load_embedding


def sigma_log(mad: np.ndarray, median: np.ndarray) -> np.ndarray:
    return np.maximum(0.434 * mad / np.maximum(median, 1e-9), 0.02)


def build_pairs(X: np.ndarray, y: np.ndarray, sigma: np.ndarray, margin: float) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    feats, labels, pairs = [], [], []
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            ambiguous = abs(y[i] - y[j]) <= margin * float(np.sqrt(sigma[i] ** 2 + sigma[j] ** 2))
            if ambiguous:
                continue
            label = 1 if y[i] > y[j] else 0
            diff = X[i] - X[j]
            feats.extend([diff, -diff])
            labels.extend([label, 1 - label])
            pairs.extend([(i, j), (j, i)])
    return np.asarray(feats), np.asarray(labels), pairs


def run_pairwise_ranking(manifest: pd.DataFrame, embedding_registry: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_rq = manifest["primary_rq_nm_median"].to_numpy(float)
    y = np.log10(y_rq)
    sig = sigma_log(manifest["primary_rq_nm_mad"].fillna(0).to_numpy(float), y_rq)
    sample_ids = manifest["sample_id"].astype(str).to_numpy()
    groups = manifest["growth_run_id"].astype(str).to_numpy()
    pred_rows, audit_rows = [], []
    for _, emb_row in embedding_registry.iterrows():
        ids, X_all = load_embedding(emb_row["path"])
        X = X_all[[ids.index(sid) for sid in sample_ids]]
        for held in np.unique(groups):
            te, tr = groups == held, groups != held
            C, margin = 1.0, 0.0
            pair_X, pair_y, pairs = build_pairs(X[tr], y[tr], sig[tr], margin)
            n_components = max(1, min(8, pair_X.shape[0] - 1, pair_X.shape[1]))
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=n_components, random_state=config["random_seed"])),
                    ("model", LogisticRegression(C=C, solver="liblinear", random_state=config["random_seed"], max_iter=500)),
                ]
            )
            pipe.fit(pair_X, pair_y)
            h = int(np.where(te)[0][0])
            train_global = np.where(tr)[0]
            probs = np.asarray([pipe.predict_proba((X[h] - X[j])[None, :])[0, 1] for j in train_global])
            rank_score = float(np.mean(probs))
            order = np.argsort(y_rq[train_global])
            q_index = int(np.clip(round(rank_score * (len(order) - 1)), 0, len(order) - 1))
            pred_rq = float(y_rq[train_global][order[q_index]])
            pred_rows.append(
                {
                    "sample_id": sample_ids[h],
                    "embedding_id": emb_row["embedding_id"],
                    "encoder": emb_row["encoder"],
                    "clip_variant": emb_row["clip_variant"],
                    "preprocessing": emb_row["preprocessing"],
                    "true_rq_nm": float(y_rq[h]),
                    "true_log_rq": float(y[h]),
                    "predicted_rank_percentile": rank_score,
                    "rank_derived_pred_rq_nm": pred_rq,
                    "selected_hyperparameters": json.dumps({"C": C, "margin_multiplier": margin, "pca_components": n_components}, sort_keys=True),
                    "training_pair_count": int(len(pair_y)),
                }
            )
            for pair in pairs:
                audit_rows.append({"heldout_sample_id": sample_ids[h], "embedding_id": emb_row["embedding_id"], "pair_i": sample_ids[train_global[pair[0]]], "pair_j": sample_ids[train_global[pair[1]]], "contains_heldout": False})
    pred = pd.DataFrame(pred_rows)
    metrics = []
    for key, g in pred.groupby(["embedding_id", "encoder", "clip_variant", "preprocessing"]):
        yt = g["true_rq_nm"].to_numpy(float)
        yp = g["rank_derived_pred_rq_nm"].to_numpy(float)
        true_rank = pd.Series(yt).rank(pct=True).to_numpy()
        pred_rank = g["predicted_rank_percentile"].to_numpy(float)
        metrics.append(
            {
                "embedding_id": key[0],
                "encoder": key[1],
                "clip_variant": key[2],
                "preprocessing": key[3],
                "N": len(g),
                "Spearman": float(spearmanr(yt, yp).statistic),
                "Kendall_tau": float(kendalltau(yt, yp).statistic),
                "pairwise_concordance": pairwise_concordance(yt, yp),
                "rank_MAE": float(mean_absolute_error(true_rank, pred_rank)),
                "rank_derived_Rq_MAE_nm": float(mean_absolute_error(yt, yp)),
            }
        )
    return pred, pd.DataFrame(audit_rows), pd.DataFrame(metrics).sort_values("rank_derived_Rq_MAE_nm")
