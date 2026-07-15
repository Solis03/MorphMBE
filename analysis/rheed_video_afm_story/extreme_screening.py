from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .embedding_regression import load_embedding


def regime_labels(rq: np.ndarray, q33: float, q67: float) -> np.ndarray:
    labels = np.full(len(rq), "middle", dtype=object)
    labels[rq <= q33] = "low"
    labels[rq >= q67] = "high"
    return labels


def classifier_candidates(config: dict[str, Any], n_train: int, n_features: int, binary: bool) -> list[tuple[str, Any, dict[str, Any]]]:
    dims = [d for d in (4, 8) if d <= min(n_train - 1, n_features)]
    if not dims:
        dims = [None]
    out = []
    for dim in dims:
        out.append(("logistic", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config["random_seed"]), {"pca_dim": dim}))
    if 3 < n_train:
        out.append(("knn", KNeighborsClassifier(n_neighbors=3, weights="distance"), {"pca_dim": min(d for d in dims if d is not None) if any(d is not None for d in dims) else None, "k": 3}))
    return out


def make_pipe(model: Any, pca_dim: int | None) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if pca_dim is not None:
        steps.append(("pca", PCA(n_components=pca_dim, random_state=17)))
    steps.append(("model", model))
    return Pipeline(steps)


def score_binary(y_true: np.ndarray, prob_high: np.ndarray) -> dict[str, float]:
    pred = np.where(prob_high >= 0.5, "high", "low")
    auroc = roc_auc_score((y_true == "high").astype(int), prob_high) if len(np.unique(y_true)) == 2 else np.nan
    auprc = average_precision_score((y_true == "high").astype(int), prob_high) if len(np.unique(y_true)) == 2 else np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=["low", "high"]).ravel()
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_F1": float(f1_score(y_true, pred, average="macro")),
        "AUROC": float(auroc) if np.isfinite(auroc) else np.nan,
        "AUPRC": float(auprc) if np.isfinite(auprc) else np.nan,
        "high_sensitivity": float(tp / max(tp + fn, 1)),
        "high_specificity": float(tn / max(tn + fp, 1)),
    }


def run_extreme_screening(manifest: pd.DataFrame, embedding_registry: pd.DataFrame, config: dict[str, Any], q33: float, q67: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rq = manifest["primary_rq_nm_median"].to_numpy(float)
    labels = regime_labels(rq, q33, q67)
    sample_ids = manifest["sample_id"].astype(str).to_numpy()
    groups = manifest["growth_run_id"].astype(str).to_numpy()
    pred_rows = []
    for _, emb_row in embedding_registry.iterrows():
        ids, X_all = load_embedding(emb_row["path"])
        X = X_all[[ids.index(sid) for sid in sample_ids]]
        for task in ("three_class", "extreme_binary"):
            for held in np.unique(groups):
                te, tr = groups == held, groups != held
                train_mask = tr.copy()
                test_i = int(np.where(te)[0][0])
                if task == "extreme_binary":
                    train_mask &= labels != "middle"
                    if labels[test_i] == "middle":
                        pred_rows.append({"sample_id": sample_ids[test_i], "embedding_id": emb_row["embedding_id"], "task": task, "true_label": labels[test_i], "pred_label": "middle_excluded", "prob_high": np.nan})
                        continue
                best = None
                y_train = labels[train_mask]
                for name, model, params in classifier_candidates(config, int(train_mask.sum()), X.shape[1], task == "extreme_binary"):
                    try:
                        pipe = make_pipe(model, params.get("pca_dim"))
                        pipe.fit(X[train_mask], y_train)
                        train_pred = pipe.predict(X[train_mask])
                        score = balanced_accuracy_score(y_train, train_pred)
                    except Exception:
                        continue
                    if best is None or score > best[0]:
                        best = (score, name, pipe, params)
                if best is None:
                    continue
                _, name, pipe, params = best
                pred = str(pipe.predict(X[te])[0])
                classes = list(pipe.named_steps["model"].classes_)
                probs = pipe.predict_proba(X[te])[0] if hasattr(pipe.named_steps["model"], "predict_proba") else np.zeros(len(classes))
                prob_high = float(probs[classes.index("high")]) if "high" in classes else np.nan
                pred_rows.append(
                    {
                        "sample_id": sample_ids[test_i],
                        "embedding_id": emb_row["embedding_id"],
                        "encoder": emb_row["encoder"],
                        "clip_variant": emb_row["clip_variant"],
                        "preprocessing": emb_row["preprocessing"],
                        "task": task,
                        "true_label": labels[test_i],
                        "pred_label": pred,
                        "prob_high": prob_high,
                        "selected_model": name,
                        "selected_hyperparameters": json.dumps(params, sort_keys=True),
                    }
                )
    pred = pd.DataFrame(pred_rows)
    metrics = []
    for keys, g in pred.groupby(["embedding_id", "task"]):
        if keys[1] == "extreme_binary":
            gg = g[g["true_label"].isin(["low", "high"])].copy()
            if len(gg) < 4:
                continue
            row = {"embedding_id": keys[0], "task": keys[1], "N": len(gg)}
            row.update(score_binary(gg["true_label"].to_numpy(), gg["prob_high"].to_numpy(float)))
            row["confusion_matrix"] = json.dumps(confusion_matrix(gg["true_label"], gg["pred_label"], labels=["low", "high"]).tolist())
        else:
            row = {
                "embedding_id": keys[0],
                "task": keys[1],
                "N": len(g),
                "balanced_accuracy": float(balanced_accuracy_score(g["true_label"], g["pred_label"])),
                "macro_F1": float(f1_score(g["true_label"], g["pred_label"], average="macro")),
                "AUROC": np.nan,
                "AUPRC": np.nan,
                "confusion_matrix": json.dumps(confusion_matrix(g["true_label"], g["pred_label"], labels=["low", "middle", "high"]).tolist()),
            }
        metrics.append(row)
    return pred, pd.DataFrame(metrics).sort_values(["task", "balanced_accuracy"], ascending=[True, False])
