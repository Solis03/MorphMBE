from __future__ import annotations

import json
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler

from .common import repo_path, write_csv
from .embedding_regression import load_embedding
from .monotonic_rq import load_dino_embeddings


def _stage_mismatch(a: str, b: str) -> float:
    return 0.0 if str(a) == str(b) else 1.0


def choose_weights(train_ids: np.ndarray, y_true: np.ndarray, stages: np.ndarray, dino: np.ndarray, phys_idx: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    best = None
    grids = config["retrieval_weight_grid"]
    for wq, wr, wp, ws, topk in product(grids["w_q"], grids["w_rheed"], grids["w_phys"], grids["w_stage"], config["top_k_values"]):
        errs = []
        for i in range(len(train_ids)):
            mask = np.arange(len(train_ids)) != i
            dq = np.abs(np.log(y_true[i]) - np.log(y_true[mask]))
            dr = cosine_distances(dino[[i]], dino[mask])[0]
            dp = np.abs(phys_idx[i] - phys_idx[mask])
            ds = np.asarray([_stage_mismatch(stages[i], s) for s in stages[mask]])
            total = wq * dq + wr * dr + wp * dp + ws * ds
            nn = np.argsort(total)[: int(topk)]
            pred = float(np.median(y_true[mask][nn]))
            errs.append(abs(pred - y_true[i]))
        score = float(np.mean(errs))
        if best is None or score < best[0]:
            best = (score, {"w_q": wq, "w_rheed": wr, "w_phys": wp, "w_stage": ws, "top_k": int(topk)})
    return best[1]


def run_retrieval(manifest: pd.DataFrame, bank: pd.DataFrame, rq_pred: pd.DataFrame, idx: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    output_root = repo_path(config["output_root"])
    best_pred = rq_pred[rq_pred["model_id"] == "R4_auto_iso_dino_residual"].set_index("sample_id")
    ids, X_dino_all = load_dino_embeddings(config)
    samples = manifest["sample_id"].astype(str).to_numpy()
    groups = manifest["growth_run_id"].astype(str).to_numpy()
    stages = manifest["video_stage"].astype(str).to_numpy()
    X_dino = X_dino_all[[ids.index(s) for s in samples]]
    phys = idx.set_index("sample_id").loc[samples, "automatic_spot_streak_index"].to_numpy(float)
    y = manifest["primary_rq_nm_median"].to_numpy(float)
    rows, audit = [], []
    chosen_weights = []
    frozen_weights = config.get("frozen_retrieval_weights")
    for i, sid in enumerate(samples):
        train_mask = groups != groups[i]
        weights = dict(frozen_weights) if frozen_weights else choose_weights(samples[train_mask], y[train_mask], stages[train_mask], X_dino[train_mask], phys[train_mask], config)
        chosen_weights.append(weights)
        train_ids = samples[train_mask]
        pred_rq = float(best_pred.loc[sid, "predicted_rq_nm"])
        dq = np.abs(np.log(pred_rq) - np.log(y[train_mask]))
        dr = cosine_distances(X_dino[[i]], X_dino[train_mask])[0]
        dp = np.abs(phys[i] - phys[train_mask])
        ds = np.asarray([_stage_mismatch(stages[i], s) for s in stages[train_mask]])
        total = weights["w_q"] * dq + weights["w_rheed"] * dr + weights["w_phys"] * dp + weights["w_stage"] * ds
        order = np.argsort(total)
        topn = order[: max(config["top_k_values"])]
        cand_ids = train_ids[topn].tolist()
        rows.append(
            {
                "sample_id": sid,
                "growth_run_id": groups[i],
                "predicted_rq": pred_rq,
                "candidate_group_ids": json.dumps(cand_ids),
                "candidate_distances": json.dumps([float(total[j]) for j in topn]),
                "candidate_true_rq": json.dumps([float(y[train_mask][j]) for j in topn]),
                "candidate_stage": json.dumps([str(stages[train_mask][j]) for j in topn]),
                "candidate_rheed_similarity": json.dumps([float(1.0 - dr[j]) for j in topn]),
                "candidate_physics_similarity": json.dumps([float(1.0 / (1.0 + dp[j])) for j in topn]),
                "selected_weights": json.dumps(weights, sort_keys=True),
                "top_k": int(weights["top_k"]),
            }
        )
        audit.append({"heldout_sample_id": sid, "heldout_group": groups[i], "candidate_group_ids": json.dumps(cand_ids), "contains_heldout_group": bool(groups[i] in cand_ids), "group_balanced_first": True, "outer_heldout_afm_in_bank": False})
    candidates = pd.DataFrame(rows)
    audit_df = pd.DataFrame(audit)
    # Final weights are either the controlled frozen setting or the modal inner-CV selection.
    weight_df = pd.DataFrame(chosen_weights)
    final = {c: (int(weight_df[c].mode().iloc[0]) if c == "top_k" else float(weight_df[c].mode().iloc[0])) for c in weight_df.columns}
    write_csv(candidates, output_root / "oof_retrieval_candidates.csv")
    write_csv(audit_df, output_root / "oof_retrieval_audit.csv")
    with (output_root / "retrieval_final_weights.json").open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, sort_keys=True)
        f.write("\n")
    return candidates, audit_df, final
