from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

from .afm_dataset import load_unit_shapes
from .afm_evaluation import reconstruction_metrics, summarize_metrics
from .common import display_path, repo_path, write_csv
from .rq_disentanglement import project_unit_rq_np


def make_group_folds(manifest: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    groups = manifest["growth_run_id"].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=int(config["outer_split"]["folds"]))
    rows = []
    dummy = np.zeros(len(manifest))
    for fold, (train_idx, test_idx) in enumerate(gkf.split(dummy, groups=groups)):
        train_groups = sorted(set(groups[train_idx]))
        test_groups = sorted(set(groups[test_idx]))
        for g in train_groups:
            rows.append({"fold": fold, "growth_run_id": g, "split": "train"})
        for g in test_groups:
            rows.append({"fold": fold, "growth_run_id": g, "split": "test"})
    split = pd.DataFrame(rows)
    write_csv(split, repo_path(config["output_root"]) / "group_outer_splits.csv")
    return split


def run_pca_decoder(manifest: pd.DataFrame, config: dict[str, Any], resolution: int, split: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_root = repo_path(config["output_root"])
    recon_root = output_root / "pca_oof_reconstructions"
    recon_root.mkdir(parents=True, exist_ok=True)
    X_img = load_unit_shapes(manifest, resolution)
    X = X_img.reshape(len(X_img), -1)
    groups = manifest["growth_run_id"].astype(str).to_numpy()
    scan_rows, registry_rows = [], []
    for n_comp in config["pca_components"]:
        if int(n_comp) >= min(X.shape[0], X.shape[1]):
            registry_rows.append({"model_id": f"pca_{n_comp}", "resolution": resolution, "components": n_comp, "status": "skipped", "skip_reason": "components_exceed_training_rank"})
            continue
        recon = np.zeros_like(X_img, dtype=np.float32)
        for fold in sorted(split["fold"].unique()):
            test_groups = set(split.query("fold == @fold and split == 'test'")["growth_run_id"].astype(str))
            test_mask = np.asarray([g in test_groups for g in groups])
            train_mask = ~test_mask
            local_comp = min(int(n_comp), int(train_mask.sum()) - 1, X.shape[1])
            if local_comp < 1:
                continue
            pca = PCA(n_components=local_comp, svd_solver="randomized", random_state=int(config["outer_split"]["random_seed"]))
            pca.fit(X[train_mask])
            pred = pca.inverse_transform(pca.transform(X[test_mask]))
            for local_i, global_i in enumerate(np.where(test_mask)[0]):
                recon[global_i] = project_unit_rq_np(pred[local_i].reshape(resolution, resolution), epsilon=float(config["epsilon"]))
        out_path = recon_root / f"pca_{n_comp}_{resolution}px_oof_reconstructions.npz"
        np.savez_compressed(out_path, reconstructions=recon, sample_ids=manifest["sample_id"].astype(str).to_numpy(), afm_file_ids=manifest["afm_file_id"].astype(str).to_numpy(), resolution=resolution, components=int(n_comp), global_transductive_development_model=False)
        for i, row in manifest.reset_index(drop=True).iterrows():
            metrics = reconstruction_metrics(X_img[i], recon[i], float(row["rq_nm"]))
            scan_rows.append(
                {
                    "model_family": "PCA",
                    "model_id": f"pca_{n_comp}_{resolution}px",
                    "resolution": resolution,
                    "components": int(n_comp),
                    "sample_id": row["sample_id"],
                    "growth_run_id": row["growth_run_id"],
                    "afm_file_id": row["afm_file_id"],
                    "true_q_used_for_afm_side_decoder_evaluation": True,
                    **metrics,
                }
            )
        registry_rows.append(
            {
                "model_id": f"pca_{n_comp}_{resolution}px",
                "resolution": resolution,
                "components": int(n_comp),
                "status": "complete",
                "fit_scope": "outer_train_groups_only",
                "reconstruction_path": display_path(out_path),
            }
        )
    scan_metrics = pd.DataFrame(scan_rows)
    summary = summarize_metrics(scan_metrics, ["model_id", "resolution", "components"]) if not scan_metrics.empty else pd.DataFrame()
    registry = pd.DataFrame(registry_rows)
    write_csv(summary, output_root / "pca_oof_metrics.csv")
    write_csv(registry, output_root / "pca_model_registry.csv")
    return scan_metrics, summary, registry
