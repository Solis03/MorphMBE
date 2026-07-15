from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .common import repo_path, write_csv, write_json


def medoid_index(X: np.ndarray) -> int:
    d = cdist(X, X)
    return int(np.argmin(d.sum(axis=1)))


def aggregate_scan_latents(manifest: pd.DataFrame, latents: np.ndarray, output_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    scan_rows = []
    for i, row in manifest.reset_index(drop=True).iterrows():
        scan_rows.append({"sample_id": row["sample_id"], "growth_run_id": row["growth_run_id"], "afm_file_id": row["afm_file_id"], **{f"z{j}": float(latents[i, j]) for j in range(latents.shape[1])}})
    scan_df = pd.DataFrame(scan_rows)
    for sid, g in scan_df.groupby("sample_id"):
        z = g[[c for c in scan_df.columns if c.startswith("z")]].to_numpy(float)
        medoid_local = medoid_index(z)
        distances = cdist(z, z)
        row = {
            "sample_id": sid,
            "growth_run_id": str(g["growth_run_id"].iloc[0]),
            "scan_count": len(g),
            "medoid_afm_file_id": str(g.iloc[medoid_local]["afm_file_id"]),
            "within_sample_latent_distance_median": float(np.median(distances[np.triu_indices_from(distances, 1)])) if len(g) > 1 else 0.0,
            "within_sample_latent_distance_iqr": float(np.percentile(distances[np.triu_indices_from(distances, 1)], 75) - np.percentile(distances[np.triu_indices_from(distances, 1)], 25)) if len(g) > 1 else 0.0,
            "latent_covariance_trace": float(np.trace(np.cov(z, rowvar=False))) if len(g) > 1 else 0.0,
        }
        for j in range(z.shape[1]):
            row[f"mean_z{j}"] = float(np.mean(z[:, j]))
            row[f"median_z{j}"] = float(np.median(z[:, j]))
            row[f"medoid_z{j}"] = float(z[medoid_local, j])
        rows.append(row)
    sample_df = pd.DataFrame(rows)
    np.savez_compressed(repo_path(output_root) / "afm_scan_latents.npz", latents=latents, sample_ids=manifest["sample_id"].astype(str).to_numpy(), afm_file_ids=manifest["afm_file_id"].astype(str).to_numpy())
    write_csv(scan_df, repo_path(output_root) / "afm_scan_latents.csv")
    write_csv(sample_df, repo_path(output_root) / "afm_sample_latent_summary.csv")
    return scan_df, sample_df


def sample_descriptor_matrix(manifest: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "unit_ra",
        "unit_robust_height_range",
        "unit_skewness",
        "unit_kurtosis",
        "unit_psd_low_fraction",
        "unit_psd_mid_fraction",
        "unit_psd_high_fraction",
        "unit_psd_slope",
        "unit_autocorr_length_nm",
        "unit_anisotropy_ratio",
        "unit_mean_abs_gradient",
        "unit_rms_gradient",
    ]
    return manifest.groupby("sample_id")[cols].median().reset_index()


def representation_matrix(sample_latents: pd.DataFrame, manifest: pd.DataFrame, representation: str) -> tuple[list[str], np.ndarray]:
    samples = sample_latents["sample_id"].astype(str).tolist()
    latent_cols = [c for c in sample_latents.columns if c.startswith("medoid_z")]
    desc = sample_descriptor_matrix(manifest).set_index("sample_id").loc[samples]
    if representation == "latent":
        X = sample_latents[latent_cols].to_numpy(float)
    elif representation == "descriptors":
        X = desc.to_numpy(float)
    elif representation == "hybrid":
        X = np.concatenate([StandardScaler().fit_transform(sample_latents[latent_cols].to_numpy(float)), StandardScaler().fit_transform(desc.to_numpy(float))], axis=1)
    else:
        raise ValueError(representation)
    return samples, StandardScaler().fit_transform(np.nan_to_num(X))


def cluster_samples(X: np.ndarray, k: int) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


def bootstrap_stability(X: np.ndarray, labels: np.ndarray, n_boot: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = len(labels)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        unique = np.unique(idx)
        if len(unique) <= len(np.unique(labels)):
            continue
        boot_labels = cluster_samples(X[unique], len(np.unique(labels)))
        scores.append(adjusted_rand_score(labels[unique], boot_labels))
    return float(np.nanmedian(scores)) if scores else np.nan


def run_prototypes(manifest: pd.DataFrame, sample_latents: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_root = repo_path(config["output_root"])
    candidate_rows, stability_rows = [], []
    assignment_frames = []
    best = None
    for rep in ("latent", "descriptors", "hybrid"):
        samples, X = representation_matrix(sample_latents, manifest, rep)
        for k in config["prototype_candidate_k"]:
            if int(k) >= len(samples):
                continue
            labels = cluster_samples(X, int(k))
            sil = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else np.nan
            stability = bootstrap_stability(X, labels, int(config["prototype_bootstrap_count"]), int(config["review_random_seed"]) + int(k))
            sizes = pd.Series(labels).value_counts().sort_index().to_dict()
            medoids = {}
            for lab in sorted(set(labels)):
                idx = np.where(labels == lab)[0]
                local = medoid_index(X[idx])
                medoids[f"P{lab}"] = samples[int(idx[local])]
            row = {
                "representation": rep,
                "K": int(k),
                "silhouette": sil,
                "bootstrap_ari_median": stability,
                "cluster_sizes": json.dumps({f"P{k0}": int(v) for k0, v in sizes.items()}, sort_keys=True),
                "medoid_sample_ids": json.dumps(medoids, sort_keys=True),
                "min_cluster_size": int(min(sizes.values())),
            }
            candidate_rows.append(row)
            stability_rows.append({"representation": rep, "K": int(k), "bootstrap_ari_median": stability, "bootstrap_unit": "sample"})
            if best is None or (np.nan_to_num(stability) + np.nan_to_num(sil)) > best[0]:
                best = (np.nan_to_num(stability) + np.nan_to_num(sil), rep, int(k), labels, samples, X, medoids)
    _, best_rep, best_k, labels, samples, X, medoids = best
    sample_to_label = {sid: f"P{int(lab)}" for sid, lab in zip(samples, labels)}
    for sid, g in manifest.groupby("sample_id"):
        pid = sample_to_label[str(sid)]
        purity = 1.0
        hetero = float(sample_latents.loc[sample_latents["sample_id"].astype(str) == str(sid), "within_sample_latent_distance_median"].iloc[0])
        assignment_frames.append(
            {
                "sample_id": str(sid),
                "growth_run_id": str(g["growth_run_id"].iloc[0]),
                "dominant_prototype": pid,
                "prototype_distribution_across_scans": json.dumps({pid: int(len(g))}),
                "prototype_purity": purity,
                "mixed_morphology": bool(hetero > sample_latents["within_sample_latent_distance_median"].quantile(0.75)),
                "within_sample_heterogeneity": hetero,
                "selected_k": best_k,
                "representation": best_rep,
                "is_prototype_medoid": str(sid) in medoids.values(),
            }
        )
    candidates = pd.DataFrame(candidate_rows)
    assignments = pd.DataFrame(assignment_frames)
    stability = pd.DataFrame(stability_rows)
    write_csv(candidates, output_root / "prototype_candidates.csv")
    write_csv(assignments, output_root / "prototype_assignments.csv")
    write_csv(stability, output_root / "prototype_stability.csv")
    template = {"selected_k": None, "representation": None, "prototypes": {f"P{i}": {"accepted": None, "display_name": "", "merge_into": None, "notes": ""} for i in range(best_k)}}
    review_root = repo_path(config["report_root"]) / "prototype_review"
    review_root.mkdir(parents=True, exist_ok=True)
    write_json(template, review_root / "prototype_review_template.json")
    write_csv(assignments.merge(manifest.groupby("sample_id")[["rq_nm", "unit_autocorr_length_nm", "unit_anisotropy_ratio"]].median().reset_index(), on="sample_id"), review_root / "prototype_summary.csv")
    return candidates, assignments, stability
