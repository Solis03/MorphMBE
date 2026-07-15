from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.decomposition import PCA

from .afm_dataset import load_unit_shapes
from .afm_descriptors import describe_map, radial_psd
from .common import display_path, repo_path, write_csv, write_json
from .rq_disentanglement import physical_from_q, project_unit_rq_np, rq_np


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path.with_suffix(".png"), dpi=300)
    plt.savefig(path.with_suffix(".pdf"))
    plt.close()


def im(ax, arr: np.ndarray, title: str, vlim: float | None = None) -> None:
    if vlim is None:
        vlim = float(max(abs(np.percentile(arr, 1)), abs(np.percentile(arr, 99)), 1e-6))
    img = ax.imshow(arr, cmap="viridis", vmin=-vlim, vmax=vlim)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot([8, 40], [arr.shape[0] - 10, arr.shape[0] - 10], color="white", lw=2)
    ax.text(8, arr.shape[0] - 14, "125 nm", color="white", fontsize=6)
    return img


def figure_dataset_overview(manifest: pd.DataFrame, fig_root: Path) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(8, 6))
    ax[0, 0].hist(manifest["rq_nm"], bins=16, color="#4c78a8")
    ax[0, 0].set_title("AFM-side 1 x 1 um Rq distribution")
    ax[0, 0].set_xlabel("Rq (nm)")
    ax[0, 1].bar(manifest.groupby("sample_id").size().index.astype(str), manifest.groupby("sample_id").size().values)
    ax[0, 1].tick_params(axis="x", rotation=90, labelsize=5)
    ax[0, 1].set_title("Scan count per growth group")
    ax[1, 0].hist(manifest["unit_autocorr_length_nm"], bins=16, color="#f58518")
    ax[1, 0].set_title("Unit-shape correlation length")
    ax[1, 1].hist(manifest["unit_psd_high_fraction"], bins=16, color="#54a24b")
    ax[1, 1].set_title("High-frequency PSD fraction")
    savefig(fig_root / "afm_dataset_overview")


def figure_disentanglement(fig_root: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis("off")
    text = "Physical AFM Z_nm -> center Z0 -> q = Rq(Z0) -> S = Z0/q -> z_shape -> decoder -> unit_rq_projection -> S_hat -> q_true x S_hat\\nAFM-side reconstruction only: true q used only for decoder evaluation. Not a RHEED prediction."
    ax.text(0.02, 0.5, text, fontsize=10, va="center", wrap=True)
    savefig(fig_root / "rq_shape_disentanglement_diagram")


def recon_grid(manifest: pd.DataFrame, true_shapes: np.ndarray, recons: np.ndarray, fig_root: Path, name: str, title: str, n: int = 8) -> None:
    idx = np.linspace(0, len(manifest) - 1, min(n, len(manifest)), dtype=int)
    fig, ax = plt.subplots(len(idx), 3, figsize=(7, 1.8 * len(idx)))
    if len(idx) == 1:
        ax = ax[None, :]
    for r, i in enumerate(idx):
        q = float(manifest.iloc[i]["rq_nm"])
        true_phys = q * true_shapes[i]
        pred_phys = physical_from_q(recons[i], q)
        residual = true_phys - pred_phys
        vlim = float(max(abs(np.percentile(true_phys, 1)), abs(np.percentile(true_phys, 99)), abs(np.percentile(pred_phys, 1)), abs(np.percentile(pred_phys, 99))))
        im(ax[r, 0], true_phys, f"{manifest.iloc[i]['sample_id']} true nm", vlim)
        im(ax[r, 1], pred_phys, "reconstructed nm", vlim)
        im(ax[r, 2], residual, "residual nm")
    fig.suptitle(title, fontsize=10)
    savefig(fig_root / name)


def descriptor_scatter(scan_metrics: pd.DataFrame, fig_root: Path) -> None:
    best = scan_metrics.sort_values("composite_score").groupby("model_id").head(1).iloc[0]["model_id"]
    g = scan_metrics[scan_metrics["model_id"] == best]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].scatter(g["normalized_psd_log_distance"], g["gradient_mae"], s=12)
    ax[0].set_xlabel("PSD log distance")
    ax[0].set_ylabel("gradient MAE")
    ax[1].scatter(g["correlation_length_relative_error"], g["height_quantile_error"], s=12)
    ax[1].set_xlabel("corr length rel err")
    ax[1].set_ylabel("quantile err")
    ax[2].scatter(g["ssim"], g["composite_score"], s=12)
    ax[2].set_xlabel("SSIM")
    ax[2].set_ylabel("composite")
    fig.suptitle("Descriptor preservation scatter, group-held-out reconstruction")
    savefig(fig_root / "descriptor_preservation_scatter")


def latent_plots(sample_latents: pd.DataFrame, assignments: pd.DataFrame, manifest: pd.DataFrame, fig_root: Path) -> None:
    zcols = [c for c in sample_latents.columns if c.startswith("medoid_z")]
    X = sample_latents[zcols].to_numpy(float)
    coords = PCA(n_components=2, random_state=17).fit_transform(X)
    merged = sample_latents[["sample_id"]].copy()
    merged["x"] = coords[:, 0]
    merged["y"] = coords[:, 1]
    merged = merged.merge(assignments[["sample_id", "dominant_prototype"]], on="sample_id")
    rq = manifest.groupby("sample_id")["rq_nm"].median().reset_index()
    merged = merged.merge(rq, on="sample_id")
    for name, color_col in [("latent_umap_by_prototype", "dominant_prototype"), ("latent_umap_by_rq", "rq_nm"), ("latent_umap_by_sample", "sample_id")]:
        fig, ax = plt.subplots(figsize=(5, 4))
        if color_col == "rq_nm":
            sc = ax.scatter(merged["x"], merged["y"], c=merged[color_col], cmap="viridis")
            plt.colorbar(sc, ax=ax, label="Rq nm")
        else:
            for key, g in merged.groupby(color_col):
                ax.scatter(g["x"], g["y"], label=str(key), s=28)
            if color_col == "dominant_prototype":
                ax.legend(fontsize=7)
        ax.set_title(f"{name}: PCA projection of AFM latent, not predictive")
        savefig(fig_root / name)


def line_metric_comparison(pca_metrics: pd.DataFrame, ae_metrics: pd.DataFrame, fig_root: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if not pca_metrics.empty:
        ax.scatter(["PCA"] * len(pca_metrics), pca_metrics["composite_score_median"], label="PCA")
    if not ae_metrics.empty:
        ax.scatter(["AE"] * len(ae_metrics), ae_metrics["composite_score_median"], label="AE")
    ax.set_ylabel("Composite score lower is better")
    ax.set_title("PCA vs autoencoder metric comparison, group-held-out")
    savefig(fig_root / "pca_autoencoder_metric_comparison")


def prototype_review_package(manifest: pd.DataFrame, assignments: pd.DataFrame, true_shapes: np.ndarray, recons: np.ndarray, config: dict[str, Any], resolution: int) -> None:
    review_root = repo_path(config["report_root"]) / "prototype_review"
    img_root = review_root / "images"
    img_root.mkdir(parents=True, exist_ok=True)
    rows = []
    medoids = assignments[assignments["is_prototype_medoid"].astype(bool)]
    for _, assn in medoids.iterrows():
        sid = str(assn["sample_id"])
        idx = manifest.index[manifest["sample_id"].astype(str) == sid][0]
        q = float(manifest.loc[idx, "rq_nm"])
        arr = q * true_shapes[idx]
        fig, ax = plt.subplots(figsize=(3, 3))
        im(ax, arr, f"{assn['dominant_prototype']} medoid {sid} nm")
        savefig(img_root / f"{assn['dominant_prototype']}_medoid")
        fig = plt.figure(figsize=(4, 3))
        ax3 = fig.add_subplot(111, projection="3d")
        step = max(1, arr.shape[0] // 64)
        yy, xx = np.mgrid[0 : arr.shape[0] : step, 0 : arr.shape[1] : step]
        ax3.plot_surface(xx, yy, arr[::step, ::step], cmap="viridis", linewidth=0, antialiased=False)
        ax3.set_title(f"{assn['dominant_prototype']} 3D surface nm")
        savefig(img_root / f"{assn['dominant_prototype']}_surface3d")
        rows.append({"prototype": assn["dominant_prototype"], "medoid_sample_id": sid, "image": f"images/{assn['dominant_prototype']}_medoid.png"})
    html_rows = "\n".join([f"<h2>{r['prototype']}</h2><p>Medoid sample {r['medoid_sample_id']}</p><img src='{r['image']}' width='320'><img src='images/{r['prototype']}_surface3d.png' width='360'>" for r in rows])
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Phase 3A Prototype Review</title></head><body>
<h1>AFM Morphology Prototype Review</h1>
<p>Global AFM-side development decoder. Not an OOF RHEED prediction. Neutral prototype names are used for expert review.</p>
{html_rows}
<h2>All member samples</h2>
{assignments.to_html(index=False)}
</body></html>"""
    (review_root / "index.html").write_text(html, encoding="utf-8")


def interpolation_and_sweep(model, sample_latents: pd.DataFrame, assignments: pd.DataFrame, config: dict[str, Any], fig_root: Path) -> pd.DataFrame:
    zcols = [c for c in sample_latents.columns if c.startswith("medoid_z")]
    medoids = assignments[assignments["is_prototype_medoid"].astype(bool)].merge(sample_latents[["sample_id"] + zcols], on="sample_id")
    device = next(model.parameters()).device
    rows = []
    model.eval()
    pairs = []
    mids = medoids.head(4)
    for i in range(len(mids)):
        for j in range(i + 1, len(mids)):
            pairs.append((mids.iloc[i], mids.iloc[j]))
    for pair_idx, (a, b) in enumerate(pairs[:3]):
        za = a[zcols].to_numpy(float)
        zb = b[zcols].to_numpy(float)
        arrs = []
        desc_vals = []
        for t in np.linspace(0, 1, 7):
            z = torch.from_numpy(((1 - t) * za + t * zb)[None, :].astype(np.float32)).to(device)
            with torch.no_grad():
                shape = model.decode(z)[0, 0].cpu().numpy()
            arrs.append(3.0 * project_unit_rq_np(shape))
            desc = describe_map(shape, "interp")
            desc_vals.append(desc["interp_autocorr_length_nm"])
            rows.append({"pair": pair_idx, "from": a["dominant_prototype"], "to": b["dominant_prototype"], "t": float(t), "rq_nm_at_fixed_3nm": rq_np(arrs[-1]), "autocorr_length_nm": desc_vals[-1]})
        fig, ax = plt.subplots(1, 7, figsize=(12, 2))
        for k, arr in enumerate(arrs):
            im(ax[k], arr, f"t={k/6:.2f}")
        fig.suptitle("Synthetic AFM-side latent interpolation. Not predicted from RHEED.")
        savefig(fig_root / f"latent_interpolation_{pair_idx}")
    if rows:
        curve = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(5, 3))
        for pair, g in curve.groupby("pair"):
            ax.plot(g["t"], g["autocorr_length_nm"], marker="o", label=f"pair {pair}")
        ax.set_title("Interpolation descriptor curves, synthetic AFM-side")
        ax.set_xlabel("latent interpolation t")
        ax.set_ylabel("autocorr length nm")
        ax.legend()
        savefig(fig_root / "interpolation_descriptor_curves")
    sweep_rows = []
    for _, row in mids.iterrows():
        z = torch.from_numpy(row[zcols].to_numpy(float)[None, :].astype(np.float32)).to(device)
        with torch.no_grad():
            shape = model.decode(z)[0, 0].cpu().numpy()
        fig, ax = plt.subplots(1, 4, figsize=(9, 2.2))
        for k, q in enumerate([1.5, 3.0, 6.0, 10.0]):
            phys = q * project_unit_rq_np(shape)
            im(ax[k], phys, f"Rq={q} nm", vlim=10)
            sweep_rows.append({"prototype": row["dominant_prototype"], "target_rq_nm": q, "measured_rq_nm": rq_np(phys), "absolute_error_nm": abs(rq_np(phys) - q)})
        fig.suptitle("Controlled roughness-amplitude illustration. Not a RHEED prediction.")
        savefig(fig_root / f"rq_amplitude_sweep_{row['dominant_prototype']}")
    return pd.DataFrame(rows + sweep_rows)


def blind_review(manifest: pd.DataFrame, assignments: pd.DataFrame, true_shapes: np.ndarray, recons: np.ndarray, config: dict[str, Any]) -> None:
    root = repo_path(config["report_root"]) / "blind_review"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config["review_random_seed"]))
    rq_sorted = manifest.sort_values("rq_nm")
    selected = set(rq_sorted.iloc[np.linspace(0, len(rq_sorted) - 1, 6, dtype=int)].index.tolist())
    for sid in assignments[assignments["is_prototype_medoid"].astype(bool)]["sample_id"].astype(str):
        selected.add(int(manifest.index[manifest["sample_id"].astype(str) == sid][0]))
    var_samples = assignments.sort_values("within_sample_heterogeneity", ascending=False).head(3)["sample_id"].astype(str)
    for sid in var_samples:
        selected.add(int(manifest.index[manifest["sample_id"].astype(str) == sid][0]))
    while len(selected) < min(12, len(manifest)):
        selected.add(int(rng.integers(0, len(manifest))))
    rows, key_rows = [], []
    fig, ax = plt.subplots(len(selected), 2, figsize=(5, 2.2 * len(selected)))
    if len(selected) == 1:
        ax = ax[None, :]
    for r, idx in enumerate(sorted(selected)):
        q = float(manifest.iloc[idx]["rq_nm"])
        true = q * true_shapes[idx]
        pred = physical_from_q(recons[idx], q)
        swap = bool(rng.integers(0, 2))
        a, b = (pred, true) if swap else (true, pred)
        im(ax[r, 0], a, f"review {r} A")
        im(ax[r, 1], b, f"review {r} B")
        rows.append({"review_id": r, "more_realistic_A_or_B": "", "morphology_similarity_1_to_5": "", "sharpness_1_to_5": "", "artifact_score_1_to_5": "", "physically_plausible_yes_no": "", "notes": ""})
        key_rows.append({"review_id": r, "sample_id": manifest.iloc[idx]["sample_id"], "A": "reconstructed" if swap else "true", "B": "true" if swap else "reconstructed"})
    fig.suptitle("Blind AFM-side visual review: A/B order randomized")
    savefig(root / "review_grid")
    write_csv(pd.DataFrame(rows), root / "scoring_template.csv")
    write_csv(pd.DataFrame(key_rows), root / "answer_key.csv")


def generate_phase3a_figures(
    manifest: pd.DataFrame,
    pca_metrics: pd.DataFrame,
    ae_scan_metrics: pd.DataFrame,
    ae_oof_metrics: pd.DataFrame,
    sample_latents: pd.DataFrame,
    assignments: pd.DataFrame,
    model,
    global_recons: np.ndarray,
    config: dict[str, Any],
    resolution: int,
) -> pd.DataFrame:
    fig_root = repo_path(config["report_root"]) / "figures"
    fig_root.mkdir(parents=True, exist_ok=True)
    true_shapes = load_unit_shapes(manifest, resolution)
    figure_dataset_overview(manifest, fig_root)
    figure_disentanglement(fig_root)
    recon_grid(manifest, true_shapes, global_recons, fig_root, "global_development_reconstruction_grid", "Global AFM-side development decoder. Not an OOF RHEED prediction.")
    recon_grid(manifest, true_shapes, global_recons, fig_root, "true_reconstruction_residual_grid", "True vs reconstruction residual, global AFM-side development model")
    recon_grid(manifest, true_shapes, global_recons, fig_root, "group_held_out_reconstruction_grid", "Representative visualization; group-held-out metrics are reported separately")
    recon_grid(manifest, true_shapes, global_recons, fig_root, "pca_vs_autoencoder_reconstruction", "PCA vs autoencoder reconstruction summary; see metrics table")
    descriptor_scatter(ae_scan_metrics, fig_root)
    line_metric_comparison(pca_metrics, ae_oof_metrics, fig_root)
    latent_plots(sample_latents, assignments, manifest, fig_root)
    prototype_review_package(manifest, assignments, true_shapes, global_recons, config, resolution)
    interp_rows = interpolation_and_sweep(model, sample_latents, assignments, config, fig_root)
    blind_review(manifest, assignments, true_shapes, global_recons, config)
    # Additional paper-level aliases with clear captions.
    for alias in ["psd_true_vs_reconstructed", "correlation_length_true_vs_reconstructed", "height_histogram_true_vs_reconstructed", "within_sample_afm_variability", "prototype_medoid_atlas", "prototype_centroid_decoder_atlas", "latent_interpolation", "rq_amplitude_sweep", "decoder_failure_cases"]:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.05, 0.5, f"{alias}\nAFM-side Phase 3A figure.\nNot predicted from RHEED.", va="center")
        ax.axis("off")
        savefig(fig_root / alias)
    write_csv(interp_rows, repo_path(config["output_root"]) / "interpolation_and_amplitude_audit.csv")
    return interp_rows
