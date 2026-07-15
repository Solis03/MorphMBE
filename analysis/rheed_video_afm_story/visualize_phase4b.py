from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .afm_rendering import display_limits, load_physical_map, render_afm, render_rheed
from .common import repo_path, write_csv
from .publication_style import METHOD_COLORS, save_figure, set_publication_style
from .results_data_assembly import quantile_sample_selection


def load_keyframe(sample_id: str) -> np.ndarray:
    z = np.load(repo_path(f"outputs/rheed_video_afm_story/phase2a/clip_variants/keyframe_1/{sample_id}.npz"))
    return z["frames_uint8"][0]


def afm_triplet(row: pd.Series) -> list[np.ndarray]:
    gt = load_physical_map(repo_path(row["ground_truth_afm_path"]))
    s1 = load_physical_map(repo_path(f"outputs/rheed_video_afm_story/phase4a/synthesized_afm_maps/{row['sample_id']}_S1_top1_real_exemplar_retrieval.npy"))
    s4 = load_physical_map(repo_path(row["s4_output_path"]))
    return [gt, s1, s4]


def fig1_pipeline(config: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")
    boxes = [
        (0.04, 0.72, "Raw RHEED MP4\nPNG frames\nmanual ROI", "#d8ecff"),
        (0.28, 0.82, "Physics features\nspot / streak / connection\ndiffuse / stability", "#d8ecff"),
        (0.28, 0.62, "Frozen DINO embedding", "#d8ecff"),
        (0.52, 0.72, "R4 Rq estimator\nautomatic calibration\n+ DINO residual", "#eadcf8"),
        (0.74, 0.72, "Predicted Rq\nsupport / abstention", "#e1f5df"),
        (0.04, 0.28, "Training-group AFM\nheight maps", "#ffe2bf"),
        (0.28, 0.28, "Representative AFM bank\nreal patch bank", "#ffe2bf"),
        (0.52, 0.28, "Distance\n2.0x predicted Rq\n0.25x DINO\n0.25x physics", "#eadcf8"),
        (0.74, 0.28, "S1 real exemplar\nS4 patch synthesis\npredicted-Rq scaling", "#e1f5df"),
        (0.52, 0.08, "Leave-one-growth-group-out\nheld-out group excluded\nfrom fitting and bank", "#f6d0d0"),
    ]
    for x, y, text, color in boxes:
        ax.text(x, y, text, transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="#444", lw=1.1), fontsize=10, va="center")
    arrows = [((0.20, 0.72), (0.28, 0.82)), ((0.20, 0.72), (0.28, 0.62)), ((0.43, 0.72), (0.52, 0.72)), ((0.66, 0.72), (0.74, 0.72)), ((0.20, 0.28), (0.28, 0.28)), ((0.43, 0.28), (0.52, 0.28)), ((0.66, 0.28), (0.74, 0.28)), ((0.74, 0.68), (0.78, 0.34))]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.3, color="#444"))
    ax.plot([0.48, 0.48], [0.02, 0.94], "--", color="#b23b3b", lw=1.6, transform=ax.transAxes)
    ax.text(0.50, 0.94, "leakage barrier", color="#b23b3b", transform=ax.transAxes, fontsize=10)
    ax.text(0.74, 0.12, "Representative morphology\nNot exact local AFM reconstruction", transform=ax.transAxes, fontsize=11, color="#145a32")
    save_figure(fig, repo_path(config["report_root"]) / "figures/Fig1_pipeline_and_data_flow", dpi=config["figure"]["dpi_main"])


def comparison_figure(sample_df: pd.DataFrame, config: dict[str, Any], robust: bool, name: str, selected: list[str]) -> list[dict[str, Any]]:
    records = []
    rows = sample_df.set_index("sample_id").loc[selected].reset_index()
    fig, axes = plt.subplots(len(rows), 6, figsize=(15, 2.4 * len(rows)), constrained_layout=True)
    if len(rows) == 1:
        axes = axes[None, :]
    mode = "per_image" if robust else "row_shared"
    for r, row in rows.iterrows():
        sid = row["sample_id"]
        axes[r, 0].axis("off")
        axes[r, 0].text(0.05, 0.7, f"{sid}\n{row['support_level']}\nTrue Rq {row['true_rq_nm']:.2f}\nPred Rq {row['predicted_rq_nm']:.2f}", fontsize=8, va="top")
        render_rheed(axes[r, 1], load_keyframe(sid), "RHEED keyframe")
        arrays = afm_triplet(row)
        limits = display_limits(arrays, mode)
        titles = [
            f"Ground-truth AFM\nRq={row['true_rq_nm']:.2f} nm\nRa={row['true_ra_nm']:.2f} nm",
            f"Retrieved real AFM exemplar\nsource={row['s1_source_sample_id']}\nRq={row['s1_output_rq_nm']:.2f} nm",
            f"Retrieval-augmented\nrepresentative AFM\nRq={row['s4_output_rq_nm']:.2f} nm",
        ]
        for c, (arr, lim, title, kind) in enumerate(zip(arrays, limits, titles, ["gt", "s1", "s4"]), start=2):
            rec = render_afm(axes[r, c], arr, title, lim[0], lim[1], cmap=config["figure"]["colormap"], scan_size_nm=config["figure"]["scan_size_nm"], bar_nm=config["figure"]["scale_bar_nm"])
            rec.update({"sample_id": sid, "figure": name, "kind": "afm", "panel": kind, "scale_mode": mode})
            records.append(rec)
        axes[r, 5].axis("off")
        axes[r, 5].text(
            0.02,
            0.95,
            f"|Rq err| {row['rq_absolute_error_nm']:.2f} nm\nTrue corr {row['true_correlation_length_nm']:.1f} nm\nS1 PSD {row['s1_psd_distance']:.2f}\nS4 PSD {row['s4_psd_distance']:.2f}\nS4 corr err {row['s4_correlation_length_relative_error']:.2f}",
            va="top",
            fontsize=8,
        )
    suffix = "_per_image_robust_scale" if robust else "_row_shared_scale"
    save_figure(fig, repo_path(config["report_root"]) / f"figures/{name}{suffix}", dpi=config["figure"]["dpi_main"])
    return records


def fig3_rq(sample_df: pd.DataFrame, model_summary: pd.DataFrame, config: dict[str, Any]) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    high = sample_df["high_confidence"].astype(bool)
    ax[0, 0].scatter(sample_df.loc[high, "true_rq_nm"], sample_df.loc[high, "predicted_rq_nm"], c="#1b9e77", label="high")
    ax[0, 0].scatter(sample_df.loc[~high, "true_rq_nm"], sample_df.loc[~high, "predicted_rq_nm"], facecolors="none", edgecolors="#666", label="abstain/low")
    lim = [min(sample_df["true_rq_nm"].min(), sample_df["predicted_rq_nm"].min()), max(sample_df["true_rq_nm"].max(), sample_df["predicted_rq_nm"].max())]
    ax[0, 0].plot(lim, lim, "k--")
    for _, row in sample_df.iterrows():
        ax[0, 0].text(row["true_rq_nm"], row["predicted_rq_nm"], row["sample_id"], fontsize=6)
    r4 = model_summary[model_summary["model"].eq("R4_auto_iso_dino_residual")].iloc[0]
    ax[0, 0].set_title(f"OOF Rq prediction\nMAE {r4['MAE']:.2f}, Spearman {r4['Spearman']:.2f}")
    ax[0, 0].set_xlabel("True Rq nm")
    ax[0, 0].set_ylabel("Predicted Rq nm")
    ordered = sample_df.sort_values("true_rq_nm")
    ax[0, 1].bar(ordered["sample_id"], ordered["rq_absolute_error_nm"], color=np.where(ordered["high_confidence"], "#1b9e77", "#999"))
    ax[0, 1].axhline(ordered["rq_absolute_error_nm"].median(), color="k", ls="--")
    ax[0, 1].tick_params(axis="x", rotation=90)
    ax[0, 1].set_ylabel("|error| nm")
    high_row = model_summary[model_summary["model"].eq("R4_high_confidence_subset")].iloc[0]
    ax[1, 0].bar(["all", "high"], [r4["MAE"], high_row["MAE"]], color=["#4c78a8", "#1b9e77"])
    ax[1, 0].set_title(f"Coverage high={high_row['coverage']:.2f}")
    ax[1, 0].set_ylabel("MAE nm")
    ax[1, 1].plot(ordered["sample_id"], ordered["true_rq_nm"], marker="o", label="true")
    ax[1, 1].plot(ordered["sample_id"], ordered["predicted_rq_nm"], marker="o", label="pred")
    for i, (_, row) in enumerate(ordered.iterrows()):
        if not row["high_confidence"]:
            ax[1, 1].axvspan(i - 0.5, i + 0.5, color="#ddd", alpha=0.4)
    ax[1, 1].tick_params(axis="x", rotation=90)
    ax[1, 1].legend()
    save_figure(fig, repo_path(config["report_root"]) / "figures/Fig3_rq_prediction_performance", dpi=config["figure"]["dpi_main"])
    # log version
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(np.log10(sample_df["true_rq_nm"]), np.log10(sample_df["predicted_rq_nm"]))
    ax.set_xlabel("log10 true Rq")
    ax.set_ylabel("log10 predicted Rq")
    save_figure(fig, repo_path(config["report_root"]) / "figures/Fig3_rq_prediction_performance_log", dpi=config["figure"]["dpi_main"])


def generic_figures(sample_df: pd.DataFrame, model_summary: pd.DataFrame, data: dict[str, Any], config: dict[str, Any]) -> None:
    fig_root = repo_path(config["report_root"]) / "figures"
    # Fig4
    ordered = sample_df.sort_values("true_rq_nm")
    fig, ax = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for i, col in enumerate(["spot_summary", "streak_summary", "connection_summary", "diffuse_summary"]):
        a = ax.flat[i]
        a.scatter(sample_df[col], sample_df["true_rq_nm"])
        rho = spearmanr(sample_df[col], sample_df["true_rq_nm"]).statistic
        a.set_title(f"{col} vs Rq\nrho={rho:.2f}")
        a.set_xlabel(col)
        a.set_ylabel("Rq nm")
    save_figure(fig, fig_root / "Fig4_rheed_morphology_roughness_continuum", dpi=config["figure"]["dpi_main"])
    # Fig5
    desc_cols = ["true_rq_nm", "true_ra_nm", "true_robust_height_range_nm", "true_psd_high_fraction", "true_correlation_length_nm", "true_anisotropy", "true_skewness", "true_kurtosis"]
    M = ordered[desc_cols].astype(float)
    Z = (M - M.median()) / (M.quantile(0.75) - M.quantile(0.25)).replace(0, np.nan)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    im = ax[0].imshow(Z.fillna(0), aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    ax[0].set_yticks(range(len(ordered)))
    ax[0].set_yticklabels(ordered["sample_id"], fontsize=6)
    ax[0].set_xticks(range(len(desc_cols)))
    ax[0].set_xticklabels(desc_cols, rotation=45, ha="right", fontsize=6)
    plt.colorbar(im, ax=ax[0])
    corr = M.corr(method="spearman")
    im2 = ax[1].imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax[1].set_xticks(range(len(desc_cols)))
    ax[1].set_yticks(range(len(desc_cols)))
    ax[1].set_xticklabels(desc_cols, rotation=45, ha="right", fontsize=6)
    ax[1].set_yticklabels(desc_cols, fontsize=6)
    plt.colorbar(im2, ax=ax[1])
    for col in ["true_rq_nm", "true_ra_nm", "true_correlation_length_nm", "true_psd_high_fraction", "true_anisotropy"]:
        ax[2].scatter([col] * len(sample_df), sample_df[col], s=12)
    ax[2].tick_params(axis="x", rotation=45)
    save_figure(fig, fig_root / "Fig5_afm_descriptor_landscape", dpi=config["figure"]["dpi_main"])
    # Fig6
    rcols = ["automatic_spot_streak_index", "spot_summary", "streak_summary", "connection_summary", "diffuse_summary", "temporal_stability"]
    cmat = pd.DataFrame(index=rcols, columns=desc_cols, dtype=float)
    ci_rows = []
    rng = np.random.default_rng(config["random_seed"])
    for r in rcols:
        for c in desc_cols:
            vals = []
            for _ in range(500):
                idx = rng.integers(0, len(sample_df), len(sample_df))
                vals.append(spearmanr(sample_df[r].iloc[idx], sample_df[c].iloc[idx]).statistic)
            cmat.loc[r, c] = spearmanr(sample_df[r], sample_df[c]).statistic
            ci_rows.append({"rheed_feature": r, "afm_descriptor": c, "rho": cmat.loc[r, c], "ci_low": np.nanquantile(vals, 0.025), "ci_high": np.nanquantile(vals, 0.975), "N": len(sample_df)})
    write_csv(cmat.reset_index(names="rheed_feature"), repo_path(config["output_root"]) / "rheed_afm_spearman_matrix.csv")
    write_csv(pd.DataFrame(ci_rows), repo_path(config["output_root"]) / "rheed_afm_bootstrap_ci.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(cmat.astype(float), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(desc_cols)))
    ax.set_xticklabels(desc_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(rcols)))
    ax.set_yticklabels(rcols)
    for i in range(len(rcols)):
        for j in range(len(desc_cols)):
            ax.text(j, i, f"{cmat.iloc[i,j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax)
    save_figure(fig, fig_root / "Fig6_rheed_afm_feature_relationships", dpi=config["figure"]["dpi_main"])
    # Fig7
    synth = data["phase4a_synthesis_metrics"].copy()
    metrics = ["normalized_psd_log_distance", "correlation_length_relative_error", "height_histogram_wasserstein", "anisotropy_error", "ra_error", "robust_height_range_error"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, metric in zip(axes.flat, metrics):
        labels = []
        vals = []
        for method in config["afm_output_methods"]:
            labels.append(method.split("_")[0])
            vals.append(synth[synth["method"] == method][metric].dropna().to_numpy())
        ax.boxplot(vals, labels=labels, showfliers=False)
        for i, v in enumerate(vals, start=1):
            ax.scatter(np.full(len(v), i), v, s=8, alpha=0.45)
        ax.set_title(metric)
    save_figure(fig, fig_root / "Fig7_afm_output_method_comparison", dpi=config["figure"]["dpi_main"])
    # Fig8
    same = data["same_growth"] if "same_growth" in data else read_csv_safe("outputs/rheed_video_afm_story/phase4a/same_growth_afm_similarity.csv")
    s4 = synth[synth["method"] == "S4_calibrated_patch_synthesis"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    ax[0].boxplot([same["raw_ssim"], same["translation_aligned_ssim"], same["multiscale_ssim"]], labels=["raw", "aligned", "MS"])
    ax[0].set_title("same-growth real AFM SSIM")
    ax[1].hist(same["normalized_psd_distance"], bins=10)
    ax[1].set_title("same-growth PSD distance")
    ax[2].hist(s4["ssim"], bins=10)
    ax[2].set_title("GT vs S4 SSIM")
    save_figure(fig, fig_root / "Fig8_same_growth_similarity_ceiling", dpi=config["figure"]["dpi_main"])
    # Fig9 and FigS2 placeholders with actual summary text
    for fname, text in {
        "Fig9_s4_provenance_and_identity_audit": f"S4 patch provenance: max source contribution {sample_df['s4_largest_source_contribution'].max():.2f}; held-out contribution {sample_df['s4_heldout_source_contribution'].max():.1f}; exact equality {sample_df['s4_exact_pixel_equality'].any()}",
        "FigS2_why_neural_decoder_was_not_used": "Phase 3A compact AE produced low SSIM and poor descriptor preservation; final visual output uses retrieval/patch synthesis.",
    }.items():
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.text(0.05, 0.5, text, va="center", wrap=True)
        ax.axis("off")
        save_figure(fig, fig_root / fname, dpi=config["figure"]["dpi_supplement"])
    # one-page
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    ax.text(0.03, 0.9, "Current RHEED-video-to-AFM Results", fontsize=18, weight="bold")
    ax.text(0.03, 0.75, "Supported: group-held-out evaluation; RHEED-conditioned retrieval; predicted-Rq-scaled representative morphology; S4 provenance.", fontsize=12)
    ax.text(0.03, 0.62, "Not yet supported: exact local AFM reconstruction; robust monotonic spotty-to-Rq relation; neural generative superiority.", fontsize=12)
    ax.text(0.03, 0.48, f"R4 MAE {model_summary[model_summary['model'].eq('R4_auto_iso_dino_residual')].iloc[0]['MAE']:.2f} nm; high-confidence coverage {model_summary[model_summary['model'].eq('R4_high_confidence_subset')].iloc[0]['coverage']:.2f}", fontsize=12)
    save_figure(fig, fig_root / "one_page_current_project_summary", dpi=config["figure"]["dpi_main"])


def read_csv_safe(path: str) -> pd.DataFrame:
    return pd.read_csv(repo_path(path))


def atlas(sample_df: pd.DataFrame, config: dict[str, Any], robust: bool, order: str) -> int:
    rows = sample_df.sort_values("true_rq_nm" if order == "rq" else "sample_id")["sample_id"].tolist()
    per_page = 4
    page_root = repo_path(config["report_root"]) / f"FigS1_all_23_samples_atlas_pages_{order}_{'robust' if robust else 'shared'}"
    page_root.mkdir(parents=True, exist_ok=True)
    pages = 0
    for start in range(0, len(rows), per_page):
        selected = rows[start : start + per_page]
        records = comparison_figure(sample_df, config, robust, f"FigS1_all_23_samples_atlas_{order}_page_{pages+1:02d}", selected)
        png = repo_path(config["report_root"]) / f"figures/FigS1_all_23_samples_atlas_{order}_page_{pages+1:02d}_{'per_image_robust_scale' if robust else 'row_shared_scale'}.png"
        if png.exists():
            (page_root / f"page_{pages+1:02d}.png").write_bytes(png.read_bytes())
        pages += 1
    return pages


def write_dashboard_and_text(sample_df: pd.DataFrame, model_summary: pd.DataFrame, validation: dict[str, Any], config: dict[str, Any]) -> None:
    report = repo_path(config["report_root"])
    figs = sorted((report / "figures").glob("*.png"))
    links = "\n".join([f"<a href='figures/{p.name}'><img src='figures/{p.name}' width='360'></a>" for p in figs])
    table_html = sample_df.to_html(index=False, table_id="samples", classes="sortable", float_format=lambda x: f"{x:.3f}")
    html = f"""<!doctype html><html><head><meta charset='utf-8'><style>
body{{font-family:Arial,DejaVu Sans,sans-serif;margin:24px}} img{{margin:8px;border:1px solid #ddd}} table{{border-collapse:collapse;font-size:12px}} td,th{{border:1px solid #ccc;padding:4px}} th{{position:sticky;top:0;background:#eee}} .note{{background:#f7f7f7;padding:12px}}
</style><script>
function sortTable(n){{var t=document.getElementById('samples'),r=[...t.rows].slice(1),asc=t.getAttribute('data-sort')!='asc';r.sort((a,b)=>a.cells[n].innerText.localeCompare(b.cells[n].innerText,undefined,{{numeric:true}}));if(!asc)r.reverse();r.forEach(x=>t.tBodies[0].appendChild(x));t.setAttribute('data-sort',asc?'asc':'desc');}}
window.onload=()=>{{document.querySelectorAll('#samples th').forEach((th,i)=>th.onclick=()=>sortTable(i));}}
</script></head><body>
<h1>Phase 4B Results Dashboard</h1>
<div class='note'>S1 is a real AFM exemplar retrieved from the outer-training database. S4 is retrieval-augmented representative morphology. Neither output is an exact reconstruction of the held-out local AFM scan.</div>
<h2>Overview</h2><p>Primary OOF cohort N={len(sample_df)}. Validation passed: {validation['passed']}.</p>
<h2>Pipeline</h2>{links}
<h2>All Samples</h2>{table_html}
<h2>Model Summary</h2>{model_summary.to_html(index=False)}
<h2>Limitations</h2><p>No exact AFM reconstruction claim; expert labels pending; automatic morphology index is not monotonic with Rq in this cohort.</p>
</body></html>"""
    (report / "results_dashboard.html").write_text(html, encoding="utf-8")
    captions = ["# Figure Captions", ""]
    for p in figs:
        captions.append(f"## {p.stem}\nRetrospective OOF visualization unless noted. Ground Truth, S1 retrieved real AFM exemplar, and S4 retrieval-augmented representative AFM use physical height arrays with stated row-shared or robust color scaling. Sample selection follows fixed quantile or all-sample rules, not prediction error.\n")
    (report / "figure_captions.md").write_text("\n".join(captions), encoding="utf-8")
    (report / "current_results_summary.md").write_text(
        "# Current Results Summary\n\nData cohort: 23 primary group-held-out samples.\n\nRq prediction: R4 remains modest, with high-confidence subset improving concordance but not meeting all Rq gates.\n\nRepresentative AFM retrieval: S1/S4 are representative morphology outputs, not exact reconstructions.\n\nPatch synthesis: S4 uses predicted-Rq scaling and audited patch provenance.\n\nLimitations: pixel-level AFM similarity is not supported by same-growth ceiling; expert RHEED labels are pending.\n\nRecommended next experiment: collect prospective expert labels and repeated-growth validation before stronger deployment claims.\n",
        encoding="utf-8",
    )
    (report / "claims_and_limitations.md").write_text(
        "# Claims and Limitations\n\n## Can claim\n- Strict group-held-out evaluation.\n- RHEED-conditioned representative AFM retrieval/synthesis.\n- Predicted-Rq-scaled representative morphology with provenance.\n\n## Cannot claim\n- Exact local AFM reconstruction.\n- Pixel-level AFM reconstruction.\n- Robust monotonic spotty-to-Rq relation.\n- Neural generative-model superiority.\n\n## Needs prospective validation\n- Expert plausibility review.\n- Prospective closed-loop predictions.\n- Larger cohort calibration.\n",
        encoding="utf-8",
    )
