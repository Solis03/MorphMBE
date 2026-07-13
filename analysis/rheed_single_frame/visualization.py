"""Figures and static HTML report for the single-frame experiment."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
from PIL import Image

from analysis.rheed_roughness.run import display_path, safe_float
from analysis.rheed_roughness.visualize_manual_pairs import (
    SelectedPair,
    load_height_nm,
    nice_scale_bar_um,
    render_afm_height_panel,
    scan_label,
)
from analysis.rheed_single_frame.connectivity_features import PHYSICAL_INTERPRETABLE_FEATURES
from analysis.rheed_single_frame.data import ExperimentPaths
from analysis.rheed_single_frame.removelist import RemovelistAudit, assert_no_removed_samples


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _prediction_arrays(predictions: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray([safe_float(row["rq_true_nm"]) for row in predictions], dtype=float)
    p = np.asarray([safe_float(row["rq_pred_nm"]) for row in predictions], dtype=float)
    return y, p


def _annotate_points(ax: plt.Axes, predictions: Sequence[dict[str, Any]], x: np.ndarray, y: np.ndarray) -> None:
    for row, xv, yv in zip(predictions, x, y):
        ax.annotate(str(row["sample_id"]), (xv, yv), fontsize=6, xytext=(2, 2), textcoords="offset points")


def plot_prediction_scatter(predictions: Sequence[dict[str, Any]], paths: ExperimentPaths) -> None:
    if not predictions:
        return
    y, p = _prediction_arrays(predictions)
    lower = np.asarray([safe_float(row["prediction_interval_lower_nm"]) for row in predictions], dtype=float)
    upper = np.asarray([safe_float(row["prediction_interval_upper_nm"]) for row in predictions], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=180)
    axes[0].errorbar(y, p, yerr=[p - lower, upper - p], fmt="o", color="#2b6f8a", ecolor="#9eb7c2", capsize=2)
    lo, hi = min(float(y.min()), float(lower.min())), max(float(y.max()), float(upper.max()))
    axes[0].plot([lo, hi], [lo, hi], color="black", lw=1)
    axes[0].set_xlabel("True Rq (nm)")
    axes[0].set_ylabel("OOF predicted Rq (nm)")
    axes[0].set_title("Original scale")
    _annotate_points(axes[0], predictions, y, p)
    ly = np.log10(y)
    lp = np.log10(p)
    axes[1].scatter(ly, lp, color="#2b6f8a")
    lo, hi = min(float(ly.min()), float(lp.min())), max(float(ly.max()), float(lp.max()))
    axes[1].plot([lo, hi], [lo, hi], color="black", lw=1)
    axes[1].set_xlabel("True log10 Rq")
    axes[1].set_ylabel("OOF predicted log10 Rq")
    axes[1].set_title("Log scale")
    _annotate_points(axes[1], predictions, ly, lp)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "oof_predicted_vs_true_rq.png", bbox_inches="tight")
    fig.savefig(paths.figures_dir / "oof_predicted_vs_true_rq.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_residuals(predictions: Sequence[dict[str, Any]], paths: ExperimentPaths) -> None:
    if not predictions:
        return
    y, p = _prediction_arrays(predictions)
    residual = p - y
    conf = np.asarray([safe_float(row["confidence_score"]) for row in predictions], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=180)
    for ax, x, label in [
        (axes[0], y, "True Rq (nm)"),
        (axes[1], p, "Predicted Rq (nm)"),
        (axes[2], conf, "Confidence score"),
    ]:
        ax.axhline(0, color="black", lw=1)
        ax.scatter(x, residual, color="#8a4b2b")
        ax.set_xlabel(label)
        ax.set_ylabel("Residual (pred - true, nm)")
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "oof_residuals.png", bbox_inches="tight")
    plt.close(fig)


def _feature_by_sample(feature_rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in feature_rows}


def plot_feature_hypothesis(predictions: Sequence[dict[str, Any]], feature_rows: Sequence[dict[str, Any]], paths: ExperimentPaths) -> None:
    features = _feature_by_sample(feature_rows)
    plot_keys = [
        ("horizontal_connectivity_score", "Horizontal connectivity"),
        ("isolation_score", "Isolation"),
        ("isolated_component_fraction", "Isolated component fraction"),
        ("horizontal_closing_gain", "Horizontal closing gain"),
        ("horizontal_neighbor_fraction", "Horizontal neighbor fraction"),
    ]
    fig, axes = plt.subplots(1, len(plot_keys), figsize=(16, 3.2), dpi=180)
    rq = np.asarray([safe_float(row["rq_true_nm"]) for row in predictions], dtype=float)
    for ax, (key, title) in zip(axes, plot_keys):
        x = np.asarray([safe_float(features.get(str(row["sample_id"]), {}).get(key), math.nan) for row in predictions], dtype=float)
        ax.scatter(x, rq, color="#306b3f")
        ax.set_xlabel(title)
        ax.set_ylabel("Rq (nm)")
        if np.isfinite(x).sum() >= 3:
            rho = safe_float(__import__("scipy").stats.spearmanr(x[np.isfinite(x)], rq[np.isfinite(x)]).statistic, math.nan)
            ax.set_title(f"rho={rho:.2f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "feature_hypothesis_plots.png", bbox_inches="tight")
    plt.close(fig)


def plot_confound(predictions: Sequence[dict[str, Any]], feature_rows: Sequence[dict[str, Any]], paths: ExperimentPaths) -> None:
    features = _feature_by_sample(feature_rows)
    plot_specs = [
        ("mean_intensity", "rq_true_nm", "Brightness vs Rq"),
        ("saturation_fraction", "rq_true_nm", "Saturation vs Rq"),
        ("pattern_centroid_x", "rq_true_nm", "Centroid x vs Rq"),
        ("horizontal_connectivity_score", "mean_intensity", "Connectivity vs brightness"),
        ("isolation_score", "mean_intensity", "Isolation vs brightness"),
    ]
    fig, axes = plt.subplots(1, len(plot_specs), figsize=(16, 3.2), dpi=180)
    for ax, (xkey, ykey, title) in zip(axes, plot_specs):
        xs = []
        ys = []
        for row in predictions:
            feat = features.get(str(row["sample_id"]), {})
            xs.append(safe_float(feat.get(xkey), math.nan))
            ys.append(safe_float(row.get(ykey, feat.get(ykey)), math.nan))
        ax.scatter(xs, ys, color="#5c5874")
        ax.set_xlabel(xkey)
        ax.set_ylabel(ykey)
        ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "confound_plots.png", bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(model_rows: Sequence[dict[str, Any]], paths: ExperimentPaths) -> None:
    if not model_rows:
        return
    rows = sorted(model_rows, key=lambda row: safe_float(row.get("mae_log"), math.inf))
    names = [str(row["model_name"]) for row in rows]
    metrics = ["mae_nm", "rmse_nm", "r2_nm", "spearman_nm", "mean_interval_width_nm"]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 11), dpi=180)
    for ax, metric in zip(axes, metrics):
        values = [safe_float(row.get(metric), math.nan) for row in rows]
        ax.barh(names, values, color="#577590")
        ax.set_xlabel(metric)
        ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "model_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_confidence_calibration(predictions: Sequence[dict[str, Any]], feature_rows: Sequence[dict[str, Any]], paths: ExperimentPaths) -> None:
    features = _feature_by_sample(feature_rows)
    if not predictions:
        return
    conf = np.asarray([safe_float(row["confidence_score"]) for row in predictions], dtype=float)
    err = np.asarray([safe_float(row["absolute_error_nm"]) for row in predictions], dtype=float)
    width = np.asarray([safe_float(row["prediction_interval_width_nm"]) for row in predictions], dtype=float)
    covered = np.asarray([
        safe_float(row["prediction_interval_lower_nm"]) <= safe_float(row["rq_true_nm"]) <= safe_float(row["prediction_interval_upper_nm"])
        for row in predictions
    ])
    isolation = np.asarray([safe_float(features.get(str(row["sample_id"]), {}).get("isolation_score"), math.nan) for row in predictions])
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), dpi=180)
    axes[0].scatter(conf, err, color="#aa5a44")
    axes[0].set_xlabel("Confidence")
    axes[0].set_ylabel("Absolute error (nm)")
    axes[1].scatter(width, covered.astype(float), color="#4b7f52")
    axes[1].set_xlabel("PI width (nm)")
    axes[1].set_ylabel("Covered")
    axes[2].scatter(conf, width, color="#4b6b95")
    axes[2].set_xlabel("Confidence")
    axes[2].set_ylabel("PI width (nm)")
    axes[3].scatter(isolation, conf, color="#6a4b95")
    axes[3].set_xlabel("Isolation score")
    axes[3].set_ylabel("Confidence")
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "confidence_calibration.png", bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(rows: Sequence[dict[str, Any]], paths: ExperimentPaths) -> None:
    rows = list(rows)[:25]
    if not rows:
        return
    labels = [str(row["feature"]) for row in rows][::-1]
    values = [safe_float(row["mean_abs_coefficient"]) for row in rows][::-1]
    colors = ["#5d8a66" if row.get("feature_group") == "physics" else "#8a775d" for row in rows][::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, len(rows) * 0.22)), dpi=180)
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("Mean absolute fold coefficient")
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)


def plot_influence(rows: Sequence[dict[str, Any]], paths: ExperimentPaths) -> None:
    if not rows:
        return
    labels = [str(row["removed_sample_id"]) for row in rows]
    mae = [safe_float(row.get("mae_nm")) for row in rows]
    rho = [safe_float(row.get("spearman_nm")) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), dpi=180, sharex=True)
    axes[0].bar(labels, mae, color="#8a4f5d")
    axes[0].set_ylabel("MAE after removal")
    axes[1].bar(labels, rho, color="#4f6f8a")
    axes[1].set_ylabel("Spearman after removal")
    axes[1].tick_params(axis="x", rotation=90, labelsize=6)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "leave_one_sample_influence.png", bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_6023(rows: Sequence[dict[str, Any]], paths: ExperimentPaths) -> None:
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=180)
    ax.axis("off")
    text = "\n".join(f"{row.get('analysis', '')}: {row.get('status', 'ok')}" for row in rows)
    ax.text(0.02, 0.7, text or "No sensitivity rows.", fontsize=10, va="top")
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "sensitivity_without_6023.png", bbox_inches="tight")
    plt.close(fig)


def plot_perturbation(rows: Sequence[dict[str, Any]], paths: ExperimentPaths) -> None:
    if not rows:
        return
    names = sorted({str(row["perturbation"]) for row in rows})
    values = [
        np.nanmedian([safe_float(row["prediction_change_relative_to_dataset_iqr"]) for row in rows if str(row["perturbation"]) == name])
        for name in names
    ]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=180)
    ax.barh(names, values, color="#7c6a4d")
    ax.set_xlabel("Median prediction change / dataset Rq IQR")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(paths.figures_dir / "perturbation_stability.png", bbox_inches="tight")
    plt.close(fig)


def _rheed_image(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3]
    return arr


def _confidence_bar(ax: plt.Axes, value: float) -> None:
    ax.add_patch(plt.Rectangle((0.0, 0.02), 1.0, 0.06, color="#e3e3e3", transform=ax.transAxes, clip_on=False))
    ax.add_patch(plt.Rectangle((0.0, 0.02), np.clip(value / 100.0, 0, 1), 0.06, color="#477e61", transform=ax.transAxes, clip_on=False))


def render_prediction_grid(
    predictions: Sequence[dict[str, Any]],
    pairs: Sequence[SelectedPair],
    paths: ExperimentPaths,
    removelist: RemovelistAudit,
    *,
    output_stem: str,
    sort_key: str,
    common_scale: tuple[float, float] | None,
    native_scale: bool = False,
    cards_per_row: int = 2,
) -> None:
    assert_no_removed_samples((row["sample_id"] for row in predictions), removelist.sample_ids, context=f"figure {output_stem}")
    pair_by_id = {pair.sample_id: pair for pair in pairs}
    rows = [row for row in predictions if str(row["sample_id"]) in pair_by_id]
    if sort_key == "predicted":
        rows.sort(key=lambda row: (safe_float(row["rq_pred_nm"]), row["sample_id"]))
    elif sort_key == "absolute_error":
        rows.sort(key=lambda row: (safe_float(row["absolute_error_nm"]), row["sample_id"]))
    elif sort_key == "confidence":
        rows.sort(key=lambda row: (-safe_float(row["confidence_score"]), row["sample_id"]))
    else:
        rows.sort(key=lambda row: (safe_float(row["rq_true_nm"]), row["sample_id"]))
    if not rows:
        return
    nrows = math.ceil(len(rows) / cards_per_row)
    fig = plt.figure(figsize=(cards_per_row * 6.6, nrows * 3.9), dpi=220)
    width_pattern: list[float] = []
    for _ in range(cards_per_row):
        width_pattern.extend([1.0, 1.0, 0.06, 1.7])
    grid = fig.add_gridspec(nrows=nrows, ncols=cards_per_row * 4, width_ratios=width_pattern, hspace=0.75, wspace=0.15)
    for idx, row in enumerate(rows):
        rr = idx // cards_per_row
        cc = (idx % cards_per_row) * 4
        pair = pair_by_id[str(row["sample_id"])]
        ax_r = fig.add_subplot(grid[rr, cc])
        ax_a = fig.add_subplot(grid[rr, cc + 1])
        cax = fig.add_subplot(grid[rr, cc + 2])
        ax_t = fig.add_subplot(grid[rr, cc + 3])
        ax_r.imshow(_rheed_image(pair.manual_rheed_path), cmap="gray")
        ax_r.set_title(f"Sample {pair.sample_id}\nRHEED input\nManual selection", fontsize=7)
        ax_r.set_xticks([])
        ax_r.set_yticks([])
        if native_scale or common_scale is None:
            vmin, vmax = pair.native_display_min_nm, pair.native_display_max_nm
            title = "Ground-truth AFM"
        else:
            vmin, vmax = common_scale
            title = "Ground-truth AFM\ncommon height scale"
        im = render_afm_height_panel(ax_a, pair, vmin=vmin, vmax=vmax, title=title)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Height (nm)", fontsize=6)
        cb.ax.tick_params(labelsize=5)
        ax_t.axis("off")
        fallback = "" if abs(pair.afm.scan_size_um - 1.0) <= 0.10 else "\nnon-1.0 um fallback scan"
        text = (
            f"Ground truth Rq = {safe_float(row['rq_true_nm']):.3f} nm\n"
            f"Predicted Rq = {safe_float(row['rq_pred_nm']):.3f} nm\n"
            f"90% PI = [{safe_float(row['prediction_interval_lower_nm']):.3f}, {safe_float(row['prediction_interval_upper_nm']):.3f}] nm\n"
            f"Absolute error = {safe_float(row['absolute_error_nm']):.3f} nm\n"
            f"Confidence = {safe_float(row['confidence_score']):.0f} / 100\n"
            f"OOF model = {html.escape(str(row.get('selected_model_name', '')))}"
            f"{fallback}"
        )
        ax_t.text(0.0, 0.92, text, va="top", fontsize=7.5, linespacing=1.18)
        _confidence_bar(ax_t, safe_float(row["confidence_score"]))
    for idx in range(len(rows), nrows * cards_per_row):
        rr = idx // cards_per_row
        cc = (idx % cards_per_row) * 4
        for sub in range(4):
            fig.add_subplot(grid[rr, cc + sub]).axis("off")
    fig.savefig(paths.reports_dir / f"{output_stem}.png", bbox_inches="tight")
    if output_stem.endswith("common_scale") or "by_true_rq" in output_stem:
        fig.savefig(paths.reports_dir / f"{output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def render_feature_overlay_grid(
    predictions: Sequence[dict[str, Any]],
    feature_rows: Sequence[dict[str, Any]],
    pairs: Sequence[SelectedPair],
    paths: ExperimentPaths,
    removelist: RemovelistAudit,
    *,
    output_stem: str,
    sort_key: str = "rq_true_nm",
    cards_per_row: int = 3,
) -> None:
    assert_no_removed_samples((row["sample_id"] for row in predictions), removelist.sample_ids, context=f"feature overlay {output_stem}")
    features = _feature_by_sample(feature_rows)
    pair_by_id = {pair.sample_id: pair for pair in pairs}
    rows = [row for row in predictions if row["sample_id"] in features]
    if sort_key == "horizontal_connectivity_score":
        rows.sort(key=lambda row: (safe_float(features[row["sample_id"]].get(sort_key)), row["sample_id"]))
    elif sort_key == "isolation_score":
        rows.sort(key=lambda row: (safe_float(features[row["sample_id"]].get(sort_key)), row["sample_id"]))
    else:
        rows.sort(key=lambda row: (safe_float(row["rq_true_nm"]), row["sample_id"]))
    if not rows:
        return
    nrows = math.ceil(len(rows) / cards_per_row)
    fig = plt.figure(figsize=(cards_per_row * 5.4, nrows * 3.5), dpi=220)
    grid = fig.add_gridspec(nrows=nrows, ncols=cards_per_row * 4, hspace=0.65, wspace=0.12)
    for idx, row in enumerate(rows):
        rr = idx // cards_per_row
        cc = (idx % cards_per_row) * 4
        feat = features[row["sample_id"]]
        pair = pair_by_id[row["sample_id"]]
        panel_paths = [
            pair.manual_rheed_path,
            paths.repo_root / str(feat["component_overlay_path"]),
            paths.repo_root / str(feat["horizontal_closing_overlay_path"]),
            paths.repo_root / str(feat["skeleton_overlay_path"]),
        ]
        titles = [
            f"{row['sample_id']} original\nRq {safe_float(row['rq_true_nm']):.3f} nm",
            "components + graph",
            f"horizontal closing\nH score {safe_float(feat.get('horizontal_connectivity_score')):.2f}",
            f"skeleton/run length\nIsolation {safe_float(feat.get('isolation_score')):.2f}",
        ]
        for c, (path, title) in enumerate(zip(panel_paths, titles)):
            ax = fig.add_subplot(grid[rr, cc + c])
            ax.imshow(np.asarray(Image.open(path)), cmap="gray")
            ax.set_title(title, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.savefig(paths.reports_dir / f"{output_stem}.png", bbox_inches="tight")
    if output_stem == "connectivity_feature_overlays_by_rq":
        fig.savefig(paths.reports_dir / f"{output_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_html_report(
    predictions: Sequence[dict[str, Any]],
    feature_rows: Sequence[dict[str, Any]],
    model_rows: Sequence[dict[str, Any]],
    skipped_rows: Sequence[dict[str, Any]],
    excluded_rows: Sequence[dict[str, Any]],
    paths: ExperimentPaths,
) -> None:
    features = _feature_by_sample(feature_rows)
    cards = []
    for row in predictions:
        feat = features.get(str(row["sample_id"]), {})
        cards.append(
            f"""
<article class="card" data-sample="{html.escape(str(row['sample_id']))}" data-true="{safe_float(row['rq_true_nm']):.12g}" data-pred="{safe_float(row['rq_pred_nm']):.12g}" data-error="{safe_float(row['absolute_error_nm']):.12g}" data-confidence="{safe_float(row['confidence_score']):.12g}" data-connectivity="{safe_float(feat.get('horizontal_connectivity_score')):.12g}" data-isolation="{safe_float(feat.get('isolation_score')):.12g}">
  <h2>Sample {html.escape(str(row['sample_id']))}</h2>
  <dl>
    <dt>True Rq</dt><dd>{safe_float(row['rq_true_nm']):.3f} nm</dd>
    <dt>Predicted Rq</dt><dd>{safe_float(row['rq_pred_nm']):.3f} nm</dd>
    <dt>90% PI</dt><dd>[{safe_float(row['prediction_interval_lower_nm']):.3f}, {safe_float(row['prediction_interval_upper_nm']):.3f}] nm</dd>
    <dt>Error</dt><dd>{safe_float(row['absolute_error_nm']):.3f} nm</dd>
    <dt>Confidence</dt><dd>{safe_float(row['confidence_score']):.0f} / 100</dd>
    <dt>Connectivity</dt><dd>{safe_float(feat.get('horizontal_connectivity_score')):.3f}</dd>
    <dt>Isolation</dt><dd>{safe_float(feat.get('isolation_score')):.3f}</dd>
    <dt>Model</dt><dd>{html.escape(str(row.get('selected_model_name', '')))}</dd>
  </dl>
</article>
"""
        )
    top_models = "".join(
        f"<tr><td>{html.escape(str(row.get('model_name', '')))}</td><td>{safe_float(row.get('mae_nm')):.3f}</td><td>{safe_float(row.get('spearman_nm')):.3f}</td><td>{safe_float(row.get('r2_nm')):.3f}</td></tr>"
        for row in model_rows[:20]
    )
    skipped = "".join(f"<li>{html.escape(str(row.get('sample_id', '')))}: {html.escape(str(row.get('skip_reason', '')))}</li>" for row in skipped_rows)
    excluded = "".join(f"<li>{html.escape(str(row.get('sample_id', '')))}: canonical removelist</li>" for row in excluded_rows)
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Single-frame manual RHEED roughness experiment</title>
<style>
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f7f7f4; color:#202124; }}
header {{ position:sticky; top:0; background:#fff; border-bottom:1px solid #d7d7d0; padding:12px 16px; z-index:2; display:flex; gap:14px; align-items:center; }}
h1 {{ font-size:18px; margin:0; }}
select {{ padding:4px 8px; }}
main {{ padding:16px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:12px; }}
.card {{ background:#fff; border:1px solid #d9d9d2; border-radius:8px; padding:12px; }}
.card h2 {{ margin:0 0 8px; font-size:15px; }}
dl {{ display:grid; grid-template-columns:110px 1fr; gap:3px 8px; font-size:13px; }}
dt {{ font-weight:600; }}
dd {{ margin:0; overflow-wrap:anywhere; }}
table {{ border-collapse:collapse; background:#fff; margin:14px 0; }}
td,th {{ border:1px solid #ddd; padding:5px 8px; font-size:13px; }}
img {{ max-width:100%; border:1px solid #ddd; background:#fff; }}
</style>
</head>
<body>
<header>
  <h1>Single-frame manual RHEED to AFM Rq</h1>
  <label>Sort <select id="sorter">
    <option value="true">true Rq</option><option value="pred">predicted Rq</option><option value="error">absolute error</option><option value="confidence">confidence</option><option value="connectivity">connectivity</option><option value="isolation">isolation</option><option value="sample">sample ID</option>
  </select></label>
</header>
<main>
<h2>Figures</h2>
<p><a href="single_frame_oof_predictions_by_true_rq_common_scale.png">Main prediction grid</a> | <a href="connectivity_feature_overlays_by_rq.png">Feature overlays</a></p>
<h2>Model comparison</h2>
<table><thead><tr><th>Model</th><th>MAE nm</th><th>Spearman</th><th>R2</th></tr></thead><tbody>{top_models}</tbody></table>
<h2>Prediction cards</h2>
<section class="grid" id="cards">{''.join(cards)}</section>
<h2>Skipped</h2><ul>{skipped}</ul>
<h2>Excluded by removelist</h2><ul>{excluded}</ul>
</main>
<script>
const cards = document.getElementById('cards');
document.getElementById('sorter').addEventListener('change', event => {{
  const mode = event.target.value;
  const items = Array.from(cards.children);
  items.sort((a, b) => {{
    if (mode === 'sample') return a.dataset.sample.localeCompare(b.dataset.sample, undefined, {{numeric:true}});
    const sign = mode === 'confidence' ? -1 : 1;
    return sign * (Number(a.dataset[mode]) - Number(b.dataset[mode])) || a.dataset.sample.localeCompare(b.dataset.sample, undefined, {{numeric:true}});
  }});
  items.forEach(item => cards.appendChild(item));
}});
</script>
</body>
</html>
"""
    (paths.reports_dir / "index.html").write_text(doc, encoding="utf-8")


def make_all_figures(
    predictions: Sequence[dict[str, Any]],
    feature_rows: Sequence[dict[str, Any]],
    model_rows: Sequence[dict[str, Any]],
    importance_rows: Sequence[dict[str, Any]],
    influence_rows: Sequence[dict[str, Any]],
    sensitivity_rows: Sequence[dict[str, Any]],
    perturbation_rows: Sequence[dict[str, Any]],
    pairs: Sequence[SelectedPair],
    paths: ExperimentPaths,
    removelist: RemovelistAudit,
    common_scale: tuple[float, float] | None,
) -> None:
    assert_no_removed_samples((row["sample_id"] for row in predictions), removelist.sample_ids, context="figure generation")
    plot_prediction_scatter(predictions, paths)
    plot_residuals(predictions, paths)
    plot_feature_hypothesis(predictions, feature_rows, paths)
    plot_confound(predictions, feature_rows, paths)
    plot_model_comparison(model_rows, paths)
    plot_confidence_calibration(predictions, feature_rows, paths)
    plot_feature_importance(importance_rows, paths)
    plot_influence(influence_rows, paths)
    plot_sensitivity_6023(sensitivity_rows, paths)
    plot_perturbation(perturbation_rows, paths)
    render_prediction_grid(
        predictions,
        pairs,
        paths,
        removelist,
        output_stem="single_frame_oof_predictions_by_true_rq_common_scale",
        sort_key="true",
        common_scale=common_scale,
    )
    render_prediction_grid(
        predictions,
        pairs,
        paths,
        removelist,
        output_stem="single_frame_oof_predictions_by_predicted_rq",
        sort_key="predicted",
        common_scale=common_scale,
    )
    render_prediction_grid(
        predictions,
        pairs,
        paths,
        removelist,
        output_stem="single_frame_oof_predictions_by_absolute_error",
        sort_key="absolute_error",
        common_scale=common_scale,
    )
    render_prediction_grid(
        predictions,
        pairs,
        paths,
        removelist,
        output_stem="single_frame_oof_predictions_by_confidence",
        sort_key="confidence",
        common_scale=common_scale,
    )
    render_prediction_grid(
        predictions,
        pairs,
        paths,
        removelist,
        output_stem="single_frame_oof_predictions_by_true_rq_native_scale",
        sort_key="true",
        common_scale=common_scale,
        native_scale=True,
    )
    render_feature_overlay_grid(predictions, feature_rows, pairs, paths, removelist, output_stem="connectivity_feature_overlays_by_rq", sort_key="rq_true_nm")
    render_feature_overlay_grid(
        predictions,
        feature_rows,
        pairs,
        paths,
        removelist,
        output_stem="connectivity_feature_overlays_by_connectivity",
        sort_key="horizontal_connectivity_score",
    )
    render_feature_overlay_grid(
        predictions,
        feature_rows,
        pairs,
        paths,
        removelist,
        output_stem="connectivity_feature_overlays_by_isolation",
        sort_key="isolation_score",
    )
