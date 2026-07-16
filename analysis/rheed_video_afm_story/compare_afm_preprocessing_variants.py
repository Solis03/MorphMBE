from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import repo_path, write_csv, write_json
from .rq_disentanglement import rq_np


def _metric(df: pd.DataFrame, method: str, seed: int | None = None) -> pd.DataFrame:
    out = df[df["method"].eq(method)].copy()
    if seed is not None:
        seeded = out[out["seed"].astype(int).eq(seed)]
        if len(seeded):
            out = seeded
    return out.sort_values("sample_id").groupby("sample_id").head(1)


def _load_selection() -> list[str]:
    path = repo_path("outputs/rheed_video_afm_story/phase4b_visualization/main_figure_sample_selection.json")
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [str(x) for x in payload.get("selected_sample_ids", [])]
    sample = pd.read_csv(repo_path("outputs/rheed_video_afm_story/phase4b_visualization/sample_level_results.csv"), dtype={"sample_id": str})
    ordered = sample.sort_values("true_rq_nm").reset_index(drop=True)
    selected = []
    for q in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        target = ordered["true_rq_nm"].quantile(q)
        for sid in ordered.assign(d=(ordered["true_rq_nm"] - target).abs()).sort_values(["d", "sample_id"])["sample_id"]:
            if sid not in selected:
                selected.append(str(sid))
                break
    return selected


def _show(ax: plt.Axes, arr: np.ndarray, title: str, vmin: float | None = None, vmax: float | None = None) -> None:
    im = ax.imshow(arr, cmap="viridis", origin="upper", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=7)
    ax.set_axis_off()
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.01).set_label("Height relative to mean (nm)", fontsize=6)


def comparison_figures(
    mapping: pd.DataFrame,
    sample_targets: pd.DataFrame,
    variant_output_root: str | Path,
    variant_report_root: str | Path,
) -> list[str]:
    out = repo_path(variant_output_root) / "comparison"
    rep = repo_path(variant_report_root) / "comparison"
    rep.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    primary = sample_targets.sort_values("sample_id")

    # Figure A.
    rows = len(primary)
    fig, axes = plt.subplots(rows, 3, figsize=(9, rows * 2.2), dpi=150, squeeze=False)
    m = mapping.set_index(["sample_id", "scan_id"])
    for i, target in enumerate(primary.itertuples()):
        first_scan = str(target.first_order_representative_scan)
        second_scan = str(target.second_order_representative_scan)
        scan = second_scan if (str(target.sample_id), second_scan) in m.index else first_scan
        map_row = m.loc[(str(target.sample_id), scan)] if (str(target.sample_id), scan) in m.index else m.loc[(str(target.sample_id), second_scan)]
        raw = np.load(repo_path(map_row["source_afm_path"]), allow_pickle=False).astype(float)
        first = np.load(repo_path(map_row["first_order_afm_path"]), allow_pickle=False).astype(float)
        second = np.load(repo_path(map_row["second_order_afm_path"]), allow_pickle=False).astype(float)
        arrays = [raw - np.nanmean(raw), first - np.nanmean(first), second - np.nanmean(second)]
        finite = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
        lo, hi = np.percentile(finite, [2, 98])
        titles = [
            f"{target.sample_id} raw\nRq={rq_np(raw):.2f}",
            f"first-order\nRq={rq_np(first):.2f}",
            f"second-order\nRq={rq_np(second):.2f}",
        ]
        for j, arr in enumerate(arrays):
            _show(axes[i, j], arr, titles[j], lo, hi)
    fig.suptitle("Comparison A: AFM preprocessing effect", y=0.995)
    fig.tight_layout()
    path = rep / "comparison_A_afm_preprocessing_effect.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path.relative_to(repo_path("."))))

    # Figure B.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=180)
    first = primary["first_order_rq_nm"].to_numpy(float)
    second = primary["second_order_rq_nm"].to_numpy(float)
    axes[0, 0].scatter(first, second)
    for row in primary.itertuples():
        axes[0, 0].annotate(row.sample_id, (row.first_order_rq_nm, row.second_order_rq_nm), fontsize=6)
    lim = [min(first.min(), second.min()), max(first.max(), second.max())]
    axes[0, 0].plot(lim, lim, "k--", lw=1)
    axes[0, 0].set_xlabel("First-order Rq (nm)")
    axes[0, 0].set_ylabel("Second-order Rq (nm)")
    mean = (first + second) / 2
    diff = second - first
    axes[0, 1].scatter(mean, diff)
    axes[0, 1].axhline(float(np.mean(diff)), color="k", lw=1)
    axes[0, 1].set_xlabel("Mean Rq (nm)")
    axes[0, 1].set_ylabel("Second - first Rq (nm)")
    axes[1, 0].bar(primary["sample_id"], primary["second_order_rq_nm"].rank().to_numpy() - primary["first_order_rq_nm"].rank().to_numpy())
    axes[1, 0].tick_params(axis="x", rotation=90)
    axes[1, 0].set_ylabel("Rq rank change")
    axes[1, 1].bar(primary["sample_id"], diff)
    axes[1, 1].tick_params(axis="x", rotation=90)
    axes[1, 1].set_ylabel("Delta Rq (nm)")
    fig.suptitle("Comparison B: sample-level Rq targets")
    fig.tight_layout()
    path = rep / "comparison_B_sample_level_rq_targets.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path.relative_to(repo_path("."))))

    # Figures C/D/E and tables.
    first_rq = pd.read_csv(repo_path("outputs/rheed_video_afm_story/phase4a/rheed_rq_oof_predictions.csv"), dtype={"sample_id": str})
    second_rq = pd.read_csv(repo_path(out.parent / "rq_models" / "second_order_rq_oof_predictions.csv"), dtype={"sample_id": str})
    first_metrics = pd.read_csv(repo_path("outputs/rheed_video_afm_story/phase4a/rheed_rq_oof_metrics.csv"))
    second_metrics = pd.read_csv(repo_path(out.parent / "rq_models" / "second_order_rq_model_metrics.csv"))
    first_synth = pd.read_csv(repo_path("outputs/rheed_video_afm_story/phase4a/synthesis_oof_metrics.csv"), dtype={"sample_id": str})
    second_synth = pd.read_csv(repo_path(out.parent / "phase4a" / "second_order_synthesis_metrics_by_sample.csv"), dtype={"sample_id": str})
    write_csv(
        pd.concat(
            [
                first_metrics.assign(afm_target_variant="first_order"),
                second_metrics.assign(afm_target_variant="second_order_y2"),
            ],
            ignore_index=True,
        ),
        out / "first_vs_second_order_model_metrics.csv",
    )
    write_csv(
        pd.concat(
            [
                first_synth.assign(afm_target_variant="first_order"),
                second_synth.assign(afm_target_variant="second_order_y2"),
            ],
            ignore_index=True,
        ),
        out / "first_vs_second_order_synthesis_metrics.csv",
    )

    r4_first = first_rq[first_rq["model_id"].eq("R4_auto_iso_dino_residual")].set_index("sample_id")
    r4_second = second_rq[second_rq["model_id"].eq("R4_auto_iso_dino_residual")].set_index("sample_id")
    sample_rows = []
    s1_first = _metric(first_synth, "S1_top1_real_exemplar_retrieval").set_index("sample_id")
    s4_first = _metric(first_synth, "S4_calibrated_patch_synthesis", 29).set_index("sample_id")
    s1_second = _metric(second_synth, "S1_top1_real_exemplar_retrieval").set_index("sample_id")
    s4_second = _metric(second_synth, "S4_calibrated_patch_synthesis", 29).set_index("sample_id")
    first_ret = pd.read_csv(repo_path("outputs/rheed_video_afm_story/phase4a/oof_retrieval_candidates.csv"), dtype={"sample_id": str}).set_index("sample_id")
    second_ret = pd.read_csv(repo_path(out.parent / "phase4a" / "second_order_oof_retrieval_candidates.csv"), dtype={"heldout_sample_id": str})
    second_top = second_ret[second_ret["rank"].astype(int).eq(1)].set_index("heldout_sample_id")
    for row in primary.itertuples():
        sid = str(row.sample_id)
        first_source = json.loads(first_ret.loc[sid, "candidate_group_ids"])[0]
        second_source = str(second_top.loc[sid, "candidate_sample_id"])
        sample_rows.append(
            {
                "sample_id": sid,
                "rq_true_first": float(r4_first.loc[sid, "true_rq_nm"]),
                "rq_pred_first": float(r4_first.loc[sid, "predicted_rq_nm"]),
                "rq_error_first": float(r4_first.loc[sid, "absolute_error_nm"]),
                "rq_true_second": float(r4_second.loc[sid, "true_rq_nm"]),
                "rq_pred_second": float(r4_second.loc[sid, "predicted_rq_nm"]),
                "rq_error_second": float(r4_second.loc[sid, "absolute_error_nm"]),
                "delta_true_rq": float(r4_second.loc[sid, "true_rq_nm"] - r4_first.loc[sid, "true_rq_nm"]),
                "delta_predicted_rq": float(r4_second.loc[sid, "predicted_rq_nm"] - r4_first.loc[sid, "predicted_rq_nm"]),
                "s1_psd_first": float(s1_first.loc[sid, "normalized_psd_log_distance"]),
                "s1_psd_second": float(s1_second.loc[sid, "normalized_psd_log_distance"]),
                "s4_psd_first": float(s4_first.loc[sid, "normalized_psd_log_distance"]),
                "s4_psd_second": float(s4_second.loc[sid, "normalized_psd_log_distance"]),
                "s4_corr_error_first": float(s4_first.loc[sid, "correlation_length_relative_error"]),
                "s4_corr_error_second": float(s4_second.loc[sid, "correlation_length_relative_error"]),
                "first_retrieved_source": first_source,
                "second_retrieved_source": second_source,
                "retrieved_source_changed": first_source != second_source,
                "first_representative_scan": row.first_order_representative_scan,
                "second_representative_scan": row.second_order_representative_scan,
                "representative_scan_changed": bool(row.representative_scan_changed),
            }
        )
    sample_results = pd.DataFrame(sample_rows)
    write_csv(sample_results, out / "first_vs_second_order_sample_results.csv")

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    labels = ["first R4", "second R4"]
    mae = [
        float(first_metrics[first_metrics["model_id"].eq("R4_auto_iso_dino_residual")]["MAE"].iloc[0]),
        float(second_metrics[second_metrics["model_id"].eq("R4_auto_iso_dino_residual")]["MAE"].iloc[0]),
    ]
    ax.bar(labels, mae)
    ax.set_ylabel("OOF MAE (nm)")
    ax.set_title("Comparison C: Rq model performance")
    path = rep / "comparison_C_rq_model_performance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path.relative_to(repo_path("."))))

    fig, ax = plt.subplots(figsize=(11, 5), dpi=180)
    x = np.arange(len(sample_results))
    ax.plot(x, sample_results["rq_true_first"], "o-", label="first true")
    ax.plot(x, sample_results["rq_pred_first"], "o--", label="first predicted")
    ax.plot(x, sample_results["rq_true_second"], "s-", label="second true")
    ax.plot(x, sample_results["rq_pred_second"], "s--", label="second predicted")
    ax.set_xticks(x, sample_results["sample_id"], rotation=90)
    ax.set_ylabel("Rq (nm)")
    ax.set_title("Comparison D: OOF predictions")
    ax.legend(fontsize=7)
    path = rep / "comparison_D_oof_predictions.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path.relative_to(repo_path("."))))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=180)
    axes[0].bar(["S1 first", "S1 second", "S4 first", "S4 second"], [sample_results["s1_psd_first"].median(), sample_results["s1_psd_second"].median(), sample_results["s4_psd_first"].median(), sample_results["s4_psd_second"].median()])
    axes[0].set_ylabel("median PSD distance")
    axes[1].bar(["S4 first", "S4 second"], [sample_results["s4_corr_error_first"].median(), sample_results["s4_corr_error_second"].median()])
    axes[1].set_ylabel("median corr length rel error")
    fig.suptitle("Comparison E: S1/S4 descriptor metrics")
    path = rep / "comparison_E_s1_s4_descriptor_metrics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(str(path.relative_to(repo_path("."))))

    # Figure F.
    selected = [sid for sid in _load_selection() if sid in set(sample_results["sample_id"])]
    if selected:
        first_sample = pd.read_csv(repo_path("outputs/rheed_video_afm_story/phase4b_visualization/sample_level_results.csv"), dtype={"sample_id": str}).set_index("sample_id")
        second_sample = pd.read_csv(repo_path(out.parent / "phase4b_visualization" / "sample_level_results.csv"), dtype={"sample_id": str}).set_index("sample_id")
        fig, axes = plt.subplots(len(selected), 4, figsize=(13, 2.4 * len(selected)), dpi=150, squeeze=False)
        for i, sid in enumerate(selected):
            arrays = [
                np.load(repo_path(first_sample.loc[sid, "ground_truth_afm_path"])).astype(float),
                np.load(repo_path(second_sample.loc[sid, "ground_truth_afm_path"])).astype(float),
                np.load(repo_path(first_sample.loc[sid, "s4_output_path"])).astype(float),
                np.load(repo_path(second_sample.loc[sid, "s4_output_path"])).astype(float),
            ]
            finite = np.concatenate([(a - np.nanmean(a))[np.isfinite(a)].ravel() for a in arrays])
            lo, hi = np.percentile(finite, [2, 98])
            titles = ["first GT", "second GT", "first S4", "second S4"]
            for j, arr in enumerate(arrays):
                _show(axes[i, j], arr - np.nanmean(arr), f"{sid} {titles[j]}", lo, hi)
        fig.suptitle("Comparison F: visual output shift")
        fig.tight_layout()
        path = rep / "comparison_F_visual_output_shift.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path.relative_to(repo_path("."))))

    summary = {
        "comparison_figures": paths,
        "sample_table": str((out / "first_vs_second_order_sample_results.csv").relative_to(repo_path("."))),
        "retrieved_source_changed_count": int(sample_results["retrieved_source_changed"].sum()),
        "representative_scan_changed_count": int(sample_results["representative_scan_changed"].sum()),
        "first_r4_mae": mae[0],
        "second_r4_mae": mae[1],
        "interpretation": "second-order AFM preprocessing changes the target definition and downstream model behavior; superiority requires joint QC, repeatability, OOF, and descriptor evidence.",
    }
    write_json(summary, out / "first_vs_second_order_summary.json")
    return paths
