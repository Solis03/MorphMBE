"""Evaluate AFM prior v4 height-calibrated generation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from rheed2morph.generative.common import display_path, read_csv_rows, read_json, resolve_repo_path, write_csv_rows, write_json
from rheed2morph.generative.condition_control_v3_utils import correlation, finite_float, rank_correlation
from rheed2morph.generative.visualization import write_panel_grid


DESCRIPTORS = ("rq", "ra", "robust_range", "psd_slope", "autocorrelation_length_px", "gradient_anisotropy", "island_count")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate AFM prior v4.")
    parser.add_argument("--v4-root", type=Path, required=True)
    parser.add_argument("--mvp3-root", type=Path, required=True)
    parser.add_argument("--mvp4-root", type=Path, required=True)
    parser.add_argument("--samples-v4", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def _metric(rows: list[dict[str, str]], prior: str, calibrated: bool, descriptor: str, top_only: bool = False) -> dict[str, Any]:
    generated = []
    requested = []
    key = f"{'calibrated' if calibrated else 'uncalibrated'}_{descriptor}"
    for row in rows:
        if row.get("prior") != prior:
            continue
        if top_only and str(row.get("rank", "")) != "1":
            continue
        gen = finite_float(row.get(key, "nan"))
        req = finite_float(row.get(f"requested_{descriptor}", "nan"))
        if math.isfinite(gen) and math.isfinite(req):
            generated.append(gen)
            requested.append(req)
    if not generated:
        return {"count": 0, "mae": float("nan"), "rmse": float("nan"), "pearson": float("nan"), "spearman": float("nan")}
    gen_arr = np.asarray(generated, dtype=np.float64)
    req_arr = np.asarray(requested, dtype=np.float64)
    delta = gen_arr - req_arr
    return {
        "count": int(gen_arr.size),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "pearson": correlation(req_arr, gen_arr),
        "spearman": rank_correlation(req_arr, gen_arr),
    }


def _write_requested_plot(out_dir: Path, rows: list[dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    for ax, descriptor in zip(axes, ("rq", "ra", "robust_range")):
        for prior, marker in (("v2", "o"), ("v3", "s")):
            x = []
            y = []
            for row in rows:
                if row.get("prior") != prior or str(row.get("rank", "")) != "1":
                    continue
                req = finite_float(row.get(f"requested_{descriptor}", "nan"))
                gen = finite_float(row.get(f"calibrated_{descriptor}", "nan"))
                if math.isfinite(req) and math.isfinite(gen):
                    x.append(req)
                    y.append(gen)
            ax.scatter(x, y, marker=marker, label=prior, alpha=0.75)
        ax.set_title(descriptor)
        ax.set_xlabel("requested physical")
        ax.set_ylabel("generated calibrated")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "requested_vs_generated_roughness_v4.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(13, 7), dpi=150, squeeze=False)
    for ax, descriptor in zip(axes.ravel(), DESCRIPTORS):
        x = []
        y = []
        for row in rows:
            if str(row.get("rank", "")) != "1":
                continue
            req = finite_float(row.get(f"requested_{descriptor}", "nan"))
            gen = finite_float(row.get(f"calibrated_{descriptor}", "nan"))
            if math.isfinite(req) and math.isfinite(gen):
                x.append(req)
                y.append(gen)
        ax.scatter(x, y, s=12, alpha=0.7)
        ax.set_title(descriptor, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "requested_vs_generated_all_descriptors_v4.png")
    plt.close(fig)


def _write_distribution_plots(out_dir: Path, rows: list[dict[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    for ax, descriptor in zip(axes, ("rq", "ra", "robust_range")):
        for prior in ("v2", "v3"):
            values = [finite_float(row.get(f"calibrated_{descriptor}", "nan")) for row in rows if row.get("prior") == prior and str(row.get("rank", "")) == "1"]
            values = [value for value in values if math.isfinite(value)]
            ax.hist(values, alpha=0.5, label=prior)
        ax.set_title(descriptor)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "descriptor_distribution_v2_v3_v4.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    labels = []
    means = []
    for prior in ("v2", "v3"):
        values = [finite_float(row.get("normalized_std", "nan")) for row in rows if row.get("prior") == prior]
        values = [value for value in values if math.isfinite(value)]
        labels.append(f"{prior} normalized std")
        means.append(float(np.mean(values)) if values else 0.0)
    ax.bar(labels, means)
    ax.set_ylabel("mean normalized std")
    fig.tight_layout()
    fig.savefig(out_dir / "visual_richness_v2_v3_v4.png")
    plt.close(fig)


def evaluate_v4(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = resolve_repo_path(args.samples_v4)
    rows = read_csv_rows(samples_dir / "calibrated_generation_metrics.csv")
    calibration_rows = read_csv_rows(samples_dir / "height_calibration_metrics_v4.csv")
    sample_summary = read_json(samples_dir / "calibrated_generation_summary.json") if (samples_dir / "calibrated_generation_summary.json").is_file() else {}
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for prior in ("v2", "v3"):
        for calibrated in (False, True):
            for descriptor in DESCRIPTORS:
                metric = _metric(rows, prior, calibrated, descriptor, top_only=calibrated)
                out = {
                    "prior": prior,
                    "calibration_state": "calibrated_top1" if calibrated else "uncalibrated_all",
                    "descriptor": descriptor,
                    **metric,
                }
                metric_rows.append(out)
                if descriptor in ("rq", "ra", "robust_range"):
                    comparison_rows.append(out)
    write_csv_rows(out_dir / "afm_prior_v4_metrics.csv", metric_rows)
    write_csv_rows(out_dir / "v2_v3_v4_descriptor_comparison.csv", comparison_rows)
    _write_requested_plot(out_dir, rows)
    _write_distribution_plots(out_dir, rows)
    src_grid = samples_dir / "calibrated_v2_v3_oracle_grid_val.png"
    if src_grid.is_file():
        # Keep a same-directory named artifact expected by downstream report readers.
        import shutil

        shutil.copyfile(src_grid, out_dir / "v2_v3_v4_visual_comparison_grid.png")
    payload = samples_dir / "generated_candidates_calibrated_v2_v3.npz"
    if payload.is_file():
        data = np.load(payload, allow_pickle=True)
        images = np.asarray(data["images"], dtype=np.float32)
        if images.size:
            write_panel_grid(out_dir / "nearest_real_diagnostic_v4.png", [list(images[: min(6, len(images))])], [f"generated {i}" for i in range(min(6, len(images)))])
    clamp_values = [str(row.get("clamped", row.get("scale_clamped", ""))).lower() in {"1", "true", "yes"} for row in calibration_rows]
    v2_rq_before = _metric(rows, "v2", False, "rq")["mae"]
    v2_rq_after = _metric(rows, "v2", True, "rq", top_only=True)["mae"]
    v2_ra_before = _metric(rows, "v2", False, "ra")["mae"]
    v2_ra_after = _metric(rows, "v2", True, "ra", top_only=True)["mae"]
    v2_range_before = _metric(rows, "v2", False, "robust_range")["mae"]
    v2_range_after = _metric(rows, "v2", True, "robust_range", top_only=True)["mae"]
    stds = [finite_float(row.get("normalized_std", "nan")) for row in rows]
    stds = [value for value in stds if math.isfinite(value)]
    summary = {
        "v4_root": display_path(resolve_repo_path(args.v4_root)),
        "samples_v4": display_path(samples_dir),
        "recommended_primary_prior": sample_summary.get("recommended_primary_prior", "calibrated_v2"),
        "v2_rq_mae_before": v2_rq_before,
        "v2_rq_mae_after_top1": v2_rq_after,
        "v2_ra_mae_before": v2_ra_before,
        "v2_ra_mae_after_top1": v2_ra_after,
        "v2_robust_range_mae_before": v2_range_before,
        "v2_robust_range_mae_after_top1": v2_range_after,
        "roughness_improved": bool(v2_rq_after < v2_rq_before and v2_ra_after < v2_ra_before and v2_range_after < v2_range_before),
        "scale_clamp_rate": float(np.mean(clamp_values)) if clamp_values else 0.0,
        "nonconstant_rate": sample_summary.get("generated_nonconstant_rate", float(np.mean(np.asarray(stds) > 1e-4)) if stds else 0.0),
        "normalized_std_mean": float(np.mean(stds)) if stds else 0.0,
        "afm_prior_v4_metrics": display_path(out_dir / "afm_prior_v4_metrics.csv"),
        "descriptor_comparison": display_path(out_dir / "v2_v3_v4_descriptor_comparison.csv"),
        "roughness_plot": display_path(out_dir / "requested_vs_generated_roughness_v4.png"),
    }
    write_json(out_dir / "afm_prior_v4_summary.json", summary)
    report = [
        "# AFM Prior V4 Evaluation Report",
        "",
        f"Recommended primary prior: `{summary['recommended_primary_prior']}`",
        f"V2 Rq MAE before/after calibration: `{v2_rq_before:.6g}` / `{v2_rq_after:.6g}`",
        f"V2 Ra MAE before/after calibration: `{v2_ra_before:.6g}` / `{v2_ra_after:.6g}`",
        f"V2 robust range MAE before/after calibration: `{v2_range_before:.6g}` / `{v2_range_after:.6g}`",
        f"Scale clamp rate: `{summary['scale_clamp_rate']:.3f}`",
        "",
        "V4 here denotes the height-calibrated production path. If no separately trained v4 diffusion checkpoint is present, calibrated v2 is evaluated as the primary v4 prior.",
    ]
    (out_dir / "evaluation_report_v4.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate_v4(args)
    print(f"Wrote AFM prior v4 evaluation to {display_path(resolve_repo_path(args.out))}")
    print(f"recommended_primary_prior={summary['recommended_primary_prior']} roughness_improved={summary['roughness_improved']}")


if __name__ == "__main__":
    main()
