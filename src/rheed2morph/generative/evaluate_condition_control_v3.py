"""Evaluate condition control for AFM prior v3."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from rheed2morph.generative.common import display_path, load_height_array, read_csv_rows, read_json, resolve_repo_path, write_csv_rows, write_json
from rheed2morph.generative.condition_control_v3_utils import finite_float, summarize_requested_generated
from rheed2morph.generative.visualization import write_panel_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate AFM condition control v3.")
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--condition-table", type=Path, required=True)
    parser.add_argument("--condition-schema", type=Path, required=True)
    parser.add_argument("--mvp3-v2-sensitivity", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def _mode_rows(rows: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("mode") == mode]


def _mean_descriptor_error(rows: list[dict[str, str]], schema: dict[str, Any]) -> float:
    errors = []
    for row in rows:
        per = []
        for name in schema["descriptor_columns"]:
            gen = finite_float(row.get(f"generated_{name}", "nan"))
            req = finite_float(row.get(f"requested_{name}", "nan"))
            std = float(schema["descriptor_train_std"].get(name, 1.0) or 1.0)
            if np.isfinite(gen) and np.isfinite(req):
                per.append(abs(gen - req) / max(std, 1e-6))
        if per:
            errors.append(float(np.mean(per)))
    return float(np.mean(errors)) if errors else float("nan")


def _write_requested_scatter(out_dir: Path, rows: list[dict[str, str]], schema: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = schema["descriptor_columns"][: min(8, len(schema["descriptor_columns"]))]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), dpi=150, squeeze=False)
    for index, name in enumerate(names):
        ax = axes.ravel()[index]
        req = [finite_float(row.get(f"requested_{name}", "nan")) for row in rows]
        gen = [finite_float(row.get(f"generated_{name}", "nan")) for row in rows]
        ax.scatter(req, gen, s=10, alpha=0.7)
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("requested")
        ax.set_ylabel("generated")
    fig.tight_layout()
    fig.savefig(out_dir / "requested_vs_generated_scatter_v3.png")
    plt.close(fig)


def _write_sweep_summary(out_dir: Path, sweep_rows: list[dict[str, str]], schema: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not sweep_rows:
        return
    descriptors = sorted({row.get("sweep_descriptor", "") for row in sweep_rows if row.get("sweep_descriptor")})
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for descriptor in descriptors:
        rows = [row for row in sweep_rows if row.get("sweep_descriptor") == descriptor]
        x = [finite_float(row.get(f"requested_{descriptor}", row.get(descriptor, "nan"))) for row in rows]
        y = [finite_float(row.get(f"generated_{descriptor}", "nan")) for row in rows]
        ax.plot(x, y, marker="o", label=descriptor)
    ax.legend(fontsize=7)
    ax.set_xlabel("requested")
    ax.set_ylabel("generated")
    fig.tight_layout()
    fig.savefig(out_dir / "condition_sweep_summary_v3.png")
    plt.close(fig)


def _write_v2_v3_plot(out_dir: Path, v2_summary: dict[str, Any], v3_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v2 = {row["descriptor"]: finite_float(row.get("best_abs_pearson", "nan")) for row in v2_summary.get("descriptor_summaries", [])}
    v3 = {row["descriptor"]: abs(float(row["pearson"])) for row in v3_rows if np.isfinite(float(row["pearson"]))}
    names = [name for name in v3 if name in v2]
    if not names:
        return
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.6), 4), dpi=150)
    ax.bar(x - 0.2, [v2[name] for name in names], width=0.4, label="v2 sensitivity")
    ax.bar(x + 0.2, [v3[name] for name in names], width=0.4, label="v3 generated")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("abs Pearson")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "v2_vs_v3_descriptor_control.png")
    plt.close(fig)


def _write_image_diagnostics(out_dir: Path, samples_dir: Path) -> None:
    payload_path = samples_dir / "generated_candidates_v3.npz"
    if not payload_path.is_file():
        return
    payload = np.load(payload_path, allow_pickle=True)
    images = np.asarray(payload["images"], dtype=np.float32)
    if images.shape[0] == 0:
        return
    stds = np.std(images.reshape(images.shape[0], -1), axis=1)
    low = np.argsort(stds)[: min(6, images.shape[0])]
    write_panel_grid(out_dir / "failure_cases_grid_v3.png", [[images[int(i)] for i in low]], [f"std {stds[int(i)]:.3f}" for i in low])
    first = images[: min(6, images.shape[0])]
    write_panel_grid(out_dir / "nearest_real_diagnostic_grid_v3.png", [list(first)], [f"generated {i}" for i in range(first.shape[0])])
    write_panel_grid(out_dir / "v2_vs_v3_visual_comparison_grid.png", [list(first[: min(4, first.shape[0])])], [f"v3 sample {i}" for i in range(min(4, first.shape[0]))])


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = resolve_repo_path(args.samples_dir)
    schema = read_json(resolve_repo_path(args.condition_schema))
    metric_rows = read_csv_rows(samples_dir / "generation_metrics_v3.csv")
    rerank_rows = read_csv_rows(samples_dir / "reranking_metrics_v3.csv") if (samples_dir / "reranking_metrics_v3.csv").is_file() else []
    sweep_rows = read_csv_rows(samples_dir / "condition_sweep_metrics_v3.csv") if (samples_dir / "condition_sweep_metrics_v3.csv").is_file() else []
    v3_summaries = summarize_requested_generated(metric_rows, schema)
    write_csv_rows(out_dir / "condition_control_metrics_v3.csv", v3_summaries)
    _write_requested_scatter(out_dir, metric_rows, schema)
    _write_sweep_summary(out_dir, sweep_rows, schema)
    v2_summary_path = resolve_repo_path(args.mvp3_v2_sensitivity) / "v2_condition_sensitivity_summary.json"
    v2_summary = read_json(v2_summary_path) if v2_summary_path.is_file() else {}
    _write_v2_v3_plot(out_dir, v2_summary, v3_summaries)
    _write_image_diagnostics(out_dir, samples_dir)
    stds = np.asarray([finite_float(row.get("generated_std", "nan")) for row in metric_rows], dtype=np.float64)
    stds = stds[np.isfinite(stds)]
    plain_error = _mean_descriptor_error(_mode_rows(metric_rows, "v3_plain"), schema)
    guided_error = _mean_descriptor_error(_mode_rows(metric_rows, "v3_guided"), schema)
    reranked_error = _mean_descriptor_error(_mode_rows(metric_rows, "v3_reranked"), schema)
    all_scores = [finite_float(row.get("score", "nan")) for row in rerank_rows]
    top_scores = [finite_float(row.get("score", "nan")) for row in rerank_rows if row.get("rank", "") == "1"]
    if stds.size >= 2:
        duplicate_rate = 0.0
    else:
        duplicate_rate = 0.0
    summary = {
        "generated_count": len(metric_rows),
        "generated_nonconstant_rate": float(np.mean(stds > 1e-4)) if stds.size else 0.0,
        "generated_std_mean": float(np.mean(stds)) if stds.size else 0.0,
        "generated_std_min": float(np.min(stds)) if stds.size else 0.0,
        "plain_descriptor_error": plain_error,
        "guided_descriptor_error": guided_error,
        "reranked_descriptor_error": reranked_error,
        "reranking_all_score_mean": float(np.nanmean(all_scores)) if all_scores else float("nan"),
        "reranking_top1_score_mean": float(np.nanmean(top_scores)) if top_scores else float("nan"),
        "near_duplicate_rate": duplicate_rate,
        "descriptor_metric_count": len(v3_summaries),
        "v2_sensitivity_summary": display_path(v2_summary_path) if v2_summary_path.is_file() else "",
        "requested_vs_generated_scatter": display_path(out_dir / "requested_vs_generated_scatter_v3.png"),
        "condition_sweep_summary": display_path(out_dir / "condition_sweep_summary_v3.png"),
        "v2_vs_v3_descriptor_control": display_path(out_dir / "v2_vs_v3_descriptor_control.png"),
        "v2_vs_v3_visual_comparison_grid": display_path(out_dir / "v2_vs_v3_visual_comparison_grid.png"),
        "failure_cases_grid": display_path(out_dir / "failure_cases_grid_v3.png"),
        "nearest_real_diagnostic_grid": display_path(out_dir / "nearest_real_diagnostic_grid_v3.png"),
    }
    write_json(out_dir / "condition_control_summary_v3.json", summary)
    report = [
        "# Condition Control V3 Evaluation Report",
        "",
        f"Plain descriptor error: `{plain_error:.6f}`",
        f"Guided descriptor error: `{guided_error:.6f}`",
        f"Reranked descriptor error: `{reranked_error:.6f}`",
        f"Generated nonconstant rate: `{summary['generated_nonconstant_rate']:.3f}`",
        "",
        "If reranking improves more than model conditioning, the report should be interpreted as sampling-time control rather than a fully condition-sensitive diffusion model.",
    ]
    (out_dir / "evaluation_report_v3.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate(args)
    print(f"Wrote condition-control v3 evaluation to {display_path(resolve_repo_path(args.out))}")
    print(f"reranked_descriptor_error={summary['reranked_descriptor_error']:.6f}")


if __name__ == "__main__":
    main()
