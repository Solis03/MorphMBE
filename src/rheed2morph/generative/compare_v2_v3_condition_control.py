"""Compare MVP-3 v2 and AFM prior v3 condition-control metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from rheed2morph.generative.common import display_path, read_csv_rows, read_json, resolve_repo_path, write_csv_rows, write_json
from rheed2morph.generative.condition_control_v3_utils import finite_float


KEY_DESCRIPTOR_NAMES = {
    "rq",
    "ra",
    "height_std",
    "robust_range",
    "psd_low_power",
    "psd_mid_power",
    "psd_high_power",
    "psd_slope",
    "autocorrelation_length_px",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare v2 and v3 condition-control summaries.")
    parser.add_argument("--v2-sensitivity", type=Path, required=True)
    parser.add_argument("--v3-evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def _summary_json(path: Path, filename: str) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    if resolved.is_dir():
        resolved = resolved / filename
    return read_json(resolved) if resolved.is_file() else {}


def _metric_rows(path: Path) -> list[dict[str, str]]:
    resolved = resolve_repo_path(path)
    if resolved.is_dir():
        resolved = resolved / "condition_control_metrics_v3.csv"
    return read_csv_rows(resolved) if resolved.is_file() else []


def _write_plot(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [str(row["descriptor"]) for row in rows if np.isfinite(finite_float(row.get("v3_abs_pearson", "nan")))]
    if not names:
        return
    x = np.arange(len(names))
    v2 = [finite_float(row.get("v2_abs_pearson", "nan")) for row in rows if row["descriptor"] in names]
    v3 = [finite_float(row.get("v3_abs_pearson", "nan")) for row in rows if row["descriptor"] in names]
    fig, ax = plt.subplots(figsize=(max(6, 0.65 * len(names)), 4), dpi=150)
    ax.bar(x - 0.2, v2, width=0.4, label="v2 sensitivity")
    ax.bar(x + 0.2, v3, width=0.4, label="v3 evaluation")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("abs Pearson")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "v2_vs_v3_condition_control_summary.png")
    plt.close(fig)


def compare(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    v2_summary = _summary_json(args.v2_sensitivity, "v2_condition_sensitivity_summary.json")
    v3_summary = _summary_json(args.v3_evaluation, "condition_control_summary_v3.json")
    v3_rows = _metric_rows(args.v3_evaluation)
    v2_by_descriptor = {
        str(row.get("descriptor")): row
        for row in v2_summary.get("descriptor_summaries", [])
        if isinstance(row, dict) and row.get("descriptor")
    }
    v3_by_descriptor = {str(row.get("descriptor")): row for row in v3_rows if row.get("descriptor")}
    descriptors = sorted(set(v2_by_descriptor) | set(v3_by_descriptor))
    comparison_rows: list[dict[str, Any]] = []
    for descriptor in descriptors:
        v2 = v2_by_descriptor.get(descriptor, {})
        v3 = v3_by_descriptor.get(descriptor, {})
        v2_abs = finite_float(v2.get("best_abs_pearson", v2.get("abs_pearson", "nan")))
        v3_pearson = finite_float(v3.get("pearson", "nan"))
        v3_abs = abs(v3_pearson) if np.isfinite(v3_pearson) else float("nan")
        v2_mae = finite_float(v2.get("mae", "nan"))
        v3_mae = finite_float(v3.get("mae", "nan"))
        comparison_rows.append(
            {
                "descriptor": descriptor,
                "is_key_descriptor": descriptor in KEY_DESCRIPTOR_NAMES,
                "v2_abs_pearson": v2_abs,
                "v3_abs_pearson": v3_abs,
                "v2_mae": v2_mae,
                "v3_mae": v3_mae,
                "v2_best_guidance_scale": v2.get("best_guidance_scale", ""),
                "v3_monotonicity": v3.get("monotonicity", ""),
                "improved_abs_pearson": bool(np.isfinite(v2_abs) and np.isfinite(v3_abs) and v3_abs > v2_abs),
                "improved_mae": bool(np.isfinite(v2_mae) and np.isfinite(v3_mae) and v3_mae < v2_mae),
            }
        )
    write_csv_rows(
        out_dir / "v2_vs_v3_condition_control_metrics.csv",
        comparison_rows,
        fieldnames=[
            "descriptor",
            "is_key_descriptor",
            "v2_abs_pearson",
            "v3_abs_pearson",
            "v2_mae",
            "v3_mae",
            "v2_best_guidance_scale",
            "v3_monotonicity",
            "improved_abs_pearson",
            "improved_mae",
        ],
    )
    _write_plot(out_dir, comparison_rows)
    comparable = [row for row in comparison_rows if np.isfinite(finite_float(row["v2_abs_pearson"])) and np.isfinite(finite_float(row["v3_abs_pearson"]))]
    key_rows = [row for row in comparable if row["is_key_descriptor"]]
    improved_corr = [str(row["descriptor"]) for row in comparable if row["improved_abs_pearson"]]
    improved_mae = [str(row["descriptor"]) for row in comparable if row["improved_mae"]]
    summary = {
        "v2_sensitivity": display_path(resolve_repo_path(args.v2_sensitivity)),
        "v3_evaluation": display_path(resolve_repo_path(args.v3_evaluation)),
        "descriptor_count": len(comparison_rows),
        "comparable_descriptor_count": len(comparable),
        "comparable_key_descriptor_count": len(key_rows),
        "improved_abs_pearson_descriptors": improved_corr,
        "improved_mae_descriptors": improved_mae,
        "v3_plain_descriptor_error": v3_summary.get("plain_descriptor_error", ""),
        "v3_guided_descriptor_error": v3_summary.get("guided_descriptor_error", ""),
        "v3_reranked_descriptor_error": v3_summary.get("reranked_descriptor_error", ""),
        "v3_generated_nonconstant_rate": v3_summary.get("generated_nonconstant_rate", ""),
        "comparison_metrics": display_path(out_dir / "v2_vs_v3_condition_control_metrics.csv"),
        "comparison_plot": display_path(out_dir / "v2_vs_v3_condition_control_summary.png"),
    }
    write_json(out_dir / "v2_vs_v3_condition_control_summary.json", summary)
    report = [
        "# V2 vs V3 Condition-Control Comparison",
        "",
        f"Comparable descriptors: `{len(comparable)}`",
        f"Key comparable descriptors: `{len(key_rows)}`",
        f"Improved abs Pearson descriptors: `{improved_corr}`",
        f"Improved MAE descriptors: `{improved_mae}`",
        f"V3 reranked descriptor error: `{v3_summary.get('reranked_descriptor_error', '')}`",
        "",
        "This comparison uses the available v2 sweep summary and v3 evaluation summary; it is a diagnostic comparison, not proof of exact AFM reconstruction.",
    ]
    (out_dir / "v2_vs_v3_condition_control_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = compare(args)
    print(f"Wrote v2/v3 condition-control comparison to {display_path(resolve_repo_path(args.out))}")
    print(f"comparable_descriptors={summary['comparable_descriptor_count']}")


if __name__ == "__main__":
    main()
