"""Summarize v2/v3/v4 AFM generation comparison artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rheed2morph.generative.common import display_path, read_csv_rows, read_json, resolve_repo_path, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare v2/v3/v4 generation outputs.")
    parser.add_argument("--evaluation-v4", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def compare(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluation = resolve_repo_path(args.evaluation_v4)
    summary = read_json(evaluation / "afm_prior_v4_summary.json")
    rows = read_csv_rows(evaluation / "v2_v3_v4_descriptor_comparison.csv")
    roughness_rows = [row for row in rows if row.get("descriptor") in {"rq", "ra", "robust_range"}]
    report_lines = [
        "# V2/V3/V4 Generation Comparison",
        "",
        f"Recommended primary prior: `{summary.get('recommended_primary_prior', '')}`",
        f"Roughness improved: `{summary.get('roughness_improved', '')}`",
        f"Nonconstant rate: `{summary.get('nonconstant_rate', '')}`",
        "",
        "## Roughness Rows",
        "",
    ]
    for row in roughness_rows:
        report_lines.append(
            f"- `{row.get('prior')}` `{row.get('calibration_state')}` `{row.get('descriptor')}`: "
            f"MAE `{row.get('mae')}`, Pearson `{row.get('pearson')}`"
        )
    payload = {
        "evaluation_v4": display_path(evaluation),
        "recommended_primary_prior": summary.get("recommended_primary_prior", ""),
        "roughness_improved": summary.get("roughness_improved", False),
        "nonconstant_rate": summary.get("nonconstant_rate", ""),
        "comparison_rows": len(rows),
        "report": display_path(out_dir / "v2_v3_v4_generation_comparison_report.md"),
    }
    write_json(out_dir / "v2_v3_v4_generation_comparison_summary.json", payload)
    (out_dir / "v2_v3_v4_generation_comparison_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    args = build_parser().parse_args()
    summary = compare(args)
    print(f"Wrote v2/v3/v4 comparison to {display_path(resolve_repo_path(args.out))}")
    print(f"recommended_primary_prior={summary['recommended_primary_prior']}")


if __name__ == "__main__":
    main()
