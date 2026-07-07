"""Generate MVP-10 evidence package for advisor/paper review."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from rheed2morph.rheed.shape_bag_trustworthy_utils import display_path, read_csv, read_json, resolve_path, write_csv, write_json


def _exists(path: Path) -> str:
    return "yes" if path.exists() else "no"


def generate_package(args: argparse.Namespace) -> dict[str, Any]:
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = resolve_path(args.mvp10_root)
    mvp9 = resolve_path(args.mvp9_root)
    mvp8 = resolve_path(args.mvp8_root)
    mvp5 = resolve_path(args.mvp5_root)
    cv_summary = read_json(root / "strict_descriptor_cv" / "strict_descriptor_cv_summary.json")
    prod = read_json(root / "production_model_selection" / "production_model_selection_summary.json")
    neg = read_json(root / "negative_controls" / "negative_control_summary.json")
    gen = read_json(root / "trustworthy_calibrated_v2_generation" / "trustworthy_generation_summary.json")
    selected_count = int(prod.get("selected_descriptor_count", 0) or 0)
    negative_pass = bool(neg.get("negative_controls_pass", False))
    claim_rows = [
        {"claim": "End-to-end pipeline works", "support": "SUPPORTED", "evidence": display_path(root)},
        {"claim": "AFM prior generates realistic representative AFM", "support": "SUPPORTED", "evidence": display_path(mvp5 / "codex_report.md")},
        {"claim": "height calibration improves roughness/range", "support": "SUPPORTED", "evidence": display_path(mvp5 / "calibrated_v2_v3" / "calibrated_generation_summary.json")},
        {"claim": "RHEED shape-bag features beat mean baseline", "support": "SUPPORTED" if selected_count else "NOT_SUPPORTED", "evidence": display_path(root / "strict_descriptor_cv" / "descriptor_predictability_table.csv")},
        {"claim": "RHEED shape-bag beats metadata-only", "support": "WEAK", "evidence": "MVP-10 metadata production input was not selected; MVP-6 metadata baseline is contextual only."},
        {"claim": "RHEED shape-bag beats brightness-only", "support": "SUPPORTED" if negative_pass else "UNRELIABLE", "evidence": display_path(root / "negative_controls" / "negative_control_report.md")},
        {"claim": "negative controls pass", "support": "SUPPORTED" if negative_pass else "UNRELIABLE", "evidence": display_path(root / "negative_controls" / "negative_control_summary.json")},
        {"claim": "elongated/bar-like RHEED features correlate with AFM descriptors", "support": "WEAK", "evidence": display_path(root / "feature_importance" / "elongated_bar_feature_report.md")},
        {"claim": "generated AFM differs meaningfully from mean-condition", "support": "WEAK", "evidence": display_path(root / "trustworthy_calibrated_v2_generation" / "trustworthy_generation_summary.json")},
        {"claim": "exact pixel-level AFM reconstruction is possible", "support": "NOT_SUPPORTED", "evidence": "MVP-10 is descriptor/prototype prediction plus representative generation only."},
    ]
    write_csv(out / "claim_support_matrix.csv", claim_rows)
    figures = [
        root / "strict_descriptor_cv" / "predicted_vs_true_by_descriptor.png",
        root / "strict_descriptor_cv" / "descriptor_error_heatmap.png",
        root / "feature_importance" / "grouped_feature_importance.png",
        root / "trustworthy_calibrated_v2_generation" / "trustworthy_shape_bag_calibrated_v2_grid.png",
    ]
    tables = [
        root / "strict_descriptor_cv" / "cv_metrics_summary.csv",
        root / "strict_descriptor_cv" / "descriptor_predictability_table.csv",
        root / "negative_controls" / "negative_control_metrics.csv",
        root / "feature_importance" / "feature_importance_summary.csv",
        root / "production_predictions" / "predicted_condition_table_oof_production.csv",
    ]
    write_csv(out / "figure_manifest.csv", [{"path": display_path(path), "exists": _exists(path)} for path in figures])
    write_csv(out / "table_manifest.csv", [{"path": display_path(path), "exists": _exists(path)} for path in tables])
    advisor = [
        "# Advisor Summary",
        "",
        "MVP-10 tests whether RHEED shape-bag features support descriptor-wise AFM morphology prediction under strict validation.",
        f"Selected production descriptors: `{prod.get('selected_descriptors', [])}`",
        f"Negative controls pass: `{negative_pass}`",
        "Exact pixel-level AFM reconstruction remains not supported.",
    ]
    (out / "advisor_summary.md").write_text("\n".join(advisor) + "\n", encoding="utf-8")
    package_report = [
        "# Evidence Package Report",
        "",
        f"MVP-8 root: `{display_path(mvp8)}`",
        f"MVP-9 root: `{display_path(mvp9)}`",
        f"MVP-10 root: `{display_path(root)}`",
        "",
        f"Strict CV descriptors: `{cv_summary.get('target_descriptors', [])}`",
        f"Selected descriptors: `{prod.get('selected_descriptors', [])}`",
        f"Generation summary: `{display_path(root / 'trustworthy_calibrated_v2_generation' / 'trustworthy_generation_summary.json')}`",
    ]
    (out / "evidence_package_report.md").write_text("\n".join(package_report) + "\n", encoding="utf-8")
    paper = [
        "# Paper Results Draft",
        "",
        "RHEED shape-bag stable geometry features were evaluated descriptor-wise against train-fold mean, brightness/exposure, random-feature, shuffled-label, and forbidden-feature diagnostics.",
        "Only descriptors passing the production policy are allowed to use RHEED-predicted conditions in representative calibrated_v2 generation.",
    ]
    (out / "paper_results_draft.md").write_text("\n".join(paper) + "\n", encoding="utf-8")
    limits = [
        "# Limitations And Next Steps",
        "",
        "- Small supervised matched set.",
        "- Descriptor support is policy-gated; unsupported descriptors are filled by train mean.",
        "- Manual frame selection remains pending or incomplete unless enough manual selections are present.",
        "- Exact AFM reconstruction is not supported.",
    ]
    (out / "limitations_and_next_steps.md").write_text("\n".join(limits) + "\n", encoding="utf-8")
    summary = {"claim_support_matrix": display_path(out / "claim_support_matrix.csv"), "selected_descriptor_count": selected_count}
    write_json(out / "evidence_package_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvp10-root", required=True)
    parser.add_argument("--mvp9-root", required=True)
    parser.add_argument("--mvp8-root", required=True)
    parser.add_argument("--mvp5-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = generate_package(args)
    print(f"Wrote evidence package to {display_path(resolve_path(args.out))}")
    print(f"claim_support_matrix={summary['claim_support_matrix']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
