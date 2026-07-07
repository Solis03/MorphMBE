"""Select MVP-10 production descriptor policy from strict CV evidence."""

from __future__ import annotations

import argparse
from typing import Any, Sequence

from rheed2morph.rheed.shape_bag_trustworthy_utils import display_path, finite_float, read_csv, read_json, resolve_path, write_csv, write_json


ELIGIBLE_FEATURE_SETS = {"stable36", "stable36_plus_consensus_summary", "stable36_plus_metadata"}


def _best_rows(cv_rows: Sequence[dict[str, str]]) -> dict[str, dict[str, str]]:
    by_desc: dict[str, dict[str, str]] = {}
    for row in cv_rows:
        if int(finite_float(row.get("negative_control", "0"), 0.0)):
            continue
        if row.get("model") == "mean":
            continue
        if row.get("feature_set") not in ELIGIBLE_FEATURE_SETS:
            continue
        desc = row.get("descriptor", "")
        if not desc:
            continue
        if desc not in by_desc or finite_float(row.get("mse", "inf"), float("inf")) < finite_float(by_desc[desc].get("mse", "inf"), float("inf")):
            by_desc[desc] = row
    return by_desc


def select_model(args: argparse.Namespace) -> dict[str, Any]:
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv_root = resolve_path(args.cv_root)
    neg_root = resolve_path(args.negative_control_root)
    fi_root = resolve_path(args.feature_importance_root)
    cv_rows = read_csv(cv_root / "cv_metrics_summary.csv")
    desc_rows = read_csv(cv_root / "descriptor_predictability_table.csv")
    neg_summary = read_json(neg_root / "negative_control_summary.json")
    grouped_importance = read_csv(fi_root / "grouped_feature_importance.csv") if (fi_root / "grouped_feature_importance.csv").is_file() else []
    negative_pass = bool(neg_summary.get("negative_controls_pass", False))
    best_by_desc = _best_rows(cv_rows)
    selected: dict[str, Any] = {}
    policy_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    for desc_item in desc_rows:
        descriptor = desc_item["descriptor"]
        best = best_by_desc.get(descriptor, {})
        original_label = desc_item.get("trust_label", "NOT_SUPPORTED")
        has_physical_group = any(row.get("descriptor") == descriptor and finite_float(row.get("mean_importance_delta_mse", "nan"), -1.0) > 0 for row in grouped_importance)
        eligible = (
            negative_pass
            and original_label in {"SUPPORTED", "WEAK"}
            and bool(best)
            and finite_float(best.get("paired_improvement_over_mean_mse", "nan"), -1.0) > 0.0
            and best.get("feature_set") in ELIGIBLE_FEATURE_SETS
        )
        if eligible:
            selected[descriptor] = {
                "model": best["model"],
                "feature_set": best["feature_set"],
                "trust_label": original_label,
                "mse": finite_float(best.get("mse", "nan")),
                "mean_baseline_mse": finite_float(best.get("mean_baseline_mse", "nan")),
            }
            action = "use_rheed_prediction"
        else:
            action = "fill_train_mean"
        policy_rows.append(
            {
                "descriptor": descriptor,
                "original_trust_label": original_label,
                "production_action": action,
                "selected_model": best.get("model", ""),
                "selected_feature_set": best.get("feature_set", ""),
                "negative_controls_pass": negative_pass,
                "physical_importance_present": has_physical_group,
                "reason": "passes descriptor policy" if eligible else "unsupported or control risk; use train mean",
            }
        )
    write_json(
        out / "selected_model_config.json",
        {
            "selection_type": "descriptorwise_policy",
            "eligible_feature_sets": sorted(ELIGIBLE_FEATURE_SETS),
            "negative_controls_pass": negative_pass,
            "selected_descriptors": selected,
            "cv_root": display_path(cv_root),
        },
    )
    write_json(out / "selected_descriptor_subset.json", {"descriptors": sorted(selected)})
    write_json(out / "unsupported_descriptor_policy.json", {"policy": policy_rows, "default_action": "fill_train_mean_for_unsupported_descriptors"})
    write_csv(out / "descriptor_policy_table.csv", policy_rows)
    claim_rows.extend(
        [
            {"claim": "RHEED shape-bag features beat mean baseline", "support": "SUPPORTED" if selected else "NOT_SUPPORTED", "evidence": display_path(cv_root / "descriptor_predictability_table.csv")},
            {"claim": "RHEED shape-bag beats brightness/exposure diagnostics", "support": "SUPPORTED" if negative_pass else "UNRELIABLE", "evidence": display_path(neg_root / "negative_control_summary.json")},
            {"claim": "negative controls pass", "support": "SUPPORTED" if negative_pass else "UNRELIABLE", "evidence": display_path(neg_root / "negative_control_report.md")},
            {"claim": "raw 240 features are production default", "support": "NOT_SUPPORTED", "evidence": "Raw features remain diagnostic only."},
            {"claim": "exact pixel-level AFM reconstruction is possible", "support": "NOT_SUPPORTED", "evidence": "MVP-10 predicts descriptors for representative generation only."},
        ]
    )
    write_csv(out / "claim_support_matrix_shape_bag.csv", claim_rows)
    report = [
        "# Production Shape-Bag Model Selection",
        "",
        f"Negative controls pass: `{negative_pass}`",
        f"Selected descriptors: `{sorted(selected)}`",
        "",
        "Unsupported descriptors are filled with train means before calibrated_v2 generation. This prevents unsupported RHEED condition claims from entering the generation table.",
    ]
    (out / "production_model_selection_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    summary = {"selected_descriptor_count": len(selected), "selected_descriptors": sorted(selected), "negative_controls_pass": negative_pass}
    write_json(out / "production_model_selection_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-root", required=True)
    parser.add_argument("--negative-control-root", required=True)
    parser.add_argument("--feature-importance-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = select_model(args)
    print(f"Wrote production selection to {display_path(resolve_path(args.out))}")
    print(f"selected_descriptor_count={summary['selected_descriptor_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
