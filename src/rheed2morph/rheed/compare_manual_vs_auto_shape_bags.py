"""Compare manual-selected and auto-candidate shape bags, or prepare review queue."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rheed2morph.rheed.shape_bag_trustworthy_utils import display_path, finite_float, load_dataset, read_csv, resolve_path, write_csv, write_json


def _manual_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    rows = read_csv(path)
    return [row for row in rows if any(str(value).strip() for key, value in row.items() if key != "sample_id")]


def _prediction_error_by_sample(mvp9_root: Path) -> dict[str, float]:
    path = mvp9_root / "predicted_conditions" / "predicted_condition_table_val.csv"
    if not path.is_file():
        return {}
    rows = read_csv(path)
    errors: dict[str, list[float]] = {}
    for row in rows:
        sample = row.get("sample_id", "")
        for key, value in row.items():
            if not key.startswith("pred_") or key.startswith("pred_cond_"):
                continue
            desc = key[len("pred_") :]
            true = finite_float(row.get(f"true_{desc}", row.get(desc, "nan")))
            pred = finite_float(value)
            if np.isfinite(true) and np.isfinite(pred):
                errors.setdefault(sample, []).append(abs(pred - true))
    return {sample: float(np.mean(vals)) for sample, vals in errors.items() if vals}


def run_manual_vs_auto(args: argparse.Namespace) -> dict[str, Any]:
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = resolve_path(args.root)
    mvp8 = resolve_path(args.mvp8_root)
    mvp9 = resolve_path(args.mvp9_root)
    manual_path = root / "manual_selected_frame_manifest.csv"
    manual = _manual_rows(manual_path)
    bundle = load_dataset(mvp8, mvp9, mvp9 / "data" / "target_schema_shape_bag.json")
    paired_samples = {row["sample_id"] for row in bundle.index_rows}
    manual_paired = [row for row in manual if row.get("sample_id", "") in paired_samples]
    status_rows = [
        {
            "manual_manifest": display_path(manual_path),
            "manual_manifest_exists": manual_path.is_file(),
            "manual_rows": len(manual),
            "paired_manual_rows": len(manual_paired),
            "status": "enough_for_comparison" if len(manual_paired) >= 10 else "pending_manual_selection",
        }
    ]
    write_csv(out / "manual_selection_status.csv", status_rows)
    error_by_sample = _prediction_error_by_sample(mvp9)
    priority_rows: list[dict[str, Any]] = []
    for row in bundle.index_rows:
        feature_values = {key[len("shape_feature::") :]: value for key, value in row.items() if key.startswith("shape_feature::")}
        confidence = max(
            [finite_float(value, float("nan")) for key, value in feature_values.items() if "mask_confidence" in key] or [float("nan")]
        )
        snr = max([finite_float(value, float("nan")) for key, value in feature_values.items() if "snr_score" in key] or [float("nan")])
        error = error_by_sample.get(row["sample_id"], float("nan"))
        priority = 0.0
        if np.isfinite(error):
            priority += error
        if np.isfinite(confidence):
            priority += max(0.0, 1.0 - confidence)
        if np.isfinite(snr):
            priority += max(0.0, 1.0 - snr)
        priority_rows.append(
            {
                "sample_id": row["sample_id"],
                "pair_id": row["pair_id"],
                "split": row.get("split", ""),
                "priority_score": priority,
                "prediction_error_proxy": error,
                "mask_confidence_proxy": confidence,
                "snr_score_proxy": snr,
                "shape_input_folder": row.get("shape_input_folder", ""),
                "preview_grid": row.get("preview_grid", ""),
                "reason": "high error / low confidence / review coverage",
            }
        )
    priority_rows.sort(key=lambda row: finite_float(row.get("priority_score", 0.0), 0.0), reverse=True)
    write_csv(out / "prioritized_manual_review_samples.csv", priority_rows)
    metrics_rows: list[dict[str, Any]] = []
    if len(manual_paired) >= 10:
        metrics_rows.append({"comparison": "manual_vs_auto", "paired_manual_rows": len(manual_paired), "status": "manual comparison requires manual feature rebuild hook"})
    write_csv(out / "manual_vs_auto_metrics.csv", metrics_rows)
    guide = [
        "# Manual Review Guide",
        "",
        "Review the highest priority samples first. Priority combines validation prediction error when available, low mask-confidence proxy, and low SNR proxy.",
        "",
        f"Prioritized list: `{display_path(out / 'prioritized_manual_review_samples.csv')}`",
    ]
    (out / "manual_review_guide.md").write_text("\n".join(guide) + "\n", encoding="utf-8")
    report = [
        "# Manual vs Auto Shape-Bag Report",
        "",
        f"Manual manifest exists: `{manual_path.is_file()}`",
        f"Manual rows: `{len(manual)}`",
        f"Paired manual rows: `{len(manual_paired)}`",
        f"Status: `{status_rows[0]['status']}`",
        "",
        "If paired manual selections reach at least 10 samples, rerun this tool after rebuilding manual-only shape-bag features for a quantitative comparison.",
    ]
    (out / "manual_vs_auto_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    summary = {"status": status_rows[0]["status"], "manual_rows": len(manual), "paired_manual_rows": len(manual_paired)}
    write_json(out / "manual_vs_auto_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--mvp8-root", required=True)
    parser.add_argument("--mvp9-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_manual_vs_auto(args)
    print(f"Wrote manual-vs-auto status to {display_path(resolve_path(args.out))}")
    print(f"status={summary['status']} paired_manual_rows={summary['paired_manual_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
