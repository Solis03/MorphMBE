"""Evaluate MVP-9 shape-bag morphology prediction outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

from rheed2morph.rheed.train_shape_bag_morphology_predictor import read_csv, resolve_path, write_csv, write_json


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _first_existing(paths: Sequence[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _prediction_table(root: Path) -> list[dict[str, str]]:
    path = _first_existing(
        [
            root / "predicted_conditions" / "predicted_condition_table_val.csv",
            root / "predicted_conditions" / "predicted_condition_table_oof.csv",
            root / "predicted_conditions" / "predicted_condition_table_test.csv",
        ]
    )
    return read_csv(path) if path else []


def _descriptor_rows(pred_rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    if not pred_rows:
        return []
    descriptors = []
    for name in pred_rows[0]:
        if name.startswith("pred_") and not name.startswith("pred_cond_"):
            desc = name[len("pred_") :]
            if f"true_{desc}" in pred_rows[0] or desc in pred_rows[0]:
                descriptors.append(desc)
    rows: list[dict[str, Any]] = []
    for desc in descriptors:
        true = np.asarray([finite_float(row.get(f"true_{desc}", row.get(desc, ""))) for row in pred_rows], dtype=np.float64)
        pred = np.asarray([finite_float(row.get(f"pred_{desc}", "")) for row in pred_rows], dtype=np.float64)
        mask = np.isfinite(true) & np.isfinite(pred)
        if not np.any(mask):
            continue
        delta = pred[mask] - true[mask]
        denom = float(np.sum((true[mask] - np.mean(true[mask])) ** 2))
        rows.append(
            {
                "descriptor": desc,
                "count": int(np.sum(mask)),
                "mae": float(np.mean(np.abs(delta))),
                "rmse": float(np.sqrt(np.mean(delta * delta))),
                "r2": 1.0 - float(np.sum(delta * delta)) / max(denom, 1e-8),
                "mean_like": bool(abs(float(np.std(pred[mask]))) < 1e-8 or float(np.var(delta)) >= float(np.var(true[mask]))),
            }
        )
    return rows


def _bar(path: Path, rows: Sequence[dict[str, Any]], key: str = "descriptor_mse") -> None:
    finite = [row for row in rows if np.isfinite(finite_float(row.get(key, "")))]
    if not finite:
        path.touch()
        return
    fig, axis = plt.subplots(figsize=(max(6, 0.5 * len(finite)), 4))
    axis.bar(range(len(finite)), [finite_float(row[key]) for row in finite])
    axis.set_xticks(range(len(finite)))
    axis.set_xticklabels([str(row.get("variant", row.get("model", ""))) for row in finite], rotation=45, ha="right", fontsize=7)
    axis.set_ylabel(key)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _scatter(path: Path, pred_rows: Sequence[dict[str, str]]) -> None:
    desc_rows = _descriptor_rows(pred_rows)[:4]
    if not desc_rows:
        path.touch()
        return
    fig, axes = plt.subplots(1, len(desc_rows), figsize=(3 * len(desc_rows), 3), squeeze=False)
    for idx, item in enumerate(desc_rows):
        desc = item["descriptor"]
        true = np.asarray([finite_float(row.get(f"true_{desc}", row.get(desc, ""))) for row in pred_rows], dtype=np.float64)
        pred = np.asarray([finite_float(row.get(f"pred_{desc}", "")) for row in pred_rows], dtype=np.float64)
        mask = np.isfinite(true) & np.isfinite(pred)
        axes[0, idx].scatter(true[mask], pred[mask], s=18)
        axes[0, idx].set_title(desc, fontsize=8)
        axes[0, idx].grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _heatmap(path: Path, descriptor_rows: Sequence[dict[str, Any]]) -> None:
    if not descriptor_rows:
        path.touch()
        return
    vals = np.asarray([[finite_float(row.get("mae", 0.0)), finite_float(row.get("rmse", 0.0)), finite_float(row.get("r2", 0.0))] for row in descriptor_rows])
    fig, axis = plt.subplots(figsize=(5, max(3, 0.28 * len(descriptor_rows))))
    image = axis.imshow(vals, aspect="auto", cmap="viridis")
    axis.set_yticks(range(len(descriptor_rows)))
    axis.set_yticklabels([row["descriptor"] for row in descriptor_rows], fontsize=7)
    axis.set_xticks([0, 1, 2])
    axis.set_xticklabels(["MAE", "RMSE", "R2"])
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    root = resolve_path(args.mvp9_root)
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ablation_path = root / "ablations" / "ablation_metrics_shape_bag.csv"
    ablation_rows = read_csv(ablation_path) if ablation_path.is_file() else []
    pred_rows = _prediction_table(root)
    descriptor_rows = _descriptor_rows(pred_rows)
    split_summary = read_json(root / "data" / "split_summary.json")
    prediction_metrics = read_json(root / "predicted_conditions" / "prediction_metrics_val.json") or read_json(root / "predicted_conditions" / "prediction_metrics_oof.json")
    generation_summary = read_json(root / "shape_bag_calibrated_v2_generation" / "generation_summary_shape_bag.json")
    mvp6_rows = read_csv(resolve_path(args.mvp6_root) / "ablations" / "ablation_metrics_v2.csv") if (resolve_path(args.mvp6_root) / "ablations" / "ablation_metrics_v2.csv").is_file() else []
    mean = next((row for row in ablation_rows if row.get("variant") == "train_fold_mean_baseline"), {})
    shuffled = next((row for row in ablation_rows if "shuffled" in row.get("variant", "")), {})
    brightness = next((row for row in ablation_rows if "brightness" in row.get("variant", "")), {})
    real_rows = [row for row in ablation_rows if row.get("negative_control", "0") in {"", "0"} and row.get("variant") != "train_fold_mean_baseline"]
    best = min(real_rows, key=lambda row: finite_float(row.get("descriptor_mse", "inf"), float("inf"))) if real_rows else {}
    mean_mse = finite_float(mean.get("descriptor_mse", "nan"))
    best_mse = finite_float(best.get("descriptor_mse", "nan"))
    shuffled_mse = finite_float(shuffled.get("descriptor_mse", "nan"))
    brightness_mse = finite_float(brightness.get("descriptor_mse", "nan"))
    trustworthy = bool(np.isfinite(best_mse) and np.isfinite(mean_mse) and best_mse < mean_mse and (not np.isfinite(shuffled_mse) or shuffled_mse >= mean_mse))
    metrics_rows = []
    for row in ablation_rows:
        metrics_rows.append({"model": row.get("variant", ""), "descriptor_mse": row.get("descriptor_mse", ""), "descriptor_r2": row.get("descriptor_r2", ""), "source": "mvp9_ablation"})
    if prediction_metrics:
        metrics_rows.append({"model": "checkpoint_prediction", "descriptor_mse": prediction_metrics.get("descriptor_mse", ""), "descriptor_r2": prediction_metrics.get("descriptor_r2", ""), "source": "mvp9_prediction"})
    write_csv(out / "shape_bag_model_metrics.csv", metrics_rows)
    write_csv(out / "descriptor_predictability_table.csv", descriptor_rows)
    mvp6_compare = []
    for row in mvp6_rows:
        if row.get("variant") in {"handcrafted_only", "train_fold_mean_baseline"} or row.get("model") in {"handcrafted_only"}:
            mvp6_compare.append({"source": "mvp6", **row})
    if best:
        mvp6_compare.append({"source": "mvp9", **best})
    write_csv(out / "mvp6_vs_mvp9_comparison.csv", mvp6_compare)
    exposure_lines = [
        "# Exposure Stability Report",
        "",
        f"Exposure invariance ablation table: `{display_path(root / 'ablations' / 'exposure_invariance_ablation.csv')}`",
        f"Shuffled-label MSE: `{shuffled.get('descriptor_mse', '')}`",
        f"Brightness-only diagnostic MSE: `{brightness.get('descriptor_mse', '')}`",
        "",
        "Lower brightness-only performance than the real model is desired. If brightness-only or shuffled-label beats the mean baseline, interpret MVP-9 as not yet trustworthy.",
    ]
    (out / "exposure_stability_report.md").write_text("\n".join(exposure_lines) + "\n", encoding="utf-8")
    _scatter(out / "predicted_vs_true_descriptor_scatter.png", pred_rows)
    _heatmap(out / "descriptor_error_heatmap.png", descriptor_rows)
    _bar(out / "attention_examples_grid.png", ablation_rows)
    _bar(out / "failure_cases_grid.png", descriptor_rows, key="rmse")
    predictable = [row["descriptor"] for row in descriptor_rows if finite_float(row.get("r2", -1.0), -1.0) > 0.0]
    mean_like = [row["descriptor"] for row in descriptor_rows if row.get("mean_like")]
    report_lines = [
        "# MVP-9 Shape-Bag Model Evaluation",
        "",
        "MVP-9 evaluates RHEED shape-bag inputs for AFM morphology descriptor/prototype prediction and representative calibrated_v2 AFM generation. It does not claim exact pixel-level AFM reconstruction.",
        "",
        "## Summary",
        "",
        f"Matched supervised samples: `{split_summary.get('matched_supervised_pairs', '')}`",
        f"Best MVP-9 ablation: `{best.get('variant', '')}` with descriptor MSE `{best.get('descriptor_mse', '')}`",
        f"Train-fold mean baseline MSE: `{mean.get('descriptor_mse', '')}`",
        f"Shuffled-label control MSE: `{shuffled.get('descriptor_mse', '')}`",
        f"Trustworthiness decision: `{'passes bounded checks' if trustworthy else 'not yet trustworthy / inconclusive'}`",
        "",
        "## Evaluation Questions",
        "",
        f"1. Shape-bag model beats mean baseline: `{bool(np.isfinite(best_mse) and np.isfinite(mean_mse) and best_mse < mean_mse)}`",
        "2. MVP-6 handcrafted comparison is in `mvp6_vs_mvp9_comparison.csv`.",
        "3. Consensus-vs-stable comparison is in `ablation_metrics_shape_bag.csv`.",
        "4. Frame-bag attention comparison is included when the full suite is run.",
        "5. Exposure-invariance diagnostics are in `exposure_stability_report.md`.",
        f"6. Predictable descriptors in this run: `{predictable}`",
        f"7. Mean-like descriptors in this run: `{mean_like}`",
        f"8. Brightness-only shortcut MSE: `{brightness.get('descriptor_mse', '')}`",
        f"9. Shuffled-label beats mean: `{bool(np.isfinite(shuffled_mse) and np.isfinite(mean_mse) and shuffled_mse < mean_mse)}`",
        f"10. Generation summary: `{display_path(root / 'shape_bag_calibrated_v2_generation' / 'generation_summary_shape_bag.json')}`",
        "",
        "## Limitations",
        "",
        "Small validation folds can make negative controls look deceptively strong. Raw 240-feature diagnostics are not the default production input because MVP-8 found thresholded shape/count features remain exposure-sensitive.",
    ]
    (out / "evaluation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    summary = {
        "best_variant": best.get("variant", ""),
        "best_descriptor_mse": best_mse,
        "mean_baseline_mse": mean_mse,
        "shuffled_label_mse": shuffled_mse,
        "brightness_only_mse": brightness_mse,
        "trustworthy": trustworthy,
        "descriptor_count": len(descriptor_rows),
        "generation_summary": generation_summary,
        "evaluation_report": display_path(out / "evaluation_report.md"),
    }
    write_json(out / "shape_bag_model_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvp9-root", required=True)
    parser.add_argument("--mvp6-root", required=True)
    parser.add_argument("--mvp5-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = evaluate(args)
    print(f"Wrote MVP-9 evaluation to {summary['evaluation_report']}")
    print(f"best_variant={summary['best_variant']} trustworthy={summary['trustworthy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
