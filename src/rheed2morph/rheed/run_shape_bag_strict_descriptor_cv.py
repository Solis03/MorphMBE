"""Run descriptor-wise strict CV for MVP-10 shape-bag trust analysis."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Any, Sequence

import matplotlib
import numpy as np

from rheed2morph.rheed.shape_bag_trustworthy_utils import (
    bootstrap_ci,
    display_path,
    feature_matrix,
    finite_float,
    fit_predict_model,
    impute_scale_train_val,
    load_dataset,
    make_folds,
    metric_row,
    parse_list,
    read_json,
    resolve_path,
    target_vector,
    trust_label,
    write_bar_plot,
    write_csv,
    write_heatmap,
    write_json,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _target_audit(bundle: Any, folds: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fold_by_pair = {}
    for fold in folds:
        for pair in fold["val_ids"]:
            fold_by_pair.setdefault(pair, str(fold["fold_id"]))
    for descriptor in bundle.descriptor_columns:
        values = target_vector(bundle, descriptor, [row["pair_id"] for row in bundle.index_rows])
        finite = values[np.isfinite(values)]
        split_stats: dict[str, float] = {}
        for split in sorted({row.get("split", "") for row in bundle.index_rows}):
            pairs = [row["pair_id"] for row in bundle.index_rows if row.get("split", "") == split]
            arr = target_vector(bundle, descriptor, pairs)
            arr = arr[np.isfinite(arr)]
            split_stats[f"{split}_mean"] = float(np.mean(arr)) if arr.size else float("nan")
            split_stats[f"{split}_std"] = float(np.std(arr)) if arr.size else float("nan")
        rows.append(
            {
                "descriptor": descriptor,
                "unit": "raw_descriptor_units",
                "count": int(finite.size),
                "missing": int(values.size - finite.size),
                "mean": float(np.mean(finite)) if finite.size else float("nan"),
                "std": float(np.std(finite)) if finite.size else float("nan"),
                "min": float(np.min(finite)) if finite.size else float("nan"),
                "max": float(np.max(finite)) if finite.size else float("nan"),
                "near_constant": bool(finite.size < 2 or float(np.std(finite)) <= 1e-8),
                "high_noise_proxy": bool(finite.size >= 2 and float(np.std(finite)) > 0 and np.max(np.abs(finite - np.median(finite))) > 5.0 * float(np.std(finite))),
                **split_stats,
            }
        )
    return rows


def _target_correlation_rows(bundle: Any) -> list[dict[str, Any]]:
    descriptors = bundle.descriptor_columns
    pairs = [row["pair_id"] for row in bundle.index_rows]
    rows: list[dict[str, Any]] = []
    for idx, left in enumerate(descriptors):
        x = target_vector(bundle, left, pairs)
        for right in descriptors[idx + 1 :]:
            y = target_vector(bundle, right, pairs)
            mask = np.isfinite(x) & np.isfinite(y)
            corr = float(np.corrcoef(x[mask], y[mask])[0, 1]) if np.sum(mask) >= 3 and np.std(x[mask]) > 1e-12 and np.std(y[mask]) > 1e-12 else float("nan")
            rows.append({"descriptor_a": left, "descriptor_b": right, "pearson": corr, "count": int(np.sum(mask))})
    return rows


def _plot_target_distribution(path: Any, bundle: Any, folds: Sequence[dict[str, Any]]) -> None:
    descriptors = bundle.descriptor_columns[: min(8, len(bundle.descriptor_columns))]
    if not descriptors:
        resolve_path(path).touch()
        return
    fig, axes = plt.subplots(len(descriptors), 1, figsize=(7, max(3, len(descriptors) * 1.6)), squeeze=False)
    for axis, descriptor in zip(axes[:, 0], descriptors):
        for split in sorted({row.get("split", "") for row in bundle.index_rows}):
            pairs = [row["pair_id"] for row in bundle.index_rows if row.get("split", "") == split]
            vals = target_vector(bundle, descriptor, pairs)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                axis.hist(vals, bins=min(8, max(3, vals.size)), alpha=0.45, label=split)
        axis.set_title(descriptor, fontsize=8)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _summarize_predictions(pred_rows: Sequence[dict[str, Any]], *, bootstrap: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pred_rows:
        grouped[(row["descriptor"], row["model"], row["feature_set"])].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (descriptor, model, feature_set), rows in grouped.items():
        y_true = np.asarray([finite_float(row["true"]) for row in rows], dtype=np.float64)
        y_pred = np.asarray([finite_float(row["prediction"]) for row in rows], dtype=np.float64)
        baseline = np.asarray([finite_float(row["train_mean_prediction"]) for row in rows], dtype=np.float64)
        train_std = np.nanmedian([finite_float(row.get("train_std", "nan")) for row in rows])
        metrics = metric_row(y_true, y_pred, baseline, train_std=float(train_std) if np.isfinite(train_std) else 1.0)
        fold_improvements = []
        fold_beats = 0
        folds = sorted({str(row["fold_key"]) for row in rows})
        for fold in folds:
            fold_rows = [row for row in rows if str(row["fold_key"]) == fold]
            f_true = np.asarray([finite_float(row["true"]) for row in fold_rows], dtype=np.float64)
            f_pred = np.asarray([finite_float(row["prediction"]) for row in fold_rows], dtype=np.float64)
            f_base = np.asarray([finite_float(row["train_mean_prediction"]) for row in fold_rows], dtype=np.float64)
            f_mse = np.mean((f_pred - f_true) ** 2)
            f_base_mse = np.mean((f_base - f_true) ** 2)
            fold_improvements.append(float(f_base_mse - f_mse))
            if f_mse < f_base_mse:
                fold_beats += 1
        paired_values = (baseline - y_true) ** 2 - (y_pred - y_true) ** 2
        ci_low, ci_high = bootstrap_ci(paired_values, bootstrap=bootstrap, seed=seed)
        item = {
            "descriptor": descriptor,
            "model": model,
            "feature_set": feature_set,
            **metrics,
            "fold_count": len(folds),
            "folds_beating_mean": int(fold_beats),
            "beats_mean_fold_rate": float(fold_beats / max(1, len(folds))),
            "improvement_ci_low": ci_low,
            "improvement_ci_high": ci_high,
            "negative_control": int("diagnostic" in feature_set or model in {"shuffled_labels", "random_features"}),
        }
        item["trust_label"] = trust_label(item)
        summary_rows.append(item)
    return sorted(summary_rows, key=lambda row: (row["descriptor"], finite_float(row.get("mse", "inf"), float("inf"))))


def _descriptor_predictability(summary_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    descriptors = sorted({row["descriptor"] for row in summary_rows})
    for descriptor in descriptors:
        candidates = [
            row
            for row in summary_rows
            if row["descriptor"] == descriptor and not int(row.get("negative_control", 0)) and row["model"] != "mean"
        ]
        best = min(candidates, key=lambda row: finite_float(row.get("mse", "inf"), float("inf"))) if candidates else {}
        if best:
            out.append(
                {
                    "descriptor": descriptor,
                    "best_model": best.get("model", ""),
                    "best_feature_set": best.get("feature_set", ""),
                    "mse": best.get("mse", ""),
                    "mean_baseline_mse": best.get("mean_baseline_mse", ""),
                    "r2_vs_train_mean": best.get("r2_vs_train_mean", ""),
                    "paired_improvement_over_mean_mse": best.get("paired_improvement_over_mean_mse", ""),
                    "improvement_ci_low": best.get("improvement_ci_low", ""),
                    "improvement_ci_high": best.get("improvement_ci_high", ""),
                    "folds_beating_mean": best.get("folds_beating_mean", ""),
                    "fold_count": best.get("fold_count", ""),
                    "trust_label": best.get("trust_label", "NOT_SUPPORTED"),
                }
            )
        else:
            out.append({"descriptor": descriptor, "trust_label": "NOT_SUPPORTED"})
    return out


def run_cv(args: argparse.Namespace) -> dict[str, Any]:
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = load_dataset(args.mvp8_root, args.mvp9_root, args.condition_schema)
    models = parse_list(args.models)
    feature_sets = parse_list(args.feature_sets)
    if args.target_set == "key_descriptors":
        key = {"rq", "ra", "robust_range", "psd_slope", "autocorrelation_length_px", "gradient_anisotropy", "mean_abs_gradient", "gradient_std", "island_count"}
        descriptors = [name for name in bundle.descriptor_columns if name in key]
    else:
        descriptors = bundle.descriptor_columns
    folds = make_folds(bundle.index_rows, bundle.fold_rows, fold_mode=args.fold_mode, n_splits=int(args.n_splits), n_repeats=int(args.n_repeats), seed=int(args.seed))
    write_csv(out / "descriptor_target_audit.csv", _target_audit(bundle, folds))
    write_csv(out / "target_correlations.csv", _target_correlation_rows(bundle))
    _plot_target_distribution(out / "target_distribution_by_fold.png", bundle, folds)
    pred_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    for fold in folds:
        train_ids = list(fold["train_ids"])
        val_ids = list(fold["val_ids"])
        fold_key = f"r{fold['repeat']}_f{fold['fold_id']}"
        if not train_ids or not val_ids:
            continue
        for feature_set in feature_sets:
            x_train_raw, feature_names = feature_matrix(bundle, feature_set, train_ids, seed=int(args.seed) + int(fold["repeat"]))
            x_val_raw, _ = feature_matrix(bundle, feature_set, val_ids, seed=int(args.seed) + int(fold["repeat"]))
            x_train, x_val, scale_info = impute_scale_train_val(x_train_raw, x_val_raw)
            missing_fraction = float(np.mean(~np.isfinite(x_train_raw))) if x_train_raw.size else 0.0
            for descriptor in descriptors:
                y_train = target_vector(bundle, descriptor, train_ids)
                y_val = target_vector(bundle, descriptor, val_ids)
                valid_train = np.isfinite(y_train)
                valid_val = np.isfinite(y_val)
                if np.sum(valid_train) < 2 or np.sum(valid_val) < 1:
                    continue
                xt = x_train[valid_train]
                yt = y_train[valid_train]
                xv = x_val[valid_val]
                yv = y_val[valid_val]
                train_mean = float(np.mean(yt))
                train_std = float(np.std(yt)) if float(np.std(yt)) > 1e-8 else 1.0
                baseline = np.full_like(yv, train_mean, dtype=np.float64)
                for model in models:
                    if model == "stable_features_mlp" and not feature_set.startswith("stable36"):
                        continue
                    pred = fit_predict_model(model, xt, yt, xv, seed=int(args.seed))
                    metrics = metric_row(yv, pred, baseline, train_std=train_std)
                    fold_metric_rows.append(
                        {
                            "fold_key": fold_key,
                            "fold_id": fold["fold_id"],
                            "repeat": fold["repeat"],
                            "descriptor": descriptor,
                            "model": model,
                            "feature_set": feature_set,
                            "feature_count": len(feature_names),
                            "missing_fraction_train_features": missing_fraction,
                            **metrics,
                        }
                    )
                    val_pairs = [pair for pair, keep in zip(val_ids, valid_val) if keep]
                    for pair, true, pred_value, base_value in zip(val_pairs, yv, pred, baseline):
                        source = next(row for row in bundle.index_rows if row["pair_id"] == pair)
                        pred_rows.append(
                            {
                                "pair_id": pair,
                                "row_id": source.get("row_id", ""),
                                "sample_id": source.get("sample_id", ""),
                                "group_id": source.get("group_id", ""),
                                "split": source.get("split", ""),
                                "fold_key": fold_key,
                                "fold_id": fold["fold_id"],
                                "repeat": fold["repeat"],
                                "descriptor": descriptor,
                                "model": model,
                                "feature_set": feature_set,
                                "true": float(true),
                                "prediction": float(pred_value),
                                "train_mean_prediction": float(base_value),
                                "train_std": train_std,
                                "feature_count": len(feature_names),
                            }
                        )
    summary_rows = _summarize_predictions(pred_rows, bootstrap=int(args.bootstrap), seed=int(args.seed))
    descriptor_rows = _descriptor_predictability(summary_rows)
    ranking = sorted(
        [row for row in summary_rows if not int(row.get("negative_control", 0))],
        key=lambda row: (row["descriptor"], finite_float(row.get("mse", "inf"), float("inf"))),
    )
    write_csv(out / "cv_metrics_by_fold.csv", fold_metric_rows)
    write_csv(out / "cv_metrics_summary.csv", summary_rows)
    write_csv(out / "cv_predictions_oof.csv", pred_rows)
    write_csv(out / "descriptor_predictability_table.csv", descriptor_rows)
    write_csv(out / "model_ranking_descriptorwise.csv", ranking)
    write_csv(out / "missingness_imputation_table.csv", [{"feature_set": fs, "note": "Feature medians, means, and standard deviations are fitted inside each train fold only."} for fs in feature_sets])
    write_bar_plot(out / "fold_stability_plot.png", descriptor_rows, "descriptor", "folds_beating_mean", title="Folds beating mean")
    write_bar_plot(out / "predicted_vs_true_by_descriptor.png", descriptor_rows, "descriptor", "r2_vs_train_mean", title="Descriptor R2 vs train mean")
    descriptors_sorted = [row["descriptor"] for row in descriptor_rows]
    heat_vals = np.asarray([[finite_float(row.get("mse", "nan")), finite_float(row.get("mean_baseline_mse", "nan"))] for row in descriptor_rows], dtype=np.float64)
    write_heatmap(out / "descriptor_error_heatmap.png", heat_vals, descriptors_sorted, ["model_mse", "mean_mse"], title="Descriptor errors")
    labels = {row["descriptor"]: row.get("trust_label", "NOT_SUPPORTED") for row in descriptor_rows}
    summary = {
        "mvp8_root": display_path(resolve_path(args.mvp8_root)),
        "mvp9_root": display_path(resolve_path(args.mvp9_root)),
        "condition_schema": display_path(resolve_path(args.condition_schema)),
        "fold_mode": args.fold_mode,
        "n_splits": int(args.n_splits),
        "n_repeats": int(args.n_repeats),
        "models": models,
        "feature_sets": feature_sets,
        "target_descriptors": descriptors,
        "sample_count": len(bundle.index_rows),
        "fold_count": len(folds),
        "descriptor_trust_labels": labels,
        "outputs": {
            "cv_metrics_summary": display_path(out / "cv_metrics_summary.csv"),
            "cv_predictions_oof": display_path(out / "cv_predictions_oof.csv"),
            "descriptor_predictability_table": display_path(out / "descriptor_predictability_table.csv"),
        },
    }
    write_json(out / "strict_descriptor_cv_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvp8-root", required=True)
    parser.add_argument("--mvp9-root", required=True)
    parser.add_argument("--condition-schema", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fold-mode", choices=["leave_one_group_out", "repeated_group_kfold", "original_mvp9", "group_kfold"], default="original_mvp9")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--models", default="mean,median,ridge,elasticnet,random_forest,gradient_boosting,stable_features_mlp")
    parser.add_argument("--feature-sets", default="stable36,stable36_plus_consensus_summary,brightness_only_diagnostic,raw240_diagnostic")
    parser.add_argument("--target-set", choices=["all", "key_descriptors"], default="all")
    parser.add_argument("--nested-hparam", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=False)
    parser.add_argument("--save-oof-predictions", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_cv(args)
    print(f"Wrote strict descriptor CV to {display_path(resolve_path(args.out))}")
    print(f"samples={summary['sample_count']} descriptors={len(summary['target_descriptors'])} folds={summary['fold_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
