"""Negative-control battery for MVP-10 shape-bag trust analysis."""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Sequence

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
    read_csv,
    read_json,
    resolve_path,
    target_vector,
    write_bar_plot,
    write_csv,
    write_json,
)


def _load_cv_config(cv_root: Any) -> dict[str, Any]:
    return read_json(resolve_path(cv_root) / "strict_descriptor_cv_summary.json")


def _evaluate_control(
    bundle: Any,
    folds: Sequence[dict[str, Any]],
    descriptors: Sequence[str],
    *,
    control: str,
    feature_set: str,
    seed: int,
    shuffle_train_labels: bool = False,
    shuffle_global_labels: bool = False,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for fold in folds:
        train_ids = list(fold["train_ids"])
        val_ids = list(fold["val_ids"])
        if not train_ids or not val_ids:
            continue
        x_train_raw, names = feature_matrix(bundle, feature_set, train_ids, seed=seed)
        x_val_raw, _ = feature_matrix(bundle, feature_set, val_ids, seed=seed)
        x_train, x_val, _scale = impute_scale_train_val(x_train_raw, x_val_raw)
        for descriptor in descriptors:
            y_train = target_vector(bundle, descriptor, train_ids)
            y_val = target_vector(bundle, descriptor, val_ids)
            valid_train = np.isfinite(y_train)
            valid_val = np.isfinite(y_val)
            if np.sum(valid_train) < 2 or np.sum(valid_val) < 1:
                continue
            yt = y_train[valid_train].copy()
            if shuffle_train_labels:
                yt = yt[rng.permutation(yt.size)]
            if shuffle_global_labels:
                pool = target_vector(bundle, descriptor, [row["pair_id"] for row in bundle.index_rows])
                pool = pool[np.isfinite(pool)]
                yt = rng.choice(pool, size=yt.size, replace=True) if pool.size else yt
            yv = y_val[valid_val]
            xt = x_train[valid_train]
            xv = x_val[valid_val]
            train_mean = float(np.mean(yt))
            train_std = float(np.std(yt)) if float(np.std(yt)) > 1e-8 else 1.0
            baseline = np.full_like(yv, train_mean, dtype=np.float64)
            pred = fit_predict_model("ridge", xt, yt, xv, seed=seed)
            rows.append(
                {
                    "control": control,
                    "descriptor": descriptor,
                    "fold_id": fold["fold_id"],
                    "repeat": fold["repeat"],
                    "feature_set": feature_set,
                    "feature_count": len(names),
                    **metric_row(yv, pred, baseline, train_std=train_std),
                }
            )
    return rows


def _summarize(rows: Sequence[dict[str, Any]], real_rows: Sequence[dict[str, str]], *, seed: int) -> list[dict[str, Any]]:
    best_real_by_desc: dict[str, float] = {}
    for row in real_rows:
        if int(finite_float(row.get("negative_control", "0"), 0.0)):
            continue
        if row.get("model") == "mean":
            continue
        desc = row.get("descriptor", "")
        mse = finite_float(row.get("mse", "nan"))
        if np.isfinite(mse):
            best_real_by_desc[desc] = min(best_real_by_desc.get(desc, float("inf")), mse)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["control"], row["descriptor"])].append(row)
    out: list[dict[str, Any]] = []
    for (control, descriptor), items in grouped.items():
        mse = np.asarray([finite_float(row.get("mse", "nan")) for row in items], dtype=np.float64)
        mean_mse = np.asarray([finite_float(row.get("mean_baseline_mse", "nan")) for row in items], dtype=np.float64)
        improvements = mean_mse - mse
        ci_low, ci_high = bootstrap_ci(improvements, bootstrap=200, seed=seed)
        finite = np.isfinite(mse) & np.isfinite(mean_mse)
        beats = float(np.mean(mse[finite] < mean_mse[finite])) if np.any(finite) else float("nan")
        real_mse = best_real_by_desc.get(descriptor, float("nan"))
        out.append(
            {
                "control": control,
                "descriptor": descriptor,
                "mse": float(np.nanmean(mse)) if mse.size else float("nan"),
                "mean_baseline_mse": float(np.nanmean(mean_mse)) if mean_mse.size else float("nan"),
                "improvement_ci_low": ci_low,
                "improvement_ci_high": ci_high,
                "beats_mean_fold_rate": beats,
                "best_real_mse": real_mse,
                "beats_real": bool(np.isfinite(real_mse) and np.nanmean(mse) < real_mse),
            }
        )
    return sorted(out, key=lambda row: (row["descriptor"], row["control"]))


def run_negative_controls(args: argparse.Namespace) -> dict[str, Any]:
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv_config = _load_cv_config(args.cv_root)
    bundle = load_dataset(args.mvp8_root, args.mvp9_root, cv_config.get("condition_schema"))
    folds = make_folds(bundle.index_rows, bundle.fold_rows, fold_mode=cv_config.get("fold_mode", "original_mvp9"), n_splits=int(cv_config.get("n_splits", 5)), n_repeats=int(cv_config.get("n_repeats", 1)), seed=int(args.seed))
    descriptors = list(cv_config.get("target_descriptors", bundle.descriptor_columns))
    real_summary = read_csv(resolve_path(args.cv_root) / "cv_metrics_summary.csv")
    metric_rows: list[dict[str, Any]] = []
    metric_rows.extend(_evaluate_control(bundle, folds, descriptors, control="shuffled_labels_within_train_folds", feature_set="stable36", seed=int(args.seed), shuffle_train_labels=True))
    metric_rows.extend(_evaluate_control(bundle, folds, descriptors, control="shuffled_labels_global_diagnostic", feature_set="stable36", seed=int(args.seed) + 1, shuffle_global_labels=True))
    metric_rows.extend(_evaluate_control(bundle, folds, descriptors, control="shuffled_shape_bags_across_groups", feature_set="random_gaussian_diagnostic", seed=int(args.seed) + 2))
    metric_rows.extend(_evaluate_control(bundle, folds, descriptors, control="random_gaussian_features", feature_set="random_gaussian_diagnostic", seed=int(args.seed) + 3))
    metric_rows.extend(_evaluate_control(bundle, folds, descriptors, control="brightness_only_diagnostic", feature_set="brightness_only_diagnostic", seed=int(args.seed) + 4))
    metric_rows.extend(_evaluate_control(bundle, folds, descriptors, control="exposure_only_diagnostic", feature_set="exposure_only_diagnostic", seed=int(args.seed) + 5))
    metric_rows.extend(_evaluate_control(bundle, folds, descriptors, control="raw240_feature_diagnostic", feature_set="raw240_diagnostic", seed=int(args.seed) + 6))
    metric_rows.extend(_evaluate_control(bundle, folds, descriptors, control="forbidden_id_path_diagnostic", feature_set="forbidden_id_path_diagnostic", seed=int(args.seed) + 7))
    summary_rows = _summarize(metric_rows, real_summary, seed=int(args.seed))
    write_csv(out / "negative_control_metrics.csv", metric_rows)
    write_csv(out / "permutation_p_values.csv", summary_rows)
    write_bar_plot(out / "real_vs_shuffled_barplot.png", summary_rows, "control", "mse", title="Negative-control MSE")
    write_bar_plot(out / "permutation_null_distribution.png", summary_rows, "descriptor", "beats_mean_fold_rate", title="Control fold beat rate")
    suspicious = [
        row
        for row in summary_rows
        if row["control"] in {"shuffled_labels_within_train_folds", "random_gaussian_features", "brightness_only_diagnostic", "exposure_only_diagnostic"}
        and (finite_float(row.get("beats_mean_fold_rate", "nan"), 0.0) >= 0.5 or bool(row.get("beats_real", False)))
    ]
    payload = {
        "controls_attempted": sorted({row["control"] for row in summary_rows}),
        "descriptor_count": len(descriptors),
        "not_trustworthy_flags": len(suspicious),
        "negative_controls_pass": len(suspicious) == 0,
        "suspicious_rows": suspicious[:20],
    }
    write_json(out / "negative_control_summary.json", payload)
    report = [
        "# Shape-Bag Negative-Control Report",
        "",
        f"Controls attempted: `{payload['controls_attempted']}`",
        f"Suspicious control rows: `{payload['not_trustworthy_flags']}`",
        f"Negative controls pass: `{payload['negative_controls_pass']}`",
        "",
        "Forbidden ID/path diagnostics are leakage demonstrations only and are never eligible for production selection.",
        "Raw 240 feature diagnostics are not production defaults unless separately audited for exposure stability.",
    ]
    (out / "negative_control_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-root", required=True)
    parser.add_argument("--mvp8-root", required=True)
    parser.add_argument("--mvp9-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_negative_controls(args)
    print(f"Wrote negative controls to {display_path(resolve_path(args.out))}")
    print(f"negative_controls_pass={summary['negative_controls_pass']} flags={summary['not_trustworthy_flags']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
