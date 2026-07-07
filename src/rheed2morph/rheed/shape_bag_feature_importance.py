"""Feature-importance and physical interpretability for MVP-10 shape-bag models."""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Sequence

import matplotlib
import numpy as np

from rheed2morph.rheed.shape_bag_trustworthy_utils import (
    correlation,
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
    spearman,
    target_vector,
    write_bar_plot,
    write_csv,
    write_json,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


GROUP_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("spot_count_density", ("spot_count", "component_count", "density")),
    ("elongated_bar_like", ("bar_like", "elongated", "spot_to_streak", "aspect_ratio", "eccentricity")),
    ("orientation_anisotropy", ("orientation", "anisotropy")),
    ("spacing_pairwise_distance", ("spacing", "pairwise_distance", "neighbor_distance")),
    ("fft_psd", ("fft", "psd", "frequency")),
    ("mask_quality", ("mask_confidence", "snr_score", "artifact")),
    ("brightness_exposure", ("brightness", "contrast", "shadow", "saturation")),
]


def feature_group(name: str) -> str:
    low = name.lower()
    for group, tokens in GROUP_RULES:
        if any(token in low for token in tokens):
            return group
    return "other_stable_geometry"


def _safe_descriptors(cv_root: Any) -> list[str]:
    rows = read_csv(resolve_path(cv_root) / "descriptor_predictability_table.csv")
    descriptors = [row["descriptor"] for row in rows if row.get("trust_label") in {"SUPPORTED", "WEAK"}]
    return descriptors or [row["descriptor"] for row in rows[: min(6, len(rows))]]


def _permutation_importance(bundle: Any, folds: Sequence[dict[str, Any]], descriptors: Sequence[str], *, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for fold in folds:
        train_ids = list(fold["train_ids"])
        val_ids = list(fold["val_ids"])
        if not train_ids or not val_ids:
            continue
        x_train_raw, names = feature_matrix(bundle, "stable36", train_ids, seed=seed)
        x_val_raw, _ = feature_matrix(bundle, "stable36", val_ids, seed=seed)
        x_train, x_val, _scale = impute_scale_train_val(x_train_raw, x_val_raw)
        for descriptor in descriptors:
            y_train = target_vector(bundle, descriptor, train_ids)
            y_val = target_vector(bundle, descriptor, val_ids)
            valid_train = np.isfinite(y_train)
            valid_val = np.isfinite(y_val)
            if np.sum(valid_train) < 2 or np.sum(valid_val) < 1:
                continue
            xt = x_train[valid_train]
            xv = x_val[valid_val]
            yt = y_train[valid_train]
            yv = y_val[valid_val]
            baseline = np.full_like(yv, float(np.mean(yt)))
            pred = fit_predict_model("ridge", xt, yt, xv, seed=seed)
            base_metrics = metric_row(yv, pred, baseline, train_std=float(np.std(yt)) if np.std(yt) > 1e-8 else 1.0)
            base_mse = finite_float(base_metrics.get("mse", "nan"))
            for col_idx, name in enumerate(names):
                xp = xv.copy()
                xp[:, col_idx] = rng.permutation(xp[:, col_idx])
                pp = fit_predict_model("ridge", xt, yt, xp, seed=seed)
                perm_mse = float(np.mean((pp - yv) ** 2))
                rows.append(
                    {
                        "fold_id": fold["fold_id"],
                        "repeat": fold["repeat"],
                        "descriptor": descriptor,
                        "feature": name,
                        "group": feature_group(name),
                        "base_mse": base_mse,
                        "permuted_mse": perm_mse,
                        "importance_delta_mse": perm_mse - base_mse,
                    }
                )
    return rows


def _summaries(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_feature: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    by_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = finite_float(row.get("importance_delta_mse", "nan"))
        if not np.isfinite(value):
            continue
        by_feature[(row["descriptor"], row["feature"], row["group"])].append(value)
        by_group[(row["descriptor"], row["group"])].append(value)
    feature_rows = [
        {
            "descriptor": descriptor,
            "feature": feature,
            "group": group,
            "mean_importance_delta_mse": float(np.mean(values)),
            "std_importance_delta_mse": float(np.std(values)),
            "fold_count": len(values),
        }
        for (descriptor, feature, group), values in by_feature.items()
    ]
    group_rows = [
        {
            "descriptor": descriptor,
            "group": group,
            "mean_importance_delta_mse": float(np.mean(values)),
            "std_importance_delta_mse": float(np.std(values)),
            "feature_fold_count": len(values),
        }
        for (descriptor, group), values in by_group.items()
    ]
    feature_rows.sort(key=lambda row: finite_float(row["mean_importance_delta_mse"], 0.0), reverse=True)
    group_rows.sort(key=lambda row: (row["descriptor"], -finite_float(row["mean_importance_delta_mse"], 0.0)))
    return feature_rows, group_rows


def _bar_scatter_report(bundle: Any, feature_rows: Sequence[dict[str, Any]], out: Any) -> list[dict[str, Any]]:
    out_path = resolve_path(out)
    pairs = [row["pair_id"] for row in bundle.index_rows]
    x, names = feature_matrix(bundle, "stable36", pairs)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    rows: list[dict[str, Any]] = []
    bar_rows = [row for row in feature_rows if row["group"] == "elongated_bar_like"][:12]
    if not bar_rows:
        (out_path / "top_feature_target_scatter.png").touch()
        return rows
    fig, axes = plt.subplots(1, min(4, len(bar_rows)), figsize=(min(4, len(bar_rows)) * 3, 3), squeeze=False)
    for axis, row in zip(axes[0], bar_rows[:4]):
        feature = row["feature"]
        descriptor = row["descriptor"]
        idx = name_to_idx.get(feature)
        if idx is None:
            continue
        y = target_vector(bundle, descriptor, pairs)
        xx = x[:, idx]
        mask = np.isfinite(xx) & np.isfinite(y)
        axis.scatter(xx[mask], y[mask], s=18)
        axis.set_title(f"{feature}\n{descriptor}", fontsize=7)
        axis.grid(alpha=0.2)
        rows.append(
            {
                "feature": feature,
                "descriptor": descriptor,
                "pearson": correlation(xx[mask], y[mask]),
                "spearman": spearman(xx[mask], y[mask]),
                "count": int(np.sum(mask)),
            }
        )
    fig.tight_layout()
    fig.savefig(out_path / "top_feature_target_scatter.png", dpi=150)
    plt.close(fig)
    return rows


def run_feature_importance(args: argparse.Namespace) -> dict[str, Any]:
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv_config = read_json(resolve_path(args.cv_root) / "strict_descriptor_cv_summary.json")
    bundle = load_dataset(args.mvp8_root, args.mvp9_root, cv_config.get("condition_schema"))
    folds = make_folds(bundle.index_rows, bundle.fold_rows, fold_mode=cv_config.get("fold_mode", "original_mvp9"), n_splits=int(cv_config.get("n_splits", 5)), n_repeats=int(cv_config.get("n_repeats", 1)), seed=int(args.seed))
    descriptors = _safe_descriptors(args.cv_root)
    rows = _permutation_importance(bundle, folds, descriptors, seed=int(args.seed))
    feature_summary, grouped = _summaries(rows)
    bar_corr = _bar_scatter_report(bundle, feature_summary, out)
    write_csv(out / "feature_importance_by_fold.csv", rows)
    write_csv(out / "feature_importance_summary.csv", feature_summary)
    write_csv(out / "grouped_feature_importance.csv", grouped)
    write_csv(out / "elongated_bar_feature_correlations.csv", bar_corr)
    write_bar_plot(out / "grouped_feature_importance.png", grouped[:30], "group", "mean_importance_delta_mse", title="Grouped importance")
    bar_supported = [row for row in bar_corr if abs(finite_float(row.get("spearman", "nan"), 0.0)) >= 0.25]
    report = [
        "# Elongated/Bar-Like Feature Report",
        "",
        f"Bar-like feature/target correlations with |Spearman| >= 0.25: `{len(bar_supported)}`",
        "",
        "These correlations are exploratory and must be read together with strict CV and negative-control outcomes.",
    ]
    (out / "elongated_bar_feature_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    physical = [
        "# Physical Interpretability Report",
        "",
        "Permutation importance was computed within validation folds after fitting only on the corresponding training fold.",
        "Grouped importance separates elongated/bar-like, orientation, spacing, FFT/PSD, mask quality, and brightness/exposure features.",
        "",
        f"Top grouped rows are available at `{display_path(out / 'grouped_feature_importance.csv')}`.",
    ]
    (out / "physical_interpretability_report.md").write_text("\n".join(physical) + "\n", encoding="utf-8")
    summary = {
        "descriptor_count": len(descriptors),
        "feature_rows": len(feature_summary),
        "bar_like_correlation_rows": len(bar_corr),
        "bar_like_supported_correlation_rows": len(bar_supported),
        "feature_importance_summary": display_path(out / "feature_importance_summary.csv"),
    }
    write_json(out / "feature_importance_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-root", required=True)
    parser.add_argument("--mvp8-root", required=True)
    parser.add_argument("--mvp9-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_feature_importance(args)
    print(f"Wrote feature importance to {display_path(resolve_path(args.out))}")
    print(f"features={summary['feature_rows']} bar_rows={summary['bar_like_correlation_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
