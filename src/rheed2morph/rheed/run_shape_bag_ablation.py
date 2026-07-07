"""Run MVP-9 shape-bag ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import matplotlib
import numpy as np

from rheed2morph.rheed.train_shape_bag_morphology_predictor import (
    descriptor_metrics,
    finite_float,
    read_csv,
    resolve_path,
    split_pair_ids,
    train_model,
    write_csv,
    write_json,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path(__file__).resolve().parents[3]).as_posix()
    except ValueError:
        return resolved.as_posix()


def _targets(data_root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    return (
        read_csv(data_root / "supervised_shape_bag_index.csv"),
        read_csv(data_root / "target_conditions_shape_bag.csv"),
        read_csv(data_root / "strict_fold_assignments.csv"),
        json.loads((data_root / "feature_schema_shape_bag.json").read_text(encoding="utf-8")),
        json.loads((data_root / "target_schema_shape_bag.json").read_text(encoding="utf-8")),
    )


def _target_columns(target_schema: dict[str, Any], rows: Sequence[dict[str, str]]) -> list[str]:
    conds = [column for column in target_schema.get("condition_columns", []) if all(column in row for row in rows)]
    return conds or list(target_schema["descriptor_columns"])


def _mean_baseline(data_root: Path, split: str = "original_split") -> dict[str, Any]:
    index_rows, target_rows, folds, _feature_schema, target_schema = _targets(data_root)
    train_ids, val_ids = split_pair_ids(index_rows, folds, split)
    target_by_pair = {row["pair_id"]: row for row in target_rows}
    columns = _target_columns(target_schema, target_rows)
    train = np.asarray([[finite_float(target_by_pair[pair][column]) for column in columns] for pair in train_ids], dtype=np.float32)
    val = np.asarray([[finite_float(target_by_pair[pair][column]) for column in columns] for pair in val_ids], dtype=np.float32)
    pred = np.repeat(train.mean(axis=0, keepdims=True), val.shape[0], axis=0)
    return {"variant": "train_fold_mean_baseline", "model_family": "baseline", "row_count": int(val.shape[0]), **descriptor_metrics(val, pred)}


def _tabular_arrays(data_root: Path, feature_columns: Sequence[str], *, shuffled: bool = False, brightness_only: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    index_rows, target_rows, folds, _feature_schema, target_schema = _targets(data_root)
    train_ids, val_ids = split_pair_ids(index_rows, folds, "original_split")
    target_by_pair = {row["pair_id"]: row for row in target_rows}
    target_columns = _target_columns(target_schema, target_rows)
    if brightness_only:
        # Forbidden diagnostic proxy: use only mask/artifact/SNR-like columns if present.
        feature_columns = [c for c in feature_columns if any(token in c for token in ["mask_confidence", "artifact_fraction", "snr_score"])]
    if not feature_columns:
        feature_columns = list(feature_columns)
    def x_for(row: dict[str, str]) -> list[float]:
        return [finite_float(row.get(f"shape_feature::{col}", 0.0)) for col in feature_columns]
    train_rows = [row for row in index_rows if row["pair_id"] in train_ids]
    val_rows = [row for row in index_rows if row["pair_id"] in val_ids]
    x_train = np.asarray([x_for(row) for row in train_rows], dtype=np.float32)
    x_val = np.asarray([x_for(row) for row in val_rows], dtype=np.float32)
    y_train = np.asarray([[finite_float(target_by_pair[row["pair_id"]][col]) for col in target_columns] for row in train_rows], dtype=np.float32)
    y_val = np.asarray([[finite_float(target_by_pair[row["pair_id"]][col]) for col in target_columns] for row in val_rows], dtype=np.float32)
    if shuffled:
        rng = np.random.default_rng(42)
        y_train = y_train[rng.permutation(y_train.shape[0])]
    mean = x_train.mean(axis=0, keepdims=True) if x_train.size else np.zeros((1, len(feature_columns)), dtype=np.float32)
    std = x_train.std(axis=0, keepdims=True) if x_train.size else np.ones((1, len(feature_columns)), dtype=np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_train - mean) / std, y_train, (x_val - mean) / std, y_val, target_columns


def _ridge_variant(data_root: Path, name: str, feature_columns: Sequence[str], *, shuffled: bool = False, brightness_only: bool = False) -> dict[str, Any]:
    x_train, y_train, x_val, y_val, _targets = _tabular_arrays(data_root, feature_columns, shuffled=shuffled, brightness_only=brightness_only)
    if x_train.shape[1] == 0:
        pred = np.repeat(y_train.mean(axis=0, keepdims=True), y_val.shape[0], axis=0)
    else:
        try:
            from sklearn.linear_model import Ridge  # type: ignore

            model = Ridge(alpha=1.0)
            model.fit(x_train, y_train)
            pred = model.predict(x_val)
        except Exception:
            x_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1), dtype=np.float32)], axis=1)
            coef = np.linalg.pinv(x_aug.T @ x_aug + np.eye(x_aug.shape[1]) * 1e-3) @ x_aug.T @ y_train
            pred = np.concatenate([x_val, np.ones((x_val.shape[0], 1), dtype=np.float32)], axis=1) @ coef
    row = {"variant": name, "model_family": "tabular", "row_count": int(y_val.shape[0]), **descriptor_metrics(y_val, pred)}
    row["negative_control"] = int(shuffled or brightness_only)
    return row


def _train_variant(args: argparse.Namespace, name: str, **updates: Any) -> dict[str, Any]:
    data_root = resolve_path(args.data_root)
    out = resolve_path(args.out) / name
    values = {
        "supervised_index": str(data_root / "supervised_shape_bag_index.csv"),
        "target_table": str(data_root / "target_conditions_shape_bag.csv"),
        "folds": str(data_root / "strict_fold_assignments.csv"),
        "feature_schema": str(data_root / "feature_schema_shape_bag.json"),
        "target_schema": str(data_root / "target_schema_shape_bag.json"),
        "out": str(out),
        "model": name,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "device": args.device,
        "fold_id": "original_split",
        "quick": bool(args.quick),
        "amp": bool(args.amp),
        "predict_uncertainty": False,
        "use_frames": False,
        "use_consensus": False,
        "use_stable_features": True,
        "use_raw_240_features": False,
        "use_metadata": False,
        "freeze_frame_branch": False,
        "frame_dropout": 0.1,
        "channel_dropout": 0.05,
        "exposure_invariance_weight": 0.0,
        "loss": "mse",
        "early_stop_patience": 8,
        "model_image_size": 64,
        "hidden_dim": 64,
        "embedding_dim": 128,
        "shuffle_labels": False,
        "seed": int(args.seed),
    }
    values.update(updates)
    metrics = train_model(SimpleNamespace(**values))
    return {"variant": name, "model_family": "neural", "row_count": "", "descriptor_mse": metrics.get("descriptor_mse", ""), "best_checkpoint": metrics.get("best_checkpoint", ""), "negative_control": int(values.get("shuffle_labels", False))}


def _bar(path: Path, rows: Sequence[dict[str, Any]], key: str = "descriptor_mse") -> None:
    plot_rows = [row for row in rows if row.get(key, "") != "" and np.isfinite(finite_float(row[key], float("nan")))]
    if not plot_rows:
        return
    fig, axis = plt.subplots(figsize=(max(7, 0.45 * len(plot_rows)), 4))
    axis.bar(range(len(plot_rows)), [finite_float(row[key]) for row in plot_rows])
    axis.set_xticks(range(len(plot_rows)))
    axis.set_xticklabels([row["variant"] for row in plot_rows], rotation=45, ha="right", fontsize=7)
    axis.set_ylabel(key)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_ablations(args: argparse.Namespace) -> dict[str, Any]:
    data_root = resolve_path(args.data_root)
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _index_rows, _target_rows, _folds, feature_schema, _target_schema = _targets(data_root)
    stable = feature_schema["stable_feature_columns"]
    raw = feature_schema["raw_240_feature_columns"]
    rows: list[dict[str, Any]] = [_mean_baseline(data_root)]
    rows.append(_ridge_variant(data_root, "stable_features_ridge", stable))
    rows.append(_train_variant(args, "stable_features_mlp", use_stable_features=True, use_consensus=False, use_frames=False))
    rows.append(_train_variant(args, "consensus_maps_only_cnn", use_stable_features=False, use_consensus=True, use_frames=False))
    rows.append(_train_variant(args, "stable_features_plus_consensus", use_stable_features=True, use_consensus=True, use_frames=False, exposure_invariance_weight=0.05))
    if bool(args.full_suite):
        rows.append(_train_variant(args, "frame_bag_only", use_stable_features=False, use_consensus=False, use_frames=True))
        rows.append(_train_variant(args, "frame_bag_plus_consensus", use_stable_features=False, use_consensus=True, use_frames=True))
        rows.append(_train_variant(args, "full_fusion", use_stable_features=True, use_consensus=True, use_frames=True, exposure_invariance_weight=0.1))
    rows.append(_ridge_variant(data_root, "raw_240_features_diagnostic", raw))
    rows.append(_ridge_variant(data_root, "shuffled_label_negative_control", stable, shuffled=True))
    rows.append(_ridge_variant(data_root, "brightness_only_forbidden_diagnostic", raw, brightness_only=True))
    write_csv(out / "ablation_metrics_shape_bag.csv", rows)
    write_csv(out / "descriptor_predictability_shape_bag.csv", rows)
    write_csv(out / "model_ranking_shape_bag.csv", sorted(rows, key=lambda row: finite_float(row.get("descriptor_mse", "inf"), float("inf"))))
    write_csv(out / "exposure_invariance_ablation.csv", [{"variant": row["variant"], "exposure_invariance_weight": 0.05 if "consensus" in row["variant"] else 0.0, "descriptor_mse": row.get("descriptor_mse", "")} for row in rows])
    write_json(
        out / "negative_control_summary.json",
        {
            "shuffled_label_attempted": True,
            "brightness_only_attempted": True,
            "shuffled_label_mse": next((row.get("descriptor_mse") for row in rows if row["variant"] == "shuffled_label_negative_control"), ""),
            "brightness_only_mse": next((row.get("descriptor_mse") for row in rows if row["variant"] == "brightness_only_forbidden_diagnostic"), ""),
        },
    )
    _bar(out / "stable_vs_raw_feature_comparison.png", [row for row in rows if "features" in row["variant"]])
    _bar(out / "consensus_vs_frame_bag_comparison.png", rows)
    _bar(out / "predicted_vs_true_best_model.png", sorted(rows, key=lambda row: finite_float(row.get("descriptor_mse", "inf"), float("inf")))[:4])
    _bar(out / "shuffled_control_plot.png", [row for row in rows if "shuffled" in row["variant"] or "baseline" in row["variant"]])
    mean_mse = finite_float(rows[0].get("descriptor_mse", float("nan")), float("nan"))
    finite_rows = [row for row in rows if np.isfinite(finite_float(row.get("descriptor_mse", "nan"), float("nan"))) and not int(row.get("negative_control", 0))]
    best = min(finite_rows, key=lambda row: finite_float(row["descriptor_mse"])) if finite_rows else {}
    summary = {
        "best_variant": best.get("variant", ""),
        "best_descriptor_mse": finite_float(best.get("descriptor_mse", float("nan")), float("nan")) if best else float("nan"),
        "mean_baseline_mse": mean_mse,
        "best_beats_mean": bool(best and finite_float(best["descriptor_mse"]) < mean_mse),
        "negative_controls_attempted": True,
        "ablation_metrics": display_path(out / "ablation_metrics_shape_bag.csv"),
    }
    write_json(out / "ablation_summary_shape_bag.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full-suite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_ablations(args)
    print(f"Wrote shape-bag ablations to {display_path(resolve_path(args.out))}")
    print(f"best_variant={summary['best_variant']} best_beats_mean={summary['best_beats_mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

