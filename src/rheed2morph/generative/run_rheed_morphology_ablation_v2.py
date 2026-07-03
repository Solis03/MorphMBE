"""Run MVP-6 RHEED morphology encoder ablations."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import numpy as np

from rheed2morph.generative.common import display_path, resolve_repo_path, write_csv_rows, write_json
from rheed2morph.generative.train_rheed_morphology_encoder_v2 import _make_loaders, descriptor_metrics, train_encoder


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RHEED morphology encoder v2 ablations.")
    parser.add_argument("--paired-index", type=Path, required=True)
    parser.add_argument("--condition-schema", type=Path, required=True)
    parser.add_argument("--mvp5-root", type=Path, required=True)
    parser.add_argument("--frame-mae-checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--label-efficiency-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full-suite", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _namespace(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        {
            "frame_encoder": "small_cnn",
            "freeze_frame_encoder": False,
            "temporal_pooling": "attention",
            "use_visual": True,
            "use_handcrafted": True,
            "use_metadata": True,
            "predict_uncertainty": False,
            "target_schema": "v3",
            "loss": "mse",
            "label_fraction": 1.0,
            "shuffle_labels": False,
            "shuffle_videos": False,
            "num_workers": 0,
        }
    )
    values.update(updates)
    return SimpleNamespace(**values)


def _mean_baseline(args: argparse.Namespace) -> dict[str, Any]:
    base = _namespace(args, use_visual=False, use_handcrafted=False, use_metadata=False)
    loaders, _schema, _scaler = _make_loaders(base)
    train_targets = []
    val_targets = []
    for batch in loaders["train"]:
        train_targets.append(batch["target"].numpy())
    for batch in loaders["val"]:
        val_targets.append(batch["target"].numpy())
    train = np.concatenate(train_targets, axis=0) if train_targets else np.zeros((0, 0), dtype=np.float32)
    val = np.concatenate(val_targets, axis=0) if val_targets else np.zeros((0, 0), dtype=np.float32)
    mean = train.mean(axis=0, keepdims=True) if train.size else np.zeros_like(val[:1])
    pred = np.repeat(mean, val.shape[0], axis=0) if val.size else val
    return {"variant": "mean_condition_baseline", "split": "val", "row_count": int(val.shape[0]), **descriptor_metrics(val, pred), "prototype_accuracy": "", "prototype_macro_f1": "", "beats_mean_condition_mse": ""}


def _run_variant(args: argparse.Namespace, name: str, visual_mode: str = "all", **updates: Any) -> dict[str, Any]:
    out = resolve_repo_path(args.out) / name
    variant_args = _namespace(args, out=out, **updates)
    metrics = train_encoder(variant_args, variant_name=name, visual_mode=visual_mode)
    return {"variant": name, "split": "val", "row_count": metrics.get("row_count", ""), **{key: metrics.get(key, "") for key in ("descriptor_mse", "descriptor_mae", "descriptor_rmse", "descriptor_r2", "descriptor_spearman", "prototype_accuracy", "prototype_macro_f1", "beats_mean_condition_mse")}, "checkpoint": metrics.get("best_checkpoint", ""), "mae_checkpoint_loaded": metrics.get("mae_checkpoint_loaded", False)}


def _bar(path: Path, rows: list[dict[str, Any]], title: str, key: str = "descriptor_mse") -> None:
    plot_rows = [row for row in rows if row.get(key, "") != "" and np.isfinite(float(row[key]))]
    if not plot_rows:
        return
    fig, axis = plt.subplots(figsize=(max(6, 0.5 * len(plot_rows)), 3.5), dpi=150)
    axis.bar(range(len(plot_rows)), [float(row[key]) for row in plot_rows])
    axis.set_xticks(range(len(plot_rows)))
    axis.set_xticklabels([str(row["variant"]) for row in plot_rows], rotation=45, ha="right", fontsize=7)
    axis.set_ylabel(key)
    axis.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _label_efficiency(args: argparse.Namespace, fractions: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        for name, updates in [
            ("temporal_random_init", {"use_visual": True, "use_handcrafted": False, "use_metadata": False, "frame_mae_checkpoint": None}),
            ("temporal_mae_init", {"use_visual": True, "use_handcrafted": False, "use_metadata": False, "frame_mae_checkpoint": args.frame_mae_checkpoint}),
            ("handcrafted_only", {"use_visual": False, "use_handcrafted": True, "use_metadata": False, "frame_mae_checkpoint": None}),
            ("metadata_only", {"use_visual": False, "use_handcrafted": False, "use_metadata": True, "frame_mae_checkpoint": None}),
        ]:
            variant = f"label_{int(fraction * 100)}_{name}"
            try:
                row = _run_variant(
                    args,
                    variant,
                    epochs=min(int(args.label_efficiency_epochs), int(args.epochs)),
                    label_fraction=fraction,
                    temporal_pooling="attention",
                    **updates,
                )
                row["label_fraction"] = fraction
                row["model_family"] = name
                rows.append(row)
            except Exception as exc:
                rows.append({"variant": variant, "label_fraction": fraction, "model_family": name, "error": str(exc)})
    return rows


def run_ablations(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = [_mean_baseline(args)]
    variants: list[tuple[str, str, dict[str, Any]]] = [
        ("metadata_only", "all", {"use_visual": False, "use_handcrafted": False, "use_metadata": True}),
        ("handcrafted_only", "all", {"use_visual": False, "use_handcrafted": True, "use_metadata": False}),
        ("final_frame_visual_only", "final", {"use_visual": True, "use_handcrafted": False, "use_metadata": False, "temporal_pooling": "final"}),
        ("temporal_attention_visual_handcrafted", "all", {"use_visual": True, "use_handcrafted": True, "use_metadata": False, "temporal_pooling": "attention", "frame_mae_checkpoint": args.frame_mae_checkpoint}),
        ("temporal_attention_visual_handcrafted_metadata", "all", {"use_visual": True, "use_handcrafted": True, "use_metadata": True, "temporal_pooling": "attention", "frame_mae_checkpoint": args.frame_mae_checkpoint}),
        ("shuffled_label_negative_control", "all", {"use_visual": True, "use_handcrafted": True, "use_metadata": False, "temporal_pooling": "attention", "shuffle_labels": True}),
    ]
    if args.full_suite:
        variants.extend(
            [
                ("final_frames_average_visual_only", "final_average", {"use_visual": True, "use_handcrafted": False, "use_metadata": False, "temporal_pooling": "mean"}),
                ("temporal_visual_mean_pooling", "all", {"use_visual": True, "use_handcrafted": False, "use_metadata": False, "temporal_pooling": "mean"}),
                ("temporal_visual_gru", "all", {"use_visual": True, "use_handcrafted": False, "use_metadata": False, "temporal_pooling": "gru"}),
                ("mae_pretrained_frozen_encoder", "all", {"use_visual": True, "use_handcrafted": False, "use_metadata": False, "temporal_pooling": "attention", "frame_mae_checkpoint": args.frame_mae_checkpoint, "freeze_frame_encoder": True}),
                ("random_initialized_encoder", "all", {"use_visual": True, "use_handcrafted": False, "use_metadata": False, "temporal_pooling": "attention", "frame_mae_checkpoint": None}),
                ("shuffled_video_negative_control", "all", {"use_visual": True, "use_handcrafted": True, "use_metadata": False, "temporal_pooling": "attention", "shuffle_videos": True}),
            ]
        )
    for name, visual_mode, updates in variants:
        try:
            rows.append(_run_variant(args, name, visual_mode=visual_mode, **updates))
        except Exception as exc:
            rows.append({"variant": name, "split": "val", "error": str(exc)})
    write_csv_rows(out_dir / "ablation_metrics_v2.csv", rows)
    _bar(out_dir / "temporal_vs_final_frame_barplot.png", [row for row in rows if row.get("variant") in {"final_frame_visual_only", "temporal_attention_visual_handcrafted", "temporal_attention_visual_handcrafted_metadata"}], "Temporal vs final-frame")
    _bar(out_dir / "ssl_vs_random_init_barplot.png", [row for row in rows if "mae" in str(row.get("variant", "")) or "random" in str(row.get("variant", "")) or row.get("variant") == "temporal_attention_visual_handcrafted"], "SSL init vs random")
    _bar(out_dir / "descriptor_scatter_baselines.png", rows, "Ablation descriptor MSE")
    _bar(out_dir / "descriptor_scatter_best_model.png", sorted([r for r in rows if r.get("descriptor_mse", "") != ""], key=lambda r: float(r["descriptor_mse"]))[:4], "Best descriptor MSE")
    _bar(out_dir / "uncertainty_calibration.png", rows, "Uncertainty placeholder")
    label_rows = _label_efficiency(args, [0.25, 0.5, 0.75, 1.0])
    write_csv_rows(out_dir / "label_efficiency_metrics.csv", label_rows)
    _bar(out_dir / "label_efficiency_plot.png", [row for row in label_rows if row.get("descriptor_mse", "") != ""], "Label efficiency")
    finite_rows = [row for row in rows if row.get("descriptor_mse", "") != "" and np.isfinite(float(row["descriptor_mse"]))]
    mean_mse = next((float(row["descriptor_mse"]) for row in rows if row.get("variant") == "mean_condition_baseline" and row.get("descriptor_mse", "") != ""), float("nan"))
    best = min(finite_rows, key=lambda row: float(row["descriptor_mse"])) if finite_rows else {}
    final = next((row for row in rows if row.get("variant") == "final_frame_visual_only"), {})
    temporal = next((row for row in rows if row.get("variant") == "temporal_attention_visual_handcrafted_metadata"), next((row for row in rows if row.get("variant") == "temporal_attention_visual_handcrafted"), {}))
    summary = {
        "ablation_metrics": display_path(out_dir / "ablation_metrics_v2.csv"),
        "best_variant": best.get("variant", ""),
        "best_descriptor_mse": float(best.get("descriptor_mse", "nan")) if best else float("nan"),
        "mean_condition_mse": mean_mse,
        "temporal_beats_final_frame": bool(temporal.get("descriptor_mse", "") != "" and final.get("descriptor_mse", "") != "" and float(temporal["descriptor_mse"]) < float(final["descriptor_mse"])),
        "best_beats_mean_condition": bool(best and np.isfinite(mean_mse) and float(best["descriptor_mse"]) < mean_mse),
        "negative_control_attempted": any("negative_control" in str(row.get("variant", "")) for row in rows),
        "label_efficiency_metrics": display_path(out_dir / "label_efficiency_metrics.csv"),
    }
    write_json(out_dir / "ablation_summary_v2.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run_ablations(args)
    print(f"Wrote RHEED morphology ablations to {display_path(resolve_repo_path(args.out))}")
    print(f"best_variant={summary['best_variant']} temporal_beats_final={summary['temporal_beats_final_frame']}")


if __name__ == "__main__":
    main()
