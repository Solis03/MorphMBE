#!/usr/bin/env python3
"""Compare processed-RHEED latent benchmark outputs against the raw baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_LATENT_DIR = (
    REPO_ROOT / "reports" / "one_to_one_plane_corrected_rerun" / "1um" / "rheed_to_afm_latent"
)
DEFAULT_PROCESSED_LATENT_DIR = (
    REPO_ROOT / "reports" / "one_to_one_plane_corrected_rerun_processed" / "1um" / "rheed_to_afm_latent"
)
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "one_to_one_plane_corrected_rerun_processed" / "1um"
RAW_SOURCE_LABEL = "raw video + 8 frames + mean/std"
PROCESSED_SOURCE_LABEL = "processed clean_frames + 64 frames + temporal stats"


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare processed-RHEED and raw-RHEED latent benchmark outputs.")
    parser.add_argument("--raw-latent-dir", type=Path, default=DEFAULT_RAW_LATENT_DIR)
    parser.add_argument("--processed-latent-dir", type=Path, default=DEFAULT_PROCESSED_LATENT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def load_metrics(latent_dir: Path) -> dict[str, Any]:
    metrics_path = latent_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def method_lookup(metrics_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["method"]: row for row in metrics_payload["methods"]}


def summarize_experiment(latent_dir: Path, source_label: str, fallback_label: str) -> dict[str, Any]:
    payload = load_metrics(latent_dir)
    methods = method_lookup(payload)
    selected_name = payload["selected_model_name"]
    learned = methods[selected_name]
    mean_latent = methods["train_mean_latent"]
    return {
        "variant": fallback_label,
        "embedding_source": payload.get("embedding_source_label") or source_label,
        "selected_model_name": selected_name,
        "learned_latent_mse": float(learned["latent_mse"]),
        "learned_latent_cosine_similarity": float(learned["latent_cosine_similarity"]),
        "nearest_neighbor_latent_distance": float(learned["nearest_neighbor_latent_distance"]),
        "nearest_neighbor_cosine_similarity": float(learned["nearest_neighbor_cosine_similarity"]),
        "retrieved_latent_mse": float(learned["retrieved_latent_mse"]),
        "topk_retrieval_hit_rate": float(learned["topk_retrieval_hit_rate"]),
        "mean_latent_mse": float(mean_latent["latent_mse"]),
        "beats_mean_latent": bool(float(learned["latent_mse"]) < float(mean_latent["latent_mse"])),
        "metrics_path": str((latent_dir / "metrics.json").resolve()),
        "nearest_latent_grid_path": str((latent_dir / "nearest_latent_grid.png").resolve()),
        "generated_afm_grid_path": str((latent_dir / "generated_afm_grid.png").resolve()),
    }


def write_barplot(path: Path, raw_row: dict[str, Any], processed_row: dict[str, Any]) -> None:
    metrics = [
        ("learned_latent_mse", "Learned latent MSE", "lower"),
        ("learned_latent_cosine_similarity", "Learned latent cosine", "higher"),
        ("nearest_neighbor_latent_distance", "Nearest latent distance", "lower"),
        ("nearest_neighbor_cosine_similarity", "Nearest latent cosine", "higher"),
        ("retrieved_latent_mse", "Retrieved latent MSE", "lower"),
        ("topk_retrieval_hit_rate", "Top-k hit rate", "higher"),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    labels = ["Raw", "Processed"]
    colors = ["#6f7d8c", "#1f8f6a"]
    for axis, (key, title, direction) in zip(axes, metrics):
        values = [raw_row[key], processed_row[key]]
        bars = axis.bar(labels, values, color=colors)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.set_title(f"{title} ({direction} better)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def comparison_conclusion(raw_row: dict[str, Any], processed_row: dict[str, Any]) -> tuple[str, str]:
    mse_improved = processed_row["learned_latent_mse"] < raw_row["learned_latent_mse"]
    cosine_improved = processed_row["learned_latent_cosine_similarity"] > raw_row["learned_latent_cosine_similarity"]
    if mse_improved and cosine_improved:
        raw_vs_processed = "Processed RHEED improves on the raw baseline on both learned latent MSE and cosine similarity."
    elif mse_improved or cosine_improved:
        raw_vs_processed = "Processed RHEED shows mixed movement versus the raw baseline: at least one headline metric improves, but not both."
    else:
        raw_vs_processed = "Processed RHEED does not beat the raw baseline on the main learned latent metrics."

    if processed_row["beats_mean_latent"]:
        mean_baseline = "Processed RHEED beats the mean-latent dummy baseline on latent MSE."
    else:
        mean_baseline = "Processed RHEED still does not beat the mean-latent dummy baseline on latent MSE."
    return raw_vs_processed, mean_baseline


def write_markdown_report(
    path: Path,
    raw_row: dict[str, Any],
    processed_row: dict[str, Any],
    raw_vs_processed: str,
    mean_baseline: str,
) -> None:
    lines = [
        "# Processed vs Raw RHEED Comparison",
        "",
        "## Metrics",
        "",
        "| Variant | Source | Model | Learned latent MSE | Learned cosine | Nearest latent distance | Nearest latent cosine | Retrieved latent MSE | Top-k hit rate | Beats mean latent? |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in (raw_row, processed_row):
        lines.append(
            f"| {row['variant']} | {row['embedding_source']} | {row['selected_model_name']} | "
            f"{row['learned_latent_mse']:.6f} | {row['learned_latent_cosine_similarity']:.6f} | "
            f"{row['nearest_neighbor_latent_distance']:.6f} | {row['nearest_neighbor_cosine_similarity']:.6f} | "
            f"{row['retrieved_latent_mse']:.6f} | {row['topk_retrieval_hit_rate']:.6f} | "
            f"{'yes' if row['beats_mean_latent'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Visual Checks",
            "",
            f"- Raw nearest-latent grid: `{display_path(Path(raw_row['nearest_latent_grid_path']))}`",
            f"- Processed nearest-latent grid: `{display_path(Path(processed_row['nearest_latent_grid_path']))}`",
            f"- Raw generated-AFM grid: `{display_path(Path(raw_row['generated_afm_grid_path']))}`",
            f"- Processed generated-AFM grid: `{display_path(Path(processed_row['generated_afm_grid_path']))}`",
            "",
            "## Conclusion",
            "",
            f"- {raw_vs_processed}",
            f"- {mean_baseline}",
            "- If processed RHEED still underperforms the dummy baseline, the likely bottleneck remains AFM latent target quality or cross-modal supervision mismatch rather than basic input cleaning alone.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_latent_dir = args.raw_latent_dir if args.raw_latent_dir.is_absolute() else (REPO_ROOT / args.raw_latent_dir)
    processed_latent_dir = (
        args.processed_latent_dir
        if args.processed_latent_dir.is_absolute()
        else (REPO_ROOT / args.processed_latent_dir)
    )
    out_dir = args.out_dir if args.out_dir.is_absolute() else (REPO_ROOT / args.out_dir)

    raw_row = summarize_experiment(raw_latent_dir, RAW_SOURCE_LABEL, "raw")
    processed_row = summarize_experiment(processed_latent_dir, PROCESSED_SOURCE_LABEL, "processed")
    raw_vs_processed, mean_baseline = comparison_conclusion(raw_row, processed_row)

    csv_rows = []
    for row in (raw_row, processed_row):
        csv_rows.append(
            {
                "variant": row["variant"],
                "embedding_source": row["embedding_source"],
                "selected_model_name": row["selected_model_name"],
                "learned_latent_mse": f"{row['learned_latent_mse']:.8f}",
                "learned_latent_cosine_similarity": f"{row['learned_latent_cosine_similarity']:.8f}",
                "nearest_neighbor_latent_distance": f"{row['nearest_neighbor_latent_distance']:.8f}",
                "nearest_neighbor_cosine_similarity": f"{row['nearest_neighbor_cosine_similarity']:.8f}",
                "retrieved_latent_mse": f"{row['retrieved_latent_mse']:.8f}",
                "topk_retrieval_hit_rate": f"{row['topk_retrieval_hit_rate']:.8f}",
                "beats_mean_latent": "yes" if row["beats_mean_latent"] else "no",
                "metrics_path": row["metrics_path"],
                "nearest_latent_grid_path": row["nearest_latent_grid_path"],
                "generated_afm_grid_path": row["generated_afm_grid_path"],
            }
        )
    write_csv(
        out_dir / "processed_vs_raw_metrics.csv",
        csv_rows,
        [
            "variant",
            "embedding_source",
            "selected_model_name",
            "learned_latent_mse",
            "learned_latent_cosine_similarity",
            "nearest_neighbor_latent_distance",
            "nearest_neighbor_cosine_similarity",
            "retrieved_latent_mse",
            "topk_retrieval_hit_rate",
            "beats_mean_latent",
            "metrics_path",
            "nearest_latent_grid_path",
            "generated_afm_grid_path",
        ],
    )
    write_barplot(out_dir / "processed_vs_raw_barplot.png", raw_row, processed_row)
    write_markdown_report(
        out_dir / "processed_vs_raw_comparison.md",
        raw_row,
        processed_row,
        raw_vs_processed,
        mean_baseline,
    )
    print(f"Wrote comparison artifacts to {display_path(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
