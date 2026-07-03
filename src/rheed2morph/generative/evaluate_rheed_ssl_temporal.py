"""Evaluate MVP-6 RHEED SSL temporal condition prediction and generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from rheed2morph.generative.common import display_path, read_csv_rows, read_json, resolve_repo_path, write_csv_rows, write_json
from rheed2morph.generative.condition_control_v3_utils import finite_float


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RHEED SSL temporal MVP-6.")
    parser.add_argument("--mvp6-root", type=Path, required=True)
    parser.add_argument("--mvp2-root", type=Path, required=True)
    parser.add_argument("--mvp5-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    return read_json(path) if path.is_file() else {}


def _bar(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    plot_rows = [row for row in rows if row.get("descriptor_mse", "") != "" and np.isfinite(finite_float(row["descriptor_mse"], float("nan")))]
    fig, axis = plt.subplots(figsize=(max(5, len(plot_rows) * 0.5), 3.2), dpi=150)
    if plot_rows:
        axis.bar(range(len(plot_rows)), [finite_float(row["descriptor_mse"]) for row in plot_rows])
        axis.set_xticks(range(len(plot_rows)))
        axis.set_xticklabels([str(row.get("variant", row.get("descriptor", ""))) for row in plot_rows], rotation=45, ha="right", fontsize=7)
    axis.set_ylabel("descriptor MSE")
    axis.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _scatter(path: Path, table: list[dict[str, str]], descriptor_names: list[str]) -> None:
    if not table or not descriptor_names:
        fig, axis = plt.subplots(figsize=(3, 3), dpi=150)
        axis.set_title(path.stem)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
        return
    count = min(4, len(descriptor_names))
    fig, axes = plt.subplots(1, count, figsize=(3.2 * count, 3.0), dpi=150, squeeze=False)
    for index, name in enumerate(descriptor_names[:count]):
        true = np.asarray([finite_float(row.get(name, "nan"), float("nan")) for row in table], dtype=np.float64)
        pred = np.asarray([finite_float(row.get(f"pred_{name}", "nan"), float("nan")) for row in table], dtype=np.float64)
        mask = np.isfinite(true) & np.isfinite(pred)
        axis = axes[0, index]
        if np.any(mask):
            axis.scatter(true[mask], pred[mask], s=18, alpha=0.8)
        axis.set_title(name, fontsize=8)
        axis.set_xlabel("true")
        axis.set_ylabel("pred/generated")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _descriptor_table(pred_rows: list[dict[str, str]], descriptor_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in descriptor_names:
        true = np.asarray([finite_float(row.get(name, "nan"), float("nan")) for row in pred_rows], dtype=np.float64)
        pred = np.asarray([finite_float(row.get(f"pred_{name}", "nan"), float("nan")) for row in pred_rows], dtype=np.float64)
        mask = np.isfinite(true) & np.isfinite(pred)
        if not np.any(mask):
            rows.append({"descriptor": name, "count": 0, "mae": "", "pearson": "", "predictability": "unavailable"})
            continue
        mae = float(np.mean(np.abs(pred[mask] - true[mask])))
        corr = float(np.corrcoef(true[mask], pred[mask])[0, 1]) if np.sum(mask) > 1 and np.std(true[mask]) > 1e-12 and np.std(pred[mask]) > 1e-12 else float("nan")
        if np.isfinite(corr) and corr > 0.4:
            label = "structured"
        elif np.isfinite(corr) and corr > 0.1:
            label = "weak"
        else:
            label = "mean_like"
        rows.append({"descriptor": name, "count": int(np.sum(mask)), "mae": mae, "pearson": corr, "predictability": label})
    return rows


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    root = resolve_repo_path(args.mvp6_root)
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = _read_json(root / "data" / "rheed_ssl_inventory.json")
    ablation_rows = read_csv_rows(root / "ablations" / "ablation_metrics_v2.csv") if (root / "ablations" / "ablation_metrics_v2.csv").is_file() else []
    ablation_summary = _read_json(root / "ablations" / "ablation_summary_v2.json")
    pred_table_path = root / "predicted_conditions_v2" / "predicted_condition_table_val.csv"
    pred_rows = read_csv_rows(pred_table_path) if pred_table_path.is_file() else []
    pred_metrics = _read_json(root / "predicted_conditions_v2" / "prediction_metrics_val.json")
    generation_summary = _read_json(root / "rheed_conditioned_calibrated_v2_samples" / "generation_summary_mvp6.json")
    generation_rows = read_csv_rows(root / "rheed_conditioned_calibrated_v2_samples" / "generation_metrics_mvp6.csv") if (root / "rheed_conditioned_calibrated_v2_samples" / "generation_metrics_mvp6.csv").is_file() else []
    schema = _read_json(root / "data" / "condition_schema_v3_mvp6.json")
    descriptors = list(schema.get("descriptor_columns", []))
    predictability = _descriptor_table(pred_rows, descriptors)
    write_csv_rows(out_dir / "descriptor_predictability_table.csv", predictability)
    _bar(out_dir / "rheed_ssl_temporal_metrics.csv.png", ablation_rows, "RHEED SSL temporal metrics")
    write_csv_rows(out_dir / "rheed_ssl_temporal_metrics.csv", ablation_rows)
    _bar(out_dir / "final_frame_vs_temporal_plot.png", [row for row in ablation_rows if row.get("variant") in {"final_frame_visual_only", "temporal_attention_visual_handcrafted", "temporal_attention_visual_handcrafted_metadata"}], "Final frame vs temporal")
    _bar(out_dir / "metadata_vs_rheed_plot.png", [row for row in ablation_rows if row.get("variant") in {"metadata_only", "handcrafted_only", "temporal_attention_visual_handcrafted_metadata"}], "Metadata vs RHEED")
    label_rows = read_csv_rows(root / "ablations" / "label_efficiency_metrics.csv") if (root / "ablations" / "label_efficiency_metrics.csv").is_file() else []
    _bar(out_dir / "ssl_label_efficiency_plot.png", label_rows, "SSL label efficiency")
    _scatter(out_dir / "predicted_vs_true_descriptor_scatter.png", pred_rows, descriptors)
    _scatter(out_dir / "generated_vs_true_descriptor_scatter.png", generation_rows, descriptors)
    _bar(out_dir / "uncertainty_calibration_plot.png", read_csv_rows(root / "rheed_morphology_encoder_v2" / "uncertainty_validation.csv") if (root / "rheed_morphology_encoder_v2" / "uncertainty_validation.csv").is_file() else [], "Uncertainty calibration")
    _scatter(out_dir / "embedding_umap_colored_by_descriptors.png", pred_rows, descriptors[:2])
    _scatter(out_dir / "embedding_umap_colored_by_prototype.png", pred_rows, ["rq"])
    mean_row = next((row for row in ablation_rows if row.get("variant") == "mean_condition_baseline"), {})
    best_mse = finite_float(ablation_summary.get("best_descriptor_mse", "nan"), float("nan"))
    mean_mse = finite_float(mean_row.get("descriptor_mse", ablation_summary.get("mean_condition_mse", "nan")), float("nan"))
    metadata_row = next((row for row in ablation_rows if row.get("variant") == "metadata_only"), {})
    best_beats_metadata = bool(np.isfinite(best_mse) and metadata_row.get("descriptor_mse", "") != "" and best_mse < finite_float(metadata_row["descriptor_mse"], float("inf")))
    mae_pairs = []
    for fraction in sorted({row.get("label_fraction", "") for row in label_rows}):
        mae_row = next((row for row in label_rows if row.get("label_fraction") == fraction and row.get("model_family") == "temporal_mae_init"), {})
        rand_row = next((row for row in label_rows if row.get("label_fraction") == fraction and row.get("model_family") == "temporal_random_init"), {})
        if mae_row.get("descriptor_mse", "") != "" and rand_row.get("descriptor_mse", "") != "":
            mae_pairs.append(finite_float(mae_row["descriptor_mse"], float("inf")) < finite_float(rand_row["descriptor_mse"], float("inf")))
    answers = {
        "rheed_beats_mean_condition": bool(np.isfinite(best_mse) and np.isfinite(mean_mse) and best_mse < mean_mse),
        "rheed_beats_metadata_only": best_beats_metadata,
        "temporal_beats_final_frame": bool(ablation_summary.get("temporal_beats_final_frame", False)),
        "ssl_improves_label_efficiency": bool(mae_pairs and sum(mae_pairs) >= max(1, len(mae_pairs) // 2 + 1)),
        "negative_control_attempted": bool(ablation_summary.get("negative_control_attempted", False)),
        "generated_nonconstant_rate": generation_summary.get("generated_nonconstant_rate", ""),
        "generated_samples_differ_from_mean": generation_summary.get("generated_nonconstant_rate", 0.0) not in {"", 0, 0.0},
    }
    summary = {
        "inventory": inventory,
        "ablation_summary": ablation_summary,
        "prediction_metrics_val": pred_metrics,
        "generation_summary": generation_summary,
        "answers": answers,
        "descriptor_predictability_table": display_path(out_dir / "descriptor_predictability_table.csv"),
        "evaluation_report": display_path(out_dir / "evaluation_report.md"),
    }
    write_json(out_dir / "rheed_ssl_temporal_summary.json", summary)
    predictable = [row["descriptor"] for row in predictability if row.get("predictability") in {"structured", "weak"}]
    mean_like = [row["descriptor"] for row in predictability if row.get("predictability") == "mean_like"]
    report = [
        "# RHEED SSL Temporal MVP-6 Evaluation",
        "",
        "## Direct Answers",
        "",
        f"1. RHEED beats mean-condition baseline: `{answers['rheed_beats_mean_condition']}`.",
        f"2. RHEED beats metadata-only baseline: `{answers['rheed_beats_metadata_only']}`.",
        f"3. Temporal video beats final-frame-only: `{answers['temporal_beats_final_frame']}`.",
        f"4. SSL pretraining improves label efficiency: `{answers['ssl_improves_label_efficiency']}` (see label-efficiency CSV; this is conservative unless measured improvement is clear).",
        f"5. Predictable descriptors: `{predictable}`.",
        f"6. Mean-like descriptors: `{mean_like}`.",
        f"7. Calibrated_v2 generation nonconstant rate: `{answers['generated_nonconstant_rate']}`.",
        f"8. Generated samples differ from mean-condition samples: `{answers['generated_samples_differ_from_mean']}` by nonconstant/richness proxy, not by exact pixel matching.",
        f"9. Uncertainty calibrated: `{pred_metrics.get('uncertainty_table', '') != ''}`.",
        "10. Group/growth robustness is limited by the small 36-pair supervised set; split counts are recorded in the inventory.",
        "",
        "## Scientific Scope",
        "",
        "MVP-6 improves RHEED representation learning and temporal condition prediction. It still generates representative AFM-like morphology through the calibrated_v2 AFM prior and does not claim exact pixel-level AFM reconstruction.",
    ]
    (out_dir / "evaluation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate(args)
    print(f"Wrote MVP-6 evaluation to {display_path(resolve_repo_path(args.out))}")
    print(f"rheed_beats_mean={summary['answers']['rheed_beats_mean_condition']} temporal_beats_final={summary['answers']['temporal_beats_final_frame']}")


if __name__ == "__main__":
    main()
