"""Evaluate MVP-2 RHEED-conditioned condition prediction and generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from rheed2morph.generative.common import load_height_array, read_csv_rows, read_json, resolve_repo_path, write_csv_rows, write_json
from rheed2morph.generative.train_rheed_condition_encoder import descriptor_metrics, prototype_metrics
from rheed2morph.generative.visualization import write_panel_grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RHEED-conditioned generation artifacts.")
    parser.add_argument("--predicted-condition-table", type=Path, required=True)
    parser.add_argument("--paired-index", type=Path, required=True)
    parser.add_argument("--condition-schema", type=Path, required=True)
    parser.add_argument("--generation-metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    return parser


def _condition_arrays(rows: list[dict[str, str]], columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray([[float(row.get(f"true_{col}", row.get(col, "nan"))) for col in columns] for row in rows], dtype=np.float32)
    pred = np.asarray([[float(row[col]) for col in columns] for row in rows], dtype=np.float32)
    return true, pred


def _mean_condition_baseline(paired_rows: list[dict[str, str]], eval_rows: list[dict[str, str]], columns: list[str]) -> dict[str, float]:
    train = [row for row in paired_rows if row.get("split") == "train"] or paired_rows
    train_matrix = np.asarray([[float(row[col]) for col in columns] for row in train], dtype=np.float32)
    eval_true = np.asarray([[float(row.get(f"true_{col}", row.get(col, "nan"))) for col in columns] for row in eval_rows], dtype=np.float32)
    mean = np.mean(train_matrix, axis=0, keepdims=True)
    return descriptor_metrics(eval_true, np.repeat(mean, eval_true.shape[0], axis=0))


def _failure_grid(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    if not rows:
        return
    scored = []
    for row in rows:
        errors = []
        for col in columns:
            try:
                errors.append(abs(float(row[col]) - float(row.get(f"true_{col}", row[col]))))
            except ValueError:
                pass
        scored.append((float(np.mean(errors)) if errors else 0.0, row))
    selected = [row for _score, row in sorted(scored, key=lambda item: item[0], reverse=True)[:6]]
    panels: list[list[np.ndarray]] = []
    titles: list[str] = []
    for row in selected:
        rheed = np.load(resolve_repo_path(Path(row["cached_tensor_path"])))["frames"][-1, 0]
        afm = load_height_array(resolve_repo_path(Path(row["network_input_path"])))
        panels.append([rheed, afm])
        titles.append(str(row.get("sample_id", row.get("row_id", ""))))
    write_panel_grid(path, panels, ["RHEED final frame", "true AFM"], titles)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = read_json(resolve_repo_path(args.condition_schema))
    columns = list(schema["condition_columns"])
    pred_rows = [row for row in read_csv_rows(resolve_repo_path(args.predicted_condition_table)) if row.get("split") == args.split]
    if not pred_rows:
        pred_rows = read_csv_rows(resolve_repo_path(args.predicted_condition_table))
    paired_rows = read_csv_rows(resolve_repo_path(args.paired_index))
    true, pred = _condition_arrays(pred_rows, columns)
    proto_true = np.asarray([int(float(row.get("true_prototype_id", "-1") or -1)) for row in pred_rows], dtype=np.int64)
    proto_pred = np.asarray([int(float(row.get("predicted_prototype_id", row.get("prototype_id", "-1")) or -1)) for row in pred_rows], dtype=np.int64)
    condition = descriptor_metrics(true, pred)
    proto = prototype_metrics(proto_true, proto_pred, int(schema.get("prototype_count", 0)))
    mean_baseline = _mean_condition_baseline(paired_rows, pred_rows, columns)
    generation_rows = read_csv_rows(resolve_repo_path(args.generation_metrics))
    generated_stds = [float(row["generated_std"]) for row in generation_rows]
    predicted_rows = [row for row in generation_rows if row.get("mode") == "predicted"]
    oracle_rows = [row for row in generation_rows if row.get("mode") == "oracle"]
    mean_rows = [row for row in generation_rows if row.get("mode") == "mean"]
    summary = {
        "split": args.split,
        "condition_prediction": condition,
        "prototype_prediction": proto,
        "mean_condition_baseline": mean_baseline,
        "beats_mean_condition_mse": bool(np.isfinite(condition["descriptor_mse"]) and condition["descriptor_mse"] < mean_baseline["descriptor_mse"]),
        "generation_realism": {
            "generated_count": len(generation_rows),
            "generated_std_mean": float(np.mean(generated_stds)) if generated_stds else 0.0,
            "generated_std_min": float(np.min(generated_stds)) if generated_stds else 0.0,
            "nonconstant_rate": float(np.mean(np.asarray(generated_stds) > 1e-4)) if generated_stds else 0.0,
        },
        "generation_modes": {
            "predicted_count": len(predicted_rows),
            "oracle_count": len(oracle_rows),
            "mean_condition_count": len(mean_rows),
            "predicted_generated_std_mean": float(np.mean([float(row["generated_std"]) for row in predicted_rows])) if predicted_rows else 0.0,
            "oracle_generated_std_mean": float(np.mean([float(row["generated_std"]) for row in oracle_rows])) if oracle_rows else 0.0,
            "mean_condition_generated_std_mean": float(np.mean([float(row["generated_std"]) for row in mean_rows])) if mean_rows else 0.0,
        },
    }
    metric_rows = [{"metric": key, "value": value} for key, value in condition.items()]
    metric_rows.extend({"metric": f"prototype_{key}", "value": value} for key, value in proto.items())
    metric_rows.extend({"metric": f"mean_baseline_{key}", "value": value} for key, value in mean_baseline.items())
    write_csv_rows(out_dir / "generation_metrics.csv", generation_rows)
    write_csv_rows(out_dir / "evaluation_metrics.csv", metric_rows)
    write_json(out_dir / "generation_summary.json", summary)
    _failure_grid(out_dir / "failure_cases_grid.png", pred_rows, columns)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate(args)
    print(f"condition_descriptor_mse={summary['condition_prediction']['descriptor_mse']:.6f}")
    print(f"generated_nonconstant_rate={summary['generation_realism']['nonconstant_rate']:.3f}")


if __name__ == "__main__":
    main()
