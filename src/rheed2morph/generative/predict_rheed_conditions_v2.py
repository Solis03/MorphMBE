"""Predict MVP-6 RHEED morphology conditions in the v3/calibrated-v2 schema."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from rheed2morph.generative.common import display_path, read_csv_rows, read_json, resolve_repo_path, resolve_torch_device, write_csv_rows, write_json
from rheed2morph.generative.condition_control_v3_utils import condition_to_raw_descriptor
from rheed2morph.generative.train_rheed_morphology_encoder_v2 import (
    RheedMorphologyDataset,
    _collate,
    descriptor_metrics,
    load_rheed_morphology_checkpoint,
    predict_arrays,
    prototype_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict RHEED morphology condition table v2.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--paired-index", type=Path, required=True)
    parser.add_argument("--condition-schema", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def _dataset(paired_index: Path, condition_schema: Path, payload: dict[str, Any], split: str) -> tuple[RheedMorphologyDataset, dict[str, Any]]:
    config = dict(payload["config"])
    rows = read_csv_rows(resolve_repo_path(paired_index))
    schema = read_json(resolve_repo_path(condition_schema))
    local = resolve_repo_path(paired_index).parent / "condition_schema_v3_mvp6.json"
    if local.is_file():
        enriched = read_json(local)
        schema.update({key: value for key, value in enriched.items() if key.startswith("rheed_") or key == "metadata_columns"})
    features = read_csv_rows(resolve_repo_path(paired_index).parent / "rheed_handcrafted_features.csv")
    feature_by_pair = {row["pair_id"]: row for row in features}
    ds = RheedMorphologyDataset(
        rows,
        feature_by_pair,
        schema,
        split,
        dict(config["scaler"]),
        use_visual=bool(config.get("use_visual", True)),
        visual_mode=str(config.get("visual_mode", "all")),
        limit=None,
        seed=42,
    )
    return ds, schema


def predict_split(checkpoint: Path, paired_index: Path, condition_schema: Path, split: str, out_dir: Path, batch_size: int, device_name: str) -> dict[str, Any] | None:
    device = resolve_torch_device(device_name)
    model, payload = load_rheed_morphology_checkpoint(checkpoint, str(device))
    config = dict(payload["config"])
    dataset, schema = _dataset(paired_index, condition_schema, payload, split)
    if len(dataset) == 0:
        return None
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0, collate_fn=_collate)
    arrays = predict_arrays(model.to(device).eval(), loader, device, config)
    y_true = arrays["y_true"]
    y_pred = arrays["y_pred"]
    proto_true = arrays["proto_true"]
    proto_pred = arrays["proto_pred"]
    cond_cols = list(schema["condition_columns"])
    descriptor_cols = list(schema["descriptor_columns"])
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(dataset.rows):
        out: dict[str, Any] = {
            "row_id": source.get("row_id", ""),
            "pair_id": source.get("pair_id", ""),
            "sample_id": source.get("sample_id", ""),
            "group_id": source.get("group_id", ""),
            "split": split,
            "rheed_video_path": source.get("rheed_video_path", ""),
            "cached_tensor_path": source.get("cached_tensor_path", ""),
            "network_input_path": source.get("network_input_path", ""),
            "descriptor_height_path": source.get("descriptor_height_path", ""),
            "prototype_id": int(proto_pred[index]) if proto_pred[index] >= 0 else source.get("prototype_id", ""),
            "true_prototype_id": int(proto_true[index]) if proto_true[index] >= 0 else source.get("prototype_id", ""),
            "predicted_prototype_id": int(proto_pred[index]) if proto_pred[index] >= 0 else "",
        }
        for col_index, cond_col in enumerate(cond_cols):
            name = descriptor_cols[col_index]
            true_cond = float(y_true[index, col_index])
            pred_cond = float(y_pred[index, col_index])
            pred_raw = condition_to_raw_descriptor(name, pred_cond, schema)
            out[cond_col] = f"{pred_cond:.10g}"
            out[f"true_{cond_col}"] = f"{true_cond:.10g}"
            out[f"pred_{cond_col}"] = f"{pred_cond:.10g}"
            out[name] = source.get(name, "")
            out[f"true_{name}"] = source.get(name, "")
            out[f"pred_{name}"] = f"{pred_raw:.10g}"
        rows.append(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / f"predicted_condition_table_{split}.csv"
    write_csv_rows(table_path, rows)
    metrics = {
        "split": split,
        "row_count": len(rows),
        **descriptor_metrics(y_true, y_pred),
        **prototype_metrics(proto_true, proto_pred, int(schema.get("prototype_count", 0))),
        "condition_table": display_path(table_path),
    }
    if arrays["log_variance"].size:
        uncertainty_rows = []
        for row, err, var in zip(rows, np.mean((y_pred - y_true) ** 2, axis=1), np.mean(np.exp(arrays["log_variance"]), axis=1)):
            uncertainty_rows.append({"row_id": row["row_id"], "pair_id": row["pair_id"], "mean_squared_error": float(err), "predicted_variance": float(var)})
        write_csv_rows(out_dir / f"prediction_uncertainty_{split}.csv", uncertainty_rows)
        metrics["uncertainty_table"] = display_path(out_dir / f"prediction_uncertainty_{split}.csv")
    write_json(out_dir / f"prediction_metrics_{split}.json", metrics)
    return metrics


def predict_conditions(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_repo_path(args.out)
    paired_index = resolve_repo_path(args.paired_index)
    condition_schema = resolve_repo_path(args.condition_schema)
    splits = [str(args.split)]
    if args.split == "val":
        rows = read_csv_rows(paired_index)
        if any(row.get("split") == "test" for row in rows):
            splits.append("test")
    metrics: dict[str, Any] = {}
    for split in splits:
        result = predict_split(resolve_repo_path(args.checkpoint), paired_index, condition_schema, split, out_dir, int(args.batch_size), args.device)
        if result is not None:
            metrics[split] = result
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    metrics = predict_conditions(args)
    print(f"Wrote MVP-6 predicted condition tables to {display_path(resolve_repo_path(args.out))}")
    for split, row in metrics.items():
        print(f"{split}: descriptor_mse={row['descriptor_mse']:.6f} descriptor_mae={row['descriptor_mae']:.6f}")


if __name__ == "__main__":
    main()
