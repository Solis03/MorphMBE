"""Predict MVP-1 diffusion condition vectors from paired RHEED inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from rheed2morph.generative.common import display_path, read_csv_rows, read_json, resolve_repo_path, resolve_torch_device, write_csv_rows, write_json
from rheed2morph.generative.train_rheed_condition_encoder import (
    RheedConditionDataset,
    _collate,
    descriptor_metrics,
    load_rheed_condition_checkpoint,
    predict_arrays,
    prototype_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict AFM descriptor/prototype conditions from RHEED.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--paired-index", type=Path, required=True)
    parser.add_argument("--condition-schema", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def _dataset_for_split(paired_index: Path, condition_schema: Path, payload: dict[str, Any], split: str) -> RheedConditionDataset:
    config = dict(payload["config"])
    scaler = dict(config["scaler"])
    rows = read_csv_rows(paired_index)
    features = read_csv_rows(condition_schema.parent / "rheed_handcrafted_features.csv")
    feature_by_pair = {row["pair_id"]: row for row in features}
    schema = read_json(condition_schema)
    return RheedConditionDataset(
        rows,
        feature_by_pair,
        schema,
        split,
        use_visual=bool(config.get("use_visual", True)),
        feature_mean=scaler.get("feature_mean", {}),
        feature_std=scaler.get("feature_std", {}),
        metadata_mean=scaler.get("metadata_mean", {}),
        metadata_std=scaler.get("metadata_std", {}),
        limit=None,
    )


def _predicted_raw_descriptor(schema: dict[str, Any], descriptor_name: str, predicted_cond: float) -> float | str:
    means = schema.get("descriptor_train_mean", {})
    stds = schema.get("descriptor_train_std", {})
    if descriptor_name not in means or descriptor_name not in stds:
        return ""
    return float(predicted_cond) * float(stds[descriptor_name]) + float(means[descriptor_name])


def predict_split(
    checkpoint: Path,
    paired_index: Path,
    condition_schema: Path,
    split: str,
    out_dir: Path,
    batch_size: int,
    device_name: str,
) -> dict[str, Any] | None:
    device = resolve_torch_device(device_name)
    model, payload = load_rheed_condition_checkpoint(checkpoint, device_name)
    model.to(device).eval()
    schema = read_json(condition_schema)
    config = dict(payload["config"])
    dataset = _dataset_for_split(paired_index, condition_schema, payload, split)
    if len(dataset) == 0:
        return None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=_collate)
    y_true, y_pred, proto_true, proto_pred, row_ids = predict_arrays(
        model,
        loader,
        device,
        bool(config.get("use_visual", True)),
        bool(config.get("use_handcrafted", True)),
        bool(config.get("use_metadata", False)),
    )
    cond_cols = list(schema["condition_columns"])
    descriptor_cols = list(schema["descriptor_columns"])
    rows: list[dict[str, Any]] = []
    for index, source_row in enumerate(dataset.rows):
        out: dict[str, Any] = {
            "row_id": source_row["row_id"],
            "pair_id": source_row["pair_id"],
            "sample_id": source_row.get("sample_id", ""),
            "group_id": source_row.get("group_id", ""),
            "split": split,
            "rheed_video_path": source_row.get("rheed_video_path", ""),
            "cached_tensor_path": source_row.get("cached_tensor_path", ""),
            "network_input_path": source_row.get("network_input_path", ""),
            "descriptor_height_path": source_row.get("descriptor_height_path", ""),
            "prototype_id": int(proto_pred[index]) if proto_pred[index] >= 0 else "",
            "true_prototype_id": int(proto_true[index]) if proto_true[index] >= 0 else "",
            "predicted_prototype_id": int(proto_pred[index]) if proto_pred[index] >= 0 else "",
        }
        for col_index, cond_col in enumerate(cond_cols):
            descriptor_name = descriptor_cols[col_index]
            true_value = float(y_true[index, col_index])
            pred_value = float(y_pred[index, col_index])
            out[cond_col] = f"{pred_value:.10g}"
            out[f"true_{cond_col}"] = f"{true_value:.10g}"
            out[f"pred_{cond_col}"] = f"{pred_value:.10g}"
            if descriptor_name in source_row:
                out[descriptor_name] = source_row[descriptor_name]
            pred_raw = _predicted_raw_descriptor(schema, descriptor_name, pred_value)
            out[f"pred_{descriptor_name}"] = "" if pred_raw == "" else f"{float(pred_raw):.10g}"
        rows.append(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(out_dir / f"predicted_condition_table_{split}.csv", rows)
    metrics = {
        "split": split,
        "row_count": len(rows),
        **descriptor_metrics(y_true, y_pred),
        **prototype_metrics(proto_true, proto_pred, int(schema.get("prototype_count", 0))),
        "condition_table": display_path(out_dir / f"predicted_condition_table_{split}.csv"),
    }
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
    print(f"Wrote predicted RHEED condition tables to {display_path(resolve_repo_path(args.out))}")
    for split, row in metrics.items():
        print(f"{split}: descriptor_mse={row['descriptor_mse']:.6f} descriptor_mae={row['descriptor_mae']:.6f}")


if __name__ == "__main__":
    main()
