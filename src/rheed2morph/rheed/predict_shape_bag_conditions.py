"""Predict AFM condition descriptors from a trained MVP-9 shape-bag model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from rheed2morph.generative.condition_control_v3_utils import condition_to_raw_descriptor
from rheed2morph.rheed.train_shape_bag_morphology_predictor import (
    ShapeBagSupervisedDataset,
    collate,
    descriptor_metrics,
    display_path,
    make_model,
    predict_loader,
    prototype_metrics,
    read_csv,
    resolve_device,
    resolve_path,
    split_pair_ids,
    write_csv,
    write_json,
)


def _sibling(path: str | Path, name: str) -> Path:
    return resolve_path(path).parent / name


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def _source_schema(target_schema: dict[str, Any]) -> dict[str, Any]:
    source = target_schema.get("source_condition_schema", "")
    if source:
        path = resolve_path(source)
        if path.is_file():
            return _load_json(path)
    return {
        "descriptor_columns": target_schema.get("descriptor_columns", []),
        "condition_columns": target_schema.get("condition_columns", []),
        "descriptor_train_mean": {name: 0.0 for name in target_schema.get("descriptor_columns", [])},
        "descriptor_train_std": {name: 1.0 for name in target_schema.get("descriptor_columns", [])},
    }


def _pair_ids_for_split(
    args: argparse.Namespace,
    rows: Sequence[dict[str, str]],
    folds: Sequence[dict[str, str]],
    checkpoint_meta: dict[str, Any],
) -> set[str]:
    if args.out_of_fold:
        return set(checkpoint_meta.get("val_ids", []))
    if args.fold_id:
        _train, val = split_pair_ids(rows, folds, str(args.fold_id))
        return val
    if args.split == "all":
        return {row["pair_id"] for row in rows}
    if args.split == "train":
        return set(checkpoint_meta.get("train_ids", [])) or {row["pair_id"] for row in rows if row.get("split") == "train"}
    if args.split == "val":
        return set(checkpoint_meta.get("val_ids", [])) or {row["pair_id"] for row in rows if row.get("split") in {"val", "test"}}
    return {row["pair_id"] for row in rows if row.get("split") == args.split}


def _raw_from_condition(name: str, value: float, schema: dict[str, Any]) -> float:
    if name in schema.get("descriptor_train_mean", {}):
        return condition_to_raw_descriptor(name, value, schema)
    return float(value)


def _prediction_rows(
    source_rows: Sequence[dict[str, str]],
    target_by_pair: dict[str, dict[str, str]],
    arrays: dict[str, Any],
    target_columns: Sequence[str],
    source_schema: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    row_by_pair = {row["pair_id"]: row for row in source_rows}
    output_rows: list[dict[str, Any]] = []
    uncertainty_rows: list[dict[str, Any]] = []
    proto_logits = arrays.get("proto_logits", np.zeros((0, 0), dtype=np.float32))
    proto_pred = np.argmax(proto_logits, axis=1) if proto_logits.size else np.full(len(arrays["pair_ids"]), -1, dtype=np.int64)
    for index, pair_id in enumerate(arrays["pair_ids"]):
        source = row_by_pair[pair_id]
        target = target_by_pair[pair_id]
        out: dict[str, Any] = {
            "pair_id": pair_id,
            "row_id": source.get("row_id", ""),
            "sample_id": source.get("sample_id", ""),
            "group_id": source.get("group_id", ""),
            "growth_id": source.get("growth_id", source.get("group_id", "")),
            "split": source.get("split", ""),
            "fold_id": "",
            "shape_bag_npz": source.get("shape_bag_npz", ""),
            "network_input_path": source.get("network_input_path", ""),
            "descriptor_height_path": source.get("descriptor_height_path", ""),
            "cached_tensor_path": source.get("cached_tensor_path", ""),
            "true_prototype_id": target.get("prototype_id", ""),
            "predicted_prototype_id": int(proto_pred[index]) if int(proto_pred[index]) >= 0 else "",
            "prototype_id": int(proto_pred[index]) if int(proto_pred[index]) >= 0 else target.get("prototype_id", ""),
        }
        unc: dict[str, Any] = {"pair_id": pair_id, "sample_id": out["sample_id"]}
        for col_index, column in enumerate(target_columns):
            pred = float(arrays["y_pred"][index, col_index])
            true = float(arrays["y_true"][index, col_index])
            if column.startswith("cond_"):
                descriptor = column[len("cond_") :]
                out[f"true_cond_{descriptor}"] = true
                out[f"pred_cond_{descriptor}"] = pred
                out[f"cond_{descriptor}"] = pred
                raw_true = target.get(descriptor, "")
                if raw_true == "":
                    raw_true = _raw_from_condition(descriptor, true, source_schema)
                out[descriptor] = raw_true
                out[f"true_{descriptor}"] = raw_true
                out[f"pred_{descriptor}"] = _raw_from_condition(descriptor, pred, source_schema)
            else:
                descriptor = column
                out[descriptor] = true
                out[f"true_{descriptor}"] = true
                out[f"pred_{descriptor}"] = pred
                if f"cond_{descriptor}" in source_schema.get("condition_columns", []):
                    mean = float(source_schema["descriptor_train_mean"].get(descriptor, 0.0))
                    std = float(source_schema["descriptor_train_std"].get(descriptor, 1.0) or 1.0)
                    out[f"true_cond_{descriptor}"] = (true - mean) / std
                    out[f"pred_cond_{descriptor}"] = (pred - mean) / std
                    out[f"cond_{descriptor}"] = (pred - mean) / std
            if arrays.get("logvar", np.zeros((0, 0))).size:
                logvar = float(arrays["logvar"][index, col_index])
                unc[f"logvar_{descriptor}"] = logvar
                unc[f"variance_{descriptor}"] = float(np.exp(logvar))
        output_rows.append(out)
        if len(unc) > 2:
            uncertainty_rows.append(unc)
    return output_rows, uncertainty_rows


def predict_conditions(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = torch.load(resolve_path(args.checkpoint), map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    if args.model_image_size is not None:
        config["model_image_size"] = int(args.model_image_size)
    for key in ("supervised_index", "target_table", "folds", "feature_schema", "target_schema"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = str(value)
    config.setdefault("batch_size", int(args.batch_size))
    config["batch_size"] = int(args.batch_size)
    config["device"] = args.device
    ns = SimpleNamespace(**config)
    meta = dict(checkpoint["meta"])
    meta["feature_mean"] = np.asarray(meta.get("feature_mean", []), dtype=np.float32)
    meta["feature_std"] = np.asarray(meta.get("feature_std", []), dtype=np.float32)
    index_rows = read_csv(resolve_path(ns.supervised_index))
    target_rows = read_csv(resolve_path(ns.target_table))
    folds = read_csv(resolve_path(ns.folds))
    target_by_pair = {row["pair_id"]: row for row in target_rows}
    pair_ids = _pair_ids_for_split(args, index_rows, folds, meta)
    if not pair_ids:
        pair_ids = set(meta.get("val_ids", [])) or {row["pair_id"] for row in index_rows}
    dataset = ShapeBagSupervisedDataset(
        index_rows,
        target_by_pair,
        pair_ids,
        meta["feature_columns"],
        meta["target_columns"],
        feature_mean=meta["feature_mean"],
        feature_std=meta["feature_std"],
        image_size=int(config.get("model_image_size", 96)),
        load_frames=bool(config.get("use_frames", False)),
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=collate)
    device = resolve_device(args.device)
    model = make_model(ns, meta).to(device)
    model.load_state_dict(checkpoint["model_state"])
    arrays = predict_loader(model, loader, device)
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_targets = np.asarray(
        [
            [float(target_by_pair[pair].get(column, 0.0) or 0.0) for column in meta["target_columns"]]
            for pair in meta.get("train_ids", [])
            if pair in target_by_pair
        ],
        dtype=np.float32,
    )
    baseline = np.repeat(train_targets.mean(axis=0, keepdims=True), arrays["y_true"].shape[0], axis=0) if train_targets.size else np.zeros_like(arrays["y_true"])
    metrics = {
        **descriptor_metrics(arrays["y_true"], arrays["y_pred"], baseline),
        **prototype_metrics(arrays["proto_true"], arrays["proto_logits"], int(meta.get("prototype_count", 0))),
        "row_count": int(arrays["y_true"].shape[0]),
        "checkpoint": display_path(resolve_path(args.checkpoint)),
        "split": "oof" if args.out_of_fold else args.split,
    }
    target_schema = _load_json(resolve_path(ns.target_schema))
    rows, uncertainty = _prediction_rows(index_rows, target_by_pair, arrays, meta["target_columns"], _source_schema(target_schema))
    suffix = "oof" if args.out_of_fold else args.split
    write_csv(out_dir / f"predicted_condition_table_{suffix}.csv", rows)
    write_json(out_dir / f"prediction_metrics_{suffix}.json", metrics)
    if uncertainty:
        write_csv(out_dir / "prediction_uncertainty.csv", uncertainty)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--supervised-index", required=True)
    parser.add_argument("--target-table", default=None)
    parser.add_argument("--folds", default=None)
    parser.add_argument("--feature-schema", default=None)
    parser.add_argument("--target-schema", required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--fold-id", default=None)
    parser.add_argument("--out-of-fold", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=False)
    parser.add_argument("--use-best-ablation", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=False)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-image-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    args.target_table = args.target_table or _sibling(args.supervised_index, "target_conditions_shape_bag.csv")
    args.folds = args.folds or _sibling(args.supervised_index, "strict_fold_assignments.csv")
    args.feature_schema = args.feature_schema or _sibling(args.supervised_index, "feature_schema_shape_bag.json")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = predict_conditions(args)
    suffix = "oof" if args.out_of_fold else args.split
    print(f"Wrote shape-bag predictions to {display_path(resolve_path(args.out) / f'predicted_condition_table_{suffix}.csv')}")
    print(f"descriptor_mse={metrics['descriptor_mse']:.6g} rows={metrics['row_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
