"""Export policy-adjusted OOF production predictions for MVP-10."""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from rheed2morph.rheed.shape_bag_trustworthy_utils import (
    descriptor_to_condition,
    display_path,
    finite_float,
    read_csv,
    read_json,
    resolve_path,
    write_csv,
    write_json,
)


def _policy_by_descriptor(path: Any) -> dict[str, dict[str, Any]]:
    payload = read_json(resolve_path(path))
    rows = payload.get("policy", [])
    return {row["descriptor"]: row for row in rows}


def export_predictions(args: argparse.Namespace) -> dict[str, Any]:
    out = resolve_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cv_root = resolve_path(args.cv_root)
    cv_config = read_json(cv_root / "strict_descriptor_cv_summary.json")
    selected = read_json(resolve_path(args.production_selection)).get("selected_descriptors", {})
    policy = _policy_by_descriptor(args.descriptor_policy)
    preds = read_csv(cv_root / "cv_predictions_oof.csv")
    mvp9_root = resolve_path(cv_config["mvp9_root"])
    condition_schema = read_json(resolve_path(cv_config["condition_schema"]))
    index_rows = read_csv(mvp9_root / "data" / "supervised_shape_bag_index.csv")
    source_by_pair = {row["pair_id"]: row for row in index_rows}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in preds:
        grouped[row["pair_id"]].append(row)
    descriptors = list(condition_schema.get("descriptor_columns", []))
    output_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for pair_id, rows in grouped.items():
        source = source_by_pair.get(pair_id, {})
        out_row: dict[str, Any] = {
            "pair_id": pair_id,
            "row_id": source.get("row_id", rows[0].get("row_id", "")),
            "sample_id": source.get("sample_id", rows[0].get("sample_id", "")),
            "group_id": source.get("group_id", rows[0].get("group_id", "")),
            "growth_id": source.get("growth_id", source.get("group_id", "")),
            "split": source.get("split", rows[0].get("split", "")),
            "shape_bag_npz": source.get("shape_bag_npz", ""),
            "network_input_path": source.get("network_input_path", ""),
            "descriptor_height_path": source.get("descriptor_height_path", ""),
            "cached_tensor_path": source.get("cached_tensor_path", ""),
            "prototype_id": source.get("prototype_id", ""),
        }
        for descriptor in descriptors:
            descriptor_rows = [row for row in rows if row.get("descriptor") == descriptor]
            if not descriptor_rows:
                continue
            true = finite_float(descriptor_rows[0].get("true", "nan"))
            selected_cfg = selected.get(descriptor)
            selected_row = None
            if selected_cfg:
                for row in descriptor_rows:
                    if row.get("model") == selected_cfg.get("model") and row.get("feature_set") == selected_cfg.get("feature_set"):
                        selected_row = row
                        break
            if selected_row is None:
                selected_row = descriptor_rows[0]
            raw_pred = finite_float(selected_row.get("prediction", "nan"))
            train_mean = finite_float(selected_row.get("train_mean_prediction", "nan"))
            action = policy.get(descriptor, {}).get("production_action", "fill_train_mean")
            if action == "use_rheed_prediction" and np.isfinite(raw_pred):
                adjusted = raw_pred
                source_flag = "predicted_by_rheed"
            else:
                adjusted = train_mean
                source_flag = "filled_by_train_mean"
            out_row[descriptor] = true
            out_row[f"true_{descriptor}"] = true
            out_row[f"raw_pred_{descriptor}"] = raw_pred
            out_row[f"pred_{descriptor}"] = adjusted
            out_row[f"policy_adjusted_{descriptor}"] = adjusted
            out_row[f"cond_{descriptor}"] = descriptor_to_condition(descriptor, adjusted, condition_schema)
            out_row[f"pred_cond_{descriptor}"] = out_row[f"cond_{descriptor}"]
            out_row[f"true_cond_{descriptor}"] = descriptor_to_condition(descriptor, true, condition_schema)
            out_row[f"policy_{descriptor}"] = source_flag
            policy_rows.append(
                {
                    "pair_id": pair_id,
                    "sample_id": out_row["sample_id"],
                    "descriptor": descriptor,
                    "action": action,
                    "source_flag": source_flag,
                    "raw_prediction": raw_pred,
                    "policy_adjusted_prediction": adjusted,
                    "true": true,
                    "selected_model": selected_cfg.get("model", "") if selected_cfg else "",
                    "selected_feature_set": selected_cfg.get("feature_set", "") if selected_cfg else "",
                }
            )
        output_rows.append(out_row)
    write_csv(out / "predicted_condition_table_oof_production.csv", output_rows)
    val_rows = [row for row in output_rows if row.get("split") in {"val", "test"}]
    if val_rows:
        write_csv(out / "predicted_condition_table_val_production.csv", val_rows)
    metrics = {
        "row_count": len(output_rows),
        "selected_descriptor_count": len(selected),
        "policy_adjusted_table": display_path(out / "predicted_condition_table_oof_production.csv"),
    }
    write_json(out / "prediction_metrics_production.json", metrics)
    write_csv(out / "descriptor_policy_applied.csv", policy_rows)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-root", required=True)
    parser.add_argument("--production-selection", required=True)
    parser.add_argument("--descriptor-policy", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = export_predictions(args)
    print(f"Wrote production predictions to {metrics['policy_adjusted_table']}")
    print(f"rows={metrics['row_count']} selected_descriptors={metrics['selected_descriptor_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
