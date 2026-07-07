"""Build the supervised MVP-9 RHEED shape-bag dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MVP8_ROOT = REPO_ROOT / "reports" / "rheed_shape_bag_input_mvp" / "20260703_165110"
DEFAULT_MVP6_ROOT = REPO_ROOT / "reports" / "rheed_ssl_temporal_mvp" / "20260703_072054"
DEFAULT_MVP9_ROOT = REPO_ROOT / "reports" / "rheed_shape_bag_model_mvp"
AGG_PREFIXES = ("weighted_mean_", "weighted_median_", "trimmed_mean_", "std_", "iqr_")


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def numeric_sample_id(value: str) -> str:
    match = re.search(r"(\d{4,})", str(value))
    return match.group(1) if match else str(value)


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def default_target_columns(schema: dict[str, Any], paired_rows: Sequence[dict[str, str]]) -> list[str]:
    preferred = [
        "rq",
        "ra",
        "robust_range",
        "psd_slope",
        "autocorrelation_length_px",
        "gradient_anisotropy",
        "island_count",
        "island_mean_area_px",
    ]
    descriptor_columns = list(schema.get("descriptor_columns", []))
    available = set(paired_rows[0].keys()) if paired_rows else set()
    out = [name for name in preferred if name in descriptor_columns and name in available]
    for name in descriptor_columns:
        if name in available and name not in out:
            out.append(name)
    return out


def feature_base_name(column: str) -> str:
    for prefix in AGG_PREFIXES:
        if column.startswith(prefix):
            return column[len(prefix) :]
    return column


def build_feature_columns(shape_feature_rows: Sequence[dict[str, str]], stable_base_names: Sequence[str]) -> tuple[list[str], list[str]]:
    if not shape_feature_rows:
        return [], []
    columns = [name for name in shape_feature_rows[0] if name != "sample_id"]
    stable_set = set(stable_base_names)
    stable_columns = [name for name in columns if feature_base_name(name) in stable_set]
    return stable_columns, columns


def split_folds(rows: Sequence[dict[str, Any]], n_splits: int, seed: int) -> list[dict[str, Any]]:
    groups = sorted({str(row["group_id"]) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    group_to_fold = {group: index % max(1, n_splits) for index, group in enumerate(groups)}
    out = []
    for row in rows:
        fold = group_to_fold[str(row["group_id"])]
        original = row.get("original_split", row.get("split", ""))
        out.append(
            {
                "pair_id": row["pair_id"],
                "sample_id": row["sample_id"],
                "group_id": row["group_id"],
                "fold_id": fold,
                "strict_split": "val",
                "original_split": original,
                "is_original_train": int(original == "train"),
                "is_original_val": int(original == "val"),
                "is_original_test": int(original == "test"),
            }
        )
    return out


def scaler_stats(rows: Sequence[dict[str, Any]], target_columns: Sequence[str], train_pair_ids: set[str]) -> dict[str, dict[str, float]]:
    train_rows = [row for row in rows if row["pair_id"] in train_pair_ids]
    stats = {"mean": {}, "std": {}}
    for column in target_columns:
        values = np.asarray([finite_float(row[column]) for row in train_rows], dtype=np.float64)
        stats["mean"][column] = float(values.mean()) if values.size else 0.0
        std = float(values.std()) if values.size else 1.0
        stats["std"][column] = std if std > 1e-8 else 1.0
    return stats


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    shape_manifest = read_csv(resolve_path(args.shape_bag_manifest))
    shape_features = read_csv(resolve_path(args.shape_features))
    paired_rows = read_csv(resolve_path(args.paired_index))
    condition_schema = json.loads(resolve_path(args.condition_schema).read_text(encoding="utf-8"))
    stable_base_names = [
        line.strip()
        for line in resolve_path(args.stable_feature_list).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    shape_by_numeric = {numeric_sample_id(row["sample_id"]): row for row in shape_manifest}
    features_by_numeric = {numeric_sample_id(row["sample_id"]): row for row in shape_features}
    target_columns = default_target_columns(condition_schema, paired_rows)
    stable_columns, raw_columns = build_feature_columns(shape_features, stable_base_names)
    if args.use_raw_240_features:
        default_feature_columns = raw_columns
    else:
        default_feature_columns = stable_columns

    supervised_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    unmatched_pairs: list[dict[str, Any]] = []
    matched_shape_ids: set[str] = set()
    for pair in paired_rows:
        key = numeric_sample_id(pair.get("sample_id", ""))
        shape = shape_by_numeric.get(key)
        features = features_by_numeric.get(key)
        if shape is None or features is None:
            unmatched_pairs.append({"sample_id": pair.get("sample_id", ""), "pair_id": pair.get("pair_id", ""), "reason": "missing_shape_bag"})
            continue
        matched_shape_ids.add(key)
        row = {
            "pair_id": pair.get("pair_id", pair.get("row_id", key)),
            "row_id": pair.get("row_id", ""),
            "sample_id": key,
            "shape_sample_id": shape["sample_id"],
            "group_id": pair.get("group_id", key),
            "growth_id": pair.get("growth_id", pair.get("group_id", key)),
            "split": pair.get("split", ""),
            "original_split": pair.get("split", ""),
            "shape_bag_npz": shape["shape_bag_npz"],
            "shape_input_folder": shape["shape_input_folder"],
            "preview_grid": shape.get("preview_grid", ""),
            "candidate_csv": shape.get("candidate_csv", ""),
            "cached_tensor_path": pair.get("cached_tensor_path", ""),
            "network_input_path": pair.get("network_input_path", ""),
            "descriptor_height_path": pair.get("descriptor_height_path", ""),
            "prototype_id": pair.get("prototype_id", ""),
        }
        for column in default_feature_columns:
            row[f"shape_feature::{column}"] = features.get(column, "")
        supervised_rows.append(row)
        target = {
            "pair_id": row["pair_id"],
            "row_id": row["row_id"],
            "sample_id": key,
            "group_id": row["group_id"],
            "split": row["split"],
            "prototype_id": pair.get("prototype_id", ""),
        }
        for column in target_columns:
            target[column] = pair.get(column, "")
            cond_column = f"cond_{column}"
            if cond_column in pair:
                target[cond_column] = pair[cond_column]
        target_rows.append(target)
        if args.limit and len(supervised_rows) >= int(args.limit):
            break

    unpaired_shapes = [
        {"sample_id": row["sample_id"], "reason": "not_in_supervised_pair_index"}
        for row in shape_manifest
        if numeric_sample_id(row["sample_id"]) not in matched_shape_ids
    ]
    folds = split_folds(supervised_rows, int(args.n_splits), int(args.seed))
    fold_ids = sorted({row["fold_id"] for row in folds})
    fold_scalers: dict[str, Any] = {}
    for fold_id in fold_ids:
        val_ids = {row["pair_id"] for row in folds if int(row["fold_id"]) == int(fold_id)}
        train_ids = {row["pair_id"] for row in folds if row["pair_id"] not in val_ids}
        fold_scalers[str(fold_id)] = scaler_stats(target_rows, target_columns, train_ids)
    original_train_ids = {row["pair_id"] for row in supervised_rows if row.get("split") == "train"}
    fold_scalers["original_split"] = scaler_stats(target_rows, target_columns, original_train_ids)

    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "supervised_shape_bag_index.csv", supervised_rows)
    target_fieldnames = ["pair_id", "row_id", "sample_id", "group_id", "split", "prototype_id"]
    for column in target_columns:
        target_fieldnames.append(column)
        if f"cond_{column}" in target_rows[0]:
            target_fieldnames.append(f"cond_{column}")
    write_csv(out_dir / "target_conditions_shape_bag.csv", target_rows, target_fieldnames)
    write_csv(out_dir / "strict_fold_assignments.csv", folds)
    write_csv(out_dir / "unmatched_shape_bags.csv", unpaired_shapes)
    write_csv(out_dir / "unmatched_pairs.csv", unmatched_pairs)
    feature_schema = {
        "stable_base_feature_names": stable_base_names,
        "stable_feature_columns": stable_columns,
        "raw_240_feature_columns": raw_columns,
        "default_feature_columns": default_feature_columns,
        "use_raw_240_features_by_default": False,
        "shape_feature_prefix": "shape_feature::",
    }
    target_schema = {
        "descriptor_columns": target_columns,
        "condition_columns": [f"cond_{column}" for column in target_columns if f"cond_{column}" in target_rows[0]],
        "prototype_column": "prototype_id" if any(row.get("prototype_id", "") != "" for row in target_rows) else "",
        "fold_target_scalers": fold_scalers,
        "source_condition_schema": display_path(resolve_path(args.condition_schema)),
    }
    write_json(out_dir / "feature_schema_shape_bag.json", feature_schema)
    write_json(out_dir / "target_schema_shape_bag.json", target_schema)
    split_summary = {
        "shape_bag_samples": len(shape_manifest),
        "matched_supervised_pairs": len(supervised_rows),
        "unmatched_shape_bags": len(unpaired_shapes),
        "unmatched_pairs": len(unmatched_pairs),
        "target_columns": target_columns,
        "stable_feature_count": len(stable_columns),
        "raw_feature_count": len(raw_columns),
        "fold_counts": {str(fold): sum(1 for row in folds if row["fold_id"] == fold) for fold in fold_ids},
        "original_split_counts": {split: sum(1 for row in supervised_rows if row.get("split") == split) for split in sorted({row.get("split", "") for row in supervised_rows})},
    }
    write_json(out_dir / "split_summary.json", split_summary)
    report_lines = [
        "# Shape-Bag Supervised Pairing Report",
        "",
        f"Shape-bag samples: {len(shape_manifest)}",
        f"Matched supervised pairs: {len(supervised_rows)}",
        f"Unmatched shape bags: {len(unpaired_shapes)}",
        f"Unmatched pair rows: {len(unmatched_pairs)}",
        f"Default stable feature columns: {len(stable_columns)}",
        f"Raw feature columns available for diagnostic ablation: {len(raw_columns)}",
        "",
        "Frames from a sample are never split independently; each `shape_bag.npz` remains one supervised sample.",
    ]
    (out_dir / "pairing_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return split_summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-bag-manifest", default=str(DEFAULT_MVP8_ROOT / "rheed_shape_bag_manifest.csv"))
    parser.add_argument("--shape-features", default=str(DEFAULT_MVP8_ROOT / "global_sample_shape_features.csv"))
    parser.add_argument("--stable-feature-list", default=str(DEFAULT_MVP8_ROOT / "default_training_feature_names.txt"))
    parser.add_argument("--paired-index", default=str(DEFAULT_MVP6_ROOT / "data" / "rheed_supervised_pair_index.csv"))
    parser.add_argument("--condition-table", default=str(REPO_ROOT / "reports" / "afm_condition_control_v3" / "20260703_060549" / "condition_schema_v3" / "condition_table_v3.csv"))
    parser.add_argument("--condition-schema", default=str(REPO_ROOT / "reports" / "afm_condition_control_v3" / "20260703_060549" / "condition_schema_v3" / "condition_schema_v3.json"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--mvp8-root", default=str(DEFAULT_MVP8_ROOT))
    parser.add_argument("--mvp5-root", default="")
    parser.add_argument("--mvp6-root", default=str(DEFAULT_MVP6_ROOT))
    parser.add_argument("--target-schema", choices=["v3", "v4", "shared"], default="v3")
    parser.add_argument("--split-mode", choices=["original_mvp6", "group_kfold", "leave_one_group_out"], default="group_kfold")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--include-metadata", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=False)
    parser.add_argument("--use-raw-240-features", action="store_true")
    parser.add_argument("--strict", type=lambda x: str(x).lower() in {"1", "true", "yes", "y"}, default=False)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_dataset(args)
    print(f"Wrote supervised shape-bag dataset to {display_path(resolve_path(args.out))}")
    print(f"matched={summary['matched_supervised_pairs']} stable_features={summary['stable_feature_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

