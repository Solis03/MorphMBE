"""Shared condition-control utilities for AFM prior v3."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rheed2morph.generative.afm_prior_v2_utils import V2_DESCRIPTOR_NAMES, bool_arg, compute_afm_descriptors_v2, format_float
from rheed2morph.generative.common import display_path, read_csv_rows, read_json, resolve_repo_path, write_csv_rows, write_json


DEFAULT_V3_DESCRIPTOR_CANDIDATES = [
    "rq",
    "ra",
    "robust_range",
    "mean_abs_gradient",
    "gradient_std",
    "gradient_anisotropy",
    "psd_low_power",
    "psd_mid_power",
    "psd_high_power",
    "psd_slope",
    "autocorrelation_length_px",
    "island_count",
    "island_mean_area_px",
]

SWEEP_DESCRIPTOR_NAMES = [
    "rq",
    "ra",
    "robust_range",
    "psd_low_power",
    "psd_mid_power",
    "psd_high_power",
    "psd_slope",
    "autocorrelation_length_px",
    "gradient_anisotropy",
    "island_count",
]


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if math.isfinite(output) else default


def correlation(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 2 or float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    try:
        from scipy.stats import spearmanr

        value = spearmanr(x, y, nan_policy="omit").correlation
        return float(value) if value is not None and math.isfinite(float(value)) else float("nan")
    except Exception:
        return correlation(np.argsort(np.argsort(np.asarray(x))), np.argsort(np.argsort(np.asarray(y))))


def monotonicity_score(requested: Sequence[float], generated: Sequence[float]) -> float:
    req = np.asarray(requested, dtype=np.float64)
    gen = np.asarray(generated, dtype=np.float64)
    mask = np.isfinite(req) & np.isfinite(gen)
    req = req[mask]
    gen = gen[mask]
    if req.size < 3:
        return float("nan")
    order = np.argsort(req)
    diffs = np.diff(gen[order])
    nonzero = diffs[np.abs(diffs) > 1e-12]
    if nonzero.size == 0:
        return 0.0
    return float(np.mean(nonzero > 0.0) * 2.0 - 1.0)


def robust_descriptor_columns(rows: Sequence[dict[str, str]], sensitivity: dict[str, Any] | None = None) -> tuple[list[str], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("Cannot select v3 descriptor columns from an empty condition table.")
    train_rows = [row for row in rows if row.get("split") == "train"] or list(rows)
    report_rows: list[dict[str, Any]] = []
    selected: list[str] = []
    sensitivity_by_name = {}
    if sensitivity:
        for item in sensitivity.get("descriptor_summaries", []):
            if isinstance(item, dict) and item.get("descriptor"):
                sensitivity_by_name[str(item["descriptor"])] = item
    for name in DEFAULT_V3_DESCRIPTOR_CANDIDATES:
        if name not in rows[0]:
            report_rows.append({"descriptor": name, "selected": False, "reason": "missing"})
            continue
        values = np.asarray([finite_float(row.get(name, "")) for row in train_rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size < max(5, len(train_rows) // 5):
            report_rows.append({"descriptor": name, "selected": False, "reason": "too_many_nonfinite"})
            continue
        std = float(np.std(finite))
        if std <= 1e-8:
            report_rows.append({"descriptor": name, "selected": False, "reason": "near_constant", "train_std": std})
            continue
        sens = sensitivity_by_name.get(name, {})
        corr_value = finite_float(sens.get("best_abs_pearson", sens.get("abs_pearson", "nan")))
        selected.append(name)
        report_rows.append({"descriptor": name, "selected": True, "reason": "robust_default", "train_std": std, "v2_abs_pearson": corr_value})
    if len(selected) < 6:
        numeric: list[tuple[str, float]] = []
        for name in V2_DESCRIPTOR_NAMES:
            if name not in rows[0] or name in selected:
                continue
            values = np.asarray([finite_float(row.get(name, "")) for row in train_rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size and float(np.std(values)) > 1e-8:
                numeric.append((name, float(np.std(values))))
        for name, std in sorted(numeric, key=lambda item: item[1], reverse=True):
            selected.append(name)
            report_rows.append({"descriptor": name, "selected": True, "reason": "fallback_high_train_std", "train_std": std})
            if len(selected) >= 10:
                break
    return selected, report_rows


def create_condition_schema_v3(
    condition_table_v2: Path,
    descriptors: Path,
    prototypes: Path,
    sensitivity_summary: Path | None,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir = resolve_repo_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    v2_rows = read_csv_rows(resolve_repo_path(condition_table_v2))
    sensitivity = read_json(resolve_repo_path(sensitivity_summary)) if sensitivity_summary is not None and resolve_repo_path(sensitivity_summary).is_file() else {}
    selected, selection_rows = robust_descriptor_columns(v2_rows, sensitivity)
    train_rows = [row for row in v2_rows if row.get("split") == "train"] or v2_rows
    matrix = np.asarray([[finite_float(row.get(name, "nan")) for name in selected] for row in train_rows], dtype=np.float64)
    medians = np.nanmedian(matrix, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    imputation_counts = {name: int(np.sum(~np.isfinite(matrix[:, i]))) for i, name in enumerate(selected)}
    imputed = matrix.copy()
    for index in range(imputed.shape[1]):
        mask = ~np.isfinite(imputed[:, index])
        imputed[mask, index] = medians[index]
    means = np.mean(imputed, axis=0)
    stds = np.std(imputed, axis=0)
    stds = np.where(stds > 1e-8, stds, 1.0)
    proto_rows = read_csv_rows(resolve_repo_path(prototypes)) if resolve_repo_path(prototypes).is_file() else []
    prototype_by_row = {row["row_id"]: row.get("prototype_id", "") for row in proto_rows}
    prototype_values = [int(float(value)) for value in prototype_by_row.values() if value != ""]
    prototype_count = max(prototype_values) + 1 if prototype_values else 0
    output_rows: list[dict[str, Any]] = []
    for row in v2_rows:
        out: dict[str, Any] = {
            "row_id": row["row_id"],
            "parent_row_id": row.get("parent_row_id", row["row_id"]),
            "sample_id": row.get("sample_id", ""),
            "group_id": row.get("group_id", ""),
            "split": row.get("split", ""),
            "network_input_path": row.get("network_input_path", ""),
            "descriptor_height_path": row.get("descriptor_height_path", ""),
            "source_path": row.get("source_path", ""),
            "source_kind": row.get("source_kind", ""),
            "prototype_id": prototype_by_row.get(row["row_id"], row.get("prototype_id", "")),
        }
        for index, name in enumerate(selected):
            raw = finite_float(row.get(name, "nan"))
            if not math.isfinite(raw):
                raw = float(medians[index])
            out[name] = format_float(raw)
            out[f"cond_{name}"] = format_float((raw - float(means[index])) / float(stds[index]))
        output_rows.append(out)
    schema = {
        "descriptor_columns": selected,
        "condition_columns": [f"cond_{name}" for name in selected],
        "condition_dim": len(selected) + int(prototype_count),
        "prototype_count": int(prototype_count),
        "prototype_one_hot": bool(prototype_count > 0),
        "descriptor_train_mean": {name: float(means[i]) for i, name in enumerate(selected)},
        "descriptor_train_std": {name: float(stds[i]) for i, name in enumerate(selected)},
        "descriptor_train_median": {name: float(medians[i]) for i, name in enumerate(selected)},
        "imputation_counts": imputation_counts,
        "source_condition_table_v2": display_path(resolve_repo_path(condition_table_v2)),
        "source_descriptors": display_path(resolve_repo_path(descriptors)),
        "source_prototypes": display_path(resolve_repo_path(prototypes)),
    }
    write_csv_rows(out_dir / "condition_table_v3.csv", output_rows)
    write_csv_rows(out_dir / "condition_selection_metrics.csv", selection_rows)
    write_json(out_dir / "condition_schema_v3.json", schema)
    dropped = [row["descriptor"] for row in selection_rows if not row.get("selected")]
    report = [
        "# Condition Schema V3 Selection Report",
        "",
        f"Selected descriptor columns: `{selected}`",
        f"Prototype count: `{prototype_count}`",
        f"Dropped or skipped descriptor columns: `{dropped}`",
        "",
        "The v3 schema intentionally uses a smaller robust descriptor subset than v2.",
        "Conditions are standardized using train-set means and standard deviations.",
    ]
    (out_dir / "condition_selection_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return schema


def build_condition_matrix_v3(
    rows: Sequence[dict[str, str]],
    selected_row_ids: Sequence[str] | np.ndarray,
    schema: dict[str, Any],
) -> np.ndarray:
    by_row = {str(row["row_id"]): row for row in rows}
    values: list[list[float]] = []
    proto_count = int(schema.get("prototype_count", 0))
    for row_id in selected_row_ids:
        row = by_row[str(row_id)]
        out = [finite_float(row[col], 0.0) for col in schema["condition_columns"]]
        if proto_count > 0:
            one_hot = [0.0] * proto_count
            proto = row.get("prototype_id", "")
            if proto != "":
                index = int(float(proto))
                if 0 <= index < proto_count:
                    one_hot[index] = 1.0
            out.extend(one_hot)
        values.append(out)
    return np.asarray(values, dtype=np.float32)


def split_condition_vector(condition: np.ndarray, schema: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    descriptor_dim = len(schema["condition_columns"])
    return condition[..., :descriptor_dim], condition[..., descriptor_dim:]


def raw_descriptor_to_condition(name: str, raw_value: float, schema: dict[str, Any]) -> float:
    mean = float(schema["descriptor_train_mean"][name])
    std = float(schema["descriptor_train_std"].get(name, 1.0) or 1.0)
    return (float(raw_value) - mean) / std


def condition_to_raw_descriptor(name: str, cond_value: float, schema: dict[str, Any]) -> float:
    mean = float(schema["descriptor_train_mean"][name])
    std = float(schema["descriptor_train_std"].get(name, 1.0) or 1.0)
    return float(cond_value) * std + mean


def condition_row_to_vector(row: dict[str, str], schema: dict[str, Any]) -> np.ndarray:
    values = [finite_float(row.get(col, "0"), 0.0) for col in schema["condition_columns"]]
    proto_count = int(schema.get("prototype_count", 0))
    if proto_count > 0:
        one_hot = [0.0] * proto_count
        proto = row.get("prototype_id", "")
        if proto != "":
            index = int(float(proto))
            if 0 <= index < proto_count:
                one_hot[index] = 1.0
        values.extend(one_hot)
    return np.asarray(values, dtype=np.float32)


def descriptor_error_score(generated: dict[str, float], target_row: dict[str, str], schema: dict[str, Any]) -> float:
    errors = []
    for name in schema["descriptor_columns"]:
        gen = finite_float(generated.get(name, float("nan")))
        target = finite_float(target_row.get(name, float("nan")))
        std = float(schema["descriptor_train_std"].get(name, 1.0) or 1.0)
        if math.isfinite(gen) and math.isfinite(target):
            errors.append(abs(gen - target) / max(std, 1e-6))
    return float(np.mean(errors)) if errors else float("inf")


def summarize_requested_generated(rows: Sequence[dict[str, Any]], schema: dict[str, Any], prefix: str = "generated_") -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for name in schema["descriptor_columns"]:
        requested = [finite_float(row.get(f"requested_{name}", row.get(name, "nan"))) for row in rows]
        generated = [finite_float(row.get(f"{prefix}{name}", "nan")) for row in rows]
        req = np.asarray(requested, dtype=np.float64)
        gen = np.asarray(generated, dtype=np.float64)
        mask = np.isfinite(req) & np.isfinite(gen)
        if not np.any(mask):
            continue
        delta = gen[mask] - req[mask]
        output.append(
            {
                "descriptor": name,
                "count": int(np.sum(mask)),
                "mae": float(np.mean(np.abs(delta))),
                "rmse": float(np.sqrt(np.mean(delta * delta))),
                "pearson": correlation(req[mask], gen[mask]),
                "spearman": rank_correlation(req[mask], gen[mask]),
                "monotonicity": monotonicity_score(req[mask], gen[mask]),
            }
        )
    return output


def adapt_external_condition_row(
    row: dict[str, str],
    schema: dict[str, Any],
    mode: str = "predicted",
    fill_missing_with_train_mean: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out: dict[str, Any] = {
        "row_id": row.get("row_id", ""),
        "sample_id": row.get("sample_id", ""),
        "group_id": row.get("group_id", ""),
        "split": row.get("split", ""),
        "network_input_path": row.get("network_input_path", ""),
        "descriptor_height_path": row.get("descriptor_height_path", ""),
        "prototype_id": "",
    }
    mapped: list[str] = []
    filled: list[str] = []
    for name in schema["descriptor_columns"]:
        if mode == "mean":
            raw = float(schema["descriptor_train_mean"][name])
            filled.append(name)
        else:
            keys = [f"pred_{name}", name] if mode == "predicted" else [name, f"true_{name}"]
            raw_text = ""
            for key in keys:
                if row.get(key, "") != "":
                    raw_text = row[key]
                    break
            if raw_text == "":
                if not fill_missing_with_train_mean:
                    raise ValueError(f"Cannot map descriptor {name}; pass fill_missing_with_train_mean to fill safely.")
                raw = float(schema["descriptor_train_mean"][name])
                filled.append(name)
            else:
                raw = finite_float(raw_text, float(schema["descriptor_train_mean"][name]))
                mapped.append(name)
        out[name] = format_float(raw)
        out[f"cond_{name}"] = format_float(raw_descriptor_to_condition(name, raw, schema))
    report = {
        "mode": mode,
        "mapped_descriptors": mapped,
        "filled_descriptors": filled,
        "mapped_descriptor_count": len(mapped),
        "filled_descriptor_count": len(filled),
    }
    return out, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create AFM condition-control v3 schema.")
    parser.add_argument("--condition-table-v2", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--prototypes", type=Path, required=True)
    parser.add_argument("--sensitivity-summary", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    schema = create_condition_schema_v3(args.condition_table_v2, args.descriptors, args.prototypes, args.sensitivity_summary, args.out)
    print(f"Wrote condition schema v3 to {display_path(resolve_repo_path(args.out))}")
    print(f"condition_dim={schema['condition_dim']} descriptors={len(schema['descriptor_columns'])}")


if __name__ == "__main__":
    main()
