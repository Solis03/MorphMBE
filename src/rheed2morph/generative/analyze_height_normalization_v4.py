"""Diagnose AFM physical-height normalization for v4 calibration."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from rheed2morph.generative.common import display_path, load_height_array, read_csv_rows, resolve_repo_path, write_csv_rows, write_json
from rheed2morph.generative.height_calibration_v4 import compute_height_descriptors, summarize_scale_values
from rheed2morph.generative.condition_control_v3_utils import correlation, finite_float


ROUGHNESS_NAMES = ("rq", "ra", "height_std", "robust_range")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose AFM height normalization for v4 calibration.")
    parser.add_argument("--mvp3-root", type=Path, required=True)
    parser.add_argument("--mvp4-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def _safe_ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) <= 1e-8:
        return float("nan")
    return float(num / den)


def _write_plots(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    for ax, name in zip(axes, ("rq", "ra", "robust_range")):
        x = np.asarray([finite_float(row.get(f"network_{name}", "nan")) for row in rows], dtype=np.float64)
        y = np.asarray([finite_float(row.get(f"physical_{name}", "nan")) for row in rows], dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], s=10, alpha=0.7)
        ax.set_xlabel(f"network {name}")
        ax.set_ylabel(f"physical {name} nm")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(out_dir / "physical_vs_network_roughness_scatter.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    for ax, name in zip(axes, ("scale_rq", "scale_ra", "scale_range")):
        values = np.asarray([finite_float(row.get(name, "nan")) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        ax.hist(values, bins=30, alpha=0.85)
        ax.set_title(name)
        ax.set_xlabel("nm per normalized unit")
    fig.tight_layout()
    fig.savefig(out_dir / "scale_factor_histograms.png")
    plt.close(fig)


def analyze_height_normalization(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = resolve_repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    mvp3_root = resolve_repo_path(args.mvp3_root)
    mvp4_root = resolve_repo_path(args.mvp4_root)
    index_rows = read_csv_rows(mvp3_root / "data" / "afm_prior_v2_index.csv")
    descriptor_rows = {row["row_id"]: row for row in read_csv_rows(mvp3_root / "data" / "afm_prior_v2_descriptors.csv")}
    v3_rows = {row["row_id"]: row for row in read_csv_rows(mvp4_root / "condition_schema_v3" / "condition_table_v3.csv")}
    if args.limit is not None:
        index_rows = index_rows[: int(args.limit)]
    output_rows: list[dict[str, Any]] = []
    for row in index_rows:
        row_id = row["row_id"]
        try:
            network = load_height_array(resolve_repo_path(Path(row["network_input_path"])))
            physical = load_height_array(resolve_repo_path(Path(row["descriptor_height_path"])))
        except Exception as exc:
            output_rows.append({"row_id": row_id, "load_error": str(exc)})
            continue
        network_desc = compute_height_descriptors(network)
        physical_desc = compute_height_descriptors(physical)
        stored = descriptor_rows.get(row_id, {})
        out: dict[str, Any] = {
            "row_id": row_id,
            "sample_id": row.get("sample_id", ""),
            "group_id": row.get("group_id", ""),
            "split": row.get("split", ""),
            "network_input_path": row.get("network_input_path", ""),
            "descriptor_height_path": row.get("descriptor_height_path", ""),
            "scan_size_um": row.get("scan_size_um", ""),
            "height_shape": row.get("height_shape", ""),
            "network_min": float(np.min(network)),
            "network_max": float(np.max(network)),
            "network_mean": float(np.mean(network)),
            "network_std": float(np.std(network)),
            "physical_min": float(np.min(physical)),
            "physical_max": float(np.max(physical)),
            "physical_mean": float(np.mean(physical)),
            "physical_std": float(np.std(physical)),
            "looks_per_image_normalized": bool(np.isclose(np.min(network), -1.0, atol=1e-4) and np.isclose(np.max(network), 1.0, atol=1e-4)),
        }
        for name in set(ROUGHNESS_NAMES + ("psd_slope", "autocorrelation_length_px", "island_count")):
            out[f"network_{name}"] = network_desc.get(name, float("nan"))
            out[f"physical_{name}"] = physical_desc.get(name, float("nan"))
            if stored:
                out[f"stored_{name}"] = stored.get(name, "")
            if row_id in v3_rows and name in v3_rows[row_id]:
                out[f"v3_table_{name}"] = v3_rows[row_id][name]
        out["scale_rq"] = _safe_ratio(float(physical_desc["rq"]), float(network_desc["rq"]))
        out["scale_ra"] = _safe_ratio(float(physical_desc["ra"]), float(network_desc["ra"]))
        out["scale_std"] = _safe_ratio(float(physical_desc["height_std"]), float(network_desc["height_std"]))
        out["scale_range"] = _safe_ratio(float(physical_desc["robust_range"]), float(network_desc["robust_range"]))
        output_rows.append(out)
    write_csv_rows(out_dir / "height_scale_table.csv", output_rows)
    train_rows = [row for row in output_rows if row.get("split") == "train"]
    if not train_rows:
        train_rows = output_rows
    scale_values = []
    for row in train_rows:
        for name in ("scale_rq", "scale_ra", "scale_range"):
            value = finite_float(row.get(name, "nan"))
            if math.isfinite(value) and value > 0:
                scale_values.append(value)
    bounds = summarize_scale_values(scale_values)
    all_per_image = [bool(row.get("looks_per_image_normalized")) for row in output_rows if "looks_per_image_normalized" in row]
    summary = {
        "mvp3_root": display_path(mvp3_root),
        "mvp4_root": display_path(mvp4_root),
        "row_count": len(output_rows),
        "train_row_count": len(train_rows),
        "per_image_normalized_rate": float(np.mean(all_per_image)) if all_per_image else 0.0,
        "network_min_median": float(np.nanmedian([finite_float(row.get("network_min", "nan")) for row in output_rows])),
        "network_max_median": float(np.nanmedian([finite_float(row.get("network_max", "nan")) for row in output_rows])),
        "network_std_median": float(np.nanmedian([finite_float(row.get("network_std", "nan")) for row in output_rows])),
        "physical_rq_median": float(np.nanmedian([finite_float(row.get("physical_rq", "nan")) for row in output_rows])),
        "network_rq_median": float(np.nanmedian([finite_float(row.get("network_rq", "nan")) for row in output_rows])),
        "scale_bounds": bounds,
        "rq_physical_network_pearson": correlation([finite_float(row.get("network_rq", "nan")) for row in output_rows], [finite_float(row.get("physical_rq", "nan")) for row in output_rows]),
        "ra_physical_network_pearson": correlation([finite_float(row.get("network_ra", "nan")) for row in output_rows], [finite_float(row.get("physical_ra", "nan")) for row in output_rows]),
        "robust_range_physical_network_pearson": correlation([finite_float(row.get("network_robust_range", "nan")) for row in output_rows], [finite_float(row.get("physical_robust_range", "nan")) for row in output_rows]),
        "height_scale_table": display_path(out_dir / "height_scale_table.csv"),
    }
    write_json(out_dir / "height_scale_summary.json", summary)
    audit = {
        "network_input_units": "per-image normalized network space" if summary["per_image_normalized_rate"] > 0.9 else "mixed_or_unknown",
        "physical_descriptor_units": "nm-like physical height units from descriptor_height_path",
        "condition_table_v2_v3_units": "raw descriptor columns are physical descriptors; cond_* columns are standardized train-set descriptors",
        "decoder_output_units": "normalized network-input height map",
        "absolute_roughness_requires_external_scale": bool(summary["per_image_normalized_rate"] > 0.9),
    }
    write_json(out_dir / "descriptor_units_audit.json", audit)
    _write_plots(out_dir, output_rows)
    report = [
        "# Height Normalization V4 Diagnosis",
        "",
        f"Rows analyzed: `{len(output_rows)}`",
        f"Per-image normalized rate: `{summary['per_image_normalized_rate']:.3f}`",
        f"Network min/max median: `{summary['network_min_median']:.4g}` / `{summary['network_max_median']:.4g}`",
        f"Network Rq median: `{summary['network_rq_median']:.4g}`",
        f"Physical Rq median: `{summary['physical_rq_median']:.4g}`",
        f"Scale bounds: `{bounds}`",
        "",
        "The standardized network inputs are effectively per-image normalized to [-1, 1]. "
        "Absolute height-scale descriptors such as Rq, Ra, and robust range cannot be recovered directly from decoder output without an external physical height-scale calibration step.",
    ]
    (out_dir / "height_normalization_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = analyze_height_normalization(args)
    print(f"Wrote height normalization diagnosis to {display_path(resolve_repo_path(args.out))}")
    print(f"rows={summary['row_count']} per_image_normalized_rate={summary['per_image_normalized_rate']:.3f}")


if __name__ == "__main__":
    main()
