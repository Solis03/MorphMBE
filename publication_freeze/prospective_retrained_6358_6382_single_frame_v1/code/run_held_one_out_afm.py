#!/usr/bin/env python3
"""Run 28-fold held-one-out A3 AFM retrieval and build a paper atlas."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


THIS = Path(__file__).resolve()
PKG = THIS.parents[1]
REPO = next(
    parent
    for parent in THIS.parents
    if (parent / "publication_freeze" / "rheed_afm_single_frame_v1_2026-07-18").is_dir()
)
FREEZE = REPO / "publication_freeze/rheed_afm_single_frame_v1_2026-07-18"
PROSPECTIVE_SOURCE = REPO / "publication_freeze/prospective_unseen_single_frame_v1"
OUTPUT = PKG / "predictions/held_one_out_afm_28"
BANK_OUTPUT = PKG / "models/visual_model/full_labeled_28_afm_bank_manifest.csv"
ATLAS_STEM = PKG / "figures/main/Figure3_held_one_out_afm_prediction_atlas"
PER_SAMPLE_FIGURES = PKG / "figures/held_one_out_afm/per_sample"
HISTORICAL_IDS = [
    "6022",
    "6028",
    "6029",
    "6033",
    "6047",
    "6048",
    "6056",
    "6057",
    "6062",
    "6063",
    "6070",
    "6072",
    "6078",
    "6080",
    "6081",
    "6082",
    "6084",
    "6085",
    "6090",
    "6094",
    "6095",
    "6099",
    "6101",
]
ADDED_TRAIN_IDS = ["6358", "6382"]
PROSPECTIVE_TEST_IDS = ["6342", "6389", "6390"]
EXTRA_IDS = ["6342", "6358", "6382", "6389", "6390"]
SAMPLE_IDS = HISTORICAL_IDS + EXTRA_IDS
FORCED_GROUND_TRUTH_SCANS = {
    "6342": 5,
    "6389": 3,
    "6390": 1,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rq_nm(array: np.ndarray) -> float:
    values = np.asarray(array, dtype=np.float64)
    values = values[np.isfinite(values)]
    values = values - float(np.mean(values))
    return float(np.sqrt(np.mean(values**2)))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_retrieval_module() -> Any:
    sys.path.insert(0, str(PROSPECTIVE_SOURCE / "code"))
    return importlib.import_module("generate_full_cohort_retrieval_images")


def display_id(sample_id: str) -> str:
    return f"N{sample_id}" if sample_id in EXTRA_IDS else sample_id


def source_role(sample_id: str) -> str:
    if sample_id in ADDED_TRAIN_IDS:
        return "original_added_training"
    if sample_id in PROSPECTIVE_TEST_IDS:
        return "original_prospective_test"
    return "historical"


def build_full_labeled_bank() -> pd.DataFrame:
    historical = pd.read_csv(
        FREEZE / "models/visual_model/full_cohort_bank/afm_bank_manifest.csv",
        dtype={"sample_id": str},
    )
    extra_manifest = pd.read_csv(
        PKG / "ground_truth_afm/quarter_afm_manifest.csv",
        dtype={"sample_id": str},
    )
    extra_rows: list[dict[str, Any]] = []
    for original in extra_manifest.to_dict("records"):
        sample_id = str(original["sample_id"]).removeprefix("N")
        row = dict(original)
        row.update(
            {
                "sample_id": sample_id,
                "growth_run_id": sample_id,
                "source_afm_file": original["source_path"],
                "plane_corrected_array_path": original["output_path"],
                "second_order_afm_path": original["output_path"],
                "scan_size_x_um": 1.0,
                "scan_size_y_um": 1.0,
                "resolution_x": 256,
                "resolution_y": 256,
                "height_unit": "nm",
                "robust_height_range_nm": original["physical_robust_height_range"],
                "height_skewness": original["physical_skewness"],
                "height_kurtosis": original["physical_kurtosis"],
                "paired_primary": True,
                "paired_exploratory": False,
                "unpaired_support": False,
                "representative_for_sample": False,
                "quality_pass": True,
                "quality_flags": "",
                "source_array_hash": original["output_sha256"],
            }
        )
        extra_rows.append(row)
    bank = pd.concat([historical, pd.DataFrame(extra_rows)], ignore_index=True, sort=False)
    bank["sample_id"] = bank["sample_id"].astype(str)
    if bank["sample_id"].drop_duplicates().tolist() != SAMPLE_IDS:
        raise RuntimeError(
            "Full labeled AFM bank does not contain the expected ordered 28 sample groups"
        )
    if len(bank) != len(historical) + len(extra_manifest):
        raise RuntimeError("Unexpected AFM bank row count")
    BANK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    bank.to_csv(BANK_OUTPUT, index=False)
    return bank


def select_ground_truth_maps(
    bank: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    target_index = predictions.set_index("sample_id")["ground_truth_T4_rq_nm"]
    rows: list[dict[str, Any]] = []
    for sample_id in SAMPLE_IDS:
        candidates = bank[
            bank["sample_id"].eq(sample_id)
            & bank["quality_pass"].astype(str).eq("True")
        ].copy()
        if candidates.empty:
            raise RuntimeError(f"No quality-passing AFM maps for {sample_id}")
        target = float(target_index.loc[sample_id])
        if sample_id in FORCED_GROUND_TRUTH_SCANS:
            scan_number = FORCED_GROUND_TRUTH_SCANS[sample_id]
            selected = candidates[
                pd.to_numeric(candidates["scan_number"], errors="coerce").eq(scan_number)
            ]
            if len(selected) != 1:
                raise RuntimeError(
                    f"Expected exactly one ground-truth scan {scan_number} for {sample_id}"
                )
            chosen = selected.iloc[0]
            method = f"user_forced_ground_truth_{scan_number}"
        else:
            candidates["distance_to_T4_rq_nm"] = (
                pd.to_numeric(candidates["rq_nm"], errors="coerce") - target
            ).abs()
            candidates = candidates.sort_values(
                ["distance_to_T4_rq_nm", "afm_file_id"]
            )
            chosen = candidates.iloc[0]
            method = "minimum_absolute_rq_distance_to_T4_target"
        selected_rq = float(chosen["rq_nm"])
        rows.append(
            {
                "sample_id": sample_id,
                "display_sample_id": display_id(sample_id),
                "sample_role": source_role(sample_id),
                "selection_method": method,
                "forced_scan_number": FORCED_GROUND_TRUTH_SCANS.get(sample_id),
                "selected_scan_number": (
                    int(chosen["scan_number"])
                    if pd.notna(chosen.get("scan_number"))
                    else None
                ),
                "ground_truth_afm_file_id": str(chosen["afm_file_id"]),
                "ground_truth_afm_path": str(chosen["second_order_afm_path"]),
                "ground_truth_rq_nm": selected_rq,
                "sample_T4_target_rq_nm": target,
                "absolute_rq_distance_to_T4_nm": abs(selected_rq - target),
                "scan_size_x_um": float(chosen["scan_size_x_um"]),
                "scan_size_y_um": float(chosen["scan_size_y_um"]),
                "available_quality_afm_count": len(candidates),
            }
        )
    frame = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "ground_truth_selection.csv", index=False)
    return frame


def run_held_one_out_retrieval(
    bank: pd.DataFrame,
    predictions: pd.DataFrame,
    selections: pd.DataFrame,
    retrieval: Any,
) -> pd.DataFrame:
    map_dir = OUTPUT / "retrieved_maps_q50"
    source_dir = OUTPUT / "source_unit_shape_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    selection_index = selections.set_index("sample_id")
    rows: list[dict[str, Any]] = []
    for prediction in predictions.to_dict("records"):
        sample_id = str(prediction["sample_id"])
        raw_rq = float(prediction["leave_one_out_predicted_rq_nm"])
        physical_rq = max(
            float(prediction["leave_one_out_predicted_rq_nm_clipped_nonnegative"]),
            1e-3,
        )
        fold_bank = bank[~bank["sample_id"].eq(sample_id)].copy()
        fold_group_ids = fold_bank["sample_id"].drop_duplicates().tolist()
        expected_group_ids = [value for value in SAMPLE_IDS if value != sample_id]
        if fold_group_ids != expected_group_ids:
            raise RuntimeError(f"Unexpected fold AFM bank for {sample_id}")
        ranked = retrieval.rank_bank(fold_bank, physical_rq)
        top5 = retrieval.top_distinct_groups(ranked, 5)
        source = ranked.iloc[0]
        source_sample_id = str(source["sample_id"])
        if source_sample_id == sample_id:
            raise RuntimeError(f"Held-out AFM leakage for {sample_id}")
        source_map = retrieval.read_map(REPO, str(source["second_order_afm_path"]))
        unit_shape = retrieval.project_unit_rq(source_map)
        retrieved_map = retrieval.physical_from_q(unit_shape, physical_rq)
        source_path = (
            source_dir
            / f"{display_id(sample_id)}_source_{display_id(source_sample_id)}_{source['afm_file_id']}_unit.npy"
        )
        map_path = (
            map_dir / f"{display_id(sample_id)}_A3_held_one_out_q50_retrieved.npy"
        )
        np.save(source_path, unit_shape.astype(np.float32))
        np.save(map_path, retrieved_map.astype(np.float32))
        selected = selection_index.loc[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "display_sample_id": display_id(sample_id),
                "sample_role": source_role(sample_id),
                "method_id": "A3_full_cohort_rq_conditioned_held_one_out",
                "raw_leave_one_out_predicted_rq_nm": raw_rq,
                "rendered_physical_rq_nm": physical_rq,
                "negative_raw_prediction_clipped_to_1e_3_for_map": raw_rq < 0,
                "source_sample_id": source_sample_id,
                "display_source_sample_id": display_id(source_sample_id),
                "source_afm_file_id": str(source["afm_file_id"]),
                "source_afm_path": str(source["second_order_afm_path"]),
                "source_scan_size_x_um": float(source["scan_size_x_um"]),
                "source_scan_size_y_um": float(source["scan_size_y_um"]),
                "source_rank_score": float(source["rank_score"]),
                "source_rq_nm": float(source["rq_nm"]),
                "top5_source_sample_ids": json.dumps(
                    top5["sample_id"].astype(str).tolist()
                ),
                "top5_source_afm_file_ids": json.dumps(
                    top5["afm_file_id"].astype(str).tolist()
                ),
                "fold_source_group_count": len(fold_group_ids),
                "fold_source_group_ids": json.dumps(fold_group_ids),
                "held_out_sample_absent_from_afm_bank": sample_id not in fold_group_ids,
                "ground_truth_selection_method": selected["selection_method"],
                "ground_truth_afm_file_id": selected["ground_truth_afm_file_id"],
                "ground_truth_afm_path": selected["ground_truth_afm_path"],
                "ground_truth_selected_rq_nm": float(selected["ground_truth_rq_nm"]),
                "ground_truth_T4_rq_nm": float(selected["sample_T4_target_rq_nm"]),
                "selected_ground_truth_rq_absolute_error_nm": abs(
                    physical_rq - float(selected["ground_truth_rq_nm"])
                ),
                "retrieved_q50_map_path": rel(map_path),
                "source_unit_shape_map_path": rel(source_path),
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT / "retrieval_results.csv", index=False)
    return results


def height_limits(array: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(array, dtype=float)
    finite = finite[np.isfinite(finite)]
    low, high = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        center = float(np.mean(finite)) if finite.size else 0.0
        return center - 0.5, center + 0.5
    return float(low), float(high)


def height_label(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 0.01:
        return f"{value:.4f}"
    if magnitude < 1.0:
        return f"{value:.2f}"
    return f"{value:.1f}"


def add_height_bar(ax: plt.Axes, low: float, high: float) -> None:
    bar = ax.inset_axes([1.015, 0.08, 0.045, 0.82])
    bar.imshow(np.linspace(high, low, 160)[:, None], cmap="viridis", aspect="auto")
    bar.set_xticks([])
    bar.set_yticks([0, 159])
    bar.set_yticklabels([height_label(high), height_label(low)], fontsize=4.5)
    bar.yaxis.tick_right()
    bar.tick_params(length=1.5, pad=1)
    bar.set_title("nm", fontsize=4.5, pad=1)


def show_afm(ax: plt.Axes, array: np.ndarray, title: str, title_color: str) -> None:
    low, high = height_limits(array)
    ax.imshow(array, cmap="viridis", vmin=low, vmax=high)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=6.4, color=title_color, pad=3, linespacing=1.18)
    add_height_bar(ax, low, high)


def selection_title(row: pd.Series, array: np.ndarray) -> str:
    size_x = float(row["scan_size_x_um"])
    size_y = float(row["scan_size_y_um"])
    return (
        f"{row['display_sample_id']} | Ground truth\n"
        f"{row['ground_truth_afm_file_id']}\n"
        f"Rq {rq_nm(array):.3f} nm | {size_x:g}×{size_y:g} µm"
    )


def prediction_title(row: pd.Series, array: np.ndarray) -> str:
    raw = float(row["raw_leave_one_out_predicted_rq_nm"])
    rendered = rq_nm(array)
    raw_note = f"raw {raw:.3f} nm → " if raw < 0 else ""
    return (
        f"HOO AFM prediction | source {row['display_source_sample_id']}\n"
        f"{row['source_afm_file_id']}\n"
        f"{raw_note}Rq {rendered:.3f} nm | "
        f"{row['source_scan_size_x_um']:g}×{row['source_scan_size_y_um']:g} µm"
    )


def role_color(role: str) -> str:
    return {
        "historical": "#2F5D8A",
        "original_added_training": "#3B873E",
        "original_prospective_test": "#C76810",
    }[role]


def render_atlas(results: pd.DataFrame, selections: pd.DataFrame) -> None:
    selection_index = selections.set_index("sample_id")
    result_index = results.set_index("sample_id")
    fig = plt.figure(figsize=(23.0, 20.5), facecolor="white")
    grid = fig.add_gridspec(7, 8, wspace=0.62, hspace=0.58)
    for index, sample_id in enumerate(SAMPLE_IDS):
        row_index = index // 4
        pair_index = index % 4
        gt_ax = fig.add_subplot(grid[row_index, pair_index * 2])
        pred_ax = fig.add_subplot(grid[row_index, pair_index * 2 + 1])
        selected = selection_index.loc[sample_id]
        result = result_index.loc[sample_id]
        ground_truth = np.load(
            REPO / selected["ground_truth_afm_path"], allow_pickle=False
        )
        predicted = np.load(
            REPO / result["retrieved_q50_map_path"], allow_pickle=False
        )
        color = role_color(str(result["sample_role"]))
        show_afm(gt_ax, ground_truth, selection_title(selected, ground_truth), color)
        show_afm(pred_ax, predicted, prediction_title(result, predicted), color)
        for spine in [*gt_ax.spines.values(), *pred_ax.spines.values()]:
            spine.set_edgecolor(color)
            spine.set_linewidth(0.8)
    fig.suptitle(
        "Held-one-out AFM prediction atlas — each Rq model and AFM bank exclude the target sample",
        fontsize=17,
        y=0.995,
    )
    fig.text(
        0.5,
        0.006,
        "Blue: historical | Green: original added training | Orange: original prospective test. "
        "Every AFM panel shows its physical height range and displayed-map Rq.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ATLAS_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ATLAS_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(ATLAS_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(ATLAS_STEM.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def render_per_sample_pairs(
    results: pd.DataFrame, selections: pd.DataFrame
) -> None:
    PER_SAMPLE_FIGURES.mkdir(parents=True, exist_ok=True)
    selection_index = selections.set_index("sample_id")
    result_index = results.set_index("sample_id")
    for sample_id in SAMPLE_IDS:
        selected = selection_index.loc[sample_id]
        result = result_index.loc[sample_id]
        ground_truth = np.load(
            REPO / selected["ground_truth_afm_path"], allow_pickle=False
        )
        predicted = np.load(
            REPO / result["retrieved_q50_map_path"], allow_pickle=False
        )
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
        color = role_color(str(result["sample_role"]))
        show_afm(axes[0], ground_truth, selection_title(selected, ground_truth), color)
        show_afm(axes[1], predicted, prediction_title(result, predicted), color)
        fig.suptitle(
            f"{display_id(sample_id)} held-one-out AFM prediction",
            fontsize=11,
            y=1.01,
        )
        fig.subplots_adjust(wspace=0.38)
        fig.savefig(
            PER_SAMPLE_FIGURES
            / f"{display_id(sample_id)}_ground_truth_vs_held_one_out_prediction.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def calculate_metrics(results: pd.DataFrame) -> dict[str, Any]:
    selected_true = results["ground_truth_selected_rq_nm"].to_numpy(float)
    rendered = results["rendered_physical_rq_nm"].to_numpy(float)
    t4_true = results["ground_truth_T4_rq_nm"].to_numpy(float)
    raw = results["raw_leave_one_out_predicted_rq_nm"].to_numpy(float)
    metrics = {
        "created_at": now(),
        "sample_count": len(results),
        "evaluation_design": "28-fold held-one-out quantitative fit and AFM retrieval bank; 27 other sample groups per fold",
        "selected_display_ground_truth_rq": {
            "MAE_nm": float(mean_absolute_error(selected_true, rendered)),
            "RMSE_nm": float(np.sqrt(mean_squared_error(selected_true, rendered))),
            "R2": float(r2_score(selected_true, rendered)),
        },
        "T4_target_vs_raw_leave_one_out_rq": {
            "MAE_nm": float(mean_absolute_error(t4_true, raw)),
            "RMSE_nm": float(np.sqrt(mean_squared_error(t4_true, raw))),
            "R2": float(r2_score(t4_true, raw)),
        },
        "negative_raw_prediction_count": int(np.count_nonzero(raw < 0)),
        "negative_raw_predictions_rendered_at_rq_nm": 1e-3,
        "visual_method": "unchanged A3_full_cohort_rq_conditioned",
        "morphology_source_is_always_another_sample": bool(
            (results["sample_id"] != results["source_sample_id"]).all()
        ),
    }
    write_json(OUTPUT / "metrics.json", metrics)
    return metrics


def write_report(
    results: pd.DataFrame, selections: pd.DataFrame, metrics: dict[str, Any]
) -> None:
    result_index = results.set_index("sample_id")
    selection_index = selections.set_index("sample_id")
    display_metrics = metrics["selected_display_ground_truth_rq"]
    lines = [
        "# Held-one-out AFM prediction report",
        "",
        "For each of all 28 labeled samples, the quantitative ensemble is trained on the other 27 samples and the A3 AFM retrieval bank excludes every AFM from the held-out sample. The representative ground-truth AFM is selected only after prediction and never enters its own fold.",
        "",
        "N6342 uses ground truth 5, N6389 uses ground truth 3, and N6390 uses ground truth 1 as explicitly requested. Every other sample uses the quality-passing AFM whose measured Rq is closest to that sample's T4 target.",
        "",
        f"Against the displayed representative AFMs, rendered-map Rq MAE is {display_metrics['MAE_nm']:.4f} nm and RMSE is {display_metrics['RMSE_nm']:.4f} nm.",
        "",
        "## Per-sample selection and prediction",
        "",
        "| Sample | GT selection | GT file | GT Rq | Raw LOO Rq | Rendered Rq | Retrieved source |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for sample_id in SAMPLE_IDS:
        result = result_index.loc[sample_id]
        selected = selection_index.loc[sample_id]
        lines.append(
            f"| {display_id(sample_id)} | {selected['selection_method']} | "
            f"{selected['ground_truth_afm_file_id']} | "
            f"{selected['ground_truth_rq_nm']:.4f} | "
            f"{result['raw_leave_one_out_predicted_rq_nm']:.4f} | "
            f"{result['rendered_physical_rq_nm']:.4f} | "
            f"{result['display_source_sample_id']} / {result['source_afm_file_id']} |"
        )
    lines += [
        "",
        "## Primary atlas",
        "",
        "- `figures/main/Figure3_held_one_out_afm_prediction_atlas.png`",
        "- `figures/main/Figure3_held_one_out_afm_prediction_atlas.pdf`",
        "- `figures/main/Figure3_held_one_out_afm_prediction_atlas.svg`",
        "",
        "All ground-truth and predicted AFM panels include a physical height bar in nm and the Rq of the displayed array.",
    ]
    path = PKG / "report/held_one_out_afm_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_provenance(
    bank: pd.DataFrame,
    results: pd.DataFrame,
    selections: pd.DataFrame,
    retrieval: Any,
) -> None:
    algorithm_path = (
        PROSPECTIVE_SOURCE / "code/generate_full_cohort_retrieval_images.py"
    )
    provenance = {
        "created_at": now(),
        "experiment": "28-fold held-one-out AFM prediction",
        "sample_ids": SAMPLE_IDS,
        "fold_count": 28,
        "quantitative_training_sample_count_per_fold": 27,
        "afm_source_group_count_per_fold": 27,
        "visual_method": "A3_full_cohort_rq_conditioned",
        "visual_algorithm_source": rel(algorithm_path),
        "visual_algorithm_source_sha256": sha256_file(algorithm_path),
        "visual_algorithm_modified": False,
        "full_labeled_bank_rows": len(bank),
        "full_labeled_bank_groups": int(bank["sample_id"].nunique()),
        "held_out_absent_from_all_fold_banks": bool(
            results["held_out_sample_absent_from_afm_bank"].all()
        ),
        "forced_ground_truth_scans": FORCED_GROUND_TRUTH_SCANS,
        "default_ground_truth_selection": "minimum absolute measured-Rq distance to sample T4 target among quality-passing AFMs",
        "ground_truth_selection_is_post_prediction_display_only": True,
        "rank_descriptor_columns": retrieval.DESCRIPTOR_COLS,
        "outputs": {
            "full_bank_manifest": rel(BANK_OUTPUT),
            "ground_truth_selection": rel(OUTPUT / "ground_truth_selection.csv"),
            "retrieval_results": rel(OUTPUT / "retrieval_results.csv"),
            "metrics": rel(OUTPUT / "metrics.json"),
            "atlas_png": rel(ATLAS_STEM.with_suffix(".png")),
            "atlas_pdf": rel(ATLAS_STEM.with_suffix(".pdf")),
            "atlas_svg": rel(ATLAS_STEM.with_suffix(".svg")),
            "per_sample_figure_count": len(selections),
        },
    }
    write_json(OUTPUT / "run_provenance.json", provenance)


def update_readme() -> None:
    readme = PKG / "README.md"
    text = readme.read_text(encoding="utf-8")
    marker = "\n## Held-one-out AFM atlas"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    section = [
        "",
        "## Held-one-out AFM atlas",
        "",
        "The 28-fold AFM experiment uses the existing leave-one-out Rq prediction for each sample, excludes that sample's entire AFM group from the A3 retrieval bank, and retrieves morphology from the other 27 sample groups. Its primary atlas is `figures/main/Figure3_held_one_out_afm_prediction_atlas.*`; full results are in `predictions/held_one_out_afm_28/` and `report/held_one_out_afm_summary.md`.",
    ]
    readme.write_text(text.rstrip() + "\n" + "\n".join(section) + "\n", encoding="utf-8")


def update_master_result_report(metrics: dict[str, Any]) -> None:
    report = PKG / "report/result_summary.md"
    text = report.read_text(encoding="utf-8")
    marker = "\n## Held-one-out AFM prediction atlas"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n"
    display_metrics = metrics["selected_display_ground_truth_rq"]
    section = [
        "",
        "## Held-one-out AFM prediction atlas",
        "",
        "A separate 28-fold visual experiment excludes the target sample from both the 27-sample quantitative fit and the 27-group A3 AFM retrieval bank. N6342/N6389/N6390 use ground truth 5/3/1, respectively; all other representative AFMs minimize absolute measured-Rq distance to the sample T4 target.",
        "",
        f"Rendered AFM Rq versus the selected displayed ground truths: MAE {display_metrics['MAE_nm']:.4f} nm, RMSE {display_metrics['RMSE_nm']:.4f} nm.",
        "",
        "Primary atlas: `figures/main/Figure3_held_one_out_afm_prediction_atlas.*`. Full selections and retrieved sources are documented in `report/held_one_out_afm_summary.md`.",
    ]
    report.write_text(text.rstrip() + "\n" + "\n".join(section) + "\n", encoding="utf-8")


def refresh_manifest() -> None:
    manifest = PKG / "provenance/MANIFEST.sha256"
    rows = []
    for path in sorted(PKG.rglob("*")):
        if path.is_file() and path != manifest:
            rows.append(f"{sha256_file(path)}  {path.relative_to(PKG)}")
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    predictions = pd.read_csv(
        PKG / "predictions/leave_one_out_28/predictions.csv",
        dtype={"sample_id": str},
    )
    if predictions["sample_id"].tolist() != SAMPLE_IDS:
        raise RuntimeError("The 28-sample leave-one-out predictions are missing or reordered")
    retrieval = load_retrieval_module()
    bank = build_full_labeled_bank()
    selections = select_ground_truth_maps(bank, predictions)
    results = run_held_one_out_retrieval(bank, predictions, selections, retrieval)
    metrics = calculate_metrics(results)
    render_atlas(results, selections)
    render_per_sample_pairs(results, selections)
    write_report(results, selections, metrics)
    write_provenance(bank, results, selections, retrieval)
    update_readme()
    update_master_result_report(metrics)
    refresh_manifest()
    print(
        json.dumps(
            {
                "status": "ok",
                "sample_count": len(results),
                "source_groups_per_fold": 27,
                "atlas": str(ATLAS_STEM.with_suffix(".png")),
                "selected_ground_truth_rq_metrics": metrics[
                    "selected_display_ground_truth_rq"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
