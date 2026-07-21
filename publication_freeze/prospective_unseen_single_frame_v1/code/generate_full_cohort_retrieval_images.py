#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DESCRIPTOR_COLS = [
    "rq_nm",
    "ra_nm",
    "robust_height_range_nm",
    "psd_low_fraction",
    "psd_mid_fraction",
    "psd_high_fraction",
    "psd_slope",
    "correlation_length_nm",
    "anisotropy",
    "height_skewness",
    "height_kurtosis",
]
DESCRIPTOR_ALIASES = {
    "correlation_length_nm": ["physical_autocorr_length_nm"],
    "anisotropy": ["physical_anisotropy_ratio"],
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def repo_root_from_here() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "publication_freeze" / "rheed_afm_single_frame_v1_2026-07-18").exists():
            return parent
    raise RuntimeError(f"Could not locate repository root from {here}")


def rq_np(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=np.float64)
    a = a - np.nanmean(a)
    return float(np.sqrt(np.nanmean(a**2)))


def project_unit_rq(arr: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    a = a - float(np.nanmean(a))
    q = float(np.sqrt(np.nanmean(a**2)))
    return (a / (q + epsilon)).astype(np.float32)


def physical_from_q(unit: np.ndarray, q_nm: float) -> np.ndarray:
    return float(q_nm) * project_unit_rq(unit)


def read_map(repo: Path, path_value: str) -> np.ndarray:
    path = Path(path_value)
    if not path.is_absolute():
        path = repo / path
    arr = np.load(path, allow_pickle=False).astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.all():
        fill = float(np.nanmean(arr[finite])) if finite.any() else 0.0
        arr = np.where(finite, arr, fill).astype(np.float32)
    return arr


def normalize_image(arr: np.ndarray) -> np.ndarray:
    data = np.squeeze(np.asarray(arr, dtype=float))
    lo, hi = np.nanpercentile(data, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(data.shape, dtype=np.uint8)
    return (np.clip((data - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def rank_bank(bank: pd.DataFrame, predicted_rq_nm: float) -> pd.DataFrame:
    candidates = bank.copy()
    candidates = candidates[candidates["quality_pass"].astype(str).eq("True")].copy()
    descriptor_data = pd.DataFrame(index=candidates.index)
    for col in DESCRIPTOR_COLS:
        possible = [col, f"physical_{col}", *DESCRIPTOR_ALIASES.get(col, [])]
        source_col = next((name for name in possible if name in candidates.columns), None)
        if source_col is None:
            raise KeyError(f"Missing descriptor column {col} / physical_{col}")
        descriptor_data[col] = pd.to_numeric(candidates[source_col], errors="coerce")
    descriptor_medians = {
        col: float(descriptor_data[col].median()) for col in DESCRIPTOR_COLS
    }
    cond = np.asarray([descriptor_medians[col] for col in DESCRIPTOR_COLS], dtype=float)
    cond[DESCRIPTOR_COLS.index("rq_nm")] = float(predicted_rq_nm)
    mat = descriptor_data.to_numpy(float)
    col_medians = np.nanmedian(mat, axis=0)
    inds = np.where(~np.isfinite(mat))
    if inds[0].size:
        mat[inds] = np.take(col_medians, inds[1])
    scale = np.maximum(np.nanstd(mat, axis=0), 1e-6)
    candidates["rank_score"] = np.sqrt((((mat - cond) / scale) ** 2).sum(axis=1))
    candidates["rank_score"] += 0.05 * (
        pd.to_numeric(candidates["rq_nm"], errors="coerce") - float(predicted_rq_nm)
    ).abs()
    return candidates.sort_values(["rank_score", "sample_id", "afm_file_id"]).reset_index(drop=True)


def top_distinct_groups(ranked: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    for _, row in ranked.iterrows():
        sid = str(row["sample_id"])
        if sid in seen:
            continue
        rows.append(row)
        seen.add(sid)
        if len(rows) == n:
            break
    return pd.DataFrame(rows)


def save_single_map(arr: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    lo, hi = np.nanpercentile(arr, [1, 99])
    im = ax.imshow(arr, cmap="viridis", vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("nm", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def add_heightbar(ax: plt.Axes, arr: np.ndarray) -> None:
    lo, hi = np.nanpercentile(arr, [1, 99])
    cax = ax.inset_axes([1.02, 0.08, 0.045, 0.84])
    cax.imshow(np.linspace(hi, lo, 128)[:, None], cmap="viridis", aspect="auto")
    cax.set_xticks([])
    cax.set_yticks([0, 127])
    cax.set_yticklabels([f"{hi:.1f}", f"{lo:.1f}"], fontsize=5)
    cax.set_title("nm", fontsize=5, pad=1)


def build_atlas(rows: pd.DataFrame, output_path: Path) -> None:
    n = len(rows)
    fig = plt.figure(figsize=(10.8, max(2.2, n * 2.0)))
    grid = fig.add_gridspec(n, 3, wspace=0.16, hspace=0.48)
    for idx, row in rows.reset_index(drop=True).iterrows():
        axes = [fig.add_subplot(grid[idx, col]) for col in range(3)]
        rheed = plt.imread(row["roi_keyframe_png"])
        retrieved = np.load(row["retrieved_q50_map_path"], allow_pickle=False)
        source = np.load(row["source_unit_shape_map_path"], allow_pickle=False)
        axes[0].imshow(rheed, cmap="gray")
        axes[0].set_title(f"{row.sample_id} ROI RHEED", fontsize=7, pad=2)
        axes[1].imshow(source, cmap="viridis")
        axes[1].set_title(f"source {row.source_sample_id}\nunit morphology", fontsize=7, pad=2)
        add_heightbar(axes[1], source)
        axes[2].imshow(retrieved, cmap="viridis")
        axes[2].set_title(f"retrieved q50\nRq {row.retrieval_q50_rq_nm:.2f} nm", fontsize=7, pad=2)
        add_heightbar(axes[2], retrieved)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Full-cohort A3-style retrieval for five unseen samples", fontsize=11, y=0.995)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=210, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo = repo_root_from_here()
    freeze = repo / "publication_freeze" / "rheed_afm_single_frame_v1_2026-07-18"
    pkg = repo / "publication_freeze" / "prospective_unseen_single_frame_v1"
    pred_path = pkg / "predictions" / "full_cohort_single_frame_v1" / "predictions.csv"
    bank_path = freeze / "models" / "visual_model" / "full_cohort_bank" / "afm_bank_manifest.csv"
    out = pkg / "predictions" / "full_cohort_single_frame_v1" / "retrieval"
    maps_dir = out / "retrieved_maps_q50"
    rendered_dir = out / "rendered_maps_q50"
    unit_dir = out / "source_unit_shape_maps"
    for d in [maps_dir, rendered_dir, unit_dir]:
        d.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(pred_path)
    bank = pd.read_csv(bank_path, dtype={"sample_id": str})
    source_group_count = int(bank["sample_id"].astype(str).nunique())
    rows: list[dict[str, object]] = []
    for pred in predictions.to_dict("records"):
        sample_id = str(pred["sample_id"])
        raw_rq = float(pred["predicted_rq_nm"])
        q50 = max(float(pred["predicted_rq_nm_clipped_nonnegative"]), 1e-3)
        ranked = rank_bank(bank, q50)
        top5 = top_distinct_groups(ranked, 5)
        source = ranked.iloc[0]
        source_map = read_map(repo, str(source["second_order_afm_path"]))
        unit = project_unit_rq(source_map)
        retrieved = physical_from_q(unit, q50)
        unit_path = unit_dir / f"{sample_id}_source_{source['sample_id']}_{source['afm_file_id']}_unit_shape.npy"
        map_path = maps_dir / f"{sample_id}_A3_full_cohort_q50_retrieved.npy"
        png_path = rendered_dir / f"{sample_id}_A3_full_cohort_q50_retrieved.png"
        np.save(unit_path, unit.astype(np.float32))
        np.save(map_path, retrieved.astype(np.float32))
        save_single_map(
            retrieved,
            png_path,
            f"{sample_id} full-cohort retrieval\nRq {q50:.2f} nm",
        )
        rows.append(
            {
                "sample_id": sample_id,
                "method_id": "A3_full_cohort_rq_conditioned",
                "raw_predicted_rq_nm": raw_rq,
                "retrieval_q50_rq_nm": q50,
                "negative_raw_prediction_clipped_for_physical_map": raw_rq < 0,
                "source_sample_id": str(source["sample_id"]),
                "source_afm_file_id": str(source["afm_file_id"]),
                "source_afm_path": str(source["second_order_afm_path"]),
                "source_rank_score": float(source["rank_score"]),
                "source_rq_nm": float(source["rq_nm"]),
                "top5_source_sample_ids": json.dumps(top5["sample_id"].astype(str).tolist()),
                "top5_source_afm_file_ids": json.dumps(top5["afm_file_id"].astype(str).tolist()),
                "source_group_count": source_group_count,
                "uses_all_23_source_groups": source_group_count == 23,
                "afm_labels_used_for_unseen": False,
                "non_rq_condition_descriptor_fill": "full_cohort_bank_median",
                "rq_condition_source": "full_cohort_single_frame_v1 predictions.csv predicted_rq_nm_clipped_nonnegative",
                "roi_keyframe_png": str(pred["roi_keyframe_png"]),
                "retrieved_q50_map_path": str(map_path),
                "rendered_q50_png_path": str(png_path),
                "source_unit_shape_map_path": str(unit_path),
            }
        )

    result = pd.DataFrame(rows)
    table_path = out / "retrieval_results.csv"
    result.to_csv(table_path, index=False)
    build_atlas(result, out / "unseen_full_cohort_a3_retrieval_atlas")
    provenance = {
        "created_at": now(),
        "method_id": "A3_full_cohort_rq_conditioned",
        "source_group_count": source_group_count,
        "uses_all_23_source_groups": source_group_count == 23,
        "afm_labels_used_for_unseen": False,
        "bank_path": str(bank_path),
        "predictions_path": str(pred_path),
        "descriptor_columns": DESCRIPTOR_COLS,
        "non_rq_condition_descriptor_fill": "full_cohort_bank_median",
        "negative_raw_predictions_are_clipped_to_1e_3_nm_for_physical_map_rendering": True,
        "outputs": {
            "retrieval_results_csv": str(table_path),
            "atlas_png": str(out / "unseen_full_cohort_a3_retrieval_atlas.png"),
            "atlas_pdf": str(out / "unseen_full_cohort_a3_retrieval_atlas.pdf"),
            "rendered_maps_q50_dir": str(rendered_dir),
            "retrieved_maps_q50_dir": str(maps_dir),
        },
    }
    (out / "retrieval_run_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "retrieval_results": str(table_path), "atlas": provenance["outputs"]["atlas_png"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
