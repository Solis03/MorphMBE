#!/usr/bin/env python3
"""Build a one-page HOO atlas comparing legacy NN retrieval with M17b generation.

The legacy publication freeze used the name Rq for areal RMS height.  This
script uses the metrologically correct Sq label throughout, while preserving
the frozen numeric values and source identities exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image


LEGACY_METHOD = "A3_full_cohort_rq_conditioned_held_one_out"
CURRENT_METHOD = "M17b_topology_sparse_peak_terrace"
LEGACY_COMMIT = "6973e08753a6008a2b7480b7400efdf29d0b1469"
CURRENT_RESULT_COMMIT = "6a42ab2"
CURRENT_PACKAGE_COMMIT = "99bb75b3ed22d385367eb6622f7f05ddbc6a754e"


def display_id(value: Any) -> str:
    """Normalize IDs while preserving the N prefix used by the extra five."""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def project_unit_sq(array: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Reproduce the legacy unit-RMS projection used to render retrieval maps."""
    centered = np.asarray(array, dtype=np.float32)
    centered = centered - float(np.nanmean(centered))
    sq = float(np.sqrt(np.nanmean(np.square(centered))))
    if not np.isfinite(sq) or sq <= 0:
        raise ValueError(f"Cannot project array with invalid Sq={sq}")
    return centered / (sq + epsilon)


def measured_sq(array: np.ndarray) -> float:
    centered = np.asarray(array, dtype=np.float64)
    centered = centered - float(np.nanmean(centered))
    return float(np.sqrt(np.nanmean(np.square(centered))))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_under(root: Path, relative: str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else root / path


def resolve_existing(relative: str, *roots: Path) -> Path:
    path = Path(relative)
    candidates = [path] if path.is_absolute() else [root / path for root in roots]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve {relative}; tried {candidates}")


def scalar(npz: np.lib.npyio.NpzFile, key: str) -> Any:
    return np.asarray(npz[key]).item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument(
        "--legacy-data-root",
        type=Path,
        required=True,
        help="Repository root containing the frozen legacy source AFM arrays.",
    )
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-provenance", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all inputs and outputs without drawing the PDF.",
    )
    return parser.parse_args()


def input_paths(package_root: Path) -> dict[str, Path]:
    legacy_freeze = (
        package_root
        / "publication_freeze/prospective_retrained_6358_6382_single_frame_v1"
    )
    current_tag = "20260804_m17_sparse_topology_line3_full27_v1/full27_loo"
    return {
        "legacy_csv": legacy_freeze
        / "predictions/held_one_out_afm_28/retrieval_results.csv",
        "legacy_provenance": legacy_freeze
        / "predictions/held_one_out_afm_28/run_provenance.json",
        "legacy_atlas": legacy_freeze
        / "figures/main/Figure3_held_one_out_afm_prediction_atlas.pdf",
        "current_csv": package_root
        / f"reports/rheed_m17_end_to_end_generation/{current_tag}/rq_crossfit_predictions.csv",
        "current_manifest": package_root
        / f"reports/rheed_m17_end_to_end_generation/{current_tag}/best_model_manifest.json",
        "current_map_root": package_root
        / f"outputs/rheed_m17_end_to_end_generation/{current_tag}/crossfit/generated_maps/{CURRENT_METHOD}",
        "phase1_manifest": package_root
        / (
            "outputs/extra_five_integration/"
            "20260729_line3_full28_orientation90_keyframe_locked_v3/"
            "machine_dataset_full28/modeling_manifest.csv"
        ),
    }


def validate_static_inputs(paths: dict[str, Path]) -> None:
    for label, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    with paths["legacy_provenance"].open(encoding="utf-8") as handle:
        legacy_provenance = json.load(handle)
    if legacy_provenance.get("visual_method") != "A3_full_cohort_rq_conditioned":
        raise AssertionError("Legacy provenance does not identify the frozen A3 visual method")
    if not bool(legacy_provenance.get("held_out_absent_from_all_fold_banks")):
        raise AssertionError("Legacy provenance does not certify held-out AFM-bank exclusion")
    with paths["current_manifest"].open(encoding="utf-8") as handle:
        current_manifest = json.load(handle)
    current_text = json.dumps(current_manifest, sort_keys=True)
    if CURRENT_METHOD not in current_text:
        raise AssertionError(f"Current manifest does not identify {CURRENT_METHOD}")


def load_rows(
    package_root: Path, legacy_data_root: Path, paths: dict[str, Path]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    legacy = pd.read_csv(paths["legacy_csv"], dtype={"display_sample_id": str})
    current = pd.read_csv(paths["current_csv"], dtype={"growth_run_id": str})
    phase1 = pd.read_csv(paths["phase1_manifest"], dtype={"growth_run_id": str})

    legacy["target_id"] = legacy["display_sample_id"].map(display_id)
    current["target_id"] = current["growth_run_id"].map(display_id)
    phase1["target_id"] = phase1["growth_run_id"].map(display_id)

    if len(legacy) != 28 or legacy["target_id"].nunique() != 28:
        raise AssertionError("Legacy freeze must contain 28 unique HOO targets")
    if len(current) != 27 or current["target_id"].nunique() != 27:
        raise AssertionError("Current full27 table must contain 27 unique HOO targets")
    if set(legacy["method_id"]) != {LEGACY_METHOD}:
        raise AssertionError("Unexpected legacy method ID")
    if not legacy["held_out_sample_absent_from_afm_bank"].astype(bool).all():
        raise AssertionError("At least one legacy target is present in its AFM bank")
    if current["outer_target_used_for_training"].astype(bool).any():
        raise AssertionError("At least one current HOO target was used for training")
    if set(current["outer_fit_growth_count"].astype(int)) != {26}:
        raise AssertionError("Current full27 folds must each fit on 26 growth groups")
    current_sq_methods = sorted(set(current["method"].astype(str)))
    if current_sq_methods != ["M16_endpoint_streak_dual_resolution"]:
        raise AssertionError(f"Unexpected current Sq predictor: {current_sq_methods}")

    intersection = sorted(
        set(legacy["target_id"]) & set(current["target_id"]),
        key=lambda value: int(value.removeprefix("N")),
    )
    if len(intersection) != 27:
        raise AssertionError(f"Expected a 27-target intersection, got {len(intersection)}")
    if "6081" in intersection:
        raise AssertionError("Audited exclusion 6081 unexpectedly entered the intersection")
    if set(legacy["target_id"]) - set(intersection) != {"6081"}:
        raise AssertionError("The only legacy-only target must be 6081")

    legacy = legacy.set_index("target_id", drop=False)
    current = current.set_index("target_id", drop=False)
    phase1 = phase1.set_index("target_id", drop=False)
    rows: list[dict[str, Any]] = []
    max_old_sq_error = 0.0
    max_current_sq_error = 0.0
    max_target_manifest_sq_error = 0.0

    for target_id in intersection:
        old = legacy.loc[target_id]
        new = current.loc[target_id]
        rheed = phase1.loc[target_id]
        if isinstance(rheed, pd.DataFrame):
            raise AssertionError(f"Duplicate phase-1 rows for {target_id}")
        target_manifest_sq = float(rheed["primary_rq_nm_median"])
        target_table_sq = float(new["true_target"])
        target_manifest_sq_error = abs(target_manifest_sq - target_table_sq)
        max_target_manifest_sq_error = max(
            max_target_manifest_sq_error, target_manifest_sq_error
        )
        if target_manifest_sq_error > 1e-10:
            raise AssertionError(
                f"Current target Sq manifest/table mismatch for {target_id}: "
                f"{target_manifest_sq} vs {target_table_sq}"
            )

        measured_afm_path = resolve_existing(
            str(rheed["representative_afm_height_array"]),
            package_root,
            legacy_data_root,
        )
        measured_afm = np.load(measured_afm_path, allow_pickle=False).astype(np.float32)
        measured_afm = measured_afm - float(np.nanmean(measured_afm))
        measured_afm_sq = measured_sq(measured_afm)
        if not np.isfinite(measured_afm_sq) or measured_afm_sq <= 0:
            raise AssertionError(f"Invalid measured AFM Sq for {target_id}: {measured_afm_sq}")

        source_target_id = display_id(old["display_source_sample_id"])
        if source_target_id == target_id:
            raise AssertionError(f"Legacy target {target_id} retrieves itself")
        fold_sources = {display_id(x) for x in json.loads(old["fold_source_group_ids"])}
        target_bank_id = target_id.removeprefix("N")
        if target_id in fold_sources or target_bank_id in fold_sources:
            raise AssertionError(f"Legacy target {target_id} appears in its fold bank")

        source_afm_path = resolve_under(legacy_data_root, str(old["source_afm_path"]))
        if not source_afm_path.exists():
            raise FileNotFoundError(source_afm_path)
        source_afm = np.load(source_afm_path, allow_pickle=False)
        old_output_sq = float(old["rendered_physical_rq_nm"])
        old_raw_sq = float(old["raw_leave_one_out_predicted_rq_nm"])
        old_output_was_clipped = bool(old["negative_raw_prediction_clipped_to_1e_3_for_map"])
        if old_output_was_clipped != (old_raw_sq < 0 and np.isclose(old_output_sq, 1e-3)):
            raise AssertionError(f"Inconsistent legacy clipping metadata for {target_id}")
        legacy_map = project_unit_sq(source_afm) * old_output_sq
        old_sq_check = measured_sq(legacy_map)
        old_sq_error = abs(old_sq_check - old_output_sq)
        max_old_sq_error = max(max_old_sq_error, old_sq_error)
        if old_sq_error > 2e-5:
            raise AssertionError(
                f"Legacy rendered Sq mismatch for {target_id}: {old_sq_check} vs {old_output_sq}"
            )

        current_npz_path = paths["current_map_root"] / f"{target_id}.npz"
        if not current_npz_path.exists():
            raise FileNotFoundError(current_npz_path)
        with np.load(current_npz_path, allow_pickle=False) as npz:
            if display_id(scalar(npz, "growth_run_id")) != target_id:
                raise AssertionError(f"Current NPZ target mismatch for {target_id}")
            if str(scalar(npz, "method")) != CURRENT_METHOD:
                raise AssertionError(f"Current NPZ method mismatch for {target_id}")
            if bool(scalar(npz, "retrieval_at_inference")):
                raise AssertionError(f"Current map retrieves AFM at inference for {target_id}")
            if bool(scalar(npz, "measured_afm_patch_used_at_inference")):
                raise AssertionError(f"Current map uses measured AFM for {target_id}")
            predicted_sq = float(scalar(npz, "predicted_rq_nm"))
            if not np.isclose(predicted_sq, float(new["predicted_target"]), atol=1e-10):
                raise AssertionError(f"Current predicted Sq table/NPZ mismatch for {target_id}")
            generated_unit_shapes = np.asarray(npz["generated_unit_shapes"], dtype=np.float32)
        if generated_unit_shapes.shape != (4, 128, 128):
            raise AssertionError(
                f"Unexpected current generated shape for {target_id}: {generated_unit_shapes.shape}"
            )
        current_map = generated_unit_shapes[0] * predicted_sq
        current_sq_check = measured_sq(current_map)
        current_sq_error = abs(current_sq_check - predicted_sq)
        max_current_sq_error = max(max_current_sq_error, current_sq_error)
        if current_sq_error > 2e-5:
            raise AssertionError(
                f"Current rendered Sq mismatch for {target_id}: {current_sq_check} vs {predicted_sq}"
            )

        frame_paths = json.loads(rheed["clip_frame_paths"])
        offset = int(rheed["keyframe_offset_in_clip_x"])
        if frame_paths:
            if not 0 <= offset < len(frame_paths):
                raise AssertionError(f"Invalid keyframe offset for {target_id}")
            keyframe_path = resolve_under(package_root, frame_paths[offset])
            if not keyframe_path.exists():
                raise FileNotFoundError(keyframe_path)
            keyframe = np.asarray(Image.open(keyframe_path).convert("L"))
            keyframe_view = "raw frame; cyan box = model ROI"
            draw_roi = True
        else:
            keyframe_path = resolve_under(package_root, str(rheed["clip_cache_path"]))
            if not keyframe_path.exists():
                raise FileNotFoundError(keyframe_path)
            with np.load(keyframe_path, allow_pickle=False) as clip:
                frames = np.asarray(clip["frames_uint8"], dtype=np.uint8)
                frame_indices = np.asarray(clip["frame_indices"], dtype=int)
            if not 0 <= offset < len(frames):
                raise AssertionError(f"Invalid cached keyframe offset for {target_id}")
            if int(frame_indices[offset]) != int(rheed["keyframe_index"]):
                raise AssertionError(f"Cached keyframe index mismatch for {target_id}")
            keyframe = frames[offset]
            keyframe_view = "frozen 224 x 224 model crop"
            draw_roi = False

        legacy_height_low, legacy_height_high = np.nanpercentile(legacy_map, [1.0, 99.0])
        current_height_low, current_height_high = np.nanpercentile(current_map, [1.0, 99.0])
        measured_height_low, measured_height_high = np.nanpercentile(
            measured_afm, [1.0, 99.0]
        )
        rows.append(
            {
                "target_id": target_id,
                "measured_sq_nm": target_table_sq,
                "measured_afm_scan_id": str(rheed["representative_afm_scan_id"]),
                "measured_afm_path": str(measured_afm_path),
                "measured_afm_sha256": sha256(measured_afm_path),
                "measured_afm_scan_sq_nm": measured_afm_sq,
                "measured_afm_shape": f"{measured_afm.shape[0]}x{measured_afm.shape[1]}",
                "measured_afm": measured_afm,
                "keyframe_index": int(rheed["keyframe_index"]),
                "keyframe_path": str(keyframe_path),
                "keyframe_sha256": sha256(keyframe_path),
                "keyframe_view": keyframe_view,
                "draw_roi": draw_roi,
                "roi_x": int(rheed["roi_x"]),
                "roi_y": int(rheed["roi_y"]),
                "roi_width": int(rheed["roi_width"]),
                "roi_height": int(rheed["roi_height"]),
                "keyframe": keyframe,
                "legacy_output_sq_nm": old_output_sq,
                "legacy_raw_predicted_sq_nm": old_raw_sq,
                "legacy_output_was_clipped": old_output_was_clipped,
                "legacy_source_sample_id": source_target_id,
                "legacy_source_afm_file_id": str(old["source_afm_file_id"]),
                "legacy_source_sq_nm": float(old["source_rq_nm"]),
                "legacy_source_afm_path": str(source_afm_path),
                "legacy_source_afm_sha256": sha256(source_afm_path),
                "legacy_map": legacy_map,
                "current_predicted_sq_nm": predicted_sq,
                "current_npz_path": str(current_npz_path),
                "current_npz_sha256": sha256(current_npz_path),
                "current_map": current_map,
                "legacy_height_scale_low_nm": float(legacy_height_low),
                "legacy_height_scale_high_nm": float(legacy_height_high),
                "current_height_scale_low_nm": float(current_height_low),
                "current_height_scale_high_nm": float(current_height_high),
                "measured_height_scale_low_nm": float(measured_height_low),
                "measured_height_scale_high_nm": float(measured_height_high),
            }
        )

    validation = {
        "legacy_target_count": int(len(legacy)),
        "current_target_count": int(len(current)),
        "intersection_target_count": len(intersection),
        "intersection_target_ids": intersection,
        "legacy_only_target_ids": sorted(set(legacy["target_id"]) - set(intersection)),
        "legacy_all_targets_absent_from_fold_banks": True,
        "current_all_targets_absent_from_outer_training": True,
        "current_all_npz_retrieval_at_inference_false": True,
        "current_all_npz_measured_afm_patch_at_inference_false": True,
        "current_outer_fit_growth_count": 26,
        "current_sq_predictor_method": current_sq_methods[0],
        "max_legacy_rendered_sq_error_nm": max_old_sq_error,
        "max_current_rendered_sq_error_nm": max_current_sq_error,
        "max_current_target_manifest_sq_error_nm": max_target_manifest_sq_error,
        "all_measured_afm_arrays_present_and_finite": True,
    }
    return rows, validation


def draw_atlas(rows: list[dict[str, Any]], output_pdf: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "pdf.compression": 6,
            "axes.titlesize": 7.2,
            "axes.titleweight": "semibold",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    block_rows = math.ceil(len(rows) / 2)
    fig = plt.figure(figsize=(40, 41.5), constrained_layout=False)
    grid = fig.add_gridspec(
        block_rows,
        9,
        width_ratios=[1.45, 1.0, 1.0, 1.0, 0.18, 1.45, 1.0, 1.0, 1.0],
        left=0.018,
        right=0.987,
        bottom=0.035,
        top=0.955,
        wspace=0.18,
        hspace=0.58,
    )

    fig.suptitle(
        "Held-one-out AFM comparison: legacy NN retrieval vs current M17b generation",
        x=0.5,
        y=0.987,
        fontsize=22,
        fontweight="bold",
        color="#14375a",
    )
    fig.text(
        0.5,
        0.970,
        (
            "Per target: RHEED keyframe | measured AFM | legacy A3 NN retrieval | current M17b generation. "
            "Both prediction routes are held-one-out (intersection n=27)."
        ),
        ha="center",
        va="center",
        fontsize=10.5,
        color="#405267",
    )

    colors = {
        "rheed": "#315e8a",
        "measured": "#6b4c9a",
        "legacy": "#c26d21",
        "current": "#117a7a",
    }
    for index, row in enumerate(rows):
        grid_row = index // 2
        base_col = 0 if index % 2 == 0 else 5
        ax_rheed = fig.add_subplot(grid[grid_row, base_col])
        ax_measured = fig.add_subplot(grid[grid_row, base_col + 1])
        ax_old = fig.add_subplot(grid[grid_row, base_col + 2])
        ax_new = fig.add_subplot(grid[grid_row, base_col + 3])

        frame = row["keyframe"]
        rlow, rhigh = np.nanpercentile(frame, [1, 99.7])
        ax_rheed.imshow(frame, cmap="gray", vmin=rlow, vmax=rhigh, interpolation="nearest")
        if row["draw_roi"]:
            ax_rheed.add_patch(
                Rectangle(
                    (row["roi_x"], row["roi_y"]),
                    row["roi_width"],
                    row["roi_height"],
                    linewidth=1.2,
                    edgecolor="#26d7e6",
                    facecolor="none",
                )
            )
        ax_rheed.set_title(
            (
                f"Target {row['target_id']} | measured Sq {row['measured_sq_nm']:.3f} nm\n"
                f"RHEED keyframe {row['keyframe_index']} ({row['keyframe_view']})"
            ),
            color=colors["rheed"],
            pad=4,
        )

        old_kwargs = {
            "cmap": "viridis",
            "vmin": row["legacy_height_scale_low_nm"],
            "vmax": row["legacy_height_scale_high_nm"],
            "interpolation": "nearest",
        }
        current_kwargs = {
            "cmap": "viridis",
            "vmin": row["current_height_scale_low_nm"],
            "vmax": row["current_height_scale_high_nm"],
            "interpolation": "nearest",
        }
        measured_kwargs = {
            "cmap": "viridis",
            "vmin": row["measured_height_scale_low_nm"],
            "vmax": row["measured_height_scale_high_nm"],
            "interpolation": "nearest",
        }
        ax_measured.imshow(row["measured_afm"], **measured_kwargs)
        ax_measured.set_title(
            (
                f"Measured AFM | scan Sq {row['measured_afm_scan_sq_nm']:.3f} nm\n"
                f"{row['measured_afm_scan_id']}\n"
                f"sample target Sq {row['measured_sq_nm']:.3f} nm"
            ),
            color=colors["measured"],
            pad=4,
        )
        ax_old.imshow(row["legacy_map"], **old_kwargs)
        clipping_label = " [clipped]" if row["legacy_output_was_clipped"] else ""
        ax_old.set_title(
            (
                f"Legacy HOO NN retrieval | output Sq {row['legacy_output_sq_nm']:.3f} nm{clipping_label}\n"
                f"bank source {row['legacy_source_sample_id']} / {row['legacy_source_afm_file_id']}\n"
                f"source Sq {row['legacy_source_sq_nm']:.3f} nm"
            ),
            color=colors["legacy"],
            pad=4,
        )
        ax_new.imshow(row["current_map"], **current_kwargs)
        ax_new.set_title(
            (
                f"Current M17b HOO generation | predicted Sq {row['current_predicted_sq_nm']:.3f} nm\n"
                "retrieval = false\n"
                "no measured AFM patch at inference"
            ),
            color=colors["current"],
            pad=4,
        )

        for axis, color in (
            (ax_rheed, colors["rheed"]),
            (ax_measured, colors["measured"]),
            (ax_old, colors["legacy"]),
            (ax_new, colors["current"]),
        ):
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.1)
                spine.set_edgecolor(color)

    fig.text(
        0.5,
        0.016,
        (
            "Sq = areal RMS height (legacy files used the label Rq). Each AFM panel is independently "
            "contrast-scaled to its 1st-99th percentile; compare height amplitude using the labeled Sq. "
            "Legacy commit 6973e08 | current result 6a42ab2 | desktop package 99bb75b."
        ),
        ha="center",
        va="center",
        fontsize=8.8,
        color="#4a5563",
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_pdf,
        format="pdf",
        dpi=300,
        metadata={
            "Title": "Legacy NN retrieval vs current M17b held-one-out AFM atlas",
            "Author": "MorphMBE reproducible atlas builder",
            "Subject": "27-sample intersection, leakage-audited HOO comparison",
            "Keywords": "RHEED, AFM, Sq, nearest-neighbor retrieval, M17b, held-one-out",
        },
    )
    plt.close(fig)


def write_outputs(
    rows: list[dict[str, Any]],
    validation: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
) -> None:
    manifest_columns = [
        "target_id",
        "measured_sq_nm",
        "measured_afm_scan_id",
        "measured_afm_path",
        "measured_afm_sha256",
        "measured_afm_scan_sq_nm",
        "measured_afm_shape",
        "keyframe_index",
        "keyframe_path",
        "keyframe_sha256",
        "keyframe_view",
        "roi_x",
        "roi_y",
        "roi_width",
        "roi_height",
        "legacy_output_sq_nm",
        "legacy_raw_predicted_sq_nm",
        "legacy_output_was_clipped",
        "legacy_source_sample_id",
        "legacy_source_afm_file_id",
        "legacy_source_sq_nm",
        "legacy_source_afm_path",
        "legacy_source_afm_sha256",
        "current_predicted_sq_nm",
        "current_npz_path",
        "current_npz_sha256",
        "legacy_height_scale_low_nm",
        "legacy_height_scale_high_nm",
        "current_height_scale_low_nm",
        "current_height_scale_high_nm",
        "measured_height_scale_low_nm",
        "measured_height_scale_high_nm",
    ]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[manifest_columns].to_csv(args.output_manifest, index=False)

    provenance = {
        "artifact": str(args.output_pdf),
        "artifact_sha256": sha256(args.output_pdf) if args.output_pdf.exists() else None,
        "legacy": {
            "method": LEGACY_METHOD,
            "commit": LEGACY_COMMIT,
            "retrieval_results_csv": str(paths["legacy_csv"]),
            "retrieval_results_sha256": sha256(paths["legacy_csv"]),
            "run_provenance": str(paths["legacy_provenance"]),
            "run_provenance_sha256": sha256(paths["legacy_provenance"]),
            "reference_atlas": str(paths["legacy_atlas"]),
            "reference_atlas_sha256": sha256(paths["legacy_atlas"]),
        },
        "current": {
            "method": CURRENT_METHOD,
            "sq_predictor_method": validation["current_sq_predictor_method"],
            "result_commit": CURRENT_RESULT_COMMIT,
            "package_commit": CURRENT_PACKAGE_COMMIT,
            "rq_crossfit_predictions_csv": str(paths["current_csv"]),
            "rq_crossfit_predictions_sha256": sha256(paths["current_csv"]),
            "best_model_manifest": str(paths["current_manifest"]),
            "best_model_manifest_sha256": sha256(paths["current_manifest"]),
            "phase1_manifest": str(paths["phase1_manifest"]),
            "phase1_manifest_sha256": sha256(paths["phase1_manifest"]),
        },
        "nomenclature": {
            "display_label": "Sq",
            "definition": "areal RMS height in nm",
            "legacy_source_label": "Rq",
            "note": "Legacy numeric values are preserved; only the display label is corrected to Sq.",
        },
        "visualization": {
            "current_generated_draw_index": 0,
            "afm_height_scale": "independent per panel, 1st-99th percentile; Sq labels preserve amplitude",
            "measured_afm": "current phase-1 representative line-3 AFM array for each target",
            "rheed_roi_overlay": "cyan rectangle from current phase-1 manifest",
        },
        "validation": validation,
    }
    args.output_provenance.parent.mkdir(parents=True, exist_ok=True)
    args.output_provenance.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    legacy_data_root = args.legacy_data_root.resolve()
    paths = input_paths(package_root)
    validate_static_inputs(paths)
    rows, validation = load_rows(package_root, legacy_data_root, paths)
    if not args.validate_only:
        draw_atlas(rows, args.output_pdf.resolve())
    write_outputs(rows, validation, paths, args)
    print(json.dumps(validation, indent=2))
    if not args.validate_only:
        print(f"PDF: {args.output_pdf.resolve()}")
    print(f"Manifest: {args.output_manifest.resolve()}")
    print(f"Provenance: {args.output_provenance.resolve()}")


if __name__ == "__main__":
    main()
