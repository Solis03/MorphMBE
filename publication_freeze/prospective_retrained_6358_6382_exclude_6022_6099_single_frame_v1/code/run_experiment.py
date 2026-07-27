#!/usr/bin/env python3
"""Run the 21+N6358+N6382 workflow after excluding 6022 and 6099."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


THIS = Path(__file__).resolve()
REPO = next(
    parent
    for parent in THIS.parents
    if (parent / "publication_freeze" / "rheed_afm_single_frame_v1_2026-07-18").is_dir()
)
PKG_REL = Path(
    "publication_freeze/prospective_retrained_6358_6382_exclude_6022_6099_single_frame_v1"
)
PKG = REPO / PKG_REL
FREEZE_REL = Path("publication_freeze/rheed_afm_single_frame_v1_2026-07-18")
FREEZE = REPO / FREEZE_REL
SOURCE_REL = Path("publication_freeze/prospective_unseen_single_frame_v1")
SOURCE = REPO / SOURCE_REL
PROCESSED_AFM = REPO / "data/processed_afm_extra_five"

EXTRA_IDS = ["N6342", "N6358", "N6382", "N6389", "N6390"]
ADDED_TRAIN_IDS = ["N6358", "N6382"]
TEST_IDS = ["N6342", "N6389", "N6390"]
IGNORED_IDS = ["N6324"]
ORIGINAL_HISTORICAL_IDS = [
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
EXCLUDED_IDS = ["6022", "6099"]
HISTORICAL_IDS = [
    sample_id
    for sample_id in ORIGINAL_HISTORICAL_IDS
    if sample_id not in EXCLUDED_IDS
]
TRAIN_IDS = HISTORICAL_IDS + [sid.removeprefix("N") for sid in ADDED_TRAIN_IDS]
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def rq_nm(array: np.ndarray) -> float:
    data = np.asarray(array, dtype=np.float64)
    finite = data[np.isfinite(data)]
    centered = finite - float(np.mean(finite))
    return float(np.sqrt(np.mean(centered**2)))


def ra_nm(array: np.ndarray) -> float:
    data = np.asarray(array, dtype=np.float64)
    finite = data[np.isfinite(data)]
    centered = finite - float(np.mean(finite))
    return float(np.mean(np.abs(centered)))


def finite_percentile(array: np.ndarray, values: list[float]) -> list[float]:
    finite = np.asarray(array, dtype=float)
    finite = finite[np.isfinite(finite)]
    return [float(x) for x in np.percentile(finite, values)]


def load_frozen_modules() -> tuple[Any, Any]:
    source_root = FREEZE / "code/scientific_source"
    sys.path.insert(0, str(source_root))
    afm_descriptors = importlib.import_module("analysis.rheed_video_afm_story.afm_descriptors")
    sys.path.insert(0, str(SOURCE / "code"))
    retrieval = importlib.import_module("generate_full_cohort_retrieval_images")
    sys.path.insert(0, str(REPO))
    second_order = importlib.import_module("scripts.fit_afm_second_order")
    return afm_descriptors, retrieval, second_order


def copy_keyframe_assets() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for kind in ["raw", "roi", "model_ready"]:
        (PKG / "inputs/keyframes" / kind).mkdir(parents=True, exist_ok=True)
    (PKG / "inputs/metadata").mkdir(parents=True, exist_ok=True)
    for sid in EXTRA_IDS:
        raw = next((SOURCE / "keyframes/raw").glob(f"{sid}_*.png"))
        roi = next((SOURCE / "keyframes/roi").glob(f"{sid}_*.png"))
        model_ready = SOURCE / "keyframes/model_ready" / f"{sid}_keyframe_1_raw_luminance.npz"
        metadata = SOURCE / "metadata/samples" / f"{sid}.json"
        destinations = {
            "raw_keyframe_png": PKG / "inputs/keyframes/raw" / raw.name,
            "roi_keyframe_png": PKG / "inputs/keyframes/roi" / roi.name,
            "model_ready_keyframe_npz": PKG / "inputs/keyframes/model_ready" / model_ready.name,
            "selection_metadata_json": PKG / "inputs/metadata" / metadata.name,
        }
        for source_path, destination in [
            (raw, destinations["raw_keyframe_png"]),
            (roi, destinations["roi_keyframe_png"]),
            (model_ready, destinations["model_ready_keyframe_npz"]),
            (metadata, destinations["selection_metadata_json"]),
        ]:
            shutil.copy2(source_path, destination)
        rows[sid] = {key: rel(value) for key, value in destinations.items()}
    fresh_embedding = (
        SOURCE
        / "predictions/full_cohort_single_frame_v1"
        / "unseen_dino_vits14_keyframe_1_raw_luminance_embeddings.npz"
    )
    embedding_copy = PKG / "inputs/embeddings/unseen_dino_vits14_keyframe_1_raw_luminance_embeddings.npz"
    embedding_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fresh_embedding, embedding_copy)
    return rows


def scan_number(path: Path) -> int:
    match = re.search(r"_2um_(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot find scan number in {path}")
    return int(match.group(1))


def source_scan_paths(sample_id: str) -> list[Path]:
    paths = sorted(
        (PROCESSED_AFM / sample_id).glob("*/*_height.npy"),
        key=lambda path: (scan_number(path), path.name),
    )
    if len(paths) != 5:
        raise RuntimeError(f"Expected five selected AFM scans for {sample_id}; found {len(paths)}")
    if [scan_number(path) for path in paths] != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"Expected scan numbers 1..5 for {sample_id}: {paths}")
    return paths


def build_quarter_ground_truth(
    afm_descriptors: Any, second_order: Any
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[np.ndarray]]]:
    manifest_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = {}
    root = PKG / "ground_truth_afm/top_left_quarter_second_order"
    for sid in EXTRA_IDS:
        arrays[sid] = []
        sample_rows: list[dict[str, Any]] = []
        for source in source_scan_paths(sid):
            number = scan_number(source)
            full = np.load(source, allow_pickle=False).astype(np.float64)
            height, width = full.shape
            crop = full[: height // 2, : width // 2].copy()
            finite = np.isfinite(crop)
            background, coefficients, rank, condition, fit_mask, iterations = second_order.robust_fit(
                crop, "y2", True
            )
            corrected = crop.copy()
            corrected[finite] = crop[finite] - background[finite]
            corrected[~finite] = crop[~finite]
            scan_dir = root / sid / f"scan_{number}"
            output = scan_dir / f"{sid}_scan_{number}_top_left_quarter_second_order.npy"
            background_path = scan_dir / f"{sid}_scan_{number}_top_left_quarter_y2_background.npy"
            output.parent.mkdir(parents=True, exist_ok=True)
            np.save(output, corrected.astype(np.float64))
            np.save(background_path, background.astype(np.float64))
            physical = afm_descriptors.describe_map(corrected, "physical", scan_size_um=1.0)
            centered = corrected - float(np.nanmean(corrected))
            unit = centered / max(rq_nm(centered), 1e-12)
            unit_desc = afm_descriptors.describe_map(unit, "unit", scan_size_um=1.0)
            rq_value = rq_nm(corrected)
            metadata = {
                "sample_id": sid,
                "scan_number": number,
                "source_path": rel(source),
                "source_shape": list(full.shape),
                "source_scan_size_um": [2.0, 2.0],
                "crop_quarter": "top_left",
                "crop_bounds_xyxy": [0, 0, width // 2, height // 2],
                "output_shape": list(corrected.shape),
                "effective_scan_size_um": [1.0, 1.0],
                "processing_order": "physical ZSensor height -> top-left quarter crop -> robust second-order y2 subtraction",
                "second_order_model": "y2",
                "second_order_robust": True,
                "second_order_iterations": int(iterations),
                "second_order_fit_fraction": float(np.count_nonzero(fit_mask) / np.count_nonzero(finite)),
                "second_order_coefficients": second_order.coefficients_dict("y2", coefficients),
                "second_order_rank": int(rank),
                "second_order_condition_number": float(condition),
                "rq_nm": rq_value,
                "ra_nm": ra_nm(corrected),
                "height_unit": "nm",
                "output_path": rel(output),
                "background_path": rel(background_path),
                "source_sha256": sha256_file(source),
                "output_sha256": sha256_file(output),
            }
            metadata_path = output.with_suffix(".json")
            write_json(metadata_path, metadata)
            row = {
                **metadata,
                "metadata_path": rel(metadata_path),
                "afm_file_id": f"{sid}_scan_{number}_top_left_quarter",
                "finite_fraction": float(np.count_nonzero(finite) / finite.size),
                "quality_pass": True,
                "quality_flags": "",
                "physical_rq": physical["physical_rq"],
                "physical_ra": physical["physical_ra"],
                "physical_robust_height_range": physical["physical_robust_height_range"],
                "height_skewness": physical["physical_skewness"],
                "height_kurtosis": physical["physical_kurtosis"],
                **physical,
                **unit_desc,
            }
            sample_rows.append(row)
            manifest_rows.append(row)
            arrays[sid].append(corrected)
        rqs = np.asarray([float(row["rq_nm"]) for row in sample_rows], dtype=float)
        trimmed = np.sort(rqs)[1:-1]
        finite_weights = np.asarray([float(row["finite_fraction"]) for row in sample_rows], dtype=float)
        median = float(np.median(rqs))
        representative = min(sample_rows, key=lambda row: (abs(float(row["rq_nm"]) - median), row["scan_number"]))
        target_rows.append(
            {
                "sample_id": sid,
                "role": "added_training" if sid in ADDED_TRAIN_IDS else "prediction_ground_truth_only",
                "T4_second_order_trimmed_mean": float(np.mean(trimmed)),
                "T6_quality_weighted_second_order": float(np.average(rqs, weights=np.maximum(finite_weights, 1e-6))),
                "T2_second_order_median_rq": median,
                "scan_count": len(rqs),
                "scan_rq_min_nm": float(np.min(rqs)),
                "scan_rq_max_nm": float(np.max(rqs)),
                "scan_rq_mad_nm": float(np.median(np.abs(rqs - median))),
                "scan_rq_iqr_nm": float(np.quantile(rqs, 0.75) - np.quantile(rqs, 0.25)),
                "representative_scan_number": int(representative["scan_number"]),
                "representative_map_path": representative["output_path"],
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    targets = pd.DataFrame(target_rows)
    write_frame(PKG / "ground_truth_afm/quarter_afm_manifest.csv", manifest)
    write_frame(PKG / "ground_truth_afm/sample_targets.csv", targets)
    return manifest, targets, arrays


def load_embedding_banks(
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray], np.ndarray]:
    historical_path = FREEZE / "models/encoder/dino_vits14_keyframe_1_raw_luminance_embeddings.npz"
    unseen_path = PKG / "inputs/embeddings/unseen_dino_vits14_keyframe_1_raw_luminance_embeddings.npz"
    historical = np.load(historical_path, allow_pickle=False)
    original_ids = [str(value) for value in historical["sample_ids"].tolist()]
    if original_ids != ORIGINAL_HISTORICAL_IDS:
        raise RuntimeError("Historical embedding order differs from the frozen training order")
    original_X = np.asarray(historical["embeddings"], dtype=float)
    keep_indices = [
        index
        for index, sample_id in enumerate(original_ids)
        if sample_id in HISTORICAL_IDS
    ]
    historical_X = original_X[keep_indices]
    unseen = np.load(unseen_path, allow_pickle=False)
    unseen_ids = [str(value) for value in unseen["sample_ids"].tolist()]
    unseen_map = {
        sid: np.asarray(unseen["embeddings"][index], dtype=float)
        for index, sid in enumerate(unseen_ids)
    }
    if set(EXTRA_IDS) - set(unseen_map):
        raise RuntimeError("Fresh unseen embedding bank is incomplete")
    return HISTORICAL_IDS, historical_X, unseen_map, original_X


def frozen_reproduction_audit(
    X: np.ndarray, historical_targets: pd.DataFrame, ensemble: dict[str, Any]
) -> dict[str, Any]:
    target_index = historical_targets.set_index("sample_id")
    member_rows: list[dict[str, Any]] = []
    max_coef = 0.0
    max_intercept = 0.0
    max_mean = 0.0
    max_scale = 0.0
    model_root = FREEZE / "models/quantitative_model/full_cohort_deployment"
    for member in ensemble["members"]:
        y = target_index.loc[
            ORIGINAL_HISTORICAL_IDS, member["target_variant"]
        ].to_numpy(float)
        scaler = StandardScaler().fit(X)
        model = Ridge(alpha=1.0).fit(scaler.transform(X), y)
        frozen = np.load(model_root / f"{member['name']}.npz", allow_pickle=False)
        row = {
            "member_name": member["name"],
            "trial_id": member["trial_id"],
            "target_variant": member["target_variant"],
            "max_abs_coef_difference": float(np.max(np.abs(model.coef_ - frozen["coef"]))),
            "abs_intercept_difference": float(abs(model.intercept_ - frozen["intercept"])),
            "max_abs_feature_mean_difference": float(np.max(np.abs(scaler.mean_ - frozen["feature_mean"]))),
            "max_abs_feature_scale_difference": float(np.max(np.abs(scaler.scale_ - frozen["feature_scale"]))),
        }
        max_coef = max(max_coef, row["max_abs_coef_difference"])
        max_intercept = max(max_intercept, row["abs_intercept_difference"])
        max_mean = max(max_mean, row["max_abs_feature_mean_difference"])
        max_scale = max(max_scale, row["max_abs_feature_scale_difference"])
        member_rows.append(row)
    audit = {
        "status": "pass" if max(max_coef, max_intercept, max_mean, max_scale) < 1e-5 else "fail",
        "tolerance": 1e-5,
        "historical_training_sample_count": len(ORIGINAL_HISTORICAL_IDS),
        "algorithm": "StandardScaler().fit(X); Ridge(alpha=1.0).fit(scaler.transform(X), y)",
        "max_abs_coef_difference": max_coef,
        "max_abs_intercept_difference": max_intercept,
        "max_abs_feature_mean_difference": max_mean,
        "max_abs_feature_scale_difference": max_scale,
        "members": member_rows,
    }
    write_json(PKG / "provenance/frozen_23_reproduction_audit.json", audit)
    if audit["status"] != "pass":
        raise RuntimeError(f"Frozen reproduction audit failed: {audit}")
    return audit


def train_reduced_historical_baseline() -> pd.DataFrame:
    historical_targets = pd.read_csv(
        FREEZE / "data_snapshot/sample_targets_all_variants.csv",
        dtype={"sample_id": str},
    ).set_index("sample_id")
    _, historical_X, unseen_map, _ = load_embedding_banks()
    ensemble = json.loads(
        (
            FREEZE
            / "models/quantitative_model/full_cohort_deployment/ensemble_definition.json"
        ).read_text(encoding="utf-8")
    )
    model_root = PKG / "models/quantitative_model/reduced_21_baseline"
    output_root = PKG / "predictions/reduced_21_baseline"
    model_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    fitted: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for member in ensemble["members"]:
        variant = member["target_variant"]
        y = historical_targets.loc[HISTORICAL_IDS, variant].to_numpy(float)
        scaler = StandardScaler().fit(historical_X)
        model = Ridge(alpha=1.0).fit(scaler.transform(historical_X), y)
        model_path = model_root / f"{member['name']}.npz"
        np.savez(
            model_path,
            coef=model.coef_,
            intercept=model.intercept_,
            feature_mean=scaler.mean_,
            feature_scale=scaler.scale_,
            target_variant=variant,
            training_sample_ids=np.asarray(HISTORICAL_IDS),
        )
        fitted.append(
            {
                "member": member,
                "scaler": scaler,
                "model": model,
                "model_path": model_path,
            }
        )
    for sample_id in EXTRA_IDS:
        values: list[float] = []
        for item in fitted:
            predicted = float(
                item["model"].predict(
                    item["scaler"].transform(unseen_map[sample_id][None, :])
                )[0]
            )
            values.append(predicted)
            member_rows.append(
                {
                    "sample_id": sample_id,
                    "member_name": item["member"]["name"],
                    "trial_id": item["member"]["trial_id"],
                    "target_variant": item["member"]["target_variant"],
                    "predicted_rq_nm": predicted,
                    "training_sample_count": len(HISTORICAL_IDS),
                }
            )
        values_array = np.asarray(values, dtype=float)
        predicted = float(np.median(values_array))
        prediction_rows.append(
            {
                "sample_id": sample_id,
                "predicted_rq_nm": predicted,
                "q50_predicted_rq_nm": predicted,
                "predicted_rq_nm_clipped_nonnegative": max(0.0, predicted),
                "physical_plausibility_flag": (
                    "negative_raw_model_output" if predicted < 0 else "ok"
                ),
                "member_q10_rq_nm": float(np.quantile(values_array, 0.10)),
                "member_q90_rq_nm": float(np.quantile(values_array, 0.90)),
                "member_min_rq_nm": float(np.min(values_array)),
                "member_max_rq_nm": float(np.max(values_array)),
                "ensemble_member_count": len(values_array),
                "training_sample_count": len(HISTORICAL_IDS),
                "training_sample_ids": json.dumps(HISTORICAL_IDS),
                "excluded_sample_ids": json.dumps(EXCLUDED_IDS),
                "prediction_role": "reduced_historical_baseline_prediction",
            }
        )
    predictions = pd.DataFrame(prediction_rows)
    members = pd.DataFrame(member_rows)
    write_frame(output_root / "predictions.csv", predictions)
    write_frame(output_root / "ensemble_member_predictions.csv", members)
    write_json(
        output_root / "prediction_run_provenance.json",
        {
            "created_at": now(),
            "algorithm": "unchanged StandardScaler plus Ridge(alpha=1.0), five-member median",
            "training_sample_ids": HISTORICAL_IDS,
            "excluded_sample_ids": EXCLUDED_IDS,
            "prediction_sample_ids": EXTRA_IDS,
            "training_sample_count": len(HISTORICAL_IDS),
            "model_root": rel(model_root),
        },
    )
    return predictions


def train_and_predict(
    extra_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    historical_targets = pd.read_csv(
        FREEZE / "data_snapshot/sample_targets_all_variants.csv", dtype={"sample_id": str}
    )
    historical_ids, historical_X, unseen_map, original_historical_X = (
        load_embedding_banks()
    )
    ensemble = json.loads(
        (FREEZE / "models/quantitative_model/full_cohort_deployment/ensemble_definition.json").read_text(
            encoding="utf-8"
        )
    )
    reproduction = frozen_reproduction_audit(
        original_historical_X, historical_targets, ensemble
    )
    added_target_index = extra_targets.set_index("sample_id")
    X_train = np.vstack([historical_X] + [unseen_map[sid][None, :] for sid in ADDED_TRAIN_IDS])
    target_table = historical_targets.set_index("sample_id").loc[
        HISTORICAL_IDS, ["T4_second_order_trimmed_mean", "T6_quality_weighted_second_order"]
    ].copy()
    for sid in ADDED_TRAIN_IDS:
        target_table.loc[sid.removeprefix("N")] = {
            "T4_second_order_trimmed_mean": float(
                added_target_index.loc[sid, "T4_second_order_trimmed_mean"]
            ),
            "T6_quality_weighted_second_order": float(
                added_target_index.loc[sid, "T6_quality_weighted_second_order"]
            ),
        }
    target_table = target_table.loc[TRAIN_IDS]
    model_root = PKG / "models/quantitative_model"
    model_root.mkdir(parents=True, exist_ok=True)
    write_frame(
        model_root / "training_sample_ids.csv",
        pd.DataFrame(
            {
                "sample_id": TRAIN_IDS,
                "source": ["frozen_historical"] * len(HISTORICAL_IDS) + ["added_extra_five"] * 2,
            }
        ),
    )
    target_export = target_table.reset_index().rename(columns={"index": "sample_id"})
    write_frame(model_root / "training_targets.csv", target_export)
    members: list[dict[str, Any]] = []
    training_prediction_columns: dict[str, np.ndarray] = {}
    test_member_rows: list[dict[str, Any]] = []
    test_predictions: list[dict[str, Any]] = []
    fitted_models: list[dict[str, Any]] = []
    for member in ensemble["members"]:
        variant = member["target_variant"]
        y = target_table[variant].to_numpy(float)
        scaler = StandardScaler().fit(X_train)
        model = Ridge(alpha=1.0).fit(scaler.transform(X_train), y)
        output = model_root / f"{member['name']}.npz"
        np.savez(
            output,
            coef=model.coef_,
            intercept=model.intercept_,
            feature_mean=scaler.mean_,
            feature_scale=scaler.scale_,
            target_variant=variant,
            training_sample_ids=np.asarray(TRAIN_IDS),
        )
        training_prediction_columns[member["name"]] = model.predict(scaler.transform(X_train))
        fitted_models.append(
            {
                "member": member,
                "scaler": scaler,
                "model": model,
                "path": output,
                "sha256": sha256_file(output),
            }
        )
        members.append(
            {
                **member,
                "ridge_alpha": 1.0,
                "training_sample_count": len(TRAIN_IDS),
                "model_path": rel(output),
                "model_sha256": sha256_file(output),
            }
        )
    for sid in TEST_IDS:
        values: list[float] = []
        for fitted in fitted_models:
            pred = float(
                fitted["model"].predict(fitted["scaler"].transform(unseen_map[sid][None, :]))[0]
            )
            values.append(pred)
            test_member_rows.append(
                {
                    "sample_id": sid,
                    "member_name": fitted["member"]["name"],
                    "trial_id": fitted["member"]["trial_id"],
                    "target_variant": fitted["member"]["target_variant"],
                    "predicted_rq_nm": pred,
                    "model_sha256": fitted["sha256"],
                }
            )
        values_np = np.asarray(values, dtype=float)
        q50 = float(np.median(values_np))
        test_predictions.append(
            {
                "sample_id": sid,
                "predicted_rq_nm": q50,
                "q50_predicted_rq_nm": q50,
                "predicted_rq_nm_clipped_nonnegative": max(0.0, q50),
                "physical_plausibility_flag": "negative_raw_model_output" if q50 < 0 else "ok",
                "member_q10_rq_nm": float(np.quantile(values_np, 0.10)),
                "member_q90_rq_nm": float(np.quantile(values_np, 0.90)),
                "member_min_rq_nm": float(np.min(values_np)),
                "member_max_rq_nm": float(np.max(values_np)),
                "ensemble_member_count": len(values_np),
                "training_sample_count": len(TRAIN_IDS),
                "training_sample_ids": json.dumps(TRAIN_IDS),
                "added_training_sample_ids": json.dumps(ADDED_TRAIN_IDS),
                "prediction_role": "held_out_extra_five_prediction",
            }
        )
    prediction_frame = pd.DataFrame(test_predictions)
    member_frame = pd.DataFrame(test_member_rows)
    training_frame = pd.DataFrame(
        {
            "sample_id": TRAIN_IDS,
            "true_T4_rq_nm": target_table["T4_second_order_trimmed_mean"].to_numpy(float),
            **training_prediction_columns,
        }
    )
    training_frame["ensemble_in_sample_prediction_nm"] = np.median(
        training_frame[list(training_prediction_columns)].to_numpy(float), axis=1
    )
    training_frame["warning"] = "FULL-COHORT TRAINING FIT; NOT TEST PERFORMANCE"
    output_root = PKG / "predictions/retrained_23"
    write_frame(output_root / "predictions.csv", prediction_frame)
    write_frame(output_root / "ensemble_member_predictions.csv", member_frame)
    write_frame(model_root / "training_fit_predictions.csv", training_frame)
    np.savez(
        PKG / "models/encoder/combined_training_and_test_embeddings.npz",
        training_sample_ids=np.asarray(TRAIN_IDS),
        training_embeddings=X_train.astype(np.float32),
        test_sample_ids=np.asarray(TEST_IDS),
        test_embeddings=np.vstack([unseen_map[sid] for sid in TEST_IDS]).astype(np.float32),
        encoder_name="DINOv2 ViT-S/14",
        preprocessing="keyframe_1_raw_luminance",
    )
    registry = {
        "created_at": now(),
        "model_name": "retrained_23_excluding_6022_6099_top5_median_ridge_ensemble",
        "frozen_algorithm_source": rel(
            REPO / "analysis/rheed_video_afm_story/build_final_paper_freeze.py"
        ),
        "frozen_source_function": "fit_full_cohort_quantitative",
        "training_sample_count": len(TRAIN_IDS),
        "historical_training_sample_count": len(HISTORICAL_IDS),
        "added_training_samples": ADDED_TRAIN_IDS,
        "excluded_historical_samples": EXCLUDED_IDS,
        "prediction_samples": TEST_IDS,
        "ignored_samples": IGNORED_IDS,
        "feature_set": "E1_dino_keyframe",
        "member_family": "Ridge",
        "ridge_alpha": 1.0,
        "feature_scaling": "StandardScaler fitted on the 23 training rows",
        "aggregation": "median",
        "members": members,
        "frozen_23_reproduction_audit": reproduction,
        "warning": "RETRAINED REDUCED-COHORT FIT; NOT THE FROZEN STRICT OOF BENCHMARK",
    }
    write_json(model_root / "model_registry.json", registry)
    write_json(model_root / "ensemble_definition.json", {"aggregation": "median", "members": members})
    return prediction_frame, member_frame, training_frame, registry, unseen_map


def expanded_retrieval_bank(
    manifest: pd.DataFrame, afm_descriptors: Any
) -> pd.DataFrame:
    historical = pd.read_csv(
        FREEZE / "models/visual_model/full_cohort_bank/afm_bank_manifest.csv",
        dtype={"sample_id": str},
    )
    historical = historical[
        ~historical["sample_id"].isin(EXCLUDED_IDS)
    ].copy()
    extra_rows: list[dict[str, Any]] = []
    for row in manifest[manifest["sample_id"].isin(ADDED_TRAIN_IDS)].to_dict("records"):
        array = np.load(REPO / row["output_path"], allow_pickle=False).astype(float)
        centered = array - float(np.nanmean(array))
        unit = centered / max(rq_nm(centered), 1e-12)
        physical = afm_descriptors.describe_map(array, "physical", scan_size_um=1.0)
        unit_desc = afm_descriptors.describe_map(unit, "unit", scan_size_um=1.0)
        extra_rows.append(
            {
                "sample_id": str(row["sample_id"]).removeprefix("N"),
                "growth_run_id": str(row["sample_id"]).removeprefix("N"),
                "source_afm_file": row["source_path"],
                "afm_file_id": row["afm_file_id"],
                "plane_corrected_array_path": row["output_path"],
                "second_order_afm_path": row["output_path"],
                "scan_size_x_um": 1.0,
                "scan_size_y_um": 1.0,
                "resolution_x": 256,
                "resolution_y": 256,
                "height_unit": "nm",
                "rq_nm": rq_nm(array),
                "ra_nm": ra_nm(array),
                "robust_height_range_nm": physical["physical_robust_height_range"],
                "height_skewness": physical["physical_skewness"],
                "height_kurtosis": physical["physical_kurtosis"],
                "paired_primary": True,
                "paired_exploratory": False,
                "unpaired_support": False,
                "representative_for_sample": int(row["scan_number"]) == 1,
                "quality_pass": True,
                "quality_flags": "",
                "source_array_hash": sha256_file(REPO / row["output_path"]),
                **physical,
                **unit_desc,
            }
        )
    expanded = pd.concat([historical, pd.DataFrame(extra_rows)], ignore_index=True, sort=False)
    if expanded["sample_id"].astype(str).nunique() != len(TRAIN_IDS):
        raise RuntimeError(
            "Expanded retrieval bank does not contain the expected 23 training groups"
        )
    if expanded["sample_id"].isin([sid.removeprefix("N") for sid in TEST_IDS]).any():
        raise RuntimeError("Prediction ground truth leaked into the retrieval bank")
    write_frame(PKG / "models/visual_model/expanded_afm_bank_manifest.csv", expanded)
    return expanded


def run_retrieval(
    predictions: pd.DataFrame, bank: pd.DataFrame, retrieval: Any
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    output = PKG / "predictions/retrained_23/retrieval"
    map_dir = output / "retrieved_maps_q50"
    source_dir = output / "source_unit_shape_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for pred in predictions.to_dict("records"):
        sid = str(pred["sample_id"])
        raw = float(pred["predicted_rq_nm"])
        q50 = max(float(pred["predicted_rq_nm_clipped_nonnegative"]), 1e-3)
        ranked = retrieval.rank_bank(bank, q50)
        top5 = retrieval.top_distinct_groups(ranked, 5)
        source = ranked.iloc[0]
        source_map = retrieval.read_map(REPO, str(source["second_order_afm_path"]))
        unit = retrieval.project_unit_rq(source_map)
        retrieved_map = retrieval.physical_from_q(unit, q50)
        source_path = source_dir / f"{sid}_source_{source['sample_id']}_{source['afm_file_id']}_unit.npy"
        map_path = map_dir / f"{sid}_A3_retrained23_q50_retrieved.npy"
        np.save(source_path, unit.astype(np.float32))
        np.save(map_path, retrieved_map.astype(np.float32))
        arrays[sid] = retrieved_map
        rows.append(
            {
                "sample_id": sid,
                "method_id": "A3_full_cohort_rq_conditioned",
                "raw_predicted_rq_nm": raw,
                "retrieval_q50_rq_nm": q50,
                "negative_raw_prediction_clipped_for_physical_map": raw < 0,
                "source_sample_id": str(source["sample_id"]),
                "source_afm_file_id": str(source["afm_file_id"]),
                "source_afm_path": str(source["second_order_afm_path"]),
                "source_rank_score": float(source["rank_score"]),
                "source_rq_nm": float(source["rq_nm"]),
                "top5_source_sample_ids": json.dumps(top5["sample_id"].astype(str).tolist()),
                "top5_source_afm_file_ids": json.dumps(top5["afm_file_id"].astype(str).tolist()),
                "source_group_count": int(bank["sample_id"].astype(str).nunique()),
                "training_bank_sample_ids": json.dumps(TRAIN_IDS),
                "test_ground_truth_in_retrieval_bank": False,
                "rq_condition_source": "retrained_23 predictions.csv",
                "retrieved_q50_map_path": rel(map_path),
                "source_unit_shape_map_path": rel(source_path),
            }
        )
    frame = pd.DataFrame(rows)
    write_frame(output / "retrieval_results.csv", frame)
    write_json(
        output / "retrieval_run_provenance.json",
        {
            "created_at": now(),
            "method": "A3_full_cohort_rq_conditioned",
            "algorithm_source": rel(SOURCE / "code/generate_full_cohort_retrieval_images.py"),
            "training_source_group_count": len(TRAIN_IDS),
            "added_source_groups": ADDED_TRAIN_IDS,
            "excluded_source_groups": EXCLUDED_IDS,
            "prediction_ground_truth_excluded": True,
            "descriptor_columns": DESCRIPTOR_COLS,
        },
    )
    return frame, arrays


def run_reduced_baseline_retrieval(
    predictions: pd.DataFrame, retrieval: Any
) -> pd.DataFrame:
    bank = pd.read_csv(
        FREEZE / "models/visual_model/full_cohort_bank/afm_bank_manifest.csv",
        dtype={"sample_id": str},
    )
    bank = bank[~bank["sample_id"].isin(EXCLUDED_IDS)].copy()
    output = PKG / "predictions/reduced_21_baseline/retrieval"
    map_dir = output / "retrieved_maps_q50"
    source_dir = output / "source_unit_shape_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for prediction in predictions.to_dict("records"):
        sample_id = str(prediction["sample_id"])
        raw = float(prediction["predicted_rq_nm"])
        q50 = max(
            float(prediction["predicted_rq_nm_clipped_nonnegative"]),
            1e-3,
        )
        ranked = retrieval.rank_bank(bank, q50)
        top5 = retrieval.top_distinct_groups(ranked, 5)
        source = ranked.iloc[0]
        source_map = retrieval.read_map(
            REPO, str(source["second_order_afm_path"])
        )
        unit = retrieval.project_unit_rq(source_map)
        retrieved = retrieval.physical_from_q(unit, q50)
        source_path = (
            source_dir
            / f"{sample_id}_source_{source['sample_id']}_{source['afm_file_id']}_unit.npy"
        )
        map_path = (
            map_dir / f"{sample_id}_A3_reduced21_q50_retrieved.npy"
        )
        np.save(source_path, unit.astype(np.float32))
        np.save(map_path, retrieved.astype(np.float32))
        rows.append(
            {
                "sample_id": sample_id,
                "method_id": "A3_full_cohort_rq_conditioned",
                "raw_predicted_rq_nm": raw,
                "retrieval_q50_rq_nm": q50,
                "negative_raw_prediction_clipped_for_physical_map": raw < 0,
                "source_sample_id": str(source["sample_id"]),
                "source_afm_file_id": str(source["afm_file_id"]),
                "source_afm_path": str(source["second_order_afm_path"]),
                "source_rank_score": float(source["rank_score"]),
                "source_rq_nm": float(source["rq_nm"]),
                "top5_source_sample_ids": json.dumps(
                    top5["sample_id"].astype(str).tolist()
                ),
                "top5_source_afm_file_ids": json.dumps(
                    top5["afm_file_id"].astype(str).tolist()
                ),
                "source_group_count": int(bank["sample_id"].nunique()),
                "training_bank_sample_ids": json.dumps(HISTORICAL_IDS),
                "excluded_sample_ids": json.dumps(EXCLUDED_IDS),
                "retrieved_q50_map_path": rel(map_path),
                "source_unit_shape_map_path": rel(source_path),
            }
        )
    frame = pd.DataFrame(rows)
    write_frame(output / "retrieval_results.csv", frame)
    write_json(
        output / "retrieval_run_provenance.json",
        {
            "created_at": now(),
            "method": "A3_full_cohort_rq_conditioned",
            "training_source_group_count": len(HISTORICAL_IDS),
            "training_source_group_ids": HISTORICAL_IDS,
            "excluded_source_groups": EXCLUDED_IDS,
            "algorithm_source": rel(
                SOURCE / "code/generate_full_cohort_retrieval_images.py"
            ),
        },
    )
    return frame


def evaluate_predictions(
    predictions: pd.DataFrame, baseline: pd.DataFrame, targets: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    truth = targets.set_index("sample_id")["T4_second_order_trimmed_mean"].astype(float)
    baseline_index = baseline.set_index("sample_id")
    rows: list[dict[str, Any]] = []
    for pred in predictions.to_dict("records"):
        sid = str(pred["sample_id"])
        true = float(truth.loc[sid])
        new = float(pred["predicted_rq_nm"])
        old = float(baseline_index.loc[sid, "predicted_rq_nm"])
        rows.append(
            {
                "sample_id": sid,
                "ground_truth_T4_rq_nm": true,
                "reduced_21_baseline_predicted_rq_nm": old,
                "reduced_21_baseline_absolute_error_nm": abs(old - true),
                "retrained_23_predicted_rq_nm": new,
                "retrained_23_absolute_error_nm": abs(new - true),
                "absolute_error_change_nm_retrained23_minus_reduced21": abs(new - true)
                - abs(old - true),
            }
        )
    frame = pd.DataFrame(rows)
    y = frame["ground_truth_T4_rq_nm"].to_numpy(float)
    pred = frame["retrained_23_predicted_rq_nm"].to_numpy(float)
    old_pred = frame["reduced_21_baseline_predicted_rq_nm"].to_numpy(float)
    metrics = {
        "evaluation_sample_count": len(frame),
        "evaluation_sample_ids": TEST_IDS,
        "ground_truth_role": "AFM labels withheld from model fit and retrieval; used after prediction for evaluation",
        "retrained_23": {
            "MAE_nm": float(mean_absolute_error(y, pred)),
            "RMSE_nm": float(math.sqrt(mean_squared_error(y, pred))),
            "R2": float(r2_score(y, pred)),
            "Spearman": float(spearmanr(y, pred).statistic),
            "Kendall": float(kendalltau(y, pred).statistic),
        },
        "reduced_21_baseline": {
            "MAE_nm": float(mean_absolute_error(y, old_pred)),
            "RMSE_nm": float(math.sqrt(mean_squared_error(y, old_pred))),
            "R2": float(r2_score(y, old_pred)),
            "Spearman": float(spearmanr(y, old_pred).statistic),
            "Kendall": float(kendalltau(y, old_pred).statistic),
        },
        "small_n_warning": "Only three prediction samples are evaluated; rank and R2 metrics are descriptive.",
    }
    write_frame(PKG / "evaluation/per_sample_evaluation.csv", frame)
    write_json(PKG / "evaluation/metrics.json", metrics)
    return frame, metrics


def height_limits(array: np.ndarray) -> tuple[float, float]:
    low, high = finite_percentile(array, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return -1.0, 1.0
    return low, high


def add_height_bar(ax: plt.Axes, low: float, high: float) -> None:
    color_axis = ax.inset_axes([1.025, 0.09, 0.045, 0.82])
    color_axis.imshow(np.linspace(high, low, 256)[:, None], cmap="viridis", aspect="auto")
    color_axis.set_xticks([])
    color_axis.set_yticks([0, 255])
    color_axis.set_yticklabels([f"{high:.1f}", f"{low:.1f}"], fontsize=5)
    color_axis.set_title("nm", fontsize=5, pad=1)


def show_afm(ax: plt.Axes, array: np.ndarray, label: str, title_size: float = 7.0) -> None:
    low, high = height_limits(array)
    ax.imshow(array, cmap="viridis", origin="upper", vmin=low, vmax=high)
    ax.set_title(f"{label}\nRq = {rq_nm(array):.2f} nm", fontsize=title_size, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    add_height_bar(ax, low, high)


def save_figure(fig: plt.Figure, root: Path, stem: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(root / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(root / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def render_individual_afm_maps(
    afm_arrays: dict[str, list[np.ndarray]], retrieved_arrays: dict[str, np.ndarray]
) -> None:
    ground_root = PKG / "figures/individual_afm/ground_truth"
    retrieved_root = PKG / "figures/individual_afm/retrieved"
    for sid, scans in afm_arrays.items():
        (ground_root / sid).mkdir(parents=True, exist_ok=True)
        for index, array in enumerate(scans, start=1):
            fig, ax = plt.subplots(figsize=(3.0, 3.0))
            show_afm(ax, array, f"{sid} ground truth {index}", title_size=8)
            fig.savefig(
                ground_root / sid / f"{sid}_ground_truth_{index}_heightbar_rq.png",
                dpi=400,
                bbox_inches="tight",
            )
            plt.close(fig)
    for sid, array in retrieved_arrays.items():
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        show_afm(ax, array, f"{sid} retrieved", title_size=8)
        path = retrieved_root / f"{sid}_retrieved_heightbar_rq.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=400, bbox_inches="tight")
        plt.close(fig)


def make_prediction_atlases(
    assets: dict[str, dict[str, str]],
    afm_arrays: dict[str, list[np.ndarray]],
    retrieved_arrays: dict[str, np.ndarray],
    predictions: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> None:
    pred_index = predictions.set_index("sample_id")
    eval_index = evaluation.set_index("sample_id")
    main_root = PKG / "figures/main"
    fig = plt.figure(figsize=(21.5, 9.3))
    grid = fig.add_gridspec(3, 7, wspace=0.58, hspace=0.42)
    for row_index, sid in enumerate(TEST_IDS):
        axes = [fig.add_subplot(grid[row_index, column]) for column in range(7)]
        keyframe = plt.imread(REPO / assets[sid]["roi_keyframe_png"])
        axes[0].imshow(keyframe, cmap="gray")
        axes[0].set_title(f"{sid} RHEED keyframe", fontsize=8, pad=3)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        for scan_index, array in enumerate(afm_arrays[sid], start=1):
            show_afm(axes[scan_index], array, f"Ground truth {scan_index}")
        pred = float(pred_index.loc[sid, "predicted_rq_nm"])
        true = float(eval_index.loc[sid, "ground_truth_T4_rq_nm"])
        show_afm(
            axes[6],
            retrieved_arrays[sid],
            f"Retrieved\nPred {pred:.2f}; GT T4 {true:.2f} nm",
        )
    fig.suptitle(
        "RHEED keyframe, five upper-left 1 µm AFM ground truths, and retrained A3 retrieval",
        fontsize=13,
        y=0.998,
    )
    save_figure(fig, main_root, "Figure1_three_sample_prediction_atlas")
    for sid in TEST_IDS:
        fig = plt.figure(figsize=(21.5, 3.2))
        grid = fig.add_gridspec(1, 7, wspace=0.58)
        axes = [fig.add_subplot(grid[0, column]) for column in range(7)]
        axes[0].imshow(plt.imread(REPO / assets[sid]["roi_keyframe_png"]), cmap="gray")
        axes[0].set_title(f"{sid} RHEED keyframe", fontsize=8)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        for scan_index, array in enumerate(afm_arrays[sid], start=1):
            show_afm(axes[scan_index], array, f"Ground truth {scan_index}")
        pred = float(pred_index.loc[sid, "predicted_rq_nm"])
        true = float(eval_index.loc[sid, "ground_truth_T4_rq_nm"])
        show_afm(axes[6], retrieved_arrays[sid], f"Retrieved\nPred {pred:.2f}; GT T4 {true:.2f} nm")
        save_figure(fig, PKG / "figures/per_sample", f"{sid}_keyframe_five_ground_truth_retrieved")


def make_supplementary_figures(
    afm_arrays: dict[str, list[np.ndarray]],
    retrieved_arrays: dict[str, np.ndarray],
    predictions: pd.DataFrame,
    members: pd.DataFrame,
    training_fit: pd.DataFrame,
    evaluation: pd.DataFrame,
    targets: pd.DataFrame,
    retrieval_results: pd.DataFrame,
) -> None:
    root = PKG / "figures/supplementary"
    fig = plt.figure(figsize=(15.5, 14.5))
    grid = fig.add_gridspec(5, 5, wspace=0.58, hspace=0.48)
    for row_index, sid in enumerate(EXTRA_IDS):
        for scan_index, array in enumerate(afm_arrays[sid]):
            show_afm(fig.add_subplot(grid[row_index, scan_index]), array, f"{sid} GT {scan_index + 1}")
    fig.suptitle("All five extra-sample AFM sets: upper-left 1 µm quarters", fontsize=13)
    save_figure(fig, root, "SuppFigure1_all_extra_five_ground_truth_atlas")

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    historical = training_fit.iloc[: len(HISTORICAL_IDS)]
    added = training_fit.iloc[len(HISTORICAL_IDS) :]
    bins = np.linspace(
        training_fit["true_T4_rq_nm"].min() * 0.9,
        training_fit["true_T4_rq_nm"].max() * 1.05,
        11,
    )
    ax.hist(
        historical["true_T4_rq_nm"],
        bins=bins,
        alpha=0.65,
        label="21 historical",
        color="#4C78A8",
    )
    for row in added.itertuples():
        ax.axvline(row.true_T4_rq_nm, linewidth=2.3, label=f"added {row.sample_id}")
    ax.set_xlabel("Training T4 Rq (nm)")
    ax.set_ylabel("Count")
    ax.set_title("Training-target distribution after adding N6358 and N6382")
    ax.legend()
    save_figure(fig, root, "SuppFigure2_training_target_distribution")

    fig, ax = plt.subplots(figsize=(7.0, 5.1))
    x = np.arange(len(evaluation))
    width = 0.25
    ax.bar(
        x - width,
        evaluation["reduced_21_baseline_predicted_rq_nm"],
        width,
        label="Reduced historical 21",
    )
    ax.bar(
        x,
        evaluation["retrained_23_predicted_rq_nm"],
        width,
        label="Retrained 23",
    )
    ax.bar(x + width, evaluation["ground_truth_T4_rq_nm"], width, label="Ground truth T4")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x, evaluation["sample_id"])
    ax.set_ylabel("Rq (nm)")
    ax.set_title("Reduced-baseline and retrained-model predictions")
    ax.legend()
    save_figure(fig, root, "SuppFigure3_reduced21_vs_retrained23_predictions")

    fig, ax = plt.subplots(figsize=(5.6, 5.3))
    true = evaluation["ground_truth_T4_rq_nm"].to_numpy(float)
    pred = evaluation["retrained_23_predicted_rq_nm"].to_numpy(float)
    ax.scatter(true, pred, s=70, color="#E45756")
    for row in evaluation.itertuples():
        ax.annotate(
            row.sample_id,
            (row.ground_truth_T4_rq_nm, row.retrained_23_predicted_rq_nm),
            xytext=(4, 4),
            textcoords="offset points",
        )
    low = min(true.min(), pred.min()) * 0.9
    high = max(true.max(), pred.max()) * 1.1
    ax.plot([low, high], [low, high], "k--", linewidth=1)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Ground-truth T4 Rq (nm)")
    ax.set_ylabel("Retrained predicted Rq (nm)")
    ax.set_title("Three held-out extra-five predictions")
    save_figure(fig, root, "SuppFigure4_retrained_prediction_vs_ground_truth")

    fig, ax = plt.subplots(figsize=(7.1, 5.0))
    pred_index = predictions.set_index("sample_id")
    x = np.arange(len(TEST_IDS))
    med = np.array([pred_index.loc[sid, "predicted_rq_nm"] for sid in TEST_IDS], dtype=float)
    q10 = np.array([pred_index.loc[sid, "member_q10_rq_nm"] for sid in TEST_IDS], dtype=float)
    q90 = np.array([pred_index.loc[sid, "member_q90_rq_nm"] for sid in TEST_IDS], dtype=float)
    ax.errorbar(x, med, yerr=np.vstack([med - q10, q90 - med]), fmt="o", capsize=5, label="Median and member q10–q90")
    for member_name, group in members.groupby("member_name"):
        values = group.set_index("sample_id").loc[TEST_IDS, "predicted_rq_nm"].to_numpy(float)
        ax.scatter(x, values, s=16, alpha=0.55, label=member_name)
    ax.set_xticks(x, TEST_IDS)
    ax.set_ylabel("Predicted Rq (nm)")
    ax.set_title("Five-member ensemble spread")
    ax.legend(fontsize=6, ncol=2)
    save_figure(fig, root, "SuppFigure5_ensemble_member_spread")

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    data = [[rq_nm(array) for array in afm_arrays[sid]] for sid in EXTRA_IDS]
    ax.boxplot(data, tick_labels=EXTRA_IDS, showmeans=True)
    for index, values in enumerate(data, start=1):
        ax.scatter(np.full(len(values), index), values, s=18, alpha=0.7)
    ax.set_ylabel("Scan-level quarter Rq (nm)")
    ax.set_title("Within-sample AFM Rq distributions")
    save_figure(fig, root, "SuppFigure6_extra_five_scan_rq_distributions")

    fig, ax = plt.subplots(figsize=(5.8, 5.3))
    ax.scatter(
        training_fit["true_T4_rq_nm"],
        training_fit["ensemble_in_sample_prediction_nm"],
        color="#4C78A8",
        alpha=0.75,
        label="21 historical",
    )
    for sid in [value.removeprefix("N") for value in ADDED_TRAIN_IDS]:
        row = training_fit[training_fit["sample_id"].astype(str).eq(sid)].iloc[0]
        ax.scatter(row["true_T4_rq_nm"], row["ensemble_in_sample_prediction_nm"], s=80, label=f"added N{sid}")
    low = min(training_fit["true_T4_rq_nm"].min(), training_fit["ensemble_in_sample_prediction_nm"].min()) * 0.9
    high = max(training_fit["true_T4_rq_nm"].max(), training_fit["ensemble_in_sample_prediction_nm"].max()) * 1.05
    ax.plot([low, high], [low, high], "k--")
    ax.set_xlabel("Training T4 Rq (nm)")
    ax.set_ylabel("In-sample ensemble prediction (nm)")
    ax.set_title("23-sample reduced-cohort training fit (not test performance)")
    ax.legend(fontsize=7)
    save_figure(fig, root, "SuppFigure7_retrained23_in_sample_fit")

    fig = plt.figure(figsize=(10.4, 3.5))
    grid = fig.add_gridspec(1, 3, wspace=0.58)
    retrieval_index = retrieval_results.set_index("sample_id")
    for index, sid in enumerate(TEST_IDS):
        source = retrieval_index.loc[sid, "source_sample_id"]
        show_afm(fig.add_subplot(grid[0, index]), retrieved_arrays[sid], f"{sid} retrieved\nsource {source}")
    fig.suptitle("Retrained A3 retrieved AFM maps", fontsize=12)
    save_figure(fig, root, "SuppFigure8_retrieved_maps_with_heightbars")

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    labels = [
        "Historical train",
        "Added train",
        "Predicted",
        "Ignored",
        "Excluded",
    ]
    values = [21, 2, 3, 1, 2]
    colors = ["#4C78A8", "#59A14F", "#E45756", "#BAB0AC", "#B279A2"]
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars)
    ax.set_ylabel("Sample count")
    ax.set_title("Experiment data split")
    ax.text(1, 1.0, "N6358, N6382", ha="center", fontsize=8)
    ax.text(2, 1.3, "N6342, N6389, N6390", ha="center", fontsize=8)
    ax.text(3, 0.35, "N6324", ha="center", fontsize=8)
    ax.text(4, 0.75, "6022, 6099", ha="center", fontsize=8)
    save_figure(fig, root, "SuppFigure9_data_split")


def algorithm_code_audit(reproduction: dict[str, Any]) -> dict[str, Any]:
    sources = [
        REPO / "analysis/rheed_video_afm_story/build_final_paper_freeze.py",
        REPO / "scripts/fit_afm_second_order.py",
        SOURCE / "code/generate_full_cohort_retrieval_images.py",
        FREEZE / "models/quantitative_model/full_cohort_deployment/ensemble_definition.json",
    ]
    diff = subprocess.run(
        ["git", "diff", "--name-only", "--", *[str(path.relative_to(REPO)) for path in sources]],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    audit = {
        "created_at": now(),
        "algorithm_sources": [
            {"path": rel(path), "sha256": sha256_file(path)}
            for path in sources
        ],
        "algorithm_source_git_diff_detected": bool(diff),
        "algorithm_source_git_diff_paths": diff.splitlines() if diff else [],
        "source_code_modified_for_this_experiment": False,
        "only_data_division_changed": {
            "historical_training_count": len(HISTORICAL_IDS),
            "added_training_samples": ADDED_TRAIN_IDS,
            "prediction_samples": TEST_IDS,
            "ignored_samples": IGNORED_IDS,
            "excluded_samples": EXCLUDED_IDS,
        },
        "frozen_23_numeric_reproduction": reproduction,
    }
    write_json(PKG / "provenance/algorithm_code_audit.json", audit)
    if audit["algorithm_source_git_diff_detected"]:
        raise RuntimeError(f"Algorithm source files have git diffs: {audit['algorithm_source_git_diff_paths']}")
    return audit


def write_split_table() -> None:
    rows = [
        *[
            {"sample_id": sid, "role": "historical_training", "AFM_label_used_for_fit": True}
            for sid in HISTORICAL_IDS
        ],
        {"sample_id": "N6358", "role": "added_training", "AFM_label_used_for_fit": True},
        {"sample_id": "N6382", "role": "added_training", "AFM_label_used_for_fit": True},
        {"sample_id": "N6342", "role": "prediction", "AFM_label_used_for_fit": False},
        {"sample_id": "N6389", "role": "prediction", "AFM_label_used_for_fit": False},
        {"sample_id": "N6390", "role": "prediction", "AFM_label_used_for_fit": False},
        {"sample_id": "N6324", "role": "ignored", "AFM_label_used_for_fit": False},
        {"sample_id": "6022", "role": "excluded", "AFM_label_used_for_fit": False},
        {"sample_id": "6099", "role": "excluded", "AFM_label_used_for_fit": False},
    ]
    write_frame(PKG / "tables/data_split.csv", pd.DataFrame(rows))


def write_report(
    predictions: pd.DataFrame,
    evaluation: pd.DataFrame,
    metrics: dict[str, Any],
    targets: pd.DataFrame,
    retrieval_results: pd.DataFrame,
) -> None:
    pred = predictions.set_index("sample_id")
    eval_index = evaluation.set_index("sample_id")
    target_index = targets.set_index("sample_id")
    retrieval_index = retrieval_results.set_index("sample_id")
    lines = [
        "# Result summary",
        "",
        "Samples 6022 and 6099 are excluded from every training set and AFM retrieval bank in this sensitivity experiment.",
        "",
        "The reduced historical baseline contains 21 samples. The retrained quantitative cohort contains those 21 samples plus N6358 and N6382, for 23 total. N6342, N6389, and N6390 remain outside the prospective fit and the 23-group AFM retrieval bank.",
        "",
        "## Added training targets",
        "",
        "| Sample | T4 trimmed-mean Rq (nm) | T6 quality-weighted Rq (nm) |",
        "|---|---:|---:|",
    ]
    for sid in ADDED_TRAIN_IDS:
        lines.append(
            f"| {sid} | {target_index.loc[sid, 'T4_second_order_trimmed_mean']:.4f} | "
            f"{target_index.loc[sid, 'T6_quality_weighted_second_order']:.4f} |"
        )
    lines += [
        "",
        "## Predictions and revealed AFM evaluation",
        "",
        "| Sample | Reduced 21 prediction | Retrained 23 prediction | GT T4 | Retrained abs. error | Retrieved source |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    baseline = pd.read_csv(
        PKG / "predictions/reduced_21_baseline/predictions.csv",
        dtype={"sample_id": str},
    ).set_index("sample_id")
    for sid in TEST_IDS:
        lines.append(
            f"| {sid} | {baseline.loc[sid, 'predicted_rq_nm']:.4f} | "
            f"{pred.loc[sid, 'predicted_rq_nm']:.4f} | "
            f"{eval_index.loc[sid, 'ground_truth_T4_rq_nm']:.4f} | "
            f"{eval_index.loc[sid, 'retrained_23_absolute_error_nm']:.4f} | "
            f"{retrieval_index.loc[sid, 'source_sample_id']} |"
        )
    lines += [
        "",
        f"Retrained three-sample MAE: {metrics['retrained_23']['MAE_nm']:.4f} nm.",
        f"Reduced-baseline three-sample MAE: {metrics['reduced_21_baseline']['MAE_nm']:.4f} nm.",
        "",
        "Only three prediction samples are evaluated, so these descriptive metrics must not replace the frozen strict OOF benchmark.",
        "",
        "## Primary figure",
        "",
        "`figures/main/Figure1_three_sample_prediction_atlas.png` contains, in one figure, each sample's RHEED keyframe, five upper-left-quarter ground-truth AFMs, and the retrieved AFM. Every AFM panel retains a height bar in nm and an Rq label.",
    ]
    report = PKG / "report/result_summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest() -> None:
    entries: list[str] = []
    manifest = PKG / "provenance/MANIFEST.sha256"
    for path in sorted(PKG.rglob("*")):
        if not path.is_file() or path == manifest:
            continue
        entries.append(f"{sha256_file(path)}  {path.relative_to(PKG)}")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")


def main() -> None:
    for directory in [
        PKG / "models/encoder",
        PKG / "models/visual_model",
        PKG / "predictions/retrained_23",
        PKG / "figures/individual_afm/ground_truth",
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    afm_descriptors, retrieval, second_order = load_frozen_modules()
    assets = copy_keyframe_assets()
    afm_manifest, targets, afm_arrays = build_quarter_ground_truth(afm_descriptors, second_order)
    baseline = train_reduced_historical_baseline()
    run_reduced_baseline_retrieval(baseline, retrieval)
    predictions, members, training_fit, registry, _ = train_and_predict(targets)
    bank = expanded_retrieval_bank(afm_manifest, afm_descriptors)
    retrieval_results, retrieved_arrays = run_retrieval(predictions, bank, retrieval)
    evaluation, metrics = evaluate_predictions(predictions, baseline, targets)
    render_individual_afm_maps(afm_arrays, retrieved_arrays)
    make_prediction_atlases(assets, afm_arrays, retrieved_arrays, predictions, evaluation)
    make_supplementary_figures(
        afm_arrays,
        retrieved_arrays,
        predictions,
        members,
        training_fit,
        evaluation,
        targets,
        retrieval_results,
    )
    write_split_table()
    code_audit = algorithm_code_audit(registry["frozen_23_reproduction_audit"])
    write_json(
        PKG / "provenance/run_provenance.json",
        {
            "created_at": now(),
            "experiment_id": PKG.name,
            "freeze_reference": str(FREEZE_REL),
            "selection_reference": str(SOURCE_REL),
            "configuration": rel(PKG / "config/experiment.yaml"),
            "data_split": {
                "historical_training_samples": HISTORICAL_IDS,
                "added_training_samples": ADDED_TRAIN_IDS,
                "prediction_samples": TEST_IDS,
                "ignored_samples": IGNORED_IDS,
                "excluded_samples": EXCLUDED_IDS,
            },
            "afm_crop": {
                "source_scan_size_um": [2.0, 2.0],
                "quarter": "top_left",
                "crop_bounds_xyxy": [0, 0, 256, 256],
                "effective_scan_size_um": [1.0, 1.0],
                "correction_after_crop": "unchanged robust second-order y2",
            },
            "n6390_selected_raw_files": [
                "N6390_2um_1.0_00000.spm",
                "N6390_2um_2.0_00000.spm",
                "N6390_2um_3.0_00000.spm",
                "N6390_2um_4.0_00000.spm",
                "N6390_2um_5.0_00000.spm",
            ],
            "n6390_retained_but_excluded_duplicate_number_candidate": "N6390_2um_1.0_00001.spm",
            "algorithm_code_audit": code_audit,
            "baseline_rerun_provenance": rel(
                PKG
                / "predictions/reduced_21_baseline/prediction_run_provenance.json"
            ),
            "primary_figure": rel(
                PKG / "figures/main/Figure1_three_sample_prediction_atlas.png"
            ),
        },
    )
    write_report(predictions, evaluation, metrics, targets, retrieval_results)
    write_manifest()
    print(
        json.dumps(
            {
                "status": "ok",
                "experiment_root": str(PKG),
                "training_sample_count": len(TRAIN_IDS),
                "prediction_sample_ids": TEST_IDS,
                "predictions": predictions.to_dict("records"),
                "metrics": metrics,
                "primary_figure": str(PKG / "figures/main/Figure1_three_sample_prediction_atlas.png"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
