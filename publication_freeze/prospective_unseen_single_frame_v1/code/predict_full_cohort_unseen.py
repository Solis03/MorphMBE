#!/usr/bin/env python3
"""Run full-cohort frozen single-frame RHEED-to-Rq prediction for five unseen samples."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))

from keyframe_selector.common import EXPECTED_SAMPLE_IDS, atomic_write_json, load_json, package_root, repo_root_from, sha256_file, write_csv
from keyframe_selector.manifests import metadata_path
from keyframe_selector.provenance import verify_frozen_manifest


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
FREEZE_REL = Path("publication_freeze/rheed_afm_single_frame_v1_2026-07-18")
RUN_REL = Path("publication_freeze/prospective_unseen_single_frame_v1/predictions/full_cohort_single_frame_v1")
OUTPUT_SIZE = 224


def luminance_uint8(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def resize_and_pad(crop: np.ndarray, output_size: int = OUTPUT_SIZE) -> tuple[np.ndarray, float, tuple[int, int, int, int]]:
    height, width = crop.shape
    scale = output_size / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = np.asarray(Image.fromarray(crop).resize((new_width, new_height), Image.Resampling.BILINEAR))
    canvas = np.zeros((output_size, output_size), dtype=np.uint8)
    pad_left = (output_size - new_width) // 2
    pad_top = (output_size - new_height) // 2
    canvas[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = resized
    pad_right = output_size - new_width - pad_left
    pad_bottom = output_size - new_height - pad_top
    return canvas, float(scale), (pad_top, pad_bottom, pad_left, pad_right)


def preprocess_roi_crop(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(path) as image:
        gray = luminance_uint8(image)
    frame, scale, padding = resize_and_pad(gray)
    return frame[None, :, :], {
        "roi_crop_shape_hw": [int(gray.shape[0]), int(gray.shape[1])],
        "model_frame_shape": [1, OUTPUT_SIZE, OUTPUT_SIZE],
        "resize_scale": scale,
        "padding_top_bottom_left_right": list(map(int, padding)),
        "preprocessing": "manual ROI crop -> luminance_uint8 -> bilinear resize preserving aspect -> zero pad to 224x224",
    }


def preprocess_frames_for_dino(frames_uint8: np.ndarray) -> torch.Tensor:
    arr = frames_uint8.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).float().unsqueeze(1).repeat(1, 3, 1, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def temporal_aggregate(frame_embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(frame_embeddings, dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    delta = values[-1] - values[0]
    t = np.arange(values.shape[0], dtype=np.float32)
    centered_t = t - t.mean()
    denom = float(np.sum(centered_t**2))
    slope = np.zeros_like(mean) if denom <= 0 else (centered_t[:, None] * (values - mean)).sum(axis=0) / denom
    return np.concatenate([mean, std, delta, slope], axis=0).astype(np.float32)


def load_dino() -> tuple[torch.nn.Module, torch.device]:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return model.to(device), device


@torch.no_grad()
def embed_frames(model: torch.nn.Module, device: torch.device, frames_uint8: np.ndarray) -> np.ndarray:
    tensor = preprocess_frames_for_dino(frames_uint8).to(device)
    frame_embeddings = model(tensor).detach().cpu().numpy().astype(np.float32)
    return temporal_aggregate(frame_embeddings)


def load_models(freeze_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    model_root = freeze_root / "models" / "quantitative_model" / "full_cohort_deployment"
    ensemble = load_json(model_root / "ensemble_definition.json")
    expected_ids = [row["sample_id"] for row in read_csv(model_root / "training_sample_ids.csv")]
    if len(expected_ids) != 23:
        raise RuntimeError(f"Expected 23 training sample IDs, found {len(expected_ids)}")
    members = []
    member_defs = []
    for member in ensemble["members"]:
        path = model_root / f"{member['name']}.npz"
        data = np.load(path, allow_pickle=False)
        ids = [str(x) for x in data["training_sample_ids"].tolist()]
        if ids != expected_ids:
            raise RuntimeError(f"{path.name} training IDs do not match full 23-sample cohort")
        members.append({
            "name": member["name"],
            "trial_id": member["trial_id"],
            "target_variant": str(data["target_variant"].item()),
            "coef": np.asarray(data["coef"], dtype=float),
            "intercept": float(data["intercept"]),
            "feature_mean": np.asarray(data["feature_mean"], dtype=float),
            "feature_scale": np.asarray(data["feature_scale"], dtype=float),
            "path": path,
            "sha256": sha256_file(path),
        })
        member_defs.append(member)
    return member_defs, members, expected_ids


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def predict_members(embedding: np.ndarray, members: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    x = np.asarray(embedding, dtype=float)
    for member in members:
        z = (x - member["feature_mean"]) / np.maximum(member["feature_scale"], 1e-12)
        pred = float(np.dot(z, member["coef"]) + member["intercept"])
        rows.append({
            "member_name": member["name"],
            "trial_id": member["trial_id"],
            "target_variant": member["target_variant"],
            "predicted_rq_nm": pred,
            "model_sha256": member["sha256"],
        })
    return rows


def sanity_check_dino(model: torch.nn.Module, device: torch.device, freeze_root: Path) -> dict[str, Any]:
    frame_path = freeze_root / "data_snapshot" / "selected_rheed_keyframes" / "6022_keyframe_1_raw_luminance.npz"
    bank_path = freeze_root / "models" / "encoder" / "dino_vits14_keyframe_1_raw_luminance_embeddings.npz"
    frames = np.load(frame_path, allow_pickle=False)["frames_uint8"]
    observed = embed_frames(model, device, frames)
    bank = np.load(bank_path, allow_pickle=False)
    ids = [str(x) for x in bank["sample_ids"].tolist()]
    expected = bank["embeddings"][ids.index("6022")]
    diff = np.abs(observed - expected)
    return {
        "sample_id": "6022",
        "embedding_dim": int(observed.shape[0]),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "allclose_atol_rtol_1e_4": bool(np.allclose(observed, expected, atol=1e-4, rtol=1e-4)),
        "reference_embedding_path": str(bank_path),
    }


def ensure_ready(pkg_root: Path) -> list[dict[str, Any]]:
    rows = []
    for sample_id in EXPECTED_SAMPLE_IDS:
        payload = load_json(metadata_path(pkg_root, sample_id))
        if payload.get("sample", {}).get("selection_status") != "completed":
            raise RuntimeError(f"{sample_id} is not completed")
        selection = payload.get("selection") or {}
        for key in ["roi_keyframe_png", "roi", "roi_xyxy", "raw_keyframe_png", "selected_frame_index_0based"]:
            if selection.get(key) in (None, ""):
                raise RuntimeError(f"{sample_id} missing selection.{key}")
        rows.append(payload)
    return rows


def main() -> None:
    repo_root = repo_root_from(THIS)
    pkg_root = package_root(repo_root)
    freeze_root = repo_root / FREEZE_REL
    run_root = repo_root / RUN_REL
    model_ready_root = pkg_root / "keyframes" / "model_ready"
    run_root.mkdir(parents=True, exist_ok=True)
    model_ready_root.mkdir(parents=True, exist_ok=True)

    frozen_check = verify_frozen_manifest(repo_root)
    if frozen_check["status"] != "ok":
        raise RuntimeError("Frozen package manifest verification failed")
    sample_payloads = ensure_ready(pkg_root)
    member_defs, models, training_ids = load_models(freeze_root)
    model, device = load_dino()
    dino_check = sanity_check_dino(model, device, freeze_root)
    if not dino_check["allclose_atol_rtol_1e_4"]:
        raise RuntimeError(f"DINO sanity check failed: {dino_check}")

    prediction_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    embedding_rows: list[dict[str, Any]] = []
    embeddings = []
    sample_ids = []
    for payload in sample_payloads:
        sample_id = payload["sample"]["sample_id"]
        selection = payload["selection"]
        roi_crop = repo_root / selection["roi_keyframe_png"]
        frames_uint8, preprocess_info = preprocess_roi_crop(roi_crop)
        model_ready_path = model_ready_root / f"{sample_id}_keyframe_1_raw_luminance.npz"
        np.savez_compressed(
            model_ready_path,
            frames_uint8=frames_uint8.astype(np.uint8),
            frame_indices=np.asarray([int(selection["selected_frame_index_0based"])], dtype=np.int32),
            sample_id=sample_id,
            clip_variant="keyframe_1",
            source_roi_keyframe_png=selection["roi_keyframe_png"],
            preprocessing=preprocess_info["preprocessing"],
        )
        embedding = embed_frames(model, device, frames_uint8)
        embeddings.append(embedding)
        sample_ids.append(sample_id)
        member_preds = predict_members(embedding, models)
        values = np.asarray([row["predicted_rq_nm"] for row in member_preds], dtype=float)
        q50 = float(np.median(values))
        q10 = float(np.quantile(values, 0.10))
        q90 = float(np.quantile(values, 0.90))
        for row in member_preds:
            member_rows.append({"sample_id": sample_id, **row})
        prediction_rows.append({
            "sample_id": sample_id,
            "sample_id_numeric": payload["sample"]["sample_id_numeric"],
            "predicted_rq_nm": q50,
            "q50_predicted_rq_nm": q50,
            "predicted_rq_nm_clipped_nonnegative": max(0.0, q50),
            "physical_plausibility_flag": "negative_raw_model_output" if q50 < 0 else "ok",
            "member_q10_rq_nm": q10,
            "member_q90_rq_nm": q90,
            "member_min_rq_nm": float(values.min()),
            "member_max_rq_nm": float(values.max()),
            "ensemble_member_count": len(values),
            "interval_note": "member_q10/member_q90 are ensemble spread only, not calibrated prediction intervals",
            "training_sample_count": len(training_ids),
            "training_sample_ids": json.dumps(training_ids),
            "uses_all_23_training_samples": len(training_ids) == 23,
            "raw_keyframe_png": selection["raw_keyframe_png"],
            "roi_keyframe_png": selection["roi_keyframe_png"],
            "model_ready_keyframe_npz": str(model_ready_path.relative_to(repo_root)),
            "roi_xyxy": json.dumps(selection["roi_xyxy"]),
            "display_transform": selection.get("display_transform") or payload.get("source_video", {}).get("display_transform", "none"),
            "selected_frame_index_0based": selection["selected_frame_index_0based"],
            "selected_timestamp_sec": selection["selected_timestamp_sec"],
        })
        embedding_rows.append({
            "sample_id": sample_id,
            "embedding_dim": int(embedding.shape[0]),
            "embedding_sha256": sha256_array(embedding),
            "model_ready_keyframe_sha256": sha256_file(model_ready_path),
        })

    embedding_matrix = np.vstack(embeddings).astype(np.float32)
    embedding_path = run_root / "unseen_dino_vits14_keyframe_1_raw_luminance_embeddings.npz"
    np.savez_compressed(
        embedding_path,
        sample_ids=np.asarray(sample_ids),
        embeddings=embedding_matrix,
        encoder_name="dino_vits14",
        weight_identifier="facebookresearch/dinov2:dinov2_vits14",
        clip_variant="keyframe_1",
        preprocessing="raw_luminance",
        embedding_dim=embedding_matrix.shape[1],
    )
    write_csv(run_root / "predictions.csv", prediction_rows, list(prediction_rows[0].keys()))
    write_csv(run_root / "ensemble_member_predictions.csv", member_rows, list(member_rows[0].keys()))
    write_csv(run_root / "embedding_manifest.csv", embedding_rows, list(embedding_rows[0].keys()))
    atomic_write_json(run_root / "predictions.json", {"rows": prediction_rows})
    atomic_write_json(
        run_root / "prediction_run_provenance.json",
        {
            "schema_version": "prospective-full-cohort-single-frame-prediction-v1",
            "freeze_path": str(FREEZE_REL),
            "selection_package": "publication_freeze/prospective_unseen_single_frame_v1",
            "model_root": str(FREEZE_REL / "models" / "quantitative_model" / "full_cohort_deployment"),
            "encoder": "DINOv2 ViT-S/14",
            "encoder_weight_identifier": "facebookresearch/dinov2:dinov2_vits14",
            "device": str(device),
            "dino_sanity_check": dino_check,
            "frozen_manifest_check": frozen_check,
            "training_sample_count": len(training_ids),
            "training_sample_ids": training_ids,
            "uses_all_23_training_samples": len(training_ids) == 23,
            "ensemble_definition": member_defs,
            "prediction_target": "T4_second_order_trimmed_mean q50 via median of frozen full-cohort deployment members",
            "uncertainty_note": "member_q10/member_q90 are ensemble spread only, not calibrated prediction intervals",
            "afm_labels_used_for_unseen": False,
            "model_ready_keyframe_dir": str(model_ready_root.relative_to(repo_root)),
            "embedding_path": str(embedding_path.relative_to(repo_root)),
        },
    )
    print(f"wrote {run_root / 'predictions.csv'}")
    print(json.dumps({"rows": prediction_rows}, indent=2))


def sha256_array(array: np.ndarray) -> str:
    import hashlib

    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(array)
    digest.update(contiguous.tobytes())
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(contiguous.shape).encode("utf-8"))
    return digest.hexdigest()


if __name__ == "__main__":
    main()
