from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .common import display_path, repo_path, sha256_object


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
VIDEO_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
VIDEO_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)


@dataclass
class EncoderStatus:
    encoder: str
    loaded: bool
    weight_identifier: str
    reason: str = ""


def frozen_parameters(model: torch.nn.Module) -> bool:
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return all(not param.requires_grad for param in model.parameters())


def preprocess_frames(frames_uint8: np.ndarray, preprocessing: str, video: bool = False) -> torch.Tensor:
    arr = frames_uint8.astype(np.float32)
    if preprocessing == "raw_luminance":
        arr = arr / 255.0
    elif preprocessing == "clip_robust_contrast":
        p01, p99 = np.percentile(arr, [1, 99])
        if not np.isfinite(p01) or not np.isfinite(p99) or p99 <= p01:
            arr = np.zeros_like(arr, dtype=np.float32)
        else:
            arr = np.clip(arr, p01, p99)
            arr = (arr - p01) / (p99 - p01)
    else:
        raise ValueError(f"Unknown preprocessing: {preprocessing}")
    tensor = torch.from_numpy(arr).float()
    if video:
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1, 1).unsqueeze(0)  # [1,C,T,H,W]
        return (tensor - VIDEO_MEAN) / VIDEO_STD
    tensor = tensor.unsqueeze(1).repeat(1, 3, 1, 1)  # [T,C,H,W]
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def temporal_aggregate(frame_embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(frame_embeddings, dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    delta = values[-1] - values[0]
    t = np.arange(values.shape[0], dtype=np.float32)
    centered_t = t - t.mean()
    denom = float(np.sum(centered_t**2))
    if denom > 0:
        slope = (centered_t[:, None] * (values - mean)).sum(axis=0) / denom
    else:
        slope = np.zeros_like(mean)
    return np.concatenate([mean, std, delta, slope], axis=0).astype(np.float32)


def load_resnet18() -> tuple[torch.nn.Module | None, EncoderStatus]:
    try:
        import torchvision.models as models

        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        ok = frozen_parameters(model)
        return model, EncoderStatus("resnet18", ok, str(weights))
    except Exception as exc:  # noqa: BLE001
        return None, EncoderStatus("resnet18", False, "torchvision.models.ResNet18_Weights.DEFAULT", repr(exc))


def load_r3d18() -> tuple[torch.nn.Module | None, EncoderStatus]:
    try:
        import torchvision.models.video as video_models

        weights = video_models.R3D_18_Weights.DEFAULT
        model = video_models.r3d_18(weights=weights)
        model.fc = torch.nn.Identity()
        ok = frozen_parameters(model)
        return model, EncoderStatus("r3d_18", ok, str(weights))
    except Exception as exc:  # noqa: BLE001
        return None, EncoderStatus("r3d_18", False, "torchvision.models.video.R3D_18_Weights.DEFAULT", repr(exc))


def load_dino() -> tuple[torch.nn.Module | None, EncoderStatus]:
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        ok = frozen_parameters(model)
        return model, EncoderStatus("dino_vits14", ok, "facebookresearch/dinov2:dinov2_vits14")
    except Exception as exc:  # noqa: BLE001
        return None, EncoderStatus("dino_vits14", False, "facebookresearch/dinov2:dinov2_vits14", repr(exc))


def _load_variant_rows(variant_manifest: pd.DataFrame, variant: str, sample_ids: list[str]) -> pd.DataFrame:
    rows = variant_manifest[
        (variant_manifest["clip_variant"] == variant)
        & (variant_manifest["available"])
        & (variant_manifest["sample_id"].astype(str).isin(sample_ids))
    ].copy()
    rows["sample_id"] = rows["sample_id"].astype(str)
    rows = rows.set_index("sample_id").loc[sample_ids].reset_index()
    return rows


@torch.no_grad()
def extract_embeddings(
    manifest: pd.DataFrame,
    variant_manifest: pd.DataFrame,
    config: dict[str, Any],
    output_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_ids = manifest["sample_id"].astype(str).tolist()
    growth_ids = manifest["growth_run_id"].astype(str).tolist()
    embed_root = output_root / "embeddings"
    embed_root.mkdir(parents=True, exist_ok=True)
    needed = {
        job["encoder"]
        for job in config["embedding_jobs"]
        if not (embed_root / f"{job['encoder']}__{job['variant']}__{job['preprocessing']}.npz").exists()
    }
    models: dict[str, torch.nn.Module | None] = {}
    statuses: dict[str, EncoderStatus] = {}
    loaders = {"resnet18": load_resnet18, "r3d_18": load_r3d18, "dino_vits14": load_dino}
    for encoder, loader in loaders.items():
        if encoder in needed:
            model, status = loader()
            statuses[status.encoder] = status
            models[status.encoder] = model.to(device) if model is not None and status.loaded else None
        else:
            statuses[encoder] = EncoderStatus(encoder, True, "cache_hit_existing_embedding")
            models[encoder] = None
    registry_rows = [
        {
            "encoder": status.encoder,
            "status": "loaded" if status.loaded else "skipped",
            "weight_identifier": status.weight_identifier,
            "skip_reason": status.reason,
            "frozen_requires_grad_false": status.loaded,
        }
        for status in statuses.values()
    ]
    embedding_rows: list[dict[str, Any]] = []
    for job in config["embedding_jobs"]:
        encoder = job["encoder"]
        variant = job["variant"]
        preprocessing = job["preprocessing"]
        out_path = embed_root / f"{encoder}__{variant}__{preprocessing}.npz"
        if out_path.exists():
            data = np.load(out_path)
            embedding_rows.append(
                {
                    "embedding_id": f"{encoder}__{variant}__{preprocessing}",
                    "encoder": encoder,
                    "clip_variant": variant,
                    "preprocessing": preprocessing,
                    "path": display_path(out_path),
                    "N": int(len(data["sample_ids"])),
                    "embedding_dim": int(data["embedding_dim"]),
                    "weight_identifier": str(data["weight_identifier"]),
                }
            )
            registry_rows.append({"encoder": encoder, "status": "embedding_cache_hit", "weight_identifier": str(data["weight_identifier"]), "skip_reason": "", "frozen_requires_grad_false": True, "clip_variant": variant, "preprocessing": preprocessing, "embedding_path": display_path(out_path)})
            continue
        model = models.get(encoder)
        if model is None:
            registry_rows.append({"encoder": encoder, "status": "job_skipped", "weight_identifier": statuses.get(encoder, EncoderStatus(encoder, False, "")).weight_identifier, "skip_reason": "encoder_not_loaded", "frozen_requires_grad_false": False, "clip_variant": variant, "preprocessing": preprocessing})
            continue
        rows = _load_variant_rows(variant_manifest, variant, sample_ids)
        embeddings: list[np.ndarray] = []
        input_shape = None
        for _, row in rows.iterrows():
            frames = np.load(repo_path(row["cache_path"]))["frames_uint8"]
            input_shape = list(frames.shape)
            if encoder == "r3d_18":
                tensor = preprocess_frames(frames, preprocessing, video=True).to(device)
                emb = model(tensor).detach().cpu().numpy()[0].astype(np.float32)
            else:
                tensor = preprocess_frames(frames, preprocessing, video=False).to(device)
                if encoder == "dino_vits14":
                    frame_emb = model(tensor).detach().cpu().numpy().astype(np.float32)
                else:
                    frame_emb = model(tensor).detach().cpu().numpy().astype(np.float32)
                emb = temporal_aggregate(frame_emb)
            embeddings.append(emb)
        matrix = np.vstack(embeddings).astype(np.float32)
        np.savez_compressed(
            out_path,
            sample_ids=np.asarray(sample_ids),
            growth_run_ids=np.asarray(growth_ids),
            embeddings=matrix,
            encoder_name=encoder,
            weight_identifier=statuses[encoder].weight_identifier,
            clip_variant=variant,
            preprocessing=preprocessing,
            input_shape=np.asarray(input_shape if input_shape is not None else []),
            embedding_dim=matrix.shape[1],
            source_manifest_hash=config["phase1_manifest_hash"],
            config_hash=sha256_object(config),
        )
        embedding_rows.append(
            {
                "embedding_id": f"{encoder}__{variant}__{preprocessing}",
                "encoder": encoder,
                "clip_variant": variant,
                "preprocessing": preprocessing,
                "path": display_path(out_path),
                "N": matrix.shape[0],
                "embedding_dim": matrix.shape[1],
                "weight_identifier": statuses[encoder].weight_identifier,
            }
        )
        registry_rows.append({"encoder": encoder, "status": "embedding_written", "weight_identifier": statuses[encoder].weight_identifier, "skip_reason": "", "frozen_requires_grad_false": True, "clip_variant": variant, "preprocessing": preprocessing, "embedding_path": display_path(out_path)})
    return pd.DataFrame(embedding_rows), pd.DataFrame(registry_rows)
