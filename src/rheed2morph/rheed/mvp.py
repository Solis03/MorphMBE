#!/usr/bin/env python3
"""Frozen-encoder MVP for predicting AFM descriptors from RHEED videos."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import matplotlib
import numpy as np
import torch
from PIL import Image
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAIR_ROOT = REPO_ROOT / "data" / "pair"
DEFAULT_DESCRIPTOR_CSV = (
    REPO_ROOT
    / "data"
    / "afm_descriptor_reconstruction"
    / "selected_descriptors"
    / "selected_descriptor_table.csv"
)
DEFAULT_DESCRIPTOR_AUX_CSV = (
    REPO_ROOT / "data" / "afm_descriptor_reconstruction" / "afm_descriptors.csv"
)
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "rheed_descriptor_mvp"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "rheed_descriptor_mvp"
DEFAULT_PROCESSED_RHEED_ROOT = REPO_ROOT / "data" / "raw_RHEED_selected_test_512"
VIDEO_SUFFIXES = {".mov", ".mp4", ".avi", ".mkv", ".m4v", ".mts", ".m2ts"}
ID_COLUMNS = {"row_id", "sample_id", "afm_path"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


@dataclass(frozen=True)
class VideoCandidate:
    sample_id: str
    path: Path
    duration_seconds: float
    frame_count: int
    contains_main: bool


@dataclass
class SampleEmbeddingRecord:
    sample_id: str
    video_path: Path
    selection_reason: str
    duration_seconds: float
    decoded_frame_count: int
    sampled_frame_count: int
    embedding: np.ndarray


@dataclass
class JoinedDataset:
    row_ids: list[str]
    sample_ids: list[str]
    group_ids: list[str]
    afm_paths: list[str]
    network_input_paths: list[str]
    feature_names: list[str]
    target_names: list[str]
    x: np.ndarray
    y: np.ndarray


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def resolve_existing_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.exists():
        return expanded.resolve()
    repo_relative = REPO_ROOT / expanded
    if repo_relative.exists():
        return repo_relative.resolve()
    return expanded.resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating | np.integer):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(child) for child in value]
    return value


def infer_target_columns(rows: Sequence[dict[str, str]]) -> list[str]:
    if not rows:
        raise ValueError("Descriptor CSV is empty.")
    target_columns: list[str] = []
    for column in rows[0]:
        if column in ID_COLUMNS:
            continue
        values: list[float] = []
        valid = True
        for row in rows:
            raw = row.get(column, "").strip()
            if raw == "":
                valid = False
                break
            try:
                values.append(float(raw))
            except ValueError:
                valid = False
                break
        if valid and values:
            target_columns.append(column)
    if not target_columns:
        raise ValueError("Could not infer any numeric target descriptor columns.")
    return target_columns


def is_visible_video_file(path: Path) -> bool:
    return path.is_file() and not path.name.startswith("._") and path.suffix.lower() in VIDEO_SUFFIXES


def list_sample_directories(pair_root: Path) -> list[Path]:
    return [path for path in sorted(pair_root.iterdir()) if path.is_dir()]


def list_sample_video_files(sample_root: Path) -> list[Path]:
    rheed_root = sample_root / "RHEED"
    if not rheed_root.is_dir():
        return []
    return [path for path in sorted(rheed_root.iterdir()) if is_visible_video_file(path)]


def require_video_backend() -> Any:
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing video dependency `imageio-ffmpeg`. Install project dependencies "
            "before running the RHEED descriptor MVP."
        ) from exc
    return imageio_ffmpeg


def probe_video_candidate(sample_id: str, path: Path) -> VideoCandidate | None:
    imageio_ffmpeg = require_video_backend()
    try:
        frame_count, duration_seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
    except Exception:
        return None
    if frame_count <= 0 or not np.isfinite(duration_seconds) or duration_seconds <= 0:
        return None
    return VideoCandidate(
        sample_id=sample_id,
        path=path.resolve(),
        duration_seconds=float(duration_seconds),
        frame_count=int(frame_count),
        contains_main="main" in path.stem.lower(),
    )


def choose_canonical_video(candidates: Sequence[VideoCandidate]) -> tuple[VideoCandidate | None, str]:
    if not candidates:
        return None, "no decodable video"
    ordered = sorted(
        candidates,
        key=lambda item: (
            1 if item.contains_main else 0,
            item.duration_seconds,
            item.frame_count,
            item.path.name.lower(),
        ),
        reverse=True,
    )
    selected = ordered[0]
    reason = "contains_main" if selected.contains_main else "longest_decodable"
    return selected, reason


def decode_video_frames(path: Path) -> np.ndarray:
    imageio_ffmpeg = require_video_backend()
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    metadata = next(reader)
    width, height = metadata.get("source_size", metadata["size"])
    frames: list[np.ndarray] = []
    for raw_frame in reader:
        frame = np.frombuffer(raw_frame, dtype=np.uint8)
        frame = frame.reshape((height, width, 3))
        frames.append(frame.copy())
    if not frames:
        raise RuntimeError(f"Decoded zero frames from {path}.")
    return np.stack(frames, axis=0)


def sample_uniform_frames(frames: np.ndarray, frame_count: int) -> np.ndarray:
    if frames.ndim != 4:
        raise ValueError(f"Expected video frames with shape [T, H, W, C], got {frames.shape}")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    indices = sample_uniform_indices(frames.shape[0], frame_count)
    return frames[indices]


def sample_uniform_indices(length: int, target_count: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive.")
    if target_count <= 0:
        raise ValueError("target_count must be positive.")
    if length <= target_count:
        return np.arange(length, dtype=int)
    return np.linspace(0, length - 1, target_count, dtype=int)


def sample_uniform_sequence(sequence: np.ndarray, target_count: int) -> np.ndarray:
    array = np.asarray(sequence)
    if array.ndim < 1:
        raise ValueError(f"Expected an array with at least one dimension, got {array.shape}")
    return array[sample_uniform_indices(array.shape[0], target_count)]


def normalize_frame_for_rgb(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        min_value = float(np.min(array))
        max_value = float(np.max(array))
        if min_value >= 0.0 and max_value <= 1.0 + 1e-6:
            array = np.clip(array, 0.0, 1.0) * 255.0
        elif max_value > min_value:
            array = (array - min_value) / (max_value - min_value)
            array = np.clip(array, 0.0, 1.0) * 255.0
        else:
            array = np.zeros_like(array, dtype=np.float32)
    else:
        array = np.clip(array, 0, 255)
    return array.astype(np.uint8)


def frame_to_chw_tensor(frame: np.ndarray, image_size: int) -> torch.Tensor:
    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.ndim == 3 and array.shape[2] >= 3:
        array = array[..., :3]
    else:
        raise ValueError(f"Unsupported frame shape: {array.shape}")

    image = Image.fromarray(normalize_frame_for_rgb(array), mode="RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    chw = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
    return (chw - IMAGENET_MEAN) / IMAGENET_STD


def preprocess_frames(frames: np.ndarray, image_size: int) -> torch.Tensor:
    tensors = [frame_to_chw_tensor(frame, image_size) for frame in frames]
    return torch.stack(tensors, dim=0)


def aggregate_frame_embeddings(frame_embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(frame_embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"Expected [num_frames, feature_dim] frame embeddings, got {array.shape}")
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    return np.concatenate([mean, std], axis=0).astype(np.float32)


def aggregate_temporal_frame_embeddings(frame_embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(frame_embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"Expected [num_frames, feature_dim] frame embeddings, got {array.shape}")
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    if array.shape[0] >= 2:
        deltas = np.diff(array, axis=0)
        delta_mean = np.mean(deltas, axis=0)
        delta_std = np.std(deltas, axis=0)
    else:
        delta_mean = np.zeros_like(mean)
        delta_std = np.zeros_like(std)
    return np.concatenate([mean, std, delta_mean, delta_std], axis=0).astype(np.float32)


def load_pretrained_encoder(
    backend: str = "auto",
    device_name: str = "auto",
) -> tuple[torch.nn.Module, str, int]:
    device = resolve_torch_device(device_name)
    last_error: Exception | None = None

    if backend in {"auto", "torchvision"}:
        try:
            from torchvision import models

            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            encoder = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
            encoder.eval()
            return encoder, "torchvision", 224
        except Exception as exc:  # pragma: no cover - environment dependent.
            last_error = exc
            if backend == "torchvision":
                raise RuntimeError("Failed to load torchvision ResNet-50 encoder.") from exc

    if backend in {"auto", "torchhub"}:
        model = None
        hub_attempts = [
            ("pytorch/vision:v0.20.1", {"weights": "DEFAULT"}),
            ("pytorch/vision:v0.20.1", {"pretrained": True}),
            ("pytorch/vision", {"weights": "DEFAULT"}),
            ("pytorch/vision", {"pretrained": True}),
        ]
        for repo, kwargs in hub_attempts:
            try:
                model = torch.hub.load(repo, "resnet50", **kwargs)
                break
            except Exception as exc:  # pragma: no cover - network dependent.
                last_error = exc
        if model is not None:
            encoder = torch.nn.Sequential(*list(model.children())[:-1]).to(device)
            encoder.eval()
            return encoder, "torchhub", 224
        if backend == "torchhub":
            raise RuntimeError("Failed to load torch.hub ResNet-50 encoder.") from last_error

    raise RuntimeError(
        "Could not load a pretrained ResNet-50 encoder. "
        "Install torchvision or allow torch.hub model download."
    ) from last_error


def resolve_torch_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def extract_sample_embedding(
    video_path: Path,
    encoder: torch.nn.Module,
    image_size: int,
    frame_count: int,
    batch_size: int,
    device_name: str,
    aggregation_mode: str = "mean_std",
) -> tuple[np.ndarray, int, int]:
    frames = decode_video_frames(video_path)
    sampled = sample_uniform_frames(frames, frame_count)
    return extract_embedding_from_frames(
        sampled_frames=sampled,
        full_frame_count=int(frames.shape[0]),
        encoder=encoder,
        image_size=image_size,
        batch_size=batch_size,
        device_name=device_name,
        aggregation_mode=aggregation_mode,
    )


def extract_embedding_from_frames(
    sampled_frames: np.ndarray,
    full_frame_count: int,
    encoder: torch.nn.Module,
    image_size: int,
    batch_size: int,
    device_name: str,
    aggregation_mode: str,
) -> tuple[np.ndarray, int, int]:
    batch = preprocess_frames(sampled_frames, image_size)
    device = resolve_torch_device(device_name)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, batch.shape[0], batch_size):
            chunk = batch[start : start + batch_size].to(device)
            features = encoder(chunk).flatten(1).detach().cpu().numpy()
            outputs.append(features.astype(np.float32))
    frame_embeddings = np.concatenate(outputs, axis=0)
    if aggregation_mode == "mean_std":
        embedding = aggregate_frame_embeddings(frame_embeddings)
    elif aggregation_mode == "temporal_stats":
        embedding = aggregate_temporal_frame_embeddings(frame_embeddings)
    else:
        raise ValueError(f"Unsupported aggregation_mode: {aggregation_mode}")
    return embedding, int(full_frame_count), int(sampled_frames.shape[0])


def processed_dir_matches_sample_id(sample_dir_name: str, sample_id: str) -> bool:
    pattern = rf"(^|[^0-9]){re.escape(sample_id)}([^0-9]|$)"
    return re.search(pattern, sample_dir_name) is not None


def resolve_processed_sample_dir(sample_id: str, processed_rheed_root: Path) -> Path:
    matches = [
        path
        for path in sorted(processed_rheed_root.iterdir())
        if path.is_dir() and processed_dir_matches_sample_id(path.name, sample_id)
    ]
    if not matches:
        raise FileNotFoundError(f"No processed RHEED directory matched sample_id {sample_id}.")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple processed RHEED directories matched sample_id {sample_id}: "
            + ", ".join(path.name for path in matches)
        )
    return matches[0]


def resolve_processed_model_input_path(
    sample_id: str,
    processed_rheed_root: Path,
    sample_map: str,
) -> Path:
    if sample_map != "manifest_sample_id_to_dataset_dir":
        raise ValueError(f"Unsupported processed sample map: {sample_map}")
    dataset_dir = resolve_processed_sample_dir(sample_id, processed_rheed_root)
    model_input_path = dataset_dir / "tensors" / "model_input.npz"
    if not model_input_path.is_file():
        raise FileNotFoundError(f"Missing processed model input for sample_id {sample_id}: {model_input_path}")
    return model_input_path


def resolve_processed_video_path(
    sample_id: str,
    processed_rheed_root: Path,
    sample_map: str,
) -> Path:
    if sample_map != "manifest_sample_id_to_dataset_dir":
        raise ValueError(f"Unsupported processed sample map: {sample_map}")
    dataset_dir = resolve_processed_sample_dir(sample_id, processed_rheed_root)
    video_dir = dataset_dir / "videos"
    search_dir = video_dir if video_dir.is_dir() else dataset_dir
    video_files = [path for path in sorted(search_dir.iterdir()) if is_visible_video_file(path)]
    raw_crop_files = [path for path in video_files if "raw_crop" in path.stem.lower()]
    selected_files = raw_crop_files if raw_crop_files else video_files
    if not selected_files:
        raise FileNotFoundError(f"Missing processed crop video for sample_id {sample_id}: {search_dir}")
    if len(selected_files) > 1:
        raise ValueError(
            f"Multiple processed crop videos matched sample_id {sample_id}: "
            + ", ".join(path.name for path in selected_files)
        )
    return selected_files[0]


def load_processed_model_input(
    path: Path,
    frame_key: str = "clean_frames",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    resolved = resolve_existing_path(path)
    with np.load(resolved) as payload:
        if frame_key not in payload:
            raise ValueError(f"Missing frame key `{frame_key}` in {resolved}")
        if "valid_mask" not in payload:
            raise ValueError(f"Missing `valid_mask` in {resolved}")
        frames = np.asarray(payload[frame_key], dtype=np.float32)
        valid_mask = np.asarray(payload["valid_mask"], dtype=bool)
        metadata = {
            name: np.asarray(payload[name])
            for name in payload.files
            if name not in {frame_key, "valid_mask"}
        }
    if frames.ndim != 3:
        raise ValueError(f"Expected processed frames [T, H, W], got {frames.shape} at {resolved}")
    if frames.shape[0] == 0:
        raise ValueError(f"Processed frames are empty at {resolved}")
    if valid_mask.ndim != 2:
        raise ValueError(f"Expected valid_mask [H, W], got {valid_mask.shape} at {resolved}")
    if frames.shape[1:] != valid_mask.shape:
        raise ValueError(
            f"Processed frame shape {frames.shape[1:]} does not match valid_mask {valid_mask.shape} at {resolved}"
        )
    frames = np.nan_to_num(frames, nan=0.0, posinf=1.0, neginf=0.0)
    frames = np.clip(frames, 0.0, 1.0)
    frames = np.where(valid_mask[None, :, :], frames, 0.0).astype(np.float32)
    return frames, valid_mask.astype(bool), metadata


def load_processed_sample_duration_seconds(model_input_path: Path) -> float:
    metadata_path = model_input_path.parents[1] / "metadata.json"
    if not metadata_path.is_file():
        return 0.0
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return 0.0
    raw_value = payload.get("duration_sec", 0.0)
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return 0.0


def extract_processed_sample_embedding(
    model_input_path: Path,
    encoder: torch.nn.Module,
    image_size: int,
    frame_key: str,
    max_frames: int,
    batch_size: int,
    device_name: str,
) -> tuple[np.ndarray, int, int]:
    frames, _, _ = load_processed_model_input(model_input_path, frame_key=frame_key)
    sampled = sample_uniform_sequence(frames, max_frames)
    return extract_embedding_from_frames(
        sampled_frames=sampled,
        full_frame_count=int(frames.shape[0]),
        encoder=encoder,
        image_size=image_size,
        batch_size=batch_size,
        device_name=device_name,
        aggregation_mode="temporal_stats",
    )


def collect_sample_embeddings(
    pair_root: Path,
    encoder_backend: str,
    device_name: str,
    frame_count: int,
    batch_size: int,
    rheed_input_mode: str = "raw_video",
    processed_rheed_root: Path | None = None,
    processed_frame_key: str = "clean_frames",
    processed_max_frames: int = 64,
    processed_sample_map: str = "manifest_sample_id_to_dataset_dir",
    selected_rheed_paths_by_sample: dict[str, Path] | None = None,
    manifest_sample_ids: Sequence[str] | None = None,
) -> tuple[dict[str, SampleEmbeddingRecord], list[dict[str, Any]], dict[str, Any]]:
    encoder, resolved_backend, image_size = load_pretrained_encoder(
        backend=encoder_backend,
        device_name=device_name,
    )
    sample_records: dict[str, SampleEmbeddingRecord] = {}
    skipped_rows: list[dict[str, Any]] = []
    summary = {
        "encoder_backend_requested": encoder_backend,
        "encoder_backend_resolved": resolved_backend,
        "image_size": image_size,
        "frame_count_requested": frame_count,
        "rheed_input_mode": rheed_input_mode,
    }

    if rheed_input_mode == "processed_npz":
        if processed_rheed_root is None:
            raise ValueError("processed_rheed_root is required when rheed_input_mode=processed_npz")
        requested_sample_ids = (
            sorted({sample_id.strip() for sample_id in manifest_sample_ids})
            if manifest_sample_ids is not None
            else [path.name for path in list_sample_directories(pair_root)]
        )
        summary["processed_rheed_root"] = display_path(processed_rheed_root)
        summary["processed_frame_key"] = processed_frame_key
        summary["processed_max_frames"] = int(processed_max_frames)
        summary["processed_sample_map"] = processed_sample_map
        summary["sample_count_requested"] = len(requested_sample_ids)
        summary["sample_count_mapped"] = 0
        summary["sample_count_mapping_failed"] = 0

        for sample_id in requested_sample_ids:
            try:
                model_input_path = resolve_processed_model_input_path(
                    sample_id=sample_id,
                    processed_rheed_root=processed_rheed_root,
                    sample_map=processed_sample_map,
                )
            except Exception as exc:
                skipped_rows.append(
                    {
                        "sample_id": sample_id,
                        "reason": f"processed mapping failed: {exc}",
                        "video_path": "",
                    }
                )
                summary["sample_count_mapping_failed"] += 1
                continue

            summary["sample_count_mapped"] += 1
            try:
                embedding, decoded_frame_count, sampled_frame_count = extract_processed_sample_embedding(
                    model_input_path=model_input_path,
                    encoder=encoder,
                    image_size=image_size,
                    frame_key=processed_frame_key,
                    max_frames=processed_max_frames,
                    batch_size=batch_size,
                    device_name=device_name,
                )
            except Exception as exc:
                skipped_rows.append(
                    {
                        "sample_id": sample_id,
                        "reason": f"embedding failed: {exc}",
                        "video_path": str(model_input_path),
                    }
                )
                continue

            sample_records[sample_id] = SampleEmbeddingRecord(
                sample_id=sample_id,
                video_path=model_input_path,
                selection_reason=f"processed_npz:{processed_sample_map}",
                duration_seconds=load_processed_sample_duration_seconds(model_input_path),
                decoded_frame_count=decoded_frame_count,
                sampled_frame_count=sampled_frame_count,
                embedding=embedding,
            )

        summary["sample_count_embedded"] = len(sample_records)
        summary["sample_count_skipped"] = len(skipped_rows)
        return sample_records, skipped_rows, summary

    if rheed_input_mode == "processed_video":
        if processed_rheed_root is None:
            raise ValueError("processed_rheed_root is required when rheed_input_mode=processed_video")
        requested_sample_ids = (
            sorted({sample_id.strip() for sample_id in manifest_sample_ids})
            if manifest_sample_ids is not None
            else [path.name for path in list_sample_directories(processed_rheed_root)]
        )
        summary["processed_rheed_root"] = display_path(processed_rheed_root)
        summary["processed_max_frames"] = int(processed_max_frames)
        summary["processed_sample_map"] = processed_sample_map
        summary["sample_count_requested"] = len(requested_sample_ids)
        summary["sample_count_mapped"] = 0
        summary["sample_count_mapping_failed"] = 0

        for sample_id in requested_sample_ids:
            try:
                video_path = resolve_processed_video_path(
                    sample_id=sample_id,
                    processed_rheed_root=processed_rheed_root,
                    sample_map=processed_sample_map,
                )
            except Exception as exc:
                skipped_rows.append(
                    {
                        "sample_id": sample_id,
                        "reason": f"processed video mapping failed: {exc}",
                        "video_path": "",
                    }
                )
                summary["sample_count_mapping_failed"] += 1
                continue

            summary["sample_count_mapped"] += 1
            candidate = probe_video_candidate(sample_id, video_path)
            if candidate is None:
                skipped_rows.append(
                    {
                        "sample_id": sample_id,
                        "reason": "processed crop video is not decodable",
                        "video_path": str(video_path),
                    }
                )
                continue
            try:
                embedding, decoded_frame_count, sampled_frame_count = extract_sample_embedding(
                    candidate.path,
                    encoder=encoder,
                    image_size=image_size,
                    frame_count=processed_max_frames,
                    batch_size=batch_size,
                    device_name=device_name,
                    aggregation_mode="temporal_stats",
                )
            except Exception as exc:
                skipped_rows.append(
                    {
                        "sample_id": sample_id,
                        "reason": f"embedding failed: {exc}",
                        "video_path": str(candidate.path),
                    }
                )
                continue

            sample_records[sample_id] = SampleEmbeddingRecord(
                sample_id=sample_id,
                video_path=candidate.path,
                selection_reason=f"processed_video:{processed_sample_map}",
                duration_seconds=candidate.duration_seconds,
                decoded_frame_count=decoded_frame_count,
                sampled_frame_count=sampled_frame_count,
                embedding=embedding,
            )

        summary["sample_count_embedded"] = len(sample_records)
        summary["sample_count_skipped"] = len(skipped_rows)
        return sample_records, skipped_rows, summary

    for sample_root in list_sample_directories(pair_root):
        sample_id = sample_root.name
        if selected_rheed_paths_by_sample is not None and sample_id not in selected_rheed_paths_by_sample:
            continue
        if selected_rheed_paths_by_sample is not None:
            video_files = [selected_rheed_paths_by_sample[sample_id]]
        else:
            video_files = list_sample_video_files(sample_root)
        if not video_files:
            skipped_rows.append(
                {"sample_id": sample_id, "reason": "no visible video file", "video_path": ""}
            )
            continue

        decodable_candidates: list[VideoCandidate] = []
        for video_path in video_files:
            candidate = probe_video_candidate(sample_id, video_path)
            if candidate is not None:
                decodable_candidates.append(candidate)

        if selected_rheed_paths_by_sample is not None:
            selected = decodable_candidates[0] if decodable_candidates else None
            selection_reason = "manifest_specified_rheed_path"
        else:
            selected, selection_reason = choose_canonical_video(decodable_candidates)
        if selected is None:
            skipped_rows.append(
                {
                    "sample_id": sample_id,
                    "reason": selection_reason,
                    "video_path": ";".join(str(path) for path in video_files),
                }
            )
            continue

        try:
            embedding, decoded_frame_count, sampled_frame_count = extract_sample_embedding(
                selected.path,
                encoder=encoder,
                image_size=image_size,
                frame_count=frame_count,
                batch_size=batch_size,
                device_name=device_name,
            )
        except Exception as exc:
            skipped_rows.append(
                {
                    "sample_id": sample_id,
                    "reason": f"embedding failed: {exc}",
                    "video_path": str(selected.path),
                }
            )
            continue

        sample_records[sample_id] = SampleEmbeddingRecord(
            sample_id=sample_id,
            video_path=selected.path,
            selection_reason=selection_reason,
            duration_seconds=selected.duration_seconds,
            decoded_frame_count=decoded_frame_count,
            sampled_frame_count=sampled_frame_count,
            embedding=embedding,
        )

    summary["sample_count_embedded"] = len(sample_records)
    summary["sample_count_skipped"] = len(skipped_rows)
    return sample_records, skipped_rows, summary


def embedding_feature_names(embedding_dim: int) -> list[str]:
    return [f"embedding_{index:04d}" for index in range(embedding_dim)]


def load_one_to_one_manifest(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    by_sample: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row["sample_id"].strip()
        if sample_id in by_sample:
            previous = by_sample[sample_id]
            if (
                previous.get("group_id", sample_id) != row.get("group_id", sample_id)
                or previous.get("afm_path", "") != row.get("afm_path", "")
                or previous.get("rheed_path", "") != row.get("rheed_path", "")
            ):
                raise ValueError(
                    "One-to-one manifest contains repeated sample_id with conflicting selection: "
                    f"{sample_id}"
                )
        by_sample[sample_id] = row
    return list(by_sample.values())


def selected_rheed_paths_by_sample(manifest_rows: Sequence[dict[str, str]]) -> dict[str, Path]:
    return {
        row["sample_id"].strip(): resolve_existing_path(Path(row["rheed_path"]))
        for row in manifest_rows
    }


def build_joined_dataset(
    descriptor_rows: Sequence[dict[str, str]],
    aux_rows: Sequence[dict[str, str]],
    sample_embeddings: dict[str, SampleEmbeddingRecord],
    target_columns: Sequence[str] | None = None,
    manifest_rows: Sequence[dict[str, str]] | None = None,
) -> tuple[JoinedDataset, list[dict[str, Any]]]:
    if target_columns is None:
        target_columns = infer_target_columns(descriptor_rows)
    target_names = list(target_columns)
    aux_by_row = {row["row_id"]: row for row in aux_rows}
    manifest_by_sample = (
        {row["sample_id"].strip(): row for row in manifest_rows}
        if manifest_rows is not None
        else {}
    )

    row_ids: list[str] = []
    sample_ids: list[str] = []
    group_ids: list[str] = []
    afm_paths: list[str] = []
    network_input_paths: list[str] = []
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    skipped_rows: list[dict[str, Any]] = []

    for row in descriptor_rows:
        sample_id = row["sample_id"]
        embedding_record = sample_embeddings.get(sample_id)
        if embedding_record is None:
            skipped_rows.append(
                {"row_id": row["row_id"], "sample_id": sample_id, "reason": "missing sample embedding"}
            )
            continue
        aux_row = aux_by_row.get(row["row_id"])
        if aux_row is None:
            skipped_rows.append(
                {"row_id": row["row_id"], "sample_id": sample_id, "reason": "missing aux descriptor row"}
            )
            continue
        network_input_path = aux_row.get("network_input_path", "").strip()
        if not network_input_path:
            skipped_rows.append(
                {"row_id": row["row_id"], "sample_id": sample_id, "reason": "missing network_input_path"}
            )
            continue
        if manifest_by_sample:
            selected = manifest_by_sample.get(sample_id)
            if selected is None:
                skipped_rows.append(
                    {"row_id": row["row_id"], "sample_id": sample_id, "reason": "not selected in one-to-one manifest"}
                )
                continue
            manifest_afm = str(resolve_existing_path(Path(selected["afm_path"])))
            candidate_paths = {
                str(resolve_existing_path(Path(network_input_path))),
                str(resolve_existing_path(Path(aux_row.get("afm_path", "")))),
                str(resolve_existing_path(Path(row.get("afm_path", "")))),
            }
            if manifest_afm not in candidate_paths:
                skipped_rows.append(
                    {
                        "row_id": row["row_id"],
                        "sample_id": sample_id,
                        "reason": "AFM path does not match one-to-one manifest selection",
                    }
                )
                continue
            group_id = selected.get("group_id", "").strip() or sample_id
        else:
            group_id = sample_id
        try:
            targets = np.asarray([float(row[column]) for column in target_names], dtype=np.float32)
        except (KeyError, ValueError) as exc:
            skipped_rows.append(
                {"row_id": row["row_id"], "sample_id": sample_id, "reason": f"bad target row: {exc}"}
            )
            continue
        row_ids.append(row["row_id"])
        sample_ids.append(sample_id)
        group_ids.append(group_id)
        afm_paths.append(row.get("afm_path", aux_row.get("afm_path", "")))
        network_input_paths.append(network_input_path)
        x_rows.append(embedding_record.embedding.astype(np.float32))
        y_rows.append(targets)

    if not x_rows:
        raise ValueError("No joined rows were available after matching embeddings and AFM descriptors.")

    x = np.stack(x_rows, axis=0)
    y = np.stack(y_rows, axis=0)
    dataset = JoinedDataset(
        row_ids=row_ids,
        sample_ids=sample_ids,
        group_ids=group_ids,
        afm_paths=afm_paths,
        network_input_paths=network_input_paths,
        feature_names=embedding_feature_names(x.shape[1]),
        target_names=target_names,
        x=x,
        y=y,
    )
    return dataset, skipped_rows


def write_sample_embeddings_csv(path: Path, sample_embeddings: dict[str, SampleEmbeddingRecord]) -> None:
    rows: list[dict[str, Any]] = []
    embedding_dim = 0
    for record in sample_embeddings.values():
        embedding_dim = int(record.embedding.shape[0])
        break
    feature_names = embedding_feature_names(embedding_dim)
    for sample_id in sorted(sample_embeddings):
        record = sample_embeddings[sample_id]
        row = {
            "sample_id": record.sample_id,
            "video_path": display_path(record.video_path),
            "selection_reason": record.selection_reason,
            "duration_seconds": f"{record.duration_seconds:.6f}",
            "decoded_frame_count": record.decoded_frame_count,
            "sampled_frame_count": record.sampled_frame_count,
        }
        for feature_name, value in zip(feature_names, record.embedding):
            row[feature_name] = f"{float(value):.8f}"
        rows.append(row)
    write_csv(
        path,
        rows,
        ["sample_id", "video_path", "selection_reason", "duration_seconds", "decoded_frame_count", "sampled_frame_count", *feature_names],
    )


def write_sample_embeddings_matrix(
    matrix_path: Path,
    index_path: Path,
    sample_embeddings: dict[str, SampleEmbeddingRecord],
) -> None:
    sample_ids = sorted(sample_embeddings)
    matrix = np.stack([sample_embeddings[sample_id].embedding for sample_id in sample_ids], axis=0).astype(np.float32)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(matrix_path, matrix)
    rows = [
        {
            "row_id": index + 1,
            "sample_id": sample_id,
            "video_path": display_path(sample_embeddings[sample_id].video_path),
            "selection_reason": sample_embeddings[sample_id].selection_reason,
        }
        for index, sample_id in enumerate(sample_ids)
    ]
    write_csv(index_path, rows, ["row_id", "sample_id", "video_path", "selection_reason"])


def write_joined_dataset_csv(path: Path, dataset: JoinedDataset) -> None:
    rows: list[dict[str, Any]] = []
    for index in range(dataset.x.shape[0]):
        row = {
            "row_id": dataset.row_ids[index],
            "sample_id": dataset.sample_ids[index],
            "group_id": dataset.group_ids[index],
            "afm_path": dataset.afm_paths[index],
            "network_input_path": dataset.network_input_paths[index],
        }
        for name, value in zip(dataset.feature_names, dataset.x[index]):
            row[name] = f"{float(value):.8f}"
        for name, value in zip(dataset.target_names, dataset.y[index]):
            row[name] = f"{float(value):.8f}"
        rows.append(row)
    write_csv(
        path,
        rows,
        ["row_id", "sample_id", "group_id", "afm_path", "network_input_path", *dataset.feature_names, *dataset.target_names],
    )


def split_group_holdout(
    group_ids: Sequence[str],
    test_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(group_ids)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_state)
    indices = np.arange(groups.shape[0])
    train_idx, test_idx = next(splitter.split(indices, groups=groups))
    return train_idx.astype(int), test_idx.astype(int)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")


def normalized_mae(y_true: np.ndarray, y_pred: np.ndarray, scale: np.ndarray) -> float:
    safe_scale = np.where(np.asarray(scale) > 0, scale, 1.0)
    return float(np.mean(np.abs((y_true - y_pred) / safe_scale)))


def build_model_candidates(x_train: np.ndarray, y_train: np.ndarray) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = [
        ("ridge", {"alphas": np.logspace(-3, 3, 13)}),
    ]
    max_components = min(x_train.shape[1], y_train.shape[1], x_train.shape[0] - 1)
    pls_choices = sorted({value for value in (2, 4, 8, 16) if 1 <= value <= max_components})
    for n_components in pls_choices:
        candidates.append(("pls", {"n_components": n_components}))
    knn_choices = sorted({value for value in (1, 3, 5, 7) if 1 <= value <= x_train.shape[0]})
    for n_neighbors in knn_choices:
        candidates.append(("knn", {"n_neighbors": n_neighbors}))
    return candidates


def instantiate_model(model_name: str, params: dict[str, Any]) -> Any:
    if model_name == "ridge":
        return RidgeCV(alphas=params["alphas"])
    if model_name == "pls":
        return PLSRegression(n_components=int(params["n_components"]))
    if model_name == "knn":
        return KNeighborsRegressor(n_neighbors=int(params["n_neighbors"]))
    raise ValueError(f"Unsupported model: {model_name}")


def fit_predict_scaled(
    model_name: str,
    params: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[Any, StandardScaler, StandardScaler, np.ndarray]:
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    x_train_scaled = x_scaler.transform(x_train)
    x_test_scaled = x_scaler.transform(x_test)
    y_train_scaled = y_scaler.transform(y_train)
    model = instantiate_model(model_name, params)
    model.fit(x_train_scaled, y_train_scaled)
    y_pred_scaled = np.asarray(model.predict(x_test_scaled), dtype=np.float32)
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = y_pred_scaled[:, None]
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    return model, x_scaler, y_scaler, y_pred.astype(np.float32)


def select_best_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_groups: Sequence[str],
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    unique_groups = sorted(set(train_groups))
    if len(unique_groups) < 2:
        raise ValueError("Need at least two training groups for grouped model selection.")
    n_splits = min(5, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)

    score_rows: list[dict[str, Any]] = []
    summary: dict[tuple[str, str], list[float]] = {}
    candidate_defs = build_model_candidates(x_train, y_train)

    for fold_index, (fit_idx, val_idx) in enumerate(splitter.split(x_train, groups=train_groups), start=1):
        x_fit = x_train[fit_idx]
        y_fit = y_train[fit_idx]
        x_val = x_train[val_idx]
        y_val = y_train[val_idx]
        for model_name, params in candidate_defs:
            try:
                _, _, y_scaler, y_pred = fit_predict_scaled(model_name, params, x_fit, y_fit, x_val)
            except Exception as exc:
                score_rows.append(
                    {
                        "fold": fold_index,
                        "model_name": model_name,
                        "params_json": json.dumps(json_ready(params), sort_keys=True),
                        "normalized_mae": "",
                        "status": f"failed: {exc}",
                    }
                )
                continue
            score = normalized_mae(y_val, y_pred, y_scaler.scale_)
            params_json = json.dumps(json_ready(params), sort_keys=True)
            score_rows.append(
                {
                    "fold": fold_index,
                    "model_name": model_name,
                    "params_json": params_json,
                    "normalized_mae": f"{score:.8f}",
                    "status": "ok",
                }
            )
            summary.setdefault((model_name, params_json), []).append(score)

    if not summary:
        raise ValueError("Model selection failed for every candidate.")

    best_model_name = ""
    best_params: dict[str, Any] = {}
    best_score = float("inf")
    for (model_name, params_json), scores in summary.items():
        mean_score = float(np.mean(scores))
        if mean_score < best_score:
            best_score = mean_score
            best_model_name = model_name
            best_params = json.loads(params_json)

    for row in score_rows:
        key = (row["model_name"], row["params_json"])
        if key in summary:
            row["mean_normalized_mae"] = f"{float(np.mean(summary[key])):.8f}"
        else:
            row["mean_normalized_mae"] = ""
    return best_model_name, best_params, score_rows


def descriptor_metrics_rows(
    method: str,
    target_names: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, target_name in enumerate(target_names):
        truth = y_true[:, index]
        pred = y_pred[:, index]
        rows.append(
            {
                "method": method,
                "descriptor": target_name,
                "mae": f"{mean_absolute_error(truth, pred):.8f}",
                "rmse": f"{math.sqrt(mean_squared_error(truth, pred)):.8f}",
                "r2": f"{safe_r2(truth, pred):.8f}",
            }
        )
    return rows


def overall_metrics(target_names: Sequence[str], y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    metric_rows = descriptor_metrics_rows("model", target_names, y_true, y_pred)
    maes = [float(row["mae"]) for row in metric_rows]
    rmses = [float(row["rmse"]) for row in metric_rows]
    r2_values = [float(row["r2"]) for row in metric_rows if row["r2"] != "nan"]
    return {
        "descriptor_count": len(target_names),
        "mean_mae": float(np.mean(maes)),
        "mean_rmse": float(np.mean(rmses)),
        "mean_r2": float(np.mean(r2_values)) if r2_values else float("nan"),
    }


def nearest_neighbor_predictions(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    train_row_ids: Sequence[str],
    train_sample_ids: Sequence[str],
) -> tuple[np.ndarray, list[int], list[dict[str, Any]]]:
    x_scaler = StandardScaler().fit(x_train)
    x_train_scaled = x_scaler.transform(x_train)
    x_test_scaled = x_scaler.transform(x_test)
    indices: list[int] = []
    metadata: list[dict[str, Any]] = []
    predictions = np.zeros((x_test.shape[0], y_train.shape[1]), dtype=np.float32)
    for index, row in enumerate(x_test_scaled):
        distances = np.sqrt(np.sum((x_train_scaled - row[None, :]) ** 2, axis=1))
        nearest_index = int(np.argmin(distances))
        indices.append(nearest_index)
        predictions[index] = y_train[nearest_index]
        metadata.append(
            {
                "nearest_train_row_id": train_row_ids[nearest_index],
                "nearest_train_sample_id": train_sample_ids[nearest_index],
                "nearest_embedding_distance": float(distances[nearest_index]),
            }
        )
    return predictions, indices, metadata


def load_network_input_image(path_text: str) -> np.ndarray | None:
    path = resolve_existing_path(Path(path_text))
    if not path.exists():
        return None
    try:
        image = np.load(path)
    except Exception:
        return None
    if image.ndim != 2:
        return None
    return np.asarray(image, dtype=np.float32)


def write_scatter_plot(
    path: Path,
    target_names: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    count = min(6, len(target_names))
    indices = np.linspace(0, len(target_names) - 1, count, dtype=int)
    figure, axes = plt.subplots(2, int(math.ceil(count / 2)), figsize=(12, 7), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    for axis, descriptor_index in zip(axes, indices):
        name = target_names[descriptor_index]
        axis.scatter(y_true[:, descriptor_index], y_pred[:, descriptor_index], s=20, alpha=0.8)
        low = min(float(np.min(y_true[:, descriptor_index])), float(np.min(y_pred[:, descriptor_index])))
        high = max(float(np.max(y_true[:, descriptor_index])), float(np.max(y_pred[:, descriptor_index])))
        axis.plot([low, high], [low, high], color="black", linewidth=1)
        axis.set_title(name)
        axis.set_xlabel("True")
        axis.set_ylabel("Predicted")
    for axis in axes[count:]:
        axis.axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def summarize_descriptor_triplet(
    target_names: Sequence[str],
    y_true_row: np.ndarray,
    y_pred_row: np.ndarray,
    y_nn_row: np.ndarray,
) -> str:
    indices = np.argsort(np.abs(y_true_row - y_pred_row))[::-1][:3]
    lines = []
    for index in indices:
        lines.append(
            f"{target_names[index]}: true={y_true_row[index]:.3f} "
            f"pred={y_pred_row[index]:.3f} nn={y_nn_row[index]:.3f}"
        )
    return "\n".join(lines)


def write_qualitative_grid(
    path: Path,
    dataset: JoinedDataset,
    test_idx: np.ndarray,
    y_pred: np.ndarray,
    y_nn: np.ndarray,
    nearest_metadata: Sequence[dict[str, Any]],
    nearest_indices: Sequence[int],
) -> None:
    count = min(8, test_idx.size)
    selected = np.linspace(0, test_idx.size - 1, count, dtype=int)
    figure, axes = plt.subplots(count, 3, figsize=(10, 3.0 * count), dpi=150)
    axes = np.atleast_2d(axes)
    for row_axis, offset in zip(axes, selected):
        dataset_index = int(test_idx[offset])
        nearest_index = int(nearest_indices[offset])
        true_image = load_network_input_image(dataset.network_input_paths[dataset_index])
        nn_image = load_network_input_image(dataset.network_input_paths[nearest_index])
        for axis, image, title in (
            (row_axis[0], true_image, "True AFM"),
            (row_axis[1], nn_image, "Nearest train AFM"),
        ):
            axis.axis("off")
            axis.set_title(title)
            if image is not None:
                axis.imshow(image, cmap="viridis", vmin=-1.0, vmax=1.0)
            else:
                axis.text(0.5, 0.5, "missing", ha="center", va="center")
        row_axis[2].axis("off")
        meta = nearest_metadata[offset]
        row_axis[2].text(
            0.0,
            1.0,
            (
                f"sample {dataset.sample_ids[dataset_index]}\n"
                f"row {dataset.row_ids[dataset_index]}\n"
                f"nearest sample {meta['nearest_train_sample_id']}\n"
                f"distance {meta['nearest_embedding_distance']:.3f}\n\n"
                f"{summarize_descriptor_triplet(dataset.target_names, dataset.y[dataset_index], y_pred[offset], y_nn[offset])}"
            ),
            ha="left",
            va="top",
            family="monospace",
            fontsize=8,
        )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def write_embedding_pca_plot(
    path: Path,
    sample_embeddings: dict[str, SampleEmbeddingRecord],
    descriptor_rows: Sequence[dict[str, str]],
    color_descriptor: str,
) -> None:
    if len(sample_embeddings) < 2:
        return
    sample_to_values: dict[str, list[float]] = {}
    for row in descriptor_rows:
        raw = row.get(color_descriptor, "").strip()
        if raw == "":
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        sample_to_values.setdefault(row["sample_id"], []).append(value)
    sample_ids = [sample_id for sample_id in sorted(sample_embeddings) if sample_id in sample_to_values]
    if len(sample_ids) < 2:
        return
    matrix = np.stack([sample_embeddings[sample_id].embedding for sample_id in sample_ids], axis=0)
    colors = np.asarray([np.mean(sample_to_values[sample_id]) for sample_id in sample_ids], dtype=float)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(matrix)
    figure, axis = plt.subplots(figsize=(6, 5), dpi=150)
    scatter = axis.scatter(coords[:, 0], coords[:, 1], c=colors, cmap="viridis", s=45)
    for sample_id, x_coord, y_coord in zip(sample_ids, coords[:, 0], coords[:, 1]):
        axis.text(x_coord, y_coord, sample_id, fontsize=7)
    axis.set_title(f"RHEED Embedding PCA colored by {color_descriptor}")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    figure.colorbar(scatter, ax=axis, label=color_descriptor)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def write_predictions_csv(
    path: Path,
    dataset: JoinedDataset,
    test_idx: np.ndarray,
    y_pred: np.ndarray,
    y_nn: np.ndarray,
    nearest_metadata: Sequence[dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    fieldnames = [
        "row_id",
        "sample_id",
        "group_id",
        "afm_path",
        "network_input_path",
        "nearest_train_row_id",
        "nearest_train_sample_id",
        "nearest_embedding_distance",
    ]
    for target_name in dataset.target_names:
        fieldnames.extend([f"true_{target_name}", f"pred_{target_name}", f"nn_{target_name}"])
    for offset, dataset_index in enumerate(test_idx):
        row = {
            "row_id": dataset.row_ids[int(dataset_index)],
            "sample_id": dataset.sample_ids[int(dataset_index)],
            "group_id": dataset.group_ids[int(dataset_index)],
            "afm_path": dataset.afm_paths[int(dataset_index)],
            "network_input_path": dataset.network_input_paths[int(dataset_index)],
            **nearest_metadata[offset],
        }
        for target_name, true_value, pred_value, nn_value in zip(
            dataset.target_names,
            dataset.y[int(dataset_index)],
            y_pred[offset],
            y_nn[offset],
        ):
            row[f"true_{target_name}"] = f"{float(true_value):.8f}"
            row[f"pred_{target_name}"] = f"{float(pred_value):.8f}"
            row[f"nn_{target_name}"] = f"{float(nn_value):.8f}"
        rows.append(row)
    write_csv(path, rows, fieldnames)


def write_summary_report(
    path: Path,
    summary: dict[str, Any],
    overall_model_metrics: dict[str, Any],
    overall_nn_metrics: dict[str, Any],
) -> None:
    lines = [
        "# RHEED-to-AFM Descriptor MVP",
        "",
        f"- Encoder backend: `{summary['encoder_backend_resolved']}`",
        f"- Embedded samples: `{summary['sample_count_embedded']}`",
        f"- Skipped samples: `{summary['sample_count_skipped']}`",
        f"- Joined rows: `{summary['joined_row_count']}`",
        f"- Training rows: `{summary['train_row_count']}`",
        f"- Test rows: `{summary['test_row_count']}`",
        f"- Best model: `{summary['best_model_name']}` `{summary['best_params_json']}`",
        "",
        "## Holdout Metrics",
        "",
        (
            f"- Learned model mean MAE / RMSE / R^2: "
            f"`{overall_model_metrics['mean_mae']:.4f}` / "
            f"`{overall_model_metrics['mean_rmse']:.4f}` / "
            f"`{overall_model_metrics['mean_r2']:.4f}`"
        ),
        (
            f"- Nearest-neighbor baseline mean MAE / RMSE / R^2: "
            f"`{overall_nn_metrics['mean_mae']:.4f}` / "
            f"`{overall_nn_metrics['mean_rmse']:.4f}` / "
            f"`{overall_nn_metrics['mean_r2']:.4f}`"
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_modeling_experiment(
    dataset: JoinedDataset,
    data_dir: Path,
    report_dir: Path,
    summary: dict[str, Any] | None = None,
    random_state: int = 42,
    test_fraction: float = 0.25,
) -> dict[str, Any]:
    train_idx, test_idx = split_group_holdout(dataset.group_ids, test_fraction, random_state)
    x_train = dataset.x[train_idx]
    y_train = dataset.y[train_idx]
    train_groups = [dataset.group_ids[index] for index in train_idx]

    best_model_name, best_params, selection_rows = select_best_model(x_train, y_train, train_groups)
    model, x_scaler, y_scaler, y_pred = fit_predict_scaled(
        best_model_name,
        best_params,
        x_train,
        y_train,
        dataset.x[test_idx],
    )
    y_true = dataset.y[test_idx]

    y_nn, nearest_indices, nearest_metadata = nearest_neighbor_predictions(
        x_train=dataset.x[train_idx],
        y_train=dataset.y[train_idx],
        x_test=dataset.x[test_idx],
        train_row_ids=[dataset.row_ids[index] for index in train_idx],
        train_sample_ids=[dataset.sample_ids[index] for index in train_idx],
    )

    metric_rows = descriptor_metrics_rows("best_model", dataset.target_names, y_true, y_pred)
    metric_rows.extend(descriptor_metrics_rows("nearest_neighbor", dataset.target_names, y_true, y_nn))
    overall_model = overall_metrics(dataset.target_names, y_true, y_pred)
    overall_nn = overall_metrics(dataset.target_names, y_true, y_nn)

    write_csv(
        data_dir / "model_selection.csv",
        selection_rows,
        ["fold", "model_name", "params_json", "normalized_mae", "status", "mean_normalized_mae"],
    )
    write_csv(data_dir / "metrics_by_descriptor.csv", metric_rows, ["method", "descriptor", "mae", "rmse", "r2"])
    write_predictions_csv(
        data_dir / "test_predictions.csv",
        dataset=dataset,
        test_idx=test_idx,
        y_pred=y_pred,
        y_nn=y_nn,
        nearest_metadata=nearest_metadata,
    )
    joblib.dump(
        {
            "model_name": best_model_name,
            "model_params": best_params,
            "model": model,
            "x_scaler": x_scaler,
            "y_scaler": y_scaler,
            "feature_names": dataset.feature_names,
            "target_names": dataset.target_names,
        },
        data_dir / "best_model.joblib",
    )

    write_scatter_plot(report_dir / "predicted_vs_true_scatter.png", dataset.target_names, y_true, y_pred)
    write_qualitative_grid(
        report_dir / "nearest_neighbor_qualitative_grid.png",
        dataset=dataset,
        test_idx=test_idx,
        y_pred=y_pred,
        y_nn=y_nn,
        nearest_metadata=nearest_metadata,
        nearest_indices=[int(train_idx[index]) for index in nearest_indices],
    )

    metrics_summary = {
        "best_model_name": best_model_name,
        "best_params": best_params,
        "overall_model_metrics": overall_model,
        "overall_nearest_neighbor_metrics": overall_nn,
        "train_row_count": int(train_idx.size),
        "test_row_count": int(test_idx.size),
        "train_group_count": len(set(train_groups)),
        "test_group_count": len({dataset.group_ids[index] for index in test_idx}),
    }
    (data_dir / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if summary is None:
        summary = {}
    summary = dict(summary)
    summary.update(
        {
            "joined_row_count": int(dataset.x.shape[0]),
            "train_row_count": int(train_idx.size),
            "test_row_count": int(test_idx.size),
            "best_model_name": best_model_name,
            "best_params_json": json.dumps(json_ready(best_params), sort_keys=True),
        }
    )
    write_summary_report(report_dir / "summary.md", summary, overall_model, overall_nn)
    return metrics_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a frozen-encoder RHEED-to-AFM descriptor MVP baseline."
    )
    parser.add_argument("--pair-root", type=Path, default=DEFAULT_PAIR_ROOT)
    parser.add_argument("--descriptor-csv", type=Path, default=DEFAULT_DESCRIPTOR_CSV)
    parser.add_argument("--descriptor-aux-csv", type=Path, default=DEFAULT_DESCRIPTOR_AUX_CSV)
    parser.add_argument("--one-to-one-manifest", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--encoder-backend", choices=("auto", "torchvision", "torchhub"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rheed-input-mode", choices=("raw_video", "processed_npz", "processed_video"), default="raw_video")
    parser.add_argument("--processed-rheed-root", type=Path, default=DEFAULT_PROCESSED_RHEED_ROOT)
    parser.add_argument("--processed-frame-key", default="clean_frames")
    parser.add_argument("--processed-max-frames", type=int, default=64)
    parser.add_argument(
        "--processed-sample-map",
        choices=("manifest_sample_id_to_dataset_dir",),
        default="manifest_sample_id_to_dataset_dir",
    )
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pair_root = resolve_existing_path(args.pair_root)
    descriptor_csv = resolve_existing_path(args.descriptor_csv)
    descriptor_aux_csv = resolve_existing_path(args.descriptor_aux_csv)
    one_to_one_manifest = None if args.one_to_one_manifest is None else resolve_existing_path(args.one_to_one_manifest)
    data_dir = args.data_dir if args.data_dir.is_absolute() else (REPO_ROOT / args.data_dir)
    report_dir = args.report_dir if args.report_dir.is_absolute() else (REPO_ROOT / args.report_dir)
    processed_rheed_root = (
        resolve_existing_path(args.processed_rheed_root)
        if args.processed_rheed_root is not None
        else None
    )

    if args.rheed_input_mode == "raw_video" and not pair_root.is_dir():
        raise SystemExit(f"Missing pair root: {pair_root}")
    if not descriptor_csv.is_file():
        raise SystemExit(f"Missing descriptor CSV: {descriptor_csv}")
    if not descriptor_aux_csv.is_file():
        raise SystemExit(f"Missing auxiliary descriptor CSV: {descriptor_aux_csv}")
    if one_to_one_manifest is not None and not one_to_one_manifest.is_file():
        raise SystemExit(f"Missing one-to-one manifest: {one_to_one_manifest}")
    if args.rheed_input_mode in {"processed_npz", "processed_video"}:
        if processed_rheed_root is None or not processed_rheed_root.is_dir():
            raise SystemExit(f"Missing processed RHEED root: {processed_rheed_root}")

    descriptor_rows = read_csv(descriptor_csv)
    aux_rows = read_csv(descriptor_aux_csv)
    target_columns = infer_target_columns(descriptor_rows)
    manifest_rows = load_one_to_one_manifest(one_to_one_manifest) if one_to_one_manifest is not None else None

    sample_embeddings, skipped_samples, summary = collect_sample_embeddings(
        pair_root=pair_root,
        encoder_backend=args.encoder_backend,
        device_name=args.device,
        frame_count=args.frame_count,
        batch_size=args.batch_size,
        rheed_input_mode=args.rheed_input_mode,
        processed_rheed_root=processed_rheed_root,
        processed_frame_key=args.processed_frame_key,
        processed_max_frames=args.processed_max_frames,
        processed_sample_map=args.processed_sample_map,
        selected_rheed_paths_by_sample=(
            selected_rheed_paths_by_sample(manifest_rows) if manifest_rows is not None else None
        ),
        manifest_sample_ids=(
            [row["sample_id"].strip() for row in manifest_rows]
            if manifest_rows is not None
            else None
        ),
    )
    if not sample_embeddings:
        raise SystemExit("No sample embeddings were produced.")

    dataset, skipped_join_rows = build_joined_dataset(
        descriptor_rows=descriptor_rows,
        aux_rows=aux_rows,
        sample_embeddings=sample_embeddings,
        target_columns=target_columns,
        manifest_rows=manifest_rows,
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_sample_embeddings_csv(data_dir / "sample_embeddings.csv", sample_embeddings)
    write_sample_embeddings_matrix(
        data_dir / "sample_embeddings.npy",
        data_dir / "sample_embedding_index.csv",
        sample_embeddings,
    )
    write_joined_dataset_csv(data_dir / "joined_dataset.csv", dataset)
    write_csv(
        data_dir / "skipped_samples.csv",
        skipped_samples,
        ["sample_id", "reason", "video_path"],
    )
    write_csv(
        data_dir / "skipped_join_rows.csv",
        skipped_join_rows,
        ["row_id", "sample_id", "reason"],
    )
    if one_to_one_manifest is not None:
        summary["one_to_one_manifest"] = display_path(one_to_one_manifest)
    (data_dir / "sample_embedding_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    write_embedding_pca_plot(
        report_dir / "embedding_pca.png",
        sample_embeddings=sample_embeddings,
        descriptor_rows=descriptor_rows,
        color_descriptor=target_columns[0],
    )
    run_modeling_experiment(
        dataset=dataset,
        data_dir=data_dir,
        report_dir=report_dir,
        summary=summary,
        random_state=args.random_state,
        test_fraction=args.test_fraction,
    )
    print(f"Wrote data artifacts to {display_path(data_dir)}")
    print(f"Wrote report artifacts to {display_path(report_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
