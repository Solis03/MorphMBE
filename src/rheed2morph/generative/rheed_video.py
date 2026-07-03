"""RHEED video loading and preprocessing for MVP-2."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from rheed2morph.generative.common import replace_nonfinite, resolve_repo_path


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


def cache_name_for_video(path: Path, frames: int, image_size: int, final_fraction: float, log1p: bool) -> str:
    digest = hashlib.sha1(f"{path.resolve()}|{frames}|{image_size}|{final_fraction}|{log1p}".encode("utf-8")).hexdigest()[:12]
    return f"rheed_{digest}.npz"


def _to_gray(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 2:
        gray = array
    elif array.ndim == 3:
        rgb = array[..., :3].astype(np.float32)
        gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    else:
        raise ValueError(f"Unsupported frame shape: {array.shape}")
    return gray.astype(np.float32)


def _normalize_stack(frames: np.ndarray, log1p: bool = False) -> np.ndarray:
    output = replace_nonfinite(frames.astype(np.float32))
    if log1p:
        minimum = float(np.min(output))
        output = np.log1p(np.maximum(output - minimum, 0.0)).astype(np.float32)
    finite = output[np.isfinite(output)]
    if finite.size == 0:
        return np.zeros_like(output, dtype=np.float32)
    p01, p99 = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(p01) or not np.isfinite(p99) or p99 <= p01:
        return np.zeros_like(output, dtype=np.float32)
    output = np.clip(output, p01, p99)
    output = (output - p01) / (p99 - p01)
    return np.clip(output, 0.0, 1.0).astype(np.float32)


def _resize_stack(frames: np.ndarray, image_size: int) -> np.ndarray:
    tensor = torch.from_numpy(frames.astype(np.float32))[:, None]
    resized = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return resized.numpy().astype(np.float32)


def _sample_indices(frame_count: int, frames: int, final_fraction: float, sampling: str) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    start = max(0, int(frame_count * (1.0 - final_fraction)))
    if sampling != "uniform":
        raise ValueError(f"Unsupported RHEED frame sampling mode: {sampling}")
    if frame_count - start <= 0:
        start = 0
    return np.linspace(start, frame_count - 1, frames).round().astype(int)


def _load_array_tensor(path: Path, frames: int, image_size: int, final_fraction: float, sampling: str, log1p: bool) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        payload = np.load(path)
        for key in ("frames", "video", "tensor", "rheed", "arr_0"):
            if key in payload:
                array = np.asarray(payload[key])
                break
        else:
            array = np.asarray(payload[payload.files[0]])
    else:
        array = np.load(path)
    array = np.asarray(array)
    if array.ndim == 4 and array.shape[1] == 1:
        stack = array[:, 0]
    elif array.ndim == 4 and array.shape[-1] in {1, 3, 4}:
        stack = np.stack([_to_gray(frame) for frame in array], axis=0)
    elif array.ndim == 3:
        stack = array
    else:
        raise ValueError(f"Unsupported RHEED tensor shape {array.shape} in {path}")
    indices = _sample_indices(stack.shape[0], frames, final_fraction, sampling)
    normalized = _normalize_stack(stack[indices], log1p=log1p)
    return _resize_stack(normalized, image_size)


def _load_video_tensor(path: Path, frames: int, image_size: int, final_fraction: float, sampling: str, log1p: bool) -> tuple[np.ndarray, int]:
    reader = imageio.get_reader(path)
    try:
        try:
            frame_count = int(reader.count_frames())
        except Exception:
            meta_count = reader.get_meta_data().get("nframes", 0)
            frame_count = int(meta_count) if np.isfinite(meta_count) else 0
        if frame_count <= 0 or frame_count > 1_000_000:
            loaded = [_to_gray(frame) for frame in reader]
            if not loaded:
                raise ValueError(f"No frames decoded from {path}")
            stack = np.stack(loaded, axis=0)
        else:
            indices = _sample_indices(frame_count, frames, final_fraction, sampling)
            loaded = []
            for index in indices:
                loaded.append(_to_gray(reader.get_data(int(index))))
            stack = np.stack(loaded, axis=0)
        normalized = _normalize_stack(stack, log1p=log1p)
        return _resize_stack(normalized, image_size), int(frame_count if frame_count > 0 else stack.shape[0])
    finally:
        reader.close()


def load_rheed_tensor(
    path: Path,
    frames: int = 8,
    image_size: int = 224,
    final_fraction: float = 0.25,
    sampling: str = "uniform",
    log1p: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    resolved = resolve_repo_path(path)
    suffix = resolved.suffix.lower()
    if suffix in {".npy", ".npz"}:
        tensor = _load_array_tensor(resolved, frames, image_size, final_fraction, sampling, log1p)
        source_frame_count = int(tensor.shape[0])
    elif suffix in VIDEO_SUFFIXES:
        tensor, source_frame_count = _load_video_tensor(resolved, frames, image_size, final_fraction, sampling, log1p)
    else:
        raise ValueError(f"Unsupported RHEED input file type: {resolved}")
    if tensor.shape != (frames, 1, image_size, image_size):
        raise ValueError(f"Expected tensor shape {(frames, 1, image_size, image_size)}, got {tensor.shape} from {resolved}")
    metadata = {
        "source_frame_count": source_frame_count,
        "frames": frames,
        "image_size": image_size,
        "final_fraction": final_fraction,
        "sampling": sampling,
        "normalization": "percentile_1_99_to_0_1" + ("_log1p" if log1p else ""),
    }
    return tensor.astype(np.float32), metadata


def load_or_cache_rheed_tensor(
    path: Path,
    cache_dir: Path,
    frames: int = 8,
    image_size: int = 224,
    final_fraction: float = 0.25,
    sampling: str = "uniform",
    log1p: bool = False,
) -> tuple[np.ndarray, Path, dict[str, Any]]:
    resolved = resolve_repo_path(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name_for_video(resolved, frames, image_size, final_fraction, log1p)
    if cache_path.is_file():
        payload = np.load(cache_path)
        tensor = np.asarray(payload["frames"], dtype=np.float32)
        metadata = {
            "source_frame_count": int(payload.get("source_frame_count", tensor.shape[0])),
            "frames": frames,
            "image_size": image_size,
            "final_fraction": final_fraction,
            "sampling": sampling,
            "normalization": str(payload.get("normalization", "percentile_1_99_to_0_1")),
            "cache_hit": True,
        }
        return tensor, cache_path, metadata
    tensor, metadata = load_rheed_tensor(resolved, frames, image_size, final_fraction, sampling, log1p)
    np.savez_compressed(
        cache_path,
        frames=tensor.astype(np.float32),
        source_path=resolved.as_posix(),
        source_frame_count=int(metadata["source_frame_count"]),
        normalization=metadata["normalization"],
    )
    metadata["cache_hit"] = False
    return tensor, cache_path, metadata
