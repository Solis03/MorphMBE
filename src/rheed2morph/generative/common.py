"""Shared utilities for the AFM latent diffusion MVP."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_repo_path(path: Path, base_dir: Path | None = None) -> Path:
    expanded = Path(path).expanduser()
    candidates = [expanded]
    if not expanded.is_absolute():
        if base_dir is not None:
            candidates.append(base_dir / expanded)
        candidates.append(REPO_ROOT / expanded)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if expanded.is_absolute():
        return expanded.resolve()
    if base_dir is not None:
        return (base_dir / expanded).resolve()
    return (REPO_ROOT / expanded).resolve()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        names: list[str] = []
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
        fieldnames = names
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_torch_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def replace_nonfinite(array: np.ndarray) -> np.ndarray:
    output = np.asarray(array, dtype=np.float32)
    if np.isfinite(output).all():
        return output
    finite = output[np.isfinite(output)]
    fill = float(np.median(finite)) if finite.size else 0.0
    return np.nan_to_num(output, nan=fill, posinf=fill, neginf=fill).astype(np.float32)


def _first_2d_npz_array(path: Path) -> np.ndarray:
    payload = np.load(path)
    preferred = (
        "network_input",
        "height",
        "height_nm",
        "processed_zsensor_nm",
        "clean_frames",
        "arr_0",
    )
    for key in preferred:
        if key not in payload:
            continue
        array = np.asarray(payload[key])
        if array.ndim == 2:
            return array
        if array.ndim == 3:
            return array[0]
    for key in payload.files:
        array = np.asarray(payload[key])
        if array.ndim == 2:
            return array
        if array.ndim == 3:
            return array[0]
    raise ValueError(f"No 2D array found in NPZ file: {path}")


def load_height_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix == ".npz":
        array = _first_2d_npz_array(path)
    elif suffix in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}:
        with Image.open(path) as image:
            array = np.asarray(image.convert("F"), dtype=np.float32)
    elif suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        array = np.loadtxt(path, delimiter=delimiter)
    else:
        raise ValueError(f"Unsupported AFM array file type: {path}")
    array = np.asarray(array, dtype=np.float32)
    array = np.squeeze(array)
    if array.ndim == 3 and 1 in array.shape:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D AFM height map, got shape {array.shape} from {path}")
    return replace_nonfinite(array)


def resize_array(array: np.ndarray, image_size: int) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))[None, None]
    resized = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return resized[0, 0].numpy().astype(np.float32)


def robust_normalize_to_unit(array: np.ndarray, image_size: int, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    resized = resize_array(replace_nonfinite(array), image_size)
    finite = resized[np.isfinite(resized)]
    if finite.size == 0:
        return np.zeros((image_size, image_size), dtype=np.float32)
    p_low, p_high = np.percentile(finite, [low, high])
    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
        return np.zeros((image_size, image_size), dtype=np.float32)
    clipped = np.clip(resized, p_low, p_high)
    normalized = (clipped - p_low) / (p_high - p_low) * 2.0 - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class AFMIndexRecord:
    row_id: str
    sample_id: str
    group_id: str
    split: str
    network_input_path: Path


def load_data_index(path: Path, split: str | None = None, limit: int | None = None) -> list[AFMIndexRecord]:
    rows = read_csv_rows(path)
    records: list[AFMIndexRecord] = []
    for row in rows:
        if split is not None and row.get("split", "") != split:
            continue
        records.append(
            AFMIndexRecord(
                row_id=row["row_id"],
                sample_id=row.get("sample_id", row["row_id"]),
                group_id=row.get("group_id", row.get("sample_id", row["row_id"])),
                split=row.get("split", ""),
                network_input_path=resolve_repo_path(Path(row["network_input_path"])),
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


class AFMImageDataset(Dataset[tuple[torch.Tensor, dict[str, str]]]):
    def __init__(self, records: Sequence[AFMIndexRecord]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, str]]:
        record = self.records[index]
        array = load_height_array(record.network_input_path)
        tensor = torch.from_numpy(array.astype(np.float32))[None]
        metadata = {
            "row_id": record.row_id,
            "sample_id": record.sample_id,
            "group_id": record.group_id,
            "split": record.split,
        }
        return tensor, metadata


def batched(iterable: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(iterable), size):
        yield iterable[start : start + size]


def numeric_columns(rows: Sequence[dict[str, str]], exclude: set[str]) -> list[str]:
    if not rows:
        return []
    columns: list[str] = []
    for key in rows[0]:
        if key in exclude:
            continue
        ok = True
        for row in rows:
            try:
                float(row.get(key, "nan"))
            except ValueError:
                ok = False
                break
        if ok:
            columns.append(key)
    return columns


def env_summary() -> dict[str, str | bool]:
    return {
        "python_version": ".".join(map(str, __import__("sys").version_info[:3])),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
