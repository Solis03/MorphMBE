"""Dataset interface for MVP-12 variable-K RHEED shape-bag v2 files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def _np_scalar_to_python(value: np.ndarray, default: Any = None) -> Any:
    if value.shape == ():
        return value.item()
    if value.size == 1:
        return value.reshape(-1)[0].item()
    return default


class RHEEDShapeBagDatasetV2(Dataset):
    """Read `rheed_shape_bag_manifest_v2.csv` and return one bag per sample."""

    def __init__(self, manifest: str | Path) -> None:
        self.manifest = resolve_path(manifest)
        with self.manifest.open("r", newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        npz_path = resolve_path(row["shape_bag_npz"])
        with np.load(npz_path, allow_pickle=False) as data:
            frames = torch.from_numpy(data["frames"].astype(np.float32, copy=False))
            frame_mask = torch.from_numpy(data["frame_mask"].astype(np.float32, copy=False))
            frame_weights = torch.from_numpy(data["frame_weights"].astype(np.float32, copy=False))
            consensus_maps = torch.from_numpy(data["consensus_maps"].astype(np.float32, copy=False))
            shape_features = torch.from_numpy(data["sample_feature_vector"].astype(np.float32, copy=False))
            feature_names = [str(item) for item in data["sample_feature_names"].tolist()]
            num_valid_frames = int(_np_scalar_to_python(data["num_valid_frames"], int(frame_mask.sum().item())))
            sample_quality = float(_np_scalar_to_python(data["sample_quality"], row.get("sample_quality", 0.0)))
            sample_status = str(_np_scalar_to_python(data["sample_status"], row.get("sample_status", "")))
            source_type = str(_np_scalar_to_python(data["source_type"], row.get("source_type", "")))

        return {
            "sample_id": row["sample_id"],
            "frames": frames,
            "frame_mask": frame_mask,
            "frame_weights": frame_weights * frame_mask,
            "consensus_maps": consensus_maps,
            "shape_features": shape_features,
            "shape_feature_names": feature_names,
            "num_valid_frames": num_valid_frames,
            "sample_quality": sample_quality,
            "sample_status": sample_status,
            "source_type": source_type,
        }
