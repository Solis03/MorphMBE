"""Dataset interface for generated RHEED shape-bag inputs."""

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


class RHEEDShapeBagDataset(Dataset):
    """Read `rheed_shape_bag_manifest.csv` and return one multi-frame bag per sample."""

    def __init__(self, manifest: str | Path, *, target_table: str | Path | None = None) -> None:
        self.manifest = resolve_path(manifest)
        with self.manifest.open("r", newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        self.targets: dict[str, dict[str, Any]] = {}
        if target_table is not None:
            target_path = resolve_path(target_table)
            with target_path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    sample_id = row.get("sample_id")
                    if sample_id:
                        self.targets[sample_id] = row

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
        item: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "frames": frames,
            "frame_mask": frame_mask,
            "frame_weights": frame_weights,
            "consensus_maps": consensus_maps,
            "shape_features": shape_features,
            "shape_feature_names": feature_names,
            "source_type": row.get("source_type", ""),
        }
        if row["sample_id"] in self.targets:
            item["target"] = self.targets[row["sample_id"]]
        return item

