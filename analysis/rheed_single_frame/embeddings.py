"""Frozen local visual embeddings for secondary benchmarks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analysis.rheed_single_frame.data import ExperimentPaths, write_csv_rows, write_parquet_or_csv_note
from analysis.rheed_single_frame.preprocessing import PreprocessedImage
from analysis.rheed_single_frame.removelist import RemovelistAudit, assert_no_removed_samples


def _local_resnet50_checkpoint() -> Path | None:
    candidates = [
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet50-11ad3fa6.pth",
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet50-0676ba61.pth",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _extract_resnet50(images: Sequence[PreprocessedImage], checkpoint: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from torchvision import models

    model = models.resnet50(weights=None)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    backbone = torch.nn.Sequential(*(list(model.children())[:-1]))
    backbone.eval()
    rows: list[dict[str, Any]] = []
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    with torch.no_grad():
        for item in images:
            arr = np.repeat(item.normalized[None, :, :], 3, axis=0).astype(np.float32)
            tensor = torch.from_numpy(arr).unsqueeze(0)
            tensor = (tensor - mean) / std
            emb = backbone(tensor).reshape(-1).cpu().numpy().astype(float)
            row: dict[str, Any] = {
                "sample_id": item.sample_id,
                "embedding_model": "torchvision_resnet50_imagenet_local_checkpoint",
                "embedding_status": "ok",
                "embedding_dimension": int(emb.size),
            }
            row.update({f"embedding_{idx:04d}": float(value) for idx, value in enumerate(emb)})
            rows.append(row)
    return rows, {"embedding_model": "torchvision_resnet50", "checkpoint": checkpoint.as_posix(), "status": "ok"}


def extract_frozen_embeddings(
    images: Sequence[PreprocessedImage],
    paths: ExperimentPaths,
    removelist: RemovelistAudit,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract frozen embeddings without internet access."""
    assert_no_removed_samples((item.sample_id for item in images), removelist.sample_ids, context="frozen embedding extraction")
    checkpoint = _local_resnet50_checkpoint()
    if checkpoint is None:
        rows = [
            {
                "sample_id": item.sample_id,
                "embedding_model": "",
                "embedding_status": "skipped_no_local_pretrained_weights",
                "embedding_dimension": 0,
            }
            for item in images
        ]
        summary = {"embedding_model": "", "checkpoint": "", "status": "skipped_no_local_pretrained_weights"}
    else:
        rows, summary = _extract_resnet50(images, checkpoint)
    write_csv_rows(paths.outputs_dir / "frozen_embeddings.csv", rows)
    write_parquet_or_csv_note(paths.outputs_dir / "frozen_embeddings.parquet", rows)
    return rows, summary


def embedding_feature_names(rows: Sequence[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    return sorted(key for key in rows[0] if key.startswith("embedding_") and key[10:].isdigit())

