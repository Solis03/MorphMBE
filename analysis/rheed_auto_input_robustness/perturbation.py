from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analysis.rheed_manual_vs_auto_selection.dataset import (
    _decode_selected16,
)
from analysis.rheed_video_afm_story.pretrained_embeddings import (
    load_r3d18,
    preprocess_frames,
)
from rheed2morph.realtime.clips import build_model_clip
from rheed2morph.rheed.automatic_roi_keyframe import Rect


@dataclass(frozen=True)
class PerturbationView:
    name: str
    frame_offset: int = 0
    x_shift_fraction: float = 0.0
    y_shift_fraction: float = 0.0
    scale: float = 1.0


DEFAULT_VIEWS = (
    PerturbationView("base"),
    PerturbationView("frame_m2", frame_offset=-2),
    PerturbationView("frame_m1", frame_offset=-1),
    PerturbationView("frame_p1", frame_offset=1),
    PerturbationView("frame_p2", frame_offset=2),
    PerturbationView("roi_left", x_shift_fraction=-0.03),
    PerturbationView("roi_right", x_shift_fraction=0.03),
    PerturbationView("roi_up", y_shift_fraction=-0.03),
    PerturbationView("roi_down", y_shift_fraction=0.03),
    PerturbationView("roi_tight", scale=0.94),
    PerturbationView("roi_wide", scale=1.06),
)


def perturb_rect(rect: Rect, view: PerturbationView) -> Rect:
    """Apply a small, deterministic deployment-like ROI perturbation."""

    center_x = rect.x + 0.5 * rect.width
    center_y = rect.y + 0.5 * rect.height
    width = max(8, int(round(rect.width * float(view.scale))))
    height = max(8, int(round(rect.height * float(view.scale))))
    center_x += float(view.x_shift_fraction) * rect.width
    center_y += float(view.y_shift_fraction) * rect.height
    return Rect(
        x=int(round(center_x - 0.5 * width)),
        y=int(round(center_y - 0.5 * height)),
        width=width,
        height=height,
        source_width=rect.source_width,
        source_height=rect.source_height,
    ).clipped()


def _rect_from_row(row: pd.Series) -> Rect:
    return Rect(
        x=int(row["machine_roi_x"]),
        y=int(row["machine_roi_y"]),
        width=int(row["machine_roi_width"]),
        height=int(row["machine_roi_height"]),
        source_width=int(row["source_width"]),
        source_height=int(row["source_height"]),
    )


@torch.inference_mode()
def extract_perturbation_embeddings(
    selection: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    views: Iterable[PerturbationView] = DEFAULT_VIEWS,
    device: str = "cpu",
    progress: bool = True,
) -> tuple[list[str], list[str], np.ndarray, str]:
    """Extract causal-8 R3D features for frame/ROI perturbation views."""

    view_list = tuple(views)
    rows = selection.copy()
    rows["growth_run_id"] = rows["growth_run_id"].astype(str)
    rows = rows.set_index("growth_run_id")
    metadata = manifest.copy()
    metadata["growth_run_id"] = metadata["growth_run_id"].astype(str)
    metadata = metadata.set_index("growth_run_id")
    groups = rows.index.astype(str).tolist()
    metadata = metadata.loc[groups]

    model, status = load_r3d18()
    if model is None or not status.loaded:
        raise RuntimeError(f"R3D-18 could not load: {status.reason}")
    torch_device = torch.device(device)
    model = model.to(torch_device).eval()
    matrices: list[np.ndarray] = []
    for position, group in enumerate(groups, start=1):
        row = rows.loc[group]
        source = Path(str(row["source_video"]))
        base_keyframe = int(row["machine_keyframe_index"])
        rotation = int(
            row.get("frame_rotation_clockwise_degrees", 0)
            if pd.notna(row.get("frame_rotation_clockwise_degrees", 0))
            else 0
        )
        base_roi = _rect_from_row(pd.concat([row, metadata.loc[group]]))
        decoded: dict[int, list[np.ndarray]] = {}
        sample_embeddings = []
        for view in view_list:
            keyframe = base_keyframe + int(view.frame_offset)
            if keyframe not in decoded:
                decoded[keyframe] = _decode_selected16(
                    source,
                    keyframe,
                    rotation_clockwise_degrees=rotation,
                )
            roi = perturb_rect(base_roi, view)
            clip = build_model_clip(decoded[keyframe], roi)
            tensor = preprocess_frames(
                clip[:8],
                "raw_luminance",
                video=True,
            ).to(torch_device)
            embedding = model(tensor).detach().cpu().numpy()[0]
            sample_embeddings.append(np.asarray(embedding, dtype=np.float32))
        matrices.append(np.stack(sample_embeddings))
        if progress:
            print(
                f"[R3D perturbation {position:02d}/{len(groups):02d}] {group}",
                flush=True,
            )
    return (
        groups,
        [view.name for view in view_list],
        np.stack(matrices).astype(np.float32),
        status.weight_identifier,
    )


def save_perturbation_embeddings(
    path: str | Path,
    *,
    groups: list[str],
    view_names: list[str],
    embeddings: np.ndarray,
    weight_identifier: str,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        growth_run_ids=np.asarray(groups),
        view_names=np.asarray(view_names),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        embedding_dim=np.asarray(embeddings.shape[-1]),
        weight_identifier=np.asarray(weight_identifier),
        temporal_semantics=np.asarray("causal_8=k-7..k; offsets {-2,-1,0,+1,+2}"),
        target_blind=np.asarray(True),
    )
    return target
