from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis.rheed_to_afm_functional_morphology.amplitude import (
    CURATED_RHEED_FEATURES,
    DYNAMIC_NUCLEATION_FEATURES,
)
from rheed2morph.realtime.catalog import (
    discover_videos,
    group_by_sample,
    read_removelist,
)
from rheed2morph.realtime.clips import build_model_clip, live_physics_row
from rheed2morph.realtime.model import MODEL_ID, load_deployment_bundle
from rheed2morph.realtime.selector import _event_rows
from rheed2morph.rheed.automatic_roi_keyframe import Rect


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY / "configs/rheed_realtime_ui.json"


def test_video_catalog_groups_samples_and_excludes_removelist(
    tmp_path: Path,
) -> None:
    (tmp_path / "N6063").mkdir()
    (tmp_path / "N6063" / "growth.MOV").touch()
    (tmp_path / "N6055").mkdir()
    (tmp_path / "N6055" / "growth.mp4").touch()
    removelist = tmp_path / "remove.txt"
    removelist.write_text("6055 # excluded\n", encoding="utf-8")
    entries = discover_videos(
        tmp_path,
        excluded_sample_ids=read_removelist(removelist),
    )
    grouped = group_by_sample(entries)
    assert list(grouped) == ["6063"]
    assert grouped["6063"][0].path.name == "growth.MOV"


def test_selected_16_clip_preserves_keyframe_position() -> None:
    frames = [
        np.full((24, 32, 3), value, dtype=np.uint8)
        for value in range(16)
    ]
    roi = Rect(
        x=0,
        y=0,
        width=32,
        height=24,
        source_width=32,
        source_height=24,
    )
    clip = build_model_clip(frames, roi, output_size=224)
    assert clip.shape == (16, 224, 224)
    assert clip.dtype == np.uint8
    # The frozen selected-16 convention places the keyframe at zero-based 7.
    assert int(clip[7].max()) == 7
    assert int(clip[8].max()) == 8


def test_live_physics_matches_frozen_m14_schema() -> None:
    yy, xx = np.mgrid[:224, :224]
    frames = np.stack(
        [
            np.clip(
                20
                + 180 * np.exp(
                    -((xx - 80 - index) ** 2 + (yy - 110) ** 2) / 120
                ),
                0,
                255,
            ).astype(np.uint8)
            for index in range(16)
        ]
    )
    physics = live_physics_row(frames)
    required = set(CURATED_RHEED_FEATURES + DYNAMIC_NUCLEATION_FEATURES)
    assert required.issubset(physics.columns)
    assert np.isfinite(
        physics[list(required)].to_numpy(float)
    ).all()


def test_event_clustering_keeps_one_tracker_per_rotation_vertex() -> None:
    rows = [
        {
            "frame_index": 100,
            "tracker": "front",
            "score": 0.70,
            "spot_visibility_rank": 0.80,
            "eligible": True,
        },
        {
            "frame_index": 103,
            "tracker": "compact",
            "score": 0.85,
            "spot_visibility_rank": 0.75,
            "eligible": True,
        },
        {
            "frame_index": 132,
            "tracker": "front",
            "score": 0.78,
            "spot_visibility_rank": 0.70,
            "eligible": True,
        },
        {
            "frame_index": 160,
            "tracker": "front",
            "score": 0.99,
            "spot_visibility_rank": 0.20,
            "eligible": False,
        },
    ]
    events = _event_rows(
        rows,
        visibility_gate=0.50,
        estimated_period=30.0,
    )
    assert [event.frame_index for event in events] == [103, 132]
    assert all(0.0 <= event.keyframe_quality <= 1.0 for event in events)


def test_deployment_cache_identifies_frozen_nonretrieval_pipeline() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    bundle_path = REPOSITORY / config["deployment_bundle"]
    if not bundle_path.exists():
        return
    bundle = load_deployment_bundle(bundle_path)
    assert bundle.model_id == MODEL_ID
    assert len(bundle.groups) == 23
    assert bundle.generation_config["selected_method"] == (
        "M12a_edge_preserving_terrace"
    )
    assert bundle.retrieval_at_inference is False
    assert bundle.measured_afm_patch_at_inference is False
