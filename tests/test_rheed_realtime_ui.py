from __future__ import annotations

import inspect
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
from rheed2morph.realtime.clips import (
    build_causal_perturbation_clips,
    build_model_clip,
    live_physics_row,
)
from rheed2morph.realtime.model import MODEL_ID, load_deployment_bundle
from rheed2morph.realtime.ui import RealtimeMainWindow
from rheed2morph.realtime.selector import _event_rows, _lattice_vertex_scores
from rheed2morph.realtime.workers import ReplayWorker
from rheed2morph.rheed.automatic_roi_keyframe import Rect
from rheed2morph.rheed.orientation import (
    rotate_frame_clockwise,
    rotation_for_sample,
)


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


def test_clockwise_orientation_correction_is_shared_by_numeric_and_n_ids() -> None:
    frame = np.arange(2 * 3, dtype=np.uint8).reshape(2, 3)
    expected = np.asarray([[3, 0], [4, 1], [5, 2]], dtype=np.uint8)
    mapping = {"N6389": 90, "N6390": 90}

    assert np.array_equal(rotate_frame_clockwise(frame, 90), expected)
    assert rotation_for_sample(mapping, "6389") == 90
    assert rotation_for_sample(mapping, "N6390") == 90
    assert rotation_for_sample(mapping, "6382") == 0


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


def test_causal_tta_views_match_k_plus_8_replay_ring() -> None:
    frames = [
        np.full((24, 32, 3), value, dtype=np.uint8)
        for value in range(18)
    ]
    roi = Rect(
        x=0,
        y=0,
        width=32,
        height=24,
        source_width=32,
        source_height=24,
    )
    names, views = build_causal_perturbation_clips(
        frames,
        roi,
        output_size=32,
    )
    lookup = {name: views[index] for index, name in enumerate(names)}
    # Input values 0..17 represent k-9..k+8.
    assert lookup["frame_m2"][:, 16, 16].tolist() == list(range(0, 8))
    assert lookup["frame_m1"][:, 16, 16].tolist() == list(range(1, 9))
    assert lookup["base"][:, 16, 16].tolist() == list(range(2, 10))
    assert lookup["frame_p1"][:, 16, 16].tolist() == list(range(3, 11))
    assert lookup["frame_p2"][:, 16, 16].tolist() == list(range(4, 12))


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


def test_lattice_vertex_score_rejects_sparse_bright_phase() -> None:
    common = {
        "spot_energy_concentration": 0.45,
        "raw_std": 0.05,
        "spot_peak_count": 10.0,
        "haze_dominance": 5.0,
    }
    rows = [
        {
            **common,
            "spot_peak_top8_mass": 1.5,
            "spot_column_alignment": 2.8,
            "spot_horizontal_spread": 0.24,
        },
        {
            **common,
            "spot_peak_top8_mass": 1.9,
            "spot_column_alignment": 5.9,
            "spot_horizontal_spread": 0.09,
        },
    ]
    scores = _lattice_vertex_scores(rows)
    assert int(np.argmax(scores)) == 1


def test_replay_worker_never_uses_tracking_roi_as_model_input() -> None:
    source = inspect.getsource(ReplayWorker.run)
    forbidden = (
        "build_model_clip(\n"
        "                                list(ring),\n"
        "                                selection.tracking_roi.rect"
    )
    assert "selection.model_input_roi.rect" in source
    assert "selection.physics_roi.rect" in source
    assert "deque(maxlen=18)" in source
    assert forbidden not in source


def test_deployment_cache_identifies_frozen_nonretrieval_pipeline() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    bundle_path = REPOSITORY / config["deployment_bundle"]
    if not bundle_path.exists():
        return
    bundle = load_deployment_bundle(bundle_path)
    assert bundle.model_id == MODEL_ID
    generation_config = json.loads(
        (REPOSITORY / config["generation_config"]).read_text(
            encoding="utf-8"
        )
    )
    assert len(bundle.groups) == generation_config["expected_growth_count"]
    assert bundle.generation_config["selected_method"] == (
        "M12a_edge_preserving_terrace"
    )
    assert "line3" in bundle.generation_config["afm_descriptors"]
    assert bundle.rq_reference.confidence_risk_reference is not None
    assert bundle.rq_reference.confidence_error_reference is not None
    assert bundle.fsmi_reference.confidence_risk_reference is not None
    assert bundle.retrieval_at_inference is False
    assert bundle.measured_afm_patch_at_inference is False


def test_realtime_config_and_ui_identify_m15b_m12a_pipeline() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest_path = REPOSITORY / config["deployment_manifest"]
    assert "m15b_m12a" in Path(config["deployment_bundle"]).name
    assert "line3" in Path(config["deployment_bundle"]).name
    assert "full28" in Path(config["deployment_bundle"]).name
    assert "orientation90_keyframe_locked" in Path(
        config["deployment_bundle"]
    ).name
    assert config["rheed_rotation_clockwise_degrees_by_sample"] == {
        "6389": 90,
        "6390": 90,
    }
    assert set(config["replay_keyframe_override_by_sample"]) == {
        "6389",
        "6390",
    }
    assert config["replay_keyframe_override_by_sample"]["6389"][
        "source_name"
    ] == "Rampdown to 300C.avi"
    assert "m15b_m12a" in manifest_path.name
    assert config["metrology_audited_mode"] is True
    assert "line3" in config["generation_config"]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["model_id"] == MODEL_ID
        assert manifest["method"]["image_generator"] == (
            "M12a_edge_preserving_terrace"
        )
        assert manifest["method"]["legacy_internal_target_name"] == "Rq_nm"
        assert "third-order" in manifest["method"]["afm_metrology"]
        assert manifest["method"]["retrieval_at_inference"] is False
        assert manifest["method"]["measured_afm_patch_at_inference"] is False
    source = inspect.getsource(RealtimeMainWindow._build_ui)
    assert "M15b causal R3D" in source
    assert "actually passed to M15b/M12a" in source
    assert "actually passed to M14i/M12a" not in source


def test_realtime_ui_source_contains_no_cjk_text() -> None:
    realtime_root = REPOSITORY / "src" / "rheed2morph" / "realtime"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(realtime_root.glob("*.py"))
    )
    assert not any("\u3400" <= character <= "\u9fff" for character in source)


def test_v8_model_input_roi_bundle_has_strict_loo_provenance() -> None:
    import joblib

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    bundle_path = REPOSITORY / config["model_input_roi_calibration"]
    if not bundle_path.exists():
        return
    bundle = joblib.load(bundle_path)
    assert bundle["validation_method"] == (
        "v8_orientation_model_input_q20_q80"
    )
    assert bundle["validation_protocol"] == "strict_leave_one_video_out"
    assert bundle["held_video_overlap_sum"] == 0
