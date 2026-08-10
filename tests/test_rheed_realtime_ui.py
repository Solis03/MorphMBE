from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.rheed_endpoint_generation.endpoint_ensemble import (
    EndpointPrediction,
)
from analysis.rheed_rough_island_redesign.connectivity import _feature_matrix
from analysis.rheed_to_afm_full_cohort_loo.run import load_config
from analysis.rheed_to_afm_functional_morphology.amplitude import (
    CURATED_RHEED_FEATURES,
    DYNAMIC_NUCLEATION_FEATURES,
)
from rheed2morph.realtime.catalog import (
    discover_videos,
    group_by_sample,
    read_removelist,
)
from rheed2morph.realtime.cli import repository_root_from_config
from rheed2morph.realtime.clips import (
    build_causal_perturbation_clips,
    build_model_clip,
    live_physics_row,
)
from rheed2morph.realtime.model import (
    M22_MODEL_ID,
    RealtimeMorphologyPredictor,
    ScalarPrediction,
    SpotConnectivityReference,
    apply_spot_connectivity_upgrade,
    load_deployment_bundle,
)
from rheed2morph.realtime.selector import (
    _event_rows,
    _lattice_vertex_scores,
    causal_candidate_rows,
    full_lattice_fallback_eligible,
)
from rheed2morph.realtime.ui import (
    RealtimeMainWindow,
    event_pipeline_complete,
)
from rheed2morph.realtime.workers import PredictionWorker, ReplayWorker
from rheed2morph.rheed.automatic_roi_keyframe import Rect
from rheed2morph.rheed.orientation import (
    rotate_frame_clockwise,
    rotation_for_sample,
)

REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY / "configs/morphmbe_m22_realtime.json"
M22_CONFIG_PATH = REPOSITORY / "configs/morphmbe_m22_realtime.json"


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
    frames = [np.full((24, 32, 3), value, dtype=np.uint8) for value in range(16)]
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
    frames = [np.full((24, 32, 3), value, dtype=np.uint8) for value in range(18)]
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
    assert lookup["frame_m2"][:, 16, 16].tolist() == list(range(8))
    assert lookup["frame_m1"][:, 16, 16].tolist() == list(range(1, 9))
    assert lookup["base"][:, 16, 16].tolist() == list(range(2, 10))
    assert lookup["frame_p1"][:, 16, 16].tolist() == list(range(3, 11))
    assert lookup["frame_p2"][:, 16, 16].tolist() == list(range(4, 12))


def test_live_physics_matches_frozen_m14_schema() -> None:
    yy, xx = np.mgrid[:224, :224]
    frames = np.stack(
        [
            np.clip(
                20 + 180 * np.exp(-((xx - 80 - index) ** 2 + (yy - 110) ** 2) / 120),
                0,
                255,
            ).astype(np.uint8)
            for index in range(16)
        ]
    )
    physics = live_physics_row(frames)
    required = set(CURATED_RHEED_FEATURES + DYNAMIC_NUCLEATION_FEATURES)
    assert required.issubset(physics.columns)
    assert np.isfinite(physics[list(required)].to_numpy(float)).all()


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


def test_causal_vertex_uses_only_bounded_past_and_four_frame_lookahead() -> None:
    history = []
    vertex = 12
    for index in range(17):
        x = 60.0 - 0.55 * (index - vertex) ** 2
        history.append(
            {
                "frame_index": index,
                "spot_x": x,
                "spot_y": 120.0 - index,
                "compact_spot_x": x - 1.0,
                "compact_spot_y": 119.0 - index,
                "clarity": 8.0,
                "sharpness": 0.2,
                "spot_energy": 0.1,
                "mean_intensity": 0.3,
                "absolute_contrast": 0.1,
            }
        )
    candidates = causal_candidate_rows(history, lookahead_frames=4)
    assert candidates
    assert {int(row["frame_index"]) for row in candidates} == {vertex}
    assert all(bool(row["direction_consistent"]) for row in candidates)


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


def test_default_ui_uses_causal_multi_event_streaming_without_drops() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["replay_detection_mode"] == "causal_stream"
    assert config["replay_event_policy"] == "all_eligible_cycles"
    assert config["prediction_queue_capacity"] == 0
    assert config["online_full_lattice_fallback_enabled"] is True
    assert config["online_fallback_confirmation_delay_frames"] == 8
    source = inspect.getsource(ReplayWorker._run_causal_stream)
    assert "analyze_replay" not in source
    assert "detector.observe" in source
    assert "pending[trigger]" in source
    assert "prediction_job.emit" in source
    assert "stream_summary.emit" in source
    assert "fallback_roi" in source
    assert "full-lattice safety fallback" in source
    worker = PredictionWorker(bundle_path="unused.joblib")
    assert worker.jobs.maxsize == 0


def test_session_completes_only_when_every_clear_moment_is_plotted() -> None:
    assert event_pipeline_complete(13, 13, 13, 13)
    assert not event_pipeline_complete(0, 0, 0, 0)
    assert not event_pipeline_complete(13, 13, 3, 3)
    assert not event_pipeline_complete(13, 12, 12, 12)
    assert not event_pipeline_complete(13, 13, 13, 12)


def test_full_lattice_fallback_rejects_shadow_and_sparse_patterns() -> None:
    clear = {
        "visibility_proxy": 1.44,
        "raw_shadow_fraction": 0.04,
        "spot_peak_count": 11.0,
        "clarity": 8.28,
    }
    thresholds = {
        "minimum_score": 0.30,
        "minimum_visibility_proxy": 1.30,
        "maximum_shadow_fraction": 0.20,
        "minimum_spot_peak_count": 8.0,
        "minimum_clarity": 8.0,
    }
    assert full_lattice_fallback_eligible(clear, 0.34, **thresholds)
    assert not full_lattice_fallback_eligible(
        {**clear, "raw_shadow_fraction": 0.60},
        0.34,
        **thresholds,
    )
    assert not full_lattice_fallback_eligible(
        {**clear, "spot_peak_count": 2.0},
        0.34,
        **thresholds,
    )
    assert not full_lattice_fallback_eligible(clear, 0.20, **thresholds)


def test_deployment_cache_identifies_frozen_nonretrieval_pipeline() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    bundle_path = REPOSITORY / config["deployment_bundle"]
    if not bundle_path.exists():
        return
    bundle = load_deployment_bundle(bundle_path)
    assert bundle.model_id == M22_MODEL_ID
    generation_config = json.loads(
        (REPOSITORY / config["generation_config"]).read_text(encoding="utf-8")
    )
    assert len(bundle.groups) == generation_config["expected_growth_count"]
    assert bundle.generation_config["selected_method"] == ("M22c_gap_completion_strong")
    assert "line3" in bundle.generation_config["afm_descriptors"]
    assert bundle.rq_reference.confidence_risk_reference is not None
    assert bundle.rq_reference.confidence_error_reference is not None
    assert bundle.fsmi_reference.confidence_risk_reference is not None
    assert bundle.endpoint_streak_reference is not None
    assert bundle.endpoint_confidence_risk_reference is not None
    assert bundle.endpoint_confidence_error_reference is not None
    assert bundle.retrieval_at_inference is False
    assert bundle.measured_afm_patch_at_inference is False


def test_realtime_config_and_ui_identify_m20_m22c_pipeline() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest_path = REPOSITORY / config["deployment_manifest"]
    assert Path(config["deployment_bundle"]).name == "morphmbe_m22.joblib"
    assert config["rheed_rotation_clockwise_degrees_by_sample"] == {
        "6389": 90,
        "6390": 90,
    }
    assert config["replay_detection_mode"] == "causal_stream"
    assert config["replay_event_policy"] == "all_eligible_cycles"
    assert set(config["replay_keyframe_override_by_sample"]) == {
        "6389",
        "6390",
    }
    assert (
        config["replay_keyframe_override_by_sample"]["6389"]["source_name"]
        == "Rampdown to 300C.avi"
    )
    assert manifest_path.name == "morphmbe_m22_manifest.json"
    assert config["metrology_audited_mode"] is True
    assert (
        "line3"
        in load_config(REPOSITORY / config["generation_config"])["afm_target_variant"]
    )
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["model_id"] == M22_MODEL_ID
        assert manifest["method"]["image_generator"] == ("M22c_gap_completion_strong")
        assert manifest["method"]["Sq_nm"] == ("M20_spot_connectivity_calibrated_sq")
        assert manifest["method"]["legacy_internal_target_name"] == "Rq_nm"
        assert "third-order" in manifest["method"]["afm_metrology"]
        assert manifest["method"]["retrieval_at_inference"] is False
        assert manifest["method"]["measured_afm_patch_at_inference"] is False
    assert config["ui_scalar_model_label"] == ("M20 spot-connectivity calibrated Sq")
    assert config["ui_generator_model_label"] == (
        "M22c dense-intermediate gap-completion AFM generation"
    )
    source = inspect.getsource(RealtimeMainWindow._build_ui)
    assert "actually passed to the scalar and" in source
    assert "actually passed to M14i/M12a" not in source


def test_standalone_nested_config_resolves_archive_root(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    nested = root / "configs" / "standalone"
    (root / "src" / "rheed2morph").mkdir(parents=True)
    nested.mkdir(parents=True)
    config_path = nested / "ui.json"
    config_path.write_text("{}\n", encoding="utf-8")
    assert repository_root_from_config(config_path, {"repository_root": "."}) == root


def test_realtime_renderer_receives_predicted_island_target() -> None:
    source = inspect.getsource(RealtimeMorphologyPredictor.predict)
    assert "island_target=island_target" in source
    assert 'config["selected_renderer"]' in source


def test_m22_realtime_config_resolves_dense_mid_generator() -> None:
    config = json.loads(M22_CONFIG_PATH.read_text(encoding="utf-8"))
    generation = load_config(REPOSITORY / config["generation_config"])

    assert generation["expected_growth_count"] == 27
    assert generation["selected_method"] == "M22c_gap_completion_strong"
    assert generation["selected_renderer"] == {
        "mode": "regime_adaptive_separated_islands",
        "island_generator_mode": ("separated_ellipse_growth_layered_gapfill_strong"),
        "rough_start_nm": 2.2,
        "rough_full_nm": 3.6,
        "rough_structure_weight": 0.85,
        "rough_texture_weight": 0.15,
        "rough_texture_sigma_px": 2.0,
        "rough_tip_sigma_px": 0.5,
        "rough_isolation_strength": 1.0,
    }
    assert config["ui_model_badge"] == "M20 + M22c | READY"
    assert config["deployment_bundle"] == "assets/models/morphmbe_m22.joblib"
    assert "M22c-DenseMidGapCompletion" in M22_MODEL_ID


def test_online_m20_spot_connectivity_upgrade_uses_query_rheed_only() -> None:
    physics = np.asarray(
        [
            [-3.0, 9.0, 0.15, 0.0005],
            [-2.8, 8.0, 0.20, 0.0008],
            [-2.6, 7.0, 0.25, 0.0010],
        ],
        dtype=float,
    )
    from analysis.rheed_rough_island_redesign.connectivity import (
        CONNECTIVITY_FEATURES,
    )

    physical_frame = {
        name: physics[:, index] for index, name in enumerate(CONNECTIVITY_FEATURES)
    }
    query = {
        name: [physics[0, index]] for index, name in enumerate(CONNECTIVITY_FEATURES)
    }
    reference = SpotConnectivityReference(
        groups=["a", "b", "c"],
        raw_features=_feature_matrix(pd.DataFrame(physical_frame)),
        log_residuals=np.full(3, 0.10),
        isolated_spot_threshold=1.1,
    )
    rq = ScalarPrediction(
        value=4.0,
        unconstrained_value=4.0,
        support_clipped=False,
        expected_absolute_error=0.5,
        confidence=0.7,
        interval_lower=3.0,
        interval_upper=5.0,
        risk_score=0.3,
        tta_confidence=0.8,
        rotation_period_risk=0.2,
        head_agreement_confidence=0.9,
    )
    endpoint = EndpointPrediction(
        value_nm=4.0,
        temporal_5_nm=4.2,
        temporal_8_nm=4.0,
        streak_expert_nm=3.0,
        streak_gate=False,
        rough_consensus_gate=True,
        streak_threshold=2.0,
        upper_threshold_nm=3.5,
        expert_log_range=0.2,
        nearest_embedding_distance=0.3,
        streak_robust_z=0.4,
    )

    upgraded, isolation = apply_spot_connectivity_upgrade(
        rq,
        endpoint,
        pd.DataFrame(query),
        reference,
    )

    assert upgraded.value > rq.value
    assert np.isclose(upgraded.interval_upper - upgraded.value, 1.0)
    assert 0.0 <= isolation <= 1.0


def test_realtime_ui_source_contains_no_cjk_text() -> None:
    realtime_root = REPOSITORY / "src" / "rheed2morph" / "realtime"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(realtime_root.glob("*.py"))
    )
    assert not any("\u3400" <= character <= "\u9fff" for character in source)


def test_v8_model_input_roi_bundle_has_strict_loo_provenance() -> None:
    import joblib

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    bundle_path = REPOSITORY / config["model_input_roi_calibration"]
    if not bundle_path.exists():
        return
    bundle = joblib.load(bundle_path)
    assert bundle["validation_method"] == ("v8_orientation_model_input_q20_q80")
    assert bundle["validation_protocol"] == "strict_leave_one_video_out"
    assert bundle["held_video_overlap_sum"] == 0
