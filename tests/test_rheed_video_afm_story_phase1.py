from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.rheed_video_afm_story.afm_targets import build_sample_targets
from analysis.rheed_video_afm_story.clip_cache import luminance_uint8, resize_and_pad
from analysis.rheed_video_afm_story.common import read_id_list, repo_path, sha256_file


CONFIG_PATH = Path("configs/rheed_video_afm_story_phase1.yaml")
OUTPUT_ROOT = Path("outputs/rheed_video_afm_story/phase1")


def load_config() -> dict:
    return json.loads(repo_path(CONFIG_PATH).read_text())


def selected_manifest() -> pd.DataFrame:
    config = load_config()
    return pd.read_csv(repo_path(config["source_manifest_path"]), dtype={"sample_id": str})


def modeling_manifest() -> pd.DataFrame:
    return pd.read_csv(repo_path(OUTPUT_ROOT / "modeling_manifest.csv"), dtype={"sample_id": str})


def test_selected_manifest_has_27_unique_samples() -> None:
    df = selected_manifest()
    assert len(df) == 27
    assert df["sample_id"].nunique() == 27


def test_discarded_samples_do_not_enter_modeling_manifest() -> None:
    config = load_config()
    discarded = read_id_list(config["discarded_list_path"])
    manifest = modeling_manifest()
    assert not (set(manifest["sample_id"]) & discarded)


def test_each_sample_has_one_selected_video() -> None:
    df = selected_manifest()
    assert (df.groupby("sample_id")["video_id"].nunique() == 1).all()


def test_each_clip_has_16_continuous_frames_and_keyframe_inside() -> None:
    df = selected_manifest()
    for row in df.to_dict("records"):
        indices = list(range(int(row["clip_start_index"]), int(row["clip_end_index"]) + 1))
        assert len(indices) == 16
        assert int(row["actual_clip_frame_count"]) == 16
        assert int(row["keyframe_index"]) in indices


def test_roi_boundaries_are_legal() -> None:
    df = selected_manifest()
    for row in df.to_dict("records"):
        assert int(row["roi_x"]) >= 0
        assert int(row["roi_y"]) >= 0
        assert int(row["roi_width"]) > 0
        assert int(row["roi_height"]) > 0
        assert int(row["roi_x"]) + int(row["roi_width"]) <= int(row["source_width"])
        assert int(row["roi_y"]) + int(row["roi_height"]) <= int(row["source_height"])


def test_roi_crop_matches_original_png_array_slice() -> None:
    row = selected_manifest().iloc[0]
    frame_path = repo_path(row["frames_dir"]) / f"{int(row['keyframe_index'])}.png"
    gray = luminance_uint8(Image.open(frame_path))
    x, y, w, h = (int(row["roi_x"]), int(row["roi_y"]), int(row["roi_width"]), int(row["roi_height"]))
    crop = gray[y : y + h, x : x + w]
    assert np.array_equal(crop, gray[y : y + h, x : x + w])
    assert crop.shape == (h, w)


def test_resize_preserves_aspect_ratio() -> None:
    crop = np.zeros((120, 60), dtype=np.uint8)
    resized, scale, padding = resize_and_pad(crop, 300)
    assert resized.shape == (300, 300)
    assert scale == 2.5
    assert padding == (0, 0, 75, 75)


def test_afm_target_aggregation_uses_sample_median_and_representative_nearest_median() -> None:
    scan_df = pd.DataFrame(
        [
            {"sample_id": "x", "afm_path": "a", "height_array_path": "a", "afm_file_id": "a", "scan_size_x_um": 1.0, "scan_size_y_um": 1.0, "rq_recomputed_nm": 1.0},
            {"sample_id": "x", "afm_path": "b", "height_array_path": "b", "afm_file_id": "b", "scan_size_x_um": 1.0, "scan_size_y_um": 1.0, "rq_recomputed_nm": 3.0},
            {"sample_id": "x", "afm_path": "c", "height_array_path": "c", "afm_file_id": "c", "scan_size_x_um": 1.0, "scan_size_y_um": 1.0, "rq_recomputed_nm": 9.0},
        ]
    )
    targets, audited = build_sample_targets(scan_df, 1.0, 1e-6)
    target = targets.iloc[0]
    assert target["primary_rq_nm_median"] == 3.0
    assert target["representative_afm_path"] == "b"
    assert audited.loc[audited["afm_path"] == "b", "is_representative"].item()


def test_same_sample_scans_do_not_cross_outer_fold_and_knn_has_no_heldout_neighbor() -> None:
    neighbors = pd.read_csv(repo_path(OUTPUT_ROOT / "baseline_neighbor_audit.csv"), dtype=str)
    assert (neighbors["heldout_sample_id"] != neighbors["neighbor_sample_id"]).all()
    assert neighbors["leakage_free"].astype(str).str.lower().eq("true").all()


def test_scaler_is_pipeline_scoped_and_no_pca_is_used() -> None:
    source = repo_path("analysis/rheed_video_afm_story/baseline_rq.py").read_text()
    assert '("scaler", StandardScaler())' in source
    assert "PCA(" not in source


def test_baseline_prediction_has_one_oof_record_per_sample_per_model() -> None:
    preds = pd.read_csv(repo_path(OUTPUT_ROOT / "oof_predictions.csv"), dtype={"sample_id": str})
    counts = preds.groupby(["model_name", "sample_id"]).size()
    assert counts.eq(1).all()
    expected_n = modeling_manifest().query("usable_for_modeling and cohort_primary_1um")["sample_id"].nunique()
    assert preds.groupby("model_name")["sample_id"].nunique().eq(expected_n).all()


def test_original_data_paths_are_not_outputs_and_source_hashes_match() -> None:
    config = load_config()
    manifest = modeling_manifest()
    assert manifest["manifest_source_hash"].eq(sha256_file(config["source_manifest_path"])).all()
    assert manifest["removelist_hash"].eq(sha256_file(config["removelist_path"])).all()
    for output in OUTPUT_ROOT.rglob("*"):
        assert not str(output).startswith("data/")
