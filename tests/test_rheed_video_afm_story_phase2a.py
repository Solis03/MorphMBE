from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.rheed_video_afm_story import confidence_support
from analysis.rheed_video_afm_story.common import repo_path, sha256_file


CONFIG_PATH = Path("configs/rheed_video_afm_story_phase2a.yaml")
OUTPUT_ROOT = Path("outputs/rheed_video_afm_story/phase2a")


def load_config() -> dict:
    return json.loads(repo_path(CONFIG_PATH).read_text())


def test_phase2a_provenance_and_primary_cohort_are_locked() -> None:
    config = load_config()
    assert sha256_file(config["phase1_manifest_path"]) == config["phase1_manifest_hash"]
    assert sha256_file(config["removelist_path"]) == config["expected_removelist_hash"]

    registry = pd.read_csv(repo_path(OUTPUT_ROOT / "embedding_registry.csv"))
    assert registry["N"].eq(config["expected_primary_n"]).all()
    assert registry["embedding_id"].nunique() == len(config["embedding_jobs"])

    excluded = set(config["excluded_samples"])
    for path in registry["path"]:
        with np.load(repo_path(path), allow_pickle=False) as npz:
            sample_ids = {str(x) for x in npz["sample_ids"].tolist()}
            assert len(sample_ids) == config["expected_primary_n"]
            assert not (sample_ids & excluded)
            assert str(npz["source_manifest_hash"]) == config["phase1_manifest_hash"]


def test_clip_variants_are_available_and_temporally_well_formed() -> None:
    config = load_config()
    manifest = pd.read_csv(repo_path(OUTPUT_ROOT / "clip_variants_manifest.csv"), dtype={"sample_id": str})
    assert manifest["sample_id"].nunique() == config["expected_primary_n"]
    assert set(manifest["clip_variant"]) == set(config["clip_variants"])
    assert manifest["available"].astype(bool).all()

    for row in manifest.to_dict("records"):
        indices = json.loads(row["frame_indices"])
        assert len(indices) == int(row["frame_count"])
        assert indices == sorted(indices)
        assert len(indices) == len(set(indices))
        params = json.loads(row["preprocessing_params"])
        assert params["clip_robust_contrast"]["scope"] == "joint_clip"
        assert params["clip_robust_contrast"]["p99"] >= params["clip_robust_contrast"]["p01"]


def test_pretrained_embeddings_are_frozen_and_target_free() -> None:
    model_registry = pd.read_csv(repo_path(OUTPUT_ROOT / "model_registry.csv"))
    loaded = model_registry[model_registry["status"].isin(["loaded", "embedding_written"])]
    assert not loaded.empty
    assert loaded["frozen_requires_grad_false"].astype(bool).all()

    source = repo_path("analysis/rheed_video_afm_story/pretrained_embeddings.py").read_text()
    assert "primary_rq" not in source
    assert "representative_afm" not in source
    assert "param.requires_grad = False" in source
    assert ".eval()" in source


def test_oof_transforms_and_pairwise_training_are_fold_scoped() -> None:
    regression_source = repo_path("analysis/rheed_video_afm_story/embedding_regression.py").read_text()
    ranking_source = repo_path("analysis/rheed_video_afm_story/pairwise_ranking.py").read_text()
    assert '("scaler", StandardScaler())' in regression_source
    assert '("pca", PCA(' in regression_source
    assert '("scaler", StandardScaler())' in ranking_source
    assert '("pca", PCA(' in ranking_source

    audit = pd.read_csv(repo_path(OUTPUT_ROOT / "ranking_pair_audit.csv"), dtype=str)
    assert audit["contains_heldout"].astype(str).str.lower().eq("false").all()
    assert (audit["heldout_sample_id"] != audit["pair_i"]).all()
    assert (audit["heldout_sample_id"] != audit["pair_j"]).all()


def test_confidence_and_retrieval_neighbors_do_not_include_self() -> None:
    confidence = pd.read_csv(repo_path(OUTPUT_ROOT / "oof_confidence_support.csv"), dtype={"sample_id": str})
    assert len(confidence) == load_config()["expected_primary_n"]
    for row in confidence.to_dict("records"):
        neighbors = json.loads(row["neighbor_ids"])
        assert str(row["sample_id"]) not in neighbors

    retrieval = pd.read_csv(repo_path(OUTPUT_ROOT / "embedding_neighbor_audit.csv"), dtype={"sample_id": str})
    for row in retrieval.to_dict("records"):
        neighbors = json.loads(row["neighbor_sample_ids"])
        assert str(row["sample_id"]) not in neighbors


def test_support_score_does_not_select_models_by_heldout_error() -> None:
    source = inspect.getsource(confidence_support.support_scores)
    assert "absolute_error" not in source
    assert "support_score" in source


def test_frozen_thresholds_and_summary_are_reproducible_outputs() -> None:
    thresholds = json.loads(repo_path(OUTPUT_ROOT / "frozen_regime_thresholds.json").read_text())
    summary = json.loads(repo_path(OUTPUT_ROOT / "phase2a_summary.json").read_text())
    assert thresholds["created_once"] is True
    assert thresholds["sample_ids"] == summary["thresholds"]["sample_ids"]
    assert summary["primary_n"] == load_config()["expected_primary_n"]
    assert set(summary["go_decisions"]) == {"Go-A", "Go-B", "Go-C", "Go-D"}
