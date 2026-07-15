from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.rheed_video_afm_story.common import repo_path, sha256_file


CONFIG_PATH = Path("configs/rheed_video_afm_story_phase4a.yaml")
OUTPUT_ROOT = Path("outputs/rheed_video_afm_story/phase4a")
REPORT_ROOT = Path("reports/rheed_video_afm_story/phase4a")


def load_config() -> dict:
    return json.loads(repo_path(CONFIG_PATH).read_text())


def test_phase4a_provenance_hashes_are_frozen() -> None:
    config = load_config()
    for key in [
        "phase1_manifest",
        "phase1_afm_audit",
        "phase2a_summary",
        "phase2a_embedding_registry",
        "phase3a_morphology_bank",
        "phase3a_decoder_manifest",
        "phase3a_summary",
    ]:
        assert sha256_file(config[f"{key}_path"]) == config[f"{key}_hash"]
    assert sha256_file(config["removelist_path"]) == config["expected_removelist_hash"]


def test_excluded_samples_do_not_enter_phase4a_artifacts() -> None:
    excluded = set(load_config()["excluded_samples"])
    tables = [
        "rheed_physics_features.csv",
        "automatic_spot_streak_index.csv",
        "rheed_rq_oof_predictions.csv",
        "oof_retrieval_candidates.csv",
        "oof_synthesis_outputs.csv",
        "synthesis_oof_metrics.csv",
        "high_confidence_support.csv",
    ]
    for table in tables:
        df = pd.read_csv(repo_path(OUTPUT_ROOT / table), dtype={"sample_id": str})
        assert not (set(df["sample_id"]) & excluded), table


def test_rheed_physics_features_are_target_blind() -> None:
    source = repo_path("analysis/rheed_video_afm_story/rheed_physics_features.py").read_text()
    assert "primary_rq" not in source
    assert "rq_nm" not in source
    assert "afm_" not in source.lower()
    features = pd.read_csv(repo_path(OUTPUT_ROOT / "rheed_physics_features.csv"), dtype={"sample_id": str})
    assert {"spot_summary_raw", "streak_summary_raw", "connection_summary_raw", "diffuse_summary_raw"}.issubset(features.columns)


def test_expert_review_is_blinded_and_missing_labels_are_pending() -> None:
    html = repo_path(REPORT_ROOT / "rheed_expert_review/index.html").read_text()
    assert "Rq" not in html
    assert "AFM" not in html
    assert "sample_id" not in html
    status = json.loads(repo_path(REPORT_ROOT / "rheed_expert_review/expert_branch_status.json").read_text())
    assert status["expert_branch_available"] is False
    assert status["expert_label_hash"] is None


def test_rq_model_leakage_audit_and_high_confidence_are_target_blind() -> None:
    audit = pd.read_csv(repo_path(OUTPUT_ROOT / "rq_model_leakage_audit.csv"))
    assert audit["outer_target_used_for_training"].astype(bool).eq(False).all()
    assert audit["global_scaler_used"].astype(bool).eq(False).all()
    idx = pd.read_csv(repo_path(OUTPUT_ROOT / "automatic_spot_streak_index.csv"))
    assert idx["scaler_fit_scope"].eq("outer_training_samples_only").all()
    support = pd.read_csv(repo_path(OUTPUT_ROOT / "high_confidence_support.csv"))
    assert support["uses_heldout_error"].astype(bool).eq(False).all()


def test_retrieval_excludes_heldout_growth_group_and_is_group_balanced() -> None:
    audit = pd.read_csv(repo_path(OUTPUT_ROOT / "oof_retrieval_audit.csv"), dtype={"heldout_sample_id": str, "heldout_group": str})
    assert audit["contains_heldout_group"].astype(bool).eq(False).all()
    assert audit["outer_heldout_afm_in_bank"].astype(bool).eq(False).all()
    assert audit["group_balanced_first"].astype(bool).all()
    for row in audit.to_dict("records"):
        assert row["heldout_group"] not in json.loads(row["candidate_group_ids"])


def test_synthesis_uses_predicted_rq_and_patch_provenance_is_complete() -> None:
    outputs = pd.read_csv(repo_path(OUTPUT_ROOT / "oof_synthesis_outputs.csv"), dtype={"sample_id": str})
    assert outputs["uses_predicted_rq_not_true_rq"].astype(bool).all()
    metrics = pd.read_csv(repo_path(OUTPUT_ROOT / "synthesis_oof_metrics.csv"), dtype={"sample_id": str})
    assert metrics["synth_rq_minus_predicted_rq_nm"].abs().max() < 1e-3
    prov = pd.read_parquet(repo_path(OUTPUT_ROOT / "synthesis_patch_provenance.parquet"))
    assert {"sample_id", "heldout_group", "source_sample_id", "heldout_source"}.issubset(prov.columns)
    assert prov["heldout_source"].astype(bool).eq(False).all()
    assert (prov["source_sample_id"].astype(str) != prov["heldout_group"].astype(str)).all()


def test_identity_audit_prevents_hidden_copy_for_s3_s4() -> None:
    ident = pd.read_csv(repo_path(OUTPUT_ROOT / "synthesis_identity_audit.csv"), dtype={"sample_id": str})
    assert ident["exact_pixel_equality"].astype(bool).eq(False).all()
    assert ident["heldout_sample_source_contribution"].eq(0).all()
    assert ident["largest_single_source_contribution"].max() <= 0.60


def test_retrieval_and_oracle_labels_are_explicit() -> None:
    outputs = pd.read_csv(repo_path(OUTPUT_ROOT / "oof_synthesis_outputs.csv"))
    s1s2 = outputs[outputs["method"].isin(["S1_top1_real_exemplar_retrieval", "S2_topk_real_scan_medoid"])]
    assert s1s2["output_label"].str.contains("retrieval|Retrieved", case=False).all()
    oracle = pd.read_csv(repo_path(OUTPUT_ROOT / "oracle_retrieval_ceiling.csv"))
    assert oracle["oracle_label"].str.contains("ORACLE DEVELOPMENT UPPER BOUND").all()
    assert oracle["oracle_label"].str.contains("NOT A DEPLOYABLE MODEL").all()


def test_phase4a_does_not_train_forbidden_generators() -> None:
    summary = json.loads(repo_path(OUTPUT_ROOT / "phase4a_summary.json").read_text())
    assert summary["no_afm_autoencoder_trained"] is True
    assert summary["no_gan_vae_diffusion_trained"] is True
    assert repo_path(REPORT_ROOT / "phase4a_report.md").exists()
