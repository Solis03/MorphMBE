from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.rheed_video_afm_story.afm_training import group_balanced_epoch_indices
from analysis.rheed_video_afm_story.common import repo_path, sha256_file
from analysis.rheed_video_afm_story.rq_disentanglement import physical_from_q, project_unit_rq_np, rq_np


CONFIG_PATH = Path("configs/rheed_video_afm_story_phase3a.yaml")
OUTPUT_ROOT = Path("outputs/rheed_video_afm_story/phase3a")


def load_config() -> dict:
    return json.loads(repo_path(CONFIG_PATH).read_text())


def manifest() -> pd.DataFrame:
    return pd.read_csv(repo_path(OUTPUT_ROOT / "afm_decoder_manifest.csv"), dtype={"sample_id": str, "growth_run_id": str})


def test_phase3a_provenance_inputs_are_unchanged() -> None:
    config = load_config()
    assert sha256_file(config["phase1_afm_audit_path"]) == config["phase1_afm_audit_hash"]
    assert sha256_file(config["phase1_manifest_path"]) == config["phase1_manifest_hash"]
    assert sha256_file(config["phase2a_summary_path"]) == config["phase2a_summary_hash"]
    assert sha256_file(config["removelist_path"]) == config["expected_removelist_hash"]


def test_removelist_applied_before_afm_training_artifacts() -> None:
    excluded = set(load_config()["excluded_samples"])
    m = manifest()
    assert not (set(m["sample_id"]) & excluded)
    for table in ["afm_morphology_bank.csv", "prototype_assignments.csv", "afm_scan_latents.csv", "afm_sample_latent_summary.csv"]:
        df = pd.read_csv(repo_path(OUTPUT_ROOT / table), dtype={"sample_id": str})
        assert not (set(df["sample_id"]) & excluded)
    audit = pd.read_csv(repo_path(OUTPUT_ROOT / "afm_decoder_audit.csv"), dtype={"sample_id": str})
    excluded_rows = audit[audit["sample_id"].isin(excluded)]
    assert not excluded_rows.empty
    assert excluded_rows["excluded_by_removelist"].astype(bool).all()
    assert not excluded_rows["quality_pass"].astype(bool).any()


def test_manifest_uses_nm_plane_corrected_1um_arrays_not_rendered_images() -> None:
    m = manifest()
    assert m["height_unit"].eq("nm").all()
    assert np.allclose(m["scan_size_x_um"], 1.0)
    assert np.allclose(m["scan_size_y_um"], 1.0)
    assert m["quality_pass"].astype(bool).all()
    assert m["plane_corrected_array_path"].str.endswith("_plane_corrected.npy").all()
    assert not m["plane_corrected_array_path"].str.contains("render|png|colorbar", case=False).any()
    assert m["source_array_hash"].str.len().eq(64).all()


def test_group_outer_split_has_no_scan_or_growth_group_leakage() -> None:
    split = pd.read_csv(repo_path(OUTPUT_ROOT / "group_outer_splits.csv"), dtype={"growth_run_id": str})
    m = manifest()
    for fold, g in split.groupby("fold"):
        duplicated = g.groupby("growth_run_id")["split"].nunique()
        assert duplicated.eq(1).all()
        test_groups = set(g.query("split == 'test'")["growth_run_id"])
        train_groups = set(g.query("split == 'train'")["growth_run_id"])
        assert test_groups.isdisjoint(train_groups)
        assert set(m["growth_run_id"]) == test_groups | train_groups
    assert split.query("split == 'test'").groupby("growth_run_id").size().eq(1).all()


def test_unit_shape_projection_and_physical_rq_recomposition() -> None:
    m = manifest().head(12)
    for row in m.to_dict("records"):
        paths = json.loads(row["unit_shape_paths"])
        shape = np.load(repo_path(paths["128"]))
        assert abs(float(shape.mean())) < 1e-4
        assert abs(rq_np(shape) - 1.0) < 1e-4
        projected = project_unit_rq_np(shape * 2.0 + 7.0)
        assert abs(float(projected.mean())) < 1e-4
        assert abs(rq_np(projected) - 1.0) < 1e-4
        physical = physical_from_q(shape, float(row["rq_nm"]))
        assert abs(rq_np(physical) - float(row["rq_nm"])) < 1e-3
        resize_audit = json.loads(row["resize_audit"])
        assert abs(resize_audit["128"]["shape_rq"] - 1.0) < 1e-4


def test_pca_and_autoencoder_are_fit_inside_training_groups_only() -> None:
    pca_registry = pd.read_csv(repo_path(OUTPUT_ROOT / "pca_model_registry.csv"))
    assert pca_registry.query("status == 'complete'")["fit_scope"].eq("outer_train_groups_only").all()
    ae_registry = pd.read_csv(repo_path(OUTPUT_ROOT / "decoder_model_registry.csv"))
    assert ae_registry["fit_scope"].eq("outer_train_groups_only").all()
    for row in ae_registry.to_dict("records"):
        val_groups = set(json.loads(row["validation_group_ids"]))
        test_groups = set(json.loads(row["test_group_ids"]))
        assert val_groups.isdisjoint(test_groups)
    history = pd.read_csv(repo_path(OUTPUT_ROOT / "autoencoder_training_history.csv"))
    assert history["validation_groups_excluded_from_gradient_update"].astype(bool).all()


def test_group_balanced_sampler_limits_scan_count_dominance() -> None:
    groups = np.asarray(["many"] * 20 + ["few"] * 1 + ["mid"] * 5)
    rng = np.random.default_rng(17)
    selected = group_balanced_epoch_indices(groups, rng)
    selected_groups = groups[selected]
    counts = pd.Series(selected_groups).value_counts()
    assert counts.to_dict() == {"few": 1, "many": 1, "mid": 1}


def test_autoencoder_architecture_constraints_and_latent_dims() -> None:
    registry = pd.read_csv(repo_path(OUTPUT_ROOT / "decoder_model_registry.csv"))
    assert set(registry["latent_dim"]).issubset({16, 32, 64})
    assert registry["has_unet_skip_connections"].astype(bool).eq(False).all()
    source = repo_path("analysis/rheed_video_afm_story/afm_autoencoder.py").read_text()
    assert "ConvTranspose2d" not in source
    assert "Sigmoid" not in source
    assert "Tanh" not in source
    assert "has_unet_skip_connections = False" in source


def test_prototypes_are_sample_level_and_bootstrap_unit_is_sample() -> None:
    assignments = pd.read_csv(repo_path(OUTPUT_ROOT / "prototype_assignments.csv"), dtype={"sample_id": str})
    assert len(assignments) == manifest()["sample_id"].nunique()
    assert assignments["sample_id"].is_unique
    stability = pd.read_csv(repo_path(OUTPUT_ROOT / "prototype_stability.csv"))
    assert stability["bootstrap_unit"].eq("sample").all()


def test_morphology_bank_is_retrieval_ready_and_global_model_is_marked_transductive() -> None:
    bank = pd.read_csv(repo_path(OUTPUT_ROOT / "afm_morphology_bank.csv"), dtype={"sample_id": str, "growth_run_id": str})
    assert len(bank) == manifest()["sample_id"].nunique()
    for col in ["growth_run_id", "physical_map_path", "unit_shape_map_path", "latent_path", "decoder_reconstruction_path", "high_frequency_residual_path", "prototype_id"]:
        assert col in bank.columns
        assert bank[col].notna().all()
    global_registry = pd.read_csv(repo_path(OUTPUT_ROOT / "global_development_model/global_development_model_registry.csv"))
    assert global_registry["global_transductive_development_model"].astype(bool).all()
    report = repo_path("reports/rheed_video_afm_story/phase3a/phase3a_report.md").read_text()
    assert "transductive" in report.lower()
    assert "No RHEED-to-latent model is trained" in report
