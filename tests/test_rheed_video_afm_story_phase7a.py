from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.rheed_video_afm_story.common import repo_path


ROOT = Path("outputs/rheed_video_afm_story/variants/afm_second_order_y2_v1/phase7a_reconstruction_first")
REPORT = Path("reports/rheed_video_afm_story/variants/afm_second_order_y2_v1/phase7a_reconstruction_first")
REMOVED = {"6023", "6087"}


def load_json(path: str | Path) -> dict:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value) == "True"


class Phase7ATest(unittest.TestCase):
    def test_primary_cohort_and_alignment_audit(self) -> None:
        idx = pd.read_csv(repo_path(ROOT / "canonical_index/canonical_sample_index.csv"), dtype={"sample_id": str})
        active = idx[idx["is_primary"].map(truthy)]
        audit = pd.read_csv(repo_path(ROOT / "provenance/phase7_alignment_audit.csv"))
        summary = load_json(ROOT / "phase7a_summary.json")
        self.assertEqual(23, active["sample_id"].nunique())
        self.assertFalse(REMOVED & set(active["sample_id"]))
        self.assertTrue(audit["passed"].map(truthy).all())
        self.assertTrue(summary["alignment_audit_passed"])

    def test_removelist_sample_zero_in_all_banks_and_outputs(self) -> None:
        for rel, col in [
            ("afm_texture_dataset/scan_manifest.csv", "sample_id"),
            ("afm_texture_dataset/patch_manifests/patch_manifest.csv", "source_sample_id"),
            ("provenance/phase7_visual_source_audit.csv", "sample_id"),
            ("all_generated_maps_manifest.csv", "sample_id"),
            ("all_patch_source_provenance.csv", "sample_id"),
            ("deployment/visual_deployment_model/active_sample_index.csv", "sample_id"),
        ]:
            df = pd.read_csv(repo_path(ROOT / rel), dtype={col: str})
            self.assertFalse(REMOVED & set(df[col]), rel)

    def test_outer_folds_are_n_minus_1(self) -> None:
        folds = pd.read_csv(repo_path(ROOT / "provenance/phase7_fold_membership.csv"), dtype={"heldout_sample_id": str})
        ids = set(folds["heldout_sample_id"])
        self.assertEqual(23, len(folds))
        for row in folds.to_dict("records"):
            train = set(json.loads(row["training_groups"]))
            self.assertEqual(22, int(row["training_group_count"]))
            self.assertEqual(ids - {str(row["heldout_sample_id"])}, train)
            self.assertTrue(truthy(row["split_valid"]))

    def test_strict_visual_sources_never_use_heldout_or_removed(self) -> None:
        prov = pd.read_csv(repo_path(ROOT / "all_patch_source_provenance.csv"), dtype={"sample_id": str})
        strict = prov[prov["track"].eq("strict")]
        self.assertGreater(len(strict), 0)
        self.assertTrue(strict["heldout_source_contribution"].astype(float).eq(0.0).all())
        for row in strict.to_dict("records"):
            sources = set(json.loads(row["source_sample_ids"]))
            self.assertNotIn(str(row["sample_id"]), sources)
            self.assertFalse(REMOVED & sources)

    def test_strict_outputs_use_predicted_rq_and_match_conditioned_rq(self) -> None:
        metrics = pd.read_csv(repo_path(ROOT / "strict_visual_metrics.csv"), dtype={"sample_id": str})
        self.assertTrue(metrics["uses_predicted_rq_not_true_rq"].map(truthy).all())
        self.assertLess(float(metrics["conditioned_rq_error_nm"].max()), 1e-3)
        sample = metrics.iloc[0]
        arr = np.load(repo_path(sample["map_path"]), allow_pickle=False)
        self.assertEqual((256, 256), arr.shape)

    def test_oracle_uses_true_descriptors_but_not_heldout_pixels(self) -> None:
        metrics = pd.read_csv(repo_path(ROOT / "oracle_visual_metrics.csv"), dtype={"sample_id": str})
        self.assertTrue(metrics["oracle_uses_true_descriptors"].map(truthy).all())
        self.assertTrue(metrics["heldout_source_contribution"].astype(float).eq(0.0).all())

    def test_full_cohort_outputs_have_warning(self) -> None:
        registry = load_json(ROOT / "deployment/visual_deployment_model/model_registry.json")
        dev = pd.read_csv(repo_path(ROOT / "development_visual_registry.csv"), dtype={"sample_id": str})
        manifest = pd.read_csv(repo_path(ROOT / "all_generated_maps_manifest.csv"), dtype={"sample_id": str})
        self.assertIn("NOT AN INDEPENDENT TEST RESULT", registry["warning"])
        self.assertTrue(dev["full_cohort_development"].map(truthy).all())
        self.assertTrue(manifest[manifest["track"].eq("development")]["warning"].str.contains("FULL-COHORT").all())

    def test_vq_and_diffusion_training_are_fold_scoped_for_strict(self) -> None:
        for path in repo_path(ROOT / "vq_models").glob("vq_registry_strict_*.json"):
            data = load_json(path)
            heldout = path.stem.split("_")[-1]
            self.assertNotIn(heldout, set(data["training_sample_ids"]))
        for path in repo_path(ROOT / "diffusion_models/checkpoints").glob("residual_diffusion_strict_*.json"):
            data = load_json(path)
            heldout = data["heldout_sample_id"]
            self.assertNotIn(heldout, set(data["training_sample_ids"]))

    def test_registry_resume_and_failures(self) -> None:
        reg = pd.read_csv(repo_path(ROOT / "visual_trial_registry.csv"))
        summary = load_json(ROOT / "phase7a_summary.json")
        self.assertEqual(204, len(reg))
        self.assertEqual(204, reg["trial_id"].nunique())
        self.assertTrue(reg["status"].eq("completed").all())
        self.assertEqual(204, summary["trial_counts"]["completed"])
        self.assertEqual(0, summary["trial_counts"]["failed"])

    def test_method_families_and_baselines_ran(self) -> None:
        metrics = pd.read_csv(repo_path(ROOT / "metrics/all_visual_metrics.csv"))
        families = set(metrics["method_family"])
        self.assertTrue({"baseline", "retrieval", "quilting", "residual", "iaaft", "texture", "vq", "diffusion"} <= families)
        self.assertIn("VB1", set(metrics["method"]))
        self.assertIn("VB2", set(metrics["method"]))
        self.assertTrue(metrics[metrics["method"].eq("VB1")]["method_label"].str.contains("Retrieved real AFM").all())

    def test_identity_audit_for_synthesis(self) -> None:
        summary = load_json(ROOT / "phase7a_summary.json")
        identity = pd.read_csv(repo_path(ROOT / "identity_audit.csv"))
        strict_synth = identity[(identity["track"].eq("strict")) & (~identity["method_family"].eq("retrieval"))]
        self.assertEqual(0.0, summary["identity_audit"]["strict_heldout_source_contribution_max"])
        self.assertFalse(strict_synth["exact_identity"].map(truthy).any())

    def test_blind_review_package_exists_without_scores(self) -> None:
        registry = load_json(ROOT / "blind_review_registry.json")
        self.assertTrue(repo_path(ROOT / "blind_review/index.html").exists())
        self.assertTrue(repo_path(ROOT / "blind_review/scoring_templates/blind_review_scoring_template.csv").exists())
        self.assertGreaterEqual(len(registry["reviews"]), 3)
        template = pd.read_csv(repo_path(ROOT / "blind_review/scoring_templates/blind_review_scoring_template.csv"))
        self.assertEqual(0, len(template))

    def test_required_reports_figures_and_hashes_exist(self) -> None:
        for rel in [
            "phase7a_report.md",
            "executive_visual_summary.md",
            "visual_methods_summary.md",
            "visual_figure_captions.md",
            "claims_and_limitations.md",
            "dashboard/results_dashboard.html",
        ]:
            self.assertTrue(repo_path(REPORT / rel).exists(), rel)
        for i in range(1, 13):
            self.assertEqual(1, len(list(repo_path(REPORT / "figures").glob(f"Fig{i}_*.png"))), f"Fig{i}")
        summary = load_json(ROOT / "phase7a_summary.json")
        self.assertEqual("8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b", summary["raw_old_hash_validation"]["removelist.txt"])

    def test_6095_6099_are_reported(self) -> None:
        summary = load_json(ROOT / "phase7a_summary.json")
        self.assertIn("6095", summary["samples_6095_6099_visual"])
        self.assertIn("6099", summary["samples_6095_6099_visual"])


if __name__ == "__main__":
    unittest.main()
