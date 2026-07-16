from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.rheed_video_afm_story.common import repo_path


VARIANT = Path("outputs/rheed_video_afm_story/variants/afm_second_order_y2_v1")
REPORT = Path("reports/rheed_video_afm_story/variants/afm_second_order_y2_v1")


def read_json(path: str | Path) -> dict:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def truthy(value: object) -> bool:
    if isinstance(value, str):
        return value == "True"
    return bool(value)


class SecondOrderRerunTest(unittest.TestCase):
    def test_mapping_is_complete_and_second_order_only(self) -> None:
        mapping = pd.read_csv(repo_path(VARIANT / "provenance/second_order_scan_mapping.csv"), dtype={"sample_id": str})
        self.assertEqual(260, len(mapping))
        self.assertTrue(mapping["mapping_status"].eq("ok").all())
        self.assertTrue(mapping["output_exists"].map(truthy).all())
        self.assertTrue(mapping["shape_matches_source"].map(truthy).all())
        self.assertTrue(mapping["finite_policy_ok"].map(truthy).all())
        self.assertTrue(mapping["height_unit"].eq("nm").all())
        self.assertTrue(mapping["second_order_afm_path"].str.startswith("data/afm_second_order/").all())
        self.assertFalse(mapping["second_order_afm_path"].str.contains("_backgrounds", regex=False).any())
        self.assertEqual(len(mapping), mapping["source_afm_path"].nunique())
        self.assertEqual(len(mapping), mapping["second_order_afm_path"].nunique())

    def test_primary_cohort_is_fixed_23_without_removed_samples(self) -> None:
        manifest = pd.read_csv(repo_path(VARIANT / "targets/second_order_modeling_manifest.csv"), dtype={"sample_id": str})
        primary = manifest[manifest["usable_for_modeling"].map(truthy) & manifest["cohort_primary_1um"].map(truthy)]
        primary = primary[~primary["sample_id"].isin(["6023", "6087"])]
        self.assertEqual(23, len(primary))
        self.assertEqual(23, primary["sample_id"].nunique())
        self.assertTrue(primary["representative_afm_path"].str.startswith("data/afm_second_order/").all())
        self.assertTrue(primary["afm_target_variant"].eq("second_order_y2").all())

    def test_frozen_retrieval_weights_are_used(self) -> None:
        expected = {"w_q": 2.0, "w_rheed": 0.25, "w_phys": 0.25, "w_stage": 0.0, "top_k": 1}
        self.assertEqual(expected, read_json(VARIANT / "phase4a/retrieval_final_weights.json"))
        self.assertEqual(expected, read_json(VARIANT / "phase4a/frozen_phase4a_settings.json"))
        config = read_json(VARIANT / "provenance/phase4a_effective_config.json")
        self.assertEqual(expected, config["frozen_retrieval_weights"])

    def test_oof_r4_predictions_are_one_per_primary_sample(self) -> None:
        manifest = pd.read_csv(repo_path(VARIANT / "targets/second_order_modeling_manifest.csv"), dtype={"sample_id": str})
        primary_ids = set(
            manifest[
                manifest["usable_for_modeling"].map(truthy)
                & manifest["cohort_primary_1um"].map(truthy)
                & ~manifest["sample_id"].isin(["6023", "6087"])
            ]["sample_id"]
        )
        rq = pd.read_csv(repo_path(VARIANT / "rq_models/second_order_rq_oof_predictions.csv"), dtype={"sample_id": str})
        r4 = rq[rq["model_id"].eq("R4_auto_iso_dino_residual")]
        self.assertEqual(primary_ids, set(r4["sample_id"]))
        self.assertEqual(23, len(r4))
        self.assertTrue(r4["afm_target_variant"].eq("second_order_y2").all())

    def test_retrieval_is_outer_training_only_and_second_order_weighted(self) -> None:
        candidates = pd.read_csv(
            repo_path(VARIANT / "phase4a/second_order_oof_retrieval_candidates.csv"),
            dtype={"heldout_sample_id": str, "candidate_sample_id": str},
        )
        self.assertFalse((candidates["heldout_sample_id"] == candidates["candidate_sample_id"]).any())
        selected = candidates[candidates["selected"].map(truthy)]
        self.assertEqual(23, selected["heldout_sample_id"].nunique())
        self.assertTrue(selected["rank"].eq(1).all())
        self.assertTrue(candidates["w_q"].eq(2.0).all())
        self.assertTrue(candidates["w_rheed"].eq(0.25).all())
        self.assertTrue(candidates["w_phys"].eq(0.25).all())
        self.assertTrue(candidates["w_stage"].eq(0.0).all())
        self.assertTrue(candidates["top_k"].eq(1).all())

    def test_synthesis_outputs_are_second_order_and_materialized_by_sample(self) -> None:
        synth = pd.read_csv(repo_path(VARIANT / "phase4a/second_order_oof_synthesis_outputs.csv"), dtype={"sample_id": str})
        self.assertEqual(23, synth["sample_id"].nunique())
        self.assertTrue(synth["afm_target_variant"].eq("second_order_y2").all())
        self.assertTrue(synth["uses_predicted_rq_not_true_rq"].map(truthy).all())
        for sample_id in synth["sample_id"].unique():
            sample_dir = repo_path(VARIANT / "phase4a/synthesized_afm_maps_by_sample" / sample_id)
            for name in ["S0.npy", "S1.npy", "S2.npy"]:
                self.assertTrue((sample_dir / name).exists(), f"{sample_id}/{name}")
            self.assertGreaterEqual(len(list(sample_dir.glob("S3_seed*.npy"))), 1)
            self.assertGreaterEqual(len(list(sample_dir.glob("S4_seed*.npy"))), 1)

    def test_phase4b_uses_second_order_ground_truth_and_old_main_selection(self) -> None:
        sample = pd.read_csv(repo_path(VARIANT / "phase4b_visualization/sample_level_results.csv"), dtype={"sample_id": str})
        self.assertEqual(23, len(sample))
        self.assertTrue(sample["ground_truth_afm_path"].str.startswith("data/afm_second_order/").all())
        self.assertTrue(sample["s1_source_afm_path"].str.startswith("data/afm_second_order/").all())
        second_sel = read_json(VARIANT / "phase4b_visualization/main_figure_sample_selection.json")
        first_sel = read_json("outputs/rheed_video_afm_story/phase4b_visualization/main_figure_sample_selection.json")
        self.assertEqual(first_sel["selected_sample_ids"], second_sel["selected_sample_ids"])
        self.assertFalse(second_sel["uses_prediction_error"])

    def test_validation_passes_and_old_inputs_are_unchanged(self) -> None:
        validation = read_json(VARIANT / "provenance/second_order_rerun_validation.json")
        self.assertTrue(validation["passed"])
        for key, value in validation.items():
            if key != "passed":
                self.assertTrue(truthy(value), key)

    def test_comparison_artifacts_and_summary_exist(self) -> None:
        summary = read_json(VARIANT / "second_order_rerun_summary.json")
        self.assertTrue(summary["mapping_complete"])
        self.assertEqual(23, summary["primary_growth_group_count"])
        self.assertEqual(10, summary["representative_scan_changed_count"])
        self.assertEqual(21, summary["comparison"]["retrieved_source_changed_count"])
        self.assertIn("second_r4_mae", summary["comparison"])
        for path in summary["comparison_figures"]:
            self.assertTrue(repo_path(path).exists(), path)
        for name in [
            "first_vs_second_order_sample_results.csv",
            "first_vs_second_order_model_metrics.csv",
            "first_vs_second_order_synthesis_metrics.csv",
            "target_rank_change.csv",
        ]:
            self.assertTrue(repo_path(VARIANT / "comparison" / name).exists(), name)

    def test_variant_registry_records_controlled_ablation(self) -> None:
        registry = read_json(VARIANT / "variant_registry.json")
        self.assertEqual("afm_second_order_y2_v1", registry["variant_id"])
        self.assertEqual("data/afm_second_order", registry["afm_preprocessing"]["source"])
        self.assertEqual("y2", registry["afm_preprocessing"]["model"])
        self.assertFalse(registry["rheed_inputs_changed"])
        self.assertFalse(registry["sample_cohort_changed"])
        self.assertFalse(registry["retrieval_settings_changed"])
        self.assertFalse(registry["model_architecture_changed"])

    def test_phase4b_and_final_reports_exist(self) -> None:
        for path in [
            REPORT / "second_order_rerun_report.md",
            REPORT / "phase4b_visualization/results_dashboard.html",
            REPORT / "phase4b_visualization/figures/Fig2_ground_truth_vs_representative_afm_main_row_shared_scale.png",
            REPORT / "phase4b_visualization/figures/Fig2_ground_truth_vs_representative_afm_main_per_image_robust_scale.png",
            REPORT / "comparison/comparison_A_afm_preprocessing_effect.png",
            REPORT / "comparison/comparison_F_visual_output_shift.png",
        ]:
            self.assertTrue(repo_path(path).exists(), str(path))


if __name__ == "__main__":
    unittest.main()
