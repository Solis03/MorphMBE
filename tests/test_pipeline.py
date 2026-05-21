from __future__ import annotations

import unittest

from rheed2morph.pipeline import run


class PipelineConfigTest(unittest.TestCase):
    def test_pipeline_paths_are_project_relative(self) -> None:
        self.assertEqual(str(run.RAW_AFM_DIR), "data/raw/raw_AFM")
        self.assertEqual(str(run.RAW_RHEED_DIR), "data/raw/raw_RHEED")
        self.assertEqual(str(run.PAIR_ROOT), "data/pair")
        self.assertEqual(str(run.PROCESSED_AFM_ROOT), "data/processed_afm")
        self.assertEqual(str(run.PLANE_CORRECTED_AFM_ROOT), "data/plane_corrected_afm")
        self.assertEqual(str(run.REPORT_FIGURES_ROOT), "reports/figures/afm_scan_size_grids")
        self.assertEqual(str(run.AFM_RECON_ROOT), "data/afm_descriptor_reconstruction")
        self.assertEqual(
            str(run.AFM_RECON_LARGE_ROOT),
            "data/afm_descriptor_reconstruction_large",
        )

    def test_reconstruction_steps_are_configured(self) -> None:
        labels = [label for label, _ in run.reconstruction_steps()]
        self.assertIn("Build 1um AFM reconstruction manifest", labels)
        self.assertIn("Train 1um descriptor PCA decoder", labels)
        self.assertIn("Train large AFM descriptor MLP decoder", labels)
