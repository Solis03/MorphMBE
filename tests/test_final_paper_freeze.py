from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]


def latest_freeze() -> Path:
    latest = REPO / "paper_freeze" / "LATEST_FREEZE.txt"
    if not latest.exists():
        raise unittest.SkipTest("paper_freeze/LATEST_FREEZE.txt is not present")
    root = REPO / latest.read_text(encoding="utf-8").strip()
    if not root.exists():
        raise unittest.SkipTest(f"latest freeze root missing: {root}")
    return root


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class FinalPaperFreezeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = latest_freeze()
        cls.manifest = json.loads((cls.root / "01_FREEZE_AND_PROVENANCE/FREEZE_MANIFEST.json").read_text(encoding="utf-8"))

    def test_freeze_folder_is_unique_and_not_symlinked(self) -> None:
        self.assertRegex(self.root.name, r"^rheed_to_afm_paper_freeze_v1_\d{8}_\d{6}$")
        self.assertEqual(self.manifest["freeze_id"], "RHEED_AFM_PAPER_FREEZE_V1_" + self.root.name.rsplit("_v1_", 1)[1].upper())
        self.assertFalse(self.root.is_symlink())
        self.assertFalse(any(p.is_symlink() for p in self.root.rglob("*")))
        self.assertTrue((self.root.parent / f"{self.root.name}.tar.gz").exists())
        self.assertTrue((self.root.parent / f"{self.root.name}.zip").exists())

    def test_canonical_cohort_and_removelist(self) -> None:
        cohort = pd.read_csv(self.root / "02_DATA_AND_COHORT/canonical_training_cohort.csv", dtype={"sample_id": str})
        afm = pd.read_csv(self.root / "12_FULL_COHORT_DEPLOYMENT/visual_model/afm_bank_manifest.csv", dtype={"sample_id": str})
        self.assertEqual(len(cohort), 23)
        self.assertEqual(len(afm), 116)
        self.assertFalse(cohort["sample_id"].isin(["6023", "6087"]).any())
        self.assertFalse(afm["sample_id"].isin(["6023", "6087"]).any())
        self.assertTrue((cohort["join_key"] == "sample_id").all())
        self.assertTrue(cohort["target_sample_id_consistent"].astype(bool).all())
        self.assertEqual(set(cohort["sample_id"]), set(pd.read_csv(self.root / "06_STRICT_OOF_RESULTS/strict_oof_predictions.csv", dtype={"sample_id": str})["sample_id"]))

    def test_strict_oof_metrics_and_source_data_are_consistent(self) -> None:
        metrics = json.loads((self.root / "06_STRICT_OOF_RESULTS/strict_oof_metrics.json").read_text(encoding="utf-8"))
        table = pd.read_csv(self.root / "09_PAPER_TABLES/Table2_rq_model_performance.csv").iloc[0].to_dict()
        numbers = json.loads((self.root / "10_FIGURE_SOURCE_DATA/paper_numbers.json").read_text(encoding="utf-8"))
        self.assertEqual(int(metrics["N"]), 23)
        self.assertAlmostEqual(metrics["MAE"], float(table["MAE"]), places=12)
        self.assertAlmostEqual(metrics["MAE"], numbers["strict_MAE"], places=12)
        self.assertTrue((self.root / "10_FIGURE_SOURCE_DATA/Fig1/source_data.csv").exists())
        self.assertTrue((self.root / "10_FIGURE_SOURCE_DATA/FigS14/source_data.json").exists())

    def test_full_cohort_model_loads_and_uses_all_23_groups(self) -> None:
        q = self.root / "12_FULL_COHORT_DEPLOYMENT/quantitative_model"
        training = pd.read_csv(q / "training_sample_ids.csv", dtype={"sample_id": str})
        self.assertEqual(len(training), 23)
        self.assertFalse(training["sample_id"].isin(["6023", "6087"]).any())
        model_paths = sorted(q.glob("model_*.npz"))
        self.assertGreaterEqual(len(model_paths), 1)
        model = np.load(model_paths[0], allow_pickle=False)
        self.assertEqual(len(model["training_sample_ids"]), 23)
        self.assertIn("coef", model.files)

    def test_checksums_and_validation_record(self) -> None:
        validation = json.loads((self.root / "15_REPRODUCIBILITY/freeze_validation.json").read_text(encoding="utf-8"))
        self.assertTrue(validation["all_passed"])
        for line in (self.root / "01_FREEZE_AND_PROVENANCE/checksums.sha256").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, rel = line.split("  ", 1)
            self.assertEqual(sha(self.root / rel), expected, rel)

    def test_readonly_input_hashes_still_match_repo(self) -> None:
        hashes = json.loads((self.root / "01_FREEZE_AND_PROVENANCE/input_artifact_hashes.json").read_text(encoding="utf-8"))
        for rel, expected in hashes.items():
            path = REPO / rel
            self.assertTrue(path.exists(), rel)
            self.assertEqual(sha(path), expected, rel)


if __name__ == "__main__":
    unittest.main()
