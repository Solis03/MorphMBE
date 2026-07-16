from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def latest_freeze() -> Path:
    latest = REPO / "paper_freeze" / "LATEST_FREEZE.txt"
    if not latest.exists():
        raise unittest.SkipTest("paper_freeze/LATEST_FREEZE.txt is not present")
    root = REPO / latest.read_text(encoding="utf-8").strip()
    if not root.exists():
        raise unittest.SkipTest(f"latest freeze root missing: {root}")
    return root


class FrozenUnseenInferenceTest(unittest.TestCase):
    def test_batch_prediction_runs_without_afm_target_access(self) -> None:
        root = latest_freeze()
        manifest = json.loads((root / "01_FREEZE_AND_PROVENANCE/FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_manifest = tmp_path / "unseen_manifest.csv"
            fields = json.loads((root / "13_UNSEEN_INFERENCE/input_schema.json").read_text(encoding="utf-8"))["required"]
            row = {k: "" for k in fields}
            row.update(
                {
                    "sample_id": "UNSEEN_TEST_001",
                    "video_path": "future_unseen_readonly.mp4",
                    "frames_dir": "future_frames",
                    "metadata_path": "future_metadata.json",
                    "keyframe_index": "10",
                    "clip_start_index": "2",
                    "clip_end_index": "17",
                    "roi_x": "1",
                    "roi_y": "2",
                    "roi_width": "128",
                    "roi_height": "96",
                    "source_width": "640",
                    "source_height": "480",
                    "growth_stage": "prospective_test",
                    "notes": "unit test synthetic manifest",
                }
            )
            with input_manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)
            out = tmp_path / "predictions"
            subprocess.check_call(
                [
                    sys.executable,
                    str(root / "13_UNSEEN_INFERENCE/predict_unseen_batch.py"),
                    "--bundle-root",
                    str(root),
                    "--manifest",
                    str(input_manifest),
                    "--output-root",
                    str(out),
                    "--freeze-id",
                    manifest["freeze_id"],
                ],
                cwd=REPO,
            )
            pred_dir = out / "UNSEEN_TEST_001"
            pred = json.loads((pred_dir / "prediction.json").read_text(encoding="utf-8"))
            self.assertEqual(pred["freeze_id"], manifest["freeze_id"])
            self.assertFalse(pred["uses_unknown_afm_target"])
            self.assertEqual(pred["support_level"], "unseen_pending_qc")
            self.assertNotIn("afm_target", row)
            training_ids = np.load(next((root / "12_FULL_COHORT_DEPLOYMENT/quantitative_model").glob("model_*.npz")), allow_pickle=False)["training_sample_ids"]
            self.assertNotIn("UNSEEN_TEST_001", {str(x) for x in training_ids.tolist()})
            source_paths = pred["retrieved_AFM_source_paths"]
            self.assertTrue(source_paths)
            self.assertTrue(all(p.startswith("12_FULL_COHORT_DEPLOYMENT/visual_model/physical_maps/") for p in source_paths))
            self.assertTrue((pred_dir / "representative_afm.png").exists())
            self.assertTrue(np.load(pred_dir / "representative_afm.npy", allow_pickle=False).size > 0)


if __name__ == "__main__":
    unittest.main()
