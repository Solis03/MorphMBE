from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

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


def write_manifest(root: Path, path: Path, sample_id: str) -> None:
    fields = json.loads((root / "13_UNSEEN_INFERENCE/input_schema.json").read_text(encoding="utf-8"))["required"]
    row = {k: "" for k in fields}
    row.update(
        {
            "sample_id": sample_id,
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
            "growth_stage": "prospective_registry_test",
            "notes": "unit test synthetic manifest",
        }
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


class ProspectiveRegistryTest(unittest.TestCase):
    def test_registry_append_only_and_reveal_does_not_modify_prediction(self) -> None:
        root = latest_freeze()
        manifest = json.loads((root / "01_FREEZE_AND_PROVENANCE/FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_manifest = tmp_path / "manifest.csv"
            write_manifest(root, input_manifest, "UNSEEN_REGISTRY_001")
            pred_root = tmp_path / "predictions"
            subprocess.check_call(
                [
                    sys.executable,
                    str(root / "13_UNSEEN_INFERENCE/predict_unseen_batch.py"),
                    "--bundle-root",
                    str(root),
                    "--manifest",
                    str(input_manifest),
                    "--output-root",
                    str(pred_root),
                    "--freeze-id",
                    manifest["freeze_id"],
                ],
                cwd=REPO,
            )
            prediction_json = pred_root / "UNSEEN_REGISTRY_001" / "prediction.json"
            before = sha(prediction_json)
            registry = tmp_path / "registry.jsonl"
            freeze_cmd = [
                sys.executable,
                str(root / "14_PROSPECTIVE_REGISTRY/freeze_predictions.py"),
                "--prediction-root",
                str(pred_root),
                "--registry",
                str(registry),
                "--freeze-id",
                manifest["freeze_id"],
            ]
            subprocess.check_call(freeze_cmd, cwd=REPO)
            self.assertEqual(sha(prediction_json), before)
            first_lines = registry.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(first_lines), 1)
            subprocess.check_call(freeze_cmd, cwd=REPO)
            second_lines = registry.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(second_lines), 2)
            self.assertTrue((pred_root / "PREDICTIONS_FROZEN_BEFORE_AFM.md").exists())
            self.assertEqual(sha(prediction_json), before)

            bank = pd.read_csv(root / "12_FULL_COHORT_DEPLOYMENT/visual_model/afm_bank_manifest.csv")
            afm_file = root / "12_FULL_COHORT_DEPLOYMENT/visual_model/physical_maps" / f"{bank.iloc[0]['sample_id']}__{bank.iloc[0]['afm_file_id']}.npy"
            reveal_root = tmp_path / "reveal"
            subprocess.check_call(
                [
                    sys.executable,
                    str(root / "14_PROSPECTIVE_REGISTRY/reveal_and_evaluate_afm.py"),
                    "--registry",
                    str(registry),
                    "--afm-file",
                    str(afm_file),
                    "--sample-id",
                    "UNSEEN_REGISTRY_001",
                    "--output-root",
                    str(reveal_root),
                ],
                cwd=REPO,
            )
            self.assertEqual(sha(prediction_json), before)
            self.assertTrue((reveal_root / "revealed_results/UNSEEN_REGISTRY_001/prospective_evaluation.json").exists())


if __name__ == "__main__":
    unittest.main()
