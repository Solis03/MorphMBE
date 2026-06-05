from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from rheed2morph.manifests.one_to_one import (
    CandidateRecord,
    build_all_size_representative_manifest,
    build_target_manifest,
    infer_afm_scan_size_um,
    parse_afm_scan_size_um,
)
from rheed2morph.rheed.mvp import (
    SampleEmbeddingRecord,
    build_joined_dataset,
)


class OneToOneManifestTest(unittest.TestCase):
    def test_parse_afm_scan_size_um_patterns(self) -> None:
        cases = {
            "N6101_1um_001.npy": 1.0,
            "N6101_1_um_001.npy": 1.0,
            "N6101_1.0um_001.npy": 1.0,
            "N6101_0p5um_001.npy": 0.5,
            "N6101_500nm_001.npy": 0.5,
            "N6101_500_nm_001.npy": 0.5,
            "N6101_1umx1um_001.npy": 1.0,
            "N6101_500x500nm_001.npy": 0.5,
            "N6101_1.0x1.0_001.npy": 1.0,
            "N6101_5um_001.npy": 5.0,
        }
        for name, expected in cases.items():
            self.assertAlmostEqual(parse_afm_scan_size_um(name), expected, places=6)

    def test_build_target_manifest_selects_one_per_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            a = root / "a.npy"
            b = root / "b.npy"
            c = root / "c.npy"
            np.save(a, np.zeros((64, 64), dtype=np.float32))
            np.save(b, np.zeros((128, 128), dtype=np.float32))
            np.save(c, np.zeros((32, 32), dtype=np.float32))
            grouped = {
                "g1": [
                    CandidateRecord("s1", "g1", "unknown", root / "r1.mov", a, None, 1.0),
                    CandidateRecord("s1", "g1", "unknown", root / "r1.mov", b, None, 1.0),
                ],
                "g2": [
                    CandidateRecord("s2", "g2", "unknown", root / "r2.mov", c, None, 0.5),
                ],
            }
            (root / "r1.mov").write_bytes(b"x")
            (root / "r2.mov").write_bytes(b"x")
            rows, summary = build_target_manifest(grouped, target_um=1.0, tolerance=0.05)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].group_id, "g1")
            self.assertEqual(rows[0].afm_path.name, "b.npy")
            self.assertEqual(summary["dropped_group_count"], 1)

    def test_infer_afm_scan_size_prefers_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            afm = root / "N66_Ctr_003_plane_corrected.npy"
            metadata = root / "N66_Ctr_003_plane_corrected_metadata.json"
            np.save(afm, np.zeros((8, 8), dtype=np.float32))
            metadata.write_text('{"scan_size_um":[1.0,1.0]}', encoding="utf-8")
            size_um, source = infer_afm_scan_size_um(afm)
            self.assertEqual(size_um, 1.0)
            self.assertIn("metadata:", source)

    def test_build_all_size_representative_prefers_one_um(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            afm_1 = root / "one.npy"
            afm_05 = root / "half.npy"
            np.save(afm_1, np.zeros((64, 64), dtype=np.float32))
            np.save(afm_05, np.zeros((64, 64), dtype=np.float32))
            rheed = root / "video.mov"
            rheed.write_bytes(b"x")
            grouped = {
                "g1": [
                    CandidateRecord("s1", "g1", "unknown", rheed, afm_05, None, 0.5),
                    CandidateRecord("s1", "g1", "unknown", rheed, afm_1, None, 1.0),
                ]
            }
            rows = build_all_size_representative_manifest(grouped, 0.05, [1.0, 0.5, 5.0], 0.5)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].afm_path.name, "one.npy")

    def test_build_joined_dataset_filters_to_manifest_selection(self) -> None:
        sample_embeddings = {
            "6022": SampleEmbeddingRecord(
                sample_id="6022",
                video_path=Path("/tmp/video.MOV"),
                selection_reason="contains_main",
                duration_seconds=1.0,
                decoded_frame_count=8,
                sampled_frame_count=8,
                embedding=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            )
        }
        descriptor_rows = [
            {"row_id": "1", "sample_id": "6022", "afm_path": "/tmp/raw_a.npy", "roughness": "0.2"},
            {"row_id": "2", "sample_id": "6022", "afm_path": "/tmp/raw_b.npy", "roughness": "0.3"},
        ]
        aux_rows = [
            {"row_id": "1", "network_input_path": "/tmp/sel_a.npy", "afm_path": "/tmp/raw_a.npy"},
            {"row_id": "2", "network_input_path": "/tmp/sel_b.npy", "afm_path": "/tmp/raw_b.npy"},
        ]
        manifest_rows = [
            {
                "sample_id": "6022",
                "group_id": "g6022",
                "rheed_path": "/tmp/video.MOV",
                "afm_path": "/tmp/sel_b.npy",
            }
        ]
        dataset, skipped = build_joined_dataset(
            descriptor_rows=descriptor_rows,
            aux_rows=aux_rows,
            sample_embeddings=sample_embeddings,
            manifest_rows=manifest_rows,
        )
        self.assertEqual(dataset.row_ids, ["2"])
        self.assertEqual(dataset.group_ids, ["g6022"])
        self.assertEqual(len(skipped), 1)


if __name__ == "__main__":
    unittest.main()
