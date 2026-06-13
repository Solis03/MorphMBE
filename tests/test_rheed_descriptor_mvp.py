from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from rheed2morph.rheed.mvp import (
    JoinedDataset,
    SampleEmbeddingRecord,
    VideoCandidate,
    aggregate_frame_embeddings,
    aggregate_temporal_frame_embeddings,
    build_joined_dataset,
    choose_canonical_video,
    load_processed_model_input,
    processed_dir_matches_sample_id,
    resolve_processed_model_input_path,
    run_modeling_experiment,
    split_group_holdout,
)


class RheedDescriptorMvpTest(unittest.TestCase):
    def test_choose_canonical_video_prefers_main(self) -> None:
        candidates = [
            VideoCandidate("6022", Path("/tmp/long.MOV"), 12.0, 120, False),
            VideoCandidate("6022", Path("/tmp/main_short.MOV"), 4.0, 40, True),
        ]
        selected, reason = choose_canonical_video(candidates)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.path.name, "main_short.MOV")
        self.assertEqual(reason, "contains_main")

    def test_choose_canonical_video_falls_back_to_longest(self) -> None:
        candidates = [
            VideoCandidate("6022", Path("/tmp/a.MOV"), 4.0, 40, False),
            VideoCandidate("6022", Path("/tmp/b.MOV"), 8.0, 80, False),
        ]
        selected, reason = choose_canonical_video(candidates)
        self.assertIsNotNone(selected)
        self.assertEqual(selected.path.name, "b.MOV")
        self.assertEqual(reason, "longest_decodable")

    def test_build_joined_dataset_matches_on_sample_id(self) -> None:
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
            {"row_id": "1", "sample_id": "6022", "afm_path": "a.npy", "roughness": "0.2", "texture": "0.8"},
            {"row_id": "2", "sample_id": "9999", "afm_path": "b.npy", "roughness": "0.3", "texture": "0.6"},
        ]
        aux_rows = [
            {"row_id": "1", "network_input_path": "proto.npy", "afm_path": "a.npy"},
            {"row_id": "2", "network_input_path": "proto2.npy", "afm_path": "b.npy"},
        ]
        dataset, skipped = build_joined_dataset(descriptor_rows, aux_rows, sample_embeddings)
        self.assertEqual(dataset.row_ids, ["1"])
        self.assertEqual(dataset.sample_ids, ["6022"])
        self.assertEqual(dataset.group_ids, ["6022"])
        self.assertEqual(dataset.target_names, ["roughness", "texture"])
        self.assertEqual(dataset.x.shape, (1, 4))
        self.assertEqual(dataset.y.shape, (1, 2))
        self.assertEqual(skipped[0]["sample_id"], "9999")

    def test_split_group_holdout_has_no_leakage(self) -> None:
        sample_ids = ["a", "a", "b", "b", "c", "c", "d", "d"]
        train_idx, test_idx = split_group_holdout(sample_ids, test_fraction=0.25, random_state=42)
        train_groups = {sample_ids[index] for index in train_idx}
        test_groups = {sample_ids[index] for index in test_idx}
        self.assertFalse(train_groups & test_groups)

    def test_aggregate_frame_embeddings_is_deterministic(self) -> None:
        frame_embeddings = np.asarray([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
        aggregated = aggregate_frame_embeddings(frame_embeddings)
        expected = np.asarray([3.0, 5.0, 2.0, 2.0], dtype=np.float32)
        np.testing.assert_allclose(aggregated, expected)

    def test_aggregate_temporal_frame_embeddings_appends_delta_stats(self) -> None:
        frame_embeddings = np.asarray([[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]], dtype=np.float32)
        aggregated = aggregate_temporal_frame_embeddings(frame_embeddings)
        expected = np.asarray([5.0, 7.0, 3.2659864, 3.2659864, 4.0, 4.0, 0.0, 0.0], dtype=np.float32)
        self.assertEqual(aggregated.shape, (8,))
        np.testing.assert_allclose(aggregated, expected, rtol=1e-6, atol=1e-6)

    def test_processed_dir_matches_sample_id_uses_numeric_boundaries(self) -> None:
        self.assertTrue(processed_dir_matches_sample_id("N6022 - Copy", "6022"))
        self.assertTrue(processed_dir_matches_sample_id("N6063", "6063"))
        self.assertFalse(processed_dir_matches_sample_id("N16022", "6022"))

    def test_resolve_processed_model_input_path_uses_unique_sample_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            target = root / "N6022 - Copy" / "tensors"
            target.mkdir(parents=True)
            np.savez(target / "model_input.npz", clean_frames=np.zeros((2, 3, 4), dtype=np.float32), valid_mask=np.ones((3, 4), dtype=bool))
            resolved = resolve_processed_model_input_path("6022", root, "manifest_sample_id_to_dataset_dir")
            self.assertEqual(resolved, target / "model_input.npz")

    def test_load_processed_model_input_applies_mask_and_validates_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "model_input.npz"
            clean_frames = np.asarray(
                [
                    [[0.1, 0.2], [0.3, 0.4]],
                    [[0.5, 0.6], [0.7, 0.8]],
                ],
                dtype=np.float32,
            )
            valid_mask = np.asarray([[True, False], [False, True]])
            timestamps_sec = np.asarray([0.0, 1.0], dtype=np.float32)
            np.savez(path, clean_frames=clean_frames, valid_mask=valid_mask, timestamps_sec=timestamps_sec)
            frames, mask, metadata = load_processed_model_input(path)
            expected = np.asarray(
                [
                    [[0.1, 0.0], [0.0, 0.4]],
                    [[0.5, 0.0], [0.0, 0.8]],
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(frames, expected)
            np.testing.assert_array_equal(mask, valid_mask)
            np.testing.assert_allclose(metadata["timestamps_sec"], timestamps_sec)

    def test_load_processed_model_input_rejects_bad_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad_model_input.npz"
            np.savez(
                path,
                clean_frames=np.zeros((2, 3, 4), dtype=np.float32),
                valid_mask=np.ones((5, 4), dtype=bool),
            )
            with self.assertRaises(ValueError):
                load_processed_model_input(path)

    def test_run_modeling_experiment_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            proto_dir = root / "prototypes"
            proto_dir.mkdir()
            row_ids: list[str] = []
            sample_ids: list[str] = []
            afm_paths: list[str] = []
            network_input_paths: list[str] = []
            x_rows: list[np.ndarray] = []
            y_rows: list[np.ndarray] = []

            sample_vectors = {
                "s1": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                "s2": np.asarray([0.2, 0.1, 0.4, 0.3], dtype=np.float32),
                "s3": np.asarray([0.9, 1.0, 1.1, 1.2], dtype=np.float32),
                "s4": np.asarray([1.0, 0.9, 1.2, 1.1], dtype=np.float32),
            }
            for sample_id, vector in sample_vectors.items():
                for replicate in range(2):
                    row_id = f"{sample_id}_{replicate}"
                    prototype = np.full((8, 8), fill_value=len(row_ids), dtype=np.float32)
                    prototype_path = proto_dir / f"{row_id}.npy"
                    np.save(prototype_path, prototype)
                    row_ids.append(row_id)
                    sample_ids.append(sample_id)
                    afm_paths.append(f"{row_id}.npy")
                    network_input_paths.append(str(prototype_path))
                    x_rows.append(vector)
                    y_rows.append(np.asarray([vector[0] + replicate, vector[1] - replicate], dtype=np.float32))

            dataset = JoinedDataset(
                row_ids=row_ids,
                sample_ids=sample_ids,
                group_ids=sample_ids,
                afm_paths=afm_paths,
                network_input_paths=network_input_paths,
                feature_names=[f"embedding_{index:04d}" for index in range(4)],
                target_names=["descriptor_a", "descriptor_b"],
                x=np.stack(x_rows, axis=0),
                y=np.stack(y_rows, axis=0),
            )
            metrics = run_modeling_experiment(
                dataset=dataset,
                data_dir=root / "data_out",
                report_dir=root / "report_out",
                summary={"encoder_backend_resolved": "synthetic", "sample_count_embedded": 4, "sample_count_skipped": 0},
                random_state=42,
                test_fraction=0.25,
            )
            self.assertIn("best_model_name", metrics)
            self.assertTrue((root / "data_out" / "best_model.joblib").is_file())
            self.assertTrue((root / "data_out" / "metrics_summary.json").is_file())
            self.assertTrue((root / "report_out" / "predicted_vs_true_scatter.png").is_file())
            self.assertTrue((root / "report_out" / "nearest_neighbor_qualitative_grid.png").is_file())


if __name__ == "__main__":
    unittest.main()
