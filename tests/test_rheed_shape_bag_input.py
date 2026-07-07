from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from rheed2morph.rheed.build_shape_bag_inputs import (
    aggregate_sample_features,
    build_consensus_maps,
    compute_frame_weights,
    write_shape_npz,
)
from rheed2morph.rheed.models.shape_bag_encoder import RHEEDShapeBagEncoder
from rheed2morph.rheed.rheed_shape_bag_dataset import RHEEDShapeBagDataset
from rheed2morph.rheed.shape_preprocessing import (
    DEFAULT_CHANNEL_NAMES,
    channels_to_tensor,
    photometric_perturbations,
    preprocess_frame_for_shape,
)
from rheed2morph.rheed.spot_streak_geometry import (
    FRAME_SHAPE_FEATURE_NAMES,
    extract_components_and_frame_features,
)


def synthetic_round_spots(size: int = 64) -> np.ndarray:
    y, x = np.indices((size, size))
    frame = np.full((size, size), 0.18, dtype=np.float32)
    for cy, cx, radius in [(22, 22, 5), (42, 38, 4), (26, 45, 3)]:
        frame += 0.75 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * radius**2))
    return np.clip(frame, 0.0, 1.0).astype(np.float32)


def synthetic_horizontal_bars(size: int = 64) -> np.ndarray:
    frame = np.full((size, size), 0.15, dtype=np.float32)
    frame[18:23, 12:52] = 0.95
    frame[38:43, 8:56] = 0.85
    frame += np.random.default_rng(1).normal(0.0, 0.01, frame.shape).astype(np.float32)
    return np.clip(frame, 0.0, 1.0).astype(np.float32)


def feature_vector_for(frame: np.ndarray) -> np.ndarray:
    processed = preprocess_frame_for_shape(frame, image_size=64)
    _, features = extract_components_and_frame_features(
        soft_mask=processed.channels["soft_spot_streak_mask"],
        enhanced_image=processed.channels["log_bgsub"],
        artifact_mask=processed.artifact_mask,
        min_area=4,
    )
    return np.asarray([features[name] for name in FRAME_SHAPE_FEATURE_NAMES], dtype=np.float32)


class RheedShapeBagInputTest(unittest.TestCase):
    def test_preprocessing_returns_finite_channels(self) -> None:
        result = preprocess_frame_for_shape(synthetic_round_spots(), image_size=64)
        self.assertEqual(set(DEFAULT_CHANNEL_NAMES), set(result.channels))
        for channel in result.channels.values():
            self.assertTrue(np.all(np.isfinite(channel)))

    def test_background_subtraction_reduces_brightness_difference(self) -> None:
        base = synthetic_round_spots()
        bright = np.clip(base * 1.7, 0.0, 1.0)
        dark = np.clip(base * 0.55, 0.0, 1.0)
        raw_diff = abs(float(bright.mean()) - float(dark.mean()))
        bright_processed = preprocess_frame_for_shape(bright, image_size=64)
        dark_processed = preprocess_frame_for_shape(dark, image_size=64)
        bgsub_diff = abs(float(bright_processed.channels["log_bgsub"].mean()) - float(dark_processed.channels["log_bgsub"].mean()))
        self.assertLess(bgsub_diff, raw_diff)

    def test_geometry_extractor_detects_round_spots(self) -> None:
        frame = synthetic_round_spots()
        mask = frame > 0.45
        components, features = extract_components_and_frame_features(
            soft_mask=mask.astype(np.float32),
            enhanced_image=frame,
            artifact_mask=np.zeros_like(frame),
            min_area=5,
        )
        self.assertGreaterEqual(len(components), 2)
        self.assertGreater(features["round_spot_count"], 0)

    def test_geometry_extractor_detects_horizontal_bars(self) -> None:
        frame = synthetic_horizontal_bars()
        bar_mask = frame > 0.6
        _, bar_features = extract_components_and_frame_features(
            soft_mask=bar_mask.astype(np.float32),
            enhanced_image=frame,
            artifact_mask=np.zeros_like(frame),
            min_area=5,
        )
        round_frame = synthetic_round_spots()
        round_mask = round_frame > 0.45
        _, round_features = extract_components_and_frame_features(
            soft_mask=round_mask.astype(np.float32),
            enhanced_image=round_frame,
            artifact_mask=np.zeros_like(round_frame),
            min_area=5,
        )
        self.assertGreater(bar_features["horizontal_bar_count"], 0)
        self.assertGreater(bar_features["bar_like_score"], round_features["bar_like_score"])

    def test_shadowed_low_snr_frame_receives_lower_weight(self) -> None:
        clean = preprocess_frame_for_shape(synthetic_round_spots(), image_size=64)
        shadowed_frame = synthetic_round_spots()
        shadowed_frame[:, :28] *= 0.02
        shadowed = preprocess_frame_for_shape(shadowed_frame, image_size=64)
        records = []
        for processed, quality in [(clean, 0.9), (shadowed, 0.4)]:
            _, features = extract_components_and_frame_features(
                soft_mask=processed.channels["soft_spot_streak_mask"],
                enhanced_image=processed.channels["log_bgsub"],
                artifact_mask=processed.artifact_mask,
                min_area=4,
            )
            records.append(
                SimpleNamespace(
                    candidate_row={"quality_score": str(quality), "low_confidence_candidate": "0"},
                    audit=processed.audit_features,
                    features=features,
                    frame_weight=0.0,
                    raw_weight=0.0,
                )
            )
        compute_frame_weights(records)
        self.assertGreater(records[0].frame_weight, records[1].frame_weight)

    def test_robust_aggregation_is_less_sensitive_to_one_bad_frame(self) -> None:
        frames = []
        for index in range(16):
            value = 1.0 if index < 15 else 100.0
            frames.append(SimpleNamespace(frame_weight=1.0 / 16.0, features={"bar_like_score": value}))
        simple_mean = np.mean([frame.features["bar_like_score"] for frame in frames])
        vector, names, summary = aggregate_sample_features(frames)
        _ = vector, names
        self.assertLess(summary["weighted_median_bar_like_score"], simple_mean)
        self.assertLess(summary["trimmed_mean_bar_like_score"], simple_mean)

    def test_shape_bag_npz_contains_required_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            processed = preprocess_frame_for_shape(synthetic_round_spots(), image_size=64)
            _, features = extract_components_and_frame_features(
                soft_mask=processed.channels["soft_spot_streak_mask"],
                enhanced_image=processed.channels["log_bgsub"],
                artifact_mask=processed.artifact_mask,
                min_area=4,
            )
            frame = SimpleNamespace(
                channels=processed.channels,
                frame_weight=1.0,
                frame_idx=7,
                timestamp_sec=0.7,
                features=features,
            )
            sample_vector = np.asarray([1.0, 2.0], dtype=np.float32)
            path = Path(tmp_dir) / "shape_bag.npz"
            write_shape_npz(
                path,
                frames=[frame],
                image_size=64,
                max_frames=16,
                sample_feature_vector=sample_vector,
                sample_feature_names=["a", "b"],
            )
            with np.load(path) as data:
                for key in ["frames", "frame_mask", "frame_weights", "consensus_maps", "sample_feature_vector"]:
                    self.assertIn(key, data.files)
                self.assertEqual(data["frames"].shape, (16, len(DEFAULT_CHANNEL_NAMES), 64, 64))

    def test_dataset_returns_correct_tensor_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            frames = np.zeros((16, 6, 32, 32), dtype=np.float32)
            frame_mask = np.ones(16, dtype=np.float32)
            frame_weights = np.full(16, 1 / 16, dtype=np.float32)
            consensus = build_consensus_maps(frames, frame_mask, frame_weights)
            npz = root / "shape_bag.npz"
            np.savez(
                npz,
                frames=frames,
                frame_mask=frame_mask,
                frame_weights=frame_weights,
                consensus_maps=consensus,
                sample_feature_vector=np.ones(5, dtype=np.float32),
                sample_feature_names=np.asarray(["a", "b", "c", "d", "e"], dtype="U"),
            )
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["sample_id", "shape_bag_npz", "source_type"])
                writer.writeheader()
                writer.writerow({"sample_id": "N0001", "shape_bag_npz": str(npz), "source_type": "auto_candidate_fallback"})
            item = RHEEDShapeBagDataset(manifest)[0]
            self.assertEqual(item["frames"].shape, (16, 6, 32, 32))
            self.assertEqual(item["consensus_maps"].shape, (6, 32, 32))
            self.assertEqual(item["shape_features"].shape, (5,))

    def test_shape_bag_encoder_forward_variable_k(self) -> None:
        model = RHEEDShapeBagEncoder(in_channels=6, consensus_channels=6, shape_feature_dim=10, embedding_dim=32)
        frames = torch.randn(2, 16, 6, 64, 64)
        mask = torch.ones(2, 16)
        mask[1, 8:] = 0
        weights = mask / mask.sum(dim=1, keepdim=True)
        consensus = torch.randn(2, 6, 64, 64)
        features = torch.randn(2, 10)
        output = model(frames, mask, weights, consensus, features)
        self.assertEqual(output["sample_embedding"].shape, (2, 32))
        self.assertEqual(output["attention_weights"].shape, (2, 16))
        self.assertTrue(torch.all(output["attention_weights"][1, 8:] < 1e-6))

    def test_exposure_audit_shape_features_more_stable_than_raw_brightness(self) -> None:
        frame = synthetic_round_spots()
        raw_means = []
        vectors = []
        for perturbed in photometric_perturbations(frame).values():
            raw_means.append(float(perturbed.mean()))
            vectors.append(feature_vector_for(perturbed))
        raw_cv = np.std(raw_means) / max(abs(np.mean(raw_means)), 1e-6)
        stack = np.stack(vectors)
        shape_cv = np.median(np.std(stack, axis=0) / np.maximum(np.mean(np.abs(stack), axis=0), 1e-6))
        self.assertLess(shape_cv, raw_cv)

    def test_new_shape_bag_sources_do_not_import_retrieval_models(self) -> None:
        root = Path(__file__).resolve().parents[1]
        files = [
            root / "src" / "rheed2morph" / "rheed" / "shape_preprocessing.py",
            root / "src" / "rheed2morph" / "rheed" / "spot_streak_geometry.py",
            root / "src" / "rheed2morph" / "rheed" / "build_shape_bag_inputs.py",
            root / "src" / "rheed2morph" / "rheed" / "rheed_shape_bag_dataset.py",
            root / "src" / "rheed2morph" / "rheed" / "models" / "shape_bag_encoder.py",
        ]
        source_text = "\n".join(path.read_text(encoding="utf-8") for path in files)
        self.assertNotIn("KNeighbors", source_text)
        self.assertNotIn("sklearn.neighbors", source_text)


if __name__ == "__main__":
    unittest.main()
