from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "afm_descriptor_reconstruction"
sys.path.insert(0, str(SCRIPT_DIR))

from train_descriptor_pca_decoder import fit_predict_insample  # noqa: E402


class PCATrainingSmokeTest(unittest.TestCase):
    def test_fit_predict_insample_runs_on_cpu(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(10, 4)).astype(np.float32)
        y = rng.normal(size=(10, 16)).astype(np.float32)

        import torch

        pred, pca_model, ridge_model, metrics = fit_predict_insample(
            x,
            y,
            n_components=3,
            device=torch.device("cpu"),
        )

        self.assertEqual(pred.shape, y.shape)
        self.assertEqual(pca_model["components"].shape, (3, 16))
        self.assertEqual(ridge_model["coef"].shape, (4, 3))
        self.assertIn("mse", metrics)


if __name__ == "__main__":
    unittest.main()
