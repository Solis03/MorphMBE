from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "afm_descriptor_reconstruction"
sys.path.insert(0, str(SCRIPT_DIR))

from train_descriptor_mlp_decoder import predict, train_model  # noqa: E402


class MLPTrainingSmokeTest(unittest.TestCase):
    def test_train_model_runs_on_cpu(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(8, 5)).astype(np.float32)
        y = rng.normal(size=(8, 16)).astype(np.float32)

        model, history = train_model(
            x,
            y,
            image_size=4,
            epochs=2,
            patience=2,
            batch_size=4,
            learning_rate=1e-3,
            seed=123,
            device_name="cpu",
            use_amp=False,
            compile_model=False,
        )
        pred = predict(model, x, batch_size=4)

        self.assertEqual(pred.shape, y.shape)
        self.assertEqual(len(history), 2)


if __name__ == "__main__":
    unittest.main()
