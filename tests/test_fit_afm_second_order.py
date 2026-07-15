from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fit_afm_second_order.py"
SPEC = importlib.util.spec_from_file_location("fit_afm_second_order", SCRIPT_PATH)
fit_afm_second_order = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = fit_afm_second_order
SPEC.loader.exec_module(fit_afm_second_order)


def coords(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    return fit_afm_second_order.normalized_coordinates(shape)


def write_afm(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    metadata = {
        "sample_id": path.parts[-3] if len(path.parts) >= 3 else "sample",
        "afm_file_id": path.stem.removesuffix("_height"),
        "primary_channel": "ZSensor",
        "height_unit_exported": "nm",
    }
    path.with_name(f"{path.name.removesuffix('_height.npy')}_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )


def process_one(source: Path, input_root: Path, output_root: Path, **kwargs: object) -> dict[str, object]:
    options = fit_afm_second_order.ProcessOptions(**kwargs)
    return fit_afm_second_order.process_file(source, input_root, output_root, options)


class FitAfmSecondOrderTests(unittest.TestCase):
    def test_y2_synthetic_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shape = (48, 64)
            x, y = coords(shape)
            texture = 0.02 * np.sin(18 * x) * np.cos(15 * y)
            background = 1.5 + 0.2 * x - 0.1 * y + 0.7 * y * y
            source = root / "input" / "s1" / "scan" / "scan_height.npy"
            write_afm(source, (texture + background).astype(np.float64))

            row = process_one(source, root / "input", root / "out", model="y2", robust=False)
            output = np.load(root / "out" / "s1" / "scan" / "scan_height.npy")
            fitted = np.load(root / "out" / "_backgrounds" / "s1" / "scan" / "scan_height.npy")

            self.assertEqual(row["status"], "success")
            self.assertEqual(output.shape, shape)
            self.assertLess(float(np.max(np.abs(fitted - background))), 0.005)
            self.assertLess(float(np.max(np.abs(output - texture))), 0.005)
            before = abs(fit_afm_second_order.vertical_edge_center_bow(texture + background))
            after = abs(fit_afm_second_order.vertical_edge_center_bow(output))
            self.assertLess(after, 0.1 * before)

    def test_full2d_synthetic_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shape = (45, 53)
            x, y = coords(shape)
            texture = 0.01 * np.sin(11 * x + 3 * y)
            background = 0.4 + 0.3 * x - 0.2 * y + 0.5 * x * x - 0.35 * x * y + 0.8 * y * y
            source = root / "input" / "s1" / "scan" / "scan_height.npy"
            write_afm(source, texture + background)

            row = process_one(source, root / "input", root / "out", model="full2d", robust=False)
            fitted = np.load(root / "out" / "_backgrounds" / "s1" / "scan" / "scan_height.npy")
            output = np.load(root / "out" / "s1" / "scan" / "scan_height.npy")

            self.assertEqual(row["status"], "success")
            self.assertLess(float(np.max(np.abs(fitted - background))), 0.005)
            self.assertLess(float(np.max(np.abs(output - texture))), 0.005)

    def test_nan_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x, y = coords((24, 28))
            array = 2.0 + 0.1 * x + 0.3 * y * y
            array[3, 4] = np.nan
            source = root / "input" / "s1" / "scan" / "scan_height.npy"
            write_afm(source, array)

            row = process_one(source, root / "input", root / "out", model="y2")
            output = np.load(root / "out" / "s1" / "scan" / "scan_height.npy")

            self.assertEqual(row["status"], "success")
            self.assertTrue(np.isnan(output[3, 4]))
            self.assertTrue(np.isfinite(np.delete(output.ravel(), 3 * output.shape[1] + 4)).all())

    def test_singleton_dimension_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x, y = coords((20, 22))
            array = (1.0 + 0.2 * x + 0.4 * y * y)[np.newaxis, :, :]
            source = root / "input" / "s1" / "scan" / "scan_height.npy"
            write_afm(source, array)

            row = process_one(source, root / "input", root / "out", model="y2")
            output = np.load(root / "out" / "s1" / "scan" / "scan_height.npy")

            self.assertEqual(row["status"], "success")
            self.assertEqual(output.shape, array.shape)

    def test_unsupported_multichannel_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input" / "s1" / "scan" / "scan_height.npy"
            write_afm(source, np.zeros((3, 10, 12), dtype=np.float64))

            row = process_one(source, root / "input", root / "out", model="y2")

            self.assertEqual(row["status"], "unsupported_shape")
            self.assertFalse((root / "out" / "s1" / "scan" / "scan_height.npy").exists())

    def test_source_immutability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x, y = coords((18, 19))
            source = root / "input" / "s1" / "scan" / "scan_height.npy"
            write_afm(source, 1.0 + x + y * y)
            before = fit_afm_second_order.sha256_file(source)

            row = process_one(source, root / "input", root / "out", model="y2")

            self.assertEqual(row["status"], "success")
            self.assertEqual(fit_afm_second_order.sha256_file(source), before)
            self.assertEqual(row["source_sha256_before"], before)
            self.assertEqual(row["source_sha256_after"], before)

    def test_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x, y = coords((18, 19))
            source = root / "input" / "s1" / "scan" / "scan_height.npy"
            write_afm(source, 1.0 + x + y * y)
            existing = root / "out" / "s1" / "scan" / "scan_height.npy"
            existing.parent.mkdir(parents=True)
            np.save(existing, np.full((2, 2), 7.0))
            before = fit_afm_second_order.sha256_file(existing)

            row = process_one(source, root / "input", root / "out", model="y2")

            self.assertEqual(row["status"], "exists_skipped")
            self.assertEqual(fit_afm_second_order.sha256_file(existing), before)

    def test_relative_path_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x, y = coords((16, 17))
            first = root / "input" / "a" / "scan" / "same_height.npy"
            second = root / "input" / "b" / "scan" / "same_height.npy"
            write_afm(first, 1.0 + x + y * y)
            write_afm(second, 2.0 - x + 0.5 * y * y)

            for source in (first, second):
                row = process_one(source, root / "input", root / "out", model="y2")
                self.assertEqual(row["status"], "success")

            self.assertTrue((root / "out" / "a" / "scan" / "same_height.npy").exists())
            self.assertTrue((root / "out" / "b" / "scan" / "same_height.npy").exists())

    def test_dtype_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x, y = coords((16, 17))
            cases = [
                ("float32", (1.0 + x + y * y).astype(np.float32), np.float32),
                ("float64", (1.0 + x + y * y).astype(np.float64), np.float64),
                ("int", np.ones((16, 17), dtype=np.int16), np.float32),
            ]
            for name, array, expected in cases:
                source = root / name / "input" / "s1" / "scan" / "scan_height.npy"
                write_afm(source, array)
                row = process_one(source, root / name / "input", root / name / "out", model="y2")
                output = np.load(root / name / "out" / "s1" / "scan" / "scan_height.npy")
                self.assertEqual(row["status"], "success")
                self.assertEqual(output.dtype, expected)

    def test_flat_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input" / "s1" / "scan" / "scan_height.npy"
            write_afm(source, np.full((20, 20), 3.0, dtype=np.float64))

            row = process_one(source, root / "input", root / "out", model="y2")
            output = np.load(root / "out" / "s1" / "scan" / "scan_height.npy")
            metadata = json.loads((root / "out" / "_metadata" / "s1" / "scan" / "scan_height.json").read_text())

            self.assertEqual(row["status"], "success")
            self.assertLess(float(np.max(np.abs(output))), 1e-10)
            self.assertIn("coefficients", metadata)


if __name__ == "__main__":
    unittest.main()
