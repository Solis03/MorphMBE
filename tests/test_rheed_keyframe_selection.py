from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from rheed2morph.rheed import keyframe_selection


def write_synthetic_mp4(path: Path, frame_count: int = 4, width: int = 10, height: int = 8) -> list[np.ndarray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []
    for index in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (index * 31 + np.arange(width, dtype=np.uint8)[None, :]) % 255
        frame[:, :, 1] = (index * 47 + np.arange(height, dtype=np.uint8)[:, None]) % 255
        frame[:, :, 2] = index * 23
        frames.append(frame)
    writer = imageio.get_writer(str(path), fps=5, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return frames


def decoded_video_frames(path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(str(path), "ffmpeg")
    try:
        return [keyframe_selection.to_rgb_array(frame) for frame in reader]
    finally:
        reader.close()


class RHEEDKeyframeSelectionTest(unittest.TestCase):
    def test_short_mp4_exports_all_pngs_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "data" / "pair" / "6022" / "RHEED" / "tiny.MP4"
            write_synthetic_mp4(source, frame_count=5, width=12, height=10)
            stat_before = source.stat()

            summary = keyframe_selection.run_extraction(
                input_root=root / "data" / "pair",
                output_root=root / "data" / "rheed_keyframe_selection",
                repo_root=root,
            )

            self.assertEqual(summary.success_count, 1)
            frames_dir = root / "data" / "rheed_keyframe_selection" / "6022" / "videos" / "tiny" / "frames"
            self.assertEqual([path.name for path in sorted(frames_dir.glob("*.png"), key=lambda p: int(p.stem))], [f"{i}.png" for i in range(5)])
            decoded = decoded_video_frames(source)
            for index, expected in enumerate(decoded):
                with Image.open(frames_dir / f"{index}.png") as image:
                    self.assertEqual(image.size, (12, 10))
                    self.assertEqual(image.mode, "RGB")
                    self.assertNotEqual(image.mode, "P")
                    self.assertTrue(np.array_equal(np.asarray(image), expected))

            metadata = json.loads((root / "data" / "rheed_keyframe_selection" / "6022" / "metadata.json").read_text())
            video_metadata = metadata["videos"]["tiny"]
            self.assertIsNone(video_metadata["selection"]["keyframe_index"])
            self.assertIsNone(video_metadata["selection"]["clip_frame_count"])
            self.assertEqual(video_metadata["video_info"]["extracted_frame_count"], 5)
            self.assertEqual(video_metadata["video_info"]["width"], 12)
            self.assertEqual(video_metadata["video_info"]["height"], 10)
            self.assertEqual(video_metadata["extraction"]["color_mode"], "RGB")
            self.assertEqual(video_metadata["extraction"]["preprocessing_applied"], [])
            stat_after = source.stat()
            self.assertEqual(stat_before.st_size, stat_after.st_size)
            self.assertEqual(stat_before.st_mtime_ns, stat_after.st_mtime_ns)

    def test_selection_is_preserved_on_rerun_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "data" / "pair" / "6022" / "RHEED" / "tiny.MP4"
            write_synthetic_mp4(source, frame_count=3)
            output_root = root / "data" / "rheed_keyframe_selection"
            keyframe_selection.run_extraction(root / "data" / "pair", output_root, root)

            metadata_path = output_root / "6022" / "metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["videos"]["tiny"]["selection"] = {"keyframe_index": 1, "clip_frame_count": 3}
            metadata["notes"] = "manual note"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            skipped = keyframe_selection.run_extraction(root / "data" / "pair", output_root, root)
            self.assertEqual(skipped.skipped_count, 1)
            preserved = json.loads(metadata_path.read_text())
            self.assertEqual(preserved["videos"]["tiny"]["selection"]["keyframe_index"], 1)
            self.assertEqual(preserved["videos"]["tiny"]["selection"]["clip_frame_count"], 3)
            self.assertEqual(preserved["notes"], "manual note")

            overwritten = keyframe_selection.run_extraction(root / "data" / "pair", output_root, root, overwrite=True)
            self.assertEqual(overwritten.success_count, 1)
            preserved = json.loads(metadata_path.read_text())
            self.assertEqual(preserved["videos"]["tiny"]["selection"]["keyframe_index"], 1)
            self.assertEqual(preserved["videos"]["tiny"]["selection"]["clip_frame_count"], 3)

    def test_duplicate_mp4_stems_do_not_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_synthetic_mp4(root / "data" / "pair" / "6022" / "RHEED" / "a" / "same.MP4", frame_count=1)
            write_synthetic_mp4(root / "data" / "pair" / "6022" / "RHEED" / "b" / "same.mp4", frame_count=1)
            output_root = root / "data" / "rheed_keyframe_selection"

            summary = keyframe_selection.run_extraction(root / "data" / "pair", output_root, root)

            self.assertEqual(summary.success_count, 2)
            video_dirs = sorted(path.name for path in (output_root / "6022" / "videos").iterdir() if path.is_dir())
            self.assertEqual(len(video_dirs), 2)
            self.assertEqual(len(set(video_dirs)), 2)
            for video_id in video_dirs:
                self.assertTrue((output_root / "6022" / "videos" / video_id / "frames" / "0.png").is_file())

    def test_mov_files_are_discovered_and_exported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "data" / "pair" / "6022" / "RHEED" / "clip.MOV"
            write_synthetic_mp4(source, frame_count=2, width=12, height=10)
            output_root = root / "data" / "rheed_keyframe_selection"

            summary = keyframe_selection.run_extraction(root / "data" / "pair", output_root, root)

            self.assertEqual(summary.success_count, 1)
            self.assertTrue((output_root / "6022" / "videos" / "clip" / "frames" / "0.png").is_file())
            self.assertTrue((output_root / "6022" / "videos" / "clip" / "frames" / "1.png").is_file())

    def test_corrupt_mp4_enters_failed_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "data" / "pair" / "6022" / "RHEED" / "bad.MP4"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"not a video")
            output_root = root / "data" / "rheed_keyframe_selection"

            summary = keyframe_selection.run_extraction(root / "data" / "pair", output_root, root)

            self.assertEqual(summary.failed_count, 1)
            with (output_root / "extraction_report.csv").open() as handle:
                report_rows = list(csv.DictReader(handle))
            self.assertEqual(report_rows[0]["status"], "failed")
            metadata = json.loads((output_root / "6022" / "metadata.json").read_text())
            self.assertFalse(metadata["videos"]["bad"]["extraction"]["completed"])

    def test_dry_run_does_not_create_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_synthetic_mp4(root / "data" / "pair" / "6022" / "RHEED" / "tiny.MP4", frame_count=2)
            output_root = root / "data" / "rheed_keyframe_selection"

            summary = keyframe_selection.run_extraction(root / "data" / "pair", output_root, root, dry_run=True)

            self.assertEqual(summary.video_count, 1)
            self.assertFalse(output_root.exists())

    def test_temp_failure_is_not_marked_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_synthetic_mp4(root / "data" / "pair" / "6022" / "RHEED" / "tiny.MP4", frame_count=2)
            output_root = root / "data" / "rheed_keyframe_selection"

            with mock.patch.object(keyframe_selection, "compare_sampled_pixels", side_effect=ValueError("forced")):
                summary = keyframe_selection.run_extraction(root / "data" / "pair", output_root, root)

            self.assertEqual(summary.failed_count, 1)
            self.assertFalse((output_root / "6022" / "videos" / "tiny" / "frames").exists())
            metadata = json.loads((output_root / "6022" / "metadata.json").read_text())
            self.assertFalse(metadata["videos"]["tiny"]["extraction"]["completed"])

    def test_png_compression_levels_preserve_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_synthetic_mp4(root / "data" / "pair" / "6022" / "RHEED" / "tiny.MP4", frame_count=3)
            out_0 = root / "out0"
            out_9 = root / "out9"

            keyframe_selection.run_extraction(root / "data" / "pair", out_0, root, png_compression=0)
            keyframe_selection.run_extraction(root / "data" / "pair", out_9, root, png_compression=9)

            for index in range(3):
                with Image.open(out_0 / "6022" / "videos" / "tiny" / "frames" / f"{index}.png") as img_0:
                    arr_0 = np.asarray(img_0)
                with Image.open(out_9 / "6022" / "videos" / "tiny" / "frames" / f"{index}.png") as img_9:
                    arr_9 = np.asarray(img_9)
                self.assertTrue(np.array_equal(arr_0, arr_9))


if __name__ == "__main__":
    unittest.main()
