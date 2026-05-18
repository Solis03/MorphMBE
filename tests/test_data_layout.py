from __future__ import annotations

from pathlib import Path
import unittest


class DataLayoutTest(unittest.TestCase):
    def test_manifest_directory_exists(self) -> None:
        manifest_dir = Path("data/manifests")

        self.assertTrue(manifest_dir.is_dir())
        self.assertTrue((manifest_dir / "README.md").is_file())
