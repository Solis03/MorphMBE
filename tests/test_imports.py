from __future__ import annotations

import unittest


class PackageImportTest(unittest.TestCase):
    def test_package_import(self) -> None:
        import rheed2morph

        self.assertTrue(rheed2morph.__version__)
