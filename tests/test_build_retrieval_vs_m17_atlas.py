from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_retrieval_vs_m17_atlas.py"
SPEC = importlib.util.spec_from_file_location("build_retrieval_vs_m17_atlas", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_display_id_preserves_extra_five_prefix() -> None:
    assert MODULE.display_id("N6342") == "N6342"
    assert MODULE.display_id("6022.0") == "6022"


def test_project_unit_sq_reproduces_requested_physical_sq() -> None:
    source = np.arange(36, dtype=np.float32).reshape(6, 6)
    requested_sq = 4.25
    rendered = MODULE.project_unit_sq(source) * requested_sq
    assert np.isclose(MODULE.measured_sq(rendered), requested_sq, atol=2e-5)
