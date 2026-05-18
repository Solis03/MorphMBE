#!/usr/bin/env python3
"""Compatibility wrapper for batch AFM extraction."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rheed2morph.afm.batch_extract import main


if __name__ == "__main__":
    raise SystemExit(main())
