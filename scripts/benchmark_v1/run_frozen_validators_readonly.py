#!/usr/bin/env python3
"""Run frozen validators on temporary copies so immutable packages stay untouched."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from rheed2morph.benchmark.hashing import repo_root, write_json


RETRO_REL = Path("publication_freeze/rheed_afm_single_frame_v1_2026-07-18")
PROSPECTIVE_REL = Path("publication_freeze/prospective_unseen_single_frame_v1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = repo_root(Path.cwd())
    temp_root = Path(tempfile.mkdtemp(prefix="rheed2morph_phase0_validators_"))
    (temp_root / ".git").mkdir()
    (temp_root / "publication_freeze").mkdir()
    shutil.copytree(root / RETRO_REL, temp_root / RETRO_REL)
    shutil.copytree(root / PROSPECTIVE_REL, temp_root / PROSPECTIVE_REL)
    for rel in ["data", "src", "tools"]:
        target = root / rel
        if target.exists():
            os.symlink(target, temp_root / rel)
    results = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "temp_root": temp_root.as_posix(),
        "note": "Validators were run on temporary copies to avoid modifying immutable publication packages.",
        "retrospective_verifier": run(
            [sys.executable, str(temp_root / RETRO_REL / "code/verify_freeze.py")],
            cwd=temp_root,
        ),
        "prospective_validator": run(
            [
                sys.executable,
                str(temp_root / PROSPECTIVE_REL / "code/validate_keyframe_selections.py"),
                "--require-complete",
            ],
            cwd=temp_root,
        ),
    }
    results["status"] = (
        "ok"
        if results["retrospective_verifier"]["exit_code"] == 0
        and results["prospective_validator"]["exit_code"] == 0
        else "failed"
    )
    write_json(root / args.output, results)
    print(json.dumps(results, indent=2, sort_keys=True))
    if results["status"] != "ok":
        raise SystemExit(1)


def run(cmd: list[str], cwd: Path) -> dict:
    start = time.monotonic()
    result = subprocess.run(cmd, cwd=cwd, check=False, text=True, capture_output=True)
    return {
        "command": cmd,
        "exit_code": result.returncode,
        "runtime_seconds": time.monotonic() - start,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


if __name__ == "__main__":
    main()
