#!/usr/bin/env python3
"""
Run the full data processing pipeline from raw AFM/RHEED folders.

Short command:
    uv run python run_pipeline.py

This script is intentionally small and idempotent. Each run rebuilds generated
outputs from the current raw data:

1. Validate raw input folders: data/raw/raw_AFM/ and data/raw/raw_RHEED/
2. Remove generated folders: data/pair/ and data/processed_afm/
3. Recreate data/pair/ from raw AFM/RHEED sample ids
4. Extract AFM height maps into data/processed_afm/

It never deletes data/raw/.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = Path("data") / "raw"
RAW_AFM_DIR = RAW_ROOT / "raw_AFM"
RAW_RHEED_DIR = RAW_ROOT / "raw_RHEED"
PAIR_ROOT = Path("data") / "pair"
PROCESSED_AFM_ROOT = Path("data") / "processed_afm"


def require_raw_inputs() -> None:
    missing = [path for path in (RAW_AFM_DIR, RAW_RHEED_DIR) if not path.is_dir()]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(
            "Missing raw input folder(s). Put raw data back in place first:\n"
            f"{missing_text}"
        )


def remove_generated_dir(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise SystemExit(f"Refusing to remove non-directory generated path: {path}")
    print(f"Removing generated directory: {path}")
    shutil.rmtree(path)


def run_step(label: str, command: list[str]) -> None:
    print()
    print(f"==> {label}")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the RHEED-to-AFM data preparation pipeline."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print planned steps without deleting or creating outputs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(PROJECT_ROOT)
    require_raw_inputs()

    if args.dry_run:
        print("Pipeline dry run.")
        print(f"Raw AFM: {RAW_AFM_DIR}")
        print(f"Raw RHEED: {RAW_RHEED_DIR}")
        print(f"Pair output: {PAIR_ROOT}")
        print(f"Processed AFM output: {PROCESSED_AFM_ROOT}")
        print("No files were changed.")
        return 0

    # Clean only generated outputs so repeated runs match the current raw data.
    remove_generated_dir(PAIR_ROOT)
    remove_generated_dir(PROCESSED_AFM_ROOT)

    run_step(
        "Build paired AFM/RHEED folders",
        [
            sys.executable,
            "scripts/make_pairs.py",
            "--afm_root",
            str(RAW_AFM_DIR),
            "--rheed_root",
            str(RAW_RHEED_DIR),
            "--pair_root",
            str(PAIR_ROOT),
        ],
    )
    run_step(
        "Extract AFM height maps",
        [
            sys.executable,
            "scripts/batch_extract_afm_by_sample.py",
            "--pair_root",
            str(PAIR_ROOT),
            "--output_root",
            str(PROCESSED_AFM_ROOT),
        ],
    )

    print()
    print("Pipeline complete.")
    print(f"AFM summary: {PROCESSED_AFM_ROOT / 'afm_summary.csv'}")
    print(f"Sample summary: {PROCESSED_AFM_ROOT / 'sample_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
