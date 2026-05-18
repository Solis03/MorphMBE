#!/usr/bin/env python3
"""
Create paired AFM/RHEED folders.

The script scans the top-level folders inside AFM and RHEED, extracts the
four-digit sample number from names that start with "N####", and creates:

    pair/####/AFM
    pair/####/RHEED

Only sample numbers present in both AFM and RHEED are paired.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


SAMPLE_RE = re.compile(r"^N(\d{4})")


def collect_sample_dirs(root: Path) -> dict[str, Path]:
    """Return {sample_number: folder_path} for immediate child folders."""
    samples: dict[str, Path] = {}

    if not root.is_dir():
        raise FileNotFoundError(f"Missing folder: {root}")

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue

        match = SAMPLE_RE.match(child.name)
        if not match:
            continue

        sample_number = match.group(1)

        # If duplicate folders map to the same number, keep the first one found
        # and warn the user instead of guessing which folder is better.
        if sample_number in samples:
            print(
                f"Warning: duplicate {root.name} folder for N{sample_number}: "
                f"keeping {samples[sample_number].name}, skipping {child.name}"
            )
            continue

        samples[sample_number] = child

    return samples


def copy_folder_contents(source: Path, destination: Path, dry_run: bool) -> None:
    """Copy all files and subfolders from source into destination."""
    if dry_run:
        print(f"  would copy {source} -> {destination}")
        return

    destination.mkdir(parents=True, exist_ok=True)

    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create pair folders for sample numbers shared by AFM and RHEED."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("."),
        help="Base folder used for default AFM/RHEED/pair paths.",
    )
    parser.add_argument(
        "--afm_root",
        type=Path,
        default=None,
        help="Folder containing raw AFM sample folders. Default: BASE/AFM.",
    )
    parser.add_argument(
        "--rheed_root",
        type=Path,
        default=None,
        help="Folder containing raw RHEED sample folders. Default: BASE/RHEED.",
    )
    parser.add_argument(
        "--pair_root",
        type=Path,
        default=None,
        help="Output folder for paired samples. Default: BASE/pair.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be created without copying files.",
    )
    args = parser.parse_args()

    base = args.base.expanduser()
    afm_root = (args.afm_root or base / "AFM").expanduser()
    rheed_root = (args.rheed_root or base / "RHEED").expanduser()
    pair_root = (args.pair_root or base / "pair").expanduser()

    afm_samples = collect_sample_dirs(afm_root)
    rheed_samples = collect_sample_dirs(rheed_root)
    common_numbers = sorted(set(afm_samples) & set(rheed_samples))

    print(f"Found {len(afm_samples)} AFM sample folders.")
    print(f"Found {len(rheed_samples)} RHEED sample folders.")
    print(f"Creating {len(common_numbers)} paired folders in {pair_root}.")

    for number in common_numbers:
        sample_root = pair_root / number
        afm_destination = sample_root / "AFM"
        rheed_destination = sample_root / "RHEED"

        print(f"N{number}")
        copy_folder_contents(afm_samples[number], afm_destination, args.dry_run)
        copy_folder_contents(rheed_samples[number], rheed_destination, args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
