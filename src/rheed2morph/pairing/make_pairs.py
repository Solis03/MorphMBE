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
VIDEO_SUFFIXES = {".mov", ".mp4", ".avi", ".mkv", ".m4v", ".mts", ".m2ts"}
HEIC_SUFFIXES = {".heic", ".heif"}


def load_removed_samples(remove_list_path: Path) -> set[str]:
    """Load four-digit sample ids from a plain-text remove list file."""
    if not remove_list_path.exists():
        return set()

    removed: set[str] = set()
    for raw_line in remove_list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        for sample_id in re.findall(r"\d{4}", line):
            removed.add(sample_id)
    return removed


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


def visible_files(root: Path) -> list[Path]:
    return [item for item in sorted(root.iterdir()) if item.is_file() and not item.name.startswith(".")]


def is_single_heic_only_sample(rheed_sample_dir: Path) -> bool:
    files = visible_files(rheed_sample_dir)
    return len(files) == 1 and files[0].suffix.lower() in HEIC_SUFFIXES


def collect_main_video_files(rheed_sample_dir: Path) -> list[Path]:
    return [
        item
        for item in visible_files(rheed_sample_dir)
        if item.stem.lower() == "main" and item.suffix.lower() in VIDEO_SUFFIXES
    ]


def copy_selected_files(files: list[Path], destination: Path, dry_run: bool) -> None:
    if dry_run:
        for source in files:
            print(f"  would copy {source} -> {destination / source.name}")
        return

    destination.mkdir(parents=True, exist_ok=True)
    for source in files:
        shutil.copy2(source, destination / source.name)


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
        "--remove_list",
        type=Path,
        default=None,
        help=(
            "Text file listing sample ids to exclude before pairing. "
            "Default: BASE/removelist.txt"
        ),
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
    remove_list_path = (args.remove_list or base / "removelist.txt").expanduser()

    afm_samples = collect_sample_dirs(afm_root)
    rheed_samples = collect_sample_dirs(rheed_root)
    removed_samples = load_removed_samples(remove_list_path)

    common_numbers_all = sorted(set(afm_samples) & set(rheed_samples))
    common_numbers_after_remove = [
        number for number in common_numbers_all if number not in removed_samples
    ]

    selected_rheed_main_files: dict[str, list[Path]] = {}
    skipped_single_heic: list[str] = []
    skipped_no_main_video: list[str] = []
    for number in common_numbers_after_remove:
        rheed_sample_dir = rheed_samples[number]
        if is_single_heic_only_sample(rheed_sample_dir):
            skipped_single_heic.append(number)
            continue

        main_video_files = collect_main_video_files(rheed_sample_dir)
        if not main_video_files:
            skipped_no_main_video.append(number)
            continue

        selected_rheed_main_files[number] = main_video_files

    common_numbers = sorted(selected_rheed_main_files)

    print(f"Found {len(afm_samples)} AFM sample folders.")
    print(f"Found {len(rheed_samples)} RHEED sample folders.")
    if removed_samples:
        removed_in_common = sorted(set(common_numbers_all) & removed_samples)
        removed_not_found = sorted(removed_samples - set(common_numbers_all))
        print(f"Loaded {len(removed_samples)} removed sample id(s) from {remove_list_path}.")
        if removed_in_common:
            print(f"Excluded {len(removed_in_common)} sample(s): {', '.join(removed_in_common)}")
        if removed_not_found:
            print(
                "Remove-list ids not found in both AFM/RHEED roots: "
                f"{', '.join(removed_not_found)}"
            )
    if skipped_single_heic:
        print(
            "Skipped samples with only one HEIC file in RHEED folder: "
            f"{', '.join(skipped_single_heic)}"
        )
    if skipped_no_main_video:
        print(
            "Skipped samples without RHEED main video file: "
            f"{', '.join(skipped_no_main_video)}"
        )
    print(f"Creating {len(common_numbers)} paired folders in {pair_root}.")

    for number in common_numbers:
        sample_root = pair_root / number
        afm_destination = sample_root / "AFM"
        rheed_destination = sample_root / "RHEED"

        print(f"N{number}")
        copy_folder_contents(afm_samples[number], afm_destination, args.dry_run)
        copy_selected_files(selected_rheed_main_files[number], rheed_destination, args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
