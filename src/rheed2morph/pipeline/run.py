#!/usr/bin/env python3
"""
Run the full data processing pipeline from raw AFM/RHEED folders.

Short command:
    uv run python run_pipeline.py

This script is intentionally small and idempotent. Each run rebuilds generated
outputs from the current raw data:

1. Validate raw input folders: data/raw/raw_AFM/ and data/raw/raw_RHEED/
2. Remove generated folders: data/pair/, data/processed_afm/, and
   data/plane_corrected_afm/
3. Recreate data/pair/ from raw AFM/RHEED sample ids
4. Extract AFM height maps into data/processed_afm/
5. Subtract fitted planes into data/plane_corrected_afm/
6. Render 1 x 1 um AFM overview grids into reports/figures/
7. Run descriptor-to-image reconstruction experiments

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
RAW_RHEED_SELECTED_DIR = RAW_ROOT / "raw_RHEED_selected"
RAW_RHEED_DIR = RAW_ROOT / "raw_RHEED"
PAIR_ROOT = Path("data") / "pair"
PROCESSED_AFM_ROOT = Path("data") / "processed_afm"
PLANE_CORRECTED_AFM_ROOT = Path("data") / "plane_corrected_afm"
REPORT_FIGURES_ROOT = Path("reports") / "figures" / "afm_scan_size_grids"
AFM_RECON_ROOT = Path("data") / "afm_descriptor_reconstruction"
AFM_RECON_LARGE_ROOT = Path("data") / "afm_descriptor_reconstruction_large"
AFM_RECON_REPORT_ROOT = Path("reports") / "afm_descriptor_reconstruction"
AFM_RECON_LARGE_REPORT_ROOT = Path("reports") / "afm_descriptor_reconstruction_large"


def resolve_rheed_input_dir() -> Path:
    """Prefer manually curated RHEED selection if present."""
    if RAW_RHEED_SELECTED_DIR.is_dir():
        return RAW_RHEED_SELECTED_DIR
    return RAW_RHEED_DIR


def require_raw_inputs(rheed_input_dir: Path) -> None:
    missing = [path for path in (RAW_AFM_DIR, rheed_input_dir) if not path.is_dir()]
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


def reconstruction_steps() -> list[tuple[str, list[str]]]:
    """Return descriptor-to-image reconstruction commands in dependency order."""
    return [
        (
            "Build 1um AFM reconstruction manifest",
            [
                sys.executable,
                "scripts/afm_descriptor_reconstruction/build_afm_manifest.py",
                "--input_dir",
                str(PLANE_CORRECTED_AFM_ROOT),
                "--output_csv",
                str(AFM_RECON_ROOT / "afm_1um_manifest.csv"),
            ],
        ),
        (
            "Extract 1um AFM descriptors",
            [
                sys.executable,
                "scripts/afm_descriptor_reconstruction/extract_afm_descriptors.py",
                "--manifest",
                str(AFM_RECON_ROOT / "afm_1um_manifest.csv"),
                "--output_csv",
                str(AFM_RECON_ROOT / "afm_descriptors.csv"),
                "--output_dir",
                str(AFM_RECON_ROOT / "descriptors"),
            ],
        ),
        (
            "Select 1um AFM descriptors",
            [
                sys.executable,
                "scripts/afm_descriptor_reconstruction/select_descriptors.py",
                "--descriptor_csv",
                str(AFM_RECON_ROOT / "afm_descriptors.csv"),
                "--output_dir",
                str(AFM_RECON_ROOT / "selected_descriptors"),
            ],
        ),
        (
            "Train 1um descriptor PCA decoder",
            [
                sys.executable,
                "scripts/afm_descriptor_reconstruction/train_descriptor_pca_decoder.py",
                "--manifest",
                str(AFM_RECON_ROOT / "afm_1um_manifest.csv"),
                "--selected_descriptor_csv",
                str(AFM_RECON_ROOT / "selected_descriptors" / "selected_descriptor_table.csv"),
                "--network_input_dir",
                str(AFM_RECON_ROOT / "network_inputs"),
                "--output_dir",
                str(AFM_RECON_ROOT / "pca_decoder"),
                "--report_dir",
                str(AFM_RECON_REPORT_ROOT / "pca_decoder"),
            ],
        ),
        (
            "Train 1um descriptor MLP decoder",
            [
                sys.executable,
                "scripts/afm_descriptor_reconstruction/train_descriptor_mlp_decoder.py",
                "--manifest",
                str(AFM_RECON_ROOT / "afm_1um_manifest.csv"),
                "--selected_descriptor_csv",
                str(AFM_RECON_ROOT / "selected_descriptors" / "selected_descriptor_table.csv"),
                "--network_input_dir",
                str(AFM_RECON_ROOT / "network_inputs"),
                "--output_dir",
                str(AFM_RECON_ROOT / "mlp_decoder"),
                "--report_dir",
                str(AFM_RECON_REPORT_ROOT / "mlp_decoder"),
            ],
        ),
        (
            "Build large AFM reconstruction manifest",
            [
                sys.executable,
                "scripts/afm_descriptor_reconstruction_large_dataset/build_large_afm_manifest.py",
                "--input_dir",
                str(PLANE_CORRECTED_AFM_ROOT),
            ],
        ),
        (
            "Extract large AFM descriptors",
            [
                sys.executable,
                "scripts/afm_descriptor_reconstruction_large_dataset/extract_large_afm_descriptors.py",
                "--manifest",
                str(AFM_RECON_LARGE_ROOT / "large_afm_manifest.csv"),
            ],
        ),
        (
            "Select large AFM descriptors",
            [
                sys.executable,
                "scripts/afm_descriptor_reconstruction_large_dataset/select_large_descriptors.py",
                "--descriptor_csv",
                str(AFM_RECON_LARGE_ROOT / "large_afm_descriptors.csv"),
            ],
        ),
        (
            "Train large AFM descriptor MLP decoder",
            [
                sys.executable,
                "scripts/afm_descriptor_reconstruction_large_dataset/train_large_mlp_decoder.py",
                "--manifest",
                str(AFM_RECON_LARGE_ROOT / "large_afm_manifest.csv"),
                "--selected_descriptor_csv",
                str(AFM_RECON_LARGE_ROOT / "selected_descriptors" / "selected_descriptors.csv"),
                "--network_input_dir",
                str(AFM_RECON_LARGE_ROOT / "network_inputs"),
                "--output_dir",
                str(AFM_RECON_LARGE_ROOT / "mlp_decoder"),
                "--report_dir",
                str(AFM_RECON_LARGE_REPORT_ROOT / "mlp_decoder"),
            ],
        ),
    ]


def require_reconstruction_inputs() -> None:
    if not PLANE_CORRECTED_AFM_ROOT.is_dir():
        raise SystemExit(
            "Missing plane-corrected AFM folder for reconstruction: "
            f"{PLANE_CORRECTED_AFM_ROOT}\n"
            "Run the full pipeline first, or restore data/plane_corrected_afm/."
        )


def run_reconstruction_steps() -> None:
    require_reconstruction_inputs()
    for label, command in reconstruction_steps():
        run_step(label, command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the RHEED-to-AFM data preparation pipeline."
    )
    parser.add_argument(
        "stage",
        nargs="?",
        choices=("all", "recon"),
        default="all",
        help=(
            "Pipeline stage to run. Default 'all' runs data preparation, visualization, "
            "and reconstruction. Use 'recon' to run only reconstruction steps."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print planned steps without deleting or creating outputs.",
    )
    parser.add_argument(
        "--skip-plane-correction",
        action="store_true",
        help="Stop after AFM extraction and do not write data/plane_corrected_afm.",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Do not render AFM scan-size overview grid figures.",
    )
    parser.add_argument(
        "--skip-reconstruction",
        action="store_true",
        help="Do not run descriptor-to-image reconstruction experiments in the full pipeline.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(PROJECT_ROOT)
    rheed_input_dir = resolve_rheed_input_dir()

    if args.stage == "recon" and args.skip_reconstruction:
        raise SystemExit("'recon' stage cannot be combined with --skip-reconstruction.")
    if args.stage == "recon" and (args.skip_plane_correction or args.skip_visualization):
        print("warning: --skip-plane-correction/--skip-visualization are ignored for stage 'recon'.")

    if args.dry_run:
        print("Pipeline dry run.")
        print(f"Stage: {args.stage}")
        if args.stage == "all":
            print(f"Raw AFM: {RAW_AFM_DIR}")
            print(f"Raw RHEED: {rheed_input_dir}")
            print(f"Pair output: {PAIR_ROOT}")
            print(f"Processed AFM output: {PROCESSED_AFM_ROOT}")
            if not args.skip_plane_correction:
                print(f"Plane-corrected AFM output: {PLANE_CORRECTED_AFM_ROOT}")
            if not args.skip_plane_correction and not args.skip_visualization:
                print(f"AFM overview figures: {REPORT_FIGURES_ROOT}")
        if args.stage == "recon" or (
            args.stage == "all" and not args.skip_plane_correction and not args.skip_reconstruction
        ):
            print(f"Reconstruction input: {PLANE_CORRECTED_AFM_ROOT}")
            for label, command in reconstruction_steps():
                print(f"Reconstruction step: {label}")
                print(f"  {' '.join(command)}")
        print("No files were changed.")
        return 0

    if args.stage == "recon":
        run_reconstruction_steps()
        print()
        print("Reconstruction pipeline complete.")
        print(f"1um reconstruction output: {AFM_RECON_ROOT}")
        print(f"Large reconstruction output: {AFM_RECON_LARGE_ROOT}")
        print(f"Large reconstruction report: {AFM_RECON_LARGE_REPORT_ROOT / 'summary.md'}")
        return 0

    require_raw_inputs(rheed_input_dir)

    # Clean only generated outputs so repeated runs match the current raw data.
    remove_generated_dir(PAIR_ROOT)
    remove_generated_dir(PROCESSED_AFM_ROOT)
    if not args.skip_plane_correction:
        remove_generated_dir(PLANE_CORRECTED_AFM_ROOT)

    run_step(
        "Build paired AFM/RHEED folders",
        [
            sys.executable,
            "scripts/make_pairs.py",
            "--afm_root",
            str(RAW_AFM_DIR),
            "--rheed_root",
            str(rheed_input_dir),
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
    if not args.skip_plane_correction:
        run_step(
            "Subtract fitted planes from AFM height maps",
            [
                sys.executable,
                "scripts/afm_plane_correct.py",
                "--input-root",
                str(PROCESSED_AFM_ROOT),
                "--output-root",
                str(PLANE_CORRECTED_AFM_ROOT),
            ],
        )
    if not args.skip_plane_correction and not args.skip_visualization:
        run_step(
            "Render 1 x 1 um AFM overview grids",
            [
                sys.executable,
                "scripts/afm_scan_size_grid.py",
                "--processed-root",
                str(PROCESSED_AFM_ROOT),
                "--plane-corrected-root",
                str(PLANE_CORRECTED_AFM_ROOT),
                "--output-dir",
                str(REPORT_FIGURES_ROOT),
            ],
        )
    if not args.skip_plane_correction and not args.skip_reconstruction:
        run_reconstruction_steps()

    print()
    print("Pipeline complete.")
    print(f"AFM summary: {PROCESSED_AFM_ROOT / 'afm_summary.csv'}")
    print(f"Sample summary: {PROCESSED_AFM_ROOT / 'sample_summary.csv'}")
    if not args.skip_plane_correction:
        print(f"Plane-corrected AFM output: {PLANE_CORRECTED_AFM_ROOT}")
    if not args.skip_plane_correction and not args.skip_visualization:
        print(f"AFM overview figures: {REPORT_FIGURES_ROOT}")
    if not args.skip_plane_correction and not args.skip_reconstruction:
        print(f"1um reconstruction output: {AFM_RECON_ROOT}")
        print(f"Large reconstruction output: {AFM_RECON_LARGE_ROOT}")
        print(f"Large reconstruction report: {AFM_RECON_LARGE_REPORT_ROOT / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
