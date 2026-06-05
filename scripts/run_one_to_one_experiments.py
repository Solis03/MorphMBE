#!/usr/bin/env python3
"""Run one-to-one manifest experiments and summarize comparable outputs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "one_to_one"
DESCRIPTOR_SCRIPT = REPO_ROOT / "scripts" / "rheed_to_afm_descriptor_mvp.py"
AUTOENCODER_SCRIPT = REPO_ROOT / "scripts" / "train_afm_autoencoder_mvp.py"
LATENT_SCRIPT = REPO_ROOT / "scripts" / "rheed_to_afm_latent_mvp.py"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def manifest_subset_name(path: Path) -> str:
    stem = path.stem
    if stem == "manifest_all_size_representative_one_to_one":
        return "all_size_representative"
    if stem.startswith("manifest_") and stem.endswith("_one_to_one"):
        return stem.removeprefix("manifest_").removesuffix("_one_to_one")
    return stem


def manifest_paths(manifest_dir: Path) -> list[Path]:
    preferred = [
        manifest_dir / "manifest_1um_one_to_one.csv",
        manifest_dir / "manifest_0p5um_one_to_one.csv",
        manifest_dir / "manifest_5um_one_to_one.csv",
        manifest_dir / "manifest_all_size_representative_one_to_one.csv",
    ]
    return [path for path in preferred if path.is_file()]


def run_command(label: str, command: list[str]) -> None:
    print()
    print(f"==> {label}")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def load_descriptor_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def notes_for_subset(subset_name: str, build_summary: dict[str, Any] | None) -> str:
    if build_summary is None:
        return ""
    if subset_name == "all_size_representative":
        return "representative one-to-one selection across all sizes"
    target_summary = build_summary.get("target_summary_by_label", {}).get(subset_name)
    if target_summary is None:
        return ""
    return f"dropped_groups={target_summary['dropped_group_count']}"


def group_count_for_manifest(manifest_path: Path) -> int:
    rows = read_csv(manifest_path)
    return len({row["group_id"] for row in rows})


def comparison_rows(
    manifest_dir: Path,
    report_root: Path,
    run_notes: dict[str, str],
) -> list[dict[str, Any]]:
    build_summary_path = manifest_dir / "manifest_build_summary.json"
    build_summary = json.loads(build_summary_path.read_text(encoding="utf-8")) if build_summary_path.is_file() else None
    rows: list[dict[str, Any]] = []
    for manifest_path in manifest_paths(manifest_dir):
        subset = manifest_subset_name(manifest_path)
        manifest_rows = read_csv(manifest_path)
        descriptor_dir = report_root / subset / "descriptor_data"
        descriptor_metrics = load_descriptor_metrics(descriptor_dir / "metrics_summary.json")
        latent_grid = report_root / subset / "rheed_to_afm_latent" / "nearest_latent_grid.png"
        rows.append(
            {
                "manifest_name": manifest_path.name,
                "manifest_path": display_path(manifest_path),
                "pair_count": len(manifest_rows),
                "material_count": len({row["material"] for row in manifest_rows}),
                "group_count": len({row["group_id"] for row in manifest_rows}),
                "descriptor_mean_mae": (
                    f"{descriptor_metrics['overall_model_metrics']['mean_mae']:.8f}"
                    if descriptor_metrics is not None
                    else ""
                ),
                "descriptor_mean_rmse": (
                    f"{descriptor_metrics['overall_model_metrics']['mean_rmse']:.8f}"
                    if descriptor_metrics is not None
                    else ""
                ),
                "descriptor_mean_r2": (
                    f"{descriptor_metrics['overall_model_metrics']['mean_r2']:.8f}"
                    if descriptor_metrics is not None
                    else ""
                ),
                "latent_nearest_neighbor_output_path": (
                    display_path(latent_grid) if latent_grid.is_file() else ""
                ),
                "notes": "; ".join(filter(None, [notes_for_subset(subset, build_summary), run_notes.get(subset, "")])),
            }
        )
    return rows


def write_comparison_summary(report_root: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = report_root / "comparison_metrics.csv"
    write_csv(
        csv_path,
        rows,
        [
            "manifest_name",
            "manifest_path",
            "pair_count",
            "material_count",
            "group_count",
            "descriptor_mean_mae",
            "descriptor_mean_rmse",
            "descriptor_mean_r2",
            "latent_nearest_neighbor_output_path",
            "notes",
        ],
    )
    md_path = report_root / "comparison_summary.md"
    lines = [
        "# One-to-One Comparison Summary",
        "",
        "| Manifest | Pairs | Materials | Groups | Descriptor MAE | Descriptor RMSE | Descriptor R2 | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['manifest_name']} | {row['pair_count']} | {row['material_count']} | "
            f"{row['group_count']} | {row['descriptor_mean_mae'] or 'n/a'} | "
            f"{row['descriptor_mean_rmse'] or 'n/a'} | {row['descriptor_mean_r2'] or 'n/a'} | "
            f"{row['notes'] or ''} |"
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one-to-one manifest experiments and summarize comparison metrics."
    )
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--autoencoder-epochs", type=int, default=100)
    parser.add_argument("--skip-descriptor", action="store_true")
    parser.add_argument("--skip-autoencoder", action="store_true")
    parser.add_argument("--skip-latent", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_dir = args.manifest_dir if args.manifest_dir.is_absolute() else (REPO_ROOT / args.manifest_dir)
    report_root = args.report_root if args.report_root.is_absolute() else (REPO_ROOT / args.report_root)
    if not manifest_dir.is_dir():
        raise SystemExit(f"Manifest directory does not exist: {manifest_dir}")

    manifests = manifest_paths(manifest_dir)
    if not manifests:
        raise SystemExit(f"No one-to-one manifest CSVs found in {manifest_dir}")

    if not args.skip_descriptor and not DESCRIPTOR_SCRIPT.is_file():
        raise SystemExit(f"Descriptor MVP script does not exist: {DESCRIPTOR_SCRIPT}")

    if not args.skip_autoencoder and not AUTOENCODER_SCRIPT.is_file():
        raise SystemExit(f"AFM autoencoder MVP script does not exist: {AUTOENCODER_SCRIPT}")
    if not args.skip_latent and not LATENT_SCRIPT.is_file():
        raise SystemExit(f"RHEED-to-AFM latent MVP script does not exist: {LATENT_SCRIPT}")

    run_notes: dict[str, str] = {}
    for manifest_path in manifests:
        subset = manifest_subset_name(manifest_path)
        subset_root = report_root / subset
        manifest_group_count = group_count_for_manifest(manifest_path)
        if manifest_group_count < 3:
            note = f"descriptor_skipped_insufficient_groups={manifest_group_count}"
            run_notes[subset] = note
            print(f"Skipping learned MVPs for {subset}: {note}")
            continue
        if not args.skip_descriptor:
            run_command(
                f"Descriptor MVP for {subset}",
                [
                    sys.executable,
                    str(DESCRIPTOR_SCRIPT),
                    "--device",
                    args.device,
                    "--one-to-one-manifest",
                    str(manifest_path),
                    "--data-dir",
                    str(subset_root / "descriptor_data"),
                    "--report-dir",
                    str(subset_root),
                ],
            )
            run_notes.setdefault(subset, "descriptor_ran")
        if not args.skip_autoencoder:
            run_command(
                f"AFM autoencoder MVP for {subset}",
                [
                    sys.executable,
                    str(AUTOENCODER_SCRIPT),
                    "--manifest",
                    str(manifest_path),
                    "--device",
                    args.device,
                    "--epochs",
                    str(args.autoencoder_epochs),
                    "--out-dir",
                    str(subset_root / "afm_autoencoder"),
                ],
            )
            run_notes[subset] = "; ".join(filter(None, [run_notes.get(subset, ""), "autoencoder_ran"]))
        if not args.skip_latent:
            run_command(
                f"RHEED-to-AFM latent MVP for {subset}",
                [
                    sys.executable,
                    str(LATENT_SCRIPT),
                    "--manifest",
                    str(manifest_path),
                    "--device",
                    args.device,
                    "--rheed-embeddings",
                    str(subset_root / "descriptor_data" / "sample_embeddings.npy"),
                    "--rheed-embedding-index",
                    str(subset_root / "descriptor_data" / "sample_embedding_index.csv"),
                    "--afm-latents",
                    str(subset_root / "afm_autoencoder" / "afm_latents.npy"),
                    "--afm-latent-index",
                    str(subset_root / "afm_autoencoder" / "afm_latent_index.csv"),
                    "--autoencoder-checkpoint",
                    str(subset_root / "afm_autoencoder" / "autoencoder_checkpoint.pt"),
                    "--out-dir",
                    str(subset_root / "rheed_to_afm_latent"),
                ],
            )
            run_notes[subset] = "; ".join(filter(None, [run_notes.get(subset, ""), "latent_ran"]))

    rows = comparison_rows(manifest_dir, report_root, run_notes)
    write_comparison_summary(report_root, rows)
    print()
    print(f"Wrote comparison metrics to {display_path(report_root / 'comparison_metrics.csv')}")
    print(f"Wrote comparison summary to {display_path(report_root / 'comparison_summary.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
