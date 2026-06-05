#!/usr/bin/env python3
"""Build a complete AFM candidate table, then derive one-to-one manifests."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rheed2morph.manifests.afm_candidates import (
    DEFAULT_PAIR_ROOT,
    DEFAULT_PLANE_CORRECTED_ROOT,
    DEFAULT_PROCESSED_ROOT,
    build_complete_candidate_rows,
    candidate_rows_to_dicts,
    coverage_rows,
    display_path,
)
from rheed2morph.manifests.one_to_one import (
    DEFAULT_OUT_DIR,
    build_one_to_one_manifests,
    write_csv,
)


def resolve_cli_path(path: Path, repo_root: Path) -> Path:
    """Resolve CLI paths predictably from the current working directory first.

    This avoids accidentally nesting `MorphMBE/...` under `repo_root` when the
    caller already runs the script from the workspace root.
    """

    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    cwd_candidate = (Path.cwd() / expanded).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (repo_root / expanded).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a complete AFM candidate table and one-to-one manifests."
    )
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--plane-corrected-root", type=Path, default=DEFAULT_PLANE_CORRECTED_ROOT)
    parser.add_argument("--pair-root", type=Path, default=DEFAULT_PAIR_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-sizes", type=float, nargs="+", default=[1.0, 0.5, 5.0])
    parser.add_argument("--size-tolerance", type=float, default=0.10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    processed_root = resolve_cli_path(args.processed_root, repo_root)
    plane_corrected_root = resolve_cli_path(args.plane_corrected_root, repo_root)
    pair_root = resolve_cli_path(args.pair_root, repo_root)
    out_dir = resolve_cli_path(args.out_dir, repo_root)

    rows = build_complete_candidate_rows(
        processed_root=processed_root,
        plane_corrected_root=plane_corrected_root,
        pair_root=pair_root,
    )
    candidate_table_path = out_dir / "afm_candidate_table_complete.csv"
    candidate_dict_rows = candidate_rows_to_dicts(rows)
    write_csv(
        candidate_table_path,
        candidate_dict_rows,
        [
            "sample_id",
            "group_id",
            "material",
            "afm_path",
            "original_source_path",
            "rheed_path",
            "scan_size_um",
            "scan_size_source",
            "resolution_h",
            "resolution_w",
            "channel_name",
            "is_plane_corrected",
            "is_rendered_image",
            "is_physical_height_map",
            "metadata_path",
        ],
    )
    summary = build_one_to_one_manifests(
        manifest_path=candidate_table_path,
        pair_root=pair_root,
        descriptor_aux_csv=out_dir / "unused.csv",
        out_dir=out_dir,
        target_sizes=args.target_sizes,
        size_tolerance=args.size_tolerance,
    )
    coverage = coverage_rows(rows)
    summary_by_label = summary.get("target_summary_by_label", {})
    for row in coverage:
        row["one_to_one_pairs_1um"] = summary_by_label.get("1um", {}).get("selected_pair_count", "")
        row["one_to_one_pairs_0p5um"] = summary_by_label.get("0p5um", {}).get("selected_pair_count", "")
        row["one_to_one_pairs_5um"] = summary_by_label.get("5um", {}).get("selected_pair_count", "")
        row["one_to_one_pairs_all_size_representative"] = summary.get("validation", [{}])[-1].get("row_count", "")
    coverage_path = out_dir / "multisize_afm_coverage_report.csv"
    write_csv(
        coverage_path,
        coverage,
        [
            "scan_size_um",
            "afm_file_count",
            "unique_group_count",
            "one_to_one_pairs_1um",
            "one_to_one_pairs_0p5um",
            "one_to_one_pairs_5um",
            "one_to_one_pairs_all_size_representative",
            "sample_count_with_multiple_scan_sizes",
            "sample_count_only_1um",
            "sample_count_with_0p5um_or_5um",
        ],
    )

    size_counts = Counter(row["scan_size_um"] for row in candidate_dict_rows if row["scan_size_um"])
    print(f"Wrote complete AFM candidate table to {display_path(candidate_table_path)} ({len(rows)} rows)")
    print(f"Wrote multi-size coverage report to {display_path(coverage_path)}")
    print(f"Scan sizes found: {dict(size_counts)}")
    print(f"One-to-one manifests written under {display_path(out_dir)}")
    print(f"Summary JSON: {display_path(out_dir / 'manifest_build_summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
