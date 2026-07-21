#!/usr/bin/env python3
"""Discover exactly the configured unseen MPG files without touching sources."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))

from keyframe_selector.common import ensure_package_dirs, package_root, repo_root_from
from keyframe_selector.manifests import discover_mpgs
from keyframe_selector.provenance import refresh_provenance, verify_frozen_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-hash-mode", choices=["none", "fast", "full"], default="fast")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = repo_root_from(THIS)
    pkg_root = package_root(repo_root)
    ensure_package_dirs(pkg_root)
    rows = discover_mpgs(repo_root, pkg_root, hash_mode=args.video_hash_mode)
    refresh_provenance(repo_root, pkg_root)
    frozen = verify_frozen_manifest(repo_root)
    zero = sorted({row["sample_id"] for row in rows if int(row["candidate_count_for_sample"] or 0) == 0})
    multiple = sorted({row["sample_id"] for row in rows if int(row["candidate_count_for_sample"] or 0) > 1})
    print(f"wrote {pkg_root / 'manifests' / 'discovered_mpg_files.csv'}")
    print(f"candidate rows: {len(rows)}")
    print(f"zero-MPG samples: {zero or 'none'}")
    print(f"multiple-MPG samples: {multiple or 'none'}")
    print(f"frozen manifest status: {frozen['status']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
