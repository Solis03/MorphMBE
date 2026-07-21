#!/usr/bin/env python3
"""Launch the prospective unseen manual keyframe selector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))

from keyframe_selector.common import ensure_package_dirs, package_root, repo_root_from, which_or_error
from keyframe_selector.gui import clear_cache, launch, smoke
from keyframe_selector.provenance import refresh_provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear-cache", action="store_true", help="Delete bounded frame/thumbnail cache and exit.")
    parser.add_argument("--smoke-test", action="store_true", help="Initialize state without opening a GUI window.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = repo_root_from(THIS)
    pkg_root = package_root(repo_root)
    ensure_package_dirs(pkg_root)
    which_or_error("ffprobe")
    which_or_error("ffmpeg")
    if args.clear_cache:
        clear_cache(pkg_root)
        refresh_provenance(repo_root, pkg_root)
        print(f"cleared cache under {pkg_root / 'cache'}")
        return
    if args.smoke_test:
        smoke(repo_root, pkg_root)
        refresh_provenance(repo_root, pkg_root)
        return
    launch(repo_root, pkg_root)
    refresh_provenance(repo_root, pkg_root)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
