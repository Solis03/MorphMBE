#!/usr/bin/env python3
"""Generate deterministic benchmark v1 historical LOGO splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from rheed2morph.benchmark.hashing import repo_root, sha256_file
from rheed2morph.benchmark.splits import generate_outer_logo_from_registry, write_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="configs/benchmark_v1/registry/sample_registry_master_v1.csv",
    )
    parser.add_argument(
        "--output",
        default="configs/benchmark_v1/splits/historical_outer_logo_v1.json",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = repo_root(Path.cwd())
    registry = root / args.registry
    output = root / args.output
    split = generate_outer_logo_from_registry(registry)
    if args.check:
        before = output.read_text(encoding="utf-8") if output.exists() else None
        from rheed2morph.benchmark.hashing import canonical_json

        after = canonical_json(split)
        if before != after:
            raise SystemExit("split output is not up to date")
    else:
        write_split(output, split)
        fingerprint = sha256_file(output)
        fingerprint_path = output.with_name("split_fingerprint_v1.sha256")
        fingerprint_path.write_text(
            f"{fingerprint}  {output.relative_to(root).as_posix()}\n",
            encoding="utf-8",
        )
    print(f"outer_folds={len(split['folds'])}")
    print(f"split_sha256={sha256_file(output)}")


if __name__ == "__main__":
    main()
