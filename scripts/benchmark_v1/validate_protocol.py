#!/usr/bin/env python3
"""Validate benchmark v1 protocol text and report its hash."""

from __future__ import annotations

import json
from pathlib import Path

from rheed2morph.benchmark.hashing import repo_root, sha256_file
from rheed2morph.benchmark.protocol import validate_protocol_text


def main() -> None:
    root = repo_root(Path.cwd())
    path = root / "configs/benchmark_v1/protocol_v1.yaml"
    errors = validate_protocol_text(path)
    payload = {
        "status": "ok" if not errors else "failed",
        "protocol_hash": sha256_file(path),
        "errors": errors,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
