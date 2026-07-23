#!/usr/bin/env python3
"""Validate benchmark v1 registry, split, metadata, protocol, and schema guards."""

from __future__ import annotations

import json
from pathlib import Path

from rheed2morph.benchmark.hashing import repo_root
from rheed2morph.benchmark.validation import validate_all


def main() -> None:
    root = repo_root(Path.cwd())
    errors = validate_all(root)
    payload = {"status": "ok" if not errors else "failed", "error_count": len(errors), "errors": errors}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
