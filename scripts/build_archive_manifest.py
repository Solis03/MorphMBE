#!/usr/bin/env python3
"""Build or verify the standalone transfer-integrity manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "provenance" / "file_manifest.sha256"
SUMMARY = ROOT / "provenance" / "file_manifest_summary.json"
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "reproduced_outputs",
    "reproduced_reports",
}
EXCLUDED_FILES = {MANIFEST.resolve(), SUMMARY.resolve(), ROOT / ".DS_Store"}


def included_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and not any(
            part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts
        )
        and path.resolve() not in EXCLUDED_FILES
        and path.name != ".DS_Store"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build() -> None:
    files = included_files()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        "\n".join(
            f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}"
            for path in files
        )
        + "\n",
        encoding="utf-8",
    )
    SUMMARY.write_text(
        json.dumps(
            {
                "archive_root_name": ROOT.name,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "file_count": len(files),
                "total_bytes": sum(path.stat().st_size for path in files),
                "manifest": str(MANIFEST.relative_to(ROOT)),
                "excluded": sorted(EXCLUDED_PARTS),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"built {MANIFEST}: {len(files)} files")


def verify() -> None:
    if not MANIFEST.exists():
        raise SystemExit(f"missing manifest: {MANIFEST}")
    failures: list[str] = []
    checked = 0
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        checked += 1
        if sha256(path) != expected:
            failures.append(f"hash mismatch: {relative}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"verified {checked} archived files")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "verify"))
    args = parser.parse_args()
    build() if args.action == "build" else verify()


if __name__ == "__main__":
    main()
