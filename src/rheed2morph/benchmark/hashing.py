"""Deterministic hashing and small-file manifest helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


def repo_root(start: Path | None = None) -> Path:
    """Return the Git repository root for this checkout."""

    start_path = (start or Path.cwd()).resolve()
    if start_path.is_file():
        start_path = start_path.parent
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=start_path,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "could not find git root")
    return Path(result.stdout.strip())


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"


def sha256_json(data: Any) -> str:
    return sha256_text(canonical_json(data))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _csv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload), encoding="utf-8")


def write_sha256_manifest(path: Path, files: Iterable[Path], root: Path) -> None:
    rows = []
    for file_path in sorted(files, key=lambda p: p.as_posix()):
        rows.append(f"{sha256_file(root / file_path)}  {file_path.as_posix()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def combined_file_hash(files: Iterable[Path], root: Path) -> str:
    lines = []
    for file_path in sorted(files, key=lambda p: p.as_posix()):
        lines.append(f"{sha256_file(root / file_path)}  {file_path.as_posix()}")
    return sha256_text("\n".join(lines) + "\n")


def file_metadata(path: Path, root: Path, existing_checksum: str = "") -> dict[str, Any]:
    full = root / path
    if not full.exists():
        return {
            "relative_path": path.as_posix(),
            "exists": False,
            "size_bytes": None,
            "mtime_ns": None,
            "existing_checksum": existing_checksum,
        }
    stat = full.stat()
    return {
        "relative_path": path.as_posix(),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "existing_checksum": existing_checksum,
    }


def file_hash_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def has_git_lfs(root: Path) -> bool:
    if (root / ".gitattributes").exists():
        try:
            if "filter=lfs" in (root / ".gitattributes").read_text(encoding="utf-8", errors="ignore"):
                return True
        except OSError:
            pass
    if (root / ".git/lfs").exists():
        return True
    return bool(os.environ.get("GIT_LFS_SKIP_SMUDGE"))

