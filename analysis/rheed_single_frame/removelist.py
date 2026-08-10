"""Canonical removelist discovery and fail-closed sample exclusion."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEADING_SAMPLE_RE = re.compile(r"^\s*(\d{4,})\b\s*(?:[-:;,]?\s*(.*))?$")


@dataclass(frozen=True)
class RemovelistRecord:
    sample_id: str
    raw_line: str
    note: str
    source_path: Path


@dataclass(frozen=True)
class RemovelistAudit:
    path: Path
    sha256: str
    mtime: str
    parser: str
    sample_ids: tuple[str, ...]
    records: tuple[RemovelistRecord, ...]


def resolve_path(repo_root: Path, raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path


def discover_removelist(repo_root: Path, configured_path: str | Path | None) -> Path:
    """Return the single active canonical removelist path or fail closed."""
    repo_root = repo_root.resolve()
    if configured_path:
        path = resolve_path(repo_root, configured_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Configured removelist_path does not exist: {path}"
            )
        return path

    candidates = sorted(
        path.resolve()
        for path in repo_root.rglob("*")
        if path.is_file()
        and re.search(r"remove_?list|removelist", path.name, re.IGNORECASE)
    )
    active = [
        path
        for path in candidates
        if "reports" not in path.relative_to(repo_root).parts
        and "outputs" not in path.relative_to(repo_root).parts
    ]
    if len(active) == 1:
        return active[0]
    if not active:
        raise FileNotFoundError(
            "No canonical removelist could be found; refusing to process samples."
        )
    formatted = "\n".join(f"- {path}" for path in active)
    raise RuntimeError(
        f"Multiple active removelists found; set removelist_path explicitly:\n{formatted}"
    )


def parse_removelist(path: Path) -> tuple[set[str], list[RemovelistRecord]]:
    """Parse leading numeric sample IDs, preserving notes for audit output."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing removelist: {path}")
    records: list[RemovelistRecord] = []
    ids: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = LEADING_SAMPLE_RE.match(raw_line)
        if match is None:
            raise ValueError(f"Could not parse removelist line in {path}: {raw_line!r}")
        sample_id = match.group(1)
        note = (match.group(2) or "").strip()
        ids.add(sample_id)
        records.append(
            RemovelistRecord(
                sample_id=sample_id, raw_line=raw_line, note=note, source_path=path
            )
        )
    if not ids:
        raise ValueError(
            f"Removelist is empty after parsing; refusing to process samples: {path}"
        )
    return ids, records


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_removelist_audit(
    repo_root: Path, configured_path: str | Path | None
) -> RemovelistAudit:
    path = discover_removelist(repo_root, configured_path)
    sample_ids, records = parse_removelist(path)
    stat = path.stat()
    return RemovelistAudit(
        path=path,
        sha256=file_sha256(path),
        mtime=f"{stat.st_mtime:.9f}",
        parser="leading numeric sample id at start of non-comment line",
        sample_ids=tuple(sorted(sample_ids)),
        records=tuple(records),
    )


def audit_to_json(audit: RemovelistAudit) -> dict[str, Any]:
    return {
        "absolute_path": audit.path.as_posix(),
        "sha256": audit.sha256,
        "mtime_epoch_seconds": audit.mtime,
        "parsed_sample_ids": list(audit.sample_ids),
        "parser": audit.parser,
        "records": [
            {
                "sample_id": record.sample_id,
                "raw_line": record.raw_line,
                "note": record.note,
                "source_path": record.source_path.as_posix(),
            }
            for record in audit.records
        ],
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8"
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_ready(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(child) for child in value]
    try:
        import numpy as np

        if isinstance(value, np.integer | np.floating):
            out = value.item()
            if isinstance(out, float) and not math.isfinite(out):
                return None
            return out
    except Exception:
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def assert_no_removed_samples(
    sample_ids: Iterable[str], removelist_ids: Iterable[str], *, context: str
) -> None:
    overlap = sorted(set(map(str, sample_ids)) & set(map(str, removelist_ids)))
    if overlap:
        raise AssertionError(
            f"Removelist samples entered {context}: {', '.join(overlap)}"
        )


def excluded_rows_for_present_samples(
    records: Sequence[RemovelistRecord],
    present_sample_ids: Iterable[str],
) -> list[dict[str, str]]:
    present = set(map(str, present_sample_ids))
    rows = []
    for record in records:
        if record.sample_id in present:
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "source_path": record.source_path.as_posix(),
                    "exclusion_reason": "canonical_removelist",
                    "note": record.note,
                }
            )
    return rows
