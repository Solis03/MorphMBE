"""Removelist checks specific to the peak-saddle staged experiment."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from analysis.rheed_single_frame.removelist import (
    RemovelistAudit,
    assert_no_removed_samples,
    audit_to_json,
    discover_removelist,
    load_removelist_audit,
    parse_removelist,
    write_json,
)


MANDATORY_REMOVELIST_IDS = ("6088",)
MISSING_6088_MESSAGE = (
    "HUMAN ACTION REQUIRED:\n"
    "Sample 6088 is not present in the canonical removelist.\n"
    "Do not run Stage 1."
)
REFERENCE_TERMS = re.compile(r"removelist|remove_list|excluded_samples|bad_samples", re.IGNORECASE)
REFERENCE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".txt", ".toml", ".cfg", ".ini"}


def assert_mandatory_removelist_ids(audit: RemovelistAudit, required_ids: Iterable[str] = MANDATORY_REMOVELIST_IDS) -> None:
    """Fail closed when a mandatory bad sample is absent from the canonical removelist."""
    parsed = set(map(str, audit.sample_ids))
    missing = [str(sample_id) for sample_id in required_ids if str(sample_id) not in parsed]
    if missing:
        if missing == ["6088"]:
            raise RuntimeError(MISSING_6088_MESSAGE)
        joined = ", ".join(missing)
        raise RuntimeError(f"HUMAN ACTION REQUIRED:\nAdd sample(s) {joined} to the canonical removelist, then rerun Stage 0.")


def find_removelist_references(repo_root: Path) -> list[dict[str, Any]]:
    """Find source/config references to removelist-like names without scanning generated outputs."""
    skip_parts = {".git", ".venv", "__pycache__", "outputs", "reports"}
    rows: list[dict[str, Any]] = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(repo_root)
        except ValueError:
            continue
        if any(part in skip_parts for part in rel.parts):
            continue
        if path.suffix.lower() not in REFERENCE_SUFFIXES:
            continue
        name_match = bool(REFERENCE_TERMS.search(path.name))
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if REFERENCE_TERMS.search(line) or (name_match and line_number == 1):
                rows.append(
                    {
                        "path": rel.as_posix(),
                        "line": line_number,
                        "text": line.strip()[:240],
                    }
                )
    return rows
