"""Read-only discovery of RHEED videos for the replay user interface."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from rheed2morph.rheed.automatic_roi_keyframe import (
    SUPPORTED_VIDEO_SUFFIXES,
)

SAMPLE_PATTERN = re.compile(r"(?<!\d)(\d{4})(?!\d)")


@dataclass(frozen=True)
class VideoEntry:
    sample_id: str
    path: Path
    label: str


def _sample_id(path: Path) -> str | None:
    for part in reversed(path.parts):
        match = SAMPLE_PATTERN.search(part)
        if match:
            return match.group(1)
    return None


def read_removelist(path: str | Path) -> set[str]:
    source = Path(path)
    if not source.exists():
        return set()
    result: set[str] = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.split("#", 1)[0].strip()
        match = SAMPLE_PATTERN.search(stripped)
        if match:
            result.add(match.group(1))
    return result


def discover_videos(
    raw_root: str | Path,
    *,
    excluded_sample_ids: Iterable[str] = (),
) -> list[VideoEntry]:
    """Return videos grouped by sample without changing the raw-data tree."""

    root = Path(raw_root)
    excluded = set(map(str, excluded_sample_ids))
    entries: list[VideoEntry] = []
    if not root.exists():
        return entries
    for path in sorted(root.rglob("*"), key=lambda value: str(value).lower()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
            continue
        sample_id = _sample_id(path)
        if sample_id is None or sample_id in excluded:
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = path
        entries.append(
            VideoEntry(
                sample_id=sample_id,
                path=path.resolve(),
                label=str(relative),
            )
        )
    return entries


def group_by_sample(entries: Iterable[VideoEntry]) -> dict[str, list[VideoEntry]]:
    grouped: dict[str, list[VideoEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.sample_id, []).append(entry)
    return {
        sample_id: sorted(rows, key=lambda row: row.label.lower())
        for sample_id, rows in sorted(grouped.items())
    }
