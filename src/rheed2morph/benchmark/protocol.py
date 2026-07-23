"""Protocol helpers for benchmark v1."""

from __future__ import annotations

from pathlib import Path

from .hashing import sha256_file


REQUIRED_PROTOCOL_PHRASES = [
    "historical_development",
    "prospective_pilot_seen",
    "T4_second_order_trimmed_mean",
    "Leave-One-Growth-Group-Out",
    "MAE_nm",
    "--allow-seen-pilot-evaluation",
    "base_seed: 20260723",
    "frame-level random train/test splitting",
    "true_rq_nm_median_second_order",
]


def protocol_hash(path: Path) -> str:
    return sha256_file(path)


def validate_protocol_text(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    missing = [phrase for phrase in REQUIRED_PROTOCOL_PHRASES if phrase not in text]
    if missing:
        return [f"missing protocol phrase: {phrase}" for phrase in missing]
    return []

