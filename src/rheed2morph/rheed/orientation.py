"""Target-blind frame-orientation transforms for RHEED acquisition data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


VALID_CLOCKWISE_ROTATIONS = (0, 90, 180, 270)


def normalize_clockwise_rotation(degrees: int | float | str) -> int:
    """Return a validated clockwise quarter-turn angle."""

    value = int(degrees) % 360
    if value not in VALID_CLOCKWISE_ROTATIONS:
        raise ValueError(
            "RHEED frame rotation must be one of "
            f"{VALID_CLOCKWISE_ROTATIONS}, got {degrees!r}"
        )
    return value


def rotate_frame_clockwise(
    frame: np.ndarray,
    degrees: int | float | str = 0,
) -> np.ndarray:
    """Rotate an image clockwise without modifying its source array."""

    angle = normalize_clockwise_rotation(degrees)
    values = np.asarray(frame)
    if angle == 0:
        return np.ascontiguousarray(values)
    return np.ascontiguousarray(np.rot90(values, k=-(angle // 90)))


def value_for_sample(
    mapping: Mapping[str, Any] | None,
    sample_id: str,
) -> Any | None:
    """Resolve numeric and N-prefixed sample IDs against one mapping."""

    if not mapping:
        return None
    sample = str(sample_id)
    candidates = [sample]
    if sample.startswith("N") and sample[1:].isdigit():
        candidates.append(sample[1:])
    elif sample.isdigit():
        candidates.append(f"N{sample}")
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    return None


def rotation_for_sample(
    mapping: Mapping[str, int | float | str] | None,
    sample_id: str,
) -> int:
    """Resolve and validate a sample-specific clockwise rotation."""

    value = value_for_sample(mapping, sample_id)
    return 0 if value is None else normalize_clockwise_rotation(value)
