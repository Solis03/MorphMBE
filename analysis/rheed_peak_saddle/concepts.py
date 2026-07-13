"""Shared concept names and cache-key helpers for peak-saddle adhesion."""

from __future__ import annotations

import hashlib
import json
from typing import Any


FEATURE_SPEC_VERSION = "peak_saddle_adhesion_v0_stage0_scaffold"

PRIMARY_CONCEPTS = (
    "adhesion_median",
    "adhesion_q25",
    "adhesion_q75",
    "adhesion_iqr",
    "strongly_connected_pair_fraction",
    "strongly_isolated_pair_fraction",
    "isolation_persistence_median",
    "spot_width_over_spacing",
    "row_consistency",
    "valid_pair_count",
)

MEASUREMENT_QUALITY_FIELDS = (
    "spot_detection_quality",
    "pair_measurement_quality",
    "background_fit_quality",
    "saturation_fraction",
    "valid_pair_fraction",
    "nuisance_perturbation_stability",
)


def stable_json_hash(payload: dict[str, Any]) -> str:
    """Return a deterministic SHA256 hash for a JSON-compatible payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_key(*, removelist_hash: str, feature_spec: dict[str, Any] | None = None) -> str:
    """Return the cache namespace for outputs that depend on exclusions and feature logic."""
    payload = {
        "removelist_sha256": removelist_hash,
        "feature_spec": feature_spec or {"version": FEATURE_SPEC_VERSION},
    }
    return stable_json_hash(payload)

