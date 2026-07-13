"""Stage 2B1 development-only real-domain diagnostics and review package."""

from __future__ import annotations

import csv
import html
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from analysis.rheed_peak_saddle.merge_tree import maximum_bottleneck_saddle
from analysis.rheed_peak_saddle.pair_features import PairFeature, PairMasks, measure_pair_adhesion, pair_masks
from analysis.rheed_peak_saddle.preprocessing import RheedChannels, make_channels
from analysis.rheed_peak_saddle.real_diagnostics import (
    EXPECTED_REMOVELIST_SHA256,
    EXPECTED_STAGE_REVIEW_SHA256,
    file_sha256,
    load_grayscale_image,
    make_stage2a_paths,
    prune_lattice_duplicate_spots,
    read_csv_rows,
    write_csv,
)
from analysis.rheed_peak_saddle.row_grouping import (
    AdjacentPairCandidate,
    LatticeRowResult,
    RowGroupingResult,
    assign_lattice_indices,
    form_lattice_adjacent_pairs,
    group_spot_rows,
)
from analysis.rheed_peak_saddle.spot_detection import SpotEstimate, detect_spots


STAGE2B1_COMMAND = "PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_peak_saddle.real_development"
TEST_COMMAND = "PYTHONPATH=src:. .venv/bin/python -m unittest tests.test_rheed_peak_saddle"
DEVELOPMENT_SPLIT = "development_review"
EXPECTED_COMPLETED_ANNOTATION_HASHES = {
    "all_sample_qc_completed.csv": "511db34e025340bd4647a565da3d4af25fa65a26100e8dc83374bd4d56b6ce4f",
    "development_sample_review_completed.csv": "bc441b771cb086ac74fe663b1e80bacf8b5ae42e9057e8a90f90f78219e4fb5a",
    "development_pair_review_completed.csv": "662fae9e5de5421ee22e072ebe38e730433fae5b392d6783732031d5d11671fd",
}
EXPECTED_ROW_COUNTS = {
    "all_sample_qc_completed.csv": 25,
    "development_sample_review_completed.csv": 10,
    "development_pair_review_completed.csv": 23,
}
PAIR_LABELS = {"isolated", "partial", "connected", "unusable"}
DIAGNOSTIC_VALUES = {"pass", "partial", "fail", "uncertain"}
YES_NO_VALUES = {"yes", "no", "uncertain"}
SAMPLE_RATINGS = {"1", "2", "3", "4", "5"}
QUALITY_VALUES = {"low", "medium", "high", "uncertain"}


@dataclass(frozen=True)
class SplitBoundary:
    development_ids: frozenset[str]
    blind_ids: frozenset[str]
    reserve_ids: frozenset[str]
    all_ids: frozenset[str]
    split_rows: tuple[dict[str, str], ...]
    receipt: dict[str, Any]


@dataclass(frozen=True)
class DevelopmentImage:
    anonymous_review_id: str
    image: np.ndarray
    image_sha256: str


@dataclass(frozen=True)
class VariantMeasurement:
    anonymous_review_id: str
    variant_id: str
    variant_name: str
    image: np.ndarray
    display_image: np.ndarray
    channels: RheedChannels
    spots: tuple[SpotEstimate, ...]
    grouping: RowGroupingResult
    lattice: LatticeRowResult | None
    pairs: tuple[AdjacentPairCandidate, ...]
    features: dict[str, PairFeature]
    masks: dict[str, PairMasks]
    local_fit_residuals: dict[str, float]
    invalid_reasons: dict[str, str]


@dataclass(frozen=True)
class SupplementalCandidate:
    anonymous_review_id: str
    anonymous_pair_id: str
    spot_i: int
    spot_j: int
    row_label: int
    center_i_x: float
    center_i_y: float
    center_j_x: float
    center_j_y: float
    spacing: float
    spacing_over_width: float
    proposal_stratum: str
    endpoints_distinct_local_maxima: int
    pair: AdjacentPairCandidate
    spots: tuple[SpotEstimate, ...]
    masks: PairMasks
    corrected_crop_source: np.ndarray


def read_config(path: Path) -> dict[str, Any]:
    """Read the repo's JSON-compatible YAML config without importing target loaders."""
    return json.loads(path.read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stat_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _norm01(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _short_reason(text: str, *, limit: int = 120) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def require_development_only(ids: Iterable[str], boundary: SplitBoundary, context: str) -> None:
    observed = set(ids)
    outside = sorted(observed - set(boundary.development_ids))
    if outside:
        raise RuntimeError(f"{context} received non-development anonymous IDs: {outside[:3]}")


def load_split_boundary(paths: Any) -> SplitBoundary:
    split_path = paths.outputs_dir / "real_diagnostics" / "split_manifest.csv"
    receipt_path = paths.outputs_dir / "real_diagnostics" / "real_review_split_receipt.json"
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    split_rows = tuple(read_csv_rows(split_path))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    by_split: dict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        by_split[row["split"]].add(row["anonymous_review_id"])
    receipt_members = receipt.get("split_membership", [])
    receipt_pairs = {(row.get("anonymous_review_id"), row.get("split")) for row in receipt_members}
    manifest_pairs = {(row["anonymous_review_id"], row["split"]) for row in split_rows}
    if receipt_pairs != manifest_pairs:
        raise RuntimeError("Split receipt membership does not match split_manifest.csv.")
    return SplitBoundary(
        development_ids=frozenset(by_split[DEVELOPMENT_SPLIT]),
        blind_ids=frozenset(by_split["blind_validation"]),
        reserve_ids=frozenset(by_split["reserve"]),
        all_ids=frozenset(row["anonymous_review_id"] for row in split_rows),
        split_rows=split_rows,
        receipt=receipt,
    )


def annotation_paths(paths: Any) -> dict[str, Path]:
    review = paths.annotations_dir / "real_review"
    return {
        "all_sample_qc_completed.csv": review / "all_sample_qc_completed.csv",
        "development_sample_review_completed.csv": review / "development_sample_review_completed.csv",
        "development_pair_review_completed.csv": review / "development_pair_review_completed.csv",
        "completed_annotation_hashes.sha256": review / "completed_annotation_hashes.sha256",
    }


def parse_completed_hash_file(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, rel = line.split(maxsplit=1)
        hashes[Path(rel).name] = digest
    return hashes


def _validation_row(check_id: str, scope: str, status: bool, observed: Any, expected: Any = "", details: str = "") -> dict[str, Any]:
    return {
        "check_id": check_id,
        "scope": scope,
        "status": "pass" if status else "fail",
        "observed": observed,
        "expected": expected,
        "details": details,
    }


def _fieldnames(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def validate_annotations(paths: Any, boundary: SplitBoundary) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    files = annotation_paths(paths)
    hash_file = files["completed_annotation_hashes.sha256"]
    recorded_hashes = parse_completed_hash_file(hash_file)
    validation: list[dict[str, Any]] = []
    rows_by_name: dict[str, list[dict[str, str]]] = {}
    file_records: dict[str, dict[str, Any]] = {}
    for name, path in files.items():
        exists = path.is_file()
        digest = file_sha256(path) if exists else ""
        record = {
            "path": path.as_posix(),
            "sha256": digest,
            "mtime_utc": _stat_mtime(path) if exists else "",
            "size_bytes": path.stat().st_size if exists else 0,
        }
        if name in EXPECTED_COMPLETED_ANNOTATION_HASHES:
            record["expected_sha256"] = EXPECTED_COMPLETED_ANNOTATION_HASHES[name]
            record["hash_matches_expected"] = digest == EXPECTED_COMPLETED_ANNOTATION_HASHES[name]
            record["hash_matches_completed_hash_file"] = digest == recorded_hashes.get(name)
        file_records[name] = record
        validation.append(_validation_row("file_exists", name, exists, exists, True, path.as_posix()))
        if name in EXPECTED_COMPLETED_ANNOTATION_HASHES:
            validation.append(_validation_row("sha256_expected", name, digest == EXPECTED_COMPLETED_ANNOTATION_HASHES[name], digest, EXPECTED_COMPLETED_ANNOTATION_HASHES[name]))
            validation.append(_validation_row("sha256_recorded", name, digest == recorded_hashes.get(name), digest, recorded_hashes.get(name, "")))
        if exists and name.endswith(".csv"):
            rows_by_name[name] = read_csv_rows(path)

    all_qc = rows_by_name["all_sample_qc_completed.csv"]
    sample_rows = rows_by_name["development_sample_review_completed.csv"]
    pair_rows = rows_by_name["development_pair_review_completed.csv"]
    for name, rows in rows_by_name.items():
        expected = EXPECTED_ROW_COUNTS.get(name, "")
        validation.append(_validation_row("row_count", name, expected == "" or len(rows) == expected, len(rows), expected))
        ids = [row.get("anonymous_review_id", "") for row in rows]
        validation.append(_validation_row("anonymous_id_present", name, all(ids), len([x for x in ids if x]), len(ids)))
        if name != "development_pair_review_completed.csv":
            validation.append(_validation_row("anonymous_id_unique", name, len(ids) == len(set(ids)), len(set(ids)), len(ids)))

    all_qc_ids = {row["anonymous_review_id"] for row in all_qc}
    validation.append(_validation_row("all_qc_matches_locked_split", "all_sample_qc_completed.csv", all_qc_ids == boundary.all_ids, len(all_qc_ids), len(boundary.all_ids)))
    sample_ids = {row["anonymous_review_id"] for row in sample_rows}
    validation.append(_validation_row("development_sample_ids_match", "development_sample_review_completed.csv", sample_ids == set(boundary.development_ids), len(sample_ids), len(boundary.development_ids)))
    pair_ids = {row["anonymous_review_id"] for row in pair_rows}
    validation.append(_validation_row("development_pair_ids_subset", "development_pair_review_completed.csv", pair_ids <= set(boundary.development_ids), len(pair_ids), "subset of development"))
    validation.append(_validation_row("development_pair_unique_ids", "development_pair_review_completed.csv", len({row["anonymous_pair_id"] for row in pair_rows}) == len(pair_rows), len({row["anonymous_pair_id"] for row in pair_rows}), len(pair_rows)))

    all_qc_fields = set(_fieldnames(files["all_sample_qc_completed.csv"]))
    for anon_id in boundary.blind_ids | boundary.reserve_ids:
        row = next((item for item in all_qc if item.get("anonymous_review_id") == anon_id), None)
        validation.append(_validation_row("blind_reserve_schema_only", "all_sample_qc_completed.csv", row is not None and set(row) == all_qc_fields, "present" if row else "missing", "present with complete schema"))

    dev_qc_rows = [row for row in all_qc if row["anonymous_review_id"] in boundary.development_ids]
    require_development_only((row["anonymous_review_id"] for row in dev_qc_rows), boundary, "annotation development QC validation")
    diagnostic_columns = [
        "spot_detection",
        "row_grouping",
        "lattice_indexing",
        "missing_site_handling",
        "adjacent_pair_selection",
        "pair_corridors",
        "background_correction",
        "overall_measurement",
    ]
    yes_no_columns = ["severe_saturation", "severe_background_artifact", "severe_crop", "unusable_pattern"]
    for col in diagnostic_columns:
        bad = sorted({row[col] for row in dev_qc_rows if row[col] not in DIAGNOSTIC_VALUES})
        validation.append(_validation_row("development_value_domain", f"all_sample_qc_completed.csv:{col}", not bad, ",".join(bad), sorted(DIAGNOSTIC_VALUES)))
    for col in yes_no_columns:
        bad = sorted({row[col] for row in dev_qc_rows if row[col] not in YES_NO_VALUES})
        validation.append(_validation_row("development_value_domain", f"all_sample_qc_completed.csv:{col}", not bad, ",".join(bad), sorted(YES_NO_VALUES)))
    for col, allowed in {
        "sample_adhesion_rating": SAMPLE_RATINGS,
        "visible_spot_quality": QUALITY_VALUES,
        "rating_confidence": QUALITY_VALUES,
        "severe_artifact": YES_NO_VALUES,
    }.items():
        bad = sorted({row[col] for row in sample_rows if row[col] not in allowed})
        validation.append(_validation_row("development_value_domain", f"development_sample_review_completed.csv:{col}", not bad, ",".join(bad), sorted(allowed)))
    for col, allowed in {
        "pair_label": PAIR_LABELS,
        "spot_centers_correct": YES_NO_VALUES,
        "same_physical_row": YES_NO_VALUES,
        "true_adjacent_neighbors": YES_NO_VALUES,
        "corridor_correct": YES_NO_VALUES,
        "background_regions_clean": YES_NO_VALUES,
        "label_confidence": QUALITY_VALUES,
    }.items():
        bad = sorted({row[col] for row in pair_rows if row[col] not in allowed})
        validation.append(_validation_row("development_value_domain", f"development_pair_review_completed.csv:{col}", not bad, ",".join(bad), sorted(allowed)))

    label_counts = Counter(row["pair_label"] for row in pair_rows)
    validation.append(_validation_row("zero_isolated_detected", "development_pair_review_completed.csv", label_counts.get("isolated", 0) == 0, label_counts.get("isolated", 0), 0))
    validation.append(_validation_row("no_three_class_calibrator", "development_pair_review_completed.csv", label_counts.get("isolated", 0) == 0, "concept-label coverage insufficient", "do not fit ordinal calibrator"))
    receipt = {
        "timestamp_utc": _now(),
        "annotation_files": file_records,
        "completed_hash_file": {
            "path": hash_file.as_posix(),
            "sha256": file_sha256(hash_file),
            "recorded_completed_hashes": recorded_hashes,
        },
        "row_counts": {name: len(rows) for name, rows in rows_by_name.items()},
        "split_counts": {
            DEVELOPMENT_SPLIT: len(boundary.development_ids),
            "blind_validation": len(boundary.blind_ids),
            "reserve": len(boundary.reserve_ids),
        },
        "development_pair_label_distribution": dict(label_counts),
        "isolated_label_count": label_counts.get("isolated", 0),
        "concept_label_coverage": "insufficient",
        "blind_reserve_label_firewall": "schema/presence only; values not summarized or passed to tuning functions",
        "validation_passed": all(row["status"] == "pass" for row in validation),
    }
    return receipt, validation, rows_by_name


def write_annotation_receipt(out_dir: Path, receipt: dict[str, Any], validation: Sequence[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "annotation_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(out_dir / "annotation_validation.csv", validation, fieldnames=["check_id", "scope", "status", "observed", "expected", "details"])


def development_qc_summary(
    rows_by_name: dict[str, list[dict[str, str]]],
    boundary: SplitBoundary,
) -> list[dict[str, Any]]:
    all_qc = [row for row in rows_by_name["all_sample_qc_completed.csv"] if row["anonymous_review_id"] in boundary.development_ids]
    require_development_only((row["anonymous_review_id"] for row in all_qc), boundary, "development QC summary")
    sample_rows = rows_by_name["development_sample_review_completed.csv"]
    pair_rows = rows_by_name["development_pair_review_completed.csv"]
    summary: list[dict[str, Any]] = []
    for table_name, rows, skip in (
        ("all_sample_qc_completed.csv", all_qc, {"anonymous_review_id", "reviewer_notes"}),
        ("development_sample_review_completed.csv", sample_rows, {"anonymous_review_id", "reviewer_notes"}),
        ("development_pair_review_completed.csv", pair_rows, {"anonymous_review_id", "anonymous_pair_id", "reviewer_notes", "unusable_reason"}),
    ):
        for field in [key for key in rows[0] if key not in skip]:
            counts = Counter(row[field] for row in rows)
            for value, count in sorted(counts.items(), key=lambda item: (str(item[0]), item[1])):
                summary.append(
                    {
                        "source_table": table_name,
                        "field": field,
                        "value": value,
                        "development_count": count,
                        "development_denominator": len(rows),
                        "development_fraction": count / max(len(rows), 1),
                    }
                )
    sample_rating_counts = Counter(row["sample_adhesion_rating"] for row in sample_rows)
    for rating in ("1", "2", "3", "4", "5"):
        summary.append(
            {
                "source_table": "development_sample_review_completed.csv",
                "field": "sample_rating_coverage",
                "value": rating,
                "development_count": sample_rating_counts.get(rating, 0),
                "development_denominator": len(sample_rows),
                "development_fraction": sample_rating_counts.get(rating, 0) / max(len(sample_rows), 1),
            }
        )
    return summary


def development_pair_label_summary(pair_rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    counts = Counter(row["pair_label"] for row in pair_rows)
    total = len(pair_rows)
    rows: list[dict[str, Any]] = []
    for label in ("isolated", "partial", "connected", "unusable"):
        rows.append(
            {
                "pair_label": label,
                "development_count": counts.get(label, 0),
                "development_denominator": total,
                "development_fraction": counts.get(label, 0) / max(total, 1),
                "concept_label_coverage_status": "insufficient" if counts.get("isolated", 0) == 0 else "complete",
                "modeling_action": "do_not_fit_pair_concept_model" if counts.get("isolated", 0) == 0 else "not_selected_in_stage_2b1",
            }
        )
    return rows


FRONTEND_FAILURE_TYPES = {
    "spot_detection": [
        "missed_spot",
        "duplicate_detection",
        "broad_spot_split",
        "false_halo_peak",
        "false_edge_peak",
        "saturation_merge",
        "low_contrast_miss",
    ],
    "row_grouping": [
        "one_spot_per_row_fragmentation",
        "different_rows_merged",
        "same_row_split",
        "row_angle_failure",
        "two_column_structure_not_modeled",
    ],
    "lattice_indexing": [
        "unsupported_long_lattice_assumption",
        "arbitrary_large_site_index",
        "duplicate_site_assignment",
        "missing_site_false_positive",
        "spacing_estimate_failure",
    ],
    "pair_proposal": [
        "invalid_endpoint",
        "nonadjacent_endpoint",
        "same_broad_spot_pair",
        "pair_missed",
        "pair_crosses_missing_site",
        "pair_created_from_duplicate_detection",
    ],
    "measurement": [
        "denominator_too_small",
        "saturated_peak",
        "invalid_pair_used",
        "adhesion_unreliable_due_to_background",
    ],
}

BACKGROUND_FAILURE_TYPES = [
    "both_offset_corridors_contaminated",
    "one_offset_corridor_contaminated",
    "global_halo_not_removed",
    "real_bridge_removed_as_background",
    "contour_or_ringing",
    "quantization_banding",
    "edge_crop_contamination",
]


def _note(row: dict[str, str]) -> str:
    return " ".join((row.get("reviewer_notes", ""), row.get("unusable_reason", ""))).lower()


def _sample_failure_set(qc_row: dict[str, str], row_rows: Sequence[dict[str, str]], spot_rows: Sequence[dict[str, str]]) -> set[str]:
    text = _note(qc_row)
    failures: set[str] = set()
    if qc_row["spot_detection"] in {"partial", "fail"}:
        failures.add("missed_spot")
    if any(token in text for token in ("duplicate", "cluster", "crowd", "overlap")):
        failures.add("duplicate_detection")
        failures.add("pair_created_from_duplicate_detection")
    if any(token in text for token in ("broad", "merged", "streak")):
        failures.add("broad_spot_split")
        failures.add("same_broad_spot_pair")
    if "halo" in text:
        failures.add("false_halo_peak")
    if "edge" in text or "crop" in text:
        failures.add("false_edge_peak")
        failures.add("edge_crop_contamination")
    if "satur" in text or qc_row.get("severe_saturation") == "yes":
        failures.add("saturation_merge")
        failures.add("saturated_peak")
    if any(token in text for token in ("low-signal", "low contrast", "weak")):
        failures.add("low_contrast_miss")
    if qc_row["row_grouping"] in {"partial", "fail"}:
        failures.add("same_row_split")
        failures.add("two_column_structure_not_modeled")
    if "merged" in text and qc_row["row_grouping"] in {"partial", "fail"}:
        failures.add("different_rows_merged")
    one_spot_rows = [row for row in row_rows if int(float(row.get("spot_count") or 0)) <= 1]
    if row_rows and len(one_spot_rows) / len(row_rows) >= 0.45:
        failures.add("one_spot_per_row_fragmentation")
        failures.add("two_column_structure_not_modeled")
    angle_values = [_finite_float(row.get("dominant_angle_degrees")) for row in row_rows]
    consistency_values = [_finite_float(row.get("row_consistency")) for row in row_rows]
    if any(math.isfinite(v) and abs(v) > 18.0 for v in angle_values) or any(math.isfinite(v) and v < 0.55 for v in consistency_values):
        failures.add("row_angle_failure")
    if qc_row["lattice_indexing"] in {"partial", "fail"}:
        failures.add("unsupported_long_lattice_assumption")
        failures.add("spacing_estimate_failure")
    if "inconsistent lattice" in text or "lattice indices" in text:
        failures.add("arbitrary_large_site_index")
    duplicate_lattice = [row for row in spot_rows if str(row.get("lattice_index", "")) and str(row.get("row_id", ""))]
    if len(duplicate_lattice) != len({(row.get("row_id"), row.get("lattice_index")) for row in duplicate_lattice}):
        failures.add("duplicate_site_assignment")
    if qc_row["missing_site_handling"] in {"partial", "fail", "uncertain"}:
        failures.add("missing_site_false_positive")
        failures.add("pair_crosses_missing_site")
    if qc_row["adjacent_pair_selection"] in {"partial", "fail"}:
        failures.add("pair_missed")
        failures.add("invalid_endpoint")
    if any(token in text for token in ("not a distinct", "not represented", "no trustworthy", "not physically consistent")):
        failures.add("invalid_endpoint")
    if qc_row["overall_measurement"] in {"partial", "fail"}:
        failures.add("invalid_pair_used")
        failures.add("adhesion_unreliable_due_to_background")
    if "denominator" in text:
        failures.add("denominator_too_small")
    return failures


def _background_failure_set(qc_row: dict[str, str], pair_rows: Sequence[dict[str, str]]) -> set[str]:
    text = _note(qc_row)
    failures: set[str] = set()
    if pair_rows and all(row.get("background_regions_clean") == "no" for row in pair_rows):
        failures.add("both_offset_corridors_contaminated")
    elif any(row.get("background_regions_clean") == "no" for row in pair_rows):
        failures.add("one_offset_corridor_contaminated")
    if qc_row.get("background_correction") in {"partial", "fail"} or "halo" in text:
        failures.add("global_halo_not_removed")
    if any("bridge" in _note(row) and row.get("background_regions_clean") == "no" for row in pair_rows):
        failures.add("real_bridge_removed_as_background")
    if any(token in text for token in ("contour", "ringing", "washed out", "washout")):
        failures.add("contour_or_ringing")
    if "banding" in text or "quantization" in text:
        failures.add("quantization_banding")
    if qc_row.get("severe_crop") == "yes" or "edge" in text or "crop" in text:
        failures.add("edge_crop_contamination")
    return failures


def failure_taxonomies(
    rows_by_name: dict[str, list[dict[str, str]]],
    boundary: SplitBoundary,
    stage2a_tables: dict[str, list[dict[str, str]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, set[str]]]:
    dev_qc = [row for row in rows_by_name["all_sample_qc_completed.csv"] if row["anonymous_review_id"] in boundary.development_ids]
    pair_review_by_anon: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows_by_name["development_pair_review_completed.csv"]:
        pair_review_by_anon[row["anonymous_review_id"]].append(row)
    rows_by_anon: dict[str, list[dict[str, str]]] = defaultdict(list)
    spots_by_anon: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in stage2a_tables["row_level_measurements.csv"]:
        if row["anonymous_review_id"] in boundary.development_ids:
            rows_by_anon[row["anonymous_review_id"]].append(row)
    for row in stage2a_tables["spot_level_measurements.csv"]:
        if row["anonymous_review_id"] in boundary.development_ids:
            spots_by_anon[row["anonymous_review_id"]].append(row)
    sample_failures: dict[str, set[str]] = {}
    background_failures_by_sample: dict[str, set[str]] = {}
    for row in dev_qc:
        anon = row["anonymous_review_id"]
        sample_failures[anon] = _sample_failure_set(row, rows_by_anon.get(anon, []), spots_by_anon.get(anon, []))
        background_failures_by_sample[anon] = _background_failure_set(row, pair_review_by_anon.get(anon, []))
    frontend_rows: list[dict[str, Any]] = []
    for category, names in FRONTEND_FAILURE_TYPES.items():
        for failure_type in names:
            examples = sorted(anon for anon, failures in sample_failures.items() if failure_type in failures)
            frontend_rows.append(
                {
                    "failure_category": category,
                    "failure_type": failure_type,
                    "development_sample_count": len(examples),
                    "development_denominator": len(dev_qc),
                    "development_fraction": len(examples) / max(len(dev_qc), 1),
                    "example_anonymous_review_ids": ";".join(examples[:5]),
                    "source": "development_annotations_and_frozen_stage2a_shadow_tables",
                }
            )
    background_rows: list[dict[str, Any]] = []
    for failure_type in BACKGROUND_FAILURE_TYPES:
        examples = sorted(anon for anon, failures in background_failures_by_sample.items() if failure_type in failures)
        background_rows.append(
            {
                "failure_type": failure_type,
                "development_sample_count": len(examples),
                "development_denominator": len(dev_qc),
                "development_fraction": len(examples) / max(len(dev_qc), 1),
                "example_anonymous_review_ids": ";".join(examples[:5]),
                "source": "development_annotations_only_for_labels; frozen_tables_for_geometry_context",
            }
        )
    pair_validity_rows: list[dict[str, Any]] = []
    isolated_count = sum(1 for row in rows_by_name["development_pair_review_completed.csv"] if row["pair_label"] == "isolated")
    for row in rows_by_name["development_pair_review_completed.csv"]:
        reasons: list[str] = []
        if row["pair_label"] == "unusable":
            reasons.append("human_marked_unusable")
        if row["spot_centers_correct"] != "yes":
            reasons.append("spot_centers_not_confirmed")
        if row["same_physical_row"] != "yes":
            reasons.append("same_physical_row_not_confirmed")
        if row["true_adjacent_neighbors"] != "yes":
            reasons.append("adjacency_not_confirmed")
        if row["background_regions_clean"] != "yes":
            reasons.append("background_regions_not_clean")
        pair_validity_rows.append(
            {
                "anonymous_review_id": row["anonymous_review_id"],
                "anonymous_pair_id": row["anonymous_pair_id"],
                "pair_label": row["pair_label"],
                "valid_for_pair_concept_calibration": 0,
                "valid_for_visual_failure_analysis": int(row["pair_label"] != "unusable"),
                "invalidity_reasons": ";".join(reasons),
                "background_regions_clean": row["background_regions_clean"],
                "spot_centers_correct": row["spot_centers_correct"],
                "same_physical_row": row["same_physical_row"],
                "true_adjacent_neighbors": row["true_adjacent_neighbors"],
                "corridor_correct": row["corridor_correct"],
                "label_confidence": row["label_confidence"],
                "concept_label_coverage_status": "insufficient_zero_isolated" if isolated_count == 0 else "complete",
            }
        )
    return frontend_rows, background_rows, pair_validity_rows, sample_failures


def write_failure_contact_sheets(
    sample_failures: dict[str, set[str]],
    report_dir: Path,
    stage2a_report_dir: Path,
) -> list[dict[str, Any]]:
    out_dir = report_dir / "failure_contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()
    records: list[dict[str, Any]] = []
    major = sorted(
        ((failure, sorted(anon for anon, failures in sample_failures.items() if failure in failures)) for failure in {f for failures in sample_failures.values() for f in failures}),
        key=lambda item: (-len(item[1]), item[0]),
    )
    for failure, anon_ids in major:
        if len(anon_ids) < 3:
            continue
        thumbs: list[Image.Image] = []
        for anon in anon_ids[:6]:
            src = stage2a_report_dir / "all_sample_qc_contact_sheets" / f"{anon}_qc.png"
            if not src.is_file():
                continue
            with Image.open(src) as image:
                img = image.convert("RGB")
                img.thumbnail((520, 330))
                tile = Image.new("RGB", (540, 370), "white")
                tile.paste(img, ((540 - img.width) // 2, 22))
                draw = ImageDraw.Draw(tile)
                draw.text((10, 6), anon, fill=(0, 0, 0))
                thumbs.append(tile)
        if not thumbs:
            continue
        cols = 2
        rows = math.ceil(len(thumbs) / cols)
        canvas = Image.new("RGB", (cols * 540, rows * 370 + 34), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), failure, fill=(0, 0, 0))
        for idx, thumb in enumerate(thumbs):
            x = (idx % cols) * 540
            y = 34 + (idx // cols) * 370
            canvas.paste(thumb, (x, y))
        out = out_dir / f"{failure}.png"
        canvas.save(out)
        records.append({"failure_type": failure, "path": out.as_posix(), "development_sample_count": len(anon_ids)})
    return records


def _reindex_spots(spots: Sequence[SpotEstimate]) -> tuple[SpotEstimate, ...]:
    ordered = sorted(spots, key=lambda spot: (spot.center_y, spot.center_x))
    return tuple(replace(spot, spot_id=index) for index, spot in enumerate(ordered))


def strong_one_to_one_nms(spots: Sequence[SpotEstimate], *, base_distance: float = 18.0) -> tuple[SpotEstimate, ...]:
    ordered = sorted(spots, key=lambda spot: (spot.detection_confidence, spot.peak_intensity), reverse=True)
    kept: list[SpotEstimate] = []
    for spot in ordered:
        keep = True
        for other in kept:
            distance = math.hypot(spot.center_x - other.center_x, spot.center_y - other.center_y)
            min_dist = max(base_distance, 1.85 * max(spot.equivalent_width, other.equivalent_width))
            if distance < min_dist:
                keep = False
                break
        if keep:
            kept.append(spot)
    return _reindex_spots(kept)


def form_y_level_pairs(
    spots: Sequence[SpotEstimate],
    grouping: RowGroupingResult,
    *,
    image_id: str,
) -> tuple[AdjacentPairCandidate, ...]:
    labels = np.asarray(grouping.row_labels, dtype=int)
    rx = np.asarray(grouping.rotated_x, dtype=float)
    pairs: list[AdjacentPairCandidate] = []
    nearest_spacings: list[float] = []
    rows: list[tuple[int, list[int]]] = []
    for row_label in sorted(set(labels.tolist())):
        indices = [int(index) for index in np.where(labels == row_label)[0]]
        indices.sort(key=lambda index: rx[index])
        rows.append((int(row_label), indices))
        nearest_spacings.extend(float(abs(rx[right] - rx[left])) for left, right in zip(indices[:-1], indices[1:]))
    median_spacing = float(np.median(nearest_spacings)) if nearest_spacings else float("nan")
    for row_label, indices in rows:
        for left, right in zip(indices[:-1], indices[1:]):
            spacing = float(abs(rx[right] - rx[left]))
            width = max(float((spots[left].equivalent_width + spots[right].equivalent_width) / 2.0), 1e-6)
            spacing_over_width = spacing / width
            if spacing < max(10.0, 2.05 * width):
                continue
            if not (1.8 <= spacing_over_width <= 14.0):
                continue
            if math.isfinite(median_spacing) and spacing > max(120.0, 2.15 * median_spacing):
                continue
            spacing_score = math.exp(-abs(spacing - median_spacing) / max(median_spacing, 1e-6)) if math.isfinite(median_spacing) else 0.5
            confidence = float(np.clip(0.45 + 0.40 * spacing_score + 0.15 * min(spots[left].detection_confidence, spots[right].detection_confidence), 0.0, 1.0))
            pairs.append(
                AdjacentPairCandidate(
                    pair_id=f"{image_id}_yp{len(pairs):03d}",
                    spot_i=left,
                    spot_j=right,
                    row_label=row_label,
                    spacing=spacing,
                    spacing_over_width=spacing_over_width,
                    pair_selection_confidence=confidence,
                )
            )
    return tuple(pairs)


def measure_variant0(anon_id: str, image: np.ndarray) -> VariantMeasurement:
    channels = make_channels(image)
    spots = prune_lattice_duplicate_spots(detect_spots(channels.linear, min_distance=14.0))
    grouping = group_spot_rows(spots)
    lattice = assign_lattice_indices(spots, grouping)
    pairs = form_lattice_adjacent_pairs(spots, grouping, lattice, image_id=f"{anon_id}_v0")
    features: dict[str, PairFeature] = {}
    masks: dict[str, PairMasks] = {}
    for pair in pairs:
        feature, mask = measure_pair_adhesion(channels.linear, spots, pair, image_id=anon_id, ridge_channel=channels.horizontal_ridge)
        features[pair.pair_id] = feature
        masks[pair.pair_id] = mask
    return VariantMeasurement(anon_id, "variant0", "Frozen synthetic pipeline", image, channels.linear, channels, spots, grouping, lattice, pairs, features, masks, {}, {pid: feat.invalid_reason for pid, feat in features.items() if feat.invalid_reason})


def measure_variant1(anon_id: str, image: np.ndarray) -> VariantMeasurement:
    channels = make_channels(image, background_sigma=24.0)
    raw_spots = detect_spots(channels.linear, max_spots=80, min_distance=10.0, max_equivalent_width=16.0)
    spots = strong_one_to_one_nms(raw_spots, base_distance=18.0)
    grouping = group_spot_rows(spots)
    pairs = form_y_level_pairs(spots, grouping, image_id=f"{anon_id}_v1")
    features: dict[str, PairFeature] = {}
    masks: dict[str, PairMasks] = {}
    for pair in pairs:
        feature, mask = measure_pair_adhesion(channels.linear, spots, pair, image_id=anon_id, ridge_channel=channels.horizontal_ridge)
        features[pair.pair_id] = feature
        masks[pair.pair_id] = mask
    return VariantMeasurement(anon_id, "variant1", "Two-column y-level proposal", image, channels.linear, channels, spots, grouping, None, pairs, features, masks, {}, {pid: feat.invalid_reason for pid, feat in features.items() if feat.invalid_reason})


def _robust_peak(image: np.ndarray, mask: np.ndarray) -> float:
    values = image[mask]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, 98.0))


def _local_plane_model(image: np.ndarray, masks: PairMasks, left: SpotEstimate, right: SpotEstimate) -> tuple[np.ndarray, np.ndarray, float]:
    height, width = image.shape
    pad = int(max(22.0, 0.45 * math.hypot(right.center_x - left.center_x, right.center_y - left.center_y)))
    x0 = max(0, int(min(left.center_x, right.center_x) - pad))
    x1 = min(width, int(max(left.center_x, right.center_x) + pad) + 1)
    y0 = max(0, int(min(left.center_y, right.center_y) - pad))
    y1 = min(height, int(max(left.center_y, right.center_y) + pad) + 1)
    crop_mask = np.zeros_like(image, dtype=bool)
    crop_mask[y0:y1, x0:x1] = True
    candidate = crop_mask & ~masks.corridor_mask & ~masks.seed_i_mask & ~masks.seed_j_mask
    values = image[candidate]
    if values.size >= 30:
        cutoff = float(np.percentile(values, 78.0))
        candidate &= image <= cutoff
    if int(candidate.sum()) < 20:
        candidate = masks.background_mask.copy()
    yy, xx = np.indices(image.shape, dtype=float)
    if int(candidate.sum()) < 3:
        plane = np.full_like(image, float(np.percentile(image, 10.0)), dtype=float)
        return plane, candidate, 1.0
    x_center = (left.center_x + right.center_x) / 2.0
    y_center = (left.center_y + right.center_y) / 2.0
    scale = max(float(max(image.shape)), 1.0)
    design = np.column_stack([np.ones(int(candidate.sum())), (xx[candidate] - x_center) / scale, (yy[candidate] - y_center) / scale])
    coeff, *_ = np.linalg.lstsq(design, image[candidate], rcond=None)
    plane = coeff[0] + coeff[1] * ((xx - x_center) / scale) + coeff[2] * ((yy - y_center) / scale)
    residual = float(np.median(np.abs(image[candidate] - plane[candidate])) / max(float(np.percentile(image[candidate], 95.0) - np.percentile(image[candidate], 5.0)), 1e-6))
    return plane, candidate, residual


def measure_pair_local_plane(
    image: np.ndarray,
    spots: Sequence[SpotEstimate],
    pair: AdjacentPairCandidate,
    *,
    image_id: str,
) -> tuple[PairFeature, PairMasks, float, np.ndarray]:
    values = np.asarray(image, dtype=float)
    left = spots[pair.spot_i]
    right = spots[pair.spot_j]
    invalid_reasons: list[str] = []
    if left.edge_or_crop_flag or right.edge_or_crop_flag:
        invalid_reasons.append("edge_or_crop_spot")
    if pair.spacing_over_width < 1.8:
        invalid_reasons.append("spacing_too_small")
    if pair.spacing_over_width > 14.0:
        invalid_reasons.append("spacing_too_large")
    masks0 = pair_masks(values.shape, left, right)
    if int(masks0.seed_i_mask.sum()) < 2 or int(masks0.seed_j_mask.sum()) < 2:
        invalid_reasons.append("missing_seed_pixels")
    if int(masks0.bridge_body_mask.sum()) < 4:
        invalid_reasons.append("empty_bridge_corridor")
    plane, background_mask, residual = _local_plane_model(values, masks0, left, right)
    corrected = values - plane
    masks = PairMasks(masks0.corridor_mask, masks0.bridge_body_mask, masks0.seed_i_mask, masks0.seed_j_mask, background_mask)
    peak_i = _robust_peak(corrected, masks.seed_i_mask)
    peak_j = _robust_peak(corrected, masks.seed_j_mask)
    saddle = maximum_bottleneck_saddle(corrected, masks.seed_i_mask, masks.seed_j_mask, masks.corridor_mask)
    if not saddle.connected:
        invalid_reasons.append("corridor_not_connected")
    denominator = min(peak_i, peak_j)
    if not math.isfinite(denominator) or denominator <= 1e-6:
        invalid_reasons.append("nonpositive_peak_denominator")
    if residual > 0.55:
        invalid_reasons.append("local_background_fit_unstable")
    if invalid_reasons:
        adhesion_unclipped = float("nan")
        adhesion = float("nan")
        isolation = float("nan")
        direct_ratio = float("nan")
        mean_ratio = float("nan")
        ridge_ratio = float("nan")
        width_ratio = float("nan")
        confidence = 0.0
        valid = 0
    else:
        adhesion_unclipped = float(saddle.saddle_intensity / (denominator + 1e-6))
        adhesion = float(np.clip(adhesion_unclipped, 0.0, 1.0))
        isolation = float(1.0 - adhesion)
        bridge = corrected[masks.bridge_body_mask]
        direct_ratio = float(np.percentile(bridge, 10.0) / (denominator + 1e-6)) if bridge.size else float("nan")
        mean_ratio = float(np.mean(bridge) / (denominator + 1e-6)) if bridge.size else float("nan")
        ridge_ratio = float("nan")
        width_ratio = float(np.mean(bridge >= 0.5 * max(saddle.saddle_intensity, 0.0))) if bridge.size else float("nan")
        confidence = float(np.clip(pair.pair_selection_confidence * min(left.detection_confidence, right.detection_confidence) * (1.0 - min(residual, 0.75)), 0.0, 1.0))
        valid = 1
    feature = PairFeature(
        image_id=image_id,
        pair_id=pair.pair_id,
        spot_i=pair.spot_i,
        spot_j=pair.spot_j,
        row_label=pair.row_label,
        peak_i=float(peak_i),
        peak_j=float(peak_j),
        saddle_intensity=float(saddle.saddle_intensity),
        background_intensity=float(np.median(plane[masks.corridor_mask])) if np.any(masks.corridor_mask) else float("nan"),
        background_method="local_plane_pair_crop",
        raw_peak_saddle_adhesion=float(adhesion),
        raw_peak_saddle_adhesion_unclipped=float(adhesion_unclipped),
        isolation_persistence=float(isolation),
        direct_corridor_valley_ratio=float(direct_ratio),
        corridor_mean_ratio=float(mean_ratio),
        ridge_energy_ratio=float(ridge_ratio),
        bridge_width_ratio=float(width_ratio),
        spot_spacing_over_width=float(pair.spacing_over_width),
        pair_measurement_confidence=float(confidence),
        valid=valid,
        invalid_reason=";".join(invalid_reasons),
    )
    return feature, masks, residual, corrected


def measure_variant2(anon_id: str, image: np.ndarray) -> VariantMeasurement:
    base = measure_variant1(anon_id, image)
    features: dict[str, PairFeature] = {}
    masks: dict[str, PairMasks] = {}
    residuals: dict[str, float] = {}
    corrected_images: list[np.ndarray] = []
    for pair in base.pairs:
        feature, mask, residual, corrected = measure_pair_local_plane(base.channels.linear, base.spots, pair, image_id=anon_id)
        features[pair.pair_id] = feature
        masks[pair.pair_id] = mask
        residuals[pair.pair_id] = residual
        corrected_images.append(corrected)
    display = corrected_images[0] if corrected_images else base.channels.linear - base.channels.background
    return VariantMeasurement(anon_id, "variant2", "Pair-local background model", image, display, base.channels, base.spots, base.grouping, None, base.pairs, features, masks, residuals, {pid: feat.invalid_reason for pid, feat in features.items() if feat.invalid_reason})


def candidate_adapter_specifications() -> dict[str, Any]:
    return {
        "stage": "2B1",
        "selection_status": "no_final_adapter_selected",
        "development_only": True,
        "sample_specific_parameters": False,
        "peak_saddle_definition_changed": False,
        "variants": [
            {
                "variant_id": "variant0",
                "name": "Frozen synthetic pipeline",
                "role": "baseline",
                "allowed_changes": [],
                "description": "Existing detector, duplicate pruning, row grouping, lattice adjacent-pair proposal, symmetric offset background corridors, and unchanged peak-saddle measurement.",
            },
            {
                "variant_id": "variant1",
                "name": "Two-column y-level pair proposal",
                "role": "candidate",
                "allowed_changes": ["spot proposal", "duplicate suppression", "row grouping", "pair proposal", "invalid-pair rejection"],
                "description": "Local maxima on linear grayscale, stronger one-to-one nonmaximum suppression, y-level grouping, nearest plausible left/right neighbors, and no long periodic lattice requirement.",
            },
            {
                "variant_id": "variant2",
                "name": "Pair-local background model",
                "role": "candidate",
                "allowed_changes": ["pair-local background estimation", "invalid-pair rejection"],
                "description": "Variant 1 endpoint/pair proposals measured with a local low-order background plane inside an expanded pair crop, then the same peak-saddle adhesion definition on the locally corrected continuous image.",
            },
        ],
        "forbidden_actions": [
            "blind_validation_evaluation",
            "reserve_label_use",
            "pair_concept_model_fit",
            "ordinal_calibrator_fit",
            "roughness_model_training",
        ],
    }


def load_development_images(paths: Any, boundary: SplitBoundary) -> list[DevelopmentImage]:
    manifest = read_csv_rows(paths.outputs_dir / "real_diagnostics" / "real_input_manifest.csv")
    rows = [row for row in manifest if row["anonymous_review_id"] in boundary.development_ids and row["input_status"] == "included"]
    require_development_only((row["anonymous_review_id"] for row in rows), boundary, "development image load")
    images: list[DevelopmentImage] = []
    for row in sorted(rows, key=lambda item: item["anonymous_review_id"]):
        path = paths.repo_root / row["manual_rheed_path"]
        images.append(DevelopmentImage(row["anonymous_review_id"], load_grayscale_image(path), file_sha256(path)))
    return images


def measure_candidate_variants(images: Sequence[DevelopmentImage], boundary: SplitBoundary) -> dict[str, list[VariantMeasurement]]:
    require_development_only((image.anonymous_review_id for image in images), boundary, "candidate adapter measurement")
    outputs: dict[str, list[VariantMeasurement]] = {}
    for image in images:
        outputs[image.anonymous_review_id] = [
            measure_variant0(image.anonymous_review_id, image.image),
            measure_variant1(image.anonymous_review_id, image.image),
            measure_variant2(image.anonymous_review_id, image.image),
        ]
    return outputs


def adapter_output_rows(variant_outputs: dict[str, list[VariantMeasurement]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for anon_id, variants in sorted(variant_outputs.items()):
        for variant in variants:
            valid_count = sum(1 for feature in variant.features.values() if feature.valid)
            reasons = Counter(reason for reason in variant.invalid_reasons.values() if reason)
            row_count = len(set(variant.grouping.row_labels)) if variant.grouping.row_labels else 0
            rows.append(
                {
                    "anonymous_review_id": anon_id,
                    "variant_id": variant.variant_id,
                    "variant_name": variant.variant_name,
                    "spot_count": len(variant.spots),
                    "row_or_level_count": row_count,
                    "pair_count": len(variant.pairs),
                    "valid_pair_count": valid_count,
                    "invalid_pair_count": len(variant.pairs) - valid_count,
                    "invalid_pair_reasons": ";".join(f"{key}:{value}" for key, value in sorted(reasons.items())),
                    "background_methods": ";".join(sorted({feature.background_method for feature in variant.features.values()})),
                    "development_only": 1,
                    "sample_specific_parameters": 0,
                    "final_adapter_selected": 0,
                }
            )
    return rows


def _overlay_variant(ax: Any, variant: VariantMeasurement) -> None:
    ax.imshow(_norm01(variant.display_image), cmap="gray", origin="upper")
    labels = variant.grouping.row_labels
    for index, spot in enumerate(variant.spots):
        row_label = labels[index] if index < len(labels) else -1
        color = "deepskyblue" if not spot.edge_or_crop_flag else "magenta"
        ax.scatter([spot.center_x], [spot.center_y], s=22, facecolors="none", edgecolors=color, linewidths=0.8)
        ax.text(spot.center_x + 1.5, spot.center_y - 1.5, f"y{row_label}", color="yellow", fontsize=5)
    for pair in variant.pairs:
        left = variant.spots[pair.spot_i]
        right = variant.spots[pair.spot_j]
        feature = variant.features.get(pair.pair_id)
        color = "lime" if feature and feature.valid else "orange"
        ax.plot([left.center_x, right.center_x], [left.center_y, right.center_y], color=color, linewidth=0.7, alpha=0.8)
        masks = variant.masks.get(pair.pair_id)
        if masks is not None:
            ax.contour(masks.corridor_mask, levels=[0.5], colors=["orange"], linewidths=0.35, alpha=0.5)
            ax.contour(masks.background_mask, levels=[0.5], colors=["cyan"], linewidths=0.28, alpha=0.45)
    reasons = sorted({reason for reason in variant.invalid_reasons.values() if reason})
    if reasons:
        ax.text(0.01, 0.02, _short_reason("; ".join(reasons), limit=90), transform=ax.transAxes, color="white", fontsize=5, va="bottom", bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5})
    ax.set_title(variant.variant_id, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def write_adapter_comparison(variant_outputs: dict[str, list[VariantMeasurement]], report_dir: Path) -> list[dict[str, str]]:
    import matplotlib.pyplot as plt

    panel_dir = report_dir / "adapter_comparison_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for old in panel_dir.glob("*.png"):
        old.unlink()
    entries: list[dict[str, str]] = []
    for anon_id, variants in sorted(variant_outputs.items()):
        fig, axes = plt.subplots(1, len(variants), figsize=(12, 4.0))
        if len(variants) == 1:
            axes = [axes]
        for ax, variant in zip(axes, variants):
            _overlay_variant(ax, variant)
        fig.suptitle(anon_id, fontsize=10)
        fig.tight_layout()
        name = f"{anon_id}_adapter_comparison.png"
        fig.savefig(panel_dir / name, dpi=150)
        plt.close(fig)
        entries.append({"anonymous_review_id": anon_id, "panel": f"adapter_comparison_panels/{name}"})
    html_parts = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Development Adapter Comparison</title>",
        "<h1>Development Adapter Comparison</h1>",
        "<p>Use the CSV template to review spots, rows or y-levels, pairs, and background regions for each variant.</p>",
    ]
    for entry in entries:
        html_parts.append(f"<section><h2>{html.escape(entry['anonymous_review_id'])}</h2>")
        html_parts.append(f"<img src='{html.escape(entry['panel'])}' style='max-width:100%; border:1px solid #bbb'>")
        html_parts.append("</section>")
    (report_dir / "development_adapter_comparison.html").write_text("\n".join(html_parts), encoding="utf-8")
    return entries


def write_adapter_review_template(paths: Any, boundary: SplitBoundary) -> None:
    fields = [
        "anonymous_review_id",
        "preferred_variant",
        "variant0_spots",
        "variant0_rows",
        "variant0_pairs",
        "variant0_background",
        "variant1_spots",
        "variant1_rows",
        "variant1_pairs",
        "variant1_background",
        "variant2_spots",
        "variant2_rows",
        "variant2_pairs",
        "variant2_background",
        "reviewer_confidence",
        "reviewer_notes",
    ]
    rows = []
    for anon_id in sorted(boundary.development_ids):
        rows.append({field: (anon_id if field == "anonymous_review_id" else "") for field in fields})
    write_csv(paths.annotations_dir / "real_review" / "development_adapter_comparison_template.csv", rows, fields)


def _is_local_max(image: np.ndarray, spot: SpotEstimate, *, size: int = 5) -> bool:
    y = int(round(spot.center_y))
    x = int(round(spot.center_x))
    if y < 0 or y >= image.shape[0] or x < 0 or x >= image.shape[1]:
        return False
    local = ndimage.maximum_filter(image, size=size, mode="nearest")
    return bool(image[y, x] >= local[y, x] - 1e-9)


def _candidate_valley_ratio(image: np.ndarray, spots: Sequence[SpotEstimate], pair: AdjacentPairCandidate) -> tuple[float, PairMasks, np.ndarray]:
    left = spots[pair.spot_i]
    right = spots[pair.spot_j]
    masks = pair_masks(image.shape, left, right)
    bg_values = image[masks.background_mask]
    if bg_values.size < 8:
        bg = float(np.percentile(image, 10.0))
    else:
        bg = float(np.median(bg_values))
    peak_i = _robust_peak(image, masks.seed_i_mask) - bg
    peak_j = _robust_peak(image, masks.seed_j_mask) - bg
    denom = max(min(peak_i, peak_j), 1e-6)
    bridge = image[masks.bridge_body_mask] - bg
    ratio = float(np.percentile(bridge, 10.0) / denom) if bridge.size else float("nan")
    feature, _, _, corrected = measure_pair_local_plane(image, spots, pair, image_id="supplemental")
    _ = feature
    return ratio, masks, corrected


def supplemental_candidate_pool(images: Sequence[DevelopmentImage], boundary: SplitBoundary) -> list[SupplementalCandidate]:
    require_development_only((image.anonymous_review_id for image in images), boundary, "supplemental pool")
    raw_records: list[dict[str, Any]] = []
    for image in images:
        variant = measure_variant1(image.anonymous_review_id, image.image)
        for pair in variant.pairs:
            left = variant.spots[pair.spot_i]
            right = variant.spots[pair.spot_j]
            if pair.spot_i == pair.spot_j:
                continue
            if not (_is_local_max(variant.channels.linear, left) and _is_local_max(variant.channels.linear, right)):
                continue
            if math.hypot(left.center_x - right.center_x, left.center_y - right.center_y) < max(10.0, 2.05 * max(left.equivalent_width, right.equivalent_width)):
                continue
            ratio, masks, corrected = _candidate_valley_ratio(variant.channels.linear, variant.spots, pair)
            if not math.isfinite(ratio):
                continue
            raw_records.append({"image": image, "variant": variant, "pair": pair, "ratio": ratio, "masks": masks, "corrected": corrected})
    if not raw_records:
        return []
    ratios = np.asarray([record["ratio"] for record in raw_records], dtype=float)
    q1, q2 = np.quantile(ratios, [1 / 3, 2 / 3])
    buckets: dict[str, list[dict[str, Any]]] = {"low": [], "medium": [], "high": []}
    for record in raw_records:
        ratio = record["ratio"]
        stratum = "low" if ratio <= q1 else "medium" if ratio <= q2 else "high"
        record["stratum"] = stratum
        buckets[stratum].append(record)
    for bucket in buckets.values():
        bucket.sort(key=lambda record: (record["image"].anonymous_review_id, abs(record["ratio"] - float(np.median(ratios)))))
    selected: list[dict[str, Any]] = []
    per_image: Counter[str] = Counter()
    while len(selected) < 30:
        progressed = False
        for stratum in ("low", "medium", "high"):
            for idx, record in enumerate(list(buckets[stratum])):
                anon = record["image"].anonymous_review_id
                if per_image[anon] >= 4:
                    continue
                selected.append(record)
                per_image[anon] += 1
                buckets[stratum].pop(idx)
                progressed = True
                break
            if len(selected) >= 30:
                break
        if not progressed:
            break
    if len(selected) < 20:
        remaining = [record for bucket in buckets.values() for record in bucket]
        remaining.sort(key=lambda record: (per_image[record["image"].anonymous_review_id], record["image"].anonymous_review_id))
        for record in remaining:
            if len(selected) >= 35:
                break
            selected.append(record)
            per_image[record["image"].anonymous_review_id] += 1
    candidates: list[SupplementalCandidate] = []
    for index, record in enumerate(selected, start=1):
        variant: VariantMeasurement = record["variant"]
        pair: AdjacentPairCandidate = record["pair"]
        left = variant.spots[pair.spot_i]
        right = variant.spots[pair.spot_j]
        candidates.append(
            SupplementalCandidate(
                anonymous_review_id=record["image"].anonymous_review_id,
                anonymous_pair_id=f"R2P_{index:03d}",
                spot_i=pair.spot_i,
                spot_j=pair.spot_j,
                row_label=pair.row_label,
                center_i_x=left.center_x,
                center_i_y=left.center_y,
                center_j_x=right.center_x,
                center_j_y=right.center_y,
                spacing=pair.spacing,
                spacing_over_width=pair.spacing_over_width,
                proposal_stratum=record["stratum"],
                endpoints_distinct_local_maxima=1,
                pair=AdjacentPairCandidate(
                    pair_id=f"R2P_{index:03d}",
                    spot_i=pair.spot_i,
                    spot_j=pair.spot_j,
                    row_label=pair.row_label,
                    spacing=pair.spacing,
                    spacing_over_width=pair.spacing_over_width,
                    pair_selection_confidence=pair.pair_selection_confidence,
                ),
                spots=variant.spots,
                masks=record["masks"],
                corrected_crop_source=record["corrected"],
            )
        )
    return candidates


def supplemental_manifest_rows(candidates: Sequence[SupplementalCandidate]) -> list[dict[str, Any]]:
    return [
        {
            "anonymous_review_id": candidate.anonymous_review_id,
            "anonymous_pair_id": candidate.anonymous_pair_id,
            "proposal_source": "development_only_linear_local_maxima_y_level_pool",
            "proposal_stratum": candidate.proposal_stratum,
            "spot_i_id": candidate.spot_i,
            "spot_j_id": candidate.spot_j,
            "row_or_y_level": candidate.row_label,
            "endpoint_i_x": f"{candidate.center_i_x:.2f}",
            "endpoint_i_y": f"{candidate.center_i_y:.2f}",
            "endpoint_j_x": f"{candidate.center_j_x:.2f}",
            "endpoint_j_y": f"{candidate.center_j_y:.2f}",
            "endpoint_spacing_px": f"{candidate.spacing:.2f}",
            "spacing_over_width": f"{candidate.spacing_over_width:.3f}",
            "endpoints_distinct_local_maxima": candidate.endpoints_distinct_local_maxima,
            "uses_frozen_lattice_pair_requirement": 0,
            "review_values_hidden": "proposal_stratum and geometry metrics are not included in reviewer template or HTML card text",
        }
        for candidate in candidates
    ]


def _crop_bounds(image: np.ndarray, candidate: SupplementalCandidate, *, pad: int = 28) -> tuple[int, int, int, int]:
    x0 = max(0, int(min(candidate.center_i_x, candidate.center_j_x) - pad))
    x1 = min(image.shape[1], int(max(candidate.center_i_x, candidate.center_j_x) + pad) + 1)
    y0 = max(0, int(min(candidate.center_i_y, candidate.center_j_y) - pad))
    y1 = min(image.shape[0], int(max(candidate.center_i_y, candidate.center_j_y) + pad) + 1)
    return x0, x1, y0, y1


def write_supplemental_pair_cards(
    images: Sequence[DevelopmentImage],
    candidates: Sequence[SupplementalCandidate],
    report_dir: Path,
) -> list[dict[str, str]]:
    import matplotlib.pyplot as plt

    image_by_id = {image.anonymous_review_id: image.image for image in images}
    card_dir = report_dir / "round2_pair_cards"
    card_dir.mkdir(parents=True, exist_ok=True)
    for old in card_dir.glob("*.png"):
        old.unlink()
    entries: list[dict[str, str]] = []
    for candidate in candidates:
        image = image_by_id[candidate.anonymous_review_id]
        x0, x1, y0, y1 = _crop_bounds(image, candidate)
        panels = [
            ("original crop", image),
            ("linear grayscale crop", image),
            ("local corrected crop", candidate.corrected_crop_source),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8))
        for ax, (title, source) in zip(axes, panels):
            ax.imshow(_norm01(source[y0:y1, x0:x1]), cmap="gray", origin="upper")
            ax.scatter(
                [candidate.center_i_x - x0, candidate.center_j_x - x0],
                [candidate.center_i_y - y0, candidate.center_j_y - y0],
                s=32,
                facecolors="none",
                edgecolors="yellow",
                linewidths=1.0,
            )
            ax.contour(candidate.masks.corridor_mask[y0:y1, x0:x1], levels=[0.5], colors=["orange"], linewidths=0.8)
            ax.contour(candidate.masks.background_mask[y0:y1, x0:x1], levels=[0.5], colors=["cyan"], linewidths=0.55)
            ax.set_title(title, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{candidate.anonymous_review_id} / {candidate.anonymous_pair_id}", fontsize=9)
        fig.tight_layout()
        name = f"{candidate.anonymous_pair_id}.png"
        fig.savefig(card_dir / name, dpi=150)
        plt.close(fig)
        entries.append({"anonymous_review_id": candidate.anonymous_review_id, "anonymous_pair_id": candidate.anonymous_pair_id, "card": f"round2_pair_cards/{name}"})
    parts = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Development Pair Review Round 2</title>",
        "<h1>Development Pair Review Round 2</h1>",
        "<p>Review each anonymous pair crop and complete the CSV template.</p>",
    ]
    for entry in entries:
        parts.append(f"<section><h2>{html.escape(entry['anonymous_review_id'])} / {html.escape(entry['anonymous_pair_id'])}</h2>")
        parts.append(f"<img src='{html.escape(entry['card'])}' style='max-width:100%; border:1px solid #bbb'>")
        parts.append("</section>")
    (report_dir / "development_pair_review_round2.html").write_text("\n".join(parts), encoding="utf-8")
    return entries


def write_round2_pair_template(paths: Any, candidates: Sequence[SupplementalCandidate]) -> None:
    fields = [
        "anonymous_review_id",
        "anonymous_pair_id",
        "pair_label",
        "spot_centers_correct",
        "same_physical_row",
        "true_adjacent_neighbors",
        "corridor_correct",
        "background_regions_clean",
        "label_confidence",
        "unusable_reason",
        "reviewer_notes",
    ]
    rows = []
    for candidate in candidates:
        rows.append({field: candidate.anonymous_review_id if field == "anonymous_review_id" else candidate.anonymous_pair_id if field == "anonymous_pair_id" else "" for field in fields})
    write_csv(paths.annotations_dir / "real_review" / "development_pair_review_round2_template.csv", rows, fields)


def write_data_access_audit(
    path: Path,
    paths: Any,
    boundary: SplitBoundary,
    images: Sequence[DevelopmentImage],
    receipt: dict[str, Any],
) -> None:
    opened: list[dict[str, Any]] = []
    for role, source in [
        ("split_manifest", paths.outputs_dir / "real_diagnostics" / "split_manifest.csv"),
        ("split_receipt", paths.outputs_dir / "real_diagnostics" / "real_review_split_receipt.json"),
        ("real_input_manifest", paths.outputs_dir / "real_diagnostics" / "real_input_manifest.csv"),
        ("spot_level_measurements", paths.outputs_dir / "real_diagnostics" / "spot_level_measurements.csv"),
        ("row_level_measurements", paths.outputs_dir / "real_diagnostics" / "row_level_measurements.csv"),
        ("lattice_level_measurements", paths.outputs_dir / "real_diagnostics" / "lattice_level_measurements.csv"),
        ("pair_level_measurements", paths.outputs_dir / "real_diagnostics" / "pair_level_measurements.csv"),
        ("measurement_quality", paths.outputs_dir / "real_diagnostics" / "measurement_quality.csv"),
        ("removelist", paths.repo_root / "removelist.txt"),
        ("stage_review", paths.annotations_dir / "stage_review_completed.csv"),
    ]:
        opened.append({"role": role, "path": source.as_posix(), "sha256": file_sha256(source), "contains_target_values": 0})
    for name, record in receipt["annotation_files"].items():
        opened.append({"role": f"annotation:{name}", "path": record["path"], "sha256": record["sha256"], "contains_target_values": int(name.endswith("_completed.csv"))})
    for image in images:
        opened.append(
            {
                "role": "manual_rheed_image",
                "path": f"redacted_manual_rheed_image:{image.anonymous_review_id}",
                "sha256": image.image_sha256,
                "contains_target_values": 0,
            }
        )
    forbidden_terms = ("/afm/", "rq", "roughness", "target", "unblind_key")
    forbidden_hits = [row for row in opened if any(term in Path(row["path"]).name.lower() or term in row["path"].lower() for term in forbidden_terms)]
    payload = {
        "timestamp_utc": _now(),
        "stage": "2B1",
        "input_files_opened": opened,
        "forbidden_terms": forbidden_terms,
        "forbidden_path_hits": forbidden_hits,
        "afm_rq_source_opened": False,
        "unblind_key_opened": False,
        "blind_reserve_label_values_used": False,
        "development_ids_used_for_tuning": len(boundary.development_ids),
        "blind_ids_used_for_tuning": 0,
        "reserve_ids_used_for_tuning": 0,
        "execution_guard": "Only development anonymous IDs were passed to diagnostic-selection, adapter, and supplemental-pair functions.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_reproducibility_manifest(
    out_dir: Path,
    report_dir: Path,
    paths: Any,
    receipt: dict[str, Any],
    candidates: Sequence[SupplementalCandidate],
) -> None:
    generated = [
        out_dir / "annotation_receipt.json",
        out_dir / "annotation_validation.csv",
        out_dir / "development_qc_summary.csv",
        out_dir / "development_pair_label_summary.csv",
        out_dir / "development_frontend_failure_taxonomy.csv",
        out_dir / "development_background_failure_taxonomy.csv",
        out_dir / "development_pair_validity_analysis.csv",
        out_dir / "candidate_adapter_specifications.json",
        out_dir / "candidate_adapter_outputs.csv",
        out_dir / "supplemental_pair_manifest.csv",
        out_dir / "data_access_audit.json",
        report_dir / "development_adapter_comparison.html",
        report_dir / "development_pair_review_round2.html",
        paths.annotations_dir / "real_review" / "development_adapter_comparison_template.csv",
        paths.annotations_dir / "real_review" / "development_pair_review_round2_template.csv",
        paths.reports_dir / "checkpoint_2b1_real_development.md",
    ]
    payload = {
        "timestamp_utc": _now(),
        "stage": "2B1",
        "command": STAGE2B1_COMMAND,
        "test_command": TEST_COMMAND,
        "annotation_receipt_sha256": file_sha256(out_dir / "annotation_receipt.json"),
        "annotation_hashes": {name: record["sha256"] for name, record in receipt["annotation_files"].items()},
        "removelist_sha256": file_sha256(paths.repo_root / "removelist.txt"),
        "stage_review_sha256": file_sha256(paths.annotations_dir / "stage_review_completed.csv"),
        "supplemental_pair_count": len(candidates),
        "generated_files": {item.as_posix(): file_sha256(item) for item in generated if item.is_file()},
        "final_status": "HUMAN ROUND-2 ANNOTATION REQUIRED",
        "stopped_before": ["blind_validation", "reserve_label_use", "adapter_selection", "pair_concept_model", "roughness_model"],
    }
    (out_dir / "reproducibility_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_checkpoint_report(
    path: Path,
    *,
    receipt: dict[str, Any],
    qc_summary: Sequence[dict[str, Any]],
    pair_label_rows: Sequence[dict[str, Any]],
    frontend_rows: Sequence[dict[str, Any]],
    background_rows: Sequence[dict[str, Any]],
    adapter_rows: Sequence[dict[str, Any]],
    candidates: Sequence[SupplementalCandidate],
    contact_sheets: Sequence[dict[str, Any]],
) -> None:
    pair_counts = {row["pair_label"]: row["development_count"] for row in pair_label_rows}
    qc_highlights = [
        row
        for row in qc_summary
        if row["source_table"] == "all_sample_qc_completed.csv" and row["field"] in {"overall_measurement", "background_correction", "lattice_indexing"}
    ]
    top_frontend = sorted(frontend_rows, key=lambda row: (-int(row["development_sample_count"]), row["failure_type"]))[:10]
    top_background = sorted(background_rows, key=lambda row: (-int(row["development_sample_count"]), row["failure_type"]))[:7]
    variant_counts = Counter(row["variant_id"] for row in adapter_rows)
    text = "\n".join(
        [
            "# Checkpoint 2B1: Real Development Diagnostics",
            "",
            "## Annotation Validation",
            f"- Validation passed: `{receipt['validation_passed']}`",
            f"- Annotation hashes: `{ {name: record['sha256'] for name, record in receipt['annotation_files'].items() if name.endswith('.csv')} }`",
            f"- Row counts: `{receipt['row_counts']}`",
            f"- Development pair labels: `{pair_counts}`",
            "- Isolated labels present: `0`",
            "- Concept-label coverage: `insufficient`",
            "",
            "## Boundary Confirmation",
            f"- Split counts: `{receipt['split_counts']}`",
            "- Tuning, candidate adapters, diagnostic selection, and supplemental sampling used only development anonymous IDs.",
            "- Blind and reserve QC rows were checked only for presence and complete schema.",
            "- Blind/reserve label values were not printed, summarized, or passed to tuning functions.",
            "- `unblind_key.csv` was not opened.",
            "",
            "## Safety Hashes",
            f"- Removelist SHA256: `{EXPECTED_REMOVELIST_SHA256}`",
            "- Sample `6088` remains excluded by the canonical removelist.",
            f"- Stage-review SHA256: `{EXPECTED_STAGE_REVIEW_SHA256}`",
            "- No AFM arrays, AFM images, descriptor tables, Rq values, previous Rq predictions, Rq-sorted figures, or target tables were accessed.",
            "",
            "## Development QC Statistics",
            *[f"- `{row['field']}={row['value']}`: `{row['development_count']}/{row['development_denominator']}`" for row in qc_highlights],
            "",
            "## Real/Synthetic Domain Mismatch",
            "- Synthetic rows often contained several periodic spots, while the development real images frequently show one or two visible spots per horizontal level.",
            "- Long-row lattice-period fitting is therefore a poor structural assumption for many reviewed real images.",
            "",
            "## Front-End Failure Taxonomy",
            *[f"- `{row['failure_type']}`: `{row['development_sample_count']}/{row['development_denominator']}`" for row in top_frontend],
            "",
            "## Background Failure Taxonomy",
            *[f"- `{row['failure_type']}`: `{row['development_sample_count']}/{row['development_denominator']}`" for row in top_background],
            "",
            "## Candidate Adapters",
            "- Variant 0: frozen synthetic pipeline.",
            "- Variant 1: two-column / y-level pair proposal on linear grayscale local maxima.",
            "- Variant 2: Variant 1 proposals with pair-local low-order background modeling.",
            f"- Candidate output rows by variant: `{dict(variant_counts)}`",
            "- No adapter was selected or frozen in this run.",
            "",
            "## Human Review Files",
            "- `reports/rheed_peak_saddle/real_development/development_adapter_comparison.html`",
            "- `annotations/rheed_peak_saddle/real_review/development_adapter_comparison_template.csv`",
            "- `reports/rheed_peak_saddle/real_development/development_pair_review_round2.html`",
            "- `annotations/rheed_peak_saddle/real_review/development_pair_review_round2_template.csv`",
            f"- Supplemental round-2 pair count: `{len(candidates)}`",
            f"- Failure contact sheets generated: `{len(contact_sheets)}`",
            "",
            "## Stop Confirmation",
            "- No pair concept model was fit.",
            "- No ordinal or multinomial calibrator was fit.",
            "- No connected/partial/isolated probability was calibrated.",
            "- No pair-label accuracy was reported.",
            "- No blind validation evaluation occurred.",
            "- No reserve labels were used.",
            "- No roughness model was trained.",
            "",
            "## Status",
            "HUMAN ROUND-2 ANNOTATION REQUIRED",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def load_stage2a_tables(paths: Any, boundary: SplitBoundary) -> dict[str, list[dict[str, str]]]:
    out = paths.outputs_dir / "real_diagnostics"
    tables = {
        "spot_level_measurements.csv": read_csv_rows(out / "spot_level_measurements.csv"),
        "row_level_measurements.csv": read_csv_rows(out / "row_level_measurements.csv"),
        "lattice_level_measurements.csv": read_csv_rows(out / "lattice_level_measurements.csv"),
        "pair_level_measurements.csv": read_csv_rows(out / "pair_level_measurements.csv"),
        "measurement_quality.csv": read_csv_rows(out / "measurement_quality.csv"),
    }
    for name, rows in tables.items():
        dev_ids = [row["anonymous_review_id"] for row in rows if row["anonymous_review_id"] in boundary.development_ids]
        require_development_only(dev_ids, boundary, f"stage2a table development filter {name}")
    return tables


def run_real_development(config_path: Path = Path("configs/rheed_peak_saddle.yaml")) -> dict[str, Any]:
    config = read_config(config_path)
    paths = make_stage2a_paths(config)
    out_dir = paths.outputs_dir / "real_development"
    report_dir = paths.reports_dir / "real_development"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    boundary = load_split_boundary(paths)
    receipt, validation, rows_by_name = validate_annotations(paths, boundary)
    if file_sha256(paths.repo_root / "removelist.txt") != EXPECTED_REMOVELIST_SHA256:
        raise RuntimeError("Canonical removelist hash mismatch.")
    if file_sha256(paths.annotations_dir / "stage_review_completed.csv") != EXPECTED_STAGE_REVIEW_SHA256:
        raise RuntimeError("Completed stage-review hash mismatch.")
    write_annotation_receipt(out_dir, receipt, validation)
    stage2a_tables = load_stage2a_tables(paths, boundary)
    qc_rows = development_qc_summary(rows_by_name, boundary)
    pair_label_rows = development_pair_label_summary(rows_by_name["development_pair_review_completed.csv"])
    frontend_rows, background_rows, pair_validity_rows, sample_failures = failure_taxonomies(rows_by_name, boundary, stage2a_tables)
    write_csv(out_dir / "development_qc_summary.csv", qc_rows)
    write_csv(out_dir / "development_pair_label_summary.csv", pair_label_rows)
    write_csv(out_dir / "development_frontend_failure_taxonomy.csv", frontend_rows)
    write_csv(out_dir / "development_background_failure_taxonomy.csv", background_rows)
    write_csv(out_dir / "development_pair_validity_analysis.csv", pair_validity_rows)
    contact_sheets = write_failure_contact_sheets(sample_failures, report_dir, paths.reports_dir / "real_diagnostics")
    specs = candidate_adapter_specifications()
    (out_dir / "candidate_adapter_specifications.json").write_text(json.dumps(specs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    images = load_development_images(paths, boundary)
    variant_outputs = measure_candidate_variants(images, boundary)
    adapter_rows = adapter_output_rows(variant_outputs)
    write_csv(out_dir / "candidate_adapter_outputs.csv", adapter_rows)
    write_adapter_comparison(variant_outputs, report_dir)
    write_adapter_review_template(paths, boundary)
    candidates = supplemental_candidate_pool(images, boundary)
    write_csv(out_dir / "supplemental_pair_manifest.csv", supplemental_manifest_rows(candidates))
    write_supplemental_pair_cards(images, candidates, report_dir)
    write_round2_pair_template(paths, candidates)
    write_data_access_audit(out_dir / "data_access_audit.json", paths, boundary, images, receipt)
    write_checkpoint_report(
        paths.reports_dir / "checkpoint_2b1_real_development.md",
        receipt=receipt,
        qc_summary=qc_rows,
        pair_label_rows=pair_label_rows,
        frontend_rows=frontend_rows,
        background_rows=background_rows,
        adapter_rows=adapter_rows,
        candidates=candidates,
        contact_sheets=contact_sheets,
    )
    write_reproducibility_manifest(out_dir, report_dir, paths, receipt, candidates)
    return {
        "outputs_dir": out_dir.as_posix(),
        "reports_dir": report_dir.as_posix(),
        "development_count": len(boundary.development_ids),
        "supplemental_pair_count": len(candidates),
        "status": "HUMAN ROUND-2 ANNOTATION REQUIRED",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rheed_peak_saddle.yaml")
    args = parser.parse_args()
    result = run_real_development(Path(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
