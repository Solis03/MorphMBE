"""Build and validate the benchmark v1 sample registry."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .constants import (
    AFM_ONLY_UNMATCHED_ID,
    HISTORICAL_INDEX,
    HISTORICAL_INDEX_FULL,
    HISTORICAL_REMOVELIST,
    HISTORICAL_SAMPLE_IDS,
    HISTORICAL_TARGETS,
    METADATA_INVENTORY_COLUMNS,
    PROSPECTIVE_JOIN,
    PROSPECTIVE_KEYFRAMES,
    PROSPECTIVE_MISMATCH,
    PROSPECTIVE_PENDING_TRUTH_ID,
    PROSPECTIVE_PILOT_IDS,
    PROSPECTIVE_PREDICTIONS,
    PROSPECTIVE_SCAN_MANIFEST,
    PROSPECTIVE_TRUTH,
    REGISTRY_COLUMNS,
    RETROSPECTIVE_FREEZE,
    PROSPECTIVE_FREEZE,
    UNMATCHED_AND_PENDING_IDS,
)
from .hashing import combined_file_hash, file_metadata, read_csv_rows, sha256_file


def source_files() -> list[Path]:
    return [
        RETROSPECTIVE_FREEZE / "README.md",
        RETROSPECTIVE_FREEZE / "data_snapshot/canonical_sample_index.csv",
        RETROSPECTIVE_FREEZE / "data_snapshot/canonical_sample_index_full.csv",
        RETROSPECTIVE_FREEZE / "data_snapshot/sample_targets.csv",
        RETROSPECTIVE_FREEZE / "data_snapshot/removelist.txt",
        RETROSPECTIVE_FREEZE / "provenance/GIT_INFO.json",
        RETROSPECTIVE_FREEZE / "provenance/PROVENANCE.json",
        PROSPECTIVE_FREEZE / "README.md",
        PROSPECTIVE_FREEZE / "ground_truth_afm/README.md",
        PROSPECTIVE_FREEZE / "manifests/unseen_keyframe_manifest.csv",
        PROSPECTIVE_FREEZE / "predictions/full_cohort_single_frame_v1/predictions.csv",
        PROSPECTIVE_FREEZE / "ground_truth_afm/manifests/afm_extra_five_sample_level_ground_truth.csv",
        PROSPECTIVE_FREEZE / "ground_truth_afm/manifests/afm_extra_five_second_order_scan_manifest.csv",
        PROSPECTIVE_FREEZE / "ground_truth_afm/manifests/full_cohort_prediction_vs_afm_truth_join.csv",
        PROSPECTIVE_FREEZE / "ground_truth_afm/manifests/sample_id_mismatch_report.json",
    ]


def build_master_registry(root: Path) -> list[dict[str, Any]]:
    index_rows = _by_id(read_csv_rows(root / HISTORICAL_INDEX), "sample_id")
    target_rows = _by_id(read_csv_rows(root / HISTORICAL_TARGETS), "sample_id")
    target_row_numbers = _row_numbers(read_csv_rows(root / HISTORICAL_TARGETS), "sample_id")
    keyframe_rows = _by_id(read_csv_rows(root / PROSPECTIVE_KEYFRAMES), "sample_id")
    truth_rows = _by_id(read_csv_rows(root / PROSPECTIVE_TRUTH), "sample_id")
    truth_row_numbers = _row_numbers(read_csv_rows(root / PROSPECTIVE_TRUTH), "sample_id")
    prediction_rows = _by_id(read_csv_rows(root / PROSPECTIVE_PREDICTIONS), "sample_id")

    historical_hash = combined_file_hash(
        [HISTORICAL_INDEX, HISTORICAL_TARGETS, HISTORICAL_REMOVELIST],
        root,
    )
    prospective_rheed_hash = combined_file_hash([PROSPECTIVE_KEYFRAMES, PROSPECTIVE_PREDICTIONS], root)
    prospective_truth_hash = combined_file_hash([PROSPECTIVE_TRUTH, PROSPECTIVE_JOIN, PROSPECTIVE_MISMATCH], root)

    rows: list[dict[str, Any]] = []
    for sample_id in HISTORICAL_SAMPLE_IDS:
        index = index_rows[sample_id]
        target = target_rows[sample_id]
        rows.append(
            _registry_row(
                sample_id=sample_id,
                growth_group_id=index["growth_run_id"],
                cohort_role="historical_development",
                paired_status="paired",
                has_rheed=True,
                has_afm=True,
                has_target=True,
                target_rq_nm=target["T4_second_order_trimmed_mean"],
                target_definition="T4_second_order_trimmed_mean",
                target_source_path=HISTORICAL_TARGETS.as_posix(),
                target_source_row=target_row_numbers[sample_id],
                rheed_source_path=(
                    RETROSPECTIVE_FREEZE
                    / f"data_snapshot/selected_rheed_keyframes/{sample_id}_keyframe_1_raw_luminance.npz"
                ).as_posix(),
                afm_source_path=index["second_order_representative_afm_path"],
                afm_scan_count=target["scan_count"],
                historical_or_prospective="historical",
                truth_visibility="retrospective_known_before_phase0",
                eligible_for_model_development=True,
                eligible_for_primary_nested_cv=True,
                eligible_for_pilot_evaluation=False,
                eligible_for_confirmatory_test=False,
                eligible_for_final_training=True,
                exclusion_reason="",
                sample_id_status="canonical_historical_strict_growth_group",
                source_manifest_hash=historical_hash,
                notes="Historical target read from frozen target table; AFM scans remain grouped by sample.",
            )
        )

    for sample_id in PROSPECTIVE_PILOT_IDS:
        truth = truth_rows[sample_id]
        keyframe = keyframe_rows[sample_id]
        rows.append(
            _registry_row(
                sample_id=sample_id,
                growth_group_id=sample_id,
                cohort_role="prospective_pilot_seen",
                paired_status="paired",
                has_rheed=True,
                has_afm=True,
                has_target=True,
                target_rq_nm=truth["true_rq_nm_median_second_order"],
                target_definition="true_rq_nm_median_second_order_not_T4",
                target_source_path=PROSPECTIVE_TRUTH.as_posix(),
                target_source_row=truth_row_numbers[sample_id],
                rheed_source_path=prediction_rows[sample_id]["model_ready_keyframe_npz"]
                or keyframe["model_ready_keyframe_relpath"],
                afm_source_path=truth["representative_local_copy_npy"] or truth["representative_second_order_height_path"],
                afm_scan_count=truth["afm_scan_count"],
                historical_or_prospective="prospective",
                truth_visibility="seen_before_benchmark_v1_model_selection",
                eligible_for_model_development=False,
                eligible_for_primary_nested_cv=False,
                eligible_for_pilot_evaluation=True,
                eligible_for_confirmatory_test=False,
                eligible_for_final_training=False,
                exclusion_reason="seen_prospective_truth_excluded_from_model_selection",
                sample_id_status="prediction_and_afm_truth_available",
                source_manifest_hash=combined_file_hash(
                    [PROSPECTIVE_KEYFRAMES, PROSPECTIVE_PREDICTIONS, PROSPECTIVE_TRUTH],
                    root,
                ),
                notes=(
                    "Pilot truth is frozen median second-order Rq, not the historical "
                    "T4 trimmed-mean target; do not use for model choice."
                ),
            )
        )

    pending_keyframe = keyframe_rows[PROSPECTIVE_PENDING_TRUTH_ID]
    pending_prediction = prediction_rows[PROSPECTIVE_PENDING_TRUTH_ID]
    rows.append(
        _registry_row(
            sample_id=PROSPECTIVE_PENDING_TRUTH_ID,
            growth_group_id=PROSPECTIVE_PENDING_TRUTH_ID,
            cohort_role="prospective_pending_truth",
            paired_status="rheed_only_pending_afm",
            has_rheed=True,
            has_afm=False,
            has_target=False,
            target_rq_nm="",
            target_definition="pending_afm_truth",
            target_source_path="",
            target_source_row="",
            rheed_source_path=pending_prediction["model_ready_keyframe_npz"]
            or pending_keyframe["model_ready_keyframe_relpath"],
            afm_source_path="",
            afm_scan_count="0",
            historical_or_prospective="prospective",
            truth_visibility="pending_truth_not_available",
            eligible_for_model_development=False,
            eligible_for_primary_nested_cv=False,
            eligible_for_pilot_evaluation=False,
            eligible_for_confirmatory_test=False,
            eligible_for_final_training=False,
            exclusion_reason="pending_afm_truth",
            sample_id_status="prediction_without_afm_truth",
            source_manifest_hash=prospective_rheed_hash,
            notes="Retained as prospective RHEED-only pending truth; excluded from all metrics.",
        )
    )

    afm_truth = truth_rows[AFM_ONLY_UNMATCHED_ID]
    rows.append(
        _registry_row(
            sample_id=AFM_ONLY_UNMATCHED_ID,
            growth_group_id=AFM_ONLY_UNMATCHED_ID,
            cohort_role="afm_only_unmatched",
            paired_status="afm_only_unmatched",
            has_rheed=False,
            has_afm=True,
            has_target=True,
            target_rq_nm=afm_truth["true_rq_nm_median_second_order"],
            target_definition="true_rq_nm_median_second_order_unmatched_not_T4",
            target_source_path=PROSPECTIVE_TRUTH.as_posix(),
            target_source_row=truth_row_numbers[AFM_ONLY_UNMATCHED_ID],
            rheed_source_path="",
            afm_source_path=afm_truth["representative_local_copy_npy"] or afm_truth["representative_second_order_height_path"],
            afm_scan_count=afm_truth["afm_scan_count"],
            historical_or_prospective="prospective",
            truth_visibility="afm_truth_without_rheed_prediction",
            eligible_for_model_development=False,
            eligible_for_primary_nested_cv=False,
            eligible_for_pilot_evaluation=False,
            eligible_for_confirmatory_test=False,
            eligible_for_final_training=False,
            exclusion_reason="afm_truth_without_rheed_prediction",
            sample_id_status="afm_truth_without_prediction",
            source_manifest_hash=prospective_truth_hash,
            notes="Retained as AFM-only unmatched sample; excluded from supervised metrics.",
        )
    )
    return rows


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["cohort_role"] in {"historical_development", "prospective_pilot_seen"}]


def historical_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["cohort_role"] == "historical_development"]


def prospective_pilot_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["cohort_role"] == "prospective_pilot_seen"]


def unmatched_pending_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["sample_id"] in set(UNMATCHED_AND_PENDING_IDS)]


def cohort_summary(rows: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    counts = Counter(row["cohort_role"] for row in rows)
    historical = historical_rows(rows)
    targets = [float(row["target_rq_nm"]) for row in historical]
    q33 = quantile(targets, 1 / 3)
    q66 = quantile(targets, 2 / 3)
    return {
        "protocol_version": "benchmark_v1",
        "master_registry_rows": len(rows),
        "paired_supervised_rows": len(paired_rows(rows)),
        "historical_development_rows": len(historical),
        "prospective_pilot_seen_rows": counts["prospective_pilot_seen"],
        "prospective_pending_truth_rows": counts["prospective_pending_truth"],
        "afm_only_unmatched_rows": counts["afm_only_unmatched"],
        "historical_sample_ids": [row["sample_id"] for row in historical],
        "prospective_pilot_seen_ids": [row["sample_id"] for row in prospective_pilot_rows(rows)],
        "unmatched_and_pending_ids": UNMATCHED_AND_PENDING_IDS,
        "primary_target": "T4_second_order_trimmed_mean",
        "primary_target_unit": "nm",
        "prospective_truth_compatibility": {
            "historical_definition": "T4_second_order_trimmed_mean",
            "prospective_available_definition": "true_rq_nm_median_second_order",
            "directly_comparable": False,
            "action": "pilot truth retained but not used for model selection or primary confirmatory claims",
        },
        "fixed_historical_strata_quantiles": {
            "low_max_nm": q33,
            "mid_max_nm": q66,
            "definition": "low <= q33, q33 < mid <= q66, high > q66 on historical targets only",
        },
        "source_files": [path.as_posix() for path in source_files()],
        "large_source_files": large_source_inventory(root),
        "canonical_removelist_path": HISTORICAL_REMOVELIST.as_posix(),
        "canonical_removelist_sha256": sha256_file(root / HISTORICAL_REMOVELIST),
    }


def large_source_inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    full_index = read_csv_rows(root / HISTORICAL_INDEX_FULL)
    for row in full_index:
        if row["sample_id"] not in HISTORICAL_SAMPLE_IDS:
            continue
        source = Path(row["source_video"])
        rows.append(file_metadata(source, root, existing_checksum=""))
    for path in sorted((root / PROSPECTIVE_FREEZE / "metadata/samples").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = payload.get("source_video") or {}
        rel = source.get("repo_relative_path")
        if not rel:
            continue
        rows.append(file_metadata(Path(rel), root, existing_checksum=str(source.get("sha256", ""))))
    scan_rows = read_csv_rows(root / PROSPECTIVE_SCAN_MANIFEST)
    for row in scan_rows:
        rows.append(file_metadata(Path(row["raw_afm_file"]), root, existing_checksum=row.get("source_sha256", "")))
    return rows


def metadata_inventory(root: Path) -> list[dict[str, Any]]:
    datasets = [
        ("canonical_sample_index_full", HISTORICAL_INDEX_FULL, read_csv_rows(root / HISTORICAL_INDEX_FULL)),
        ("historical_sample_targets", HISTORICAL_TARGETS, read_csv_rows(root / HISTORICAL_TARGETS)),
        ("prospective_keyframe_manifest", PROSPECTIVE_KEYFRAMES, read_csv_rows(root / PROSPECTIVE_KEYFRAMES)),
        ("prospective_predictions", PROSPECTIVE_PREDICTIONS, read_csv_rows(root / PROSPECTIVE_PREDICTIONS)),
        ("prospective_afm_truth", PROSPECTIVE_TRUTH, read_csv_rows(root / PROSPECTIVE_TRUTH)),
        ("prospective_afm_scan_manifest", PROSPECTIVE_SCAN_MANIFEST, read_csv_rows(root / PROSPECTIVE_SCAN_MANIFEST)),
    ]
    out: list[dict[str, Any]] = []
    for dataset_name, path, rows in datasets:
        if not rows:
            continue
        for column in rows[0]:
            values = [row.get(column, "") for row in rows]
            role, leakage, notes = classify_metadata_field(dataset_name, column)
            missing = sum(1 for value in values if value in {"", "nan", "NaN", "None", "null"})
            coverage = len(values) - missing
            out.append(
                {
                    "feature_name": f"{dataset_name}.{column}",
                    "source_path": path.as_posix(),
                    "source_column": column,
                    "dtype": infer_dtype(values),
                    "unit": infer_unit(column),
                    "sample_coverage_count": coverage,
                    "missing_count": missing,
                    "missing_fraction": f"{(missing / len(values)):.6f}",
                    "constant_or_variable": "constant" if len({v for v in values if v != ''}) <= 1 else "variable",
                    "available_before_growth": role in {"identifier", "context"} and not _is_path_or_hash(column),
                    "available_during_growth": role in {"context", "in_situ_observation", "action"},
                    "available_after_growth": True,
                    "controllable_by_operator": False,
                    "control_role": role,
                    "leakage_risk": leakage,
                    "allowed_metadata_only_baseline": False,
                    "allowed_target_control_model": False,
                    "notes": notes,
                }
            )
    return out


def classify_metadata_field(dataset_name: str, column: str) -> tuple[str, str, str]:
    lower = f"{dataset_name}.{column}".lower()
    if column in {"sample_id", "sample_id_numeric", "growth_run_id", "growth_group_id"}:
        return "identifier", "medium", "Identifier only; not a predictive metadata feature."
    if _is_path_or_hash(column):
        return "identifier", "medium", "Path, checksum, or provenance field; not a biological or growth feature."
    if "prediction" in lower or "predicted" in lower or "error" in lower:
        return "post_growth_outcome", "high", "Model-output or error field; forbidden as input."
    if "afm" in lower or "target" in lower or "rq" in lower or "ra_" in lower or "height" in lower:
        return "post_growth_outcome", "high", "AFM, target, or post-growth morphology descriptor; forbidden as input."
    if any(token in lower for token in ["roi", "keyframe", "frame", "fps", "codec", "video", "timestamp", "duration"]):
        if column in {"source_video", "source_video_filename", "filename"}:
            return (
                "unknown",
                "medium",
                "Unstructured filename may contain growth hints; do not use until curated by domain experts.",
            )
        return "in_situ_observation", "low", "RHEED acquisition or selection provenance; not a control variable."
    if any(token in lower for token in ["operator", "hostname", "git", "schema", "status", "notes"]):
        return "context", "medium", "Operational provenance; not a predictive feature."
    return "unknown", "unknown", "Ambiguous field; not enabled for model inputs in benchmark v1."


def infer_dtype(values: list[str]) -> str:
    clean = [v for v in values if v not in {"", "nan", "NaN", "None", "null"}]
    if not clean:
        return "empty"
    if all(v in {"True", "False", "true", "false"} for v in clean):
        return "bool"
    try:
        for value in clean:
            int(value)
        return "int"
    except ValueError:
        pass
    try:
        for value in clean:
            float(value)
        return "float"
    except ValueError:
        pass
    if all((v.startswith("[") and v.endswith("]")) or (v.startswith("{") and v.endswith("}")) for v in clean):
        return "json_string"
    return "string"


def infer_unit(column: str) -> str:
    lower = column.lower()
    if lower.endswith("_nm") or "_nm_" in lower or lower.endswith("_rq_nm"):
        return "nm"
    if lower.endswith("_um") or "_um_" in lower:
        return "um"
    if lower.endswith("_sec") or "duration" in lower or "timestamp" in lower:
        return "s"
    if lower == "fps":
        return "frames/s"
    if "frame_index" in lower or "frame_count" in lower or lower.endswith("_frame"):
        return "frames"
    if "width" in lower or "height" in lower or "resolution" in lower:
        return "pixels"
    if lower.endswith("_bytes"):
        return "bytes"
    if lower.endswith("_ns"):
        return "ns"
    return ""


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute quantile of empty values")
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _registry_row(**kwargs: Any) -> dict[str, Any]:
    return {key: kwargs.get(key, "") for key in REGISTRY_COLUMNS}


def _by_id(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


def _row_numbers(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    return {row[key]: idx for idx, row in enumerate(rows, start=1)}


def _is_path_or_hash(column: str) -> bool:
    lower = column.lower()
    return (
        "path" in lower
        or "sha" in lower
        or "hash" in lower
        or lower.endswith("_id")
        or "manifest" in lower
        or lower in {"notes"}
    )

