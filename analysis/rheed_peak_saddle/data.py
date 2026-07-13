"""Stage 0 data and growth-stage audit for peak-saddle adhesion."""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from analysis.rheed_roughness.run import csv_value, display_path, resolve_path, safe_float
from analysis.rheed_roughness.visualize_manual_pairs import (
    AFMCandidate,
    ManualSelection,
    discover_manual_rheed_images,
    select_representative_afm_scan,
    valid_physical_afm,
)
from analysis.rheed_single_frame.data import load_afm_candidates_filtered
from analysis.rheed_single_frame.removelist import RemovelistAudit, assert_no_removed_samples

from analysis.rheed_peak_saddle.concepts import FEATURE_SPEC_VERSION, cache_key


DATASET_AUDIT_FIELDS = (
    "sample_id",
    "sample_group_id",
    "growth_run_id",
    "manual_rheed_path",
    "manual_rheed_filename",
    "manual_selection_status",
    "removelist_status",
    "has_representative_afm",
    "included_in_candidate_set",
    "skip_reason",
    "inferred_stage",
    "material",
    "afm_candidate_scan_count",
    "selected_afm_scan_id",
    "selected_height_map_path",
    "scan_size_um",
    "afm_resolution",
    "warnings",
    "cache_key",
)

PRELIMINARY_MANIFEST_FIELDS = (
    "sample_id",
    "sample_group_id",
    "growth_run_id",
    "manual_rheed_path",
    "manual_rheed_filename",
    "inferred_stage",
    "selected_afm_scan_id",
    "selected_height_map_path",
    "scan_size_um",
    "scan_size_x_um",
    "scan_size_y_um",
    "afm_resolution",
    "material",
    "afm_candidate_scan_count",
    "afm_selection_reason",
    "rq_status",
    "rq_source",
    "removelist_sha256",
    "visual_feature_spec_version",
    "cache_key",
)

EXCLUDED_FIELDS = (
    "sample_id",
    "source_path",
    "exclusion_reason",
    "note",
    "present_in_manual_selection",
    "manual_rheed_filename",
    "present_in_plane_corrected_afm_tree",
)

STAGE_REVIEW_FIELDS = (
    "sample_id",
    "manual_rheed_filename",
    "inferred_stage",
    "approved_stage",
    "comparable_stage_group",
    "stage_confidence",
    "user_approved",
    "user_notes",
)

ALLOWED_STAGE_CATEGORIES = (
    "oxide_or_substrate",
    "active_growth",
    "after_growth",
    "rampdown_or_cooldown",
    "rampup_or_heating",
    "unknown",
)


@dataclass(frozen=True)
class PeakSaddlePaths:
    repo_root: Path
    outputs_dir: Path
    reports_dir: Path
    annotations_dir: Path
    approvals_dir: Path
    manual_root: Path
    plane_corrected_afm_root: Path


@dataclass(frozen=True)
class Stage0Bundle:
    paths: PeakSaddlePaths
    dataset_rows: tuple[dict[str, Any], ...]
    preliminary_manifest_rows: tuple[dict[str, Any], ...]
    excluded_rows: tuple[dict[str, Any], ...]
    stage_review_rows: tuple[dict[str, Any], ...]
    missing_manual_sample_ids: tuple[str, ...]
    missing_afm_sample_ids: tuple[str, ...]
    stage_counts: dict[str, int]


def make_paths(config: dict[str, Any]) -> PeakSaddlePaths:
    repo_root = resolve_path(Path(__file__).resolve().parents[2], config.get("repo_root", ".")).resolve()
    outputs = resolve_path(repo_root, config["outputs_dir"]).resolve()
    reports = resolve_path(repo_root, config["reports_dir"]).resolve()
    annotations = resolve_path(repo_root, config.get("annotations_dir", "annotations/rheed_peak_saddle")).resolve()
    approvals = annotations / "approvals"
    manual_root = resolve_path(repo_root, config.get("manual_selection_root", "data/manual_selection")).resolve()
    plane_root = resolve_path(repo_root, config.get("data_roots", {}).get("plane_corrected_afm_root", "data/plane_corrected_afm")).resolve()
    for path in (outputs, reports, annotations, approvals):
        path.mkdir(parents=True, exist_ok=True)
    return PeakSaddlePaths(
        repo_root=repo_root,
        outputs_dir=outputs,
        reports_dir=reports,
        annotations_dir=annotations,
        approvals_dir=approvals,
        manual_root=manual_root,
        plane_corrected_afm_root=plane_root,
    )


def write_csv_rows(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def infer_growth_stage(filename: str) -> str:
    """Infer a provisional growth-stage category from a selected RHEED filename."""
    name = filename.lower().replace("_", " ").replace("-", " ")
    if any(token in name for token in ("rampdown", "ramp down", "ramping down", "ramped down", "cooldown", "cool down", "cooling")):
        return "rampdown_or_cooldown"
    if any(token in name for token in ("rampup", "ramp up", "ramping up", "heating", "outgassing")):
        return "rampup_or_heating"
    if any(token in name for token in ("oxide", "substrate", "desorption")):
        return "oxide_or_substrate"
    if any(token in name for token in ("after", "post growth", "postgrowth", "end growth", "final growth")):
        return "after_growth"
    if any(token in name for token in ("growth", "gasb", "alsb", "gdsb", "min")):
        return "active_growth"
    return "unknown"


def _afm_by_sample(candidates: Sequence[AFMCandidate]) -> dict[str, list[AFMCandidate]]:
    by_sample: dict[str, list[AFMCandidate]] = {}
    for candidate in candidates:
        by_sample.setdefault(candidate.sample_id, []).append(candidate)
    return by_sample


def _afm_tree_sample_ids(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {child.name for child in root.iterdir() if child.is_dir()}


def _selection_by_sample(selections: Sequence[ManualSelection]) -> dict[str, ManualSelection]:
    return {selection.sample_id: selection for selection in selections if selection.sample_id}


def _select_afm(
    sample_candidates: Sequence[AFMCandidate],
    config: dict[str, Any],
) -> tuple[AFMCandidate | None, float, float, str, list[AFMCandidate]]:
    valid = [candidate for candidate in sample_candidates if valid_physical_afm(candidate)]
    selected, median, distance, reason = select_representative_afm_scan(
        valid,
        primary_scan_size_um=float(config.get("afm", {}).get("primary_scan_size_um", 1.0)),
        tolerance_um=float(config.get("afm", {}).get("primary_scan_size_tolerance_um", 0.10)),
    )
    return selected, median, distance, reason, valid


def build_stage0_dataset(config: dict[str, Any], paths: PeakSaddlePaths, removelist: RemovelistAudit) -> Stage0Bundle:
    all_selections = discover_manual_rheed_images(paths.manual_root)
    selection_lookup = _selection_by_sample(all_selections)
    removed_ids = set(removelist.sample_ids)
    cache_namespace = cache_key(removelist_hash=removelist.sha256)

    afm_candidates = load_afm_candidates_filtered(paths, config, removed_ids)
    assert_no_removed_samples((candidate.sample_id for candidate in afm_candidates), removed_ids, context="Stage 0 AFM candidate loading")
    afm_lookup = _afm_by_sample(afm_candidates)
    afm_tree_ids = _afm_tree_sample_ids(paths.plane_corrected_afm_root)

    excluded_rows: list[dict[str, Any]] = []
    for record in removelist.records:
        selection = selection_lookup.get(record.sample_id)
        excluded_rows.append(
            {
                "sample_id": record.sample_id,
                "source_path": record.source_path.as_posix(),
                "exclusion_reason": "canonical_removelist",
                "note": record.note,
                "present_in_manual_selection": int(selection is not None),
                "manual_rheed_filename": selection.selected_path.name if selection and selection.selected_path else "",
                "present_in_plane_corrected_afm_tree": int(record.sample_id in afm_tree_ids),
            }
        )

    dataset_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    stage_review_rows: list[dict[str, Any]] = []
    missing_manual: list[str] = []
    missing_afm: list[str] = []

    for selection in all_selections:
        if selection.sample_id in removed_ids:
            continue
        filename = selection.selected_path.name if selection.selected_path else ""
        inferred_stage = infer_growth_stage(filename)
        selected_afm: AFMCandidate | None = None
        selected_reason = ""
        valid_candidates: list[AFMCandidate] = []
        skip_reason = ""
        if selection.status != "ok":
            skip_reason = selection.status
            missing_manual.append(selection.sample_id)
        else:
            selected_afm, _, _, selected_reason, valid_candidates = _select_afm(afm_lookup.get(selection.sample_id, ()), config)
            if selected_afm is None:
                skip_reason = selected_reason
                missing_afm.append(selection.sample_id)
        included = selection.status == "ok" and selected_afm is not None
        material = selected_afm.material if selected_afm is not None else (valid_candidates[0].material if valid_candidates else "")
        row = {
            "sample_id": selection.sample_id,
            "sample_group_id": selection.sample_id,
            "growth_run_id": selection.sample_id,
            "manual_rheed_path": display_path(selection.selected_path, paths.repo_root) if selection.selected_path else "",
            "manual_rheed_filename": filename,
            "manual_selection_status": selection.status,
            "removelist_status": "kept",
            "has_representative_afm": int(selected_afm is not None),
            "included_in_candidate_set": int(included),
            "skip_reason": skip_reason,
            "inferred_stage": inferred_stage,
            "material": material,
            "afm_candidate_scan_count": len(valid_candidates),
            "selected_afm_scan_id": selected_afm.afm_scan_id if selected_afm else "",
            "selected_height_map_path": display_path(selected_afm.selected_height_map_path, paths.repo_root) if selected_afm else "",
            "scan_size_um": selected_afm.scan_size_um if selected_afm else "",
            "afm_resolution": f"{selected_afm.resolution_y}x{selected_afm.resolution_x}" if selected_afm else "",
            "warnings": ";".join(selection.warnings),
            "cache_key": cache_namespace,
        }
        dataset_rows.append(row)
        if included and selected_afm is not None:
            manifest_rows.append(
                {
                    "sample_id": selection.sample_id,
                    "sample_group_id": selection.sample_id,
                    "growth_run_id": selection.sample_id,
                    "manual_rheed_path": row["manual_rheed_path"],
                    "manual_rheed_filename": filename,
                    "inferred_stage": inferred_stage,
                    "selected_afm_scan_id": selected_afm.afm_scan_id,
                    "selected_height_map_path": display_path(selected_afm.selected_height_map_path, paths.repo_root),
                    "scan_size_um": selected_afm.scan_size_um,
                    "scan_size_x_um": selected_afm.scan_size_x_um,
                    "scan_size_y_um": selected_afm.scan_size_y_um,
                    "afm_resolution": f"{selected_afm.resolution_y}x{selected_afm.resolution_x}",
                    "material": selected_afm.material,
                    "afm_candidate_scan_count": len(valid_candidates),
                    "afm_selection_reason": selected_reason,
                    "rq_status": "available_but_blinded_until_model_stage" if math.isfinite(safe_float(selected_afm.rq_nm)) else "missing",
                    "rq_source": selected_afm.rq_source,
                    "removelist_sha256": removelist.sha256,
                    "visual_feature_spec_version": FEATURE_SPEC_VERSION,
                    "cache_key": cache_namespace,
                }
            )
            stage_review_rows.append(
                {
                    "sample_id": selection.sample_id,
                    "manual_rheed_filename": filename,
                    "inferred_stage": inferred_stage,
                    "approved_stage": "",
                    "comparable_stage_group": "",
                    "stage_confidence": "provisional_filename_rule",
                    "user_approved": "",
                    "user_notes": "",
                }
            )

    included_ids = [row["sample_id"] for row in manifest_rows]
    assert_no_removed_samples(included_ids, removed_ids, context="Stage 0 preliminary manifest")
    stage_counts = dict(Counter(row["inferred_stage"] for row in stage_review_rows))
    return Stage0Bundle(
        paths=paths,
        dataset_rows=tuple(dataset_rows),
        preliminary_manifest_rows=tuple(manifest_rows),
        excluded_rows=tuple(excluded_rows),
        stage_review_rows=tuple(stage_review_rows),
        missing_manual_sample_ids=tuple(sorted(missing_manual)),
        missing_afm_sample_ids=tuple(sorted(missing_afm)),
        stage_counts=stage_counts,
    )


def write_stage0_outputs(bundle: Stage0Bundle) -> None:
    write_csv_rows(bundle.paths.outputs_dir / "excluded_by_removelist.csv", bundle.excluded_rows, EXCLUDED_FIELDS)
    write_csv_rows(bundle.paths.outputs_dir / "dataset_audit.csv", bundle.dataset_rows, DATASET_AUDIT_FIELDS)
    write_csv_rows(bundle.paths.outputs_dir / "preliminary_manifest.csv", bundle.preliminary_manifest_rows, PRELIMINARY_MANIFEST_FIELDS)
    write_csv_rows(bundle.paths.annotations_dir / "stage_review_template.csv", bundle.stage_review_rows, STAGE_REVIEW_FIELDS)


def checkpoint_0_text(bundle: Stage0Bundle, removelist_payload: dict[str, Any]) -> str:
    stage_lines = [f"- `{name}`: {bundle.stage_counts.get(name, 0)}" for name in ALLOWED_STAGE_CATEGORIES]
    missing_manual = ", ".join(bundle.missing_manual_sample_ids) if bundle.missing_manual_sample_ids else "none"
    missing_afm = ", ".join(bundle.missing_afm_sample_ids) if bundle.missing_afm_sample_ids else "none"
    included_count = len(bundle.preliminary_manifest_rows)
    return "\n".join(
        [
            "# Checkpoint 0: Peak-Saddle Stage Audit",
            "",
            "## Canonical Removelist",
            "",
            f"- Path: `{removelist_payload['absolute_path']}`",
            f"- SHA256: `{removelist_payload['sha256']}`",
            f"- Sample `6088` excluded: `{int('6088' in set(removelist_payload['parsed_sample_ids']))}`",
            "",
            "## Candidate Dataset",
            "",
            f"- Candidate included sample count: `{included_count}`",
            f"- Missing manual images among non-removelist samples: {missing_manual}",
            f"- Missing AFM data among non-removelist manual selections: {missing_afm}",
            "",
            "## Provisional Growth Stages",
            "",
            *stage_lines,
            "",
            "## Exact User Action Required",
            "",
            "1. Open `annotations/rheed_peak_saddle/stage_review_template.csv`.",
            "2. Review each filename-derived `inferred_stage`.",
            "3. Fill `approved_stage`, `comparable_stage_group`, `user_approved`, and `user_notes`.",
            "4. Save the completed file as `annotations/rheed_peak_saddle/stage_review_completed.csv`.",
            "5. Ensure every included row has `user_approved = 1` before requesting Stage 1.",
            "",
            "STOP: Stage 1 synthetic validation has not been run.",
        ]
    )
