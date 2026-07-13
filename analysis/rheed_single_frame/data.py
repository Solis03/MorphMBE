"""Dataset construction for the manual single-frame RHEED experiment."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analysis.rheed_roughness.run import convert_height_to_nm, csv_value, display_path, read_config, safe_float
from analysis.rheed_roughness.visualize_manual_pairs import (
    AFMCandidate,
    ManualSelection,
    SelectedPair,
    common_scale_for_pairs,
    descriptor_lookup,
    discover_manual_rheed_images,
    infer_material,
    load_json,
    metadata_path_for_height,
    parse_scan_size_pair,
    recompute_height_stats,
    robust_display_limits,
    scan_id_from_height_path,
    scan_size_from_filename_pair,
    select_representative_afm_scan,
    valid_physical_afm,
)
from analysis.rheed_single_frame.removelist import RemovelistAudit, assert_no_removed_samples, excluded_rows_for_present_samples


DATASET_AUDIT_FIELDS = [
    "sample_id",
    "sample_group_id",
    "growth_run_id",
    "manual_rheed_path",
    "manual_rheed_filename",
    "manual_selection_status",
    "match_method",
    "removelist_status",
    "included_in_model",
    "skip_reason",
    "warnings",
]


TARGET_FIELDS = [
    "sample_id",
    "sample_group_id",
    "growth_run_id",
    "rq_nm",
    "rq_source",
    "metadata_rq_nm",
    "recomputed_rq_nm",
    "selected_afm_scan_id",
    "selected_afm_path",
    "selected_height_map_path",
    "scan_size_um",
    "scan_size_x_um",
    "scan_size_y_um",
    "afm_resolution",
    "height_unit",
    "native_display_min_nm",
    "native_display_max_nm",
    "common_display_min_nm",
    "common_display_max_nm",
    "selection_reason",
    "qc_flags",
]


@dataclass(frozen=True)
class ExperimentPaths:
    repo_root: Path
    outputs_dir: Path
    reports_dir: Path
    figures_dir: Path
    assets_dir: Path
    manual_root: Path
    plane_corrected_afm_root: Path


@dataclass(frozen=True)
class DatasetBundle:
    paths: ExperimentPaths
    selections: tuple[ManualSelection, ...]
    pairs: tuple[SelectedPair, ...]
    dataset_rows: tuple[dict[str, Any], ...]
    target_rows: tuple[dict[str, Any], ...]
    manifest_rows: tuple[dict[str, Any], ...]
    skipped_rows: tuple[dict[str, Any], ...]
    excluded_rows: tuple[dict[str, Any], ...]
    common_scale: tuple[float, float] | None
    afm_available_fields: tuple[str, ...]


def resolve_path(repo_root: Path, raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path


def make_paths(config: dict[str, Any]) -> ExperimentPaths:
    repo_root = resolve_path(Path(__file__).resolve().parents[2], config.get("repo_root", ".")).resolve()
    outputs = resolve_path(repo_root, config["outputs_dir"]).resolve()
    reports = resolve_path(repo_root, config["reports_dir"]).resolve()
    figures = reports / "figures"
    assets = reports / "assets"
    manual_root = resolve_path(repo_root, config.get("manual_selection_root", "data/manual_selection")).resolve()
    plane_root = resolve_path(repo_root, config.get("data_roots", {}).get("plane_corrected_afm_root", "data/plane_corrected_afm")).resolve()
    for path in (outputs, reports, figures, assets):
        path.mkdir(parents=True, exist_ok=True)
    return ExperimentPaths(
        repo_root=repo_root,
        outputs_dir=outputs,
        reports_dir=reports,
        figures_dir=figures,
        assets_dir=assets,
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


def write_parquet_or_csv_note(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path, index=False)
    except Exception as exc:  # pragma: no cover - depends on optional engines.
        path.with_suffix(path.suffix + ".unavailable.txt").write_text(
            f"Parquet export unavailable: {type(exc).__name__}: {exc}\nCSV with the same stem is authoritative.\n",
            encoding="utf-8",
        )


def metadata_rq_nm(metadata: dict[str, Any]) -> float:
    for key in ("height_std_nm", "Rq_nm", "rq_nm", "Sq_nm", "sq_nm"):
        value = safe_float(metadata.get(key), math.nan)
        if math.isfinite(value):
            return value
    return math.nan


def load_afm_candidates_filtered(paths: ExperimentPaths, config: dict[str, Any], removelist_ids: set[str]) -> list[AFMCandidate]:
    """Load AFM candidates only after path-level removelist exclusion."""
    desc = descriptor_lookup(paths.repo_root / "data" / "afm_descriptor_reconstruction" / "afm_descriptors.csv", paths.repo_root)
    candidates: list[AFMCandidate] = []
    for afm_path in sorted(paths.plane_corrected_afm_root.glob("*/*/*_plane_corrected.npy")):
        sample_id_from_folder = afm_path.parts[-3] if len(afm_path.parts) >= 3 else ""
        if sample_id_from_folder in removelist_ids:
            continue
        metadata_path = metadata_path_for_height(afm_path)
        metadata = load_json(metadata_path)
        sample_id = str(metadata.get("sample_id") or sample_id_from_folder)
        if not sample_id or sample_id in removelist_ids:
            continue
        rel_path = display_path(afm_path, paths.repo_root)
        desc_row = desc.get(rel_path, {})
        unit = str(
            metadata.get("height_unit_exported")
            or metadata.get("height_unit_original")
            or config.get("afm", {}).get("height_unit_default", "nm")
        )
        resolution = metadata.get("resolution")
        if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
            resolution_y = int(safe_float(resolution[0], 0))
            resolution_x = int(safe_float(resolution[1], 0))
        else:
            arr = np.load(afm_path, mmap_mode="r")
            resolution_y, resolution_x = int(arr.shape[-2]), int(arr.shape[-1])
        scan_size, scan_x, scan_y = parse_scan_size_pair(metadata)
        filename_scan_size, filename_scan_x, filename_scan_y = scan_size_from_filename_pair(metadata)
        channel = str(metadata.get("primary_channel") or "ZSensor")
        rq = safe_float(desc_row.get("Rq"), math.nan)
        ra = safe_float(desc_row.get("Ra"), math.nan)
        robust_range = safe_float(desc_row.get("p95"), math.nan) - safe_float(desc_row.get("p05"), math.nan)
        peak_to_valley = safe_float(desc_row.get("peak_to_valley"), math.nan)
        rq_source = "loaded_from_descriptor_table" if math.isfinite(rq) else "recomputed_from_height_map"
        recomputed_rq = math.nan
        qc_flags: list[str] = []
        if not math.isfinite(rq) or not math.isfinite(ra) or not math.isfinite(robust_range) or not math.isfinite(peak_to_valley):
            stats = recompute_height_stats(afm_path, unit)
            rq = rq if math.isfinite(rq) else stats["rq"]
            ra = ra if math.isfinite(ra) else stats["ra"]
            robust_range = robust_range if math.isfinite(robust_range) else stats["robust_height_range"]
            peak_to_valley = peak_to_valley if math.isfinite(peak_to_valley) else stats["peak_to_valley"]
            recomputed_rq = stats["rq"]
        else:
            stats = recompute_height_stats(afm_path, unit)
            recomputed_rq = stats["rq"]
            if math.isfinite(recomputed_rq) and abs(recomputed_rq - rq) > max(0.2, 0.2 * abs(rq)):
                qc_flags.append("rq_descriptor_recompute_disagreement")
        _, unit_status = convert_height_to_nm(np.asarray([0.0]), unit)
        if unit_status != "ok":
            qc_flags.append(unit_status)
        if not math.isfinite(scan_size) or scan_size <= 0:
            qc_flags.append("invalid_scan_size")
        elif (
            math.isfinite(filename_scan_size)
            and abs(filename_scan_x - scan_x) > 0.05
            and abs(filename_scan_y - scan_y) > 0.05
        ):
            qc_flags.append(
                f"scan_size_metadata_filename_disagreement:{scan_x:g}x{scan_y:g}_metadata_vs_{filename_scan_x:g}x{filename_scan_y:g}_filename"
            )
        if not math.isfinite(rq):
            qc_flags.append("missing_rq")
        candidates.append(
            AFMCandidate(
                sample_id=sample_id,
                sample_group_id=sample_id,
                growth_run_id=sample_id,
                material=infer_material(afm_path, metadata),
                afm_scan_id=str(metadata.get("afm_file_id") or scan_id_from_height_path(afm_path)),
                afm_path=afm_path,
                selected_height_map_path=afm_path,
                channel=channel,
                height_unit_exported=unit,
                scan_size_um=scan_size,
                scan_size_x_um=scan_x,
                scan_size_y_um=scan_y,
                resolution_x=resolution_x,
                resolution_y=resolution_y,
                rq_nm=rq,
                rq_source=rq_source,
                rq_recomputed_nm=recomputed_rq,
                ra_nm=ra,
                robust_height_range_nm=robust_range,
                peak_to_valley_nm=peak_to_valley,
                qc_flags=";".join(qc_flags),
            )
        )
    return candidates


def _selection_for_id(selections: Sequence[ManualSelection], sample_id: str) -> ManualSelection | None:
    for selection in selections:
        if selection.sample_id == sample_id:
            return selection
    return None


def _build_nonremoved_pairs(
    selections: Sequence[ManualSelection],
    afm_candidates: Sequence[AFMCandidate],
    config: dict[str, Any],
) -> tuple[list[SelectedPair], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    afm_by_sample: dict[str, list[AFMCandidate]] = {}
    for candidate in afm_candidates:
        afm_by_sample.setdefault(candidate.sample_id, []).append(candidate)
    primary_size = float(config.get("afm", {}).get("primary_scan_size_um", 1.0))
    tolerance = float(config.get("afm", {}).get("primary_scan_size_tolerance_um", 0.10))
    pairs: list[SelectedPair] = []
    dataset_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for selection in selections:
        warnings = list(selection.warnings)
        skip_reason = ""
        match_method = "exact_normalized_sample_id"
        selected_pair: SelectedPair | None = None
        selected_afm: AFMCandidate | None = None
        if selection.status != "ok":
            skip_reason = selection.status
        elif selection.sample_id not in afm_by_sample:
            skip_reason = "missing_afm_pair"
        else:
            sample_candidates = [c for c in afm_by_sample[selection.sample_id] if valid_physical_afm(c)]
            selected_afm, median_rq, distance, reason = select_representative_afm_scan(
                sample_candidates,
                primary_scan_size_um=primary_size,
                tolerance_um=tolerance,
            )
            if selected_afm is None:
                skip_reason = reason
            else:
                native_min, native_max = robust_display_limits(selected_afm.selected_height_map_path, selected_afm.height_unit_exported)
                selected_pair = SelectedPair(
                    sample_id=selection.sample_id,
                    sample_group_id=selection.sample_id,
                    growth_run_id=selection.sample_id,
                    material=selected_afm.material,
                    manual_folder=selection.manual_folder,
                    manual_rheed_path=selection.selected_path or Path(),
                    manual_rheed_filename=(selection.selected_path.name if selection.selected_path else ""),
                    all_manual_candidates=selection.candidates,
                    manual_warnings=selection.warnings,
                    afm=selected_afm,
                    number_of_candidate_scans=len(sample_candidates),
                    sample_median_rq_nm=median_rq,
                    distance_from_median_rq_nm=distance,
                    selection_reason=reason,
                    native_display_min_nm=native_min,
                    native_display_max_nm=native_max,
                )
                pairs.append(selected_pair)
        dataset_rows.append(
            {
                "sample_id": selection.sample_id,
                "sample_group_id": selection.sample_id,
                "growth_run_id": selection.sample_id,
                "manual_rheed_path": display_path(selection.selected_path, Path.cwd()) if selection.selected_path else "",
                "manual_rheed_filename": selection.selected_path.name if selection.selected_path else "",
                "manual_selection_status": selection.status,
                "match_method": match_method,
                "removelist_status": "kept",
                "included_in_model": int(selected_pair is not None),
                "skip_reason": skip_reason,
                "warnings": ";".join(warnings),
            }
        )
        if selected_afm is not None:
            metadata = load_json(metadata_path_for_height(selected_afm.selected_height_map_path))
            target_rows.append(
                {
                    "sample_id": selection.sample_id,
                    "sample_group_id": selection.sample_id,
                    "growth_run_id": selection.sample_id,
                    "rq_nm": selected_afm.rq_nm,
                    "rq_source": selected_afm.rq_source,
                    "metadata_rq_nm": metadata_rq_nm(metadata),
                    "recomputed_rq_nm": selected_afm.rq_recomputed_nm,
                    "selected_afm_scan_id": selected_afm.afm_scan_id,
                    "selected_afm_path": display_path(selected_afm.afm_path, Path.cwd()),
                    "selected_height_map_path": display_path(selected_afm.selected_height_map_path, Path.cwd()),
                    "scan_size_um": selected_afm.scan_size_um,
                    "scan_size_x_um": selected_afm.scan_size_x_um,
                    "scan_size_y_um": selected_afm.scan_size_y_um,
                    "afm_resolution": f"{selected_afm.resolution_y}x{selected_afm.resolution_x}",
                    "height_unit": selected_afm.height_unit_exported,
                    "native_display_min_nm": selected_pair.native_display_min_nm if selected_pair else "",
                    "native_display_max_nm": selected_pair.native_display_max_nm if selected_pair else "",
                    "common_display_min_nm": "",
                    "common_display_max_nm": "",
                    "selection_reason": selected_pair.selection_reason if selected_pair else "",
                    "qc_flags": selected_afm.qc_flags,
                }
            )
        manifest_rows.append(
            {
                **dataset_rows[-1],
                "selected_afm_scan_id": selected_afm.afm_scan_id if selected_afm else "",
                "selected_height_map_path": display_path(selected_afm.selected_height_map_path, Path.cwd()) if selected_afm else "",
                "rq_nm": selected_afm.rq_nm if selected_afm else "",
                "scan_size_um": selected_afm.scan_size_um if selected_afm else "",
            }
        )
    return pairs, dataset_rows, target_rows, manifest_rows


def build_dataset(config: dict[str, Any], paths: ExperimentPaths, removelist: RemovelistAudit) -> DatasetBundle:
    selections_all = discover_manual_rheed_images(paths.manual_root)
    excluded_rows = [
        {
            "sample_id": record.sample_id,
            "source_path": record.source_path.as_posix(),
            "exclusion_reason": "canonical_removelist",
            "note": record.note,
        }
        for record in removelist.records
    ]
    kept_selections = tuple(selection for selection in selections_all if selection.sample_id not in set(removelist.sample_ids))
    afm_candidates = load_afm_candidates_filtered(paths, config, set(removelist.sample_ids))
    assert_no_removed_samples((c.sample_id for c in afm_candidates), removelist.sample_ids, context="AFM candidate loading")
    pairs, dataset_rows, target_rows, manifest_rows = _build_nonremoved_pairs(kept_selections, afm_candidates, config)
    common_scale = common_scale_for_pairs(pairs, float(config["afm"]["primary_scan_size_um"]), float(config["afm"]["primary_scan_size_tolerance_um"]))
    for row in target_rows:
        if common_scale is not None and abs(safe_float(row.get("scan_size_um")) - float(config["afm"]["primary_scan_size_um"])) <= float(
            config["afm"]["primary_scan_size_tolerance_um"]
        ):
            row["common_display_min_nm"] = common_scale[0]
            row["common_display_max_nm"] = common_scale[1]
    skipped_rows = [row for row in dataset_rows if not int(row.get("included_in_model", 0))]
    model_ids = [pair.sample_id for pair in pairs]
    assert_no_removed_samples(model_ids, removelist.sample_ids, context="model dataset")
    if len({row["sample_id"] for row in target_rows}) != len(target_rows):
        raise AssertionError("More than one selected AFM target was produced for a sample.")
    if set(model_ids) != {str(row["sample_id"]) for row in target_rows}:
        raise AssertionError("Selected RHEED pairs and AFM target rows do not match one-to-one.")
    afm_available_fields = (
        "sample_id",
        "afm_scan_id",
        "selected_height_map_path",
        "scan_size_um",
        "resolution",
        "height_unit",
        "metadata_rq_nm",
        "descriptor_Rq_nm",
        "recomputed_rq_nm",
        "Rq_nm",
        "Ra_nm",
        "robust_height_range_nm",
        "peak_to_valley_nm",
    )
    return DatasetBundle(
        paths=paths,
        selections=tuple(selections_all),
        pairs=tuple(pairs),
        dataset_rows=tuple(dataset_rows),
        target_rows=tuple(target_rows),
        manifest_rows=tuple(manifest_rows),
        skipped_rows=tuple(skipped_rows),
        excluded_rows=tuple(excluded_rows),
        common_scale=common_scale,
        afm_available_fields=afm_available_fields,
    )


def write_dataset_outputs(bundle: DatasetBundle, removelist: RemovelistAudit) -> None:
    assert_no_removed_samples((pair.sample_id for pair in bundle.pairs), removelist.sample_ids, context="dataset output writing")
    write_csv_rows(bundle.paths.outputs_dir / "dataset_audit.csv", bundle.dataset_rows, DATASET_AUDIT_FIELDS)
    write_csv_rows(bundle.paths.outputs_dir / "dataset_manifest.csv", bundle.manifest_rows)
    write_csv_rows(bundle.paths.outputs_dir / "excluded_by_removelist.csv", bundle.excluded_rows, ["sample_id", "source_path", "exclusion_reason", "note"])
    write_csv_rows(bundle.paths.outputs_dir / "skipped_samples.csv", bundle.skipped_rows, DATASET_AUDIT_FIELDS)
    write_csv_rows(bundle.paths.outputs_dir / "selected_afm_targets.csv", bundle.target_rows, TARGET_FIELDS)
