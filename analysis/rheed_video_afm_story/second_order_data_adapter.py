from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import display_path, repo_path, save_parquet, sha256_file, write_csv, write_json


VARIANT_ID = "afm_second_order_y2_v1"


def read_processing_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(repo_path(path), dtype=str).fillna("")
    valid = df[df["status"].isin(["success", "exists_skipped"])].copy()
    return valid.sort_values("source_relative_path").reset_index(drop=True)


def second_order_metadata_path(output_path: str | Path, output_root: str | Path) -> Path:
    out = repo_path(output_path)
    root = repo_path(output_root)
    return root / "_metadata" / out.relative_to(root).with_suffix(".json")


def load_json(path: str | Path) -> dict[str, Any]:
    p = repo_path(path)
    if not p.exists():
        return {}
    value = json.loads(p.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def first_order_path_for_source(source_path: str | Path) -> Path:
    source = repo_path(source_path)
    rel = source.relative_to(repo_path("data/processed_afm"))
    base = rel.name.removesuffix("_height.npy")
    return repo_path("data/plane_corrected_afm") / rel.parent / f"{base}_plane_corrected.npy"


def source_metadata_path(source_path: str | Path) -> Path:
    source = repo_path(source_path)
    base = source.name.removesuffix("_height.npy")
    return source.with_name(f"{base}_metadata.json")


def scan_id_from_source(source_path: str | Path) -> str:
    source = repo_path(source_path)
    return source.parent.name


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def scan_size_pair(metadata: dict[str, Any]) -> tuple[float, float]:
    value = metadata.get("scan_size_um")
    if isinstance(value, list) and len(value) >= 2:
        x, y = as_float(value[0]), as_float(value[1])
    else:
        x = y = as_float(value)
    if x > 20.0 or y > 20.0:
        x /= 1000.0
        y /= 1000.0
    return x, y


def finite_policy_ok(source: np.ndarray, second: np.ndarray) -> bool:
    if source.shape != second.shape:
        return False
    source_finite = np.isfinite(source)
    if not np.isfinite(second[source_finite]).all():
        return False
    nan_mask = np.isnan(source)
    if nan_mask.any() and not np.isnan(second[nan_mask]).all():
        return False
    posinf = np.isposinf(source)
    neginf = np.isneginf(source)
    if posinf.any() and not np.isposinf(second[posinf]).all():
        return False
    if neginf.any() and not np.isneginf(second[neginf]).all():
        return False
    return True


def build_scan_mapping(config: dict[str, Any]) -> pd.DataFrame:
    manifest = read_processing_manifest(config["second_order_manifest_path"])
    output_root = repo_path(config["second_order_root"])
    rows: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    seen_outputs: set[str] = set()
    for record in manifest.to_dict("records"):
        source = repo_path(record["source_path"])
        second = repo_path(record["output_path"])
        first = first_order_path_for_source(source)
        meta_path = source_metadata_path(source)
        meta = load_json(meta_path)
        so_meta = second_order_metadata_path(second, output_root)
        sample_id = str(meta.get("sample_id") or Path(record["source_relative_path"]).parts[0])
        scan_id = str(meta.get("afm_file_id") or scan_id_from_source(source))
        sx, sy = scan_size_pair(meta)
        output_exists = second.exists()
        mapping_notes: list[str] = []
        mapping_status = "ok"
        if not source.exists():
            mapping_status = "missing_source"
        if not first.exists():
            mapping_status = "missing_first_order"
        if not output_exists:
            mapping_status = "missing_second_order"
        if str(source) in seen_sources:
            mapping_status = "duplicate_source"
        if str(second) in seen_outputs:
            mapping_status = "duplicate_second_order"
        seen_sources.add(str(source))
        seen_outputs.add(str(second))
        source_sha = sha256_file(source) if source.exists() else ""
        first_sha = sha256_file(first) if first.exists() else ""
        second_sha = sha256_file(second) if second.exists() else ""
        shape_ok = False
        finite_ok = False
        resolution_y = resolution_x = 0
        if source.exists() and second.exists():
            src = np.load(source, allow_pickle=False)
            sec = np.load(second, allow_pickle=False)
            shape_ok = src.shape == sec.shape
            finite_ok = finite_policy_ok(src, sec)
            if src.ndim >= 2:
                resolution_y, resolution_x = int(src.shape[-2]), int(src.shape[-1])
            if not shape_ok:
                mapping_notes.append("shape_mismatch")
                mapping_status = "shape_mismatch"
            if not finite_ok:
                mapping_notes.append("finite_policy_failed")
                mapping_status = "finite_policy_failed"
        height_unit = str(meta.get("height_unit_exported") or meta.get("height_unit_original") or "")
        if height_unit != "nm":
            mapping_notes.append("height_unit_not_nm")
            mapping_status = "height_unit_not_nm"
        if "_backgrounds" in second.parts:
            mapping_notes.append("background_array_selected")
            mapping_status = "background_array_selected"
        rows.append(
            {
                "sample_id": sample_id,
                "growth_run_id": sample_id,
                "scan_id": scan_id,
                "raw_afm_path": str(meta.get("raw_afm_file") or meta.get("raw_file") or ""),
                "source_afm_path": display_path(source),
                "first_order_afm_path": display_path(first),
                "second_order_afm_path": display_path(second),
                "second_order_metadata_path": display_path(so_meta),
                "source_sha256": source_sha,
                "first_order_sha256": first_sha,
                "second_order_sha256": second_sha,
                "scan_size_x_um": sx,
                "scan_size_y_um": sy,
                "resolution_x": resolution_x,
                "resolution_y": resolution_y,
                "height_unit": height_unit,
                "second_order_manifest_status": record["status"],
                "output_exists": bool(output_exists),
                "shape_matches_source": bool(shape_ok),
                "finite_policy_ok": bool(finite_ok),
                "mapping_status": mapping_status,
                "mapping_notes": ";".join(mapping_notes),
            }
        )
    mapping = pd.DataFrame(rows).sort_values(["sample_id", "scan_id"]).reset_index(drop=True)
    out = repo_path(config["variant_output_root"]) / "provenance"
    rep = repo_path(config["variant_report_root"]) / "provenance"
    out.mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)
    write_csv(mapping, out / "second_order_scan_mapping.csv")
    audit_lines = [
        "# Second-Order Mapping Audit",
        "",
        f"- variant_id: `{VARIANT_ID}`",
        f"- manifest rows loaded as valid: {len(mapping)}",
        f"- outputs exist: {int(mapping['output_exists'].sum())}",
        f"- mapping status counts: `{mapping['mapping_status'].value_counts().to_dict()}`",
        f"- source duplicate count: {int(mapping['source_afm_path'].duplicated().sum())}",
        f"- second-order duplicate count: {int(mapping['second_order_afm_path'].duplicated().sum())}",
        f"- nm unit rows: {int(mapping['height_unit'].eq('nm').sum())}",
        f"- shape matches source: {int(mapping['shape_matches_source'].sum())}",
        f"- finite policy ok: {int(mapping['finite_policy_ok'].sum())}",
        "",
        "No `_backgrounds`, `_metadata`, `_qc`, rendered PNG, RHEED, latent, cache, or network-input arrays are used.",
    ]
    (rep / "second_order_mapping_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    return mapping


def write_variant_registry(config: dict[str, Any]) -> dict[str, Any]:
    registry = {
        "variant_id": VARIANT_ID,
        "experiment_type": "controlled AFM-preprocessing ablation",
        "afm_preprocessing": {
            "source": config["second_order_root"],
            "model": "y2",
            "formula": "c0 + c1*x + c2*y + c3*y^2",
            "robust_fitting": True,
            "normalization": False,
            "smoothing": False,
            "resize_during_fitting": False,
        },
        "rheed_inputs_changed": False,
        "sample_cohort_changed": False,
        "removelist_changed": False,
        "model_architecture_changed": False,
        "model_search_space_changed": False,
        "retrieval_settings_changed": False,
        "patch_synthesis_settings_changed": False,
        "visualization_style_changed": False,
        "second_order_manifest_sha256": sha256_file(config["second_order_manifest_path"]),
        "phase1_manifest_sha256": sha256_file(config["phase1_manifest_path"]),
        "phase2a_embedding_registry_sha256": sha256_file(config["phase2a_embedding_registry_path"]),
        "phase4a_config_sha256": sha256_file(config["phase4a_base_config_path"]),
        "removelist_sha256": sha256_file(config["removelist_path"]),
    }
    write_json(registry, repo_path(config["variant_output_root"]) / "variant_registry.json")
    return registry


def write_reused_rheed_artifacts(config: dict[str, Any], primary_ids: list[str]) -> list[dict[str, Any]]:
    artifacts = [
        ("phase1_manifest", config["phase1_manifest_path"], "fixed cohort, keyframe, ROI, clip indices"),
        ("phase1_clip_cache_dir", "outputs/rheed_video_afm_story/phase1/clip_cache", "target-blind cached RHEED clips"),
        ("phase2a_embedding_registry", config["phase2a_embedding_registry_path"], "target-blind embedding registry"),
        ("phase2a_dino_embedding", "outputs/rheed_video_afm_story/phase2a/embeddings/dino_vits14__keyframe_1__raw_luminance.npz", "target-blind DINO keyframe embedding"),
        ("phase4a_physics_features", "outputs/rheed_video_afm_story/phase4a/rheed_physics_features.csv", "target-blind RHEED physics features"),
        ("phase1_rheed_quality", "outputs/rheed_video_afm_story/phase1/rheed_quality_metrics.csv", "target-blind RHEED quality metrics"),
    ]
    rows = []
    for name, path, reason in artifacts:
        p = repo_path(path)
        if p.is_dir():
            child_hashes = {child.name: sha256_file(child) for child in sorted(p.glob("*")) if child.is_file() and child.stem in primary_ids}
            shape: Any = {"file_count_for_primary_ids": len(child_hashes)}
            digest = str(hash(json.dumps(child_hashes, sort_keys=True)))
        else:
            digest = sha256_file(p)
            shape = ""
            if p.suffix == ".npz":
                data = np.load(p, allow_pickle=False)
                shape = {key: list(data[key].shape) for key in data.files}
            elif p.suffix == ".csv":
                shape = {"rows": int(len(pd.read_csv(p)))}
        rows.append({"name": name, "path": display_path(p), "sha256": digest, "shape": shape, "sample_ids": primary_ids, "reuse_reason": reason, "target_blind": True})
    write_json(rows, repo_path(config["variant_output_root"]) / "provenance" / "reused_rheed_artifacts.json")
    return rows
