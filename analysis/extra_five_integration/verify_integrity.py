from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.rheed_video_afm_story.common import (
    repo_path,
    sha256_file,
    write_csv,
    write_json,
)


EXTRA = {"N6342", "N6358", "N6382", "N6389", "N6390"}
EXCLUDED = {"6043", "6055", "N6324"}
M12A = "M12a_edge_preserving_terrace"


def _load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def _verify_raw_afm(inventory_path: Path) -> pd.DataFrame:
    inventory = pd.read_csv(
        inventory_path,
        dtype={"sample_id": str},
    )
    records = []
    for row in inventory.itertuples(index=False):
        path = repo_path(row.raw_afm_path)
        actual = sha256_file(path)
        records.append(
            {
                "modality": "AFM",
                "sample_id": str(row.sample_id),
                "path": str(row.raw_afm_path),
                "expected_sha256": str(row.raw_afm_sha256),
                "actual_sha256": actual,
                "hash_match": actual == str(row.raw_afm_sha256),
                "expected_size_bytes": int(row.size_bytes),
                "actual_size_bytes": int(path.stat().st_size),
                "size_match": int(path.stat().st_size)
                == int(row.size_bytes),
                "selected_for_modeling": str(row.decision) == "include",
            }
        )
    return pd.DataFrame(records)


def _verify_raw_rheed(inventory_path: Path) -> pd.DataFrame:
    inventory = pd.read_csv(
        inventory_path,
        dtype={"sample_id": str},
    )
    records = []
    for row in inventory.itertuples(index=False):
        path = repo_path(row.raw_rheed_video)
        selected = bool(row.selected_for_modeling)
        expected_hash = (
            str(row.raw_rheed_sha256)
            if selected
            else ""
        )
        actual_hash = sha256_file(path) if selected else ""
        records.append(
            {
                "modality": "RHEED",
                "sample_id": str(row.sample_id),
                "path": str(row.raw_rheed_video),
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "hash_match": (
                    actual_hash == expected_hash if selected else True
                ),
                "expected_size_bytes": int(row.size_bytes),
                "actual_size_bytes": int(path.stat().st_size),
                "size_match": int(path.stat().st_size)
                == int(row.size_bytes),
                "expected_mtime_ns": int(row.mtime_ns),
                "actual_mtime_ns": int(path.stat().st_mtime_ns),
                "mtime_match": int(path.stat().st_mtime_ns)
                == int(row.mtime_ns),
                "selected_for_modeling": selected,
            }
        )
    return pd.DataFrame(records)


def _verify_model_artifacts(
    *, config: dict[str, Any], integration_root: Path
) -> dict[str, Any]:
    machine = pd.read_csv(
        integration_root
        / "machine_dataset_full28"
        / "modeling_manifest.csv",
        dtype={"growth_run_id": str},
    )
    scans = pd.read_csv(
        integration_root / "combined_primary_1um_scans.csv",
        dtype={"growth_run_id": str},
    )
    predictions_path = repo_path(
        config["external_confidence_predictions"]
    )
    predictions = pd.read_csv(
        predictions_path,
        dtype={"growth_run_id": str},
    )
    report = (
        repo_path(config["report_root"])
        / str(config.get("full_run_suffix", "full23_loo"))
    )
    folds = pd.read_csv(
        report / "fold_integrity_audit.csv",
        dtype={"held_growth_run_id": str},
    )
    map_root = (
        repo_path(config["output_root"])
        / str(config.get("full_run_suffix", "full23_loo"))
        / "crossfit"
        / "generated_maps"
        / M12A
    )
    maps = []
    for path in sorted(map_root.glob("*.npz")):
        payload = np.load(path, allow_pickle=False)
        maps.append(
            {
                "growth_run_id": path.stem,
                "draw_count": int(
                    np.asarray(payload["generated_unit_shapes"]).shape[0]
                ),
                "retrieval_at_inference": bool(
                    np.asarray(payload["retrieval_at_inference"]).item()
                ),
                "measured_afm_patch_used_at_inference": bool(
                    np.asarray(
                        payload["measured_afm_patch_used_at_inference"]
                    ).item()
                ),
            }
        )
    map_audit = pd.DataFrame(maps)
    machine_groups = set(machine["growth_run_id"].astype(str))
    scan_groups = set(scans["growth_run_id"].astype(str))
    prediction_groups = set(predictions["growth_run_id"].astype(str))
    map_groups = set(map_audit["growth_run_id"].astype(str))
    checks = {
        "machine_growth_count_28": len(machine_groups) == 28,
        "scan_growth_count_28": len(scan_groups) == 28,
        "combined_scan_count_214": len(scans) == 214,
        "extra_five_all_present": EXTRA.issubset(machine_groups),
        "excluded_absent_from_machine": not EXCLUDED & machine_groups,
        "excluded_absent_from_scans": not EXCLUDED & scan_groups,
        "excluded_absent_from_predictions": (
            not EXCLUDED & prediction_groups
        ),
        "prediction_growth_count_28": len(prediction_groups) == 28,
        "prediction_target_row_count_56": len(predictions) == 56,
        "outer_target_never_used": not predictions[
            "outer_target_used_for_training"
        ].astype(bool).any(),
        "generator_fold_count_28": len(folds) == 28,
        "generator_fit_count_27": (
            folds["fit_growth_count"].astype(int) == 27
        ).all(),
        "held_never_overlaps_generator_fit": not folds[
            "held_overlap_with_fit"
        ].astype(bool).any(),
        "generated_map_count_28": len(map_groups) == 28,
        "generated_map_groups_match_predictions": (
            map_groups == prediction_groups
        ),
        "generated_draw_count_4": (
            map_audit["draw_count"].astype(int) == 4
        ).all(),
        "retrieval_never_used": not map_audit[
            "retrieval_at_inference"
        ].any(),
        "measured_patch_never_used_at_inference": not map_audit[
            "measured_afm_patch_used_at_inference"
        ].any(),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(f"model integrity checks failed: {failed}")
    return {
        "checks": checks,
        "machine_growth_ids": sorted(machine_groups),
        "prediction_sha256": sha256_file(predictions_path),
        "map_audit": map_audit,
    }


def run(config_path: str | Path) -> None:
    config = _load_config(config_path)
    integration_output = repo_path(
        "outputs/extra_five_integration/20260729_line3_full28_v1"
    )
    integration_report = repo_path(
        "reports/extra_five_integration/20260729_line3_full28_v1"
    )
    afm = _verify_raw_afm(
        integration_report / "raw_afm_source_inventory.csv"
    )
    rheed = _verify_raw_rheed(
        integration_report / "raw_rheed_source_inventory.csv"
    )
    raw = pd.concat([afm, rheed], ignore_index=True, sort=False)
    if not raw["hash_match"].all() or not raw["size_match"].all():
        raise RuntimeError("raw source integrity check failed")
    if "mtime_match" in raw and not raw["mtime_match"].fillna(True).all():
        raise RuntimeError("raw RHEED source mtime check failed")
    artifacts = _verify_model_artifacts(
        config=config,
        integration_root=integration_output,
    )
    write_csv(raw, integration_report / "raw_source_integrity_audit.csv")
    write_csv(
        artifacts.pop("map_audit"),
        integration_report / "generated_map_integrity_audit.csv",
    )
    manifest = {
        "experiment_id": config["experiment_id"],
        "raw_afm_file_count": int((raw["modality"] == "AFM").sum()),
        "raw_rheed_inventory_file_count": int(
            (raw["modality"] == "RHEED").sum()
        ),
        "selected_rheed_full_hash_count": int(
            (
                (raw["modality"] == "RHEED")
                & raw["selected_for_modeling"].astype(bool)
            ).sum()
        ),
        "all_raw_hash_size_mtime_checks_passed": True,
        **artifacts,
        "raw_data_modified": False,
        "standalone_modified": False,
    }
    write_json(
        manifest,
        integration_report / "final_integrity_manifest.json",
    )
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_m15b_end_to_end_generation_line3_full28_v1.json",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
