from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.rheed_video_afm_story.afm_dataset import build_afm_manifest
from analysis.rheed_video_afm_story.afm_pca_decoder import make_group_folds
from analysis.rheed_video_afm_story.common import (
    load_config,
    repo_path,
    sha256_file,
    write_json,
)


def run(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    manifest, audit, descriptors = build_afm_manifest(config)
    folds = make_group_folds(manifest, config)
    groups = sorted(manifest["growth_run_id"].astype(str).unique())
    if len(groups) != 23:
        raise RuntimeError(f"expected 23 corrected groups, found {len(groups)}")
    if manifest["afm_file_id"].duplicated().any():
        raise RuntimeError("duplicate AFM file IDs remain after metrology audit")
    if manifest["source_array_hash"].duplicated().any():
        raise RuntimeError("duplicate selected AFM arrays remain in descriptors")
    output = repo_path(config["output_root"])
    summary: dict[str, object] = {
        "afm_scan_count": int(len(manifest)),
        "growth_group_count": int(len(groups)),
        "growth_run_ids": groups,
        "quality_rejection_count": int((~audit["quality_pass"]).sum()),
        "roughness_metric": "Sq_areal_RMS_height_nm",
        "preprocessing": "third_order_polynomial_per_scan_line",
        "sample_aggregation": (
            "arithmetic median of deduplicated scan Sq in nm; log afterwards"
        ),
        "descriptor_sha256": sha256_file(output / "afm_descriptors.csv"),
        "group_fold_sha256": sha256_file(output / "group_outer_splits.csv"),
    }
    write_json(summary, output / "descriptor_build_summary.json")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_video_afm_story_phase3a_line3_v1.yaml",
    )
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
