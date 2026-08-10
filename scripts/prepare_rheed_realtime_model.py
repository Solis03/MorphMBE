#!/usr/bin/env python3
"""Prepare an audited real-time MorphMBE deployment refit."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

from rheed2morph.realtime.cli import repository_root_from_config
from rheed2morph.realtime.model import (
    build_deployment_bundle,
    save_deployment_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_realtime_ui.json",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only the derived deployment cache, never a publication freeze.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repository = repository_root_from_config(config_path, config)
    config["repository_root"] = str(repository)
    destination = repository / config["deployment_bundle"]
    manifest = repository / config["deployment_manifest"]
    if destination.exists() and not args.force:
        print(
            json.dumps(
                {
                    "status": "cache_exists",
                    "bundle": str(destination),
                    "hint": "pass --force to rebuild this derived cache",
                },
                indent=2,
            )
        )
        return

    started = time.perf_counter()

    def progress(message: str) -> None:
        print(f"[model] {message}", flush=True)

    bundle = build_deployment_bundle(config, progress=progress)
    save_deployment_bundle(bundle, destination)
    payload = {
        "model_id": bundle.model_id,
        "bundle": str(destination.relative_to(repository)),
        "created_at": bundle.created_at,
        "training_growth_count": len(bundle.groups),
        "training_growth_ids": bundle.groups,
        "frozen_parameter_hashes": bundle.frozen_parameter_hashes,
        "method": {
            "Sq_nm": (
                "M20_spot_connectivity_calibrated_sq"
                if bundle.spot_connectivity_reference is not None
                else (
                    "M16_endpoint_streak_dual_resolution"
                    if bundle.endpoint_streak_reference is not None
                    else bundle.rq_reference.method
                )
            ),
            "legacy_internal_target_name": "Rq_nm",
            "FSMI_nm": bundle.fsmi_reference.method,
            "image_generator": bundle.generation_config["selected_method"],
            "afm_metrology": (
                "third-order independent polynomial flatten per fast-scan "
                "line; sample target is median scan Sq in nm"
            ),
            "retrieval_at_inference": False,
            "measured_afm_patch_at_inference": False,
            "confidence": (
                "M16 target-blind endpoint-support risk with causal "
                "angular-TTA/head risk; strict-LOO error calibration"
            ),
            "rotation_period_reference_available": (
                bundle.period_frames_reference is not None
            ),
            "spot_connectivity_reference_available": (
                bundle.spot_connectivity_reference is not None
            ),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "build_seconds": time.perf_counter() - started
        }
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
