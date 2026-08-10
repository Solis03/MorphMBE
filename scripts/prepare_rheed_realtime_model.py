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
        default="configs/morphmbe_m22_realtime.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/models/morphmbe_m22.joblib"),
        help="Derived bundle path; frozen assets/models is never overwritten.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    repository = repository_root_from_config(config_path, config)
    config["repository_root"] = str(repository)
    destination = args.output.resolve()
    frozen_assets = (repository / "assets/models").resolve()
    if frozen_assets in destination.parents:
        raise SystemExit("refusing to overwrite a frozen release asset")
    manifest = destination.with_name(f"{destination.stem}_manifest.json")
    if destination.exists():
        raise SystemExit(
            f"output already exists: {destination}; choose a fresh --output"
        )

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
            "build_seconds": time.perf_counter() - started,
        },
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
