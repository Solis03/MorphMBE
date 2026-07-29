#!/usr/bin/env python3
"""Prepare the full-23 deployment refit of M15b + frozen M12a."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import torch

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
    repository = config_path.parent.parent
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
            "Rq_nm": bundle.rq_reference.method,
            "FSMI_nm": bundle.fsmi_reference.method,
            "image_generator": bundle.generation_config["selected_method"],
            "retrieval_at_inference": False,
            "measured_afm_patch_at_inference": False,
            "confidence": (
                "rotation-angular-coverage x keyframe/ROI TTA centrality "
                "with extreme head-conflict diagnostic"
            ),
            "rotation_period_reference_available": (
                bundle.period_frames_reference is not None
            )
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
