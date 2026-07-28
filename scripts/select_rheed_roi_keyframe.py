#!/usr/bin/env python3
"""Select a RHEED ROI and rotation-phase keyframe from one complete video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rheed2morph.rheed.automatic_roi_keyframe import (
    ROI_METHODS,
    save_selection_artifacts,
    select_from_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically find a border-safe RHEED ROI and the clearest "
            "right-most diffraction-trajectory vertex."
        )
    )
    parser.add_argument(
        "source",
        help=(
            "Input MOV/MP4/AVI/MKV video or a directory containing numeric "
            "lossless PNG frames."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Derived artifact directory; the source is never modified.",
    )
    parser.add_argument(
        "--roi-method",
        choices=ROI_METHODS,
        default="calibrated_safe",
    )
    parser.add_argument(
        "--selected-method",
        choices=(
            "quality_only",
            "vertex_clarity",
            "physics_vertex",
            "front_visibility",
            "compact_physics",
            "compact_visibility",
            "supervised_phase_ranker",
        ),
        default="supervised_phase_ranker",
    )
    parser.add_argument("--roi-aspect-ratio", type=float, default=1.54)
    parser.add_argument("--roi-scale", type=float, default=0.90)
    parser.add_argument("--roi-sample-count", type=int, default=48)
    parser.add_argument(
        "--phase-ranker",
        default=(
            "outputs/rheed_auto_roi_keyframe/"
            "20260728_diffraction_front_visibility_v2/"
            "supervised_phase_ranker/"
            "gradient_boosting_phase_ranker.joblib"
        ),
        help=(
            "Fitted phase-candidate ranker. Pass an empty string to run only "
            "the deterministic physics heuristics."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection, _, _ = select_from_source(
        args.source,
        roi_method=args.roi_method,
        aspect_ratio=args.roi_aspect_ratio,
        calibrated_scale=args.roi_scale,
        roi_sample_count=args.roi_sample_count,
        phase_ranker_path=args.phase_ranker or None,
    )
    paths = save_selection_artifacts(
        selection,
        output_dir=args.output_dir,
        selected_method=args.selected_method,
    )
    print(
        json.dumps(
            {
                "selection": selection.to_dict(),
                "artifacts": {
                    key: str(Path(value)) for key, value in paths.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
