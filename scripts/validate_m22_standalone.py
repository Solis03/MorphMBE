#!/usr/bin/env python3
"""Validate the M20+M22c real-time standalone and its visual evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.rheed_to_afm_full_cohort_loo.run import load_config
from rheed2morph.realtime.cli import repository_root_from_config
from rheed2morph.realtime.model import M22_MODEL_ID, load_deployment_bundle

ATLAS_NAMES = tuple(f"Atlas_{page:02d}_of_06.png" for page in range(1, 7))
EXTRA_FIGURES = (
    "Focus_true_Sq_3p5_to_6p0_M17_vs_M22_dual.png",
    "M22_Sq_measured_vs_predicted_ordered.png",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_realtime_ui_m22_full27_dense_mid_v10.json",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    ui_config = json.loads(config_path.read_text(encoding="utf-8"))
    root = repository_root_from_config(config_path, ui_config)
    generation = load_config(root / ui_config["generation_config"])
    bundle = load_deployment_bundle(root / ui_config["deployment_bundle"])
    failures: list[str] = []

    if bundle.model_id != M22_MODEL_ID:
        failures.append(f"unexpected model id: {bundle.model_id}")
    if len(bundle.groups) != 27 or "6081" in bundle.groups:
        failures.append(f"invalid deployment cohort: {bundle.groups}")
    if generation.get("selected_method") != "M22c_gap_completion_strong":
        failures.append("M22c is not the selected image generator")
    renderer = generation.get("selected_renderer", {})
    if renderer.get("island_generator_mode") != (
        "separated_ellipse_growth_layered_gapfill_strong"
    ):
        failures.append("M22c gap-completion generator is not active")
    if bundle.spot_connectivity_reference is None:
        failures.append("M20 spot-connectivity deployment head is absent")
    if bundle.retrieval_at_inference or bundle.measured_afm_patch_at_inference:
        failures.append("deployment violates the non-retrieval boundary")

    required_paths = (
        root / ui_config["deployment_manifest"],
        root / ui_config["deep_visibility_ranker"],
        root / ui_config["model_input_roi_calibration"],
        root / ui_config["physics_roi_calibration"],
        root / ui_config["online_clear_moment_detector"],
        root / "tmp/torch/hub/checkpoints/r3d_18-b3b3357e.pth",
        root / "reports/rheed_m22_dense_mid_20260809.md",
        root / "docs/M22_STANDALONE_RUNBOOK.md",
        root / "docs/M22_VISUALIZATION_INDEX.md",
        root / "reproduced_outputs/model_smoke_6063/result.json",
        root / "reproduced_outputs/ui_offscreen_6063/ui_offscreen.png",
    )
    for path in required_paths:
        if not path.exists():
            failures.append(f"missing required path: {path.relative_to(root)}")

    figure_root = (
        root
        / "reports/rheed_m22_dense_mid/20260809_m22_paired_comparison"
        / "figures/gwyddion_individual_height_atlas_M17_vs_M22_dual"
    )
    for name in (*ATLAS_NAMES, *EXTRA_FIGURES):
        if not (figure_root / name).is_file():
            failures.append(f"missing required visualization: {name}")

    smoke_path = root / "reproduced_outputs/model_smoke_6063/result.json"
    smoke = (
        json.loads(smoke_path.read_text(encoding="utf-8"))
        if smoke_path.is_file()
        else {}
    )
    smoke_prediction = smoke.get("prediction", {})
    if smoke_prediction.get("model_id") != M22_MODEL_ID:
        failures.append("6063 smoke result does not identify M20+M22c")
    generated_sq = smoke_prediction.get("generated_sq_nm")
    predicted_sq = smoke_prediction.get("Sq_nm", {}).get("value")
    if (
        generated_sq is None
        or predicted_sq is None
        or abs(float(generated_sq) - float(predicted_sq)) > 1e-4
    ):
        failures.append("6063 generated and predicted Sq are inconsistent")

    if failures:
        raise SystemExit("\n".join(failures))
    print(
        json.dumps(
            {
                "status": "PASS",
                "standalone_root": str(root),
                "model_id": bundle.model_id,
                "training_growth_count": len(bundle.groups),
                "6081_excluded": "6081" not in bundle.groups,
                "selected_generator": generation["selected_method"],
                "spot_connectivity_head": True,
                "retrieval_at_inference": bundle.retrieval_at_inference,
                "visualization_count": len(ATLAS_NAMES + EXTRA_FIGURES) + 2,
                "smoke_6063_predicted_sq_nm": predicted_sq,
                "smoke_6063_generated_sq_nm": generated_sq,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
