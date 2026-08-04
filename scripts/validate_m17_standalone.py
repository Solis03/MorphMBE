#!/usr/bin/env python3
"""Validate a self-contained M17b/N6342 standalone archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rheed2morph.realtime.cli import repository_root_from_config
from rheed2morph.realtime.model import M17_MODEL_ID, load_deployment_bundle


REQUIRED_FIGURES = (
    "Fig10_N6342_renderer_ablation.png",
    "Fig11_N6342_peak_signature.png",
    "Fig2_full27_target_scatter.png",
    "Fig5_confidence_audit.png",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "configs/"
            "rheed_realtime_ui_m17_full27_line3_exclude6081_v9.json"
        ),
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = repository_root_from_config(config_path, config)
    bundle = load_deployment_bundle(root / config["deployment_bundle"])
    generation = json.loads(
        (root / config["generation_config"]).read_text(encoding="utf-8")
    )
    failures: list[str] = []
    if bundle.model_id != M17_MODEL_ID:
        failures.append(f"unexpected model id: {bundle.model_id}")
    if len(bundle.groups) != 27 or "6081" in bundle.groups:
        failures.append(f"invalid deployment cohort: {bundle.groups}")
    if generation.get("selected_method") != (
        "M17b_topology_sparse_peak_terrace"
    ):
        failures.append("M17b is not the selected image generator")
    if bundle.retrieval_at_inference or bundle.measured_afm_patch_at_inference:
        failures.append("deployment violates the non-retrieval boundary")
    required_paths = (
        root / "data" / "compressedfile" / "N6342",
        root / "data" / "AFM-extra-five",
        root / "data" / "afm_metrology_line3_v1",
        root / "reports" / "rheed_n6342_sparse_island" / "REPORT.md",
        root / "reports" / "rheed_n6342_sparse_island" / "literature_review.md",
        root / config["endpoint_streak_features"],
        root / config["stability_predictions"],
    )
    for path in required_paths:
        if not path.exists():
            failures.append(f"missing required path: {path.relative_to(root)}")
    figure_root = (
        root
        / "reports"
        / "rheed_m17_end_to_end_generation"
        / "20260804_m17_sparse_topology_line3_full27_v1"
        / "full27_loo"
        / "figures"
    )
    for name in REQUIRED_FIGURES:
        if not (figure_root / name).is_file():
            failures.append(f"missing required figure: {name}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(
        json.dumps(
            {
                "status": "PASS",
                "archive_root": str(root),
                "model_id": bundle.model_id,
                "training_growth_count": len(bundle.groups),
                "6081_excluded": "6081" not in bundle.groups,
                "selected_generator": generation["selected_method"],
                "retrieval_at_inference": bundle.retrieval_at_inference,
                "required_figures": list(REQUIRED_FIGURES),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
