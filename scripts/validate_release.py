#!/usr/bin/env python3
"""Validate the self-contained MorphMBE M22 release boundary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

from analysis.rheed_to_afm_full_cohort_loo.run import load_config
from rheed2morph.realtime.cli import repository_root_from_config
from rheed2morph.realtime.model import M22_MODEL_ID, load_deployment_bundle

EXPECTED_METRICS = {
    "mean_absolute_error": 0.6853452351430823,
    "rmse": 0.829067335675652,
    "pearson_r": 0.9234250316048422,
}
EXPECTED_GROWTHS = {
    "6022",
    "6028",
    "6029",
    "6033",
    "6047",
    "6048",
    "6056",
    "6057",
    "6062",
    "6063",
    "6070",
    "6072",
    "6078",
    "6080",
    "6082",
    "6084",
    "6085",
    "6090",
    "6094",
    "6095",
    "6099",
    "6101",
    "N6342",
    "N6358",
    "N6382",
    "N6389",
    "N6390",
}
MAX_TRACKED_FILE_BYTES = 20 * 1024 * 1024


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tracked_files(root: Path) -> list[Path]:
    process = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return [root / item.decode() for item in process.stdout.split(b"\0") if item]


def validate(config_path: Path) -> dict[str, object]:
    config_path = config_path.resolve()
    ui_config = json.loads(config_path.read_text(encoding="utf-8"))
    root = repository_root_from_config(config_path, ui_config)
    generation = load_config(root / ui_config["generation_config"])
    bundle = load_deployment_bundle(root / ui_config["deployment_bundle"])
    failures: list[str] = []

    if bundle.model_id != M22_MODEL_ID:
        failures.append(f"unexpected model id: {bundle.model_id}")
    if set(bundle.groups) != EXPECTED_GROWTHS:
        failures.append("deployment cohort differs from the frozen 27 growths")
    if generation.get("selected_method") != "M22c_gap_completion_strong":
        failures.append("M22c is not the selected AFM generator")
    renderer = generation.get("selected_renderer", {})
    if renderer.get("island_generator_mode") != (
        "separated_ellipse_growth_layered_gapfill_strong"
    ):
        failures.append("M22c gap-completion renderer is not active")
    if bundle.spot_connectivity_reference is None:
        failures.append("M20 spot-connectivity Sq head is absent")
    if bundle.retrieval_at_inference or bundle.measured_afm_patch_at_inference:
        failures.append("deployment violates the non-retrieval boundary")

    required = (
        root / "README.md",
        root / "docs/ARCHITECTURE.md",
        root / "docs/MODEL_CARD.md",
        root / "docs/REPRODUCIBILITY.md",
        root / "docs/RELEASE_VERIFICATION.md",
        root / "docs/assets/m22_overview.png",
        root / "docs/assets/m22_results.png",
        root / "tests/fixtures/m22_6063_result.json",
        root / ui_config["deployment_manifest"],
        root / ui_config["deep_visibility_ranker"],
        root / ui_config["model_input_roi_calibration"],
        root / ui_config["physics_roi_calibration"],
        root / ui_config["online_clear_moment_detector"],
    )
    for path in required:
        if not path.is_file():
            failures.append(f"missing release file: {path.relative_to(root)}")

    manifest_path = root / "assets/manifest.sha256"
    if not manifest_path.is_file():
        failures.append("missing assets/manifest.sha256")
    else:
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            expected_hash, relative = line.split("  ", 1)
            asset = root / relative
            if not asset.is_file():
                failures.append(f"missing checksummed asset: {relative}")
            elif _sha256(asset) != expected_hash:
                failures.append(f"asset checksum mismatch: {relative}")

    summary_rows = _read_csv(root / "results/m22/target_prediction_summary.csv")
    sq_row = next((row for row in summary_rows if row["target"] == "Rq_nm"), None)
    if sq_row is None or int(sq_row["group_count"]) != 27:
        failures.append("Sq summary is not the frozen 27-growth result")
    elif any(
        abs(float(sq_row[name]) - expected) > 1e-12
        for name, expected in EXPECTED_METRICS.items()
    ):
        failures.append("frozen Sq metrics drifted")

    predictions = _read_csv(root / "results/m22/sq_outer_loo_predictions.csv")
    predicted_groups = {row["growth_run_id"] for row in predictions}
    if len(predictions) != 27 or predicted_groups != EXPECTED_GROWTHS:
        failures.append("outer-LOO prediction cohort is invalid")
    if any(
        row["outer_target_used_for_training"].lower() == "true" for row in predictions
    ):
        failures.append("a held Sq target entered its training fold")
    if any(int(row["outer_fit_growth_count"]) != 26 for row in predictions):
        failures.append("outer-LOO prediction fit count is not 26")

    folds = _read_csv(root / "results/m22/fold_integrity_audit.csv")
    held_groups = {row["held_growth_run_id"] for row in folds}
    if len(folds) != 27 or held_groups != EXPECTED_GROWTHS:
        failures.append("fold-integrity audit is incomplete")
    if any(row["held_overlap_with_fit"].lower() == "true" for row in folds):
        failures.append("held growth overlaps a training fold")
    if any(int(row["fit_growth_count"]) != 26 for row in folds):
        failures.append("fold-integrity fit count is not 26")

    fixture = json.loads(
        (root / "tests/fixtures/m22_6063_result.json").read_text(encoding="utf-8")
    )
    fixture_prediction = fixture.get("prediction", {})
    if fixture_prediction.get("model_id") != M22_MODEL_ID:
        failures.append("6063 frozen fixture has the wrong model identity")
    predicted_sq = fixture_prediction.get("Sq_nm", {}).get("value")
    generated_sq = fixture_prediction.get("generated_sq_nm")
    if (
        predicted_sq is None
        or generated_sq is None
        or abs(float(predicted_sq) - float(generated_sq)) > 1e-4
    ):
        failures.append("6063 generated and predicted Sq are inconsistent")

    forbidden_roots = {
        "checkpoints",
        "outputs",
        "paper_freeze",
        "publication_freeze",
        "reports",
    }
    oversized: list[str] = []
    forbidden: list[str] = []
    for path in _tracked_files(root):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in forbidden_roots:
            forbidden.append(str(relative))
        if path.stat().st_size > MAX_TRACKED_FILE_BYTES:
            oversized.append(str(relative))
    if forbidden:
        failures.append(f"historical artifact roots remain tracked: {forbidden[:3]}")
    if oversized:
        failures.append(f"tracked files exceed 20 MiB: {oversized}")

    if failures:
        raise RuntimeError("\n".join(failures))
    return {
        "status": "PASS",
        "root": str(root),
        "model_id": bundle.model_id,
        "training_growths": len(bundle.groups),
        "held_growth_overlap": False,
        "selected_generator": generation["selected_method"],
        "retrieval_at_inference": False,
        "Sq_metrics": {name: float(sq_row[name]) for name in EXPECTED_METRICS},
        "fixture_6063_predicted_Sq_nm": float(predicted_sq),
        "fixture_6063_generated_Sq_nm": float(generated_sq),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/morphmbe_m22_realtime.json"),
    )
    args = parser.parse_args()
    print(json.dumps(validate(args.config), indent=2))


if __name__ == "__main__":
    main()
