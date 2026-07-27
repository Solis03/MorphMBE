#!/usr/bin/env python3
"""Validate prompt references and write a portable asset manifest."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


PACKAGE = Path(__file__).resolve().parents[1]
OUTPUT_NAMES = {"ASSET_MANIFEST.csv", "SHA256SUMS.txt", "PACKAGE_VALIDATION.json"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def referenced_paths() -> set[str]:
    references: set[str] = set()
    for markdown in PACKAGE.glob("*.md"):
        for token in re.findall(r"`([^`]+)`", markdown.read_text(encoding="utf-8")):
            if token.startswith(("01_", "02_", "03_", "04_", "05_")):
                references.add(token.rstrip(".,;:"))
    for line in (PACKAGE / "FILES_TO_ATTACH.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            references.add(line.strip())
    return references


def main() -> None:
    checks: list[dict[str, object]] = []
    references = sorted(referenced_paths())
    missing = [
        path
        for path in references
        if not (
            (PACKAGE / path).is_dir()
            if path.endswith("/")
            else (PACKAGE / path).is_file()
        )
    ]
    checks.append(
        {
            "check": "all_prompt_file_references_exist",
            "passed": not missing,
            "detail": {"reference_count": len(references), "missing": missing},
        }
    )

    raw = sorted((PACKAGE / "01_rheed_inputs/raw_keyframes").glob("*.png"))
    roi = sorted((PACKAGE / "01_rheed_inputs/roi_keyframes").glob("*.png"))
    ready_png = sorted((PACKAGE / "01_rheed_inputs/model_ready").glob("*.png"))
    ready_npz = sorted((PACKAGE / "01_rheed_inputs/model_ready").glob("*.npz"))
    gt = sorted((PACKAGE / "02_afm_ground_truth_selected").glob("*.png"))
    retrieved = sorted((PACKAGE / "03_afm_retrieval_outputs").glob("*.png"))
    candidates = sorted((PACKAGE / "04_afm_candidate_bank").glob("*.png"))
    references_png = sorted((PACKAGE / "05_reference_only").glob("*.png"))
    counts = {
        "raw_rheed_png": len(raw),
        "roi_rheed_png": len(roi),
        "model_ready_png": len(ready_png),
        "model_ready_npz": len(ready_npz),
        "selected_gt_afm_png": len(gt),
        "retrieved_afm_png": len(retrieved),
        "candidate_bank_png": len(candidates),
        "reference_only_png": len(references_png),
    }
    expected = {
        "raw_rheed_png": 5,
        "roi_rheed_png": 5,
        "model_ready_png": 6,
        "model_ready_npz": 5,
        "selected_gt_afm_png": 5,
        "retrieved_afm_png": 3,
        "candidate_bank_png": 6,
        "reference_only_png": 2,
    }
    checks.append(
        {
            "check": "asset_counts_exact",
            "passed": counts == expected,
            "detail": {"actual": counts, "expected": expected},
        }
    )

    npz_ok = True
    npz_details = {}
    for path in ready_npz:
        archive = np.load(path, allow_pickle=False)
        shape = list(archive["frames_uint8"].shape)
        dtype = str(archive["frames_uint8"].dtype)
        npz_ok &= shape == [1, 224, 224] and dtype == "uint8"
        npz_details[path.name] = {"shape": shape, "dtype": dtype}
    checks.append(
        {
            "check": "model_ready_tensors_are_exact_single_224_frames",
            "passed": npz_ok,
            "detail": npz_details,
        }
    )

    image_details = {}
    images_ok = True
    for path in raw + roi + ready_png + gt + retrieved + candidates + references_png:
        with Image.open(path) as image:
            width, height = image.size
            images_ok &= width > 0 and height > 0
            image_details[str(path.relative_to(PACKAGE))] = {
                "width_px": width,
                "height_px": height,
                "mode": image.mode,
            }
    checks.append(
        {
            "check": "all_png_assets_decode",
            "passed": images_ok,
            "detail": {"image_count": len(image_details)},
        }
    )

    parameter_audit = json.loads(
        (PACKAGE / "MODEL_PARAMETER_AUDIT.json").read_text(encoding="utf-8")
    )
    parameters = parameter_audit["encoder"]["parameter_counts"]
    dino_sum = (
        parameters["patch_embedding"]
        + parameters["class_token"]
        + parameters["pretrained_positional_embedding"]
        + parameters["mask_token_present_but_unused_for_inference"]
        + parameters["twelve_transformer_blocks"]
        + parameters["final_layer_norm"]
    )
    accounting = parameter_audit["parameter_count_interpretation"]
    parameter_ok = (
        dino_sum == 22056576
        and parameters["one_transformer_block"] * 12
        == parameters["twelve_transformer_blocks"]
        and accounting["encoder_plus_ridge_parameter_values"]
        == 22056576 + 7685
        and accounting["complete_serialized_numeric_state_including_scalers"]
        == 22056576 + 7685 + 15360
    )
    checks.append(
        {
            "check": "parameter_arithmetic_exact",
            "passed": parameter_ok,
            "detail": {
                "summed_dino_parameters": dino_sum,
                "encoder_plus_ridge": accounting["encoder_plus_ridge_parameter_values"],
                "state_including_scalers": accounting[
                    "complete_serialized_numeric_state_including_scalers"
                ],
            },
        }
    )

    manifest_rows = []
    checksum_rows = []
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file() or path.name in OUTPUT_NAMES:
            continue
        relative = str(path.relative_to(PACKAGE))
        digest = sha256_file(path)
        row = {
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": digest,
            "file_type": path.suffix.lower().lstrip("."),
        }
        row.update(image_details.get(relative, {}))
        manifest_rows.append(row)
        checksum_rows.append(f"{digest}  {relative}")

    manifest_path = PACKAGE / "ASSET_MANIFEST.csv"
    fields = [
        "relative_path",
        "bytes",
        "sha256",
        "file_type",
        "width_px",
        "height_px",
        "mode",
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    (PACKAGE / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_rows) + "\n", encoding="utf-8"
    )

    passed = all(bool(check["passed"]) for check in checks)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "check_count": len(checks),
        "manifest_entry_count": len(manifest_rows),
        "checks": checks,
    }
    (PACKAGE / "PACKAGE_VALIDATION.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
