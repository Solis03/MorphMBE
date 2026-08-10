#!/usr/bin/env python3
"""Validate the publication and editable assets in the M22 figure package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

FIGURES = {
    "Figure_1_AutoRHEED_M22_overview": (4200, 2670),
    "Figure_2_M20_M22_model_and_validation": (4200, 3690),
    "Figure_3_M22_selected_results_and_Sq": (4200, 5010),
    "Figure_4_M22_full_cohort_atlas": (5100, 15600),
}
FORMATS = ("png", "tiff", "pdf", "svg")
EXPECTED_METRICS = {
    "pearson_r": 0.9234250316048422,
    "mae_nm": 0.6853452351430823,
    "rmse_nm": 0.829067335675652,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pdf_pages(path: Path) -> int:
    executable = shutil.which("pdfinfo")
    if executable is None:
        raise RuntimeError("pdfinfo is required for PDF validation")
    result = subprocess.run(
        [executable, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"could not read page count from {path}")


def validate(package_root: Path) -> dict[str, object]:
    package_root = package_root.resolve()
    figure_root = package_root / "figures"
    editable_root = package_root / "editable"
    svg_text = []

    for stem, expected_size in FIGURES.items():
        for suffix in FORMATS:
            path = figure_root / f"{stem}.{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"missing figure asset: {path}")
        for suffix in ("png", "tiff"):
            path = figure_root / f"{stem}.{suffix}"
            with Image.open(path) as image:
                if image.size != expected_size:
                    raise RuntimeError(
                        f"{path.name}: expected {expected_size}, got {image.size}"
                    )
                dpi = image.info.get("dpi", (0.0, 0.0))
                if min(float(dpi[0]), float(dpi[1])) < 599.0:
                    raise RuntimeError(f"{path.name}: expected 600 dpi, got {dpi}")
                if suffix == "tiff" and image.info.get("compression") != "tiff_lzw":
                    raise RuntimeError(f"{path.name}: TIFF is not LZW-compressed")
        pdf = figure_root / f"{stem}.pdf"
        if _pdf_pages(pdf) != 1:
            raise RuntimeError(f"{pdf.name}: expected one PDF page")
        svg = figure_root / f"{stem}.svg"
        root = ET.parse(svg).getroot()
        text_nodes = [
            "".join(element.itertext())
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] == "text"
        ]
        if len(text_nodes) < 10:
            raise RuntimeError(f"{svg.name}: editable text was not retained")
        svg_text.extend(text_nodes)

    public_text = "\n".join(svg_text)
    mapping = pd.read_csv(
        package_root / "private/sample_id_mapping_internal.csv",
        dtype={"public_sample_id": str, "internal_growth_run_id": str},
    )
    if len(mapping) != 27 or mapping["public_sample_id"].nunique() != 27:
        raise RuntimeError("private public-ID mapping is incomplete")
    leaked = [
        growth for growth in mapping["internal_growth_run_id"] if growth in public_text
    ]
    if leaked:
        raise RuntimeError(f"internal growth IDs leaked into SVG text: {leaked}")

    predictions = pd.read_csv(
        editable_root / "data/M22_Sq_outer_LOO.csv",
        dtype={"public_sample_id": str},
    )
    if len(predictions) != 27 or predictions["public_sample_id"].nunique() != 27:
        raise RuntimeError("outer-LOO prediction table is not a 27-growth cohort")
    truth = predictions["measured_Sq_nm"].to_numpy(float)
    predicted = predictions["predicted_Sq_nm"].to_numpy(float)
    metrics = {
        "pearson_r": float(np.corrcoef(truth, predicted)[0, 1]),
        "mae_nm": float(np.mean(np.abs(predicted - truth))),
        "rmse_nm": float(np.sqrt(np.mean(np.square(predicted - truth)))),
    }
    for name, expected in EXPECTED_METRICS.items():
        if not np.isclose(metrics[name], expected, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"{name} drifted: expected {expected}, got {metrics[name]}"
            )

    manifest = json.loads(
        (package_root / "provenance/figure_package_manifest.json").read_text()
    )
    if manifest["growth_count"] != 27:
        raise RuntimeError("provenance manifest growth count is not 27")
    for boundary in (
        "retrieval_at_inference",
        "measured_afm_patch_at_inference",
        "outer_target_used_for_training",
    ):
        if manifest[boundary]:
            raise RuntimeError(f"inference/split boundary failed: {boundary}")

    excluded_names = {
        "package_validation.json",
        "file_manifest.sha256",
    }
    manifest_files = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and path.name not in excluded_names
        and not path.name.endswith(".inspect.ndjson")
        and not any(part.endswith("_editable") for part in path.parts)
    )
    checksum_lines = [
        f"{_sha256(path)}  {path.relative_to(package_root)}" for path in manifest_files
    ]
    checksum_path = package_root / "provenance/file_manifest.sha256"
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    result: dict[str, object] = {
        "status": "PASS",
        "package": str(package_root),
        "figures": len(FIGURES),
        "publication_assets": len(FIGURES) * len(FORMATS),
        "editable_svg_figures": len(FIGURES),
        "growths": len(predictions),
        "metrics": metrics,
        "checksum_files": len(manifest_files),
    }
    validation_path = package_root / "provenance/package_validation.json"
    validation_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.package), indent=2))


if __name__ == "__main__":
    main()
