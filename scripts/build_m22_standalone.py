#!/usr/bin/env python3
"""Materialize the M22 standalone from the read-only M17 archive and M22 tree."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_M17 = Path(
    "/Users/ziyi/Desktop/"
    "MorphMBE_M17_N6342_SparsePeak_UI_Standalone_20260804"
)
DEFAULT_DESTINATION = (
    ROOT / "standalone/MorphMBE_M22_DenseMid_UI_Standalone_20260810"
)
TEMPLATE_ROOT = ROOT / "packaging/m22_standalone"


def _copy(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m17-root", type=Path, default=DEFAULT_M17)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    source = args.m17_root.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        raise SystemExit(
            f"destination already exists; move it aside before rebuilding: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-cR", str(source), str(destination)], check=True)

    for relative in ("src", "analysis", "scripts", "configs", "tests"):
        _copy(ROOT / relative, destination / relative)
    for relative in (
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "removelist.txt",
        "reports/rheed_m22_dense_mid_20260809.md",
    ):
        _copy(ROOT / relative, destination / relative)

    for relative in (
        "outputs/rheed_m22_dense_mid",
        "reports/rheed_m22_dense_mid",
        "outputs/rheed_m20_spot_connectivity/20260808_m20_connectivity_full27_v2",
        "outputs/rheed_m19_separated_rough_islands/20260807_m19_source_predictions",
        "outputs/rheed_endpoint_generation/m16_full27_exclude6081_v2",
    ):
        _copy(ROOT / relative, destination / relative)
    _copy(
        ROOT / "reproduced_outputs/model_smoke_6063",
        destination / "reproduced_outputs/model_smoke_6063",
    )
    _copy(
        ROOT / "reproduced_outputs/m22_ui_offscreen_6063",
        destination / "reproduced_outputs/ui_offscreen_6063",
    )
    for name in (
        "morphmbe_m20_m22c_dense_mid_full27_exclude6081_live_v10.joblib",
        "morphmbe_m20_m22c_dense_mid_full27_exclude6081_live_v10_manifest.json",
    ):
        relative = Path("outputs/rheed_realtime_ui") / name
        _copy(ROOT / relative, destination / relative)

    torch_weight = Path.home() / ".cache/torch/hub/checkpoints/r3d_18-b3b3357e.pth"
    _copy(
        torch_weight,
        destination / "tmp/torch/hub/checkpoints/r3d_18-b3b3357e.pth",
    )
    _copy(TEMPLATE_ROOT / "README.md", destination / "README.md")
    _copy(TEMPLATE_ROOT / "docs", destination / "docs")
    _copy(
        TEMPLATE_ROOT / "run_m22_standalone.sh",
        destination / "scripts/run_m22_standalone.sh",
    )
    (destination / "scripts/run_m22_standalone.sh").chmod(0o755)

    provenance = {
        "package": destination.name,
        "research_branch": "codex/m22-dense-mid-dual-cohort-20260809",
        "research_commit": "94f20d081964cb38986ff9c1221f3af78cf4a3d2",
        "research_tag": "m22-dense-mid-dual-cohort-v1",
        "standalone_source_branch": _git("branch", "--show-current"),
        "standalone_source_commit": _git("rev-parse", "HEAD"),
        "standalone_source_tree_dirty": bool(_git("status", "--porcelain")),
        "m17_reference_root": str(source),
        "m17_reference_policy": "read_only",
        "active_ui_config": "configs/rheed_realtime_ui_m22_full27_dense_mid_v10.json",
        "active_model": "M20 spot-connectivity Sq + M22c gap completion",
        "raw_data_policy": "copied_from_read_only_reference; never modified",
    }
    provenance_path = destination / "provenance/m22_standalone_manifest.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "built", "destination": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
