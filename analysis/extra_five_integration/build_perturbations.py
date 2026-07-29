from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.rheed_auto_input_robustness.perturbation import (
    extract_perturbation_embeddings,
    save_perturbation_embeddings,
)
from analysis.rheed_video_afm_story.common import repo_path, write_json


def run(config_path: str | Path, *, device: str) -> dict[str, object]:
    config = json.loads(repo_path(config_path).read_text(encoding="utf-8"))
    machine_root = repo_path(config["output_root"]) / "machine_dataset_full28"
    selection = pd.read_csv(
        machine_root / "selection_comparison.csv",
        dtype={"sample_id": str, "growth_run_id": str},
    )
    manifest = pd.read_csv(
        machine_root / "modeling_manifest.csv",
        dtype={"sample_id": str, "growth_run_id": str},
    )
    base_path_value = config.get("base_perturbation_embeddings")
    if base_path_value:
        base = np.load(repo_path(base_path_value), allow_pickle=False)
        base_groups = [
            str(value) for value in base["growth_run_ids"].tolist()
        ]
        extra_groups = list(map(str, config["included_samples"]))
        selection_index = selection.set_index("growth_run_id")
        manifest_index = manifest.set_index("growth_run_id")
        extra_selection = selection_index.loc[extra_groups].reset_index()
        extra_manifest = manifest_index.loc[extra_groups].reset_index()
        (
            extracted_groups,
            views,
            extra_embeddings,
            weight,
        ) = extract_perturbation_embeddings(
            extra_selection,
            extra_manifest,
            device=device,
        )
        base_views = [str(value) for value in base["view_names"].tolist()]
        if extracted_groups != extra_groups or views != base_views:
            raise RuntimeError("base and extra perturbation views do not match")
        if str(base["weight_identifier"]) != str(weight):
            raise RuntimeError("base and extra R3D perturbation weights differ")
        groups = base_groups + extra_groups
        embeddings = np.concatenate(
            [
                np.asarray(base["embeddings"], dtype=np.float32),
                np.asarray(extra_embeddings, dtype=np.float32),
            ],
            axis=0,
        )
    else:
        groups, views, embeddings, weight = extract_perturbation_embeddings(
            selection,
            manifest,
            device=device,
        )
    if len(groups) != int(config["expected_combined_growth_count"]):
        raise RuntimeError("unexpected perturbation cohort size")
    if "N6324" in set(groups):
        raise RuntimeError("N6324 entered perturbation embeddings")
    destination = repo_path(config["expanded_m15_output_root"])
    destination.mkdir(parents=True, exist_ok=True)
    path = save_perturbation_embeddings(
        destination / "r3d_causal8_input_perturbations.npz",
        groups=groups,
        view_names=views,
        embeddings=embeddings,
        weight_identifier=weight,
    )
    manifest_payload: dict[str, object] = {
        "growth_count": len(groups),
        "growth_run_ids": groups,
        "view_names": views,
        "embedding_shape": list(embeddings.shape),
        "weight_identifier": weight,
        "target_blind": True,
        "n6324_used": False,
        "base_perturbation_embeddings": str(base_path_value or ""),
        "output": str(path),
    }
    write_json(
        manifest_payload,
        destination / "perturbation_embedding_manifest.json",
    )
    print(json.dumps(manifest_payload, indent=2), flush=True)
    return manifest_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/extra_five_line3_full28_v1.json",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.config, device=str(args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
