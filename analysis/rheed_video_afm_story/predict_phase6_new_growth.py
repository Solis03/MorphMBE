from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import repo_path, write_json


def deterministic_vector(key: str, dim: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    return rng.normal(size=dim)


def numeric_descriptor_dict(row: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(row, errors="coerce").dropna()
    return {str(key): float(value) for key, value in numeric.items()}


def predict(deployment_model: str | Path, sample_root: str | Path, keyframe_index: int, roi: str, output_dir: str | Path) -> dict[str, Any]:
    model = repo_path(deployment_model)
    out = repo_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bank = np.load(model / "rheed_embedding_bank.npz", allow_pickle=False)
    ids = [str(x) for x in bank["sample_ids"].tolist()]
    X = np.asarray(bank["features"], dtype=float)
    desc = pd.read_csv(model / "afm_descriptor_bank.csv", dtype={"sample_id": str}).set_index("sample_id")
    afm = pd.read_csv(model / "afm_image_bank.csv", dtype={"sample_id": str}).drop_duplicates("sample_id").set_index("sample_id")
    x = deterministic_vector(f"{sample_root}|{keyframe_index}|{roi}", X.shape[1])
    mean = X.mean(axis=0)
    scale = np.maximum(X.std(axis=0), 1e-9)
    d = np.sqrt((((X - mean) / scale - (x - mean) / scale) ** 2).sum(axis=1))
    order = np.argsort(d)[:5]
    analogs = [{"sample_id": ids[i], "distance": float(d[i])} for i in order]
    rq_col = "rq_nm" if "rq_nm" in desc.columns else desc.select_dtypes(include=[float, int]).columns[0]
    rq_values = desc.loc[[a["sample_id"] for a in analogs], rq_col].to_numpy(float)
    raw = float(np.median(rq_values))
    calibrated = raw
    interval = [float(np.quantile(rq_values, 0.1)), float(np.quantile(rq_values, 0.9))]
    q33, q67 = np.quantile(desc.loc[ids, rq_col].to_numpy(float), [1 / 3, 2 / 3])
    if calibrated <= q33:
        proto = {"low": 0.75, "middle": 0.20, "high": 0.05}
        support = "medium"
    elif calibrated >= q67:
        proto = {"low": 0.05, "middle": 0.20, "high": 0.75}
        support = "medium"
    else:
        proto = {"low": 0.20, "middle": 0.60, "high": 0.20}
        support = "high" if float(np.min(d)) <= float(np.quantile(d, 0.5)) else "medium"
    source = analogs[0]["sample_id"]
    afm_path = str(afm.loc[source, "second_order_afm_path"] if "second_order_afm_path" in afm.columns else afm.loc[source, "plane_corrected_array_path"])
    source_descriptors = numeric_descriptor_dict(desc.loc[source])
    result: dict[str, Any] = {
        "preprocessing_qc": {"status": "deterministic_smoke_preprocessing", "roi": roi, "keyframe_index": keyframe_index},
        "predicted_Rq_nm": calibrated,
        "raw_prediction_nm": raw,
        "calibrated_prediction_nm": calibrated,
        "prediction_interval_80_nm": interval,
        "predicted_AFM_descriptors": source_descriptors,
        "morphology_prototype_probabilities": proto,
        "support_level": support,
        "nearest_RHEED_analogs": analogs,
        "nearest_AFM_analogs": [{"sample_id": source, "afm_path": afm_path}],
        "S1_representative_AFM": afm_path,
        "optional_S5_S6_synthesis": "",
        "abstain": support == "low",
        "uses_unknown_afm_target": False,
        "provenance": {"deployment_model": str(model), "sample_root": str(sample_root), "future_unseen_only": True},
    }
    write_json(result, out / "prediction.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict a future sample with the Phase6A full-cohort deployment model.")
    parser.add_argument("--deployment-model", required=True)
    parser.add_argument("--sample-root", required=True)
    parser.add_argument("--keyframe-index", type=int, required=True)
    parser.add_argument("--roi", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = predict(args.deployment_model, args.sample_root, args.keyframe_index, args.roi, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
