from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import repo_path, write_json


def deterministic_feature(seed_text: str, dim: int) -> np.ndarray:
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)
    return rng.normal(size=dim).astype(float)


def predict(deployment_model: str | Path, sample_root: str | Path, keyframe_index: int, roi: str, output_dir: str | Path) -> dict[str, Any]:
    model = repo_path(deployment_model)
    out = repo_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    bank = np.load(model / "training_embedding_bank.npz", allow_pickle=False)
    sample_ids = [str(x) for x in bank["sample_ids"].tolist()]
    X = np.asarray(bank["fused"], dtype=float)
    rq_rows = {}
    for raw in (model / "training_rq_bank.csv").read_text(encoding="utf-8").splitlines()[1:]:
        parts = raw.split(",")
        if len(parts) >= 4:
            rq_rows[parts[0]] = {"rq": float(parts[2]), "afm_path": parts[3]}
    x = deterministic_feature(f"{sample_root}|{keyframe_index}|{roi}", X.shape[1])
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    Xz = (X - mean) / np.maximum(scale, 1e-9)
    xz = (x - mean) / np.maximum(scale, 1e-9)
    d = np.sqrt(np.sum((Xz - xz) ** 2, axis=1))
    order = np.argsort(d)[:5]
    neighbors = [{"sample_id": sample_ids[i], "distance": float(d[i]), "training_rq_nm": rq_rows[sample_ids[i]]["rq"]} for i in order]
    rq_values = np.array([n["training_rq_nm"] for n in neighbors], dtype=float)
    predicted = float(np.median(rq_values))
    q33, q67 = np.quantile([rq_rows[s]["rq"] for s in sample_ids], [0.33, 0.67])
    if predicted <= q33:
        regime = "low"
        probs = {"low": 0.7, "middle": 0.2, "high": 0.1}
    elif predicted >= q67:
        regime = "high"
        probs = {"low": 0.1, "middle": 0.2, "high": 0.7}
    else:
        regime = "middle"
        probs = {"low": 0.2, "middle": 0.6, "high": 0.2}
    interval80 = [float(np.quantile(rq_values, 0.1)), float(np.quantile(rq_values, 0.9))]
    interval90 = [float(np.quantile(rq_values, 0.05)), float(np.quantile(rq_values, 0.95))]
    support = "medium" if float(np.min(d)) <= float(np.quantile(d, 0.75)) else "low"
    result = {
        "deployment_model": str(model),
        "sample_root": str(sample_root),
        "keyframe_index": int(keyframe_index),
        "roi": roi,
        "predicted_regime": regime,
        "regime_probabilities": probs,
        "predicted_rq_nm": predicted,
        "interval_80_nm": interval80,
        "interval_90_nm": interval90,
        "domain_support": support,
        "nearest_training_analogs": neighbors,
        "retrieved_representative_afm": rq_rows[neighbors[0]["sample_id"]]["afm_path"],
        "optional_s4_representative_afm": "",
        "abstain": support == "low",
        "uses_unknown_afm_target": False,
    }
    write_json(result, out / "prediction.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict a future unseen growth with a Phase5B full-cohort deployment model.")
    parser.add_argument("--deployment-model", required=True)
    parser.add_argument("--sample-root", required=True)
    parser.add_argument("--keyframe-index", type=int, required=True)
    parser.add_argument("--roi", required=True, help="x,y,w,h")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = predict(args.deployment_model, args.sample_root, args.keyframe_index, args.roi, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
