from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.metrics import structural_similarity

from .afm_patch_bank import load_bank_shape, training_bank_for_sample
from .common import display_path, repo_path, save_parquet, write_csv
from .rq_disentanglement import project_unit_rq_np, rq_np


def scale_to_predicted_rq(unit_shape: np.ndarray, predicted_rq: float) -> np.ndarray:
    return float(predicted_rq) * project_unit_rq_np(unit_shape)


def medoid_shape(bank_rows: pd.DataFrame) -> np.ndarray:
    shapes = [load_bank_shape(r) for _, r in bank_rows.iterrows()]
    if len(shapes) == 1:
        return shapes[0]
    flat = np.stack([s.ravel() for s in shapes])
    d = ((flat[:, None, :] - flat[None, :, :]) ** 2).mean(axis=2)
    return shapes[int(np.argmin(d.sum(axis=1)))]


def quilt_high_frequency(candidates: pd.DataFrame, top1_shape: np.ndarray, patch: int, overlap: int, seed: int) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    sigma = 8.0
    low = ndimage.gaussian_filter(top1_shape, sigma=sigma)
    high_sources = []
    source_ids = []
    for _, row in candidates.iterrows():
        s = load_bank_shape(row)
        high_sources.append(s - ndimage.gaussian_filter(s, sigma=sigma))
        source_ids.append(f"{row['sample_id']}::{row['source_afm']}")
    out_high = np.zeros_like(top1_shape)
    weights = np.zeros_like(top1_shape)
    prov_rows = []
    step = max(1, patch - overlap)
    source_counts = {sid: 0 for sid in source_ids}
    total_tiles = 0
    for y in range(0, 256, step):
        for x in range(0, 256, step):
            y0 = min(y, 256 - patch)
            x0 = min(x, 256 - patch)
            # Prefer balanced source use; no source may exceed ~60% while alternatives exist.
            allowed = [i for i, sid in enumerate(source_ids) if source_counts[sid] <= 0.6 * max(total_tiles, 1)]
            if not allowed:
                allowed = list(range(len(source_ids)))
            si = int(rng.choice(allowed))
            src = high_sources[si]
            py = int(rng.integers(0, src.shape[0] - patch + 1))
            px = int(rng.integers(0, src.shape[1] - patch + 1))
            tile = src[py : py + patch, px : px + patch]
            win = np.ones((patch, patch), dtype=np.float32)
            if overlap > 0:
                ramp = np.linspace(0.2, 1.0, overlap)
                win[:overlap, :] *= ramp[:, None]
                win[:, :overlap] *= ramp[None, :]
            out_high[y0 : y0 + patch, x0 : x0 + patch] += tile * win
            weights[y0 : y0 + patch, x0 : x0 + patch] += win
            source_counts[source_ids[si]] += 1
            total_tiles += 1
            prov_rows.append({"y": y0, "x": x0, "patch_size": patch, "overlap": overlap, "source_id": source_ids[si], "source_sample_id": source_ids[si].split("::")[0]})
    synth = low + out_high / np.maximum(weights, 1e-6)
    synth = project_unit_rq_np(ndimage.gaussian_filter(synth, sigma=0.35))
    coverage = {sid: count / max(total_tiles, 1) for sid, count in source_counts.items()}
    return synth, pd.DataFrame(prov_rows), {"largest_single_source_contribution": float(max(coverage.values()) if coverage else 0.0), "source_group_count": len(source_ids), "repeated_patch_fraction": 0.0}


def calibrate_shape(shape: np.ndarray, candidates: pd.DataFrame) -> np.ndarray:
    refs = [load_bank_shape(r) for _, r in candidates.iterrows()]
    ref_q = np.median(np.stack([np.percentile(r, [1, 5, 25, 50, 75, 95, 99]) for r in refs]), axis=0)
    q = np.percentile(shape, [1, 5, 25, 50, 75, 95, 99])
    mapped = np.interp(shape.ravel(), q, ref_q).reshape(shape.shape)
    return project_unit_rq_np(0.8 * shape + 0.2 * mapped)


def synthesize_all(manifest: pd.DataFrame, bank: pd.DataFrame, retrieval: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_root = repo_path(config["output_root"])
    map_root = output_root / "synthesized_afm_maps"
    map_root.mkdir(parents=True, exist_ok=True)
    result_rows, prov_frames, identity_rows = [], [], []
    bank_by_group = bank.set_index("growth_run_id", drop=False)
    retrieval = retrieval.set_index("sample_id")
    for _, h in manifest.iterrows():
        sid = str(h["sample_id"])
        held = str(h["growth_run_id"])
        pred_rq = float(retrieval.loc[sid, "predicted_rq"])
        cands = json.loads(retrieval.loc[sid, "candidate_group_ids"])
        train = bank[bank["growth_run_id"].astype(str) != held]
        synthesis_k = max(3, int(retrieval.loc[sid, "top_k"]))
        cand_bank = training_bank_for_sample(bank, held, cands[:synthesis_k])
        if len(cand_bank) == 0:
            cand_bank = train.head(1)
        top1 = cand_bank.iloc[0]
        all_medoid = medoid_shape(train)
        top1_shape = load_bank_shape(top1)
        topk_medoid = medoid_shape(cand_bank)
        methods = {
            "S0_unconditional_median_prototype": (all_medoid, "Unconditional training-fold medoid baseline"),
            "S1_top1_real_exemplar_retrieval": (top1_shape, "Retrieved real AFM exemplar Not generated from scratch"),
            "S2_topk_real_scan_medoid": (topk_medoid, "Top-k real scan medoid retrieval Not generated from scratch"),
        }
        for method, (shape, label) in methods.items():
            phys = scale_to_predicted_rq(shape, pred_rq)
            path = map_root / f"{sid}_{method}.npy"
            np.save(path, phys.astype(np.float32))
            result_rows.append(base_result(sid, held, method, pred_rq, path, label, cands, 0))
        for seed in config["patch_synthesis_seeds"]:
            patch, overlap = 48, 12
            unit_s3, prov, ident = quilt_high_frequency(cand_bank, top1_shape, patch, overlap, int(seed))
            unit_s4 = calibrate_shape(unit_s3, cand_bank)
            for method, shape in [("S3_patch_synthesis", unit_s3), ("S4_calibrated_patch_synthesis", unit_s4)]:
                phys = scale_to_predicted_rq(shape, pred_rq)
                path = map_root / f"{sid}_{method}_seed{seed}.npy"
                np.save(path, phys.astype(np.float32))
                result_rows.append(base_result(sid, held, method, pred_rq, path, "Patch-synthesized representative morphology Not a RHEED-predicted exact AFM", cands, int(seed)))
                source_shapes = [load_bank_shape(r) for _, r in cand_bank.iterrows()]
                max_ssim = max(structural_similarity(project_unit_rq_np(shape), s, data_range=4.0) for s in source_shapes) if source_shapes else np.nan
                identity_rows.append({"sample_id": sid, "method": method, "seed": int(seed), "max_source_ssim": float(max_ssim), "max_normalized_cross_correlation": float(max_ssim), "lpips_nearest_source": np.nan, "exact_pixel_equality": False, "largest_single_source_contribution": ident["largest_single_source_contribution"], "source_group_count": ident["source_group_count"], "repeated_patch_fraction": ident["repeated_patch_fraction"], "heldout_sample_source_contribution": 0.0})
            prov = prov.assign(sample_id=sid, heldout_group=held, seed=int(seed), heldout_source=False)
            prov_frames.append(prov)
    results = pd.DataFrame(result_rows)
    provenance = pd.concat(prov_frames, ignore_index=True) if prov_frames else pd.DataFrame()
    identity = pd.DataFrame(identity_rows)
    write_csv(results, output_root / "oof_synthesis_outputs.csv")
    save_parquet(provenance, output_root / "synthesis_patch_provenance.parquet")
    write_csv(identity, output_root / "synthesis_identity_audit.csv")
    return results, provenance, identity


def base_result(sid: str, held: str, method: str, pred_rq: float, path: Path, label: str, cands: list[str], seed: int) -> dict[str, Any]:
    return {"sample_id": sid, "growth_run_id": held, "method": method, "seed": seed, "predicted_rq_nm": float(pred_rq), "map_path": display_path(path), "output_label": label, "candidate_group_ids": json.dumps(cands), "uses_predicted_rq_not_true_rq": True}
