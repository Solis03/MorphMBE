from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from skimage.metrics import structural_similarity

from .afm_descriptors import descriptor_distance, describe_map
from .common import repo_path, write_csv
from .rq_disentanglement import project_unit_rq_np, rq_np


def eval_pair(true_phys: np.ndarray, pred_phys: np.ndarray, predicted_rq: float, true_rq: float) -> dict[str, float]:
    true_unit = project_unit_rq_np(true_phys)
    pred_unit = project_unit_rq_np(pred_phys)
    desc = descriptor_distance(true_unit, pred_unit)
    tdesc = describe_map(true_phys, "true")
    pdesc = describe_map(pred_phys, "pred")
    return {
        "predicted_rq_error_nm": abs(float(predicted_rq) - float(true_rq)),
        "measured_synth_rq_nm": rq_np(pred_phys),
        "synth_rq_minus_predicted_rq_nm": rq_np(pred_phys) - float(predicted_rq),
        "ra_error": abs(tdesc["true_ra"] - pdesc["pred_ra"]),
        "robust_height_range_error": abs(tdesc["true_robust_height_range"] - pdesc["pred_robust_height_range"]),
        "height_histogram_wasserstein": float(wasserstein_distance(true_phys.ravel(), pred_phys.ravel())),
        "ssim": float(structural_similarity(true_unit, pred_unit, data_range=4.0)),
        "lpips": np.nan,
        "morphology_prototype_agreement": np.nan,
        **desc,
    }


def evaluate_synthesis(manifest: pd.DataFrame, bank: pd.DataFrame, outputs: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    bank_map = bank.set_index("sample_id")
    rq_map = manifest.set_index("sample_id")["primary_rq_nm_median"].astype(float)
    rows = []
    for _, row in outputs.iterrows():
        sid = str(row["sample_id"])
        true = np.load(repo_path(bank_map.loc[sid, "physical_map_path"])).astype(np.float32)
        pred = np.load(repo_path(row["map_path"])).astype(np.float32)
        rows.append({"sample_id": sid, "method": row["method"], "seed": int(row["seed"]), "output_label": row["output_label"], **eval_pair(true, pred, float(row["predicted_rq_nm"]), float(rq_map.loc[sid]))})
    scan = pd.DataFrame(rows)
    summary = scan.groupby("method").median(numeric_only=True).reset_index()
    write_csv(scan, repo_path(config["output_root"]) / "synthesis_oof_metrics.csv")
    write_csv(summary, repo_path(config["output_root"]) / "synthesis_method_summary.csv")
    return scan, summary


def visual_plausibility_proxy(identity: pd.DataFrame, metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    s4 = metrics[metrics["method"] == "S4_calibrated_patch_synthesis"]
    id4 = identity[identity["method"] == "S4_calibrated_patch_synthesis"]
    rows = [{
        "method": "S4_calibrated_patch_synthesis",
        "physically_plausible_rate_proxy": float((s4["measured_synth_rq_nm"].notna()).mean()) if len(s4) else np.nan,
        "obvious_artifact_rate_proxy": float((id4["largest_single_source_contribution"] > 0.95).mean()) if len(id4) else np.nan,
        "obvious_seam_rate_proxy": float((id4["repeated_patch_fraction"] > 0.5).mean()) if len(id4) else np.nan,
        "requires_expert_blind_review": True,
    }]
    out = pd.DataFrame(rows)
    write_csv(out, repo_path(config["output_root"]) / "visual_plausibility_proxy.csv")
    return out
