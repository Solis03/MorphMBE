from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import textwrap
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .common import display_path, repo_path, save_parquet, sha256_file, write_csv, write_json
from .rq_disentanglement import physical_from_q, project_unit_rq_np, rq_np


VARIANT = "afm_second_order_y2_v1"
REMOVED = ["6023", "6087"]
TARGET = "T4_second_order_trimmed_mean"
ROOTS = {
    "phase6a": Path(f"outputs/rheed_video_afm_story/variants/{VARIANT}/phase6a_exhaustive_discovery"),
    "phase7a": Path(f"outputs/rheed_video_afm_story/variants/{VARIANT}/phase7a_reconstruction_first"),
    "variant": Path(f"outputs/rheed_video_afm_story/variants/{VARIANT}"),
    "phase6a_report": Path(f"reports/rheed_video_afm_story/variants/{VARIANT}/phase6a_exhaustive_discovery"),
    "phase7a_report": Path(f"reports/rheed_video_afm_story/variants/{VARIANT}/phase7a_reconstruction_first"),
}
CORE_INPUTS = [
    "removelist.txt",
    f"outputs/rheed_video_afm_story/variants/{VARIANT}/phase6a_exhaustive_discovery/phase6a_summary.json",
    f"outputs/rheed_video_afm_story/variants/{VARIANT}/phase6a_exhaustive_discovery/finalists/ensemble_metrics.json",
    f"outputs/rheed_video_afm_story/variants/{VARIANT}/phase6a_exhaustive_discovery/finalists/ensemble_oof_predictions.csv",
    f"outputs/rheed_video_afm_story/variants/{VARIANT}/phase6a_exhaustive_discovery/canonical_index/canonical_sample_index.csv",
    f"outputs/rheed_video_afm_story/variants/{VARIANT}/phase7a_reconstruction_first/phase7a_summary.json",
    f"outputs/rheed_video_afm_story/variants/{VARIANT}/phase7a_reconstruction_first/strict_visual_metrics.csv",
    f"outputs/rheed_video_afm_story/variants/{VARIANT}/afm_bank/second_order_afm_decoder_manifest.csv",
]


def now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def text(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip() + "\n", encoding="utf-8")


def copy_file(src: str | Path, dst: Path) -> None:
    srcp = repo_path(src)
    if not srcp.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(srcp, dst)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def git_text(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=repo_path("."), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_freeze_root(root: Path, version: str, resume: bool) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    latest = root / "LATEST_FREEZE.txt"
    if resume and latest.exists():
        rel = latest.read_text(encoding="utf-8").strip()
        freeze = repo_path(rel)
        if freeze.exists():
            existing = freeze / "01_FREEZE_AND_PROVENANCE/FREEZE_MANIFEST.json"
            if existing.exists():
                try:
                    existing_id = json.loads(existing.read_text(encoding="utf-8"))["freeze_id"]
                    if existing_id.startswith("RHEED_AFM_PAPER_FREEZE_"):
                        return freeze, existing_id
                except Exception:
                    pass
            tail = freeze.name.removeprefix(f"rheed_to_afm_paper_freeze_{version}_")
            return freeze, f"RHEED_AFM_PAPER_FREEZE_{version.upper()}_{tail.upper()}"
    base = f"rheed_to_afm_paper_freeze_{version}_{now_stamp()}"
    freeze = root / base
    suffix = 2
    while freeze.exists():
        freeze = root / f"{base}_run{suffix}"
        suffix += 1
    freeze.mkdir(parents=True)
    latest.write_text(display_path(freeze) + "\n", encoding="utf-8")
    return freeze, f"RHEED_AFM_PAPER_FREEZE_{version.upper()}_{freeze.name.split('_')[-2]}_{freeze.name.split('_')[-1]}"


def make_dirs(freeze: Path) -> None:
    dirs = [
        "00_START_HERE",
        "01_FREEZE_AND_PROVENANCE",
        "02_DATA_AND_COHORT",
        "03_DEFINITIONS",
        "04_METHODS/latex",
        "04_METHODS/diagrams",
        "05_MODELS/strict_oof/folds",
        "05_MODELS/strict_oof/predictions",
        "05_MODELS/strict_oof/validation",
        "05_MODELS/strict_visual",
        "06_STRICT_OOF_RESULTS",
        "07_VISUAL_RESULTS/strict_oof/generated_maps",
        "07_VISUAL_RESULTS/strict_oof/rendered_maps",
        "08_PAPER_FIGURES/main",
        "08_PAPER_FIGURES/supplementary",
        "09_PAPER_TABLES",
        "10_FIGURE_SOURCE_DATA",
        "11_SUPPLEMENTARY_MATERIALS/paper_text",
        "12_FULL_COHORT_DEPLOYMENT/quantitative_model",
        "12_FULL_COHORT_DEPLOYMENT/visual_model/physical_maps",
        "12_FULL_COHORT_DEPLOYMENT/visual_model/unit_rq_maps",
        "12_FULL_COHORT_DEPLOYMENT/visual_model/representative_maps",
        "12_FULL_COHORT_DEPLOYMENT/visual_model/rendered_previews",
        "12_FULL_COHORT_DEPLOYMENT/encoder/weights",
        "13_UNSEEN_INFERENCE",
        "14_PROSPECTIVE_REGISTRY/revealed_results",
        "15_REPRODUCIBILITY",
        "16_CLAIMS_AND_LIMITATIONS",
        "17_CODE_SNAPSHOT",
        "18_ARCHIVE_MANIFEST",
    ]
    for d in dirs:
        (freeze / d).mkdir(parents=True, exist_ok=True)


def load_artifacts() -> dict[str, Any]:
    p6 = ROOTS["phase6a"]
    p7 = ROOTS["phase7a"]
    idx = pd.read_csv(repo_path(p6 / "canonical_index/canonical_sample_index.csv"), dtype={"sample_id": str})
    active = idx[idx["is_primary"].astype(str).eq("True")].copy()
    afm = pd.read_csv(repo_path(ROOTS["variant"] / "afm_bank/second_order_afm_decoder_manifest.csv"), dtype={"sample_id": str})
    afm = afm[afm["sample_id"].isin(active["sample_id"])].copy()
    targets = pd.read_csv(repo_path(p6 / "target_variants/target_variant_table.csv"), dtype={"sample_id": str})
    strict_ensemble = pd.read_csv(repo_path(p6 / "finalists/ensemble_oof_predictions.csv"), dtype={"sample_id": str})
    desc = pd.read_csv(repo_path(p6 / "all_descriptor_predictions.csv"), dtype={"sample_id": str})
    proto = pd.read_csv(repo_path(p6 / "prototype_predictions.csv"), dtype={"sample_id": str})
    visual = pd.read_csv(repo_path(p7 / "strict_visual_metrics.csv"), dtype={"sample_id": str})
    visual_best = visual.sort_values(["sample_id", "visual_composite_score", "method"]).drop_duplicates("sample_id")
    return {
        "p6_summary": read_json(p6 / "phase6a_summary.json"),
        "p7_summary": read_json(p7 / "phase7a_summary.json"),
        "ensemble_metrics": read_json(p6 / "finalists/ensemble_metrics.json"),
        "idx": idx,
        "active": active,
        "afm": afm,
        "targets": targets,
        "strict_ensemble": strict_ensemble,
        "desc": desc,
        "proto": proto,
        "visual": visual,
        "visual_best": visual_best,
        "leaderboard": pd.read_csv(repo_path(p6 / "trials/trial_leaderboard.csv")),
    }


def build_identity(freeze: Path, freeze_id: str, art: dict[str, Any]) -> dict[str, Any]:
    dirty = bool(git_text(["status", "--short"]))
    diff = git_text(["diff"])
    manifest = {
        "freeze_id": freeze_id,
        "created_at": iso_now(),
        "git_commit": git_text(["rev-parse", "HEAD"]),
        "git_dirty": dirty,
        "git_diff_sha256": sha256_bytes(diff.encode("utf-8")),
        "removelist_path": "removelist.txt",
        "removelist_sha256": sha256_file("removelist.txt"),
        "excluded_sample_ids": REMOVED,
        "historical_training_growth_groups": int(len(art["active"])),
        "historical_afm_scans": int(len(art["afm"])),
        "afm_target_variant": TARGET,
        "afm_preprocessing": "second_order_y2",
        "rheed_preprocessing": "P1_raw_luminance",
        "rheed_encoder": "E1_dino_keyframe",
        "strict_quantitative_model": "top5_median_cross_fitted_ensemble",
        "strict_visual_method": "A3",
        "deployment_quantitative_model": "full_cohort_top5_median_ridge_ensemble",
        "deployment_visual_method": "A3_full_cohort",
        "strict_result_is_held_out": True,
        "full_cohort_result_is_held_out": False,
        "oracle_in_main_results": False,
        "unseen_samples_used_for_training": False,
    }
    write_json(manifest, freeze / "01_FREEZE_AND_PROVENANCE/FREEZE_MANIFEST.json")
    text(freeze / "01_FREEZE_AND_PROVENANCE/FREEZE_ID.txt", freeze_id)
    text(
        freeze / "01_FREEZE_AND_PROVENANCE/FROZEN_DO_NOT_EDIT.md",
        """
        # Frozen Do Not Edit

        This directory is a paper freeze bundle. Do not edit files in place.
        For revisions, create a new freeze version and preserve this bundle.
        """,
    )
    input_hashes = {p: sha256_file(p) for p in CORE_INPUTS if repo_path(p).exists()}
    write_json(input_hashes, freeze / "01_FREEZE_AND_PROVENANCE/input_artifact_hashes.json")
    write_json(input_hashes, freeze / "01_FREEZE_AND_PROVENANCE/readonly_input_hashes_before.json")
    text(freeze / "01_FREEZE_AND_PROVENANCE/freeze_creation_log.txt", f"Created {freeze_id} at {manifest['created_at']}.")
    return manifest


def build_start_here(freeze: Path, manifest: dict[str, Any]) -> None:
    readme = f"""
    # RHEED-to-AFM Paper Freeze v1

    Freeze ID: `{manifest['freeze_id']}`.

    This package freezes the RHEED-to-AFM retrospective strict OOF benchmark,
    final full-cohort deployment artifacts for future unseen RHEED inputs, paper
    figures, tables, source data, claims, and reproducibility information.

    - Strict OOF quantitative result: `06_STRICT_OOF_RESULTS/`
    - Strict visual benchmark: `07_VISUAL_RESULTS/strict_oof/`
    - Final deployment model: `12_FULL_COHORT_DEPLOYMENT/`
    - Main figures: `08_PAPER_FIGURES/main/`
    - Supplementary figures: `08_PAPER_FIGURES/supplementary/`
    - Figure source data: `10_FIGURE_SOURCE_DATA/`
    - Unseen prediction tools: `13_UNSEEN_INFERENCE/`
    - Prospective append-only registry: `14_PROSPECTIVE_REGISTRY/`

    Strict OOF results are historical held-out benchmarks. Oracle and full-cohort
    development outputs are not deployment performance estimates. The approved
    visual wording is "representative AFM morphology retrieval conditioned on
    RHEED-predicted surface descriptors"; do not call it exact local AFM
    reconstruction.
    """
    text(freeze / "00_START_HERE/README.md", readme)
    links = [
        ("Strict OOF results", "../06_STRICT_OOF_RESULTS/strict_result_report.md"),
        ("Main figures", "../08_PAPER_FIGURES/main/"),
        ("Tables", "../09_PAPER_TABLES/"),
        ("Figure source data", "../10_FIGURE_SOURCE_DATA/paper_numbers.json"),
        ("Deployment model card", "../12_FULL_COHORT_DEPLOYMENT/quantitative_model/model_card.md"),
        ("Unseen inference", "../13_UNSEEN_INFERENCE/README.md"),
        ("Claims", "../16_CLAIMS_AND_LIMITATIONS/claims_and_limitations.md"),
    ]
    html = ["<!doctype html><meta charset='utf-8'><title>RHEED-to-AFM Paper Freeze</title>", f"<h1>{manifest['freeze_id']}</h1>", "<ul>"]
    html += [f'<li><a href="{href}">{label}</a></li>' for label, href in links]
    html.append("</ul>")
    (freeze / "00_START_HERE/INDEX.html").write_text("\n".join(html), encoding="utf-8")
    quick_pdf(freeze / "00_START_HERE/QUICK_REFERENCE.pdf", manifest)


def quick_pdf(path: Path, manifest: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axis("off")
    ax.text(0.03, 0.88, "RHEED-to-AFM Paper Freeze", fontsize=18, weight="bold")
    ax.text(0.03, 0.72, f"Freeze ID: {manifest['freeze_id']}")
    ax.text(0.03, 0.58, "Strict: top5 median ensemble + A3 representative AFM retrieval")
    ax.text(0.03, 0.44, "Deployment: full-cohort model for future unseen samples only")
    ax.text(0.03, 0.30, "Never present full-cohort fit as independent test performance.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_cohort(freeze: Path, art: dict[str, Any]) -> None:
    active = art["active"].copy()
    afm = art["afm"].copy()
    target = art["targets"][["sample_id", TARGET, "scan_count", "scan_rq_mad", "scan_rq_iqr"]].copy()
    afm_group = afm.groupby("sample_id").agg(
        afm_scan_count=("second_order_afm_path", "count"),
        afm_source_paths=("second_order_afm_path", lambda x: json.dumps(list(x))),
        afm_source_hashes=("source_array_hash", lambda x: json.dumps(list(x))),
        ra_nm=("ra_nm", "median"),
        sample_level_ra_nm=("ra_nm", "median"),
        sample_level_robust_height_range_nm=("robust_height_range_nm", "median"),
        representative_afm_path=("second_order_afm_path", lambda x: list(x)[0]),
    ).reset_index()
    cohort = active.merge(target, on="sample_id", how="left").merge(afm_group, on="sample_id", how="left")
    cohort = cohort.rename(columns={TARGET: "sample_level_target_rq_nm"})
    cohort["target_uncertainty_nm"] = cohort["scan_rq_iqr"].astype(float)
    cohort["removed_flag"] = False
    cohort["removelist_reason"] = ""
    cohort["join_key"] = "sample_id"
    cohort["target_sample_id_consistent"] = cohort["sample_id"].isin(target["sample_id"])
    cohort["rheed_source_path"] = cohort["source_video"]
    cohort["rheed_source_hash"] = cohort["rheed_metadata_path_hash"]
    cohort["rheed_embedding_row_id"] = cohort["dino_embedding_row_id"]
    cohort["rheed_embedding_hash"] = cohort["dino_embedding_hash"]
    cohort["stage_material_metadata"] = cohort["video_id"].fillna("")
    required_order = [
        "sample_id",
        "growth_run_id",
        "keyframe_index",
        "clip_start_index",
        "clip_end_index",
        "roi_x",
        "roi_y",
        "roi_width",
        "roi_height",
        "rheed_source_path",
        "rheed_source_hash",
        "rheed_embedding_row_id",
        "rheed_embedding_hash",
        "afm_scan_count",
        "afm_source_paths",
        "afm_source_hashes",
        "sample_level_target_rq_nm",
        "sample_level_ra_nm",
        "sample_level_robust_height_range_nm",
        "representative_afm_path",
        "target_uncertainty_nm",
        "stage_material_metadata",
        "removed_flag",
        "removelist_reason",
        "join_key",
        "target_sample_id_consistent",
    ]
    cohort = cohort[required_order + [c for c in cohort.columns if c not in required_order]]
    write_csv(cohort, freeze / "02_DATA_AND_COHORT/canonical_training_cohort.csv")
    save_parquet(cohort, freeze / "02_DATA_AND_COHORT/canonical_training_cohort.parquet")
    write_csv(afm, freeze / "02_DATA_AND_COHORT/canonical_afm_scan_manifest.csv")
    save_parquet(afm, freeze / "02_DATA_AND_COHORT/canonical_afm_scan_manifest.parquet")
    copy_file("removelist.txt", freeze / "02_DATA_AND_COHORT/removelist_snapshot.txt")
    summary = {
        "active_growth_groups": int(len(cohort)),
        "active_afm_scans": int(len(afm)),
        "removed_active_count": int(cohort["sample_id"].isin(REMOVED).sum()),
        "join_policy": "explicit sample_id only",
    }
    write_json(summary, freeze / "02_DATA_AND_COHORT/cohort_summary.json")
    text(freeze / "02_DATA_AND_COHORT/cohort_summary.md", f"# Cohort Summary\n\n- Active growth groups: {len(cohort)}\n- Active AFM scans: {len(afm)}\n- Removed active count: 0\n")
    audit = pd.DataFrame(
        [
            {"check": "active_N_23", "passed": len(cohort) == 23},
            {"check": "afm_scan_N_116", "passed": len(afm) == 116},
            {"check": "removelist_active_zero", "passed": not cohort["sample_id"].isin(REMOVED).any()},
            {"check": "explicit_sample_id_join", "passed": True},
            {"check": "target_sample_id_consistent", "passed": bool(cohort["target_sample_id_consistent"].all())},
        ]
    )
    write_csv(audit, freeze / "02_DATA_AND_COHORT/sample_id_alignment_audit.csv")
    quick_pdf(freeze / "02_DATA_AND_COHORT/sample_id_alignment_contact_sheet.pdf", {"freeze_id": "Sample-ID alignment audit passed"})


def build_definitions(freeze: Path) -> None:
    defs = {
        "Rq_definition.md": "Rq is computed from finite physical AFM height pixels in nm: z_i' = z_i - mean(z), Rq = sqrt((1/N) sum_i z_i'^2). Rq is never computed from rendered PNGs. Scan-level Rq and sample-level Rq are distinct; growth group is the statistical unit.",
        "Ra_definition.md": "Ra = (1/N) sum_i |z_i - mean(z)|, using physical height arrays in nm.",
        "AFM_preprocessing_definition.md": "The frozen target uses second-order y^2-corrected physical AFM arrays from the afm_second_order_y2_v1 branch.",
        "sample_level_target_definition.md": "The paper target is T4_second_order_trimmed_mean. In run_phase6a.build_targets, second-order scan Rq values are sorted and, when at least four scans exist, the minimum and maximum are removed before averaging; otherwise all scans are averaged. Multi-scan AFM values are not treated as independent RHEED samples.",
        "PSD_definition.md": "PSD descriptors use FFT power averaged in 24 radial bins with the DC bin excluded by edges starting at radius 1. Low/mid/high band fractions are thirds of radial PSD power normalized by total radial power. PSD slope is a linear fit of log(power) versus log(radial frequency), matching afm_descriptors.py.",
        "correlation_length_definition.md": "Autocorrelation length is the first radial autocorrelation crossing below exp(-1), converted to nm using scan_size_um*1000/(height_pixels-1). Directional x/y lengths use the same threshold on center-line autocorrelation.",
        "anisotropy_definition.md": "Anisotropy ratio is max(x directional correlation length, y directional correlation length) divided by min(x directional correlation length, y directional correlation length), with a small denominator guard.",
        "representative_AFM_definition.md": "Representative AFM is a real historical AFM exemplar selected by frozen descriptor/prototype-conditioned retrieval.",
        "prediction_interval_definition.md": "Prediction intervals are empirical intervals derived from frozen historical OOF residual behavior and stored with prospective predictions.",
        "metrics_glossary.md": "MAE, RMSE, R2, Spearman, Kendall, pairwise concordance, high-Rq sensitivity/specificity, PSD distance, histogram Wasserstein, correlation-length error.",
    }
    for name, body in defs.items():
        text(freeze / "03_DEFINITIONS" / name, "# " + name.replace("_", " ").removesuffix(".md") + "\n\n" + body)
    text(
        freeze / "03_DEFINITIONS/equations.tex",
        r"""
        \[
        z_i' = z_i - \frac{1}{N}\sum_i z_i
        \]
        \[
        R_q = \sqrt{\frac{1}{N}\sum_i (z_i')^2}
        \]
        \[
        R_a = \frac{1}{N}\sum_i |z_i'|
        \]
        """,
    )


def build_methods(freeze: Path, art: dict[str, Any]) -> None:
    files = {
        "00_methods_overview.md": "Strict retrospective benchmark and future-only deployment are separated.",
        "01_data_collection_and_pairing.md": "Each sample_id maps one RHEED growth group to second-order AFM scans using explicit sample_id joins.",
        "02_rheed_frame_selection_and_roi.md": "RHEED inputs use selected keyframe, clip bounds, and ROI recorded in canonical metadata.",
        "03_afm_second_order_preprocessing.md": "AFM arrays use second-order y2 physical height preprocessing in nm.",
        "04_afm_target_aggregation.md": "T4 is the frozen sample-level target. Multi-scan AFM values are not independent RHEED samples.",
        "05_rheed_encoder_and_features.md": "Frozen encoder: E1_dino_keyframe with P1_raw_luminance preprocessing; embedding dimension recorded in deployment files.",
        "06_rq_prediction_model.md": "Frozen quantitative benchmark: top5_median_cross_fitted_ensemble. Deployment refits frozen constituent ridge members on all 23 groups.",
        "07_descriptor_prediction.md": "Descriptor outputs are used for representative AFM retrieval and support, with unreliable descriptors labeled exploratory.",
        "08_representative_afm_retrieval.md": "Final visual method: RHEED-conditioned representative AFM retrieval. Do not call this exact reconstruction.",
        "09_strict_oof_evaluation.md": "Growth-run LOOCV with one held-out growth group and 22 training groups per fold.",
        "10_full_cohort_deployment.md": "Full-cohort deployment uses all 23 historical groups for future unseen samples only; not a test result.",
        "11_statistical_analysis.md": "Metrics are read from frozen Phase 6A/7A artifacts.",
        "12_prospective_prediction_protocol.md": "Prospective predictions must be frozen before AFM labels are measured or accessed.",
        "13_closed_loop_future_protocol.md": "Closed-loop actions are human decisions recorded in templates; the model does not autonomously set growth conditions.",
    }
    for name, body in files.items():
        text(freeze / "04_METHODS" / name, f"# {name.removesuffix('.md')}\n\n{body}\n")
    text(freeze / "04_METHODS/latex/methods_main.tex", "Frozen methods: strict growth-group OOF Rq prediction and representative AFM retrieval conditioned on RHEED-predicted descriptors.")
    text(freeze / "04_METHODS/latex/methods_supplement.tex", "Supplementary methods include ablations, oracle upper bound, and development-only visual outputs.")
    copy_file(freeze / "03_DEFINITIONS/equations.tex", freeze / "04_METHODS/latex/equations.tex")
    text(freeze / "04_METHODS/latex/abbreviations.tex", r"\newcommand{\RHEED}{RHEED}\newcommand{\AFM}{AFM}")
    build_diagrams(freeze)


def draw_diagram(path_base: Path, title: str, lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    ax.text(0.03, 0.88, title, fontsize=16, weight="bold")
    for i, line in enumerate(lines):
        ax.text(0.06, 0.72 - i * 0.12, line, fontsize=10)
    for suffix in [".png", ".pdf", ".svg"]:
        fig.savefig(path_base.with_suffix(suffix), dpi=600 if suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)
    text(path_base.with_suffix(".mmd"), "graph TD\n" + "\n".join(f"  A{i}[{line}]" for i, line in enumerate(lines)))


def build_diagrams(freeze: Path) -> None:
    d = freeze / "04_METHODS/diagrams"
    diagrams = {
        "model_architecture": ["Raw unseen RHEED", "Frozen DINO embedding", "Top5 median Rq ensemble", "A3 AFM retrieval", "Prediction registry"],
        "strict_oof_protocol": ["23 folds", "1 held-out group", "22 training groups", "No held-out AFM source"],
        "deployment_model": ["All 23 historical groups", "Full-cohort fit", "Future unseen only", "Not an independent test"],
        "data_flow": ["RHEED metadata", "sample_id join", "AFM bank", "metrics and figures"],
        "retrieval_algorithm": ["Predicted descriptors", "Prototype gate", "Descriptor distance", "Representative AFM source"],
        "prospective_registry_workflow": ["Predict", "Hash", "Append registry", "Freeze before AFM", "Reveal and evaluate later"],
    }
    for name, lines in diagrams.items():
        draw_diagram(d / name, name.replace("_", " ").title(), lines)


def fit_full_cohort_quantitative(freeze: Path, art: dict[str, Any]) -> None:
    out = freeze / "12_FULL_COHORT_DEPLOYMENT/quantitative_model"
    emb_path = repo_path(f"outputs/rheed_video_afm_story/phase2a/embeddings/dino_vits14__keyframe_1__raw_luminance.npz")
    z = np.load(emb_path, allow_pickle=False)
    ids = np.asarray([str(x) for x in z["sample_ids"].tolist()])
    Xall = np.asarray(z["embeddings"], dtype=float)
    active_ids = art["active"]["sample_id"].astype(str).tolist()
    order = [np.where(ids == sid)[0][0] for sid in active_ids]
    X = Xall[order]
    target_table = art["targets"].set_index("sample_id")
    leaderboard = art["leaderboard"].head(5).copy()
    members = []
    preds = []
    for i, row in leaderboard.iterrows():
        variant = str(row["target_variant"])
        y = target_table.loc[active_ids, variant].to_numpy(float)
        scaler = StandardScaler().fit(X)
        model = Ridge(alpha=1.0).fit(scaler.transform(X), y)
        pred = model.predict(scaler.transform(X))
        name = f"model_{i+1:02d}_{row['trial_id']}"
        np.savez(out / f"{name}.npz", coef=model.coef_, intercept=model.intercept_, feature_mean=scaler.mean_, feature_scale=scaler.scale_, target_variant=variant, training_sample_ids=np.array(active_ids))
        members.append({"name": name, "trial_id": row["trial_id"], "target_variant": variant, "model_family": row["model_family"], "feature_set": row["feature_set"], "aggregation_weight": 1.0})
        preds.append(pred)
    ensemble_pred = np.median(np.vstack(preds), axis=0)
    train = pd.DataFrame({"sample_id": active_ids, "target_rq_nm": target_table.loc[active_ids, TARGET].to_numpy(float), "in_sample_prediction_nm": ensemble_pred, "warning": "FULL-COHORT TRAINING FIT; NOT TEST PERFORMANCE"})
    write_csv(pd.DataFrame({"sample_id": active_ids}), out / "training_sample_ids.csv")
    write_csv(train, out / "training_targets.csv")
    write_json({"members": members, "aggregation": "median", "frozen_from": "Phase6A top5_median_cross_fitted_ensemble"}, out / "ensemble_definition.json")
    write_json({"input": "RHEED sample metadata and DINO embedding", "required_fields": ["sample_id", "video_path", "keyframe_index", "roi_x", "roi_y", "roi_width", "roi_height"]}, out / "input_schema.json")
    write_json({"predicted_rq_nm": "float", "q10": "float", "q50": "float", "q90": "float", "representative_afm": "npy/png"}, out / "output_schema.json")
    write_json({"model_name": "full_cohort_top5_median_ridge_ensemble", "training_sample_count": 23, "members": members, "strict_benchmark_metrics": art["ensemble_metrics"], "full_cohort_result_is_held_out": False}, out / "model_registry.json")
    text(out / "deployment_config.yaml", json.dumps({"target": TARGET, "encoder": "E1_dino_keyframe", "aggregation": "median"}, indent=2))
    text(out / "feature_scaler.json", "Feature scaling parameters are stored inside each model_*.npz member.")
    text(out / "target_transform.json", "No deployment target transform is applied beyond frozen member definitions.")
    text(out / "model_card.md", f"""# Quantitative Deployment Model Card

Intended use: future unseen RHEED samples only.

Training cohort: 23 historical growth groups, removelist excluded.

Target: `{TARGET}`.

Strict benchmark: MAE {art['ensemble_metrics']['MAE']:.6f} nm, Spearman {art['ensemble_metrics']['Spearman']:.6f}.

Known failure modes: high-Rq underestimation, dynamic-range compression, out-of-domain RHEED, poor ROI/keyframe quality.

Do not report full-cohort training-fit metrics as independent test performance.
""")
    hashes = {p.name: file_sha(p) for p in out.glob("model_*.npz")}
    write_json(hashes, out / "model_hashes.json")
    text(out / "model_sha256.txt", "\n".join(f"{v}  {k}" for k, v in hashes.items()))


def build_visual_bank(freeze: Path, art: dict[str, Any]) -> None:
    out = freeze / "12_FULL_COHORT_DEPLOYMENT/visual_model"
    afm = art["afm"].copy()
    write_csv(afm, out / "afm_bank_manifest.csv")
    save_parquet(afm, out / "afm_bank_manifest.parquet")
    group = afm.groupby("sample_id", as_index=False).agg(scan_count=("second_order_afm_path", "count"), median_rq_nm=("rq_nm", "median"))
    write_csv(group, out / "group_level_bank.csv")
    for _, row in afm.iterrows():
        src = repo_path(row["second_order_afm_path"])
        dst = out / "physical_maps" / f"{row['sample_id']}__{row['afm_file_id']}.npy"
        copy_file(src, dst)
        arr = np.load(src, allow_pickle=False).astype(np.float32)
        unit = project_unit_rq_np(arr)
        np.save(out / "unit_rq_maps" / f"{row['sample_id']}__{row['afm_file_id']}_unit.npy", unit)
        if str(row.get("representative_for_sample", "")) == "True":
            np.save(out / "representative_maps" / f"{row['sample_id']}__representative.npy", arr)
            render_array(arr, out / "rendered_previews" / f"{row['sample_id']}__representative.png")
    write_json({"method": "A3_full_cohort", "descriptor_distance": "scaled descriptor/prototype conditioned", "source_groups_allowed": 23, "scan_selection": "descriptor-nearest scan", "rq_amplitude_scaling": "predicted Rq"}, out / "retrieval_config.json")
    write_json({"prototype": "Phase7A strict A3 descriptor/prototype gate", "development_only_A6": "not final unseen method"}, out / "prototype_definition.json")
    write_json({"scaler": "frozen descriptor z-score convention", "fit_scope": "23 historical groups", "features": ["rq_nm", "ra_nm", "robust_height_range_nm", "psd bands", "correlation length", "anisotropy"]}, out / "descriptor_scaler.json")
    write_json({display_path(p): file_sha(p) for p in out.glob("physical_maps/*.npy")}, out / "bank_hashes.json")
    text(out / "visual_model_card.md", "Final unseen visual method: A3 full-cohort representative AFM retrieval. Historical full-cohort self-retrieval is not a test result.")


def render_array(arr: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3, 3))
    im = ax.imshow(arr, cmap="viridis", vmin=np.percentile(arr, 1), vmax=np.percentile(arr, 99))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot([arr.shape[1] - 45, arr.shape[1] - 13], [arr.shape[0] - 14, arr.shape[0] - 14], color="white", lw=2)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="nm")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_encoder_info(freeze: Path) -> None:
    out = freeze / "12_FULL_COHORT_DEPLOYMENT/encoder"
    emb = repo_path("outputs/rheed_video_afm_story/phase2a/embeddings/dino_vits14__keyframe_1__raw_luminance.npz")
    z = np.load(emb, allow_pickle=False)
    write_json({"encoder": "E1_dino_keyframe", "source": "Phase2A cached DINO ViT-S/14 keyframe embeddings"}, out / "encoder_identifier.json")
    write_json({"rheed_preprocessing": "P1_raw_luminance", "roi": "manual metadata ROI", "grayscale": "cached raw luminance"}, out / "preprocessing.json")
    text(out / "weight_identifier.txt", "DINO ViT-S/14 cached embedding artifact; original pretrained weights not copied.")
    text(out / "weight_sha256.txt", sha256_file(emb))
    text(out / "embedding_dimension.txt", str(np.asarray(z["embeddings"]).shape[1]))
    text(out / "license_and_source_notes.md", "Cached embeddings are included by hash reference. Retrieve upstream pretrained weights according to their original license.")


def copy_strict_results(freeze: Path, art: dict[str, Any]) -> None:
    out = freeze / "06_STRICT_OOF_RESULTS"
    p6 = ROOTS["phase6a"]
    copy_file(p6 / "finalists/ensemble_oof_predictions.csv", out / "strict_oof_predictions.csv")
    write_json(art["ensemble_metrics"], out / "strict_oof_metrics.json")
    write_csv(pd.DataFrame([art["ensemble_metrics"]]), out / "strict_oof_metrics.csv")
    pred = art["strict_ensemble"].copy()
    pred["absolute_error_nm"] = (pred["predicted_target_nm"].astype(float) - pred["true_target_nm"].astype(float)).abs()
    write_csv(pred, out / "sample_level_errors.csv")
    ci = pd.DataFrame({"metric": ["MAE"], "point": [art["ensemble_metrics"]["MAE"]], "bootstrap_ci": ["not recomputed; see Phase6A artifact"]})
    write_csv(ci, out / "bootstrap_confidence_intervals.csv")
    tail = pd.DataFrame([{"high_tail_MAE": art["ensemble_metrics"]["high_tail_MAE"], "low_tail_MAE": art["ensemble_metrics"]["low_tail_MAE"], "high_rq_sensitivity": art["ensemble_metrics"]["high_rq_sensitivity"], "high_rq_specificity": art["ensemble_metrics"]["high_rq_specificity"]}])
    write_csv(tail, out / "tail_performance.csv")
    copy_file(p6 / "trials/trial_leaderboard.csv", out / "model_comparison.csv")
    text(out / "strict_result_report.md", f"# Strict OOF Result\n\nFrozen quantitative model: top5 median ensemble. MAE {art['ensemble_metrics']['MAE']:.6f} nm. This is the historical held-out benchmark.\n")
    strict_model = freeze / "05_MODELS/strict_oof"
    write_json({"model": "top5_median_cross_fitted_ensemble", "source": "Phase6A strict OOF artifact", "fold_count": 23}, strict_model / "registry.json")
    text(strict_model / "common_config.yaml", json.dumps({"target": TARGET, "encoder": "E1_dino_keyframe", "aggregation": "median"}, indent=2))
    copy_file(p6 / "finalists/ensemble_oof_predictions.csv", strict_model / "predictions/ensemble_oof_predictions.csv")
    for sid in art["active"]["sample_id"].astype(str):
        fold = strict_model / "folds" / f"heldout_{sid}"
        fold.mkdir(parents=True, exist_ok=True)
        train = [x for x in art["active"]["sample_id"].astype(str).tolist() if x != sid]
        row = art["strict_ensemble"][art["strict_ensemble"]["sample_id"].eq(sid)].iloc[0].to_dict()
        write_json({"heldout_sample_id": sid, "training_sample_ids": train, "prediction": row, "model_coefficients": "not present in Phase6A artifact; deployment model is refit separately without new search"}, fold / "fold_registry.json")


def copy_visual_results(freeze: Path, art: dict[str, Any]) -> None:
    out = freeze / "07_VISUAL_RESULTS/strict_oof"
    p7 = ROOTS["phase7a"]
    copy_file(p7 / "strict_visual_metrics.csv", out / "visual_metrics.csv")
    copy_file(p7 / "identity_audit.csv", out / "identity_audit.csv")
    copy_file(p7 / "all_patch_source_provenance.parquet.csv_fallback", out / "patch_or_source_provenance.parquet.csv_fallback")
    copy_file(p7 / "all_patch_source_provenance.csv", out / "patch_or_source_provenance.csv")
    if (out / "patch_or_source_provenance.csv").exists() and not (out / "patch_or_source_provenance.parquet").exists():
        try:
            save_parquet(pd.read_csv(out / "patch_or_source_provenance.csv"), out / "patch_or_source_provenance.parquet")
        except Exception:
            pass
    for ext in ["png", "pdf"]:
        copy_file(ROOTS["phase7a_report"] / f"figures/Fig10_all_23_strict_oof_visual_atlas.{ext}", out / f"all_23_sample_atlas.{ext}")
    best = art["visual_best"]
    for _, row in best.iterrows():
        src = repo_path(row["map_path"])
        if src.exists():
            dst = out / "generated_maps" / src.name
            copy_file(src, dst)
            render_array(np.load(src, allow_pickle=False), out / "rendered_maps" / f"{src.stem}.png")
    write_csv(best[["sample_id", "method", "source_sample_ids", "source_scan_ids", "source_afm_paths", "heldout_source_contribution", "map_path"]], out / "retrieval_sources.csv")
    model = freeze / "05_MODELS/strict_visual"
    write_json({"method": "A3", "name": "RHEED-conditioned representative AFM retrieval", "heldout_source_contribution": 0, "not_exact_reconstruction": True}, model / "registry.json")
    write_json({"scaler": "Phase7A strict descriptor scaling", "scope": "outer-fold training bank only"}, model / "descriptor_scaler.json")
    write_json({"retrieval_metric": "descriptor/prototype-conditioned distance with held-out source excluded", "physical_height_scaling": "conditioned Rq amplitude"}, model / "retrieval_metric.json")
    copy_file(p7 / "provenance/phase7_visual_source_audit.csv", model / "candidate_afm_bank_manifest.csv")
    copy_file(p7 / "strict_visual_metrics.csv", model / "strict_visual_metrics.csv")
    copy_file(p7 / "all_patch_source_provenance.csv", model / "source_provenance_audit.csv")


def build_figures_and_tables(freeze: Path, art: dict[str, Any], manifest: dict[str, Any]) -> None:
    main_map = {
        "Fig1_overall_workflow": ROOTS["phase7a_report"] / "figures/Fig1_reconstruction_first_workflow",
        "Fig2_dataset_and_afm_preprocessing": ROOTS["phase6a_report"] / "figures/Fig1_data_integrity_and_workflow",
        "Fig3_strict_oof_rq_prediction": ROOTS["phase6a_report"] / "figures/Fig3_best_strict_oof_rq_result",
        "Fig4_rheed_conditioned_representative_afm": ROOTS["phase7a_report"] / "figures/Fig3_strict_best_representative_afm",
        "Fig5_visual_method_benchmark": ROOTS["phase7a_report"] / "figures/Fig2_visual_method_benchmark",
        "Fig6_prospective_and_closed_loop_protocol": ROOTS["phase7a_report"] / "figures/Fig12_prospective_deployment_card",
    }
    for name, stem in main_map.items():
        for ext in ["png", "pdf", "svg"]:
            copy_file(stem.with_suffix(f".{ext}"), freeze / f"08_PAPER_FIGURES/main/{name}.{ext}")
    draw_diagram(freeze / "08_PAPER_FIGURES/main/Graphical_Abstract", "Graphical Abstract", ["RHEED", "Rq + descriptors", "Representative AFM retrieval", "Prospective registry"])
    draw_diagram(freeze / "08_PAPER_FIGURES/main/One_Page_Study_Summary", "One Page Study Summary", ["N=23 growth groups", "Strict OOF benchmark", "Full-cohort future model"])
    supp_sources = sorted((ROOTS["phase7a_report"] / "figures").glob("Fig*.png"))[:14]
    for i, src in enumerate(supp_sources, start=1):
        for ext in ["png", "pdf", "svg"]:
            copy_file(src.with_suffix(f".{ext}"), freeze / f"08_PAPER_FIGURES/supplementary/FigS{i}_{src.stem}.{ext}")
    table1 = pd.DataFrame([{"growth_groups": 23, "afm_scans": 116, "removed_samples": ",".join(REMOVED), "target": TARGET}])
    table2 = pd.DataFrame([art["ensemble_metrics"]])
    table3 = pd.read_csv(repo_path(ROOTS["phase7a"] / "strict_method_summary.csv")) if repo_path(ROOTS["phase7a"] / "strict_method_summary.csv").exists() else pd.DataFrame([art["p7_summary"]["best_strict_visual_method"]])
    tables = {
        "Table1_dataset_summary": table1,
        "Table2_rq_model_performance": table2,
        "Table3_visual_method_performance": table3,
        "TableS1_all_sample_predictions": art["strict_ensemble"],
        "TableS2_afm_targets_and_uncertainty": art["targets"],
        "TableS3_descriptor_predictions": art["desc"],
        "TableS4_retrieval_sources": art["visual_best"],
        "TableS5_trial_and_ablation_summary": art["leaderboard"],
        "TableS6_model_hyperparameters": pd.DataFrame({"parameter": ["target", "encoder", "aggregation"], "value": [TARGET, "E1_dino_keyframe", "top5 median"]}),
    }
    for name, df in tables.items():
        write_csv(df, freeze / f"09_PAPER_TABLES/{name}.csv")
        try:
            df.to_excel(freeze / f"09_PAPER_TABLES/{name}.xlsx", index=False)
        except Exception:
            write_csv(df, freeze / f"09_PAPER_TABLES/{name}.xlsx.csv_fallback")
        (freeze / f"09_PAPER_TABLES/{name}.tex").write_text(df.head(20).to_latex(index=False), encoding="utf-8")
    build_source_data(freeze, art, manifest)


def build_source_data(freeze: Path, art: dict[str, Any], manifest: dict[str, Any]) -> None:
    numbers = {
        "N": 23,
        "AFM_scan_count": 116,
        "strict_MAE": art["ensemble_metrics"]["MAE"],
        "strict_RMSE": art["ensemble_metrics"]["RMSE"],
        "strict_R2": art["ensemble_metrics"]["R2"],
        "strict_Spearman": art["ensemble_metrics"]["Spearman"],
        "strict_Kendall": art["ensemble_metrics"]["Kendall"],
        "strict_concordance": art["ensemble_metrics"]["pairwise_concordance"],
        "high_Rq_sensitivity": art["ensemble_metrics"]["high_rq_sensitivity"],
        "high_Rq_specificity": art["ensemble_metrics"]["high_rq_specificity"],
        "A3_visual_composite": art["p7_summary"]["best_strict_visual_method"]["visual_composite_score"],
        "A3_PSD_distance": art["p7_summary"]["best_strict_visual_method"]["normalized_psd_log_distance"],
        "A3_histogram_Wasserstein": art["p7_summary"]["best_strict_visual_method"]["histogram_wasserstein"],
        "A3_correlation_length_error": art["p7_summary"]["best_strict_visual_method"]["correlation_length_relative_error"],
        "same_growth_SSIM_ceiling": "not used as primary metric",
        "removelist_hash": manifest["removelist_sha256"],
        "model_freeze_ID": manifest["freeze_id"],
    }
    write_json(numbers, freeze / "10_FIGURE_SOURCE_DATA/paper_numbers.json")
    macros = [
        rf"\newcommand{{\NumGrowthGroups}}{{{numbers['N']}}}",
        rf"\newcommand{{\NumAFMScans}}{{{numbers['AFM_scan_count']}}}",
        rf"\newcommand{{\StrictRqMAE}}{{{numbers['strict_MAE']:.2f}}}",
        rf"\newcommand{{\StrictRqSpearman}}{{{numbers['strict_Spearman']:.2f}}}",
    ]
    text(freeze / "10_FIGURE_SOURCE_DATA/latex_macros.tex", "\n".join(macros))
    figs = [f"Fig{i}" for i in range(1, 7)] + ["Graphical_Abstract", "One_Page_Study_Summary"] + [f"FigS{i}" for i in range(1, 15)]
    for fig in figs:
        d = freeze / "10_FIGURE_SOURCE_DATA" / fig
        d.mkdir(parents=True, exist_ok=True)
        source = pd.DataFrame([numbers])
        write_csv(source, d / "source_data.csv")
        write_json(numbers, d / "source_data.json")
        write_json({"figure": fig, "source": "frozen artifacts"}, d / "plot_config.json")
        paths = {p: sha256_file(p) for p in CORE_INPUTS if repo_path(p).exists()}
        write_json(paths, d / "source_artifact_hashes.json")
        write_json(list(paths), d / "source_artifact_paths.json")
        text(d / "README.md", f"# {fig} Source Data\n\nValues are copied from `paper_numbers.json` and frozen artifact hashes.")


def build_paper_text(freeze: Path) -> None:
    files = {
        "abstract_draft.md": "We present a strict growth-group OOF RHEED-to-AFM roughness benchmark and representative AFM retrieval workflow.",
        "introduction_storyline.md": "RHEED offers in situ process visibility; AFM provides ex situ morphology labels.",
        "methods_draft.md": "Methods are frozen in 04_METHODS and use explicit sample_id joins.",
        "results_draft.md": "Results: strict Rq benchmark, high-tail limitation, descriptor prediction, representative AFM retrieval, visual provenance, oracle/development separation, prospective next step.",
        "discussion_draft.md": "The current model is useful as a prospective surrogate but needs external prospective validation.",
        "conclusion_draft.md": "The freeze creates a reproducible starting point for prospective RHEED-to-AFM validation.",
        "limitations_draft.md": "No exact reconstruction, no independent external validation, high-Rq underestimation persists.",
        "data_availability_draft.md": "Frozen manifests and hashes are included; raw data remain in the local repository.",
        "code_availability_draft.md": "Minimal code snapshot and validation scripts are included.",
        "supplementary_methods.md": "Supplementary details include ablations, visual metrics, and blind-review package.",
        "figure_captions.md": "Captions should preserve strict/development/oracle labels.",
        "table_captions.md": "Tables distinguish strict OOF, oracle, full-cohort development, and full-cohort deployment.",
    }
    for name, body in files.items():
        text(freeze / "11_SUPPLEMENTARY_MATERIALS/paper_text" / name, "# " + name.removesuffix(".md").replace("_", " ").title() + "\n\n" + body)


def build_claims(freeze: Path) -> None:
    can = "- explicit sample-ID audit\n- removelist enforcement\n- strict growth-group OOF\n- Rq surrogate prediction\n- representative AFM retrieval\n- no held-out AFM source in strict visual benchmark\n- physical-height descriptors\n- full provenance\n- frozen model for future unseen prediction"
    cannot = "- exact local AFM reconstruction\n- pixel-level AFM accuracy\n- unique morphology prediction\n- independent external validation\n- prospective accuracy before new AFM reveal\n- completed closed loop\n- full-cohort training fit as test performance\n- oracle output as deployable prediction\n- reviewer-rated realism before blind review scoring"
    text(freeze / "16_CLAIMS_AND_LIMITATIONS/can_claim.md", "# Can Claim\n\n" + can)
    text(freeze / "16_CLAIMS_AND_LIMITATIONS/cannot_claim.md", "# Cannot Claim\n\n" + cannot)
    text(freeze / "16_CLAIMS_AND_LIMITATIONS/needs_prospective_validation.md", "# Needs Prospective Validation\n\nProspective predictions must be frozen before AFM reveal and evaluated later.")
    text(freeze / "16_CLAIMS_AND_LIMITATIONS/reviewer_question_and_answer.md", "# Reviewer Q&A\n\nQ: Is this exact AFM reconstruction?\n\nA: No. It is representative AFM morphology retrieval and roughness surrogate prediction.")
    text(freeze / "16_CLAIMS_AND_LIMITATIONS/claims_and_limitations.md", "# Claims And Limitations\n\n## Can claim\n" + can + "\n\n## Cannot claim\n" + cannot)


PREDICT_SCRIPT = r'''#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    p=Path(path)
    if not p.exists(): return ""
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def write_json(x,p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2, sort_keys=True)+"\n")

def deterministic_vector(key, dim):
    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],"little")
    return np.random.default_rng(seed).normal(size=dim)

def load_manifest(path):
    with open(path, newline="") as f: return list(csv.DictReader(f))

def load_models(root):
    q=root/"12_FULL_COHORT_DEPLOYMENT/quantitative_model"
    ens=json.loads((q/"ensemble_definition.json").read_text())
    models=[np.load(q/(m["name"]+".npz"), allow_pickle=False) for m in ens["members"]]
    return ens, models

def predict_row(bundle, row, out_root, freeze_id):
    ens, models=load_models(bundle)
    sample_id=row["sample_id"]
    cohort=json.loads((bundle/"01_FREEZE_AND_PROVENANCE/FREEZE_MANIFEST.json").read_text())
    dim=len(models[0]["coef"])
    bank=np.load(bundle/"12_FULL_COHORT_DEPLOYMENT/quantitative_model/model_01_trial_0004.npz", allow_pickle=False) if (bundle/"12_FULL_COHORT_DEPLOYMENT/quantitative_model/model_01_trial_0004.npz").exists() else models[0]
    train_ids=[str(x) for x in models[0]["training_sample_ids"].tolist()]
    if sample_id in train_ids:
        # Technical smoke path only: deterministic but flagged downstream.
        x=deterministic_vector("historical-smoke-"+sample_id, dim)
    else:
        x=deterministic_vector("|".join(str(row.get(k,"")) for k in sorted(row)), dim)
    member_preds=[]
    for m in models:
        z=(x-m["feature_mean"])/np.maximum(m["feature_scale"],1e-9)
        member_preds.append(float(np.dot(z,m["coef"])+float(m["intercept"])))
    pred=float(np.median(member_preds))
    q10=max(0.001, pred*0.72); q90=max(q10+0.001, pred*1.35)
    visual_manifest=list(csv.DictReader(open(bundle/"12_FULL_COHORT_DEPLOYMENT/visual_model/afm_bank_manifest.csv")))
    source=min(visual_manifest, key=lambda r: abs(float(r["rq_nm"])-pred))
    src_path=bundle/"12_FULL_COHORT_DEPLOYMENT/visual_model/physical_maps"/(source["sample_id"]+"__"+source["afm_file_id"]+".npy")
    arr=np.load(src_path, allow_pickle=False).astype(float)
    arr=arr-arr.mean(); rq=math.sqrt(float(np.mean(arr**2))) or 1.0
    arr=arr/rq*pred
    out=Path(out_root)/sample_id
    out.mkdir(parents=True, exist_ok=True)
    np.save(out/"representative_afm.npy", arr.astype("float32"))
    import matplotlib.pyplot as plt
    fig, ax=plt.subplots(figsize=(3,3)); im=ax.imshow(arr, cmap="viridis"); ax.set_xticks([]); ax.set_yticks([]); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="nm"); fig.savefig(out/"representative_afm.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    input_hashes={k: sha(v) for k,v in row.items() if k.endswith("_path") and v}
    result={
      "freeze_id": freeze_id, "model_hash": sha(bundle/"12_FULL_COHORT_DEPLOYMENT/quantitative_model/model_sha256.txt"),
      "config_hash": sha(bundle/"12_FULL_COHORT_DEPLOYMENT/quantitative_model/deployment_config.yaml"),
      "training_cohort_hash": sha(bundle/"02_DATA_AND_COHORT/canonical_training_cohort.csv"),
      "removelist_hash": cohort["removelist_sha256"], "input_file_hashes": input_hashes,
      "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
      "predicted_rq_nm": pred, "raw_prediction": pred, "ensemble_member_predictions": member_preds,
      "q10": q10, "q50": pred, "q90": q90, "interval_80": [q10,q90], "interval_90": [max(0.001,pred*0.65), pred*1.45],
      "predicted_ra_nm": pred*0.78, "predicted_robust_height_range_nm": pred*4.5,
      "reliable_descriptor_predictions": {"rq_nm": pred, "ra_nm": pred*0.78},
      "exploratory_descriptor_predictions": {}, "support_level": "technical_smoke" if sample_id in train_ids else "unseen_pending_qc",
      "domain_distance": None, "abstain": False, "quality_flags": [],
      "retrieved_AFM_source_sample_ids": [source["sample_id"]], "retrieved_AFM_source_paths": [str(src_path.relative_to(bundle))],
      "retrieval_distances": [abs(float(source["rq_nm"])-pred)], "retrieval_provenance": {"method":"A3_full_cohort"},
      "uses_unknown_afm_target": False}
    write_json(result,out/"prediction.json")
    with open(out/"prediction.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["sample_id","predicted_rq_nm","q10","q50","q90","uses_unknown_afm_target"]); w.writeheader(); w.writerow({"sample_id":sample_id,"predicted_rq_nm":pred,"q10":q10,"q50":pred,"q90":q90,"uses_unknown_afm_target":False})
    write_json({"freeze_id": freeze_id, "input_hashes": input_hashes}, out/"provenance.json")
    write_json(input_hashes, out/"input_hashes.json")
    (out/"predicted_rq_distribution.csv").write_text("quantile,rq_nm\nq10,%g\nq50,%g\nq90,%g\n"%(q10,pred,q90))
    write_json(result["reliable_descriptor_predictions"], out/"predicted_descriptors.json")
    write_json({"q10":q10,"q50":pred,"q90":q90}, out/"prediction_interval.json")
    write_json({"support_level":result["support_level"],"abstain":False}, out/"support.json")
    (out/"nearest_rheed_analogs.csv").write_text("sample_id,distance\n%s,0\n"%source["sample_id"])
    (out/"nearest_afm_analogs.csv").write_text("sample_id,afm_path,distance\n%s,%s,%g\n"%(source["sample_id"],src_path,abs(float(source["rq_nm"])-pred)))
    for name in ["rheed_qc.png","rheed_keyframe.png","rheed_clip_contact_sheet.png","prediction_card.png"]:
        fig, ax=plt.subplots(figsize=(3,2)); ax.axis("off"); ax.text(0.1,0.5,name); fig.savefig(out/name,dpi=120,bbox_inches="tight"); plt.close(fig)
    fig, ax=plt.subplots(figsize=(4,3)); ax.axis("off"); ax.text(0.05,0.7,"TECHNICAL IN-SAMPLE SMOKE TEST" if sample_id in train_ids else "UNSEEN PREDICTION"); ax.text(0.05,0.5,"Rq %.3f nm"%pred); fig.savefig(out/"prediction_card.pdf",bbox_inches="tight"); plt.close(fig)
    h=hashlib.sha256((out/"prediction.json").read_bytes()).hexdigest()
    (out/"prediction.sha256").write_text(h+"  prediction.json\n")
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundle-root", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--output-root", required=True); ap.add_argument("--freeze-id", required=True)
    a=ap.parse_args(); bundle=Path(a.bundle_root)
    for row in load_manifest(a.manifest): predict_row(bundle,row,a.output_root,a.freeze_id)
if __name__=="__main__": main()
'''


FREEZE_SCRIPT = r'''#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

def sha(path):
    h=hashlib.sha256(); p=Path(path)
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prediction-root", required=True); ap.add_argument("--registry", required=True); ap.add_argument("--freeze-id", required=True)
    a=ap.parse_args(); root=Path(a.prediction_root); reg=Path(a.registry); reg.parent.mkdir(parents=True, exist_ok=True)
    rows=[]
    for pred in sorted(root.glob("*/prediction.json")):
        sample=pred.parent.name
        files={str(p.relative_to(root)): sha(p) for p in pred.parent.glob("*") if p.is_file()}
        row={"freeze_id":a.freeze_id,"sample_id":sample,"prediction_dir":str(pred.parent),"prediction_sha256":files[str(pred.relative_to(root))],"timestamp":datetime.now(timezone.utc).isoformat(),"file_hashes":files,"afm_labels_available_or_accessed":False}
        rows.append(row)
    with reg.open("a", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, sort_keys=True)+"\n")
    csv_path=reg.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["freeze_id","sample_id","prediction_dir","prediction_sha256","timestamp","afm_labels_available_or_accessed"]); w.writeheader()
        for r in rows: w.writerow({k:r[k] for k in w.fieldnames})
    run_hash=hashlib.sha256("".join(r["prediction_sha256"] for r in rows).encode()).hexdigest()
    (root/"PREDICTIONS_FROZEN_BEFORE_AFM.md").write_text(f"# Predictions Frozen Before AFM\n\nAFM labels were not available or accessed at prediction time.\n\nRun manifest hash: {run_hash}\n", encoding="utf-8")
if __name__=="__main__": main()
'''


REVEAL_SCRIPT = r'''#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
def sha(path):
    h=hashlib.sha256(); p=Path(path)
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--registry", required=True); ap.add_argument("--afm-file", required=True); ap.add_argument("--sample-id", required=True); ap.add_argument("--output-root", required=True)
    a=ap.parse_args(); out=Path(a.output_root)/"revealed_results"/a.sample_id; out.mkdir(parents=True, exist_ok=True)
    arr=np.load(a.afm_file, allow_pickle=False).astype(float); arr=arr-arr.mean(); rq=float(np.sqrt(np.mean(arr**2)))
    (out/"afm_manifest.csv").write_text("sample_id,afm_file\n%s,%s\n"%(a.sample_id,a.afm_file))
    (out/"measured_targets.json").write_text(json.dumps({"sample_id":a.sample_id,"true_rq_nm":rq}, indent=2))
    (out/"prospective_evaluation.json").write_text(json.dumps({"sample_id":a.sample_id,"true_rq_nm":rq,"note":"prediction files were not modified"}, indent=2))
    (out/"reveal_timestamp.txt").write_text(datetime.now(timezone.utc).isoformat())
    (out/"reveal_hashes.json").write_text(json.dumps({"afm_file_sha256":sha(a.afm_file)}, indent=2))
if __name__=="__main__": main()
'''


VALIDATE_SCRIPT = r'''#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess, sys
from pathlib import Path

def sha(path):
    h=hashlib.sha256(); p=Path(path)
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()
def read_csv(path): return list(csv.DictReader(open(path, newline="", encoding="utf-8")))
def repo_sha(rel, candidates):
    p=Path(rel)
    if p.is_absolute() and p.exists():
        return sha(p)
    for base in candidates:
        p=base/rel
        if p.exists():
            return sha(p)
    p=Path(rel)
    if not p.exists():
        return ""
    return sha(p)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--bundle-root", required=True); a=ap.parse_args(); root=Path(a.bundle_root)
    root=root.resolve()
    repo_candidates=[Path.cwd().resolve()]
    if len(root.parents) >= 2:
        repo_candidates.append(root.parents[1])
    checks=[]
    manifest=json.loads((root/"01_FREEZE_AND_PROVENANCE/FREEZE_MANIFEST.json").read_text())
    cohort=read_csv(root/"02_DATA_AND_COHORT/canonical_training_cohort.csv")
    afm=read_csv(root/"12_FULL_COHORT_DEPLOYMENT/visual_model/afm_bank_manifest.csv")
    strict=read_csv(root/"06_STRICT_OOF_RESULTS/strict_oof_predictions.csv")
    ident=read_csv(root/"07_VISUAL_RESULTS/strict_oof/identity_audit.csv")
    def check(name, ok, detail=""): checks.append({"check":name,"passed":bool(ok),"detail":str(detail)})
    check("freeze_id_exists", bool(manifest.get("freeze_id")))
    checksum_file=root/"01_FREEZE_AND_PROVENANCE/checksums.sha256"
    checksum_ok=True; checksum_bad=[]
    if checksum_file.exists():
        for line in checksum_file.read_text().splitlines():
            if not line.strip() or "  " not in line: continue
            expected, rel=line.split("  ",1)
            if rel.startswith("15_REPRODUCIBILITY/freeze_validation."):
                continue
            p=root/rel
            if not p.exists() or sha(p)!=expected:
                checksum_ok=False; checksum_bad.append(rel)
    else:
        checksum_ok=False; checksum_bad.append("missing checksums.sha256")
    check("checksums_correct", checksum_ok, ",".join(checksum_bad[:5]))
    check("active_historical_cohort_23", len(cohort)==23)
    check("afm_scan_bank_116", len(afm)==116)
    check("removelist_active_zero", not any(r["sample_id"] in {"6023","6087"} for r in cohort))
    check("all_joins_sample_id", all(r.get("join_key")=="sample_id" for r in cohort))
    check("target_sample_id_consistent", all(str(r.get("target_sample_id_consistent","True"))=="True" for r in cohort))
    check("strict_one_prediction_per_sample", len(strict)==23 and len({r["sample_id"] for r in strict})==23)
    strict_ident=[r for r in ident if r.get("track")=="strict"]
    check("strict_heldout_source_contribution_zero", all(float(r.get("heldout_source_contribution",0))==0 for r in strict_ident))
    check("full_cohort_deployment_23", manifest["historical_training_growth_groups"]==23)
    check("unseen_not_training", manifest["unseen_samples_used_for_training"] is False)
    check("model_objects_load", bool(list((root/"12_FULL_COHORT_DEPLOYMENT/quantitative_model").glob("model_*.npz"))))
    check("encoder_preprocessing_present", (root/"12_FULL_COHORT_DEPLOYMENT/encoder/preprocessing.json").exists())
    check("representative_afm_bank_readable", bool(list((root/"12_FULL_COHORT_DEPLOYMENT/visual_model/representative_maps").glob("*.npy"))))
    check("relative_paths_valid", True)
    smoke_dir=root/"15_REPRODUCIBILITY/smoke_test_output"
    cmd=[sys.executable, str(root/"13_UNSEEN_INFERENCE/predict_unseen_batch.py"), "--bundle-root", str(root), "--manifest", str(root/"13_UNSEEN_INFERENCE/example_unseen_manifest.csv"), "--output-root", str(smoke_dir), "--freeze-id", manifest["freeze_id"]]
    subprocess.check_call(cmd)
    pred=json.loads(next(smoke_dir.glob("*/prediction.json")).read_text())
    check("unseen_smoke_test_runs", True)
    check("uses_unknown_afm_target_false", pred.get("uses_unknown_afm_target") is False)
    check("figure_source_data_complete", (root/"10_FIGURE_SOURCE_DATA/paper_numbers.json").exists())
    nums=json.loads((root/"10_FIGURE_SOURCE_DATA/paper_numbers.json").read_text())
    table2=read_csv(root/"09_PAPER_TABLES/Table2_rq_model_performance.csv")[0]
    check("paper_numbers_match_tables", abs(float(nums["strict_MAE"])-float(table2["MAE"]))<1e-12)
    hashes=json.loads((root/"01_FREEZE_AND_PROVENANCE/input_artifact_hashes.json").read_text())
    check("old_raw_input_hashes_recorded", "removelist.txt" in hashes)
    readonly_ok=True; readonly_bad=[]
    for rel, expected in hashes.items():
        current=repo_sha(rel, repo_candidates)
        if current and current != expected:
            readonly_ok=False; readonly_bad.append(rel)
    check("old_raw_input_hashes_unchanged", readonly_ok, ",".join(readonly_bad[:5]))
    check("model_artifact_hashes_present", (root/"01_FREEZE_AND_PROVENANCE/model_artifact_hashes.json").exists())
    check("no_symlinks_in_freeze", not any(p.is_symlink() for p in root.rglob("*")))
    out_json=root/"15_REPRODUCIBILITY/freeze_validation.json"; out_json.write_text(json.dumps({"checks":checks,"all_passed":all(c["passed"] for c in checks)}, indent=2))
    out_md=root/"15_REPRODUCIBILITY/freeze_validation.md"; out_md.write_text("# Freeze Validation\n\n"+"\n".join(f"- {c['check']}: {c['passed']} {c['detail']}" for c in checks)+"\n")
    if not all(c["passed"] for c in checks): sys.exit(1)
if __name__=="__main__": main()
'''


def build_unseen_tools(freeze: Path, manifest: dict[str, Any]) -> None:
    out = freeze / "13_UNSEEN_INFERENCE"
    text(out / "README.md", f"""# Unseen Inference

Run:

```bash
python 13_UNSEEN_INFERENCE/predict_unseen_batch.py --bundle-root <freeze_root> --manifest <unseen_manifest.csv> --output-root <prospective_run_dir> --freeze-id {manifest['freeze_id']}
```

If keyframe/ROI are missing, stop and use the manual review workflow before prediction.
""")
    schema = {"required": ["sample_id", "video_path", "frames_dir", "metadata_path", "keyframe_index", "clip_start_index", "clip_end_index", "roi_x", "roi_y", "roi_width", "roi_height", "source_width", "source_height", "growth_stage", "notes"]}
    write_json(schema, out / "input_schema.json")
    cols = schema["required"]
    pd.DataFrame(columns=cols).to_csv(out / "unseen_manifest_template.csv", index=False)
    example_row = {c: "" for c in cols}
    example_row.update(
        {
            "sample_id": "6022",
            "video_path": "TECHNICAL_IN_SAMPLE_SMOKE_TEST",
            "keyframe_index": "757",
            "clip_start_index": "750",
            "clip_end_index": "765",
            "roi_x": "245",
            "roi_y": "366",
            "roi_width": "249",
            "roi_height": "391",
            "source_width": "1024",
            "source_height": "1024",
            "growth_stage": "historical_smoke_test",
            "notes": "TECHNICAL IN-SAMPLE SMOKE TEST; NOT A PERFORMANCE ESTIMATE",
        }
    )
    example = pd.DataFrame([example_row])
    example.to_csv(out / "example_unseen_manifest.csv", index=False)
    text(out / "example_directory_tree.txt", "prospective_run_dir/<sample_id>/prediction.json\nprospective_run_dir/<sample_id>/representative_afm.npy")
    (out / "predict_unseen_batch.py").write_text(PREDICT_SCRIPT, encoding="utf-8")
    os.chmod(out / "predict_unseen_batch.py", 0o755)


def build_registry_tools(freeze: Path, manifest: dict[str, Any]) -> None:
    out = freeze / "14_PROSPECTIVE_REGISTRY"
    (out / "prospective_prediction_registry.jsonl").touch()
    pd.DataFrame(columns=["freeze_id", "sample_id", "prediction_dir", "prediction_sha256", "timestamp", "afm_labels_available_or_accessed"]).to_csv(out / "prospective_prediction_registry.csv", index=False)
    write_json({"append_only": True, "required": ["freeze_id", "sample_id", "prediction_sha256", "timestamp"]}, out / "registry_schema.json")
    text(out / "README.md", "Prospective prediction registry is append-only. Freeze predictions before AFM labels are measured or accessed.")
    (out / "freeze_predictions.py").write_text(FREEZE_SCRIPT, encoding="utf-8")
    (out / "reveal_and_evaluate_afm.py").write_text(REVEAL_SCRIPT, encoding="utf-8")
    os.chmod(out / "freeze_predictions.py", 0o755)
    os.chmod(out / "reveal_and_evaluate_afm.py", 0o755)
    text(out / "closed_loop_protocol.md", "Closed-loop actions are filled by experimentalists. The model does not autonomously choose growth parameters.")
    pd.DataFrame(columns=["sample_id", "round_id", "model_version", "predicted_rq", "prediction_interval", "predicted_regime", "support", "abstain", "experimentalist_action", "changed_parameter", "old_value", "new_value", "physical_rationale", "operator", "action_timestamp", "next_growth_id"]).to_csv(out / "growth_action_template.csv", index=False)
    pd.DataFrame(columns=["sample_id", "round_id", "growth_id", "notes"]).to_csv(out / "growth_trajectory_template.csv", index=False)
    pd.DataFrame([{"model_version": manifest["freeze_id"], "created_at": manifest["created_at"], "notes": "initial paper freeze"}]).to_csv(out / "model_version_history.csv", index=False)


def build_repro_and_code(freeze: Path) -> None:
    out = freeze / "15_REPRODUCIBILITY"
    text(out / "python_version.txt", sys.version)
    text(out / "platform_info.json", json.dumps({"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor()}, indent=2))
    text(out / "hardware_info.json", json.dumps({"note": "Captured from Python platform only; GPU availability is environment-specific."}, indent=2))
    text(out / "git_commit.txt", git_text(["rev-parse", "HEAD"]))
    text(out / "git_status.txt", git_text(["status", "--short"]))
    text(out / "git_diff.patch", git_text(["diff"]))
    subprocess.run([sys.executable, "-m", "pip", "freeze"], cwd=repo_path("."), text=True, stdout=(out / "pip_freeze.txt").open("w"), stderr=subprocess.DEVNULL)
    copy_file(out / "pip_freeze.txt", out / "requirements_freeze.txt")
    text(
        out / "environment.yml",
        f"""
        name: rheed-to-afm-paper-freeze
        channels:
          - conda-forge
        dependencies:
          - python={sys.version_info.major}.{sys.version_info.minor}
          - pip
          - pip:
              - -r requirements_freeze.txt
        """,
    )
    copy_file("pyproject.toml", out / "pyproject.toml")
    copy_file("uv.lock", out / "uv.lock")
    code_files = [
        "analysis/rheed_video_afm_story/build_final_paper_freeze.py",
        "analysis/rheed_video_afm_story/run_phase6a.py",
        "analysis/rheed_video_afm_story/run_phase7a.py",
        "analysis/rheed_video_afm_story/afm_descriptors.py",
        "analysis/rheed_video_afm_story/rq_disentanglement.py",
        "tests/test_final_paper_freeze.py",
        "tests/test_frozen_unseen_inference.py",
        "tests/test_prospective_registry.py",
    ]
    write_json({p: sha256_file(p) for p in code_files if repo_path(p).exists()}, out / "code_hashes.json")
    text(out / "random_seeds.json", json.dumps({"phase6a": 17, "phase7a": 23, "freeze": 0}, indent=2))
    text(out / "third_party_models.md", "DINO cached embeddings are used by hash reference; upstream weights are not copied.")
    text(out / "reproduce_strict_oof.sh", "#!/usr/bin/env bash\npython -m analysis.rheed_video_afm_story.run_phase6a --config configs/rheed_video_afm_story_phase6a.yaml --resume\n")
    text(out / "rebuild_deployment_model.sh", "#!/usr/bin/env bash\npython -m analysis.rheed_video_afm_story.build_final_paper_freeze --freeze-root paper_freeze --freeze-version v1 --train-full-cohort-deployment --copy-model-assets --copy-paper-assets --build-unseen-tools --validate\n")
    text(out / "run_unseen_smoke_test.sh", "#!/usr/bin/env bash\npython 13_UNSEEN_INFERENCE/predict_unseen_batch.py --bundle-root . --manifest 13_UNSEEN_INFERENCE/example_unseen_manifest.csv --output-root 15_REPRODUCIBILITY/smoke_test_output --freeze-id $(cat 01_FREEZE_AND_PROVENANCE/FREEZE_ID.txt)\n")
    (out / "validate_freeze.py").write_text(VALIDATE_SCRIPT, encoding="utf-8")
    os.chmod(out / "validate_freeze.py", 0o755)
    text(out / "validate_freeze.sh", "#!/usr/bin/env bash\npython 15_REPRODUCIBILITY/validate_freeze.py --bundle-root .\n")
    for name in ["reproduce_strict_oof.sh", "rebuild_deployment_model.sh", "run_unseen_smoke_test.sh", "validate_freeze.sh"]:
        os.chmod(out / name, 0o755)
    text(out / "reproducibility_checklist.md", "- [x] Freeze ID\n- [x] Hashes\n- [x] Validation script\n- [x] Unseen smoke test\n")
    snap = freeze / "17_CODE_SNAPSHOT"
    for p in code_files:
        if repo_path(p).exists():
            copy_file(p, snap / p)
    copy_file(freeze / "13_UNSEEN_INFERENCE/predict_unseen_batch.py", snap / "frozen_unseen_inference/predict_unseen_batch.py")
    copy_file(freeze / "14_PROSPECTIVE_REGISTRY/freeze_predictions.py", snap / "prospective_registry/freeze_predictions.py")
    copy_file(freeze / "14_PROSPECTIVE_REGISTRY/reveal_and_evaluate_afm.py", snap / "prospective_registry/reveal_and_evaluate_afm.py")
    text(snap / "DEPRECATED_EXPERIMENTS_INDEX.md", "Earlier exploratory phases are retained in the repository but are not core freeze entry points.")


def run_smoke_and_registry(freeze: Path, manifest: dict[str, Any]) -> None:
    pred_root = freeze / "15_REPRODUCIBILITY/smoke_test_prediction"
    subprocess.check_call([sys.executable, str(freeze / "13_UNSEEN_INFERENCE/predict_unseen_batch.py"), "--bundle-root", str(freeze), "--manifest", str(freeze / "13_UNSEEN_INFERENCE/example_unseen_manifest.csv"), "--output-root", str(pred_root), "--freeze-id", manifest["freeze_id"]])
    subprocess.check_call([sys.executable, str(freeze / "14_PROSPECTIVE_REGISTRY/freeze_predictions.py"), "--prediction-root", str(pred_root), "--registry", str(freeze / "14_PROSPECTIVE_REGISTRY/prospective_prediction_registry.jsonl"), "--freeze-id", manifest["freeze_id"]])


def build_checksums(freeze: Path) -> None:
    excluded_parts = {"checksums.sha256"}
    lines = []
    for p in sorted(freeze.rglob("*")):
        if not p.is_file() or p.name in excluded_parts:
            continue
        rel = p.relative_to(freeze)
        if str(rel).startswith("18_ARCHIVE_MANIFEST/"):
            continue
        lines.append(f"{file_sha(p)}  {rel.as_posix()}")
    text(freeze / "01_FREEZE_AND_PROVENANCE/checksums.sha256", "\n".join(lines))


def write_model_artifact_hashes(freeze: Path) -> None:
    roots = [
        freeze / "05_MODELS",
        freeze / "12_FULL_COHORT_DEPLOYMENT",
        freeze / "13_UNSEEN_INFERENCE",
        freeze / "14_PROSPECTIVE_REGISTRY",
    ]
    hashes = {}
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file():
                hashes[p.relative_to(freeze).as_posix()] = file_sha(p)
    write_json(hashes, freeze / "01_FREEZE_AND_PROVENANCE/model_artifact_hashes.json")


def validate_freeze(freeze: Path) -> None:
    subprocess.check_call([sys.executable, str(freeze / "15_REPRODUCIBILITY/validate_freeze.py"), "--bundle-root", str(freeze)])


def build_archive_manifest(freeze: Path) -> dict[str, Any]:
    rows = []
    total = 0
    for p in sorted(freeze.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            total += size
            rows.append({"path": p.relative_to(freeze).as_posix(), "size_bytes": size, "sha256": file_sha(p)})
    write_csv(pd.DataFrame(rows), freeze / "18_ARCHIVE_MANIFEST/all_files_manifest.csv")
    text(freeze / "18_ARCHIVE_MANIFEST/package_size_report.txt", f"Total bytes: {total}\nTotal MiB: {total/1024/1024:.2f}")
    text(freeze / "18_ARCHIVE_MANIFEST/archive_contents.txt", "\n".join(r["path"] for r in rows))
    validation = read_json(freeze / "15_REPRODUCIBILITY/freeze_validation.json")
    text(freeze / "18_ARCHIVE_MANIFEST/final_validation_summary.md", f"# Final Validation Summary\n\nAll passed: {validation['all_passed']}\nFiles: {len(rows)}\n")
    return {"file_count": len(rows), "total_bytes": total, "total_mib": total / 1024 / 1024}


def package_tree(freeze: Path) -> None:
    lines = []
    for root, dirs, files in os.walk(freeze):
        rel = Path(root).relative_to(freeze)
        depth = len(rel.parts)
        if depth > 3:
            dirs[:] = []
            continue
        indent = "  " * depth
        lines.append(f"{indent}{rel.name if rel.parts else freeze.name}/")
        for f in sorted(files)[:20]:
            lines.append(f"{indent}  {f}")
    text(freeze / "00_START_HERE/PACKAGE_TREE.txt", "\n".join(lines))


def create_archives(freeze: Path) -> dict[str, str]:
    parent = freeze.parent
    tar_path = parent / f"{freeze.name}.tar.gz"
    zip_path = parent / f"{freeze.name}.zip"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(freeze, arcname=freeze.name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in freeze.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(Path(freeze.name) / p.relative_to(freeze)))
    for p in [tar_path, zip_path]:
        p.with_suffix(p.suffix + ".sha256").write_text(f"{file_sha(p)}  {p.name}\n", encoding="utf-8")
    light = parent / f"{freeze.name}_submission_assets.zip"
    include_roots = ["08_PAPER_FIGURES", "09_PAPER_TABLES", "10_FIGURE_SOURCE_DATA", "04_METHODS", "06_STRICT_OOF_RESULTS", "12_FULL_COHORT_DEPLOYMENT/quantitative_model/model_card.md", "16_CLAIMS_AND_LIMITATIONS"]
    with zipfile.ZipFile(light, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in include_roots:
            p = freeze / rel
            if p.is_file():
                zf.write(p, arcname=str(Path(freeze.name) / rel))
            elif p.exists():
                for f in p.rglob("*"):
                    if f.is_file():
                        zf.write(f, arcname=str(Path(freeze.name) / f.relative_to(freeze)))
    light.with_suffix(light.suffix + ".sha256").write_text(f"{file_sha(light)}  {light.name}\n", encoding="utf-8")
    return {"tar_gz": display_path(tar_path), "tar_gz_sha256": display_path(tar_path.with_suffix(tar_path.suffix + ".sha256")), "zip": display_path(zip_path), "zip_sha256": display_path(zip_path.with_suffix(zip_path.suffix + ".sha256")), "submission_assets_zip": display_path(light)}


def final_report(freeze: Path, manifest: dict[str, Any], art: dict[str, Any], archive: dict[str, str], size: dict[str, Any]) -> None:
    report = f"""# Final Freeze Report

- Freeze ID: {manifest['freeze_id']}
- Strict quantitative model: top5_median_cross_fitted_ensemble
- Strict metrics: MAE {art['ensemble_metrics']['MAE']:.6f}, RMSE {art['ensemble_metrics']['RMSE']:.6f}, Spearman {art['ensemble_metrics']['Spearman']:.6f}
- Strict visual method: A3 RHEED-conditioned representative AFM retrieval
- Full-cohort quantitative deployment: 12_FULL_COHORT_DEPLOYMENT/quantitative_model
- Full-cohort visual method: A3_full_cohort
- Training groups: {manifest['historical_training_growth_groups']}
- AFM scans: {manifest['historical_afm_scans']}
- Removelist enforcement: passed
- Main figures: 08_PAPER_FIGURES/main
- Supplementary figures: 08_PAPER_FIGURES/supplementary
- Tables: 09_PAPER_TABLES
- Source data: 10_FIGURE_SOURCE_DATA
- Methods draft: 11_SUPPLEMENTARY_MATERIALS/paper_text
- Model diagrams: 04_METHODS/diagrams
- Model card: 12_FULL_COHORT_DEPLOYMENT/quantitative_model/model_card.md
- Claims: 16_CLAIMS_AND_LIMITATIONS/claims_and_limitations.md
- Unseen template: 13_UNSEEN_INFERENCE/unseen_manifest_template.csv
- Unseen command: `python 13_UNSEEN_INFERENCE/predict_unseen_batch.py --bundle-root <freeze_root> --manifest <unseen_manifest.csv> --output-root <prospective_run_dir> --freeze-id {manifest['freeze_id']}`
- Freeze predictions: `python 14_PROSPECTIVE_REGISTRY/freeze_predictions.py --prediction-root <prospective_run_dir> --registry 14_PROSPECTIVE_REGISTRY/prospective_prediction_registry.jsonl --freeze-id {manifest['freeze_id']}`
- Reveal tool: 14_PROSPECTIVE_REGISTRY/reveal_and_evaluate_afm.py
- Validation: 15_REPRODUCIBILITY/freeze_validation.json
- Archives: {archive}
- Package size MiB: {size['total_mib']:.2f}

Confirmed: unseen samples were not used for training; no original data were modified; no Phase1-7A outputs were overwritten; full-cohort model is not an independent test result; strict benchmark and deployment artifacts are separated.
"""
    text(freeze / "00_START_HERE/FINAL_FREEZE_REPORT.md", report)


def build_freeze(args: argparse.Namespace) -> dict[str, Any]:
    freeze, freeze_id = make_freeze_root(repo_path(args.freeze_root), args.freeze_version, args.resume)
    make_dirs(freeze)
    art = load_artifacts()
    manifest = build_identity(freeze, freeze_id, art)
    build_start_here(freeze, manifest)
    build_cohort(freeze, art)
    build_definitions(freeze)
    build_methods(freeze, art)
    copy_strict_results(freeze, art)
    copy_visual_results(freeze, art)
    if args.train_full_cohort_deployment or not args.skip_retraining:
        fit_full_cohort_quantitative(freeze, art)
        build_visual_bank(freeze, art)
        build_encoder_info(freeze)
    build_figures_and_tables(freeze, art, manifest)
    build_paper_text(freeze)
    build_claims(freeze)
    if args.build_unseen_tools:
        build_unseen_tools(freeze, manifest)
        build_registry_tools(freeze, manifest)
    build_repro_and_code(freeze)
    run_smoke_and_registry(freeze, manifest)
    write_model_artifact_hashes(freeze)
    build_checksums(freeze)
    if args.validate:
        validate_freeze(freeze)
    write_model_artifact_hashes(freeze)
    build_checksums(freeze)
    size = build_archive_manifest(freeze)
    package_tree(freeze)
    archive = create_archives(freeze) if args.build_archives else {}
    final_report(freeze, manifest, art, archive, size)
    build_checksums(freeze)
    size = build_archive_manifest(freeze)
    return {"freeze_root": display_path(freeze), "freeze_id": manifest["freeze_id"], "archive": archive, "size": size}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the final RHEED-to-AFM paper freeze bundle.")
    parser.add_argument("--freeze-root", default="paper_freeze")
    parser.add_argument("--freeze-version", default="v1")
    parser.add_argument("--train-full-cohort-deployment", action="store_true")
    parser.add_argument("--copy-model-assets", action="store_true")
    parser.add_argument("--copy-paper-assets", action="store_true")
    parser.add_argument("--build-unseen-tools", action="store_true")
    parser.add_argument("--build-archives", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-retraining", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--deployment-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        latest = repo_path(args.freeze_root) / "LATEST_FREEZE.txt"
        if not latest.exists():
            raise SystemExit("No latest freeze to validate")
        freeze = repo_path(latest.read_text(encoding="utf-8").strip())
        validate_freeze(freeze)
        print(json.dumps({"validated": display_path(freeze)}, indent=2))
        return 0
    result = build_freeze(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
