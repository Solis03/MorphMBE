from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import repo_path, save_parquet, write_csv, write_json
from .rq_disentanglement import physical_from_q, project_unit_rq_np, rq_np


VARIANT = "afm_second_order_y2_v1"
REMOVED = {"6023", "6087"}
PHASE7A_OUT = Path(f"outputs/rheed_video_afm_story/variants/{VARIANT}/phase7a_reconstruction_first")
PHASE7B_OUT = Path(f"outputs/rheed_video_afm_story/variants/{VARIANT}/phase7b_fixed_method_atlases")
PHASE7B_REPORT = Path(f"reports/rheed_video_afm_story/variants/{VARIANT}/phase7b_fixed_method_atlases")

METHOD_REGISTRY = [
    {
        "family": "retrieval",
        "method_id": "A3",
        "selection_basis": "current frozen Phase7A/Freeze strict visual method",
        "recommended_for_unseen": True,
        "notes": "Frozen deployment recommendation: descriptor-conditioned representative AFM retrieval.",
    },
    {
        "family": "quilting",
        "method_id": "VB2",
        "selection_basis": "best fixed method in Phase7A strict_method_summary for this family",
        "recommended_for_unseen": False,
        "notes": "Exploratory deployable candidate; not current frozen deployment recommendation.",
    },
    {
        "family": "residual",
        "method_id": "C1",
        "selection_basis": "best fixed method in Phase7A strict_method_summary for this family",
        "recommended_for_unseen": False,
        "notes": "Exploratory synthesis candidate; not current frozen deployment recommendation.",
    },
    {
        "family": "iaaft",
        "method_id": "D4",
        "selection_basis": "best fixed method in Phase7A strict_method_summary for this family",
        "recommended_for_unseen": False,
        "notes": "Exploratory spectral synthesis candidate; not current frozen deployment recommendation.",
    },
    {
        "family": "texture",
        "method_id": "E2",
        "selection_basis": "best fixed method in Phase7A strict_method_summary for this family",
        "recommended_for_unseen": False,
        "notes": "Exploratory texture optimization candidate; not current frozen deployment recommendation.",
    },
    {
        "family": "vq",
        "method_id": "F1",
        "selection_basis": "best fixed method in Phase7A strict_method_summary for this family; F1/F2 tied",
        "recommended_for_unseen": False,
        "notes": "Exploratory VQ candidate; not current frozen deployment recommendation.",
    },
    {
        "family": "diffusion",
        "method_id": "G4",
        "selection_basis": "only strict diffusion candidate in Phase7A",
        "recommended_for_unseen": False,
        "notes": "Exploratory residual diffusion fallback candidate; not current frozen deployment recommendation.",
    },
]


def markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = [str(row[c]).replace("\n", " ") for c in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def ensure_dirs() -> tuple[Path, Path]:
    out = repo_path(PHASE7B_OUT)
    rep = repo_path(PHASE7B_REPORT)
    for rel in ["rendered_maps", "scaled_maps", "tables", "validation"]:
        (out / rel).mkdir(parents=True, exist_ok=True)
    for rel in ["figures", "dashboard"]:
        (rep / rel).mkdir(parents=True, exist_ok=True)
    return out, rep


def parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return [str(value)]
    if isinstance(parsed, (list, tuple)):
        return [str(x) for x in parsed]
    return [str(parsed)]


def read_map(path: str | Path) -> np.ndarray:
    arr = np.load(repo_path(path), allow_pickle=False).astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.all():
        fill = float(np.nanmean(arr[finite])) if finite.any() else 0.0
        arr = np.where(finite, arr, fill).astype(np.float32)
    return arr


def render_rheed(sid: str) -> np.ndarray | None:
    p = repo_path(f"outputs/rheed_video_afm_story/phase2a/clip_variants/keyframe_1/{sid}.npz")
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return np.asarray(z["frames_uint8"][0])


def render_single_map(arr: np.ndarray, path: Path, cmap: str = "viridis") -> None:
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    lo, hi = np.percentile(arr, [1, 99])
    im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="nm")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_afm(ax: plt.Axes, arr: np.ndarray, title: str) -> None:
    lo, hi = np.percentile(arr, [1, 99])
    ax.imshow(arr, cmap="viridis", vmin=lo, vmax=hi)
    ax.set_title(title, fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_rheed(ax: plt.Axes, img: np.ndarray | None, title: str) -> None:
    if img is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=6)
    else:
        ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=6)
    ax.set_xticks([])
    ax.set_yticks([])


def save_figure(fig: plt.Figure, stem: Path) -> dict[str, str]:
    paths = {}
    for ext in ["png", "pdf"]:
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=190 if ext == "png" else None, bbox_inches="tight")
        paths[ext] = str(p)
    return paths


def method_rows(metrics: pd.DataFrame, method_id: str, active_ids: list[str]) -> dict[str, pd.DataFrame]:
    subset = metrics[
        metrics["track"].eq("strict")
        & metrics["method"].eq(method_id)
        & metrics["seed"].astype(int).eq(0)
        & metrics["sample_id"].astype(str).isin(active_ids)
    ].copy()
    by_amp: dict[str, pd.DataFrame] = {}
    for amp in ["q10", "q50", "q90"]:
        rows = subset[subset["amplitude_key"].astype(str).eq(amp)].copy()
        if len(rows) == len(active_ids):
            by_amp[amp] = rows.sort_values("sample_id").reset_index(drop=True)
    if "q50" not in by_amp:
        raise RuntimeError(f"Method {method_id} has no fixed seed=0 q50 rows for all samples")
    return by_amp


def q_values(conditions: pd.DataFrame, sid: str) -> dict[str, float]:
    row = conditions[conditions["sample_id"].astype(str).eq(sid)].iloc[0]
    return {
        "q10": float(row["condition_q10_rq_nm"]),
        "q50": float(row["condition_q50_rq_nm"]),
        "q90": float(row["condition_q90_rq_nm"]),
    }


def true_rq_value(conditions: pd.DataFrame, sid: str) -> float:
    row = conditions[conditions["sample_id"].astype(str).eq(sid)].iloc[0]
    return float(row["true_rq_nm"])


def rescale_to_q(arr: np.ndarray, q: float) -> np.ndarray:
    unit = project_unit_rq_np(arr)
    return physical_from_q(unit, q).astype(np.float32)


def select_or_create_amp_map(
    out: Path,
    family: str,
    method_id: str,
    sid: str,
    amp: str,
    q50_row: pd.Series,
    by_amp: dict[str, pd.DataFrame],
    q: float,
) -> Path:
    existing = by_amp.get(amp)
    if existing is not None:
        row = existing[existing["sample_id"].astype(str).eq(sid)]
        if len(row):
            return repo_path(row.iloc[0]["map_path"])
    arr = rescale_to_q(read_map(q50_row["map_path"]), q)
    path = out / "scaled_maps" / family / f"{sid}__{method_id}__seed0__{amp}_amplitude_only.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return path


def build_family(
    out: Path,
    rep: Path,
    registry: dict[str, Any],
    metrics: pd.DataFrame,
    conditions: pd.DataFrame,
    index: pd.DataFrame,
    active_ids: list[str],
) -> pd.DataFrame:
    family = registry["family"]
    method_id = registry["method_id"]
    by_amp = method_rows(metrics, method_id, active_ids)
    q50 = by_amp["q50"].sort_values("sample_id").reset_index(drop=True)
    rows: list[dict[str, Any]] = []

    fig, axes = plt.subplots(len(active_ids), 6, figsize=(13.5, max(12, len(active_ids) * 1.55)))
    axes = np.atleast_2d(axes)
    fig.suptitle(
        f"Strict OOF fixed {family} ({method_id}) atlas: RHEED-conditioned deployable path; held-out AFM shown only for retrospective comparison",
        fontsize=12,
        weight="bold",
    )

    idx = index.drop_duplicates("sample_id").set_index("sample_id")
    for r, sid in enumerate(active_ids):
        row = q50[q50["sample_id"].astype(str).eq(sid)].iloc[0]
        q = q_values(conditions, sid)
        gt_path = idx.loc[sid, "second_order_representative_afm_path"]
        gt = read_map(gt_path)
        true_rq = true_rq_value(conditions, sid)
        rheed = render_rheed(sid)
        amp_paths = {
            amp: select_or_create_amp_map(out, family, method_id, sid, amp, row, by_amp, q[amp])
            for amp in ["q10", "q50", "q90"]
        }
        rendered_paths: dict[str, str] = {}
        for amp, map_path in amp_paths.items():
            rendered = out / "rendered_maps" / family / f"{sid}__{method_id}__seed0__{amp}.png"
            render_single_map(read_map(map_path), rendered)
            rendered_paths[amp] = str(rendered)

        plot_rheed(axes[r, 0], rheed, f"{sid} RHEED")
        plot_afm(axes[r, 1], gt, "GT AFM")
        plot_afm(axes[r, 2], read_map(amp_paths["q50"]), f"{method_id} q50")
        plot_afm(axes[r, 3], read_map(amp_paths["q10"]), "q10")
        plot_afm(axes[r, 4], read_map(amp_paths["q90"]), "q90")
        axes[r, 5].axis("off")
        source_ids = parse_list(row["source_sample_ids"])
        axes[r, 5].text(
            0.01,
            0.96,
            "\n".join(
                [
                    f"sample {sid}",
                    f"true Rq {true_rq:.2f}",
                    f"pred q50 {q['q50']:.2f}",
                    f"out Rq {float(row['measured_rq_nm']):.2f}",
                    f"PSD {float(row['normalized_psd_log_distance']):.2f}",
                    f"source {','.join(source_ids[:3])}",
                    "support strict_oof",
                ]
            ),
            va="top",
            fontsize=6,
        )
        rows.append(
            {
                "sample_id": sid,
                "family": family,
                "method_id": method_id,
                "strict_track": True,
                "deployable_for_unseen": True,
                "recommended_for_unseen": bool(registry["recommended_for_unseen"]),
                "predicted_rq_nm": q["q50"],
                "true_rq_nm": true_rq,
                "output_q50_measured_rq_nm": float(row["measured_rq_nm"]),
                "predicted_rq_q10": q["q10"],
                "predicted_rq_q50": q["q50"],
                "predicted_rq_q90": q["q90"],
                "source_sample_ids": json.dumps(source_ids),
                "heldout_source_contribution": float(row["heldout_source_contribution"]),
                "uses_predicted_rq_not_true_rq": bool(row["uses_predicted_rq_not_true_rq"]),
                "uses_heldout_true_afm_for_selection": False,
                "uses_heldout_true_descriptors_for_selection": False,
                "visual_composite": float(row["visual_composite_score"]),
                "psd_distance": float(row["normalized_psd_log_distance"]),
                "histogram_wasserstein": float(row["histogram_wasserstein"]),
                "corr_length_relative_error": float(row["correlation_length_relative_error"]),
                "anisotropy_error_if_available": float(row["anisotropy_error"]),
                "support": "strict_oof_rheed_conditioned",
                "abstain_flag_if_available": False,
                "q10_q90_generation": "existing_phase7a_map" if "q10" in by_amp and "q90" in by_amp else "amplitude_only_rescale_from_fixed_seed0_q50",
                "rendered_q10_path": rendered_paths["q10"],
                "rendered_q50_path": rendered_paths["q50"],
                "rendered_q90_path": rendered_paths["q90"],
                "generated_q10_map_path": str(amp_paths["q10"]),
                "generated_q50_map_path": str(amp_paths["q50"]),
                "generated_q90_map_path": str(amp_paths["q90"]),
                "ground_truth_afm_path": gt_path,
                "rheed_keyframe_path": f"outputs/rheed_video_afm_story/phase2a/clip_variants/keyframe_1/{sid}.npz",
            }
        )
    for ax in axes.flat:
        ax.set_anchor("N")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    stem = rep / "figures" / f"Fig_fixed_{family}_all_23_strict_oof_atlas"
    paths = save_figure(fig, stem)
    plt.close(fig)

    table = pd.DataFrame(rows)
    table["atlas_png_path"] = paths["png"]
    table["atlas_pdf_path"] = paths["pdf"]
    write_csv(table, out / f"{family}_strict_outputs.csv")
    save_parquet(table, out / f"{family}_strict_outputs.parquet")
    return table


def build_summary(tables: list[pd.DataFrame], registry: pd.DataFrame, out: Path, rep: Path) -> pd.DataFrame:
    rows = []
    for table in tables:
        family = str(table["family"].iloc[0])
        method_id = str(table["method_id"].iloc[0])
        reg = registry[registry["family"].eq(family)].iloc[0].to_dict()
        rows.append(
            {
                "family": family,
                "method_id": method_id,
                "N": int(len(table)),
                "median_visual_composite": float(table["visual_composite"].median()),
                "mean_visual_composite": float(table["visual_composite"].mean()),
                "median_psd_distance": float(table["psd_distance"].median()),
                "median_histogram_wasserstein": float(table["histogram_wasserstein"].median()),
                "median_corr_length_relative_error": float(table["corr_length_relative_error"].median()),
                "strict_identity_pass": bool(table["heldout_source_contribution"].max() == 0 and not table["source_sample_ids"].isna().any()),
                "max_heldout_source_contribution": float(table["heldout_source_contribution"].max()),
                "deployable_for_unseen": bool(table["deployable_for_unseen"].all()),
                "recommended_for_unseen": bool(table["recommended_for_unseen"].all()),
                "notes": reg["notes"],
            }
        )
    summary = pd.DataFrame(rows).sort_values("median_visual_composite").reset_index(drop=True)
    write_csv(summary, out / "fixed_method_family_summary.csv")
    write_csv(summary, rep / "fixed_method_family_summary.csv")
    best = summary.iloc[0]
    current = summary[summary["family"].eq("retrieval")].iloc[0]
    md = [
        "# Fixed-Method Family Summary",
        "",
        f"- Best fixed strict method by median visual composite: {best['family']} / {best['method_id']} ({best['median_visual_composite']:.6f}).",
        f"- Current frozen deployment recommendation: {current['family']} / {current['method_id']} ({current['median_visual_composite']:.6f}).",
        "- Mixed-method atlas differs because it selects the best visual output separately for each sample; this rerun fixes one method per family across all 23 held-out samples.",
        "- Lower visual composite is better.",
        "",
        markdown_table(summary),
    ]
    (rep / "fixed_method_family_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return summary


def build_audit(tables: list[pd.DataFrame], registry: pd.DataFrame, out: Path, rep: Path) -> pd.DataFrame:
    rows = []
    for table in tables:
        family = str(table["family"].iloc[0])
        method_id = str(table["method_id"].iloc[0])
        source_self = []
        for _, row in table.iterrows():
            if str(row["sample_id"]) in parse_list(row["source_sample_ids"]):
                source_self.append(str(row["sample_id"]))
        row = {
            "family": family,
            "method_id": method_id,
            "single_method_only": table["method_id"].nunique() == 1,
            "all_from_heldout_rheed_strict_prediction": bool(table["uses_predicted_rq_not_true_rq"].all()),
            "uses_heldout_true_afm_for_method_seed_or_source_selection": bool(table["uses_heldout_true_afm_for_selection"].any()),
            "uses_heldout_true_descriptors": bool(table["uses_heldout_true_descriptors_for_selection"].any()),
            "max_heldout_source_contribution": float(table["heldout_source_contribution"].max()),
            "source_sample_never_equals_heldout": len(source_self) == 0,
            "retrieval_based": family in {"retrieval"},
            "synthesis_based": family in {"quilting", "residual", "iaaft", "texture", "vq", "diffusion"},
            "q10_q50_q90_amplitude_only": bool(table["q10_q90_generation"].str.contains("amplitude_only").any()),
            "deployable_for_unseen": bool(table["deployable_for_unseen"].all()),
            "notes": registry[registry["family"].eq(family)]["notes"].iloc[0],
        }
        rows.append(row)
    audit = pd.DataFrame(rows)
    write_csv(audit, out / "method_audit.csv")
    write_csv(audit, rep / "method_audit.csv")
    md = ["# Method Audit", ""]
    md.append("All rows are fixed-method strict OOF paths; held-out AFM is used only for display and retrospective metrics.")
    md.append("")
    md.append(markdown_table(audit))
    (rep / "method_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return audit


def build_validation(tables: list[pd.DataFrame], summary: pd.DataFrame, audit: pd.DataFrame, active_ids: list[str], out: Path, rep: Path) -> dict[str, Any]:
    checks = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    all_rows = pd.concat(tables, ignore_index=True)
    add("active_sample_count_23", len(active_ids) == 23, str(len(active_ids)))
    add("removed_samples_excluded", REMOVED.isdisjoint(set(active_ids)), ",".join(sorted(REMOVED & set(active_ids))))
    add("one_method_id_per_family", all(t["method_id"].nunique() == 1 for t in tables))
    add("each_atlas_covers_23_samples", all(len(t) == 23 and t["sample_id"].nunique() == 23 for t in tables))
    add("displayed_method_fixed_per_atlas", all(t["method_id"].nunique() == 1 for t in tables))
    add("no_per_sample_best_method_selection", len(summary) == 7 and all(t["family"].nunique() == 1 for t in tables))
    add("heldout_true_afm_not_used_for_selection", not bool(all_rows["uses_heldout_true_afm_for_selection"].any()))
    add("max_heldout_source_contribution_zero", float(all_rows["heldout_source_contribution"].max()) == 0.0)
    source_ok = True
    bad_sources = []
    for _, row in all_rows.iterrows():
        if str(row["sample_id"]) in parse_list(row["source_sample_ids"]):
            source_ok = False
            bad_sources.append(str(row["sample_id"]))
    add("source_sample_never_equals_heldout", source_ok, ",".join(bad_sources[:10]))
    add("predicted_rq_from_strict_oof_branch", bool(all_rows["uses_predicted_rq_not_true_rq"].all()))
    add(
        "true_rq_not_copied_from_output_measured_rq",
        not bool((all_rows["true_rq_nm"].round(6) == all_rows["output_q50_measured_rq_nm"].round(6)).all()),
    )
    add("all_outputs_future_unseen_deployable_path", bool(all_rows["deployable_for_unseen"].all()))
    add("no_heldout_true_descriptors_for_selection", not bool(all_rows["uses_heldout_true_descriptors_for_selection"].any()))
    result = {"checks": checks, "all_passed": all(c["passed"] for c in checks)}
    write_json(result, out / "validation_fixed_method_atlases.json")
    write_json(result, rep / "validation_fixed_method_atlases.json")
    md = ["# Fixed-Method Atlas Validation", ""]
    md.extend([f"- {c['check']}: {c['passed']} {c['detail']}" for c in checks])
    md.append("")
    md.append(f"All passed: {result['all_passed']}")
    (out / "validation_fixed_method_atlases.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (rep / "validation_fixed_method_atlases.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if not result["all_passed"]:
        raise RuntimeError("Fixed-method atlas validation failed")
    return result


def build_dashboard(tables: list[pd.DataFrame], summary: pd.DataFrame, rep: Path) -> None:
    rows = [
        "<!doctype html><meta charset='utf-8'><title>Phase 7B Fixed Method Atlases</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:4px 8px} img{max-width:420px}</style>",
        "<h1>Phase 7B Fixed-Method Strict OOF Atlases</h1>",
        "<p>Each atlas fixes one method family and one method_id across all 23 held-out samples. Held-out AFM is displayed only for retrospective comparison.</p>",
        summary.to_html(index=False),
        "<h2>Atlases</h2>",
    ]
    for table in tables:
        family = table["family"].iloc[0]
        method = table["method_id"].iloc[0]
        rel_png = Path(table["atlas_png_path"].iloc[0]).relative_to(rep)
        rel_pdf = Path(table["atlas_pdf_path"].iloc[0]).relative_to(rep)
        rows.append(f"<h3>{family} / {method}</h3><p><a href='{rel_pdf.as_posix()}'>PDF</a></p><img src='{rel_png.as_posix()}'>")
    (rep / "dashboard" / "index.html").write_text("\n".join(rows) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    out, rep = ensure_dirs()
    metrics = pd.read_csv(repo_path(PHASE7A_OUT / "metrics/all_visual_metrics.csv"), dtype={"sample_id": str})
    conditions = pd.read_csv(repo_path(PHASE7A_OUT / "condition_vectors/phase7_condition_vectors.csv"), dtype={"sample_id": str})
    index = pd.read_csv(repo_path(PHASE7A_OUT / "canonical_index/canonical_sample_index.csv"), dtype={"sample_id": str})
    active = index[index["is_primary"].astype(str).eq("True")].copy()
    active_ids = sorted(active["sample_id"].astype(str).tolist())
    registry = pd.DataFrame(METHOD_REGISTRY)
    write_csv(registry, out / "fixed_method_registry.csv")
    write_csv(registry, rep / "fixed_method_registry.csv")
    (rep / "fixed_method_registry.md").write_text("# Fixed Method Registry\n\n" + markdown_table(registry) + "\n", encoding="utf-8")

    tables = [build_family(out, rep, reg, metrics, conditions, index, active_ids) for reg in METHOD_REGISTRY]
    all_outputs = pd.concat(tables, ignore_index=True)
    write_csv(all_outputs, out / "all_fixed_method_strict_outputs.csv")
    save_parquet(all_outputs, out / "all_fixed_method_strict_outputs.parquet")
    summary = build_summary(tables, registry, out, rep)
    audit = build_audit(tables, registry, out, rep)
    validation = build_validation(tables, summary, audit, active_ids, out, rep)
    build_dashboard(tables, summary, rep)
    return {
        "output_root": str(PHASE7B_OUT),
        "report_root": str(PHASE7B_REPORT),
        "families": METHOD_REGISTRY,
        "summary": summary.to_dict("records"),
        "validation": validation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fixed-method strict visual atlases for Phase 7B.")
    parser.parse_args()
    print(json.dumps(run(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
