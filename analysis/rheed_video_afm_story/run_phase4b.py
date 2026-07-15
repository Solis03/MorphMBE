from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .common import display_path, ensure_dirs, load_config, repo_path, sha256_file, write_json
from .publication_style import set_publication_style
from .results_data_assembly import (
    audit_schema,
    build_model_summary,
    build_sample_table,
    load_artifacts,
    quantile_sample_selection,
    validate_visualization,
    write_model_summary,
    write_sample_table,
)
from .visualize_phase4b import (
    atlas,
    comparison_figure,
    fig1_pipeline,
    fig3_rq,
    generic_figures,
    write_dashboard_and_text,
)


def artifact_hashes(config: dict[str, Any]) -> dict[str, str]:
    return {key: sha256_file(item["path"]) for key, item in config["artifacts"].items()}


def verify_config_hashes(config: dict[str, Any], hashes: dict[str, str]) -> None:
    mismatches = []
    for key, item in config["artifacts"].items():
        expected = item.get("sha256")
        if expected and expected != hashes[key]:
            mismatches.append(f"{key}: expected {expected}, observed {hashes[key]}")
    if mismatches:
        raise RuntimeError("Phase 4B input hash mismatch:\n" + "\n".join(mismatches))


def write_selection_file(sample_df: pd.DataFrame, selected: list[str], config: dict[str, Any]) -> None:
    rows = sample_df.set_index("sample_id").loc[selected].reset_index()
    write_json(
        {
            "selection_rule": config["sample_selection_rule"],
            "uses_prediction_error": False,
            "selected_sample_ids": selected,
            "selected_true_rq_nm": {str(row.sample_id): float(row.true_rq_nm) for row in rows.itertuples()},
        },
        repo_path(config["output_root"]) / "main_figure_sample_selection.json",
    )


def write_all_samples_visual_table(sample_df: pd.DataFrame, config: dict[str, Any]) -> None:
    display_cols = [
        "sample_id",
        "support_level",
        "true_rq_nm",
        "predicted_rq_nm",
        "rq_absolute_error_nm",
        "automatic_spot_streak_index",
        "s1_source_sample_id",
        "s1_psd_distance",
        "s4_psd_distance",
        "s4_largest_source_contribution",
    ]
    table = sample_df[display_cols].copy()
    table["FigS1_row_shared_page"] = table["sample_id"].rank(method="first").sub(1).floordiv(4).add(1).astype(int)
    html = table.to_html(index=False, float_format=lambda x: f"{x:.3f}", classes="sortable", table_id="all_samples")
    out = repo_path(config["output_root"]) / "all_samples_visual_table.html"
    out.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        "body{font-family:Arial,DejaVu Sans,sans-serif;margin:24px}"
        "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:4px}"
        "th{background:#eee}</style></head><body><h1>All Samples Visual Table</h1>"
        "<p>Rows are ordered by sample ID. FigS1 pages show the corresponding RHEED/AFM visual panels.</p>"
        f"{html}</body></html>",
        encoding="utf-8",
    )


def write_summary(
    sample_df: pd.DataFrame,
    model_summary: pd.DataFrame,
    validation: dict[str, Any],
    atlas_pages: dict[str, int],
    config: dict[str, Any],
) -> None:
    report_root = repo_path(config["report_root"])
    output_root = repo_path(config["output_root"])
    figures = sorted(display_path(p) for p in (report_root / "figures").glob("*.png"))
    r4 = model_summary[model_summary["model"].eq(config["preferred_rq_model"])].iloc[0]
    high = model_summary[model_summary["model"].eq("R4_high_confidence_subset")].iloc[0]
    write_json(
        {
            "phase": "4B",
            "primary_sample_count": int(len(sample_df)),
            "excluded_samples": config["cohort"]["excluded_samples"],
            "main_figure_sample_ids": quantile_sample_selection(sample_df, [0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            "preferred_rq_model": config["preferred_rq_model"],
            "r4_mae_nm": float(r4["MAE"]),
            "r4_spearman": float(r4["Spearman"]),
            "high_confidence_coverage": float(high["coverage"]),
            "validation_passed": bool(validation["passed"]),
            "atlas_pages": atlas_pages,
            "figures": figures,
            "outputs": {
                "sample_level_results": display_path(output_root / "sample_level_results.csv"),
                "model_level_summary": display_path(output_root / "model_level_summary.csv"),
                "validation": display_path(output_root / "visualization_validation.json"),
                "dashboard": display_path(report_root / "results_dashboard.html"),
            },
        },
        output_root / "phase4b_summary.json",
    )


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    ensure_dirs(config)
    before_hashes = artifact_hashes(config)
    verify_config_hashes(config, before_hashes)

    data = load_artifacts(config)
    data["same_growth"] = pd.read_csv(repo_path("outputs/rheed_video_afm_story/phase4a/same_growth_afm_similarity.csv"))

    sample_df = build_sample_table(data, config)
    model_summary = build_model_summary(data, config)
    write_sample_table(sample_df, config)
    write_model_summary(model_summary, config)
    write_all_samples_visual_table(sample_df, config)
    audit_schema(data, sample_df, config, before_hashes)

    set_publication_style()
    selected = quantile_sample_selection(sample_df, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    write_selection_file(sample_df, selected, config)

    render_records: list[dict[str, Any]] = []
    fig1_pipeline(config)
    render_records += comparison_figure(sample_df, config, robust=False, name="Fig2_ground_truth_vs_representative_afm_main", selected=selected)
    render_records += comparison_figure(sample_df, config, robust=True, name="Fig2_ground_truth_vs_representative_afm_main", selected=selected)
    fig3_rq(sample_df, model_summary, config)
    generic_figures(sample_df, model_summary, data, config)

    atlas_pages = {
        "sample_id_row_shared": atlas(sample_df, config, robust=False, order="sample_id"),
        "sample_id_robust": atlas(sample_df, config, robust=True, order="sample_id"),
        "true_rq_row_shared": atlas(sample_df, config, robust=False, order="rq"),
        "true_rq_robust": atlas(sample_df, config, robust=True, order="rq"),
    }

    validation = validate_visualization(sample_df, config, render_records, before_hashes)
    write_dashboard_and_text(sample_df, model_summary, validation, config)
    write_summary(sample_df, model_summary, validation, atlas_pages, config)
    return {
        "samples": len(sample_df),
        "figures": len(list((repo_path(config["report_root"]) / "figures").glob("*.png"))),
        "validation_passed": validation["passed"],
        "dashboard": display_path(repo_path(config["report_root"]) / "results_dashboard.html"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 4B current-results tables and publication visualizations.")
    parser.add_argument("--config", default="configs/rheed_video_afm_story_phase4b.yaml")
    args = parser.parse_args()
    summary = run(args.config)
    print(summary)


if __name__ == "__main__":
    main()
