"""Audit M23 smooth-island tone, M22 rough protection, and exclusion flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.rheed_rough_island_redesign.dense_mid_evaluate import (
    _display_metrics,
)
from analysis.rheed_to_afm_full_cohort_loo.run import load_config
from analysis.rheed_to_afm_functional_morphology.visualization import (
    _real_afm,
)
from analysis.rheed_video_afm_story.common import repo_path, write_json

FROZEN_M22_METHOD = "M22c_frozen_dense_mid"


def _paths(config: dict[str, Any]) -> tuple[Path, Path]:
    suffix = str(config["full_run_suffix"])
    return (
        repo_path(config["output_root"]) / suffix,
        repo_path(config["report_root"]) / suffix,
    )


def _generated_payload(
    output: Path, *, method: str, group: str
) -> np.lib.npyio.NpzFile:
    return np.load(
        output
        / "crossfit"
        / "generated_maps"
        / method
        / f"{group}.npz",
        allow_pickle=False,
    )


def _source_rows(
    *,
    phase1: pd.DataFrame,
    predictions: pd.DataFrame,
    output: Path,
    method: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for prediction in predictions.to_dict(orient="records"):
        group = str(prediction["growth_run_id"])
        sources = {"real": _real_afm(phase1, group)}
        for source, source_method in (
            ("m22_frozen", FROZEN_M22_METHOD),
            ("m23", method),
        ):
            with _generated_payload(
                output, method=source_method, group=group
            ) as payload:
                sources[source] = (
                    np.asarray(
                        payload["generated_unit_shapes"][0], dtype=float
                    )
                    * float(payload["predicted_rq_nm"])
                )
        for source, array in sources.items():
            rows.append(
                {
                    "growth_run_id": group,
                    "source": source,
                    "true_sq_nm": float(prediction["true_target"]),
                    "predicted_sq_nm": float(prediction["predicted_target"]),
                    **_display_metrics(array),
                }
            )
    return pd.DataFrame(rows)


def _subset_summary(
    rows: pd.DataFrame, *, focus_groups: set[str]
) -> pd.DataFrame:
    subsets = {
        "all26": np.ones(len(rows), dtype=bool),
        "low_predicted_Sq_le_3p3_nm": rows["predicted_sq_nm"].le(3.3),
        "user_low_Sq_focus": rows["growth_run_id"].isin(focus_groups),
        "intermediate_true_Sq_3p5_to_6p0_nm": rows["true_sq_nm"].between(
            3.5, 6.0, inclusive="both"
        ),
        "frozen_rough_predicted_Sq_ge_3p8_nm": rows[
            "predicted_sq_nm"
        ].ge(3.8),
    }
    metrics = (
        "dark_fraction",
        "largest_dark_component_fraction",
        "display_median",
        "display_mean",
        "height_skewness",
    )
    summary: list[dict[str, float | int | str]] = []
    for subset, mask in subsets.items():
        selected = rows.loc[mask]
        for source, group in selected.groupby("source", sort=False):
            row: dict[str, float | int | str] = {
                "subset": subset,
                "source": str(source),
                "growth_count": int(group["growth_run_id"].nunique()),
            }
            row.update(
                {
                    f"mean_{metric}": float(group[metric].mean())
                    for metric in metrics
                }
            )
            summary.append(row)
    return pd.DataFrame(summary)


def _rough_protection(
    *,
    output: Path,
    predictions: pd.DataFrame,
    method: str,
    threshold_nm: float,
) -> pd.DataFrame:
    rows = []
    for prediction in predictions.loc[
        predictions["predicted_target"].ge(threshold_nm)
    ].to_dict(orient="records"):
        group = str(prediction["growth_run_id"])
        with _generated_payload(
            output, method=FROZEN_M22_METHOD, group=group
        ) as m22, _generated_payload(
            output, method=method, group=group
        ) as m23:
            m22_maps = np.asarray(m22["generated_unit_shapes"])
            m23_maps = np.asarray(m23["generated_unit_shapes"])
            rows.append(
                {
                    "growth_run_id": group,
                    "predicted_sq_nm": float(prediction["predicted_target"]),
                    "draw_count": len(m23_maps),
                    "array_equal_all_draws": bool(
                        np.array_equal(m22_maps, m23_maps)
                    ),
                    "maximum_absolute_pixel_difference": float(
                        np.max(np.abs(m22_maps - m23_maps))
                    ),
                }
            )
    return pd.DataFrame(rows)


def _exclusion_audit(
    *,
    config: dict[str, Any],
    output: Path,
    report: Path,
    predictions: pd.DataFrame,
    excluded: str,
) -> dict[str, Any]:
    cohort = pd.read_csv(
        report / "cohort_manifest.csv", dtype={"growth_run_id": str}
    )
    fold_manifests = sorted(
        (report / "crossfit" / "folds").glob("*/fold_manifest.json")
    )
    fold_hits: list[str] = []
    for path in fold_manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, value in payload.items():
            if key.endswith("growth_run_ids") and excluded in map(str, value):
                fold_hits.append(f"{path.parent.name}:{key}")
    external_hits: dict[str, bool] = {}
    for target, path in config["external_target_predictions"].items():
        table = pd.read_csv(repo_path(path), dtype={"growth_run_id": str})
        external_hits[str(target)] = bool(
            table["growth_run_id"].astype(str).eq(excluded).any()
        )
    removelist_ids = {
        line.split(maxsplit=1)[0]
        for line in repo_path(config["removelist_path"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    generated_hits = [
        str(path.relative_to(output))
        for path in output.rglob(f"{excluded}.npz")
    ]
    audit = {
        "excluded_growth_run_id": excluded,
        "listed_in_removelist": excluded in removelist_ids,
        "prediction_row_count": int(
            predictions["growth_run_id"].astype(str).eq(excluded).sum()
        ),
        "cohort_row_count": int(
            cohort["growth_run_id"].astype(str).eq(excluded).sum()
        ),
        "outer_fold_manifest_count": len(fold_manifests),
        "fold_training_or_morphology_hits": fold_hits,
        "external_prediction_hits": external_hits,
        "generated_map_hits": generated_hits,
    }
    audit["passed"] = bool(
        audit["listed_in_removelist"]
        and audit["prediction_row_count"] == 0
        and audit["cohort_row_count"] == 0
        and not fold_hits
        and not any(external_hits.values())
        and not generated_hits
    )
    return audit


def run(config: dict[str, Any]) -> None:
    output, report = _paths(config)
    method = str(config["selected_method"])
    phase1 = pd.read_csv(
        repo_path(config["phase1_manifest"]), dtype={"growth_run_id": str}
    )
    predictions = pd.read_csv(
        report / "rq_crossfit_predictions.csv",
        dtype={"growth_run_id": str},
    )
    rows = _source_rows(
        phase1=phase1,
        predictions=predictions,
        output=output,
        method=method,
    )
    summary = _subset_summary(
        rows,
        focus_groups=set(map(str, config["visualization_focus_groups"])),
    )
    rough = _rough_protection(
        output=output,
        predictions=predictions,
        method=method,
        threshold_nm=float(
            config["selected_renderer"]["m22_full_above_nm"]
        ),
    )
    exclusion = _exclusion_audit(
        config=config,
        output=output,
        report=report,
        predictions=predictions,
        excluded="6022",
    )
    rows.to_csv(report / "m23_display_metrics_per_group.csv", index=False)
    summary.to_csv(report / "m23_display_metrics_summary.csv", index=False)
    rough.to_csv(report / "m23_rough_branch_protection_audit.csv", index=False)
    write_json(exclusion, report / "m23_exclusion_6022_audit.json")
    if not exclusion["passed"]:
        raise RuntimeError("6022 exclusion audit failed")
    if rough.empty or not rough["array_equal_all_draws"].all():
        raise RuntimeError("frozen M22 rough-branch protection failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(Path(args.config)))


if __name__ == "__main__":
    main()
