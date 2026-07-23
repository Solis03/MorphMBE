#!/usr/bin/env python3
"""Validate the retrained prospective package and refresh its SHA256 manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


THIS = Path(__file__).resolve()
REPO = next(
    parent
    for parent in THIS.parents
    if (parent / "publication_freeze" / "rheed_afm_single_frame_v1_2026-07-18").is_dir()
)
PKG = THIS.parents[1]
HISTORICAL_IDS = [
    "6022", "6028", "6029", "6033", "6047", "6048", "6056", "6057", "6062",
    "6063", "6070", "6072", "6078", "6080", "6081", "6082", "6084", "6085",
    "6090", "6094", "6095", "6099", "6101",
]
TRAIN_IDS = HISTORICAL_IDS + ["6358", "6382"]
LOO_EXTRA_IDS = ["6342", "6358", "6382", "6389", "6390"]
LOO_IDS = HISTORICAL_IDS + LOO_EXTRA_IDS
TEST_IDS = ["N6342", "N6389", "N6390"]
EXTRA_IDS = ["N6342", "N6358", "N6382", "N6389", "N6390"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rq_nm(array: np.ndarray) -> float:
    values = np.asarray(array, dtype=np.float64)
    values = values[np.isfinite(values)]
    values = values - float(np.mean(values))
    return float(np.sqrt(np.mean(values**2)))


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def validate_existing_manifest(checks: list[dict[str, Any]]) -> None:
    manifest = PKG / "provenance/MANIFEST.sha256"
    rows = []
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, relative = line.split("  ", 1)
            path = PKG / relative
            rows.append(path.is_file() and sha256_file(path) == digest)
    add_check(checks, "existing_manifest_entries_match", bool(rows) and all(rows), f"{sum(rows)}/{len(rows)}")


def refresh_manifest() -> None:
    manifest = PKG / "provenance/MANIFEST.sha256"
    rows = []
    for path in sorted(PKG.rglob("*")):
        if path.is_file() and path != manifest:
            rows.append(f"{sha256_file(path)}  {path.relative_to(PKG)}")
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    checks: list[dict[str, Any]] = []
    validate_existing_manifest(checks)

    split = pd.read_csv(PKG / "tables/data_split.csv", dtype={"sample_id": str})
    train = split[split["role"].isin(["historical_training", "added_training"])]["sample_id"].tolist()
    normalized_train = [sid.removeprefix("N") for sid in train]
    add_check(checks, "training_ids_exactly_23_plus_6358_6382", normalized_train == TRAIN_IDS, normalized_train)
    add_check(
        checks,
        "prediction_ids_exactly_6342_6389_6390",
        split[split["role"].eq("prediction")]["sample_id"].tolist() == TEST_IDS,
        split[split["role"].eq("prediction")]["sample_id"].tolist(),
    )
    add_check(
        checks,
        "N6324_ignored",
        split.loc[split["sample_id"].eq("N6324"), "role"].tolist() == ["ignored"],
        split.loc[split["sample_id"].eq("N6324"), "role"].tolist(),
    )

    baseline = pd.read_csv(PKG / "predictions/frozen_23_baseline/predictions.csv", dtype={"sample_id": str})
    add_check(checks, "frozen_rerun_has_five_non6324_samples", baseline["sample_id"].tolist() == EXTRA_IDS, baseline["sample_id"].tolist())
    predictions = pd.read_csv(PKG / "predictions/retrained_25/predictions.csv", dtype={"sample_id": str})
    add_check(checks, "retrained_prediction_ids_exact", predictions["sample_id"].tolist() == TEST_IDS, predictions["sample_id"].tolist())
    add_check(checks, "retrained_training_count_25", predictions["training_sample_count"].eq(25).all(), predictions["training_sample_count"].tolist())

    quarter = pd.read_csv(PKG / "ground_truth_afm/quarter_afm_manifest.csv", dtype={"sample_id": str})
    add_check(checks, "quarter_manifest_has_25_maps", len(quarter) == 25, len(quarter))
    add_check(
        checks,
        "five_maps_per_extra_sample",
        quarter.groupby("sample_id").size().to_dict() == {sid: 5 for sid in EXTRA_IDS},
        quarter.groupby("sample_id").size().to_dict(),
    )
    shape_ok = True
    rq_ok = True
    for row in quarter.to_dict("records"):
        array = np.load(REPO / row["output_path"], allow_pickle=False)
        shape_ok &= array.shape == (256, 256)
        rq_ok &= abs(rq_nm(array) - float(row["rq_nm"])) < 1e-10
    add_check(checks, "all_quarter_maps_are_256_square", shape_ok, "expected 256x256")
    add_check(checks, "all_quarter_rq_values_recompute", rq_ok, "tolerance 1e-10 nm")

    for model_path in sorted((PKG / "models/quantitative_model").glob("model_*.npz")):
        model = np.load(model_path, allow_pickle=False)
        ids = [str(value) for value in model["training_sample_ids"].tolist()]
        add_check(checks, f"{model_path.stem}_training_ids", ids == TRAIN_IDS, ids)

    audit = json.loads((PKG / "provenance/frozen_23_reproduction_audit.json").read_text(encoding="utf-8"))
    add_check(checks, "frozen_23_numeric_reproduction_passes", audit["status"] == "pass", audit["status"])
    code_audit = json.loads((PKG / "provenance/algorithm_code_audit.json").read_text(encoding="utf-8"))
    add_check(checks, "algorithm_sources_have_no_git_diff", not code_audit["algorithm_source_git_diff_detected"], code_audit["algorithm_source_git_diff_paths"])

    retrieval = pd.read_csv(
        PKG / "predictions/retrained_25/retrieval/retrieval_results.csv",
        dtype={"sample_id": str, "source_sample_id": str},
    )
    retrieved_rq_ok = True
    for row in retrieval.to_dict("records"):
        array = np.load(REPO / row["retrieved_q50_map_path"], allow_pickle=False)
        retrieved_rq_ok &= abs(rq_nm(array) - float(row["retrieval_q50_rq_nm"])) < 1e-5
    add_check(checks, "retrieved_maps_match_predicted_rq", retrieved_rq_ok, "tolerance 1e-5 nm")
    add_check(checks, "retrieval_bank_has_25_groups", retrieval["source_group_count"].eq(25).all(), retrieval["source_group_count"].tolist())
    add_check(checks, "test_ground_truth_excluded_from_retrieval", retrieval["test_ground_truth_in_retrieval_bank"].eq(False).all(), retrieval["test_ground_truth_in_retrieval_bank"].tolist())

    figure_stem = PKG / "figures/main/Figure1_three_sample_prediction_atlas"
    figure_paths = [figure_stem.with_suffix(suffix) for suffix in [".png", ".pdf", ".svg"]]
    add_check(checks, "primary_figure_all_formats_exist", all(path.is_file() and path.stat().st_size > 0 for path in figure_paths), [str(path) for path in figure_paths])
    individual_ground_truth = list((PKG / "figures/individual_afm/ground_truth").glob("*/*.png"))
    individual_retrieved = list((PKG / "figures/individual_afm/retrieved").glob("*.png"))
    add_check(checks, "25_individual_ground_truth_renders", len(individual_ground_truth) == 25, len(individual_ground_truth))
    add_check(checks, "3_individual_retrieved_renders", len(individual_retrieved) == 3, len(individual_retrieved))

    loo_root = PKG / "predictions/leave_one_out_28"
    loo = pd.read_csv(loo_root / "predictions.csv", dtype={"sample_id": str})
    loo_members = pd.read_csv(
        loo_root / "ensemble_member_predictions.csv",
        dtype={"held_out_sample_id": str},
    )
    loo_folds = pd.read_csv(
        loo_root / "fold_manifest.csv",
        dtype={"held_out_sample_id": str},
    )
    add_check(
        checks,
        "loo_prediction_ids_exactly_all_28",
        loo["sample_id"].tolist() == LOO_IDS,
        loo["sample_id"].tolist(),
    )
    add_check(
        checks,
        "loo_has_one_prediction_per_sample",
        len(loo) == 28 and loo["sample_id"].is_unique,
        {"rows": len(loo), "unique_ids": int(loo["sample_id"].nunique())},
    )
    add_check(
        checks,
        "loo_all_folds_train_on_27",
        loo["training_sample_count"].eq(27).all()
        and loo_folds["training_sample_count"].eq(27).all()
        and loo_members["training_sample_count"].eq(27).all(),
        {
            "prediction_counts": sorted(loo["training_sample_count"].unique().tolist()),
            "fold_counts": sorted(loo_folds["training_sample_count"].unique().tolist()),
            "member_counts": sorted(loo_members["training_sample_count"].unique().tolist()),
        },
    )
    leakage_ok = True
    fold_membership_ok = True
    for row in loo_folds.to_dict("records"):
        training_ids = json.loads(row["training_sample_ids"])
        held_out = row["held_out_sample_id"]
        leakage_ok &= held_out not in training_ids
        fold_membership_ok &= training_ids == [sid for sid in LOO_IDS if sid != held_out]
    add_check(checks, "loo_held_out_absent_from_every_fold", leakage_ok, "28 fold manifests checked")
    add_check(
        checks,
        "loo_each_fold_uses_exact_other_27_ids",
        fold_membership_ok,
        "ordered training IDs checked",
    )
    add_check(
        checks,
        "loo_has_five_member_predictions_per_sample",
        len(loo_members) == 140
        and loo_members.groupby("held_out_sample_id").size().eq(5).all(),
        {
            "rows": len(loo_members),
            "counts": loo_members.groupby("held_out_sample_id").size().to_dict(),
        },
    )

    embedding_bank = np.load(
        PKG / "models/encoder/combined_training_and_test_embeddings.npz",
        allow_pickle=False,
    )
    training_embedding_ids = [
        str(value) for value in embedding_bank["training_sample_ids"].tolist()
    ]
    test_embedding_ids = [
        str(value).removeprefix("N")
        for value in embedding_bank["test_sample_ids"].tolist()
    ]
    embedding_map = {
        sample_id: np.asarray(vector, dtype=np.float64)
        for sample_id, vector in zip(
            training_embedding_ids,
            embedding_bank["training_embeddings"],
            strict=True,
        )
    }
    embedding_map.update(
        {
            sample_id: np.asarray(vector, dtype=np.float64)
            for sample_id, vector in zip(
                test_embedding_ids,
                embedding_bank["test_embeddings"],
                strict=True,
            )
        }
    )
    embedding_ids = LOO_IDS
    embeddings = np.vstack([embedding_map[sample_id] for sample_id in embedding_ids])
    training_loo_targets = pd.read_csv(
        PKG / "models/quantitative_model/training_targets.csv",
        dtype={"sample_id": str},
    )
    extra_loo_targets = pd.read_csv(
        PKG / "ground_truth_afm/sample_targets.csv",
        dtype={"sample_id": str},
    )
    extra_loo_targets["sample_id"] = extra_loo_targets["sample_id"].str.removeprefix("N")
    training_target_index = training_loo_targets.set_index("sample_id")
    extra_target_index = extra_loo_targets.set_index("sample_id")
    loo_target_rows = []
    for sample_id in LOO_IDS:
        source = extra_target_index if sample_id in LOO_EXTRA_IDS else training_target_index
        loo_target_rows.append(
            {
                "sample_id": sample_id,
                "T4_second_order_trimmed_mean": float(
                    source.loc[sample_id, "T4_second_order_trimmed_mean"]
                ),
                "T6_quality_weighted_second_order": float(
                    source.loc[sample_id, "T6_quality_weighted_second_order"]
                ),
            }
        )
    loo_targets = pd.DataFrame(loo_target_rows).set_index("sample_id")
    ensemble = json.loads(
        (
            REPO
            / "publication_freeze/rheed_afm_single_frame_v1_2026-07-18"
            / "models/quantitative_model/full_cohort_deployment/ensemble_definition.json"
        ).read_text(encoding="utf-8")
    )
    recomputed_predictions: dict[str, float] = {}
    maximum_member_difference = 0.0
    member_index = loo_members.set_index(["held_out_sample_id", "member_name"])
    for held_index, held_out in enumerate(embedding_ids):
        train_mask = np.arange(len(embedding_ids)) != held_index
        member_values = []
        for member in ensemble["members"]:
            scaler = StandardScaler().fit(embeddings[train_mask])
            model = Ridge(alpha=1.0).fit(
                scaler.transform(embeddings[train_mask]),
                loo_targets.loc[
                    [sid for sid in embedding_ids if sid != held_out],
                    member["target_variant"],
                ].to_numpy(float),
            )
            value = float(
                model.predict(scaler.transform(embeddings[held_index : held_index + 1]))[0]
            )
            saved = float(member_index.loc[(held_out, member["name"]), "predicted_rq_nm"])
            maximum_member_difference = max(maximum_member_difference, abs(value - saved))
            member_values.append(value)
        recomputed_predictions[held_out] = float(np.median(member_values))
    saved_loo = loo.set_index("sample_id")["leave_one_out_predicted_rq_nm"]
    maximum_ensemble_difference = max(
        abs(recomputed_predictions[sid] - float(saved_loo.loc[sid]))
        for sid in LOO_IDS
    )
    add_check(
        checks,
        "loo_member_predictions_recompute_exactly",
        maximum_member_difference < 1e-10,
        maximum_member_difference,
    )
    add_check(
        checks,
        "loo_ensemble_predictions_recompute_exactly",
        maximum_ensemble_difference < 1e-10,
        maximum_ensemble_difference,
    )
    loo_metrics = json.loads((loo_root / "metrics.json").read_text(encoding="utf-8"))
    recomputed_mae = float(
        mean_absolute_error(
            loo["ground_truth_T4_rq_nm"],
            loo["leave_one_out_predicted_rq_nm"],
        )
    )
    add_check(
        checks,
        "loo_mae_recomputes_exactly",
        abs(recomputed_mae - float(loo_metrics["raw_model_output"]["MAE_nm"])) < 1e-12,
        recomputed_mae,
    )
    add_check(
        checks,
        "loo_extra_five_display_ids",
        loo[loo["sample_id"].isin(LOO_EXTRA_IDS)]["display_sample_id"].tolist()
        == ["N6342", "N6358", "N6382", "N6389", "N6390"],
        loo[loo["sample_id"].isin(LOO_EXTRA_IDS)]["display_sample_id"].tolist(),
    )
    loo_figure_stems = [
        PKG / "figures/main/Figure2_leave_one_out_prediction_scatter",
        PKG / "figures/supplementary/SuppFigure10_leave_one_out_diagnostics",
    ]
    loo_figure_paths = [
        stem.with_suffix(suffix)
        for stem in loo_figure_stems
        for suffix in [".png", ".pdf", ".svg"]
    ]
    add_check(
        checks,
        "loo_paper_figures_all_formats_exist",
        all(path.is_file() and path.stat().st_size > 0 for path in loo_figure_paths),
        [str(path) for path in loo_figure_paths],
    )

    source_files = [
        "analysis/rheed_video_afm_story/build_final_paper_freeze.py",
        "scripts/fit_afm_second_order.py",
        "publication_freeze/prospective_unseen_single_frame_v1/code/generate_full_cohort_retrieval_images.py",
    ]
    diff = subprocess.run(
        ["git", "diff", "--name-only", "--", *source_files],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    add_check(checks, "algorithm_source_files_still_unmodified", not diff, diff.splitlines() if diff else [])

    passed = all(row["passed"] for row in checks)
    report = {"created_at": now(), "status": "pass" if passed else "fail", "check_count": len(checks), "checks": checks}
    report_path = PKG / "provenance/validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    refresh_manifest()
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
