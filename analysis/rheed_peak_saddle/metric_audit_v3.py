"""Plot-only Stage 1C-R metric-lineage audit for synthetic v3 outputs."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import stats

from analysis.rheed_peak_saddle.data import make_paths
from analysis.rheed_peak_saddle.merge_tree import maximum_bottleneck_saddle
from analysis.rheed_peak_saddle.pair_features import pair_masks
from analysis.rheed_peak_saddle.semantic_v3 import (
    independent_maximin_saddle,
    make_semantic_templates,
    render_semantic_template,
    spot_estimate_from_truth_v3,
)
from analysis.rheed_peak_saddle.synthetic import SyntheticRheed, SyntheticSpotTruth, make_synthetic_split_v2


EXPECTED_REMOVELIST_SHA256 = "8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b"
EXPECTED_STAGE_REVIEW_SHA256 = "862df0397683a19c24d616b2ba42b088538048750a63a89eb17593c1b4c9081e"
SUCCESS_SOLVER_STATUSES = {"converged"}
MINIMUM_FAMILY_POINTS = 3
INVERSION_TOLERANCE = 0.01
AUDIT_COMMAND = (
    "PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_peak_saddle.run "
    "--config configs/rheed_peak_saddle.yaml --stage synthetic_v3_metric_audit"
)

IMMUTABLE_EVALUATION_FILENAMES = (
    "evaluation_receipt.json",
    "frozen_semantic_spec.json",
    "frozen_semantic_spec.sha256",
    "holdout_v3_manifest.csv",
    "holdout_v3_metrics.csv",
    "target_adhesion_solver_results.csv",
    "pair_level_measurement_fidelity.csv",
    "image_level_measurement_fidelity.csv",
    "topology_regression_metrics.csv",
    "rank_inversion_manifest.csv",
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_row_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _fmt(value: Any) -> str:
    number = _num(value)
    if math.isfinite(number):
        return f"{number:.6g}"
    return str(value)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < MINIMUM_FAMILY_POINTS or len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    return float(stats.spearmanr(xs, ys).statistic)


def _kendall_tau_b(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < MINIMUM_FAMILY_POINTS or len(set(xs)) < 2 or len(set(ys)) < 2:
        return float("nan")
    return float(stats.kendalltau(xs, ys, variant="b").statistic)


def _rank_diagnostics(points: Sequence[tuple[float, float]], *, tolerance: float = INVERSION_TOLERANCE) -> dict[str, Any]:
    concordant = 0
    discordant = 0
    strict_inversions = 0
    tolerance_inversions = 0
    ordered = sorted(points, key=lambda item: (item[0], item[1]))
    for i, (x_a, y_a) in enumerate(ordered):
        for x_b, y_b in ordered[i + 1 :]:
            if x_a == x_b:
                continue
            if y_a < y_b:
                concordant += 1
            elif y_a > y_b:
                discordant += 1
                strict_inversions += 1
                if y_a > y_b + tolerance:
                    tolerance_inversions += 1
    denom = concordant + discordant
    return {
        "pairwise_concordance": concordant / denom if denom else float("nan"),
        "strict_inversion_count": strict_inversions,
        "tolerance_inversion_count": tolerance_inversions,
    }


def _family_metric_row(
    *,
    analysis: str,
    family_id: str,
    split: str,
    points: Sequence[tuple[float, float]],
    errors: Sequence[float],
    solver_failure_count: int,
    unattainable_target_count: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    rank = _rank_diagnostics(points)
    metadata = metadata or {}
    return {
        "analysis": analysis,
        "split": split,
        "family_id": family_id,
        "number_of_successful_targets": len(points),
        "number_of_unique_target_values": len(set(xs)),
        "attainable_target_min": min(xs) if xs else "",
        "attainable_target_max": max(xs) if xs else "",
        "spearman_rho": _spearman(xs, ys),
        "kendall_tau_b": _kendall_tau_b(xs, ys),
        "pairwise_concordance": rank["pairwise_concordance"],
        "strict_inversion_count": rank["strict_inversion_count"],
        "tolerance_inversion_count": rank["tolerance_inversion_count"],
        "target_achievement_mae": float(np.mean(errors)) if errors else "",
        "target_achievement_p90": float(np.percentile(errors, 90)) if errors else "",
        "maximum_target_error": max(errors) if errors else "",
        "solver_failure_count": solver_failure_count,
        "unattainable_target_count": unattainable_target_count,
        **metadata,
    }


def nominal_control_family_metrics(old_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in old_rows:
        grouped[str(row["template_id"])].append(row)
    out: list[dict[str, Any]] = []
    for family_id, rows in sorted(grouped.items()):
        points = []
        errors = []
        first = rows[0]
        for row in rows:
            x = _num(row.get("nominal_bridge_control"))
            y = _num(row.get("oracle_visual_adhesion_clean"))
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))
                errors.append(abs(y - x))
        out.append(
            _family_metric_row(
                analysis="old_nominal_control_identifiability",
                split=str(first.get("split", "development_v3_diagnostic")),
                family_id=family_id,
                points=points,
                errors=errors,
                solver_failure_count=0,
                unattainable_target_count=0,
                metadata={
                    "x_column": "nominal_bridge_control",
                    "y_column": "oracle_visual_adhesion_clean",
                    "spacing_width_ratio": first.get("spacing_width_ratio", ""),
                    "profile_family": first.get("profile_family", ""),
                    "psf_blur_sigma": first.get("psf_blur_sigma", ""),
                    "row_count": first.get("row_count", ""),
                },
            )
        )
    return out


def calibrated_target_family_metrics(solver_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in solver_rows:
        grouped[(str(row["split"]), str(row["template_id"]))].append(row)
    out: list[dict[str, Any]] = []
    for (split, family_id), rows in sorted(grouped.items()):
        points: list[tuple[float, float]] = []
        errors: list[float] = []
        failure_count = 0
        unattainable_count = 0
        for row in rows:
            status = str(row.get("solver_status", ""))
            if status == "unattainable":
                unattainable_count += 1
                continue
            if status not in SUCCESS_SOLVER_STATUSES:
                failure_count += 1
                continue
            x = _num(row.get("target_visual_adhesion"))
            y = _num(row.get("achieved_oracle_visual_adhesion"))
            if not math.isfinite(x) or not math.isfinite(y):
                failure_count += 1
                continue
            points.append((x, y))
            errors.append(abs(y - x))
        attainable_mins = [_num(row.get("attainable_min")) for row in rows]
        attainable_maxs = [_num(row.get("attainable_max")) for row in rows]
        attainable_mins = [v for v in attainable_mins if math.isfinite(v)]
        attainable_maxs = [v for v in attainable_maxs if math.isfinite(v)]
        out.append(
            _family_metric_row(
                analysis="calibrated_target_acceptance",
                split=split,
                family_id=family_id,
                points=points,
                errors=errors,
                solver_failure_count=failure_count,
                unattainable_target_count=unattainable_count,
                metadata={
                    "x_column": "target_visual_adhesion",
                    "y_column": "achieved_oracle_visual_adhesion",
                    "solver_success_statuses": ";".join(sorted(SUCCESS_SOLVER_STATUSES)),
                    "solver_attainable_min_observed": min(attainable_mins) if attainable_mins else "",
                    "solver_attainable_max_observed": max(attainable_maxs) if attainable_maxs else "",
                },
            )
        )
    return out


def family_fraction_ge(rows: Sequence[dict[str, Any]], *, split: str, threshold: float = 0.99) -> float:
    values = [_num(row["spearman_rho"]) for row in rows if row.get("split") == split]
    values = [v for v in values if math.isfinite(v)]
    return float(np.mean(np.asarray(values) >= threshold)) if values else float("nan")


def family_median_spearman(rows: Sequence[dict[str, Any]], *, split: str) -> float:
    values = [_num(row["spearman_rho"]) for row in rows if row.get("split") == split]
    values = [v for v in values if math.isfinite(v)]
    return float(np.median(values)) if values else float("nan")


def build_metric_lineage_rows() -> list[dict[str, Any]]:
    return [
        {
            "metric_name": "historical_reported_within_family_fraction_ge_0_99",
            "implementation_file": "analysis/rheed_peak_saddle/run.py",
            "implementation_function": "aggregate_holdout_v3_metrics -> old_control_identifiability_v3",
            "source_table": "outputs/rheed_peak_saddle/synthetic_v3/within_family_monotonicity.csv",
            "group_column": "template_id",
            "x_column": "nominal_bridge_control",
            "y_column": "oracle_visual_adhesion_clean",
            "row_filter": "all old-control development diagnostic rows; nominal control grid swept as renderer input",
            "minimum_points": MINIMUM_FAMILY_POINTS,
            "tie_handling": "scipy.stats.spearmanr average-rank ties",
            "nan_handling": "semantic_v3.spearman keeps finite x/y pairs only; NaN if fewer than three pairs",
            "matches_preregistered_definition": 0,
            "audit_notes": "Lineage bug: this is the historical nominal-control identifiability audit, not target_visual_adhesion versus achieved_oracle_visual_adhesion.",
        },
        {
            "metric_name": "corrected_preregistered_within_family_fraction_ge_0_99",
            "implementation_file": "analysis/rheed_peak_saddle/metric_audit_v3.py",
            "implementation_function": "calibrated_target_family_metrics",
            "source_table": "outputs/rheed_peak_saddle/synthetic_v3/target_adhesion_solver_results.csv",
            "group_column": "split,template_id",
            "x_column": "target_visual_adhesion",
            "y_column": "achieved_oracle_visual_adhesion",
            "row_filter": "solver_status in {'converged'}; excludes unattainable, failed, missing, and nonfinite target/achieved rows",
            "minimum_points": MINIMUM_FAMILY_POINTS,
            "tie_handling": "scipy.stats.spearmanr average-rank ties; duplicate targets counted once by unique-target audit field",
            "nan_handling": "finite x/y pairs only; NaN if fewer than three successful targets or fewer than two unique x/y values",
            "matches_preregistered_definition": 1,
            "audit_notes": "This matches the pre-registered target-calibrated family criterion.",
        },
    ]


def corrected_metric_summary(
    original_metrics: Sequence[dict[str, Any]],
    calibrated_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    replacements = {
        "within_family_median_spearman": {
            "development_value": family_median_spearman(calibrated_rows, split="development_v3"),
            "holdout_v3_value": family_median_spearman(calibrated_rows, split="holdout_v3"),
            "threshold": ">= 0.995",
        },
        "within_family_fraction_ge_0_99": {
            "development_value": family_fraction_ge(calibrated_rows, split="development_v3"),
            "holdout_v3_value": family_fraction_ge(calibrated_rows, split="holdout_v3"),
            "threshold": ">= 0.95",
        },
    }
    rows: list[dict[str, Any]] = []
    overall_pass = True
    for row in original_metrics:
        criterion = row["criterion"]
        corrected = replacements.get(criterion)
        historical_pass = str(row.get("pass", ""))
        if corrected:
            threshold = corrected["threshold"]
            value = corrected["holdout_v3_value"]
            if criterion == "within_family_median_spearman":
                corrected_pass = value >= 0.995
            else:
                corrected_pass = value >= 0.95
            status = "PASS" if corrected_pass else "FAIL"
            rows.append(
                {
                    "criterion": criterion,
                    "threshold": threshold,
                    "historical_reported_development_value": row.get("development_value", ""),
                    "historical_reported_value": row.get("holdout_v3_value", ""),
                    "historical_reported_pass": historical_pass,
                    "corrected_development_value": corrected["development_value"],
                    "corrected_preregistered_value": value,
                    "corrected_pass": status,
                    "correction_applied": 1,
                    "correction_note": "Recomputed from target_visual_adhesion vs achieved_oracle_visual_adhesion successful solver rows.",
                }
            )
            overall_pass = overall_pass and corrected_pass
        else:
            passed = historical_pass == "PASS"
            rows.append(
                {
                    "criterion": criterion,
                    "threshold": row.get("threshold", ""),
                    "historical_reported_development_value": row.get("development_value", ""),
                    "historical_reported_value": row.get("holdout_v3_value", ""),
                    "historical_reported_pass": historical_pass,
                    "corrected_development_value": row.get("development_value", ""),
                    "corrected_preregistered_value": row.get("holdout_v3_value", ""),
                    "corrected_pass": historical_pass,
                    "correction_applied": 0,
                    "correction_note": "Unchanged original mandatory criterion.",
                }
            )
            overall_pass = overall_pass and passed
    status = "STAGE 1C PASS AFTER METRIC-LINEAGE CORRECTION" if overall_pass else "STAGE 1C REMAINS FAIL"
    return rows, status


def assert_provenance(paths: Any) -> dict[str, Any]:
    out = paths.outputs_dir / "synthetic_v3"
    receipt_path = out / "evaluation_receipt.json"
    if file_sha256(paths.repo_root / "removelist.txt") != EXPECTED_REMOVELIST_SHA256:
        raise RuntimeError("Canonical removelist hash mismatch.")
    if file_sha256(paths.annotations_dir / "stage_review_completed.csv") != EXPECTED_STAGE_REVIEW_SHA256:
        raise RuntimeError("Completed stage-review hash mismatch.")
    if "6088" not in (paths.repo_root / "removelist.txt").read_text(encoding="utf-8").split():
        text = (paths.repo_root / "removelist.txt").read_text(encoding="utf-8")
        if not any(line.strip().startswith("6088") for line in text.splitlines()):
            raise RuntimeError("Sample 6088 is not present in the canonical removelist.")
    if not receipt_path.is_file():
        raise RuntimeError("Stage 1C evaluation receipt is missing.")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not receipt.get("evaluation_completed"):
        raise RuntimeError("Stage 1C evaluation receipt is not marked complete.")
    spec_hash = (out / "frozen_semantic_spec.sha256").read_text(encoding="utf-8").strip()
    if spec_hash != receipt.get("semantic_spec_sha256"):
        raise RuntimeError("Frozen semantic spec hash does not match the evaluation receipt.")
    return {
        "receipt": receipt,
        "receipt_sha256": file_sha256(receipt_path),
        "frozen_semantic_spec_sha256": spec_hash,
        "frozen_semantic_spec_file_sha256": file_sha256(out / "frozen_semantic_spec.json"),
        "holdout_manifest_sha256": receipt.get("holdout_manifest_sha256", ""),
        "removelist_sha256": EXPECTED_REMOVELIST_SHA256,
        "stage_review_sha256": EXPECTED_STAGE_REVIEW_SHA256,
    }


def immutable_hashes(out: Path) -> dict[str, str]:
    return {
        name: file_sha256(out / name)
        for name in IMMUTABLE_EVALUATION_FILENAMES
        if (out / name).is_file()
    }


def _template_for_manifest_row(row: dict[str, Any]) -> Any:
    templates = {template.template_id: template for template in make_semantic_templates(row["split"], count=8)}
    return templates[row["template_id"]]


def reconstruct_semantic_render(row: dict[str, Any]) -> Any:
    return render_semantic_template(
        _template_for_manifest_row(row),
        target=_num(row["target_visual_adhesion"]),
        nominal_bridge_control=_num(row["solved_nominal_bridge_amplitude"]),
        image_id=row["image_id"],
    )


def _spot_by_id(spots: Sequence[SyntheticSpotTruth], spot_id: int) -> SyntheticSpotTruth | None:
    for spot in spots:
        if spot.spot_id == spot_id:
            return spot
    return None


def _semantic_pair_for_row(render: Any, pair_row: dict[str, Any]) -> Any | None:
    truth_pair_id = pair_row.get("truth_pair_id", "")
    for pair in render.pairs:
        if pair.pair_id == truth_pair_id:
            return pair
    return None


def _overlay_semantic_pair(ax: Any, render: Any, pair_row: dict[str, Any], *, clean: bool) -> None:
    image = render.morphology_signal_clean if clean else render.displayed_image
    ax.imshow(image, cmap="gray", origin="upper")
    pair = _semantic_pair_for_row(render, pair_row)
    for spot in render.spots:
        color = "deepskyblue" if not spot.missing else "crimson"
        marker = "o" if not spot.missing else "x"
        ax.scatter([spot.center_x], [spot.center_y], s=18, marker=marker, facecolors="none", edgecolors=color, linewidths=0.8)
        ax.text(spot.center_x + 1.5, spot.center_y - 1.5, f"r{spot.row_id}:k{spot.site_index}", color="yellow", fontsize=5)
    if pair is not None:
        left = spot_estimate_from_truth_v3(render.spots[pair.spot_i], pair.spot_i)
        right = spot_estimate_from_truth_v3(render.spots[pair.spot_j], pair.spot_j)
        masks = pair_masks(render.morphology_signal_clean.shape, left, right)
        ax.contour(masks.corridor_mask, levels=[0.5], colors=["orange"], linewidths=0.7)
        ax.contour(masks.background_mask, levels=[0.5], colors=["lime"], linewidths=0.5, alpha=0.7)
        saddle, path = independent_maximin_saddle(render.morphology_signal_clean, masks.seed_i_mask, masks.seed_j_mask, masks.corridor_mask)
        prod = maximum_bottleneck_saddle(render.morphology_signal_clean, masks.seed_i_mask, masks.seed_j_mask, masks.corridor_mask)
        if path:
            ys = [p[0] for p in path]
            xs = [p[1] for p in path]
            ax.plot(xs, ys, color="red", linewidth=0.7)
        ax.set_title(
            f"{'clean' if clean else 'observed'} {pair_row.get('pair_id', '')[-5:]}\n"
            f"target={_fmt(pair_row.get('target_visual_adhesion'))} oracle={_fmt(pair_row.get('oracle_visual_adhesion_clean'))} "
            f"est={_fmt(pair_row.get('estimated_adhesion_observed'))}\n"
            f"prod saddle={_fmt(prod.saddle_intensity)} oracle saddle={_fmt(saddle)}",
            fontsize=6,
        )
    ax.set_xticks([])
    ax.set_yticks([])


def _overlay_lattice_example(ax: Any, example: SyntheticRheed, *, title: str) -> None:
    ax.imshow(example.display_image, cmap="gray", origin="upper")
    for spot in example.spots:
        if spot.missing:
            ax.scatter([spot.center_x], [spot.center_y], s=38, marker="x", c="crimson", linewidths=1.1)
            label = f"r{spot.row_id}:k{spot.site_index} missing"
        elif spot.edge_or_crop_flag:
            ax.scatter([spot.center_x], [spot.center_y], s=32, marker="s", facecolors="none", edgecolors="magenta", linewidths=1.0)
            label = f"r{spot.row_id}:k{spot.site_index} crop"
        else:
            ax.scatter([spot.center_x], [spot.center_y], s=20, marker="o", facecolors="none", edgecolors="deepskyblue", linewidths=0.9)
            label = f"r{spot.row_id}:k{spot.site_index}"
        ax.text(spot.center_x + 1.5, spot.center_y - 1.5, label, color="yellow", fontsize=5)
    for pair in example.pairs:
        left = _spot_by_id(example.spots, pair.spot_i)
        right = _spot_by_id(example.spots, pair.spot_j)
        if left is None or right is None:
            continue
        color = "lime" if pair.valid_expected else "crimson"
        style = "-" if pair.valid_expected else "--"
        ax.plot([left.center_x, right.center_x], [left.center_y, right.center_y], color=color, linestyle=style, linewidth=0.7, alpha=0.85)
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_nominal_control_sweeps(path: Path, rows: Sequence[dict[str, Any]], metrics: Sequence[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metric_by_family = {row["family_id"]: row for row in metrics}
    for row in rows:
        by_family[row["template_id"]].append(row)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.8), sharex=True, sharey=True)
    for ax, (family, family_rows) in zip(axes.ravel(), sorted(by_family.items())):
        xs = [_num(row["nominal_bridge_control"]) for row in family_rows]
        ys = [_num(row["oracle_visual_adhesion_clean"]) for row in family_rows]
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.1)
        m = metric_by_family.get(family, {})
        first = family_rows[0]
        ax.set_title(
            f"{family}\nrho={_fmt(m.get('spearman_rho'))} s/w={_fmt(first.get('spacing_width_ratio'))} {first.get('profile_family')}",
            fontsize=8,
        )
        ax.grid(alpha=0.25)
        ax.set_xlabel("nominal_bridge_control")
        ax.set_ylabel("oracle_visual_adhesion_clean")
    fig.suptitle("Historical Nominal-Control Within-Family Sweeps", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_calibrated_target_sweeps(path: Path, rows: Sequence[dict[str, Any]], metrics: Sequence[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    by_family: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    metric_by_family = {(row["split"], row["family_id"]): row for row in metrics}
    for row in rows:
        by_family[(row["split"], row["template_id"])].append(row)
    chosen = sorted(by_family.items())
    fig, axes = plt.subplots(4, 4, figsize=(15, 13.5), sharex=True, sharey=True)
    for ax, ((split, family), family_rows) in zip(axes.ravel(), chosen):
        success = [row for row in family_rows if row.get("solver_status") in SUCCESS_SOLVER_STATUSES and math.isfinite(_num(row.get("achieved_oracle_visual_adhesion")))]
        unattainable = [row for row in family_rows if row.get("solver_status") == "unattainable"]
        xs = [_num(row["target_visual_adhesion"]) for row in success]
        ys = [_num(row["achieved_oracle_visual_adhesion"]) for row in success]
        ax.plot([0, 1], [0, 1], color="0.65", linewidth=0.8, linestyle=":")
        if xs:
            ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.1, color="tab:blue")
        if unattainable:
            ux = [_num(row["target_visual_adhesion"]) for row in unattainable]
            ax.scatter(ux, [0.02] * len(ux), marker="x", c="crimson", s=18, label="unattainable")
        m = metric_by_family.get((split, family), {})
        ax.set_title(
            f"{family.replace(split + '_', '')} {split.replace('_v3', '')}\n"
            f"rho={_fmt(m.get('spearman_rho'))} tau={_fmt(m.get('kendall_tau_b'))} inv={m.get('strict_inversion_count', '')}\n"
            f"MAE={_fmt(m.get('target_achievement_mae'))} n={m.get('number_of_successful_targets', '')} "
            f"[{_fmt(m.get('attainable_target_min'))},{_fmt(m.get('attainable_target_max'))}]",
            fontsize=7,
        )
        ax.grid(alpha=0.25)
        ax.set_xlim(-0.02, 1.0)
        ax.set_ylim(-0.02, 1.0)
        ax.set_xlabel("target_visual_adhesion")
        ax.set_ylabel("achieved_oracle_visual_adhesion")
    fig.suptitle("Target-Calibrated Within-Family Sweeps", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_rank_inversions_visual(
    png_path: Path,
    pdf_path: Path,
    inversion_rows: Sequence[dict[str, Any]],
    pair_rows: Sequence[dict[str, Any]],
    manifest_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    pair_by_id = {row["pair_id"]: row for row in pair_rows}
    manifest_by_image = {row["image_id"]: row for row in manifest_rows}
    render_cache: dict[str, Any] = {}
    chosen = list(inversion_rows[:6])
    fig, axes = plt.subplots(len(chosen) * 2, 2, figsize=(10.8, max(6.0, len(chosen) * 3.0)))
    reconstruction_rows: list[dict[str, Any]] = []
    if len(chosen) == 1:
        axes = np.asarray(axes).reshape(2, 2)
    for idx, inv in enumerate(chosen):
        for side, pair_key in enumerate(("pair_a", "pair_b")):
            pair_id = inv[pair_key]
            pair_row = pair_by_id[pair_id]
            image_id = pair_row["image_id"]
            if image_id not in render_cache:
                render_cache[image_id] = reconstruct_semantic_render(manifest_by_image[image_id])
                reconstruction_rows.append(
                    {
                        "figure": "largest_rank_inversions_visual",
                        "image_id": image_id,
                        "source_table": "holdout_v3_manifest.csv",
                        "source_row_sha256": stable_row_hash(manifest_by_image[image_id]),
                        "display_only": 1,
                    }
                )
            render = render_cache[image_id]
            row = idx * 2 + side
            _overlay_semantic_pair(axes[row, 0], render, pair_row, clean=True)
            _overlay_semantic_pair(axes[row, 1], render, pair_row, clean=False)
            axes[row, 0].text(
                0.01,
                0.02,
                f"inversion {idx + 1} {'A' if side == 0 else 'B'}: "
                f"oracle A={_fmt(inv.get('oracle_a'))} B={_fmt(inv.get('oracle_b'))}; "
                f"estimated A={_fmt(inv.get('estimated_a'))} B={_fmt(inv.get('estimated_b'))}",
                transform=axes[row, 0].transAxes,
                color="white",
                fontsize=6,
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1},
            )
    fig.suptitle("Largest Rank Inversions: Reconstructed Display Panels", fontsize=12)
    fig.tight_layout()
    fig.savefig(png_path, dpi=170)
    fig.savefig(pdf_path)
    plt.close(fig)
    return reconstruction_rows


def plot_high_adhesion_errors(
    png_path: Path,
    pdf_path: Path,
    pair_rows: Sequence[dict[str, Any]],
    manifest_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    candidates = []
    for row in pair_rows:
        oracle = _num(row.get("oracle_visual_adhesion_clean"))
        estimate = _num(row.get("estimated_adhesion_observed"))
        if row.get("split") == "holdout_v3" and math.isfinite(oracle) and math.isfinite(estimate) and oracle >= 0.55:
            candidates.append((abs(estimate - oracle), row))
    chosen = [row for _, row in sorted(candidates, key=lambda item: item[0], reverse=True)[:8]]
    manifest_by_image = {row["image_id"]: row for row in manifest_rows}
    render_cache: dict[str, Any] = {}
    fig, axes = plt.subplots(len(chosen), 2, figsize=(10.8, max(5.5, len(chosen) * 1.55)))
    reconstruction_rows: list[dict[str, Any]] = []
    if len(chosen) == 1:
        axes = np.asarray(axes).reshape(1, 2)
    for row_index, pair_row in enumerate(chosen):
        image_id = pair_row["image_id"]
        if image_id not in render_cache:
            render_cache[image_id] = reconstruct_semantic_render(manifest_by_image[image_id])
            reconstruction_rows.append(
                {
                    "figure": "high_adhesion_error_cases",
                    "image_id": image_id,
                    "source_table": "holdout_v3_manifest.csv",
                    "source_row_sha256": stable_row_hash(manifest_by_image[image_id]),
                    "display_only": 1,
                }
            )
        render = render_cache[image_id]
        _overlay_semantic_pair(axes[row_index, 0], render, pair_row, clean=True)
        _overlay_semantic_pair(axes[row_index, 1], render, pair_row, clean=False)
        error = abs(_num(pair_row.get("estimated_adhesion_observed")) - _num(pair_row.get("oracle_visual_adhesion_clean")))
        axes[row_index, 1].text(
            0.01,
            0.02,
            f"abs error={_fmt(error)} spacing/width={_fmt(pair_row.get('spacing_over_width_truth'))} "
            f"peak ratio={_fmt(pair_row.get('peak_amplitude_ratio'))} blur={_fmt(pair_row.get('psf_blur_sigma'))} "
            f"bg={_fmt(pair_row.get('smooth_background_strength'))} confidence={_fmt(pair_row.get('pair_measurement_confidence'))}",
            transform=axes[row_index, 1].transAxes,
            color="white",
            fontsize=6,
            bbox={"facecolor": "black", "alpha": 0.45, "pad": 1},
        )
    fig.suptitle("High-Adhesion Error Cases: Reconstructed Display Panels", fontsize=12)
    fig.tight_layout()
    fig.savefig(png_path, dpi=170)
    fig.savefig(pdf_path)
    plt.close(fig)
    return reconstruction_rows


def plot_lattice_examples_visual(png_path: Path, pdf_path: Path) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    wanted = [
        "development_v2_sweep_00_0",
        "development_v2_challenge_missing_one_site",
        "development_v2_challenge_missing_two_sites",
        "development_v2_challenge_duplicate_local_maximum",
        "development_v2_challenge_nearby_rows",
        "development_v2_challenge_mild_curvature",
        "development_v2_challenge_partial_crop_endpoint",
        "development_v2_challenge_valid_near_missing_site",
    ]
    examples = {example.image_id: example for example in make_synthetic_split_v2("development_v2")}
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    reconstruction_rows: list[dict[str, Any]] = []
    for ax, image_id in zip(axes.ravel(), wanted):
        example = examples[image_id]
        _overlay_lattice_example(ax, example, title=image_id.replace("development_v2_", ""))
        reconstruction_rows.append(
            {
                "figure": "lattice_indexing_examples_visual",
                "image_id": image_id,
                "source_table": "synthetic_v2/development_manifest.csv",
                "source_row_sha256": hashlib.sha256(image_id.encode("utf-8")).hexdigest(),
                "display_only": 1,
            }
        )
    fig.suptitle("Lattice Indexing Overlays: r<row_id>:k<site_index>", fontsize=12)
    fig.tight_layout()
    fig.savefig(png_path, dpi=170)
    fig.savefig(pdf_path)
    plt.close(fig)
    return reconstruction_rows


def write_human_review_index(report_dir: Path, out: Path) -> None:
    links = [
        ("Nominal-control family sweeps", "nominal_control_within_family_sweeps.png"),
        ("Target-calibrated family sweeps", "calibrated_target_within_family_sweeps.png"),
        ("Target versus achieved", "target_vs_achieved_oracle_adhesion.png"),
        ("Observed versus oracle", "estimated_vs_oracle_adhesion.png"),
        ("Rank-inversion panels", "largest_rank_inversions_visual.png"),
        ("Lattice-index overlays", "lattice_indexing_examples_visual.png"),
        ("High-adhesion error panels", "high_adhesion_error_cases.png"),
    ]
    table_links = [
        ("Metric-lineage table", out / "metric_lineage_audit.csv"),
        ("Nominal-control family metrics", out / "nominal_control_family_metrics.csv"),
        ("Calibrated-target family metrics", out / "calibrated_target_family_metrics.csv"),
        ("Metric correction summary", out / "metric_correction_summary.csv"),
    ]
    parts = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Stage 1C-R Human Review</title>",
        "<h1>Stage 1C-R Human Review Package</h1>",
        "<p>All panels are plot-only diagnostics derived from immutable Stage 1C tables. Reconstructed image panels are for display only.</p>",
        "<h2>Figures</h2>",
    ]
    for label, filename in links:
        parts.append(f"<h3>{html.escape(label)}</h3>")
        parts.append(f"<p><a href='{html.escape(filename)}'>{html.escape(filename)}</a></p>")
        parts.append(f"<img src='{html.escape(filename)}' style='max-width:100%; border:1px solid #ccc'>")
    parts.append("<h2>Tables</h2><ul>")
    for label, path in table_links:
        rel = path.relative_to(report_dir) if path.is_relative_to(report_dir) else Path("../../../outputs/rheed_peak_saddle/synthetic_v3") / path.name
        parts.append(f"<li><a href='{html.escape(rel.as_posix())}'>{html.escape(label)}</a></li>")
    parts.append("</ul>")
    (report_dir / "human_review_index.html").write_text("\n".join(parts), encoding="utf-8")


def write_approval_template(paths: Any) -> None:
    approval_dir = paths.annotations_dir / "approvals"
    approval_dir.mkdir(parents=True, exist_ok=True)
    path = approval_dir / "checkpoint_1c_visual_review_template.txt"
    if path.is_file() and path.read_text(encoding="utf-8").strip() == "APPROVED":
        return
    path.write_text(
        "\n".join(
            [
                "Review the repaired visual diagnostics.",
                "",
                "Check:",
                "[ ] Target-calibrated family sweeps are monotonic.",
                "[ ] Rank-inversion panels show physically understandable failures.",
                "[ ] Lattice labels are correctly overlaid on actual images.",
                "[ ] Missing sites are not paired across.",
                "[ ] High-adhesion outliers do not reveal an obvious systematic",
                "    measurement failure.",
                "[ ] I understand the amended PASS/FAIL decision.",
                "",
                "Replace this file with exactly:",
                "",
                "    APPROVED",
                "",
                "only after all checks are satisfactory.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_metric_audit_report(
    path: Path,
    *,
    provenance: dict[str, Any],
    lineage_rows: Sequence[dict[str, Any]],
    nominal_rows: Sequence[dict[str, Any]],
    calibrated_rows: Sequence[dict[str, Any]],
    correction_rows: Sequence[dict[str, Any]],
    amended_status: str,
    before_hashes: dict[str, str],
    after_hashes: dict[str, str],
) -> None:
    original_fraction = next(row for row in correction_rows if row["criterion"] == "within_family_fraction_ge_0_99")
    median_row = next(row for row in correction_rows if row["criterion"] == "within_family_median_spearman")
    changed_hashes = [name for name in before_hashes if before_hashes.get(name) != after_hashes.get(name)]
    calibrated_table = [
        "| Split | Family | n | rho | tau-b | inversions | MAE | unattainable |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in calibrated_rows:
        calibrated_table.append(
            f"| {row['split']} | `{row['family_id']}` | {row['number_of_successful_targets']} | "
            f"{_fmt(row['spearman_rho'])} | {_fmt(row['kendall_tau_b'])} | {row['strict_inversion_count']} | "
            f"{_fmt(row['target_achievement_mae'])} | {row['unattainable_target_count']} |"
        )
    text = "\n".join(
        [
            "# Checkpoint 1C-R: Metric Lineage Audit",
            "",
            "## Provenance And Immutability",
            f"- Evaluation receipt SHA256: `{provenance['receipt_sha256']}`",
            f"- Frozen semantic spec hash: `{provenance['frozen_semantic_spec_sha256']}`",
            f"- Frozen semantic spec file SHA256: `{provenance['frozen_semantic_spec_file_sha256']}`",
            f"- Holdout-v3 manifest hash from receipt: `{provenance['holdout_manifest_sha256']}`",
            f"- Removelist SHA256: `{provenance['removelist_sha256']}`",
            f"- Stage-review SHA256: `{provenance['stage_review_sha256']}`",
            f"- Immutable evaluation files changed during audit: `{','.join(changed_hashes) if changed_hashes else 'none'}`",
            "",
            "## Original Stage 1C Result",
            "- Original reported status: `STAGE 1C FAIL`",
            f"- Original reported failing value: `{original_fraction['historical_reported_value']}`",
            f"- Original threshold: `{original_fraction['threshold']}`",
            "",
            "## Failing Metric Lineage",
            "- Historical source function: `aggregate_holdout_v3_metrics()` consumed `family_rows` from `old_control_identifiability_v3()`.",
            "- Historical x/y: `nominal_bridge_control` versus `oracle_visual_adhesion_clean`.",
            "- Pre-registered x/y: `target_visual_adhesion` versus `achieved_oracle_visual_adhesion`.",
            "- Lineage decision: `BRANCH A - CLEAR METRIC-LINEAGE BUG`.",
            "",
            "## Corrected Preregistered Metrics",
            f"- Corrected within-family median Spearman: `{_fmt(median_row['corrected_preregistered_value'])}`.",
            f"- Corrected within-family fraction >= 0.99: `{_fmt(original_fraction['corrected_preregistered_value'])}`.",
            "- Successful solver rows use `solver_status == converged`; unattainable and failed rows are excluded from the acceptance denominator.",
            "",
            "## Family-Level Calibrated Metrics",
            *calibrated_table,
            "",
            "## Nominal-Control Diagnostic",
            f"- Nominal-control families audited: `{len(nominal_rows)}`.",
            "- This remains diagnostic only and is not the target-calibrated acceptance metric.",
            "",
            "## Visual Diagnostics",
            "- Repaired nominal-control sweep: `reports/rheed_peak_saddle/synthetic_v3/nominal_control_within_family_sweeps.png`.",
            "- Repaired target-calibrated sweep: `reports/rheed_peak_saddle/synthetic_v3/calibrated_target_within_family_sweeps.png`.",
            "- Repaired rank inversion panels: `largest_rank_inversions_visual.png` and `.pdf`.",
            "- Repaired lattice overlay panels: `lattice_indexing_examples_visual.png` and `.pdf`.",
            "- High-adhesion error panels: `high_adhesion_error_cases.png` and `.pdf`.",
            "",
            "## Boundary Confirmations",
            "- No holdout-v3 image, target, oracle, prediction, manifest, semantic spec, or receipt was modified.",
            "- No holdout-v3 predictions were rerun.",
            "- No renderer, detector, measurement algorithm, or model parameter changed.",
            "- No real RHEED images, AFM data, Rq targets, model training, or Stage 2 were used.",
            "",
            "## Final Amended Status",
            amended_status,
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_metric_audit(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    paths = make_paths(config)
    out = paths.outputs_dir / "synthetic_v3"
    report_dir = paths.reports_dir / "synthetic_v3"
    out.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    provenance = assert_provenance(paths)
    before_hashes = immutable_hashes(out)

    old_rows = read_csv_rows(out / "old_control_identifiability.csv")
    solver_rows = read_csv_rows(out / "target_adhesion_solver_results.csv")
    original_metrics = read_csv_rows(out / "holdout_v3_metrics.csv")
    pair_rows = read_csv_rows(out / "pair_level_measurement_fidelity.csv")
    inversion_rows = read_csv_rows(out / "rank_inversion_manifest.csv")
    manifest_rows = read_csv_rows(out / "holdout_v3_manifest.csv")

    lineage_rows = build_metric_lineage_rows()
    nominal_rows = nominal_control_family_metrics(old_rows)
    calibrated_rows = calibrated_target_family_metrics(solver_rows)
    correction_rows, amended_status = corrected_metric_summary(original_metrics, calibrated_rows)

    write_csv(out / "metric_lineage_audit.csv", lineage_rows)
    write_csv(out / "nominal_control_family_metrics.csv", nominal_rows)
    write_csv(out / "calibrated_target_family_metrics.csv", calibrated_rows)
    write_csv(out / "metric_correction_summary.csv", correction_rows)

    plot_nominal_control_sweeps(report_dir / "nominal_control_within_family_sweeps.png", old_rows, nominal_rows)
    plot_calibrated_target_sweeps(report_dir / "calibrated_target_within_family_sweeps.png", solver_rows, calibrated_rows)
    reconstruction_rows = []
    reconstruction_rows.extend(
        plot_rank_inversions_visual(
            report_dir / "largest_rank_inversions_visual.png",
            report_dir / "largest_rank_inversions_visual.pdf",
            inversion_rows,
            pair_rows,
            manifest_rows,
        )
    )
    reconstruction_rows.extend(
        plot_high_adhesion_errors(
            report_dir / "high_adhesion_error_cases.png",
            report_dir / "high_adhesion_error_cases.pdf",
            pair_rows,
            manifest_rows,
        )
    )
    reconstruction_rows.extend(
        plot_lattice_examples_visual(
            report_dir / "lattice_indexing_examples_visual.png",
            report_dir / "lattice_indexing_examples_visual.pdf",
        )
    )
    write_csv(out / "visual_reconstruction_manifest.csv", reconstruction_rows)

    after_hashes = immutable_hashes(out)
    hash_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "audit_command": AUDIT_COMMAND,
        "immutable_hashes_before": before_hashes,
        "immutable_hashes_after": after_hashes,
        "immutable_hashes_changed": [name for name in before_hashes if before_hashes.get(name) != after_hashes.get(name)],
        "derived_outputs": {
            "metric_lineage_audit.csv": file_sha256(out / "metric_lineage_audit.csv"),
            "nominal_control_family_metrics.csv": file_sha256(out / "nominal_control_family_metrics.csv"),
            "calibrated_target_family_metrics.csv": file_sha256(out / "calibrated_target_family_metrics.csv"),
            "metric_correction_summary.csv": file_sha256(out / "metric_correction_summary.csv"),
            "visual_reconstruction_manifest.csv": file_sha256(out / "visual_reconstruction_manifest.csv"),
        },
    }
    (out / "metric_audit_hashes_before_after.json").write_text(json.dumps(hash_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    write_metric_audit_report(
        paths.reports_dir / "checkpoint_1c_metric_audit.md",
        provenance=provenance,
        lineage_rows=lineage_rows,
        nominal_rows=nominal_rows,
        calibrated_rows=calibrated_rows,
        correction_rows=correction_rows,
        amended_status=amended_status,
        before_hashes=before_hashes,
        after_hashes=after_hashes,
    )
    write_human_review_index(report_dir, out)
    write_approval_template(paths)

    return {
        "outputs_dir": out,
        "reports_dir": report_dir,
        "amended_status": amended_status,
        "historical_reported_value": next(row for row in correction_rows if row["criterion"] == "within_family_fraction_ge_0_99")["historical_reported_value"],
        "corrected_preregistered_value": next(row for row in correction_rows if row["criterion"] == "within_family_fraction_ge_0_99")["corrected_preregistered_value"],
        "immutable_hashes_changed": hash_payload["immutable_hashes_changed"],
    }
