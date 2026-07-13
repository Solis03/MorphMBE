"""Stage 2A real RHEED shadow-mode diagnostics without AFM/Rq access."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from analysis.rheed_peak_saddle.pair_features import PairFeature, PairMasks, measure_pair_adhesion
from analysis.rheed_peak_saddle.preprocessing import RheedChannels, make_channels
from analysis.rheed_peak_saddle.row_grouping import (
    LatticeRowResult,
    RowGroupingResult,
    assign_lattice_indices,
    form_lattice_adjacent_pairs,
    group_spot_rows,
)
from analysis.rheed_peak_saddle.spot_detection import SpotEstimate, detect_spots
from analysis.rheed_peak_saddle.synthetic import make_synthetic_split_v2


EXPECTED_REMOVELIST_SHA256 = "8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b"
EXPECTED_STAGE_REVIEW_SHA256 = "862df0397683a19c24d616b2ba42b088538048750a63a89eb17593c1b4c9081e"
EXPECTED_EVALUATION_RECEIPT_SHA256 = "a3619bad7a8517d083c4fd73852a6666c235bf21a2f46c5a8ce02f0869541e9f"
EXPECTED_SEMANTIC_SPEC_CONTENT_SHA256 = "ffa417f8c5a67f8a3ede3e532464b3a82a783c47a3362dc4abb85cc4f8ed0689"
EXPECTED_SEMANTIC_SPEC_FILE_SHA256 = "98aebd38a1e59f1f120cc143fd6b068f6113bfd092b7a1aae3106b8887272a26"
EXPECTED_HOLDOUT_MANIFEST_HASH = "0b107e3433de5d757ca70e72cf3a302cfa97fe1dd1fa80d410394b658305dc84"
REAL_DIAGNOSTIC_SPLIT_SEED = 2026071302
STAGE2A_COMMAND = (
    "PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_peak_saddle.run "
    "--config configs/rheed_peak_saddle.yaml --stage real_diagnostics"
)
ANON_PREFIX = "RPS2A"


@dataclass(frozen=True)
class RealSample:
    sample_id: str
    anonymous_review_id: str
    manual_rheed_path: Path
    manual_rheed_filename: str
    approved_stage: str
    comparable_stage_group: str
    stage_confidence: str
    image_sha256: str
    input_status: str
    skip_reason: str = ""


@dataclass(frozen=True)
class RealMeasurement:
    sample: RealSample
    image: np.ndarray
    channels: RheedChannels
    spots: tuple[SpotEstimate, ...]
    grouping: RowGroupingResult
    lattice: LatticeRowResult
    pairs: tuple[Any, ...]
    features: dict[str, PairFeature]
    masks: dict[str, PairMasks]
    quality: dict[str, Any]


def make_stage2a_paths(config: dict[str, Any]) -> Any:
    repo_root = (Path(__file__).resolve().parents[2] / str(config.get("repo_root", "."))).resolve()
    outputs_dir = (repo_root / str(config.get("outputs_dir", "outputs/rheed_peak_saddle"))).resolve()
    reports_dir = (repo_root / str(config.get("reports_dir", "reports/rheed_peak_saddle"))).resolve()
    annotations_dir = (repo_root / str(config.get("annotations_dir", "annotations/rheed_peak_saddle"))).resolve()
    approvals_dir = annotations_dir / "approvals"
    manual_root = (repo_root / str(config.get("manual_selection_root", "data/manual_selection"))).resolve()
    for path in (outputs_dir, reports_dir, annotations_dir, approvals_dir):
        path.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        repo_root=repo_root,
        outputs_dir=outputs_dir,
        reports_dir=reports_dir,
        annotations_dir=annotations_dir,
        approvals_dir=approvals_dir,
        manual_root=manual_root,
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


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
    return f"{number:.4g}" if math.isfinite(number) else str(value)


def _median(values: Iterable[Any]) -> float:
    nums = [_num(v) for v in values]
    nums = [v for v in nums if math.isfinite(v)]
    return float(np.median(nums)) if nums else float("nan")


def _percentile(values: Iterable[Any], q: float) -> float:
    nums = [_num(v) for v in values]
    nums = [v for v in nums if math.isfinite(v)]
    return float(np.percentile(nums, q)) if nums else float("nan")


def validate_stage2a_gates(config: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    paths = make_stage2a_paths(config)
    approval_path = paths.approvals_dir / "checkpoint_1c_visual_review_template.txt"
    if not approval_path.is_file() or approval_path.read_text(encoding="utf-8").strip() != "APPROVED":
        raise SystemExit(
            "HUMAN ACTION REQUIRED:\n"
            "Review the Stage 1C repaired synthetic diagnostics and replace the\n"
            "approval file contents with exactly APPROVED."
        )
    removelist_path = paths.repo_root / "removelist.txt"
    stage_review_path = paths.annotations_dir / "stage_review_completed.csv"
    receipt_path = paths.outputs_dir / "synthetic_v3" / "evaluation_receipt.json"
    semantic_spec_path = paths.outputs_dir / "synthetic_v3" / "frozen_semantic_spec.json"
    semantic_spec_hash_path = paths.outputs_dir / "synthetic_v3" / "frozen_semantic_spec.sha256"
    metric_audit_path = paths.reports_dir / "checkpoint_1c_metric_audit.md"
    required = [removelist_path, stage_review_path, receipt_path, semantic_spec_path, semantic_spec_hash_path, metric_audit_path]
    for path in required:
        if not path.is_file():
            raise SystemExit(f"Required Stage 2A gate file is missing: {path}")
    if file_sha256(removelist_path) != EXPECTED_REMOVELIST_SHA256:
        raise SystemExit("Canonical removelist hash mismatch.")
    if not any(line.strip().startswith("6088") for line in removelist_path.read_text(encoding="utf-8").splitlines()):
        raise SystemExit("Sample 6088 is not present in the canonical removelist.")
    if file_sha256(stage_review_path) != EXPECTED_STAGE_REVIEW_SHA256:
        raise SystemExit("Completed stage-review hash mismatch.")
    if file_sha256(receipt_path) != EXPECTED_EVALUATION_RECEIPT_SHA256:
        raise SystemExit("Stage 1C evaluation receipt hash mismatch.")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    semantic_content_hash = semantic_spec_hash_path.read_text(encoding="utf-8").strip()
    if semantic_content_hash != EXPECTED_SEMANTIC_SPEC_CONTENT_SHA256:
        raise SystemExit("Frozen semantic-spec content hash mismatch.")
    if file_sha256(semantic_spec_path) != EXPECTED_SEMANTIC_SPEC_FILE_SHA256:
        raise SystemExit("Frozen semantic-spec file hash mismatch.")
    if receipt.get("holdout_manifest_sha256") != EXPECTED_HOLDOUT_MANIFEST_HASH:
        raise SystemExit("Holdout-v3 manifest hash mismatch.")
    if "STAGE 1C PASS AFTER METRIC-LINEAGE CORRECTION" not in metric_audit_path.read_text(encoding="utf-8"):
        raise SystemExit("Checkpoint 1C-R amended PASS was not found.")
    approval_stat = approval_path.stat()
    return paths, {
        "approval_path": approval_path.as_posix(),
        "approval_sha256": file_sha256(approval_path),
        "approval_mtime": datetime.fromtimestamp(approval_stat.st_mtime, timezone.utc).isoformat(),
        "removelist_path": removelist_path.as_posix(),
        "removelist_sha256": EXPECTED_REMOVELIST_SHA256,
        "stage_review_path": stage_review_path.as_posix(),
        "stage_review_sha256": EXPECTED_STAGE_REVIEW_SHA256,
        "evaluation_receipt_sha256": EXPECTED_EVALUATION_RECEIPT_SHA256,
        "frozen_semantic_spec_sha256": semantic_content_hash,
        "frozen_semantic_spec_file_sha256": file_sha256(semantic_spec_path),
        "holdout_manifest_sha256": receipt.get("holdout_manifest_sha256", ""),
    }


def load_removed_ids(path: Path) -> set[str]:
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        token = line.split()[0]
        if token.isdigit():
            ids.add(token)
    return ids


def anonymous_id(sample_id: str, index: int) -> str:
    digest = hashlib.sha256(f"{REAL_DIAGNOSTIC_SPLIT_SEED}:{sample_id}".encode("utf-8")).hexdigest()[:8]
    first = chr(ord("A") + ((index - 1) // 26) % 26)
    second = chr(ord("A") + (index - 1) % 26)
    letter_digest = "".join(chr(ord("a") + int(char, 16)) for char in digest)
    return f"RPSA_{first}{second}_{letter_digest}"


def build_real_input_manifest(paths: Any) -> tuple[list[RealSample], list[dict[str, Any]]]:
    removed = load_removed_ids(paths.repo_root / "removelist.txt")
    review_rows = read_csv_rows(paths.annotations_dir / "stage_review_completed.csv")
    samples: list[RealSample] = []
    manifest_rows: list[dict[str, Any]] = []
    for index, row in enumerate(sorted(review_rows, key=lambda r: str(r["sample_id"]))):
        sid = str(row["sample_id"])
        anon = anonymous_id(sid, index + 1)
        rel = Path("data/manual_selection") / sid / "RHEED" / row["manual_rheed_filename"]
        path = paths.repo_root / rel
        status = "included"
        skip = ""
        if sid in removed:
            status = "skipped"
            skip = "canonical_removelist"
        elif str(row.get("user_approved", "")).strip() != "1":
            status = "skipped"
            skip = "stage_review_not_user_approved"
        elif not row["manual_rheed_filename"].lower().startswith("select"):
            status = "skipped"
            skip = "manual_filename_not_select"
        elif not path.is_file():
            status = "skipped"
            skip = "manual_screenshot_missing"
        image_hash = file_sha256(path) if path.is_file() else ""
        sample = RealSample(
            sample_id=sid,
            anonymous_review_id=anon,
            manual_rheed_path=path,
            manual_rheed_filename=row["manual_rheed_filename"],
            approved_stage=row["approved_stage"],
            comparable_stage_group=row["comparable_stage_group"],
            stage_confidence=row.get("stage_confidence", ""),
            image_sha256=image_hash,
            input_status=status,
            skip_reason=skip,
        )
        if status == "included":
            samples.append(sample)
        manifest_rows.append(
            {
                "sample_id": sid,
                "anonymous_review_id": anon,
                "manual_rheed_path": rel.as_posix(),
                "manual_rheed_filename": row["manual_rheed_filename"],
                "approved_stage": row["approved_stage"],
                "comparable_stage_group": row["comparable_stage_group"],
                "removelist_status": "removed" if sid in removed else "not_removed",
                "input_status": status,
                "skip_reason": skip,
                "image_sha256": image_hash,
            }
        )
    return samples, manifest_rows


def load_grayscale_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"Empty image: {path}")
    return arr / 255.0


def prune_lattice_duplicate_spots(detected: Sequence[SpotEstimate]) -> tuple[SpotEstimate, ...]:
    if len(detected) < 3:
        return tuple(detected)
    grouping = group_spot_rows(detected)
    lattice = assign_lattice_indices(detected, grouping)
    keep = {assignment.detection_index for assignment in lattice.assignments if not assignment.duplicate_candidate and assignment.lattice_assignment_confidence >= 0.20}
    return tuple(spot for index, spot in enumerate(detected) if index in keep) if keep else tuple(detected)


def measure_real_sample(sample: RealSample) -> RealMeasurement:
    image = load_grayscale_image(sample.manual_rheed_path)
    channels = make_channels(image)
    detected = prune_lattice_duplicate_spots(detect_spots(channels.linear, min_distance=14.0))
    grouping = group_spot_rows(detected)
    lattice = assign_lattice_indices(detected, grouping)
    pairs = form_lattice_adjacent_pairs(detected, grouping, lattice, image_id=sample.anonymous_review_id)
    features: dict[str, PairFeature] = {}
    masks: dict[str, PairMasks] = {}
    for pair in pairs:
        feature, pair_mask = measure_pair_adhesion(channels.linear, detected, pair, image_id=sample.anonymous_review_id, ridge_channel=channels.horizontal_ridge)
        features[pair.pair_id] = feature
        masks[pair.pair_id] = pair_mask
    valid_features = [feature for feature in features.values() if feature.valid]
    valid_fraction = len(valid_features) / max(len(features), 1)
    detection_quality = float(np.median([spot.detection_confidence for spot in detected])) if detected else 0.0
    background_fit_quality = float(1.0 / (1.0 + np.nanmedian(np.abs(channels.linear - channels.background)))) if image.size else 0.0
    quality_score = float(np.clip(0.35 * detection_quality + 0.35 * valid_fraction + 0.30 * grouping.row_consistency, 0.0, 1.0))
    quality_label = "high" if quality_score >= 0.75 else "medium" if quality_score >= 0.45 else "low"
    quality = {
        "spot_detection_quality": detection_quality,
        "background_fit_quality": background_fit_quality,
        "overall_measurement_quality": quality_score,
        "measurement_quality_label": quality_label,
        "row_consistency": grouping.row_consistency,
        "valid_pair_fraction": valid_fraction,
    }
    return RealMeasurement(sample, image, channels, detected, grouping, lattice, pairs, features, masks, quality)


def assignment_maps(lattice: LatticeRowResult) -> dict[int, Any]:
    return {assignment.detection_index: assignment for assignment in lattice.assignments}


def measurement_tables(measurements: Sequence[RealMeasurement]) -> dict[str, list[dict[str, Any]]]:
    spot_rows: list[dict[str, Any]] = []
    row_rows: list[dict[str, Any]] = []
    lattice_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for measurement in measurements:
        sample = measurement.sample
        amap = assignment_maps(measurement.lattice)
        for idx, spot in enumerate(measurement.spots):
            assignment = amap.get(idx)
            spot_rows.append(
                {
                    "anonymous_review_id": sample.anonymous_review_id,
                    "spot_i_id": idx,
                    "center_x": spot.center_x,
                    "center_y": spot.center_y,
                    "peak_intensity": spot.peak_intensity,
                    "sigma_x": spot.sigma_x,
                    "sigma_y": spot.sigma_y,
                    "equivalent_width": spot.equivalent_width,
                    "eccentricity": spot.eccentricity,
                    "local_background": spot.local_background,
                    "fit_residual": spot.fit_residual,
                    "saturation_flag": spot.saturation_flag,
                    "edge_or_crop_flag": spot.edge_or_crop_flag,
                    "detection_confidence": spot.detection_confidence,
                    "row_id": assignment.row_label if assignment else "",
                    "lattice_index": assignment.lattice_index if assignment else "",
                }
            )
        labels = sorted(set(measurement.grouping.row_labels))
        for label in labels:
            indices = [i for i, row_label in enumerate(measurement.grouping.row_labels) if row_label == label]
            row_rows.append(
                {
                    "anonymous_review_id": sample.anonymous_review_id,
                    "row_id": label,
                    "dominant_angle_degrees": measurement.grouping.dominant_angle_degrees,
                    "row_consistency": measurement.grouping.row_consistency,
                    "spot_count": len(indices),
                    "median_rotated_y": _median(measurement.grouping.rotated_y[i] for i in indices),
                }
            )
        for assignment in measurement.lattice.assignments:
            lattice_rows.append(
                {
                    "anonymous_review_id": sample.anonymous_review_id,
                    "spot_i_id": assignment.detection_index,
                    "row_id": assignment.row_label,
                    "lattice_index": assignment.lattice_index,
                    "local_u": assignment.local_u,
                    "local_v": assignment.local_v,
                    "row_spacing": assignment.row_spacing,
                    "lattice_fit_residual": assignment.lattice_fit_residual,
                    "lattice_assignment_confidence": assignment.lattice_assignment_confidence,
                    "duplicate_candidate": assignment.duplicate_candidate,
                }
            )
        valid_adhesion = []
        invalid_count = 0
        for pair in measurement.pairs:
            feature = measurement.features[pair.pair_id]
            left_assignment = amap.get(pair.spot_i)
            right_assignment = amap.get(pair.spot_j)
            row = {
                "anonymous_review_id": sample.anonymous_review_id,
                "anonymous_pair_id": feature.pair_id,
                "spot_i_id": feature.spot_i,
                "spot_j_id": feature.spot_j,
                "row_id": feature.row_label,
                "lattice_index_i": left_assignment.lattice_index if left_assignment else "",
                "lattice_index_j": right_assignment.lattice_index if right_assignment else "",
                "peak_i": feature.peak_i,
                "peak_j": feature.peak_j,
                "saddle": feature.saddle_intensity,
                "local_background": feature.background_intensity,
                "raw_adhesion": feature.raw_peak_saddle_adhesion_unclipped,
                "clipped_adhesion": feature.raw_peak_saddle_adhesion,
                "isolation_persistence": feature.isolation_persistence,
                "direct_corridor_valley_ratio": feature.direct_corridor_valley_ratio,
                "corridor_mean_ratio": feature.corridor_mean_ratio,
                "ridge_energy_ratio": feature.ridge_energy_ratio,
                "bridge_width_ratio": feature.bridge_width_ratio,
                "spot_spacing_over_width": feature.spot_spacing_over_width,
                "pair_measurement_confidence": feature.pair_measurement_confidence,
                "valid_pair": feature.valid,
                "invalid_reason": feature.invalid_reason,
            }
            pair_rows.append(row)
            if feature.valid:
                valid_adhesion.append(feature.raw_peak_saddle_adhesion)
            else:
                invalid_count += 1
                invalid_rows.append(row)
        sample_rows.append(
            {
                "anonymous_review_id": sample.anonymous_review_id,
                "adhesion_median": _median(valid_adhesion),
                "adhesion_q25": _percentile(valid_adhesion, 25),
                "adhesion_q75": _percentile(valid_adhesion, 75),
                "adhesion_iqr": _percentile(valid_adhesion, 75) - _percentile(valid_adhesion, 25) if valid_adhesion else "",
                "strongly_connected_pair_fraction": sum(v >= 0.5 for v in valid_adhesion) / max(len(valid_adhesion), 1),
                "strongly_isolated_pair_fraction": sum(v <= 0.1 for v in valid_adhesion) / max(len(valid_adhesion), 1),
                "isolation_persistence_median": _median(1.0 - v for v in valid_adhesion),
                "valid_spot_count": len(measurement.spots),
                "valid_pair_count": len(valid_adhesion),
                "valid_pair_fraction": measurement.quality["valid_pair_fraction"],
                "row_consistency": measurement.quality["row_consistency"],
                "spot_detection_quality": measurement.quality["spot_detection_quality"],
                "background_fit_quality": measurement.quality["background_fit_quality"],
                "overall_measurement_quality": measurement.quality["overall_measurement_quality"],
            }
        )
        quality_rows.append(
            {
                "anonymous_review_id": sample.anonymous_review_id,
                "image_height": measurement.image.shape[0],
                "image_width": measurement.image.shape[1],
                "spot_count": len(measurement.spots),
                "row_count": len(labels),
                "pair_count": len(measurement.pairs),
                "valid_pair_count": len(valid_adhesion),
                "invalid_pair_count": invalid_count,
                **measurement.quality,
            }
        )
    return {
        "spot_level_measurements.csv": spot_rows,
        "row_level_measurements.csv": row_rows,
        "lattice_level_measurements.csv": lattice_rows,
        "pair_level_measurements.csv": pair_rows,
        "invalid_pair_audit.csv": invalid_rows,
        "sample_level_concepts.csv": sample_rows,
        "measurement_quality.csv": quality_rows,
    }


def choose_split(samples: Sequence[RealSample], quality_rows: Sequence[dict[str, Any]], sample_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    quality_by_id = {row["anonymous_review_id"]: row for row in quality_rows}
    sample_by_id = {row["anonymous_review_id"]: row for row in sample_rows}
    targets = {"development_review": 10, "blind_validation": 10, "reserve": max(0, len(samples) - 20)}
    assigned = {key: 0 for key in targets}
    split_rows: list[dict[str, Any]] = []
    buckets: dict[str, list[RealSample]] = defaultdict(list)
    for sample in samples:
        buckets[sample.approved_stage].append(sample)
    split_order = ["development_review", "blind_validation", "reserve"]
    for stage in sorted(buckets):
        bucket = sorted(
            buckets[stage],
            key=lambda sample: hashlib.sha256(f"{REAL_DIAGNOSTIC_SPLIT_SEED}:{sample.sample_id}:{stage}".encode("utf-8")).hexdigest(),
        )
        for sample in bucket:
            candidates = [split for split in split_order if assigned[split] < targets[split]]
            if not candidates:
                candidates = split_order
            split = min(candidates, key=lambda name: (assigned[name] / max(targets[name], 1), assigned[name], split_order.index(name)))
            assigned[split] += 1
            quality = quality_by_id.get(sample.anonymous_review_id, {})
            concepts = sample_by_id.get(sample.anonymous_review_id, {})
            split_rows.append(
                {
                    "anonymous_review_id": sample.anonymous_review_id,
                    "sample_id": sample.sample_id,
                    "split": split,
                    "approved_stage": sample.approved_stage,
                    "comparable_stage_group": sample.comparable_stage_group,
                    "image_height": quality.get("image_height", ""),
                    "image_width": quality.get("image_width", ""),
                    "measurement_quality_label": quality.get("measurement_quality_label", ""),
                    "valid_spot_count": quality.get("spot_count", ""),
                    "valid_pair_count": quality.get("valid_pair_count", ""),
                    "adhesion_median": concepts.get("adhesion_median", ""),
                    "split_seed": REAL_DIAGNOSTIC_SPLIT_SEED,
                }
            )
    return sorted(split_rows, key=lambda row: (row["split"], row["anonymous_review_id"]))


def rows_for_split(split_rows: Sequence[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in split_rows if row["split"] == split]


def _norm01(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=float)
    lo = float(np.nanpercentile(values, 1))
    hi = float(np.nanpercentile(values, 99))
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _overlay_measurement(ax: Any, measurement: RealMeasurement, *, show_pairs: bool = True, show_corridors: bool = False) -> None:
    ax.imshow(_norm01(measurement.channels.linear), cmap="gray", origin="upper")
    amap = assignment_maps(measurement.lattice)
    for idx, spot in enumerate(measurement.spots):
        assignment = amap.get(idx)
        label = f"r{assignment.row_label}:k{assignment.lattice_index}" if assignment else "unassigned"
        color = "deepskyblue" if not spot.edge_or_crop_flag else "magenta"
        ax.scatter([spot.center_x], [spot.center_y], s=22, facecolors="none", edgecolors=color, linewidths=0.8)
        ax.text(spot.center_x + 1.5, spot.center_y - 1.5, label, color="yellow", fontsize=5)
    if show_pairs:
        for pair in measurement.pairs:
            left = measurement.spots[pair.spot_i]
            right = measurement.spots[pair.spot_j]
            ax.plot([left.center_x, right.center_x], [left.center_y, right.center_y], color="lime", linewidth=0.7, alpha=0.75)
    if show_corridors:
        for pair_id, masks in list(measurement.masks.items())[:8]:
            ax.contour(masks.corridor_mask, levels=[0.5], colors=["orange"], linewidths=0.45, alpha=0.65)
            ax.contour(masks.background_mask, levels=[0.5], colors=["cyan"], linewidths=0.35, alpha=0.55)
    ax.set_xticks([])
    ax.set_yticks([])


def write_sample_contact_sheet(measurement: RealMeasurement, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    axes = axes.ravel()
    panels = [
        ("Original", _norm01(measurement.image)),
        ("Linear grayscale", _norm01(measurement.channels.linear)),
        ("Smooth background", _norm01(measurement.channels.background)),
        ("Background corrected", _norm01(measurement.channels.log_corrected)),
    ]
    for ax, (title, image) in zip(axes[:4], panels):
        ax.imshow(image, cmap="gray", origin="upper")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    _overlay_measurement(axes[4], measurement, show_pairs=True, show_corridors=False)
    axes[4].set_title("Spots, rows, lattice, neutral pairs", fontsize=8)
    _overlay_measurement(axes[5], measurement, show_pairs=True, show_corridors=True)
    axes[5].set_title("Pair/background corridors", fontsize=8)
    fig.suptitle(measurement.sample.anonymous_review_id, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_pair_crop_panel(measurement: RealMeasurement, pair_ids: Sequence[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    chosen = [pair for pair in measurement.pairs if pair.pair_id in set(pair_ids)]
    if not chosen:
        return
    fig, axes = plt.subplots(len(chosen), 2, figsize=(8.5, max(3.0, len(chosen) * 2.0)))
    if len(chosen) == 1:
        axes = np.asarray(axes).reshape(1, 2)
    for row_idx, pair in enumerate(chosen):
        left = measurement.spots[pair.spot_i]
        right = measurement.spots[pair.spot_j]
        pad = 18
        x0 = max(0, int(min(left.center_x, right.center_x) - pad))
        x1 = min(measurement.image.shape[1], int(max(left.center_x, right.center_x) + pad))
        y0 = max(0, int(min(left.center_y, right.center_y) - pad))
        y1 = min(measurement.image.shape[0], int(max(left.center_y, right.center_y) + pad))
        for col, image in enumerate((measurement.channels.linear, measurement.channels.log_corrected)):
            ax = axes[row_idx, col]
            ax.imshow(_norm01(image[y0:y1, x0:x1]), cmap="gray", origin="upper")
            masks = measurement.masks[pair.pair_id]
            ax.contour(masks.corridor_mask[y0:y1, x0:x1], levels=[0.5], colors=["orange"], linewidths=0.8)
            ax.contour(masks.background_mask[y0:y1, x0:x1], levels=[0.5], colors=["cyan"], linewidths=0.6)
            ax.scatter([left.center_x - x0, right.center_x - x0], [left.center_y - y0, right.center_y - y0], s=28, facecolors="none", edgecolors="yellow")
            ax.set_title(("original crop" if col == 0 else "corrected crop") + f" {pair.pair_id}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(measurement.sample.anonymous_review_id, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def select_development_pairs(measurement: RealMeasurement, *, count: int = 6) -> list[str]:
    valid = [feature for feature in measurement.features.values() if feature.valid]
    if not valid:
        return []
    valid.sort(key=lambda feature: feature.raw_peak_saddle_adhesion)
    picks = [valid[0], valid[len(valid) // 2], valid[-1]]
    by_conf = sorted(valid, key=lambda feature: feature.pair_measurement_confidence)
    picks.extend(by_conf[:2])
    for feature in valid:
        if feature not in picks:
            picks.append(feature)
        if len({p.pair_id for p in picks}) >= count:
            break
    return list(dict.fromkeys(p.pair_id for p in picks))[:count]


def write_review_html(path: Path, title: str, entries: Sequence[tuple[str, str]], *, instructions: str) -> None:
    parts = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p>{html.escape(instructions)}</p>",
    ]
    for anon_id, rel_image in entries:
        parts.append(f"<section><h2>{html.escape(anon_id)}</h2>")
        parts.append(f"<img src='{html.escape(rel_image)}' style='max-width:100%; border:1px solid #ccc'>")
        parts.append("</section>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_algorithm_shadow_audit(path: Path, sample_rows: Sequence[dict[str, Any]], quality_rows: Sequence[dict[str, Any]], pair_rows: Sequence[dict[str, Any]]) -> None:
    sample_by_id = {row["anonymous_review_id"]: row for row in sample_rows}
    quality_by_id = {row["anonymous_review_id"]: row for row in quality_rows}
    parts = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Algorithm Shadow Audit</title>",
        "<h1>DO NOT OPEN BEFORE COMPLETING THE BLINDED REVIEW.</h1>",
        "<p>Algorithm-only shadow-mode diagnostics. No AFM or Rq data are included.</p>",
        "<table border='1'><tr><th>anonymous_review_id</th><th>adhesion_median</th><th>valid_pairs</th><th>quality</th><th>invalid reasons</th></tr>",
    ]
    reasons_by_id: dict[str, Counter[str]] = defaultdict(Counter)
    for row in pair_rows:
        if row.get("invalid_reason"):
            reasons_by_id[row["anonymous_review_id"]][row["invalid_reason"]] += 1
    for anon_id in sorted(sample_by_id):
        sample = sample_by_id[anon_id]
        quality = quality_by_id.get(anon_id, {})
        reasons = "; ".join(f"{k}:{v}" for k, v in reasons_by_id.get(anon_id, {}).items())
        parts.append(
            "<tr>"
            f"<td>{html.escape(anon_id)}</td>"
            f"<td>{html.escape(_fmt(sample.get('adhesion_median')))}</td>"
            f"<td>{html.escape(str(sample.get('valid_pair_count', '')))}</td>"
            f"<td>{html.escape(_fmt(quality.get('overall_measurement_quality')))}</td>"
            f"<td>{html.escape(reasons)}</td>"
            "</tr>"
        )
    parts.append("</table>")
    path.write_text("\n".join(parts), encoding="utf-8")


def template_rows_all_sample(measurements: Sequence[RealMeasurement]) -> list[dict[str, Any]]:
    fields = [
        "anonymous_review_id",
        "spot_detection",
        "row_grouping",
        "lattice_indexing",
        "missing_site_handling",
        "adjacent_pair_selection",
        "pair_corridors",
        "background_correction",
        "overall_measurement",
        "severe_saturation",
        "severe_background_artifact",
        "severe_crop",
        "unusable_pattern",
        "reviewer_notes",
    ]
    return [{field: (measurement.sample.anonymous_review_id if field == "anonymous_review_id" else "") for field in fields} for measurement in measurements]


def write_instructions(path: Path, report_dir: Path) -> None:
    examples = make_synthetic_split_v2("development_v2")[:3]
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    for ax, example in zip(axes, examples):
        ax.imshow(example.display_image, cmap="gray", origin="upper")
        ax.set_title("synthetic example", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    example_path = report_dir / "synthetic_instruction_examples.png"
    fig.savefig(example_path, dpi=150)
    plt.close(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Stage 2A Blinded Review Instructions",
                "",
                "Judge visible spot-to-spot grayscale adhesion, not expected roughness.",
                "Do not consult AFM images.",
                "Do not consult Rq values.",
                "Do not use prior memory of sample IDs.",
                "Do not infer labels from filename or growth stage.",
                "Judge a bridge relative to the local spot peaks and local background.",
                "Geometric alignment alone does not mean connected.",
                "A broad diffuse halo alone does not mean connected.",
                "Mark unusable rather than guessing when evidence is insufficient.",
                "Complete the review in the randomized order presented.",
                "",
                "Synthetic schematic examples are available in the report folder as `synthetic_instruction_examples.png`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_annotation_templates(paths: Any, measurements: Sequence[RealMeasurement], split_rows: Sequence[dict[str, Any]], selected_pair_rows: Sequence[dict[str, Any]]) -> None:
    review_dir = paths.annotations_dir / "real_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    write_csv(review_dir / "all_sample_qc_template.csv", template_rows_all_sample(measurements))
    dev_ids = {row["anonymous_review_id"] for row in split_rows if row["split"] == "development_review"}
    write_csv(
        review_dir / "development_sample_review_template.csv",
        [
            {
                "anonymous_review_id": measurement.sample.anonymous_review_id,
                "sample_adhesion_rating": "",
                "visible_spot_quality": "",
                "rating_confidence": "",
                "severe_artifact": "",
                "reviewer_notes": "",
            }
            for measurement in measurements
            if measurement.sample.anonymous_review_id in dev_ids
        ],
    )
    write_csv(
        review_dir / "development_pair_review_template.csv",
        [
            {
                "anonymous_review_id": row["anonymous_review_id"],
                "anonymous_pair_id": row["anonymous_pair_id"],
                "pair_label": "",
                "spot_centers_correct": "",
                "same_physical_row": "",
                "true_adjacent_neighbors": "",
                "corridor_correct": "",
                "background_regions_clean": "",
                "label_confidence": "",
                "unusable_reason": "",
                "reviewer_notes": "",
            }
            for row in selected_pair_rows
        ],
    )
    write_instructions(review_dir / "instructions.md", paths.reports_dir / "real_diagnostics")
    write_csv(
        review_dir / "unblind_key.csv",
        [
            {
                "warning": "DO NOT OPEN UNTIL BLINDED HUMAN REVIEW IS COMPLETE.",
                "anonymous_review_id": row["anonymous_review_id"],
                "sample_id": row["sample_id"],
                "manual_rheed_filename": next(m.sample.manual_rheed_filename for m in measurements if m.sample.anonymous_review_id == row["anonymous_review_id"]),
                "approved_stage": row["approved_stage"],
                "comparable_stage_group": row["comparable_stage_group"],
                "split": row["split"],
            }
            for row in split_rows
        ],
    )


def write_data_access_audit(path: Path, input_paths: Sequence[Path], gate_info: dict[str, Any]) -> None:
    forbidden_terms = ("afm", "rq", "roughness", "selected_afm_targets", "sample_level_analysis_table", "oof")
    rows = []
    for input_path in input_paths:
        lower = input_path.as_posix().lower()
        forbidden_hit = any(term in lower for term in forbidden_terms)
        rows.append(
            {
                "path": input_path.as_posix(),
                "sha256": file_sha256(input_path) if input_path.is_file() else "",
                "forbidden_afm_rq_path_hit": forbidden_hit,
            }
        )
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_files_opened": rows,
        "forbidden_terms": forbidden_terms,
        "afm_rq_source_opened": any(row["forbidden_afm_rq_path_hit"] for row in rows),
        "execution_guard": "Stage 2A code reads only approval/removelist/stage review/frozen synthetic metadata/manual RHEED screenshots.",
        "forbidden_loader_calls": [],
        "gate_info": gate_info,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_split_receipt(out: Path, split_rows: Sequence[dict[str, Any]], manifest_path: Path, gate_info: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "deterministic_seed": REAL_DIAGNOSTIC_SPLIT_SEED,
        "input_manifest_sha256": file_sha256(manifest_path),
        "removelist_sha256": gate_info["removelist_sha256"],
        "stage_review_sha256": gate_info["stage_review_sha256"],
        "split_membership": [
            {"anonymous_review_id": row["anonymous_review_id"], "split": row["split"]}
            for row in sorted(split_rows, key=lambda row: row["anonymous_review_id"])
        ],
    }
    path = out / "real_review_split_receipt.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": path.as_posix(), "sha256": file_sha256(path), **payload}


def write_checkpoint_2a(
    path: Path,
    *,
    gate_info: dict[str, Any],
    manifest_rows: Sequence[dict[str, Any]],
    split_rows: Sequence[dict[str, Any]],
    quality_rows: Sequence[dict[str, Any]],
    split_receipt: dict[str, Any],
    template_hashes: dict[str, str],
) -> None:
    included = [row for row in manifest_rows if row["input_status"] == "included"]
    skipped = [row for row in manifest_rows if row["input_status"] != "included"]
    split_counts = Counter(row["split"] for row in split_rows)
    stage_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for row in split_rows:
        stage_by_split[row["split"]][row["approved_stage"]] += 1
    text = "\n".join(
        [
            "# Checkpoint 2A: Real RHEED Shadow Diagnostics",
            "",
            "## Recovered State",
            "- Stage 1C-R status: `STAGE 1C PASS AFTER METRIC-LINEAGE CORRECTION`",
            f"- Frozen semantic spec hash: `{gate_info['frozen_semantic_spec_sha256']}`",
            f"- Evaluation receipt SHA256: `{gate_info['evaluation_receipt_sha256']}`",
            "",
            "## Gates",
            f"- Visual approval path: `{gate_info['approval_path']}`",
            f"- Visual approval SHA256: `{gate_info['approval_sha256']}`",
            f"- Visual approval mtime: `{gate_info['approval_mtime']}`",
            f"- Removelist path: `{gate_info['removelist_path']}`",
            f"- Removelist SHA256: `{gate_info['removelist_sha256']}`",
            "- Sample `6088` excluded: `1`",
            f"- Stage-review path: `{gate_info['stage_review_path']}`",
            f"- Stage-review SHA256: `{gate_info['stage_review_sha256']}`",
            "",
            "## Real Inputs",
            f"- Eligible real-image count: `{len(included)}`",
            f"- Skipped images: `{len(skipped)}`",
            f"- Skipped details: `{'; '.join(row['sample_id'] + ':' + row['skip_reason'] for row in skipped) if skipped else 'none'}`",
            "- No AFM/Rq source was opened: `1`",
            "",
            "## Split",
            f"- Split seed: `{REAL_DIAGNOSTIC_SPLIT_SEED}`",
            f"- Split receipt SHA256: `{split_receipt['sha256']}`",
            f"- Development/blind/reserve counts: `{dict(split_counts)}`",
            f"- Approved-stage distribution by split: `{ {split: dict(counter) for split, counter in stage_by_split.items()} }`",
            "",
            "## Algorithm Shadow-Mode Summary",
            f"- Spot count median: `{_fmt(_median(row['spot_count'] for row in quality_rows))}`",
            f"- Row count median: `{_fmt(_median(row['row_count'] for row in quality_rows))}`",
            f"- Valid-pair count median: `{_fmt(_median(row['valid_pair_count'] for row in quality_rows))}`",
            f"- Invalid-pair count median: `{_fmt(_median(row['invalid_pair_count'] for row in quality_rows))}`",
            f"- Measurement-quality labels: `{dict(Counter(row['measurement_quality_label'] for row in quality_rows))}`",
            "",
            "## Files Requiring Human Completion",
            "- `annotations/rheed_peak_saddle/real_review/all_sample_qc_template.csv`",
            "- `annotations/rheed_peak_saddle/real_review/development_sample_review_template.csv`",
            "- `annotations/rheed_peak_saddle/real_review/development_pair_review_template.csv`",
            f"- Template hashes: `{template_hashes}`",
            "",
            "## Human Instructions",
            "Judge visible spot-to-spot grayscale adhesion only. Do not consult AFM images, Rq values, filenames, stage, or prior sample identity memory. Mark unusable rather than guessing.",
            "",
            "## Status",
            "HUMAN ANNOTATION REQUIRED",
            "",
            "## Stop Confirmation",
            "- No annotation validation was run.",
            "- No real-image tuning was run.",
            "- No AFM data were accessed.",
            "- No Rq values were accessed.",
            "- No model training was run.",
            "- No Stage 2B/3 was run.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_real_diagnostics(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    paths, gate_info = validate_stage2a_gates(config)
    out = paths.outputs_dir / "real_diagnostics"
    report_dir = paths.reports_dir / "real_diagnostics"
    contact_dir = report_dir / "all_sample_qc_contact_sheets"
    pair_dir = report_dir / "development_pair_contact_sheets"
    out.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    contact_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    for generated_dir in (contact_dir, pair_dir):
        for old_file in generated_dir.glob("*.png"):
            old_file.unlink()

    samples, manifest_rows = build_real_input_manifest(paths)
    manifest_path = out / "real_input_manifest.csv"
    write_csv(manifest_path, manifest_rows)
    measurements = [measure_real_sample(sample) for sample in samples]
    tables = measurement_tables(measurements)
    for filename, rows in tables.items():
        write_csv(out / filename, rows)

    split_rows = choose_split(samples, tables["measurement_quality.csv"], tables["sample_level_concepts.csv"])
    write_csv(out / "split_manifest.csv", split_rows)
    for split, filename in (
        ("development_review", "development_review_manifest.csv"),
        ("blind_validation", "blind_validation_manifest.csv"),
        ("reserve", "reserve_manifest.csv"),
    ):
        write_csv(out / filename, rows_for_split(split_rows, split))
    split_receipt = write_split_receipt(out, split_rows, manifest_path, gate_info)

    entries = []
    for measurement in measurements:
        image_name = f"{measurement.sample.anonymous_review_id}_qc.png"
        write_sample_contact_sheet(measurement, contact_dir / image_name)
        entries.append((measurement.sample.anonymous_review_id, f"all_sample_qc_contact_sheets/{image_name}"))
    write_review_html(
        report_dir / "all_sample_qc_review.html",
        "All-Sample Blinded QC Review",
        entries,
        instructions="Review geometry and measurement regions. Identity, source-name, growth-context, external target data, and algorithm scores are hidden.",
    )

    split_by_anon = {row["anonymous_review_id"]: row["split"] for row in split_rows}
    dev_measurements = [m for m in measurements if split_by_anon.get(m.sample.anonymous_review_id) == "development_review"]
    write_review_html(
        report_dir / "development_sample_review.html",
        "Development Sample-Level Adhesion Review",
        [(m.sample.anonymous_review_id, f"all_sample_qc_contact_sheets/{m.sample.anonymous_review_id}_qc.png") for m in dev_measurements],
        instructions="Rate visible grayscale adhesion. Algorithmic values and sample identity are hidden.",
    )

    selected_pair_rows = []
    pair_entries = []
    for measurement in dev_measurements:
        chosen = select_development_pairs(measurement)
        if not chosen:
            continue
        panel_name = f"{measurement.sample.anonymous_review_id}_pairs.png"
        write_pair_crop_panel(measurement, chosen, pair_dir / panel_name)
        pair_entries.append((measurement.sample.anonymous_review_id, f"development_pair_contact_sheets/{panel_name}"))
        for pair_id in chosen:
            selected_pair_rows.append({"anonymous_review_id": measurement.sample.anonymous_review_id, "anonymous_pair_id": pair_id})
    write_csv(out / "development_pair_selection.csv", selected_pair_rows)
    write_review_html(
        report_dir / "development_pair_review.html",
        "Development Pair-Level Adhesion Review",
        pair_entries,
        instructions="Review selected pair crops. Identity, source-name, growth-context, external target data, and algorithm scores are hidden.",
    )

    write_annotation_templates(paths, measurements, split_rows, selected_pair_rows)
    write_algorithm_shadow_audit(
        report_dir / "algorithm_shadow_audit.html",
        tables["sample_level_concepts.csv"],
        tables["measurement_quality.csv"],
        tables["pair_level_measurements.csv"],
    )

    input_paths = [
        Path(gate_info["approval_path"]),
        Path(gate_info["removelist_path"]),
        Path(gate_info["stage_review_path"]),
        paths.outputs_dir / "synthetic_v3" / "evaluation_receipt.json",
        paths.outputs_dir / "synthetic_v3" / "frozen_semantic_spec.json",
        paths.outputs_dir / "synthetic_v3" / "frozen_semantic_spec.sha256",
        paths.reports_dir / "checkpoint_1c_metric_audit.md",
        *[sample.manual_rheed_path for sample in samples],
    ]
    write_data_access_audit(out / "data_access_audit.json", input_paths, gate_info)
    template_hashes = {
        name: file_sha256(paths.annotations_dir / "real_review" / name)
        for name in (
            "all_sample_qc_template.csv",
            "development_sample_review_template.csv",
            "development_pair_review_template.csv",
            "instructions.md",
            "unblind_key.csv",
        )
    }
    repro = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": STAGE2A_COMMAND,
        "config_path": config_path.as_posix(),
        "config_sha256": file_sha256(config_path),
        "real_input_manifest_sha256": file_sha256(manifest_path),
        "split_receipt_sha256": split_receipt["sha256"],
        "annotation_template_hashes": template_hashes,
        "gate_info": gate_info,
        "stage2a_stop": "HUMAN ANNOTATION REQUIRED",
    }
    (out / "reproducibility_manifest.json").write_text(json.dumps(repro, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_checkpoint_2a(
        paths.reports_dir / "checkpoint_2a_real_diagnostics.md",
        gate_info=gate_info,
        manifest_rows=manifest_rows,
        split_rows=split_rows,
        quality_rows=tables["measurement_quality.csv"],
        split_receipt=split_receipt,
        template_hashes=template_hashes,
    )
    return {
        "outputs_dir": out,
        "reports_dir": report_dir,
        "eligible_count": len(samples),
        "split_counts": dict(Counter(row["split"] for row in split_rows)),
        "status": "HUMAN ANNOTATION REQUIRED",
    }
