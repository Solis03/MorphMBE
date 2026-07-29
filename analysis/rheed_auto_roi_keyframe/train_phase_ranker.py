#!/usr/bin/env python3
"""Train and strictly leave-one-video-out evaluate phase-candidate rankers."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from scipy import ndimage, stats
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.metrics import structural_similarity
import joblib


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rheed2morph.rheed.automatic_roi_keyframe import (  # noqa: E402
    Rect,
    _select_keyframes_core,
    load_frame,
)


FEATURES = [
    "spot_x",
    "spot_y",
    "clarity",
    "sharpness",
    "spot_energy",
    "mean_intensity",
    "absolute_contrast",
    "prominence",
    "pre_dx",
    "post_dx",
    "upward_dy",
    "q_spot_x",
    "q_spot_y",
    "q_clarity",
    "q_sharpness",
    "q_spot_energy",
    "q_mean_intensity",
    "q_absolute_contrast",
    "q_prominence",
    "q_pre_dx",
    "q_post_dx",
    "q_upward_dy",
    "direction_consistent",
    "tracker_front",
    "cross_tracker_distance",
    "cross_tracker_agreement",
    "cross_tracker_direction_support",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_auto_roi_keyframe_v2.json",
    )
    parser.add_argument("--rebuild-candidates", action="store_true")
    return parser.parse_args()


def crop_for_compare(frame: np.ndarray, rect: Rect) -> np.ndarray:
    ys, xs = rect.as_slices()
    gray = np.asarray(frame)[ys, xs, :3].astype(np.float32).mean(axis=2)
    gray /= 255.0
    image = np.asarray(
        Image.fromarray(gray).resize(
            (128, 192), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    low, high = np.percentile(image, [3.0, 99.7])
    normalized = np.clip(
        (image - low) / max(float(high - low), 1e-5), 0.0, 1.0
    )
    response = normalized - ndimage.gaussian_filter(normalized, 12.0)
    scale = float(np.percentile(np.abs(response), 99.0))
    return np.clip(0.5 + response / max(2.0 * scale, 1e-5), 0.0, 1.0)


def pattern_metrics(
    first: np.ndarray, second: np.ndarray, rect: Rect
) -> tuple[float, float, float]:
    a = crop_for_compare(first, rect)
    b = crop_for_compare(second, rect)
    ncc = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    ssim = float(structural_similarity(a, b, data_range=1.0))
    ga = np.hypot(*np.gradient(a))
    gb = np.hypot(*np.gradient(b))
    gradient_ncc = float(np.corrcoef(ga.ravel(), gb.ravel())[0, 1])
    return ncc, ssim, gradient_ncc


def candidate_rows_for_tracker(
    trajectory: list[dict[str, Any]],
    *,
    tracker: str,
) -> list[dict[str, Any]]:
    if tracker == "compact":
        values = [
            {
                **row,
                "spot_x": row["compact_spot_x"],
                "spot_y": row["compact_spot_y"],
            }
            for row in trajectory
        ]
    else:
        values = trajectory
    _, candidates = _select_keyframes_core(values)
    for candidate in candidates:
        candidate["tracker"] = tracker
    return candidates


def build_candidate_dataset(
    manifest: pd.DataFrame,
    *,
    experiment_root: Path,
    output_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for count, (_, record) in enumerate(manifest.iterrows(), start=1):
        sample = str(record["sample_id"])
        print(
            f"[candidate labels {count:02d}/{len(manifest):02d}] "
            f"{sample}",
            flush=True,
        )
        selection = json.loads(
            (
                experiment_root / "selections" / f"{sample}.json"
            ).read_text(encoding="utf-8")
        )
        roi = Rect(
            **selection["roi_predictions"]["calibrated_safe"]["rect"]
        )
        trajectory = json.loads(
            (
                experiment_root / "trajectories" / f"{sample}.json"
            ).read_text(encoding="utf-8")
        )["automatic"]
        frames_dir = ROOT / str(record["frames_dir"])
        manual_index = int(record["keyframe_index"])
        manual_frame = load_frame(frames_dir, manual_index)
        candidates_by_tracker = {
            tracker: candidate_rows_for_tracker(
                trajectory, tracker=tracker
            )
            for tracker in ("front", "compact")
        }
        for tracker in ("front", "compact"):
            candidates = candidates_by_tracker[tracker]
            other_candidates = candidates_by_tracker[
                "compact" if tracker == "front" else "front"
            ]
            for candidate in candidates:
                frame_index = int(candidate["frame_index"])
                nearest_other = min(
                    other_candidates,
                    key=lambda item: abs(
                        int(item["frame_index"]) - frame_index
                    ),
                )
                cross_distance = abs(
                    int(nearest_other["frame_index"]) - frame_index
                )
                frame = load_frame(frames_dir, frame_index)
                ncc, ssim, gradient_ncc = pattern_metrics(
                    manual_frame, frame, roi
                )
                rows.append(
                    {
                        "sample_id": sample,
                        "tracker": tracker,
                        "frame_index": frame_index,
                        "manual_frame_index": manual_index,
                        "absolute_frame_error": abs(
                            frame_index - manual_index
                        ),
                        "pattern_ncc": ncc,
                        "pattern_ssim": ssim,
                        "gradient_ncc": gradient_ncc,
                        "target_similarity": (
                            0.60 * ncc
                            + 0.25 * ssim
                            + 0.15 * gradient_ncc
                        ),
                        "tracker_front": float(tracker == "front"),
                        "cross_tracker_distance": float(
                            min(cross_distance, 60)
                        ),
                        "cross_tracker_agreement": float(
                            math.exp(-cross_distance / 3.0)
                        ),
                        "cross_tracker_direction_support": float(
                            bool(
                                nearest_other["direction_consistent"]
                            )
                            and cross_distance <= 4
                        ),
                        **{
                            key: value
                            for key, value in candidate.items()
                            if key not in {"position", "tracker"}
                        },
                    }
                )
    dataset = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    return dataset


def make_models() -> dict[str, Any]:
    return {
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=10.0),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=350,
                max_depth=5,
                min_samples_leaf=5,
                max_features=0.8,
                random_state=17,
                n_jobs=-1,
            ),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesRegressor(
                n_estimators=350,
                max_depth=6,
                min_samples_leaf=5,
                max_features=0.8,
                random_state=17,
                n_jobs=-1,
            ),
        ),
        "gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingRegressor(
                loss="huber",
                n_estimators=120,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=7,
                random_state=17,
            ),
        ),
    }


def grouped_leave_one_out(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()
    dataset["target_rank_within_video"] = dataset.groupby(
        "sample_id"
    )["target_similarity"].rank(pct=True, method="average")
    dataset["top_quintile_within_video"] = (
        dataset["target_rank_within_video"] >= 0.80
    ).astype(int)
    predictions: list[dict[str, Any]] = []
    groups = list(dataset["sample_id"].drop_duplicates())
    for fold, held in enumerate(groups, start=1):
        train = dataset.loc[dataset["sample_id"] != held].copy()
        test = dataset.loc[dataset["sample_id"] == held].copy()
        regression_jobs = [
            (name, model, "target_similarity")
            for name, model in make_models().items()
        ]
        regression_jobs.extend(
            [
                (
                    "ridge_within_video_rank",
                    make_pipeline(
                        SimpleImputer(strategy="median"),
                        StandardScaler(),
                        Ridge(alpha=10.0),
                    ),
                    "target_rank_within_video",
                ),
                (
                    "gradient_boosting_within_video_rank",
                    make_pipeline(
                        SimpleImputer(strategy="median"),
                        GradientBoostingRegressor(
                            loss="huber",
                            n_estimators=140,
                            learning_rate=0.03,
                            max_depth=2,
                            min_samples_leaf=7,
                            random_state=17,
                        ),
                    ),
                    "target_rank_within_video",
                ),
            ]
        )
        for name, model, target_column in regression_jobs:
            model.fit(train[FEATURES], train[target_column])
            scores = model.predict(test[FEATURES])
            position = int(np.argmax(scores))
            selected = test.iloc[position]
            ordered = np.sort(scores)[::-1]
            margin = (
                float(ordered[0] - ordered[1])
                if len(ordered) > 1
                else float(ordered[0])
            )
            predictions.append(
                {
                    "fold": fold,
                    "sample_id": held,
                    "method": name,
                    "train_video_count": len(groups) - 1,
                    "held_video_overlap": 0,
                    "selected_frame_index": int(
                        selected["frame_index"]
                    ),
                    "manual_frame_index": int(
                        selected["manual_frame_index"]
                    ),
                    "tracker": selected["tracker"],
                    "predicted_similarity": float(scores[position]),
                    "prediction_margin": margin,
                    "pattern_ncc": float(selected["pattern_ncc"]),
                    "pattern_ssim": float(selected["pattern_ssim"]),
                    "gradient_ncc": float(selected["gradient_ncc"]),
                    "target_similarity": float(
                        selected["target_similarity"]
                    ),
                    "absolute_frame_error": int(
                        selected["absolute_frame_error"]
                    ),
                }
            )
        ridge_model = make_models()["ridge"]
        gradient_model = make_models()["gradient_boosting"]
        ridge_model.fit(train[FEATURES], train["target_similarity"])
        gradient_model.fit(train[FEATURES], train["target_similarity"])
        ridge_scores = ridge_model.predict(test[FEATURES])
        gradient_scores = gradient_model.predict(test[FEATURES])
        ridge_ranks = pd.Series(ridge_scores).rank(pct=True).to_numpy()
        gradient_ranks = (
            pd.Series(gradient_scores).rank(pct=True).to_numpy()
        )
        scores = 0.55 * ridge_ranks + 0.45 * gradient_ranks
        position = int(np.argmax(scores))
        selected = test.iloc[position]
        ordered = np.sort(scores)[::-1]
        predictions.append(
            {
                "fold": fold,
                "sample_id": held,
                "method": "ridge_gradient_rank_ensemble",
                "train_video_count": len(groups) - 1,
                "held_video_overlap": 0,
                "selected_frame_index": int(selected["frame_index"]),
                "manual_frame_index": int(
                    selected["manual_frame_index"]
                ),
                "tracker": selected["tracker"],
                "predicted_similarity": float(
                    0.55 * ridge_scores[position]
                    + 0.45 * gradient_scores[position]
                ),
                "prediction_margin": (
                    float(ordered[0] - ordered[1])
                    if len(ordered) > 1
                    else float(ordered[0])
                ),
                "pattern_ncc": float(selected["pattern_ncc"]),
                "pattern_ssim": float(selected["pattern_ssim"]),
                "gradient_ncc": float(selected["gradient_ncc"]),
                "target_similarity": float(
                    selected["target_similarity"]
                ),
                "absolute_frame_error": int(
                    selected["absolute_frame_error"]
                ),
            }
        )
        classifiers = {
            "logistic_top_quintile": make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(
                    C=0.3,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=17,
                ),
            ),
            "gradient_boosting_top_quintile": make_pipeline(
                SimpleImputer(strategy="median"),
                GradientBoostingClassifier(
                    n_estimators=120,
                    learning_rate=0.03,
                    max_depth=2,
                    min_samples_leaf=7,
                    random_state=17,
                ),
            ),
        }
        for name, model in classifiers.items():
            model.fit(train[FEATURES], train["top_quintile_within_video"])
            scores = model.predict_proba(test[FEATURES])[:, 1]
            position = int(np.argmax(scores))
            selected = test.iloc[position]
            ordered = np.sort(scores)[::-1]
            margin = (
                float(ordered[0] - ordered[1])
                if len(ordered) > 1
                else float(ordered[0])
            )
            predictions.append(
                {
                    "fold": fold,
                    "sample_id": held,
                    "method": name,
                    "train_video_count": len(groups) - 1,
                    "held_video_overlap": 0,
                    "selected_frame_index": int(
                        selected["frame_index"]
                    ),
                    "manual_frame_index": int(
                        selected["manual_frame_index"]
                    ),
                    "tracker": selected["tracker"],
                    "predicted_similarity": float(scores[position]),
                    "prediction_margin": margin,
                    "pattern_ncc": float(selected["pattern_ncc"]),
                    "pattern_ssim": float(selected["pattern_ssim"]),
                    "gradient_ncc": float(selected["gradient_ncc"]),
                    "target_similarity": float(
                        selected["target_similarity"]
                    ),
                    "absolute_frame_error": int(
                        selected["absolute_frame_error"]
                    ),
                }
            )
    return pd.DataFrame(predictions)


def visual_descriptor(frame: np.ndarray, rect: Rect) -> np.ndarray:
    image = crop_for_compare(frame, rect)
    small = np.asarray(
        Image.fromarray(image).resize(
            (32, 48), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    vector = small.ravel().astype(np.float64)
    vector -= float(vector.mean())
    vector /= max(float(vector.std()), 1e-6)
    vector /= max(float(np.linalg.norm(vector)), 1e-8)
    return vector


def template_leave_one_out(
    dataset: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    experiment_root: Path,
) -> pd.DataFrame:
    manifest_lookup = manifest.set_index("sample_id")
    manual_descriptors: dict[str, np.ndarray] = {}
    candidate_descriptors: dict[tuple[str, int], np.ndarray] = {}
    for sample in dataset["sample_id"].drop_duplicates():
        record = manifest_lookup.loc[sample]
        selection = json.loads(
            (
                experiment_root / "selections" / f"{sample}.json"
            ).read_text(encoding="utf-8")
        )
        roi = Rect(
            **selection["roi_predictions"]["calibrated_safe"]["rect"]
        )
        frames_dir = ROOT / str(record["frames_dir"])
        manual_index = int(record["keyframe_index"])
        manual_descriptors[sample] = visual_descriptor(
            load_frame(frames_dir, manual_index), roi
        )
        for frame_index in dataset.loc[
            dataset["sample_id"] == sample, "frame_index"
        ].unique():
            key = (sample, int(frame_index))
            candidate_descriptors[key] = visual_descriptor(
                load_frame(frames_dir, int(frame_index)), roi
            )

    results: list[dict[str, Any]] = []
    groups = list(dataset["sample_id"].drop_duplicates())
    for fold, held in enumerate(groups, start=1):
        train_manual = np.stack(
            [
                manual_descriptors[sample]
                for sample in groups
                if sample != held
            ]
        )
        test = dataset.loc[dataset["sample_id"] == held].copy()
        candidate_matrix = np.stack(
            [
                candidate_descriptors[(held, int(frame_index))]
                for frame_index in test["frame_index"]
            ]
        )
        mean_template = train_manual.mean(axis=0)
        mean_template -= mean_template.mean()
        mean_template /= max(np.linalg.norm(mean_template), 1e-8)
        mean_scores = candidate_matrix @ mean_template
        maximum_scores = np.max(
            candidate_matrix @ train_manual.T, axis=1
        )

        components = min(10, len(train_manual) - 2)
        pca = PCA(n_components=components, random_state=17)
        pca.fit(train_manual)
        reconstructed = pca.inverse_transform(
            pca.transform(candidate_matrix)
        )
        reconstruction_error = np.mean(
            (candidate_matrix - reconstructed) ** 2, axis=1
        )
        mean_rank = pd.Series(mean_scores).rank(pct=True).to_numpy()
        maximum_rank = (
            pd.Series(maximum_scores).rank(pct=True).to_numpy()
        )
        reconstruction_rank = (
            pd.Series(-reconstruction_error).rank(pct=True).to_numpy()
        )
        score_sets = {
            "manual_mean_template": mean_scores,
            "manual_max_template": maximum_scores,
            "manual_pca_template": (
                0.55 * mean_rank
                + 0.25 * maximum_rank
                + 0.20 * reconstruction_rank
            ),
        }
        for method, scores in score_sets.items():
            position = int(np.argmax(scores))
            selected = test.iloc[position]
            ordered = np.sort(scores)[::-1]
            margin = (
                float(ordered[0] - ordered[1])
                if len(ordered) > 1
                else float(ordered[0])
            )
            results.append(
                {
                    "fold": fold,
                    "sample_id": held,
                    "method": method,
                    "train_video_count": len(groups) - 1,
                    "held_video_overlap": 0,
                    "selected_frame_index": int(
                        selected["frame_index"]
                    ),
                    "manual_frame_index": int(
                        selected["manual_frame_index"]
                    ),
                    "tracker": selected["tracker"],
                    "predicted_similarity": float(scores[position]),
                    "prediction_margin": margin,
                    "pattern_ncc": float(selected["pattern_ncc"]),
                    "pattern_ssim": float(selected["pattern_ssim"]),
                    "gradient_ncc": float(selected["gradient_ncc"]),
                    "target_similarity": float(
                        selected["target_similarity"]
                    ),
                    "absolute_frame_error": int(
                        selected["absolute_frame_error"]
                    ),
                }
            )
    return pd.DataFrame(results)


def save_summary(
    predictions: pd.DataFrame, output_root: Path, report_root: Path
) -> pd.DataFrame:
    summary = (
        predictions.groupby("method", sort=False)
        .agg(
            n=("sample_id", "size"),
            median_pattern_ncc=("pattern_ncc", "median"),
            mean_pattern_ncc=("pattern_ncc", "mean"),
            median_pattern_ssim=("pattern_ssim", "median"),
            median_gradient_ncc=("gradient_ncc", "median"),
            median_target_similarity=("target_similarity", "median"),
            median_absolute_frame_error=(
                "absolute_frame_error", "median"
            ),
            front_tracker_rate=("tracker", lambda x: (x == "front").mean()),
            leakage_overlap=("held_video_overlap", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(output_root / "phase_ranker_summary.csv", index=False)

    order = list(summary.sort_values(
        "median_target_similarity", ascending=False
    )["method"])
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
    for axis, metric, title in (
        (axes[0], "pattern_ncc", "Human–machine pattern NCC"),
        (axes[1], "pattern_ssim", "Human–machine pattern SSIM"),
        (
            axes[2],
            "absolute_frame_error",
            "Absolute frame difference\n(other cycles are allowed)",
        ),
    ):
        values = [
            predictions.loc[predictions["method"] == method, metric]
            for method in order
        ]
        boxes = axis.boxplot(values, patch_artist=True, showfliers=True)
        for patch, color in zip(
            boxes["boxes"],
            [
                "#228833",
                "#4477aa",
                "#cc6677",
                "#aa3377",
                "#66ccee",
                "#ee7733",
                "#999933",
            ],
        ):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        axis.set_xticks(range(1, len(order) + 1))
        axis.set_xticklabels(
            [item.replace("_", "\n") for item in order], fontsize=8
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Strict leave-one-video-out supervised phase ranking",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    report_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        report_root / "phase_ranker_benchmark.png",
        dpi=240,
        bbox_inches="tight",
    )
    fig.savefig(
        report_root / "phase_ranker_benchmark.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    return summary


def fit_final_ridge_json(
    dataset: pd.DataFrame, output_path: Path
) -> None:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    features = imputer.fit_transform(dataset[FEATURES])
    scaled = scaler.fit_transform(features)
    model = Ridge(alpha=10.0)
    model.fit(scaled, dataset["target_similarity"])
    payload = {
        "schema_version": 1,
        "model_family": "ridge_candidate_similarity_ranker",
        "training_video_count": int(dataset["sample_id"].nunique()),
        "training_candidate_count": int(len(dataset)),
        "features": FEATURES,
        "imputer_median": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficient": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "ridge_alpha": 10.0,
        "target": (
            "0.60*pattern_ncc + 0.25*pattern_ssim + "
            "0.15*gradient_ncc"
        ),
        "provenance": (
            "Fit on all annotated videos only after strict leave-one-video-"
            "out comparison; intended for future prospective videos."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fit_final_gradient_bundle(
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    output_root: Path,
) -> None:
    model = make_models()["gradient_boosting"]
    model.fit(dataset[FEATURES], dataset["target_similarity"])
    loo = predictions.loc[
        predictions["method"] == "gradient_boosting"
    ].copy()
    calibrator = IsotonicRegression(
        y_min=0.0, y_max=1.0, out_of_bounds="clip"
    )
    calibrator.fit(
        loo["predicted_similarity"], loo["target_similarity"]
    )
    bundle_path = output_root / "gradient_boosting_phase_ranker.joblib"
    joblib.dump(
        {
            "schema_version": 1,
            "model": model,
            "calibrator": calibrator,
            "features": FEATURES,
            "training_video_count": int(
                dataset["sample_id"].nunique()
            ),
            "training_candidate_count": int(len(dataset)),
            "tracker_models": ["diffraction_front", "compact_bright_spot"],
        },
        bundle_path,
        compress=3,
    )
    rho, p_value = stats.spearmanr(
        loo["predicted_similarity"],
        1.0 - loo["target_similarity"],
    )
    metadata = {
        "schema_version": 1,
        "model_family": "gradient_boosting_candidate_similarity_ranker",
        "bundle_path": bundle_path.name,
        "features": FEATURES,
        "training_video_count": int(dataset["sample_id"].nunique()),
        "training_candidate_count": int(len(dataset)),
        "validation_protocol": "strict_leave_one_video_out",
        "validation_video_count": int(len(loo)),
        "held_video_overlap_sum": int(loo["held_video_overlap"].sum()),
        "median_pattern_ncc": float(loo["pattern_ncc"].median()),
        "median_pattern_ssim": float(loo["pattern_ssim"].median()),
        "median_absolute_frame_error": float(
            loo["absolute_frame_error"].median()
        ),
        "confidence_error_spearman_rho": float(rho),
        "confidence_error_spearman_p": float(p_value),
        "confidence_definition": (
            "LOO-isotonic expected composite human-frame similarity; "
            "not a probability of correctness."
        ),
        "prospective_note": (
            "The exported model is refit on all 27 annotated videos and "
            "must be prospectively validated on future unseen videos."
        ),
    }
    (output_root / "gradient_boosting_phase_ranker.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fit_final_ridge_bundle(
    dataset: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    output_root: Path,
) -> None:
    model = make_models()["ridge"]
    model.fit(dataset[FEATURES], dataset["target_similarity"])
    loo = predictions.loc[predictions["method"] == "ridge"].copy()
    calibrator = IsotonicRegression(
        y_min=0.0, y_max=1.0, out_of_bounds="clip"
    )
    calibrator.fit(
        loo["predicted_similarity"], loo["target_similarity"]
    )
    bundle_path = output_root / "ridge_phase_ranker.joblib"
    joblib.dump(
        {
            "schema_version": 1,
            "model": model,
            "calibrator": calibrator,
            "features": FEATURES,
            "training_video_count": int(
                dataset["sample_id"].nunique()
            ),
            "training_candidate_count": int(len(dataset)),
            "tracker_models": ["diffraction_front", "compact_bright_spot"],
        },
        bundle_path,
        compress=3,
    )
    rho, p_value = stats.spearmanr(
        loo["predicted_similarity"],
        1.0 - loo["target_similarity"],
    )
    metadata = {
        "schema_version": 1,
        "model_family": "ridge_candidate_similarity_ranker",
        "bundle_path": bundle_path.name,
        "features": FEATURES,
        "training_video_count": int(dataset["sample_id"].nunique()),
        "training_candidate_count": int(len(dataset)),
        "validation_protocol": "strict_leave_one_video_out",
        "validation_video_count": int(len(loo)),
        "held_video_overlap_sum": int(loo["held_video_overlap"].sum()),
        "median_pattern_ncc": float(loo["pattern_ncc"].median()),
        "mean_pattern_ncc": float(loo["pattern_ncc"].mean()),
        "median_pattern_ssim": float(loo["pattern_ssim"].median()),
        "median_absolute_frame_error": float(
            loo["absolute_frame_error"].median()
        ),
        "confidence_error_spearman_rho": float(rho),
        "confidence_error_spearman_p": float(p_value),
        "confidence_definition": (
            "LOO-isotonic expected composite human-frame similarity; "
            "not a probability of correctness."
        ),
        "prospective_note": (
            "The exported model is refit on all retained annotated videos "
            "and must be prospectively validated on future unseen videos."
        ),
    }
    (output_root / "ridge_phase_ranker_bundle.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def crop_display(frame: np.ndarray, rect: Rect) -> Image.Image:
    image = Image.fromarray(frame).convert("RGB")
    crop = image.crop((rect.x, rect.y, rect.x2, rect.y2))
    crop.thumbnail((360, 420), Image.Resampling.LANCZOS)
    return ImageEnhance.Contrast(crop).enhance(1.8)


def save_selected_ranker_atlases(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    experiment_root: Path,
    report_root: Path,
) -> None:
    manifest_lookup = manifest.set_index("sample_id")
    for method in ("gradient_boosting", "ridge"):
        selected = predictions.loc[
            predictions["method"] == method
        ].sort_values("fold")
        pages = math.ceil(len(selected) / 5)
        method_root = report_root / method
        method_root.mkdir(parents=True, exist_ok=True)
        for page in range(pages):
            subset = selected.iloc[page * 5 : (page + 1) * 5]
            fig, axes = plt.subplots(
                len(subset), 2, figsize=(7.7, 3.0 * len(subset))
            )
            axes = np.atleast_2d(axes)
            for axis_row, (_, prediction) in zip(
                axes, subset.iterrows()
            ):
                sample = str(prediction["sample_id"])
                record = manifest_lookup.loc[sample]
                selection = json.loads(
                    (
                        experiment_root
                        / "selections"
                        / f"{sample}.json"
                    ).read_text(encoding="utf-8")
                )
                auto_roi = Rect(
                    **selection["roi_predictions"][
                        "calibrated_safe"
                    ]["rect"]
                )
                manual_roi = Rect(
                    int(record["roi_x"]),
                    int(record["roi_y"]),
                    int(record["roi_width"]),
                    int(record["roi_height"]),
                    int(record["source_width"]),
                    int(record["source_height"]),
                )
                frames_dir = ROOT / str(record["frames_dir"])
                human = load_frame(
                    frames_dir, int(record["keyframe_index"])
                )
                machine = load_frame(
                    frames_dir,
                    int(prediction["selected_frame_index"]),
                )
                axis_row[0].imshow(crop_display(human, manual_roi))
                axis_row[1].imshow(crop_display(machine, auto_roi))
                for axis in axis_row:
                    axis.axis("off")
                axis_row[0].set_title(
                    f"{sample} | human f{int(record['keyframe_index'])}",
                    fontsize=9,
                )
                axis_row[1].set_title(
                    f"machine f{int(prediction['selected_frame_index'])} "
                    f"| NCC {prediction['pattern_ncc']:.2f} "
                    f"| score {prediction['predicted_similarity']:.2f}",
                    fontsize=9,
                )
            fig.suptitle(
                f"Strict leave-one-video-out: {method} "
                f"(all samples, page {page + 1}/{pages})",
                fontsize=12,
                fontweight="bold",
            )
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            fig.savefig(
                method_root / f"all_samples_page_{page + 1:02d}.png",
                dpi=210,
                bbox_inches="tight",
            )
            fig.savefig(
                method_root / f"all_samples_page_{page + 1:02d}.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)


def save_confidence_figure(
    predictions: pd.DataFrame,
    report_root: Path,
    *,
    selected_method: str,
) -> None:
    selected = predictions.loc[
        predictions["method"] == selected_method
    ].copy()
    rho, p_value = stats.spearmanr(
        selected["predicted_similarity"],
        1.0 - selected["target_similarity"],
    )
    order = np.argsort(selected["predicted_similarity"].to_numpy())
    selected = selected.iloc[order]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    scatter = axes[0].scatter(
        selected["predicted_similarity"],
        selected["target_similarity"],
        c=selected["pattern_ncc"],
        cmap="viridis",
        s=55,
        edgecolor="black",
        linewidth=0.4,
    )
    for _, row in selected.iterrows():
        if (
            row["target_similarity"]
            < selected["target_similarity"].quantile(0.20)
        ):
            axes[0].annotate(
                row["sample_id"],
                (row["predicted_similarity"], row["target_similarity"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
    axes[0].set_xlabel("Model confidence score (predicted similarity)")
    axes[0].set_ylabel("Realized human-frame composite similarity")
    axes[0].set_title(
        f"Confidence audit | error ρ={rho:.2f}, p={p_value:.4f}"
    )
    axes[0].grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes[0], label="Pattern NCC")

    axes[1].bar(
        np.arange(len(selected)),
        selected["target_similarity"],
        color=plt.cm.viridis(
            np.clip(selected["predicted_similarity"], 0.0, 1.0)
        ),
    )
    axes[1].set_xticks(np.arange(len(selected)))
    axes[1].set_xticklabels(
        selected["sample_id"], rotation=90, fontsize=7
    )
    axes[1].set_ylabel("Realized composite similarity")
    axes[1].set_xlabel("Videos ordered by predicted confidence")
    axes[1].set_title("All held videos; no omitted failures")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(
        f"Supervised keyframe ranker confidence validation: "
        f"{selected_method}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(
        report_root / "confidence_validation.png",
        dpi=240,
        bbox_inches="tight",
    )
    fig.savefig(
        report_root / "confidence_validation.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    experiment_id = config["experiment_id"]
    source_experiment_id = config.get(
        "source_experiment_id", experiment_id
    )
    source_experiment_root = (
        ROOT
        / "outputs"
        / "rheed_auto_roi_keyframe"
        / source_experiment_id
    )
    output_root = (
        ROOT
        / "outputs"
        / "rheed_auto_roi_keyframe"
        / experiment_id
        / "supervised_phase_ranker"
    )
    report_root = (
        ROOT
        / "reports"
        / "rheed_auto_roi_keyframe"
        / experiment_id
        / "supervised_phase_ranker"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(
        ROOT / config["manifest"], dtype={"sample_id": str}
    ).drop_duplicates(subset=["sample_id", "source_video"])
    if bool(config.get("exclude_removelist", False)):
        if "excluded_by_removelist" not in manifest.columns:
            raise KeyError(
                "Manifest lacks excluded_by_removelist required by config"
            )
        manifest = manifest.loc[
            ~manifest["excluded_by_removelist"].astype(bool)
        ].copy()
    candidate_path = output_root / "phase_candidates.csv"
    if args.rebuild_candidates or not candidate_path.exists():
        dataset = build_candidate_dataset(
            manifest,
            experiment_root=source_experiment_root,
            output_path=candidate_path,
        )
    else:
        dataset = pd.read_csv(candidate_path, dtype={"sample_id": str})
    predictions = pd.concat(
        [
            grouped_leave_one_out(dataset),
            template_leave_one_out(
                dataset,
                manifest,
                experiment_root=source_experiment_root,
            ),
        ],
        ignore_index=True,
    )
    predictions.to_csv(
        output_root / "phase_ranker_loo_predictions.csv", index=False
    )
    summary = save_summary(predictions, output_root, report_root)
    # Ridge is the only exported ranker: it is compact, auditable and can be
    # evaluated without loading a pickle.  The final selection decision is
    # revisited after inspecting the complete LOO summary.
    fit_final_ridge_json(
        dataset, output_root / "ridge_phase_ranker.json"
    )
    fit_final_gradient_bundle(
        dataset, predictions, output_root=output_root
    )
    fit_final_ridge_bundle(
        dataset, predictions, output_root=output_root
    )
    save_selected_ranker_atlases(
        predictions,
        manifest,
        experiment_root=source_experiment_root,
        report_root=report_root,
    )
    save_confidence_figure(
        predictions,
        report_root,
        selected_method=config.get(
            "selected_ranker_method", "gradient_boosting"
        ),
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
