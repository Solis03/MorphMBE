#!/usr/bin/env python3
"""Evaluate image-aware RHEED phase-candidate rankers.

This experiment keeps the V4 physical candidate generator fixed and asks a
more specific question: among physically plausible rotation vertices, which
frame contains a compact, high-contrast diffraction-spot family rather than
diffuse phosphor-screen haze?

Every supervised result uses strict leave-one-video-out evaluation.  Frozen
DINOv2 features are extracted without labels; all PCA, scaling, regression
and pairwise-ranking heads are fitted inside each training fold.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rheed2morph.rheed.automatic_roi_keyframe import (  # noqa: E402
    Rect,
    load_frame,
)
from rheed2morph.rheed.spot_visibility import (  # noqa: E402
    DinoV2Embedder,
    SPOT_VISIBILITY_FEATURES,
    analyze_spot_visibility,
    prepare_dinov2_image,
    visibility_proxy,
)


V4_FEATURES = [
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

VISIBILITY_RANK_FEATURES = [
    "qv_spot_peak_top8_mass",
    "qv_spot_peak_snr_top4",
    "qv_high_to_low_frequency_ratio",
    "qv_spot_energy_concentration",
    "qv_spot_vertical_span",
    "qv_spot_column_alignment",
    "qv_normalized_laplacian_variance",
    "qv_visibility_proxy",
    "qv_haze_rejection",
]

METHODS_FOR_ATLASES = (
    "v4_ridge_reproduced",
    "v5_visual_ridge",
    "v5_visual_extra_trees",
    "v5_dinov2_ridge_a10",
    "v5_dinov2_pairwise",
    "v5_dinov2_top_quintile",
    "v5_dinov2_tree_hybrid_gate25",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_auto_roi_keyframe_v5.json",
    )
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument(
        "--device",
        choices=("mps", "cpu", "cuda"),
        default=None,
    )
    return parser.parse_args()


def read_config(path: str | Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def load_manifest(config: dict[str, Any]) -> pd.DataFrame:
    manifest = pd.read_csv(
        ROOT / config["manifest"], dtype={"sample_id": str}
    ).drop_duplicates(subset=["sample_id", "source_video"])
    excluded = manifest["excluded_by_removelist"]
    if excluded.dtype != bool:
        excluded = (
            excluded.astype(str).str.strip().str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )
        )
        if excluded.isna().any():
            raise ValueError("Could not parse excluded_by_removelist")
    retained = manifest.loc[~excluded].copy()
    expected = int(config["expected_video_count"])
    if len(retained) != expected:
        raise ValueError(
            f"Expected {expected} retained videos, found {len(retained)}"
        )
    return retained.sort_values("sample_id").reset_index(drop=True)


def load_roi(sample: str, config: dict[str, Any]) -> Rect:
    selection_path = (
        ROOT
        / config["trajectory_experiment_root"]
        / "selections"
        / f"{sample}.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    return Rect(
        **selection["roi_predictions"][config["roi_method"]]["rect"]
    )


def feature_keys(
    candidates: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    candidate_keys = candidates[["sample_id", "frame_index"]].copy()
    manual_keys = manifest[["sample_id", "keyframe_index"]].rename(
        columns={"keyframe_index": "frame_index"}
    )
    keys = (
        pd.concat([candidate_keys, manual_keys], ignore_index=True)
        .drop_duplicates()
        .sort_values(["sample_id", "frame_index"])
        .reset_index(drop=True)
    )
    keys["feature_row"] = np.arange(len(keys), dtype=int)
    return keys


def build_image_features(
    keys: pd.DataFrame,
    manifest: pd.DataFrame,
    config: dict[str, Any],
    *,
    output_root: Path,
    device: str | None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    records_by_sample = {
        str(row.sample_id): row
        for row in manifest.itertuples(index=False)
    }
    analyses = []
    dino_images: list[Image.Image] = []
    started = time.perf_counter()
    for count, row in enumerate(keys.itertuples(index=False), start=1):
        sample = str(row.sample_id)
        record = records_by_sample[sample]
        roi = load_roi(sample, config)
        frame = load_frame(
            ROOT / str(record.frames_dir), int(row.frame_index)
        )
        analysis = analyze_spot_visibility(frame, roi)
        analyses.append(analysis)
        dino_images.append(prepare_dinov2_image(frame, roi))
        if count % 100 == 0 or count == len(keys):
            print(
                f"[image descriptors {count:04d}/{len(keys):04d}]",
                flush=True,
            )

    descriptor_rows = []
    for row, analysis in zip(keys.itertuples(index=False), analyses):
        descriptor_rows.append(
            {
                "sample_id": str(row.sample_id),
                "frame_index": int(row.frame_index),
                "feature_row": int(row.feature_row),
                **analysis.features,
                "visibility_proxy": visibility_proxy(analysis.features),
            }
        )
    descriptors = pd.DataFrame(descriptor_rows)

    cache_dir = ROOT / config["foundation_model_cache"]
    embedder = DinoV2Embedder(
        config["foundation_model_id"],
        revision=config.get("foundation_model_revision"),
        device=device,
        cache_dir=cache_dir,
    )
    embeddings = embedder.encode(
        dino_images, batch_size=int(config["embedding_batch_size"])
    )
    elapsed = time.perf_counter() - started
    metadata = {
        "foundation_model_id": embedder.model_id,
        "foundation_model_requested_revision": embedder.revision,
        "foundation_model_resolved_revision": embedder.commit_hash,
        "embedding_width": int(embeddings.shape[1]),
        "image_count": int(len(keys)),
        "device": str(embedder.device),
        "elapsed_seconds": elapsed,
        "images_per_second": len(keys) / max(elapsed, 1e-6),
        "input_representation": (
            "ROI max-channel brightness, 2-99.8 percentile normalization, "
            "aspect-preserving 224x224 padding"
        ),
        "embedding_representation": (
            "concatenated DINOv2 CLS, mean patch and patch standard "
            "deviation vectors"
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    descriptors.to_csv(output_root / "image_descriptors.csv", index=False)
    np.savez_compressed(
        output_root / "dinov2_embeddings.npz",
        embeddings=embeddings,
    )
    (output_root / "feature_extraction_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return descriptors, embeddings, metadata


def load_or_build_features(
    candidates: pd.DataFrame,
    manifest: pd.DataFrame,
    config: dict[str, Any],
    *,
    output_root: Path,
    rebuild: bool,
    device: str | None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    keys = feature_keys(candidates, manifest)
    descriptor_path = output_root / "image_descriptors.csv"
    embedding_path = output_root / "dinov2_embeddings.npz"
    metadata_path = output_root / "feature_extraction_metadata.json"
    if rebuild or not (
        descriptor_path.exists()
        and embedding_path.exists()
        and metadata_path.exists()
    ):
        return build_image_features(
            keys,
            manifest,
            config,
            output_root=output_root,
            device=device,
        )
    descriptors = pd.read_csv(
        descriptor_path, dtype={"sample_id": str}
    )
    embeddings = np.load(embedding_path)["embeddings"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = keys[["sample_id", "frame_index", "feature_row"]]
    actual = descriptors[["sample_id", "frame_index", "feature_row"]]
    if not expected.equals(actual):
        raise ValueError("Cached image feature index does not match candidates")
    if len(descriptors) != len(embeddings):
        raise ValueError("Descriptor and DINOv2 cache lengths differ")
    return descriptors, embeddings, metadata


def add_visibility_ranks(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    for name in (
        "spot_peak_top8_mass",
        "spot_peak_snr_top4",
        "high_to_low_frequency_ratio",
        "spot_energy_concentration",
        "spot_vertical_span",
        "spot_column_alignment",
        "normalized_laplacian_variance",
        "visibility_proxy",
    ):
        table[f"qv_{name}"] = table.groupby("sample_id")[name].rank(
            pct=True, method="average"
        )
    table["qv_haze_rejection"] = 1.0 - table.groupby("sample_id")[
        "haze_dominance"
    ].rank(pct=True, method="average")
    table["spot_visibility_rank"] = table[VISIBILITY_RANK_FEATURES].mean(
        axis=1
    )
    return table


def assemble_candidate_features(
    candidates: pd.DataFrame,
    descriptors: pd.DataFrame,
    embeddings: np.ndarray,
) -> tuple[pd.DataFrame, list[str]]:
    table = candidates.merge(
        descriptors,
        on=["sample_id", "frame_index"],
        validate="many_to_one",
    )
    dino_names = [
        f"dinov2_{index:04d}" for index in range(embeddings.shape[1])
    ]
    embedding_frame = pd.DataFrame(embeddings, columns=dino_names)
    embedding_frame["feature_row"] = np.arange(len(embeddings), dtype=int)
    table = table.merge(
        embedding_frame, on="feature_row", validate="many_to_one"
    )
    table = add_visibility_ranks(table)
    return table, dino_names


def fit_transform(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    pca_components: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = imputer.fit_transform(train[list(feature_names)])
    test_values = imputer.transform(test[list(feature_names)])
    train_values = scaler.fit_transform(train_values)
    test_values = scaler.transform(test_values)
    pca = None
    if pca_components is not None:
        component_count = min(
            pca_components,
            train_values.shape[0] - 1,
            train_values.shape[1],
        )
        pca = PCA(
            n_components=component_count,
            svd_solver="randomized",
            random_state=17,
        )
        train_values = pca.fit_transform(train_values)
        test_values = pca.transform(test_values)
    return train_values, test_values, {
        "imputer": imputer,
        "scaler": scaler,
        "pca": pca,
        "feature_names": list(feature_names),
    }


def pairwise_rank_scores(
    train_values: np.ndarray,
    test_values: np.ndarray,
    train: pd.DataFrame,
    *,
    minimum_difference: float = 0.08,
) -> tuple[np.ndarray, LogisticRegression]:
    differences: list[np.ndarray] = []
    labels: list[int] = []
    for sample in sorted(train["sample_id"].unique()):
        positions = np.flatnonzero(
            train["sample_id"].to_numpy() == sample
        )
        targets = train.iloc[positions]["target_similarity"].to_numpy()
        for left in range(len(positions)):
            for right in range(left + 1, len(positions)):
                delta = float(targets[left] - targets[right])
                if abs(delta) < minimum_difference:
                    continue
                difference = (
                    train_values[positions[left]]
                    - train_values[positions[right]]
                )
                positive = int(delta > 0.0)
                differences.extend((difference, -difference))
                labels.extend((positive, 1 - positive))
    matrix = np.asarray(differences, dtype=np.float32)
    target = np.asarray(labels, dtype=int)
    model = LogisticRegression(
        C=0.2,
        fit_intercept=False,
        max_iter=3000,
        random_state=17,
    )
    model.fit(matrix, target)
    return model.decision_function(test_values), model


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(pct=True, method="average").to_numpy()


def selected_record(
    test: pd.DataFrame,
    scores: np.ndarray,
    *,
    fold: int,
    held: str,
    method: str,
    train_count: int,
    gate: float | None = None,
) -> dict[str, Any]:
    effective = np.asarray(scores, dtype=float).copy()
    eligible_count = len(effective)
    if gate is not None:
        eligible = test["spot_visibility_rank"].to_numpy() >= gate
        eligible_count = int(eligible.sum())
        if eligible_count:
            effective[~eligible] = -np.inf
    position = int(np.argmax(effective))
    row = test.iloc[position]
    ordered = np.sort(effective[np.isfinite(effective)])[::-1]
    return {
        "fold": fold,
        "sample_id": held,
        "method": method,
        "train_video_count": train_count,
        "held_video_overlap": 0,
        "selected_frame_index": int(row["frame_index"]),
        "manual_frame_index": int(row["manual_frame_index"]),
        "tracker": str(row["tracker"]),
        "predicted_similarity": float(scores[position]),
        "prediction_margin": (
            float(ordered[0] - ordered[1])
            if len(ordered) > 1
            else float(ordered[0])
        ),
        "visibility_gate": gate,
        "eligible_candidate_count": eligible_count,
        "spot_visibility_rank": float(row["spot_visibility_rank"]),
        "visibility_proxy": float(row["visibility_proxy"]),
        "spot_peak_top8_mass": float(row["spot_peak_top8_mass"]),
        "haze_dominance": float(row["haze_dominance"]),
        "pattern_ncc": float(row["pattern_ncc"]),
        "pattern_ssim": float(row["pattern_ssim"]),
        "gradient_ncc": float(row["gradient_ncc"]),
        "target_similarity": float(row["target_similarity"]),
        "absolute_frame_error": int(row["absolute_frame_error"]),
    }


def grouped_leave_one_out(
    table: pd.DataFrame,
    *,
    dino_names: list[str],
) -> pd.DataFrame:
    predictions: list[dict[str, Any]] = []
    groups = sorted(table["sample_id"].unique())
    visual_names = (
        V4_FEATURES
        + list(SPOT_VISIBILITY_FEATURES)
        + ["visibility_proxy"]
        + VISIBILITY_RANK_FEATURES
        + ["spot_visibility_rank"]
    )
    deep_names = visual_names + dino_names

    for fold, held in enumerate(groups):
        print(
            f"[strict LOO {fold + 1:02d}/{len(groups):02d}] {held}",
            flush=True,
        )
        train = table.loc[table["sample_id"] != held].reset_index(drop=True)
        test = table.loc[table["sample_id"] == held].reset_index(drop=True)
        train_count = int(train["sample_id"].nunique())
        if held in set(train["sample_id"]):
            raise AssertionError("Held video leaked into training fold")

        base_train, base_test, _ = fit_transform(
            train, test, V4_FEATURES
        )
        base_model = Ridge(alpha=10.0)
        base_model.fit(base_train, train["target_similarity"])
        base_scores = base_model.predict(base_test)
        predictions.append(
            selected_record(
                test,
                base_scores,
                fold=fold,
                held=held,
                method="v4_ridge_reproduced",
                train_count=train_count,
            )
        )
        predictions.append(
            selected_record(
                test,
                base_scores,
                fold=fold,
                held=held,
                method="v4_ridge_spot_gate40",
                train_count=train_count,
                gate=0.40,
            )
        )

        visual_train, visual_test, _ = fit_transform(
            train, test, visual_names
        )
        visual_ridge = Ridge(alpha=10.0)
        visual_ridge.fit(visual_train, train["target_similarity"])
        visual_scores = visual_ridge.predict(visual_test)
        predictions.append(
            selected_record(
                test,
                visual_scores,
                fold=fold,
                held=held,
                method="v5_visual_ridge",
                train_count=train_count,
            )
        )
        visual_trees = ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=3,
            max_features=0.75,
            random_state=17,
            n_jobs=-1,
        )
        visual_trees.fit(
            train[visual_names].replace([np.inf, -np.inf], np.nan).fillna(
                train[visual_names].median()
            ),
            train["target_similarity"],
        )
        tree_scores = visual_trees.predict(
            test[visual_names].replace([np.inf, -np.inf], np.nan).fillna(
                train[visual_names].median()
            )
        )
        predictions.append(
            selected_record(
                test,
                tree_scores,
                fold=fold,
                held=held,
                method="v5_visual_extra_trees",
                train_count=train_count,
            )
        )

        deep_train, deep_test, _ = fit_transform(
            train, test, deep_names, pca_components=64
        )
        deep_scores: dict[str, np.ndarray] = {}
        for alpha in (1.0, 10.0, 100.0):
            model = Ridge(alpha=alpha)
            model.fit(deep_train, train["target_similarity"])
            scores = model.predict(deep_test)
            name = f"v5_dinov2_ridge_a{int(alpha)}"
            deep_scores[name] = scores
            predictions.append(
                selected_record(
                    test,
                    scores,
                    fold=fold,
                    held=held,
                    method=name,
                    train_count=train_count,
                )
            )

        pair_scores, _ = pairwise_rank_scores(
            deep_train, deep_test, train
        )
        predictions.append(
            selected_record(
                test,
                pair_scores,
                fold=fold,
                held=held,
                method="v5_dinov2_pairwise",
                train_count=train_count,
            )
        )
        within_video_rank = train.groupby("sample_id")[
            "target_similarity"
        ].rank(pct=True, method="average")
        top_quintile = (within_video_rank >= 0.80).astype(int)
        top_classifier = ExtraTreesClassifier(
            n_estimators=600,
            min_samples_leaf=3,
            max_features=0.75,
            class_weight="balanced",
            random_state=17,
            n_jobs=-1,
        )
        top_classifier.fit(deep_train, top_quintile)
        top_scores = top_classifier.predict_proba(deep_test)[:, 1]
        predictions.append(
            selected_record(
                test,
                top_scores,
                fold=fold,
                held=held,
                method="v5_dinov2_top_quintile",
                train_count=train_count,
            )
        )
        hybrid_scores = (
            0.55 * percentile_rank(deep_scores["v5_dinov2_ridge_a10"])
            + 0.30 * percentile_rank(pair_scores)
            + 0.15 * test["spot_visibility_rank"].to_numpy()
        )
        for gate in (None, 0.25, 0.30, 0.35, 0.40, 0.50):
            suffix = "none" if gate is None else str(int(100 * gate))
            predictions.append(
                selected_record(
                    test,
                    hybrid_scores,
                    fold=fold,
                    held=held,
                    method=f"v5_dinov2_hybrid_gate{suffix}",
                    train_count=train_count,
                    gate=gate,
                )
            )
        tree_hybrid_scores = (
            0.38 * percentile_rank(deep_scores["v5_dinov2_ridge_a10"])
            + 0.27 * percentile_rank(pair_scores)
            + 0.25 * percentile_rank(tree_scores)
            + 0.10 * test["spot_visibility_rank"].to_numpy()
        )
        for gate in (None, 0.25, 0.30, 0.35, 0.40):
            suffix = "none" if gate is None else str(int(100 * gate))
            predictions.append(
                selected_record(
                    test,
                    tree_hybrid_scores,
                    fold=fold,
                    held=held,
                    method=f"v5_dinov2_tree_hybrid_gate{suffix}",
                    train_count=train_count,
                    gate=gate,
                )
            )
    return pd.DataFrame(predictions)


def add_manual_visibility_metrics(
    predictions: pd.DataFrame,
    descriptors: pd.DataFrame,
) -> pd.DataFrame:
    lookup = descriptors.set_index(["sample_id", "frame_index"])
    rows = []
    for prediction in predictions.to_dict(orient="records"):
        sample = str(prediction["sample_id"])
        manual = lookup.loc[
            (sample, int(prediction["manual_frame_index"]))
        ]
        selected = lookup.loc[
            (sample, int(prediction["selected_frame_index"]))
        ]
        manual_mass = float(manual["spot_peak_top8_mass"])
        selected_mass = float(selected["spot_peak_top8_mass"])
        manual_proxy = float(manual["visibility_proxy"])
        selected_proxy = float(selected["visibility_proxy"])
        prediction["manual_visibility_proxy"] = manual_proxy
        prediction["selected_visibility_proxy"] = selected_proxy
        prediction["spot_peak_mass_ratio_to_manual"] = (
            selected_mass / max(manual_mass, 1e-6)
        )
        prediction["visibility_proxy_ratio_to_manual"] = (
            selected_proxy / max(manual_proxy, 1e-6)
        )
        prediction["diffuse_shadow_proxy_failure"] = float(
            prediction["spot_peak_mass_ratio_to_manual"] < 0.60
        )
        rows.append(prediction)
    return pd.DataFrame(rows)


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    summary = (
        predictions.groupby("method", as_index=False)
        .agg(
            n_videos=("sample_id", "nunique"),
            median_pattern_ncc=("pattern_ncc", "median"),
            mean_pattern_ncc=("pattern_ncc", "mean"),
            median_pattern_ssim=("pattern_ssim", "median"),
            median_gradient_ncc=("gradient_ncc", "median"),
            median_target_similarity=("target_similarity", "median"),
            median_absolute_frame_error=("absolute_frame_error", "median"),
            median_spot_mass_ratio=(
                "spot_peak_mass_ratio_to_manual",
                "median",
            ),
            diffuse_shadow_proxy_rate=(
                "diffuse_shadow_proxy_failure",
                "mean",
            ),
            median_visibility_rank=("spot_visibility_rank", "median"),
            held_video_overlap=("held_video_overlap", "sum"),
        )
        .sort_values(
            [
                "diffuse_shadow_proxy_rate",
                "median_target_similarity",
                "mean_pattern_ncc",
            ],
            ascending=[True, False, False],
        )
    )
    return summary


def display_crop(frame: np.ndarray, rect: Rect) -> Image.Image:
    ys, xs = rect.as_slices()
    image = Image.fromarray(np.asarray(frame)[ys, xs, :3]).convert("RGB")
    image = ImageOps.fit(
        image, (250, 350), method=Image.Resampling.BICUBIC
    )
    return ImageEnhance.Contrast(image).enhance(1.35)


def full_frame_with_rois(
    frame: np.ndarray, auto_roi: Rect, human_roi: Rect
) -> Image.Image:
    image = Image.fromarray(np.asarray(frame)[..., :3]).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (auto_roi.x, auto_roi.y, auto_roi.x2, auto_roi.y2),
        outline=(0, 255, 255),
        width=max(3, image.width // 350),
    )
    draw.rectangle(
        (human_roi.x, human_roi.y, human_roi.x2, human_roi.y2),
        outline=(255, 215, 0),
        width=max(3, image.width // 350),
    )
    image.thumbnail((350, 350), Image.Resampling.LANCZOS)
    return image


def save_atlases(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    config: dict[str, Any],
    *,
    report_root: Path,
) -> None:
    records = {
        str(row.sample_id): row
        for row in manifest.itertuples(index=False)
    }
    for method in METHODS_FOR_ATLASES:
        selected = predictions.loc[
            predictions["method"] == method
        ].sort_values("sample_id")
        method_root = report_root / method
        method_root.mkdir(parents=True, exist_ok=True)
        for page, start in enumerate(range(0, len(selected), 5), start=1):
            batch = selected.iloc[start : start + 5]
            fig, axes = plt.subplots(
                len(batch),
                3,
                figsize=(11.5, 3.45 * len(batch)),
                squeeze=False,
            )
            for row_index, prediction in enumerate(
                batch.itertuples(index=False)
            ):
                sample = str(prediction.sample_id)
                record = records[sample]
                auto_roi = load_roi(sample, config)
                human_roi = Rect(
                    x=int(record.roi_x),
                    y=int(record.roi_y),
                    width=int(record.roi_width),
                    height=int(record.roi_height),
                    source_width=int(record.source_width),
                    source_height=int(record.source_height),
                )
                manual = load_frame(
                    ROOT / str(record.frames_dir),
                    int(prediction.manual_frame_index),
                )
                machine = load_frame(
                    ROOT / str(record.frames_dir),
                    int(prediction.selected_frame_index),
                )
                axes[row_index, 0].imshow(
                    full_frame_with_rois(manual, auto_roi, human_roi)
                )
                axes[row_index, 0].set_title(
                    f"{sample} | ROI: cyan auto, yellow human",
                    fontsize=9,
                )
                axes[row_index, 1].imshow(display_crop(manual, auto_roi))
                axes[row_index, 1].set_title(
                    f"human f{prediction.manual_frame_index}",
                    fontsize=9,
                )
                axes[row_index, 2].imshow(display_crop(machine, auto_roi))
                axes[row_index, 2].set_title(
                    f"machine f{prediction.selected_frame_index} | "
                    f"NCC {prediction.pattern_ncc:.2f} | "
                    f"spot ratio {prediction.spot_peak_mass_ratio_to_manual:.2f}",
                    fontsize=9,
                )
                for axis in axes[row_index]:
                    axis.axis("off")
            fig.suptitle(
                f"Strict leave-one-video-out: {method} "
                f"(page {page}/{math.ceil(len(selected) / 5)})",
                fontsize=14,
                fontweight="bold",
            )
            fig.tight_layout(rect=(0, 0, 1, 0.975))
            stem = method_root / f"all_samples_page_{page:02d}"
            fig.savefig(stem.with_suffix(".png"), dpi=180)
            fig.savefig(stem.with_suffix(".pdf"))
            plt.close(fig)


def save_benchmark(
    summary: pd.DataFrame, report_root: Path
) -> None:
    order = summary.sort_values(
        "median_target_similarity", ascending=True
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
    axes[0].barh(order["method"], order["median_pattern_ncc"])
    axes[0].set_xlabel("Median human–machine NCC")
    axes[0].set_xlim(0.0, 1.0)
    axes[1].barh(order["method"], order["median_pattern_ssim"])
    axes[1].set_xlabel("Median SSIM")
    axes[1].set_xlim(0.0, 1.0)
    axes[2].barh(
        order["method"], 100.0 * order["diffuse_shadow_proxy_rate"]
    )
    axes[2].set_xlabel("Diffuse-shadow proxy failures (%)")
    axes[2].set_xlim(0.0, 100.0)
    for axis in axes:
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle(
        "RHEED image-aware keyframe benchmark: strict video LOO",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(report_root / "deep_visibility_benchmark.png", dpi=200)
    fig.savefig(report_root / "deep_visibility_benchmark.pdf")
    plt.close(fig)


def save_failure_panel(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    config: dict[str, Any],
    *,
    method: str,
    report_root: Path,
) -> None:
    records = {
        str(row.sample_id): row
        for row in manifest.itertuples(index=False)
    }
    failures = (
        predictions.loc[predictions["method"] == method]
        .nsmallest(6, "target_similarity")
        .sort_values("target_similarity")
    )
    fig, axes = plt.subplots(3, 4, figsize=(13, 12))
    for index, prediction in enumerate(failures.itertuples(index=False)):
        sample = str(prediction.sample_id)
        record = records[sample]
        roi = load_roi(sample, config)
        manual = load_frame(
            ROOT / str(record.frames_dir),
            int(prediction.manual_frame_index),
        )
        machine = load_frame(
            ROOT / str(record.frames_dir),
            int(prediction.selected_frame_index),
        )
        row_index, col = divmod(index, 2)
        left = axes[row_index, 2 * col]
        right = axes[row_index, 2 * col + 1]
        left.imshow(display_crop(manual, roi))
        right.imshow(display_crop(machine, roi))
        left.set_title(f"{sample} human")
        right.set_title(
            f"machine | NCC {prediction.pattern_ncc:.2f}\n"
            f"spot ratio {prediction.spot_peak_mass_ratio_to_manual:.2f}"
        )
        left.axis("off")
        right.axis("off")
    fig.suptitle(
        f"Lowest-similarity held videos: {method}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(report_root / "lowest_similarity_cases.png", dpi=200)
    fig.savefig(report_root / "lowest_similarity_cases.pdf")
    plt.close(fig)


def save_confidence_validation(
    predictions: pd.DataFrame,
    *,
    method: str,
    report_root: Path,
) -> dict[str, float]:
    selected = predictions.loc[
        predictions["method"] == method
    ].copy()
    uncertainty = selected["prediction_margin"].to_numpy()
    similarity = selected["target_similarity"].to_numpy()
    raw_reliability = -uncertainty
    rho, p_value = stats.spearmanr(raw_reliability, 1.0 - similarity)
    calibrator = IsotonicRegression(
        y_min=0.0, y_max=1.0, out_of_bounds="clip"
    )
    confidence = calibrator.fit_transform(raw_reliability, similarity)
    cross_fitted_confidence = np.empty(len(selected), dtype=float)
    for held_index in range(len(selected)):
        training_mask = np.arange(len(selected)) != held_index
        fold_calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        fold_calibrator.fit(
            raw_reliability[training_mask],
            similarity[training_mask],
        )
        cross_fitted_confidence[held_index] = fold_calibrator.predict(
            [raw_reliability[held_index]]
        )[0]
    cross_fitted_rho, cross_fitted_p = stats.spearmanr(
        cross_fitted_confidence, 1.0 - similarity
    )
    selected["confidence"] = confidence
    ordered = selected.sort_values("confidence")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    scatter = axes[0].scatter(
        confidence,
        similarity,
        c=selected["pattern_ncc"],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        edgecolors="black",
        linewidths=0.6,
        s=70,
    )
    axes[0].set_xlabel("Calibrated reliability confidence")
    axes[0].set_ylabel("Realized human-frame composite similarity")
    axes[0].set_title(
        "Development-calibrated confidence vs held similarity\n"
        f"raw reliability vs error: rho={rho:.2f}, p={p_value:.3f}"
    )
    axes[0].grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes[0], label="Pattern NCC")
    axes[1].bar(
        np.arange(len(ordered)),
        ordered["target_similarity"],
        color=plt.cm.viridis(ordered["confidence"]),
    )
    axes[1].set_xticks(np.arange(len(ordered)))
    axes[1].set_xticklabels(
        ordered["sample_id"], rotation=90, fontsize=8
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Realized composite similarity")
    axes[1].set_title(
        "All 25 held videos ordered by confidence; no omitted failures"
    )
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle(
        "V5 deep spot-visibility reliability audit",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(report_root / "confidence_validation.png", dpi=200)
    fig.savefig(report_root / "confidence_validation.pdf")
    plt.close(fig)
    return {
        "confidence_error_spearman_rho": float(rho),
        "confidence_error_spearman_p": float(p_value),
        "cross_fitted_calibration_error_spearman_rho": float(
            cross_fitted_rho
        ),
        "cross_fitted_calibration_error_spearman_p": float(
            cross_fitted_p
        ),
    }


def fit_final_bundle(
    table: pd.DataFrame,
    *,
    visual_feature_names: list[str],
    dino_feature_names: list[str],
    model_id: str,
    model_revision: str | None,
    output_root: Path,
    visibility_gate: float,
    predictions: pd.DataFrame,
    selected_method: str,
) -> None:
    feature_names = visual_feature_names + dino_feature_names
    train_values, _, transforms = fit_transform(
        table,
        table.iloc[:1],
        feature_names,
        pca_components=64,
    )
    deep_ridge = Ridge(alpha=10.0)
    deep_ridge.fit(train_values, table["target_similarity"])
    _, pairwise_model = pairwise_rank_scores(
        train_values, train_values[:1], table
    )
    visual_imputer = SimpleImputer(strategy="median")
    visual_values = visual_imputer.fit_transform(
        table[visual_feature_names]
    )
    visual_tree = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        max_features=0.75,
        random_state=17,
        n_jobs=-1,
    )
    visual_tree.fit(visual_values, table["target_similarity"])
    loo = predictions.loc[predictions["method"] == selected_method].copy()
    calibrator = IsotonicRegression(
        y_min=0.0, y_max=1.0, out_of_bounds="clip"
    )
    calibrator.fit(
        -loo["prediction_margin"], loo["target_similarity"]
    )
    joblib.dump(
        {
            "schema_version": 2,
            "model_family": (
                "dinov2_ridge_pairwise_visual_tree_spot_visibility_ranker"
            ),
            "foundation_model_id": model_id,
            "foundation_model_revision": model_revision,
            "input_representation": (
                "contrast-normalized grayscale ROI, aspect-preserving "
                "224x224 padding"
            ),
            "dino_embedding_representation": (
                "CLS + mean patch + patch standard deviation"
            ),
            "feature_names": feature_names,
            "visual_feature_names": visual_feature_names,
            "dino_feature_names": dino_feature_names,
            "imputer": transforms["imputer"],
            "scaler": transforms["scaler"],
            "pca": transforms["pca"],
            "deep_ridge": deep_ridge,
            "pairwise_ranker": pairwise_model,
            "visual_imputer": visual_imputer,
            "visual_tree": visual_tree,
            "score_weights": {
                "deep_ridge_rank": 0.38,
                "pairwise_rank": 0.27,
                "visual_tree_rank": 0.25,
                "spot_visibility_rank": 0.10,
            },
            "visibility_gate": visibility_gate,
            "confidence_calibrator": calibrator,
            "confidence_input": "negative_selection_margin",
            "confidence_definition": (
                "Isotonic expected composite human-frame similarity from "
                "the negative top-two selection margin. In strict LOO, a "
                "large isolated winning margin marks unsupported "
                "extrapolation and lower reliability. This is not a "
                "probability of correctness."
            ),
            "training_video_count": int(table["sample_id"].nunique()),
            "training_candidate_count": int(len(table)),
            "validation_protocol": "strict_leave_one_video_out",
            "validation_method": selected_method,
            "validation_video_count": int(len(loo)),
            "held_video_overlap_sum": int(
                loo["held_video_overlap"].sum()
            ),
            "prospective_note": (
                "Refit on all retained annotations; requires prospective "
                "validation on newly acquired videos."
            ),
        },
        output_root / "dinov2_spot_visibility_ranker.joblib",
        compress=3,
    )


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    experiment_id = config["experiment_id"]
    output_root = (
        ROOT
        / "outputs"
        / "rheed_auto_roi_keyframe"
        / experiment_id
    )
    report_root = (
        ROOT
        / "reports"
        / "rheed_auto_roi_keyframe"
        / experiment_id
    )
    output_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(config)
    candidates = pd.read_csv(
        ROOT / config["candidate_table"], dtype={"sample_id": str}
    )
    candidates = candidates.loc[
        candidates["sample_id"].isin(manifest["sample_id"])
    ].copy()
    if candidates["sample_id"].nunique() != len(manifest):
        raise ValueError("Candidate table does not cover retained manifest")
    descriptors, embeddings, feature_metadata = load_or_build_features(
        candidates,
        manifest,
        config,
        output_root=output_root,
        rebuild=args.rebuild_features,
        device=args.device,
    )
    table, dino_names = assemble_candidate_features(
        candidates, descriptors, embeddings
    )
    table.to_csv(output_root / "deep_candidate_table.csv", index=False)
    predictions = grouped_leave_one_out(table, dino_names=dino_names)
    predictions = add_manual_visibility_metrics(predictions, descriptors)
    predictions.to_csv(
        output_root / "deep_visibility_loo_predictions.csv", index=False
    )
    summary = summarize(predictions)
    summary.to_csv(
        output_root / "deep_visibility_summary.csv", index=False
    )
    save_benchmark(summary, report_root)
    save_atlases(
        predictions, manifest, config, report_root=report_root
    )
    selected_method = config["selected_method"]
    save_failure_panel(
        predictions,
        manifest,
        config,
        method=selected_method,
        report_root=report_root,
    )
    confidence_metrics = save_confidence_validation(
        predictions,
        method=selected_method,
        report_root=report_root,
    )
    visual_names = (
        V4_FEATURES
        + list(SPOT_VISIBILITY_FEATURES)
        + ["visibility_proxy"]
        + VISIBILITY_RANK_FEATURES
        + ["spot_visibility_rank"]
    )
    fit_final_bundle(
        table,
        visual_feature_names=visual_names,
        dino_feature_names=dino_names,
        model_id=config["foundation_model_id"],
        model_revision=feature_metadata[
            "foundation_model_resolved_revision"
        ],
        output_root=output_root,
        visibility_gate=float(config["visibility_gate"]),
        predictions=predictions,
        selected_method=selected_method,
    )
    metadata = {
        "experiment_id": experiment_id,
        "config": config,
        "retained_video_count": int(len(manifest)),
        "candidate_count": int(len(table)),
        "removelist_overlap": sorted(
            set(candidates["sample_id"])
            & set(
                pd.read_csv(
                    ROOT / config["manifest"], dtype={"sample_id": str}
                )
                .loc[lambda frame: frame["excluded_by_removelist"]]
                ["sample_id"]
            )
        ),
        "selected_method": selected_method,
        "selected_summary": {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in (
                summary.loc[summary["method"] == selected_method]
                .iloc[0]
                .to_dict()
                .items()
            )
        },
        "confidence_validation": confidence_metrics,
        "feature_extraction": feature_metadata,
    }
    (output_root / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
