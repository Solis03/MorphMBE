"""Image-content features for judging RHEED spot-pattern visibility.

The phase tracker in :mod:`automatic_roi_keyframe` deliberately emphasizes
rotation geometry.  This module supplies the complementary image-quality
signal: a useful frame must contain several compact, locally prominent
diffraction spots rather than only a smooth bright phosphor-screen haze.

All deterministic descriptors are computed inside the predicted ROI.  The
optional DINOv2 encoder is frozen and is used only as a general visual feature
extractor; fitted ranking heads remain video-group cross-validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops

from rheed2morph.rheed.automatic_roi_keyframe import Rect


SPOT_VISIBILITY_FEATURES = (
    "raw_mean",
    "raw_std",
    "raw_dynamic_range",
    "raw_shadow_fraction",
    "raw_saturation_fraction",
    "normalized_gradient_mean",
    "normalized_laplacian_variance",
    "fine_dog_std",
    "medium_dog_std",
    "medium_dog_noise",
    "medium_dog_snr",
    "low_frequency_std",
    "high_to_low_frequency_ratio",
    "haze_dominance",
    "spot_peak_count",
    "spot_peak_top1",
    "spot_peak_top4_mean",
    "spot_peak_top8_mass",
    "spot_peak_snr_top4",
    "spot_component_count",
    "spot_component_area_median",
    "spot_component_compactness",
    "spot_energy_concentration",
    "spot_vertical_span",
    "spot_horizontal_spread",
    "spot_verticality",
    "spot_column_alignment",
)


@dataclass(frozen=True)
class SpotVisibilityAnalysis:
    """Descriptor values plus normalized images used for visual diagnostics."""

    features: dict[str, float]
    normalized: np.ndarray
    response: np.ndarray
    peak_coordinates: np.ndarray


def _brightness_crop(frame: np.ndarray, rect: Rect) -> np.ndarray:
    values = np.asarray(frame)
    ys, xs = rect.as_slices()
    crop = values[ys, xs]
    if crop.ndim == 2:
        brightness = crop.astype(np.float32)
    else:
        brightness = crop[..., :3].astype(np.float32).max(axis=2)
    if brightness.size == 0:
        raise ValueError("ROI produced an empty image crop")
    if float(np.nanmax(brightness)) > 1.5:
        brightness /= 255.0
    return np.clip(
        np.nan_to_num(brightness, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    )


def _resize(image: np.ndarray, size: tuple[int, int] = (128, 192)) -> np.ndarray:
    return np.asarray(
        Image.fromarray(np.asarray(image, dtype=np.float32)).resize(
            size, Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )


def analyze_spot_visibility(
    frame: np.ndarray,
    rect: Rect,
    *,
    analysis_size: tuple[int, int] = (128, 192),
) -> SpotVisibilityAnalysis:
    """Measure compact diffraction-spot visibility within ``rect``.

    Percentile normalization makes the shape descriptors robust to camera
    gain, while the raw-intensity descriptors preserve the ability to detect
    shadow, saturation and diffuse over-exposure.  Multi-scale
    difference-of-Gaussian responses distinguish compact spots from broad
    low-frequency haze.
    """

    raw = _resize(_brightness_crop(frame, rect), analysis_size)
    low, high = np.percentile(raw, [2.0, 99.8])
    dynamic_range = max(float(high - low), 1e-5)
    normalized = np.clip((raw - low) / dynamic_range, 0.0, 1.0)

    fine = ndimage.gaussian_filter(normalized, 0.7) - ndimage.gaussian_filter(
        normalized, 2.0
    )
    medium = ndimage.gaussian_filter(
        normalized, 1.1
    ) - ndimage.gaussian_filter(normalized, 4.2)
    low_frequency = ndimage.gaussian_filter(normalized, 12.0)
    high_frequency = normalized - low_frequency

    medium_center = float(np.median(medium))
    medium_noise = float(
        1.4826 * np.median(np.abs(medium - medium_center)) + 1e-6
    )
    threshold = max(
        3.0 * medium_noise,
        float(np.quantile(medium, 0.985)),
        0.012,
    )
    peaks = peak_local_max(
        medium,
        min_distance=4,
        threshold_abs=threshold,
        exclude_border=5,
    )
    peak_values = (
        np.sort(medium[tuple(peaks.T)])[::-1]
        if len(peaks)
        else np.empty(0, dtype=np.float32)
    )
    top1 = float(peak_values[0]) if len(peak_values) else 0.0
    top4 = peak_values[:4]
    top8 = peak_values[:8]

    mask = medium > threshold
    components = []
    for region in regionprops(label(mask, connectivity=2)):
        if 3 <= region.area <= 420:
            components.append(region)
    component_areas = np.asarray(
        [float(region.area) for region in components], dtype=float
    )
    component_compactness = []
    for region in components:
        perimeter = max(float(region.perimeter), 1.0)
        component_compactness.append(
            float(4.0 * np.pi * region.area / (perimeter * perimeter))
        )

    positive = np.maximum(medium, 0.0)
    positive_total = float(positive.sum()) + 1e-8
    peak_energy = 0.0
    for y, x in peaks[:12]:
        y0, y1 = max(0, y - 4), min(positive.shape[0], y + 5)
        x0, x1 = max(0, x - 4), min(positive.shape[1], x + 5)
        peak_energy += float(positive[y0:y1, x0:x1].sum())

    if len(peaks) >= 3:
        y_values = peaks[:, 0].astype(float)
        x_values = peaks[:, 1].astype(float)
        centered = np.column_stack(
            (x_values - x_values.mean(), y_values - y_values.mean())
        )
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        major = eigenvectors[:, int(np.argmax(eigenvalues))]
        verticality = float(abs(major[1]))
        vertical_span = float(np.ptp(y_values) / max(normalized.shape[0], 1))
        horizontal_spread = float(
            np.std(x_values) / max(normalized.shape[1], 1)
        )
        column_alignment = float(
            vertical_span / max(horizontal_spread, 1e-3)
        )
    else:
        verticality = 0.0
        vertical_span = 0.0
        horizontal_spread = 1.0
        column_alignment = 0.0

    gy, gx = np.gradient(normalized)
    laplacian = ndimage.laplace(normalized)
    low_std = float(np.std(low_frequency))
    medium_std = float(np.std(medium))
    high_low_ratio = medium_std / max(low_std, 1e-6)
    haze_dominance = low_std / max(medium_std, 1e-6)
    raw_std = float(np.std(raw))
    features = {
        "raw_mean": float(np.mean(raw)),
        "raw_std": raw_std,
        "raw_dynamic_range": dynamic_range,
        "raw_shadow_fraction": float(np.mean(raw < 0.025)),
        "raw_saturation_fraction": float(np.mean(raw > 0.985)),
        "normalized_gradient_mean": float(np.mean(np.hypot(gx, gy))),
        "normalized_laplacian_variance": float(np.var(laplacian)),
        "fine_dog_std": float(np.std(fine)),
        "medium_dog_std": medium_std,
        "medium_dog_noise": medium_noise,
        "medium_dog_snr": float(
            np.mean(top4) / medium_noise if len(top4) else 0.0
        ),
        "low_frequency_std": low_std,
        "high_to_low_frequency_ratio": float(high_low_ratio),
        "haze_dominance": float(haze_dominance),
        "spot_peak_count": float(len(peaks)),
        "spot_peak_top1": top1,
        "spot_peak_top4_mean": float(np.mean(top4)) if len(top4) else 0.0,
        "spot_peak_top8_mass": float(np.sum(top8)),
        "spot_peak_snr_top4": float(
            np.mean(top4) / medium_noise if len(top4) else 0.0
        ),
        "spot_component_count": float(len(components)),
        "spot_component_area_median": (
            float(np.median(component_areas)) if len(component_areas) else 0.0
        ),
        "spot_component_compactness": (
            float(np.median(component_compactness))
            if component_compactness
            else 0.0
        ),
        "spot_energy_concentration": float(
            np.clip(peak_energy / positive_total, 0.0, 1.0)
        ),
        "spot_vertical_span": vertical_span,
        "spot_horizontal_spread": horizontal_spread,
        "spot_verticality": verticality,
        "spot_column_alignment": float(min(column_alignment, 50.0)),
    }
    return SpotVisibilityAnalysis(
        features=features,
        normalized=normalized,
        response=medium,
        peak_coordinates=np.asarray(peaks, dtype=int),
    )


def prepare_dinov2_image(
    frame: np.ndarray,
    rect: Rect,
    *,
    output_size: int = 224,
) -> Image.Image:
    """Create a contrast-normalized, aspect-preserving DINOv2 input."""

    analysis = analyze_spot_visibility(frame, rect)
    values = np.clip(analysis.normalized * 255.0, 0.0, 255.0).astype(np.uint8)
    image = Image.fromarray(values, mode="L").convert("RGB")
    return ImageOps.pad(
        image,
        (output_size, output_size),
        method=Image.Resampling.BICUBIC,
        color=(0, 0, 0),
        centering=(0.5, 0.5),
    )


class DinoV2Embedder:
    """Frozen DINOv2 CLS and patch-statistic feature extractor."""

    def __init__(
        self,
        model_id: str = "facebook/dinov2-small",
        *,
        revision: str | None = None,
        device: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, Dinov2Model
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "DINOv2 extraction requires torch and transformers"
            ) from exc

        if device is None:
            device = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        self.torch = torch
        self.device = torch.device(device)
        self.model_id = model_id
        self.revision = revision
        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            use_fast=False,
        )
        self.model = Dinov2Model.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        self.model.eval().to(self.device)
        self.commit_hash = getattr(self.model.config, "_commit_hash", None)

    def encode(
        self,
        images: Sequence[Image.Image],
        *,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Return ``[CLS, mean(patch), std(patch)]`` per image."""

        encoded: list[np.ndarray] = []
        torch = self.torch
        for start in range(0, len(images), batch_size):
            batch = list(images[start : start + batch_size])
            inputs = self.processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            with torch.inference_mode():
                hidden = self.model(pixel_values=pixel_values).last_hidden_state
            cls = hidden[:, 0]
            patches = hidden[:, 1:]
            vector = torch.cat(
                (cls, patches.mean(dim=1), patches.std(dim=1)), dim=1
            )
            encoded.append(vector.float().cpu().numpy())
        if not encoded:
            width = int(self.model.config.hidden_size) * 3
            return np.empty((0, width), dtype=np.float32)
        return np.concatenate(encoded, axis=0).astype(np.float32)


def visibility_proxy(features: dict[str, float]) -> float:
    """A monotonic, interpretable spot-versus-haze score.

    This score is intended for diagnostics and hard gating.  Supervised
    rankers receive the individual descriptors rather than this formula.
    """

    peak_mass = np.log1p(max(features["spot_peak_top8_mass"], 0.0) * 8.0)
    peak_snr = np.log1p(max(features["spot_peak_snr_top4"], 0.0))
    frequency = np.log1p(
        max(features["high_to_low_frequency_ratio"], 0.0) * 10.0
    )
    lattice = min(max(features["spot_vertical_span"], 0.0), 1.0)
    concentration = min(
        max(features["spot_energy_concentration"], 0.0), 1.0
    )
    diffuse_penalty = np.log1p(max(features["haze_dominance"], 0.0))
    return float(
        0.30 * peak_mass
        + 0.22 * peak_snr
        + 0.18 * frequency
        + 0.12 * lattice
        + 0.18 * concentration
        - 0.18 * diffuse_penalty
    )


def feature_matrix(
    analyses: Iterable[SpotVisibilityAnalysis],
) -> np.ndarray:
    """Convert analyses to a stable deterministic descriptor matrix."""

    return np.asarray(
        [
            [analysis.features[name] for name in SPOT_VISIBILITY_FEATURES]
            for analysis in analyses
        ],
        dtype=np.float32,
    )


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    if not len(values):
        return np.empty(0, dtype=float)
    return rankdata(values, method="average") / float(len(values))


def score_deep_visibility_candidates(
    candidates: Sequence[dict[str, object]],
    frames: dict[int, np.ndarray],
    rect: Rect,
    bundle_path: str | Path,
    *,
    foundation_cache_dir: str | Path | None = None,
    device: str | None = None,
) -> dict[str, object]:
    """Score one video's physical candidates with a fitted V5 bundle.

    The function returns the selected candidate, calibrated confidence and
    per-candidate component scores.  Frames must correspond to the candidate
    indices only; source video decoding remains the caller's responsibility.
    """

    try:
        import joblib
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "Deep visibility inference requires joblib and pandas"
        ) from exc

    bundle = joblib.load(Path(bundle_path))
    if int(bundle.get("schema_version", -1)) != 2:
        raise ValueError("Expected a schema-version 2 deep visibility bundle")
    unique_indices = sorted({int(row["frame_index"]) for row in candidates})
    missing = [index for index in unique_indices if index not in frames]
    if missing:
        raise KeyError(f"Missing {len(missing)} candidate frames")

    analyses = [
        analyze_spot_visibility(frames[index], rect)
        for index in unique_indices
    ]
    descriptor_lookup = {
        index: {
            **analysis.features,
            "visibility_proxy": visibility_proxy(analysis.features),
        }
        for index, analysis in zip(unique_indices, analyses)
    }
    dino_images = [
        prepare_dinov2_image(frames[index], rect)
        for index in unique_indices
    ]
    embedder = DinoV2Embedder(
        str(bundle["foundation_model_id"]),
        revision=bundle.get("foundation_model_revision"),
        device=device,
        cache_dir=foundation_cache_dir,
    )
    embeddings = embedder.encode(dino_images)
    embedding_lookup = {
        index: embedding
        for index, embedding in zip(unique_indices, embeddings)
    }

    rows = []
    for candidate in candidates:
        index = int(candidate["frame_index"])
        rows.append(
            {
                **candidate,
                **descriptor_lookup[index],
                **{
                    name: float(value)
                    for name, value in zip(
                        bundle["dino_feature_names"],
                        embedding_lookup[index],
                    )
                },
            }
        )
    table = pd.DataFrame(rows)
    rank_sources = (
        "spot_peak_top8_mass",
        "spot_peak_snr_top4",
        "high_to_low_frequency_ratio",
        "spot_energy_concentration",
        "spot_vertical_span",
        "spot_column_alignment",
        "normalized_laplacian_variance",
        "visibility_proxy",
    )
    for name in rank_sources:
        table[f"qv_{name}"] = table[name].rank(
            pct=True, method="average"
        )
    table["qv_haze_rejection"] = 1.0 - table["haze_dominance"].rank(
        pct=True, method="average"
    )
    rank_names = [
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
    table["spot_visibility_rank"] = table[rank_names].mean(axis=1)

    feature_names = list(bundle["feature_names"])
    deep_values = bundle["imputer"].transform(table[feature_names])
    deep_values = bundle["scaler"].transform(deep_values)
    if bundle.get("pca") is not None:
        deep_values = bundle["pca"].transform(deep_values)
    ridge_scores = bundle["deep_ridge"].predict(deep_values)
    pair_scores = bundle["pairwise_ranker"].decision_function(deep_values)
    visual_values = bundle["visual_imputer"].transform(
        table[list(bundle["visual_feature_names"])]
    )
    tree_scores = bundle["visual_tree"].predict(visual_values)
    weights = bundle["score_weights"]
    final_scores = (
        float(weights["deep_ridge_rank"]) * _percentile_rank(ridge_scores)
        + float(weights["pairwise_rank"]) * _percentile_rank(pair_scores)
        + float(weights["visual_tree_rank"]) * _percentile_rank(tree_scores)
        + float(weights["spot_visibility_rank"])
        * table["spot_visibility_rank"].to_numpy()
    )
    gate = float(bundle["visibility_gate"])
    eligible = table["spot_visibility_rank"].to_numpy() >= gate
    effective = final_scores.copy()
    if eligible.any():
        effective[~eligible] = -np.inf
    position = int(np.argmax(effective))
    ordered = np.sort(effective[np.isfinite(effective)])[::-1]
    selection_margin = (
        float(ordered[0] - ordered[1])
        if len(ordered) > 1
        else float(ordered[0])
    )
    confidence = float(
        bundle["confidence_calibrator"].predict(
            [-selection_margin]
        )[0]
    )
    selected = dict(candidates[position])
    return {
        "selected_candidate": selected,
        "score": float(final_scores[position]),
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "visibility_gate": gate,
        "visibility_rank": float(
            table.iloc[position]["spot_visibility_rank"]
        ),
        "selection_margin": selection_margin,
        "candidate_count": int(len(table)),
        "eligible_candidate_count": int(eligible.sum()),
        "foundation_model_id": str(bundle["foundation_model_id"]),
        "component_scores": {
            "deep_ridge": float(ridge_scores[position]),
            "pairwise": float(pair_scores[position]),
            "visual_tree": float(tree_scores[position]),
        },
        "candidate_scores": [
            {
                "frame_index": int(row.frame_index),
                "tracker": str(row.tracker),
                "score": float(score),
                "spot_visibility_rank": float(row.spot_visibility_rank),
                "eligible": bool(is_eligible),
            }
            for row, score, is_eligible in zip(
                table.itertuples(index=False), final_scores, eligible
            )
        ],
    }
