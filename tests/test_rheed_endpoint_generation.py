from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis.rheed_endpoint_generation.endpoint_ensemble import (
    predict_endpoint,
)
from analysis.rheed_endpoint_generation.streak_features import (
    PRIMARY_STREAK_FEATURE,
    extract_streak_features,
)
from analysis.rheed_to_afm_functional_morphology.render import (
    edge_preserving_terrace_blend,
    regime_adaptive_microisland_terrace_blend,
    regime_adaptive_terrace_blend,
)


def _unit_sq(array: np.ndarray) -> float:
    centered = np.asarray(array, dtype=float) - float(np.mean(array))
    return float(np.sqrt(np.mean(np.square(centered))))


def test_local_peak_feature_separates_horizontal_streaks_from_spots() -> None:
    yy, xx = np.indices((224, 224))
    streak = np.zeros((224, 224), dtype=float)
    spots = np.zeros((224, 224), dtype=float)
    for y in (55, 88, 121, 154):
        streak += np.exp(
            -0.5
            * (
                np.square((yy - y) / 2.0)
                + np.square((xx - 150) / 13.0)
            )
        )
        spots += np.exp(
            -0.5
            * (
                np.square((yy - y) / 3.0)
                + np.square((xx - 150) / 3.0)
            )
        )
    streak_clip = np.repeat(
        np.clip(streak * 255, 0, 255).astype(np.uint8)[None], 8, axis=0
    )
    spot_clip = np.repeat(
        np.clip(spots * 255, 0, 255).astype(np.uint8)[None], 8, axis=0
    )
    streak_value = extract_streak_features(streak_clip)[
        PRIMARY_STREAK_FEATURE
    ]
    spot_value = extract_streak_features(spot_clip)[
        PRIMARY_STREAK_FEATURE
    ]
    assert streak_value > spot_value + 0.5


def test_endpoint_prediction_does_not_use_query_target() -> None:
    rng = np.random.default_rng(12)
    embeddings = rng.normal(size=(12, 16))
    streak = rng.uniform(1.0, 2.5, size=12)
    target = np.exp(rng.normal(np.log(2.0), 0.5, size=12))
    train = np.arange(11)
    original = predict_endpoint(
        embeddings=embeddings,
        streak=streak,
        target_nm=target,
        train=train,
        query_embedding=embeddings[11],
        query_streak=float(streak[11]),
    )
    changed = target.copy()
    changed[11] = 1.0e6
    repeated = predict_endpoint(
        embeddings=embeddings,
        streak=streak,
        target_nm=changed,
        train=train,
        query_embedding=embeddings[11],
        query_streak=float(streak[11]),
    )
    assert original == repeated


def test_regime_adaptive_renderer_preserves_requested_unit_sq() -> None:
    rng = np.random.default_rng(7)
    structure = rng.normal(size=(128, 128))
    prior = rng.normal(size=(128, 128))
    keywords = {
        "boundary_width_px": 0.75,
        "structure_weight": 0.50,
        "plateau_weight": 0.14,
        "relief_weight": 0.18,
        "spectral_weight": 0.13,
        "texture_weight": 0.05,
    }
    smooth = regime_adaptive_terrace_blend(
        structure,
        prior,
        conditioning_sq_nm=0.6,
        smooth_full_below_nm=0.8,
        terrace_full_above_nm=1.6,
        **keywords,
    )
    rough = regime_adaptive_terrace_blend(
        structure,
        prior,
        conditioning_sq_nm=2.0,
        smooth_full_below_nm=0.8,
        terrace_full_above_nm=1.6,
        **keywords,
    )
    frozen_terrace = edge_preserving_terrace_blend(
        structure, prior, **keywords
    )
    assert np.isclose(_unit_sq(smooth), 1.0, atol=1e-5)
    assert np.allclose(rough, frozen_terrace)


def test_microisland_renderer_is_unit_sq_and_suppresses_pixel_noise() -> None:
    rng = np.random.default_rng(19)
    structure = rng.normal(size=(128, 128))
    prior = rng.normal(size=(128, 128))
    keywords = {
        "boundary_width_px": 0.75,
        "structure_weight": 0.50,
        "plateau_weight": 0.14,
        "relief_weight": 0.18,
        "spectral_weight": 0.13,
        "texture_weight": 0.05,
        "smooth_full_below_nm": 0.8,
        "terrace_full_above_nm": 1.6,
    }
    smooth = regime_adaptive_microisland_terrace_blend(
        structure, prior, conditioning_sq_nm=0.6, **keywords
    )
    gy, gx = np.gradient(smooth)
    assert np.isclose(_unit_sq(smooth), 1.0, atol=1e-5)
    assert float(np.mean(np.hypot(gx, gy))) < 0.75


def test_m16_generation_config_marks_retrospective_development_boundary() -> None:
    repository = Path(__file__).resolve().parents[1]
    config = json.loads(
        (
            repository
            / "configs"
            / "rheed_m16_end_to_end_generation_line3_full28_smooth_v2.json"
        ).read_text(encoding="utf-8")
    )
    assert config["selected_method_frozen_before_expanded_cohort_run"] is False
    assert "Retrospective" in config["claim_boundary"]
    assert "not a prospective untouched test" in config["claim_boundary"]
