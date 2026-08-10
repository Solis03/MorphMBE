"""Deploy the endpoint-aware Sq head and non-retrieval AFM generator."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler

from analysis.rheed_auto_input_robustness.confidence import (
    _empirical_risk,
    _isotonic_expected_error,
    angular_coverage_risk,
    combine_tta_and_head_confidence,
)
from analysis.rheed_endpoint_generation.endpoint_ensemble import (
    EndpointPrediction,
    predict_endpoint,
)
from analysis.rheed_endpoint_generation.run_endpoint_ensemble import (
    _support_risk,
)
from analysis.rheed_endpoint_generation.streak_features import (
    PRIMARY_STREAK_FEATURE,
    extract_streak_features,
)
from analysis.rheed_rough_island_redesign.connectivity import (
    _feature_matrix,
)
from analysis.rheed_to_afm_distinct_confidence.matern import (
    DescriptorMaternGenerator,
)
from analysis.rheed_to_afm_distinct_confidence.run import (
    _blend_ensembles,
    _group_targets,
    _predictor_factory,
)
from analysis.rheed_to_afm_distinct_confidence.variance import (
    VarianceCalibrator,
    fit_variance_calibrator,
)
from analysis.rheed_to_afm_full_cohort_loo.run import (
    GROWTH_LAYER_GENERATOR_MODES,
    STABLE_GENERATOR_SEED_OFFSETS,
    _condition_with_amplitude,
    _target_series,
    load_config,
    prepare_full_cohort,
)
from analysis.rheed_to_afm_functional_morphology.amplitude import (
    _higher_quantile,
    _range_calibrate,
)
from analysis.rheed_to_afm_functional_morphology.metrics import (
    group_metric_table,
    scan_metric_table,
)
from analysis.rheed_to_afm_functional_morphology.render import render_ensemble
from analysis.rheed_to_afm_functional_morphology.run import _physics_table
from analysis.rheed_to_afm_generation.data import (
    PHYSICS_COLUMNS,
    ConditionScaler,
)
from analysis.rheed_to_afm_generation.run import _load_tables
from analysis.rheed_to_afm_island_generation.evaluate_topology_renderers import (
    _structure_blend,
)
from analysis.rheed_to_afm_island_generation.islands import (
    IslandConditionModel,
    IslandPrimitiveGenerator,
    fit_island_condition_model,
)
from analysis.rheed_to_afm_ood_robust.prediction import (
    DENSITY_WEIGHTED,
    MULTIVIEW_60,
    R3D_TEMPORAL,
    CandidateConfig,
    _expected_error_and_confidence,
    _nested_calibrated_errors,
    _r3d_prediction,
    predict_candidates,
)
from analysis.rheed_to_afm_sharp_generation.cross_validation import (
    _calibrate,
    _generate,
)
from analysis.rheed_to_afm_sharp_generation.spectral import (
    ConditionalSpectralModel,
    fit_conditional_spectral_model,
)
from analysis.rheed_video_afm_story.pretrained_embeddings import (
    load_r3d18,
    preprocess_frames,
)

from .clips import live_physics_row

M22_MODEL_ID = (
    "MorphMBE-M20-SpotConnectivitySq + "
    "M22c-DenseMidGapCompletion-line3-metrology-live-v10"
)
MODEL_ID = M22_MODEL_ID
QUERY_ID = "__live_stream__"

_FROZEN_GENERATOR_KEYS = (
    "resolution",
    "scan_size_nm",
    "analysis_scale_nm",
    "ridge_alpha",
    "morphology_head_weight",
    "pca_dim",
    "hybrid_roughness_embedding_id",
    "hybrid_morphology_embedding_id",
    "variance_cap",
    "minimum_predicted_std",
    "matern_blend_weight",
    "spectral_iaaft_iterations",
    "spectral_ridge_alphas",
    "descriptor_calibration_steps",
    "descriptor_calibration_learning_rate",
    "descriptor_calibration_content_weight",
    "laguerre_count_factor",
    "fine_count_factor",
    "island_ridge_alphas",
    "m10_structure_weight",
    "selected_method",
    "selected_renderer",
    "condition_columns",
)


@dataclass
class TargetReference:
    name: str
    method: str
    log_target: pd.Series
    raw_loo_predictions: np.ndarray
    calibrated_loo_predictions: np.ndarray
    calibrated_loo_errors: np.ndarray
    diagnostics: pd.DataFrame
    tta_centrality_reference: np.ndarray | None = None
    confidence_risk_reference: np.ndarray | None = None
    confidence_error_reference: np.ndarray | None = None


@dataclass
class SpotConnectivityReference:
    groups: list[str]
    raw_features: np.ndarray
    log_residuals: np.ndarray
    neighbors: int = 3
    correction_strength: float = 1.0
    bridge_merge_threshold: float = -2.5
    nonround_fraction_threshold: float = 0.30
    isolated_spot_threshold: float = 0.75
    isolated_interval_blend: float = 0.45


@dataclass
class DeploymentBundle:
    model_id: str
    created_at: str
    groups: list[str]
    physics: pd.DataFrame
    causal_embeddings: pd.DataFrame
    target_config: CandidateConfig
    period_frames_reference: np.ndarray | None
    rq_reference: TargetReference
    fsmi_reference: TargetReference
    condition_scaler: ConditionScaler
    morphology_predictor: Any
    variance_calibrator: VarianceCalibrator
    island_model: IslandConditionModel
    spectral_model: ConditionalSpectralModel
    generation_config: dict[str, Any]
    frozen_parameter_hashes: dict[str, str]
    retrieval_at_inference: bool
    measured_afm_patch_at_inference: bool
    automatic_input_domain: bool = False
    endpoint_streak_reference: np.ndarray | None = None
    endpoint_confidence_risk_reference: np.ndarray | None = None
    endpoint_confidence_error_reference: np.ndarray | None = None
    spot_connectivity_reference: SpotConnectivityReference | None = None


@dataclass(frozen=True)
class ScalarPrediction:
    value: float
    unconstrained_value: float
    support_clipped: bool
    expected_absolute_error: float
    confidence: float
    interval_lower: float
    interval_upper: float
    risk_score: float
    tta_confidence: float
    rotation_period_risk: float
    head_agreement_confidence: float


@dataclass(frozen=True)
class MorphologyPrediction:
    unit_shape: np.ndarray
    height_nm: np.ndarray
    rq: ScalarPrediction
    fsmi: ScalarPrediction
    model_confidence: float
    keyframe_quality: float
    combined_confidence: float
    inference_seconds: float
    model_id: str
    generated_rq_nm: float


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_spot_connectivity_reference(
    *,
    repository: Path,
    model_config: dict[str, Any],
    physics: pd.DataFrame,
    groups: list[str],
) -> SpotConnectivityReference:
    prediction_path = repository / str(
        model_config["external_target_predictions"]["Rq_nm"]
    )
    predictions = pd.read_csv(prediction_path, dtype={"growth_run_id": str}).set_index(
        "growth_run_id"
    )
    rows = predictions.loc[groups]
    required = {
        "true_target",
        "m19_predicted_target_nm",
        "method",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise RuntimeError(f"M20 deployment predictions are incomplete: {missing}")
    if set(rows["method"].astype(str)) != {"M20_spot_connectivity_calibrated_sq"}:
        raise RuntimeError("M22 deployment requires the audited M20 Sq head")
    manifest_path = prediction_path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("query_target_used_for_calibration", True)):
        raise RuntimeError("M20 connectivity calibration used a query target")
    raw_features = _feature_matrix(physics.loc[groups])
    truth = rows["true_target"].to_numpy(float)
    m19 = rows["m19_predicted_target_nm"].to_numpy(float)
    log_residuals = np.log1p(np.maximum(truth, 0.0)) - np.log1p(np.maximum(m19, 0.0))
    return SpotConnectivityReference(
        groups=list(groups),
        raw_features=raw_features,
        log_residuals=log_residuals,
        neighbors=int(manifest["neighbors"]),
        correction_strength=float(manifest["correction_strength"]),
        bridge_merge_threshold=float(manifest["bridge_merge_threshold"]),
        nonround_fraction_threshold=float(manifest["nonround_fraction_threshold"]),
        isolated_spot_threshold=float(manifest["isolated_spot_threshold"]),
        isolated_interval_blend=float(manifest["isolated_interval_blend"]),
    )


def apply_spot_connectivity_upgrade(
    rq: ScalarPrediction,
    endpoint: EndpointPrediction | None,
    query_physics: pd.DataFrame,
    reference: SpotConnectivityReference | None,
) -> tuple[ScalarPrediction, float]:
    """Apply the audited M19/M20 rough-tail and spot-topology rules online."""

    if endpoint is None or reference is None:
        return rq, 0.50
    query_raw = _feature_matrix(query_physics)
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    train_values = scaler.fit_transform(imputer.fit_transform(reference.raw_features))
    query_values = scaler.transform(imputer.transform(query_raw))
    count = min(int(reference.neighbors), len(reference.groups))
    correction_model = KNeighborsRegressor(
        n_neighbors=count,
        weights="distance",
    ).fit(train_values, reference.log_residuals)
    correction = float(correction_model.predict(query_values)[0])
    topology_z = query_values[0]
    isolation = float(
        expit(
            np.mean(
                [
                    topology_z[0],
                    topology_z[1],
                    topology_z[2],
                    -topology_z[3],
                ]
            )
        )
    )

    base = float(rq.value)
    rough_support = bool(
        endpoint.rough_consensus_gate
        or (base < 1.50 and endpoint.streak_expert_nm >= 2.50)
    )
    m19_value = max(base, float(endpoint.streak_expert_nm)) if rough_support else base
    merge_rate = float(query_raw[0, 0])
    round_fraction = float(query_raw[0, 2])
    connectivity_gate = bool(
        rough_support
        and merge_rate <= float(reference.bridge_merge_threshold)
        and (
            base > float(endpoint.streak_expert_nm)
            or round_fraction < float(reference.nonround_fraction_threshold)
        )
    )
    corrected = m19_value
    if connectivity_gate:
        corrected = float(
            np.expm1(
                np.log1p(max(m19_value, 0.0))
                + float(reference.correction_strength) * correction
            )
        )
    radius = max(float(rq.interval_upper) - base, 0.0)
    if rough_support and isolation >= float(reference.isolated_spot_threshold):
        source_upper = m19_value + radius
        corrected += float(reference.isolated_interval_blend) * max(
            source_upper - corrected, 0.0
        )
    corrected = max(float(corrected), 0.0)
    upgraded = ScalarPrediction(
        value=corrected,
        unconstrained_value=corrected,
        support_clipped=False,
        expected_absolute_error=rq.expected_absolute_error,
        confidence=rq.confidence,
        interval_lower=max(corrected - radius, 0.0),
        interval_upper=corrected + radius,
        risk_score=rq.risk_score,
        tta_confidence=rq.tta_confidence,
        rotation_period_risk=rq.rotation_period_risk,
        head_agreement_confidence=rq.head_agreement_confidence,
    )
    return upgraded, isolation


def _embedding_frame(registry: pd.DataFrame, embedding_id: str) -> pd.DataFrame:
    row = registry.loc[registry["embedding_id"] == str(embedding_id)]
    if len(row) != 1:
        raise RuntimeError(f"embedding id is not unique: {embedding_id}")
    payload = np.load(str(row.iloc[0]["path"]), allow_pickle=False)
    ids = [str(value) for value in payload["sample_ids"].tolist()]
    frame = pd.DataFrame(
        np.asarray(payload["embeddings"], dtype=np.float32),
        index=ids,
    )
    frame.index.name = "growth_run_id"
    return frame


def _target_reference(
    *,
    name: str,
    method: str,
    physics: pd.DataFrame,
    embeddings: pd.DataFrame,
    log_target: pd.Series,
    config: CandidateConfig,
) -> TargetReference:
    groups = list(map(str, log_target.index))
    raw_rows: list[float] = []
    diagnostics: list[dict[str, float]] = []
    for held in groups:
        fit = [group for group in groups if group != held]
        predictions, diagnostic = predict_candidates(
            physics=physics,
            embeddings=embeddings,
            log_target=log_target,
            fit_groups=fit,
            query_group=held,
            config=config,
        )
        raw_rows.append(float(np.exp(predictions[method])))
        diagnostics.append(diagnostic)
    raw = np.asarray(raw_rows, dtype=float)
    truth = np.exp(log_target.loc[groups].to_numpy(float))
    calibrated, errors = _nested_calibrated_errors(raw, truth)
    return TargetReference(
        name=name,
        method=method,
        log_target=log_target.copy(),
        raw_loo_predictions=raw,
        calibrated_loo_predictions=calibrated,
        calibrated_loo_errors=errors,
        diagnostics=pd.DataFrame(diagnostics, index=groups),
    )


def _resolve_freeze_file(
    configured_root: str | Path,
    relative: str,
    fallback_root: Path,
) -> Path:
    configured = Path(configured_root).expanduser() / relative
    if configured.exists():
        return configured
    fallback = fallback_root / relative
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"frozen model metadata is unavailable: {configured} or {fallback}"
    )


def build_deployment_bundle(
    config: dict[str, Any],
    *,
    progress: Callable[[str], None] | None = None,
) -> DeploymentBundle:
    """Refit the configured audited deployment on the allowed cohort."""

    log = progress or (lambda _: None)
    repository = Path(config.get("repository_root", ".")).resolve()
    freeze_fallback = repository / str(config["repository_freeze_root"])
    m12_parameters = _resolve_freeze_file(
        config["standalone_root"],
        str(config["standalone_m12_parameters"]),
        freeze_fallback,
    )
    m14_parameters = _resolve_freeze_file(
        config["standalone_root"],
        str(config["standalone_m14_parameters"]),
        freeze_fallback,
    )
    m14_generator_parameters = _resolve_freeze_file(
        config["standalone_root"],
        str(config["standalone_m14_generator_parameters"]),
        freeze_fallback,
    )
    generation = json.loads(m12_parameters.read_text(encoding="utf-8"))
    target_parameters = json.loads(m14_parameters.read_text(encoding="utf-8"))
    if generation["selected_method"] != "M12a_edge_preserving_terrace":
        raise RuntimeError("frozen generator is not M12a")
    if target_parameters["generation_prediction_methods"] != {
        "Rq_nm": MULTIVIEW_60,
        "FSMI_nm": DENSITY_WEIGHTED,
    }:
        raise RuntimeError("frozen M14i target-head mapping changed")

    model_config_path = repository / str(config["generation_config"])
    model_config = load_config(model_config_path)
    frozen_generator_config = json.loads(
        m14_generator_parameters.read_text(encoding="utf-8")
    )
    endpoint_upgrade = bool(config.get("endpoint_rq_upgrade", False))
    if bool(config.get("metrology_audited_mode", False)):
        upgrade_keys = {"selected_method", "selected_renderer"}
        mismatched = [
            key
            for key in _FROZEN_GENERATOR_KEYS
            if model_config.get(key) != frozen_generator_config.get(key)
            and not (endpoint_upgrade and key in upgrade_keys)
        ]
        if mismatched:
            raise RuntimeError(
                "the metrology repair changed frozen M12a architecture or "
                f"hyperparameters: {mismatched}"
            )
        metrology_variant = str(model_config.get("afm_target_variant", ""))
        phase1_path = str(model_config.get("phase1_manifest", ""))
        line3_phase1 = (
            "afm_metrology_line3_v1" in phase1_path
            or "line3_scanline_flatten" in metrology_variant
        )
        if (
            "line3" not in str(model_config.get("afm_descriptors", ""))
            or not line3_phase1
        ):
            raise RuntimeError(
                "metrology-audited deployment must use the line-3 AFM "
                "descriptor and manifest paths"
            )
        selected_generator = str(model_config.get("selected_method", ""))
        selected_mode = str(model_config.get("selected_renderer", {}).get("mode", ""))
        audited_endpoint_generators = {
            "M22c_gap_completion_strong": ("regime_adaptive_separated_islands")
        }
        if (
            endpoint_upgrade
            and audited_endpoint_generators.get(selected_generator) != selected_mode
        ):
            raise RuntimeError(
                "the release deployment requires the audited M22c renderer/mode pair"
            )
    elif _hash_file(model_config_path) != _hash_file(m14_generator_parameters):
        raise RuntimeError(
            "deployment generation config differs from the frozen M14i "
            "generator parameters"
        )
    log("Loading the configured training cohort and AFM descriptors")
    tables = _load_tables(model_config)
    descriptors, _ = prepare_full_cohort(tables, model_config)
    scan_metrics = scan_metric_table(
        descriptors,
        scan_size_nm=float(model_config["scan_size_nm"]),
        analysis_scale_nm=float(model_config["analysis_scale_nm"]),
    )
    group_metrics = group_metric_table(scan_metrics)
    log_rq, log_fsmi = _target_series(descriptors, group_metrics)
    groups = list(map(str, log_rq.index))
    physics = _physics_table(tables["physics"]).loc[groups].copy()
    causal_embeddings = _embedding_frame(
        tables["registry"],
        str(target_parameters["temporal_embedding_id"]),
    ).loc[groups]
    target_config = CandidateConfig(
        density_strength=float(target_parameters["density_strength"]),
        density_floor=float(target_parameters["density_floor"]),
        residual_strength=float(target_parameters["residual_strength"]),
        residual_floor=float(target_parameters["residual_floor"]),
        r3d_pca_components=int(target_parameters["r3d_pca_components"]),
        ridge_alpha=float(target_parameters["robust_ridge_alpha"]),
        baseline_alpha=float(target_parameters["baseline_ridge_alpha"]),
        morphology_weight=float(target_parameters["baseline_morphology_weight"]),
    )
    automatic_domain = bool(config.get("automatic_target_physics"))
    target_physics = physics
    target_embeddings = causal_embeddings
    rq_method = MULTIVIEW_60
    fsmi_method = DENSITY_WEIGHTED
    period_frames_reference: np.ndarray | None = None
    if automatic_domain:
        target_physics = _physics_table(
            pd.read_csv(
                repository / str(config["automatic_target_physics"]),
                dtype={"sample_id": str, "growth_run_id": str},
            )
        ).loc[groups]
        target_registry = pd.read_csv(
            repository / str(config["automatic_target_embedding_registry"])
        )
        target_embeddings = _embedding_frame(
            target_registry,
            str(target_parameters["temporal_embedding_id"]),
        ).loc[groups]
        rq_method = str(config.get("automatic_rq_method", R3D_TEMPORAL))
        fsmi_method = str(config.get("automatic_fsmi_method", R3D_TEMPORAL))
        if (rq_method, fsmi_method) != (R3D_TEMPORAL, R3D_TEMPORAL):
            raise RuntimeError(
                "automatic-domain continuation requires the nested-selected "
                "causal R3D head for both targets"
            )
        selection = (
            pd.read_csv(
                repository / str(config["automatic_selection_table"]),
                dtype={"growth_run_id": str},
            )
            .set_index("growth_run_id")
            .loc[groups]
        )
        period_frames_reference = selection["estimated_period_frames"].to_numpy(float)
        if not np.isfinite(period_frames_reference).all():
            raise RuntimeError("automatic deployment requires finite rotation periods")

    log(
        "Fitting automatic-input Sq/FSMI deployment heads and strict-LOO "
        "confidence references"
    )
    rq_reference = _target_reference(
        name="Rq_nm",
        method=rq_method,
        physics=target_physics,
        embeddings=target_embeddings,
        log_target=log_rq,
        config=target_config,
    )
    fsmi_reference = _target_reference(
        name="FSMI_nm",
        method=fsmi_method,
        physics=target_physics,
        embeddings=target_embeddings,
        log_target=log_fsmi,
        config=target_config,
    )
    if automatic_domain:
        stability = pd.read_csv(
            repository / str(config["stability_predictions"]),
            dtype={"growth_run_id": str},
        )
        for reference in (rq_reference, fsmi_reference):
            rows = (
                stability.loc[stability["target"] == reference.name]
                .set_index("growth_run_id")
                .loc[groups]
            )
            if not np.allclose(
                rows["true_target"].to_numpy(float),
                np.exp(reference.log_target.loc[groups].to_numpy(float)),
                rtol=1e-6,
                atol=1e-8,
            ):
                raise RuntimeError(
                    f"{reference.name} confidence targets do not match the "
                    "metrology-audited training targets"
                )
            reference.tta_centrality_reference = rows["base_to_tta_median_nm"].to_numpy(
                float
            )
            reference.confidence_risk_reference = rows[
                "uncertainty_risk_score"
            ].to_numpy(float)
            reference.confidence_error_reference = rows["absolute_error"].to_numpy(
                float
            )

    endpoint_streak_reference: np.ndarray | None = None
    endpoint_confidence_risk_reference: np.ndarray | None = None
    endpoint_confidence_error_reference: np.ndarray | None = None
    if endpoint_upgrade:
        streak_table = pd.read_csv(
            repository / str(config["endpoint_streak_features"]),
            dtype={"growth_run_id": str},
        ).set_index("growth_run_id")
        endpoint_predictions = pd.read_csv(
            repository / str(config["endpoint_strict_predictions"]),
            dtype={"growth_run_id": str},
        )
        endpoint_predictions = (
            endpoint_predictions.loc[
                (endpoint_predictions["target"] == "Rq_nm")
                & (
                    endpoint_predictions["method"]
                    == "M16_endpoint_streak_dual_resolution"
                )
            ]
            .set_index("growth_run_id")
            .loc[groups]
        )
        endpoint_truth = np.exp(rq_reference.log_target.loc[groups].to_numpy(float))
        if not np.allclose(
            endpoint_predictions["true_target"].to_numpy(float),
            endpoint_truth,
            rtol=1e-6,
            atol=1e-8,
        ):
            raise RuntimeError(
                "M16 endpoint confidence targets do not match the "
                "metrology-audited deployment targets"
            )
        endpoint_streak_reference = streak_table.loc[
            groups, PRIMARY_STREAK_FEATURE
        ].to_numpy(float)
        endpoint_confidence_risk_reference = endpoint_predictions[
            "uncertainty_risk_score"
        ].to_numpy(float)
        endpoint_confidence_error_reference = endpoint_predictions[
            "absolute_error"
        ].to_numpy(float)

    condition_columns = list(model_config["condition_columns"])
    condition_scaler = ConditionScaler.fit(descriptors, condition_columns, set(groups))
    group_targets = _group_targets(descriptors, condition_columns)
    fit_predictor = _predictor_factory(
        config=model_config,
        tables=tables,
        group_targets=group_targets,
    )
    log("Fitting the R3D-18 temporal morphology-conditioning head")
    morphology_predictor = fit_predictor(groups, condition_scaler)
    variance_calibrator, _ = fit_variance_calibrator(
        groups=groups,
        group_targets=group_targets,
        enclosing_scaler=condition_scaler,
        fit_predictor=fit_predictor,
        registry=tables["registry"],
        physics=tables["physics"],
        cap=float(model_config["variance_cap"]),
        minimum_predicted_std=float(model_config["minimum_predicted_std"]),
    )
    log("Fitting the island-statistics model and non-retrieval spectral prior")
    island_model, _, _ = fit_island_condition_model(
        train_rows=descriptors,
        condition_scaler=condition_scaler,
        resolution=int(model_config["resolution"]),
        alphas=model_config["island_ridge_alphas"],
    )
    spectral_model, _, _ = fit_conditional_spectral_model(
        train_rows=descriptors,
        condition_scaler=condition_scaler,
        alphas=model_config["spectral_ridge_alphas"],
        resolution=int(model_config["resolution"]),
        removelist_sample_ids=tables["removelist"].sample_ids,
    )
    selected_generator = str(model_config.get("selected_method", ""))
    spot_connectivity_reference = None
    if selected_generator == "M22c_gap_completion_strong":
        log("Fitting the full-cohort M20 spot-connectivity deployment head")
        spot_connectivity_reference = _build_spot_connectivity_reference(
            repository=repository,
            model_config=model_config,
            physics=target_physics,
            groups=groups,
        )
    if selected_generator != "M22c_gap_completion_strong":
        raise RuntimeError(
            "the release deployment supports only M22c_gap_completion_strong"
        )
    connectivity_prediction_path = (
        repository / str(model_config["external_target_predictions"]["Rq_nm"])
        if spot_connectivity_reference is not None
        else None
    )
    return DeploymentBundle(
        model_id=M22_MODEL_ID,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        groups=groups,
        physics=target_physics,
        causal_embeddings=target_embeddings,
        target_config=target_config,
        period_frames_reference=period_frames_reference,
        rq_reference=rq_reference,
        fsmi_reference=fsmi_reference,
        condition_scaler=condition_scaler,
        morphology_predictor=morphology_predictor,
        variance_calibrator=variance_calibrator,
        island_model=island_model,
        spectral_model=spectral_model,
        generation_config=model_config,
        frozen_parameter_hashes={
            "m12_parameters": _hash_file(m12_parameters),
            "m14_parameters": _hash_file(m14_parameters),
            "m14_generator_parameters": _hash_file(m14_generator_parameters),
            "deployment_config": _hash_file(model_config_path),
            "metrology_manifest": _hash_file(
                repository / str(model_config["phase1_manifest"])
            ),
            "afm_descriptors": _hash_file(
                repository / str(model_config["afm_descriptors"])
            ),
            "confidence_reference": _hash_file(
                repository / str(config["stability_predictions"])
            ),
            **(
                {
                    "endpoint_streak_features": _hash_file(
                        repository / str(config["endpoint_streak_features"])
                    ),
                    "endpoint_strict_predictions": _hash_file(
                        repository / str(config["endpoint_strict_predictions"])
                    ),
                }
                if endpoint_upgrade
                else {}
            ),
            **(
                {
                    "m20_connectivity_predictions": _hash_file(
                        connectivity_prediction_path
                    ),
                    "m20_connectivity_manifest": _hash_file(
                        connectivity_prediction_path.with_name("manifest.json")
                    ),
                }
                if connectivity_prediction_path is not None
                else {}
            ),
        },
        retrieval_at_inference=False,
        measured_afm_patch_at_inference=False,
        automatic_input_domain=automatic_domain,
        endpoint_streak_reference=endpoint_streak_reference,
        endpoint_confidence_risk_reference=(endpoint_confidence_risk_reference),
        endpoint_confidence_error_reference=(endpoint_confidence_error_reference),
        spot_connectivity_reference=spot_connectivity_reference,
    )


def save_deployment_bundle(
    bundle: DeploymentBundle,
    path: str | Path,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination, compress=3)
    return destination


def load_deployment_bundle(path: str | Path) -> DeploymentBundle:
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, DeploymentBundle) or bundle.model_id != M22_MODEL_ID:
        raise RuntimeError("deployment bundle identity does not match")
    return bundle


def _select_device(name: str) -> torch.device:
    if name == "auto":
        # R3D-18 is slightly faster and bitwise identical to the frozen
        # embeddings on CPU on the target M1 Pro.
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


class RealtimeMorphologyPredictor:
    """Generate a new AFM height field from one detected RHEED event."""

    def __init__(
        self,
        bundle: DeploymentBundle,
        *,
        device: str = "auto",
    ) -> None:
        self.bundle = bundle
        self.device = _select_device(device)
        model, status = load_r3d18()
        if model is None or not status.loaded:
            raise RuntimeError(f"R3D-18 could not be loaded: {status.reason}")
        self.r3d = model.to(self.device).eval()

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
    ) -> RealtimeMorphologyPredictor:
        return cls(load_deployment_bundle(path), device=device)

    @torch.inference_mode()
    def _embedding(self, frames: np.ndarray) -> np.ndarray:
        tensor = preprocess_frames(
            np.asarray(frames, dtype=np.uint8),
            "raw_luminance",
            video=True,
        ).to(self.device)
        result = self.r3d(tensor).detach().cpu().numpy()[0]
        return np.asarray(result, dtype=np.float32)

    def _scalar(
        self,
        reference: TargetReference,
        *,
        physics: pd.DataFrame,
        embeddings: pd.DataFrame,
        tta_embeddings: np.ndarray | None = None,
        estimated_period_frames: float | None = None,
    ) -> ScalarPrediction:
        truth = np.exp(reference.log_target.loc[self.bundle.groups].to_numpy(float))
        if (
            reference.method == R3D_TEMPORAL
            and tta_embeddings is not None
            and reference.tta_centrality_reference is not None
        ):
            calibrated_views = []
            for view_index, embedding in enumerate(
                np.asarray(tta_embeddings, dtype=np.float32)
            ):
                query = f"{QUERY_ID}_tta_{view_index}"
                view_frame = pd.concat(
                    [
                        self.bundle.causal_embeddings,
                        pd.DataFrame(
                            embedding[None],
                            index=[query],
                        ),
                    ]
                )
                raw_log = _r3d_prediction(
                    embeddings=view_frame,
                    log_target=reference.log_target,
                    fit_groups=self.bundle.groups,
                    query_group=query,
                    alpha=self.bundle.target_config.ridge_alpha,
                    components=(self.bundle.target_config.r3d_pca_components),
                )
                calibrated, _ = _range_calibrate(
                    reference.raw_loo_predictions,
                    truth,
                    float(np.exp(raw_log)),
                )
                calibrated_views.append(calibrated)
            view_values = np.asarray(calibrated_views, dtype=float)
            unconstrained_value = float(view_values[0])
            centrality = float(abs(unconstrained_value - np.median(view_values)))
            centrality_reference = np.asarray(
                reference.tta_centrality_reference,
                dtype=float,
            )
            centrality_risk = _empirical_risk(
                centrality,
                centrality_reference,
            )
            period_reference = self.bundle.period_frames_reference
            rotation_period_risk = 0.0
            angular_risk = centrality_risk
            if (
                period_reference is not None
                and estimated_period_frames is not None
                and np.isfinite(estimated_period_frames)
            ):
                rotation_period_risk = _empirical_risk(
                    float(estimated_period_frames),
                    np.asarray(period_reference, dtype=float),
                )
                angular_risk = float(
                    angular_coverage_risk(
                        centrality_risk,
                        rotation_period_risk,
                    ).item()
                )
            tta_confidence = float(np.clip(1.0 - angular_risk, 0.0, 1.0))
            _, query_diagnostics = predict_candidates(
                physics=physics,
                embeddings=embeddings,
                log_target=reference.log_target,
                fit_groups=self.bundle.groups,
                query_group=QUERY_ID,
                config=self.bundle.target_config,
            )
            head_reference = reference.diagnostics[
                "core_head_disagreement_log_std"
            ].to_numpy(float)
            head_risk = _empirical_risk(
                float(query_diagnostics["core_head_disagreement_log_std"]),
                head_reference,
            )
            head_confidence = float(np.clip(1.0 - head_risk, 0.0, 1.0))
            amplitude_risk = _empirical_risk(
                unconstrained_value,
                reference.calibrated_loo_predictions,
            )
            range_aware_risk = float(0.75 * amplitude_risk + 0.25 * angular_risk)
            combined, _veto = combine_tta_and_head_confidence(
                1.0 - range_aware_risk,
                head_confidence,
                extreme_quantile=0.10,
            )
            confidence = float(combined.item())
            composite_risk = 1.0 - confidence
            reference_risk = reference.confidence_risk_reference
            reference_errors = reference.confidence_error_reference
            if reference_risk is None or reference_errors is None:
                raise RuntimeError(
                    f"{reference.name} lacks the audited M15b confidence reference"
                )
            expected_error = _isotonic_expected_error(
                reference_risk,
                reference_errors,
                composite_risk,
            )
            adaptive = reference_errors / np.maximum(
                0.50 + reference_risk,
                0.25,
            )
            radius = _higher_quantile(adaptive, 0.90) * (0.50 + composite_risk)
            support_lower = float(np.min(truth))
            support_upper = float(np.max(truth))
            value = float(
                np.clip(
                    unconstrained_value,
                    support_lower,
                    support_upper,
                )
            )
            support_clipped = not np.isclose(
                value,
                unconstrained_value,
                rtol=1e-8,
                atol=1e-8,
            )
            if support_clipped:
                confidence *= 0.5
            return ScalarPrediction(
                value=value,
                unconstrained_value=unconstrained_value,
                support_clipped=bool(support_clipped),
                expected_absolute_error=float(expected_error),
                confidence=float(confidence),
                interval_lower=float(max(value - radius, 0.0)),
                interval_upper=float(value + radius),
                risk_score=float(composite_risk),
                tta_confidence=tta_confidence,
                rotation_period_risk=float(rotation_period_risk),
                head_agreement_confidence=head_confidence,
            )

        predicted, diagnostics = predict_candidates(
            physics=physics,
            embeddings=embeddings,
            log_target=reference.log_target,
            fit_groups=self.bundle.groups,
            query_group=QUERY_ID,
            config=self.bundle.target_config,
        )
        raw = float(np.exp(predicted[reference.method]))
        unconstrained_value, _ = _range_calibrate(
            reference.raw_loo_predictions,
            truth,
            raw,
        )
        support_lower = float(np.min(truth))
        support_upper = float(np.max(truth))
        value = float(np.clip(unconstrained_value, support_lower, support_upper))
        support_clipped = not np.isclose(
            value, unconstrained_value, rtol=1e-8, atol=1e-8
        )
        expected_error, confidence, risk, inner_risk = _expected_error_and_confidence(
            reference.diagnostics,
            reference.calibrated_loo_errors,
            reference.calibrated_loo_predictions,
            diagnostics,
            unconstrained_value,
        )
        if support_clipped:
            confidence *= 0.5
        adaptive = reference.calibrated_loo_errors / (1.0 + inner_risk)
        radius = _higher_quantile(adaptive, 0.90) * (1.0 + risk)
        return ScalarPrediction(
            value=float(value),
            unconstrained_value=float(unconstrained_value),
            support_clipped=bool(support_clipped),
            expected_absolute_error=float(expected_error),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            interval_lower=float(max(value - radius, 0.0)),
            interval_upper=float(value + radius),
            risk_score=float(risk),
            tta_confidence=float(np.clip(confidence, 0.0, 1.0)),
            rotation_period_risk=0.0,
            head_agreement_confidence=float(np.clip(confidence, 0.0, 1.0)),
        )

    def _endpoint_rq(
        self,
        *,
        query_embedding: np.ndarray,
        query_streak: float,
        legacy: ScalarPrediction,
    ) -> tuple[ScalarPrediction, EndpointPrediction | None]:
        """Apply M16 to Sq while retaining target-blind angular diagnostics."""

        # ``getattr`` preserves read compatibility with pre-M16 joblib bundles.
        # Pickle does not synthesize newly added dataclass attributes when an
        # older instance is loaded against the current class definition.
        streak_reference = getattr(self.bundle, "endpoint_streak_reference", None)
        risk_reference = getattr(
            self.bundle, "endpoint_confidence_risk_reference", None
        )
        error_reference = getattr(
            self.bundle, "endpoint_confidence_error_reference", None
        )
        if (
            streak_reference is None
            or risk_reference is None
            or error_reference is None
        ):
            return legacy, None
        truth = np.exp(
            self.bundle.rq_reference.log_target.loc[self.bundle.groups].to_numpy(float)
        )
        embeddings = self.bundle.causal_embeddings.loc[self.bundle.groups].to_numpy(
            float
        )
        endpoint = predict_endpoint(
            embeddings=embeddings,
            streak=np.asarray(streak_reference, dtype=float),
            target_nm=truth,
            train=np.arange(len(self.bundle.groups)),
            query_embedding=np.asarray(query_embedding, dtype=float),
            query_streak=float(query_streak),
        )
        risk = _support_risk(
            endpoint,
            training_targets=truth,
            legacy_risk=float(legacy.risk_score),
        )
        expected_error = _isotonic_expected_error(
            np.asarray(risk_reference, dtype=float),
            np.asarray(error_reference, dtype=float),
            risk,
        )
        adaptive = np.asarray(error_reference, dtype=float) / np.maximum(
            0.50 + np.asarray(risk_reference, dtype=float), 0.25
        )
        radius = _higher_quantile(adaptive, 0.90) * (0.50 + risk)
        unconstrained = float(endpoint.value_nm)
        value = float(
            np.clip(unconstrained, float(np.min(truth)), float(np.max(truth)))
        )
        support_clipped = not np.isclose(value, unconstrained, rtol=1e-8, atol=1e-8)
        confidence = float(np.clip(1.0 - risk, 0.0, 1.0))
        if support_clipped:
            confidence *= 0.5
        return (
            ScalarPrediction(
                value=value,
                unconstrained_value=unconstrained,
                support_clipped=bool(support_clipped),
                expected_absolute_error=float(expected_error),
                confidence=confidence,
                interval_lower=float(max(value - radius, 0.0)),
                interval_upper=float(value + radius),
                risk_score=float(risk),
                tta_confidence=legacy.tta_confidence,
                rotation_period_risk=legacy.rotation_period_risk,
                head_agreement_confidence=(legacy.head_agreement_confidence),
            ),
            endpoint,
        )

    def predict(
        self,
        selected_16: np.ndarray,
        *,
        physics_selected_16: np.ndarray | None = None,
        causal_view_names: list[str] | None = None,
        causal_8_views: np.ndarray | None = None,
        estimated_period_frames: float | None = None,
        keyframe_quality: float,
        seed: int,
    ) -> MorphologyPrediction:
        started = time.perf_counter()
        frames = np.asarray(selected_16, dtype=np.uint8)
        if frames.shape != (16, 224, 224):
            raise ValueError(f"model input must be [16,224,224], got {frames.shape}")
        tta_embeddings: np.ndarray | None = None
        if causal_8_views is not None:
            causal_views = np.asarray(causal_8_views, dtype=np.uint8)
            names = list(causal_view_names or [])
            if causal_views.ndim != 4 or causal_views.shape[1:] != (
                8,
                224,
                224,
            ):
                raise ValueError("causal TTA input must be [V,8,224,224]")
            if len(names) != len(causal_views) or "base" not in names:
                raise ValueError("causal TTA view names must identify one base view")
            ordered = [names.index("base")] + [
                index for index, name in enumerate(names) if name != "base"
            ]
            tta_embeddings = np.stack(
                [self._embedding(causal_views[index]) for index in ordered]
            ).astype(np.float32)
            causal_embedding = tta_embeddings[0]
        else:
            causal_embedding = self._embedding(frames[:8])
        selected_embedding = self._embedding(frames)
        physics_frames = (
            frames
            if physics_selected_16 is None
            else np.asarray(physics_selected_16, dtype=np.uint8)
        )
        if physics_frames.shape != (16, 224, 224):
            raise ValueError(
                f"physics input must be [16,224,224], got {physics_frames.shape}"
            )
        query_physics = live_physics_row(
            physics_frames,
            sample_id=QUERY_ID,
        )
        physics = pd.concat([self.bundle.physics, query_physics], axis=0)
        embeddings = pd.concat(
            [
                self.bundle.causal_embeddings,
                pd.DataFrame(
                    causal_embedding[None],
                    index=[QUERY_ID],
                ),
            ],
            axis=0,
        )
        legacy_rq = self._scalar(
            self.bundle.rq_reference,
            physics=physics,
            embeddings=embeddings,
            tta_embeddings=tta_embeddings,
            estimated_period_frames=estimated_period_frames,
        )
        streak_features = extract_streak_features(frames[:8])
        rq, endpoint = self._endpoint_rq(
            query_embedding=causal_embedding,
            query_streak=streak_features[PRIMARY_STREAK_FEATURE],
            legacy=legacy_rq,
        )
        rq, predicted_isolation = apply_spot_connectivity_upgrade(
            rq,
            endpoint,
            query_physics,
            getattr(self.bundle, "spot_connectivity_reference", None),
        )
        fsmi = self._scalar(
            self.bundle.fsmi_reference,
            physics=physics,
            embeddings=embeddings,
            tta_embeddings=tta_embeddings,
            estimated_period_frames=estimated_period_frames,
        )

        morphology = self.bundle.morphology_predictor.morphology_predictor
        _, raw_z = morphology.predict(
            selected_embedding[None],
            query_physics[PHYSICS_COLUMNS].to_numpy(np.float32),
        )
        raw_z = np.asarray(raw_z[0], dtype=np.float32)
        selected_z = self.bundle.variance_calibrator.transform_z(raw_z)
        condition_z = _condition_with_amplitude(
            selected_z,
            self.bundle.condition_scaler,
            rq.value,
        )
        config = self.bundle.generation_config
        generator = IslandPrimitiveGenerator(
            resolution=int(config["resolution"]),
            laguerre_count_factor=float(config["laguerre_count_factor"]),
            fine_count_factor=float(config["fine_count_factor"]),
        )
        matern = DescriptorMaternGenerator(
            self.bundle.condition_scaler,
            resolution=int(config["resolution"]),
        ).generate_ensemble(selected_z, draws=1, seed=int(seed))
        spectral_raw = _generate(
            self.bundle.spectral_model,
            raw_z,
            draws=1,
            iterations=int(config["spectral_iaaft_iterations"]),
            seed=int(seed) + 700_000,
        )
        spectral = _calibrate(
            spectral_raw,
            raw_z,
            scaler=self.bundle.condition_scaler,
            config=config,
            device=torch.device("cpu"),
        )
        prior = _blend_ensembles(
            matern,
            spectral,
            primary_weight=float(config["matern_blend_weight"]),
        )
        island_target = dict(self.bundle.island_model.predict(condition_z))
        island_target["rheed_spot_isolation_score"] = predicted_isolation
        island_target["conditioning_sq_nm"] = rq.value
        baseline_structure = generator.generate_ensemble(
            island_target,
            draws=1,
            seed=int(seed) + 300_000,
            mode="laguerre",
        )
        # Retain this intermediate construction because it is part of the
        # frozen M12a provenance even though the final renderer uses the same
        # structure and prior directly.
        _structure_blend(
            baseline_structure,
            prior,
            weight=float(config["m10_structure_weight"]),
        )
        renderer = dict(config.get("candidate_renderer_defaults", {}))
        renderer.update(config["selected_renderer"])
        generator_mode = str(renderer.pop("island_generator_mode", "laguerre"))
        if generator_mode in GROWTH_LAYER_GENERATOR_MODES and rq.value >= 7.6:
            generator_mode = "separated_ellipse_strict_sparse"
        if generator_mode == "laguerre":
            structure = baseline_structure
        else:
            structure = generator.generate_ensemble(
                island_target,
                draws=1,
                seed=(
                    int(seed)
                    + 300_000
                    + STABLE_GENERATOR_SEED_OFFSETS.get(generator_mode, 100_000)
                ),
                mode=generator_mode,
            )
        unit_shape = render_ensemble(
            structure,
            prior,
            baseline_structure=baseline_structure,
            conditioning_sq_nm=rq.value,
            island_target=island_target,
            rough_isolation_score=predicted_isolation,
            **renderer,
        )[0].astype(np.float32)
        height_nm = (unit_shape * float(rq.value)).astype(np.float32)
        generated_rq = float(np.sqrt(np.mean(np.square(height_nm - height_nm.mean()))))
        model_confidence = float(
            np.sqrt(max(rq.confidence, 0.0) * max(fsmi.confidence, 0.0))
        )
        input_quality = float(np.clip(keyframe_quality, 0.0, 1.0))
        combined = float(
            np.cbrt(
                max(rq.confidence, 1e-8)
                * max(fsmi.confidence, 1e-8)
                * max(input_quality, 1e-8)
            )
        )
        return MorphologyPrediction(
            unit_shape=unit_shape,
            height_nm=height_nm,
            rq=rq,
            fsmi=fsmi,
            model_confidence=model_confidence,
            keyframe_quality=input_quality,
            combined_confidence=combined,
            inference_seconds=float(time.perf_counter() - started),
            model_id=self.bundle.model_id,
            generated_rq_nm=generated_rq,
        )
