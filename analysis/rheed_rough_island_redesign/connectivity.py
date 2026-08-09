"""Target-blind RHEED spot-connectivity correction for the M19 Sq tail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler

MODEL_NAME = "M20_spot_connectivity_calibrated_sq"
CONNECTIVITY_FEATURES = (
    "selected_16__component_merge_rate_p97_to_p90_median",
    "selected_16__component_count_p90_median",
    "selected_16__round_component_fraction_p90_median",
    "selected_16__component_area_q90_p90_median",
)


def _feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(CONNECTIVITY_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"RHEED connectivity features missing: {missing}")
    values = frame.loc[:, CONNECTIVITY_FEATURES].to_numpy(float, copy=True)
    # The largest low-threshold component spans several orders of magnitude.
    values[:, 3] = np.log1p(np.maximum(values[:, 3], 0.0) * 1e4)
    return values


def _metric_record(
    frame: pd.DataFrame, *, label: str, mask: np.ndarray
) -> dict[str, float | int | str]:
    subset = frame.loc[mask]
    residual = (
        subset["predicted_target"].to_numpy(float)
        - subset["true_target"].to_numpy(float)
    )
    return {
        "stratum": label,
        "count": len(subset),
        "mae_nm": float(np.mean(np.abs(residual))),
        "rmse_nm": float(np.sqrt(np.mean(np.square(residual)))),
        "bias_nm": float(np.mean(residual)),
    }


def crossfit_spot_connectivity_calibration(
    predictions: pd.DataFrame,
    physics: pd.DataFrame,
    *,
    neighbors: int = 3,
    correction_strength: float = 1.0,
    bridge_merge_threshold: float = -2.5,
    nonround_fraction_threshold: float = 0.30,
    isolated_spot_threshold: float = 0.75,
    isolated_interval_blend: float = 0.45,
) -> pd.DataFrame:
    """Correct M19 rough-tail residuals using held-target-blind spot topology.

    Every query correction is fitted on the other growth groups.  The gate is
    intentionally conservative: M19 rough support must already be active, the
    spot cores must merge strongly as the threshold is relaxed, and either the
    temporal expert must be rougher than the streak expert or the low-threshold
    components must be distinctly non-round.  At the opposite extreme, a
    highly isolated spot pattern is allowed to use a fixed fraction of M19's
    target-blind upper uncertainty headroom.  This prevents the high-roughness
    tail from being compressed while retaining the original uncertainty bound
    as the only source of uplift magnitude.
    """

    required = {
        "growth_run_id",
        "true_target",
        "predicted_target",
        "base_endpoint_prediction_nm",
        "streak_expert_nm",
        "rough_tail_rescue_activated",
        "interval_radius",
        "interval_upper",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"M19 predictions missing columns: {missing}")
    if predictions["growth_run_id"].astype(str).duplicated().any():
        raise ValueError("M19 predictions contain duplicate growth groups")
    if physics["growth_run_id"].astype(str).duplicated().any():
        raise ValueError("RHEED physics contain duplicate growth groups")
    frame = predictions.copy()
    frame["growth_run_id"] = frame["growth_run_id"].astype(str)
    physical = physics.copy()
    physical["growth_run_id"] = physical["growth_run_id"].astype(str)
    frame = frame.merge(
        physical.loc[:, ["growth_run_id", *CONNECTIVITY_FEATURES]],
        on="growth_run_id",
        how="left",
        validate="one_to_one",
    )
    features = _feature_matrix(frame)
    truth = frame["true_target"].to_numpy(float)
    base = frame["predicted_target"].to_numpy(float)
    log_base = np.log1p(np.maximum(base, 0.0))
    log_residual = np.log1p(np.maximum(truth, 0.0)) - log_base
    merge_rate = frame[CONNECTIVITY_FEATURES[0]].to_numpy(float)
    round_fraction = frame[CONNECTIVITY_FEATURES[2]].to_numpy(float)
    temporal_rougher = (
        frame["base_endpoint_prediction_nm"].to_numpy(float)
        > frame["streak_expert_nm"].to_numpy(float)
    )
    rough_support = frame["rough_tail_rescue_activated"].astype(bool).to_numpy()
    gate = (
        rough_support
        & (merge_rate <= float(bridge_merge_threshold))
        & (
            temporal_rougher
            | (round_fraction < float(nonround_fraction_threshold))
        )
    )

    corrected = base.copy()
    corrections = np.zeros(len(frame), dtype=float)
    isolation_scores = np.zeros(len(frame), dtype=float)
    neighbor_ids: list[str] = []
    neighbor_distances: list[str] = []
    for query in range(len(frame)):
        train = np.arange(len(frame)) != query
        imputer = SimpleImputer(strategy="median")
        scaler = RobustScaler()
        train_values = scaler.fit_transform(
            imputer.fit_transform(features[train])
        )
        query_values = scaler.transform(
            imputer.transform(features[query : query + 1])
        )
        count = min(int(neighbors), int(np.sum(train)))
        model = KNeighborsRegressor(
            n_neighbors=count,
            weights="distance",
        ).fit(train_values, log_residual[train])
        correction = float(model.predict(query_values)[0])
        corrections[query] = correction
        # Positive values consistently mean more numerous, rounder and more
        # persistent isolated spots; large low-threshold components reverse it.
        topology_z = query_values[0]
        isolation_scores[query] = float(
            expit(np.mean([topology_z[0], topology_z[1], topology_z[2], -topology_z[3]]))
        )
        distances = np.sqrt(np.sum(np.square(train_values - query_values), axis=1))
        order = np.argsort(distances)[:count]
        train_indices = np.flatnonzero(train)[order]
        neighbor_ids.append("|".join(frame.iloc[train_indices]["growth_run_id"]))
        neighbor_distances.append(
            "|".join(f"{distances[index]:.6g}" for index in order)
        )
        if gate[query]:
            corrected[query] = float(
                np.expm1(
                    log_base[query]
                    + float(correction_strength) * correction
                )
            )

    isolated_spot_gate = rough_support & (
        isolation_scores >= float(isolated_spot_threshold)
    )
    source_upper = frame["interval_upper"].to_numpy(float)
    isolated_headroom = np.maximum(source_upper - corrected, 0.0)
    isolated_uplift = (
        isolated_spot_gate.astype(float)
        * float(isolated_interval_blend)
        * isolated_headroom
    )
    corrected += isolated_uplift

    result = frame.copy()
    result["m19_predicted_target_nm"] = base
    result["spot_connectivity_gate"] = gate
    result["spot_connectivity_log_residual_correction"] = corrections
    result["rheed_spot_isolation_score"] = isolation_scores
    result["isolated_spot_uplift_gate"] = isolated_spot_gate
    result["isolated_spot_uncertainty_uplift_nm"] = isolated_uplift
    result["connectivity_neighbor_growth_ids"] = neighbor_ids
    result["connectivity_neighbor_distances"] = neighbor_distances
    result["predicted_target"] = np.maximum(corrected, 0.0)
    radius = result["interval_radius"].to_numpy(float)
    result["interval_lower"] = np.maximum(corrected - radius, 0.0)
    result["interval_upper"] = corrected + radius
    result["absolute_error"] = np.abs(corrected - truth)
    result["interval_covered"] = np.abs(corrected - truth) <= radius
    result["outer_target_used_for_training"] = False
    result["method"] = MODEL_NAME
    result["confidence_method"] = (
        "M19 uncertainty with strict outer-LOO spot-connectivity residual "
        "calibration; query AFM target excluded"
    )
    return result


def run(
    input_path: Path,
    physics_path: Path,
    output_dir: Path,
    *,
    neighbors: int,
    correction_strength: float,
    isolated_spot_threshold: float,
    isolated_interval_blend: float,
) -> None:
    source = pd.read_csv(input_path, dtype={"growth_run_id": str})
    physics = pd.read_csv(physics_path, dtype={"growth_run_id": str})
    result = crossfit_spot_connectivity_calibration(
        source,
        physics,
        neighbors=neighbors,
        correction_strength=correction_strength,
        isolated_spot_threshold=isolated_spot_threshold,
        isolated_interval_blend=isolated_interval_blend,
    )
    truth = result["true_target"].to_numpy(float)
    metrics = pd.DataFrame(
        [
            _metric_record(
                result,
                label="all",
                mask=np.ones(len(result), dtype=bool),
            ),
            _metric_record(
                result,
                label="smooth_below_1p6_nm",
                mask=truth < 1.6,
            ),
            _metric_record(
                result,
                label="rough_3_to_10_nm",
                mask=(truth >= 3.0) & (truth <= 10.0),
            ),
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "m20_strict_loo_predictions.csv", index=False)
    metrics.to_csv(output_dir / "m20_connectivity_metrics.csv", index=False)
    manifest = {
        "model": MODEL_NAME,
        "source_predictions": str(input_path),
        "physics_features": str(physics_path),
        "connectivity_features": list(CONNECTIVITY_FEATURES),
        "neighbors": int(neighbors),
        "correction_strength": float(correction_strength),
        "bridge_merge_threshold": -2.5,
        "nonround_fraction_threshold": 0.30,
        "isolated_spot_threshold": float(isolated_spot_threshold),
        "isolated_interval_blend": float(isolated_interval_blend),
        "query_target_used_for_calibration": False,
        "activation_growth_ids": result.loc[
            result["spot_connectivity_gate"].astype(bool), "growth_run_id"
        ].tolist(),
        "isolated_spot_uplift_growth_ids": result.loc[
            result["isolated_spot_uplift_gate"].astype(bool), "growth_run_id"
        ].tolist(),
        "metrics": metrics.to_dict(orient="records"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--correction-strength", type=float, default=1.0)
    parser.add_argument("--isolated-spot-threshold", type=float, default=0.75)
    parser.add_argument("--isolated-interval-blend", type=float, default=0.45)
    args = parser.parse_args()
    run(
        args.input,
        args.physics,
        args.output,
        neighbors=args.neighbors,
        correction_strength=args.correction_strength,
        isolated_spot_threshold=args.isolated_spot_threshold,
        isolated_interval_blend=args.isolated_interval_blend,
    )


if __name__ == "__main__":
    main()
