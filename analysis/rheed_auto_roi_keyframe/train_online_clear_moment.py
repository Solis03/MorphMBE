#!/usr/bin/env python3
"""Fit an absolute, causal clear-moment ranker for live RHEED streams."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline

from rheed2morph.realtime.selector import ONLINE_CLEAR_MOMENT_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-table",
        default=(
            "outputs/rheed_auto_roi_keyframe/"
            "20260728_dinov2_spot_visibility_v5/deep_candidate_table.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/rheed_realtime_ui/"
            "causal_clear_moment_detector_v1"
        ),
    )
    parser.add_argument("--minimum-score", type=float, default=0.40)
    return parser.parse_args()


def model_pipeline() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=4,
            max_features=0.75,
            random_state=17,
            n_jobs=1,
        ),
    )


def main() -> None:
    args = parse_args()
    source = Path(args.candidate_table).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(source)
    feature_names = list(ONLINE_CLEAR_MOMENT_FEATURES)
    missing = sorted(set(feature_names) - set(table))
    if missing:
        raise KeyError(f"Candidate table is missing features: {missing}")
    values = table[feature_names]
    target = table["target_similarity"].to_numpy(float)
    groups = table["sample_id"].astype(str).to_numpy()

    oof = cross_val_predict(
        model_pipeline(),
        values,
        target,
        groups=groups,
        cv=LeaveOneGroupOut(),
        n_jobs=1,
    )
    fitted = model_pipeline()
    fitted.fit(values, target)
    fitted_scores = np.asarray(fitted.predict(values), dtype=float)
    accepted = oof >= float(args.minimum_score)
    good = target >= 0.50
    metrics = {
        "validation_protocol": "strict_leave_one_video_out",
        "held_video_overlap_sum": 0,
        "video_count": int(len(np.unique(groups))),
        "candidate_count": int(len(table)),
        "mae": float(mean_absolute_error(target, oof)),
        "pearson_r": float(pearsonr(target, oof).statistic),
        "spearman_rho": float(spearmanr(target, oof).statistic),
        "good_frame_auc": float(roc_auc_score(good, oof)),
        "minimum_score": float(args.minimum_score),
        "accepted_candidate_count": int(accepted.sum()),
        "accepted_good_frame_precision": float(
            np.mean(good[accepted]) if accepted.any() else float("nan")
        ),
        "accepted_mean_target_similarity": float(
            np.mean(target[accepted]) if accepted.any() else float("nan")
        ),
    }
    bundle = {
        "schema_version": 1,
        "model_family": "absolute_visual_extra_trees_clear_moment",
        "feature_names": feature_names,
        "model": fitted,
        "score_reference": np.sort(fitted_scores),
        "minimum_score": float(args.minimum_score),
        "minimum_visibility_proxy": 1.15,
        "maximum_shadow_fraction": 0.35,
        "minimum_spot_peak_count": 3.0,
        "lookahead_frames": 4,
        "target": "human_keyframe_pattern_similarity",
        "validation": metrics,
        "candidate_table": str(source),
        "candidate_table_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "prospective_note": (
            "The detector is causal and target-blind at inference. Thresholds "
            "require prospective validation on newly acquired camera streams."
        ),
    }
    joblib.dump(bundle, output / "online_clear_moment_detector.joblib")
    pd.DataFrame(
        {
            "sample_id": groups,
            "frame_index": table["frame_index"].to_numpy(int),
            "target_similarity": target,
            "strict_loo_prediction": oof,
            "accepted_at_default_threshold": accepted,
        }
    ).to_csv(output / "strict_loo_predictions.csv", index=False)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
