from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_NAME = "M19_rough_support_tail_rescue"


def apply_rough_tail_rescue(frame: pd.DataFrame) -> pd.DataFrame:
    """Rescue compressed rough predictions without consulting query truth.

    M17 already stores two independently trained endpoint estimates: the
    temporal video expert and a local-streak expert.  The latter is used as an
    upper support only when the existing rough-consensus gate fires, or when
    an implausibly smooth temporal prediction strongly conflicts with the
    streak estimate.  The rule deliberately leaves the established smooth
    regime unchanged.
    """

    required = {
        "predicted_target",
        "streak_expert_nm",
        "rough_consensus_gate",
        "interval_radius",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"endpoint predictions missing columns: {missing}")
    result = frame.copy()
    base = result["predicted_target"].to_numpy(float)
    streak = result["streak_expert_nm"].to_numpy(float)
    rough_support = result["rough_consensus_gate"].astype(bool).to_numpy()
    discordant_support = (base < 1.50) & (streak >= 2.50)
    activated = rough_support | discordant_support
    rescued = np.where(activated, np.maximum(base, streak), base)
    result["base_endpoint_prediction_nm"] = base
    result["rough_tail_rescue_activated"] = activated
    result["rough_tail_rescue_reason"] = np.select(
        [rough_support, discordant_support],
        ["rough_consensus", "discordant_streak_support"],
        default="inactive",
    )
    result["predicted_target"] = rescued
    radius = result["interval_radius"].to_numpy(float)
    result["interval_lower"] = np.maximum(rescued - radius, 0.0)
    result["interval_upper"] = rescued + radius
    if "true_target" in result:
        truth = result["true_target"].to_numpy(float)
        result["absolute_error"] = np.abs(rescued - truth)
        result["interval_covered"] = np.abs(rescued - truth) <= radius
    result["method"] = MODEL_NAME
    return result


def _metric_record(
    frame: pd.DataFrame, *, label: str, mask: np.ndarray
) -> dict[str, float | int | str]:
    subset = frame.loc[mask]
    truth = subset["true_target"].to_numpy(float)
    predicted = subset["predicted_target"].to_numpy(float)
    residual = predicted - truth
    return {
        "stratum": label,
        "count": len(subset),
        "mae_nm": float(np.mean(np.abs(residual))),
        "rmse_nm": float(np.sqrt(np.mean(np.square(residual)))),
        "bias_nm": float(np.mean(residual)),
    }


def run(input_path: Path, output_dir: Path) -> None:
    source = pd.read_csv(input_path, dtype={"growth_run_id": str})
    rescued = apply_rough_tail_rescue(source)
    truth = rescued["true_target"].to_numpy(float)
    metrics = pd.DataFrame(
        [
            _metric_record(
                rescued, label="all", mask=np.ones(len(rescued), dtype=bool)
            ),
            _metric_record(
                rescued,
                label="smooth_below_1p6_nm",
                mask=truth < 1.6,
            ),
            _metric_record(
                rescued,
                label="rough_3_to_10_nm",
                mask=(truth >= 3.0) & (truth <= 10.0),
            ),
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rescued.to_csv(output_dir / "m19_strict_loo_predictions.csv", index=False)
    metrics.to_csv(output_dir / "m19_tail_rescue_metrics.csv", index=False)
    manifest = {
        "model": MODEL_NAME,
        "source_predictions": str(input_path),
        "decision_uses_query_target": False,
        "rule": (
            "max(M17 endpoint, independent streak expert) when the existing "
            "rough-consensus gate is active or endpoint<1.50 nm and streak "
            "support>=2.50 nm; otherwise retain M17 exactly"
        ),
        "activation_count": int(
            rescued["rough_tail_rescue_activated"].sum()
        ),
        "metrics": metrics.to_dict(orient="records"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
