from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analysis.rheed_video_afm_story.common import repo_path, write_csv

from .comparison import PROTOCOL_AUTO, PROTOCOL_HUMAN, PROTOCOL_SHIFT
from .dataset import load_config


def paired_error_statistics(
    predictions: pd.DataFrame,
    *,
    draws: int = 20_000,
    seed: int = 20260729,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for target in ("Rq_nm", "FSMI_nm"):
        target_rows = predictions.loc[predictions["target"] == target]
        human = (
            target_rows.loc[target_rows["protocol"] == PROTOCOL_HUMAN]
            .set_index("growth_run_id")["absolute_error"]
            .sort_index()
        )
        for protocol in (PROTOCOL_AUTO, PROTOCOL_SHIFT):
            alternative = (
                target_rows.loc[target_rows["protocol"] == protocol]
                .set_index("growth_run_id")
                .loc[human.index, "absolute_error"]
            )
            delta = alternative.to_numpy(float) - human.to_numpy(float)
            indices = rng.integers(0, len(delta), size=(draws, len(delta)))
            bootstrap = delta[indices].mean(axis=1)
            test = wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
            records.append(
                {
                    "target": target,
                    "comparison": f"{protocol} minus {PROTOCOL_HUMAN}",
                    "growth_group_count": len(delta),
                    "mean_absolute_error_change_nm": float(np.mean(delta)),
                    "median_absolute_error_change_nm": float(np.median(delta)),
                    "paired_bootstrap_mean_change_lower_95_nm": float(
                        np.quantile(bootstrap, 0.025)
                    ),
                    "paired_bootstrap_mean_change_upper_95_nm": float(
                        np.quantile(bootstrap, 0.975)
                    ),
                    "bootstrap_probability_improved": float(
                        np.mean(bootstrap < 0.0)
                    ),
                    "wilcoxon_signed_rank_statistic": float(test.statistic),
                    "wilcoxon_signed_rank_p": float(test.pvalue),
                    "negative_change_means_lower_error": True,
                }
            )
    return pd.DataFrame(records)


def run(config_path: str | Path) -> None:
    config = load_config(config_path)
    report = repo_path(config["report_root"])
    predictions = pd.read_csv(
        report / "paired_target_predictions.csv",
        dtype={"growth_run_id": str},
    )
    table = paired_error_statistics(predictions)
    write_csv(table, report / "paired_error_statistics.csv")
    print(table.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rheed_manual_vs_auto_selection.json",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
