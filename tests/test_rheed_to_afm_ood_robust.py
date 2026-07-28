from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.rheed_to_afm_functional_morphology.amplitude import (
    CURATED_RHEED_FEATURES,
    DYNAMIC_NUCLEATION_FEATURES,
)
from analysis.rheed_to_afm_ood_robust.prediction import (
    CandidateConfig,
    crossfit_robust_candidates,
)
from analysis.rheed_to_afm_ood_robust.support import (
    density_weights,
    leave_one_out_support_audit,
)


def _physics(groups: list[str]) -> pd.DataFrame:
    values = np.arange(len(groups), dtype=float)
    return pd.DataFrame(
        {
            column: values + 0.01 * position
            for position, column in enumerate(
                CURATED_RHEED_FEATURES + DYNAMIC_NUCLEATION_FEATURES
            )
        },
        index=groups,
    )


def test_target_blind_support_audit_finds_extreme_covariate() -> None:
    groups = [f"g{index}" for index in range(7)]
    physics = _physics(groups)
    physics.loc["g6", CURATED_RHEED_FEATURES] += 100.0
    audit = leave_one_out_support_audit(physics, groups)

    assert audit.iloc[0]["growth_run_id"] == "g6"
    assert bool(audit.iloc[0]["excluded_in_top2_sensitivity"])
    assert not bool(audit["selection_used_afm_target"].any())


def test_density_weights_downweight_sparse_training_group() -> None:
    groups = [f"g{index}" for index in range(7)]
    physics = _physics(groups)
    physics.loc["g6", CURATED_RHEED_FEATURES] += 100.0
    weights = density_weights(physics, groups).set_index("growth_run_id")

    assert weights.loc["g6", "density_sample_weight"] < 0.5
    assert weights.loc["g6", "density_sample_weight"] < weights.loc[
        "g3", "density_sample_weight"
    ]


def test_robust_crossfit_never_uses_outer_target() -> None:
    groups = [f"g{index}" for index in range(7)]
    physics = _physics(groups)
    embeddings = pd.DataFrame(
        np.arange(7 * 8, dtype=float).reshape(7, 8),
        index=groups,
    )
    target = pd.Series(
        np.log(np.linspace(1.0, 4.0, len(groups))),
        index=groups,
        name="log_target",
    )
    fixed, selected, inner = crossfit_robust_candidates(
        physics=physics,
        embeddings=embeddings,
        log_target=target,
        config=CandidateConfig(r3d_pca_components=2),
    )

    assert fixed["growth_run_id"].nunique() == len(groups)
    assert selected["growth_run_id"].nunique() == len(groups)
    assert not bool(fixed["outer_target_used_for_training"].any())
    assert set(inner["outer_held_growth_group"]) == set(groups)
