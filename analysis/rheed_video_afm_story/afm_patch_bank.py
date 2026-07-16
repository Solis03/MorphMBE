from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from skimage.transform import resize

from .common import repo_path
from .rq_disentanglement import project_unit_rq_np


def load_bank_shape(row: pd.Series) -> np.ndarray:
    arr = np.load(repo_path(row["unit_shape_map_path"])).astype(np.float32)
    if arr.shape != (256, 256):
        arr = resize(arr, (256, 256), order=1, mode="reflect", anti_aliasing=True, preserve_range=True).astype(np.float32)
    return project_unit_rq_np(arr)


def training_bank_for_sample(bank: pd.DataFrame, heldout_group: str, candidate_groups: list[str]) -> pd.DataFrame:
    train = bank[bank["growth_run_id"].astype(str) != str(heldout_group)].copy()
    ordered = []
    for group in candidate_groups:
        g = train[train["growth_run_id"].astype(str) == str(group)]
        if len(g):
            ordered.append(g.iloc[0])
    return pd.DataFrame(ordered) if ordered else train.head(0)


def weighted_candidate_stats(candidates: pd.DataFrame) -> dict[str, float]:
    return {
        "rq_median": float(candidates["rq_nm"].median()),
        "psd_low_fraction": float(candidates["psd_low_fraction"].median()),
        "psd_mid_fraction": float(candidates["psd_mid_fraction"].median()),
        "psd_high_fraction": float(candidates["psd_high_fraction"].median()),
        "correlation_length_nm": float(candidates["correlation_length_nm"].median()),
        "anisotropy": float(candidates["anisotropy"].median()),
    }
