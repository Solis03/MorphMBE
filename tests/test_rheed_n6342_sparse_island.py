from __future__ import annotations

import numpy as np

from analysis.rheed_n6342_sparse_island.evaluate import peak_signature


def test_peak_signature_detects_sparser_prominent_peaks() -> None:
    y, x = np.mgrid[:128, :128]
    sparse = np.zeros((128, 128), dtype=float)
    dense = np.zeros_like(sparse)
    for cy, cx in ((24, 24), (48, 78), (92, 36), (100, 104)):
        sparse += np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 2.0**2))
    for cy in range(12, 128, 20):
        for cx in range(12, 128, 20):
            dense += np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 2.0**2))
    sparse_signature = peak_signature(sparse)
    dense_signature = peak_signature(dense)

    assert (
        sparse_signature["persistent_peak_count_h050"]
        < dense_signature["persistent_peak_count_h050"]
    )
    assert all(np.isfinite(list(sparse_signature.values())))
