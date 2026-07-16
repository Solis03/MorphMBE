# Final Algorithm Audit

This directory is a read-only audit layer over the existing frozen RHEED-to-AFM artifacts. It does not train models, reselect methods, edit raw data, or modify the paper freeze.

Core conclusion: the strict visual result should be described as RHEED-conditioned representative AFM retrieval using fixed A3, not as a neural AFM pixel decoder and not as Phase3A AFM autoencoder performance.

Quantitative model: `full_cohort_top5_median_ridge_ensemble`.

Visual method for the final strict architecture: `A3` representative AFM retrieval. Phase7B comparison methods are benchmark-only.

Known deployment gap: `13_UNSEEN_INFERENCE/predict_unseen_batch.py` is a technical smoke script and does not yet implement actual DINO feature extraction plus descriptor A3 ranking.
