# RHEED-to-AFM Single-Frame Publication Freeze v1

Freeze ID: `rheed_afm_single_frame_v1_2026-07-18`

This directory freezes the final selected single-frame retrospective path only: selected RHEED keyframe -> frozen DINOv2 embedding -> top-five Ridge median ensemble -> predicted AFM Rq q50 -> strict A3 historical AFM retrieval.

- Cohort: N = 23 strict samples.
- Target: `T4_second_order_trimmed_mean`, Rq in nm.
- Quantitative MAE: 1.2600983408 nm; RMSE: 1.8393101334 nm; R2: 0.2939669043; Spearman: 0.4288537549.
- Visual method: strict training-only `A3` retrieval, q50 atlas.

Prospective unseen deployment is blocked in `BLOCKER_PROSPECTIVE_DEPLOYMENT.md`; no production `predict_unseen.py` is included.
