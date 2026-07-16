# Worked Example: 6081 (low_true_rq)

True T4 Rq: 1.122543 nm

Strict OOF predicted q50 Rq: 2.935414 nm

Strict q10/q50/q90: 0.526422, 2.751703, 4.976985 nm

Full-cohort in-sample prediction: 1.144733 nm

Selected A3 source: sample 6056, scan GdSb_N6056_ctr_013.

Source path: `data/afm_second_order/6056/GdSb_N6056_ctr_013/GdSb_N6056_ctr_013_height.npy`.

## Ensemble Members

| member | trial_id | model_family | feature_set | target_variant | input_dim | prediction_nm |
| --- | --- | --- | --- | --- | --- | --- |
| model_01_trial_0004 | trial_0004 | ridge | E1_dino_keyframe | T4_second_order_trimmed_mean | 1536 | 1.14473258136 |
| model_02_trial_0012 | trial_0012 | ridge | E1_dino_keyframe | T4_second_order_trimmed_mean | 1536 | 1.14473258136 |
| model_03_trial_0006 | trial_0006 | ridge | E1_dino_keyframe | T6_quality_weighted_second_order | 1536 | 1.27553393861 |
| model_04_trial_0014 | trial_0014 | ridge | E1_dino_keyframe | T6_quality_weighted_second_order | 1536 | 1.27553393861 |
| model_05_trial_0028 | trial_0028 | ridge | E1_dino_keyframe | T4_second_order_trimmed_mean | 1536 | 1.14473258136 |


## Top A3 Sources

| sample_id | scan_id | rq_nm | descriptor_z_euclidean | rq_penalty_abs | rank_score | second_order_afm_path |
| --- | --- | --- | --- | --- | --- | --- |
| 6056 | GdSb_N6056_ctr_013 | 2.82553553581 | 1.8345656318 | 0.00369160585783 | 1.83825723766 | data/afm_second_order/6056/GdSb_N6056_ctr_013/GdSb_N6056_ctr_013_height.npy |
| 6057 | GdSb_N6057_edge_014 | 4.55192947388 | 2.35977579491 | 0.0900113027611 | 2.44978709767 | data/afm_second_order/6057/GdSb_N6057_edge_014/GdSb_N6057_edge_014_height.npy |
| 6072 | 6072_500_nm_004 | 1.47022557259 | 2.99175438105 | 0.0640738923035 | 3.05582827336 | data/afm_second_order/6072/6072_500_nm_004/6072_500_nm_004_height.npy |
| 6078 | N78_Ctr_000 | 1.63835644722 | 3.24994959746 | 0.0556673485718 | 3.30561694603 | data/afm_second_order/6078/N78_Ctr_000/N78_Ctr_000_height.npy |
| 6048 | N6048_1um_027 | 1.86817789078 | 3.43052411262 | 0.0441762763939 | 3.47470038902 | data/afm_second_order/6048/N6048_1um_027/N6048_1um_027_height.npy |
