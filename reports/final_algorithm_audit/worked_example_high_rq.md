# Worked Example: 6099 (high_true_rq)

True T4 Rq: 10.267897 nm

Strict OOF predicted q50 Rq: 3.725471 nm

Strict q10/q50/q90: 1.483611, 3.376097, 5.268584 nm

Full-cohort in-sample prediction: 10.232314 nm

Selected A3 source: sample 6047, scan N6047_1um_008.

Source path: `data/afm_second_order/6047/N6047_1um_008/N6047_1um_008_height.npy`.

## Ensemble Members

| member | trial_id | model_family | feature_set | target_variant | input_dim | prediction_nm |
| --- | --- | --- | --- | --- | --- | --- |
| model_01_trial_0004 | trial_0004 | ridge | E1_dino_keyframe | T4_second_order_trimmed_mean | 1536 | 10.2323143482 |
| model_02_trial_0012 | trial_0012 | ridge | E1_dino_keyframe | T4_second_order_trimmed_mean | 1536 | 10.2323143482 |
| model_03_trial_0006 | trial_0006 | ridge | E1_dino_keyframe | T6_quality_weighted_second_order | 1536 | 10.8271184309 |
| model_04_trial_0014 | trial_0014 | ridge | E1_dino_keyframe | T6_quality_weighted_second_order | 1536 | 10.8271184309 |
| model_05_trial_0028 | trial_0028 | ridge | E1_dino_keyframe | T4_second_order_trimmed_mean | 1536 | 10.2323143482 |


## Top A3 Sources

| sample_id | scan_id | rq_nm | descriptor_z_euclidean | rq_penalty_abs | rank_score | second_order_afm_path |
| --- | --- | --- | --- | --- | --- | --- |
| 6047 | N6047_1um_008 | 4.69486045837 | 2.09626015624 | 0.0659381524391 | 2.16219830868 | data/afm_second_order/6047/N6047_1um_008/N6047_1um_008_height.npy |
| 6057 | GdSb_N6057_edge_014 | 4.55192947388 | 2.88717789301 | 0.0587916032143 | 2.94596949623 | data/afm_second_order/6057/GdSb_N6057_edge_014/GdSb_N6057_edge_014_height.npy |
| 6072 | 6072_500_nm_004 | 1.47022557259 | 2.8972994404 | 0.0952935918503 | 2.99259303225 | data/afm_second_order/6072/6072_500_nm_004/6072_500_nm_004_height.npy |
| 6078 | N78_Ctr_000 | 1.63835644722 | 2.99159876355 | 0.0868870481186 | 3.07848581167 | data/afm_second_order/6078/N78_Ctr_000/N78_Ctr_000_height.npy |
| 6056 | GdSb_N6056_ctr_013 | 2.82553553581 | 3.10378639642 | 0.027528093689 | 3.13131449011 | data/afm_second_order/6056/GdSb_N6056_ctr_013/GdSb_N6056_ctr_013_height.npy |
