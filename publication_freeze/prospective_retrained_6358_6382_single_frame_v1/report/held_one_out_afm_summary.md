# Held-one-out AFM prediction report

For each of all 28 labeled samples, the quantitative ensemble is trained on the other 27 samples and the A3 AFM retrieval bank excludes every AFM from the held-out sample. The representative ground-truth AFM is selected only after prediction and never enters its own fold.

N6342 uses ground truth 5, N6389 uses ground truth 3, and N6390 uses ground truth 1 as explicitly requested. Every other sample uses the quality-passing AFM whose measured Rq is closest to that sample's T4 target.

Against the displayed representative AFMs, rendered-map Rq MAE is 1.6339 nm and RMSE is 2.2851 nm.

## Per-sample selection and prediction

| Sample | GT selection | GT file | GT Rq | Raw LOO Rq | Rendered Rq | Retrieved source |
|---|---|---|---:|---:|---:|---|
| 6022 | minimum_absolute_rq_distance_to_T4_target | N6022_Ctr_004 | 1.3614 | 6.7805 | 6.7805 | 6057 / GdSb_N6057_ctr_020 |
| 6028 | minimum_absolute_rq_distance_to_T4_target | N6028_1um_003 | 5.7831 | 5.0155 | 5.0155 | 6057 / GdSb_N6057_ctr_020 |
| 6029 | minimum_absolute_rq_distance_to_T4_target | N6029_1um_007 | 2.2898 | 3.7576 | 3.7576 | 6028 / N6028_500_nm_006 |
| 6033 | minimum_absolute_rq_distance_to_T4_target | N6033_1um_016 | 2.2942 | 2.5958 | 2.5958 | 6028 / N6028_500_nm_006 |
| 6047 | minimum_absolute_rq_distance_to_T4_target | N6047_1um_008 | 4.6949 | 3.0403 | 3.0403 | 6028 / N6028_500_nm_006 |
| 6048 | minimum_absolute_rq_distance_to_T4_target | N6048_1um_027 | 1.8682 | 1.8112 | 1.8112 | 6028 / N6028_500_nm_006 |
| 6056 | minimum_absolute_rq_distance_to_T4_target | GdSb_N6056_ctr_009 | 2.7709 | 2.9245 | 2.9245 | 6028 / N6028_500_nm_006 |
| 6057 | minimum_absolute_rq_distance_to_T4_target | GdSb_N6057_edge_014 | 4.5519 | 3.1116 | 3.1116 | 6028 / N6028_500_nm_006 |
| 6062 | minimum_absolute_rq_distance_to_T4_target | N62_Ctr_003 | 3.0082 | 3.7602 | 3.7602 | 6028 / N6028_500_nm_006 |
| 6063 | minimum_absolute_rq_distance_to_T4_target | N63_Ctr_004 | 5.7035 | 3.5983 | 3.5983 | 6028 / N6028_500_nm_006 |
| 6070 | minimum_absolute_rq_distance_to_T4_target | N69_center_003 | 2.7649 | 5.1308 | 5.1308 | 6057 / GdSb_N6057_ctr_020 |
| 6072 | minimum_absolute_rq_distance_to_T4_target | N6072_1um_002 | 1.2756 | 2.8899 | 2.8899 | 6028 / N6028_500_nm_006 |
| 6078 | minimum_absolute_rq_distance_to_T4_target | N78_1um_005 | 1.2683 | 0.8691 | 0.8691 | 6028 / N6028_500_nm_006 |
| 6080 | minimum_absolute_rq_distance_to_T4_target | N80_Ctr_000 | 3.6452 | 3.6163 | 3.6163 | 6028 / N6028_500_nm_006 |
| 6081 | minimum_absolute_rq_distance_to_T4_target | N6081_1um_000 | 0.9417 | 4.9129 | 4.9129 | 6057 / GdSb_N6057_ctr_017 |
| 6082 | minimum_absolute_rq_distance_to_T4_target | N6082_1um_001 | 1.8639 | 3.7911 | 3.7911 | 6028 / N6028_500_nm_006 |
| 6084 | minimum_absolute_rq_distance_to_T4_target | N6084_1um_005 | 1.6138 | 2.9732 | 2.9732 | 6028 / N6028_500_nm_006 |
| 6085 | minimum_absolute_rq_distance_to_T4_target | N6085_1um_000 | 2.7688 | 2.4167 | 2.4167 | 6028 / N6028_500_nm_006 |
| 6090 | minimum_absolute_rq_distance_to_T4_target | N6090_1um_001 | 2.8617 | 1.9244 | 1.9244 | 6028 / N6028_500_nm_006 |
| 6094 | minimum_absolute_rq_distance_to_T4_target | N6094_1_um_008 | 1.3909 | 5.3971 | 5.3971 | 6057 / GdSb_N6057_ctr_020 |
| 6095 | minimum_absolute_rq_distance_to_T4_target | N6095_1um_003 | 8.1496 | 4.7523 | 4.7523 | 6056 / GdSb_N6056_edge_008 |
| 6099 | minimum_absolute_rq_distance_to_T4_target | N6099_1um_003 | 10.2962 | 3.8817 | 3.8817 | 6056 / GdSb_N6056_edge_008 |
| 6101 | minimum_absolute_rq_distance_to_T4_target | N6101_1um_002 | 1.3161 | 0.8167 | 0.8167 | 6028 / N6028_500_nm_006 |
| N6342 | user_forced_ground_truth_5 | N6342_scan_5_top_left_quarter | 0.8773 | 0.6927 | 0.6927 | 6094 / N6094_1_um_008 |
| N6358 | minimum_absolute_rq_distance_to_T4_target | N6358_scan_4_top_left_quarter | 1.0902 | -0.2750 | 0.0010 | 6094 / N6094_1_um_008 |
| N6382 | minimum_absolute_rq_distance_to_T4_target | N6382_scan_2_top_left_quarter | 1.3054 | 0.3736 | 0.3736 | 6094 / N6094_1_um_008 |
| N6389 | user_forced_ground_truth_3 | N6389_scan_3_top_left_quarter | 2.5396 | 3.4146 | 3.4146 | 6028 / N6028_500_nm_006 |
| N6390 | user_forced_ground_truth_1 | N6390_scan_1_top_left_quarter | 2.3975 | 3.6747 | 3.6747 | 6028 / N6028_500_nm_006 |

## Primary atlas

- `figures/main/Figure3_held_one_out_afm_prediction_atlas.png`
- `figures/main/Figure3_held_one_out_afm_prediction_atlas.pdf`
- `figures/main/Figure3_held_one_out_afm_prediction_atlas.svg`

All ground-truth and predicted AFM panels include a physical height bar in nm and the Rq of the displayed array.
