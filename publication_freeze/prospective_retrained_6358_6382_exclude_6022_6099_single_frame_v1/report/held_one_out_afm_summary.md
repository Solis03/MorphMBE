# Held-one-out AFM prediction report

For each of the 26 retained labeled samples, the quantitative ensemble is trained on the other 25 retained samples and the A3 AFM retrieval bank excludes every AFM from the held-out sample. Samples 6022 and 6099 are absent from every fold. The representative ground-truth AFM is selected only after prediction and never enters its own fold.

N6342 uses ground truth 5, N6389 uses ground truth 3, and N6390 uses ground truth 1 as explicitly requested. Every other sample uses the quality-passing AFM whose measured Rq is closest to that sample's T4 target.

Against the displayed representative AFMs, rendered-map Rq MAE is 1.3522 nm and RMSE is 1.6667 nm.

## Per-sample selection and prediction

| Sample | GT selection | GT file | GT Rq | Raw LOO Rq | Rendered Rq | Retrieved source |
|---|---|---|---:|---:|---:|---|
| 6028 | minimum_absolute_rq_distance_to_T4_target | N6028_1um_003 | 5.7831 | 4.3779 | 4.3779 | 6056 / GdSb_N6056_edge_008 |
| 6029 | minimum_absolute_rq_distance_to_T4_target | N6029_1um_007 | 2.2898 | 3.7471 | 3.7471 | 6056 / GdSb_N6056_edge_008 |
| 6033 | minimum_absolute_rq_distance_to_T4_target | N6033_1um_016 | 2.2942 | 4.2824 | 4.2824 | 6056 / GdSb_N6056_edge_008 |
| 6047 | minimum_absolute_rq_distance_to_T4_target | N6047_1um_008 | 4.6949 | 4.0347 | 4.0347 | 6056 / GdSb_N6056_edge_008 |
| 6048 | minimum_absolute_rq_distance_to_T4_target | N6048_1um_027 | 1.8682 | 3.4421 | 3.4421 | 6028 / N6028_500_nm_006 |
| 6056 | minimum_absolute_rq_distance_to_T4_target | GdSb_N6056_ctr_009 | 2.7709 | 2.2286 | 2.2286 | 6028 / N6028_500_nm_006 |
| 6057 | minimum_absolute_rq_distance_to_T4_target | GdSb_N6057_edge_014 | 4.5519 | 1.7885 | 1.7885 | 6028 / N6028_500_nm_006 |
| 6062 | minimum_absolute_rq_distance_to_T4_target | N62_Ctr_003 | 3.0082 | 0.2768 | 0.2768 | 6094 / N6094_1_um_008 |
| 6063 | minimum_absolute_rq_distance_to_T4_target | N63_Ctr_004 | 5.7035 | 4.6683 | 4.6683 | 6056 / GdSb_N6056_edge_008 |
| 6070 | minimum_absolute_rq_distance_to_T4_target | N69_center_003 | 2.7649 | 3.5578 | 3.5578 | 6028 / N6028_500_nm_006 |
| 6072 | minimum_absolute_rq_distance_to_T4_target | N6072_1um_002 | 1.2756 | 2.5664 | 2.5664 | 6028 / N6028_500_nm_006 |
| 6078 | minimum_absolute_rq_distance_to_T4_target | N78_1um_005 | 1.2683 | 0.8600 | 0.8600 | 6028 / N6028_500_nm_006 |
| 6080 | minimum_absolute_rq_distance_to_T4_target | N80_Ctr_000 | 3.6452 | 2.7905 | 2.7905 | 6028 / N6028_500_nm_006 |
| 6081 | minimum_absolute_rq_distance_to_T4_target | N6081_1um_000 | 0.9417 | 2.7999 | 2.7999 | 6028 / N6028_500_nm_006 |
| 6082 | minimum_absolute_rq_distance_to_T4_target | N6082_1um_001 | 1.8639 | 4.1611 | 4.1611 | 6056 / GdSb_N6056_edge_008 |
| 6084 | minimum_absolute_rq_distance_to_T4_target | N6084_1um_005 | 1.6138 | 2.8802 | 2.8802 | 6028 / N6028_500_nm_006 |
| 6085 | minimum_absolute_rq_distance_to_T4_target | N6085_1um_000 | 2.7688 | 3.9433 | 3.9433 | 6056 / GdSb_N6056_edge_008 |
| 6090 | minimum_absolute_rq_distance_to_T4_target | N6090_1um_001 | 2.8617 | 1.4810 | 1.4810 | 6028 / N6028_500_nm_006 |
| 6094 | minimum_absolute_rq_distance_to_T4_target | N6094_1_um_008 | 1.3909 | 3.7067 | 3.7067 | 6056 / GdSb_N6056_edge_008 |
| 6095 | minimum_absolute_rq_distance_to_T4_target | N6095_1um_003 | 8.1496 | 3.6032 | 3.6032 | 6056 / GdSb_N6056_edge_008 |
| 6101 | minimum_absolute_rq_distance_to_T4_target | N6101_1um_002 | 1.3161 | 1.9570 | 1.9570 | 6028 / N6028_500_nm_006 |
| N6342 | user_forced_ground_truth_5 | N6342_scan_5_top_left_quarter | 0.8773 | 1.4807 | 1.4807 | 6028 / N6028_500_nm_006 |
| N6358 | minimum_absolute_rq_distance_to_T4_target | N6358_scan_4_top_left_quarter | 1.0902 | 0.0697 | 0.0697 | 6094 / N6094_1_um_008 |
| N6382 | minimum_absolute_rq_distance_to_T4_target | N6382_scan_2_top_left_quarter | 1.3054 | 1.4915 | 1.4915 | 6028 / N6028_500_nm_006 |
| N6389 | user_forced_ground_truth_3 | N6389_scan_3_top_left_quarter | 2.5396 | 2.5836 | 2.5836 | 6028 / N6028_500_nm_006 |
| N6390 | user_forced_ground_truth_1 | N6390_scan_1_top_left_quarter | 2.3975 | 2.0773 | 2.0773 | 6028 / N6028_500_nm_006 |

## Primary atlas

- `figures/main/Figure3_held_one_out_afm_prediction_atlas.png`
- `figures/main/Figure3_held_one_out_afm_prediction_atlas.pdf`
- `figures/main/Figure3_held_one_out_afm_prediction_atlas.svg`

All ground-truth and predicted AFM panels include a physical height bar in nm and the Rq of the displayed array.
