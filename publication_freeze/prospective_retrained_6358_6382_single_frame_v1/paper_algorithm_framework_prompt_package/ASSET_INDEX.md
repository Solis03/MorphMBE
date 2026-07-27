# Scientific image asset index

All paths below are relative to this package. Files in `05_reference_only/`
show previous layouts and must not be pasted into the new schematic as data
panels.

## Recommended worked example: N6390

| Pipeline stage | Exact file | Required use |
|---|---|---|
| Raw keyframe | `01_rheed_inputs/raw_keyframes/N6390_frame_000137.png` | First RHEED panel; real 1376 × 1100 RGB frame |
| Manual ROI | `01_rheed_inputs/roi_keyframes/N6390_frame_000137_roi.png` | Second RHEED panel |
| Model-ready luminance | `01_rheed_inputs/model_ready/N6390_model_ready_224x224_luminance.png` | Exact 224 × 224 DINO input image before channel duplication/normalization |
| Patch visualization | `01_rheed_inputs/model_ready/N6390_model_ready_patch14_grid.png` | Optional patch-embedding inset; the cyan grid is exactly 16 × 16 |
| A3 top-five bank | `04_afm_candidate_bank/N6390_A3_top5_candidate_bank_montage.png` | Candidate-bank visual; already ordered by A3 score |
| Selected source AFM | `04_afm_candidate_bank/rank1_6028_N6028_500_nm_006_selected_heightbar_Rq.png` | Rank-1 morphology selected by A3 |
| Retrieved AFM | `03_afm_retrieval_outputs/N6390_retrieved_heightbar_rq.png` | Final q50 representative AFM output |
| Experimental AFM | `02_afm_ground_truth_selected/N6390_GT1_top_left_quarter_heightbar_Rq.png` | Post-hoc evaluation only; never an input |

N6390 numerical annotations:

- predicted Rq: 2.250672 nm;
- selected source: sample 6028 / `N6028_500_nm_006`;
- source Rq: 2.319988 nm;
- sample-level T4 ground truth: 2.297728 nm;
- displayed ground truth 1 Rq: 2.397455 nm.

## RHEED inputs for the five extra samples

| Sample | Role | Raw keyframe | ROI keyframe | Model-ready PNG |
|---|---|---|---|---|
| N6342 | prospective test | `01_rheed_inputs/raw_keyframes/N6342_frame_001501.png` | `01_rheed_inputs/roi_keyframes/N6342_frame_001501_roi.png` | `01_rheed_inputs/model_ready/N6342_model_ready_224x224_luminance.png` |
| N6358 | added training | `01_rheed_inputs/raw_keyframes/N6358_frame_000620.png` | `01_rheed_inputs/roi_keyframes/N6358_frame_000620_roi.png` | `01_rheed_inputs/model_ready/N6358_model_ready_224x224_luminance.png` |
| N6382 | added training | `01_rheed_inputs/raw_keyframes/N6382_frame_000159.png` | `01_rheed_inputs/roi_keyframes/N6382_frame_000159_roi.png` | `01_rheed_inputs/model_ready/N6382_model_ready_224x224_luminance.png` |
| N6389 | prospective test | `01_rheed_inputs/raw_keyframes/N6389_frame_000600.png` | `01_rheed_inputs/roi_keyframes/N6389_frame_000600_roi.png` | `01_rheed_inputs/model_ready/N6389_model_ready_224x224_luminance.png` |
| N6390 | prospective test | `01_rheed_inputs/raw_keyframes/N6390_frame_000137.png` | `01_rheed_inputs/roi_keyframes/N6390_frame_000137_roi.png` | `01_rheed_inputs/model_ready/N6390_model_ready_224x224_luminance.png` |

The corresponding `.npz` files in `01_rheed_inputs/model_ready/` retain the
exact uint8 tensor and provenance. They are supplied for audit, not as drawing
panels.

## Selected AFM ground truths

| Sample | File | Displayed-map Rq | Sample-level T4 | Figure role |
|---|---|---:|---:|---|
| N6342 | `02_afm_ground_truth_selected/N6342_GT5_top_left_quarter_heightbar_Rq.png` | 0.877334 nm | 0.894464 nm | forced GT 5; evaluation only |
| N6358 | `02_afm_ground_truth_selected/N6358_GT4_top_left_quarter_heightbar_Rq.png` | 1.090183 nm | 1.092313 nm | closest-Rq representative; added training sample |
| N6382 | `02_afm_ground_truth_selected/N6382_GT2_top_left_quarter_heightbar_Rq.png` | 1.305372 nm | 1.303937 nm | closest-Rq representative; added training sample |
| N6389 | `02_afm_ground_truth_selected/N6389_GT3_top_left_quarter_heightbar_Rq.png` | 2.539615 nm | 2.335015 nm | forced GT 3; evaluation only |
| N6390 | `02_afm_ground_truth_selected/N6390_GT1_top_left_quarter_heightbar_Rq.png` | 2.397455 nm | 2.297728 nm | forced GT 1; evaluation only |

The N6358/N6382 scalar training targets are five-scan sample-level aggregates;
the single maps above are representative visual examples, not the sole scalar
labels.

## Prospective retrieved AFMs

| Sample | Exact file | Predicted/retrieved Rq |
|---|---|---:|
| N6342 | `03_afm_retrieval_outputs/N6342_retrieved_heightbar_rq.png` | 1.447468 nm |
| N6389 | `03_afm_retrieval_outputs/N6389_retrieved_heightbar_rq.png` | 2.516366 nm |
| N6390 | `03_afm_retrieval_outputs/N6390_retrieved_heightbar_rq.png` | 2.250672 nm |

## N6390 A3 candidate bank

| Rank | Exact file | Source | Source Rq |
|---:|---|---|---:|
| 1 | `04_afm_candidate_bank/rank1_6028_N6028_500_nm_006_selected_heightbar_Rq.png` | 6028 / N6028_500_nm_006 | 2.319988 nm |
| 2 | `04_afm_candidate_bank/rank2_6085_N6085_1um_001_candidate_heightbar_Rq.png` | 6085 / N6085_1um_001 | 2.120037 nm |
| 3 | `04_afm_candidate_bank/rank3_6048_N6048_1um_027_candidate_heightbar_Rq.png` | 6048 / N6048_1um_027 | 1.868178 nm |
| 4 | `04_afm_candidate_bank/rank4_6047_N6047_1um_013_candidate_heightbar_Rq.png` | 6047 / N6047_1um_013 | 2.003680 nm |
| 5 | `04_afm_candidate_bank/rank5_6056_N56_Ctr_000_candidate_heightbar_Rq.png` | 6056 / N56_Ctr_000 | 2.658612 nm |

Every AFM file in the three AFM asset directories includes its height bar in
nm and an Rq label. Do not crop either element.

## Reference-only files

- `05_reference_only/reference_Figure1_three_sample_prediction_atlas.png`:
  confirms how RHEED, five ground truths, and retrieved maps relate.
- `05_reference_only/reference_Figure3_held_one_out_AFMs.png`: confirms the
  full held-one-out atlas and AFM display conventions.

These are composition references only. Do not embed them as primary panels and
do not copy their crowded layout into the new framework figure.
