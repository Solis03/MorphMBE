# RHEED-to-AFM Publication Freeze Technical and Experimental Report

Report version: 2026-07-22  
Repository root: `/Users/ziyi/Desktop/LAB/code`  
Primary frozen package: `publication_freeze/rheed_afm_single_frame_v1_2026-07-18`  
Prospective package: `publication_freeze/prospective_unseen_single_frame_v1`

This document summarizes the frozen RHEED-to-AFM single-frame method at the code, data, model, and experiment-result levels. It is intended as a technical basis for the final manuscript and supplementary methods. Numeric summaries are derived from the frozen package and the current prospective AFM truth processing outputs. The companion machine-readable summary is `reports/publication_freeze_technical_report_stats_2026-07-22.json`.

## Executive Summary

The frozen publication method maps one manually selected RHEED keyframe to AFM morphology in two stages:

1. Quantitative route: manual RHEED ROI keyframe -> frozen DINOv2 ViT-S/14 encoder -> 1536-D feature vector -> five-member Ridge ensemble -> predicted second-order AFM Rq in nm.
2. Visual route: predicted Rq and full-cohort AFM descriptor bank -> A3-style retrieval of a historical AFM morphology -> amplitude rescale to predicted Rq -> representative AFM height map.

The frozen retrospective benchmark uses 23 strict historical growth groups with leave-one-growth-group-out predictions. The final frozen strict OOF quantitative result is:

| Metric | Value |
|---|---:|
| N | 23 |
| MAE | 1.2600983408 nm |
| RMSE | 1.8393101334 nm |
| R2 | 0.2939669043 |
| Spearman | 0.4288537549 |
| Kendall | 0.2806324111 |
| Median absolute error | 1.1205674311 nm |
| Pairwise concordance | 0.6403162055 |

For prospective deployment, the current package uses full-cohort models trained with all 23 historical samples. Five RHEED unseen samples were predicted: `N6342`, `N6358`, `N6382`, `N6389`, and `N6390`. Current AFM truth has been processed for `N6324`, `N6342`, `N6358`, `N6382`, and `N6389`. Therefore, only four samples currently have both prediction and AFM truth: `N6342`, `N6358`, `N6382`, and `N6389`.

The current prospective validation is provisional. Three matched samples (`N6342`, `N6358`, `N6382`) have negative raw Rq predictions. This is recorded as `negative_raw_model_output` and is consistent with the current interpretation that these RHEED patterns are unusually streaky relative to the training distribution. For physical map rendering, negative Rq is clipped to nonnegative values, but the raw negative outputs should be reported as a failure mode, not hidden.

## Directory and Code Architecture

### Frozen Retrospective Package

`publication_freeze/rheed_afm_single_frame_v1_2026-07-18` is the immutable publication freeze for retrospective evidence. Important subdirectories are:

| Path | Role |
|---|---|
| `00_README/README.md` | Top-level freeze summary and canonical headline metrics. |
| `01_SUMMARY/`, `02_MODEL_AND_DATA/`, `03_METHODS/`, `04_RESULTS/`, `05_CLAIMS_AND_LIMITATIONS/` | Human-readable frozen summaries. |
| `data_snapshot/` | Minimal frozen data snapshot: sample index, target values, selected keyframes, representative AFM maps, schemas, and split assignments. |
| `models/encoder/` | Frozen DINO embedding bank and encoder card. |
| `models/quantitative_model/` | Ridge ensemble specification, OOF member predictions, and full-cohort deployment artifacts. |
| `models/visual_model/` | A3 retrieval specification and full-cohort AFM retrieval bank. |
| `results/strict_oof/` | Strict retrospective predictions, errors, metrics, retrieval results, and uncertainty summaries. |
| `figures/`, `tables/`, `report/` | Frozen manuscript figures, tables, and HTML/Markdown report. |
| `provenance/` | Source map, manifest hashes, git info, verification report, and pre-freeze audit. |
| `code/verify_freeze.py` | Freeze verification script. |
| `code/reproduce_frozen_results.py` | Reproduction script for frozen outputs. |
| `code/regenerate_figures.py` | Figure regeneration script. |

The frozen package explicitly states that prospective unseen deployment was blocked in the original freeze because earlier unseen scripts were not validated as canonical deployment paths. The later `publication_freeze/prospective_unseen_single_frame_v1` package is a separate prospective execution package built after the original freeze.

### Prospective Package

`publication_freeze/prospective_unseen_single_frame_v1` contains the current unseen selection, prediction, retrieval, and AFM truth comparison artifacts.

| Path | Role |
|---|---|
| `code/launch_keyframe_selector.py` | GUI entry point for manual keyframe/ROI selection. |
| `code/keyframe_selector/` | GUI support, metadata, decoding, provenance, and manifest helpers. |
| `code/validate_keyframe_selections.py` | Selection validation. |
| `code/finalize_keyframe_selections.py` | Finalizes selected frames into package artifacts. |
| `code/predict_full_cohort_unseen.py` | Current full-cohort unseen RHEED-to-Rq inference script. |
| `code/generate_full_cohort_retrieval_images.py` | Current full-cohort A3-style retrieval and AFM map rendering script. |
| `manifests/unseen_keyframe_manifest.csv` | Five selected prospective RHEED keyframes. |
| `metadata/samples/*.json` | Per-sample selection metadata. |
| `keyframes/raw/`, `keyframes/roi/`, `keyframes/model_ready/` | Raw selected frames, manually cropped ROI images, and model-ready 224x224 luminance arrays. |
| `predictions/full_cohort_single_frame_v1/` | Predictions, member predictions, embeddings, provenance, and retrieval maps. |
| `ground_truth_afm/` | Newly processed AFM truth for the extra AFM batch and prediction/truth joins. |

### AFM Processing Code

The AFM-processing route used for both historical data and the new AFM truth batch is:

1. Raw Bruker/Nanoscope extraction:
   - Script: `scripts/batch_extract_afm_by_sample.py`
   - Core implementation: `src/rheed2morph/afm/inspect.py`
   - Preferred channel: `ZSensor`
   - Fallback channel: `Height`
   - Output: `*_height.npy`, render PNGs, metadata JSON, inspection text.
2. First-order plane correction:
   - Script: `scripts/afm_plane_correct.py`
   - Input: `*_height.npy`
   - Output: `*_plane_corrected.npy`, fitted plane array, render PNG, metadata JSON.
3. Second-order background subtraction:
   - Script: `scripts/fit_afm_second_order.py`
   - Input: raw physical AFM `*_height.npy` files with `primary_channel == ZSensor`
   - Model: robust second-order background subtraction with terms selected by `--model y2`
   - Output: second-order corrected `*_height.npy`, background arrays, metadata, QC grid, processing manifest.

The current publication target uses second-order corrected physical height arrays in nm, not rendered PNG or TIFF images.

## Historical Dataset

### Strict Cohort

The strict frozen retrospective cohort contains 23 nonremoved historical growth groups:

`6022`, `6028`, `6029`, `6033`, `6047`, `6048`, `6056`, `6057`, `6062`, `6063`, `6070`, `6072`, `6078`, `6080`, `6081`, `6082`, `6084`, `6085`, `6090`, `6094`, `6095`, `6099`, `6101`.

The full canonical index has 27 rows, with 2 rows marked removed. The strict publication subset has N = 23. The modeling unit is the growth group/sample ID; strict OOF evaluation holds out one growth group at a time and trains on the other 22 groups.

### RHEED Inputs

Each strict sample contributes one manually selected RHEED keyframe ROI:

| Item | Value |
|---|---:|
| Strict RHEED samples | 23 |
| Model-ready keyframe array shape | `[1, 224, 224]` |
| Model-ready dtype | `uint8` |
| ROI width, min/median/max | 249 / 516 / 639 px |
| ROI height, min/median/max | 391 / 783 / 1026 px |

Preprocessing is:

1. Manual ROI crop from selected RHEED frame.
2. Convert RGB to luminance using `0.2126 R + 0.7152 G + 0.0722 B`.
3. Resize while preserving aspect ratio so the longer edge is 224 px.
4. Zero-pad to 224 x 224.
5. Convert to 3-channel ImageNet-normalized tensor for DINOv2.

The frozen embedding bank is:

| Array | Shape | Dtype |
|---|---:|---|
| `embeddings` | `[23, 1536]` | `float32` |
| `sample_ids` | `[23]` | string |
| `growth_run_ids` | `[23]` | string |
| `input_shape` | `[3]` | `int64` |

The 1536-D feature vector is produced by temporal aggregation of DINO frame embeddings:

`feature = concat(mean, std, delta_last_minus_first, linear_time_slope)`.

For the final single-frame variant, the DINO frame embedding dimension is 384, so the aggregated feature dimension is `4 x 384 = 1536`. With one frame, the standard-deviation, delta, and slope components are zero-valued by construction, but they are retained for compatibility with the frozen feature schema and model artifacts.

### AFM Targets

Historical AFM target values are second-order corrected Rq values in nm. The frozen target column is `T4_second_order_trimmed_mean`.

| Statistic | Value |
|---|---:|
| Count | 23 |
| Min | 1.011548 nm |
| Q25 | 1.786071 nm |
| Median | 2.808947 nm |
| Mean | 3.461093 nm |
| Q75 | 4.099714 nm |
| Max | 10.296223 nm |
| Std | 2.399461 nm |

The representative AFM arrays in `data_snapshot/representative_afm/` have:

| Shape | Count |
|---|---:|
| `[256, 256]` | 21 |
| `[512, 512]` | 2 |

All representative AFM arrays in the frozen snapshot are `float64`. The recomputed representative-array Rq distribution is close to the frozen target distribution, but the target table is authoritative because it can aggregate multiple scans per sample.

Historical AFM target scan counts per sample range from 2 to 26. Examples:

| Sample | Target Rq nm | Scan count | Scan Rq MAD nm | Scan Rq IQR nm |
|---|---:|---:|---:|---:|
| 6022 | 1.447127 | 6 | 0.490076 | 0.807126 |
| 6047 | 4.191616 | 8 | 1.340669 | 3.554280 |
| 6056 | 2.764360 | 26 | 0.142206 | 0.310455 |
| 6095 | 7.420876 | 8 | 3.853116 | 6.704930 |
| 6099 | 10.267897 | 5 | 0.129515 | 0.174052 |

## Model Architecture

### Frozen Encoder: DINOv2 ViT-S/14

The encoder is `facebookresearch/dinov2:dinov2_vits14`, loaded through `torch.hub`. It is fully frozen during both retrospective feature extraction and prospective inference.

Local parameter audit:

| Component | Value |
|---|---:|
| Model class | `DinoVisionTransformer` |
| Total DINO parameters | 22,056,576 |
| Trainable parameters during this pipeline | 0 |
| Embedding dimension | 384 |
| Transformer blocks | 12 |
| Attention heads | 6 |
| Patch size | 14 x 14 |
| Patch stride | 14 x 14 |
| Patch projection weight | `[384, 3, 14, 14]` |
| Patch projection parameters | 226,176 including bias |
| Stored positional embedding shape | `[1, 1370, 384]` |
| Token and final norm parameters | 527,616 |
| Parameters per transformer block | 1,775,232 |
| Total transformer block parameters | 21,302,784 |

For a 224 x 224 model input, the patch grid is 16 x 16 = 256 patches. The transformer sequence is 257 tokens after adding the CLS token. The DINO output used here is a 384-D image-level embedding. The frozen feature aggregator maps this to 1536-D by concatenating mean, standard deviation, first-last delta, and temporal slope.

One representative transformer block has the following parameter tensors:

| Tensor | Shape |
|---|---:|
| `norm1.weight` | `[384]` |
| `norm1.bias` | `[384]` |
| `attn.qkv.weight` | `[1152, 384]` |
| `attn.qkv.bias` | `[1152]` |
| `attn.proj.weight` | `[384, 384]` |
| `attn.proj.bias` | `[384]` |
| `ls1.gamma` | `[384]` |
| `norm2.weight` | `[384]` |
| `norm2.bias` | `[384]` |
| `mlp.fc1.weight` | `[1536, 384]` |
| `mlp.fc1.bias` | `[1536]` |
| `mlp.fc2.weight` | `[384, 1536]` |
| `mlp.fc2.bias` | `[384]` |
| `ls2.gamma` | `[384]` |

### Quantitative Regression Model

The quantitative model is a five-member Ridge ensemble:

| Item | Value |
|---|---|
| Model family | Ridge regression |
| Feature scaler | StandardScaler-style feature mean and scale stored per member |
| Input feature dimension | 1536 |
| Output | Scalar Rq in nm |
| Ensemble aggregation | Median of five member predictions in Rq nm space |
| Full-cohort training samples | 23 |
| Strict OOF training samples per heldout sample | 22 |

The selected frozen member artifacts are:

| Member | Trial | Stored target variant | Coef shape | Regression params |
|---|---|---|---:|---:|
| `model_01_trial_0004.npz` | `trial_0004` | `T4_second_order_trimmed_mean` | `[1536]` | 1537 |
| `model_02_trial_0012.npz` | `trial_0012` | `T4_second_order_trimmed_mean` | `[1536]` | 1537 |
| `model_03_trial_0006.npz` | `trial_0006` | `T6_quality_weighted_second_order` | `[1536]` | 1537 |
| `model_04_trial_0014.npz` | `trial_0014` | `T6_quality_weighted_second_order` | `[1536]` | 1537 |
| `model_05_trial_0028.npz` | `trial_0028` | `T4_second_order_trimmed_mean` | `[1536]` | 1537 |

Each member stores:

| Stored array | Shape | Dtype |
|---|---:|---|
| `coef` | `[1536]` | `float64` |
| `intercept` | scalar | `float64` |
| `feature_mean` | `[1536]` | `float64` |
| `feature_scale` | `[1536]` | `float64` |
| `training_sample_ids` | `[23]` | string |
| `target_variant` | scalar | string |

Parameter count:

| Scope | Count |
|---|---:|
| Trainable regression parameters per member, including intercept | 1,537 |
| Trainable regression parameters across 5 members | 7,685 |
| Stored numeric parameters per member, including scaler | 4,609 |
| Stored numeric parameters across 5 members, including scaler | 23,045 |

The Ridge model is linear after fixed DINO feature extraction and feature standardization:

`z = (x - feature_mean) / max(feature_scale, 1e-12)`

`prediction_member = dot(z, coef) + intercept`

`prediction_ensemble = median(prediction_member_1 ... prediction_member_5)`

### Visual Retrieval Model

The visual output is not generated by an image decoder. It is a retrieval-and-rescaling model based on real historical AFM maps.

The strict retrospective visual method is A3. For a heldout sample, the heldout sample group is excluded from the retrieval bank. The retrieval bank condition vector uses 11 AFM descriptors:

1. `rq_nm`
2. `ra_nm`
3. `robust_height_range_nm`
4. `psd_low_fraction`
5. `psd_mid_fraction`
6. `psd_high_fraction`
7. `psd_slope`
8. `correlation_length_nm`
9. `anisotropy`
10. `height_skewness`
11. `height_kurtosis`

The A3 rank score is:

`sqrt(sum(((bank_descriptor - condition_descriptor) / training_bank_std)^2)) + 0.05 * abs(bank_rq_nm - condition_rq_nm)`

For prospective full-cohort deployment, source groups allowed = 23. The AFM bank has:

| Item | Value |
|---|---:|
| Rows | 116 |
| Sample groups | 23 |
| Quality-pass rows | 116 |
| Rq min / median / max | 0.706984 / 2.825301 / 20.481377 nm |

The prospective retrieval code fills non-Rq conditioning descriptors with full-cohort bank medians and sets the Rq coordinate to the predicted nonnegative Rq. It then selects the closest AFM source, projects its morphology to unit Rq, and rescales it to the predicted Rq:

`unit = (source_map - mean(source_map)) / (Rq(source_map) + epsilon)`

`retrieved_map = predicted_Rq_nm * unit`

If raw predicted Rq is negative, the prediction table records the negative value, but the physical map generation clips it to a small nonnegative value for rendering.

## Retrospective Experimental Results

The strict OOF evaluation uses 23 heldout predictions. Each heldout sample is predicted using models trained without that growth group. The frozen strict quantitative result is the primary retrospective evidence.

### Error Distribution

| Statistic | Absolute error nm |
|---|---:|
| Count | 23 |
| Min | 0.015686 |
| Q25 | 0.391751 |
| Median | 1.120567 |
| Mean | 1.260098 |
| Q75 | 1.552503 |
| Max | 6.542426 |
| Std | 1.339856 |

The largest strict OOF error is sample `6099`, where the true Rq is 10.267897 nm and the predicted q50 Rq is 3.725471 nm, giving an absolute error of 6.542426 nm. This is consistent with the known high-Rq underestimation and dynamic-range compression failure mode. The high-tail MAE is 2.166008 nm, higher than the low-tail MAE of 1.062787 nm.

### Claims Supported by the Retrospective Freeze

Supported:

- A single manually selected RHEED ROI keyframe contains predictive information about final AFM Rq.
- The frozen DINOv2 + Ridge ensemble provides a strict OOF MAE of 1.260098 nm on the 23-sample strict cohort.
- The A3 retrieval route can produce physically interpretable representative AFM maps by retrieving and rescaling real historical AFM morphologies.

Not supported:

- Full-cohort training-fit performance as an independent test result.
- Fully calibrated uncertainty intervals from member q10/q90 spread.
- Reliable extrapolation to very high Rq or strongly out-of-domain RHEED patterns.
- Claiming generated AFM images are direct pixel-accurate reconstructions. They are representative retrieval maps conditioned by predicted morphology descriptors, dominated by predicted Rq.

## Prospective Five-Point Experiment

### Current Input and Prediction State

The prospective RHEED prediction package currently contains five RHEED unseen samples:

`N6342`, `N6358`, `N6382`, `N6389`, `N6390`.

The full-cohort prediction script confirms `uses_all_23_training_samples == True` for every row. The historical training set for prospective deployment is:

`6022`, `6028`, `6029`, `6033`, `6047`, `6048`, `6056`, `6057`, `6062`, `6063`, `6070`, `6072`, `6078`, `6080`, `6081`, `6082`, `6084`, `6085`, `6090`, `6094`, `6095`, `6099`, `6101`.

Prospective predictions:

| Sample | Raw predicted Rq nm | Clipped nonnegative Rq nm | Flag | Display transform |
|---|---:|---:|---|---|
| N6342 | -2.224453 | 0.000000 | `negative_raw_model_output` | none |
| N6358 | -1.909976 | 0.000000 | `negative_raw_model_output` | none |
| N6382 | -1.670822 | 0.000000 | `negative_raw_model_output` | none |
| N6389 | 2.202156 | 2.202156 | `ok` | rotate_clockwise_90 |
| N6390 | 2.589034 | 2.589034 | `ok` | rotate_clockwise_90 |

Three of the five raw predictions are negative. This is physically impossible for Rq and should be treated as a model/domain failure. The current interpretation is that these three RHEED patterns are too streaky relative to the training distribution; the linear Ridge head maps their DINO features outside the calibrated target range. The clipping step prevents nonsensical negative amplitude in visual retrieval, but it does not fix the quantitative failure.

### Current AFM Truth State

The newly supplied AFM batch is under `data/AFM-extra-five`. It contains raw AFM files for:

`N6324`, `N6342`, `N6358`, `N6382`, `N6389`.

There is a sample-ID mismatch with the RHEED prediction set:

| Status | Samples |
|---|---|
| Prediction and AFM truth both available | `N6342`, `N6358`, `N6382`, `N6389` |
| AFM truth available but no current prediction | `N6324` |
| Prediction available but AFM truth missing | `N6390` |

The expectation is that `N6390` AFM will be added in the next few days. After adding `N6390`, the intended five-point prospective validation set should be complete. Until then, the current matched validation set has N = 4.

### AFM Truth Processing for Extra Batch

The new AFM truth was processed using the same raw AFM route:

| Stage | Output path | Count |
|---|---|---:|
| Pair-like symlink entry | `data/pair_afm_extra_five` | 25 symlinked raw files |
| Raw ZSensor extraction | `data/processed_afm_extra_five` | 25 success, 0 failed |
| First-order plane correction | `data/plane_corrected_afm_extra_five` | 25 corrected maps |
| Second-order background subtraction | `data/afm_second_order_extra_five` | 25 corrected maps |
| Paper comparison package | `publication_freeze/prospective_unseen_single_frame_v1/ground_truth_afm` | 5 sample summaries |

All 25 AFM files parsed as:

| Property | Value |
|---|---|
| Primary channel | `ZSensor` |
| Scan size | 2.0 x 2.0 um |
| Resolution | 512 x 512 |
| Height unit | nm |
| Second-order arrays finite | yes |

Representative AFM selection follows the previous target-building convention: compute the median second-order Rq across the five scans for each sample, then select the scan closest to that median as the representative AFM.

Current AFM truth summaries:

| Sample | Scan count | True median second-order Rq nm | IQR nm | Representative scan |
|---|---:|---:|---:|---|
| N6324 | 5 | 1.486911 | 0.158409 | `N6324_2um_5_000` |
| N6342 | 5 | 1.272999 | 0.606784 | `N6342_2um_4_000` |
| N6358 | 5 | 1.591984 | 0.269545 | `N6358_2um_5_000` |
| N6382 | 5 | 1.577852 | 0.327553 | `N6382_2um_4_000` |
| N6389 | 5 | 2.628967 | 0.253652 | `N6389_2um_3_000` |

The five AFM truth samples have median Rq distribution:

| Statistic | Rq nm |
|---|---:|
| Min | 1.272999 |
| Q25 | 1.486911 |
| Median | 1.577852 |
| Mean | 1.711743 |
| Q75 | 1.591984 |
| Max | 2.628967 |
| Std | 0.472571 |

The all-scan preview images include 500 nm scale bars and explicitly state that each field is 2 um x 2 um:

`publication_freeze/prospective_unseen_single_frame_v1/ground_truth_afm/all_scan_previews/`

### Current Matched Prospective Accuracy

Only four samples currently have both prediction and AFM truth: `N6342`, `N6358`, `N6382`, `N6389`.

| Sample | Raw predicted Rq nm | Clipped predicted Rq nm | True median Rq nm | Clipped absolute error nm | Flag |
|---|---:|---:|---:|---:|---|
| N6342 | -2.224453 | 0.000000 | 1.272999 | 1.272999 | `negative_raw_model_output` |
| N6358 | -1.909976 | 0.000000 | 1.591984 | 1.591984 | `negative_raw_model_output` |
| N6382 | -1.670822 | 0.000000 | 1.577852 | 1.577852 | `negative_raw_model_output` |
| N6389 | 2.202156 | 2.202156 | 2.628967 | 0.426811 | `ok` |

Current N = 4 matched provisional metrics:

| Metric | Raw predictions | Clipped nonnegative predictions |
|---|---:|---:|
| MAE | 2.668724 nm | 1.217411 nm |
| RMSE | 2.967825 nm | 1.306400 nm |

These metrics should not be presented as the final prospective validation result. They are useful as a diagnosis of model behavior on the currently available matched subset. The final prospective report should be regenerated after `N6390` AFM is added and the extraneous `N6324` status is resolved.

## Interpretation of the Negative Predictions

The three negative raw predictions are important. Rq is nonnegative by definition, so a negative linear-model output means the feature vector has landed outside the range where the Ridge head is physically well constrained. This is not merely a cosmetic issue in the plotting code.

Likely contributing factors:

- The model is a linear Ridge head over frozen DINO features. It has no nonnegative output constraint.
- The training cohort is small: 23 historical growth groups.
- The prospective RHEED patterns for `N6342`, `N6358`, and `N6382` are described as very streaky. This may place them outside or near the edge of the training feature distribution.
- The frozen retrospective model already shows high-tail underestimation and dynamic-range compression; the same limited calibration can also produce implausible extrapolations in prospective inputs.

Recommended manuscript language:

- Report raw negative predictions explicitly.
- State that nonnegative clipping was used only for physical-map visualization/retrieval.
- Treat the three negative outputs as a prospective failure mode of the current linear Rq head.
- Avoid reporting clipped prospective performance alone without explaining the raw model outputs.

## Data Structures and Artifacts for Manuscript Figures

Recommended source files for final figures and tables:

| Purpose | File |
|---|---|
| Strict retrospective metrics | `publication_freeze/rheed_afm_single_frame_v1_2026-07-18/results/strict_oof/metrics.json` |
| Strict per-sample predictions/errors | `publication_freeze/rheed_afm_single_frame_v1_2026-07-18/results/strict_oof/per_sample_errors.csv` |
| Frozen sample index | `publication_freeze/rheed_afm_single_frame_v1_2026-07-18/data_snapshot/canonical_sample_index.csv` |
| Frozen target table | `publication_freeze/rheed_afm_single_frame_v1_2026-07-18/data_snapshot/sample_targets.csv` |
| Full-cohort prospective predictions | `publication_freeze/prospective_unseen_single_frame_v1/predictions/full_cohort_single_frame_v1/predictions.csv` |
| Full-cohort prospective member predictions | `publication_freeze/prospective_unseen_single_frame_v1/predictions/full_cohort_single_frame_v1/ensemble_member_predictions.csv` |
| Prospective retrieval outputs | `publication_freeze/prospective_unseen_single_frame_v1/predictions/full_cohort_single_frame_v1/retrieval/retrieval_results.csv` |
| Extra AFM truth sample summary | `publication_freeze/prospective_unseen_single_frame_v1/ground_truth_afm/manifests/afm_extra_five_sample_level_ground_truth.csv` |
| Extra AFM scan-level manifest | `publication_freeze/prospective_unseen_single_frame_v1/ground_truth_afm/manifests/afm_extra_five_second_order_scan_manifest.csv` |
| Prediction/truth join | `publication_freeze/prospective_unseen_single_frame_v1/ground_truth_afm/manifests/full_cohort_prediction_vs_afm_truth_join.csv` |
| Sample-ID mismatch report | `publication_freeze/prospective_unseen_single_frame_v1/ground_truth_afm/manifests/sample_id_mismatch_report.json` |
| AFM all-scan previews with scale bars | `publication_freeze/prospective_unseen_single_frame_v1/ground_truth_afm/all_scan_previews/*.png` |
| Representative AFM maps | `publication_freeze/prospective_unseen_single_frame_v1/ground_truth_afm/representative_maps/*.npy` and `*.png` |

## Recommended Next Steps Before Final Submission

1. Add and process `N6390` AFM using the same AFM-extra route.
2. Regenerate `ground_truth_afm` manifests and the prediction/truth join after `N6390` is present.
3. Decide whether `N6324` should be predicted as an additional sixth unseen RHEED sample or excluded as an unmatched AFM-only sample.
4. Recompute final prospective metrics on exactly the intended five matched samples.
5. Include both raw and clipped prospective predictions in the supplementary table.
6. Add a failure-mode paragraph explaining negative Rq outputs for streaky RHEED patterns.
7. Consider a constrained or transformed regression head for future work, such as predicting `log(Rq)` or using nonnegative post-calibration, but keep this separate from the frozen publication result.

## Reproducibility Notes

The following checks were completed while generating this report:

- DINOv2 ViT-S/14 loaded locally via `torch.hub` and parameter count was computed.
- Frozen DINO embedding bank shape was verified as `[23, 1536]`.
- All five full-cohort Ridge deployment members were loaded and had `coef.shape == [1536]`.
- Each full-cohort Ridge member had 23 training sample IDs.
- Prospective prediction rows all recorded `uses_all_23_training_samples == True`.
- New AFM truth processing produced 25 successful ZSensor extractions, 25 first-order corrected arrays, and 25 second-order corrected arrays.
- New second-order AFM truth arrays were verified finite with shape `[512, 512]`.
- All five all-scan AFM preview images were regenerated with 500 nm scale bars.

