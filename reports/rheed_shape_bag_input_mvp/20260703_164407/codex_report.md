# RHEED shape-bag input MVP report

Generated: 2026-07-03T16:48:56+00:00

This task builds exposure-invariant multi-frame RHEED shape-bag inputs. It does not train or validate a RHEED-to-AFM prediction model.

## Git Status

Before:

```text
?? reports/rheed_frame_selection_mvp/
?? reports/rheed_shape_bag_input_mvp/
?? src/rheed2morph/rheed/build_shape_bag_inputs.py
?? src/rheed2morph/rheed/frame_quality.py
?? src/rheed2morph/rheed/manual_frame_selection.py
?? src/rheed2morph/rheed/models/
?? src/rheed2morph/rheed/pretrain_shape_bag_invariance.py
?? src/rheed2morph/rheed/rheed_shape_bag_dataset.py
?? src/rheed2morph/rheed/select_representative_frames.py
?? src/rheed2morph/rheed/shape_preprocessing.py
?? src/rheed2morph/rheed/spot_streak_geometry.py
?? tests/test_rheed_frame_selection.py
?? tests/test_rheed_shape_bag_input.py
```

After:

```text
?? reports/rheed_frame_selection_mvp/
?? reports/rheed_shape_bag_input_mvp/
?? src/rheed2morph/rheed/build_shape_bag_inputs.py
?? src/rheed2morph/rheed/frame_quality.py
?? src/rheed2morph/rheed/manual_frame_selection.py
?? src/rheed2morph/rheed/models/
?? src/rheed2morph/rheed/pretrain_shape_bag_invariance.py
?? src/rheed2morph/rheed/rheed_shape_bag_dataset.py
?? src/rheed2morph/rheed/select_representative_frames.py
?? src/rheed2morph/rheed/shape_preprocessing.py
?? src/rheed2morph/rheed/spot_streak_geometry.py
?? tests/test_rheed_frame_selection.py
?? tests/test_rheed_shape_bag_input.py
```

## Commands

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.build_shape_bag_inputs --root data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 --out-root data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 --mode manual_or_candidates --candidate-count 16 --image-size 256 --seed 42 --exposure-audit true --make-global-report true
```

## Environment

| package | version |
| --- | --- |
| python | 3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0] |
| platform | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 |
| numpy | 2.4.5 |
| scipy | 1.17.1 |
| skimage | 0.26.0 |
| cv2 | not available |
| torch | 2.12.0+cu130 |

## Input Inventory

- Sample folders found: 62
- Candidate CSVs found: 62
- Manual selections used: 0
- Auto-candidate fallbacks: 62
- Samples processed: 62
- Failures: 0

## Representation Design

Preprocessing channels: `pclip_norm, log_bgsub, local_zscore, dog_response, ridge_or_edge_response, soft_spot_streak_mask`. Background subtraction uses `log1p(alpha * pclip_norm)` followed by Gaussian low-frequency background subtraction and robust rescaling. Local normalization uses local mean/std z-scoring with a 31-pixel default window. Mask extraction combines positive background-subtracted signal, DoG response, local z-score, and artifact masking. Components are classified with deterministic aspect-ratio, eccentricity, orientation, area, border, and artifact rules.

## Multi-Frame Aggregation

Frame weights use `candidate_quality_score * mask_confidence * non_artifact_score * SNR_score`, with low-confidence candidates penalized. Scalar shape features save weighted mean, weighted median, trimmed mean, weighted std, and IQR. Consensus maps include weighted mean log-bgsub, median log-bgsub, mask vote, persistent mask, DoG max response, and uncertainty. Translation alignment is currently not applied; all selected crop frames are assumed pre-cropped and comparable.

## Exposure Invariance

Perturbations tested: brightness scales, contrast scales, gamma shifts, low-frequency gradient, mild noise, and mild blur. Raw brightness stability and shape-feature stability are summarized in `reports/rheed_shape_bag_input_mvp/20260703_164407/global_exposure_invariance_summary.csv`. Exposure-sensitive feature names are recorded in each sample's `exposure_invariance_audit.json` and excluded from that sample's audit-recommended feature list.

## Geometry Feature Summary

- Frame feature count: 48
- Low-confidence samples: 14
- Component type summary: `reports/rheed_shape_bag_input_mvp/20260703_164407/global_component_type_summary.png`
- High `bar_like_score` examples: `reports/rheed_shape_bag_input_mvp/20260703_164407/global_bar_like_score_examples.png`
- Low-quality examples: `reports/rheed_shape_bag_input_mvp/20260703_164407/global_low_quality_examples.png`

## Output Locations

- Global manifest: `reports/rheed_shape_bag_input_mvp/20260703_164407/rheed_shape_bag_manifest.csv`
- Global feature table: `reports/rheed_shape_bag_input_mvp/20260703_164407/global_sample_shape_features.csv`
- Global overview grid: `reports/rheed_shape_bag_input_mvp/20260703_164407/global_shape_bag_overview.png`
- Example `shape_input_overview.png` paths:
  - `data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256/N6011 - Copy/rheed_shape_input/shape_input_overview.png`
  - `data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256/N6018 - Copy/rheed_shape_input/shape_input_overview.png`
  - `data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256/N6019 - Copy/rheed_shape_input/shape_input_overview.png`
  - `data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256/N6022 - Copy/rheed_shape_input/shape_input_overview.png`
  - `data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256/N6023 - Copy/rheed_shape_input/shape_input_overview.png`

## Dataset And Encoder Interface

`shape_bag.npz` contains `frames [K,C,H,W]`, `frame_mask [K]`, `frame_weights [K]`, `consensus_maps [6,H,W]`, `sample_feature_vector [F]`, `sample_feature_names`, `frame_indices`, and `timestamps_sec`. `RHEEDShapeBagDataset` returns those arrays as tensors plus `sample_id` and `source_type`. `RHEEDShapeBagEncoder` accepts variable `K` with `frame_mask` and `frame_weights` for multi-instance pooling and emits `sample_embedding`, `attention_weights`, and `frame_embeddings`.

## Known Limitations

- Component rules are transparent heuristics, not a trained RHEED detector.
- Alignment is interface-ready but not enabled by default in this MVP.
- Exposure-invariance audit is diagnostic and may still flag threshold-sensitive count features.
- Manual visual verification remains required before using the representation in a supervised RHEED-to-AFM experiment.

## Recommended Next Command

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests/test_rheed_shape_bag_input.py
```
