# MVP-12 RHEED frame selection v2 report

Generated: 2026-07-09T12:52:18+00:00

MVP-12 improves RHEED frame selection and variable-K shape-bag inputs. It does not train or validate a RHEED-to-AFM prediction model.

## Git Status Before

```text
?? src/rheed2morph/rheed/build_shape_bag_inputs_v2.py
?? src/rheed2morph/rheed/frame_quality_v2.py
?? src/rheed2morph/rheed/rheed_shape_bag_dataset_v2.py
?? src/rheed2morph/rheed/select_representative_frames_v2.py
?? tests/test_rheed_frame_selection_v2.py
```

## Git Status After Selection

```text
?? reports/rheed_frame_selection_v2_mvp/
?? src/rheed2morph/rheed/build_shape_bag_inputs_v2.py
?? src/rheed2morph/rheed/frame_quality_v2.py
?? src/rheed2morph/rheed/rheed_shape_bag_dataset_v2.py
?? src/rheed2morph/rheed/select_representative_frames_v2.py
?? tests/test_rheed_frame_selection_v2.py
```

## Files Created Or Modified

- `src/rheed2morph/rheed/frame_quality_v2.py`
- `src/rheed2morph/rheed/select_representative_frames_v2.py`
- `src/rheed2morph/rheed/build_shape_bag_inputs_v2.py`
- `src/rheed2morph/rheed/rheed_shape_bag_dataset_v2.py`
- `tests/test_rheed_frame_selection_v2.py`
- Per-sample `frame_selection_v2/` folders under `data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256`
- `reports/rheed_frame_selection_v2_mvp/20260709_125138/video_inventory_v2.csv`
- `reports/rheed_frame_selection_v2_mvp/20260709_125138/frame_selection_summary_v2.csv`
- `reports/rheed_frame_selection_v2_mvp/20260709_125138/failed_videos_v2.csv`
- `reports/rheed_frame_selection_v2_mvp/20260709_125138/global_status_summary.json`
- `reports/rheed_frame_selection_v2_mvp/20260709_125138/global_accepted_overview_grid.png`
- `reports/rheed_frame_selection_v2_mvp/20260709_125138/global_rejected_artifact_examples.png`
- `reports/rheed_frame_selection_v2_mvp/20260709_125138/global_low_confidence_samples.png`
- `reports/rheed_frame_selection_v2_mvp/20260709_125138/specific_sample_audit.md`

## Exact Commands Run

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.select_representative_frames_v2 --video-root data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 --out-root data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 --max-candidates 16 --max-frames-per-video 600 --debug --seed 42
```

Additional test and shape-bag commands are appended after the shape-bag v2 run.

## Environment

| package | version |
| --- | --- |
| python | 3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0] |
| platform | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 |
| numpy | 2.4.5 |
| scipy | 1.17.1 |
| skimage | 0.26.0 |
| cv2 | not available |
| torch | 2.12.0+cu130 |

## Input Inventory

- Video root: `data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256`
- MP4 files found: 62
- Number processed: 4
- Failures: 0
- Used all-MP4 fallback: False

## Difference From V1

- No forced 16 accepted frames; accepted bags are variable-K up to `16`.
- Hard artifact rejection separates validity from information quality.
- Temporal consistency penalizes isolated quality spikes when enabled.
- Accepted and rejected frames are written to separate CSVs, grids, and folders.
- Shape-bag v2 uses accepted rows and preserves valid frames with `frame_mask`.

## Frame Selection Summary

- GOOD: 4
- USABLE: 0
- LOW_CONFIDENCE: 0
- EXCLUDE: 0
- Average accepted frames per sample: 16
- Min/max accepted frames: 16 / 16
- Common rejection reasons: `{'not_selected_lower_rank': 1084, 'no_plausible_rheed_pattern': 844, 'almost_black': 396, 'strong_shadow': 368, 'very_low_dynamic_range': 348, 'blocky_artifact': 274, 'largest_rectangular_component_too_large': 205, 'binary_artifact': 200, 'extreme_pixel_fraction_too_high': 141, 'too_few_gray_levels': 19}`
- Binary/block artifacts rejected: 474

## Specific Sample Audit

See `reports/rheed_frame_selection_v2_mvp/20260709_125138/specific_sample_audit.md` for N6041, N6047, N6027, and N6043 if present.

## Manual Review Workflow

Review LOW_CONFIDENCE and EXCLUDE samples first, then inspect accepted/rejected grids for samples with many binary or block artifacts.

- Accepted grids: `<sample>/frame_selection_v2/accepted_candidate_frames_grid.png`
- Rejected grids: `<sample>/frame_selection_v2/rejected_bad_frames_grid.png`
- Summary table: `reports/rheed_frame_selection_v2_mvp/20260709_125138/frame_selection_summary_v2.csv`

## Known Limitations

- The detector is a transparent heuristic gate, not a trained RHEED physics model.
- Some unusual but real patterns may need manual review if they resemble saturated or block-like artifacts.
- Temporal consistency uses local image similarity; abrupt real changes can be down-weighted.

## Recommended Next Command For Future Model Run

After manual review, run the future supervised experiment against `rheed_shape_input_v2/shape_bag_v2.npz` files listed in `rheed_shape_bag_manifest_v2.csv`.

## Git Status After Report Write

```text
?? reports/rheed_frame_selection_v2_mvp/
?? src/rheed2morph/rheed/build_shape_bag_inputs_v2.py
?? src/rheed2morph/rheed/frame_quality_v2.py
?? src/rheed2morph/rheed/rheed_shape_bag_dataset_v2.py
?? src/rheed2morph/rheed/select_representative_frames_v2.py
?? tests/test_rheed_frame_selection_v2.py
```
