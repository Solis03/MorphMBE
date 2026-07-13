# MVP-12 RHEED frame selection v2 report

Generated: 2026-07-09T13:01:37+00:00

MVP-12 improves RHEED frame selection and variable-K shape-bag inputs. It does not train or validate a RHEED-to-AFM prediction model.

## Git Status Before

```text
?? reports/rheed_frame_selection_v2_mvp/
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
- `reports/rheed_frame_selection_v2_mvp/20260709_125225/video_inventory_v2.csv`
- `reports/rheed_frame_selection_v2_mvp/20260709_125225/frame_selection_summary_v2.csv`
- `reports/rheed_frame_selection_v2_mvp/20260709_125225/failed_videos_v2.csv`
- `reports/rheed_frame_selection_v2_mvp/20260709_125225/global_status_summary.json`
- `reports/rheed_frame_selection_v2_mvp/20260709_125225/global_accepted_overview_grid.png`
- `reports/rheed_frame_selection_v2_mvp/20260709_125225/global_rejected_artifact_examples.png`
- `reports/rheed_frame_selection_v2_mvp/20260709_125225/global_low_confidence_samples.png`
- `reports/rheed_frame_selection_v2_mvp/20260709_125225/specific_sample_audit.md`

## Exact Commands Run

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.select_representative_frames_v2 --video-root data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 --out-root data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 --max-candidates 16 --sample-every-n-frames 1 --max-frames-per-video 1200 --display-size 256 --seed 42
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
- Number processed: 62
- Failures: 0
- Used all-MP4 fallback: False

## Difference From V1

- No forced 16 accepted frames; accepted bags are variable-K up to `16`.
- Hard artifact rejection separates validity from information quality.
- Temporal consistency penalizes isolated quality spikes when enabled.
- Accepted and rejected frames are written to separate CSVs, grids, and folders.
- Shape-bag v2 uses accepted rows and preserves valid frames with `frame_mask`.

## Frame Selection Summary

- GOOD: 36
- USABLE: 24
- LOW_CONFIDENCE: 1
- EXCLUDE: 1
- Average accepted frames per sample: 15.35
- Min/max accepted frames: 0 / 16
- Common rejection reasons: `{'no_plausible_rheed_pattern': 17757, 'not_selected_lower_rank': 10457, 'almost_black': 6262, 'strong_shadow': 5307, 'very_low_dynamic_range': 5211, 'blocky_artifact': 4438, 'largest_rectangular_component_too_large': 3649, 'binary_artifact': 3557, 'extreme_pixel_fraction_too_high': 2417, 'too_few_gray_levels': 533}`
- Binary/block artifacts rejected: 7995

## Specific Sample Audit

See `reports/rheed_frame_selection_v2_mvp/20260709_125225/specific_sample_audit.md` for N6041, N6047, N6027, and N6043 if present.

## Manual Review Workflow

Review LOW_CONFIDENCE and EXCLUDE samples first, then inspect accepted/rejected grids for samples with many binary or block artifacts.

- Accepted grids: `<sample>/frame_selection_v2/accepted_candidate_frames_grid.png`
- Rejected grids: `<sample>/frame_selection_v2/rejected_bad_frames_grid.png`
- Summary table: `reports/rheed_frame_selection_v2_mvp/20260709_125225/frame_selection_summary_v2.csv`

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


## Shape-Bag V2 Summary

Generated: 2026-07-09T13:06:23+00:00

### Shape Command

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.build_shape_bag_inputs_v2 --root data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 --out-root data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 --frame-selection-version v2 --max-frames-per-sample 16 --image-size 256 --seed 42
```

### Shape Environment

| package | version |
| --- | --- |
| python | 3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0] |
| platform | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 |
| numpy | 2.4.5 |
| scipy | 1.17.1 |
| skimage | 0.26.0 |
| cv2 | not available |
| torch | 2.12.0+cu130 |

### Shape-Bag Outputs

- Manifest: `reports/rheed_frame_selection_v2_mvp/20260709_125225/rheed_shape_bag_manifest_v2.csv`
- Global feature table: `reports/rheed_frame_selection_v2_mvp/20260709_125225/global_sample_shape_features_v2.csv`
- Default training feature names: `reports/rheed_frame_selection_v2_mvp/20260709_125225/default_training_feature_names_v2.txt`
- Shape bags written: 60
- Samples included by status: `{'GOOD': 36, 'USABLE': 24}`
- Samples excluded or failed: 2
- Manual selections used: 0
- Accepted-v2 fallback used: 60
- `pad_to_max`: True
- Example tensor shapes: `N6011 - Copy: frames=(16, 6, 256, 256) mask=(16,) valid=16 status=GOOD; N6018 - Copy: frames=(16, 6, 256, 256) mask=(16,) valid=16 status=GOOD; N6019 - Copy: frames=(16, 6, 256, 256) mask=(16,) valid=16 status=GOOD; N6022 - Copy: frames=(16, 6, 256, 256) mask=(16,) valid=16 status=GOOD; N6023 - Copy: frames=(16, 6, 256, 256) mask=(16,) valid=16 status=GOOD`

`shape_bag_v2.npz` preserves variable-K through `frame_mask`; padded all-zero frames have mask value 0 and zero frame weight.

## Git Status After Shape-Bag V2

```text
?? reports/rheed_frame_selection_v2_mvp/
?? src/rheed2morph/rheed/build_shape_bag_inputs_v2.py
?? src/rheed2morph/rheed/frame_quality_v2.py
?? src/rheed2morph/rheed/rheed_shape_bag_dataset_v2.py
?? src/rheed2morph/rheed/select_representative_frames_v2.py
?? tests/test_rheed_frame_selection_v2.py
```

## Final Verification

Initial pre-edit git status was clean. Final git status is:

```text
?? reports/rheed_frame_selection_v2_mvp/
?? src/rheed2morph/rheed/build_shape_bag_inputs_v2.py
?? src/rheed2morph/rheed/frame_quality_v2.py
?? src/rheed2morph/rheed/rheed_shape_bag_dataset_v2.py
?? src/rheed2morph/rheed/select_representative_frames_v2.py
?? tests/test_rheed_frame_selection_v2.py
```

Commands run for implementation validation and data generation:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests/test_rheed_frame_selection_v2.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.select_representative_frames_v2 --video-root data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 --out-root data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 --max-candidates 16 --max-frames-per-video 600 --debug --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.select_representative_frames_v2 --video-root data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 --out-root data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 --max-candidates 16 --sample-every-n-frames 1 --max-frames-per-video 1200 --display-size 256 --seed 42
PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.build_shape_bag_inputs_v2 --root data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 --out-root data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 --frame-selection-version v2 --max-frames-per-sample 16 --image-size 256 --seed 42
PYTHONPATH=src .venv/bin/python - <<'PY'
from pathlib import Path
from rheed2morph.rheed.rheed_shape_bag_dataset_v2 import RHEEDShapeBagDatasetV2
manifest = sorted(Path("reports/rheed_frame_selection_v2_mvp").glob("*/rheed_shape_bag_manifest_v2.csv"))[-1]
ds = RHEEDShapeBagDatasetV2(manifest)
print("n=", len(ds))
item = ds[0]
print(item["sample_id"], item["frames"].shape, item["frame_mask"].shape, item["num_valid_frames"], item["sample_status"])
PY
```

Validation results:

- Targeted v2 tests: 10 tests passed.
- Full test discovery: 107 tests passed; only unrelated numerical warnings were emitted.
- Dataset loader check: `n=60`; first sample `N6011 - Copy` loaded as `frames=[16,6,256,256]`, `frame_mask=[16]`, `num_valid_frames=16`, `sample_status=GOOD`.
- Manifest rows: 60.
- `shape_bag_v2.npz` files on disk: 60.
- Variable-K audit: `N6058 - Copy` has `num_valid_frames=7` and `frame_mask.sum()==7.0` in a padded `[16,6,256,256]` bag.
- Default exclusions: `N6066` is LOW_CONFIDENCE with 1 accepted frame and no default shape bag; `N6099` is EXCLUDE with 0 accepted frames and no default shape bag.
