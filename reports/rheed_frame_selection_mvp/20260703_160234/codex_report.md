# RHEED frame selection MVP report

Generated: 2026-07-03T16:02:42Z

## Scope

This task only selects candidate RHEED frames for human verification. It does not train or validate a RHEED-to-AFM prediction model.

## Git Status

Before run:

```text
?? src/rheed2morph/rheed/frame_quality.py
?? src/rheed2morph/rheed/manual_frame_selection.py
?? src/rheed2morph/rheed/select_representative_frames.py
?? tests/test_rheed_frame_selection.py
```

After run:

```text
?? reports/rheed_frame_selection_mvp/
?? src/rheed2morph/rheed/frame_quality.py
?? src/rheed2morph/rheed/manual_frame_selection.py
?? src/rheed2morph/rheed/select_representative_frames.py
?? tests/test_rheed_frame_selection.py
```

## Files Created Or Modified

- Per-sample `frame_selection/` folders under `data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256`
- `reports/rheed_frame_selection_mvp/20260703_160234/video_inventory.csv`
- `reports/rheed_frame_selection_mvp/20260703_160234/frame_selection_summary.csv`
- `reports/rheed_frame_selection_mvp/20260703_160234/failed_videos.csv`
- `reports/rheed_frame_selection_mvp/20260703_160234/global_candidate_overview_grid.png`
- `reports/rheed_frame_selection_mvp/20260703_160234/sample_quality_histograms.png`
- `reports/rheed_frame_selection_mvp/20260703_160234/codex_report.md`

## Exact Command

```bash
PYTHONPATH=src /home/wangziyi/MorphMBE/MorphMBE/src/rheed2morph/rheed/select_representative_frames.py --video-root data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 --out-root data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 --num-candidates 12 --max-frames-per-video 300 --debug
```

## Environment

| package | version |
| --- | --- |
| python | 3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0] |
| platform | Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39 |
| cv2 | not available |
| imageio | 2.37.3 |
| numpy | 2.4.5 |
| scipy | 1.17.1 |
| skimage | not used |

## Input Data Inventory

- Video root: `data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256`
- MP4 files found: 62
- Files matching `*raw_crop*.mp4`: 62
- Used all MP4 fallback: False
- Videos processed successfully: 3
- Failures: 0
- Example video paths:
  - `data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256/N6011 - Copy/videos/N6011 - Copy_raw_crop_256x256.mp4`
  - `data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256/N6018 - Copy/videos/N6018 - Copy_raw_crop_256x256.mp4`
  - `data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256/N6019 - Copy/videos/N6019 - Copy_raw_crop_256x256.mp4`
  - `data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256/N6022 - Copy/videos/N6022 - Copy_raw_crop_256x256.mp4`
  - `data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256/N6023 - Copy/videos/N6023 - Copy_raw_crop_256x256.mp4`

## Sample Folder Behavior

Sample IDs are inferred from the first directory under the video root when videos already live inside sample folders, or from the video filename with the raw-crop suffix removed for shared video directories. Outputs are written to `<out-root>/<sample_id>/frame_selection/`. Source videos are never moved or modified.

## Frame Scoring Summary

Features used: brightness percentiles, dynamic range, edge and center intensity ratios, dark obstruction proxies, Laplacian variance, Tenengrad gradients, local contrast, entropy, FFT low/mid/high frequency power, FFT anisotropy, projection prominence, and projection entropy.

Quality score formula:

```text
quality_score =
  0.18 * brightness_score
  + 0.20 * dynamic_range_score
  + 0.22 * sharpness_score
  + 0.22 * pattern_visibility_score
  + 0.18 * contrast_score
  - 0.35 * shadow_penalty
  - 0.25 * saturation_penalty
  - 0.20 * blur_penalty
  - 0.20 * low_dynamic_range_penalty
```

Scores are clipped to `[0, 1]`, with component normalization performed within each video. Critical rejection flags are `almost_black`, `almost_white`, `over_saturated`, `very_low_dynamic_range`, and `strong_shadow`. The requested candidate count is 12 per video, with `min_frame_gap=5` and `min_ssim_distance=0.03`.

## Quality Summary

- Low-confidence videos: 0
- Common failure modes: none
- Shadow, saturation, and blur statistics are summarized in `reports/rheed_frame_selection_mvp/20260703_160234/sample_quality_histograms.png`.

## Output Examples

- `data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256/N6011 - Copy/frame_selection/candidate_frames_grid.png`
- `data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256/N6018 - Copy/frame_selection/candidate_frames_grid.png`
- `data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256/N6019 - Copy/frame_selection/candidate_frames_grid.png`

Global overview grid: `reports/rheed_frame_selection_mvp/20260703_160234/global_candidate_overview_grid.png`

## Manual Selection Workflow

Open each sample's `frame_selection/candidate_frames_grid.png` and `candidate_frames_grid_raw_and_equalized.png`, then edit `manual_selected_frames.txt`. Uncomment `rank01` or add lines such as `frame_idx=123` for selected frame(s).

Build the manifest for future experiments with:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.manual_frame_selection --root data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 --out data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256/manual_selected_frame_manifest.csv
```

## Known Limitations

- The scoring is transparent image-quality triage, not physics-aware RHEED interpretation.
- Component normalization is per video, so scores are best compared within a sample.
- Similarity suppression uses lightweight image-distance checks for diversity.
- Manual review remains required before future experiments consume selected frames.

## Next Recommended Command

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.rheed.manual_frame_selection --root data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 --out data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256/manual_selected_frame_manifest.csv
```

## Git Status After Report Write

```text
?? reports/rheed_frame_selection_mvp/
?? src/rheed2morph/rheed/frame_quality.py
?? src/rheed2morph/rheed/manual_frame_selection.py
?? src/rheed2morph/rheed/select_representative_frames.py
?? tests/test_rheed_frame_selection.py
```
