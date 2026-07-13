# RHEED Roughness Data Audit

## Reused Files and Functions
- RHEED video reading: `imageio.v2.get_reader over pre-cropped crop videos`
- ROI extraction: `existing data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256 crop-video dataset`
- video stabilization: `not available; centroid shift proxies are computed`
- streak_spot_feature_extraction: `rheed2morph.rheed.spot_streak_geometry.extract_components_and_frame_features`
- rheed_preprocessing: `rheed2morph.rheed.shape_preprocessing.preprocess_frame_for_shape`
- frame_quality: `rheed2morph.rheed.frame_quality.extract_frame_quality_features`
- AFM processing: `plane-corrected height maps from data/plane_corrected_afm; Rq/Ra recomputed`
- metadata_parsing: `recursive metadata JSON schema discovery in processed and plane-corrected AFM roots`
- sample_pairing: `existing manifest_all_size_representative_one_to_one.csv and afm_candidate_table_complete.csv`

## Counts

| key | value |
| --- | ---: |
| `growth_runs` | 3 |
| `sample_groups` | 3 |
| `representative_pairs` | 3 |
| `candidate_table_rows` | 96 |
| `afm_scan_level_targets` | 16 |
| `rheed_videos_processed` | 3 |
| `metadata_json_files` | 520 |
| `representative_pair_count` | 3 |
| `paired_count` | 3 |
| `unmatched_count` | 0 |
| `crop_video_issue_count` | 0 |
| `duplicate_afm_pairing_count` | 0 |
| `duplicate_rheed_pairing_count` | 0 |
| `crop_video_issues` | [] |
| `duplicate_afm_pairings` | [] |
| `duplicate_rheed_pairings` | [] |

## Distributions

- materials: `{'Ctr': 2, 'unknown': 1}`
- AFM scan sizes: `{1.0: 9, 0.5: 6, 0.496: 1}`
- AFM resolutions: `{'256x256': 16}`
- video resolutions: `{'256x256': 3}`

## Metadata Roughness Candidates

- `raw_file`: 520
- `raw_afm_file`: 520
- `available_channels`: 520
- `available_channels[0]`: 520
- `available_channels[1]`: 520
- `primary_channel`: 520
- `secondary_channel`: 520
- `scan_size_um`: 520
- `scan_size_um[0]`: 520
- `scan_size_um[1]`: 520
- `scan_size_from_filename_um`: 520
- `height_unit_original`: 520
- `height_unit_exported`: 520
- `height_min_nm`: 520
- `height_max_nm`: 520
- `channels.ZSensor.shape`: 520
- `channels.ZSensor.shape[0]`: 520
- `channels.ZSensor.shape[1]`: 520
- `channels.ZSensor.unit`: 520
- `channels.ZSensor.scan_size_um`: 520
- `channels.ZSensor.scan_size_um[0]`: 520
- `channels.ZSensor.scan_size_um[1]`: 520
- `channels.ZSensor.stats_original.min`: 520
- `channels.ZSensor.stats_original.max`: 520
- `channels.ZSensor.stats_original.mean`: 520
- `channels.ZSensor.stats_original.std`: 520
- `channels.ZSensor.stats_nm.min`: 520
- `channels.ZSensor.stats_nm.max`: 520
- `channels.ZSensor.stats_nm.mean`: 520
- `channels.ZSensor.stats_nm.std`: 520
- `channels.ZSensor.description`: 520
- `channels.ZSensor.source`: 520
- `channels.Peak Force Error.shape`: 520
- `channels.Peak Force Error.shape[0]`: 520
- `channels.Peak Force Error.shape[1]`: 520
- `channels.Peak Force Error.unit`: 520
- `channels.Peak Force Error.scan_size_um`: 520
- `channels.Peak Force Error.scan_size_um[0]`: 520
- `channels.Peak Force Error.scan_size_um[1]`: 520
- `channels.Peak Force Error.stats_original`: 520

## Pairing Notes

- Paired rows: 3
- Unmatched rows: 0
- Crop-video issues: 0
- Ambiguous pairings are reported in `data_audit.json`; no AFM-outcome-based filtering is applied.
