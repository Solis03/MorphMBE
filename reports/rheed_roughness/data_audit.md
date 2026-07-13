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
| `growth_runs` | 40 |
| `sample_groups` | 40 |
| `representative_pairs` | 40 |
| `candidate_table_rows` | 1560 |
| `afm_scan_level_targets` | 260 |
| `rheed_videos_processed` | 40 |
| `metadata_json_files` | 520 |
| `representative_pair_count` | 40 |
| `paired_count` | 40 |
| `unmatched_count` | 0 |
| `crop_video_issue_count` | 0 |
| `duplicate_afm_pairing_count` | 0 |
| `duplicate_rheed_pairing_count` | 0 |
| `crop_video_issues` | [] |
| `duplicate_afm_pairings` | [] |
| `duplicate_rheed_pairings` | [] |

## Distributions

- materials: `{'Ctr': 2, 'unknown': 2, '1um': 16, 'GdSb': 3, 'N56': 1, 'N58': 1, 'N62': 1, 'N63': 1, 'N65': 1, 'N66': 1, 'N68': 1, 'N69': 1, 'N73': 1, 'N74': 1, 'N80': 1, 'N81': 1, 'N87': 1, 'N88': 1, '2um': 2, '5um': 1}`
- AFM scan sizes: `{1.0: 164, 0.5: 35, 0.496: 1, 5.0: 6, 1.016: 3, 0.332: 2, 0.1: 2, 0.203: 1, 0.102: 2, 0.201: 2, 2.0: 8, 0.498: 1, 0.8: 3, 39.429: 1, 0.301: 3, 0.305: 1, 0.219: 1, 0.27: 1, 0.309: 1, 0.488: 1, 0.891: 1, 0.15: 1, 0.168: 1, 0.207: 1, 0.508: 1, 0.098: 1, 0.094: 1, 0.086: 1, 0.238: 1, 0.072: 1, 0.395: 1, 0.096: 1, 0.076: 1, 0.664: 1, 1.328: 1, 0.801: 1, 1.641: 1, 0.263: 1, 0.179: 1}`
- AFM resolutions: `{'256x256': 195, '512x512': 57, '104x104': 3, '131x132': 1, '128x128': 2, '164x164': 1, '131x144': 1}`
- video resolutions: `{'256x256': 40}`

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

- Paired rows: 40
- Unmatched rows: 0
- Crop-video issues: 0
- Ambiguous pairings are reported in `data_audit.json`; no AFM-outcome-based filtering is applied.
