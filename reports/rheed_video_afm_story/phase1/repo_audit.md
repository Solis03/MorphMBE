# Phase 1 Repository Audit

- Selected manifest: `data/rheed_keyframe_selection/selected_roi_manifest.csv`
- Selected manifest schema: `['sample_id', 'video_id', 'source_video', 'metadata_path', 'frames_dir', 'keyframe_index', 'clip_frame_count', 'roi_x', 'roi_y', 'roi_width', 'roi_height', 'source_width', 'source_height', 'clip_start_index', 'clip_end_index', 'actual_clip_frame_count']`
- Selected sample count: 27
- Selected sample IDs: 6022, 6023, 6028, 6029, 6033, 6043, 6047, 6048, 6055, 6056, 6057, 6062, 6063, 6070, 6072, 6078, 6080, 6081, 6082, 6084, 6085, 6087, 6090, 6094, 6095, 6099, 6101
- Canonical removelist: `removelist.txt`
- Removelist hash: `8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b`
- Discarded overlap in selected manifest: []
- Removelist conflict in selected manifest: ['6023', '6087']
- Modeling-eligible selected samples after removelist: 25
- Selected samples with any valid AFM: 27
- Modeling-eligible samples with valid AFM: 25
- Samples with primary 1 x 1 um AFM: 23
- AFM scan count per sample range: 2 to 26
- AFM target source: recomputed Rq from `data/plane_corrected_afm/*/*_plane_corrected.npy`; descriptor/metadata values are retained in `afm_scan_audit.csv` for comparison.
- AFM unit conflicts: 0
- Rq existing/recomputed conflicts > 1e-5 nm: 0
- Pairing conflicts: []

## Per-Sample AFM Scan Counts

sample_id
6022     6
6023     6
6028     4
6029     4
6033     4
6043     4
6047     8
6048     2
6055     5
6056    26
6057     7
6062     6
6063     6
6070     8
6072    14
6078     8
6080     4
6081     6
6082     3
6084     7
6085     4
6087     2
6090     3
6094    16
6095     8
6099     5
6101     4

## Warnings

- selected samples in canonical removelist and excluded from modeling: ['6023', '6087']
