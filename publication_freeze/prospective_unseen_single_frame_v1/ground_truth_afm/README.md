# AFM Extra Five Ground Truth

Generated from raw AFM files in `data/AFM-extra-five` using the existing raw AFM pipeline:

1. symlink-only pair entry: `data/pair_afm_extra_five`
2. raw ZSensor extraction: `data/processed_afm_extra_five`
3. first-order plane correction: `data/plane_corrected_afm_extra_five`
4. second-order background subtraction: `data/afm_second_order_extra_five`

All 25 raw files parsed successfully as ZSensor, 512 x 512, 2.0 x 2.0 um, nm units.

Representative selection uses the prior target-building convention: sample-level median Rq across scans, then the scan closest to that median.

Important sample-id mismatch: AFM truth samples are `N6324, N6342, N6358, N6382, N6389`; existing prediction samples are `N6342, N6358, N6382, N6389, N6390`. This means `N6324` has AFM truth but no current prediction, and `N6390` has a prediction but no AFM truth in this added AFM batch.

Key files:

- `manifests/afm_extra_five_second_order_scan_manifest.csv`
- `manifests/afm_extra_five_sample_level_ground_truth.csv`
- `manifests/full_cohort_prediction_vs_afm_truth_join.csv`
- `manifests/sample_id_mismatch_report.json`
- `representative_maps/*_ground_truth_second_order_representative.png`
- `all_scan_previews/*_all_scans_second_order.png`
