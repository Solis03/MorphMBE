# RHEED Roughness Limitations

- Material labels are inferred from existing manifests and filenames; several
  labels are scan-size or sample-token-like values rather than clean chemistry.
- Substrate, azimuth, rotation period, exposure, and camera metadata are mostly
  unavailable, so confound adjustment is limited to measurable image proxies.
- Human validation cannot be completed until blinded ratings are entered.
- The RHEED score operates on pre-cropped videos; raw full-frame ROI failures
  are not fully represented.
- No samples were removed because they weakened the expected relationship.
- Peak-to-valley/color-bar span is reported separately and is not interpreted as
  RMS roughness.

Data audit summary: {"counts": {"afm_scan_level_targets": 260, "candidate_table_rows": 1560, "crop_video_issue_count": 0, "crop_video_issues": [], "duplicate_afm_pairing_count": 0, "duplicate_afm_pairings": [], "duplicate_rheed_pairing_count": 0, "duplicate_rheed_pairings": [], "growth_runs": 40, "metadata_json_files": 520, "paired_count": 40, "representative_pair_count": 40, "representative_pairs": 40, "rheed_videos_processed": 40, "sample_groups": 40, "unmatched_count": 0}, "data_roots": {"crop_video_root": "data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256", "pair_root": "data/pair", "plane_corrected_afm_root": "data/plane_corrected_afm", "processed_afm_root": "data/processed_afm"}, "distributions": {"afm_resolutions": {"104x104": 3, "128x128": 2, "131x132": 1, "131x144": 1, "164x164": 1, "256x256": 195, "512x512": 57}, "afm_scan_sizes_um": {"0.072": 1, "0.076": 1, "0.086": 1, "0.094": 1, "0.096": 1, "0.098": 1, "0.1": 2, "0.102": 2, "0.15": 1, "0.168": 1, "0.179": 1, "0.201": 2, "0.203": 1, "0.207": 1, "0.219": 1, "0.238": 1, "0.263": 1, "0.27": 1, "0.301": 3, "0.305": 1, "0.309": 1, "0.332": 2, "0.395": 1, "0.488": 1, "0.496": 1, "0.498": 1, "0.5": 35, "0.508": 1, "0.664": 1, "0.8": 3, "0.8

AFM audit summary: {"afm_samples_with_targets": 40, "afm_scan_level_target_count": 260, "height_unit_status_counts": {"ok": 260}, "primary_scan_size_tolerance_um": 0.1, "primary_scan_size_um": 1.0, "resolution_counts": {"104x104": 3, "128x128": 2, "131x132": 1, "131x144": 1, "164x164": 1, "256x256": 195, "512x512": 57}, "sample_level_target_count": 40, "samples_with_primary_scan_size": 36, "scan_size_counts": {"0.072": 1, "0.076": 1, "0.086": 1, "0.094": 1, "0.096": 1, "0.098": 1, "0.1": 2, "0.102": 2, "0.15": 1, "0.168": 1, "0.179": 1, "0.201": 2, "0.203": 1, "0.207": 1, "0.219": 1, "0.238": 1, "0.263": 1, "0.27": 1, "0.301": 3, "0.305": 1, "0.309": 1, "0.332": 2, "0.395": 1, "0.488": 1, "0.496": 1, "0.498": 1, "0.5": 35, "0.508": 1, "0.664": 1, "0.8": 3, "0.801": 1, "0.891": 1, "1.0": 164, "1.016": 3, "1.328": 1, "1.641": 1, "2.0": 8, "39.429": 1, "5.0": 6}}
