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

Data audit summary: {"counts": {"afm_scan_level_targets": 16, "candidate_table_rows": 96, "crop_video_issue_count": 0, "crop_video_issues": [], "duplicate_afm_pairing_count": 0, "duplicate_afm_pairings": [], "duplicate_rheed_pairing_count": 0, "duplicate_rheed_pairings": [], "growth_runs": 3, "metadata_json_files": 520, "paired_count": 3, "representative_pair_count": 3, "representative_pairs": 3, "rheed_videos_processed": 3, "sample_groups": 3, "unmatched_count": 0}, "data_roots": {"crop_video_root": "data/rheed_roi_shadow_right_v2_main_raw_crop_videos_256", "pair_root": "data/pair", "plane_corrected_afm_root": "data/plane_corrected_afm", "processed_afm_root": "data/processed_afm"}, "distributions": {"afm_resolutions": {"256x256": 16}, "afm_scan_sizes_um": {"0.496": 1, "0.5": 6, "1.0": 9}, "height_units": {"nm": 16}, "materials": {"Ctr": 2, "unknown": 1}, "video_durations_sec": {"max": 34.03, "median": 27.0, "min": 23.37}, "video_resolutions": {"256x256": 3}}, "metadata_schema_summary": {"metadata_file_count": 520, "roughness_candidate_keys": {"available_channels": 520, "available_channels[0]": 520, "available_channels[1]": 520, "available_channels[2]": 152, "channels.Adhesion.description": 152, "chan

AFM audit summary: {"afm_samples_with_targets": 3, "afm_scan_level_target_count": 16, "height_unit_status_counts": {"ok": 16}, "primary_scan_size_tolerance_um": 0.1, "primary_scan_size_um": 1.0, "resolution_counts": {"256x256": 16}, "sample_level_target_count": 3, "samples_with_primary_scan_size": 3, "scan_size_counts": {"0.496": 1, "0.5": 6, "1.0": 9}}
