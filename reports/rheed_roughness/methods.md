# RHEED Roughness Methods

## Reused Implementation

- RHEED preprocessing: `rheed2morph.rheed.shape_preprocessing.preprocess_frame_for_shape`
- Frame quality: `rheed2morph.rheed.frame_quality.extract_frame_quality_features`
- Spot/streak geometry: `rheed2morph.rheed.spot_streak_geometry.extract_components_and_frame_features`

## Reproduction Command

Run from the repository root:

```bash
PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_roughness.run --config configs/rheed_roughness.yaml
```

## Frozen RHEED Score

The pre-specified score is:

- spottiness = round_spot_count + 0.6 * elongated_spot_count + 0.25 * diffuse_blob_count
- streakiness = horizontal_bar_count + vertical_streak_count + 0.6 * elongated_spot_count + bar_like_score * max(total_component_count, 1)
- morphology_index = spottiness / (spottiness + streakiness + epsilon)

The formula was fixed before reading AFM correlations.

## AFM Target

Plane-corrected ZSensor height maps were converted to nanometers and Rq was
recomputed directly from the physical height map. Metadata and descriptor Rq
values are retained for validation.

Primary target: sample-level median Rq for scans within
1.0 +/- 0.1 um.

## Statistics

The independent unit is `sample_id`/`growth_run_id`. Primary association uses
Spearman and Kendall rank correlations, sample-level bootstrap confidence
intervals, and sample-level permutation tests. Predictive metrics use
leave-one-growth-run-out predictions only.
