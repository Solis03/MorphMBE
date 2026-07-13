# RHEED-AFM Pair Visualization

## Reused Inputs and Functions

- `outputs/rheed_roughness/frame_level_scores.parquet`
- `outputs/rheed_roughness/sample_level_analysis_table.csv`
- `outputs/rheed_roughness/afm_scan_level_targets.csv`
- `outputs/rheed_roughness/pairing_audit.csv`
- `analysis.rheed_roughness.run.read_config`, path helpers, CSV helpers, and AFM unit conversion
- `rheed2morph.rheed.shape_preprocessing.preprocess_frame_for_shape`

## RHEED Frame Selection

Frames are selected only from the previous final-window frame-level table. The
quality score combines detector confidence, sharpness, contrast/dynamic range,
valid signal, and ROI completeness, then penalizes saturation, underexposure,
motion/shift, clipping, and detector-failure flags. AFM roughness and whether a
frame is streaky or spotty are not used.

## AFM Scan Selection

The primary rule uses physical ZSensor plane-corrected 1.0 +/- 0.1 um height
maps. If multiple valid scans exist, the selected scan is the one with Rq
closest to that sample's median Rq in the fixed subset. If no 1.0 um scan is
available, the dominant valid scan size for that sample is used and marked as a
fallback.

## Counts

- total paired samples visualized: 36
- primary 1.0 um samples: 32
- strict-QC samples: 31
- non-1.0 um AFM fallback samples: 4 (6054, 6055, 6100, 6102)
- samples with uncertain RHEED frame selection: 3 (6054, 6058, 6099)
- pre-rendered AFM fallback samples: 0

## Removelist Exclusion

- removelist source: `removelist.txt`
- parsed removelist sample IDs: 6018, 6023, 6061, 6066, 6068, 6087, 6104
- present in previous paired roughness outputs and excluded here: 6023, 6066, 6068, 6087
- not present in previous paired roughness outputs: 6018, 6061, 6104
- exclusion audit: `outputs/rheed_roughness/pair_visualization/excluded_samples_audit.csv`

## Display Choices

RHEED panels use the cropped ROI and percentile display normalization already
used by the roughness audit; no aggressive sharpening is applied. AFM panels are
rendered from physical height arrays in nanometers with viridis and a lateral
scale bar. Native-scale figures use each scan's own min/max. Common-scale
figures use a symmetric 2nd/98th percentile range across selected 1.0 um AFM
height maps: [-13.65, 13.65] nm.

## Reproduction Command

```bash
PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_roughness.visualize_pairs --config configs/rheed_roughness.yaml
```

## Generated Figures

- `pair_grid_primary_1um_by_roughness_native.png/pdf`
- `pair_grid_primary_1um_by_roughness_common_scale.png/pdf`
- `pair_grid_primary_1um_by_rheed_morphology.png/pdf`
- `pair_grid_all_samples_by_roughness.png/pdf`
- `pair_grid_strict_qc_by_roughness.png/pdf`
- `pair_grid_largest_rank_disagreements.png/pdf`
- `selected_rheed_raw_processed_audit.pdf`
- `index.html`
