# RHEED orientation-correction rerun

## Decision and data provenance

N6389 and N6390 are decoded **clockwise by 90 degrees** before every
model-visible crop, temporal clip, RHEED physics feature, embedding, live UI
frame, and generated-AFM prediction. Raw videos are read-only and were not
transcoded or overwritten. The other 26 growths have byte-identical model
embeddings relative to the prior full-28 run.

The first correction experiment rotated the videos and reran the automatic
keyframe selector. That changed both spatial orientation and temporal sample,
so it was not a controlled test. The final protocol locks the previously
target-blind V5 rotation vertex and recomputes the complete-lattice ROI in the
corrected coordinate system:

| sample | original vertex | rotate + reselect | final locked vertex |
|---|---:|---:|---:|
| N6389 | 1238 | 508 | 1238 |
| N6390 | 1048 | 964 | 1048 |

No AFM target was used to select these frames or ROIs.

## Strict held-growth scalar results

Every point below is a complete outer leave-one-growth-out prediction
(27 growths fitted, one growth held out), repeated for all 28 growths.

| protocol | Sq MAE (nm) | Sq Pearson r (p) | Sq confidence-error rho (p) | FSMI MAE (nm) | FSMI Pearson r (p) | FSMI confidence-error rho (p) |
|---|---:|---:|---:|---:|---:|---:|
| original orientation | 1.284 | 0.661 (0.0001299) | -0.529 (0.003814) | 1.134 | 0.661 (0.0001278) | -0.362 (0.05807) |
| CW 90 degrees + reselect | 1.380 | 0.333 (0.08296) | -0.297 (0.1246) | 1.222 | 0.334 (0.08281) | -0.250 (0.1997) |
| **CW 90 degrees + locked vertex (final)** | **1.321** | **0.622 (0.0004087)** | **-0.458 (0.01414)** | **1.168** | **0.630 (0.000325)** | **-0.403 (0.03349)** |

The reselected run is retained as a negative ablation. Its degradation shows
that the selector, which was calibrated in the original acquisition
coordinate system, moved to a different rotation cycle when the frame was
rotated. Locking the target-blind temporal vertex isolates the requested
spatial correction and recovers most of the full-cohort association.

## Generated AFM

The final run generates four 128 x 128 AFM height-field draws for every held
growth with both M10 and M12a; no measured AFM patch or nearest-neighbor image
is available at inference.

| renderer | texture-gate pass | median sharpness ratio | median island-feature MAE (z) |
|---|---:|---:|---:|
| M10 dense-island spectral | 1.000 | 0.795 | 1.371 |
| M12a edge-preserving terrace | 0.750 | 0.728 | 1.814 |

The figure atlas includes all 28 held growths; the dedicated orientation panel
shows measured AFM, the old-orientation M12a result, corrected M12a, and
corrected M10 for N6389/N6390. Generated maps are morphology samples
conditioned on RHEED, not pixel-registered reconstructions of a unique AFM
field of view.

## Key artifacts

- Input/ROI audit: `figures/Fig5_orientation_keyframe_roi_audit`
- All-28 parameter comparison: `figures/Fig6a_orientation_protocol_metrics`
  and `figures/Fig6b_orientation_protocol_scatter`
- Corrected generated AFM: `figures/Fig10_orientation_corrected_generated_afm`
- Complete final generator atlas: `reports/rheed_m15b_end_to_end_generation/20260729_m15b_m12a_line3_auto_full28_orientation90_keyframe_locked_v3/full28_loo/figures`
- Per-sample scalar comparison: `orientation_corrected_sample_predictions.csv`
- Machine-readable protocol metrics: `orientation_parameter_comparison.csv`
- Non-target embedding invariance: `orientation_embedding_isolation_audit.csv`
- AFM-target invariance: `afm_target_invariance_audit.json`

## Limitations

This is strict retrospective leave-one-growth-out evaluation, not a new
prospective acquisition batch. Only two videos required orientation
correction, so the comparison cannot establish a general law about arbitrary
camera rotations. Confidence is a cross-fitted relative risk index, not a
probability of correctness. The final live replay override applies only to
archived N6389/N6390; unseen streaming samples continue to use the automatic
selector without an ID-specific temporal override.
