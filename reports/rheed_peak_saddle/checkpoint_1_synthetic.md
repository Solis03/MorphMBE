# Checkpoint 1: Synthetic Peak-Saddle Validation

## Session-Recovery Audit

- Repository root: `/home/wangziyi/MorphMBE/MorphMBE`
- Git commit: `45ae06d4b87e08b460abf9997fbd9d096d8b2266`
- Related uncommitted files: `?? analysis/rheed_peak_saddle/; ?? annotations/rheed_peak_saddle/; ?? configs/rheed_peak_saddle.yaml; ?? reports/rheed_peak_saddle/; ?? tests/test_rheed_peak_saddle.py`
- Checkpoint 0 candidate manifest rows: `25`

## Canonical Removelist

- Path: `/home/wangziyi/MorphMBE/MorphMBE/removelist.txt`
- SHA256: `8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b`
- Checkpoint 0 SHA256: `8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b`
- Sample `6088` remains excluded: `1`

## Completed Stage Review

- Reviewed sample count: `25`
- Approved-stage counts: `{'after_growth': 1, 'active_growth': 7, 'rampdown_or_cooldown': 16, 'rampup_or_heating': 1}`
- Comparable-stage-group counts: `{'post_GaSb_before_cooldown': 1, 'GaSb_growth_time_unspecified': 1, 'temperature_transition_to_GaSb_conditions': 3, 'GaSb_growth_15min': 1, 'post_growth_cooldown_200C': 6, 'rampdown_temperature_unspecified': 1, 'GaSb_growth_early_2_6min': 4, 'post_growth_cooldown_300C': 4, 'post_growth_cooldown_250C': 2, 'temperature_ramp_to_650C': 1, 'GaSb_growth_150min': 1}`
- Stage-confidence counts: `{'high': 19, 'medium': 5, 'low': 1}`
- Rows with `unknown`: `none`
- Rows with low confidence: `6048`
- Completed CSV unchanged by this run: `1`

## Synthetic Renderer Specification

The renderer generated continuous linear grayscale RHEED-like images with one to three approximately horizontal rows, Gaussian/Moffat spots, varied amplitudes, widths, eccentricities and spacings, missing spots, unequal neighbors, known horizontal bridges from 0 to 1, diffuse screen background, central halo/direct-beam-like blobs, gradients, noise, blur, saturation, rotation, translation, and partial-crop adversarial cases. It also emitted spot centers, widths, profile families, row IDs, pair IDs, bridge map, background map, nuisance parameters, and valid-region mask.

## Development vs Locked Holdout

- Development seeds: `2026071300...2026071370`
- Locked holdout seeds: `2026072300...2026072370`
- The holdout used disjoint seeds plus different row counts, profile mixtures, background axes, spacing jitter, blur, and curvature.
- The locked holdout was evaluated once after implementing the Stage 1 algorithm in this run; no real-image diagnostics were run.

## Peak-Saddle Algorithm

Detected compact local maxima are grouped into rows after robust row-angle voting. Adjacent neighbors are measured in a pair corridor. Local background is estimated from parallel offset corridors, and a union-find superlevel merge level gives the maximum-bottleneck saddle connecting the two spot-core seeds. Final adhesion is clipped to `[0, 1]` only after recording the unclipped value and invalid reason codes.

## Analytical Unit Tests

| Test | Expected | Observed | PASS/FAIL |
|---|---:|---:|---|
| `equal_peaks_constant_bridge` | 0.5 | 0.5 | PASS |
| `unequal_peaks_constant_bridge` | 0.5 | 0.5 | PASS |
| `isolated_constant_background` | 0 | 0 | PASS |
| `isolated_over_smooth_halo` | 0 | 0 | PASS |
| `vertical_bridge_excluded` | 0 | 0 | PASS |
| `outside_corridor_bright_path` | 0 | 0 | PASS |
| `affine_intensity_base` | 0.4615 | 0.4615 | PASS |
| `affine_intensity_transformed` | 0.4615 | 0.4615 | PASS |
| `small_translation_rotation` | 0.4819 | 0.4819 | PASS |
| `partial_crop_invalid` | nan | nan | PASS |

## Acceptance Criteria

| Criterion | Threshold | Development | Holdout | Status |
|---|---:|---:|---:|---|
| `bridge_strength_monotonicity_spearman` | >= 0.95 | 0.965 | 0.9457 | FAIL |
| `connected_vs_isolated_auroc` | >= 0.95 | 1 | 1 | PASS |
| `false_connected_rate_isolated` | <= 0.05 | 0 | 0 | PASS |
| `exposure_gamma_offset_median_abs_delta` | <= 0.05 | 1.875e-07 | 2.233e-07 | PASS |
| `halo_zero_bridge_median_delta` | <= 0.05 | 0.002593 | 0.003329 | PASS |
| `vertical_bridge_false_positive_rate` | <= 0.10 | 0 | 0 | PASS |
| `spot_center_error_median_px` | <= 2.0 | 0.5074 | 0.5141 | PASS |
| `spot_center_error_median_width_norm` | reported | 0.09386 | 0.09087 | PASS |
| `row_grouping_accuracy` | >= 0.90 | 1 | 1 | PASS |
| `adjacent_pair_precision` | >= 0.90 | 0.7273 | 0.8 | FAIL |
| `adjacent_pair_recall` | >= 0.90 | 0.9 | 0.8 | FAIL |
| `valid_pair_measurement_coverage` | >= 0.90 | 0.9 | 0.8 | FAIL |
| `analytical_saddle_tests` | all pass | 1 | 1 | PASS |
| `no_afm_rq_access_dependency_audit` | no forbidden synthetic dependency | 1 | 1 | PASS |

## Overall Status

STAGE 1 FAIL

**DO NOT RUN REAL-IMAGE DIAGNOSTICS.**

## Dominant Failure Modes

- `development` `adjacent_pair_precision`: 0.7272727272727273 vs >= 0.90
- `holdout` `bridge_strength_monotonicity_spearman`: 0.9457039426343716 vs >= 0.95
- `holdout` `adjacent_pair_precision`: 0.8 vs >= 0.90
- `holdout` `adjacent_pair_recall`: 0.8 vs >= 0.90
- `holdout` `valid_pair_measurement_coverage`: 0.8 vs >= 0.90

## Development-Set Implementation Choices

- Continuous local-maxima detector with broad-background subtraction, width filtering, and non-maximum suppression.
- Pair corridors and background-offset corridors scaled by detected spot width.
- Connected/isolated synthetic boundary fixed at true bridge strength `>= 0.50` and `<= 0.10` before holdout evaluation.
- Display gamma is recorded as a display-channel nuisance; the primary synthetic measurement uses the preserved linear channel.

## Boundary Confirmation

- Locked holdout not used for tuning: `1`
- No AFM/Rq synthetic dependency audit passed: `1`
- Dependency hits: `{}`

## Reproduction

- Exact command: `PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_peak_saddle.run --config configs/rheed_peak_saddle.yaml --stage synthetic`

## Next Step

If Stage 1 is accepted by the human reviewer, the next checkpoint would be a real-image diagnostic stage run under the staged protocol. It was not executed here.
