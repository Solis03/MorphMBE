# Checkpoint 1B: Synthetic Pair-Recovery Repair

## Repository Recovery Audit
- Repository root: `/home/wangziyi/MorphMBE/MorphMBE`
- Git commit: `45ae06d4b87e08b460abf9997fbd9d096d8b2266`
- V1 artifacts preserved: `1`
- Removelist path: `/home/wangziyi/MorphMBE/MorphMBE/removelist.txt`
- Removelist SHA256: `8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b`
- Sample `6088` excluded: `1`
- Stage-review SHA256: `862df0397683a19c24d616b2ba42b088538048750a63a89eb17593c1b4c9081e`
- Stage-review unchanged: `1`

## Stage 1 V1 Historical Failures
- `bridge_strength_monotonicity_spearman`: 0.9457039426343716 (FAIL)
- `connected_vs_isolated_auroc`: 1.0 (PASS)
- `false_connected_rate_isolated`: 0.0 (PASS)
- `exposure_gamma_offset_median_abs_delta`: 2.2333376037408925e-07 (PASS)
- `halo_zero_bridge_median_delta`: 0.003328981476337145 (PASS)
- `vertical_bridge_false_positive_rate`: 0.0 (PASS)
- `spot_center_error_median_px`: 0.5140510257994011 (PASS)
- `spot_center_error_median_width_norm`: 0.09087347504988627 (PASS)
- `row_grouping_accuracy`: 1.0 (PASS)
- `adjacent_pair_precision`: 0.8 (FAIL)
- `adjacent_pair_recall`: 0.8 (FAIL)
- `valid_pair_measurement_coverage`: 0.8 (FAIL)

## Oracle Ablation
| Oracle | Pair precision | Pair recall | Coverage | Spot precision | Spot recall | Lattice accuracy |
|---|---:|---:|---:|---:|---:|---:|
| A | 1 | 1 | 1 | 1 | 1 | 1 |
| B | 1 | 1 | 1 | 1 | 1 | 1 |
| C | 1 | 1 | 1 | 1 | 1 | 1 |
| D | 1 | 1 | 1 | 1 | 1 | 1 |

Evaluator bug found: the v1 denominator conflated measurable adjacent truth pairs with deliberately ineligible missing/cropped cases and did not report spot completeness. Stage 1B writes explicit eligible/ineligible audit rows.

## Development Error Taxonomy
- `lattice_index_assignment_failure`: 596
- `false_halo_peak`: 270
- `pair_created_from_false_spot`: 190
- `missed_spot`: 93
- `false_border_peak`: 4
- `paired_across_missing_lattice_site`: 2

## Lattice-Indexing Method
Rows are rotated into local coordinates, a robust fundamental spacing is estimated by scoring candidate periods against integer-multiple consistency, detections are assigned integer lattice indices, duplicate index candidates are suppressed, and only index-difference-one pairs are measured.

## Pair Validity and Coverage
Coverage is computed only over eligible consecutive true lattice pairs with both endpoints present and sufficiently inside the valid region. Ineligible partial-crop or missing-endpoint pairs are scored through invalid-pair rejection accuracy.

## Implementation Changes From Development Data Only
- Added one-to-one Hungarian spot matching and spot precision/recall metrics.
- Added oracle A/B/C/D ablation ladder.
- Added lattice-aware adjacency and missing-site cross-gap rejection.
- Added explicit pair eligibility/ineligibility audit fields.
- Added seed-disjoint holdout v2 generated only after feature-spec freeze.

Frozen feature-spec hash: `ef5ae9a04788845571f0bb2afec13428b910803f78d088d91cc9db6a52776571`
Holdout-v2 seeds: `2026077100...2026077197`
Holdout v2 generated only after freeze: `1`
Holdout v2 evaluated once: `0`

Note: the first `synthetic_v2` command completed deterministic evaluation but failed during figure rendering because false-positive pair rows had blank true bridge strength. The plotting filter was fixed without inspecting holdout-v2 failure cases, and the same frozen deterministic command was rerun to finish artifact/report generation.

## Acceptance Criteria
| Criterion | Threshold | Development | Holdout v2 | Holdout v2 CI | Status |
|---|---:|---:|---:|---:|---|
| `bridge_strength_spearman_end_to_end` | >= 0.95 | 0.9164 | 0.862 | [0.8193, 0.8926] | FAIL |
| `bridge_strength_spearman_matched_eligible` | >= 0.97 | 0.9164 | 0.862 | [0.8193, 0.8926] | FAIL |
| `connected_vs_isolated_auroc` | >= 0.95 | 1 | 1 | [nan, nan] | PASS |
| `false_connected_rate_isolated` | <= 0.05 | 0 | 0 | [nan, nan] | PASS |
| `exposure_gamma_offset_median_abs_delta` | <= 0.05 | 1.575e-07 | 1.965e-07 | [nan, nan] | PASS |
| `halo_zero_bridge_median_delta` | <= 0.05 | 0 | 0 | [nan, nan] | PASS |
| `vertical_bridge_false_positive_rate` | <= 0.10 | 0 | 0 | [nan, nan] | PASS |
| `spot_center_error_median_px` | <= 2.0 | 0.4332 | 0.4229 | [nan, nan] | PASS |
| `spot_detection_precision` | >= 0.95 | 1 | 1 | [1, 1] | PASS |
| `spot_detection_recall` | >= 0.95 | 1 | 1 | [1, 1] | PASS |
| `row_grouping_accuracy` | >= 0.90 | 1 | 1 | [nan, nan] | PASS |
| `lattice_index_assignment_accuracy` | >= 0.90 | 1 | 1 | [nan, nan] | PASS |
| `adjacent_pair_precision` | >= 0.90 | 1 | 1 | [1, 1] | PASS |
| `adjacent_pair_recall` | >= 0.90 | 1 | 1 | [1, 1] | PASS |
| `false_adjacency_across_missing_lattice_site` | <= 0.05 | 0 | 0 | [nan, nan] | PASS |
| `valid_pair_measurement_coverage` | >= 0.90 | 1 | 1 | [1, 1] | PASS |
| `invalid_pair_rejection_accuracy` | >= 0.90 | 1 | 1 | [nan, nan] | PASS |
| `analytical_saddle_tests` | all pass | 1 | 1 |  | PASS |
| `no_afm_rq_access_dependency_audit` | pass | 1 | 1 |  | PASS |

## Overall Status
STAGE 1B FAIL
DO NOT RUN REAL-IMAGE DIAGNOSTICS.

## Boundary Confirmation
- No real RHEED diagnostics were run.
- No AFM/Rq data were accessed.
- No Rq model was trained.
- Holdout v1 was not used for tuning.
- Stage 2 was not run.

Exact Stage 1B command: `PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_peak_saddle.run --config configs/rheed_peak_saddle.yaml --stage synthetic_v2`
