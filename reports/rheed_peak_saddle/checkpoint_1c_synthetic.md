# Checkpoint 1C: Synthetic Semantic Ground Truth Repair

## Repository and Safety Audit
- Repository root: `/home/wangziyi/MorphMBE/MorphMBE`
- Git commit: `45ae06d4b87e08b460abf9997fbd9d096d8b2266`
- Preserved v1/v2 artifacts: `1`
- Removelist SHA256: `8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b`
- Sample `6088` excluded: `1`
- Stage-review SHA256: `862df0397683a19c24d616b2ba42b088538048750a63a89eb17593c1b4c9081e`

## Stage 1B Context
Stage 1B fixed the pair-topology subsystem but failed Spearman because `nominal_bridge_control` was not the same as the final visible clean adhesion.

## Semantic Repair
`nominal_bridge_control` is now separated from `oracle_visual_adhesion_clean` and `estimated_adhesion_observed`. The renderer writes clean spot signal, explicit bridge signal, clean morphology, acquisition background/halo, observed linear image, display image, and valid mask.

## Independent Oracle
The clean oracle uses a priority-queue maximum-capacity path implementation independent of the production union-find saddle code.
- Semantic spec hash: `ffa417f8c5a67f8a3ede3e532464b3a82a783c47a3362dc4abb85cc4f8ed0689`
- Holdout manifest hash: `0b107e3433de5d757ca70e72cf3a302cfa97fe1dd1fa80d410394b658305dc84`
- Evaluation completed: `1`
- Holdout-v3 seeds: `2026083100...2026083119`

## Within-Family Monotonicity
- Families audited: `8`

## Acceptance Criteria
| Criterion | Threshold | Development | Holdout v3 | Status |
|---|---:|---:|---:|---|
| `production_vs_oracle_saddle_median_abs_diff` | <= 1e-5 | 0 | 0 | PASS |
| `production_vs_oracle_saddle_max_abs_diff` | <= 1e-3 | 0 | 0 | PASS |
| `target_vs_achieved_oracle_spearman` | >= 0.995 | 0.9883 | 0.9976 | PASS |
| `target_vs_achieved_oracle_mae` | <= 0.02 | 0.004247 | 0.004247 | PASS |
| `target_vs_achieved_oracle_p90_abs_error` | <= 0.03 | 0.008727 | 0.008727 | PASS |
| `within_family_median_spearman` | >= 0.995 | 0.9981 | 0.9981 | PASS |
| `within_family_fraction_ge_0_99` | >= 0.95 | 0.75 | 0.75 | FAIL |
| `estimated_vs_oracle_spearman_matched` | >= 0.97 | 0.9904 | 0.9913 | PASS |
| `estimated_vs_oracle_median_abs_error` | <= 0.05 | 0.0087 | 0.007293 | PASS |
| `estimated_vs_oracle_p90_abs_error` | <= 0.10 | 0.07807 | 0.05187 | PASS |
| `end_to_end_estimated_vs_oracle_spearman` | >= 0.95 | 0.9904 | 0.9913 | PASS |
| `end_to_end_estimated_vs_target_spearman` | >= 0.95 | 0.9783 | 0.9849 | PASS |
| `connected_vs_isolated_auroc` | >= 0.95 | 1 | 1 | PASS |
| `false_connected_rate_clean_oracle_isolated` | <= 0.05 | 0 | 0 | PASS |
| `exposure_gamma_offset_median_abs_delta` | <= 0.05 | 0 | 0 | PASS |
| `halo_induced_error_zero_oracle` | <= 0.05 | 0 | 0 | PASS |
| `vertical_bridge_false_positive_rate` | <= 0.10 | 0 | 0 | PASS |
| `spot_precision` | >= 0.95 | 1 | 1 | PASS |
| `spot_recall` | >= 0.95 | 1 | 1 | PASS |
| `row_grouping_accuracy` | >= 0.90 | 1 | 1 | PASS |
| `lattice_index_accuracy` | >= 0.90 | 1 | 1 | PASS |
| `adjacent_pair_precision` | >= 0.90 | 1 | 1 | PASS |
| `adjacent_pair_recall` | >= 0.90 | 1 | 1 | PASS |
| `false_adjacency_across_missing_site` | <= 0.05 | 0 | 0 | PASS |
| `valid_eligible_pair_coverage` | >= 0.90 | 1 | 1 | PASS |
| `invalid_pair_rejection` | >= 0.90 | 1 | 1 | PASS |
| `analytical_saddle_tests` | all pass | 1 | 1 | PASS |
| `no_real_rheed_access` | pass | 1 | 1 | PASS |
| `no_afm_rq_access` | pass | 1 | 1 | PASS |

## Rank-Inversion Analysis
- Rank inversions recorded: `80`

## Lattice-Index Visualization
The Stage 1C report figures explicitly label lattice examples using `r<row_id>:k<lattice_site_index>` notation.

## Overall Status
STAGE 1C FAIL
DO NOT RUN REAL-IMAGE DIAGNOSTICS.

## Boundary Confirmations
- No real RHEED images were read.
- No AFM height maps were accessed.
- No AFM Rq targets were accessed.
- No model training was run.
- Stage 2 was not run.

Evaluation command: `PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_peak_saddle.run --config configs/rheed_peak_saddle.yaml --stage synthetic_v3_evaluate`
Plot-only command: `PYTHONPATH=src:. .venv/bin/python -m analysis.rheed_peak_saddle.run --config configs/rheed_peak_saddle.yaml --stage synthetic_v3_report`
