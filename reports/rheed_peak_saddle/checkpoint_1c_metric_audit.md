# Checkpoint 1C-R: Metric Lineage Audit

## Provenance And Immutability
- Evaluation receipt SHA256: `a3619bad7a8517d083c4fd73852a6666c235bf21a2f46c5a8ce02f0869541e9f`
- Frozen semantic spec hash: `ffa417f8c5a67f8a3ede3e532464b3a82a783c47a3362dc4abb85cc4f8ed0689`
- Frozen semantic spec file SHA256: `98aebd38a1e59f1f120cc143fd6b068f6113bfd092b7a1aae3106b8887272a26`
- Holdout-v3 manifest hash from receipt: `0b107e3433de5d757ca70e72cf3a302cfa97fe1dd1fa80d410394b658305dc84`
- Removelist SHA256: `8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b`
- Stage-review SHA256: `862df0397683a19c24d616b2ba42b088538048750a63a89eb17593c1b4c9081e`
- Immutable evaluation files changed during audit: `none`

## Original Stage 1C Result
- Original reported status: `STAGE 1C FAIL`
- Original reported failing value: `0.75`
- Original threshold: `>= 0.95`

## Failing Metric Lineage
- Historical source function: `aggregate_holdout_v3_metrics()` consumed `family_rows` from `old_control_identifiability_v3()`.
- Historical x/y: `nominal_bridge_control` versus `oracle_visual_adhesion_clean`.
- Pre-registered x/y: `target_visual_adhesion` versus `achieved_oracle_visual_adhesion`.
- Lineage decision: `BRANCH A - CLEAR METRIC-LINEAGE BUG`.

## Corrected Preregistered Metrics
- Corrected within-family median Spearman: `1`.
- Corrected within-family fraction >= 0.99: `1`.
- Successful solver rows use `solver_status == converged`; unattainable and failed rows are excluded from the acceptance denominator.

## Family-Level Calibrated Metrics
| Split | Family | n | rho | tau-b | inversions | MAE | unattainable |
|---|---|---:|---:|---:|---:|---:|---:|
| development_v3 | `development_v3_template_00` | 12 | 1 | 1 | 0 | 0.00404244 | 8 |
| development_v3 | `development_v3_template_01` | 15 | 1 | 1 | 0 | 0.00372301 | 5 |
| development_v3 | `development_v3_template_02` | 14 | 1 | 1 | 0 | 0.00303533 | 6 |
| development_v3 | `development_v3_template_03` | 11 | 1 | 1 | 0 | 0.00486509 | 9 |
| development_v3 | `development_v3_template_04` | 11 | 1 | 1 | 0 | 0.00774894 | 9 |
| development_v3 | `development_v3_template_05` | 15 | 1 | 1 | 0 | 0.00391585 | 5 |
| development_v3 | `development_v3_template_06` | 12 | 1 | 1 | 0 | 0.00349197 | 8 |
| development_v3 | `development_v3_template_07` | 15 | 1 | 1 | 0 | 0.00497698 | 5 |
| holdout_v3 | `holdout_v3_template_00` | 15 | 1 | 1 | 0 | 0.0048211 | 5 |
| holdout_v3 | `holdout_v3_template_01` | 15 | 1 | 1 | 0 | 0.00396981 | 5 |
| holdout_v3 | `holdout_v3_template_02` | 11 | 1 | 1 | 0 | 0.00311871 | 9 |
| holdout_v3 | `holdout_v3_template_03` | 14 | 1 | 1 | 0 | 0.00291375 | 6 |
| holdout_v3 | `holdout_v3_template_04` | 11 | 1 | 1 | 0 | 0.00525649 | 9 |
| holdout_v3 | `holdout_v3_template_05` | 12 | 1 | 1 | 0 | 0.00534472 | 8 |
| holdout_v3 | `holdout_v3_template_06` | 15 | 1 | 1 | 0 | 0.00602569 | 5 |
| holdout_v3 | `holdout_v3_template_07` | 14 | 1 | 1 | 0 | 0.00250706 | 6 |

## Nominal-Control Diagnostic
- Nominal-control families audited: `8`.
- This remains diagnostic only and is not the target-calibrated acceptance metric.

## Visual Diagnostics
- Repaired nominal-control sweep: `reports/rheed_peak_saddle/synthetic_v3/nominal_control_within_family_sweeps.png`.
- Repaired target-calibrated sweep: `reports/rheed_peak_saddle/synthetic_v3/calibrated_target_within_family_sweeps.png`.
- Repaired rank inversion panels: `largest_rank_inversions_visual.png` and `.pdf`.
- Repaired lattice overlay panels: `lattice_indexing_examples_visual.png` and `.pdf`.
- High-adhesion error panels: `high_adhesion_error_cases.png` and `.pdf`.

## Boundary Confirmations
- No holdout-v3 image, target, oracle, prediction, manifest, semantic spec, or receipt was modified.
- No holdout-v3 predictions were rerun.
- No renderer, detector, measurement algorithm, or model parameter changed.
- No real RHEED images, AFM data, Rq targets, model training, or Stage 2 were used.

## Final Amended Status
STAGE 1C PASS AFTER METRIC-LINEAGE CORRECTION
