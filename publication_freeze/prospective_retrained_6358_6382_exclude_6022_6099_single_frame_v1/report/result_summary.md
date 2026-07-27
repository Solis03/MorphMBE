# Result summary

Samples 6022 and 6099 are excluded from every training set and AFM retrieval bank in this sensitivity experiment.

The reduced historical baseline contains 21 samples. The retrained quantitative cohort contains those 21 samples plus N6358 and N6382, for 23 total. N6342, N6389, and N6390 remain outside the prospective fit and the 23-group AFM retrieval bank.

## Added training targets

| Sample | T4 trimmed-mean Rq (nm) | T6 quality-weighted Rq (nm) |
|---|---:|---:|
| N6358 | 1.0923 | 1.0943 |
| N6382 | 1.3039 | 1.2902 |

## Predictions and revealed AFM evaluation

| Sample | Reduced 21 prediction | Retrained 23 prediction | GT T4 | Retrained abs. error | Retrieved source |
|---|---:|---:|---:|---:|---:|
| N6342 | 0.0735 | 1.4475 | 0.8945 | 0.5530 | 6028 |
| N6389 | 1.8638 | 2.5164 | 2.3350 | 0.1814 | 6028 |
| N6390 | 1.8461 | 2.2507 | 2.2977 | 0.0471 | 6028 |

Retrained three-sample MAE: 0.2605 nm.
Reduced-baseline three-sample MAE: 0.5813 nm.

Only three prediction samples are evaluated, so these descriptive metrics must not replace the frozen strict OOF benchmark.

## Primary figure

`figures/main/Figure1_three_sample_prediction_atlas.png` contains, in one figure, each sample's RHEED keyframe, five upper-left-quarter ground-truth AFMs, and the retrieved AFM. Every AFM panel retains a height bar in nm and an Rq label.

## Strict leave-one-out evaluation

Each of the 26 retained labeled samples (21 historical plus all five extra samples) was predicted by refitting the unchanged five-member ensemble on the other 25 samples. Samples 6022 and 6099 are absent from every fold, and the held-out sample is excluded from both `StandardScaler` and `Ridge` fitting.

This is a post-hoc all-labeled-sample analysis. It is separate from the original three-sample prospective evaluation because N6342, N6389, and N6390 contribute labels to the training folds for other held-out samples.

| Metric | Raw LOO ensemble output |
|---|---:|
| MAE | 1.2572 nm |
| Median absolute error | 1.1579 nm |
| RMSE | 1.5288 nm |
| R² | 0.0836 |
| Pearson r | 0.4485 |
| Spearman ρ | 0.4229 |

Largest absolute LOO errors:

| Sample | Ground truth T4 | LOO prediction | Absolute error |
|---|---:|---:|---:|
| 6095 | 7.4209 nm | 3.6032 nm | 3.8177 nm |
| 6062 | 3.0743 nm | 0.2768 nm | 2.7974 nm |
| 6057 | 4.5484 nm | 1.7885 nm | 2.7599 nm |
| 6082 | 1.8730 nm | 4.1611 nm | 2.2881 nm |
| 6033 | 2.2942 nm | 4.2824 nm | 1.9882 nm |

Paper figures: `figures/main/Figure2_leave_one_out_prediction_scatter.*` and `figures/supplementary/SuppFigure10_leave_one_out_diagnostics.*`.

The complete per-sample table is `predictions/leave_one_out_26/predictions.csv`.

## Held-one-out AFM prediction atlas

A separate 26-fold visual experiment excludes the target sample from both the 25-sample quantitative fit and the 25-group A3 AFM retrieval bank; samples 6022 and 6099 are absent globally. N6342/N6389/N6390 use ground truth 5/3/1, respectively; all other representative AFMs minimize absolute measured-Rq distance to the sample T4 target.

Rendered AFM Rq versus the selected displayed ground truths: MAE 1.3522 nm, RMSE 1.6667 nm.

Primary atlas: `figures/main/Figure3_held_one_out_afm_prediction_atlas.*`. Full selections and retrieved sources are documented in `report/held_one_out_afm_summary.md`.

## Comparison with the original experiment

Prospective three-sample MAE changed from 1.1418 to 0.2605 nm. On the fair common-26 LOO cohort, MAE changed from 1.2292 to 1.2572 nm.

See `report/exclusion_impact_summary.md` for the full per-sample and retrieval comparison.
