# Result summary

The frozen 23-sample model was rerun for N6342, N6358, N6382, N6389, and N6390; N6324 was not included.

The retrained quantitative cohort contains the original 23 samples plus N6358 and N6382. N6342, N6389, and N6390 remain outside both the quantitative fit and the 25-group AFM retrieval bank.

## Added training targets

| Sample | T4 trimmed-mean Rq (nm) | T6 quality-weighted Rq (nm) |
|---|---:|---:|
| N6358 | 1.0923 | 1.0943 |
| N6382 | 1.3039 | 1.2902 |

## Predictions and revealed AFM evaluation

| Sample | Frozen 23 prediction | Retrained 25 prediction | GT T4 | Retrained abs. error | Retrieved source |
|---|---:|---:|---:|---:|---:|
| N6342 | -2.2245 | 0.8842 | 0.8945 | 0.0103 | 6048 |
| N6389 | 2.2022 | 3.7759 | 2.3350 | 1.4409 | 6028 |
| N6390 | 2.5890 | 4.2718 | 2.2977 | 1.9741 | 6057 |

Retrained three-sample MAE: 1.1418 nm.
Frozen-baseline three-sample MAE: 1.1810 nm.

Only three prediction samples are evaluated, so these descriptive metrics must not replace the frozen strict OOF benchmark.

## Primary figure

`figures/main/Figure1_three_sample_prediction_atlas.png` contains, in one figure, each sample's RHEED keyframe, five upper-left-quarter ground-truth AFMs, and the retrieved AFM. Every AFM panel retains a height bar in nm and an Rq label.

## Strict leave-one-out evaluation

Each of all 28 labeled samples (the historical 23 plus all five extra samples) was predicted by refitting the unchanged five-member ensemble on the other 27 samples. The held-out sample was excluded from both `StandardScaler` and `Ridge` fitting in its fold.

This is a post-hoc all-labeled-sample analysis. It is separate from the original three-sample prospective evaluation because N6342, N6389, and N6390 contribute labels to the training folds for other held-out samples.

| Metric | Raw LOO ensemble output |
|---|---:|
| MAE | 1.5600 nm |
| Median absolute error | 1.1794 nm |
| RMSE | 2.1665 nm |
| R² | -0.0649 |
| Pearson r | 0.3506 |
| Spearman ρ | 0.4521 |

Largest absolute LOO errors:

| Sample | Ground truth T4 | LOO prediction | Absolute error |
|---|---:|---:|---:|
| 6099 | 10.2679 nm | 3.8817 nm | 6.3862 nm |
| 6022 | 1.4471 nm | 6.7805 nm | 5.3334 nm |
| 6081 | 1.1225 nm | 4.9129 nm | 3.7904 nm |
| 6094 | 2.4915 nm | 5.3971 nm | 2.9056 nm |
| 6095 | 7.4209 nm | 4.7523 nm | 2.6686 nm |

Paper figures: `figures/main/Figure2_leave_one_out_prediction_scatter.*` and `figures/supplementary/SuppFigure10_leave_one_out_diagnostics.*`.

The complete per-sample table is `predictions/leave_one_out_28/predictions.csv`.

## Held-one-out AFM prediction atlas

A separate 28-fold visual experiment excludes the target sample from both the 27-sample quantitative fit and the 27-group A3 AFM retrieval bank. N6342/N6389/N6390 use ground truth 5/3/1, respectively; all other representative AFMs minimize absolute measured-Rq distance to the sample T4 target.

Rendered AFM Rq versus the selected displayed ground truths: MAE 1.6339 nm, RMSE 2.2851 nm.

Primary atlas: `figures/main/Figure3_held_one_out_afm_prediction_atlas.*`. Full selections and retrieved sources are documented in `report/held_one_out_afm_summary.md`.
