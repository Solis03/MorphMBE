# Strict leave-one-out prediction report

All 26 retained labeled samples are evaluated with strict leave-one-out prediction. For each row, the displayed sample is absent from both `StandardScaler` fitting and all five `Ridge(alpha=1.0)` member fits; the other 25 retained samples form that fold's training set. The five member outputs are aggregated by their unchanged median rule.

This is a post-hoc all-labeled-sample analysis and is kept separate from the original three-sample prospective test. In these LOO folds, N6342, N6389, and N6390 are allowed to train predictions for other samples, while each remains excluded from its own fold.

The table and figures use raw model outputs to preserve the frozen algorithm. Negative Rq predictions are flagged in the CSV; a separate nonnegative-clipped column is supplied but is not used for the primary metrics.

## Cohort-level metrics

| Metric | Value |
|---|---:|
| n | 26 |
| MAE | 1.2572 nm |
| Median absolute error | 1.1579 nm |
| RMSE | 1.5288 nm |
| Mean bias | -0.0008 nm |
| R² | 0.0836 |
| Pearson r | 0.4485 (p=0.0216) |
| Spearman ρ | 0.4229 (p=0.0314) |
| Kendall τ | 0.3108 (p=0.0264) |

## Per-sample predictions

| Sample | Source | Ground truth T4 | LOO prediction | Residual | Absolute error | Member q10–q90 |
|---|---|---:|---:|---:|---:|---:|
| 6028 | Historical | 5.7479 | 4.3779 | -1.3700 | 1.3700 | 4.3779–4.5821 |
| 6029 | Historical | 2.3621 | 3.7471 | +1.3849 | 1.3849 | 3.4367–3.7471 |
| 6033 | Historical | 2.2942 | 4.2824 | +1.9882 | 1.9882 | 4.2824–4.4943 |
| 6047 | Historical | 4.1916 | 4.0347 | -0.1569 | 0.1569 | 3.8484–4.0347 |
| 6048 | Historical | 1.9803 | 3.4421 | +1.4618 | 1.4618 | 2.9980–3.4421 |
| 6056 | Historical | 2.7644 | 2.2286 | -0.5357 | 0.5357 | 1.9544–2.2286 |
| 6057 | Historical | 4.5484 | 1.7885 | -2.7599 | 2.7599 | 1.7885–1.8092 |
| 6062 | Historical | 3.0743 | 0.2768 | -2.7974 | 2.7974 | 0.2768–0.3618 |
| 6063 | Historical | 5.7637 | 4.6683 | -1.0954 | 1.0954 | 4.4031–4.6683 |
| 6070 | Historical | 2.6383 | 3.5578 | +0.9194 | 0.9194 | 3.5578–3.7308 |
| 6072 | Historical | 1.2697 | 2.5664 | +1.2967 | 1.2967 | 2.5664–2.8013 |
| 6078 | Historical | 1.4314 | 0.8600 | -0.5713 | 0.5713 | 0.7323–0.8600 |
| 6080 | Historical | 3.5165 | 2.7905 | -0.7260 | 0.7260 | 2.7905–3.0018 |
| 6081 | Historical | 1.1225 | 2.7999 | +1.6773 | 1.6773 | 2.7999–3.1097 |
| 6082 | Historical | 1.8730 | 4.1611 | +2.2881 | 2.2881 | 4.1223–4.1611 |
| 6084 | Historical | 1.7656 | 2.8802 | +1.1146 | 1.1146 | 2.8802–2.9468 |
| 6085 | Historical | 2.7420 | 3.9433 | +1.2012 | 1.2012 | 3.9433–4.4006 |
| 6090 | Historical | 2.7626 | 1.4810 | -1.2816 | 1.2816 | 1.4810–1.8398 |
| 6094 | Historical | 2.4915 | 3.7067 | +1.2152 | 1.2152 | 3.7067–3.9196 |
| 6095 | Historical | 7.4209 | 3.6032 | -3.8177 | 3.8177 | 3.4390–3.6032 |
| 6101 | Historical | 1.1937 | 1.9570 | +0.7634 | 0.7634 | 1.9570–2.0293 |
| N6342 | Original prospective test | 0.8945 | 1.4807 | +0.5862 | 0.5862 | 1.2440–1.4807 |
| N6358 | Original added train | 1.0923 | 0.0697 | -1.0226 | 1.0226 | 0.0233–0.0697 |
| N6382 | Original added train | 1.3039 | 1.4915 | +0.1876 | 0.1876 | 1.4915–1.4942 |
| N6389 | Original prospective test | 2.3350 | 2.5836 | +0.2486 | 0.2486 | 2.5836–2.6739 |
| N6390 | Original prospective test | 2.2977 | 2.0773 | -0.2204 | 0.2204 | 2.0773–2.2698 |

## Figures and machine-readable outputs

- `figures/main/Figure2_leave_one_out_prediction_scatter.*`
- `figures/supplementary/SuppFigure10_leave_one_out_diagnostics.*`
- `predictions/leave_one_out_26/predictions.csv`
- `predictions/leave_one_out_26/ensemble_member_predictions.csv`
- `predictions/leave_one_out_26/fold_manifest.csv`
- `predictions/leave_one_out_26/metrics.json`
