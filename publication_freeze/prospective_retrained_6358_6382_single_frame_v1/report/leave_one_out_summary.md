# Strict leave-one-out prediction report

All 28 labeled samples are evaluated with strict leave-one-out prediction. For each row, the displayed sample is absent from both `StandardScaler` fitting and all five `Ridge(alpha=1.0)` member fits; the other 27 samples form that fold's training set. The five member outputs are aggregated by their unchanged median rule.

This is a post-hoc all-labeled-sample analysis and is kept separate from the original three-sample prospective test. In these LOO folds, N6342, N6389, and N6390 are allowed to train predictions for other samples, while each remains excluded from its own fold.

The table and figures use raw model outputs to preserve the frozen algorithm. Negative Rq predictions are flagged in the CSV; a separate nonnegative-clipped column is supplied but is not used for the primary metrics.

## Cohort-level metrics

| Metric | Value |
|---|---:|
| n | 28 |
| MAE | 1.5600 nm |
| Median absolute error | 1.1794 nm |
| RMSE | 2.1665 nm |
| Mean bias | 0.1805 nm |
| R² | -0.0649 |
| Pearson r | 0.3506 (p=0.0674) |
| Spearman ρ | 0.4521 (p=0.0157) |
| Kendall τ | 0.3386 (p=0.0112) |

## Per-sample predictions

| Sample | Source | Ground truth T4 | LOO prediction | Residual | Absolute error | Member q10–q90 |
|---|---|---:|---:|---:|---:|---:|
| 6022 | Historical | 1.4471 | 6.7805 | +5.3334 | 5.3334 | 6.6450–6.7805 |
| 6028 | Historical | 5.7479 | 5.0155 | -0.7324 | 0.7324 | 5.0155–5.2337 |
| 6029 | Historical | 2.3621 | 3.7576 | +1.3954 | 1.3954 | 3.4752–3.7576 |
| 6033 | Historical | 2.2942 | 2.5958 | +0.3016 | 0.3016 | 2.5958–2.6987 |
| 6047 | Historical | 4.1916 | 3.0403 | -1.1513 | 1.1513 | 2.7932–3.0403 |
| 6048 | Historical | 1.9803 | 1.8112 | -0.1691 | 0.1691 | 1.3514–1.8112 |
| 6056 | Historical | 2.7644 | 2.9245 | +0.1601 | 0.1601 | 2.7750–2.9245 |
| 6057 | Historical | 4.5484 | 3.1116 | -1.4368 | 1.4368 | 3.1116–3.2242 |
| 6062 | Historical | 3.0743 | 3.7602 | +0.6859 | 0.6859 | 3.7602–4.1988 |
| 6063 | Historical | 5.7637 | 3.5983 | -2.1654 | 2.1654 | 3.3032–3.5983 |
| 6070 | Historical | 2.6383 | 5.1308 | +2.4925 | 2.4925 | 5.1308–5.2606 |
| 6072 | Historical | 1.2697 | 2.8899 | +1.6202 | 1.6202 | 2.8899–3.0620 |
| 6078 | Historical | 1.4314 | 0.8691 | -0.5622 | 0.5622 | 0.7195–0.8691 |
| 6080 | Historical | 3.5165 | 3.6163 | +0.0998 | 0.0998 | 3.6163–3.8856 |
| 6081 | Historical | 1.1225 | 4.9129 | +3.7904 | 3.7904 | 4.9129–5.3811 |
| 6082 | Historical | 1.8730 | 3.7911 | +1.9181 | 1.9181 | 3.6972–3.7911 |
| 6084 | Historical | 1.7656 | 2.9732 | +1.2076 | 1.2076 | 2.9732–2.9885 |
| 6085 | Historical | 2.7420 | 2.4167 | -0.3254 | 0.3254 | 2.4167–2.8107 |
| 6090 | Historical | 2.7626 | 1.9244 | -0.8383 | 0.8383 | 1.9244–2.3024 |
| 6094 | Historical | 2.4915 | 5.3971 | +2.9056 | 2.9056 | 5.3971–5.6263 |
| 6095 | Historical | 7.4209 | 4.7523 | -2.6686 | 2.6686 | 4.6677–4.7523 |
| 6099 | Historical | 10.2679 | 3.8817 | -6.3862 | 6.3862 | 3.8817–4.1156 |
| 6101 | Historical | 1.1937 | 0.8167 | -0.3770 | 0.3770 | 0.8167–0.9691 |
| N6342 | Original prospective test | 0.8945 | 0.6927 | -0.2018 | 0.2018 | 0.3899–0.6927 |
| N6358 | Original added train | 1.0923 | -0.2750 | -1.3673 | 1.3673 | -0.2916–-0.2750 |
| N6382 | Original added train | 1.3039 | 0.3736 | -0.9303 | 0.9303 | 0.3263–0.3736 |
| N6389 | Original prospective test | 2.3350 | 3.4146 | +1.0796 | 1.0796 | 3.4146–3.5528 |
| N6390 | Original prospective test | 2.2977 | 3.6747 | +1.3769 | 1.3769 | 3.6747–3.8537 |

## Figures and machine-readable outputs

- `figures/main/Figure2_leave_one_out_prediction_scatter.*`
- `figures/supplementary/SuppFigure10_leave_one_out_diagnostics.*`
- `predictions/leave_one_out_28/predictions.csv`
- `predictions/leave_one_out_28/ensemble_member_predictions.csv`
- `predictions/leave_one_out_28/fold_manifest.csv`
- `predictions/leave_one_out_28/metrics.json`
