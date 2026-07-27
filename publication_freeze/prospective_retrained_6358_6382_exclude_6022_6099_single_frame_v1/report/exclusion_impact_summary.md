# Impact of excluding samples 6022 and 6099

The algorithms, feature definitions, ensemble members, AFM preprocessing, and A3 retrieval ranking are unchanged. Only the two sample groups 6022 and 6099 are removed.

## Data-count changes

| Experiment | Original | After exclusion |
|---|---:|---:|
| Historical baseline training | 23 | 21 |
| Prospective retraining | 25 | 23 |
| All-labeled LOO cohort | 28 | 26 |
| LOO training rows per fold | 27 | 25 |
| AFM held-one-out source groups per fold | 27 | 25 |

## Main findings

- Three-sample prospective MAE changed from 1.1418 to 0.2605 nm (Δ -0.8813 nm).
- On the identical retained 26-sample LOO cohort, MAE changed from 1.2292 to 1.2572 nm (Δ +0.0280 nm); 12/26 individual samples improved.
- On the identical retained 26-sample AFM held-one-out cohort, displayed-map Rq MAE changed from 1.3045 to 1.3522 nm (Δ +0.0478 nm).
- The lower full-cohort LOO and AFM-HOO MAEs are partly caused by removing two unusually high-error samples. The common-26 comparison is the fair test of model changes on retained data.

## Prospective prediction changes

| Sample | Ground truth | Original retrained 25 | Excluded retrained 23 | Original abs. error | New abs. error | Error change |
|---|---:|---:|---:|---:|---:|---:|
| N6342 | 0.8945 | 0.8842 | 1.4475 | 0.0103 | 0.5530 | +0.5427 |
| N6389 | 2.3350 | 3.7759 | 2.5164 | 1.4409 | 0.1814 | -1.2595 |
| N6390 | 2.2977 | 4.2718 | 2.2507 | 1.9741 | 0.0471 | -1.9270 |

## Aggregate fair comparisons

| Experiment | n | Original MAE | New MAE | ΔMAE | Original RMSE | New RMSE | ΔRMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Historical baseline on extra five | 5 | 1.9040 | 0.8394 | -1.0647 | 2.3534 | 0.9267 | -1.4268 |
| Retrained prospective test | 3 | 1.1418 | 0.2605 | -0.8813 | 1.4111 | 0.3371 | -1.0740 |
| LOO common retained cohort | 26 | 1.2292 | 1.2572 | +0.0280 | 1.5467 | 1.5288 | -0.0179 |
| Held-one-out AFM displayed Rq common cohort | 26 | 1.3045 | 1.3522 | +0.0478 | 1.7063 | 1.6667 | -0.0396 |

## Detailed common-cohort metrics

| Experiment | Version | Median AE | Bias | R² | Pearson r | Spearman ρ | Kendall τ | CCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LOO common retained cohort | Original | 1.1155 | +0.2349 | 0.0620 | 0.5179 | 0.5084 | 0.3846 | 0.5113 |
| LOO common retained cohort | After exclusion | 1.1579 | -0.0008 | 0.0836 | 0.4485 | 0.4229 | 0.3108 | 0.4365 |
| Held-one-out AFM displayed Rq common cohort | Original | 1.0132 | +0.2395 | 0.0334 | 0.4602 | 0.4270 | 0.3046 | 0.4500 |
| Held-one-out AFM displayed Rq common cohort | After exclusion | 1.2204 | -0.0069 | 0.0777 | 0.4175 | 0.4106 | 0.3046 | 0.3974 |

## Raw full-cohort summaries (different n; descriptive only)

| Experiment | Original n | New n | Original MAE | New MAE | ΔMAE | Original RMSE | New RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| LOO full available cohort | 28 | 26 | 1.5600 | 1.2572 | -0.3027 | 2.1665 | 1.5288 |
| Held-one-out AFM full available cohort | 28 | 26 | 1.6339 | 1.3522 | -0.2817 | 2.2851 | 1.6667 |

## Retrieval-source changes

Prospective retrained retrieval source changed for 2/3 samples.
Reduced historical-baseline retrieval source changed for 0/5 samples.
AFM held-one-out retrieval source changed for 13/26 retained samples.

## Model-parameter sensitivity

All 5 ensemble members were refit on 23 rather than 25 rows. Full coefficient/intercept changes are in `comparison/model_parameter_changes.csv`.

## Figures

- `figures/comparison/Figure4_exclusion_impact_summary.*`
- `figures/comparison/Figure5_leave_one_out_common26_before_after.*`
