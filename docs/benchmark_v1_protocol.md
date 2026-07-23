# Benchmark v1 Protocol

Benchmark v1 freezes the scientific evaluation rules before any new computational
experiments are run. Model selection is restricted to the 23 historical growth
groups because those are the only paired samples whose target definition is the
historical primary target, `T4_second_order_trimmed_mean`.

The four current prospective matched samples, `N6342, N6358, N6382, N6389`,
are labeled `prospective_pilot_seen`. Their AFM truth has already influenced
discussion, so they are not a blind confirmatory set. They may be used only after
a candidate is selected with historical data alone, and only with pilot labeling.

Nested leave-one-growth-group-out uses 23 outer folds. Each fold holds out one
historical growth group and trains on the other 22. Hyperparameters, scaling,
PCA, feature selection, imputation, calibration, target-transform fitting,
conformal calibration, OOD thresholds, ensemble choices, and temporal frame-count
tuning must occur inside the outer-training data. The inner loop is another
leave-one-growth-group-out over those 22 training groups.

Preprocessing must occur inside folds because any statistic fit before the split
can encode the held-out growth group. That includes StandardScaler, PCA, feature
selection, metadata imputation, calibration, target transformations, and any OOD
thresholds.

Future confirmatory samples must be newly grown, selected before AFM truth is
known, and appended in a new registry version. Benchmark v1 is never overwritten.

All model families compare on the same sample IDs, target definition, split file,
metric names, strata boundaries, and run-recording schema. The fixed historical
strata thresholds are 2.08493
nm and 2.97097 nm.
