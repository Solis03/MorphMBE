# Global Exposure Invariance Summary

Samples audited: 62
Median raw brightness CV: 0.228966
Median shape feature CV: 0.320492
Default stable feature count: 36 / 48
Default stable feature list: `reports/rheed_shape_bag_input_mvp/20260703_165110/default_training_feature_names.txt`

Some shape features were more exposure-sensitive than raw brightness by the median CV metric.

Recommendation: Use `default_training_feature_names.txt`, prioritize continuous geometry and consensus-map inputs, and tune mask thresholds before relying on raw component-count features.

Inspect sample-level audit JSON files for unstable feature names and perturbation-level behavior.
