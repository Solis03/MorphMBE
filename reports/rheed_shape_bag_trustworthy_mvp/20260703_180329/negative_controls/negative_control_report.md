# Shape-Bag Negative-Control Report

Controls attempted: `['brightness_only_diagnostic', 'exposure_only_diagnostic', 'forbidden_id_path_diagnostic', 'random_gaussian_features', 'raw240_feature_diagnostic', 'shuffled_labels_global_diagnostic', 'shuffled_labels_within_train_folds', 'shuffled_shape_bags_across_groups']`
Suspicious control rows: `23`
Negative controls pass: `False`

Forbidden ID/path diagnostics are leakage demonstrations only and are never eligible for production selection.
Raw 240 feature diagnostics are not production defaults unless separately audited for exposure stability.
