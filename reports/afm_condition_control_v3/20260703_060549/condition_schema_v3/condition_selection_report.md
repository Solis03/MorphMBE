# Condition Schema V3 Selection Report

Selected descriptor columns: `['rq', 'ra', 'robust_range', 'mean_abs_gradient', 'gradient_std', 'gradient_anisotropy', 'psd_low_power', 'psd_mid_power', 'psd_high_power', 'psd_slope', 'autocorrelation_length_px', 'island_count', 'island_mean_area_px']`
Prototype count: `4`
Dropped or skipped descriptor columns: `[]`

The v3 schema intentionally uses a smaller robust descriptor subset than v2.
Conditions are standardized using train-set means and standard deviations.
