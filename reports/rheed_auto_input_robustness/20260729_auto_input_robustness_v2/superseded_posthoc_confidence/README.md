# Superseded post-hoc confidence audit

These files preserve the first M15a conflict-veto attempt. They are retained
for failure-analysis provenance and **must not be cited as strict final
evidence**.

The point predictions were outer-target held out, but the first
temporal-versus-physics disagreement reference reused global LOO diagnostic
rows. Other rows in that global reference could have been trained with the
current outer target. Independent review detected this indirect confidence
calibration leak before the result was committed.

The valid replacement is:

- `../m15a_tta_centrality_ablation_predictions.csv` for strict M15a
  TTA-centrality-only ablation;
- `../m15b_strict_loo_predictions.csv` and `../m15b_metrics.csv` for the
  fully nested angular-coverage/TTA result.

The superseded files are never loaded by the UI or current configuration.
