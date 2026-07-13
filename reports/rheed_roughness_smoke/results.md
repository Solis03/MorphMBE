# RHEED Roughness Results

## Direct Answer

The primary inclusive 1.0 um analysis found Spearman rho =
-1 for the frozen morphology index
versus AFM Rq, with bootstrap CI
[nan,
nan] and sample-level
permutation p = nan
(n = 3). The strict-QC counterpart found rho =
nan (n = 2).

This should be read as an association audit. It is not evidence of causation,
and it is material/setup specific unless replicated outside this dataset.

## Required Questions

1. Visual measurement of streaky-to-spotty axis: the existing component
   geometry detector was reused. Inspect `index.html` galleries sorted by score;
   the report shows both agreement cases and counterexamples.
2. Human ratings: blinded review materials were generated, but ratings were not
   fabricated. `human_validation_results.csv` is pending annotation.
3. Sensitivity: the largest median perturbation effect was
   `brightness_1.75` with median absolute score change
   0.04858.
4. AFM association: see the primary rho and permutation p above.
5. Confound adjustment: see `regression_results.csv`; nuisance residualization
   and group-aware models are reported separately from raw correlation.
6. Out-of-fold prediction: see `cv_model_comparison.csv`; all metrics are
   leave-one-growth-run-out.
7. Materials/batches: see `material_stratified_results.csv`; small strata are
   explicitly labeled too few samples.
8. Counterexamples: see the agreement/disagreement quadrants in `index.html`.
9. Color-bar/peak-to-valley: `sample_level_analysis_table.csv` separates Rq,
   Ra, robust range, and peak-to-valley span.
10. Use recommendation: treat the score as an exploratory morphology diagnostic
    unless the strict-QC and confound-adjusted results are strong enough for the
    specific material/setup in question.

## Nuisance Warning

The score-predictability audit from nuisance variables produced out-of-fold R2 =
-22.97 and Spearman =
1. High values here
would mean the score is partly an acquisition artifact proxy.
