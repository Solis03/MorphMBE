# Single-frame manual RHEED to AFM Rq results

- Included independent samples: 26
- Canonical removelist hash: `840b17a0061a72340272ced9803e2f7d30871d0c28a145627d9625b764350177`
- Best OOF model row by log MAE: `physics_extra_trees_shallow`
- Nested selector Spearman (nm): `-0.8786`; permutation p `0.0004998`
- Nested MAE improvement vs median baseline (nm): `-0.3099`; permutation p `1`
- 90% prediction interval empirical coverage: `0.885`

## Scientific answers

1. Horizontal connectivity versus Rq: Spearman rho `0.142` on the included non-removelist samples.
2. Isolation score versus Rq: Spearman rho `0.0462`.
3. Model comparison is in `outputs/rheed_single_frame_manual/model_comparison.csv`; the nested row is selected only from inner folds.
4. Strongly isolated patterns are not hard-coded as confident. Isolation versus confidence rho is `-0.0174` and isolation versus absolute error rho is `0.173`.
5. Sample 6023 is not analyzed because it is in the canonical removelist.

## Largest errors

- sample 6088: true `1.832` nm, predicted `11.073` nm, error `9.241` nm
- sample 6099: true `10.321` nm, predicted `3.181` nm, error `7.140` nm
- sample 6095: true `9.867` nm, predicted `3.181` nm, error `6.686` nm
- sample 6097: true `9.224` nm, predicted `3.181` nm, error `6.043` nm
- sample 6043: true `7.372` nm, predicted `3.181` nm, error `4.191` nm

## Caution

This is a small-data hypothesis-testing regression experiment. The result should be read as an out-of-fold association test, not as causal evidence or a universal RHEED-to-roughness law.