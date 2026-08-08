# M19 separated rough-island redesign

Date: 2026-08-07

Branch: `codex/rough-afm-separated-islands-20260807`

Baseline: desktop `MorphMBE_M17_N6342_SparsePeak_UI_Standalone_20260804`

## Outcome

M19 replaces M17's rough-regime connected-cell morphology with explicit finite
round/elliptical islands on a deep connected substrate. Island centres use
repulsive placement, island sizes and shapes are learned inside each strict
outer-LOO fold, and only high-pass AFM texture is mixed back into the rough
branch. No negative Gaussian valley primitives are used, so the generator no
longer creates the broad, deep, flat basins seen in the rejected direction.

The M17 branch is returned byte-for-byte for samples whose predicted Sq is at
or below 2.2 nm. A smoothstep transition reaches the full separated-island
branch at 3.6 nm.

## Scientific rationale

The implementation follows three recurring observations in the literature:

1. Strain-mediated island growth creates spatial correlations and denuded
   zones, motivating repulsive rather than independent centre placement
   ([Surface Science, 1997](https://www.sciencedirect.com/science/article/abs/pii/S0039602897007073)).
2. Island density, growth and coalescence are coupled finite-object processes,
   motivating explicit islands instead of a stationary random field
   ([Physical Review E, 2025](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.111.035501);
   [Acta Materialia, 2026](https://www.sciencedirect.com/science/article/pii/S1359645426001394)).
3. Object-centric image generation benefits from separating objects from the
   background, motivating an explicit deep-substrate channel
   ([Object-Centric Image Generation, 2020](https://arxiv.org/abs/2004.00642)).

These papers motivate the representation but do not validate this dataset's
prediction accuracy; all performance claims below come from this repository's
strict outer-LOO evaluation.

## Experiments and selected model

Four full 27-growth experiments were run from the M17 baseline:

- v1: balanced, sparse, round, hierarchical and textured ellipse layouts.
- v2: strict-sparse density and separation ablations.
- v3: larger-island strict-sparse ablations.
- v4: selected large separated-island renderer plus target-blind Sq rough-tail
  rescue.

The selected method is `M19k_rough_tail_large_separated_islands`. Its rough
renderer uses a strict-sparse ellipse field, 0.85 structure / 0.15 high-pass
texture mixing, and a 0.5 px final tip blur. The scalar Sq rescue uses only two
independently trained M17 endpoint experts and M17's existing rough-consensus
gate; it never reads the held-out sample's target when deciding whether to
activate.

## Strict outer-LOO results

Rough stratum: measured Sq in [3, 10] nm, 9 growth groups.

| Metric | M17 | M19 | Interpretation |
|---|---:|---:|---|
| Median island-feature MAE (z) | 1.955 | 1.475 | 24.6% lower |
| Median q70 island count | 15.7 | 29.0 | measured 29.0 |
| Median q70 island area (px) | 37.0 | 82.5 | measured 95.4 |
| Median q55 footprint (px) | 38.0 | 139.2 | measured 151.0 |
| Median flat-basin fraction | 0.137 | 0.089 | measured 0.099 |
| Median PSD log distance | 0.976 | 0.907 | lower is better |
| Median sharpness ratio | 0.743 | 0.857 | closer to 1 is better |
| Sq MAE, 3–10 nm | 1.606 nm | 1.095 nm | 31.8% lower |
| Sq MAE, all 27 | 1.107 nm | 0.966 nm | 12.7% lower |
| Sq MAE, smooth <1.6 nm | 0.584 nm | 0.584 nm | unchanged |

The fold-integrity audit passed for all 27 outer folds. Every tested generated
array below the 2.2 nm predicted-Sq threshold was exactly equal to M17.

## Reproduction

```bash
.venv/bin/python -m analysis.rheed_rough_island_redesign.amplitude \
  --input outputs/rheed_m19_separated_rough_islands/20260807_m19_source_predictions/m16_strict_loo_predictions.csv \
  --output outputs/rheed_m19_separated_rough_islands/20260807_m19_rough_tail_endpoint_full27

.venv/bin/python -m analysis.rheed_to_afm_full_cohort_loo.run \
  --config configs/rheed_m19_separated_rough_islands_full27_v4.json

.venv/bin/python -m analysis.rheed_rough_island_redesign.evaluate \
  --config configs/rheed_m19_separated_rough_islands_full27_v4.json \
  --m17-predictions outputs/rheed_m19_separated_rough_islands/20260807_m19_source_predictions/m16_strict_loo_predictions.csv

.venv/bin/python -m analysis.rheed_rough_island_redesign.atlas \
  --config configs/rheed_m19_separated_rough_islands_full27_v4.json
```

## Artifacts

The final full atlas is in:

`reports/rheed_m19_separated_rough_islands/20260807_m19_m17base_full27_v4/full27_loo/figures/atlas_compare_m17_m19/`

The dedicated numeric audit is in:

`reports/rheed_m19_separated_rough_islands/20260807_m19_m17base_full27_v4/full27_loo/rough_island_audit/`

Generated reports and maps total about 182 MB and are intentionally excluded
from Git. The two small frozen source-prediction CSV files remain under the
ignored output tree and have SHA-256 checksums recorded in their run provenance.

## Integrity and limitations

- No raw RHEED or AFM file was modified.
- The desktop M17 standalone was not written. The three core source files used
  as the baseline still match commit `99bb75b` by SHA-256.
- This is retrospective method development on 27 growth groups. A new
  prospective 3–10 nm cohort is still required before a deployment claim.
- M19's median q70 island area remains about 14% below measured AFM, and the
  generated islands remain somewhat more uniform and round than some real
  rough scans.
- Sq errors are improved as a stratum, not eliminated sample-by-sample; the
  scalar branch can still overpredict individual intermediate-roughness cases.
