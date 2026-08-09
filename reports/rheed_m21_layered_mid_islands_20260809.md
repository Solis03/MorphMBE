# M21 connectivity-adaptive layered-island redesign

## Outcome

M21 improves the intermediate-growth AFM morphology while preserving the
accepted M20 smooth and rough endpoints exactly. The selected renderer is
`M21c_strong_growth_layer` from
`configs/rheed_m21_layered_mid_islands_full27_v4.json`.

The desktop M17 standalone package and its result folders were read only for
the presentation baseline. All new implementation and derived results are in
this repository.

## Diagnosis

M20 used one population of mutually repulsive islands for every rough surface.
That is appropriate for the early, high-Sq stage, but it leaves broad exposed
substrate regions around Sq 4–6 nm. The real intermediate AFMs instead show a
dense, partially coalesced coat: additional islands occupy first-layer gaps or
grow on existing islands, leaving narrow cracks rather than large empty areas.

The learned M20 component count was already reasonable. For example, 6028 had
about 30 generated q70 components versus 24 measured, so merely increasing the
number of independent bright objects would not address the topology. The
missing mechanism was overlapping second-layer growth and gap filling.

## Method

1. Pass the strict outer-LOO predicted Sq into the island generator as a
   target-blind growth-stage coordinate. The held growth's measured Sq or AFM
   is never used for generation.
2. Keep the accepted M20 separated-island layer as the base.
3. Smoothly activate a second island population as predicted Sq falls from the
   rough tail toward the intermediate regime. These islands are additive, so a
   nucleus can partly fill a substrate gap or increase an older island's
   height.
4. Increase second-layer coverage and island radius for the strong candidate.
   The second population is large enough to create a dense coat rather than
   decorative fine speckles.
5. Use the target-blind RHEED spot-isolation score to interpolate placement:
   bridged patterns favor gap nucleation and isolated spot patterns retain more
   random stacked growth.
6. Below predicted Sq 2.2 nm, retain the accepted fine-grain renderer. At and
   above 7.6 nm, reuse the accepted M20 separated-island arrays exactly.

Weak, medium, strong, global-gap, and connectivity-adaptive variants were
evaluated. Strong connectivity-adaptive layering was selected because the task
specifically targets intermediate morphology; it has the best intermediate
island-feature score, while medium layering has a slightly better all-cohort
average.

## Results

| Evaluation subset | Growths | M20 island MAE z | M21 island MAE z | Relative improvement |
|---|---:|---:|---:|---:|
| Measured Sq 3.5–6.0 nm | 5 | 1.600 | 1.243 | 22.3% |
| Predicted layered-intermediate regime | 3 | 1.368 | 1.101 | 19.5% |
| All 27 growths | 27 | 1.413 | 1.348 | 4.6% |

The Sq prediction head is intentionally unchanged from accepted M20:

- all-27 MAE: 0.685 nm;
- all-27 RMSE: 0.829 nm;
- Pearson r: 0.923;
- 6028: measured 5.214 nm, predicted 5.190 nm;
- 6063: measured 5.246 nm, predicted 4.675 nm;
- 6099: measured 9.395 nm, predicted 8.459 nm.

The new model visibly converts the isolated M20 objects around 6028/6063 into
a denser, overlapping island coat with smaller intervening gaps. It does not
alter 6099's accepted early-growth islands or N6342's accepted smooth texture.

## Endpoint and leakage protection

- Twelve growths with predicted Sq below 2.2 nm or at least 7.6 nm are exactly
  equal to the accepted M20 arrays (`maximum height difference = 0.0 nm`).
- This protected set includes N6342, 6095, and 6099.
- Every outer fold fits 26 growth groups and holds out one complete growth
  group.
- `all_outer_fold_leakage_checks_passed = true`.
- Retrieval and measured query-AFM use at inference are both false.
- Raw RHEED/AFM data, removal lists, publication freezes, desktop standalone
  code, and desktop standalone results were not modified.

The layer strengths and regime thresholds were selected retrospectively.
Prospective confirmation, especially for new 4–6 nm samples, is still required.

## Visualization

The six-page presentation atlas retains the requested format:

- held-out RHEED, measured AFM, original standalone M17, and M21;
- exact Gwyddion.net black–rust-orange–gold–white palette;
- an independent 0.5–99.5% physical height range and height bar for every AFM;
- a focused measured/M20/M21 comparison for 6028, 6063, and 6099;
- an ordered measured-versus-predicted Sq plot with intervals and signed error.

## Main artifacts

- Final config: `configs/rheed_m21_layered_mid_islands_full27_v4.json`
- Full derived maps:
  `outputs/rheed_m21_layered_mid_islands/20260809_m21_full27_v4/full27_loo/`
- Full reports and audits:
  `reports/rheed_m21_layered_mid_islands/20260809_m21_full27_v4/full27_loo/`
- Gwyddion atlas:
  `reports/rheed_m21_layered_mid_islands/20260809_m21_full27_v4/full27_loo/figures/gwyddion_individual_height_atlas_M17_standalone_vs_M21/`
- Reproducible comparison:
  `layered_growth_summary.csv`, `layered_growth_per_group_audit.csv`,
  `protected_regime_array_audit.csv`, and
  `layered_growth_audit_manifest.json` inside the report directory.

## Known limitation

M21 materially improves coverage and coalescence, but some intermediate
realizations still under-represent the continuity of broad measured plateaus.
The present target-blind dataset is too small to justify a more complex learned
nucleation policy without overfitting; a prospective intermediate-Sq batch is
the appropriate next validation step.
