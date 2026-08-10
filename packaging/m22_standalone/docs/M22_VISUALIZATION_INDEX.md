# M22 visualization index

All AFMs use the Gwyddion black-rust-gold-white palette and an independent
physical height bar. Atlas columns are held-out RHEED, measured AFM, desktop
standalone M17, M22 inclusive, and M22 morphology-exclusion ablation.

Figure root:

`reports/rheed_m22_dense_mid/20260809_m22_paired_comparison/figures/gwyddion_individual_height_atlas_M17_vs_M22_dual/`

## Full atlas

| Page | Growth IDs | File |
|---|---|---|
| 1/6 | 6101, N6342, N6358, N6382, 6084 | `Atlas_01_of_06.png` |
| 2/6 | 6072, 6078, 6022, 6048, 6082 | `Atlas_02_of_06.png` |
| 3/6 | N6390, N6389, 6033, 6085, 6029 | `Atlas_03_of_06.png` |
| 4/6 | 6090, 6056, 6070, 6062, 6047 | `Atlas_04_of_06.png` |
| 5/6 | 6080, 6057, 6094, 6028, 6063 | `Atlas_05_of_06.png` |
| 6/6 | 6095, 6099 | `Atlas_06_of_06.png` |

## Focus and scalar plots

- `Focus_true_Sq_3p5_to_6p0_M17_vs_M22_dual.png`: five intermediate-state
  growths 6080, 6057, 6094, 6028, and 6063.
- `M22_Sq_measured_vs_predicted_ordered.png`: all 27 strict outer-LOO Sq
  measurements and predictions.
- `reproduced_outputs/model_smoke_6063/rheed_to_generated_afm_panel.png`:
  command-line/UI deployment smoke showing source RHEED, selected model ROI,
  and the new M22 AFM generation.
- `reproduced_outputs/ui_offscreen_6063/ui_offscreen.png`: actual headless UI
  launch on the 6063 rampdown video, including the `M20 + M22c | READY` badge,
  causal event, generated AFM, Sq/FSMI/confidence cards, timeline, and pipeline
  log.

Associated CSV audits are beside the figures and under
`reports/rheed_m22_dense_mid/20260809_m22_paired_comparison/`.
