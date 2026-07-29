# AFM metrology audit and corrected-model retraining

Date: 2026-07-29

Branch: `codex/afm-metrology-repair-20260729`

## Executive decision

The previous publication freeze is preserved byte-for-byte, but a sibling
status record marks it `superseded_pending_AFM_metrology_audit`. It must not be
used for new roughness claims until the metrology correction is reviewed.

The repaired target is the areal RMS height **Sq** calculated from a decoded
ZSensor height map after an independent cubic polynomial is removed from every
fast-scan line. NanoScope values embedded in exported TIF filenames retain the
software's original **Rq** label and are used only as independent QC.

All corrected model targets are constructed by:

1. restricting to the documented primary 1 × 1 µm scans;
2. excluding unresolved provenance;
3. deduplicating exact height arrays by SHA-256;
4. computing each scan's Sq in nm;
5. taking the arithmetic sample median Sq in nm and retaining its IQR;
6. applying the natural log only after physical-space aggregation.

## Why the correction is defensible

The old global areal plane subtraction does not reproduce the NanoScope export
workflow closely enough. A scan-line sensitivity audit over flatten orders
0/1/2/3 shows a monotonic improvement, with the cubic line correction selected
before retraining.

| Scope | order | matched scans | MAE (nm) | median AE (nm) | Pearson r | within 0.2 nm |
|---|---:|---:|---:|---:|---:|---:|
| all matched scans | 0 | 77 | 0.3758 | 0.0987 | 0.9522 | 58.4% |
| all matched scans | 1 | 77 | 0.2773 | 0.0789 | 0.9673 | 75.3% |
| all matched scans | 2 | 77 | 0.1543 | 0.0613 | 0.9893 | 87.0% |
| all matched scans | 3 | 77 | 0.0606 | 0.0043 | 0.9970 | 94.8% |
| active 23, primary 1 µm, deduplicated | 3 | 42 | **0.0224** | **0.0042** | **0.9998** | **100%** |

The selected active-cohort result also has Spearman ρ = 0.9974. This is an
independent QC comparison; the exported NanoScope number is never used as the
training target.

## Deduplication and provenance

The audit processed 180 decoded ZSensor maps. SHA-256 identified 14 rows in
seven exact-array duplicate groups, including copied 6056 scans, a duplicated
6033 scan, and the same 6094 array stored under incompatible scale/name labels.
Only duplicate rank zero enters aggregation.

Manual local provenance review reached these conservative decisions:

- `6094/N6081_1um_000` is excluded. Its explicit four-digit ID conflicts with
  the containing sample, and its raw hash is different from the genuine
  `6081/N6081_1um_000`; local evidence cannot assign it safely.
- `6078/N74_ctr_004` and `N74_ctr_005` are retained but flagged because their
  corrected Sq is consistent with the surrounding N78 scan family.
- The 6070 files use N69 names throughout; they are retained but flagged,
  because excluding that legacy alias would remove the entire sample.

The latter two decisions still require lab-notebook confirmation before a new
paper freeze. No missing measurement has been fabricated.

## Corrected targets

The final modeling table contains 110 deduplicated, provenance-valid primary
1 µm scans across 23 growth groups. Sample-level Sq spans 0.481–9.395 nm.
Within-sample heterogeneity is explicit: for example, 6047 has median
Sq 3.35 nm and IQR 3.02 nm.

Every real-AFM comparison panel now reports both:

- `displayed scan Sq`, for the exact image being shown; and
- `sample median Sq ± IQR`, for the model target.

This prevents a selected image patch from being mislabeled as the sample-level
measurement.

## Retraining results

All scalar numbers below are strict leave-one-growth-out over all 23 growths:
22 growths fit the fold, and the held AFM target is not used for point
prediction, confidence, interval calibration, or method selection.

| input / model | target | MAE (nm) | Pearson r | Spearman ρ | confidence vs absolute error ρ |
|---|---|---:|---:|---:|---:|
| human M14i | Sq | 1.662 | 0.233 | 0.237 | -0.524 |
| human M14i | FSMI | 1.332 | 0.189 | 0.318 | -0.549 |
| automatic corrected M14i | Sq | 1.525 | 0.435 | 0.242 | -0.111 |
| automatic corrected M14i | FSMI | 1.707 | -0.287 | -0.399 | -0.112 |
| **automatic M15b** | **Sq** | **1.090** | **0.746** | **0.600** | **-0.617** |
| **automatic M15b** | **FSMI** | **0.980** | **0.726** | **0.580** | **-0.602** |

M15b uses the automatically selected causal R3D clip. Its confidence is
strictly nested and combines 75% predicted-amplitude support risk with 25%
angular/key-frame/ROI TTA risk, plus a discrete 10% extreme
temporal-versus-physics disagreement veto. The confidence-error correlations
are statistically significant (Sq p=0.0017; FSMI p=0.0024). Empirical 90%
interval coverage is 20/23 = 87.0% for both targets.

The result is scientifically different from the superseded version: M14i's old
target-specific mapping is weak after correcting the AFM target, and this
negative result is retained. M15b is the best locally defensible corrected
scalar model.

## M12a generation rerun

M12a was re-fit inside every outer fold using only the other 22 growths.
Integrity checks confirm:

- 23/23 generator-fold leakage checks pass;
- four independent generated 128 × 128 maps per held growth;
- `retrieval_at_inference = false`;
- `measured_afm_patch_at_inference = false`;
- the amplitude metadata exactly matches the strict M15b prediction hash.

Corrected M12a image metrics:

| metric | result |
|---|---:|
| generated Sq MAE | 1.090 nm |
| generated FSMI MAE | 0.982 |
| texture-gate pass fraction | 73.9% |
| median sharpness ratio | 0.719 |
| median AFM-likeness percentile | 13.0% |
| mean island-feature MAE | 1.945 z |

The generated maps are genuine non-retrieval outputs and preserve the predicted
height amplitude, but image realism is not solved: several high-Sq or unusual
held samples remain visibly over-smoothed, and the low AFM-likeness percentile
quantifies that limitation. The strongest claim of this repair is the corrected
metrology and scalar/uncertainty model, not photorealistic AFM reconstruction.

## Realtime UI integration

The current workspace UI now loads:

`outputs/rheed_realtime_ui/morphmbe_m15b_m12a_line3_metrology_live_v4.joblib`

The builder verifies that all frozen M12a architecture and renderer
hyperparameters remain identical, while the AFM descriptors, sample targets,
and confidence reference must hash to the audited line-3 artifacts. The UI
labels the result as Sq, uses nearest-neighbor display interpolation to avoid
visual smoothing, and reports range-aware confidence.

A full headless replay of raw sample 6056 selected frame 160 and the same
complete-lattice ROI used by the UI. It predicted Sq 2.435 nm, FSMI 2.064,
model confidence 55.2%, and generated-map Sq 2.435 nm. Single-event inference
was 7.0 s on the local Apple Silicon environment. This is a deployment smoke,
not a held-out performance estimate.

## Integrity and claim boundary

- All 180 raw AFM source hashes and all 180 decoded-source hashes still match
  the values captured before derivation (360/360).
- `data/raw`, `data/pair`, and `data/processed_afm` have no Git changes.
- The desktop standalone was not used as an output destination and was not
  edited by this pipeline.
- The 23-fold results are retrospective cross-validation. The M12a family and
  M15b confidence design were developed while this cohort existed; neither is
  a prospective external validation.
- A pre-registered new growth with AFM measured only after RHEED inference is
  the required next credibility step.
- The local M1 Pro completed all relevant fits in minutes, so a CUDA handoff is
  not warranted for this repair.

## Reproducibility

Key commands:

```bash
PYTHONPATH=. .venv/bin/python -m analysis.afm_metrology_repair.run \
  --config configs/afm_metrology_line3_v1.json
PYTHONPATH=. .venv/bin/python -m analysis.afm_metrology_repair.build_descriptors \
  --config configs/rheed_video_afm_story_phase3a_line3_v1.yaml
PYTHONPATH=. .venv/bin/python -m analysis.rheed_to_afm_ood_robust.run \
  --config configs/rheed_to_afm_ood_robust_line3_v1.json
PYTHONPATH=. .venv/bin/python -m analysis.rheed_auto_input_robustness.run \
  --config configs/rheed_auto_input_robustness_line3_v1.json
PYTHONPATH=. .venv/bin/python -m analysis.rheed_to_afm_full_cohort_loo.run \
  --config configs/rheed_m15b_end_to_end_generation_line3_v1.json --device cpu
PYTHONPATH=. .venv/bin/python scripts/prepare_rheed_realtime_model.py \
  --config configs/rheed_realtime_ui.json
PYTHONPATH=. .venv/bin/python -m analysis.afm_metrology_repair.verify_integrity
```

Primary artifacts:

- metrology audit: `outputs/afm_metrology_line3_v1`
- corrected derived maps: `data/afm_metrology_line3_v1`
- corrected descriptors: `outputs/rheed_video_afm_story/phase3a_line3_v1`
- M14i report: `reports/rheed_to_afm_ood_robust/20260729_m14i_line3_metrology_full23_v1`
- M15b/confidence report: `reports/rheed_auto_input_robustness/20260729_m15b_line3_metrology_v1`
- M12a end-to-end report:
  `reports/rheed_m15b_end_to_end_generation/20260729_m15b_m12a_line3_auto_full23_v1/full23_loo`
- UI smoke:
  `outputs/rheed_realtime_ui/headless_smoke_line3_v4_6056`

Publication figures:

- flatten-order/NanoScope QC:
  `figures/Fig1_flatten_order_nanoscope_qc.png`
- corrected sample Sq targets:
  `figures/Fig2_corrected_sample_sq_targets.png`
- strict LOO Sq/FSMI scatter:
  `../rheed_auto_input_robustness/20260729_m15b_line3_metrology_v1/figures/Fig1_m15b_target_predictions.png`
- confidence versus error:
  `../rheed_auto_input_robustness/20260729_m15b_line3_metrology_v1/figures/Fig2_confidence_vs_error.png`
- RHEED/generated AFM/measured AFM overview:
  `../rheed_m15b_end_to_end_generation/20260729_m15b_m12a_line3_auto_full23_v1/full23_loo/figures/Fig0_m15b_end_to_end_overview.png`
