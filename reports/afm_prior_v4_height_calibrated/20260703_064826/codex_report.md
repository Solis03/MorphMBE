# AFM Prior V4 MVP-5 Height Calibration Report

Run root: `reports/afm_prior_v4_height_calibrated/20260703_064826`

## Scope Statement

MVP-5 implements physical height-scale calibration for generated AFM maps and evaluates a V2-richness/V3-control hybrid path. It reuses the MVP-3 AE v2, MVP-3 diffusion v2, and MVP-4 condition schema/regressor artifacts. It does not retrain the RHEED encoder and does not use retrieval generation.

MVP-5 improves physical height-scale calibration and AFM prior selection. It does not by itself prove strong RHEED-to-AFM predictive accuracy.

## Environment And Git Status

- `pwd`: `/home/wangziyi/MorphMBE/MorphMBE`
- Python: `3.12.3`
- Torch: `2.12.0+cu130`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 5090`

Git status before MVP-5:

```text
?? reports/afm_condition_control_v3/
?? reports/afm_prior_v2/
?? reports/conditional_latent_diffusion_mvp/
?? reports/rheed_conditioned_latent_diffusion_mvp/
?? src/rheed2morph/generative/
?? tests/test_generative_afm_latent_diffusion.py
?? tests/test_generative_afm_prior_v2.py
?? tests/test_generative_afm_prior_v4_height_calibration.py
?? tests/test_generative_condition_control_v3.py
?? tests/test_generative_rheed_conditioned_diffusion.py
```

Git status after MVP-5:

```text
?? reports/afm_condition_control_v3/
?? reports/afm_prior_v2/
?? reports/afm_prior_v4_height_calibrated/
?? reports/conditional_latent_diffusion_mvp/
?? reports/rheed_conditioned_latent_diffusion_mvp/
?? src/rheed2morph/generative/
?? tests/test_generative_afm_latent_diffusion.py
?? tests/test_generative_afm_prior_v2.py
?? tests/test_generative_afm_prior_v4_height_calibration.py
?? tests/test_generative_condition_control_v3.py
?? tests/test_generative_rheed_conditioned_diffusion.py
```

## Files Created Or Modified

New MVP-5 source files:

- `src/rheed2morph/generative/height_calibration_v4.py`
- `src/rheed2morph/generative/analyze_height_normalization_v4.py`
- `src/rheed2morph/generative/sample_calibrated_v2_v3.py`
- `src/rheed2morph/generative/sample_afm_prior_v4.py`
- `src/rheed2morph/generative/evaluate_afm_prior_v4.py`
- `src/rheed2morph/generative/compare_v2_v3_v4_generation.py`
- `src/rheed2morph/generative/rerun_rheed_conditioned_with_v4_prior.py`

New test:

- `tests/test_generative_afm_prior_v4_height_calibration.py`

Updated:

- `src/rheed2morph/generative/rerun_rheed_conditioned_with_v4_prior.py`

## Dependency Summary

MVP-3:

- Report: `reports/afm_prior_v2/20260703_052537/codex_report.md`
- AE checkpoint: `reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt`
- Diffusion v2 checkpoint: `reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/checkpoints/ema_last.pt`
- Latent shape: `[16, 16, 16]`
- Diffusion v2 generated std mean from prior report: `0.631862`

MVP-4:

- Report: `reports/afm_condition_control_v3/20260703_060549/codex_report.md`
- Condition schema: `reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json`
- Condition table: `reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_table_v3.csv`
- Diffusion v3 checkpoint: `reports/afm_condition_control_v3/20260703_060549/latent_diffusion_v3/checkpoints/ema_last.pt`
- V3 generated std mean from prior report: `0.376785`

Condition schema used for v4 calibration:

```text
rq, ra, robust_range, mean_abs_gradient, gradient_std, gradient_anisotropy,
psd_low_power, psd_mid_power, psd_high_power, psd_slope,
autocorrelation_length_px, island_count, island_mean_area_px
```

## Exact Commands Run

Height diagnosis:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.analyze_height_normalization_v4 --mvp3-root reports/afm_prior_v2/20260703_052537 --mvp4-root reports/afm_condition_control_v3/20260703_060549 --out reports/afm_prior_v4_height_calibrated/20260703_064826/height_diagnosis
```

Calibrated v2/v3 smoke:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.sample_calibrated_v2_v3 --mvp3-root reports/afm_prior_v2/20260703_052537 --mvp4-root reports/afm_condition_control_v3/20260703_060549 --v2-diffusion reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/checkpoints/ema_last.pt --v3-diffusion reports/afm_condition_control_v3/20260703_060549/latent_diffusion_v3/checkpoints/ema_last.pt --autoencoder reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --condition-table reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_table_v3.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --split val --num-samples-per-condition 2 --keep-top-k 1 --ddim-steps 10 --guidance-scale 1.5 --calibration-mode weighted_rq_ra_range --rerank true --max-conditions 2 --out reports/afm_prior_v4_height_calibrated/20260703_064826/calibrated_v2_v3_smoke
```

Full calibrated v2/v3 sampling:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.sample_calibrated_v2_v3 --mvp3-root reports/afm_prior_v2/20260703_052537 --mvp4-root reports/afm_condition_control_v3/20260703_060549 --v2-diffusion reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/checkpoints/ema_last.pt --v3-diffusion reports/afm_condition_control_v3/20260703_060549/latent_diffusion_v3/checkpoints/ema_last.pt --autoencoder reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --condition-table reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_table_v3.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --split val --num-samples-per-condition 16 --keep-top-k 4 --ddim-steps 100 --guidance-scale 1.5 --calibration-mode weighted_rq_ra_range --rerank true --max-conditions 4 --out reports/afm_prior_v4_height_calibrated/20260703_064826/calibrated_v2_v3
```

V4 production sampling, using calibrated v2 as the primary generator:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.sample_afm_prior_v4 --fallback-v2-diffusion reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/checkpoints/ema_last.pt --fallback-v3-diffusion reports/afm_condition_control_v3/20260703_060549/latent_diffusion_v3/checkpoints/ema_last.pt --autoencoder-checkpoint reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --condition-table reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_table_v3.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --split val --num-samples-per-condition 32 --keep-top-k 4 --ddim-steps 100 --guidance-scale 1.5 --descriptor-guidance-weight 0.03 --calibration-mode weighted_rq_ra_range --rerank true --max-conditions 4 --out reports/afm_prior_v4_height_calibrated/20260703_064826/samples_v4
```

V4 evaluation and comparison:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.evaluate_afm_prior_v4 --v4-root reports/afm_prior_v4_height_calibrated/20260703_064826 --mvp3-root reports/afm_prior_v2/20260703_052537 --mvp4-root reports/afm_condition_control_v3/20260703_060549 --samples-v4 reports/afm_prior_v4_height_calibrated/20260703_064826/samples_v4 --out reports/afm_prior_v4_height_calibrated/20260703_064826/evaluation_v4
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.compare_v2_v3_v4_generation --evaluation-v4 reports/afm_prior_v4_height_calibrated/20260703_064826/evaluation_v4 --out reports/afm_prior_v4_height_calibrated/20260703_064826/v2_v3_v4_comparison
```

RHEED-conditioned v4 rerun:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.rerun_rheed_conditioned_with_v4_prior --mvp2-root reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816 --mvp3-root reports/afm_prior_v2/20260703_052537 --mvp4-root reports/afm_condition_control_v3/20260703_060549 --v4-root reports/afm_prior_v4_height_calibrated/20260703_064826 --primary-generator auto --autoencoder reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --v2-diffusion reports/afm_prior_v2/20260703_052537/latent_diffusion_v2/checkpoints/ema_last.pt --v3-diffusion reports/afm_condition_control_v3/20260703_060549/latent_diffusion_v3/checkpoints/ema_last.pt --v4-diffusion reports/afm_prior_v4_height_calibrated/20260703_064826/latent_diffusion_v4/checkpoints/ema_last.pt --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --split val --num-samples-per-condition 32 --keep-top-k 4 --ddim-steps 100 --guidance-scale 1.5 --descriptor-guidance-weight 0.03 --calibration-mode weighted_rq_ra_range --rerank true --fill-missing-with-train-mean --out reports/afm_prior_v4_height_calibrated/20260703_064826/rheed_conditioned_v4_prior
```

Tests and scans:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests/test_generative_afm_prior_v4_height_calibration.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
rg -n "kneighbors|nearestneighbors|nearest_neighbors" src/rheed2morph/generative/height_calibration_v4.py src/rheed2morph/generative/analyze_height_normalization_v4.py src/rheed2morph/generative/sample_calibrated_v2_v3.py src/rheed2morph/generative/sample_afm_prior_v4.py src/rheed2morph/generative/evaluate_afm_prior_v4.py src/rheed2morph/generative/rerun_rheed_conditioned_with_v4_prior.py || true
```

## Height Normalization Diagnosis

Output:

- `reports/afm_prior_v4_height_calibrated/20260703_064826/height_diagnosis`

Key findings:

- Rows analyzed: `168`
- Train rows: `115`
- Per-image normalized rate: `1.000`
- Network min/max median: `-1.0 / 1.0`
- Network Rq median: `0.408728`
- Physical Rq median: `3.718932`
- Physical-vs-network Pearson: Rq `0.473648`, Ra `0.538248`, robust range `0.121351`

Scale distribution over train rows:

- 1st percentile: `2.273653`
- Median: `7.755424`
- Mean: `9.764219`
- 99th percentile: `37.017104`
- Min/max: `1.620463 / 49.422011`

Units audit:

- Network input units: per-image normalized network space.
- Raw condition-table descriptor columns: physical descriptors.
- `cond_*` columns: standardized train-set descriptors.
- Decoder output units: normalized network-input height map.
- Absolute roughness requires external scale: `true`.

Artifacts:

- `height_diagnosis/height_normalization_report.md`
- `height_diagnosis/height_scale_table.csv`
- `height_diagnosis/height_scale_summary.json`
- `height_diagnosis/descriptor_units_audit.json`
- `height_diagnosis/physical_vs_network_roughness_scatter.png`
- `height_diagnosis/scale_factor_histograms.png`

## Height Calibration Method

For generated normalized decoder output `y`, MVP-5 writes calibrated physical height:

```text
h_nm = offset_nm + scale_nm_per_unit * y
```

For descriptors that scale linearly with amplitude, the scale is fit by weighted least squares:

```text
scale = sum_j w_j * generated_d_norm_j * target_d_nm_j
        / sum_j w_j * generated_d_norm_j^2
```

Default mode:

- `weighted_rq_ra_range`
- weights: Rq `1.0`, Ra `0.8`, robust range `0.7`
- offset: target median/height mean if present, otherwise `0`
- clamp bounds: train-scale 1st to 99th percentiles, `2.273653` to `37.017104`

Clamp rate:

- Full calibrated v2/v3: `0.000`
- Samples v4: `0.000`

## Calibrated V2/V3 Results

Output:

- `reports/afm_prior_v4_height_calibrated/20260703_064826/calibrated_v2_v3`

Configuration:

- Split: `val`
- Conditions: `4`
- Candidates per condition: `16`
- DDIM steps: `100`
- Guidance scale: `1.5`
- Calibration mode: `weighted_rq_ra_range`
- Keep top K: `4`

Calibrated v2 results:

| Descriptor | Before MAE | After top1 MAE |
| --- | ---: | ---: |
| Rq | 4.900323 | 2.280160 |
| Ra | 3.713067 | 2.598685 |
| robust range | 24.482457 | 1.864465 |

Calibrated v3 result:

- Rq MAE before/after top1: `5.157971 / 0.210826`

Decision:

- Calibrated v3 gives stronger roughness calibration.
- Calibrated v2 preserves substantially more normalized visual richness.
- Calibrated v2 is the recommended production prior; calibrated v3 is useful as a control/calibration auxiliary.

Visual richness proxy from `samples_v4/calibrated_generation_metrics.csv`:

| Prior | Candidate count | Mean normalized std | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| v2 | 128 | 0.631805 | 0.607923 | 0.652868 |
| v3 | 128 | 0.374579 | 0.355875 | 0.397120 |

Artifacts:

- `calibrated_v2_v3/calibrated_v2_v3_oracle_grid_val.png`
- `calibrated_v2_v3/roughness_calibration_examples.png`
- `calibrated_v2_v3/calibration_failure_cases.png`
- `calibrated_v2_v3/calibrated_generation_metrics.csv`
- `calibrated_v2_v3/calibrated_generation_summary.json`
- `calibrated_v2_v3/height_calibration_metrics_v4.csv`

## V4 Training Decision

No separate v4 diffusion checkpoint was trained in this run.

Reason:

- The fast calibrated v2/v3 diagnostic satisfied the main decision rule: calibrated v2 preserves MVP-3-style visual richness and improves Rq/Ra/range calibration versus uncalibrated v2/v3.
- Training a separate v4 diffusion risked repeating the MVP-4 v3 failure mode: better calibration with lower visual richness.

`sample_afm_prior_v4.py` therefore writes v4 outputs using calibrated v2 as the primary generator and records:

- `primary_generator`: `calibrated_v2`
- `decision`: `calibrated_v2_as_v4_primary`
- `used_trained_v4_checkpoint`: `false`

## V4 Sampling And Evaluation

Samples output:

- `reports/afm_prior_v4_height_calibrated/20260703_064826/samples_v4`

Evaluation output:

- `reports/afm_prior_v4_height_calibrated/20260703_064826/evaluation_v4`

V4 summary:

- Recommended primary prior: `calibrated_v2`
- Roughness improved: `true`
- Nonconstant rate: `1.000`
- Normalized std mean: `0.503192`
- Scale clamp rate: `0.000`

V2 before/after calibration in `samples_v4`:

| Descriptor | Before MAE | After top1 MAE |
| --- | ---: | ---: |
| Rq | 4.900190 | 2.245276 |
| Ra | 3.712865 | 2.533406 |
| robust range | 24.481661 | 1.811918 |

V3 calibrated top1, for comparison:

| Descriptor | After top1 MAE |
| --- | ---: |
| Rq | 0.164624 |
| Ra | 0.295143 |
| robust range | 0.109429 |

V4 artifacts:

- `samples_v4/afm_prior_v4_oracle_grid_val.png`
- `samples_v4/afm_prior_v4_roughness_sweep.png`
- `samples_v4/afm_prior_v4_range_sweep.png`
- `samples_v4/afm_prior_v4_psd_autocorr_sweep.png`
- `samples_v4/afm_prior_v4_random_grid.png`
- `samples_v4/afm_prior_v4_failure_cases.png`
- `samples_v4/generated_candidates_v4.npz`
- `samples_v4/generation_metrics_v4.csv`
- `samples_v4/generation_summary_v4.json`
- `samples_v4/reranking_metrics_v4.csv`
- `samples_v4/height_calibration_metrics_v4.csv`
- `samples_v4/roughness_sweep_metrics_v4.csv`
- `evaluation_v4/afm_prior_v4_summary.json`
- `evaluation_v4/afm_prior_v4_metrics.csv`
- `evaluation_v4/v2_v3_v4_descriptor_comparison.csv`
- `evaluation_v4/requested_vs_generated_roughness_v4.png`
- `evaluation_v4/requested_vs_generated_all_descriptors_v4.png`
- `evaluation_v4/descriptor_distribution_v2_v3_v4.png`
- `evaluation_v4/visual_richness_v2_v3_v4.png`
- `evaluation_v4/v2_v3_v4_visual_comparison_grid.png`
- `evaluation_v4/nearest_real_diagnostic_v4.png`

## RHEED-Conditioned V4 Prior

Output:

- `reports/afm_prior_v4_height_calibrated/20260703_064826/rheed_conditioned_v4_prior`

Condition adapter:

- Predicted table: `reports/rheed_conditioned_latent_diffusion_mvp/20260703_043816/predicted_conditions_10epoch_visual_handcrafted/predicted_condition_table_val.csv`
- Exact descriptor-name mapping was used.
- `--fill-missing-with-train-mean` was explicitly enabled.
- Predicted/oracle filled descriptors: `[]`
- Mean-condition baseline filled descriptors by design.

Summary:

- Primary generator: `calibrated_v2`
- Generated nonconstant rate: `1.000`
- Generated normalized std mean: `0.510669`
- This remains two-stage RHEED-conditioned generation and does not retrain the RHEED encoder.

Artifacts:

- `rheed_conditioned_v4_prior/rheed_conditioned_v4_prior_grid.png`
- `rheed_conditioned_v4_prior/rheed_conditioned_v4_metrics.csv`
- `rheed_conditioned_v4_prior/rheed_conditioned_v4_calibration_metrics.csv`
- `rheed_conditioned_v4_prior/condition_adapter_report.md`
- `rheed_conditioned_v4_prior/rheed_conditioned_v4_summary.json`

## Tests

Focused v4 tests:

```text
Ran 8 tests in 1.284s
OK
```

Full test discovery after all MVP-5 changes:

```text
Ran 63 tests in 11.440s
OK
```

The full suite emitted existing NumPy/sklearn warnings in tiny synthetic tests, but no failures.

KNN marker scan:

- No matches for `kneighbors`, `nearestneighbors`, or `nearest_neighbors` in the v4 path.

## Acceptance Check

- Unit tests pass: yes.
- Existing tests pass: yes.
- `height_normalization_report.md` exists: yes.
- `height_scale_table.csv` exists: yes.
- `calibrated_generation_summary.json` exists: yes.
- At least one calibrated v2/v3 comparison grid exists: yes.
- `afm_prior_v4_summary.json` exists: yes.
- RHEED-conditioned v4 comparison attempted and documented: yes.
- New v4 path does not use KNN: yes.
- No exact AFM reconstruction claim is made: yes.
- If v4 training worsens visual richness: separate v4 training was skipped by decision rule; calibrated v2 is recommended as the production prior.

## Known Limitations

- V4 calibration fixes amplitude scale after generation; it does not make the diffusion model intrinsically control physical roughness.
- Calibrated v3 gives better Rq/Ra/range MAE but remains visually less rich than v2 in normalized std.
- Calibrated v2 preserves visual richness but roughness MAE remains higher than calibrated v3.
- Evaluation is on a small representative validation subset, not all val/test rows.
- Calibrated physical descriptors depend on the target condition quality. For RHEED-predicted conditions, bad descriptor predictions still produce bad physical targets.
- The nearest-real diagnostic grid is diagnostic only; generation uses diffusion sampling, not retrieval or copying.

## Recommended Next Command

If you want to test whether v4 diffusion can match calibrated v2 richness while inheriting v3 control, train from the v2 prior with very weak auxiliary shape loss and use calibrated descriptors during sampling:

```bash
PYTHONPATH=src .venv/bin/python -m rheed2morph.generative.train_afm_latent_diffusion_v3 --latents-dir reports/afm_prior_v2/20260703_052537/latents_v2 --autoencoder-checkpoint reports/afm_prior_v2/20260703_052537/afm_autoencoder_v2/checkpoints/best.pt --condition-table reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_table_v3.csv --condition-schema reports/afm_condition_control_v3/20260703_060549/condition_schema_v3/condition_schema_v3.json --latent-descriptor-regressor reports/afm_condition_control_v3/20260703_060549/latent_descriptor_regressor/checkpoints/best.pt --out reports/afm_prior_v4_height_calibrated/20260703_064826/latent_diffusion_v4_probe --epochs 200 --batch-size 64 --lr 5e-5 --timesteps 1000 --prediction-target epsilon --beta-schedule cosine --cond-dropout 0.10 --aux-cond-loss-weight 0.02 --prototype-balance true --sample-every 50 --amp --ema --seed 42
```
