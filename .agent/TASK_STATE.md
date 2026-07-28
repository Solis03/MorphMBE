# RHEED-to-AFM Generative Modeling Task State

Last updated: 2026-07-27 (America/Detroit)

## Dynamic-range and functional-morphology continuation (started 2026-07-27)

- Working branch:
  `codex/rheed-afm-morphology-index-20260727`.
- Immutable parent milestone: M10 dense-island generation is preserved at
  commit `72ec10d17bbcd8e02cbbd2ba51d79b073531485c`.
- New user-identified failures:
  1. island edges and object shapes remain visually too soft;
  2. Rq predictions are compressed toward a narrow low range, particularly
     missing high-roughness growths;
  3. Rq alone is only an RMS height statistic and does not fully express
     texture, spatial scale, bearing/contact behavior or functional surface
     state.
- New objective: jointly improve generated AFM topology and amplitude
  calibration, design a literature/standards-grounded multiscale functional
  morphology index, predict it in strict held-one-growth evaluation, and
  attach an error-correlated confidence score to every held prediction.
- The historical test remains closed. Model selection remains strict
  training-growth LOO plus the pre-existing validation cohort. All
  `removelist.txt` exclusions remain mandatory.
- [x] Audit Rq compression, high-Rq failures and topology softness.
- [x] Review semiconductor/metrology standards and multiscale surface
  descriptors.
- [x] Define and validate a leakage-safe composite morphology index.
- [x] Implement and compare amplitude-calibrated, sharper island generators.
- [x] Complete strict grouped evaluation and confidence calibration.
- [x] Produce figures, report and verification.
- [x] Complete local commits.

## Dynamic-range and functional-morphology final evidence

- Selected development method: M12a edge-preserving terrace generator. It is
  a stochastic conditional generator with random Laguerre capture zones,
  continuous plateaus/grooves, signed-distance island shoulders and a
  low-weight learned spectral prior. It uses no AFM patch or retrieval at
  inference.
- A strictly nested log-range calibration raises Rq predicted/true standard
  deviation ratio from 0.652 to 0.796. Strict 15-growth LOO Rq mean MAE is
  0.664 nm, median MAE 0.281 nm, Pearson r 0.836 and Spearman rho 0.932.
- The extreme 6095 growth remains underpredicted (9.87 to 5.79 nm) but is
  assigned the cohort's second-lowest confidence, 18.75/100, and is shown in
  the failure figure.
- Experimental FSMI is the RMS of Sq, 31.25 nm height increment, 31.25 nm
  curvature relief, one-quarter p90-p10 bearing span and q70 island
  prominence. It is not presented as an ISO/SEMI standard. Strict LOO FSMI
  mean/median MAE are 0.722/0.417 nm; Pearson r 0.765 and Spearman rho 0.832.
- M12a versus frozen M10 in strict LOO: median Rq error 0.316 vs 0.829 nm;
  FSMI error 0.438 vs 0.684 nm; q70 area log error 0.574 vs 0.811; boundary
  contrast 1.584 vs 1.324; composite 7.963 vs 8.106. Island MAE and AFM-prior
  distance are worse and are reported as explicit sharpness tradeoffs.
- Pre-existing validation, M12a versus M10: median Rq error 0.477 vs
  0.833 nm; FSMI error 0.329 vs 0.659 nm; composite 7.792 vs 7.910; texture
  gate 3/3 for both.
- Confidence versus realized joint-error rank is Spearman rho -0.554
  (p=0.0320). Rq and FSMI 90% interval coverages are 13/15 and 14/15;
  confidence is explicitly a rank index, not a correctness probability.
- M12b was rejected as over-sharpened; M12c improved validation island
  metrics but its confidence-error relation was not significant. M11b remains
  the conservative softer-contour ablation.
- Historical test use is false; 24 historical test rows remain unselected.
  The 11 canonical removelist IDs have zero overlap with retained tables.
- Final report:
  `reports/rheed_to_afm_functional_morphology_report.md`.
- Literature/standards review:
  `reports/rheed_to_afm_functional_morphology_literature_review.md`.
- Final artifacts:
  `reports/rheed_to_afm_functional_morphology/20260727_m12_range_terrace_v1/development`.
- Ten PNG and ten PDF figures were generated. Fig. 1, Fig. 3 and Fig. 8 PDFs
  were rendered with Poppler and visually inspected.
- All RHEED-to-AFM tests pass 23/23. The smoke run is about 30 seconds and the
  full M12 run about 75 seconds on Apple Silicon; MPS is available and CUDA is
  unavailable. The CUDA-handoff condition is not met.
- Implementation/config/test commit:
  `73ea4ae` (`feat: add range-calibrated functional AFM generator`).

## Island-realism continuation (started 2026-07-27)

- Working branch:
  `codex/rheed-afm-island-generation-20260727`.
- Immutable parent milestone: M5 distinct morphology/confidence remains
  preserved at commit
  `d358781ffafccfcd77f7ab1f305081b432b0e5a7`; its reports and outputs must
  not be overwritten.
- New user-identified failure: M5 improves Rq/PSD/condition separation, but
  its Gaussian/Matérn height fields look cloud-like or fibrous rather than
  like measured AFM island, hillock, valley, terrace and coalescence
  topography.
- New scientific hypothesis: the missing inductive bias is object topology.
  RHEED should predict a low-dimensional distribution over nucleation/island
  density, size, aspect ratio, prominence, coalescence, valley fraction and
  multiscale residual texture. A stochastic compositor or
  structure-conditioned refiner should then generate a novel AFM field from
  these quantities.
- Candidate families:
  1. learned AFM island primitive extraction plus marked-point-process
     composition;
  2. nucleation/coalescence-aware procedural growth fields;
  3. structure-conditioned patch diffusion/refinement if the local MPS smoke
     run is scientifically and computationally viable.
- Selection must add island-level distribution metrics and an AFM-realism
  discriminator/embedding audit to the existing Rq, PSD, sharpness,
  condition-sensitivity, diversity and non-copying checks.
- The canonical removelist and growth-group leakage boundary remain
  mandatory. The consumed historical test partition remains closed; new
  method selection uses strict training-group LOO plus the pre-existing
  validation split.
- [x] Preserve and locally commit the M5 parent milestone.
- [x] Complete targeted GaSb/MBE island-growth and structure-conditioned
  generation literature review.
- [x] Build leakage-safe AFM island extraction, statistics and realism
  evaluation.
- [x] Implement and smoke-test multiple island-aware generators.
- [x] Run strict grouped experiments and select the best defensible method.
- [x] Produce figures, confidence audit and final report.
- [x] Complete the final verification and local commit.

## Island-realism final development evidence

- Literature synthesis:
  `reports/rheed_to_afm_island_generation_literature_review.md`. The mandatory
  Na et al. PDF was read in full and pages 3-8 were rendered/visually checked.
- Added 16 object/topology descriptors from q55/q70/q82 island level sets,
  valley level sets, boundary gradients, high-gradient/laplacian texture, and
  flat-pixel fraction. Every scaler and AFM support model is fit inside the
  applicable growth-group fold.
- M6-v1 was retained as a partially negative experiment: its islands were too
  large and sparse. M6-v2 derives capture-zone seed counts from threshold
  coverage, with approximately three times the q70 component count.
- Pre-existing validation, M6c versus M5: condition-descriptor MAE
  0.754 vs 0.940 z; composite error 7.922 vs 8.513; island-feature MAE
  1.596 vs 1.726 z; AFM-prior distance 6.708 vs 7.991; texture gate 3/3
  for both. M6c has explicit islands but still inherits some M5 cloud texture.
- M7 is a true image-space residual DDPM trained on random, growth-group-safe
  real-AFM crops. Full-noise sampling is retained as a failed result because
  it adds excessive granular high-frequency noise. Weak SDEdit-like strength
  0.25 preserves the predicted island map and adds learned edge/terrace
  texture, with validation texture gate 3/3.
- M8 is the fixed 50:50 Pareto combination of weak M7 refinement and M6c.
  It looked promising on the three pre-existing validation groups, but strict
  15-growth LOO did not preserve the object-metric gain. It is retained as an
  important negative experiment rather than selected.
- M9 edge enhancement improved PSD/texture statistics but made too little
  visible topology change; terrace quantization worsened aggregate metrics.
- Selected M10 uses sixfold dense multiscale Laguerre capture-zone
  populations and a 65:35 island/spectral blend. It uses no retrieved or
  measured AFM at inference.
- Strict 15-growth LOO, M10 versus M5: AFM-support distance 6.438 versus
  7.985 (19.4% lower); q70 median-area log error 0.811 versus 1.112 (27.1%
  lower); q70 count log error 0.540 versus 0.611 (11.5% lower);
  RHEED-condition MAE 0.876 versus 0.986 z; PSD distance 0.860 versus 0.925;
  composite 8.106 versus 8.545. Aggregate island MAE is slightly worse
  (1.514 versus 1.492 z) and is reported as a trade-off.
- Pre-existing validation, M10 versus M5: island MAE 1.423 versus 1.726 z;
  AFM-support distance 5.951 versus 7.991; q70 median-area error 0.765 versus
  1.138; composite 7.910 versus 8.513. PSD distance is worse (0.879 versus
  0.627) and is reported explicitly.
- Nested morphology confidence predicts held-group island error from
  inference-time support diagnostics. Confidence versus realized error has
  Spearman rho -0.589 (p=0.0208); the 90% upper error bound covers 14/15
  held groups. The index is not a correctness probability.
- Final report:
  `reports/rheed_to_afm_island_generation_report.md`.
- Selected artifacts:
  `reports/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development/selected`.
- Local MPS timing is approximately 55 seconds per 900-step diffusion fold,
  about 14 minutes total; M10 is approximately two minutes. The CUDA handoff
  condition is not met.
- Final focused verification: package compilation passed; the focused
  island/diffusion/confidence tests pass 9/9 and all RHEED-to-AFM tests pass
  19/19. The wider `tests/` suite has 326 passes and 22 unrelated failures:
  20 require absent historical `rheed_peak_saddle` checkpoint artifacts and
  two require an unavailable parquet engine.
- Final PDF QA: ten PNG and ten PDF figures are present; Fig. 1 and Fig. 7
  were rendered from PDF with Poppler and visually inspected.
- Raw-data and exclusion audit: `git diff -- data` is empty; the 11 canonical
  removal-list IDs have zero overlap with retained model tables.

## Distinct-morphology and confidence continuation (started 2026-07-27)

- Working branch: `codex/rheed-afm-distinct-confidence-20260727`.
- Preserved milestone: branch
  `codex/rheed-afm-sharp-generation-20260727` remains frozen at commit
  `67d35c5d9b85eb7b6134d2d2c73416ad24e1aef7`.
- User-reported failure confirmed quantitatively: validation groups 6022 and
  6056 have visibly different RHEED observations but the selected M2b
  conditions differ by only 0.063 mean absolute standardized units, producing
  nearly indistinguishable fine texture. Unit-Rq rendering also hid amplitude
  differences; all new main figures will use physical nanometre height scales.
- Development policy remains unchanged: the consumed historical test cohort
  is not reopened. Method work uses the 15 training groups, strict nested
  leave-one-growth-group-out checks, and the three pre-existing validation
  groups. A future prospective cohort is required for a new final claim.
- Canonical removelist remains mandatory, SHA-256
  `8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b`.
- Literature-informed direction:
  1. counter regression-to-the-mean with strictly nested variance calibration;
  2. use a descriptor-driven multiscale Matérn random-field generator whose
     correlation scale, anisotropy, spectrum, height distribution, and Rq
     respond explicitly to the RHEED-predicted condition;
  3. report 90% group Jackknife+/cross-conformal intervals and a conservative
     confidence index (explicitly not a probability);
  4. include learning curves to quantify the value of additional growth groups.
- Preliminary strict nested evidence:
  - uncalibrated cross-fitted descriptor MAE 0.825 z, Rq MAE 1.119 nm,
    sensitivity 0.485;
  - a variance-factor cap of 2.0 is the preselected Pareto knee: descriptor
    MAE 0.979 z, Rq MAE 0.739 nm, sensitivity 0.908;
  - nested 90% interval component coverage is 0.933, and interval width versus
    realized point error has Spearman rho 0.536;
  - median descriptor MAE decreases from 1.013 z with 5 training groups to
    0.638 z with 14 groups (strict repeated held-group learning curve).
- Handcrafted RHEED spot/streak/quality temporal summaries were explored with
  target selection repeated inside every outer fold. Their apparent
  full-dataset correlations reversed or became unstable under nested
  evaluation, so they are retained as a negative method result rather than
  used in the selected model.
- [x] Freeze the previous milestone.
- [x] Inspect the cited figure and quantify conditional collapse.
- [x] Review continuous-conditioning, condition-consistency, small-data
  generation, PSD, and conformal uncertainty literature.
- [x] Implement the distinct generator, nested variance calibration, and
  group-conformal confidence package.
- [x] Run smoke tests, full cross-validation, ablations, controls, and failure
  analysis.
- [x] Produce expanded nm-scale comparison, confidence, learning-curve, and
  failure figures with rendered-PDF visual QA.
- [x] Complete removelist/split/raw-data audits, report, independent review,
  and local commit.

## Distinct-confidence selected development evidence

- Selected method: M5 multiscale spectral hybrid. It blends a
  descriptor-driven, condition-sensitive Matérn random field (65%) with the
  learned M2b spectral random-field prior (35%). Both inputs are generated;
  inference uses no measured AFM or retrieval bank.
- Strict 15-group leave-one-growth-group-out versus prior M2b:
  - median Rq absolute error 0.829 vs 1.098 nm;
  - PSD log distance 0.925 vs 0.957;
  - sharpness ratio 0.939 vs 1.174 (target 1);
  - morphology composite 8.545 vs 9.572;
  - texture gate 13/15 vs 14/15;
  - descriptor MAE 0.986 vs 0.826 z (explicit trade-off);
  - median max-training SSIM 0.0372, incompatible with copied AFM output.
- Pre-existing validation versus prior M2b:
  texture gates 3/3 vs 2/3; Rq MAE 0.833 vs 1.205 nm; PSD distance
  0.666 vs 0.860; sharpness 1.065 vs 1.284; composite 8.513 vs 9.604.
- Validation generated-descriptor separation increased from M2b
  0.062/0.464/0.451 z to M5 0.221/1.418/1.285 z. The formerly collapsed
  6022--6056 pair is 3.6x farther apart; the median pair is 2.8x farther.
- Group CV+/Jackknife+ 90% component coverage: 0.933; Rq interval coverage:
  0.933. Confidence versus realized descriptor error Spearman rho: -0.536.
  Intervals are wide, so the maximum conservative confidence index is only
  20/100 and is explicitly not a probability.
- Learning curve median descriptor MAE: 1.013 z at 5 independent growth
  groups, 0.824 at 8, 0.765 at 11, and 0.638 at 14.
- Final development artifacts:
  `reports/rheed_to_afm_distinct_confidence/20260727_m5_hybrid_v4_confidence/development`.
- Final report:
  `reports/rheed_to_afm_distinct_confidence_report.md`.
- Historical consumed test groups remain unused for current model fitting,
  selection, generation, or evaluation. Prospective groups are required for a
  new final claim.
- Local runtime is approximately 30 seconds; CUDA handoff is not recommended.

## Sharp-generation continuation (started 2026-07-27 15:14 -0400)

- Working branch: `codex/rheed-afm-sharp-generation-20260727`
- Starting commit: `ee56f8e0e6fe4a4a8bddb2a1c805bc4cb7bdf7d1`
- New objective: replace the visibly blurred CVAE with a generator that first
  passes AFM-only texture/edge/border-artifact gates, then demonstrate that
  RHEED conditioning changes scientifically meaningful morphology.
- Canonical exclusion source: `removelist.txt`, SHA-256
  `8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b`.
  All 11 listed sample IDs must be excluded before AFM, RHEED, split, training,
  evaluation, or figure construction.
- The prior held-out test cohort is consumed and will not be reused for model
  selection. New development is validation-only; any final claim must use
  group-aware cross-validation or genuinely prospective groups.
- Candidate families are organized separately:
  1. learned conditional spectral random field (no retrieval);
  2. physics-seeded conditional adversarial refiner;
  3. prior CVAE retained only as a blur baseline.
- Mandatory-paper design elements adopted: stochastic conditional input,
  projection discriminator, conditional normalization, spectral
  normalization, hinge loss, differentiable translation/cutout augmentation,
  random-crop expansion, and FFT-domain early stopping/evaluation.
- [x] Enforce the canonical removelist in AFM tables, fold tables, physics
  tables, phase-1 manifests, and embedding payloads (zero surviving overlap).
- [x] Implement and evaluate M2 conditional spectral random fields.
- [x] Implement and evaluate M2b RHEED-descriptor-calibrated random fields.
- [x] Implement and evaluate M3/M3b circular adversarial residual refinement.
- [x] Replace the single PLS condition path with a hybrid Rq/morphology model.
- [x] Add an otherwise identical mean-condition stochastic baseline.
- [x] Complete 15-group leave-one-growth-group-out cross-fitted generation.
- [x] Complete fixed-seed qualitative review and automatic failure analysis.
- [x] Produce PNG/PDF paper figures, registry, best-model manifest, report,
  and reproducibility runbook.

## Sharp-generation final evidence

- Selected development method: hybrid RHEED condition -> conditional spectral
  random field -> 50-step descriptor calibration (M2b).
- This is genuine stochastic generation: no AFM retrieval at inference, zero
  exact training matches, cross-fitted max training SSIM median 0.037.
- Prior CVAE sharpness ratio: 0.441; selected cross-fitted sharpness ratio:
  1.174; selected texture gate: 14/15 growth groups.
- Cross-fitted RHEED-conditioned versus mean condition:
  descriptor MAE 0.826 versus 0.849 z; Rq MAE 1.098 versus 1.392 nm.
- Separate validation: M2b descriptor MAE 0.659 z, sharpness 1.284, texture
  gate 2/3, cyclic condition wins 3/3. The mean condition is slightly better
  on descriptor MAE (0.631 z), so strong conditional generalization is not
  claimed.
- Optional M3b adversarial refiner: validation descriptor MAE 0.628 z,
  sharpness 1.231, texture gate 3/3, condition wins 2/3.
- Primary residual failure: 6022 large connected islands are under-resolved;
  15-group cyclic condition wins are 53%; raw cross-fold Rq rank remains
  negative because predictions strongly shrink toward the training mean.
- The old five-group test cohort was not reused.
- Full GAN runtime: 395 seconds on MPS. CUDA handoff not recommended.
- Final focused verification: 11/11 tests pass and package compilation passes.
  The wider active suite has 316 passes and 23 unrelated failures caused by
  absent historical checkpoint artifacts or an unavailable parquet engine.
- Final PDF figures open successfully; Fig. 8 was rendered from PDF and
  visually checked. No data file changed after branch creation, and
  `git diff -- data` is empty.
- Final report: `reports/rheed_to_afm_sharp_generation_report.md`.

## Objective

Build and rigorously validate a genuine RHEED-conditioned AFM morphology
generator on leakage-safe held-out data. Preserve all retrieval methods as
baselines only, keep raw data immutable, and produce reproducible experiments,
scientific figures, manifests, and a final report.

## Repository state

- Working branch: `codex/rheed-afm-generative-20260727`
- Starting commit: `aa00163f3cd560c0d0561ea979590f2de0f62551`
- Inherited untracked file: `AGENTS.md` (read and obeyed; do not edit or commit)
- Compute: Apple M1 Pro, 32 GiB unified memory
- PyTorch environment: `.venv` via `uv run`
- PyTorch 2.12.0; MPS built and available; CUDA unavailable
- External derived-artifact capacity: `/Volumes/Portable1TB` (526 GiB free)

## Safety invariants

- Never modify files under raw RHEED or AFM source-data paths.
- Treat growth/sample groups as split boundaries.
- Select methods and hyperparameters with training/validation data only.
- Evaluate the frozen selected method on the held-out test set once.
- Do not alter publication freezes.
- Do not push, publish, deploy, use secrets, or delete irreplaceable data.

## Progress

- [x] Read `AGENTS.md`.
- [x] Inspect initial Git state and history.
- [x] Create dedicated local branch.
- [x] Detect local compute and storage.
- [x] Inventory existing data, splits, models, results, tests, and reports.
- [x] Read and visually inspect the mandatory PDF.
- [x] Complete broad literature search and write the synthesis with citations.
- [x] Reproduce local nearest-RHEED retrieval and unconditional baselines.
- [x] Audit split integrity and leakage controls.
- [x] Implement and smoke-test the chosen generative improvement.
- [x] Run validation-guided experiments and ablations.
- [x] Freeze the selected model, then evaluate the test set exactly once.
- [x] Produce publication-quality tables and figures.
- [x] Write report, experiment registry, best-model manifest, and run history.
- [x] Run the full test suite and classify all non-passing checks.
- [x] Inspect logs/diff, review independently, and verify raw-data immutability.
- [x] Commit implementation locally (`9292da2`).
- [x] Commit reports, figures, artifacts, and final task state locally
  (`f9c72a1`).

## Final scientific status

The implemented model is genuinely generative and not retrieval. It produces
non-identical stochastic AFM ensembles with zero exact training-image matches.
However, the one-time held-out condition-permutation control passed for only
1/5 test groups, Rq rank correlation was -0.40, and visual QA found smoothing
and a systematic lower-edge decoder artifact. Strong RHEED-conditioned
generalization is therefore not supported. This is a rigorous negative result,
not a success claim.

## Decisions and current evidence

- Fixed pre-existing group folds: train = 15 growth groups / 68 AFM scans;
  validation = fold 0, 3 groups / 24 scans; test = fold 1, 5 groups / 24 scans.
- Model: RHEED visual/physics features -> regularized AFM morphology descriptor
  predictor -> learned conditional Gaussian prior -> compact conditional VAE
  decoder. Inference never receives or retrieves an AFM image.
- Conditions: log Rq, unit-Rq Ra, PSD mid/high fractions and slope,
  autocorrelation length, anisotropy, skewness, and kurtosis.
- Validation-only temporal ablation compares DINOv2 key frame, DINOv2 centered
  8-frame window, and R3D-18 selected 16-frame window.
- Selected temporal input: centered 8-frame DINOv2, ridge alpha 100.
- Validation variants: bottleneck CVAE; FiLM; diversity-aware FiLM; strong
  condition/diversity balance; intermediate condition/diversity trade-off.
- Predeclared model gate: zero training identities, diversity ratio >= 0.5,
  correct condition wins >= 2/3 validation groups, then lowest morphology
  composite.
- Selected v5: validation composite 6.500 vs retrieval 8.820; diversity 0.875;
  correct condition wins 3/3; zero exact identities; epoch 65.
- Frozen checkpoint SHA-256:
  `7025b3398a18b516e686ceba5033d594d6c6871093b5588effab3c03e5149e52`.
- Held-out test: 5 growth groups / 24 scans; CVAE composite 6.248 vs retrieval
  8.508; Rq MAE 0.822 nm vs 5.775 nm; PSD distance 0.396 vs 0.424; diversity
  0.676; exact identities 0; condition control 1/5; Rq Spearman -0.40.
- Full runs took 55--128 seconds on MPS, so the CUDA handoff condition was not
  triggered. The immediate blocker is scientific generalization and decoder
  design, not local throughput.

## Approval gates / stop conditions

- Explicit user approval is required before any remote push, PR, publication,
  deployment, secret access, paid compute, raw-data deletion, or irreversible
  external action.
- If the next scientifically important experiment is estimated to exceed
  30 minutes locally and an NVIDIA CUDA machine is likely to be at least 10x
  faster, freeze the local state and produce the requested CUDA handoff package.
