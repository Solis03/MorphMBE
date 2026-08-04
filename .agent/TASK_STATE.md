# RHEED-to-AFM Generative Modeling Task State

Last updated: 2026-08-04 (America/Detroit)

## N6342 sparse-island morphology continuation (completed 2026-08-04)

- Working branch/worktree:
  `codex/n6342-sparse-island-20260804` at
  `/Users/ziyi/Desktop/LAB/code-worktrees/n6342-sparse-island-20260804`.
- Baseline source is the frozen M16/M16b commit `a6aecb6`; desktop archives,
  previous branches, and raw AFM/RHEED data are read-only.
- Domain-expert correction: growth 6081 is now explicitly listed in
  `removelist.txt` and must be absent from all new fitting, validation,
  confidence, tables, atlases, and UI artifacts. The previous 28-growth result
  remains preserved as historical evidence rather than silently overwritten.
- Scientific objective: reduce the excess bright micro-island/peak density in
  the strict held-growth N6342 generated AFM while retaining its Sq/FSMI and
  avoiding degradation across the remaining 26 growths.
- Added 6081 to the canonical removelist before any new fit. The final cohort
  is exactly 27 growths; every outer fold fits 26, and no fold contains 6081
  or its held growth in condition, spectral, or island training.
- Preserved M16b and evaluated ten additional paired renderers using the same
  strict-fold fields and seeds. M17b topology-conditioned sparse-peak terrace
  is selected. It replaces M16b's fixed dense maxima/tanh compression with a
  peak layer whose count comes from the outer-fold RHEED-conditioned q82
  component prediction, while retaining fine spectral texture.
- N6342 strict LOO Sq is 0.833 nm versus measured 0.804 nm (absolute error
  0.029 nm). Relative to M16b, N6342 normalized PSD distance improves
  2.832→0.260, island-feature MAE 1.010→0.890, bright area fraction
  0.0680→0.0509 (measured 0.0540), bright median area 17.0→19.5 px
  (measured 26.0), and height kurtosis 2.651→3.337 (measured 3.327).
- Cohort-wide M17b versus M16b: mean PSD distance 1.454→0.969, island MAE
  1.738→1.718, peak-signature MAE 1.457→1.327, and sharpness ratio
  0.810→0.836. Texture-gate rate decreases 0.852→0.815 and is explicitly
  retained as a limitation. All predictions at/above the 1.6-nm terrace gate
  are bitwise unchanged; 18/27 ensembles are exactly unchanged overall.
- Full 27-growth scalar results after excluding 6081: Sq MAE 1.107 nm,
  Pearson r=0.741, Spearman rho=0.684; FSMI MAE 1.126 nm, r=0.675,
  rho=0.583. M17b joint confidence versus realized error has Spearman
  rho=-0.565 (p=0.0021); it is a reliability index, not a probability.
- All 27 selected maps pass ID, strict-LOO condition, non-retrieval,
  no-measured-patch and SHA-256 integrity checks. Complete atlas, scalar,
  confidence, failure, extra-five, N6342 ablation and peak-topology figures
  are saved as PNG/PDF under
  `reports/rheed_m17_end_to_end_generation/20260804_m17_sparse_topology_line3_full27_v1/full27_loo/figures`.
- Final report and registries:
  `reports/rheed_n6342_sparse_island/REPORT.md`,
  `experiment_registry.csv`, `baseline_vs_final_metrics.csv`,
  `selected_map_integrity.csv`, and `renderer_branch_invariance.csv`.
- N6342 was used for renderer development; its result is retrospective LOO
  evidence, not an untouched prospective-test claim. Desktop archives, raw
  AFM/RHEED data and UI deployment files were not modified in this task.

## M15b automatic-video end-to-end AFM generation (2026-07-29)

- Connected the improved automatic-input M15b scalar/confidence predictions
  to the existing M12a non-retrieval island/terrace generator under a strict
  23-growth outer leave-one-growth-out protocol. Every held growth is absent
  from the 22-growth generator fit; the output-map metadata exactly matches
  the strict M15b outer prediction for that growth.
- Generalized the full-cohort experiment loader to accept one combined
  target table with an explicit `target` column, and generalized publication
  visualization to use a configured external confidence method rather than a
  hard-coded M14i label. Added regression tests for both contracts.
- Final scalar performance is Rq MAE 1.212 nm / Pearson r 0.757 and FSMI MAE
  1.036 nm / r 0.748. Relative to the previous M14i automatic-input baseline,
  MAE improves by 0.325 nm and 0.588 nm respectively.
- Re-measured generated-image performance is Rq MAE 1.197 nm and FSMI MAE
  1.064 nm, versus 1.510/1.350 nm for the previous automatic-input pipeline.
  Median sharpness ratio is effectively retained (0.711 versus 0.713), while
  the texture-gate rate decreases from 0.609 to 0.565; the latter is reported
  as a limitation rather than hidden.
- Produced a fixed-order complete 23-growth atlas, stratified overview,
  scalar scatter, confidence/error audit, renderer strata and largest-failure
  panels as both PDF and PNG. The overview shows automatic RHEED, generated
  AFM and measured AFM for four fixed Rq strata plus a deterministic failure.
- M15b joint reliability confidence is negatively associated with joint
  target error (Spearman rho -0.646, p≈0.001). It is explicitly labeled a
  relative reliability index, not a calibrated probability.
- Verified the configured UI path on the original 6056 MOV. It selects frame
  160, predicts Rq 2.687 nm / FSMI 2.324 nm at 61.5% model confidence and
  generates a field with measured Rq 2.687 nm. Inference is 7.02 s and the
  selector-plus-inference path is 28.75 s on the M1 Pro. Saved the actual
  16-frame input, generated field, three-panel figure and offscreen UI
  screenshot under
  `outputs/rheed_realtime_ui/20260729_m15b_m12a_end_to_end_ui_verification_6056`.
- Corrected stale UI/worker/CLI text that still named the superseded M14i
  scalar head. Added a regression check that the configured bundle, manifest,
  UI subtitle and ROI note all identify M15b+M12a and retain
  retrieval/AFM-patch prohibitions.
- Main report:
  `reports/rheed_m15b_end_to_end_generation/20260729_m15b_m12a_auto_full23_v1/REPORT.md`.
- Final verification: 51 targeted selector, automatic-input, realtime and
  generative regression tests pass. The 23 outer maps pass target-identity,
  22-growth fit, held-growth exclusion, non-retrieval and no-measured-patch
  assertions; the artifact SHA-256 manifest verifies.
- Raw data, `removelist.txt`, prior publication freezes and the desktop
  Standalone were not modified.

## Automatic-input confidence/FSMI robustness continuation (2026-07-29)

- Audited all 23 human and automatic clip caches. Both domains use exactly
  `keyframe_1=k`, `causal_8=k-7..k` and `selected_16=k-7..k+8`; every clip is
  contiguous and the keyframe is always at zero-based index 7. Temporal frame
  count mismatch is ruled out.
- A frame × ROI factorial audit isolates the FSMI failure. Automatic
  keyframes with human ROIs improve strict physics-head FSMI LOO to MAE
  1.225 nm / Pearson r 0.544, versus 1.625 / -0.130 for automatic
  keyframes + V8 ROI. Human keyframes + V8 ROIs extrapolate catastrophically.
  The primary failure is therefore ROI/feature-domain mismatch, not V5
  keyframe selection.
- Current V8 crops preserve the full diffraction family for R3D/generation
  but destabilize percentile-normalized connected-component, skeleton and
  temporal-difference features. A dedicated median-geometry physics ROI
  improves FSMI to MAE 1.359 nm / r 0.455, but remains below the
  automatic-keyframe + human-ROI diagnostic upper bound.
- More importantly, the existing strict nested candidate audit selects the
  causal-8 R3D head in every one of the 23 automatic-domain outer folds for
  both targets. Its held-one-out FSMI is MAE 1.036 nm / r 0.748 and Rq is
  MAE 1.212 nm / r 0.757. The frozen human-domain target-specific mapping,
  not information loss in automatic inputs, selected the wrong deployment
  head.
- Existing automatic-domain confidence was weak because its risk score only
  included R3D density OOD and upper-amplitude extrapolation. It ignored
  frame/ROI perturbation sensitivity and representation conflict.
- Implemented eleven target-blind causal-8 input views: keyframe offsets
  -2/-1/0/+1/+2, ROI shifts left/right/up/down by 3%, and ROI scale 94/100/
  106%. Raw TTA variance fails on stable-but-wrong sample 6057; base-to-TTA
  median displacement is the useful input-instability statistic.
- Identified a second temporal issue: every clip has eight frames, but video
  periods range from about 25 to 40 frames, so the fixed window covers a
  different rotation angle. M15b combines the strictly nested TTA-centrality
  risk with a target-blind empirical rotation-period/angular-coverage risk.
- Independent review rejected the first post-hoc multi-head veto because its
  global LOO reference could indirectly include the outer-held target. That
  result is preserved under `superseded_posthoc_confidence/` and is explicitly
  non-citable. The replacement recomputes temporal/physics disagreement in
  every inner fold. It remains a separate conflict alert and only vetoes a
  genuinely inner-95th-percentile conflict.
- Final strictly nested outer-held confidence-versus-absolute-error Spearman
  is -0.538 (p=0.0081) for Rq and -0.710 (p=0.00015) for FSMI. AURC is 0.817
  nm for Rq and 0.602 nm for FSMI; 50%-coverage risks are 0.753/0.666 nm.
- Integrated `MorphMBE-M15b-AutoR3D-AngularTTA` into the current-workspace
  UI. V8 remains the full-pattern R3D/M12a ROI; Q50 is explicitly labeled as
  a separate physics-feature diagnostic ROI. The replay ring is 18 frames
  so all TTA views are available at the existing k+8 trigger without added
  video lookahead.
- Corrected the headless smoke script to exercise that exact 18-frame path
  rather than silently falling back to a single base view. Raw-video 6056
  selects frame 160 versus human 161, predicts Rq 2.687 nm / FSMI 2.324 nm,
  assigns 61.5% angular-TTA reliability confidence, and generates an AFM
  field with measured Rq 2.687 nm in 7.14 seconds of inference.
- Strict point prediction and confidence evidence uses outer AFM-target LOO.
  The fixed selector is an AFM-target-blind prior preprocessor and is not
  retrained inside these outer folds. Angular-risk composition and the
  95th-percentile conflict diagnostic are exploratory and require prospective
  validation; UI all-23 refits are not held-out evidence.
- Complete report:
  `reports/rheed_auto_input_robustness/20260729_auto_input_robustness_v2/REPORT.md`.
- Final verification: 48 targeted robustness/realtime/generative regression
  tests pass; strict table integrity confirms 23 rows per target, 22-growth
  outer fits, 22-growth confidence calibration and zero outer-target usage.
  Publication figures were visually inspected and the frozen artifacts pass
  their SHA-256 manifest.
- Raw data and the desktop Standalone remain read-only.

## Human-vs-automatic M14i/M12a input-domain comparison (2026-07-29)

- Built a parallel automatic-input dataset for the exact frozen M14i Full23
  cohort, excluding 6043 and 6055, from the same source video/stage as every
  human row. V5 selects the visible rotation cycle, V8 supplies the model-input
  ROI, and `selected_16` remains `k-7..k+8`.
- Dataset geometry: median folded cycle-phase residual 2 frames, ROI IoU
  0.753, human-ROI coverage 0.996 and keyframe quality 0.717.
- Ran three target protocols with the frozen M14i methods/hyperparameters:
  frozen human→human strict LOO, auto→auto strict LOO, and human→auto strict
  cross-domain LOO. Every outer target and confidence calibration is held out.
- Rq remains usable: human / auto / cross-domain Pearson r =
  0.509 / 0.536 / 0.554 and MAE = 1.466 / 1.536 / 1.480 nm. Paired MAE changes
  versus human have bootstrap CIs spanning zero.
- FSMI transfer is weak: auto→auto Pearson r = -0.130. Machine-domain
  confidence is not significantly error-related, unlike the frozen human
  confidence. This negative result is retained and explicitly limits UI
  claims.
- Completed full 23-fold M12a automatic-input generation with 23/23 leakage
  audits passing. Machine-domain AFM sharpness, texture-gate rate and
  AFM-likeness are essentially unchanged from the human-input generator, so
  input automation does not collapse the image renderer.
- Same-weight paired deployment sensitivity (diagnostic, not held-out) finds
  median input-induced shifts of 0.333 nm Rq and 0.480 nm FSMI. For 6056,
  automatic frame 160 vs human 161 gives strict auto LOO Rq 3.277 nm vs true
  3.225 nm; the all-23 deployment gives 3.048 nm with 90.4% model confidence.
- Machine dataset:
  `outputs/rheed_manual_vs_auto_selection/20260729_m14i_human_vs_auto_full23_v1/machine_dataset`.
- Complete report:
  `reports/rheed_manual_vs_auto_selection/20260729_m14i_human_vs_auto_full23_v1/REPORT.md`.
- The desktop Standalone and all raw RHEED/AFM files remain unmodified.

## Realtime ROI/model-input alignment correction (2026-07-29)

- User review of 6056 exposed a real implementation defect: the compact V5
  tracking ROI was both mislabeled and passed to `build_model_clip`, while the
  complete-lattice rectangle was display-only. Secondary dark-cycle events
  could then overwrite a supported prediction. The supplied 4%-confidence
  screenshot is consistent with session frame 203.
- Raw and curated 6056 MOV files have identical SHA-256. The manually curated
  frame-161 clip is reproduced byte-for-byte by the current preprocessing
  code, isolating the fault to runtime ROI/event routing rather than source
  video, decoding or luminance conversion.
- Added a role-specific V8 model-input ROI, fitted on 25 removelist-compliant
  videos with orientation-conditioned q20/q80 boundaries and strict
  leave-one-video-out evaluation. Held-video overlap is zero; median IoU is
  0.727, manual-area coverage 0.986, compact-spot energy coverage 1.000 and
  circular-edge intrusion 0.
- V5 tracking remains internal. The best V5 physical-cycle candidate is
  locally refined in the actual model-input ROI using compact spot energy,
  column alignment and horizontal spread. For 6056 this changes candidate
  frame 146 to frame 160, versus human frame 161. Only this supported cycle
  is predicted during complete-video replay.
- ReplayWorker now constructs the 16-frame clip only from
  `selection.model_input_roi`. The UI shows one cyan
  `模型输入 / 完整点阵 ROI`; tracking and conservative audit ROIs remain
  explicit in session provenance but are not presented as model inputs.
- UI main confidence now reports the error-related M14i model confidence,
  comparable in definition to the frozen LOO confidence. Input quality and
  the more conservative combined score remain separately visible.
- Corrected 6056 raw-video smoke: Rq 3.0484 nm versus measured 3.2254
  (absolute error 0.1770), FSMI 2.9172 nm, model confidence 90.4%,
  non-clipped support and generated-map Rq 3.0484 nm. The prior faulty frame
  203 gave Rq 1.2270 nm, model confidence 1.0% and combined confidence 4.0%.
- Cross-sample 6063 sanity: refined frame 187 versus human 186, Rq
  5.3935 nm and model confidence 45.9%, versus the initial tracking-input
  smoke's 5.2082 nm and 38.2%.
- Realtime/selector tests pass 16/16. Corrected 1540x980 Qt offscreen
  rendering and headless 6056/6063 raw-MOV runs pass. No raw or standalone
  file was modified.

## Simulated real-time RHEED→AFM UI (2026-07-29)

- Working branch:
  `codex/rheed-realtime-morphology-ui-20260729`.
- Implemented a PySide6 desktop application for raw-video replay with sample
  and video dropdowns, adjustable slow playback, RHEED display, V7 full-lattice
  and model-input ROI overlays, generated AFM display, Rq/FSMI cards,
  confidence-colored roughness timeline and terminal-style pipeline logs.
- The V5 DINOv2-S selector and V7 ROI calibration are reused without changing
  their frozen fitted artifacts. The user-approved standalone directory is
  read only; all new caches and sessions are inside this repository.
- The deployed model refits the frozen M14i methods on all 23 allowed growths:
  M14g for Rq and M14b for FSMI. M12a uses selected-16 R3D morphology
  conditions, stochastic Laguerre islands, a non-retrieval spectral prior and
  the edge-preserving terrace renderer. No measured AFM patch or nearest
  neighbor is loaded at inference.
- Temporal semantics are explicit: keyframe at selected-16 index 7, causal-8
  frames `k-7..k`, selected-16 frames `k-7..k+8`, with the event emitted only
  after frame `k+8` arrives.
- On the M1 Pro, deployment-cache fitting takes about 11.3 s; one full
  morphology prediction takes 3.8–4.3 s. Video and prediction run in separate
  threads, and a bounded one-job queue prevents unbounded lag.
- 6063 raw-MOV end-to-end smoke processes 813 frames, finds a 27-frame period,
  selects frame 189 versus human ~186, predicts Rq 5.2082 nm / FSMI 3.7834 nm,
  and generates a height map whose measured Rq is 5.2081 nm. Total selector
  plus inference time is 31.58 s.
- Failure analysis found that a few secondary cycle vertices extrapolate to
  nearly zero Rq with very low confidence. The final deploy layer preserves
  the unconstrained value, clips the displayed/generated amplitude to the
  observed 23-growth support boundary, flags `support_clipped`, expands the
  interval and further reduces confidence. This keeps low-confidence outputs
  visibly AFM-like without pretending the extrapolation is supported.
- Current interface is a complete simulated replay. V5 still performs a
  full-video analysis pass before playback; a truly causal camera selector is
  a documented next step. Historical replay is not new held-out evidence.
- User guide:
  `docs/RHEED_REALTIME_MORPHOLOGY_UI.md`.
- Implementation report:
  `reports/rheed_realtime_ui_report.md`.
- Verification: 13/13 realtime/selector tests and 16/16 adjacent
  M14i/M12a/island/full-cohort tests pass. Installed console entry-point help,
  offscreen Qt rendering, raw 6063 replay and support-clipped failure handling
  were exercised. No file under `data/raw` was modified. The standalone's only
  same-day mtimes predate this task and are pre-existing Finder `.DS_Store`
  files; no standalone artifact was written by this implementation.
- Implementation commit:
  `24ae914` (`feat: add realtime RHEED morphology monitoring UI`).

## Complete-lattice ROI continuation (2026-07-28)

- Working branch:
  `codex/rheed-roi-full-lattice-v7-20260728`.
- Frozen implementation/configuration commit: `ea25f2d`.
- Frozen calibration/results/figure commit: `1ec763b`.
- The user-approved V5 DINOv2-S keyframe selector is unchanged. Its
  `calibrated_safe` ROI remains the internal tracking/scoring geometry; V7
  predicts a separate full-lattice ROI only after frame selection.
- V7 learns four independent aperture-relative boundaries from the 25
  removelist-compliant annotated videos, grouped by landscape/portrait
  orientation. Conservative 5th/95th-percentile geometry, explicit
  top/bottom padding, right transition inclusion and a row-wise circular-arc
  constraint implement the requested full-dot-family crop.
- Evaluation is strict 25-fold leave-one-video-out: each fold fits on 24
  videos and predicts one entirely excluded video. Held-video and removelist
  overlap are both zero.
- Compared with the frozen V4/V5 tracking ROI, selected V7 median compact-spot
  energy coverage is 1.000 versus 0.501; worst coverage is 0.9965 versus
  0.0633; median manual area coverage is 0.975 versus 0.798; right reference
  boundary inclusion is 25/25 versus 1/25; vertical envelope inclusion is
  24/25 versus 9/25; circular eyepiece-edge intrusion is 0/25 versus 18/25.
- The evaluation-only compact-spot metric uses DoG response within the manual
  reference ROI. It does not enter inference. All 25 fixed-order comparison
  crops and the eight numerically lowest-coverage cases are shown rather than
  cherry-picked.
- Final original-MOV smoke inference for 6063 processes 813 frames in
  28.7 seconds, selects frame 189 and exports ROI
  `(x=474, y=54, w=588, h=984)` with zero arc intrusion. A second smoke run
  on 6048 processes 371 PNG frames in 25.1 seconds, selects frame 191 versus
  human 199, and exports a complete-lattice crop with zero arc intrusion.
- Calibration bundle:
  `outputs/rheed_auto_roi_keyframe/20260728_full_lattice_roi_v7/full_lattice_roi_calibration.joblib`.
- Experiment report:
  `reports/rheed_full_lattice_roi_report.md`.
- Verification report:
  `reports/rheed_full_lattice_roi_verification.md`.
- Local compute remains sufficient; no CUDA handoff is indicated for this
  ROI task. Raw data and `removelist.txt` remain unchanged.

## Deep spot-visibility keyframe continuation (2026-07-28)

- Working branch:
  `codex/rheed-keyframe-deep-visibility-20260728`.
- Frozen implementation/configuration commit: `500b9b8`.
- Frozen model/results/figure commit: `e9dac16`.
- Immutable parent evidence is the removelist-compliant V4 selector frozen at
  commits `94b3ed0` and `90b1c4a`; it will not be overwritten.
- User review identifies the primary V4 failure: some selected frames lie in
  diffuse shadow/haze and lack the distinct RHEED spot family visible in the
  human frame (for example sample 6063).
- V5 evaluation keeps the same 25-video removelist-compliant cohort
  and strict leave-one-video-out boundary, adds an explicit shadow/spot
  visibility endpoint, and tests image-content-aware ranking without using
  held-video labels.
- Apple MPS is available; PyTorch 2.12 is installed. Local disk has about
  236 GiB free. CUDA handoff is required only if the next meaningful model is
  estimated to exceed the established local time/speedup threshold.
- Implemented 27 multi-scale spot/haze/frequency descriptors and frozen
  DINOv2 features. The selected 22M-parameter DINOv2-S model uses fold-local
  PCA plus Ridge, pairwise ranking, ExtraTrees and a 25th-percentile
  visibility gate.
- Strict 25-fold leave-one-video-out V5 result: median/mean NCC
  0.820/0.730, SSIM 0.559, gradient NCC 0.583 and median absolute frame
  difference 3. Held-video overlap and removelist overlap are zero.
- The evaluation-only diffuse-shadow proxy rate falls from 16% for V4 to 4%
  for V5. Sample 6063 changes from V4 frame 461 (NCC 0.279) to V5 frame 188
  (NCC 0.784), versus human frame 186. Sample 6022 is the one remaining
  diffuse proxy failure.
- V5 reliability confidence is error-related (Spearman rho -0.459,
  p 0.021). It is an expected-similarity score calibrated from strict-LOO
  selection margins, not a probability of correctness. A stricter
  leave-one-prediction-out calibration audit is nonsignificant (rho -0.130,
  p 0.537); therefore the raw reliability ordering is supported, while
  absolute confidence calibration remains prospective.
- An 86M-parameter DINOv2-Base V6 ablation was tested and rejected: median
  NCC 0.701, SSIM 0.476 and frame difference 148. Larger was not better for
  this small, specialized image domain.
- MPS extracted 598 unique DINOv2-S representations in 35.5 seconds
  (16.9 images/s). Complete original-MOV inference for 6063 processed 813
  frames in 30.1 seconds and selected frame 189 versus human frame 186.
  Local execution is well below the CUDA handoff threshold.
- V5 report:
  `reports/rheed_deep_visibility_keyframe_report.md`.
- V5 fitted ensemble:
  `outputs/rheed_auto_roi_keyframe/20260728_dinov2_spot_visibility_v5/dinov2_spot_visibility_ranker.joblib`.
- Verification: 49/49 targeted tests pass; repository-wide tests report
  335 passed with the same 24 failures and 6 errors from missing unrelated
  freeze/checkpoint/parquet artifacts. V5/V6 split, bundle and figure
  integrity checks pass. Details:
  `reports/rheed_deep_visibility_verification.md`.

## Automatic RHEED ROI and keyframe selection (2026-07-28)

- Working branch:
  `codex/rheed-auto-roi-keyframe-20260728`.
- New read-only tool accepts MOV/MP4/AVI/MKV/MPEG/MPG or a numeric PNG frame
  directory, predicts a circular-border-safe ROI, tracks diffraction motion
  and selects a rotation-phase vertex.
- Ground truth contains 27 manually annotated source videos. Final compliance
  audit excludes 6023 and 6087 because they overlap `removelist.txt`;
  canonical V4 fitting, evaluation and confidence calibration use 25 videos.
  Earlier 27-video V1–V3 results are noncanonical diagnostics only.
- Three ROI methods and six deterministic keyframe methods were compared.
  V1 physical vertex median NCC was 0.602. V2 visibility gating raised it to
  0.669. V3's larger aperture-inscribed ROI was rejected because its median
  NCC fell to 0.568 despite greater human ROI overlap.
- The selected V4 method uses `calibrated_safe` ROI, compact bright-feature
  and whole-diffraction-front trajectories, physical vertex candidates and an
  auditable Ridge candidate ranker.
- Strict leave-one-video-out evaluation trains on 24 retained videos and
  holds one video completely out in each of 25 folds. Held overlap and
  removelist overlap are zero. Median pattern NCC is 0.714, mean NCC 0.670,
  SSIM 0.482, gradient NCC 0.362 and absolute frame difference 46 frames.
  Repeated rotation cycles make absolute difference a secondary metric.
- Predicted similarity is error-related (Spearman rho -0.548, p 0.0046) and
  is isotonic-calibrated as the inference confidence. It is explicitly not a
  correctness probability. Samples 6048, 6063 and 6056 remain visible
  failures.
- Final fitted model is refit on all 25 retained annotations for future prospective
  video use. It must not be described as prospectively validated.
- Direct original-video smoke benchmark: 810 frames / 27 seconds processed in
  12.9 seconds on the M1 Pro (~62.7 frames/s). CUDA is unnecessary.
- Final report:
  `reports/rheed_auto_roi_keyframe_report.md`.
- User guide:
  `docs/AUTOMATIC_RHEED_ROI_KEYFRAME.md`.
- Selected experiment:
  `outputs/rheed_auto_roi_keyframe/20260728_removelist_compliant_final_v4`.
- Complete deterministic ROI/keyframe comparison atlases:
  `reports/rheed_auto_roi_keyframe/20260728_diffraction_front_visibility_v2`.
- Canonical V4 Ridge held-video atlases and confidence figures:
  `reports/rheed_auto_roi_keyframe/20260728_removelist_compliant_final_v4`.
- Verification complete: adjacent selector tests pass 48/48; 167 PNG and 56
  PDF artifacts pass integrity checks; 27/27 raw-source size/mtime audits
  pass; source video inference succeeds. The repository-wide `tests/` run has
  334 passes and unrelated missing-freeze/checkpoint/parquet failures recorded
  in `reports/rheed_auto_roi_keyframe/verification.md`.
- Final diff/raw-data audit and independent scientific review are complete.
- Local implementation commit:
  `7089a66` (`feat: automate RHEED ROI and phase keyframe selection`).
- Frozen experiments/report/figure commit:
  `1eaba50` (`docs: freeze automatic RHEED selection experiments`).
- Canonical removelist-compliant V4 freeze:
  `94b3ed0` (`fix: freeze removelist-compliant RHEED selector`).
- No raw data or `removelist.txt` changes are present. Pre-existing
  `.pytest_cache`, untracked `AGENTS.md` and untracked `tmp/` remain outside
  the task commits. No push was requested or performed.

## Dual paper-model freeze and publication handoff (2026-07-28)

- Freeze ID:
  `rheed_to_afm_dual_generative_models_v1_20260728`.
- MODEL_A canonical name:
  `MorphMBE-M12a-Strict15-RangeTerrace-v1` (`M12a-Strict15`).
  Its primary evidence is strict LOO over 15 development growths
  (14 fit, one held); three pre-existing validation growths are separate;
  five historical-test growths / 24 AFM scans remain closed for that model.
- MODEL_B canonical name:
  `MorphMBE-M14i-Full23-OODAware-v1` (`M14i-Full23`).
  Its primary evidence is retrospective full23 LOO (22 fit, one held).
  Rq uses M14g, FSMI uses M14b, and image generation uses frozen M12a.
- MODEL_A is pinned to source commit
  `dafc94c177becc0015c03f29025e7fa065f0171e`; MODEL_B is pinned to
  `e8ca3012a770bf9e06269670086db349c3da844a`.
- The freeze records 133 MODEL_A artifacts and 277 MODEL_B artifacts with
  SHA-256, Git blob ID and role. This covers 33/41 experiment-code files,
  all frozen parameters, 20/34 result figures, result tables and reports.
- Six derived input snapshots freeze the modeling manifest, AFM descriptor
  table, group-fold table, RHEED physics table, embedding registry and
  canonical removelist. No raw RHEED or AFM data is copied.
- Validation command:
  `PYTHONPATH=. .venv/bin/python scripts/freeze_rheed_to_afm_paper_models.py --validate`.
  Freeze validation passes, and the current RHEED-to-AFM tests pass 30/30.
- Freeze root:
  `paper_freeze/rheed_to_afm_dual_generative_models_v1_20260728`.

## OOD-aware robust continuation (started 2026-07-28)

- Working branch: `codex/rheed-afm-ood-robust-20260728`.
- Immutable parent evidence:
  - M12a 15-growth development milestone remains preserved.
  - M13 full-23 retrospective LOO negative audit remains preserved at
    commits `d4a07a6` and `f329129`.
- New user-requested experiments:
  1. define two to four clearly out-of-domain growths using RHEED-only,
     target-blind support diagnostics, exclude them in a separately labelled
     sensitivity cohort, and rerun the otherwise frozen M12a LOO pipeline;
  2. test fold-local density/leverage weighting and robust regression so rare,
     low-support training samples have less influence while low-support
     queries receive visibly lower confidence.
- Scientific guardrails:
  - Do not modify the canonical `removelist.txt` merely because a sample has
    high AFM prediction error.
  - Keep experiment-specific OOD exclusions in a separate manifest/config.
  - Select exclusions without AFM target values or outer-fold errors.
  - Treat exclusion results as domain-restricted sensitivity evidence, not an
    independent test or proof that removed measurements are invalid.
  - Refit all weighting, scaling, model selection and uncertainty transforms
    inside each outer growth fold.
  - Evaluate confidence through error ranking, interval coverage and
    selective risk-versus-coverage, not by visually assigned scores.
- [x] Complete broad robust-learning, OOD, selective-prediction and
  small-data literature review.
- [x] Freeze a target-blind RHEED OOD detector and exclusion manifest.
- [x] Run the unchanged M12a pipeline on the domain-restricted cohort.
- [x] Implement and test fold-local density/robust weighting candidates.
- [x] Complete nested full-cohort evaluation, failure analysis and iteration.
- [x] Produce figures and reports.
- [x] Complete final Git diff audit and local commits.

## OOD-aware robust continuation evidence

- The target-blind RHEED OOD ranking identifies 6101, 6063, 6029 and
  6028 as the top four atypical growths. AFM targets and held errors are not
  inputs to the ranking. Separate top-2, top-3 and top-4 sensitivity
  manifests were used; the canonical `removelist.txt` was not modified.
- Hard exclusion is rejected. Frozen M12a Rq MAE is 1.910 nm with all 23
  growths, 1.960 nm after top-2 exclusion, 2.321 nm after top-3 and 2.041 nm
  after top-4. Growth 6099, the largest high-Rq failure, ranks 23/23 (most
  in-domain) by the handcrafted RHEED audit, indicating conditional
  ambiguity rather than simple input OOD.
- Target-blind RHEED-density weighting helps, but target-residual self-paced
  weighting is weaker because it can suppress valid rare high-Rq examples.
  A causal temporal R3D view and fixed curated/temporal fusion provide the
  largest balanced Rq improvement.
- Selected M14i uses M14g (60% curated RHEED physics + 40% R3D temporal) for
  Rq and M14b (RHEED-density weighted) for FSMI. Full23 LOO Rq MAE is
  1.466 nm, Pearson r 0.509 and Spearman rho 0.499; FSMI MAE is 1.316 nm,
  r 0.281 and rho 0.430. Relative to the frozen M12a full23 head, MAE falls
  23.3% for Rq and 24.7% for FSMI.
- Confidence uses fold-local temporal support and high-amplitude
  extrapolation risk, never the held AFM target. Confidence versus realized
  absolute error has rho -0.601 for Rq and -0.677 for FSMI. Both 90%
  intervals cover 20/23 growths. Joint target confidence versus realized
  joint error has rho -0.696.
- Retaining the highest-confidence 12/23 growths reduces Rq MAE from
  1.466 to 0.707 nm and FSMI MAE from 1.316 to 0.715 nm. This is reported as
  a selective operational mode, not as a replacement for complete-cohort
  evaluation.
- The integrated full23 experiment reuses the immutable, genuinely
  stochastic M12a island/terrace generator with M14i target predictions.
  No measured AFM patch or AFM retrieval is used at inference. Every one of
  23 generator fold audits has 22 fit growths and zero held overlap.
- Five atlas pages show all 23 growths, and six more figures show target
  correlation, protocol provenance, ordered roughness, confidence,
  renderer strata and the four largest failures. Each is saved as PNG and
  PDF. High-Rq 6099 remains underpredicted (10.32 to 4.64 nm) but receives
  confidence 6/100; 6095 is 9.87 to 4.90 nm at confidence 22/100.
- The result remains retrospective because all 23 growths have now informed
  method development. M14i must be frozen before a new prospective test.
- Final report:
  `reports/rheed_to_afm_ood_robust_report.md`.
- Literature review:
  `reports/rheed_to_afm_ood_robust_literature_review.md`.
- Selected robust-head artifacts:
  `reports/rheed_to_afm_ood_robust/20260728_m14_ood_robust_multiview_v3_final`.
- Selected integrated generator artifacts:
  `reports/rheed_to_afm_ood_robust_generation/20260728_m14_target_specific_m12a_generator_v1/full23_loo`.
- Verification completed so far: all RHEED-to-AFM tests pass 30/30; all 23
  generator folds pass leakage checks; 17 PNG and 17 PDF final figures pass
  integrity checks; three representative PDFs were rendered with Poppler and
  visually inspected; raw-data and canonical-removelist diffs are empty.
- The repository-wide suite is blocked by a missing, ignored human-checkpoint
  artifact in the unrelated `rheed_peak_saddle` workflow after 134 tests
  pass. This is recorded in
  `reports/rheed_to_afm_ood_robust/verification.md`; no placeholder was
  fabricated.
- Implementation/config/test commit:
  `e6a063d` (`feat: add OOD-aware robust RHEED-AFM head`).
- Experiment/report/figure commit:
  `00b412e` (`docs: freeze M14 robust full23 evidence`).
- Final Git audit confirms that raw-data paths and `removelist.txt` have no
  diff. Untracked `AGENTS.md` and `tmp/` and tracked pytest-cache timestamp
  effects are outside the research commits and were not staged.

## Full 23-growth leave-one-out continuation (started 2026-07-28)

- User-approved cohort: the existing harmonized 23-growth, 1 x 1 um AFM
  cohort only. Sample 6043 remains excluded; sample 6055 remains excluded.
- The M12 milestone and its 15-growth development artifacts are immutable
  comparison evidence and will not be overwritten.
- New objective: freeze the M12a method, run retrospective nested
  leave-one-growth-out over all 23 growths (22 fit, one held, repeated 23
  times), refit every target-dependent condition/texture/island transform
  inside each outer fold, and generate a complete quantitative and visual
  comparison against the prior 15-growth result.
- This is retrospective full-cohort cross-validation, not a new prospective
  untouched test. All historical source split labels are retained only as
  provenance and do not restrict the new outer LOO.
- [x] Confirm the source descriptor, RHEED embedding and RHEED physics tables
  contain exactly the same 23 independent growth groups.
- [x] Implement a separate full-cohort nested-LOO runner and config.
- [x] Run leakage smoke checks and the full 23-fold experiment.
- [x] Generate the 23-growth atlas, target scatter, confidence, comparison and
  failure figures.
- [x] Verify reports, raw-data integrity and final diff.

## Full 23-growth LOO evidence

- The fixed cohort contains exactly 23 growth groups / 116 AFM scans. Growths
  6043 and 6055 are absent; all canonical removelist IDs have zero overlap.
- The M12a renderer and method family were frozen before this audit. Every
  outer fold refits all target-, condition-, spectrum- and island-dependent
  components on 22 growths, and all 23 leakage manifests pass.
- The full-cohort hypothesis is rejected. Rq mean/median MAE is
  1.910/1.322 nm, Pearson r is 0.265 and Spearman rho is 0.303. FSMI
  mean/median MAE is 1.748/0.979 nm, r is 0.158 and rho is 0.237.
- The new 22-fit predictions are also worse when restricted to the identical
  15 IDs shown in the prior M12 result. The degradation is not only an effect
  of plotting eight additional difficult groups.
- High-Rq growths 6099 and 6095 are underpredicted by 6.642 and 5.316 nm.
  Growth 6101 is an unstable near-zero extrapolation and produces a visibly
  flat map.
- M12a retains clearer island boundaries than M10 (median contrast 1.631
  versus 1.356), but M10 is better on PSD distance, texture-gate rate,
  island-feature MAE, AFM-prior distance and composite error over all 23.
- Rq/FSMI 90% interval coverage is 20/23 and topology upper coverage is
  22/23. Pointwise confidence does not validate: confidence versus realized
  joint error has rho +0.043 (p=0.846). It must not be presented as a
  correctness probability or reliable error ranking.
- Five atlas pages show every held-out growth. Six additional figures show
  target scatter, old/new protocol comparison, Rq range, confidence audit,
  five roughness strata, and the four largest failures. Eleven PNG and eleven
  PDF files were produced.
- Final audit report:
  `reports/rheed_to_afm_full_cohort_loo_report.md`.

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
- Reports/figures/final-state commit:
  `dafc94c` (`docs: freeze M11 and M12 functional morphology evidence`).

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

---

## AFM metrology repair — 2026-07-29

Status: in progress on branch
`codex/afm-metrology-repair-20260729`.

### Scope and safety

- [x] Preserve the 2026-07-28 publication freeze without editing it.
- [x] Add a sibling status marker:
  `superseded_pending_AFM_metrology_audit`.
- [x] Create a new derived AFM hierarchy; never modify raw AFM/RHEED files.
- [x] Retain per-scan-line polynomial flattening orders 0/1/2/3.
- [x] Use exported NanoScope Rq labels only as independent QC.
- [x] Hash-deduplicate decoded scans and record provenance decisions.
- [x] Aggregate independent scans as median Sq in nm, then take log for models.
- [x] Separate displayed-scan Sq from sample median Sq ± IQR in AFM panels.
- [x] Re-run corrected-target M14i and the M12a generator.
- [x] Develop corrected-target automatic-input M15b under strict outer LOO.
- [x] Integrate the corrected M15b/M12a bundle into the realtime UI.
- [x] Complete raw-source integrity verification, final tests, diff review,
  and report.
- [x] Create the final local commit.

### Metrology evidence

- Derived root: `data/afm_metrology_line3_v1` (180 decoded ZSensor maps).
- Audit root: `outputs/afm_metrology_line3_v1`.
- The 23-sample primary 1 µm QC subset contains 42 NanoScope-labelled scans.
- Order-3 line flattening versus independent NanoScope Rq:
  MAE 0.02240 nm, median absolute error 0.00420 nm,
  Pearson r 0.99979, Spearman rho 0.99737, and 100% within 0.2 nm.
- Exact-array hashing found 14 rows in seven duplicate groups; only one
  representative per group enters aggregation.
- `6094/N6081_1um_000` is conservatively excluded pending lab-notebook
  confirmation. The N69/N74 legacy aliases are retained but explicitly flagged.
- Corrected targets contain 110 deduplicated/provenance-valid primary 1 µm
  scans across 23 growth groups.

### Corrected-target model evidence

- M14i (human input, target-specific historical selection) no longer predicts
  corrected Sq strongly: MAE 1.662 nm, Pearson r 0.233. This negative result is
  retained and must not be hidden.
- M15b (automatic ROI/key-frame R3D angular TTA, strict 23-fold LOO):
  Sq MAE 1.090 nm, Pearson r 0.746, Spearman rho 0.600;
  FSMI MAE 0.980, Pearson r 0.726, Spearman rho 0.580.
- Nested range-aware confidence is inversely related to absolute error:
  Spearman rho = -0.617 for Sq and -0.602 for FSMI.
- M12a was re-fit inside every outer fold (22 training growths, one held out);
  all leakage checks pass, retrieval at inference is false, and measured AFM
  patches are not used at inference.
- The corrected M12a image generator remains limited: median sharpness ratio
  0.719, texture-gate pass 73.9%, median AFM-likeness percentile 13.0%.
  Scalar metrology is substantially stronger than image realism.

### Claim boundary

All 23 samples are strict leave-one-growth-out for scalar and generator fitting.
The method family was developed on earlier retrospective partitions, so this
is retrospective cross-validation rather than a prospective external test.

### Final verification

- AFM source integrity: 180/180 raw AFM hashes and 180/180 decoded ZSensor
  hashes match the pre-derivation audit.
- Changed-component regression suite: 43/43 tests pass; rebuilt UI bundle
  regression suite: 10/10 tests pass.
- New deployment cache:
  `outputs/rheed_realtime_ui/morphmbe_m15b_m12a_line3_metrology_live_v4.joblib`.
- Raw-data Git status is empty for `data/raw`, `data/pair`,
  `data/processed_afm`, and `removelist.txt`.
- Desktop standalone received no task output and no source/model/report edits.
  Finder-created `.DS_Store` activity is external to this pipeline and is not
  included in the repository work.
- Final report:
  `reports/afm_metrology_line3_v1/FINAL_RETRAIN_REPORT.md`.
- Local implementation/results commit:
  `b063d5e` (`repair AFM metrology and retrain corrected models`).

---

## Extra-five line-3 expansion and generated AFM audit — 2026-07-29

Status: completed locally on branch
`codex/afm-extra-five-line3-20260729`.

### Requested scope

- [x] Preserve all prior freezes and the completed 23-growth metrology repair.
- [x] Consolidate the second-batch RHEED/AFM sources without modifying raw data.
- [x] Explicitly exclude N6324; include N6342, N6358, N6382, N6389 and N6390.
- [x] Resolve the `M6358` source-folder alias to canonical sample ID `N6358`.
- [x] Harmonize each 2 × 2 µm AFM scan to non-overlapping 1 × 1 µm subfields
  before third-order per-scan-line flattening; keep all subfields within their
  growth group for every split.
- [x] Rebuild AFM descriptors, automatic RHEED clips, embeddings and manifests
  for the combined cohort.
- [x] Re-run strict leave-one-growth-out Sq/FSMI prediction and confidence.
- [x] Re-fit M10 and M12a inside every outer fold and save generated AFM maps.
- [x] Produce all-sample and extra-five-specific AFM visualizations, scalar
  plots, failure panels and an auditable final report.
- [x] Update the realtime UI deployment bundle only after the expanded
  experiment passes integrity checks.
- [x] Verify raw-source hashes, tests, leakage boundaries and the final diff;
  create local commits only and do not push.

### Safety decisions

- `data/compressedfile` and `data/AFM-extra-five` are immutable raw-source
  roots for this task.
- Existing `*_extra_five` folders are retained as historical derived data.
  Cleanup is additive: a canonical versioned derived root and a provenance
  manifest will supersede the scattered folders without deleting them.
- N6324 may appear only in the exclusion/provenance audit and must never enter
  descriptors, training, validation, generation, confidence calibration or UI
  deployment.

### Final cohort and metrology

- Canonical derived root: `data/extra_five_consolidated_v1`.
- Accepted extra growths: N6342, N6358, N6382, N6389 and N6390.
- Extra AFM inputs: 26 raw 2 × 2 µm ZSensor scans, yielding 104
  non-overlapping 1 × 1 µm subfields after line-3 flattening.
- Combined AFM cohort: 214 1 × 1 µm scans across 28 growth groups.
- N6324's five raw AFM scans remain in the source inventory with decision
  `excluded`; N6324 is absent from all 28-group modeling artifacts.
- The legacy machine-readable key `Rq_nm` is retained only for API
  compatibility; its audited line-3 value is the areal RMS height Sq. Figures,
  report prose and the UI label it Sq.

### Strict full-28 evidence

- Outer protocol: leave one complete growth out, fit 27, repeat 28 times.
- M15b Sq: MAE 1.284 nm, Pearson r 0.661 (p=0.000130),
  Spearman rho 0.506 (p=0.00598).
- M15b FSMI: MAE 1.134 nm, Pearson r 0.661 (p=0.000128),
  Spearman rho 0.499 (p=0.00692).
- Sq confidence versus absolute error: Spearman rho -0.529 (p=0.00381).
  FSMI has the expected negative direction (-0.362) but is only borderline
  at p=0.0581.
- Extra-five-only MAE is 0.585 nm for Sq and 0.536 nm for FSMI, but
  within-five Pearson correlations are negative because this small batch has a
  narrow low-roughness range. This limitation is reported explicitly.
- Adding the extra batch does not improve the original-23 subset:
  Sq/FSMI MAE becomes 1.436/1.265 nm versus 1.090/0.980 nm in the prior
  23-only M15b run.

### Generated AFM evidence

- Both fixed generators save four 128 × 128 draws for every held growth.
- M10 full-28 texture-gate pass is 96.4%, median sharpness ratio 0.821,
  median island-feature MAE 1.421 z.
- M12a full-28 texture-gate pass is 78.6%, median sharpness ratio 0.724,
  median island-feature MAE 1.772 z.
- M10 is therefore the stronger expanded-cohort image-metric comparator.
  M12a remains in the live UI for frozen-version continuity; both result
  families and a direct extra-five comparison figure are preserved.
- Exact training-pixel equality is zero; retrieval and measured AFM patch use
  at inference are false for all maps.
- High-Sq amplitude compression and failure to resolve the fine ordering of
  the extra five remain scientific limitations.

### Final verification and deployment

- Raw integrity: 31/31 extra AFM files match full SHA-256; five selected RHEED
  videos match full SHA-256; all 28 RHEED inventory entries match size/mtime.
- Leakage/integrity: 28 folds, 27 fit growths per fold, zero held overlap;
  28 generated map files and 56 scalar target rows; N6324 absent everywhere.
- Rebuilt M15b prediction SHA-256:
  `b7da65b1bd73516ad50fd0a21f6aed9570e7da0fc3380fe14f2922fb9453deb7`.
- Changed-component test suite: 29/29 passed. The broader historical
  `tests/` collection reports 366 passes, 24 failures and 6 errors. All
  non-passing cases are outside this change and are attributable to missing
  historical paper-freeze manifests, missing peak/saddle and human-review
  checkpoints, or the optional unavailable Parquet reader dependency; none
  is suppressed or represented as passing.
- All 24 delivered full-28 PDF figures are valid one-page PDFs; representative
  rasterized PDF inspection has no clipped or overlapping labels.
- Default UI now uses the additive 28-growth v5 deployment bundle; the prior
  full-23 v4 config remains at
  `configs/rheed_realtime_ui_line3_full23_v4.json`.
- Real 6056 replay smoke: frame 160, Sq 1.66 nm, FSMI 1.39 nm,
  model confidence 72%, inference 6.52 s, retrieval false.
- Final report:
  `reports/extra_five_integration/20260729_line3_full28_v1/FULL28_GENERATION_REPORT.md`.

## 2026-07-29 — N6389/N6390 RHEED orientation correction

### Objective and immutable constraints

- Operator correction: rotate N6389 and N6390 RHEED frames clockwise 90°
  so their diffraction orientation matches the other 26 growths.
- Apply the correction before key-frame selection, ROI inference, physics
  features, DINO/R3D embeddings, perturbation confidence views, UI display and
  UI model input. A display-only transform is invalid.
- Raw videos and raw AFM remain read-only. The transform is an additive,
  versioned derived-data operation with the source hashes preserved.
- Keep AFM line-3 targets and all model hyperparameters fixed; rerun the same
  strict 28-fold leave-one-growth-out scalar and generative protocols so the
  orientation effect is interpretable.
- N6324 remains excluded.

### Current state

- Branch: `codex/rheed-orientation-correction-20260729`.
- Added a shared target-blind frame-orientation transform supporting only
  validated quarter-turns.
- Preserved the provisional `orientation90_v2` experiment as a negative
  ablation. Rotating and then rerunning the selector changed N6389/N6390
  vertices from 1238/1048 to 508/964 and confounded orientation with temporal
  sample choice.
- Final controlled data protocol:
  `20260729_line3_full28_orientation90_keyframe_locked_v3`.
  Model-visible frames and ROIs are clockwise 90°, while the earlier
  target-blind V5 temporal vertices remain locked at 1238/1048.
- The final rotated ROIs are 804 × 899 (N6389) and 791 × 899 (N6390);
  visual audit confirms complete spot-lattice coverage without the eyepiece
  edge entering the model crop.
- Embedding isolation check: the other 26 growths have exact zero change in
  causal-R3D, selected-16 R3D and DINO keyframe embeddings. Only N6389/N6390
  changed.

### Strict full-28 orientation results

- Uncorrected v1 reference:
  Sq MAE 1.284 nm, r 0.661; FSMI MAE 1.134 nm, r 0.661.
- Rotate + automatic reselect negative ablation:
  Sq MAE 1.380 nm, r 0.333; FSMI MAE 1.222 nm, r 0.334.
- Final rotate + locked target-blind vertex:
  Sq MAE 1.321 nm, Pearson r 0.622 (p=0.000409), Spearman rho 0.524
  (p=0.00423); FSMI MAE 1.168 nm, r 0.630 (p=0.000325), Spearman rho
  0.504 (p=0.00623).
- Final confidence versus absolute error:
  Sq rho -0.458 (p=0.0141); FSMI rho -0.403 (p=0.0335).
- The correction is scientifically accepted because it fixes operator-known
  acquisition orientation while retaining significant full-cohort
  association. It is not claimed to improve over the incorrectly oriented
  v1 input; its small metric cost is reported.

### Corrected generated AFM and UI

- All 28 outer folds completed with 27 fit growths and four generated draws
  per renderer/held growth.
- M10: 100% AFM texture-gate pass, median sharpness ratio 0.795, median
  island-feature MAE 1.371 z.
- M12a: 75% texture-gate pass, median sharpness ratio 0.728, median
  island-feature MAE 1.814 z.
- Retrieval, exact training-pixel equality and measured held-AFM patch use at
  inference are all false.
- Final live bundle:
  `outputs/rheed_realtime_ui/morphmbe_m15b_m12a_line3_full28_orientation90_keyframe_locked_live_v7.joblib`.
- Default UI config points to v7 and discovers `data/compressedfile`, excludes
  N6324, rotates 6389/6390 CW 90° and applies the archived-sample vertex lock.
- Real-video smoke:
  N6389 frame 1238, Sq 2.90 nm, FSMI 2.58 nm, model confidence 45%,
  inference 7.26 s; N6390 frame 1048, Sq 2.32 nm, FSMI 2.11 nm,
  confidence 58%, inference 6.80 s.
- Actual offscreen Qt replay screenshot passed for N6389.

### Verification status

- Relevant regression tests: 35/35 passed.
- Broad historical `tests/` collection: 368 passed, 24 failed and 6 errored.
  All non-passing tests are pre-existing artifact/environment gates outside
  this change: missing old paper-freeze manifests, missing peak/saddle human
  checkpoint outputs, or absent optional Parquet engines. The initial
  repository-root collection additionally encounters three duplicate-module
  names inside an archived code snapshot; `pytest tests` avoids that unrelated
  collection-path collision.
- Raw/model audit: 31 AFM source hashes and five selected RHEED hashes pass;
  28 growths, 214 scans, 56 scalar rows, 28 generator folds, 27 fit growths
  per fold, 28 generated map archives, four draws each; zero held overlap.
- N6324/6043/6055 are absent from modeling artifacts.
- Nineteen PNG/PDF figure pairs are present and all PDFs are valid one-page
  documents. PDF raster inspection passed for the corrected generated-AFM,
  all-28 confidence scatter and corrected-sample atlas figures.
- Primary report:
  `reports/extra_five_integration/20260729_line3_full28_orientation90_keyframe_locked_v3/ORIENTATION_CORRECTION_REPORT.md`.

## 2026-07-29 — English-only real-time UI

- Replaced every Chinese user-visible string in
  `src/rheed2morph/realtime/` with consistent scientific English, including
  the window title, controls, ROI annotations, metric cards, confidence
  details, dialogs, replay/selector messages, and pipeline logs.
- Added a regression test that rejects CJK characters anywhere in the
  real-time UI Python package.
- `tests/test_rheed_realtime_ui.py`: 12/12 passed.
- Real N6389 offscreen replay passed using the unchanged orientation-corrected
  v7 bundle: CW 90°, frame 1238, Sq 2.90 nm, FSMI 2.58 nm, confidence 45%.
- English UI verification screenshot:
  `outputs/rheed_realtime_ui/full28_orientation90_v7_ui_N6389_english.png`.
- Raw data, model weights, predictions, and the desktop standalone freeze were
  not modified.

## 2026-07-29 — Causal multi-event real-time UI repair

- Root cause: the UI inherited the paper-evaluation combination of complete
  video pre-analysis plus `best_visible_cycle`, so it knew the event count in
  advance and emitted only one prediction.
- Default UI now uses `causal_stream`; offline LOO/data-building code retains
  its separate frozen single-best-frame protocol.
- The first 48 arrived frames initialize ROI. A vertex at `k` is confirmed
  with bounded lookahead at `k+4`, and prediction is submitted at `k+8` after
  selected-16 context is complete.
- Added an absolute online clear-moment ExtraTrees model trained from 642
  physical candidates in 25 annotated videos. Strict leave-one-video-out:
  MAE 0.0950, Pearson r 0.7089, Spearman rho 0.7117, good-frame AUC 0.8600,
  zero held-video overlap.
- Prediction queue is unbounded in live mode: every accepted event is queued
  rather than dropped when M12a is busy.
- N6342 full 2201-frame causal audit detected 13 clear moments:
  129, 221, 312, 495, 678, 862, 1045, 1228, 1321, 1595, 1962, 2054, 2145.
- Real offscreen Qt test displayed three completed predictions and three
  timeline points; final screenshot:
  `outputs/rheed_realtime_ui/causal_stream_v1_ui_N6342_three_predictions_final.png`.
  Closing during a queued prediction is now safe.
- UI catalog language now distinguishes 58 selectable video sample IDs from
  the 28-growth model-training cohort; out-of-cohort inputs are labeled
  prospective/OOD rather than eligible training samples.
- Report:
  `reports/rheed_realtime_ui/causal_stream_v1_N6342/REPORT.md`.

## 2026-07-29 — N6342 every-event inference completion barrier

- Added separate detected, worker-triggered, queue-accepted, inference-complete
  and scatter-point counters. Video EOF now enters `DRAINING x/N`; the UI only
  declares `COMPLETE N/N` when every count agrees.
- Starting another video is disabled while queued morphology predictions remain,
  preventing cross-session contamination.
- Replayed all 2201 N6342 frames causally at 0.25x duration. The 13 detected
  frames `129, 221, 312, 495, 678, 862, 1045, 1228, 1321, 1595, 1962, 2054,
  2145` produced 13 accepted triggers, 13 completed M15b + M12a predictions,
  13 generated AFM archives and exactly 13 Sq timeline points.
- Generated AFM conditioning fidelity passed for all events; maximum
  `|generated Sq - predicted Sq| = 1.41e-6 nm`. Per-event inference time was
  6.40--7.79 s on the M1 Pro.
- Final UI:
  `outputs/rheed_realtime_ui/causal_stream_v1_ui_N6342_thirteen_predictions_final.png`.
- All-event AFM atlas and timeline:
  `reports/rheed_realtime_ui/causal_stream_v1_N6342/N6342_all_13_generated_afm.{png,pdf}`
  and `ui_thirteen_prediction_timeline.csv`.
- N6342 raw-video SHA-256 remains
  `44d5fd8442380d2e8cbbc7481461c104b6645160518de557da841f760c659986`.

## 2026-07-29 — N6056 zero-event full-lattice fallback

- Root cause: N6056 produced nine primary geometric vertices, but the fixed
  0.40 score threshold and warm-up tracking ROI rejected all of them. The
  compact tracking ROI ended at source x=1044 and omitted the rightmost
  diffraction lattice; the model-input ROI extended to x=1308.
- Added a conservative second causal path on the full-lattice ROI. It is active
  only while the strict path has never succeeded, requires score >=0.30,
  visibility >=1.30, shadow <=0.20, at least eight spots and clarity >=8.0,
  waits through k+8 and separates fallback events by at least 3.0 seconds.
- A four-zero stream is no longer marked `COMPLETE 0/0`; the explicit terminal
  state is `NO CLEAR MOMENT`.
- N6056 full 488-frame audit recovered frames 157 and 314. Frame 157 is four
  frames from the human frame 161. End-to-end Qt replay produced two M15b
  scalar predictions, two M12a AFM maps and exactly two timeline points with
  confidence 59.8% and 80.2%.
- N6342 non-regression passed: all original 13 strict frames are identical and
  zero fallback events were added.
- Tests, figures and report:
  `reports/rheed_realtime_ui/causal_stream_v2_N6056_fallback/`.
- N6056 raw SHA-256 remains
  `8e36f1a697af4986a0f004de8e46be1181f32f0c7eb13ab19bd378f93907c0e6`.
## 2026-07-29 — Sq/Rq metrology re-audit and smooth/extreme morphology model

- Branch: `codex/sq-metrology-smooth-extremes-20260729`
- Worktree:
  `/Users/ziyi/Desktop/LAB/code-worktrees/sq-metrology-smooth-extremes-20260729`
- Base: `db74604991f7fe3bfd099a87b72fdb5e292e0307`
- Historical Desktop standalone folders are read-only and must not be changed.
- Task A: independently re-audit the AFM roughness foundation using the
  mathematical Sq/Rq definition, raw AFM headers, the current third-order
  line-by-line implementation, NanoScope records, official Gwyddion behavior,
  and direct Gwyddion calculations.
- Task B: improve strict full-28 held-growth prediction and AFM generation at
  both physical extremes: push genuinely streaky/smooth 6101 and N6342 below
  1 nm when supported, retain high predictions for spotty 6095/6099, improve
  smooth-surface fine texture, and prevent degradation of global scalar,
  confidence, leakage, or generative metrics.
- Raw data policy: read-only. New derived outputs/reports receive a new
  experiment ID and never overwrite the 20260729 freezes.

### Completed evidence

- Local Gwyddion 2.71 independently reproduced the repository's row-levelled
  Sq on 110 original 1 µm ZSensor maps and 104 extra-five 1 µm subfields.
  Maximum line-3 discrepancy was below `4e-9 nm`; all order-0/1/2/3
  comparisons agreed to numerical round-off.
- NanoScope exported-value QC remains independent of the training target:
  active original-23 deduplicated scans have line-3 MAE `0.0224 nm`,
  Pearson `r=0.9998`, and 42/42 within `0.2 nm`.
- Strict retrospective 28-growth LOO M16 Sq:
  MAE `1.0701 nm`, median AE `0.6748 nm`, Pearson `r=0.7377`,
  smooth-end MAE `0.7590 nm`, rough-end MAE `1.2330 nm`.
- Strict held-growth endpoint predictions: 6101 `0.5666 nm`, N6342
  `0.9545 nm`, N6358 `0.8001 nm`, N6382 `1.067 nm`, 6095 `7.691 nm`,
  and 6099 `6.500 nm`.
- M16 confidence remains target-blind for the outer query; confidence versus
  absolute Sq error Spearman is `-0.4691`, with 25/28 nominal 90% intervals
  covering the held target.
- M16a smooth generation was rejected and preserved as an ablation because it
  amplified high-frequency residuals. M16b micro-island/terrace generation
  improved mean gradient relative error `0.463 -> 0.263`, Laplacian relative
  error `0.897 -> 0.450`, high-PSD relative error `12.35 -> 0.82`, and the
  AFM texture gate `60.7% -> 82.1%`.
- The final 28-fold M16b generator run reports no retrieval, no measured
  held-AFM patch at inference, and no outer-fold overlap.
- New UI bundle:
  `outputs/rheed_realtime_ui/morphmbe_m16_m16b_line3_full28_orientation90_keyframe_locked_live_v8.joblib`.
  Raw-video smoke predictions were 6101 `0.481 nm` in `6.74 s` and N6342
  `0.895 nm` in `7.30 s`; generated map Sq matched conditioning to `~1e-6 nm`.
- 32 endpoint/generator/UI regression tests passed after the final
  retrospective-claim-boundary and legacy-bundle compatibility audit.
- A final source-integrity rerun checked 180 raw AFM files and 180 decoded
  ZSensor arrays; all 360 SHA-256 values match the pre-task audit.
- M16a/M16b configs and manifests now explicitly record that the architecture
  was developed retrospectively on the 28-growth cohort and was not frozen
  before cohort expansion. Strict outer-fold exclusion remains valid, but this
  is not presented as a prospective untouched test.
- The wider `pytest tests/` run produced `377 passed`, `24 failed`, `6 errors`.
  Every failure/error is outside the changed pipeline and is caused by
  worktree-local absence of historical paper-freeze/peak-saddle artifacts or
  the optional `pyarrow` dependency. A repository-root `pytest` additionally
  has three pre-existing duplicate module names inside a historical
  `paper_freeze/.../tests` snapshot. No frozen directory was modified to make
  these unrelated tests pass.
