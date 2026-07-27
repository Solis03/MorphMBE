# RHEED-to-AFM Generative Modeling Task State

Last updated: 2026-07-27 (America/Detroit)

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
- [ ] Commit reports, figures, artifacts, and final task state locally.

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
