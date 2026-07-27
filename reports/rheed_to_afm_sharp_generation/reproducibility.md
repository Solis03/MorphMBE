# Reproducibility and command history

Environment: Apple M1 Pro, 32 GiB unified memory, PyTorch 2.12.0, MPS
available, CUDA unavailable. Commands run from the repository root with the
existing `uv` environment.

## Canonical run

```bash
PYTHONPATH=. uv run python -m analysis.rheed_to_afm_sharp_generation.run gan \
  --config configs/rheed_to_afm_sharp_generation.json \
  --device auto
```

The run uses fixed seed `314159`, writes only derived artifacts, and executes:

1. removelist filtering and split-integrity checks;
2. RHEED predictor candidate selection;
3. conditional spectral model fitting;
4. stochastic IAAFT generation;
5. descriptor calibration;
6. conditional adversarial refinement and validation early stopping;
7. validation metrics, mean-condition control, cyclic condition control,
   nearest-training identity audit, and figures;
8. leave-one-training-growth-group-out cross-fitted generation.

## Tests

```bash
PYTHONPATH=. uv run pytest -q \
  tests/test_rheed_single_frame_manual.py::SingleFrameManualTest::test_canonical_removelist_discovery_and_fail_closed \
  tests/test_rheed_to_afm_conditional_vae.py \
  tests/test_rheed_to_afm_sharp_generation.py

uv run python -m compileall -q \
  analysis/rheed_to_afm_generation \
  analysis/rheed_to_afm_sharp_generation
```

The focused command passes 11/11 checks. `PYTHONPATH=. uv run pytest -q tests`
passes 316 checks and fails 23 unrelated artifact-dependent checks: the
historical peak/saddle workflow expects untracked checkpoint outputs, and two
video/AFM-story checks require an unavailable parquet engine. Unrestricted
root collection additionally finds duplicate module basenames inside the
immutable paper-freeze code snapshot.

## Earlier organized experiments

```bash
# v1 ridge-conditioned spectral generator and adversarial refiner
PYTHONPATH=. uv run python -m analysis.rheed_to_afm_sharp_generation.run gan \
  --config configs/experiments/rheed_to_afm_sharp_generation_v1_ridge.json

# v2 PLS-conditioned spectral generator
PYTHONPATH=. uv run python -m analysis.rheed_to_afm_sharp_generation.run spectral \
  --config configs/experiments/rheed_to_afm_sharp_generation_v2_pls.json

# v3 descriptor-calibrated spectral generator and adversarial refiner
PYTHONPATH=. uv run python -m analysis.rheed_to_afm_sharp_generation.run gan \
  --config configs/experiments/rheed_to_afm_sharp_generation_v3_calibrated.json
```

The old test fold was not rerun during any sharp-generation experiment.
Machine-readable hashes and split/removelist audits are in
`20260727_sharp_v4_hybrid/development/development_manifest.json`.
