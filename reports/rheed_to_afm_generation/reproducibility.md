# Reproducibility runbook

Run from `/Users/ziyi/Desktop/LAB/code` with the repository `.venv` through
`uv run`. No global Python packages are required.

## Environment check

```bash
uv run python -c 'import platform, torch; print(platform.platform()); print(torch.__version__); print("MPS", torch.backends.mps.is_available()); print("CUDA", torch.cuda.is_available())'
```

Observed: Apple M1 Pro, PyTorch 2.12.0, MPS available, CUDA unavailable.

## Tests

```bash
PYTHONPATH=. uv run python -m unittest tests.test_rheed_to_afm_conditional_vae
PYTHONPATH=. uv run python -m unittest discover -s tests
PYTHONPATH=. uv run python -m compileall -q analysis/rheed_to_afm_generation
```

Observed for this task: the focused suite passed 4/4. Repository discovery ran
285 tests: 264 passed, while 1 failed and 20 errored in unrelated pre-existing
tests. `test_rheed_peak_saddle.py` expects ignored artifacts absent from
`outputs/rheed_peak_saddle/`; `test_rheed_single_frame_manual.py` also has a
macOS `/private/var` versus `/var` temporary-path normalization error. The new
generative module byte-compiled successfully. `ruff` was not installed in the
project environment, so no lint result is claimed.

## Selected experiment

The selected config is `configs/rheed_to_afm_generation.json`. A content-
identical archival copy is
`configs/experiments/rheed_to_afm_generation_v5_selected.json`.

Smoke:

```bash
PYTHONPATH=. uv run python -m analysis.rheed_to_afm_generation.run smoke \
  --config configs/rheed_to_afm_generation.json \
  --device auto
```

Development selection:

```bash
PYTHONPATH=. uv run python -m analysis.rheed_to_afm_generation.run develop \
  --config configs/rheed_to_afm_generation.json \
  --device auto
```

Held-out test:

```bash
PYTHONPATH=. uv run python -m analysis.rheed_to_afm_generation.run test \
  --config configs/rheed_to_afm_generation.json \
  --device auto
```

The test command is intentionally single-use. It verifies the config,
checkpoint, predictor, and split hashes and raises `FileExistsError` when
`test_evaluation_manifest.json` already exists. Do not remove that manifest to
rerun or tune against the test set.

## Validation experiments actually run

```text
v1  outputs/reports root: 20260727_cvae
v2  outputs/reports root: 20260727_cvae_film
v3  outputs/reports root: 20260727_cvae_film_diverse
v4  outputs/reports root: 20260727_cvae_film_balanced
v5  outputs/reports root: 20260727_cvae_film_tradeoff
```

Each full run first had a two-epoch smoke run. The machine-readable outcomes
are in `reports/rheed_to_afm_generation/experiment_registry.csv`.

## Artifact verification

```bash
shasum -a 256 \
  reports/rheed_to_afm_generation/artifacts/selected_conditional_vae.pt \
  reports/rheed_to_afm_generation/artifacts/selected_rheed_descriptor_predictor.joblib \
  configs/rheed_to_afm_generation.json \
  reports/rheed_to_afm_generation/20260727_cvae_film_tradeoff/split_manifest.csv
```

Expected:

```text
7025b3398a18b516e686ceba5033d594d6c6871093b5588effab3c03e5149e52  selected_conditional_vae.pt
5037364e93c2792da414405f71362c709ea9fcfeaf8e984a50e2963b182b7800  selected_rheed_descriptor_predictor.joblib
0942a72475a8f620aa3031a05cab55c78a8107a518ed0494b9ebd3a8cd0171ae  rheed_to_afm_generation.json
5a3954e002544ae0b981f518505859863ab656a994b506b7629436396243e9ae  split_manifest.csv
```

The canonical-JSON config hash used by the runner is
`859eb3409a8697e7142396467d418a56c21ad4fc22215149ded50c81b65a3c71`.

## Raw-data audit

Task branch creation occurred at 2026-07-27 14:09:56 −0400. The final audit
uses:

```bash
find data -type f -newermt '2026-07-27 14:00:00' -print
git status --short
```

The `find` command returned zero files. All derived arrays, models, metrics,
and figures were written under `outputs/` or `reports/`.

## Storage

The selected tracked artifacts are about 2.8 MB. The full ignored experiment
tree is about 44 MB under `outputs/rheed_to_afm_generation`. The mounted
`/Volumes/Portable1TB` had about 526 GB free, so offloading was unnecessary.
