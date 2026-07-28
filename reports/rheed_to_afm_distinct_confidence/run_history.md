# Reproduction run history

All commands were run from `/Users/ziyi/Desktop/LAB/code` using the repository
environment and `PYTHONPATH=.`.

## Environment

- Hardware: Apple M1 Pro, 32 GiB unified memory.
- CUDA: unavailable.
- PyTorch MPS: available; the selected M4/M5 experiments are CPU/scikit/NumPy
  dominated and finish in approximately 30 seconds.
- Canonical removelist SHA-256:
  `8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b`.

## Main commands

```bash
PYTHONPATH=. uv run pytest -q \
  tests/test_rheed_to_afm_distinct_confidence.py \
  tests/test_rheed_to_afm_sharp_generation.py

PYTHONPATH=. uv run python -m \
  analysis.rheed_to_afm_distinct_confidence.run development \
  --config configs/rheed_to_afm_distinct_confidence.json
```

The development command performs:

1. removelist and growth-group split audits;
2. strict 15-group leave-one-growth-group-out prediction and generation;
3. nested variance-cap ablation;
4. M2b, M4a, M4b and M5 comparison;
5. three-group pre-existing validation evaluation;
6. nested group CV+/Jackknife+ uncertainty audit;
7. repeated learning-curve evaluation at 5, 8, 11 and 14 training groups;
8. PNG/PDF figure generation.

## Final focused checks

```bash
PYTHONPATH=. uv run python -m compileall -q \
  analysis/rheed_to_afm_distinct_confidence

PYTHONPATH=. uv run pytest -q \
  tests/test_rheed_to_afm_distinct_confidence.py \
  tests/test_rheed_to_afm_sharp_generation.py
```

Publication PDFs are rendered with the bundled Poppler `pdftoppm` executable
and inspected as raster images before handoff.
