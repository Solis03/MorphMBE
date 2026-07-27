# RHEED-conditioned AFM generation

This package implements the leakage-safe research pipeline documented in
`reports/rheed_to_afm_generation_report.md`.

## Inference graph

```text
RHEED centered 8-frame window ─┐
                               ├─ train-fitted PCA/scalers + ridge
RHEED physics summaries ───────┘             │
                                             ▼
                              predicted AFM morphology descriptors
                                             │
                                             ▼
                              learned Gaussian prior p(z | condition)
                                             │ sample
                                             ▼
                                 FiLM-conditioned AFM decoder
                                             │
                                             ▼
                            unit-Rq morphology × predicted Rq (nm)
```

No AFM image or retrieval bank is accessed at inference. Retrieval is
implemented only inside evaluation as a comparator.

Entry point:

```bash
PYTHONPATH=. uv run python -m analysis.rheed_to_afm_generation.run \
  {smoke,develop,test} \
  --config configs/rheed_to_afm_generation.json \
  --device auto
```

The test mode is intentionally single-use and hash guarded. The current test
result must not be removed or rerun for tuning.

Important scientific status: the model is genuinely generative, but its
held-out condition-permutation control failed. See the final report before
using the checkpoint.
