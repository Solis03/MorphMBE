# M10 dense-island generation reproducibility runbook

Run from `/Users/ziyi/Desktop/LAB/code` in the existing repository
environment. The commands never open the consumed historical test cohort.

## Environment and focused tests

```bash
PYTHONPATH=. uv run python -c \
  "import platform, torch; print(platform.platform()); print(torch.__version__); print(torch.backends.mps.is_available()); print(torch.cuda.is_available())"

PYTHONPATH=. uv run pytest -q \
  tests/test_rheed_to_afm_island_generation.py \
  tests/test_rheed_to_afm_island_diffusion.py \
  tests/test_rheed_to_afm_distinct_confidence.py
```

## Dense island candidates

```bash
PYTHONPATH=. uv run python \
  -m analysis.rheed_to_afm_island_generation.run \
  --config configs/rheed_to_afm_island_generation_v3_dense.json
```

This produces M5, M6a, M6b and M6c on the three pre-existing validation
growths and in strict leave-one-training-growth-out folds. Island seed
populations, regression scalers and AFM support models are fitted inside each
fold.

## Structure-weight and topology ablations

```bash
PYTHONPATH=. uv run python \
  -m analysis.rheed_to_afm_island_generation.evaluate_topology_renderers \
  --config configs/rheed_to_afm_island_generation_v3_dense.json \
  --source outputs/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development/crossfit \
  --output reports/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development/structure_weight_ablation \
  --edge-gains 0.5 \
  --terrace-levels 7 \
  --structure-weights 0.55 0.65 0.75 0.85
```

The preselected Pareto choice is 0.65 island structure and 0.35 stochastic
spectral texture. Higher island weights visibly sharpen objects but degrade
texture and island-distribution metrics.

## Freeze selected development model

```bash
PYTHONPATH=. uv run python \
  -m analysis.rheed_to_afm_island_generation.finalize_dense_island_model \
  --config configs/rheed_to_afm_island_generation_v3_dense.json \
  --cross-source outputs/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development/crossfit \
  --validation-source outputs/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development \
  --output outputs/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development/selected \
  --report reports/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development/selected \
  --weight 0.65

PYTHONPATH=. uv run python \
  -m analysis.rheed_to_afm_island_generation.morphology_confidence \
  --selected-report reports/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development/selected \
  --output reports/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development/selected/confidence
```

## Figures

```bash
PYTHONPATH=. uv run python \
  -m analysis.rheed_to_afm_island_generation.visualization \
  --config configs/rheed_to_afm_island_generation_v3_dense.json \
  --cross-report reports/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development/selected/crossfit \
  --cross-root outputs/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development/selected/crossfit \
  --validation-root outputs/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development/selected/validation \
  --validation-report reports/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development/validation \
  --m6-validation-root outputs/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development/generated_maps \
  --m6-report reports/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development \
  --diffusion-root outputs/rheed_to_afm_island_generation/20260727_m7_guided_diffusion_v1 \
  --parent-report reports/rheed_to_afm_distinct_confidence/20260727_m5_hybrid_v4_confidence/development \
  --morphology-confidence-report reports/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development/selected/confidence \
  --output reports/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development/selected/figures
```

## Safety audit

```bash
shasum -a 256 removelist.txt
git diff -- data
rg -n \
  "historical_test_used|retrieval_at_inference|measured_afm_patch_used_at_inference" \
  reports/rheed_to_afm_island_generation/20260727_m10_dense_islands_v3/development/selected
```

Expected removal-list SHA-256:
`8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b`.
Generation uses no measured AFM or retrieved AFM patch at inference.
