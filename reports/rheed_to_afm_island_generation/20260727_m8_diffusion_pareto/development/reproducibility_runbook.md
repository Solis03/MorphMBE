# M8 island-diffusion reproducibility runbook

Run from `/Users/ziyi/Desktop/LAB/code` with the repository `.venv`:

```bash
PYTHONPATH=. uv run pytest -q \
  tests/test_rheed_to_afm_island_diffusion.py \
  tests/test_rheed_to_afm_island_generation.py
```

## M6 object generators

```bash
PYTHONPATH=. uv run python \
  -m analysis.rheed_to_afm_island_generation.run \
  --config configs/rheed_to_afm_island_generation_v2.json
```

This writes M5, M6a, M6b, and M6c results without opening the historical test
cohort.

## M7 AFM-prior model and strength ablation

```bash
PYTHONPATH=. uv run python \
  -m analysis.rheed_to_afm_island_generation.train_guided_diffusion \
  --config configs/rheed_to_afm_island_generation_v2.json \
  --output outputs/rheed_to_afm_island_generation/20260727_m7_guided_diffusion_v1 \
  --steps 1600 --evaluate-every 50 --batch-size 16 \
  --base-channels 24 --embedding-dim 96
```

For each strength in `0.25 0.45 0.70 1.00`:

```bash
PYTHONPATH=. uv run python \
  -m analysis.rheed_to_afm_island_generation.sample_guided_diffusion \
  --checkpoint outputs/rheed_to_afm_island_generation/20260727_m7_guided_diffusion_v1/best_guided_diffusion.pt \
  --guide-root outputs/rheed_to_afm_island_generation/20260727_m6_island_v2/development/generated_maps \
  --output outputs/rheed_to_afm_island_generation/20260727_m7_guided_diffusion_v1/validation_strength_STRENGTH \
  --sampling-steps 48 --draws 3 --strength STRENGTH
```

The selected weak-refinement strength is `0.25`. Full-noise strength `1.0`
is intentionally preserved as a failed small-data ablation.

## Strict 15-growth LOO

```bash
PYTHONPATH=. uv run python \
  -m analysis.rheed_to_afm_island_generation.run_guided_diffusion_crossfit \
  --guide-root outputs/rheed_to_afm_island_generation/20260727_m6_island_v2/development/crossfit/generated_maps \
  --output outputs/rheed_to_afm_island_generation/20260727_m8_diffusion_pareto/development/crossfit \
  --report reports/rheed_to_afm_island_generation/20260727_m8_diffusion_pareto/development/crossfit \
  --steps 900 --strength 0.25 --blend-weight 0.50
```

Each held growth group gets a separate AFM residual-prior checkpoint trained
on the other 14 groups. The held AFM is used only after generation for
scoring.

## Frozen all-training AFM prior and validation generation

```bash
PYTHONPATH=. uv run python \
  -m analysis.rheed_to_afm_island_generation.train_guided_diffusion \
  --config configs/rheed_to_afm_island_generation_v2.json \
  --output outputs/rheed_to_afm_island_generation/20260727_m8_diffusion_pareto/development/final_afm_prior \
  --steps 900 --evaluate-every 50 --batch-size 16 \
  --base-channels 24 --embedding-dim 96 --fit-all
```

```bash
PYTHONPATH=. uv run python \
  -m analysis.rheed_to_afm_island_generation.sample_guided_diffusion \
  --checkpoint outputs/rheed_to_afm_island_generation/20260727_m8_diffusion_pareto/development/final_afm_prior/best_guided_diffusion.pt \
  --guide-root outputs/rheed_to_afm_island_generation/20260727_m6_island_v2/development/generated_maps \
  --output outputs/rheed_to_afm_island_generation/20260727_m8_diffusion_pareto/development/validation \
  --sampling-steps 48 --draws 6 --strength 0.25 \
  --reference-blend-weights 0.50
```

## Figures

```bash
PYTHONPATH=. uv run python \
  -m analysis.rheed_to_afm_island_generation.visualization \
  --cross-report reports/rheed_to_afm_island_generation/20260727_m8_diffusion_pareto/development/crossfit \
  --cross-root outputs/rheed_to_afm_island_generation/20260727_m8_diffusion_pareto/development/crossfit \
  --validation-root outputs/rheed_to_afm_island_generation/20260727_m8_diffusion_pareto/development/validation \
  --m6-validation-root outputs/rheed_to_afm_island_generation/20260727_m6_island_v2/development/generated_maps \
  --m6-report reports/rheed_to_afm_island_generation/20260727_m6_island_v2/development \
  --diffusion-root outputs/rheed_to_afm_island_generation/20260727_m7_guided_diffusion_v1 \
  --parent-report reports/rheed_to_afm_distinct_confidence/20260727_m5_hybrid_v4_confidence/development \
  --output reports/rheed_to_afm_island_generation/20260727_m8_diffusion_pareto/development/figures
```

## Safety checks

```bash
git diff -- data
rg -n "historical_test_used|retrieval_at_inference|measured_afm_patch_used_at_inference" \
  reports/rheed_to_afm_island_generation/20260727_m8_diffusion_pareto/development \
  outputs/rheed_to_afm_island_generation/20260727_m8_diffusion_pareto/development
```

The canonical exclusions are loaded through `_load_tables`, using
`removelist.txt` with SHA-256
`8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b`.
