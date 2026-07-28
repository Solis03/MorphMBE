# M12 reproducibility runbook

## Environment

- repository: `/Users/ziyi/Desktop/LAB/code`
- Python: project-local `.venv`
- platform: Apple Silicon (`arm64`)
- PyTorch: 2.12.0, MPS available, CUDA unavailable
- canonical config:
  `configs/rheed_to_afm_functional_morphology_m12.json`
- removal-list SHA-256:
  `8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b`

## Commands

```bash
cd /Users/ziyi/Desktop/LAB/code

.venv/bin/python -m analysis.rheed_to_afm_functional_morphology.run \
  --config configs/rheed_to_afm_functional_morphology_m12.json --smoke

.venv/bin/python -m analysis.rheed_to_afm_functional_morphology.run \
  --config configs/rheed_to_afm_functional_morphology_m12.json

.venv/bin/python -m analysis.rheed_to_afm_functional_morphology.visualization \
  --config configs/rheed_to_afm_functional_morphology_m12.json

PYTHONPATH=. .venv/bin/pytest -q tests/test_rheed_to_afm_*.py
```

## Expected outputs

- maps:
  `outputs/rheed_to_afm_functional_morphology/20260727_m12_range_terrace_v1`
- metrics and manifest:
  `reports/rheed_to_afm_functional_morphology/20260727_m12_range_terrace_v1/development`
- paper figures:
  `reports/rheed_to_afm_functional_morphology/20260727_m12_range_terrace_v1/development/figures`
- logs:
  `reports/rheed_to_afm_functional_morphology/20260727_m12_range_terrace_v1/logs`

## Integrity checks

1. `best_model_manifest.json` must report
   `historical_test_used=false`,
   `retrieval_at_inference=false`, and
   `measured_afm_patch_used_at_inference=false`.
2. `test_rows_present_but_unselected` must remain 24.
3. Every `outer_target_used_for_training` value in
   `rq_crossfit_predictions.csv` and `fsmi_crossfit_predictions.csv` must be
   false.
4. The current `removelist.txt` hash must match the manifest.
5. `git diff -- data` must be empty.
6. Figure manifest must report 10 PNG and 10 PDF files.

## Runtime and compute handoff

The smoke run is about 30 seconds and the full run about 75 seconds locally.
The next locally meaningful M12 rerun is far below 30 minutes. A CUDA handoff
is therefore not recommended for this milestone.
