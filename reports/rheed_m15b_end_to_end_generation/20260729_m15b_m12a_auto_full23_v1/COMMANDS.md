# Reproduction commands

Run from `/Users/ziyi/Desktop/LAB/code` with the project environment.

## Three-fold smoke

```bash
PYTHONPATH=. .venv/bin/python \
  -m analysis.rheed_to_afm_full_cohort_loo.run \
  --config configs/rheed_m15b_end_to_end_generation.json \
  --device mps --smoke
```

## Strict 23-fold end-to-end generation

```bash
PYTHONPATH=. .venv/bin/python \
  -m analysis.rheed_to_afm_full_cohort_loo.run \
  --config configs/rheed_m15b_end_to_end_generation.json \
  --device mps
```

## Publication figures and integrity report

```bash
PYTHONPATH=. .venv/bin/python \
  -m analysis.rheed_to_afm_full_cohort_loo.visualization \
  --config configs/rheed_m15b_end_to_end_generation.json

PYTHONPATH=. .venv/bin/python \
  -m analysis.rheed_m15b_end_to_end_generation.report \
  --config configs/rheed_m15b_end_to_end_generation.json
```

## Actual UI-path raw-video verification

```bash
PYTHONPATH=src:. .venv/bin/python \
  scripts/smoke_rheed_realtime_pipeline.py \
  "data/raw/raw_RHEED/N6056 - Copy/After rampdown to 200 C.MOV" \
  --sample-id 6056 \
  --output-dir \
  outputs/rheed_realtime_ui/20260729_m15b_m12a_end_to_end_ui_verification_6056

QT_QPA_PLATFORM=offscreen PYTHONPATH=src:. .venv/bin/python \
  scripts/capture_rheed_realtime_ui.py \
  --sample-id 6056 \
  --video-contains "After rampdown to 200 C" \
  --output \
  outputs/rheed_realtime_ui/20260729_m15b_m12a_end_to_end_ui_verification_6056/ui_offscreen.png \
  --timeout-seconds 120
```

## Regression tests

```bash
PYTHONPATH=src:. .venv/bin/pytest -q \
  tests/test_rheed_auto_input_robustness.py \
  tests/test_rheed_realtime_ui.py \
  tests/test_automatic_rheed_roi_keyframe.py \
  tests/test_rheed_manual_vs_auto_selection.py \
  tests/test_rheed_to_afm_ood_robust.py \
  tests/test_rheed_to_afm_full_cohort_loo.py \
  tests/test_rheed_to_afm_distinct_confidence.py \
  tests/test_rheed_to_afm_functional_morphology.py \
  tests/test_rheed_to_afm_island_generation.py \
  tests/test_rheed_to_afm_sharp_generation.py
```
