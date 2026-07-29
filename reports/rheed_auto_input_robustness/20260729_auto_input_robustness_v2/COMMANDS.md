# Reproduction commands

Run from `/Users/ziyi/Desktop/LAB/code` with the repository `.venv`.

## Strict 23-fold scalar/confidence evaluation

The perturbation embedding cache and Q50 physics ROI files are preserved in
the output artifact directory. Recompute the strict scalar/confidence tables
and all publication figures with:

```bash
PYTHONPATH=src:. .venv/bin/python \
  -m analysis.rheed_auto_input_robustness.run \
  --config configs/rheed_auto_input_robustness.json
```

## Refit the all-23 UI deployment bundle

```bash
PYTHONPATH=src:. .venv/bin/python \
  scripts/prepare_rheed_realtime_model.py --force
```

This is a deployment fit, not held-out evidence.

## Raw-video 6056 smoke

```bash
PYTHONPATH=src:. .venv/bin/python \
  scripts/smoke_rheed_realtime_pipeline.py \
  "data/raw/raw_RHEED/N6056 - Copy/After rampdown to 200 C.MOV" \
  --sample-id 6056 \
  --output-dir \
  outputs/rheed_realtime_ui/20260729_6056_m15b_auto_robustness_v3
```

## Offscreen UI verification

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=src:. .venv/bin/python \
  scripts/capture_rheed_realtime_ui.py \
  --sample-id 6056 \
  --video-contains "After rampdown to 200 C" \
  --output \
  outputs/rheed_realtime_ui/20260729_6056_m15b_auto_robustness_v3/ui_offscreen.png \
  --timeout-seconds 120
```

## Tests

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

Expected result for this freeze: `47 passed`.
