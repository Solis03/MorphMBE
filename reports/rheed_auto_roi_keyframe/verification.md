# Verification record

Date: 2026-07-28

## Passed checks

- `PYTHONPATH=src .venv/bin/python -m pytest
  tests/test_automatic_rheed_roi_keyframe.py
  tests/test_manual_rheed_roi_reviewer.py
  tests/test_rheed_frame_selection.py
  tests/test_rheed_frame_selection_v2.py
  tests/test_rheed_keyframe_selection.py
  tests/test_rheed_single_frame_manual.py -q`
  - 48 passed.
- Python compile check passed for the automatic selector, CLI, experiment
  scripts and tests.
- 155 generated PNG files opened and passed Pillow verification.
- 44 generated PDF files have valid PDF headers, EOF markers and nontrivial
  size. Primary V2 benchmark, confidence and atlas PNGs were also visually
  inspected.
- V2 row-integrity checks passed:
  - 81 ROI predictions = 27 videos x 3 ROI methods;
  - 162 heuristic keyframe predictions = 27 videos x 6 methods;
  - 27 strict leave-one-video-out gradient-boosting predictions;
  - zero held-video overlap;
  - 724 labelled physical vertex candidates in the final training bundle.
- The fitted `gradient_boosting_phase_ranker.joblib` loads successfully and
  records 27 training videos.
- Every V1/V2/V3 run audit reports unchanged raw source video size and
  modification time.
- Direct source-video inference on
  `data/pair/6022/RHEED/After GaSb.MP4` completed successfully in 12.9 seconds
  for 810 frames and wrote JSON, overlay and ROI crop outputs.
- `git diff -- data removelist.txt` is empty.

## Repository-wide test boundary

`PYTHONPATH=src .venv/bin/python -m pytest tests -q` produced:

- 334 passed;
- 24 failed;
- 6 errors.

The failures are outside this task:

- the latest paper-freeze directory is missing ignored
  `FREEZE_MANIFEST.json` and related runtime artifacts;
- the unrelated `rheed_peak_saddle` workflow lacks required human-checkpoint
  and ignored output files;
- two historical phase-4 tests require `pyarrow` or `fastparquet`, neither of
  which is installed in the existing environment.

No placeholder checkpoint, freeze manifest or raw/derived scientific record
was fabricated to make those tests pass.

Running plain `pytest` from the repository root also collects duplicated test
module names inside a historical code snapshot and stops on import-file
mismatch. Scoping to `tests/` avoids that unrelated discovery problem.

## Independent review notes

- The final supervised metrics use strict video-level leave-one-out; candidate
  labels from the held video never enter its fold model.
- ROI and keyframe development used the same 27-video cohort retrospectively.
  The final all-data model is correctly described as awaiting prospective
  validation.
- Repeated rotation cycles mean absolute frame difference is not treated as
  the primary scientific endpoint.
- Failure cases 6023 and 6056 are included in both tables and figures.
- Confidence is described as expected human-frame similarity, not a
  correctness probability.
- V1 and rejected V3 artifacts remain separate and were not overwritten.
