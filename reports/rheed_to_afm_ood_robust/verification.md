# M14 verification record

Date: 2026-07-28

## Passed checks

- `compileall` passed for `analysis/rheed_to_afm_ood_robust` and
  `analysis/rheed_to_afm_full_cohort_loo`.
- Focused modified-package tests: 7/7 passed.
- All current RHEED-to-AFM tests: 30/30 passed.
- Outer generator fold audit: 23/23 folds contain 22 fit growths and no held
  growth overlap.
- Final target table: 46/46 rows (23 Rq and 23 FSMI) report
  `outer_target_used_for_training = False`.
- Cohort manifest: 23 unique groups; 6043 and 6055 are absent.
- Canonical `removelist.txt` SHA-256:
  `8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b`,
  matching the final experiment manifest.
- `git diff -- data Data raw_data removelist.txt` is empty.
- Final figure integrity: 17/17 PNG files meet the minimum dimension check;
  17/17 PDFs pass Poppler `pdfinfo` and contain one page.
- The final method-ablation PDF, confidence/risk-coverage PDF and high-Rq
  atlas PDF were rendered with Poppler and visually inspected. Labels,
  physical units, scale bars, color bars and titles are readable.
- `git diff --check` passed.

## Full-suite limitation outside this task

The repository-wide command

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests --import-mode=importlib
```

reached 134 passing tests and then stopped at
`tests/test_rheed_peak_saddle.py::PeakSaddleStage0Test::test_completed_stage_review_file_not_modified_by_validation`.
That separate workflow requires the ignored human-checkpoint artifact
`outputs/rheed_peak_saddle/preliminary_manifest.csv`, which is absent. No
placeholder was fabricated and no `rheed_peak_saddle` file was changed.

Running plain `pytest -q` also collects duplicated test modules inside the
immutable `paper_freeze/.../17_CODE_SNAPSHOT/tests` tree and raises module
name mismatch errors. Explicitly limiting collection to `tests/` with
`--import-mode=importlib` avoids that unrelated snapshot-collection problem.
