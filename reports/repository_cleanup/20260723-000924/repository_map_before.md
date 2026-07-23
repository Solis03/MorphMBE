# Repository Map Before Cleanup (20260723-000924)

- Branch: `cleanup/natcomms-publication-baseline-2026-07-22-20260723-000924`
- HEAD: `c6d748141f011b8b72abe2c3b1f6e125dcb6e62e`
- Safety backup: `/Users/ziyi/Desktop/LAB/code_cleanup_backups/20260723-000924`

## Top-Level Directories

| directory | files | size_bytes | cleanup stance |
|---|---:|---:|---|
| `.pytest_cache/` | 5 | 32867 | cache/local env; .venv out of scope |
| `.venv/` | 36479 | 2036927952 | cache/local env; .venv out of scope |
| `__pycache__/` | 2 | 5799 | cache/local env; .venv out of scope |
| `analysis/` | 265 | 4009237 | KEEP_ACTIVE_CODE / KEEP_REUSABLE_BASELINE_CODE |
| `annotations/` | 16 | 35518 | KEEP_HUMAN_ANNOTATION |
| `checkpoints/` | 0 | 0 | UNKNOWN_KEEP; no broad deletion |
| `configs/` | 12 | 26657 | KEEP_ACTIVE_CODE / KEEP_REUSABLE_BASELINE_CODE |
| `data/` | 92206 | 294283773199 | KEEP_RAW_SOURCE / derived data review only |
| `docs/` | 6 | 74256 | review conservatively |
| `notebooks/` | 1 | 145 | review conservatively |
| `outputs/` | 7896 | 2049479638 | UNKNOWN_KEEP; no broad deletion |
| `paper_freeze/` | 815 | 855456852 | UNKNOWN_KEEP pending comparison |
| `publication_freeze/` | 383 | 104877377 | KEEP_IMMUTABLE_CANONICAL for named freeze packages |
| `reports/` | 6538 | 1480183366 | mixed: compact record + generated artifact review |
| `scripts/` | 37 | 589874 | KEEP_ACTIVE_CODE / KEEP_REUSABLE_BASELINE_CODE |
| `src/` | 202 | 2990030 | KEEP_ACTIVE_CODE / KEEP_REUSABLE_BASELINE_CODE |
| `tests/` | 106 | 1310839 | KEEP_ACTIVE_CODE / KEEP_REUSABLE_BASELINE_CODE |
| `tools/` | 1 | 1770 | KEEP_ACTIVE_CODE / KEEP_REUSABLE_BASELINE_CODE |

## Canonical Current State

- Retrospective package: `publication_freeze/rheed_afm_single_frame_v1_2026-07-18`, 23 strict historical groups, frozen single-frame DINOv2 + top-five Ridge median ensemble, strict A3 retrieval.
- Prospective package: `publication_freeze/prospective_unseen_single_frame_v1`, five RHEED samples N6342/N6358/N6382/N6389/N6390.
- Prospective AFM truth: four matched prediction/truth samples N6342/N6358/N6382/N6389; N6390 prediction without AFM; N6324 AFM without prediction.
- Raw negative predictions for N6342, N6358, N6382 are retained in canonical predictions and join files.
