# Benchmark Phase 0 Report

Branch: `experiment/natcomms-computational-benchmark-v1-20260723-110844`
Baseline: `natcomms-publication-baseline-v0-20260723-000924` -> `0a2414a06c4155123dc61cd8a95ded638fb725dc`

## Cohorts

- Master registry: 29 rows
- Paired supervised cohort: 27 rows
- Historical development: 23 rows
- Prospective pilot seen: 4 rows
- Prospective pending truth: 1 row (`N6390`)
- AFM-only unmatched: 1 row (`N6324`)

Historical sample IDs: `6022, 6028, 6029, 6033, 6047, 6048, 6056, 6057, 6062, 6063, 6070, 6072, 6078, 6080, 6081, 6082, 6084, 6085, 6090, 6094, 6095, 6099, 6101`
Prospective pilot IDs: `N6342, N6358, N6382, N6389`

## Hashes

- Protocol: `143ef5de1b7e3cc553a3d925b32a75e7875c8cccfbac25a202d59f841a89c945`
- Master registry: `21cb4e7bdb5f0a57e4cfb66956d2063cd703f35c44638453ebdfa5ac47653256`
- Split: `705f62f6eec27506f596b6dbe305f47bb6f8b500c9a0a91f29b2f64cd286d13f`
- Environment: `83a9d88eb9809662c05ed19935e16b230adfe0fa9ee16fe19701ea861c0f2bcb`
- Lock: `c36cd447e2b122f105f69d57380eea037436f567fb2a4debe85403628dae875e`

## Validation

- Registry/protocol/split/schema guards: passed
- `uv run pytest -q`: 25 Phase 0 benchmark tests passed
- `uv run python -m compileall src scripts tools`: passed
- Frozen retrospective verifier: passed on a temporary copy to avoid modifying immutable files
- Prospective validator: passed on a temporary copy to avoid modifying immutable files
- Immutable publication hash comparison: zero differences
- Dry-run: passed with run ID `run_d4c1159ed570f1f21f239e2e`; no tensors, DINO, scaler, model, predictions, or checkpoint
- `git fsck --full`: exit 0; dangling unreachable objects reported
- `git lfs fsck`: Git LFS fsck OK

## Target Compatibility

Historical primary target is `T4_second_order_trimmed_mean` in nm. The available
prospective pilot truth is `true_rq_nm_median_second_order`, so it is retained as
pilot/exploratory and marked not directly comparable to the historical primary
T4 aggregation.

## Metadata And Control Variables

Post-growth outcome fields with leakage risk are present in AFM truth, target,
prediction, and error tables and are all disallowed as predictive metadata.
No structured machine-readable software-controllable growth variables were
found. Ambiguous filename tokens for ramp-down temperature, process duration,
and material stage require expert curation before any future controller work.

## Remaining Before B00

No registry, split, or dry-run blocker remains for historical B00. Phase 0 does
not execute B00. Prospective pilot aggregate interpretation remains blocked by
seen-truth status and the median-vs-T4 target mismatch.
