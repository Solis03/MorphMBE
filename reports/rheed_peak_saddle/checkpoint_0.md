# Checkpoint 0: Peak-Saddle Stage Audit

## Canonical Removelist

- Path: `/home/wangziyi/MorphMBE/MorphMBE/removelist.txt`
- SHA256: `8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b`
- Sample `6088` excluded: `1`

## Candidate Dataset

- Candidate included sample count: `25`
- Missing manual images among non-removelist samples: 6027, 6054, 6055, 6058, 6065, 6093, 6100, 6102
- Missing AFM data among non-removelist manual selections: none

## Provisional Growth Stages

- `oxide_or_substrate`: 0
- `active_growth`: 7
- `after_growth`: 2
- `rampdown_or_cooldown`: 15
- `rampup_or_heating`: 0
- `unknown`: 1

## Exact User Action Required

1. Open `annotations/rheed_peak_saddle/stage_review_template.csv`.
2. Review each filename-derived `inferred_stage`.
3. Fill `approved_stage`, `comparable_stage_group`, `user_approved`, and `user_notes`.
4. Save the completed file as `annotations/rheed_peak_saddle/stage_review_completed.csv`.
5. Ensure every included row has `user_approved = 1` before requesting Stage 1.

STOP: Stage 1 synthetic validation has not been run.