# Checkpoint 2A: Real RHEED Shadow Diagnostics

## Recovered State
- Stage 1C-R status: `STAGE 1C PASS AFTER METRIC-LINEAGE CORRECTION`
- Frozen semantic spec hash: `ffa417f8c5a67f8a3ede3e532464b3a82a783c47a3362dc4abb85cc4f8ed0689`
- Evaluation receipt SHA256: `a3619bad7a8517d083c4fd73852a6666c235bf21a2f46c5a8ce02f0869541e9f`

## Gates
- Visual approval path: `/home/wangziyi/MorphMBE/MorphMBE/annotations/rheed_peak_saddle/approvals/checkpoint_1c_visual_review_template.txt`
- Visual approval SHA256: `82c8f8af98b335ba9df3b77d297742d4fbf789e4b0632315dd21be7e02a5d9a3`
- Visual approval mtime: `2026-07-13T20:03:44.101941+00:00`
- Removelist path: `/home/wangziyi/MorphMBE/MorphMBE/removelist.txt`
- Removelist SHA256: `8fe844f8c8c9ab6e457b8b9ebbd4e80284b784f80bbfd9602a315c9a5cd7fe3b`
- Sample `6088` excluded: `1`
- Stage-review path: `/home/wangziyi/MorphMBE/MorphMBE/annotations/rheed_peak_saddle/stage_review_completed.csv`
- Stage-review SHA256: `862df0397683a19c24d616b2ba42b088538048750a63a89eb17593c1b4c9081e`

## Real Inputs
- Eligible real-image count: `25`
- Skipped images: `0`
- Skipped details: `none`
- No AFM/Rq source was opened: `1`

## Split
- Split seed: `2026071302`
- Split receipt SHA256: `04eb4931e262997b454555c8ab3257ff2c0756c5d3670e3a9b00801863b9f5cd`
- Development/blind/reserve counts: `{'blind_validation': 10, 'development_review': 10, 'reserve': 5}`
- Approved-stage distribution by split: `{'blind_validation': {'after_growth': 1, 'rampdown_or_cooldown': 6, 'active_growth': 2, 'rampup_or_heating': 1}, 'development_review': {'rampdown_or_cooldown': 7, 'active_growth': 3}, 'reserve': {'active_growth': 2, 'rampdown_or_cooldown': 3}}`

## Algorithm Shadow-Mode Summary
- Spot count median: `12`
- Row count median: `8`
- Valid-pair count median: `2`
- Invalid-pair count median: `1`
- Measurement-quality labels: `{'medium': 11, 'high': 14}`

## Files Requiring Human Completion
- `annotations/rheed_peak_saddle/real_review/all_sample_qc_template.csv`
- `annotations/rheed_peak_saddle/real_review/development_sample_review_template.csv`
- `annotations/rheed_peak_saddle/real_review/development_pair_review_template.csv`
- Template hashes: `{'all_sample_qc_template.csv': 'f16ec2c45037515e086112a6badcb4b7209569fdc14905554581a689f0c47797', 'development_sample_review_template.csv': '1aaedb7ca676237b00aa647509b60b6f322f254b6a52bd90624596f38cc1907e', 'development_pair_review_template.csv': 'f93d47343a3604c06730087e874dd59a4df96f5101f780e126ec292cd2cfd43b', 'instructions.md': '20c7ca19281229579cf412d0013a6adc98d1db8a399887cdc0aaa60d950eae77', 'unblind_key.csv': '484db8ea958620c6519eb59cba72426f26f338bd635c166f12ff50a901913cd0'}`

## Human Instructions
Judge visible spot-to-spot grayscale adhesion only. Do not consult AFM images, Rq values, filenames, stage, or prior sample identity memory. Mark unusable rather than guessing.

## Status
HUMAN ANNOTATION REQUIRED

## Stop Confirmation
- No annotation validation was run.
- No real-image tuning was run.
- No AFM data were accessed.
- No Rq values were accessed.
- No model training was run.
- No Stage 2B/3 was run.
