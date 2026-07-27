# Sensitivity experiment excluding N6022 and N6099

This is an independent rerun of
`publication_freeze/prospective_retrained_6358_6382_single_frame_v1`.
The only experimental change is the global removal of samples N6022 and N6099
from quantitative fitting and AFM retrieval banks. Feature definitions,
`StandardScaler`, the five frozen `Ridge(alpha=1.0)` member definitions, median
aggregation, AFM preprocessing, and A3 retrieval ranking are unchanged.

## Data division

- Reduced historical baseline: 21 retained historical samples.
- Prospective retraining: the retained historical 21 plus N6358 and N6382
  (23 training samples).
- Prospective predictions: N6342, N6389, and N6390.
- Strict leave-one-out analysis: 26 retained labeled samples, with 25 training
  samples per fold.
- Held-one-out AFM analysis: 26 targets, with 25 quantitative training samples
  and 25 AFM source groups per fold.
- N6022 and N6099 are excluded globally. N6324 remains ignored.

All five extra-sample AFM sets are 2 µm × 2 µm scans. The model-facing AFM is
the upper-left 256 × 256 quarter (1 µm × 1 µm), cropped from the physical
ZSensor array before the unchanged robust second-order `y2` correction. Every
displayed ground-truth and retrieved AFM retains a height bar in nm and its Rq.

N6390 uses `N6390_2um_1.0_00000.spm` and scans 2–5. The duplicate-number
candidate `N6390_2um_1.0_00001.spm` is retained in the raw data but excluded
from this five-map series.

## Reproduction

Run from the repository root:

```bash
uv run python publication_freeze/prospective_retrained_6358_6382_exclude_6022_6099_single_frame_v1/code/run_experiment.py
uv run python publication_freeze/prospective_retrained_6358_6382_exclude_6022_6099_single_frame_v1/code/run_leave_one_out.py
uv run python publication_freeze/prospective_retrained_6358_6382_exclude_6022_6099_single_frame_v1/code/run_held_one_out_afm.py
uv run python publication_freeze/prospective_retrained_6358_6382_exclude_6022_6099_single_frame_v1/code/compare_exclusion_impact.py
uv run python publication_freeze/prospective_retrained_6358_6382_exclude_6022_6099_single_frame_v1/code/validate_experiment.py
```

The frozen original 23-sample model is also reproduced exactly as an algorithm
audit before any reduced-cohort fit is made. It is not used as the reduced
baseline.

## Main outputs

- `figures/main/Figure1_three_sample_prediction_atlas.*`
- `figures/main/Figure2_leave_one_out_prediction_scatter.*`
- `figures/main/Figure3_held_one_out_afm_prediction_atlas.*`
- `figures/comparison/Figure4_exclusion_impact_summary.*`
- `figures/comparison/Figure5_leave_one_out_common26_before_after.*`
- `figures/supplementary/` (the complete matched supplementary figure set)
- `figures/per_sample/` and `figures/held_one_out_afm/per_sample/`
- `predictions/reduced_21_baseline/`
- `predictions/retrained_23/`
- `predictions/leave_one_out_26/`
- `predictions/held_one_out_afm_26/`
- `comparison/` (machine-readable before/after tables)
- `report/result_summary.md`
- `report/exclusion_impact_summary.md`
- `provenance/validation_report.json`

The comparison report recomputes the original experiment on the identical
retained 26-sample cohort. This common-cohort comparison separates genuine
model changes from the arithmetic effect of deleting two high-error rows.

## Exclusion impact comparison

This sensitivity experiment removes samples 6022 and 6099 from every fit and AFM bank without changing algorithms. The complete before/after analysis is `report/exclusion_impact_summary.md`, with comparison figures in `figures/comparison/`.
