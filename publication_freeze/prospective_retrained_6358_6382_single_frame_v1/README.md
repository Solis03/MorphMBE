# Prospective retraining with N6358 and N6382

This experiment is a separate extension of
`publication_freeze/prospective_unseen_single_frame_v1`. It does not alter the
algorithm source or the frozen retrospective package.

The quantitative training cohort is the original 23 frozen samples plus N6358
and N6382. Predictions are made only for N6342, N6389, and N6390. N6324 is
explicitly ignored.

All five extra-sample AFM sets were acquired as 2 µm × 2 µm, 512 × 512 scans.
The model-facing ground truth is the upper-left 256 × 256 quarter, corresponding
to 1 µm × 1 µm. The quarter is cropped from the physical ZSensor height array
before the unchanged robust second-order `y2` correction is applied. Rq is
always calculated from the physical corrected array, never from a rendered
image.

N6390 contains two files with scan number 1. To retain the same five-scan
structure as the other samples, `N6390_2um_1.0_00000.spm` and scans 2–5 are
used. `N6390_2um_1.0_00001.spm` remains untouched in the raw folder and is
listed in the audit as an excluded duplicate-number candidate.

The frozen 23-sample model is first rerun for N6342, N6358, N6382, N6389, and
N6390. The separate retrained model then uses the exact deployment fit:
`StandardScaler` followed by `Ridge(alpha=1.0)` for each of the five frozen
ensemble member definitions and median member aggregation. A 23-sample
reproduction check is saved before adding N6358 and N6382.

Run the complete experiment from the repository root:

```bash
uv run python publication_freeze/prospective_retrained_6358_6382_single_frame_v1/code/run_experiment.py
uv run python publication_freeze/prospective_retrained_6358_6382_single_frame_v1/code/run_leave_one_out.py
uv run python publication_freeze/prospective_retrained_6358_6382_single_frame_v1/code/validate_experiment.py
```

The post-hoc leave-one-out extension uses all 28 labeled samples: the historical
23 plus all five extra samples. It refits the same `StandardScaler` plus
five-member `Ridge(alpha=1.0)` ensemble 28 times. In each fold, the target
sample is absent from both scaler fitting and model fitting, and the remaining
27 samples are used. Its raw ensemble-median prediction is compared with the
held-out sample's T4 Rq target. This is distinct from both the 25-sample
in-sample fit and the original three-sample prospective test; in the post-hoc
LOO analysis, the three revealed prospective labels may train other folds but
never their own folds.

Primary outputs:

- `figures/main/Figure1_three_sample_prediction_atlas.*`
- `figures/main/Figure2_leave_one_out_prediction_scatter.*`
- `figures/supplementary/SuppFigure10_leave_one_out_diagnostics.*`
- `predictions/retrained_25/predictions.csv`
- `predictions/leave_one_out_28/predictions.csv`
- `predictions/leave_one_out_28/metrics.json`
- `report/leave_one_out_summary.md`
- `predictions/retrained_25/retrieval/retrieval_results.csv`
- `evaluation/per_sample_evaluation.csv`
- `models/quantitative_model/`
- `ground_truth_afm/top_left_quarter_second_order/`
- `provenance/algorithm_code_audit.json`
- `report/result_summary.md`

Every displayed AFM ground-truth or retrieved map includes a height bar in nm
and its Rq value. The full-cohort retraining fit and its three-sample evaluation
must not be presented as a replacement for the frozen 23-sample strict OOF
benchmark.
