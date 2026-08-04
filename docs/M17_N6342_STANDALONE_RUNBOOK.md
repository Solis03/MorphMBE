# M17b N6342 sparse-peak standalone runbook

## Active model

The active deployment is **M16 scalar prediction + M17b topology-conditioned
sparse-peak terrace generation**. It uses automatic ROI/keyframe selection,
causal R3D-18 temporal features, the endpoint-aware Sq head, the FSMI head,
strict-LOO error calibration and a non-retrieval AFM generator. The AFM target
is the sample median areal Sq after third-order independent polynomial
flattening of every fast-scan line.

M17b replaces the dense bright-local-maximum field used by M16b in the smooth
surface regime. The number of visually persistent peaks is conditioned on the
RHEED-predicted island topology. The rough-surface branch is unchanged.

Growth 6081 is operator-invalid. It is listed in `removelist.txt`, absent from
the 27-growth deployment cohort and must not be used or displayed.

## Launch

From the archive root:

```bash
./scripts/run_m17_standalone.sh run-ui
```

Every accepted clear rotational moment is submitted once. Each completed
event adds one Sq/FSMI/confidence point and updates the generated AFM image.

## Validation

```bash
./scripts/run_m17_standalone.sh validate
./scripts/run_m17_standalone.sh verify-checksums
./scripts/run_m17_standalone.sh test
./scripts/run_m17_standalone.sh smoke-model-6342
```

The smoke command reads the archived N6342 video and writes only to
`reproduced_outputs/`; frozen reports and data remain unchanged.

## Full experiment reproduction

```bash
./scripts/run_m17_standalone.sh reproduce-m17
```

This is the complete retrospective leave-one-growth-out run. Each of 27 folds
fits the other 26 growths. N6342 motivated method selection, so its result is
retrospective method-development evidence, not a prospective untouched test.

## Important paths

- `configs/rheed_m17_end_to_end_generation_line3_full27_sparse_v1.json`:
  complete experiment configuration and all renderer ablations.
- `configs/rheed_realtime_ui_m17_full27_line3_exclude6081_v9.json`: active UI.
- `outputs/rheed_realtime_ui/morphmbe_m16_m17b_line3_full27_exclude6081_live_v9.joblib`:
  active deployment bundle.
- `reports/rheed_n6342_sparse_island/REPORT.md`: scientific report.
- `reports/rheed_n6342_sparse_island/literature_review.md`: method literature.
- `reports/rheed_m17_end_to_end_generation/20260804_m17_sparse_topology_line3_full27_v1/full27_loo/figures/`:
  complete publication figures.

Older M12a/M14i/M15b/M16b files are retained only because they provide frozen
parameters, dependencies, and historical provenance. They are not the active
image generator.
