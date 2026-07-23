# Current Scientific Baseline

This baseline records the current frozen state. It is not a new analysis and does not change any result.

## Retrospective Model

The frozen retrospective method is a single-frame selected RHEED keyframe route using frozen DINOv2 embeddings and a top-five Ridge median ensemble to predict AFM Rq. The target is `T4_second_order_trimmed_mean`, with strict A3 historical AFM retrieval for the visual route.

Current retrospective cohort:

- N = 23 strict historical growth groups.
- MAE = 1.2600983407909774 nm.
- RMSE = 1.8393101333714263 nm.
- R2 = 0.2939669042578328.
- Spearman = 0.42885375494071143.
- Median absolute error = 1.1205674310532974 nm.

## Prospective State

The prospective unseen prediction package contains five RHEED samples: N6342, N6358, N6382, N6389, and N6390. Current AFM truth is matched for four prediction samples: N6342, N6358, N6382, and N6389.

N6390 is still present as prediction without AFM truth. N6324 is still present as AFM truth without prediction in the mismatch report.

## Known Limitations

- Raw prospective predictions for N6342, N6358, and N6382 are negative and must remain intact.
- Nonnegative clipped values are visual/rendering aids only and do not replace raw predictions.
- The current method has dynamic-range compression and does not claim robust prospective generalization.
- The frozen retrospective package is evidence for the selected single-frame method only. It does not claim that retired diffusion, VQ, quilting, residual, oracle, or mixed-best visual families are canonical.

## Current Claim Boundary

The repository currently claims a frozen retrospective baseline and a traceable prospective unseen prediction/truth-processing state. It does not claim a fixed negative-output solution, a completed N6390 AFM truth result, an added N6390 AFM dataset, or a new benchmark.

