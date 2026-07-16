# Architecture Layout

Use a left-to-right layout with two visible lanes after the shared RHEED preprocessing block.

Top lane: quantitative RHEED-to-Rq route from frozen DINO features to five ridge members and median q50 Rq.

Bottom lane: representative-AFM route from predicted Rq/descriptors to fixed A3 candidate ranking, selected historical morphology, unit-Rq normalization, and q10/q50/q90 amplitude rescaling.

Place strict OOF and deployment as separate callouts. The strict OOF callout must say "22 candidate groups per held-out fold; held-out AFM excluded." The deployment callout must say "23 historical groups / 116 scans; future-only; not an independent test." Do not draw a neural AFM pixel decoder in the final pipeline.
