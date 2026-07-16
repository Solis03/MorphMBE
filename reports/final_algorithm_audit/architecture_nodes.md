| node_id | label | track | shape | role |
| --- | --- | --- | --- | --- |
| N01 | Raw RHEED keyframe PNG | shared | H x W x 3 uint8 | input |
| N02 | Manual ROI crop and luminance | shared | roi_height x roi_width -> 224 x 224 uint8 | preprocessing |
| N03 | DINOv2 ViT-S/14 frozen encoder | quantitative | [1,3,224,224] -> CLS [1,384] | encoder |
| N04 | Phase2A temporal aggregation | quantitative | [1,384] -> [1,1536] | feature |
| N05 | Five full-cohort ridge members | quantitative | 5 x scalar Rq | regression |
| N06 | Median Rq ensemble | quantitative | q50 scalar nm | prediction |
| N07 | Strict interval q10/q50/q90 | strict_oof | 3 scalar Rq values | uncertainty |
| N08 | Predicted descriptor vector | visual | [1,11] | condition |
| N09 | Strict A3 representative AFM bank | strict_oof | 22 x 11 candidates per held-out fold | retrieval bank |
| N10 | A3 distance and source selection | visual | 22 distances -> one source | selection |
| N11 | Selected historical AFM morphology | visual | 256 x 256 | source map |
| N12 | Unit-Rq morphology projection | visual | 256 x 256 | normalization |
| N13 | Representative AFM q10/q50/q90 maps | visual | 3 x 256 x 256 | output |
| N14 | Full-cohort deployment visual bank | deployment | 23 groups / 116 scans | future-only bank |
| N15 | Current freeze unseen script | deployment caveat | deterministic placeholder embedding; Rq-nearest scan | discrepancy |
