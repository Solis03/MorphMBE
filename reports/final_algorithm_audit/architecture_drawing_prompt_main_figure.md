Draw a publication-grade methods schematic for a RHEED-to-AFM surrogate pipeline.

Main message: the deployable quantitative route predicts AFM roughness Rq from a frozen DINO RHEED embedding; the visual route retrieves a representative historical AFM morphology using fixed A3 descriptor ranking and rescales it to q10/q50/q90 Rq.

Required blocks: Raw RHEED keyframe PNG; manual ROI crop; 224 x 224 luminance image; DINOv2 ViT-S/14 frozen encoder; 1536-D temporal aggregate feature; five ridge regressors; median q50 Rq; strict q10/q90 interval; 11-D descriptor vector; strict A3 AFM bank; A3 distance; selected historical AFM; unit-Rq morphology; q10/q50/q90 representative AFM maps.

Required warnings in small callouts: "Representative retrieval, not pixel reconstruction"; "Final visual method: fixed A3"; "Phase3A AFM autoencoder not used"; "Strict OOF separated from full-cohort future deployment"; "Do not show per-sample mixed best as the final method."
