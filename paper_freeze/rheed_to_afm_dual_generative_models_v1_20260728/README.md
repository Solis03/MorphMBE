# Dual generative RHEED-to-AFM paper model freeze

This freeze separates the two models intended to support the first paper
draft. They answer different scientific questions and must not be pooled into
one evaluation table without an explicit protocol column.

## MODEL_A - MorphMBE-M12a-Strict15-RangeTerrace-v1

- Short name: **M12a-Strict15**
- Purpose: strongest development-cohort morphology generation result.
- Primary evaluation: strict leave-one-growth-out over 15 development
  growths; each point fits 14 and predicts one.
- Separate evidence: three pre-existing validation growths, fit from the 15
  development growths.
- Closed evidence: five historical-test growths / 24 AFM scans were not used
  for M12 development, selection or primary evaluation.
- Generator: stochastic edge-preserving Laguerre island/terrace generator.
- Claim boundary: development evidence, not a prospective untouched test.

## MODEL_B - MorphMBE-M14i-Full23-OODAware-v1

- Short name: **M14i-Full23**
- Purpose: broader retrospective robustness, OOD and confidence evaluation.
- Primary evaluation: all 23 growths held out once; each point fits 22.
- Rq head: M14g curated/R3D 60:40 multiview blend.
- FSMI head: M14b target-blind RHEED-density weighted regression.
- Image generator: frozen M12a edge-preserving island/terrace generator.
- Claim boundary: retrospective method-development evidence, not a future
  untouched test.

## Frozen material

Each model directory contains the exact cohort roles, copied parameter files,
frozen headline metrics and a manifest of every code, result, report and
figure artifact. Each manifest records SHA-256, Git blob ID and source commit.
The shared directory snapshots only derived manifests and feature tables; no
raw RHEED video, raw AFM file or height array is copied.

Run `python3 scripts/freeze_rheed_to_afm_paper_models.py --validate` from the
repository root to verify the freeze.
