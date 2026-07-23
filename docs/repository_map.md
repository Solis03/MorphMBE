# Repository Map

This map records the repository layout after the Nature Communications baseline cleanup. It does not redefine scientific results; the immutable freeze packages remain the source of truth.

## Canonical Publication Evidence

- `publication_freeze/rheed_afm_single_frame_v1_2026-07-18/`: frozen retrospective single-frame package. It contains the 23 strict historical growth groups, selected keyframes, frozen model artifacts, strict OOF predictions, metrics, retrieval outputs, figures, provenance, and reproduction helpers.
- `publication_freeze/prospective_unseen_single_frame_v1/`: prospective unseen package for N6342, N6358, N6382, N6389, and N6390. It contains manual keyframe selections, full-cohort predictions, ensemble-member predictions, retrieval outputs, AFM truth processing for the current extra-five batch, and the N6324/N6390 mismatch record.
- `paper_freeze/`: prior paper-oriented freeze package. It is retained as `UNKNOWN_KEEP` until a separate provenance and byte-level comparison proves redundancy.

## Data Locations

- `data/raw/`: raw historical experimental inputs.
- `data/compressedfile/`: raw/source prospective RHEED files used by the unseen keyframe package.
- `data/AFM-extra-five/`: raw AFM files for the extra-five AFM batch.
- `data/rheed_keyframe_selection/` and `annotations/`: human-created selections, ROIs, reviews, and annotation records.
- `data/processed_afm/`, `data/plane_corrected_afm/`, `data/afm_second_order/`: historical derived AFM processing products.
- `data/processed_afm_extra_five/`, `data/plane_corrected_afm_extra_five/`, `data/afm_second_order_extra_five/`: derived AFM processing products for the prospective AFM batch.
- `data/manifests/` and `removelist.txt`: canonical manifests and exclusion/removelist rules.

## Active Code

- `src/rheed2morph/`: reusable package code for AFM inspection/extraction, pairing, manifests, pipeline entry points, RHEED utilities, and generative/benchmark-related modules.
- `scripts/`: command-line research utilities, including AFM extraction, plane correction, second-order fitting, one-to-one manifest construction, and descriptor reconstruction helpers.
- `analysis/`: reusable exploratory research code for single-frame RHEED, RHEED roughness, peak/saddle descriptors, and RHEED-to-AFM story experiments.
- `configs/`, `tools/`, and `tests/`: configuration, manual tools, and tests.

## Experiment Outputs

- `reports/`: lightweight summaries and retained compact scientific records. Large generated figures and smoke outputs should remain isolated and should not be treated as canonical unless explicitly frozen.
- `outputs/` and `checkpoints/`: ignored local/generated experiment products. They require experiment-specific review before cleanup.
- New benchmark or exploratory runs should write to a timestamped subdirectory under an ignored output root and include a small summary, command record, config, seed, code commit, and retained-best-artifact rationale.

