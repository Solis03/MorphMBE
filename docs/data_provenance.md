# Data Provenance

This document summarizes the current data state without changing any sample ID, label, target, prediction, or exclusion rule.

## Historical Cohort

The retrospective publication package freezes 23 strict historical growth groups. The split unit is the growth run/sample group, not individual scans or frames. The canonical sample index, target table, fold assignments, selected RHEED keyframes, representative AFM maps, model outputs, and removelist snapshot are in `publication_freeze/rheed_afm_single_frame_v1_2026-07-18/`.

## Prospective Samples

The prospective unseen RHEED cohort contains five sample IDs:

- N6342
- N6358
- N6382
- N6389
- N6390

The current prediction package contains all five samples. The current AFM truth join has four matched prediction/AFM samples: N6342, N6358, N6382, and N6389.

N6390 remains a prediction-without-AFM-truth case. N6324 remains an AFM-truth-without-prediction mismatch case from the AFM extra-five batch. Neither is renamed, merged, removed, or corrected by this cleanup.

## Raw And Derived Data

- Raw historical AFM/RHEED data are retained under `data/raw/` and related source pairing folders.
- Prospective source RHEED files are retained under `data/compressedfile/`.
- Extra-five raw AFM files are retained under `data/AFM-extra-five/`.
- Derived AFM processing follows raw ZSensor extraction, first-order plane correction, then second-order background subtraction.
- Prospective AFM truth manifests and representative maps are stored under `publication_freeze/prospective_unseen_single_frame_v1/ground_truth_afm/`.

## RHEED Selection Route

Historical keyframe and ROI selections are retained as human-created data under `data/rheed_keyframe_selection/` and frozen snapshots. Prospective keyframe selections are retained in the prospective package metadata, raw keyframe PNGs, ROI crops, model-ready arrays, selection manifests, and provenance files.

## Exclusion Rules

The canonical removelist is retained at repository level as `removelist.txt` and inside the retrospective freeze snapshot. Exclusion and group identity are part of the scientific provenance and are not cleanup candidates.

## Data Classes

- Raw: original AFM/RHEED files, archives, manually paired source trees.
- Human-created: annotations, manual keyframes, ROIs, review manifests, unblind keys.
- Derived: processed AFM arrays, rendered previews, model-ready keyframes, embeddings, retrieval maps.
- Frozen: immutable retrospective and prospective publication evidence under `publication_freeze/`.

