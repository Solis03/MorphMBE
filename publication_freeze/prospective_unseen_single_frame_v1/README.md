# Prospective Unseen RHEED Keyframe Selection

This package prepares manual keyframe selections for five prospective unseen RHEED samples. It is separate from the retrospective strict OOF evidence in `publication_freeze/rheed_afm_single_frame_v1_2026-07-18`, and it does not modify that frozen package.

The original videos are not copied here. The tools read `.mpg` files in place under `data/compressedfile/`, ignore `.imm` and `.avi` files, and export only the selected raw keyframe PNG plus metadata, manifests, previews, logs, and bounded cache files.

No model prediction is run at this stage. The current frozen model remains single-frame. `frames_before` and `frames_after` are recorded only for provenance and possible future temporal modeling.

## Command Sequence

Step 1: discover videos

```bash
python publication_freeze/prospective_unseen_single_frame_v1/code/discover_unseen_videos.py
```

Step 2: launch GUI

```bash
python publication_freeze/prospective_unseen_single_frame_v1/code/launch_keyframe_selector.py
```

Step 3: manually select five keyframes

Use the GUI to select one keyframe per sample. If a sample has multiple `.mpg` files, choose the intended MPG from the dropdown. Use `Set as keyframe`, draw or enter a source-pixel ROI, confirm `frames_before`, `frames_after`, and `frame_stride`, then use `Save selection`.

Step 4: finalize selections

```bash
python publication_freeze/prospective_unseen_single_frame_v1/code/finalize_keyframe_selections.py
```

Step 5: validate selections

```bash
python publication_freeze/prospective_unseen_single_frame_v1/code/validate_keyframe_selections.py --require-complete
```

## Resume and Revise

The GUI loads existing `metadata/samples/*.json` files and starts at the first sample whose keyframe plus ROI selection is not complete. Existing keyframe-only records are treated as `needs_roi_review` so the original frame can be retained while adding an ROI. To revise a sample, select it from the sample dropdown, choose a new frame if needed, draw or edit the ROI, click `Set as keyframe`, and save again. The per-sample JSON and consolidated manifests are rewritten atomically.

## Cache

The GUI cache is under `cache/` and contains only individually requested decoded frames and small support artifacts, not complete videos. It can be cleared with:

```bash
python publication_freeze/prospective_unseen_single_frame_v1/code/launch_keyframe_selector.py --clear-cache
```

## Outputs

- Per-sample metadata: `metadata/samples/N6342.json` through `metadata/samples/N6390.json`
- Raw selected keyframes: `keyframes/raw/`
- ROI crops from selected keyframes: `keyframes/roi/`
- Optional model-ready keyframes: `keyframes/model_ready/` remains empty unless a later task defines a fully deterministic transform
- Discovery manifest: `manifests/discovered_mpg_files.csv`
- Selection manifest: `manifests/unseen_keyframe_manifest.csv` and `.json`
- Context frame index without copying context frames: `manifests/unseen_context_frame_index.csv`
- Validation report: `provenance/selection_validation_report.json`

For the later prediction step, provide this package with completed `metadata/samples/*.json`, `keyframes/raw/*.png`, `manifests/unseen_keyframe_manifest.csv`, and `manifests/unseen_context_frame_index.csv`.

## Dependencies

Python dependencies are listed in `requirements-keyframe-selector.txt`. The system dependency is `ffmpeg`/`ffprobe`; on macOS install it with:

```bash
brew install ffmpeg
```
