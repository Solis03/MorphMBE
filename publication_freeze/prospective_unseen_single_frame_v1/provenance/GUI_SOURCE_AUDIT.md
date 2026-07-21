# GUI Source Audit

This audit covers the existing manual graphical RHEED keyframe/ROI tools found in the repository before creating the prospective unseen package.

## Candidate GUI and Selection Scripts

- `tools/manual_rheed_roi_reviewer.py`: most recent local GUI launcher. It discovers pre-extracted PNG frame libraries under `data/rheed_keyframe_selection`, supports smoke testing, starts at the first incomplete record, and launches the PySide6 reviewer.
- `src/rheed2morph/rheed/manual_roi_qt.py`: functional PySide6 GUI used most recently. It shows one extracted PNG frame at a time, provides frame slider/spinbox navigation, lets the user set `keyframe_index`, supports ROI drawing in source-frame pixel coordinates, saves atomically through helper functions, and can resume/edit existing metadata.
- `src/rheed2morph/rheed/manual_roi.py`: non-GUI helper module used by the PySide6 reviewer. It provides numeric PNG frame sorting, source/display coordinate mapping, ROI validation, clip index calculation, atomic JSON metadata writes, and ROI preview contact-sheet generation.
- `src/rheed2morph/rheed/keyframe_selection.py`: non-interactive preparation tool that decodes videos into full-frame PNG libraries for manual keyframe selection. It is not the final GUI, but it documents the older manual-selection data layout.
- `src/rheed2morph/rheed/manual_frame_selection.py`: older text/manual selection support for candidate frame outputs, not the most recent functional GUI.
- `src/rheed2morph/rheed/select_representative_frames.py` and `select_representative_frames_v2.py`: older frame-candidate workflows with OpenCV/imageio-style decoding and manual text selection files, not suitable as the current GUI.

## Most Recent Functional Tool

The most recent functional graphical implementation is:

`tools/manual_rheed_roi_reviewer.py` -> `src/rheed2morph/rheed/manual_roi_qt.py` -> `src/rheed2morph/rheed/manual_roi.py`

Memory and repository tests indicate this PySide6 path replaced Tkinter in this environment because Tkinter runtime support was unreliable, while PySide6 worked with an offscreen smoke path.

## Dependencies and Decoding

The existing GUI depends on:

- PySide6 for the GUI;
- Pillow for PNG display, validation, and ROI preview generation;
- `src/rheed2morph/rheed/manual_roi.py` helpers;
- pre-extracted PNG frames in each video's `frames` directory.

The existing GUI does not read MPG files directly. It assumes an upstream extraction step has already decoded full video frame PNGs. That upstream extraction code uses imageio/ffmpeg in `keyframe_selection.py`; older representative-frame scripts also include OpenCV-based decoding, but OpenCV exact seeking is not relied on here.

For the prospective unseen package, the GUI was minimally repaired/wrapped rather than copied byte-for-byte:

- the PySide6 local GUI pattern, resume behavior, and explicit save flow were preserved;
- the older source-pixel ROI drawing behavior was integrated into the MPG-in-place GUI;
- the pre-extracted PNG dependency was replaced with an ffprobe/ffmpeg decoder abstraction that reads the original `.mpg` paths in place;
- only individually requested frames are cached under the package cache;
- the saved raw keyframe PNG is copied from the exact frame displayed by the GUI;
- the selected ROI crop is saved as a derived PNG and its source-pixel coordinates are recorded;
- deterministic re-extraction of the selected frame is checked and recorded.

## Save Behavior and Schema Fit

The older GUI saved selections into the original `data/rheed_keyframe_selection/<sample>/metadata.json` layout under each video payload:

- `selection.keyframe_index`;
- `selection.clip_frame_count`;
- `selection.roi` in `source_frame_pixels`;
- optional `roi_preview.png` as QC-only output.

The prospective unseen package uses a separate schema:

- one JSON file per sample under `metadata/samples/`;
- selected source MPG metadata;
- selected frame index and timestamp;
- requested and effective `frames_before`, `frames_after`, and `frame_stride`;
- raw selected keyframe PNG path and hash;
- ROI coordinates in `source_frame_pixels`, ROI crop PNG path, and ROI crop hash;
- `model_ready_keyframe: null` because the frozen prospective input bridge is blocked and the model-ready transform is not fully specified by the frozen package.

## ROI and Context Support

The previous GUI supports ROI selection. That behavior is now integrated into the prospective GUI, using source-frame pixel coordinates after ffmpeg decodes the displayed MPG frame. The current frozen single-frame package describes a manually selected RHEED keyframe encoded as raw luminance, and the packaged prospective input schema is blocked rather than a deployable predictor schema. Because the model-ready transformation is not completely unambiguous from the freeze, this prospective package still leaves `model_ready_keyframe: null`; it saves the raw selected frame plus a separate ROI crop and ROI coordinate metadata for the later prediction bridge.

The previous GUI records a centered `clip_frame_count`, but it does not expose separate `frames_before` and `frames_after`. The prospective GUI adds explicit `frames_before`, `frames_after`, and `frame_stride` fields with defaults 0, 0, and 1, and records both requested and effective context windows.

## Frozen Model Compatibility Finding

The freeze documents a single manually selected RHEED keyframe encoded by DINOv2 ViT-S/14 with `E1_dino_keyframe` / raw-luminance wording. The freeze's prospective deployment input schema is `{"status": "blocked"}` and not a complete unseen prediction contract. Therefore this package always saves the raw, lossless selected frame and leaves model-ready conversion unperformed until the later prediction step defines the exact deterministic transform.
