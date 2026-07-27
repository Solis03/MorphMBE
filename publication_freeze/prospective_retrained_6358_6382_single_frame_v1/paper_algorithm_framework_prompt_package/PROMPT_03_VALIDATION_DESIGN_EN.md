# Data-division and leakage-control prompt

Create a third coordinated methods schematic titled:

**“Prospective, leave-one-out, and held-one-out evaluation design”**

Use the same palette and typography as `PROMPT_01_MAIN_FRAMEWORK_EN.md`.
The purpose is to make the data division and absence of leakage immediately
clear to reviewers.

Draw three side-by-side panels:

### A. Prospective prediction

- 23 original historical samples;
- N6022 and N6099 removed;
- 21 retained historical + N6358 + N6382 = 23 training samples;
- model predicts N6342, N6389, N6390;
- their RHEED images are inputs, but their AFM labels are sealed until
  evaluation;
- prospective A3 bank: 23 source groups / 118 AFM maps.

Use small real-image thumbnails from:

- `01_rheed_inputs/roi_keyframes/`;
- `02_afm_ground_truth_selected/`;
- `03_afm_retrieval_outputs/`.

Place all ground-truth AFM thumbnails behind a dashed vertical “label firewall”
until the post-hoc evaluation step.

### B. Quantitative leave-one-out

- retained labeled cohort n = 26;
- 26 folds;
- one target held out;
- StandardScaler and all five Ridge members fit only on the other 25 rows;
- rotate the held-out marker around a circular cohort;
- output one Rq prediction for every sample.

### C. AFM held-one-out

- same 26 targets;
- target Rq model trained on the other 25 samples;
- remove every AFM belonging to the target from the A3 source bank;
- 25 source groups remain in each fold;
- retrieve one representative morphology and amplitude-scale it to predicted
  Rq;
- ground-truth selection occurs only after prediction.

For the representative AFM choices, annotate:

- N6342: ground truth 5;
- N6389: ground truth 3;
- N6390: ground truth 1;
- all other samples: quality-passing AFM with measured Rq closest to sample T4.

Use a red dashed “never crossed” boundary to show that the held-out label and
held-out AFM group never enter fitting or retrieval. Do not imply nested
hyperparameter selection or a generative model. Add a footer:

“N6022 and N6099 are absent globally; N6324 is ignored.”

Use a white background, thin elegant arrows, aligned grids, clean cohort icons,
and real microscopy thumbnails without altering their scientific content.
Return high-resolution PNG and editable SVG/PDF if supported.
