#!/usr/bin/env bash
python -m analysis.rheed_video_afm_story.build_final_paper_freeze --freeze-root paper_freeze --freeze-version v1 --train-full-cohort-deployment --copy-model-assets --copy-paper-assets --build-unseen-tools --validate

