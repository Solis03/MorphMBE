#!/usr/bin/env bash
python 13_UNSEEN_INFERENCE/predict_unseen_batch.py --bundle-root . --manifest 13_UNSEEN_INFERENCE/example_unseen_manifest.csv --output-root 15_REPRODUCIBILITY/smoke_test_output --freeze-id $(cat 01_FREEZE_AND_PROVENANCE/FREEZE_ID.txt)

