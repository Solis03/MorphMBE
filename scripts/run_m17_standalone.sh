#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
ARCHIVE_ROOT="${SCRIPT_DIR:h}"
PYTHON_BIN="${ARCHIVE_ROOT}/.venv/bin/python"
CONFIG="configs/rheed_realtime_ui_m17_full27_line3_exclude6081_v9.json"
export PYTHONPATH="${ARCHIVE_ROOT}/src:${ARCHIVE_ROOT}"
export HF_HOME="${ARCHIVE_ROOT}/tmp/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "${ARCHIVE_ROOT}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  print -u2 "Missing .venv/bin/python. Run: uv sync --frozen --extra test"
  exit 2
fi

command_name="${1:-help}"
case "${command_name}" in
  run-ui)
    exec "${PYTHON_BIN}" scripts/run_rheed_realtime_ui.py --config "${CONFIG}"
    ;;
  prepare-model)
    exec "${PYTHON_BIN}" scripts/prepare_rheed_realtime_model.py \
      --config "${CONFIG}" --force
    ;;
  validate)
    exec "${PYTHON_BIN}" scripts/validate_m17_standalone.py --config "${CONFIG}"
    ;;
  verify-checksums)
    exec "${PYTHON_BIN}" scripts/build_archive_manifest.py verify
    ;;
  test)
    exec "${PYTHON_BIN}" -m pytest -p no:cacheprovider -q \
      tests/test_rheed_realtime_ui.py \
      tests/test_rheed_n6342_sparse_island.py \
      tests/test_rheed_to_afm_functional_morphology.py
    ;;
  smoke-model-6342)
    mkdir -p reproduced_outputs/model_smoke_N6342
    exec "${PYTHON_BIN}" scripts/smoke_rheed_realtime_pipeline.py \
      "data/compressedfile/N6342/Ramp down to 200C.avi" \
      --sample-id N6342 --config "${CONFIG}" \
      --output-dir "reproduced_outputs/model_smoke_N6342"
    ;;
  reproduce-m17)
    exec "${PYTHON_BIN}" -m analysis.rheed_to_afm_full_cohort_loo.run \
      --config configs/rheed_m17_end_to_end_generation_line3_full27_sparse_v1.json \
      --device mps
    ;;
  help|*)
    print "Usage: $0 COMMAND"
    print "  run-ui             Open the real-time RHEED-to-AFM UI"
    print "  prepare-model       Rebuild the derived full-27 deployment bundle"
    print "  validate            Check model/data/report/package invariants"
    print "  verify-checksums    Verify the frozen archive SHA-256 manifest"
    print "  test                Run focused UI and generator tests"
    print "  smoke-model-6342    Run N6342 raw video through the model"
    print "  reproduce-m17       Re-run the complete retrospective LOO experiment"
    ;;
esac
