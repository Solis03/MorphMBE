#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
ARCHIVE_ROOT="${SCRIPT_DIR:h}"
PYTHON_BIN="${ARCHIVE_ROOT}/.venv/bin/python"
CONFIG="configs/rheed_realtime_ui_m22_full27_dense_mid_v10.json"
export PYTHONPATH="${ARCHIVE_ROOT}/src:${ARCHIVE_ROOT}"
export HF_HOME="${ARCHIVE_ROOT}/tmp/huggingface"
export TORCH_HOME="${ARCHIVE_ROOT}/tmp/torch"
export MPLCONFIGDIR="${ARCHIVE_ROOT}/tmp/matplotlib"
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
  predict-video)
    if [[ $# -lt 3 ]]; then
      print -u2 "Usage: $0 predict-video VIDEO_PATH SAMPLE_ID [OUTPUT_DIR]"
      exit 2
    fi
    video_path="$2"
    sample_id="$3"
    output_dir="${4:-reproduced_outputs/predict_${sample_id}}"
    exec "${PYTHON_BIN}" scripts/smoke_rheed_realtime_pipeline.py \
      "${video_path}" --sample-id "${sample_id}" --config "${CONFIG}" \
      --output-dir "${output_dir}"
    ;;
  smoke-model-6063)
    exec "${PYTHON_BIN}" scripts/smoke_rheed_realtime_pipeline.py \
      "data/raw/raw_RHEED/N6063/rampdown to 300C.MOV" \
      --sample-id 6063 --config "${CONFIG}" \
      --output-dir "reproduced_outputs/model_smoke_6063"
    ;;
  prepare-model)
    exec "${PYTHON_BIN}" scripts/prepare_rheed_realtime_model.py \
      --config "${CONFIG}" --force
    ;;
  validate)
    exec "${PYTHON_BIN}" scripts/validate_m22_standalone.py --config "${CONFIG}"
    ;;
  list-visualizations)
    exec "${PYTHON_BIN}" scripts/list_m22_visualizations.py
    ;;
  verify-checksums)
    exec "${PYTHON_BIN}" scripts/build_archive_manifest.py verify
    ;;
  test)
    exec "${PYTHON_BIN}" -m pytest -p no:cacheprovider -q \
      tests/test_rheed_realtime_ui.py \
      tests/test_rheed_to_afm_full_cohort_loo.py \
      tests/test_rheed_to_afm_functional_morphology.py \
      tests/test_rheed_to_afm_island_generation.py
    ;;
  reproduce-m22-inclusive)
    exec "${PYTHON_BIN}" -m analysis.rheed_to_afm_full_cohort_loo.run \
      --config configs/rheed_m22_dense_mid_full27_inclusive_v1.json \
      --device auto
    ;;
  reproduce-m22-exclusion)
    exec "${PYTHON_BIN}" -m analysis.rheed_to_afm_full_cohort_loo.run \
      --config configs/rheed_m22_dense_mid_full27_exclude_6022_6101_v1.json \
      --device auto
    ;;
  help|*)
    print "Usage: $0 COMMAND"
    print "  run-ui                    Open the M20+M22c real-time UI"
    print "  predict-video PATH ID [OUT]  Analyze one RHEED video from the CLI"
    print "  smoke-model-6063           Re-run the archived 6063 end-to-end smoke"
    print "  prepare-model              Rebuild only the derived deployment bundle"
    print "  validate                   Check model, assets, figures, and smoke result"
    print "  list-visualizations        Print every delivered M22 visual result"
    print "  verify-checksums           Verify the transfer-integrity manifest"
    print "  test                       Run focused real-time and M22 tests"
    print "  reproduce-m22-inclusive    Re-run selected 27-fold M22 experiment"
    print "  reproduce-m22-exclusion    Re-run morphology-exclusion ablation"
    ;;
esac
