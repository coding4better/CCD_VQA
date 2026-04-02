#!/usr/bin/env bash
set -euo pipefail

# Batch download local models and run benchmark model-by-model.
# Usage:
#   bash batch_download_and_run.sh
# Optional env vars:
#   MODELS_CSV="internvl3-2b,internvl3-8b"
#   BENCHMARK_VIDEO_DIR="/root/autodl-tmp/video"
#   BENCHMARK_QA_CSV_LIST="/root/autodl-tmp/CCD_VQA/VRU/vid_list/multi_version_data/generated_options_5opts_20260327_085451.csv"
#   HF_ENDPOINT="https://hf-mirror.com"
#   HF_LOCAL_DIR_ROOT="/root/autodl-tmp/hf_models"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DEFAULT_MODELS="internvl3-2b,internvl3-8b"
MODELS_CSV="${MODELS_CSV:-${DEFAULT_MODELS}}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_LOCAL_DIR_ROOT="${HF_LOCAL_DIR_ROOT:-/root/autodl-tmp/hf_models}"

if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi

export BENCHMARK_VIDEO_DIR="${BENCHMARK_VIDEO_DIR:-/root/autodl-tmp/video}"
export BENCHMARK_MAX_VIDEOS="${BENCHMARK_MAX_VIDEOS:-0}"
export BENCHMARK_PARALLEL_MODELS="0"
export BENCHMARK_QA_CSV_LIST="${BENCHMARK_QA_CSV_LIST:-${PROJECT_ROOT}/VRU/vid_list/multi_version_data/generated_options_5opts_20260327_085451.csv}"

echo "[INFO] Script dir: ${SCRIPT_DIR}"
echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Models: ${MODELS_CSV}"
echo "[INFO] Video dir: ${BENCHMARK_VIDEO_DIR}"
echo "[INFO] QA CSV: ${BENCHMARK_QA_CSV_LIST}"
echo "[INFO] Model local dir root: ${HF_LOCAL_DIR_ROOT}"

python - <<'PY'
import importlib.util
import sys
if importlib.util.find_spec("huggingface_hub") is None:
    sys.exit("[ERROR] huggingface_hub is not installed. Run: pip install -U huggingface_hub hf_transfer")
print("[INFO] huggingface_hub detected")
PY

model_to_repo() {
  case "$1" in
    qwen2_5_vl_7b) echo "Qwen/Qwen2.5-VL-7B-Instruct" ;;
    llava-onevision-7b|llava-ov-7b|llaba-ov-7b) echo "llava-hf/llava-onevision-qwen2-7b-ov-hf" ;;
    internvl3-2b) echo "OpenGVLab/InternVL3-2B" ;;
    internvl3-8b) echo "OpenGVLab/InternVL3-8B" ;;
    internvl2.5-2b) echo "OpenGVLab/InternVL2_5-2B" ;;
    internvl2.5-1b) echo "OpenGVLab/InternVL2_5-1B" ;;
    internvl2.5-4b) echo "OpenGVLab/InternVL2_5-4B" ;;
    internvl2.5-8b) echo "OpenGVLab/InternVL2_5-8B" ;;
    *) echo "" ;;
  esac
}

repo_to_local_dir() {
  local repo="$1"
  local short_name
  short_name="${repo##*/}"
  echo "${HF_LOCAL_DIR_ROOT}/${short_name}"
}

IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"
for raw_model in "${MODELS[@]}"; do
  model="$(echo "${raw_model}" | xargs)"
  [ -z "${model}" ] && continue

  repo="$(model_to_repo "${model}")"
  if [ -z "${repo}" ]; then
    echo "[WARN] Skip download: no repo mapping for model '${model}'"
    continue
  fi

  local_dir="$(repo_to_local_dir "${repo}")"
  mkdir -p "${local_dir}"
  if [ -f "${local_dir}/config.json" ]; then
    echo "[INFO] Already downloaded: ${repo} -> ${local_dir}"
    continue
  fi

  echo "[INFO] Downloading ${repo} -> ${local_dir}"
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${repo}", local_dir="${local_dir}")
print("[OK] Download complete: ${repo}")
PY
done

cd "${SCRIPT_DIR}"
for raw_model in "${MODELS[@]}"; do
  model="$(echo "${raw_model}" | xargs)"
  [ -z "${model}" ] && continue

  echo "============================================================"
  echo "[RUN] ${model}"
  echo "============================================================"

  export BENCHMARK_MODELS="${model}"
  python run_benchmark_v2.py
done

echo "[DONE] Batch run finished. Check outputs in ${SCRIPT_DIR}/result"
