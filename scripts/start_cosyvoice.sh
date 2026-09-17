#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

: "${SPEECH_GPU_ID:=5}"
: "${COSYVOICE_PORT:=8011}"
: "${MODEL_SERVER_HOST:=0.0.0.0}"
: "${COSYVOICE_SOURCE_DIR:=${HOME}/.cache/cosyvoice/CosyVoice}"

[[ -n "${COSYVOICE_PROMPT_WAV:-}" ]] || { echo "COSYVOICE_PROMPT_WAV is required." >&2; exit 1; }
[[ -n "${COSYVOICE_PROMPT_TEXT:-}" ]] || { echo "COSYVOICE_PROMPT_TEXT is required." >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${SPEECH_GPU_ID}"

# onnxruntime-gpu is the CUDA 12 build; its CUDAExecutionProvider needs cuDNN 8,
# which is only shipped inside the torch wheel. Without this the provider fails to
# load and CosyVoice silently falls back to CPU for the speech tokenizer.
CUDNN_LIB_DIR="$(
  "${ROOT_DIR}/cosyvoice_runtime/.venv/bin/python" -c \
    'import os, nvidia.cudnn; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), "lib"))' 2>/dev/null || true
)"
if [[ -n "${CUDNN_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${CUDNN_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export PYTHONPATH="${COSYVOICE_SOURCE_DIR}:${COSYVOICE_SOURCE_DIR}/third_party/Matcha-TTS:${ROOT_DIR}"

exec "${ROOT_DIR}/cosyvoice_runtime/.venv/bin/uvicorn" cosyvoice_runtime.main:app \
  --host "${MODEL_SERVER_HOST}" \
  --port "${COSYVOICE_PORT}" \
  --no-access-log
