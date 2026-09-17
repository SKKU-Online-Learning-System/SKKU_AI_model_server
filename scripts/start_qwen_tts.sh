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
: "${QWEN_TTS_PORT:=8012}"
: "${MODEL_SERVER_HOST:=0.0.0.0}"

export CUDA_VISIBLE_DEVICES="${SPEECH_GPU_ID}"
export PYTHONPATH="${ROOT_DIR}"

exec "${ROOT_DIR}/qwen_tts_runtime/.venv/bin/uvicorn" qwen_tts_runtime.main:app \
  --host "${MODEL_SERVER_HOST}" \
  --port "${QWEN_TTS_PORT}" \
  --no-access-log
