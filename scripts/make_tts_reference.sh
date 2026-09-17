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
export CUDA_VISIBLE_DEVICES="${SPEECH_GPU_ID}"
export PYTHONPATH="${ROOT_DIR}"

exec "${ROOT_DIR}/qwen_tts_runtime/.venv/bin/python" "${ROOT_DIR}/scripts/make_tts_reference.py"
