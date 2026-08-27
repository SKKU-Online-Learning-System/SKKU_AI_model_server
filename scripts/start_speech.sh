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
: "${SPEECH_PORT:=8010}"
: "${MODEL_SERVER_HOST:=0.0.0.0}"

export CUDA_VISIBLE_DEVICES="${SPEECH_GPU_ID}"
export UV_TORCH_BACKEND=cu128

# start_all.sh prepares the cu128 Speech environment. --no-sync keeps that
# exact runtime instead of allowing uv run to resolve another CUDA backend.
exec uv run --no-sync --project "${ROOT_DIR}/speech_server" \
  uvicorn speech_server.main:app \
  --host "${MODEL_SERVER_HOST}" \
  --port "${SPEECH_PORT}" \
  --no-access-log
