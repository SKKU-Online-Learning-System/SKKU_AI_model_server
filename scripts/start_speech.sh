#!/usr/bin/env bash
set -euo pipefail

: "${SPEECH_GPU_ID:=5}"
: "${SPEECH_PORT:=8010}"
: "${MODEL_SERVER_HOST:=0.0.0.0}"

# In compose this variable is already remapped to the physical SPEECH_GPU_ID.
# For direct host execution, map the requested physical GPU here.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SPEECH_GPU_ID}"
fi

exec uv run --project /app/speech_server \
  uvicorn speech_server.main:app \
  --host "${MODEL_SERVER_HOST}" \
  --port "${SPEECH_PORT}" \
  --no-access-log
