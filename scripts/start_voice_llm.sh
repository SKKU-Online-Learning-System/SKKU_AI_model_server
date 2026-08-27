#!/usr/bin/env bash
set -euo pipefail

: "${VOICE_MODEL:=Qwen/Qwen3.5-9B}"
: "${VOICE_PORT:=8002}"
: "${VOICE_GPU_IDS:=4}"
: "${VOICE_MAX_MODEL_LEN:=8192}"
: "${VOICE_GPU_MEMORY_UTILIZATION:=0.88}"
: "${VOICE_LANGUAGE_MODEL_ONLY:=true}"
: "${MODEL_SERVER_HOST:=0.0.0.0}"

export CUDA_VISIBLE_DEVICES="${VOICE_GPU_IDS}"

args=(
  vllm serve "${VOICE_MODEL}"
  --host "${MODEL_SERVER_HOST}"
  --port "${VOICE_PORT}"
  --tensor-parallel-size 1
  --max-model-len "${VOICE_MAX_MODEL_LEN}"
  --gpu-memory-utilization "${VOICE_GPU_MEMORY_UTILIZATION}"
  --dtype bfloat16
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --disable-log-requests
  --enable-request-id-headers
)

if [[ "${VOICE_LANGUAGE_MODEL_ONLY}" == "true" ]]; then
  args+=(--language-model-only)
fi
if [[ -n "${MODEL_SERVER_API_KEY:-}" ]]; then
  args+=(--api-key "${MODEL_SERVER_API_KEY}")
fi

exec uv run --project /app/llm_runtime "${args[@]}"
