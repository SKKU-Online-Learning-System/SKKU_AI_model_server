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

: "${VOICE_MODEL:=Qwen/Qwen3.5-9B}"
: "${VOICE_PORT:=8002}"
: "${VOICE_GPU_IDS:=4}"
: "${VOICE_MAX_MODEL_LEN:=8192}"
: "${VOICE_MAX_NUM_SEQS:=32}"
: "${VOICE_GPU_MEMORY_UTILIZATION:=0.88}"
: "${VOICE_LANGUAGE_MODEL_ONLY:=false}"
: "${MODEL_SERVER_HOST:=0.0.0.0}"

export CUDA_VISIBLE_DEVICES="${VOICE_GPU_IDS}"

VLLM_BIN="${ROOT_DIR}/llm_runtime/.venv/bin/vllm"
[[ -x "${VLLM_BIN}" ]] || { echo "LLM uv environment is missing. Run ./scripts/start_all.sh first." >&2; exit 1; }

args=(
  serve "${VOICE_MODEL}"
  --host "${MODEL_SERVER_HOST}"
  --port "${VOICE_PORT}"
  --tensor-parallel-size 1
  --max-model-len "${VOICE_MAX_MODEL_LEN}"
  --max-num-seqs "${VOICE_MAX_NUM_SEQS}"
  --gpu-memory-utilization "${VOICE_GPU_MEMORY_UTILIZATION}"
  --dtype bfloat16
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --no-enable-log-requests
  --enable-request-id-headers
)

if [[ "${VOICE_LANGUAGE_MODEL_ONLY}" == "true" ]]; then
  args+=(--language-model-only)
else
  args+=(--limit-mm-per-prompt '{"image":1,"video":0}'
         --mm-processor-kwargs '{"max_pixels":1048576}')
fi
if [[ -n "${MODEL_SERVER_API_KEY:-}" ]]; then
  args+=(--api-key "${MODEL_SERVER_API_KEY}")
fi

exec "${VLLM_BIN}" "${args[@]}"
