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

: "${TEXT_MODEL:=Qwen/Qwen3.8-27B}"
: "${TEXT_PORT:=8001}"
: "${TEXT_GPU_IDS:=0,1,2,3}"
: "${TEXT_TENSOR_PARALLEL_SIZE:=4}"
: "${TEXT_MAX_MODEL_LEN:=16384}"
: "${TEXT_GPU_MEMORY_UTILIZATION:=0.90}"
: "${TEXT_LANGUAGE_MODEL_ONLY:=true}"
: "${MODEL_SERVER_HOST:=0.0.0.0}"

export CUDA_VISIBLE_DEVICES="${TEXT_GPU_IDS}"

VLLM_BIN="${ROOT_DIR}/llm_runtime/.venv/bin/vllm"
[[ -x "${VLLM_BIN}" ]] || { echo "LLM uv environment is missing. Run ./scripts/start_all.sh first." >&2; exit 1; }

args=(
  serve "${TEXT_MODEL}"
  --host "${MODEL_SERVER_HOST}"
  --port "${TEXT_PORT}"
  --tensor-parallel-size "${TEXT_TENSOR_PARALLEL_SIZE}"
  --max-model-len "${TEXT_MAX_MODEL_LEN}"
  --gpu-memory-utilization "${TEXT_GPU_MEMORY_UTILIZATION}"
  --dtype bfloat16
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --no-enable-log-requests
  --enable-request-id-headers
)

if [[ "${TEXT_LANGUAGE_MODEL_ONLY}" == "true" ]]; then
  args+=(--language-model-only)
fi
if [[ -n "${MODEL_SERVER_API_KEY:-}" ]]; then
  args+=(--api-key "${MODEL_SERVER_API_KEY}")
fi

exec "${VLLM_BIN}" "${args[@]}"
