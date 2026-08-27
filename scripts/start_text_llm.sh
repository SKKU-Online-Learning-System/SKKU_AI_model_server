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
: "${TEXT_MAX_NUM_SEQS:=32}"
: "${TEXT_GPU_MEMORY_UTILIZATION:=0.90}"
: "${TEXT_LANGUAGE_MODEL_ONLY:=true}"
: "${TEXT_DISABLE_CUSTOM_ALL_REDUCE:=true}"
: "${MODEL_SERVER_HOST:=0.0.0.0}"

# start_all.sh writes the NCCL transport that actually passed a 4-GPU
# all-reduce preflight on this container. Reuse it for later manual restarts.
RUN_DIR="${MODEL_SERVER_RUN_DIR:-${ROOT_DIR}/.run}"
NCCL_ENV_FILE="${RUN_DIR}/text-nccl.env"
if [[ -f "${NCCL_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${NCCL_ENV_FILE}"
  set +a
fi

# R570 + CUDA 12.x containers can fail NCCL cuMem host allocation even when
# single-GPU CUDA works. This is the conservative baseline recommended by NCCL
# troubleshooting; start_all.sh falls back further if its collective test fails.
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_VISIBLE_DEVICES="${TEXT_GPU_IDS}"

VLLM_BIN="${ROOT_DIR}/llm_runtime/.venv/bin/vllm"
[[ -x "${VLLM_BIN}" ]] || { echo "LLM uv environment is missing. Run ./scripts/start_all.sh first." >&2; exit 1; }

args=(
  serve "${TEXT_MODEL}"
  --host "${MODEL_SERVER_HOST}"
  --port "${TEXT_PORT}"
  --tensor-parallel-size "${TEXT_TENSOR_PARALLEL_SIZE}"
  --max-model-len "${TEXT_MAX_MODEL_LEN}"
  --max-num-seqs "${TEXT_MAX_NUM_SEQS}"
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
if [[ "${TEXT_DISABLE_CUSTOM_ALL_REDUCE}" == "true" ]]; then
  args+=(--disable-custom-all-reduce)
fi
if [[ -n "${MODEL_SERVER_API_KEY:-}" ]]; then
  args+=(--api-key "${MODEL_SERVER_API_KEY}")
fi

exec "${VLLM_BIN}" "${args[@]}"
