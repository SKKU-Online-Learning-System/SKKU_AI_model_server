#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
: "${EMBEDDING_MODEL:=Qwen/Qwen3-VL-Embedding-2B}"
: "${EMBEDDING_PORT:=8003}"
: "${EMBEDDING_GPU_IDS:=5}"
: "${EMBEDDING_MAX_MODEL_LEN:=8192}"
: "${EMBEDDING_GPU_MEMORY_UTILIZATION:=0.27}"
export CUDA_VISIBLE_DEVICES="${EMBEDDING_GPU_IDS}"
args=(serve "${EMBEDDING_MODEL}"
  --host "${MODEL_SERVER_HOST:-0.0.0.0}" --port "${EMBEDDING_PORT}"
  --dtype bfloat16 --max-model-len "${EMBEDDING_MAX_MODEL_LEN}"
  --max-num-seqs 1 --max-num-batched-tokens 8192
  --gpu-memory-utilization "${EMBEDDING_GPU_MEMORY_UTILIZATION}"
  --enforce-eager --limit-mm-per-prompt '{"image":1,"video":0}'
  --mm-processor-kwargs '{"max_pixels":1048576}'
  --no-enable-log-requests
  --runner pooling
  --pooler-config '{"pooling_type":"LAST","use_activation":true}'
)
if [[ -n "${MODEL_SERVER_API_KEY:-}" ]]; then
  args+=(--api-key "${MODEL_SERVER_API_KEY}")
fi
exec "${ROOT_DIR}/llm_runtime/.venv/bin/vllm" "${args[@]}"
