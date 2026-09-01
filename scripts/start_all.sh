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

command -v uv >/dev/null || { echo "uv is required but was not found in PATH." >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required but was not found in PATH." >&2; exit 1; }
command -v curl >/dev/null || { echo "curl is required but was not found in PATH." >&2; exit 1; }
command -v timeout >/dev/null || { echo "GNU timeout is required but was not found in PATH." >&2; exit 1; }

nvidia-smi >/dev/null

RUN_DIR="${MODEL_SERVER_RUN_DIR:-${ROOT_DIR}/.run}"
LOG_DIR="${MODEL_SERVER_LOG_DIR:-${ROOT_DIR}/logs}"
STARTUP_TIMEOUT_SECONDS="${MODEL_SERVER_STARTUP_TIMEOUT_SECONDS:-900}"
NCCL_PREFLIGHT_TIMEOUT_SECONDS="${NCCL_PREFLIGHT_TIMEOUT_SECONDS:-30}"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

: "${TEXT_GPU_IDS:=0,1,2,3}"
: "${TEXT_TENSOR_PARALLEL_SIZE:=4}"
: "${TEXT_PORT:=8001}"
: "${VOICE_PORT:=8002}"
: "${SPEECH_PORT:=8010}"
: "${VOICE_GPU_IDS:=4}"
: "${SPEECH_GPU_ID:=5}"

# This is a single-user development server. Always start from a clean set of
# managed processes so newly validated CUDA/NCCL/runtime settings are actually
# applied instead of silently reusing processes from an earlier attempt.
echo "==> Stopping previously managed model services for a clean start"
"${ROOT_DIR}/scripts/stop_all.sh" || true

# CUDA selection is encoded in each project's pyproject.toml via explicit
# PyTorch indexes. The LLM project routes torch, torchaudio, torchvision, and
# torchcodec to the same CUDA 12.9 index so compiled extensions stay ABI-aligned.
echo "==> Syncing LLM runtime (PyTorch CUDA 12.9 index)"
uv sync --project "${ROOT_DIR}/llm_runtime" --no-dev

echo "==> Verifying LLM CUDA runtime"
CUDA_VISIBLE_DEVICES="${VOICE_GPU_IDS}" \
"${ROOT_DIR}/llm_runtime/.venv/bin/python" -c '
from importlib.metadata import version
import torch, vllm, torchcodec
torchcodec_version = version("torchcodec")
print(f"llm vllm={vllm.__version__} torch={torch.__version__} torchcodec={torchcodec_version} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if not (torch.version.cuda or "").startswith("12.9"):
    raise SystemExit("LLM runtime is not using a CUDA 12.9 PyTorch build")
if "+cu129" not in torchcodec_version:
    raise SystemExit("LLM runtime torchcodec is not using the CUDA 12.9 build")
if not torch.cuda.is_available():
    raise SystemExit("LLM runtime cannot initialize CUDA")
torch.cuda.set_device(0)
print(f"llm gpu={torch.cuda.get_device_name(0)}")
'

# Tensor-parallel Text needs working NCCL collectives, not just working
# single-GPU CUDA. Test the exact Text GPU set before loading model weights.
# Every attempt is bounded so a hung NCCL transport can never hang start_all.sh.
NCCL_ENV_FILE="${RUN_DIR}/text-nccl.env"
write_nccl_env() {
  local safe_mode="$1"
  if [[ "${safe_mode}" == "1" ]]; then
    cat >"${NCCL_ENV_FILE}" <<EOF
NCCL_CUMEM_HOST_ENABLE=0
NCCL_IB_DISABLE=1
NCCL_NET=Socket
NCCL_SOCKET_IFNAME=lo
NCCL_DEBUG=WARN
NCCL_P2P_DISABLE=1
NCCL_SHM_DISABLE=1
EOF
  else
    cat >"${NCCL_ENV_FILE}" <<EOF
NCCL_CUMEM_HOST_ENABLE=0
NCCL_IB_DISABLE=1
NCCL_DEBUG=WARN
NCCL_P2P_DISABLE=0
NCCL_SHM_DISABLE=0
EOF
  fi
}

run_nccl_preflight() {
  set -a
  # shellcheck disable=SC1090
  source "${NCCL_ENV_FILE}"
  set +a

  echo "    NCCL preflight timeout: ${NCCL_PREFLIGHT_TIMEOUT_SECONDS}s"
  timeout --signal=TERM --kill-after=10s "${NCCL_PREFLIGHT_TIMEOUT_SECONDS}s" \
    env CUDA_VISIBLE_DEVICES="${TEXT_GPU_IDS}" \
    "${ROOT_DIR}/llm_runtime/.venv/bin/python" -m torch.distributed.run \
      --standalone \
      --nproc_per_node="${TEXT_TENSOR_PARALLEL_SIZE}" \
      "${ROOT_DIR}/scripts/nccl_preflight.py"
}

if (( TEXT_TENSOR_PARALLEL_SIZE > 1 )); then
  echo "==> Checking shared-memory mount"
  df -h /dev/shm || true

  echo "==> Verifying ${TEXT_TENSOR_PARALLEL_SIZE}-GPU NCCL all-reduce"
  write_nccl_env 0
  if ! run_nccl_preflight; then
    echo "==> Default NCCL transport failed or timed out; retrying container-safe socket transport" >&2
    write_nccl_env 1
    if ! run_nccl_preflight; then
      echo "NCCL preflight failed even in safe mode; Text LLM will not be started." >&2
      exit 1
    fi
    echo "==> NCCL safe mode selected for Text LLM"
  else
    echo "==> NCCL standard transport passed"
  fi
fi

echo "==> Syncing Speech runtime (PyTorch CUDA 12.8 index)"
uv sync --project "${ROOT_DIR}/speech_server" --no-dev

# FlashAttention is an optional acceleration layer rather than a hard startup
# dependency. Install the wheel that matches the pinned Torch/Python ABI when
# possible; QwenTTSService falls back to PyTorch SDPA if this preparation fails.
echo "==> Preparing FlashAttention 2 for Speech TTS"
if ! bash "${ROOT_DIR}/scripts/install_flash_attn.sh"; then
  echo "WARNING: FlashAttention preparation failed; TTS will fall back to SDPA." >&2
fi

echo "==> Verifying Speech CUDA runtime"
CUDA_VISIBLE_DEVICES="${SPEECH_GPU_ID}" \
"${ROOT_DIR}/speech_server/.venv/bin/python" -c '
import torch
print(f"speech torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if not (torch.version.cuda or "").startswith("12.8"):
    raise SystemExit("Speech runtime is not using a CUDA 12.8 PyTorch build")
if not torch.cuda.is_available():
    raise SystemExit("Speech runtime cannot initialize CUDA")
torch.cuda.set_device(0)
print(f"speech gpu={torch.cuda.get_device_name(0)}")
try:
    import flash_attn
    print(f"speech flash_attn={flash_attn.__version__}")
except ImportError:
    print("speech flash_attn=unavailable (SDPA fallback will be used)")
'

start_service() {
  local name="$1"
  local script="$2"
  local pid_file="${RUN_DIR}/${name}.pid"
  local log_file="${LOG_DIR}/${name}.log"

  if [[ -f "${pid_file}" ]]; then
    local old_pid
    old_pid="$(cat "${pid_file}")"
    if kill -0 "${old_pid}" 2>/dev/null; then
      echo "==> ${name} is already running (pid ${old_pid})"
      return 0
    fi
    rm -f "${pid_file}"
  fi

  echo "==> Starting ${name}; log: ${log_file}"
  : >"${log_file}"
  # Each start_* script execs the final service process, so nohup + $! gives
  # us the real long-lived service PID. Avoid setsid here because its optional
  # fork can make $! differ from the actual service process in some shells.
  nohup "${script}" >"${log_file}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" >"${pid_file}"

  sleep 2
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "${name} exited during startup. Last log lines:" >&2
    tail -n 120 "${log_file}" >&2 || true
    rm -f "${pid_file}"
    return 1
  fi

  echo "    pid: ${pid}"
}

headers=()
if [[ -n "${MODEL_SERVER_API_KEY:-}" ]]; then
  headers=(-H "Authorization: Bearer ${MODEL_SERVER_API_KEY}")
fi

service_ready() {
  local name="$1"
  local response
  case "${name}" in
    text-llm)
      curl --fail --silent --max-time 5 "${headers[@]}" "http://127.0.0.1:${TEXT_PORT}/v1/models" >/dev/null
      ;;
    voice-llm)
      curl --fail --silent --max-time 5 "${headers[@]}" "http://127.0.0.1:${VOICE_PORT}/v1/models" >/dev/null
      ;;
    speech)
      response="$(curl --fail --silent --max-time 5 "${headers[@]}" "http://127.0.0.1:${SPEECH_PORT}/health")" || return 1
      grep -Eq '"ready"[[:space:]]*:[[:space:]]*true' <<<"${response}"
      ;;
    *)
      return 1
      ;;
  esac
}

wait_for_ready() {
  local name="$1"
  local pid_file="${RUN_DIR}/${name}.pid"
  local log_file="${LOG_DIR}/${name}.log"
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
  local last_notice=${SECONDS}

  echo "==> Waiting for ${name} readiness (timeout ${STARTUP_TIMEOUT_SECONDS}s)"
  while (( SECONDS < deadline )); do
    if service_ready "${name}"; then
      echo "    ${name}: READY"
      return 0
    fi

    if [[ ! -f "${pid_file}" ]]; then
      echo "${name} has no pid file." >&2
      tail -n 160 "${log_file}" >&2 || true
      return 1
    fi

    local pid
    pid="$(cat "${pid_file}")"
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${name} exited before becoming ready. Last log lines:" >&2
      tail -n 160 "${log_file}" >&2 || true
      rm -f "${pid_file}"
      return 1
    fi

    if (( SECONDS - last_notice >= 30 )); then
      echo "    ${name}: still loading..."
      last_notice=${SECONDS}
    fi
    sleep 5
  done

  echo "${name} did not become ready within ${STARTUP_TIMEOUT_SECONDS}s. Last log lines:" >&2
  tail -n 160 "${log_file}" >&2 || true
  return 1
}

cleanup_on_failure() {
  echo "==> Startup failed; stopping services launched by start_all.sh" >&2
  "${ROOT_DIR}/scripts/stop_all.sh" || true
}

start_service "text-llm" "${ROOT_DIR}/scripts/start_text_llm.sh" || { cleanup_on_failure; exit 1; }
start_service "voice-llm" "${ROOT_DIR}/scripts/start_voice_llm.sh" || { cleanup_on_failure; exit 1; }
start_service "speech" "${ROOT_DIR}/scripts/start_speech.sh" || { cleanup_on_failure; exit 1; }

wait_for_ready "text-llm" || { cleanup_on_failure; exit 1; }
wait_for_ready "voice-llm" || { cleanup_on_failure; exit 1; }
wait_for_ready "speech" || { cleanup_on_failure; exit 1; }

echo
echo "All model services are READY."
echo "Text LLM : http://127.0.0.1:${TEXT_PORT}/v1"
echo "Voice LLM: http://127.0.0.1:${VOICE_PORT}/v1"
echo "Speech   : http://127.0.0.1:${SPEECH_PORT}"
echo "Logs     : ${LOG_DIR}"
