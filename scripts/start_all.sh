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
command -v setsid >/dev/null || { echo "setsid is required but was not found in PATH." >&2; exit 1; }

nvidia-smi >/dev/null

RUN_DIR="${MODEL_SERVER_RUN_DIR:-${ROOT_DIR}/.run}"
LOG_DIR="${MODEL_SERVER_LOG_DIR:-${ROOT_DIR}/logs}"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

# Resolve/install the two isolated uv environments before starting concurrent services.
echo "==> Syncing LLM runtime"
uv sync --project "${ROOT_DIR}/llm_runtime" --no-dev

echo "==> Syncing Speech runtime"
uv sync --project "${ROOT_DIR}/speech_server" --no-dev

start_service() {
  local name="$1"
  local script="$2"
  local pid_file="${RUN_DIR}/${name}.pid"
  local log_file="${LOG_DIR}/${name}.log"

  if [[ -f "${pid_file}" ]]; then
    local old_pid
    old_pid="$(cat "${pid_file}")"
    if kill -0 -- "-${old_pid}" 2>/dev/null; then
      echo "==> ${name} is already running (process group ${old_pid})"
      return 0
    fi
    rm -f "${pid_file}"
  fi

  echo "==> Starting ${name}; log: ${log_file}"
  nohup setsid "${script}" >"${log_file}" 2>&1 < /dev/null &
  local pid=$!
  echo "${pid}" > "${pid_file}"

  sleep 2
  if ! kill -0 -- "-${pid}" 2>/dev/null; then
    echo "${name} exited during startup. Last log lines:" >&2
    tail -n 80 "${log_file}" >&2 || true
    rm -f "${pid_file}"
    return 1
  fi

  echo "    process group: ${pid}"
}

start_service "text-llm" "${ROOT_DIR}/scripts/start_text_llm.sh"
start_service "voice-llm" "${ROOT_DIR}/scripts/start_voice_llm.sh"
start_service "speech" "${ROOT_DIR}/scripts/start_speech.sh"

echo
echo "All services were launched. Model loading can take several minutes."
echo "Logs: ${LOG_DIR}"
echo "Check readiness with: ./scripts/healthcheck.sh"
