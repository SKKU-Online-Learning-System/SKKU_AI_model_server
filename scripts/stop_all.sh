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

RUN_DIR="${MODEL_SERVER_RUN_DIR:-${ROOT_DIR}/.run}"

wait_pid_exit() {
  local pid="$1"
  local seconds="${2:-30}"
  for _ in $(seq 1 "${seconds}"); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

stop_service() {
  local name="$1"
  local pid_file="${RUN_DIR}/${name}.pid"

  if [[ ! -f "${pid_file}" ]]; then
    echo "==> ${name}: no pid file"
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}")"

  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "==> ${name}: stale pid file (${pid})"
    rm -f "${pid_file}"
    return 0
  fi

  echo "==> Stopping ${name} (pid ${pid})"
  kill -TERM "${pid}" 2>/dev/null || true

  if wait_pid_exit "${pid}" 30; then
    rm -f "${pid_file}"
    echo "    stopped"
    return 0
  fi

  echo "    graceful shutdown timed out; sending SIGKILL"
  kill -KILL "${pid}" 2>/dev/null || true
  wait_pid_exit "${pid}" 10 || true
  rm -f "${pid_file}"
}

stop_service "qwen-tts"
stop_service "cosyvoice"
stop_service "speech"
stop_service "voice-llm"
stop_service "text-llm"
