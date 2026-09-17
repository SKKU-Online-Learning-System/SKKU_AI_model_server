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

# All live descendants of a pid, deepest first. vLLM's API server forks an
# EngineCore that forks one worker per GPU; SIGKILL on the API server alone
# orphans them and they keep the GPUs until killed by hand.
descendants() {
  local pid="$1" child
  for child in $(pgrep -P "${pid}" 2>/dev/null); do
    descendants "${child}"
    echo "${child}"
  done
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
  local children
  children="$(descendants "${pid}")"
  kill -TERM "${pid}" 2>/dev/null || true

  if wait_pid_exit "${pid}" 30; then
    rm -f "${pid_file}"
    echo "    stopped"
    return 0
  fi

  echo "    graceful shutdown timed out; sending SIGKILL"
  kill -KILL "${pid}" 2>/dev/null || true
  wait_pid_exit "${pid}" 10 || true
  # The parent is gone; reap whatever it left behind so the GPUs are released.
  local child
  for child in ${children}; do
    if kill -0 "${child}" 2>/dev/null; then
      echo "    killing orphaned child ${child}"
      kill -KILL "${child}" 2>/dev/null || true
    fi
  done
  rm -f "${pid_file}"
}

stop_service "embedding"
stop_service "qwen-tts"
stop_service "speech"
stop_service "voice-llm"
stop_service "text-llm"
