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

stop_service() {
  local name="$1"
  local pid_file="${RUN_DIR}/${name}.pid"

  if [[ ! -f "${pid_file}" ]]; then
    echo "==> ${name}: no pid file"
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}")"

  if ! kill -0 -- "-${pid}" 2>/dev/null; then
    echo "==> ${name}: stale pid file (${pid})"
    rm -f "${pid_file}"
    return 0
  fi

  echo "==> Stopping ${name} (process group ${pid})"
  kill -TERM -- "-${pid}" 2>/dev/null || true

  for _ in $(seq 1 30); do
    if ! kill -0 -- "-${pid}" 2>/dev/null; then
      rm -f "${pid_file}"
      echo "    stopped"
      return 0
    fi
    sleep 1
  done

  echo "    graceful shutdown timed out; sending SIGKILL"
  kill -KILL -- "-${pid}" 2>/dev/null || true
  rm -f "${pid_file}"
}

stop_service "speech"
stop_service "voice-llm"
stop_service "text-llm"
