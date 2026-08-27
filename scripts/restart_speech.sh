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
LOG_DIR="${MODEL_SERVER_LOG_DIR:-${ROOT_DIR}/logs}"
SPEECH_PORT="${SPEECH_PORT:-8010}"
TIMEOUT_SECONDS="${MODEL_SERVER_STARTUP_TIMEOUT_SECONDS:-900}"
PID_FILE="${RUN_DIR}/speech.pid"
LOG_FILE="${LOG_DIR}/speech.log"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

wait_pid_exit() {
  local pid="$1"
  local seconds="${2:-60}"
  for _ in $(seq 1 "${seconds}"); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

stop_pid() {
  local pid="$1"
  if ! kill -0 "${pid}" 2>/dev/null; then
    return 0
  fi
  kill -TERM "${pid}" 2>/dev/null || true
  if ! wait_pid_exit "${pid}" 60; then
    kill -KILL "${pid}" 2>/dev/null || true
    wait_pid_exit "${pid}" 10 || true
  fi
}

# Stop the process recorded by our lifecycle manager.
if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
  if kill -0 "${pid}" 2>/dev/null; then
    echo "==> Stopping managed speech process ${pid}"
    stop_pid "${pid}"
  fi
  rm -f "${PID_FILE}"
fi

# Clean up a leftover user-owned uvicorn from an older process-group based
# launcher. The speech port is dedicated to this service on this single-user
# development server, so only the exact speech_server.main:app command is
# matched.
mapfile -t leftover_pids < <(
  pgrep -u "$(id -u)" -f "speech_server\.main:app.*--port[ =]${SPEECH_PORT}" 2>/dev/null || true
)
if (( ${#leftover_pids[@]} > 0 )); then
  echo "==> Cleaning up leftover speech process(es): ${leftover_pids[*]}"
  for old_pid in "${leftover_pids[@]}"; do
    stop_pid "${old_pid}"
  done
fi

# A cheap sync keeps the speech venv aligned with the checked-out code without
# touching the already-running Text/Voice LLM processes.
uv sync --project "${ROOT_DIR}/speech_server" --no-dev >/dev/null

echo "==> Starting speech; log: ${LOG_FILE}"
: >"${LOG_FILE}"
# nohup is sufficient for persistence here. start_speech.sh execs uvicorn, so
# $! remains the actual uvicorn PID and can be tracked reliably.
nohup "${ROOT_DIR}/scripts/start_speech.sh" >"${LOG_FILE}" 2>&1 < /dev/null &
pid=$!
echo "${pid}" >"${PID_FILE}"

headers=()
if [[ -n "${MODEL_SERVER_API_KEY:-}" ]]; then
  headers=(-H "Authorization: Bearer ${MODEL_SERVER_API_KEY}")
fi

deadline=$((SECONDS + TIMEOUT_SECONDS))
last_notice=${SECONDS}
echo "==> Waiting for speech readiness (timeout ${TIMEOUT_SECONDS}s)"
while (( SECONDS < deadline )); do
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "Speech exited before becoming ready. Last log lines:" >&2
    tail -n 200 "${LOG_FILE}" >&2 || true
    rm -f "${PID_FILE}"
    exit 1
  fi

  response="$(curl --fail --silent --max-time 5 "${headers[@]}" "http://127.0.0.1:${SPEECH_PORT}/health" 2>/dev/null || true)"
  if grep -Eq '"ready"[[:space:]]*:[[:space:]]*true' <<<"${response}"; then
    echo "    speech: READY (pid ${pid})"
    exit 0
  fi

  if grep -Eq '"status"[[:space:]]*:[[:space:]]*"error"' <<<"${response}"; then
    echo "Speech reported model-loading error: ${response}" >&2
    tail -n 200 "${LOG_FILE}" >&2 || true
    exit 1
  fi

  if (( SECONDS - last_notice >= 30 )); then
    echo "    speech: still loading..."
    last_notice=${SECONDS}
  fi
  sleep 5
done

echo "Speech did not become ready within ${TIMEOUT_SECONDS}s. Last log lines:" >&2
tail -n 200 "${LOG_FILE}" >&2 || true
exit 1
