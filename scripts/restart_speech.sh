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

if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
  if kill -0 -- "-${pid}" 2>/dev/null; then
    echo "==> Stopping speech (process group ${pid})"
    kill -TERM -- "-${pid}" 2>/dev/null || true
    for _ in $(seq 1 60); do
      if ! kill -0 -- "-${pid}" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 -- "-${pid}" 2>/dev/null; then
      echo "    graceful shutdown timed out; sending SIGKILL"
      kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
  fi
  rm -f "${PID_FILE}"
fi

echo "==> Starting speech; log: ${LOG_FILE}"
: >"${LOG_FILE}"
nohup setsid "${ROOT_DIR}/scripts/start_speech.sh" >"${LOG_FILE}" 2>&1 < /dev/null &
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
  if ! kill -0 -- "-${pid}" 2>/dev/null; then
    echo "Speech exited before becoming ready. Last log lines:" >&2
    tail -n 160 "${LOG_FILE}" >&2 || true
    rm -f "${PID_FILE}"
    exit 1
  fi

  response="$(curl --fail --silent --max-time 5 "${headers[@]}" "http://127.0.0.1:${SPEECH_PORT}/health" 2>/dev/null || true)"
  if grep -Eq '"ready"[[:space:]]*:[[:space:]]*true' <<<"${response}"; then
    echo "    speech: READY"
    exit 0
  fi

  if (( SECONDS - last_notice >= 30 )); then
    echo "    speech: still loading..."
    last_notice=${SECONDS}
  fi
  sleep 5
done

echo "Speech did not become ready within ${TIMEOUT_SECONDS}s. Last log lines:" >&2
tail -n 160 "${LOG_FILE}" >&2 || true
exit 1
