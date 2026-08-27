#!/usr/bin/env bash
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

: "${TEXT_PORT:=8001}"
: "${VOICE_PORT:=8002}"
: "${SPEECH_PORT:=8010}"

headers=()
if [[ -n "${MODEL_SERVER_API_KEY:-}" ]]; then
  headers=(-H "Authorization: Bearer ${MODEL_SERVER_API_KEY}")
fi

failures=0
check_http() {
  local name="$1"
  local url="$2"
  echo "==> ${name}: ${url}"
  if curl --fail --silent --show-error --max-time 5 "${headers[@]}" "${url}"; then
    echo
  else
    echo "[UNHEALTHY] ${name}" >&2
    failures=$((failures + 1))
  fi
}

check_speech() {
  local url="http://127.0.0.1:${SPEECH_PORT}/health"
  local response
  echo "==> Speech: ${url}"
  if ! response="$(curl --fail --silent --show-error --max-time 5 "${headers[@]}" "${url}")"; then
    echo "[UNHEALTHY] Speech" >&2
    failures=$((failures + 1))
    return
  fi
  echo "${response}"
  if ! grep -Eq '"ready"[[:space:]]*:[[:space:]]*true' <<<"${response}"; then
    echo "[UNHEALTHY] Speech is reachable but models are not ready" >&2
    failures=$((failures + 1))
  fi
}

check_http "Text LLM" "http://127.0.0.1:${TEXT_PORT}/v1/models"
check_http "Voice LLM" "http://127.0.0.1:${VOICE_PORT}/v1/models"
check_speech

if (( failures > 0 )); then
  echo "${failures} service(s) are not healthy." >&2
  exit 1
fi

echo "All services are healthy and ready."
