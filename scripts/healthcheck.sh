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
: "${COSYVOICE_PORT:=8011}"
: "${COSYVOICE_ENABLED:=false}"
: "${QWEN_TTS_PORT:=8012}"
: "${QWEN_TTS_ENABLED:=false}"

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

if [[ "${QWEN_TTS_ENABLED}" == "true" ]]; then
  check_http "Qwen3-TTS" "http://127.0.0.1:${QWEN_TTS_PORT}/health"
fi
if [[ "${COSYVOICE_ENABLED}" == "true" ]]; then
  check_http "CosyVoice TTS" "http://127.0.0.1:${COSYVOICE_PORT}/health"
fi

if (( failures > 0 )); then
  echo "${failures} service(s) are not healthy." >&2
  exit 1
fi

echo "All services are healthy and ready."
