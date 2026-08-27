#!/usr/bin/env bash
set -euo pipefail
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

check() {
  local name="$1"
  local url="$2"
  echo "==> ${name}: ${url}"
  curl --fail --silent --show-error "${headers[@]}" "${url}"
  echo
}

check "Text LLM" "http://127.0.0.1:${TEXT_PORT}/v1/models"
check "Voice LLM" "http://127.0.0.1:${VOICE_PORT}/v1/models"
check "Speech" "http://127.0.0.1:${SPEECH_PORT}/health"
