#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

: "${HF_HOME:=/models/huggingface}"
: "${TEXT_MODEL:=Qwen/Qwen3.8-27B}"
: "${VOICE_MODEL:=Qwen/Qwen3.5-9B}"
: "${ASR_MODEL:=Qwen/Qwen3-ASR-0.6B}"
: "${TTS_MODEL:=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
export HF_HOME

mkdir -p "${HF_HOME}"

hf=(uvx --from huggingface-hub==1.28.0 hf download)
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

for model in "${TEXT_MODEL}" "${VOICE_MODEL}" "${ASR_MODEL}" "${TTS_MODEL}"; do
  echo "==> Ensuring model is cached: ${model}"
  "${hf[@]}" "${model}"
done

echo "All configured model snapshots are present in ${HF_HOME}."
