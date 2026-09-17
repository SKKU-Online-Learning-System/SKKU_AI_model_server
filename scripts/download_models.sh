#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

: "${HF_HOME:=${HOME}/.cache/huggingface}"
: "${TEXT_MODEL:=Qwen/Qwen3.8-27B}"
: "${VOICE_MODEL:=Qwen/Qwen3.5-9B}"
: "${ASR_MODEL:=Qwen/Qwen3-ASR-0.6B}"
: "${TTS_MODEL:=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
: "${QWEN_TTS_ENABLED:=false}"
: "${QWEN_TTS_MODEL:=Qwen/Qwen3-TTS-12Hz-1.7B-Base}"
: "${COSYVOICE_ENABLED:=false}"
: "${COSYVOICE_MODEL:=FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
: "${COSYVOICE_MODEL_DIR:=${HF_HOME}/Fun-CosyVoice3-0.5B-2512}"
export HF_HOME

mkdir -p "${HF_HOME}"

hf=(uvx --from huggingface-hub==1.28.0 hf download)
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

models=("${TEXT_MODEL}" "${VOICE_MODEL}" "${ASR_MODEL}" "${TTS_MODEL}")
if [[ "${QWEN_TTS_ENABLED}" == "true" ]]; then
  models+=("${QWEN_TTS_MODEL}")
fi
for model in "${models[@]}"; do
  echo "==> Ensuring model is cached: ${model}"
  "${hf[@]}" "${model}"
done

if [[ "${COSYVOICE_ENABLED}" == "true" ]]; then
  echo "==> Ensuring streaming CosyVoice model is cached: ${COSYVOICE_MODEL}"
  "${hf[@]}" "${COSYVOICE_MODEL}" \
    config.json configuration.json cosyvoice3.yaml campplus.onnx \
    flow.pt hift.pt llm.pt speech_tokenizer_v3.onnx \
    CosyVoice-BlankEN/config.json CosyVoice-BlankEN/generation_config.json \
    CosyVoice-BlankEN/merges.txt CosyVoice-BlankEN/model.safetensors \
    CosyVoice-BlankEN/tokenizer_config.json CosyVoice-BlankEN/vocab.json \
    --local-dir "${COSYVOICE_MODEL_DIR}"
fi

echo "All configured model snapshots are present in ${HF_HOME}."
