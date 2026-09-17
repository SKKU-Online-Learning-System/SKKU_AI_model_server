#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${COSYVOICE_SOURCE_DIR:-${HOME}/.cache/cosyvoice/CosyVoice}"
SOURCE_REF="${COSYVOICE_SOURCE_REF:-074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc}"

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${SOURCE_DIR}")"
  git clone --recursive https://github.com/QwenAudio/CosyVoice.git "${SOURCE_DIR}"
  git -C "${SOURCE_DIR}" checkout --detach "${SOURCE_REF}"
  git -C "${SOURCE_DIR}" submodule update --init --recursive
elif [[ "$(git -C "${SOURCE_DIR}" rev-parse HEAD)" != "${SOURCE_REF}" ]]; then
  echo "CosyVoice source at ${SOURCE_DIR} is not the pinned revision ${SOURCE_REF}." >&2
  echo "Use another COSYVOICE_SOURCE_DIR or update it explicitly." >&2
  exit 1
fi

uv sync --project "${ROOT_DIR}/cosyvoice_runtime" --python 3.10 --no-dev
