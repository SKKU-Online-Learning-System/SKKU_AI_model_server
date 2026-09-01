#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/speech_server/.venv/bin/python"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"

[[ -x "${PYTHON_BIN}" ]] || {
  echo "Speech Python environment is missing: ${PYTHON_BIN}" >&2
  exit 1
}
command -v uv >/dev/null || {
  echo "uv is required but was not found in PATH." >&2
  exit 1
}

if "${PYTHON_BIN}" -c 'import flash_attn' >/dev/null 2>&1; then
  "${PYTHON_BIN}" -c 'import flash_attn; print(f"FlashAttention already installed: {flash_attn.__version__}")'
  exit 0
fi

read -r PY_TAG TORCH_SERIES ABI ARCH < <(
  "${PYTHON_BIN}" - <<'PY'
import platform
import sys
import torch

py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
torch_series = ".".join(torch.__version__.split("+")[0].split(".")[:2])
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
print(py_tag, torch_series, abi, platform.machine())
PY
)

echo "==> FlashAttention target: version=${FLASH_ATTN_VERSION} python=${PY_TAG} torch=${TORCH_SERIES} abi=${ABI} arch=${ARCH}"

# Dao-AILab publishes matching Linux wheels for the Speech runtime's Torch 2.8
# and Python 3.12/3.13 combinations. Prefer these so startup does not spend
# minutes compiling CUDA extensions. --no-deps prevents the wheel from changing
# the explicitly pinned PyTorch CUDA runtime.
if [[ "${FLASH_ATTN_VERSION}" == "2.8.3" \
   && "${TORCH_SERIES}" == "2.8" \
   && "${ARCH}" == "x86_64" \
   && ( "${PY_TAG}" == "cp312" || "${PY_TAG}" == "cp313" ) ]]; then
  WHEEL="flash_attn-2.8.3+cu12torch2.8cxx11abi${ABI}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
  URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${WHEEL/+/%2B}"
  echo "==> Installing prebuilt FlashAttention wheel"
  if uv pip install --python "${PYTHON_BIN}" --no-deps "${URL}"; then
    "${PYTHON_BIN}" -c 'import flash_attn; print(f"FlashAttention ready: {flash_attn.__version__}")'
    exit 0
  fi
  echo "Prebuilt FlashAttention wheel install failed; trying source package." >&2
fi

# Fallback for an unexpected Python/ABI combination. The caller treats failure
# as non-fatal because the TTS service can fall back to PyTorch SDPA.
MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-4}" \
uv pip install \
  --python "${PYTHON_BIN}" \
  --no-deps \
  --no-build-isolation \
  "flash-attn==${FLASH_ATTN_VERSION}"

"${PYTHON_BIN}" -c 'import flash_attn; print(f"FlashAttention ready: {flash_attn.__version__}")'
