#!/usr/bin/env bash
set -euo pipefail

NODE_PORT="${1:-50051}"
API_PORT="${2:-52415}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export EXO_HOME="${EXO_HOME:-/root/.cache/exo}"
EXO_UID="${EXO_UID:-1000}"
EXO_GID="${EXO_GID:-1000}"

# Prefer WSL's CUDA driver libs when available to avoid the stub libcuda.
WSL_CUDA_DIR="$(find /usr/lib/wsl/drivers -name 'libcuda.so.1.1' -printf '%h' -quit 2>/dev/null || true)"
if [[ -n "${WSL_CUDA_DIR}" ]]; then
  if [[ ! -e "${WSL_CUDA_DIR}/libcuda.so.1" ]]; then
    ln -sf "${WSL_CUDA_DIR}/libcuda.so.1.1" "${WSL_CUDA_DIR}/libcuda.so.1"
  fi
  export LD_LIBRARY_PATH="${WSL_CUDA_DIR}:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "${EXO_HOME}/downloads"

if ! command -v exo >/dev/null 2>&1; then
  echo "exo not found in PATH. Check conda env PATH."
  exit 1
fi

# Try to align cache ownership with host user to avoid permission issues.
if [[ -n "${EXO_UID}" && -n "${EXO_GID}" ]]; then
  chown -R "${EXO_UID}:${EXO_GID}" "${EXO_HOME}" || true
fi

cd /home/blue16/dev_workspace/exo-simulation
exec exo --node-port "${NODE_PORT}" --chatgpt-api-port "${API_PORT}"
