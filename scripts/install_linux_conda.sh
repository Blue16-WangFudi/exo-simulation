#!/usr/bin/env bash
set -e
set -o pipefail

ENV_NAME="${1:-exo-gpu}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PROJECT_ROOT="$(pwd)"
ENV_PATH="${PROJECT_ROOT}/.conda/${ENV_NAME}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda, then re-run." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

mkdir -p "${PROJECT_ROOT}/.conda"
conda create -p "${ENV_PATH}" "python=${PYTHON_VERSION}" -y
conda activate "${ENV_PATH}"
conda install -c nvidia cuda cudnn -y

mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
cat > "${CONDA_PREFIX}/etc/conda/activate.d/exo_cuda.sh" <<'EOF'
export CUDA_HOME="${CONDA_PREFIX}/targets/x86_64-linux"
export CUDA_PATH="${CONDA_PREFIX}/targets/x86_64-linux"
export CUDA_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include"
export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
EOF

# Fix libgcc_s.so if conda ships a linker script instead of ELF
if [ -f "${CONDA_PREFIX}/lib/libgcc_s.so" ]; then
  if ! file "${CONDA_PREFIX}/lib/libgcc_s.so" | grep -q "ELF"; then
    if [ -f "${CONDA_PREFIX}/lib/libgcc_s.so.1" ]; then
      mv "${CONDA_PREFIX}/lib/libgcc_s.so" "${CONDA_PREFIX}/lib/libgcc_s.so.bak"
      ln -s "${CONDA_PREFIX}/lib/libgcc_s.so.1" "${CONDA_PREFIX}/lib/libgcc_s.so"
    fi
  fi
fi

pip install -e .

echo "Done. Activate with: conda activate ${ENV_PATH}"
echo "Run: exo"
