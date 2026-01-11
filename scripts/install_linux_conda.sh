#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-exo-gpu}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda, then re-run." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
conda activate "${ENV_NAME}"
conda install -c nvidia cuda cudnn -y

mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
cat > "${CONDA_PREFIX}/etc/conda/activate.d/exo_cuda.sh" <<'EOF'
export CUDA_HOME="${CONDA_PREFIX}/targets/x86_64-linux"
export CUDA_PATH="${CONDA_PREFIX}/targets/x86_64-linux"
export CUDA_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include"
export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
EOF

pip install -e .

echo "Done. Activate with: conda activate ${ENV_NAME}"
echo "Run: exo"
