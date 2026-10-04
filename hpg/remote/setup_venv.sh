#!/usr/bin/env bash
# Run ONCE on a HiPerGator LOGIN node after first code push:
#   bash /blue/azare/zhou.m/MMCTL/hpg/remote/setup_venv.sh
# Login nodes have internet; compute nodes do not.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/blue/azare/zhou.m/MMCTL}"
cd "${PROJECT_DIR}"

echo "=== MMCTL HPG venv setup in ${PROJECT_DIR} ==="

if command -v module &>/dev/null; then
  module purge 2>/dev/null || true
  module load cuda/12.3.0 2>/dev/null || module load cuda 2>/dev/null || true
  module load gcc/12.2.0 2>/dev/null || module load gcc 2>/dev/null || true
  module load python/3.10 2>/dev/null || module load python/3.11 2>/dev/null || true
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" &>/dev/null; then
  echo "ERROR: ${PYTHON_BIN} not found. Set PYTHON_BIN or load a python module."
  exit 1
fi

if [[ "${RECREATE_VENV:-0}" == "1" && -d .venv ]]; then
  echo "Removing existing .venv (RECREATE_VENV=1) ..."
  rm -rf .venv
fi

"${PYTHON_BIN}" -m venv --copies .venv
# shellcheck disable=SC1091
source .venv/bin/activate
unset PYTHONPATH
unset PYTHONHOME
export PYTHONNOUSERSITE=1

pip install --upgrade pip
# Pin the scientific stack first so the resolver does not pull incompatible versions.
pip install --no-cache-dir 'numpy==1.26.4' 'scipy==1.11.4' 'scikit-learn==1.4.2'
pip install --no-cache-dir -r requirements.txt

# HPG GPU: requirements.txt pins plain `tensorflow` (CPU). Reinstall with the bundled
# CUDA runtime + cuDNN so jobs use the GPU instead of silently falling back to CPU.
TF_VERSION="${TF_VERSION:-2.15.1}"
pip install --no-cache-dir "tensorflow[and-cuda]==${TF_VERSION}" "numpy==1.26.4"

python -c "
import numpy as np
import numpy._typing  # noqa: F401
import sklearn, torch
print('numpy', np.__version__, 'scikit-learn', sklearn.__version__, 'torch', torch.__version__, 'OK')
print('torch cuda', torch.cuda.is_available())
"
python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPUs', tf.config.list_physical_devices('GPU'))" || true

echo "=== Done. Activate with: source ${PROJECT_DIR}/.venv/bin/activate ==="
