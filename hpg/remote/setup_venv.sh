#!/usr/bin/env bash
# Run ONCE on a HiPerGator LOGIN node after first code push:
#   bash /blue/azare/zhou.m/MMCTL/hpg/remote/setup_venv.sh
# Login nodes have internet; compute nodes do not.
#
# TensorFlow and PyTorch GPU pip wheels ship conflicting CUDA runtimes (TF 2.15 ->
# cu12.2 / nccl 2.16; torch 2.2 -> cu12.1 / nccl 2.19), which breaks torch in a
# shared env ("libtorch_cuda.so: undefined symbol: ncclCommRegister"). We therefore
# build TWO venvs:
#   .venv        TensorFlow[and-cuda] (GPU) + torch CPU-only  -> TF scripts
#                (train_mmctl, train_taskheads, eval_*; torch heads are tiny/CPU)
#   .venv_torch  torch (GPU) only                             -> torch experiments
#                (scripts/ablation_loss_components_neon.py, retune_transformer_neon.py)
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
command -v "${PYTHON_BIN}" &>/dev/null || { echo "ERROR: ${PYTHON_BIN} not found"; exit 1; }

TF_VERSION="${TF_VERSION:-2.15.1}"
TORCH_VERSION="${TORCH_VERSION:-2.2.2}"

build_base () {  # $1 = venv dir
  local v="$1"
  "${PYTHON_BIN}" -m venv --copies "$v"
  # shellcheck disable=SC1091
  source "$v/bin/activate"
  unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1
  pip install --upgrade pip
  pip install --no-cache-dir 'numpy==1.26.4' 'scipy==1.11.4' 'scikit-learn==1.4.2'
  pip install --no-cache-dir -r requirements.txt
  deactivate
}

# ---------- .venv : TensorFlow GPU + torch CPU (no CUDA wheels -> no clash) -----
if [[ "${RECREATE_VENV:-0}" == "1" && -d .venv ]]; then rm -rf .venv; fi
echo "--- building .venv (TF GPU + torch CPU) ---"
build_base .venv
source .venv/bin/activate
unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1
pip install --no-cache-dir "tensorflow[and-cuda]==${TF_VERSION}" "numpy==1.26.4"
# CPU-only torch: the CPU wheel pulls NO nvidia-* packages, so it cannot clobber
# TensorFlow's CUDA runtime. (Torch here only runs small task heads.)
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}"
python -c "import tensorflow as tf, torch; print('TF', tf.__version__, 'GPUs', tf.config.list_physical_devices('GPU')); print('torch', torch.__version__, 'cpu-only')"
deactivate

# ---------- .venv_torch : torch GPU only ---------------------------------------
if [[ "${RECREATE_VENV:-0}" == "1" && -d .venv_torch ]]; then rm -rf .venv_torch; fi
echo "--- building .venv_torch (torch GPU) ---"
build_base .venv_torch
source .venv_torch/bin/activate
unset PYTHONPATH PYTHONHOME; export PYTHONNOUSERSITE=1
pip install --no-cache-dir "torch==${TORCH_VERSION}"   # default PyPI = CUDA build on Linux
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
deactivate

echo "=== Done. TF jobs use .venv; torch experiments use VENV=.venv_torch ==="
