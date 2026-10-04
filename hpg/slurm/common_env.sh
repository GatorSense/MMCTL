#!/usr/bin/env bash
# Sourced by SLURM scripts — not submitted directly.
# Sets modules, activates .venv, loads optional .env.

: "${PROJECT_DIR:=/blue/azare/zhou.m/MMCTL}"
# Which environment to activate: .venv (TF + CPU torch) or .venv_torch (GPU torch).
: "${VENV:=.venv}"

module purge 2>/dev/null || true
module load cuda/12.3.0 2>/dev/null || module load cuda 2>/dev/null || true
module load gcc/12.2.0 2>/dev/null || module load gcc 2>/dev/null || true
# Venv interpreters on HPG are often symlinks to the module python — load it, then
# scrub PYTHONPATH so cluster site-packages do not shadow .venv packages.
module load python/3.10 2>/dev/null || module load python/3.11 2>/dev/null || true

VENV_ACTIVATE="${PROJECT_DIR}/${VENV}/bin/activate"
VENV_PYTHON="${PROJECT_DIR}/${VENV}/bin/python"
if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "ERROR: Virtualenv not found at ${PROJECT_DIR}/${VENV}"
  echo ""
  echo "Create it once on an HPG login node (not inside sbatch):"
  echo "  ssh zhou.m@hpg.rc.ufl.edu"
  echo "  cd ${PROJECT_DIR}"
  echo "  bash hpg/remote/setup_venv.sh"
  exit 1
fi

# Drop any PYTHONPATH from the submission shell (sbatch --export=ALL,PYTHONPATH,...).
unset PYTHONPATH
unset PYTHONHOME
unset PYTHONUSERBASE
unset PYTHONSTARTUP
export PYTHONNOUSERSITE=1

# shellcheck disable=SC1090
source "${VENV_ACTIVATE}"
export PATH="${PROJECT_DIR}/${VENV}/bin:${PATH}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "ERROR: venv python not executable: ${VENV_PYTHON}"
  exit 1
fi

_venv_err="$(
  env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 \
    "${VENV_PYTHON}" -c "
import numpy as np
import numpy._typing  # noqa: F401
import sklearn
from sklearn.linear_model import LogisticRegression
print('numpy', np.__version__, 'sklearn', sklearn.__version__)
" 2>&1
)" || {
  echo "ERROR: venv scientific stack import failed in ${VENV_PYTHON}"
  echo "${_venv_err}"
  echo "Recreate once on a login node: RECREATE_VENV=1 bash hpg/remote/setup_venv.sh"
  exit 1
}

# Optional project env (e.g. COMET_API_KEY for the legacy reconstruction script).
# MMCTL training (mmctl/train_mmctl.py) does not require it.
if [[ -f "${PROJECT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${PROJECT_DIR}/.env"
  set +a
fi

export MMCTL_HPG=1
export PROJECT_PYTHON="${VENV_PYTHON}"

# oneDNN rewrites LayerNormalization into a fused _MklLayerNorm op that is CPU-only
# and crashes GPU training ("No registered '_MklLayerNorm' OpKernel for 'GPU'").
# Disable it so TF uses the standard GPU kernels.
export TF_ENABLE_ONEDNN_OPTS=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
