#!/bin/bash
#SBATCH --job-name=mmctl_run
#SBATCH --output=/blue/azare/zhou.m/MMCTL/logs/%x_%j.out
#SBATCH --error=/blue/azare/zhou.m/MMCTL/logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --distribution=cyclic:cyclic
#SBATCH --time=24:00:00
#SBATCH --mem=50gb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhou.m@ufl.edu
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#
# Generic GPU runner: pass the command via the CMD env var.
#   sbatch --export=ALL,CMD="python scripts/freeze_vs_e2e_neon.py"       hpg/slurm/run.slurm
#   sbatch --export=ALL,CMD="python scripts/ablation_loss_components_neon.py" hpg/slurm/run.slurm
#   sbatch --export=ALL,CMD="python scripts/retune_transformer_neon.py"  hpg/slurm/run.slurm
#   sbatch --export=ALL,CMD="python mmctl/smoke_test_train_mmctl.py"      hpg/slurm/run.slurm
set -euo pipefail

PROJECT_DIR="/blue/azare/zhou.m/MMCTL"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/hpg/slurm/common_env.sh"

if [[ -z "${CMD:-}" ]]; then
  echo "ERROR: set CMD, e.g. --export=ALL,CMD=\"python scripts/freeze_vs_e2e_neon.py\"" >&2
  exit 1
fi

echo "Job ${SLURM_JOB_ID}: ${CMD}"
echo "Node: $(hostname)"
nvidia-smi || true

eval "${CMD}"

echo "Finished."
