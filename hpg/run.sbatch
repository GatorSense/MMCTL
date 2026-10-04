#!/usr/bin/env bash
#SBATCH --job-name=mmctl_run
#SBATCH --output=hpg/logs/%x_%j.out
#SBATCH --error=hpg/logs/%x_%j.err
#SBATCH --account=azare
#SBATCH --qos=azare
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48gb
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=CHANGEME@ufl.edu
#
# Generic GPU runner: everything after the script name is executed in the uv env.
# Examples (submit from the repo root):
#
#   sbatch hpg/run.sbatch python scripts/freeze_vs_e2e_neon.py
#   sbatch hpg/run.sbatch python scripts/ablation_loss_components_neon.py
#   sbatch hpg/run.sbatch python scripts/retune_transformer_neon.py
#   sbatch hpg/run.sbatch python mmctl/smoke_test_train_mmctl.py
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p hpg/logs
export PATH="$HOME/.local/bin:$PATH"
module load cuda/12.4.1 2>/dev/null || true
export TF_FORCE_GPU_ALLOW_GROWTH=true

if [[ $# -eq 0 ]]; then
    echo "usage: sbatch hpg/run.sbatch <command...>   (e.g. python scripts/freeze_vs_e2e_neon.py)" >&2
    exit 1
fi

echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "Command: $*"
uv run --offline "$@"
