#!/bin/bash
#SBATCH --job-name=mmctl_train
#SBATCH --output=/blue/azare/zhou.m/MMCTL/logs/%x_%j.out
#SBATCH --error=/blue/azare/zhou.m/MMCTL/logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --distribution=cyclic:cyclic
#SBATCH --time=72:00:00
#SBATCH --mem=50gb
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhou.m@ufl.edu
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#
# Train the MMCTL latent space on a HiPerGator GPU node.
#   sbatch --export=ALL,DATA=awir_data.npz hpg/slurm/train_mmctl.slurm
#   sbatch --export=ALL,MODE=demo hpg/slurm/train_mmctl.slurm      # quick smoke
set -euo pipefail

PROJECT_DIR="/blue/azare/zhou.m/MMCTL"
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs" outputs

# shellcheck disable=SC1091
source "${PROJECT_DIR}/hpg/slurm/common_env.sh"

DATA="${DATA:-awir_data.npz}"
RUNS=8; ITERATIONS=10; EPOCHS=100
if [[ "${MODE:-}" == "demo" ]]; then
  echo "DEMO MODE"; RUNS=1; ITERATIONS=1; EPOCHS=1
fi

echo "Job ${SLURM_JOB_ID}: train_mmctl data=${DATA}"
echo "Node: $(hostname)"
nvidia-smi || true

python mmctl/train_mmctl.py \
  --data_path "${DATA}" \
  --output_dir outputs/mmctl \
  --margin 0.1 \
  --embedding_dimension 1024 \
  --batch_size 32 \
  --test_size 0.5 \
  --cls_weight 0.5 \
  --runs "${RUNS}" \
  --iterations "${ITERATIONS}" \
  --epochs "${EPOCHS}"

echo "Training finished."
