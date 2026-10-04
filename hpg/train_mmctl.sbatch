#!/usr/bin/env bash
#SBATCH --job-name=mmctl_train
#SBATCH --output=hpg/logs/%x_%j.out
#SBATCH --error=hpg/logs/%x_%j.err
#SBATCH --account=azare          # <-- your SLURM group/allocation
#SBATCH --qos=azare              # <-- or azare-b for burst QOS
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1            # 1 A100 (use "--gpus=1" for any GPU)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=CHANGEME@ufl.edu
#
# Train the MMCTL latent space on a HiPerGator GPU node.
# Submit from the repo root (after running hpg/setup_env.sh on a login node):
#
#   sbatch hpg/train_mmctl.sbatch                # full training
#   sbatch hpg/train_mmctl.sbatch demo           # 1-run smoke
#   DATA=/blue/azare/$USER/MMCTL/awir_data.npz sbatch hpg/train_mmctl.sbatch
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p hpg/logs outputs
export PATH="$HOME/.local/bin:$PATH"

module load cuda/12.4.1 2>/dev/null || true   # match the toolkit TF/torch expect
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Dataset (override with: DATA=/path/to/awir_data.npz sbatch ...)
DATA="${DATA:-awir_data.npz}"

# Full vs. demo schedule
RUNS=8; ITERATIONS=10; EPOCHS=100
if [[ "${1:-}" == "demo" ]]; then
    echo "DEMO MODE"; RUNS=1; ITERATIONS=1; EPOCHS=1
fi

echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}  Data: $DATA"
uv run --offline python -c "import torch; print('torch cuda', torch.cuda.is_available())"

uv run --offline python mmctl/train_mmctl.py \
    --data_path "$DATA" \
    --output_dir outputs/mmctl \
    --margin 0.1 \
    --embedding_dimension 1024 \
    --batch_size 32 \
    --test_size 0.5 \
    --cls_weight 0.5 \
    --runs "$RUNS" \
    --iterations "$ITERATIONS" \
    --epochs "$EPOCHS"

echo "Training finished."
