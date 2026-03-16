#!/bin/bash

set -e

echo "Starting MMCTL training"

# Initialize conda
source ~/anaconda3/etc/profile.d/conda.sh

# Activate environment
conda activate mmctl_env

# Move to repo directory
cd "$(dirname "$0")"

# Default parameters (full training)
RUNS=8
ITERATIONS=10
EPOCHS=100

# Quick demo mode
if [ "$1" == "demo" ]; then
    echo "Running in DEMO MODE"
    RUNS=1
    ITERATIONS=1
    EPOCHS=1
fi

python train_mmctl.py \
    --data_path data.npz \
    --output_dir outputs/mmctl \
    --margin 0.1 \
    --embedding_dimension 1024 \
    --batch_size 32 \
    --test_size 0.5 \
    --cls_weight 0.5 \
    --runs $RUNS \
    --iterations $ITERATIONS \
    --epochs $EPOCHS

echo "Training finished"