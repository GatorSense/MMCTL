#!/usr/bin/env bash
# One-time environment setup for MMCTL on HiPerGator.
# RUN THIS ON A LOGIN NODE (login nodes have internet; compute nodes do not).
#
#   bash hpg/setup_env.sh
#
# It builds a uv-managed .venv in the repo root. Jobs then use that .venv
# offline via `uv run --offline` or by activating `.venv/bin/activate`.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

# --- uv ---------------------------------------------------------------------
# Install uv locally if it is not already on PATH (static binary, no modules).
if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv to ~/.local/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# --- create the environment from pyproject.toml / uv.lock -------------------
# uv will fetch a compatible CPython automatically.
uv sync

# --- GPU wheels -------------------------------------------------------------
# On Linux x86_64 the default PyTorch wheel is already a CUDA build.
# Replace the CPU TensorFlow wheel with the CUDA-enabled one for GPU nodes.
echo "Installing GPU TensorFlow wheel ..."
uv pip install --upgrade "tensorflow[and-cuda]"

echo
echo "Done. Quick GPU check (run on a GPU node, e.g. via: srun --partition=gpu --gpus=1 --time=10:00 --pty bash):"
echo "  uv run --offline python -c \"import torch,tensorflow as tf; print('torch cuda', torch.cuda.is_available()); print('tf gpus', tf.config.list_physical_devices('GPU'))\""
