# Training MMCTL on HiPerGator

SLURM job scripts for running MMCTL training and experiments on UF HiPerGator GPU
nodes. The environment is the same uv setup used locally (`pyproject.toml` /
`uv.lock`), with GPU wheels installed on top.

## 0. Get the code and data onto HiPerGator
```bash
cd /blue/azare/$USER            # your group's blue storage
git clone https://github.com/GatorSense/MMCTL.git
cd MMCTL
# copy the datasets here (they are gitignored), e.g. from Zenodo or your local box:
#   awir_data.npz, neon_data.npz  (and the released embeddings if running evals)
```

## 1. One-time environment setup (on a LOGIN node)
Login nodes have internet; compute nodes do not, so build the env first:
```bash
bash hpg/setup_env.sh
```
This installs `uv`, runs `uv sync` (creating `.venv`), and installs the
GPU TensorFlow wheel. PyTorch from PyPI is already a CUDA build on Linux.

## 2. Submit jobs (from the repo root)
Edit the `#SBATCH` header in each script first: set `--account` / `--qos` to your
allocation (default `azare`) and `--mail-user` to your UFL email.

Train the MMCTL latent space:
```bash
sbatch hpg/train_mmctl.sbatch                 # full (runs=8, iter=10, epochs=100)
sbatch hpg/train_mmctl.sbatch demo            # quick smoke
DATA=/blue/azare/$USER/MMCTL/awir_data.npz sbatch hpg/train_mmctl.sbatch
```

Run any other script on a GPU with the generic runner:
```bash
sbatch hpg/run.sbatch python mmctl/train_taskheads.py --data_npz awir_data.npz --mmctl_dir outputs/mmctl
sbatch hpg/run.sbatch python scripts/freeze_vs_e2e_neon.py
sbatch hpg/run.sbatch python scripts/ablation_loss_components_neon.py
sbatch hpg/run.sbatch python scripts/retune_transformer_neon.py
```

## 3. Monitor
```bash
squeue -u $USER
tail -f hpg/logs/<jobname>_<jobid>.out
```

## Notes
- GPU check on a node: `srun --partition=gpu --gpus=1 --time=10:00 --pty bash`,
  then `uv run --offline python -c "import torch,tensorflow as tf; print(torch.cuda.is_available(), tf.config.list_physical_devices('GPU'))"`.
- `uv run --offline` uses the prebuilt `.venv` without touching the network
  (required on compute nodes). Re-run `hpg/setup_env.sh` on a login node if you
  change dependencies.
- Adjust `--gpus`, `--mem`, `--cpus-per-task`, and `--time` to your workload;
  `--gpus=a100:1` requests an A100 specifically, `--gpus=1` takes any GPU.
- The round-2 experiment scripts default to small/CPU-friendly settings; on a GPU
  node you can raise their epoch counts for the full-scale versions.
