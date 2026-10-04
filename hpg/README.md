# Training MMCTL on HiPerGator

SLURM setup for UF HiPerGator, mirroring the CTL / CVPR2027 workflow: a project
`.venv` built once on a login node, a shared `common_env.sh`, and per-task
`.slurm` files submitted with `sbatch`.

Layout:
```
hpg/
  env.sh                  # HPG host/dir vars (PROJECT_DIR=/blue/azare/zhou.m/MMCTL)
  sync_push.sh            # rsync code → HPG (+ git push); excludes *.npz
  sync_exclude.txt
  remote/setup_venv.sh    # run ONCE on a login node to build .venv (pip + GPU TF)
  slurm/
    sbatch_header.sh      # shared #SBATCH defaults (reference)
    common_env.sh         # modules + .venv activation (sourced by jobs)
    train_mmctl.slurm     # train the MMCTL latent space
    run.slurm             # generic GPU runner (CMD="...")
```

## 1. Push code to HiPerGator
From your Mac (repo root), after committing:
```bash
bash hpg/sync_push.sh           # rsync to /blue/azare/$USER/MMCTL + git push
```
Datasets (`*.npz`) are excluded from the sync — copy them once:
```bash
scp awir_data.npz neon_data.npz zhou.m@hpg.rc.ufl.edu:/blue/azare/zhou.m/MMCTL/
```
(Or `git clone` the repo directly under `/blue/azare/$USER/`.)

## 2. Build the environment (ONCE, on a LOGIN node)
Login nodes have internet; compute nodes do not.
```bash
ssh zhou.m@hpg.rc.ufl.edu
cd /blue/azare/zhou.m/MMCTL
bash hpg/remote/setup_venv.sh          # builds .venv (TF GPU + CPU torch) and .venv_torch (GPU torch)
```
Recreate after dependency changes: `RECREATE_VENV=1 bash hpg/remote/setup_venv.sh`.

## 3. Submit jobs (from `/blue/azare/zhou.m/MMCTL`)
```bash
# Train the MMCTL latent space
sbatch --export=ALL,DATA=awir_data.npz hpg/slurm/train_mmctl.slurm
sbatch --export=ALL,MODE=demo          hpg/slurm/train_mmctl.slurm   # quick smoke

# Anything else, on a GPU, via the generic runner
# TF scripts use the default .venv:
sbatch --export=ALL,CMD="python mmctl/train_taskheads.py --data_npz awir_data.npz --mmctl_dir outputs/mmctl" hpg/slurm/run.slurm
sbatch --export=ALL,CMD="python scripts/freeze_vs_e2e_neon.py"            hpg/slurm/run.slurm
# torch-only experiments use the GPU-torch env:
sbatch --export=ALL,VENV=.venv_torch,CMD="python scripts/ablation_loss_components_neon.py" hpg/slurm/run.slurm
sbatch --export=ALL,VENV=.venv_torch,CMD="python scripts/retune_transformer_neon.py"       hpg/slurm/run.slurm
```

## 4. Monitor & pull results
```bash
squeue -u $USER
tail -f /blue/azare/zhou.m/MMCTL/logs/<jobname>_<jobid>.out
```

When jobs finish, pull the artifacts back to your local repo (run on your Mac):
```bash
bash hpg/sync_pull.sh results     # results/*.json,*.md (metrics, ablations)
bash hpg/sync_pull.sh outputs     # outputs/ (checkpoints, embeddings)
bash hpg/sync_pull.sh all         # both (default)
bash hpg/sync_pull.sh results --dry-run   # preview without copying
```

## Notes
- Partition `hpg-turin`, `--gpus=1`, 12 CPUs, 50 GB, 72 h — adjust in the `.slurm`
  headers (and `--mail-user`) for your job.
- `common_env.sh` sets `TF_ENABLE_ONEDNN_OPTS=0` (avoids the `_MklLayerNorm` GPU
  crash) and `TF_FORCE_GPU_ALLOW_GROWTH=true`.
- Two pip venvs avoid the TF/torch CUDA-wheel clash: `.venv` has
  `tensorflow[and-cuda]==2.15.1` (GPU) + CPU-only torch; `.venv_torch` has GPU
  torch only. Select with `VENV=.venv_torch` on torch-only jobs. Local dev uses
  uv (`pyproject.toml`).
- GPU sanity check on a node:
  `srun --partition=hpg-turin --gpus=1 --time=10:00 --pty bash`, then
  `source /blue/azare/zhou.m/MMCTL/hpg/slurm/common_env.sh && python -c "import torch,tensorflow as tf; print(torch.cuda.is_available(), tf.config.list_physical_devices('GPU'))"`.
