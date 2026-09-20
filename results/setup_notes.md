# Setup Notes (Agent A)

## Machine
- **Host:** Apple M4 (arm64), macOS 15.7.7
- **CUDA:** unavailable (no NVIDIA)
- **Accelerators used:** PyTorch MPS available (`torch.backends.mps.is_available() == True`); TensorFlow CPU on this Mac

## Environment
- `conda` **not on PATH**; `environment.yml` is Windows/Python-3.8-oriented (includes `pywin32` / MSVC runtimes) and pins TF 2.4.1 — not usable as-is on Apple Silicon.
- Created local venv: **`.venv_exp/`** (Python 3.9.6 from `/usr/bin/python3`)
- Key packages installed via pip:
  - `numpy==1.26.4`, `pandas==2.0.3`, `scikit-learn==1.3.2`, `scipy==1.13.1`, `matplotlib==3.9.4`
  - `torch==2.8.0` (macOS arm64 wheel from PyPI; MPS enabled)
  - `tensorflow==2.20.0` (macOS arm64)
  - `psutil`, `tqdm`, `pyyaml`
- Activate: `source .venv_exp/bin/activate` and set `PYTHONPATH=<repo root>` when running scripts under `scripts/`.

## Data paths
- Symlinks in repo root:
  - `awir_data.npz` → `/Users/meilunzhou/Documents/GitHub/CTL/awir_data.npz` (N=240)
  - `neon_data.npz` → `/Users/meilunzhou/Documents/GitHub/CTL/neon_data.npz` (N=779; RGB/HSI/LiDAR)
  - `data.npz` → `sample_data.npz` (N=24 mini set for smoke/timing)
- Pretrained AWIR embeddings: `pretrained_embeddings/run{01..08}_{rgb,thermal,mmctl,concat,deepset_*}_test.npy` (**test-only**, N=120)

## Smoke / verification
- `demo_metrics.py --data .../awir_data.npz` **succeeded** (structure table matches paper; see `results/replication_awir_delta.md`).
- TensorFlow import **segfaults inside Cursor sandbox**; works with full/unsandboxed permissions — run TF jobs outside sandbox.
- Full MMCTL retrain + Keras checkpoint extraction **not** required for structure replication (pretrained embeddings suffice).

## Blockers / tradeoffs
1. **No train embeddings / `.h5` checkpoints** shipped → cannot byte-reproduce paper downstream protocol (`train_taskheads.py` expects per-run Keras checkpoints). Proxy: nested CV on released test embeddings (`scripts/eval_pretrained_taskheads.py`).
2. **NEON:** no pretrained NEON embeddings in this repo; training scripts are AWIR-shaped (rgb+thermal). See `results/neon_status.md`.
3. **Full AWIR retrain** (100 epochs × 10 iterations × 8 runs) is too expensive for interactive M4 work; compute cost measured via microbenchmark + theoretical triplet counts (`results/compute_cost.json`).
4. Prefer CTL local NPZ over Zenodo downloads (already present).
