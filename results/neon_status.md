# NEON Status

## Data used
- **Authoritative path:** `/Users/meilunzhou/Documents/GitHub/CTL/neon_data.npz`
- Symlink in MMCTL repo: `neon_data.npz` → that file
- Keys: `rgb` (779,110,110,3), `hsi` (779,11,11,426), `lidar` (779,11,11,1), `class_labels`, `box_labels`, `height_info`, …

## What was run (2026-09-20)
- Proxy Torch continuous-triplet on CTL NPZ (`scripts/reproduce_neon_proxy.py`)
- Outputs: `results/replication_neon.json`, `results/replication_neon_delta.md`
- **Absolute match to paper tables: FAIL** (no released NEON Keras embeddings/checkpoints; proxy architecture differs)
- **Keep paper NEON structure/downstream numbers in the manuscript**

## Gap for faithful reproduction
- Repo training path was AWIR-oriented; Zenodo only ships raw NPZs (no NEON `pretrained_embeddings/`)
- Need original Keras autoencoder + MMCTL checkpoints, or a full paper-faithful retrain on GPU

## Next script (better proxy protocol)
- `scripts/reproduce_neon_ae_mmctl.py` — AE-pretrain baselines, then separate MMCTL (so Concat ≠ MMCTL)

## Paper NEON numbers (keep)
Structure MMCTL: C −1.49±0.23, B −2.07±0.29, H −1.58±0.25  
Downstream MMCTL: C 86.95±1.7, B R² 0.43±0.04, H R² 0.74±0.02
