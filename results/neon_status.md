# NEON Status

## Data available
- Local NPZ: `/Users/meilunzhou/Documents/GitHub/CTL/neon_data.npz` (symlinked as `neon_data.npz`)
- Keys: `rgb` (779,110,110,3), `hsi` (779,11,11,426), `lidar` (779,11,11,1), `class_labels`, `box_labels`, `height_info`, …

## Gap
- This repository’s training/eval path (`mmctl/train_mmctl.py`, `demo_metrics.py`, `pretrained_embeddings/`) is **AWIR-specific** (RGB + thermal, 300×300, class/box/position).
- **No NEON pretrained embeddings** are included under `pretrained_embeddings/`.
- NEON uses three modalities (RGB / HSI / LiDAR) and height instead of position; adapting encoders + annotation weighing would be a non-minimal code change.

## Decision
- Documented and deferred for this revision pass.
- Agent B should **not** claim NEON code replication from this repo’s current scripts.
- If NEON numbers must be regenerated, need original CTL/NEON training artifacts or a dedicated multi-encoder port.

## Paper NEON numbers (reference only; not re-run here)
Structure (lower better): MMCTL C −1.49±0.23, B −2.07±0.29, H −1.58±0.25  
Downstream: MMCTL C 86.95±1.7, B R² 0.43±0.04, H R² 0.74±0.02
