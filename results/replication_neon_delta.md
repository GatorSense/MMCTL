# NEON Replication Delta (paper reconstruction code)

Data: `/Users/meilunzhou/Documents/GitHub/CTL/neon_data.npz`
Protocol: Paper 3-mod recon (RGB+HSI+LiDAR) + CTL; recon epochs=1000×1, runs=5; task heads Adam lr=0.001 wd=0.0001 epochs=3000 patience=1000 bs=32 clip=5.0 hidden=256; data=/Users/meilunzhou/Documents/GitHub/CTL/neon_data.npz; emb=1024; weights=1.0/1.0/1.0

## Structure (lower better)

| Method | Src | Paper | Code | Δmean |
|---|---|---|---|---|
| RGB | C | -1.66±0.88 | -0.16±0.38 | +1.50 |
| RGB | B | -0.34±0.87 | 1.04±0.56 | +1.38 |
| RGB | H | 0.48±0.97 | 1.63±0.24 | +1.15 |
| HSI | C | 0.72±0.11 | -0.01±0.07 | -0.73 |
| HSI | B | 0.94±0.56 | -0.84±0.29 | -1.78 |
| HSI | H | 1.49±0.23 | -0.22±0.52 | -1.71 |
| LiDAR | C | 2.60±0.08 | 1.92±0.23 | -0.68 |
| LiDAR | B | 1.60±0.72 | 2.39±0.24 | +0.79 |
| LiDAR | H | -0.35±0.17 | 1.96±0.79 | +2.31 |
| Concat | C | 0.39±0.15 | 0.39±0.08 | +0.00 |
| Concat | B | 0.19±0.22 | 0.12±0.29 | -0.07 |
| Concat | H | 0.36±0.05 | -0.19±0.18 | -0.55 |
| DS-M | C | -0.27±0.12 | -2.53±0.07 | -2.26 |
| DS-M | B | 0.24±0.29 | -2.85±0.27 | -3.09 |
| DS-M | H | -0.27±0.23 | -2.98±0.22 | -2.71 |
| MMCTL | C | -1.49±0.23 | 0.39±0.08 | +1.88 |
| MMCTL | B | -2.07±0.29 | 0.12±0.29 | +2.19 |
| MMCTL | H | -1.58±0.25 | -0.19±0.18 | +1.39 |

## Downstream

| Method | Paper C% | Code C% | Paper B | Code B | Paper H | Code H |
|---|---|---|---|---|---|---|
| RGB | 74.59 | 80.27±1.81 | 0.07 | 0.01±0.03 | 0.34 | 0.45±0.04 |
| HSI | 76.69 | 85.91±3.13 | 0.12 | 0.00±0.07 | 0.40 | 0.55±0.03 |
| LiDAR | 49.66 | 44.92±2.18 | 0.22 | 0.00±0.03 | 0.68 | 0.60±0.01 |
| Concat | 81.69 | 85.85±1.43 | 0.39 | 0.08±0.06 | 0.69 | 0.67±0.02 |
| DS-M | 81.35 | 86.71±2.04 | 0.40 | 0.10±0.05 | 0.61 | 0.69±0.01 |
| MMCTL | 86.95 | 85.05±1.88 | 0.43 | 0.10±0.04 | 0.74 | 0.68±0.02 |
