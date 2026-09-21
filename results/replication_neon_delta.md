# NEON Replication (released embeddings, MMCTL w_cls=1.0)

N=592; MMCTL file: `neon_mmctl_all_6_1.0.npy`
Protocol: Released full-set NEON embeddings (N=592); MMCTL=neon_mmctl_all_6_1.0.npy; independent RGB/HSI/LiDAR; baselines Concat/SupCon/CrossAttn/Transformer; 50/50 + 5-fold; paper task heads 3000/1000

## Structure (lower better)

| Method | Src | Paper | Code | Δmean |
|---|---|---|---|---|
| RGB | C | -1.66±0.88 | -0.23±0.03 | +1.43 |
| RGB | B | -0.34±0.87 | 0.99±0.08 | +1.33 |
| RGB | H | 0.48±0.97 | 1.49±0.17 | +1.01 |
| HSI | C | 0.72±0.11 | 0.63±0.01 | -0.09 |
| HSI | B | 0.94±0.56 | 1.11±0.12 | +0.17 |
| HSI | H | 1.49±0.23 | 1.06±0.11 | -0.43 |
| LiDAR | C | 2.60±0.08 | 2.45±0.06 | -0.15 |
| LiDAR | B | 1.60±0.72 | 2.43±0.05 | +0.83 |
| LiDAR | H | -0.35±0.17 | -0.47±0.07 | -0.12 |
| Concat | C | 0.39±0.15 | 0.29±0.01 | -0.10 |
| Concat | B | 0.19±0.22 | 0.53±0.16 | +0.34 |
| Concat | H | 0.36±0.05 | -0.05±0.07 | -0.41 |
| SupCon | C | — | -0.22±0.03 | — |
| SupCon | B | — | 0.73±0.60 | — |
| SupCon | H | — | 1.17±0.19 | — |
| CrossAttn | C | — | -0.96±0.29 | — |
| CrossAttn | B | — | -2.41±0.43 | — |
| CrossAttn | H | — | -1.35±0.23 | — |
| Transformer | C | — | -0.46±0.26 | — |
| Transformer | B | — | -1.23±0.28 | — |
| Transformer | H | — | -0.35±0.23 | — |
| MMCTL | C | -1.49±0.23 | -1.51±0.08 | -0.02 |
| MMCTL | B | -2.07±0.29 | -2.17±0.27 | -0.10 |
| MMCTL | H | -1.58±0.25 | -1.50±0.10 | +0.08 |

## Downstream

| Method | Paper C% | Code C% | Paper B | Code B | Paper H | Code H |
|---|---|---|---|---|---|---|
| RGB | 74.59 | 76.22±2.33 | 0.07 | 0.07±0.09 | 0.34 | 0.32±0.03 |
| HSI | 76.69 | 76.22±1.38 | 0.12 | 0.13±0.06 | 0.40 | 0.40±0.03 |
| LiDAR | 49.66 | 48.38±2.09 | 0.22 | 0.15±0.08 | 0.68 | 0.67±0.02 |
| Concat | 81.69 | 80.81±1.96 | 0.39 | 0.36±0.08 | 0.69 | 0.71±0.01 |
| SupCon | — | 81.42±1.97 | — | 0.07±0.14 | — | 0.47±0.06 |
| CrossAttn | — | 68.85±2.17 | — | 0.27±0.10 | — | 0.60±0.03 |
| Transformer | — | 43.11±4.50 | — | 0.24±0.13 | — | 0.29±0.06 |
| MMCTL | 86.95 | 80.34±1.03 | 0.43 | 0.33±0.05 | 0.74 | 0.63±0.03 |
