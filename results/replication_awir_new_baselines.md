# AWIR Replication (released full-set embeddings)

N=240; files: `awir_rgb_all.npy`, `awir_thermal_all.npy`, `awir_mmctl_all.npy`
Protocol: Released full-set AWIR embeddings (N=240); RGB/Thermal/MMCTL; baselines Concat/SupCon/CrossAttn/Transformer; 50/50 + 8-fold; paper task heads 3000/1000

## Structure (lower better)

| Method | Src | Paper | Code | Δmean |
|---|---|---|---|---|
| RGB | C | 0.43±0.28 | 1.85±0.16 | +1.42 |
| RGB | B | -0.41±0.36 | -0.77±0.11 | -0.36 |
| RGB | P | 0.23±0.38 | -0.07±0.15 | -0.30 |
| Thermal | C | 0.54±0.15 | 2.02±0.22 | +1.48 |
| Thermal | B | 0.56±0.14 | 1.50±0.10 | +0.94 |
| Thermal | P | 0.89±0.49 | 2.40±0.24 | +1.51 |
| Concat | C | -0.28±0.17 | -0.10±0.17 | +0.18 |
| Concat | B | -0.19±0.15 | 0.05±0.07 | +0.24 |
| Concat | P | 0.31±0.27 | 1.02±0.18 | +0.71 |
| SupCon | C | — | 0.17±0.40 | — |
| SupCon | B | — | 0.87±0.27 | — |
| SupCon | P | — | 0.38±0.26 | — |
| CrossAttn | C | — | -0.31±0.71 | — |
| CrossAttn | B | — | 0.04±0.50 | — |
| CrossAttn | P | — | 0.03±0.58 | — |
| Transformer | C | — | -0.47±0.53 | — |
| Transformer | B | — | 0.04±0.37 | — |
| Transformer | P | — | -0.28±0.34 | — |
| MMCTL | C | -3.10±0.04 | -3.15±0.10 | -0.05 |
| MMCTL | B | -2.93±0.18 | -1.73±0.15 | +1.20 |
| MMCTL | P | -3.79±0.61 | -3.50±0.11 | +0.29 |

## Downstream

| Method | Paper C% | Code C% | Paper B | Code B | Paper P | Code P |
|---|---|---|---|---|---|---|
| RGB | 77.08 | 79.48±1.50 | 0.34 | 0.41±0.05 | -0.13 | 0.23±0.08 |
| Thermal | 78.96 | 76.98±3.14 | 0.30 | 0.29±0.03 | -0.32 | 0.01±0.04 |
| Concat | 89.06 | 85.21±2.16 | 0.32 | 0.45±0.02 | -0.15 | 0.27±0.02 |
| SupCon | — | 89.06±1.47 | — | -0.15±0.25 | — | -0.37±0.18 |
| CrossAttn | — | 86.67±1.38 | — | 0.31±0.06 | — | 0.06±0.08 |
| Transformer | — | 86.25±1.50 | — | 0.04±0.12 | — | -0.16±0.13 |
| MMCTL | 94.06 | 91.46±1.49 | 0.52 | 0.62±0.02 | 0.12 | 0.50±0.05 |
