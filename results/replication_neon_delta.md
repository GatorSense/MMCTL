# NEON Replication Delta (PROXY)

Data: `/Users/meilunzhou/Documents/GitHub/CTL/neon_data.npz`
Protocol: PROXY sampled continuous-triplet MMCTL-style on /Users/meilunzhou/Documents/GitHub/CTL/neon_data.npz; emb_dim=128, epochs=25, runs=5, device=mps. Not bit-identical to paper Keras pipeline. DS-M=mean pool; DS-A not run.

## Structure (lower better)

| Method | Src | Paper | Proxy | Δmean |
|---|---|---|---|---|
| RGB | C | -1.66±0.88 | -2.41±0.48 | -0.75 |
| RGB | B | -0.34±0.87 | -2.10±0.99 | -1.76 |
| RGB | H | 0.48±0.97 | -0.40±0.42 | -0.88 |
| HSI | C | 0.72±0.11 | 0.97±0.54 | +0.25 |
| HSI | B | 0.94±0.56 | 1.23±1.13 | +0.29 |
| HSI | H | 1.49±0.23 | 2.20±0.87 | +0.71 |
| LiDAR | C | 2.60±0.08 | 2.99±0.42 | +0.39 |
| LiDAR | B | 1.60±0.72 | 2.34±0.99 | +0.74 |
| LiDAR | H | -0.35±0.17 | -0.40±0.68 | -0.05 |
| Concat | C | 0.39±0.15 | -0.11±0.29 | -0.50 |
| Concat | B | 0.19±0.22 | -0.03±0.23 | -0.22 |
| Concat | H | 0.36±0.05 | 0.03±0.22 | -0.33 |
| DS-M | C | -0.27±0.12 | -1.33±0.10 | -1.06 |
| DS-M | B | 0.24±0.29 | -1.40±0.59 | -1.64 |
| DS-M | H | -0.27±0.23 | -1.45±0.02 | -1.18 |
| MMCTL | C | -1.49±0.23 | -0.11±0.29 | +1.38 |
| MMCTL | B | -2.07±0.29 | -0.03±0.23 | +2.04 |
| MMCTL | H | -1.58±0.25 | 0.03±0.22 | +1.61 |

## Downstream

| Method | Paper C% | Proxy C% | Paper B | Proxy B | Paper H | Proxy H |
|---|---|---|---|---|---|---|
| RGB | 74.59 | 65.44±2.28 | 0.07 | -70533422973187194880.00±88923906882540765184.00 | 0.34 | 0.27±0.03 |
| HSI | 76.69 | 54.46±3.97 | 0.12 | -85957861155231023104.00±105557038245558042624.00 | 0.40 | 0.27±0.06 |
| LiDAR | 49.66 | 40.92±1.84 | 0.22 | -43312848412881149952.00±55437344207454650368.00 | 0.68 | 0.83±0.03 |
| Concat | 81.69 | 69.59±1.95 | 0.39 | -139339078865500274688.00±176052654041683492864.00 | 0.69 | 0.84±0.02 |
| DS-M | 81.35 | 67.08±2.90 | 0.40 | -95777425592083513344.00±122584162162452709376.00 | 0.61 | 0.85±0.02 |
| MMCTL | 86.95 | 69.59±1.95 | 0.43 | -139339078865500274688.00±176052654041683492864.00 | 0.74 | 0.84±0.02 |

## Notes
- Keep paper NEON absolute tables unless proxy matches closely.
- Use proxy for ranking / feasibility evidence in response-to-reviewers.
