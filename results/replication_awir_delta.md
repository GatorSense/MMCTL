# AWIR Replication Delta (Structure)

Source: `demo_metrics.py --data awir_data.npz` vs paper Table `tab:latent_alignment`.

Lower is better. Large-delta flag if |Δmean|>0.15 or |Δstd|>0.15.

**Gate verdict: PASS** (0 flagged cells).

| Method | Src | Paper | Code | Δmean | Status |
|---|---|---|---|---|---|
| RGB | C | 0.43±0.28 | 0.43±0.28 | +0.00 | ok |
| RGB | B | -0.41±0.36 | -0.42±0.36 | -0.01 | ok |
| RGB | P | 0.23±0.38 | 0.23±0.38 | -0.00 | ok |
| Thermal | C | 0.54±0.15 | 0.54±0.15 | +0.00 | ok |
| Thermal | B | 0.56±0.14 | 0.63±0.15 | +0.07 | ok |
| Thermal | P | 0.89±0.49 | 0.89±0.49 | +0.00 | ok |
| Concat | C | -0.28±0.17 | -0.28±0.17 | -0.00 | ok |
| Concat | B | -0.19±0.15 | -0.14±0.15 | +0.05 | ok |
| Concat | P | 0.31±0.27 | 0.31±0.27 | -0.00 | ok |
| DS-A | C | 1.17±0.44 | 1.17±0.44 | +0.00 | ok |
| DS-A | B | 1.51±0.44 | 1.52±0.46 | +0.01 | ok |
| DS-A | P | 1.19±0.43 | 1.19±0.43 | +0.00 | ok |
| DS-M | C | 1.23±0.11 | 1.23±0.11 | +0.00 | ok |
| DS-M | B | 1.47±0.46 | 1.47±0.40 | +0.00 | ok |
| DS-M | P | 1.17±0.46 | 1.17±0.46 | +0.00 | ok |
| MMCTL | C | -3.10±0.04 | -3.10±0.04 | +0.00 | ok |
| MMCTL | B | -2.93±0.18 | -3.06±0.15 | -0.13 | ok |
| MMCTL | P | -3.79±0.61 | -3.79±0.61 | +0.00 | ok |

## Notes

- Thermal **B**: paper 0.56±0.14 vs code ~0.63±0.15 (small drift).
- Concat **B**: paper -0.19±0.15 vs code ~-0.14±0.15 (small drift).
- MMCTL **B**: paper -2.93±0.18 vs code ~-3.06±0.15 (small drift; MMCTL still dominates).
- All other cells match paper to printed precision (≤0.01–0.02).
- Downstream task-head numbers: see later section once proxy eval finishes; train embeddings/checkpoints not shipped.
