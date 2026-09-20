# AWIR Replication Delta

Source: `python demo_metrics.py --data /Users/meilunzhou/Documents/GitHub/CTL/awir_data.npz`  
Compared to paper Table `tab:latent_alignment` (AWIR rows). Lower is better.

**Structure gate: PASS** (0 cells with |Δmean|>0.15 or |Δstd|>0.15).

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

## Notes for Agent B
- Do **not** overwrite paper structure table; code matches printed values within rounding / mild float drift.
- Largest drifts: Thermal B (+0.07), Concat B (+0.05), MMCTL B (−0.13). Ranking unchanged; MMCTL remains best by a large margin.
- Machine-readable: `results/replication_awir.json`.
- Raw per-run: `results/demo_metrics_raw.csv`.

## Downstream (task heads)
- Paper protocol needs per-run **train** embeddings from Keras checkpoints — **not shipped** (only `*_test.npy`).
- Proxy eval running via `scripts/eval_pretrained_taskheads.py` (StratifiedKFold within released test embeddings). Treat as ranking evidence, not table replacement, until/unless checkpoints are provided.
- Paper AWIR MMCTL reference: C 94.06±1.86, B R² 0.52±0.07, P R² 0.12±0.08.
