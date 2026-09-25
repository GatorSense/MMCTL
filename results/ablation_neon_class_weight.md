# NEON class-weight ablation (R1.4)

Released full-set MMCTL embeddings (N=592); class-label weight swept, box/height weights fixed at 1.0.

## Downstream (higher better)

| Class weight | Acc % | Box R² | Height R² |
|---|---|---|---|
| 0.5 | 74.86±1.24 | 0.167±0.043 | 0.663±0.021 |
| 0.8 | 79.32±1.78 | 0.203±0.062 | 0.602±0.022 |
| 1.0 | 80.14±2.18 | 0.283±0.044 | 0.653±0.025 |
| 1.5 | 84.59±1.43 | 0.252±0.076 | 0.619±0.028 |

## Latent structure (alignment ratio + local smoothness, lower better)

| Class weight | Align cls | Align box | Align hgt | Smooth cls | Smooth box | Smooth hgt |
|---|---|---|---|---|---|---|
| 0.5 | 0.700 | 133267.578 | 0.081 | 0.574 | 0.055 | 1.949 |
| 0.8 | 0.728 | 147784.125 | 0.082 | 0.359 | 0.050 | 1.791 |
| 1.0 | 0.834 | 166142.344 | 0.093 | 0.291 | 0.053 | 1.798 |
| 1.5 | 1.018 | 188550.375 | 0.111 | 0.141 | 0.055 | 1.817 |

Figures: `paper_files/figures/neon_ablation_classweight_{boxsize,height}.pdf`.
