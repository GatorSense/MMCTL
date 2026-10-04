# Loss-component ablation (NEON proxy, advisor #4)

NEON torch-proxy MMCTL (cuda); emb=128, mm_ep=20, runs=3; loss-term ablation on L_mm_adaptive / L_mm_reg.

| Loss | Acc % | Box R² | Height R² | Align-cls (lower=better) |
|---|---|---|---|---|
| full (adaptive + reg) | 67.6$\pm$1.0 | 0.243 | 0.784 | 0.002 |
| adaptive only | 67.2$\pm$2.1 | 0.256 | 0.786 | 0.014 |
| reg only | 62.0$\pm$3.7 | 0.220 | 0.771 | 0.001 |
