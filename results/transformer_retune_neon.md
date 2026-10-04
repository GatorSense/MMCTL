# NEON Transformer-fusion re-tuning (advisor #5)

LogReg probe accuracy (%) on Transformer-fused NEON embeddings, 5-fold.

| Config | Acc % |
|---|---|
| default (cls, lr1e-3, 300ep, multitask) | 36.8$\pm$2.4 |
| mean-pool (lr1e-3, 300ep, multitask) | 41.5$\pm$4.0 |
| cls, lr3e-4, 600ep, multitask | 76.1$\pm$1.8 |
| mean-pool, lr3e-4, 600ep, cls-only | 80.7$\pm$1.5 |
