# Per-class within-class continuous-transition metric

within-class Spearman rho between pairwise cosine embedding distance and pairwise |delta continuous label|; higher=smoother

## AWIR

### box area

| Class | n | Concat | SupCon | CrossAttn | Transformer | MMCTL | Best |
|---|---|---|---|---|---|---|---|
| All (dataset-wide) | 240 | 0.233 | 0.209 | 0.188 | 0.238 | **0.272** | MMCTL |
| cow | 116 | **0.289** | 0.201 | 0.190 | 0.254 | 0.212 | Concat |
| deer | 51 | 0.040 | **0.124** | 0.115 | 0.048 | 0.105 | SupCon |
| horse | 73 | **0.584** | 0.078 | 0.437 | 0.575 | 0.361 | Concat |

## NEON

### box area

| Class | n | Concat | SupCon | CrossAttn | Transformer | MMCTL | Best |
|---|---|---|---|---|---|---|---|
| All (dataset-wide) | 592 | 0.037 | 0.011 | 0.236 | **0.533** | 0.202 | Transformer |
| ABLA | 30 | 0.092 | -0.015 | 0.196 | 0.247 | **0.432** | MMCTL |
| ACRU | 29 | -0.006 | 0.033 | **0.054** | 0.053 | 0.050 | CrossAttn |
| BENE4 | 36 | 0.169 | -0.089 | **0.299** | 0.235 | 0.173 | CrossAttn |
| PIEN | 36 | **0.236** | -0.010 | 0.188 | 0.145 | 0.057 | Concat |
| PIMA | 42 | 0.252 | 0.281 | 0.363 | **0.384** | 0.291 | Transformer |
| PIPA2 | 138 | 0.145 | 0.054 | 0.266 | **0.665** | 0.344 | Transformer |
| PIPO | 24 | -0.094 | -0.146 | 0.058 | **0.269** | -0.005 | Transformer |
| POTR5 | 44 | 0.134 | 0.066 | **0.276** | 0.273 | 0.128 | CrossAttn |
| PSME | 183 | 0.115 | 0.185 | 0.393 | **0.538** | 0.415 | Transformer |
| TSCA | 30 | 0.170 | -0.124 | 0.192 | **0.562** | 0.280 | Transformer |

### canopy height

| Class | n | Concat | SupCon | CrossAttn | Transformer | MMCTL | Best |
|---|---|---|---|---|---|---|---|
| All (dataset-wide) | 592 | 0.121 | 0.066 | 0.252 | **0.478** | 0.274 | Transformer |
| ABLA | 30 | **0.380** | -0.089 | 0.346 | 0.117 | 0.298 | Concat |
| ACRU | 29 | -0.009 | -0.044 | **0.230** | -0.037 | 0.118 | CrossAttn |
| BENE4 | 36 | 0.155 | -0.108 | **0.181** | 0.107 | 0.142 | CrossAttn |
| PIEN | 36 | **0.387** | -0.050 | 0.224 | 0.041 | 0.371 | Concat |
| PIMA | 42 | **0.602** | 0.129 | 0.424 | 0.439 | 0.593 | Concat |
| PIPA2 | 138 | 0.310 | 0.185 | 0.171 | 0.298 | **0.504** | MMCTL |
| PIPO | 24 | 0.102 | 0.272 | **0.355** | 0.150 | 0.017 | CrossAttn |
| POTR5 | 44 | 0.633 | 0.351 | **0.826** | 0.555 | 0.667 | CrossAttn |
| PSME | 183 | 0.300 | 0.266 | 0.563 | **0.662** | 0.593 | Transformer |
| TSCA | 30 | 0.023 | -0.011 | -0.026 | **0.187** | 0.127 | Transformer |
