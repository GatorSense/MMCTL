# Embedding-space quality metrics

StratifiedKFold on released full-set embeddings (8 AWIR / 5 NEON); silhouette (cosine, class) and Spearman rho between pairwise cosine embedding distances and Euclidean annotation distances (box, position/height); learned fusions trained per fold.

Silhouette (class separation) and Spearman rho (continuous-annotation distance preservation); higher is better for all columns.

## AWIR

| Method | Silhouette (class) | Spearman ρ (B) | Spearman ρ (P) |
|---|---|---|---|
| RGB | 0.101$\pm$0.065 | 0.191$\pm$0.078 | 0.062$\pm$0.096 |
| Thermal | 0.108$\pm$0.023 | 0.221$\pm$0.104 | 0.023$\pm$0.065 |
| Concat | 0.157$\pm$0.040 | 0.237$\pm$0.071 | 0.059$\pm$0.058 |
| SupCon | 0.723$\pm$0.112 | 0.222$\pm$0.073 | -0.021$\pm$0.045 |
| CrossAttn | 0.516$\pm$0.117 | 0.209$\pm$0.089 | 0.023$\pm$0.030 |
| Transformer | 0.678$\pm$0.139 | 0.202$\pm$0.137 | -0.022$\pm$0.052 |
| MMCTL | 0.427$\pm$0.050 | 0.278$\pm$0.073 | 0.223$\pm$0.063 |

Paired one-sided Wilcoxon (MMCTL > best competitor):
- Sil: vs SupCon, p=1.0000
- rho B: vs Concat, p=0.0977
- rho P: vs RGB, p=0.0078

## NEON

| Method | Silhouette (class) | Spearman ρ (B) | Spearman ρ (H) |
|---|---|---|---|
| RGB | -0.146$\pm$0.031 | 0.081$\pm$0.049 | 0.072$\pm$0.035 |
| HSI | -0.040$\pm$0.053 | 0.024$\pm$0.023 | 0.045$\pm$0.043 |
| LiDAR | -0.232$\pm$0.035 | 0.167$\pm$0.022 | 0.571$\pm$0.017 |
| Concat | -0.003$\pm$0.047 | 0.055$\pm$0.021 | 0.118$\pm$0.037 |
| SupCon | 0.562$\pm$0.031 | 0.049$\pm$0.039 | 0.058$\pm$0.010 |
| CrossAttn | -0.208$\pm$0.059 | 0.209$\pm$0.019 | 0.314$\pm$0.051 |
| Transformer | -0.584$\pm$0.024 | 0.402$\pm$0.085 | 0.409$\pm$0.099 |
| MMCTL | 0.197$\pm$0.034 | 0.188$\pm$0.040 | 0.447$\pm$0.049 |

Paired one-sided Wilcoxon (MMCTL > best competitor):
- Sil: vs SupCon, p=1.0000
- rho B: vs Transformer, p=1.0000
- rho H: vs LiDAR, p=1.0000
