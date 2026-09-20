# AWIR Failure / Retrieval Analysis

## Setup
- Embeddings: `pretrained_embeddings/run01_mmctl_test.npy` (joint 2048-d).
- Cross-modal probe: RGB half (0:1024) queries Thermal half (1024:2048) with cosine NN.
- Same held-out test split as `demo_metrics.py` (seed=42, 50%).

## Quantitative
- Rank-1 **identity** retrieval (paired object): **0.075**
- Rank-5 identity: **0.300**
- Rank-1 **class** match: **0.808**
- Identity successes: 9 / 120
- Class failures at R1: 23 / 120

## Qualitative takeaways
- Successes: paired RGB/thermal halves of the same object land as nearest neighbors, consistent with the paper's shared-latent claim.
- Failures: nearest thermal half belongs to a different class; often visually ambiguous scale/pose or weak thermal contrast.
  - query `060421_DJI_0676.JPG` (deer) → NN class horse (cos-dist=0.094)
  - query `H_DJI_0584.JPG` (horse) → NN class cow (cos-dist=0.078)
  - query `H_DJI_0624.JPG` (horse) → NN class cow (cos-dist=0.097)

## Figures
- PCA: `results/figures/awir_mmctl_pca_run01.png`
- Success grid: `results/figures/awir_retrieval_successes.png`
- Failure grid: `results/figures/awir_retrieval_failures.png`
