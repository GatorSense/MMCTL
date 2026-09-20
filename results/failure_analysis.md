# AWIR Failure / Retrieval Analysis

## Setup
- Embeddings: `pretrained_embeddings/run01_mmctl_test.npy` (joint 2048-d).
- Cross-modal probe: treat dims `0:1024` as RGB half and `1024:2048` as thermal half (matches training concat layout), L2-normalize each half, cosine NN.
- Same held-out test split as `demo_metrics.py` (seed=42, 50%).

## Quantitative
- Rank-1 **identity** retrieval (paired object): **0.075** (9 / 120)
- Rank-5 identity: **0.300**
- Rank-1 **class** match: **0.808**
- Class failures at R1: 23 / 120

## Interpretation (for Agent B)
- **Class-level** cross-modal neighborhood structure is strong (~81% R1 class match), consistent with annotation-driven alignment.
- **Exact instance** retrieval is weak (7.5% R1 identity). Do **not** over-claim pixel-perfect RGB↔thermal instance pairing from these halves alone; the loss organizes annotation geometry/class more than guaranteeing identity nearest neighbors after splitting the joint vector.
- Confusion patterns in failures are semantically adjacent livestock classes (deer→horse, horse→cow).

## Example failures
- query `060421_DJI_0676.JPG` (deer) → NN class horse (cos-dist=0.094)
- query `H_DJI_0584.JPG` (horse) → NN class cow (cos-dist=0.078)
- query `H_DJI_0624.JPG` (horse) → NN class cow (cos-dist=0.097)

## Figures
- PCA: `results/figures/awir_mmctl_pca_run01.png`
- Success grid (identity matches): `results/figures/awir_retrieval_successes.png`
- Failure grid (class mismatches): `results/figures/awir_retrieval_failures.png`

Machine-readable: `results/failure_analysis_metrics.json`.
