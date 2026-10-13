# MMCTL — Reproduction Code (anonymous)

Reproduction code for **Multi-Modal Continuous Triplet Loss (MMCTL)**: training a
shared annotation-driven latent space, training downstream task heads, and
reproducing the quantitative metrics, ablations, and figures reported in the
paper.

> This repository is anonymized for double-blind review. It contains **code
> only**; datasets and pretrained embeddings are not distributed here (see
> *Data* below for where to place them).

## Repository layout

- `mmctl/` — core library and entrypoints
  - `train_mmctl.py` — learn the multi-modal shared embedding space
  - `train_taskheads.py` — train downstream classifiers/regressors on frozen embeddings
  - `continuous_triplet_loss.py`, `losses.py`, `models.py` — loss and model definitions
  - `embedding_metrics.py` — silhouette / distance-preservation / smoothness metrics
  - `utils.py` — joint annotation vector construction and helpers
  - `smoke_test_*.py` — fast end-to-end checks for each stage
- `scripts/` — experiment drivers that reproduce the paper's numbers and figures
  - `eval_awir_released_embeddings.py`, `eval_neon_released_embeddings.py` — main downstream tables
  - `embedding_space_metrics.py`, `perclass_smoothness.py`, `significance_tests.py` — latent-space metrics + Wilcoxon tests
  - `ablation_neon_class_weight.py`, `ablation_loss_components_neon.py` — class-weight and loss-component ablations
  - `freeze_vs_e2e_neon.py`, `retune_transformer_neon.py` — frozen-vs-end-to-end and Transformer-fusion studies
  - `build_figure_embeddings.py`, `make_embedding_figures.py` — embedding figures (overall + per-class zoom)
  - `reproduce_neon_*.py`, `run_neon_paper_reconstruction.py` — NEON reconstruction / proxy pipelines
- `demo_metrics.py` — load a dataset NPZ + embeddings and print the headline metrics
- `TripletNetwork_Online.py`, `neon_custom_losses.py`, `neon_utilities.py` — supporting modules for the NEON reconstruction path
- `pyproject.toml` / `uv.lock` — [uv](https://docs.astral.sh/uv/)-managed environment (primary)
- `environment.yml` — conda alternative

## Installation

With uv (recommended):

```bash
uv sync                       # core environment
uv sync --extra full          # adds umap-learn for the full NEON reconstruction path
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate mmctl
```

## Data

The scripts expect the dataset NPZ files and (for the evaluation scripts) the
released embeddings under a local `data/` directory, e.g.:

```
data/awir_data.npz
data/neon_data.npz
pretrained_embeddings/...          # released per-method embeddings
```

Paths can be overridden per-script via the `--npz_path` / `--data` arguments, or
by setting `MMCTL_DIR` for scripts that resolve a repository root. The datasets
themselves are not included in this anonymized repository.

## Reproducing results

```bash
# Latent-space quality metrics (silhouette + Spearman distance-preservation) + Wilcoxon
uv run python scripts/embedding_space_metrics.py
uv run python scripts/significance_tests.py

# Downstream performance tables
uv run python scripts/eval_awir_released_embeddings.py
uv run python scripts/eval_neon_released_embeddings.py

# Ablations
uv run python scripts/ablation_neon_class_weight.py
uv run python scripts/ablation_loss_components_neon.py
uv run python scripts/freeze_vs_e2e_neon.py
uv run python scripts/retune_transformer_neon.py

# Per-class smoothness + embedding figures
uv run python scripts/perclass_smoothness.py
uv run python scripts/build_figure_embeddings.py
uv run python scripts/make_embedding_figures.py
```

Each script writes its JSON/Markdown summary (and figures) to a local
`results/` directory.

## Quick check

```bash
uv run python mmctl/smoke_test_train_mmctl.py
uv run python mmctl/smoke_test_taskheads.py
uv run python mmctl/smoke_test_embedding_metrics.py
```

## License

Released under the MIT License (see `LICENSE`).
