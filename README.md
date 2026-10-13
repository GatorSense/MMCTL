# MMCTL Paper Repository

This repository contains code for training MMCTL latent spaces, training downstream task heads, and reproducing lightweight metric summaries from the paper.


## Files

- `mmctl/train_mmctl.py` : train the MMCTL latent space model
- `mmctl/train_taskheads.py` : train downstream task heads on learned embeddings
- `demo_metrics.py` : load sample results and display the main metrics
- `sample_results.csv` : lightweight example results

## File Architecture

- `mmctl/` : core library and training scripts
  - `train_mmctl.py` : entrypoint for learning multimodal embedding space
  - `train_taskheads.py` : entrypoint for training downstream classifiers/regressors on embeddings
  - `embedding_metrics.py` : functions for computing alignment/smoothness metrics
  - `utils.py` : helper utilities used by both training and metrics logic
  - `smoke_test_*` : integrated end-to-end validations for each stage
- `demo_metrics.py` : loads dataset NPZ + pretrained embeddings, computes metrics, saves reports
- `pyproject.toml` / `uv.lock` : uv-managed dependencies (primary; see Installation)
- `environment.yml` : conda dependency versions (alternative to uv)
- `sample_data.npz` : lightweight mini dataset for quick demo iteration
- `pretrained_embeddings/` : predicted embedding outputs used in metric evaluation

## Installation (uv)

This project uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management. Install uv (e.g. `curl -LsSf https://astral.sh/uv/install.sh | sh`),
then from the repository root:

```bash
uv sync
```

This creates a local `.venv` and installs the pinned dependencies from
`pyproject.toml` / `uv.lock` (uv also fetches a compatible Python if needed).
Run any command in the environment with `uv run`, e.g. `uv run python demo_metrics.py`
(or activate the env with `source .venv/bin/activate`).

Optional extras:

```bash
uv sync --extra full   # + umap-learn, for the full NEON reconstruction path
uv sync --extra docs   # + reportlab, to regenerate the reviewer-response PDF
```

<details>
<summary>Alternative: conda</summary>

```bash
conda env create -f environment.yml
conda activate mmctl
```
</details>

## Download Dataset

The datasets used in the paper are hosted on Zenodo.

Download them here:

https://zenodo.org/records/19040867

After downloading, extract the archive and place the dataset file in the repository directory (or specify the path when running the training scripts).

For repro, you need to download the `awir` dataset and save it in the same location as this repository (or point to it explicitly with a general path like `/path/to/awir_data.npz`). The `demo_metrics.py` script should be run with that file path, for example:

```bash
uv run python demo_metrics.py --data /path/to/awir_data.npz
```

`sample_data.npz` is a lightweight dataset used for a single demo iteration to show how models are trained from MMCTL to task heads and then evaluated with embedding metrics. It can be replaced with any other dataset NPZ file (e.g. `awir_data.npz`) during training.

## Update Info

- Date: 2026-09-26
- Functionality updated: migrated the reproduction setup to [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`); training code and reproduction path for the main tables.

## Quick Repro (recommended)

This is the easiest path to replicate Table 1 from the paper using precomputed sample results.

1. Install dependencies:

```bash
uv sync
```

2. Run the demo metrics script:

```bash
uv run python demo_metrics.py
```

This script loads the provided `sample_results.csv` and prints the same metric summary used for Table 1.

## Full training workflow (optional)

If you want to run the full training pipeline end-to-end, execute the smoke tests in this order:

1. Train MMCTL representation space:

```bash
uv run python mmctl/smoke_test_train_mmctl.py
```

2. Train downstream task heads:

```bash
uv run python mmctl/smoke_test_taskheads.py
```

3. Compute embedding metrics:

```bash
uv run python mmctl/smoke_test_embedding_metrics.py
```

The smoke tests are designed to verify each stage and recreate the evaluation chain.

## Direct training commands

`mmctl/train_mmctl.py` : train the MMCTL latent space model

`mmctl/train_taskheads.py` : train downstream task heads on learned embeddings

`demo_metrics.py` : load sample results and display the main metrics

`sample_results.csv` : lightweight example results

