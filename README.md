# MMCTL Paper Repository

This repository contains code for training MMCTL latent spaces, training downstream task heads, and reproducing lightweight metric summaries from the paper.

## Files

- `mmctl/train_mmctl.py` : train the MMCTL latent space model
- `mmctl/train_taskheads.py` : train downstream task heads on learned embeddings
- `demo_metrics.py` : load sample results and display the main metrics
- `sample_results.csv` : lightweight example results

## Installation

```bash
pip install -r requirements.txt
```

## Download Dataset

The datasets used in the paper are hosted on Zenodo.

Download them here:

https://zenodo.org/records/19040867

After downloading, extract the archive and place the dataset file in the repository directory (or specify the path when running the training scripts).

For repro, you need to download the `awir` dataset and save it in the same location as this repository (or point to it explicitly with a general path like `/path/to/awir_data.npz`). The `demo_metrics.py` script should be run with that file path, for example:

```bash
python demo_metrics.py --data /path/to/awir_data.npz
```

`sample_data.npz` is a lightweight dataset used for a single demo iteration to show how models are trained from MMCTL to task heads and then evaluated with embedding metrics. It can be replaced with any other dataset NPZ file (e.g. `awir_data.npz`) during training.

## Quick Repro (recommended)

This is the easiest path to replicate Table 1 from the paper using precomputed sample results.

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Run the demo metrics script:

```bash
python demo_metrics.py
```

This script loads the provided `sample_results.csv` and prints the same metric summary used for Table 1.

## Full training workflow (optional)

If you want to run the full training pipeline end-to-end, execute the smoke tests in this order:

1. Train MMCTL representation space:

```bash
python mmctl/smoke_test_train_mmctl.py
```

2. Train downstream task heads:

```bash
python mmctl/smoke_test_taskheads.py
```

3. Compute embedding metrics:

```bash
python mmctl/smoke_test_embedding_metrics.py
```

The smoke tests are designed to verify each stage and recreate the evaluation chain.

## Direct training commands

`mmctl/train_mmctl.py` : train the MMCTL latent space model

`mmctl/train_taskheads.py` : train downstream task heads on learned embeddings

`demo_metrics.py` : load sample results and display the main metrics

`sample_results.csv` : lightweight example results

