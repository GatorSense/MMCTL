# MMCTL Paper Repository

This repository contains code for training MMCTL latent spaces, training downstream task heads, and reproducing lightweight metric summaries from the paper.

## Files

- `train_mmctl.py` : train the MMCTL latent space model
- `train_task_heads.py` : train downstream task heads on learned embeddings
- `demo_metrics.py` : load sample results and display the main metrics
- `sample_results.csv` : lightweight example results

## Installation

```bash
pip install -r requirements.txt
```

## Run the demo
`python demo_metrics.py`

## Train MMCTL 
`python train_mmctl.py`

## Train task heads 
`python train_task_heads.py`

