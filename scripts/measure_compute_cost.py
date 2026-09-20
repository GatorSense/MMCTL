#!/usr/bin/env python3
"""Measure / estimate MMCTL compute cost on this machine (Apple Silicon)."""

from __future__ import annotations

import json
import os
import platform
import time
import tracemalloc

import numpy as np
import psutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def theoretical_triplets(batch_size: int) -> dict:
    """
    batch-all continuous multimodal loss duplicates modality embeddings:
    N = 2 * B. Counts all ordered distinct (a,p,n) triples; the loss further
    masks to valid annotation orderings, so this is an upper bound per batch.
    """
    n = 2 * batch_size
    max_ordered = n * (n - 1) * (n - 2)
    return {
        "batch_size": batch_size,
        "embeddings_after_modality_concat": n,
        "max_ordered_triplets_per_batch": max_ordered,
        "note": "Upper bound before annotation-order mask; actual positive count is data-dependent.",
    }


def main():
    import tensorflow as tf
    from mmctl.models import image_reconstruction_multi_modalv1
    from mmctl.losses import keras_batch_all_triplet_continuous_loss_multimodal

    data_path = os.path.join(REPO, "sample_data.npz")
    f = np.load(data_path, allow_pickle=True)
    rgb = f["rgb"].astype(np.float32)
    thermal = f["thermal"].astype(np.float32)
    rgb = rgb / max(float(rgb.max()), 1.0)
    thermal = thermal / max(float(thermal.max()), 1.0)

    # Paper defaults
    paper_bs = 32
    paper_epochs = 100
    paper_iterations = 10
    paper_runs = 8

    # Timed microbenchmark (memory-safe on sample_data)
    B = min(8, len(rgb))
    emb_dim = 1024
    model = image_reconstruction_multi_modalv1(
        rgb.shape[1:],
        thermal.shape[1:],
        emb_dim,
    )
    loss_fn = keras_batch_all_triplet_continuous_loss_multimodal(margin=0.1, emb_dim=emb_dim)
    ann_dim = 8
    xb, xt = rgb[:B], thermal[:B]
    ann = np.random.randn(B, ann_dim).astype(np.float32)

    # warmup
    _ = model([xb, xt], training=True)

    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    tracemalloc.start()
    t0 = time.perf_counter()
    n_steps = 5
    for _ in range(n_steps):
        with tf.GradientTape() as tape:
            outs = model([xb, xt], training=True)
            joint = outs[0]
            tl = loss_fn(ann, joint)
            if isinstance(tl, (list, tuple)):
                tl = tl[0]
            loss = tl
        grads = tape.gradient(loss, model.trainable_variables)
        del grads
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = proc.memory_info().rss

    sec_per_step = (t1 - t0) / n_steps
    # Rough epoch estimate for AWIR: ~120 train samples after 50% split → ceil(120/32)=4 steps
    # (resampling may increase). Use 4 as lower bound; 8 after resampling heuristic.
    steps_per_epoch_awir_est = 8
    sec_per_epoch_est = sec_per_step * steps_per_epoch_awir_est * (paper_bs / B)
    # Scale by batch size ratio is imperfect (triplet is O(N^3)); report both measured and scaled.

    trip_paper = theoretical_triplets(paper_bs)
    trip_meas = theoretical_triplets(B)

    # Also time one short fit epoch on sample if feasible
    fit_epoch_sec = None
    fit_n = min(16, len(rgb))
    try:
        from tensorflow.keras.optimizers import Adam
        from mmctl.losses import ssim_loss

        small = image_reconstruction_multi_modalv1(
            rgb.shape[1:], thermal.shape[1:], emb_dim
        )
        small.compile(
            optimizer=Adam(1e-4),
            loss=[
                keras_batch_all_triplet_continuous_loss_multimodal(0.1, emb_dim),
                ssim_loss(0.1),
                ssim_loss(0.1),
            ],
        )
        # dummy annotation matching weigh_features dim — use random for timing only
        y_ann = np.random.randn(fit_n, ann_dim).astype(np.float32)
        t_fit0 = time.perf_counter()
        small.fit(
            [rgb[:fit_n], thermal[:fit_n]],
            [y_ann, rgb[:fit_n], thermal[:fit_n]],
            epochs=1,
            batch_size=min(4, fit_n),
            verbose=0,
        )
        fit_epoch_sec = time.perf_counter() - t_fit0
    except Exception as e:
        fit_epoch_sec = f"failed: {e}"

    out = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_brand": _cpu_brand(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "tf_devices": [d.name for d in tf.config.list_physical_devices()],
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        },
        "paper_training_defaults": {
            "batch_size": paper_bs,
            "epochs_per_iteration": paper_epochs,
            "iterations_per_run": paper_iterations,
            "runs_awir": paper_runs,
            "total_epochs_awir": paper_epochs * paper_iterations * paper_runs,
        },
        "triplets": {
            "paper_batch": trip_paper,
            "measured_batch": trip_meas,
            "triplets_per_epoch_upper_bound_awir": trip_paper["max_ordered_triplets_per_batch"]
            * steps_per_epoch_awir_est,
        },
        "measured": {
            "microbatch_size": B,
            "sec_per_train_step_microbatch": sec_per_step,
            "estimated_sec_per_epoch_awir_naive_scale": sec_per_epoch_est,
            "sample_fit_1epoch_n16_bs4_sec": fit_epoch_sec,
            "tracemalloc_peak_mb": peak / (1024**2),
            "rss_delta_mb": (rss_after - rss_before) / (1024**2),
            "rss_after_mb": rss_after / (1024**2),
            "caveat": (
                "Full AWIR retrain (800 effective epochs across 8 runs) not executed on M4. "
                "Triplet cost scales roughly O((2B)^3); naive linear batch scaling underestimates."
            ),
        },
    }

    out_path = os.path.join(REPO, "results/compute_cost.json")
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2)
    print(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


def _cpu_brand():
    try:
        import subprocess

        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
