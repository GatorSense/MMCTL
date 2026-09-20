#!/usr/bin/env python3

import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from mmctl.utils import generate_triplet_box_label_normalized

# TensorFlow is only required when loading Keras checkpoints. Keep it lazy so
# pretrained-embedding metric evaluation can run on machines where TF crashes.
tf = None


def _require_tf():
    global tf
    if tf is None:
        import tensorflow as _tf

        tf = _tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    return tf


def _require_losses():
    from mmctl.losses import (
        keras_batch_all_triplet_continuous_loss_multimodal,
        ssim_loss,
    )

    return keras_batch_all_triplet_continuous_loss_multimodal, ssim_loss


# =========================================================
# Metric functions
# =========================================================
def _subsample_idx(n: int, max_n: int = 1500, seed: int = 42) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_n, replace=False)


def embedding_annotation_alignment_ratio(
    z: np.ndarray,
    y: np.ndarray,
    kind: str,
    max_n: int = 1200,
    seed: int = 42,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Global alignment between embedding distances and annotation distances.

    kind="cls":
        Uses only different-class pairs. Returns:
            align_ratio
            emb_within_mean
            emb_between_mean

    kind="cont":
        Uses Euclidean annotation distances for continuous targets. Returns:
            align_ratio
    """
    n = z.shape[0]
    idx = _subsample_idx(n, max_n=max_n, seed=seed)

    z_ = z[idx]
    dz_mat = pairwise_distances(z_, metric="euclidean")
    iu = np.triu_indices(dz_mat.shape[0], k=1)
    dz = dz_mat[iu]

    out: Dict[str, float] = {}

    if kind == "cls":
        y_ = y[idx].astype(np.int64)
        same = (y_[:, None] == y_[None, :])
        diff_mask = (~same)[iu]

        if diff_mask.sum() == 0:
            out["align_ratio"] = np.nan
            out["emb_within_mean"] = np.nan
            out["emb_between_mean"] = np.nan
            return out

        within_mask = same[iu]
        out["align_ratio"] = float(np.mean(dz[diff_mask]))
        out["emb_within_mean"] = float(np.mean(dz[within_mask])) if within_mask.sum() > 0 else np.nan
        out["emb_between_mean"] = float(np.mean(dz[diff_mask]))
        return out

    if kind == "cont":
        y_ = y[idx].astype(np.float32)
        dy_mat = pairwise_distances(y_, metric="euclidean")
        dy = dy_mat[iu]
        valid = dy > eps

        if valid.sum() == 0:
            out["align_ratio"] = np.nan
            return out

        out["align_ratio"] = float(np.mean(dz[valid] / (dy[valid] + eps)))
        return out

    raise ValueError(f"Unknown kind: {kind}")


def local_smoothness_std(
    z: np.ndarray,
    y: np.ndarray,
    kind: str,
    ks: Tuple[int, ...] = (1, 2, 3, 4, 5),
    max_n: int = None,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Local smoothness:
      For each sample, compute std of annotation values among k nearest neighbors.
    """
    n = z.shape[0]
    if max_n is not None:
        idx = _subsample_idx(n, max_n=max_n, seed=seed)
        z_use = z[idx]
        y_use = y[idx]
    else:
        z_use = z
        y_use = y

    max_k = max(ks)
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean")
    nn.fit(z_use)
    neigh = nn.kneighbors(z_use, return_distance=False)
    neigh = neigh[:, 1:]  # remove self

    out: Dict[str, float] = {}

    if kind == "cls":
        y_int = y_use.astype(np.int64)
        for k in ks:
            nbr = neigh[:, :k]
            nbr_labels = y_int[nbr]
            out[f"smooth_std_k{k}"] = float(np.mean(np.std(nbr_labels, axis=1)))
            center = y_int[:, None]
            out[f"nbr_agree_k{k}"] = float(np.mean(nbr_labels == center))
        return out

    if kind == "cont":
        y_cont = y_use.astype(np.float32)
        for k in ks:
            nbr = neigh[:, :k]
            nbr_vals = y_cont[nbr]
            std_per_dim = np.std(nbr_vals, axis=1)
            out[f"smooth_std_k{k}"] = float(np.mean(std_per_dim))
        return out

    raise ValueError(f"Unknown kind: {kind}")


def compute_all_alignment_and_smoothness(
    z: np.ndarray,
    y_cls_int: np.ndarray,
    y_box_feat: np.ndarray,
    y_pos: np.ndarray,
    seed: int = 42,
    pair_max_n: int = 1200,
    smooth_max_n: int = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}

    a_cls = embedding_annotation_alignment_ratio(z, y_cls_int, kind="cls", max_n=pair_max_n, seed=seed)
    out["align_ratio_cls"] = a_cls["align_ratio"]
    out["emb_within_mean_cls"] = a_cls.get("emb_within_mean", np.nan)
    out["emb_between_mean_cls"] = a_cls.get("emb_between_mean", np.nan)

    a_box = embedding_annotation_alignment_ratio(z, y_box_feat, kind="cont", max_n=pair_max_n, seed=seed)
    out["align_ratio_box"] = a_box["align_ratio"]

    a_pos = embedding_annotation_alignment_ratio(z, y_pos, kind="cont", max_n=pair_max_n, seed=seed)
    out["align_ratio_pos"] = a_pos["align_ratio"]

    s_cls = local_smoothness_std(z, y_cls_int, kind="cls", ks=(1, 2, 3, 4, 5), max_n=smooth_max_n, seed=seed)
    s_box = local_smoothness_std(z, y_box_feat, kind="cont", ks=(1, 2, 3, 4, 5), max_n=smooth_max_n, seed=seed)
    s_pos = local_smoothness_std(z, y_pos, kind="cont", ks=(1, 2, 3, 4, 5), max_n=smooth_max_n, seed=seed)

    for k, v in s_cls.items():
        out[f"{k}_cls"] = v
    for k, v in s_box.items():
        out[f"{k}_box"] = v
    for k, v in s_pos.items():
        out[f"{k}_pos"] = v

    return out


# =========================================================
# Data and model helpers
# =========================================================
def convert_box_to_center_width_height(y_box: np.ndarray) -> np.ndarray:
    num_samples = y_box.shape[0]
    y_box_cenwh = np.zeros((num_samples, 4), dtype=np.float32)

    for i in range(num_samples):
        xmin, xmax, ymin, ymax = y_box[i]
        xcenter = (xmin + xmax) / 2.0
        ycenter = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin

        y_box_cenwh[i, 0] = xcenter / 300.0
        y_box_cenwh[i, 1] = ycenter / 300.0
        y_box_cenwh[i, 2] = width / 300.0
        y_box_cenwh[i, 3] = height / 300.0

    return y_box_cenwh


def load_targets(data_npz: str) -> Dict[str, np.ndarray]:
    f = np.load(data_npz, allow_pickle=True)

    rgb = f["rgb"].astype(np.float32)
    thermal = f["thermal"].astype(np.float32)
    y_cls_raw = f["class_label"]
    y_box = f["box_label"].astype(np.float32)
    y_box = np.clip(y_box, None, 300)

    y_box_cenwh = convert_box_to_center_width_height(y_box)
    y_triplet_box_label, box_feat = generate_triplet_box_label_normalized(y_box, y_cls_raw, n_clusters=3)
    y_pos = y_box_cenwh[:, :2].astype(np.float32)
    y_cls_int = LabelEncoder().fit_transform(y_cls_raw).astype(np.int64)

    composite_label = [f"{c}_{t}" for c, t in zip(y_cls_int, y_triplet_box_label)]
    composite_label_encoded = LabelEncoder().fit_transform(composite_label)

    return {
        "rgb": rgb,
        "thermal": thermal,
        "y_cls_int": y_cls_int,
        "y_box_feat": box_feat.astype(np.float32),
        "y_pos": y_pos,
        "composite_label_encoded": composite_label_encoded,
    }


def build_custom_objects(embedding_dimension: int, margin: float) -> Dict[str, object]:
    keras_batch_all_triplet_continuous_loss_multimodal, ssim_loss = _require_losses()
    return {
        "keras_batch_all_triplet_continuous_loss_multimodal":
            keras_batch_all_triplet_continuous_loss_multimodal,
        "ssim_loss": ssim_loss,
        "loss_fn": keras_batch_all_triplet_continuous_loss_multimodal(
            margin=margin,
            emb_dim=embedding_dimension,
        ),
    }


def extract_embeddings(
    checkpoint_path: str,
    rgb: np.ndarray,
    thermal: np.ndarray,
    embedding_dimension: int,
    margin: float,
    batch_size: int,
) -> np.ndarray:
    _tf = _require_tf()
    custom_objects = build_custom_objects(embedding_dimension, margin)

    model = _tf.keras.models.load_model(
        checkpoint_path,
        custom_objects=custom_objects,
        compile=False,
    )

    outputs = model.predict(
        [rgb.astype(np.float32), thermal.astype(np.float32)],
        batch_size=batch_size,
        verbose=0,
    )

    return np.asarray(outputs[0], dtype=np.float32)


def make_fold_splits(
    y_cls_int: np.ndarray,
    composite_label_encoded: np.ndarray,
    test_size: float,
    runs: int,
    seed: int,
) -> List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    idx_all = np.arange(len(y_cls_int))

    idx_trainval, idx_test, cls_trainval, _ = train_test_split(
        idx_all,
        y_cls_int,
        test_size=test_size,
        stratify=composite_label_encoded,
        random_state=seed,
    )

    skf = StratifiedKFold(n_splits=runs, shuffle=True, random_state=seed)

    fold_splits = []
    for run_id, (tr, va) in enumerate(skf.split(idx_trainval, cls_trainval), start=1):
        idx_tr = idx_trainval[tr]
        idx_va = idx_trainval[va]
        fold_splits.append((run_id, idx_tr, idx_va, idx_test))

    return fold_splits


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_npz", type=str, default="data.npz")
    parser.add_argument("--mmctl_dir", type=str, default="outputs/mmctl")
    parser.add_argument("--out_dir", type=str, default="outputs/embedding_metrics")
    parser.add_argument("--embedding_dimension", type=int, default=1024)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_size", type=float, default=0.5)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair_max_n", type=int, default=1200)
    parser.add_argument("--smooth_max_n", type=int, default=1200)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = load_targets(args.data_npz)
    fold_splits = make_fold_splits(
        y_cls_int=data["y_cls_int"],
        composite_label_encoded=data["composite_label_encoded"],
        test_size=args.test_size,
        runs=args.runs,
        seed=args.seed,
    )

    results: List[Dict[str, object]] = []

    for run_id, idx_tr, idx_va, idx_te in fold_splits:
        checkpoint_path = os.path.join(args.mmctl_dir, f"run_{run_id}", "best_model.h5")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        if args.split == "train":
            idx = idx_tr
        elif args.split == "val":
            idx = idx_va
        else:
            idx = idx_te

        z = extract_embeddings(
            checkpoint_path=checkpoint_path,
            rgb=data["rgb"][idx],
            thermal=data["thermal"][idx],
            embedding_dimension=args.embedding_dimension,
            margin=args.margin,
            batch_size=args.batch_size,
        )

        metrics = compute_all_alignment_and_smoothness(
            z=z,
            y_cls_int=data["y_cls_int"][idx],
            y_box_feat=data["y_box_feat"][idx],
            y_pos=data["y_pos"][idx],
            seed=args.seed,
            pair_max_n=args.pair_max_n,
            smooth_max_n=args.smooth_max_n,
        )

        row = {
            "run": run_id,
            "split": args.split,
            "n_samples": len(idx),
            **metrics,
        }
        results.append(row)

        print(
            f"[run{run_id:02d}] split={args.split} "
            f"align_cls={row['align_ratio_cls']:.4f} "
            f"align_box={row['align_ratio_box']:.4f} "
            f"align_pos={row['align_ratio_pos']:.4f}"
        )

    df = pd.DataFrame(results)

    csv_path = os.path.join(args.out_dir, "embedding_metrics.csv")
    json_path = os.path.join(args.out_dir, "embedding_metrics.json")
    summary_csv_path = os.path.join(args.out_dir, "embedding_metrics_summary.csv")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as fp:
        json.dump(results, fp, indent=2)

    summary = df.drop(columns=["run"]).mean(numeric_only=True).to_frame().T
    summary.to_csv(summary_csv_path, index=False)

    print(f"\nSaved:\n  {csv_path}\n  {json_path}\n  {summary_csv_path}")


if __name__ == "__main__":
    main()