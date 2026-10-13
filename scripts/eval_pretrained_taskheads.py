#!/usr/bin/env python3
"""
Train frozen task heads on released AWIR pretrained *test* embeddings.

Protocol note
-------------
`pretrained_embeddings/` only contains held-out test embeddings (N=120) for each
of 8 runs. Paper protocol trains heads on fold train/val embeddings from Keras
checkpoints, which are not shipped. This script therefore:

  1. Reconstructs the same stratified 50/50 split used by demo_metrics.py
  2. For each run/method, runs StratifiedKFold *within the fixed test set*
     (default 4 folds) to train/eval linear heads
  3. Aggregates mean±std across outer runs

Results are a **proxy** for Table downstream performance, not a byte-identical
replication. Ranking and relative MMCTL gains are the intended signal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from mmctl.train_taskheads import (  # noqa: E402
    HeadTrainConfig,
    standardize_z,
    train_three_heads,
)
from mmctl.utils import generate_triplet_box_label_normalized  # noqa: E402


METHOD_SPECS = {
    "RGB": "run{run:02d}_rgb_test.npy",
    "Thermal": "run{run:02d}_thermal_test.npy",
    "MMCTL": "run{run:02d}_mmctl_test.npy",
    "DeepSetMean": "run{run:02d}_deepset_mean_test_fused.npy",
    "DeepSetAttn": "run{run:02d}_deepset_attention_test_fused.npy",
    "Concat": None,  # built from rgb+thermal
}


def convert_box_to_center_width_height(y_box: np.ndarray) -> np.ndarray:
    out = np.zeros((y_box.shape[0], 4), dtype=np.float32)
    for i in range(y_box.shape[0]):
        xmin, xmax, ymin, ymax = y_box[i]
        out[i, 0] = ((xmin + xmax) / 2.0) / 300.0
        out[i, 1] = ((ymin + ymax) / 2.0) / 300.0
        out[i, 2] = (xmax - xmin) / 300.0
        out[i, 3] = (ymax - ymin) / 300.0
    return out


def l2_normalize_rows(z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    return z / (norms + eps)


def load_method_z(emb_dir: str, method: str, run: int, z_rgb=None, z_th=None) -> np.ndarray:
    if method == "Concat":
        z = np.concatenate([z_rgb, z_th], axis=1)
    else:
        path = os.path.join(emb_dir, METHOD_SPECS[method].format(run=run))
        z = np.load(path).astype(np.float32)
    return l2_normalize_rows(z)


def mask_modality(z: np.ndarray, mode: str) -> np.ndarray:
    """For MMCTL (2048=1024+1024) zero one modality half."""
    if z.shape[1] != 2048:
        raise ValueError(f"Expected 2048-d MMCTL joint emb, got {z.shape}")
    out = z.copy()
    if mode == "drop_thermal":
        out[:, 1024:] = 0.0
    elif mode == "drop_rgb":
        out[:, :1024] = 0.0
    elif mode == "full":
        pass
    else:
        raise ValueError(mode)
    return l2_normalize_rows(out)


def eval_one_embedding(
    z: np.ndarray,
    y_cls: np.ndarray,
    y_box: np.ndarray,
    y_pos: np.ndarray,
    head_cfg: HeadTrainConfig,
    n_inner: int,
    seed: int,
) -> Dict[str, float]:
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(np.arange(len(y_cls)), y_cls), start=1):
        # carve a small val from train
        if len(tr) < 8:
            tr_i, va_i = tr, tr
        else:
            tr_i, va_i = train_test_split(
                tr, test_size=0.25, stratify=y_cls[tr], random_state=seed + fold
            )
        z_tr, z_va, z_te = standardize_z(z[tr_i], z[va_i], z[te])
        stats = train_three_heads(
            cfg=head_cfg,
            z_tr=z_tr,
            z_va=z_va,
            z_te=z_te,
            y_cls_tr=y_cls[tr_i],
            y_cls_va=y_cls[va_i],
            y_cls_te=y_cls[te],
            y_box_tr=y_box[tr_i],
            y_box_va=y_box[va_i],
            y_box_te=y_box[te],
            y_pos_tr=y_pos[tr_i],
            y_pos_va=y_pos[va_i],
            y_pos_te=y_pos[te],
        )
        rows.append(stats)

    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/awir_data.npz")
    parser.add_argument("--emb_dir", default=os.path.join(REPO, "pretrained_embeddings"))
    parser.add_argument("--out_json", default=os.path.join(REPO, "results/taskhead_proxy_awir.json"))
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--inner_folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--methods", nargs="+", default=list(METHOD_SPECS.keys()))
    parser.add_argument(
        "--missing_modality",
        action="store_true",
        help="Also evaluate MMCTL with thermal/rgb halves zeroed.",
    )
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    head_cfg = HeadTrainConfig(
        epochs=args.epochs,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        hidden=0,
        dropout=0.0,
        early_patience=args.patience,
        device=device,
    )

    f = np.load(args.data, allow_pickle=True)
    y_cls_raw = f["class_label"]
    y_box = np.clip(f["box_label"].astype(np.float32), None, 300)
    y_box_cenwh = convert_box_to_center_width_height(y_box)
    y_triplet_box_label, box_feat_all = generate_triplet_box_label_normalized(
        y_box, y_cls_raw, n_clusters=3
    )
    box_feat_all = box_feat_all.astype(np.float32)
    y_pos_all = y_box_cenwh[:, :2].astype(np.float32)
    y_cls_int_all = LabelEncoder().fit_transform(y_cls_raw).astype(np.int64)

    composite = [f"{c}_{t}" for c, t in zip(y_cls_int_all, y_triplet_box_label)]
    composite_enc = LabelEncoder().fit_transform(composite)
    idx_all = np.arange(len(y_cls_raw))
    _, idx_test = train_test_split(
        idx_all, test_size=0.5, stratify=composite_enc, random_state=args.seed
    )

    y_cls = y_cls_int_all[idx_test]
    y_box_feat = box_feat_all[idx_test]
    y_pos = y_pos_all[idx_test]

    results: List[Dict] = []
    for run in range(1, args.runs + 1):
        z_rgb = np.load(os.path.join(args.emb_dir, f"run{run:02d}_rgb_test.npy")).astype(np.float32)
        z_th = np.load(os.path.join(args.emb_dir, f"run{run:02d}_thermal_test.npy")).astype(np.float32)
        if z_rgb.shape[0] != len(idx_test):
            raise ValueError(f"run{run}: emb N={z_rgb.shape[0]} vs test={len(idx_test)}")

        for method in args.methods:
            print(f"run={run:02d} method={method}")
            z = load_method_z(args.emb_dir, method, run, z_rgb=z_rgb, z_th=z_th)
            stats = eval_one_embedding(
                z, y_cls, y_box_feat, y_pos, head_cfg, args.inner_folds, args.seed
            )
            results.append({"run": run, "method": method, "setup": "full", **stats})
            print(
                f"  acc={stats['acc']*100:.2f}  "
                f"r2_box={stats['r2_box_mean']:.3f}  r2_pos={stats['r2_pos_mean']:.3f}"
            )

        if args.missing_modality:
            z_mm = load_method_z(args.emb_dir, "MMCTL", run, z_rgb=z_rgb, z_th=z_th)
            for mode in ["drop_thermal", "drop_rgb"]:
                print(f"run={run:02d} method=MMCTL setup={mode}")
                z = mask_modality(z_mm, mode)
                stats = eval_one_embedding(
                    z, y_cls, y_box_feat, y_pos, head_cfg, args.inner_folds, args.seed
                )
                results.append({"run": run, "method": "MMCTL", "setup": mode, **stats})
                print(
                    f"  acc={stats['acc']*100:.2f}  "
                    f"r2_box={stats['r2_box_mean']:.3f}  r2_pos={stats['r2_pos_mean']:.3f}"
                )

    # aggregate
    def agg(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
        out = {}
        for method in sorted({r["method"] for r in rows}):
            for setup in sorted({r["setup"] for r in rows if r["method"] == method}):
                sub = [r for r in rows if r["method"] == method and r["setup"] == setup]
                key = f"{method}|{setup}"
                out[key] = {
                    "acc_pct_mean": float(np.mean([r["acc"] for r in sub]) * 100),
                    "acc_pct_std": float(np.std([r["acc"] for r in sub], ddof=1) * 100),
                    "r2_box_mean": float(np.mean([r["r2_box_mean"] for r in sub])),
                    "r2_box_std": float(np.std([r["r2_box_mean"] for r in sub], ddof=1)),
                    "r2_pos_mean": float(np.mean([r["r2_pos_mean"] for r in sub])),
                    "r2_pos_std": float(np.std([r["r2_pos_mean"] for r in sub], ddof=1)),
                    "n_runs": len(sub),
                }
        return out

    payload = {
        "protocol": (
            "Proxy: StratifiedKFold within released test embeddings only "
            f"(inner_folds={args.inner_folds}). Not identical to paper train/val/test heads."
        ),
        "device": device,
        "per_run": results,
        "aggregate": agg(results),
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
