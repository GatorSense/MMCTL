#!/usr/bin/env python3

import os
import math
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, r2_score

from mmctl.utils import generate_triplet_box_label_normalized
from mmctl.losses import (
    keras_batch_all_triplet_continuous_loss_multimodal,
    ssim_loss,
)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# =========================================================
# Configs
# =========================================================
@dataclass
class SplitConfig:
    test_size: float = 0.5
    runs: int = 8
    seed: int = 42


@dataclass
class HeadTrainConfig:
    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: int = 0
    dropout: float = 0.0
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    early_patience: int = 50
    grad_clip: float = 0.0


# =========================================================
# Models for frozen embeddings
# =========================================================
class ClsModel(nn.Module):
    def __init__(self, d_in: int, n_classes: int, hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        if hidden > 0:
            self.backbone = nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            d_h = hidden
        else:
            self.backbone = nn.Identity()
            d_h = d_in

        self.head = nn.Linear(d_h, n_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(z))


class RegModel(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        if hidden > 0:
            self.backbone = nn.Sequential(
                nn.Linear(d_in, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            d_h = hidden
        else:
            self.backbone = nn.Identity()
            d_h = d_in

        self.head = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_h, d_out),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(z))


# =========================================================
# Datasets
# =========================================================
class FrozenEmbeddingClsDataset(Dataset):
    def __init__(self, z: np.ndarray, y: np.ndarray):
        self.z = z.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.z[idx]), torch.tensor(self.y[idx], dtype=torch.long)


class FrozenEmbeddingRegDataset(Dataset):
    def __init__(self, z: np.ndarray, y: np.ndarray):
        self.z = z.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.z[idx]), torch.from_numpy(self.y[idx])


# =========================================================
# Evaluation helpers
# =========================================================
@torch.no_grad()
def eval_cls(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    ys, ps, losses = [], [], []

    for z, y in loader:
        z = z.to(device)
        y = y.to(device)

        logits = model(z)
        loss = F.cross_entropy(logits, y)
        pred = torch.argmax(logits, dim=1)

        losses.append(float(loss.detach().cpu()))
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    y_hat = np.concatenate(ps, axis=0)

    return {
        "cls_loss": float(np.mean(losses)),
        "acc": float(accuracy_score(y_true, y_hat)),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro")),
    }


@torch.no_grad()
def eval_reg(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    ys, ps, losses = [], [], []

    for z, y in loader:
        z = z.to(device)
        y = y.to(device)

        pred = model(z)
        loss = F.mse_loss(pred, y)

        losses.append(float(loss.detach().cpu()))
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    y_hat = np.concatenate(ps, axis=0)

    r2s = []
    for d in range(y_true.shape[1]):
        try:
            r2s.append(r2_score(y_true[:, d], y_hat[:, d]))
        except Exception:
            r2s.append(float("nan"))

    return {
        "mse": float(np.mean(losses)),
        "r2_mean": float(np.nanmean(r2s)),
    }


# =========================================================
# Training helpers
# =========================================================
def train_cls_model(
    cfg: HeadTrainConfig,
    z_tr: np.ndarray, y_tr: np.ndarray,
    z_va: np.ndarray, y_va: np.ndarray,
    z_te: np.ndarray, y_te: np.ndarray,
    n_classes: int,
) -> Dict[str, float]:
    model = ClsModel(
        d_in=z_tr.shape[1],
        n_classes=n_classes,
        hidden=cfg.hidden,
        dropout=cfg.dropout,
    ).to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(FrozenEmbeddingClsDataset(z_tr, y_tr), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(FrozenEmbeddingClsDataset(z_va, y_va), batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(FrozenEmbeddingClsDataset(z_te, y_te), batch_size=cfg.batch_size, shuffle=False)

    best_val = math.inf
    best_state = None
    bad = 0

    for epoch in range(cfg.epochs):
        model.train()
        for z, y in train_loader:
            z = z.to(cfg.device)
            y = y.to(cfg.device)

            logits = model(z)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        val_stats = eval_cls(model, val_loader, cfg.device)
        val_loss = val_stats["cls_loss"]

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= cfg.early_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return eval_cls(model, test_loader, cfg.device)


def train_reg_model(
    cfg: HeadTrainConfig,
    z_tr: np.ndarray, y_tr: np.ndarray,
    z_va: np.ndarray, y_va: np.ndarray,
    z_te: np.ndarray, y_te: np.ndarray,
) -> Dict[str, float]:
    model = RegModel(
        d_in=z_tr.shape[1],
        d_out=y_tr.shape[1],
        hidden=cfg.hidden,
        dropout=cfg.dropout,
    ).to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(FrozenEmbeddingRegDataset(z_tr, y_tr), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(FrozenEmbeddingRegDataset(z_va, y_va), batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(FrozenEmbeddingRegDataset(z_te, y_te), batch_size=cfg.batch_size, shuffle=False)

    best_val = math.inf
    best_state = None
    bad = 0

    for epoch in range(cfg.epochs):
        model.train()
        for z, y in train_loader:
            z = z.to(cfg.device)
            y = y.to(cfg.device)

            pred = model(z)
            loss = F.mse_loss(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        val_stats = eval_reg(model, val_loader, cfg.device)
        val_loss = val_stats["mse"]

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= cfg.early_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return eval_reg(model, test_loader, cfg.device)


def train_three_heads(
    cfg: HeadTrainConfig,
    z_tr: np.ndarray, z_va: np.ndarray, z_te: np.ndarray,
    y_cls_tr: np.ndarray, y_cls_va: np.ndarray, y_cls_te: np.ndarray,
    y_box_tr: np.ndarray, y_box_va: np.ndarray, y_box_te: np.ndarray,
    y_pos_tr: np.ndarray, y_pos_va: np.ndarray, y_pos_te: np.ndarray,
) -> Dict[str, float]:
    n_classes = int(np.max(np.concatenate([y_cls_tr, y_cls_va, y_cls_te]))) + 1

    cls_stats = train_cls_model(cfg, z_tr, y_cls_tr, z_va, y_cls_va, z_te, y_cls_te, n_classes)
    box_stats = train_reg_model(cfg, z_tr, y_box_tr, z_va, y_box_va, z_te, y_box_te)
    pos_stats = train_reg_model(cfg, z_tr, y_pos_tr, z_va, y_pos_va, z_te, y_pos_te)

    return {
        "cls_loss": cls_stats["cls_loss"],
        "acc": cls_stats["acc"],
        "f1_macro": cls_stats["f1_macro"],
        "box_mse": box_stats["mse"],
        "r2_box_mean": box_stats["r2_mean"],
        "pos_mse": pos_stats["mse"],
        "r2_pos_mean": pos_stats["r2_mean"],
    }


# =========================================================
# Embedding helpers
# =========================================================
def standardize_z(
    z_tr: np.ndarray,
    z_va: np.ndarray,
    z_te: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = z_tr.mean(axis=0, keepdims=True)
    sigma = z_tr.std(axis=0, keepdims=True)
    sigma = np.maximum(sigma, eps)

    return (
        ((z_tr - mu) / sigma).astype(np.float32),
        ((z_va - mu) / sigma).astype(np.float32),
        ((z_te - mu) / sigma).astype(np.float32),
    )


def load_deepset_embeddings(path: str) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    if isinstance(z, np.ndarray) and z.dtype == object:
        z = z.item()
        if isinstance(z, dict):
            for key in ["embeddings", "z", "deepset", "deepset_embeddings"]:
                if key in z:
                    return np.asarray(z[key], dtype=np.float32)
    return np.asarray(z, dtype=np.float32)


def build_custom_objects(embedding_dimension: int, margin: float) -> Dict[str, object]:
    return {
        "loss_fn": keras_batch_all_triplet_continuous_loss_multimodal(
            margin=margin,
            emb_dim=embedding_dimension,
        ),
        "keras_batch_all_triplet_continuous_loss_multimodal":
            keras_batch_all_triplet_continuous_loss_multimodal,
        "ssim_loss": ssim_loss,
    }


def extract_mmctl_embeddings_for_indices(
    checkpoint_path: str,
    rgb: np.ndarray,
    thermal: np.ndarray,
    embedding_dimension: int,
    margin: float,
    batch_size: int = 64,
) -> np.ndarray:
    custom_objects = build_custom_objects(embedding_dimension, margin)

    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects=custom_objects,
        compile=False,
    )

    outputs = model.predict(
        [rgb.astype(np.float32), thermal.astype(np.float32)],
        batch_size=batch_size,
        verbose=0,
    )

    # model outputs: [joint_embedding, rgb_recon, thermal_recon]
    joint_embedding = outputs[0]
    return np.asarray(joint_embedding, dtype=np.float32)


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_npz", type=str, default="data.npz")
    parser.add_argument("--mmctl_dir", type=str, default="outputs/mmctl")
    parser.add_argument("--deepset_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="outputs/taskheads")

    parser.add_argument("--embedding_dimension", type=int, default=1024)
    parser.add_argument("--margin", type=float, default=0.1)

    parser.add_argument("--head_epochs", type=int, default=300)
    parser.add_argument("--head_bs", type=int, default=64)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--head_wd", type=float, default=1e-4)
    parser.add_argument("--head_hidden", type=int, default=0)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--head_patience", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=0.0)

    parser.add_argument("--test_size", type=float, default=0.5)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    safe_makedirs(args.out_dir)

    split_cfg = SplitConfig(test_size=args.test_size, runs=args.runs, seed=args.seed)
    head_cfg = HeadTrainConfig(
        epochs=args.head_epochs,
        batch_size=args.head_bs,
        lr=args.head_lr,
        weight_decay=args.head_wd,
        hidden=args.head_hidden,
        dropout=args.head_dropout,
        early_patience=args.head_patience,
        grad_clip=args.grad_clip,
    )

    # -----------------------------------------------------
    # Load data and targets
    # -----------------------------------------------------
    f = np.load(args.data_npz, allow_pickle=True)

    rgb = f["rgb"].astype(np.float32)
    thermal = f["thermal"].astype(np.float32)
    y_cls_raw = f["class_label"]
    y_box = f["box_label"].astype(np.float32)
    y_box = np.clip(y_box, None, 300)

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

    y_triplet_box_label, box_feat = generate_triplet_box_label_normalized(y_box, y_cls_raw, n_clusters=3)
    box_loc = y_box_cenwh[:, :2].astype(np.float32)
    cls_int = LabelEncoder().fit_transform(y_cls_raw).astype(np.int64)

    composite_label = [f"{c}_{t}" for c, t in zip(cls_int, y_triplet_box_label)]
    composite_label_encoded = LabelEncoder().fit_transform(composite_label)

    # -----------------------------------------------------
    # Train/test split
    # -----------------------------------------------------
    idx_all = np.arange(num_samples)
    idx_trainval, idx_test, cls_trainval, _ = train_test_split(
        idx_all,
        cls_int,
        test_size=split_cfg.test_size,
        stratify=composite_label_encoded,
        random_state=split_cfg.seed,
    )

    skf = StratifiedKFold(n_splits=split_cfg.runs, shuffle=True, random_state=split_cfg.seed)
    fold_splits = []
    for fold_id, (tr, va) in enumerate(skf.split(idx_trainval, cls_trainval), start=1):
        fold_splits.append((fold_id, idx_trainval[tr], idx_trainval[va]))

    # -----------------------------------------------------
    # If user passed DeepSet embeddings, use them once
    # Otherwise use MMCTL checkpoints per run
    # -----------------------------------------------------
    user_deepset_all = None
    if args.deepset_path is not None:
        user_deepset_all = load_deepset_embeddings(args.deepset_path)
        if user_deepset_all.shape[0] != num_samples:
            raise ValueError(
                f"DeepSet embedding count does not match dataset size: "
                f"{user_deepset_all.shape[0]} vs {num_samples}"
            )

    results: List[Dict[str, object]] = []

    for run_id, idx_tr, idx_va in fold_splits:
        print(f"\n========== run {run_id:02d}/{split_cfg.runs} ==========")

        y_cls_tr = cls_int[idx_tr]
        y_cls_va = cls_int[idx_va]
        y_cls_te = cls_int[idx_test]

        y_box_tr = box_feat[idx_tr]
        y_box_va = box_feat[idx_va]
        y_box_te = box_feat[idx_test]

        y_pos_tr = box_loc[idx_tr]
        y_pos_va = box_loc[idx_va]
        y_pos_te = box_loc[idx_test]

        if user_deepset_all is not None:
            z_tr = user_deepset_all[idx_tr]
            z_va = user_deepset_all[idx_va]
            z_te = user_deepset_all[idx_test]
            setup_name = "deepset_user"
        else:
            checkpoint_path = os.path.join(args.mmctl_dir, f"run_{run_id}", "best_model.h5")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Missing MMCTL checkpoint: {checkpoint_path}")

            z_tr = extract_mmctl_embeddings_for_indices(
                checkpoint_path=checkpoint_path,
                rgb=rgb[idx_tr],
                thermal=thermal[idx_tr],
                embedding_dimension=args.embedding_dimension,
                margin=args.margin,
                batch_size=args.head_bs,
            )
            z_va = extract_mmctl_embeddings_for_indices(
                checkpoint_path=checkpoint_path,
                rgb=rgb[idx_va],
                thermal=thermal[idx_va],
                embedding_dimension=args.embedding_dimension,
                margin=args.margin,
                batch_size=args.head_bs,
            )
            z_te = extract_mmctl_embeddings_for_indices(
                checkpoint_path=checkpoint_path,
                rgb=rgb[idx_test],
                thermal=thermal[idx_test],
                embedding_dimension=args.embedding_dimension,
                margin=args.margin,
                batch_size=args.head_bs,
            )
            setup_name = "mmctl"

        z_tr, z_va, z_te = standardize_z(z_tr, z_va, z_te)

        stats = train_three_heads(
            cfg=head_cfg,
            z_tr=z_tr, z_va=z_va, z_te=z_te,
            y_cls_tr=y_cls_tr, y_cls_va=y_cls_va, y_cls_te=y_cls_te,
            y_box_tr=y_box_tr, y_box_va=y_box_va, y_box_te=y_box_te,
            y_pos_tr=y_pos_tr, y_pos_va=y_pos_va, y_pos_te=y_pos_te,
        )

        row = {
            "run": run_id,
            "setup": setup_name,
            **stats,
        }
        results.append(row)

        print(
            f"[run{run_id:02d}][{setup_name}] "
            f"acc={row['acc']:.4f}  "
            f"f1={row['f1_macro']:.4f}  "
            f"r2_box={row['r2_box_mean']:.4f}  "
            f"r2_pos={row['r2_pos_mean']:.4f}"
        )

    out_csv = os.path.join(args.out_dir, "taskhead_results.csv")
    out_json = os.path.join(args.out_dir, "taskhead_results.json")

    keys = sorted(results[0].keys()) if results else []
    with open(out_csv, "w") as fp:
        fp.write(",".join(keys) + "\n")
        for row in results:
            fp.write(",".join(str(row.get(k, "")) for k in keys) + "\n")

    with open(out_json, "w") as fp:
        json.dump(results, fp, indent=2)

    print(f"\nSaved:\n  {out_csv}\n  {out_json}")


if __name__ == "__main__":
    main()