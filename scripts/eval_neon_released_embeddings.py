#!/usr/bin/env python3
"""
Evaluate released full-set NEON embeddings + stronger fusion baselines.

Embeddings (N=592, train+test together):
  neon_rgb_all_6.npy, neon_hsi_all_6.npy, neon_lidar_all_6.npy
  neon_mmctl_all_6_1.0.npy   <-- MMCTL at class weight 1.0

Index mask: top-10 species + RGB 60x60 crop feasible from box_labels*10.

Baselines (DeepSet removed): Concat, SupCon, CrossAttn, Transformer, MMCTL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)

from mmctl.embedding_metrics import (  # noqa: E402
    embedding_annotation_alignment_ratio,
    local_smoothness_std,
)
from mmctl.train_taskheads import (  # noqa: E402
    HeadTrainConfig,
    standardize_z,
    train_three_heads,
)
from mmctl.utils import generate_triplet_box_label_normalized  # noqa: E402

PAPER_STRUCTURE = {
    "RGB": {"C": (-1.66, 0.88), "B": (-0.34, 0.87), "H": (0.48, 0.97)},
    "HSI": {"C": (0.72, 0.11), "B": (0.94, 0.56), "H": (1.49, 0.23)},
    "LiDAR": {"C": (2.60, 0.08), "B": (1.60, 0.72), "H": (-0.35, 0.17)},
    "Concat": {"C": (0.39, 0.15), "B": (0.19, 0.22), "H": (0.36, 0.05)},
    "SupCon": {"C": (float("nan"), float("nan")), "B": (float("nan"), float("nan")), "H": (float("nan"), float("nan"))},
    "CrossAttn": {"C": (float("nan"), float("nan")), "B": (float("nan"), float("nan")), "H": (float("nan"), float("nan"))},
    "Transformer": {"C": (float("nan"), float("nan")), "B": (float("nan"), float("nan")), "H": (float("nan"), float("nan"))},
    "MMCTL": {"C": (-1.49, 0.23), "B": (-2.07, 0.29), "H": (-1.58, 0.25)},
}
PAPER_DOWN = {
    "RGB": {"C": (74.59, 1.90), "B": (0.07, 0.15), "H": (0.34, 0.02)},
    "HSI": {"C": (76.69, 1.39), "B": (0.12, 0.08), "H": (0.40, 0.05)},
    "LiDAR": {"C": (49.66, 1.15), "B": (0.22, 0.01), "H": (0.68, 0.02)},
    "Concat": {"C": (81.69, 1.08), "B": (0.39, 0.03), "H": (0.69, 0.02)},
    "SupCon": {"C": (float("nan"), float("nan")), "B": (float("nan"), float("nan")), "H": (float("nan"), float("nan"))},
    "CrossAttn": {"C": (float("nan"), float("nan")), "B": (float("nan"), float("nan")), "H": (float("nan"), float("nan"))},
    "Transformer": {"C": (float("nan"), float("nan")), "B": (float("nan"), float("nan")), "H": (float("nan"), float("nan"))},
    "MMCTL": {"C": (86.95, 1.7), "B": (0.43, 0.04), "H": (0.74, 0.02)},
}

ORDER = ["RGB", "HSI", "LiDAR", "Concat", "SupCon", "CrossAttn", "Transformer", "MMCTL"]


def l2(z: np.ndarray) -> np.ndarray:
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def neon_index_592(npz_path: str) -> np.ndarray:
    """Top-10 species + box center*10 allows 60x60 RGB crop in 110x110."""
    d = np.load(npz_path, allow_pickle=True)
    yp = d["box_labels"].astype(np.float64)
    ys = LabelEncoder().fit_transform(d["string_label"])
    counts = np.bincount(ys)
    top = np.argsort(counts)[::-1][:10]
    keep_cls = np.isin(ys, top)
    H, W = d["rgb"].shape[1:3]
    cx = (yp[:, 0] + yp[:, 1]) / 2.0 * 10.0
    cy = (yp[:, 2] + yp[:, 3]) / 2.0 * 10.0
    ok = (cx - 30 >= 0) & (cy - 30 >= 0) & (cx + 30 <= W) & (cy + 30 <= H)
    return np.where(keep_cls & ok)[0]


def structure_raw(z, y_cls, y_box, y_h, seed=42):
    z = l2(z.astype(np.float32))
    return {
        "align_cls": embedding_annotation_alignment_ratio(z, y_cls, "cls", seed=seed)["align_ratio"],
        "align_box": embedding_annotation_alignment_ratio(z, y_box, "cont", seed=seed)["align_ratio"],
        "align_height": embedding_annotation_alignment_ratio(z, y_h, "cont", seed=seed)["align_ratio"],
        "smooth_cls": local_smoothness_std(z, y_cls, "cls", ks=(3,), seed=seed)["smooth_std_k3"],
        "smooth_box": local_smoothness_std(z, y_box, "cont", ks=(3,), seed=seed)["smooth_std_k3"],
        "smooth_height": local_smoothness_std(z, y_h, "cont", ks=(3,), seed=seed)["smooth_std_k3"],
    }


def zscore_struct(rows: List[dict]) -> List[dict]:
    out = []
    for run in sorted({r["run"] for r in rows}):
        sub = [dict(r) for r in rows if r["run"] == run]
        for src in ("cls", "box", "height"):
            a = np.array([r[f"align_{src}"] for r in sub], float)
            s = np.array([r[f"smooth_{src}"] for r in sub], float)
            za = (a - a.mean()) / (a.std() + 1e-12)
            zs = (s - s.mean()) / (s.std() + 1e-12)
            for i in range(len(sub)):
                sub[i][f"score_{src}"] = float(za[i] + zs[i])
        out.extend(sub)
    return out


def paper_task_heads(z_tr, z_va, z_te, y_cls_tr, y_cls_va, y_cls_te, y_box_tr, y_box_va, y_box_te, y_h_tr, y_h_va, y_h_te, head_cfg):
    y_box_tr = np.nan_to_num(y_box_tr).astype(np.float32)
    y_box_va = np.nan_to_num(y_box_va).astype(np.float32)
    y_box_te = np.nan_to_num(y_box_te).astype(np.float32)
    y_h_tr = np.nan_to_num(y_h_tr).astype(np.float32)
    y_h_va = np.nan_to_num(y_h_va).astype(np.float32)
    y_h_te = np.nan_to_num(y_h_te).astype(np.float32)
    z_tr, z_va, z_te = standardize_z(l2(z_tr), l2(z_va), l2(z_te))
    stats = train_three_heads(
        cfg=head_cfg,
        z_tr=z_tr,
        z_va=z_va,
        z_te=z_te,
        y_cls_tr=y_cls_tr,
        y_cls_va=y_cls_va,
        y_cls_te=y_cls_te,
        y_box_tr=y_box_tr,
        y_box_va=y_box_va,
        y_box_te=y_box_te,
        y_pos_tr=y_h_tr,
        y_pos_va=y_h_va,
        y_pos_te=y_h_te,
    )
    return {
        "acc_pct": 100.0 * float(stats["acc"]),
        "r2_box": float(stats["r2_box_mean"]),
        "r2_height": float(stats["r2_pos_mean"]),
    }


# -------------------- fusion baselines --------------------
class SupConProjector(nn.Module):
    def __init__(self, d_in: int, d_proj: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(inplace=True),
            nn.Linear(d_in, d_proj),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def supcon_loss(z: torch.Tensor, y: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al.)."""
    device = z.device
    y = y.view(-1, 1)
    mask = torch.eq(y, y.T).float().to(device)
    logits = z @ z.T / temp
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
    pos = mask.sum(1)
    mean_log_prob = (mask * log_prob).sum(1) / pos.clamp_min(1.0)
    return -(mean_log_prob[pos > 0]).mean()


class CrossAttnFusion(nn.Module):
    def __init__(self, d: int = 1024, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, d)

    def forward(self, mods: torch.Tensor) -> torch.Tensor:
        # mods: (B, 3, D)
        q = mods.mean(dim=1, keepdim=True)
        h, _ = self.attn(q, mods, mods)
        h = self.norm(h.squeeze(1) + q.squeeze(1))
        return self.out(h)


class TransformerFusion(nn.Module):
    def __init__(self, d: int = 1024, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=2 * d, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, mods: torch.Tensor) -> torch.Tensor:
        b = mods.size(0)
        tok = torch.cat([self.cls.expand(b, -1, -1), mods], dim=1)
        h = self.enc(tok)
        return h[:, 0]


class TaskFusionHead(nn.Module):
    """Trainable fusion + shallow multitask heads (task-driven baseline)."""

    def __init__(self, fusion: nn.Module, d: int, n_classes: int, d_box: int, d_h: int):
        super().__init__()
        self.fusion = fusion
        self.cls = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, n_classes))
        self.box = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, d_box))
        self.hgt = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, d_h))

    def encode(self, mods: torch.Tensor) -> torch.Tensor:
        return self.fusion(mods)

    def forward(self, mods: torch.Tensor):
        z = self.encode(mods)
        return z, self.cls(z), self.box(z), self.hgt(z)


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_supcon(
    z_cat_tr: np.ndarray,
    y_tr: np.ndarray,
    z_cat_va: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns projected embeddings for tr/va given full arrays passed later via apply."""
    torch.manual_seed(seed)
    device = _device()
    model = SupConProjector(z_cat_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(torch.from_numpy(z_cat_tr), torch.from_numpy(y_tr.astype(np.int64)))
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, drop_last=len(ds) > batch_size)
    best, bad, state = float("inf"), 0, None
    z_va_t = torch.from_numpy(z_cat_va).to(device)
    for ep in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            if len(yb) < 2:
                continue
            xb, yb = xb.to(device), yb.to(device)
            loss = supcon_loss(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            # proxy val: class-center compactness
            zv = model(z_va_t)
            # use train loss EMA as monitor
            vloss = float(np.mean(losses)) if losses else 0.0
        if vloss < best:
            best, bad = vloss, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= max(20, epochs // 10):
            break
    if state:
        model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def apply_supcon(model: nn.Module, z: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    outs = []
    for i in range(0, len(z), 256):
        outs.append(model(torch.from_numpy(z[i : i + 256]).to(device)).cpu().numpy())
    return np.concatenate(outs, 0)


def train_task_fusion(
    fusion_kind: str,
    mods_tr: np.ndarray,
    mods_va: np.ndarray,
    y_cls_tr,
    y_cls_va,
    y_box_tr,
    y_box_va,
    y_h_tr,
    y_h_va,
    epochs: int,
    batch_size: int,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    device = _device()
    d = mods_tr.shape[-1]
    n_classes = int(max(y_cls_tr.max(), y_cls_va.max())) + 1
    if fusion_kind == "CrossAttn":
        fusion = CrossAttnFusion(d=d)
    elif fusion_kind == "Transformer":
        fusion = TransformerFusion(d=d)
    else:
        raise ValueError(fusion_kind)
    model = TaskFusionHead(fusion, d, n_classes, y_box_tr.shape[1], y_h_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = TensorDataset(
        torch.from_numpy(mods_tr),
        torch.from_numpy(y_cls_tr.astype(np.int64)),
        torch.from_numpy(y_box_tr.astype(np.float32)),
        torch.from_numpy(y_h_tr.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)
    mods_va_t = torch.from_numpy(mods_va).to(device)
    y_va = torch.from_numpy(y_cls_va.astype(np.int64)).to(device)
    yb_va = torch.from_numpy(y_box_va.astype(np.float32)).to(device)
    yh_va = torch.from_numpy(y_h_va.astype(np.float32)).to(device)
    best, bad, state = float("inf"), 0, None
    for ep in range(epochs):
        model.train()
        for xb, yc, yb, yh in loader:
            xb, yc, yb, yh = xb.to(device), yc.to(device), yb.to(device), yh.to(device)
            _, logit, pb, ph = model(xb)
            loss = F.cross_entropy(logit, yc) + F.mse_loss(pb, yb) + F.mse_loss(ph, yh)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            _, logit, pb, ph = model(mods_va_t)
            vloss = float(
                (
                    F.cross_entropy(logit, y_va) + F.mse_loss(pb, yb_va) + F.mse_loss(ph, yh_va)
                ).cpu()
            )
        if vloss < best:
            best, bad = vloss, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= max(30, epochs // 10):
            break
    if state:
        model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def apply_task_fusion(model: nn.Module, mods: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    outs = []
    for i in range(0, len(mods), 256):
        z = model.encode(torch.from_numpy(mods[i : i + 256]).to(device))
        outs.append(z.cpu().numpy())
    return np.concatenate(outs, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz_path", default="/Users/meilunzhou/Documents/GitHub/CTL/neon_data.npz")
    p.add_argument("--rgb", default=os.path.join(ROOT, "neon_rgb_all_6.npy"))
    p.add_argument("--hsi", default=os.path.join(ROOT, "neon_hsi_all_6.npy"))
    p.add_argument("--lidar", default=os.path.join(ROOT, "neon_lidar_all_6.npy"))
    p.add_argument("--mmctl", default=os.path.join(ROOT, "neon_mmctl_all_6_1.0.npy"))
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--fusion_epochs", type=int, default=300)
    p.add_argument("--head_epochs", type=int, default=3000)
    p.add_argument("--head_patience", type=int, default=1000)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.runs = 2
        args.fusion_epochs = 40
        args.head_epochs, args.head_patience = 80, 25

    idx = neon_index_592(args.npz_path)
    assert len(idx) == 592, len(idx)
    np.save(os.path.join(ROOT, "outputs", "neon_emb_index_592.npy"), idx)

    d = np.load(args.npz_path, allow_pickle=True)
    y_cls_raw = d["class_labels"][idx]
    # remap to 0..9 by frequency on this subset
    le = LabelEncoder()
    y_cls = le.fit_transform(y_cls_raw)
    y_pos = d["box_labels"][idx].astype(np.float32)
    y_box = d["box_info"][idx].astype(np.float32)
    y_h = d["height_info"][idx].astype(np.float32)
    _, box_feat = generate_triplet_box_label_normalized(y_pos, np.asarray(y_cls).astype(str), n_clusters=3)
    box_feat = box_feat.astype(np.float32)

    z_rgb = np.load(args.rgb).astype(np.float32)
    z_hsi = np.load(args.hsi).astype(np.float32)
    z_lid = np.load(args.lidar).astype(np.float32)
    z_mm = np.load(args.mmctl).astype(np.float32)
    assert z_rgb.shape[0] == 592 and z_mm.shape == (592, 3072), (z_rgb.shape, z_mm.shape)

    device = _device()
    head_cfg = HeadTrainConfig(
        epochs=args.head_epochs,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        hidden=256,
        dropout=0.1,
        early_patience=args.head_patience,
        grad_clip=5.0,
        device=device,
    )
    print(
        f"N=592 mmctl={args.mmctl} runs={args.runs} fusion_ep={args.fusion_epochs} "
        f"heads={args.head_epochs}/{args.head_patience} device={device}",
        flush=True,
    )

    # Outer protocol: 50/50 holdout, then StratifiedKFold on trainval (paper-like runs)
    (
        z_rgb_tv,
        z_rgb_te,
        z_hsi_tv,
        z_hsi_te,
        z_lid_tv,
        z_lid_te,
        z_mm_tv,
        z_mm_te,
        y_tv,
        y_te,
        box_tv,
        box_te,
        h_tv,
        h_te,
        bf_tv,
        bf_te,
    ) = train_test_split(
        z_rgb,
        z_hsi,
        z_lid,
        z_mm,
        y_cls,
        y_box,
        y_h,
        box_feat,
        test_size=0.5,
        stratify=y_cls,
        random_state=42,
    )

    skf = StratifiedKFold(n_splits=args.runs, shuffle=True, random_state=42)
    struct_rows, down_rows = [], []

    for run, (tr, va) in enumerate(skf.split(z_rgb_tv, y_tv), start=1):
        print(f"\n=== fold {run}/{args.runs} ===", flush=True)
        mods_tr = np.stack([z_rgb_tv[tr], z_hsi_tv[tr], z_lid_tv[tr]], axis=1).astype(np.float32)
        mods_va = np.stack([z_rgb_tv[va], z_hsi_tv[va], z_lid_tv[va]], axis=1).astype(np.float32)
        mods_te = np.stack([z_rgb_te, z_hsi_te, z_lid_te], axis=1).astype(np.float32)
        cat_tr = np.concatenate([z_rgb_tv[tr], z_hsi_tv[tr], z_lid_tv[tr]], 1)
        cat_va = np.concatenate([z_rgb_tv[va], z_hsi_tv[va], z_lid_tv[va]], 1)
        cat_te = np.concatenate([z_rgb_te, z_hsi_te, z_lid_te], 1)

        # SupCon on concat
        print("  train SupCon...", flush=True)
        sc_model = train_supcon(cat_tr, y_tv[tr], cat_va, args.fusion_epochs, 32, seed=40 + run)
        z_sc_tr = apply_supcon(sc_model, cat_tr)
        z_sc_va = apply_supcon(sc_model, cat_va)
        z_sc_te = apply_supcon(sc_model, cat_te)

        print("  train CrossAttn...", flush=True)
        ca = train_task_fusion(
            "CrossAttn",
            mods_tr,
            mods_va,
            y_tv[tr],
            y_tv[va],
            bf_tv[tr],
            bf_tv[va],
            h_tv[tr],
            h_tv[va],
            args.fusion_epochs,
            32,
            50 + run,
        )
        z_ca_tr = apply_task_fusion(ca, mods_tr)
        z_ca_va = apply_task_fusion(ca, mods_va)
        z_ca_te = apply_task_fusion(ca, mods_te)

        print("  train Transformer...", flush=True)
        tfm = train_task_fusion(
            "Transformer",
            mods_tr,
            mods_va,
            y_tv[tr],
            y_tv[va],
            bf_tv[tr],
            bf_tv[va],
            h_tv[tr],
            h_tv[va],
            args.fusion_epochs,
            32,
            60 + run,
        )
        z_tf_tr = apply_task_fusion(tfm, mods_tr)
        z_tf_va = apply_task_fusion(tfm, mods_va)
        z_tf_te = apply_task_fusion(tfm, mods_te)

        methods = {
            "RGB": (z_rgb_tv[tr], z_rgb_tv[va], z_rgb_te),
            "HSI": (z_hsi_tv[tr], z_hsi_tv[va], z_hsi_te),
            "LiDAR": (z_lid_tv[tr], z_lid_tv[va], z_lid_te),
            "Concat": (cat_tr, cat_va, cat_te),
            "SupCon": (z_sc_tr, z_sc_va, z_sc_te),
            "CrossAttn": (z_ca_tr, z_ca_va, z_ca_te),
            "Transformer": (z_tf_tr, z_tf_va, z_tf_te),
            "MMCTL": (z_mm_tv[tr], z_mm_tv[va], z_mm_te),
        }

        for name, (ztr, zva, zte) in methods.items():
            print(f"  eval {name}...", flush=True)
            m = structure_raw(zte, y_te, bf_te, h_te, seed=40 + run)
            struct_rows.append({"run": run, "method": name, **m})
            pr = paper_task_heads(
                ztr,
                zva,
                zte,
                y_tv[tr],
                y_tv[va],
                y_te,
                bf_tv[tr],
                bf_tv[va],
                bf_te,
                h_tv[tr],
                h_tv[va],
                h_te,
                head_cfg,
            )
            down_rows.append({"run": run, "method": name, **pr})
            print(
                f"    {name:<12} acc={pr['acc_pct']:.1f} B={pr['r2_box']:.3f} H={pr['r2_height']:.3f}",
                flush=True,
            )

    scored = zscore_struct(struct_rows)

    def aggs(method, src):
        v = [r[f"score_{src}"] for r in scored if r["method"] == method]
        return float(np.mean(v)), float(np.std(v))

    def aggd(method, key):
        v = [r[key] for r in down_rows if r["method"] == method]
        return float(np.mean(v)), float(np.std(v))

    structure = {m: {"C": aggs(m, "cls"), "B": aggs(m, "box"), "H": aggs(m, "height")} for m in ORDER}
    downstream = {
        m: {"C": aggd(m, "acc_pct"), "B": aggd(m, "r2_box"), "H": aggd(m, "r2_height")} for m in ORDER
    }

    out_json = os.path.join(ROOT, "results", "replication_neon.json")
    out_md = os.path.join(ROOT, "results", "replication_neon_delta.md")
    payload = {
        "protocol": (
            "Released full-set NEON embeddings (N=592); MMCTL=neon_mmctl_all_6_1.0.npy; "
            "independent RGB/HSI/LiDAR; baselines Concat/SupCon/CrossAttn/Transformer; "
            f"50/50 + {args.runs}-fold; paper task heads "
            f"{args.head_epochs}/{args.head_patience}"
        ),
        "data_path": args.npz_path,
        "mmctl_file": args.mmctl,
        "n": 592,
        "structure": structure,
        "downstream": downstream,
        "paper_structure": PAPER_STRUCTURE,
        "paper_downstream": PAPER_DOWN,
        "methods": ORDER,
    }
    with open(out_json, "w") as fp:
        json.dump(payload, fp, indent=2)

    lines = [
        "# NEON Replication (released embeddings, MMCTL w_cls=1.0)",
        "",
        f"N={592}; MMCTL file: `{os.path.basename(args.mmctl)}`",
        f"Protocol: {payload['protocol']}",
        "",
        "## Structure (lower better)",
        "",
        "| Method | Src | Paper | Code | Δmean |",
        "|---|---|---|---|---|",
    ]
    for m in ORDER:
        for s in ("C", "B", "H"):
            pm, ps = PAPER_STRUCTURE[m][s]
            cm, cs = structure[m][s]
            if np.isnan(pm):
                lines.append(f"| {m} | {s} | — | {cm:.2f}±{cs:.2f} | — |")
            else:
                lines.append(f"| {m} | {s} | {pm:.2f}±{ps:.2f} | {cm:.2f}±{cs:.2f} | {cm-pm:+.2f} |")
    lines += [
        "",
        "## Downstream",
        "",
        "| Method | Paper C% | Code C% | Paper B | Code B | Paper H | Code H |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in ORDER:
        pc, _ = PAPER_DOWN[m]["C"]
        pb, _ = PAPER_DOWN[m]["B"]
        ph, _ = PAPER_DOWN[m]["H"]
        cc, ccs = downstream[m]["C"]
        cb, cbs = downstream[m]["B"]
        ch, chs = downstream[m]["H"]
        pcs = f"{pc:.2f}" if not np.isnan(pc) else "—"
        pbs = f"{pb:.2f}" if not np.isnan(pb) else "—"
        phs = f"{ph:.2f}" if not np.isnan(ph) else "—"
        lines.append(
            f"| {m} | {pcs} | {cc:.2f}±{ccs:.2f} | {pbs} | {cb:.2f}±{cbs:.2f} | {phs} | {ch:.2f}±{chs:.2f} |"
        )
    with open(out_md, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print("Wrote", out_json, flush=True)
    print("Wrote", out_md, flush=True)


if __name__ == "__main__":
    main()
