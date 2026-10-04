#!/usr/bin/env python3
"""
Diagnose the low NEON Transformer-fusion accuracy (Reviewer R2.5 / advisor #5).

The reported NEON Transformer downstream accuracy (~43%) is far below RGB (~76%).
We test whether this is a configuration/tuning artifact by sweeping the readout
(CLS token vs. mean pooling), learning rate, epochs, and training objective
(multitask vs. classification-only), then probing NEON classification accuracy
with a logistic-regression head (5-fold) on the fused embeddings.

Outputs:
  results/transformer_retune_neon.json / .md

Run:
  .venv_exp/bin/python -u scripts/retune_transformer_neon.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def neon_index_592(npz_path):
    """Top-10 species + box center*10 allows a 60x60 RGB crop in 110x110.
    Inlined from scripts.eval_neon_released_embeddings to keep this script
    TensorFlow-free (so it runs in a torch-only venv on HiPerGator)."""
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


MAIN = os.environ.get("MMCTL_DIR", ROOT)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # force CPU on mac (MPS slow for small Transformer)


class TFusion(nn.Module):
    """Transformer fusion with selectable readout (cls | mean)."""
    def __init__(self, d=1024, n_heads=4, n_layers=2, readout="cls"):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads,
                                           dim_feedforward=2 * d, batch_first=True,
                                           activation="gelu", dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.readout = readout
        self.cls = nn.Parameter(torch.zeros(1, 1, d)); nn.init.normal_(self.cls, std=0.02)

    def forward(self, mods):  # mods: (B, n_mod, D)
        b = mods.size(0)
        if self.readout == "cls":
            tok = torch.cat([self.cls.expand(b, -1, -1), mods], dim=1)
            return self.enc(tok)[:, 0]
        return self.enc(mods).mean(dim=1)


class Head(nn.Module):
    def __init__(self, fusion, d, n_classes, d_box, d_h):
        super().__init__()
        self.fusion = fusion
        self.cls = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, n_classes))
        self.box = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, d_box))
        self.hgt = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, d_h))

    def forward(self, m):
        z = self.fusion(m)
        return z, self.cls(z), self.box(z), self.hgt(z)


def train_fusion(mods_tr, y_tr, box_tr, h_tr, cfg, seed):
    torch.manual_seed(seed)
    d = mods_tr.shape[-1]
    n_classes = int(y_tr.max()) + 1
    model = Head(TFusion(d, readout=cfg["readout"]), d, n_classes,
                 box_tr.shape[1], h_tr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    mt = torch.from_numpy(mods_tr).float().to(DEVICE)
    yt = torch.from_numpy(y_tr.astype(np.int64)).to(DEVICE)
    bt = torch.from_numpy(box_tr.astype(np.float32)).to(DEVICE)
    ht = torch.from_numpy(h_tr.astype(np.float32)).to(DEVICE)
    n = len(mods_tr)
    model.train()
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, 128):
            b = perm[i:i + 128]
            _, logit, pb, ph = model(mt[b])
            loss = F.cross_entropy(logit, yt[b])
            if cfg["multitask"]:
                loss = loss + F.mse_loss(pb, bt[b]) + F.mse_loss(ph, ht[b])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    model.eval()
    return model


@torch.no_grad()
def embed(model, mods):
    out = []
    for i in range(0, len(mods), 256):
        z = model.fusion(torch.from_numpy(mods[i:i + 256]).float().to(DEVICE))
        out.append(z.cpu().numpy())
    return np.concatenate(out, 0)


def probe_acc(z_tr, y_tr, z_te, y_te):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    clf.fit(z_tr, y_tr)
    return accuracy_score(y_te, clf.predict(z_te))


def main():
    d = np.load(os.path.join(MAIN, "neon_data.npz"), allow_pickle=True)
    idx = neon_index_592(os.path.join(MAIN, "neon_data.npz"))
    y = LabelEncoder().fit_transform(d["string_label"][idx]).astype(np.int64)
    box = d["box_info"][idx].astype(np.float32)
    h = d["height_info"][idx].astype(np.float32)
    z_rgb = np.load(os.path.join(MAIN, "neon_rgb_all_6.npy")).astype(np.float32)
    z_hsi = np.load(os.path.join(MAIN, "neon_hsi_all_6.npy")).astype(np.float32)
    z_lid = np.load(os.path.join(MAIN, "neon_lidar_all_6.npy")).astype(np.float32)
    mods = np.stack([z_rgb, z_hsi, z_lid], axis=1)

    configs = {
        "default (cls, lr1e-3, 300ep, multitask)": dict(readout="cls", lr=1e-3, epochs=150, multitask=True),
        "mean-pool (lr1e-3, 300ep, multitask)":    dict(readout="mean", lr=1e-3, epochs=150, multitask=True),
        "cls, lr3e-4, 600ep, multitask":           dict(readout="cls", lr=3e-4, epochs=300, multitask=True),
        "mean-pool, lr3e-4, 600ep, cls-only":      dict(readout="mean", lr=3e-4, epochs=300, multitask=False),
    }

    tv, te = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=42)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    results = {}
    for name, cfg in configs.items():
        accs = []
        for fold, (tr, va) in enumerate(skf.split(tv, y[tv]), 1):
            tri = tv[tr]
            model = train_fusion(mods[tri], y[tri], box[tri], h[tri], cfg, seed=42 + fold)
            z_tr = embed(model, mods[tri]); z_te = embed(model, mods[te])
            accs.append(probe_acc(z_tr, y[tri], z_te, y[te]))
        results[name] = (float(np.mean(accs) * 100), float(np.std(accs) * 100))
        print(f"  {name:<45} acc={results[name][0]:.1f}+/-{results[name][1]:.1f}", flush=True)

    out = {"device": DEVICE, "n": int(len(y)), "results": results,
           "note": "LogReg probe accuracy (%) on Transformer-fused NEON embeddings, 5-fold."}
    with open(os.path.join(ROOT, "results", "transformer_retune_neon.json"), "w") as fp:
        json.dump(out, fp, indent=2)
    lines = ["# NEON Transformer-fusion re-tuning (advisor #5)", "",
             out["note"], "", "| Config | Acc % |", "|---|---|"]
    for k, (m, s) in results.items():
        lines.append(f"| {k} | {m:.1f}$\\pm${s:.1f} |")
    with open(os.path.join(ROOT, "results", "transformer_retune_neon.md"), "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print("wrote results/transformer_retune_neon.{json,md}", flush=True)


if __name__ == "__main__":
    main()
