#!/usr/bin/env python3
"""
Loss-component ablation for MMCTL (advisor #4).

The MMCTL objective has two terms: the margin-free adaptive continuous triplet
(L_mm_adaptive) and the pairwise distance-alignment regularizer (L_mm_reg).
We ablate them by training the shared latent space with (a) the full loss,
(b) only L_mm_adaptive, and (c) only L_mm_reg, then evaluating latent structure
and downstream probes on the resulting joint embeddings.

Uses the NEON torch proxy encoders (same as scripts/reproduce_neon_ae_mmctl.py)
so the ablation runs without the heavy Keras reconstruction pipeline.

Outputs:
  results/ablation_loss_components_neon.json / .md

Run:
  .venv_exp/bin/python -u scripts/ablation_loss_components_neon.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.reproduce_neon_ae_mmctl import (  # noqa: E402
    Tiny, Spec, cos_rows, build_ann, norm_in, struct, l2,
)
from mmctl.utils import generate_triplet_box_label_normalized  # noqa: E402

from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import accuracy_score, r2_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.multioutput import MultiOutputRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def probe(z, yc, yb, yh, seed=42):
    """5-fold LogReg (cls) + Ridge (box/height) probe; higher is better."""
    yb, yh = np.nan_to_num(yb), np.nan_to_num(yh)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, r2b, r2h = [], [], []
    for tr, te in skf.split(z, yc):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        clf.fit(z[tr], yc[tr]); accs.append(accuracy_score(yc[te], clf.predict(z[te])))
        for Y, bucket in ((yb, r2b), (yh, r2h)):
            Y2 = Y[:, :2] if Y.ndim == 2 and Y.shape[1] >= 2 else Y.reshape(len(Y), -1)[:, :1]
            reg = make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(1.0)))
            reg.fit(z[tr], Y2); pred = reg.predict(z[te])
            scores = [r2_score(Y2[te, d], pred[:, d]) for d in range(Y2.shape[1])
                      if np.std(Y2[te, d]) >= 1e-8]
            bucket.append(float(np.mean(scores)) if scores else 0.0)
    return {"acc_pct": 100 * float(np.mean(accs)),
            "r2_box": float(np.mean(r2b)), "r2_height": float(np.mean(r2h))}

MAIN = "/Users/meilunzhou/Documents/GitHub/MMCTL"
NEON = os.path.join(MAIN, "neon_data.npz")
EMB_DIM, BS, MM_EP, RUNS, SEED = 128, 64, 20, 3, 42
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def ctl_loss_weighted(ann, zr, zh, zl, w_trip, w_reg, n=512):
    """Continuous triplet (adaptive) + distance-alignment reg, with term weights."""
    z = torch.cat([zr, zh, zl], 0)
    y = torch.cat([ann, ann, ann], 0)
    nall = z.shape[0]
    a = torch.randint(0, nall, (n,), device=z.device)
    p = torch.randint(0, nall, (n,), device=z.device)
    q = torch.randint(0, nall, (n,), device=z.device)
    dyap, dyaq = cos_rows(y[a], y[p]), cos_rows(y[a], y[q])
    m = dyap < dyaq
    if m.sum() == 0:
        return z.new_zeros(())
    a, p, q, dyap, dyaq = a[m], p[m], q[m], dyap[m], dyaq[m]
    dzap, dzaq = cos_rows(z[a], z[p]), cos_rows(z[a], z[q])
    trip = (dzap ** 2 - dzaq ** 2 + dyaq - dyap).clamp_min(0).mean()      # L_mm_adaptive
    reg = ((dzap - dyap) ** 2 + (dzaq - dyaq) ** 2).mean()               # L_mm_reg
    return w_trip * trip + w_reg * reg


class Trio(torch.nn.Module):
    def __init__(self, hsi_bands):
        super().__init__()
        self.rgb = Tiny(3, EMB_DIM)
        self.hsi = Spec(hsi_bands, EMB_DIM)
        self.lid = Tiny(1, EMB_DIM)

    def forward(self, r, h, l):
        zr, zh, zl = self.rgb.encode(r), self.hsi.encode(h), self.lid.encode(l)
        return torch.cat([zr, zh, zl], 1), zr, zh, zl


def train(rgb, hsi, lid, ann, tr, w_trip, w_reg, seed):
    torch.manual_seed(seed)
    model = Trio(hsi.shape[-1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rt = torch.from_numpy(rgb[tr]).float(); ht = torch.from_numpy(hsi[tr]).float()
    lt = torch.from_numpy(lid[tr]).float(); at = torch.from_numpy(ann[tr]).float()
    model.train()
    for ep in range(MM_EP):
        perm = np.random.permutation(len(tr))
        for i in range(0, len(tr), BS):
            b = perm[i:i + BS]
            if len(b) < 8:
                continue
            opt.zero_grad(set_to_none=True)
            _, zr, zh, zl = model(rt[b].to(DEVICE), ht[b].to(DEVICE), lt[b].to(DEVICE))
            loss = ctl_loss_weighted(at[b].to(DEVICE), zr, zh, zl, w_trip, w_reg)
            loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def joint(model, rgb, hsi, lid):
    parts = []
    for i in range(0, len(rgb), 128):
        sl = slice(i, i + 128)
        z, _, _, _ = model(torch.from_numpy(rgb[sl]).float().to(DEVICE),
                           torch.from_numpy(hsi[sl]).float().to(DEVICE),
                           torch.from_numpy(lid[sl]).float().to(DEVICE))
        parts.append(z.cpu().numpy())
    return np.concatenate(parts, 0)


def main():
    f = np.load(NEON, allow_pickle=True)
    rgb0, hsi0, lid0 = f["rgb"], f["hsi"], f["lidar"]
    yc = LabelEncoder().fit_transform(f["class_labels"]).astype(np.int64)
    yb_raw = f["box_labels"].astype(np.float32)
    yh = f["height_info"].astype(np.float32)
    ys = f["string_label"] if "string_label" in f.files else yc.astype(str)
    _, box = generate_triplet_box_label_normalized(yb_raw, ys, n_clusters=3)
    box = box.astype(np.float32)

    variants = {
        "full (adaptive + reg)": (1.0, 1.0),
        "adaptive only":         (1.0, 0.0),
        "reg only":              (0.0, 1.0),
    }
    agg = {k: {"acc": [], "r2_box": [], "r2_height": [], "align_cls": []} for k in variants}

    for run in range(1, RUNS + 1):
        seed = SEED + run - 1
        keep = np.isin(yc, np.where(np.bincount(yc) >= 2)[0])
        idx = np.arange(len(yc))[keep]
        tr, te = train_test_split(idx, test_size=0.5, stratify=yc[keep], random_state=seed)
        rgb, hsi, lid = norm_in(rgb0, hsi0, lid0, tr)
        ann = build_ann(yc, box, yh, tr)
        for name, (wt, wr) in variants.items():
            model = train(rgb, hsi, lid, ann, tr, wt, wr, seed)
            z = joint(model, rgb, hsi, lid)
            s = struct(z[te], yc[te], box[te], yh[te], seed)
            p = probe(l2(z[te].astype(np.float32)), yc[te], box[te], yh[te], seed)
            agg[name]["acc"].append(p["acc_pct"])
            agg[name]["r2_box"].append(p["r2_box"])
            agg[name]["r2_height"].append(p["r2_height"])
            agg[name]["align_cls"].append(s["align_cls"])
            print(f"  run{run} {name:<22} acc={p['acc_pct']:.1f} "
                  f"B={p['r2_box']:.3f} H={p['r2_height']:.3f} align_cls={s['align_cls']:.3f}",
                  flush=True)

    def ms(v):
        v = np.array(v, float)
        return float(v.mean()), float(v.std())

    results = {k: {m: ms(agg[k][m]) for m in agg[k]} for k in variants}
    out = {"protocol": (f"NEON torch-proxy MMCTL ({DEVICE}); emb={EMB_DIM}, mm_ep={MM_EP}, "
                        f"runs={RUNS}; loss-term ablation on L_mm_adaptive / L_mm_reg."),
           "results": results}
    with open(os.path.join(ROOT, "results", "ablation_loss_components_neon.json"), "w") as fp:
        json.dump(out, fp, indent=2)
    lines = ["# Loss-component ablation (NEON proxy, advisor #4)", "", out["protocol"], "",
             "| Loss | Acc % | Box R² | Height R² | Align-cls (lower=better) |",
             "|---|---|---|---|---|"]
    for k in variants:
        r = results[k]
        lines.append(f"| {k} | {r['acc'][0]:.1f}$\\pm${r['acc'][1]:.1f} | "
                     f"{r['r2_box'][0]:.3f} | {r['r2_height'][0]:.3f} | {r['align_cls'][0]:.3f} |")
    with open(os.path.join(ROOT, "results", "ablation_loss_components_neon.md"), "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print("wrote results/ablation_loss_components_neon.{json,md}", flush=True)


if __name__ == "__main__":
    main()
