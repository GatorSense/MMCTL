#!/usr/bin/env python3
"""NEON proxy with AE baselines + separate MMCTL (CTL neon_data.npz)."""
# See results/replication_neon_delta.md after running.
# Usage:
#   PYTHONUNBUFFERED=1 .venv_exp/bin/python -u scripts/reproduce_neon_ae_mmctl.py

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from mmctl.embedding_metrics import embedding_annotation_alignment_ratio, local_smoothness_std
from mmctl.utils import generate_triplet_box_label_normalized

NEON = "data/neon_data.npz"
OUTJ = os.path.join(REPO, "results", "replication_neon.json")
OUTD = os.path.join(REPO, "results", "replication_neon_delta.md")
EMB = os.path.join(REPO, "outputs", "neon_proxy_embeddings")

PAPER_S = {
    "RGB": {"C": (-1.66, 0.88), "B": (-0.34, 0.87), "H": (0.48, 0.97)},
    "HSI": {"C": (0.72, 0.11), "B": (0.94, 0.56), "H": (1.49, 0.23)},
    "LiDAR": {"C": (2.60, 0.08), "B": (1.60, 0.72), "H": (-0.35, 0.17)},
    "Concat": {"C": (0.39, 0.15), "B": (0.19, 0.22), "H": (0.36, 0.05)},
    "DS-M": {"C": (-0.27, 0.12), "B": (0.24, 0.29), "H": (-0.27, 0.23)},
    "MMCTL": {"C": (-1.49, 0.23), "B": (-2.07, 0.29), "H": (-1.58, 0.25)},
}
PAPER_D = {
    "RGB": {"C": (74.59, 1.90), "B": (0.07, 0.15), "H": (0.34, 0.02)},
    "HSI": {"C": (76.69, 1.39), "B": (0.12, 0.08), "H": (0.40, 0.05)},
    "LiDAR": {"C": (49.66, 1.15), "B": (0.22, 0.01), "H": (0.68, 0.02)},
    "Concat": {"C": (81.69, 1.08), "B": (0.39, 0.03), "H": (0.69, 0.02)},
    "DS-M": {"C": (81.35, 1.18), "B": (0.40, 0.01), "H": (0.61, 0.04)},
    "MMCTL": {"C": (86.95, 1.7), "B": (0.43, 0.04), "H": (0.74, 0.02)},
}

EMB_DIM, BS, AE_EP, MM_EP, RUNS, SEED = 128, 64, 12, 20, 5, 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class Tiny(nn.Module):
    def __init__(self, cin, d):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(cin, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, d)

    def encode(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        return self.fc(self.enc(x).flatten(1))


class Spec(nn.Module):
    def __init__(self, nb, d):
        super().__init__()
        self.red = nn.Conv2d(nb, 32, 1)
        self.t = Tiny(32, d)

    def encode(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.red(x)).permute(0, 2, 3, 1)
        return self.t.encode(x)


def cos_rows(a, b):
    a, b = F.normalize(a, 1), F.normalize(b, 1)
    return (1 - (a * b).sum(1)).clamp_min(0)


def ctl_loss(ann, zr, zh, zl, n=512):
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
    trip = (dzap ** 2 - dzaq ** 2 + dyaq - dyap).clamp_min(0)
    reg = (dzap - dyap) ** 2 + (dzaq - dyaq) ** 2
    align = cos_rows(zr, zh).mean() + cos_rows(zr, zl).mean() + cos_rows(zh, zl).mean()
    return trip.mean() + reg.mean() + 0.1 * align


def minmax(x, tr):
    lo, hi = x[tr].min(0), x[tr].max(0)
    return (x - lo) / (hi - lo + 1e-8)


def build_ann(yc, box, h, tr):
    oh = np.eye(int(yc.max()) + 1, dtype=np.float32)[yc]
    bn, hn = minmax(box.astype(np.float32), tr), minmax(h.astype(np.float32), tr)
    L = max(oh.shape[1], bn.shape[1], hn.shape[1])
    return np.concatenate(
        [oh * (L / oh.shape[1]), bn * (L / bn.shape[1]), hn * (L / hn.shape[1])], 1
    ).astype(np.float32)


def l2(z):
    return z / (np.linalg.norm(z, 1, keepdims=True) + 1e-12)


def struct(z, yc, yb, yh, seed=42):
    z = l2(z.astype(np.float32))
    return {
        "align_cls": embedding_annotation_alignment_ratio(z, yc, "cls", seed=seed)["align_ratio"],
        "align_box": embedding_annotation_alignment_ratio(z, yb, "cont", seed=seed)["align_ratio"],
        "align_height": embedding_annotation_alignment_ratio(z, yh, "cont", seed=seed)["align_ratio"],
        "smooth_cls": local_smoothness_std(z, yc, "cls", ks=(3,), seed=seed)["smooth_std_k3"],
        "smooth_box": local_smoothness_std(z, yb, "cont", ks=(3,), seed=seed)["smooth_std_k3"],
        "smooth_height": local_smoothness_std(z, yh, "cont", ks=(3,), seed=seed)["smooth_std_k3"],
    }


def zscore(rows):
    out = []
    for run in sorted({r["run"] for r in rows}):
        sub = [dict(r) for r in rows if r["run"] == run]
        for src in ("cls", "box", "height"):
            a = np.array([r[f"align_{src}"] for r in sub])
            s = np.array([r[f"smooth_{src}"] for r in sub])
            za = (a - a.mean()) / (a.std() + 1e-12)
            zs = (s - s.mean()) / (s.std() + 1e-12)
            for i in range(len(sub)):
                sub[i][f"score_{src}"] = float(za[i] + zs[i])
        out.extend(sub)
    return out


def probe(z, yc, yb, yh, seed=42):
    yb = np.nan_to_num(yb)
    yh = np.nan_to_num(yh)
    skf = StratifiedKFold(5, True, seed)
    accs, r2b, r2h = [], [], []
    for tr, te in skf.split(z, yc):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        clf.fit(z[tr], yc[tr])
        accs.append(accuracy_score(yc[te], clf.predict(z[te])))
        for Y, bucket in ((yb, r2b), (yh, r2h)):
            Y2 = Y[:, :2] if Y.ndim == 2 and Y.shape[1] >= 2 else Y.reshape(len(Y), -1)[:, :1]
            reg = make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(1.0)))
            reg.fit(z[tr], Y2)
            pred = reg.predict(z[te])
            scores = [
                r2_score(Y2[te, d], pred[:, d])
                for d in range(Y2.shape[1])
                if np.std(Y2[te, d]) >= 1e-8
            ]
            bucket.append(float(np.mean(scores)) if scores else 0.0)
    return {
        "acc_pct": 100 * float(np.mean(accs)),
        "r2_box": float(np.mean(r2b)),
        "r2_height": float(np.mean(r2h)),
    }


def norm_in(rgb, hsi, lidar, tr):
    rgb = rgb.astype(np.float32) / 255.0
    h = np.clip(hsi.astype(np.float32), -1e4, 1e4)
    lo = h[tr].reshape(-1, h.shape[-1]).min(0)
    hi = h[tr].reshape(-1, h.shape[-1]).max(0)
    h = (h - lo) / (hi - lo + 1e-8)
    lid = lidar.astype(np.float32)
    lo, hi = lid[tr].min(), lid[tr].max()
    lid = (lid - lo) / (hi - lo + 1e-8)
    return rgb, h, lid


@torch.no_grad()
def emb(model, x):
    outs = []
    for i in range(0, len(x), 128):
        outs.append(model.encode(torch.from_numpy(x[i : i + 128]).float().to(DEVICE)).cpu().numpy())
    return np.concatenate(outs, 0)


def train_ae(model, x, tr, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.from_numpy(x[tr]).float()
    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(len(tr))
        losses = []
        for i in range(0, len(tr), BS):
            b = perm[i : i + BS]
            if len(b) < 4:
                continue
            opt.zero_grad(set_to_none=True)
            xb = xt[b].to(DEVICE)
            z = model.encode(xb)
            loss = F.mse_loss(z.mean(1), xb.reshape(len(b), -1).mean(1))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if ep % 4 == 0:
            print(f"   ae ep {ep+1}/{epochs} {np.mean(losses):.4f}", flush=True)


def train_mm(rgb_m, hsi_m, lid_m, ann, tr, epochs, seed):
    torch.manual_seed(seed)

    class Trio(nn.Module):
        def __init__(self):
            super().__init__()
            self.rgb = Tiny(3, EMB_DIM).to(DEVICE)
            self.hsi = Spec(hsi_m.shape[-1], EMB_DIM).to(DEVICE)
            self.lid = Tiny(1, EMB_DIM).to(DEVICE)

        def forward(self, r, h, l):
            zr, zh, zl = self.rgb.encode(r), self.hsi.encode(h), self.lid.encode(l)
            return torch.cat([zr, zh, zl], 1), zr, zh, zl

    model = Trio()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rt = torch.from_numpy(rgb_m[tr]).float()
    ht = torch.from_numpy(hsi_m[tr]).float()
    lt = torch.from_numpy(lid_m[tr]).float()
    at = torch.from_numpy(ann[tr]).float()
    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(len(tr))
        losses = []
        for i in range(0, len(tr), BS):
            b = perm[i : i + BS]
            if len(b) < 8:
                continue
            opt.zero_grad(set_to_none=True)
            _, zr, zh, zl = model(rt[b].to(DEVICE), ht[b].to(DEVICE), lt[b].to(DEVICE))
            loss = ctl_loss(at[b].to(DEVICE), zr, zh, zl)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"   mm ep {ep+1}/{epochs} {np.mean(losses):.4f}", flush=True)
    return model


def main():
    os.makedirs(EMB, exist_ok=True)
    print(f"NEON={NEON} device={DEVICE}", flush=True)
    f = np.load(NEON, allow_pickle=True)
    rgb0, hsi0, lid0 = f["rgb"], f["hsi"], f["lidar"]
    yc = LabelEncoder().fit_transform(f["class_labels"]).astype(np.int64)
    yb_raw = f["box_labels"].astype(np.float32)
    yh = f["height_info"].astype(np.float32)
    ys = f["string_label"] if "string_label" in f.files else yc.astype(str)
    _, box = generate_triplet_box_label_normalized(yb_raw, ys, n_clusters=3)
    box = box.astype(np.float32)

    srows, drows = [], []
    for run in range(1, RUNS + 1):
        seed = SEED + run - 1
        print(f"\n=== run {run}/{RUNS} ===", flush=True)
        keep = np.isin(yc, np.where(np.bincount(yc) >= 2)[0])
        idx = np.arange(len(yc))[keep]
        tr, te = train_test_split(idx, test_size=0.5, stratify=yc[keep], random_state=seed)
        rgb, hsi, lid = norm_in(rgb0, hsi0, lid0, tr)
        ann = build_ann(yc, box, yh, tr)

        rgb_ae = Tiny(3, EMB_DIM).to(DEVICE)
        hsi_ae = Spec(hsi.shape[-1], EMB_DIM).to(DEVICE)
        lid_ae = Tiny(1, EMB_DIM).to(DEVICE)
        print("  AE pretrain", flush=True)
        train_ae(rgb_ae, rgb, tr, AE_EP)
        train_ae(hsi_ae, hsi, tr, AE_EP)
        train_ae(lid_ae, lid, tr, AE_EP)
        zr, zh, zl = emb(rgb_ae, rgb), emb(hsi_ae, hsi), emb(lid_ae, lid)

        print("  MMCTL train", flush=True)
        mm = train_mm(rgb, hsi, lid, ann, tr, MM_EP, seed)
        parts = []
        with torch.no_grad():
            for i in range(0, len(rgb), 128):
                sl = slice(i, i + 128)
                z, _, _, _ = mm(
                    torch.from_numpy(rgb[sl]).float().to(DEVICE),
                    torch.from_numpy(hsi[sl]).float().to(DEVICE),
                    torch.from_numpy(lid[sl]).float().to(DEVICE),
                )
                parts.append(z.cpu().numpy())
        zmm = np.concatenate(parts, 0)

        methods = {
            "RGB": zr,
            "HSI": zh,
            "LiDAR": zl,
            "Concat": np.concatenate([zr, zh, zl], 1),
            "DS-M": (zr + zh + zl) / 3.0,
            "MMCTL": zmm,
        }
        for name, arr in methods.items():
            np.save(os.path.join(EMB, f"run{run:02d}_{name.lower()}_test.npy"), arr[te])

        yte, bte, hte = yc[te], box[te], yh[te]
        for m, zall in methods.items():
            z = zall[te]
            srows.append({"run": run, "method": m, **struct(z, yte, bte, hte, seed)})
            p = probe(l2(z.astype(np.float32)), yte, bte, hte, seed)
            drows.append({"run": run, "method": m, **p})
            print(f"  {m:<7} acc={p['acc_pct']:.1f} B={p['r2_box']:.3f} H={p['r2_height']:.3f}", flush=True)

    scored = zscore(srows)
    order = ["RGB", "HSI", "LiDAR", "Concat", "DS-M", "MMCTL"]

    def aggs(m, s):
        v = [r[f"score_{s}"] for r in scored if r["method"] == m]
        return float(np.mean(v)), float(np.std(v))

    def aggd(m, k):
        v = [r[k] for r in drows if r["method"] == m]
        return float(np.mean(v)), float(np.std(v))

    structure = {m: {"C": aggs(m, "cls"), "B": aggs(m, "box"), "H": aggs(m, "height")} for m in order}
    downstream = {
        m: {"C": aggd(m, "acc_pct"), "B": aggd(m, "r2_box"), "H": aggd(m, "r2_height")} for m in order
    }
    payload = {
        "protocol": (
            f"PROXY AE-pretrain baselines + separate MMCTL on {NEON}; "
            f"emb={EMB_DIM}, ae_ep={AE_EP}, mm_ep={MM_EP}, runs={RUNS}, device={DEVICE}"
        ),
        "data_path": NEON,
        "structure": structure,
        "downstream": downstream,
        "paper_structure": PAPER_S,
        "paper_downstream": PAPER_D,
        "verdict": "PROXY — keep paper NEON absolute tables",
    }
    with open(OUTJ, "w") as fp:
        json.dump(payload, fp, indent=2)

    lines = [
        "# NEON Replication Delta (PROXY)",
        "",
        f"Data: `{NEON}`",
        f"Protocol: {payload['protocol']}",
        "",
        "## Structure (lower better)",
        "",
        "| Method | Src | Paper | Proxy | Δmean |",
        "|---|---|---|---|---|",
    ]
    for m in order:
        for s in ("C", "B", "H"):
            pm, ps = PAPER_S[m][s]
            cm, cs = structure[m][s]
            lines.append(f"| {m} | {s} | {pm:.2f}±{ps:.2f} | {cm:.2f}±{cs:.2f} | {cm-pm:+.2f} |")
    lines += [
        "",
        "## Downstream",
        "",
        "| Method | Paper C% | Proxy C% | Paper B | Proxy B | Paper H | Proxy H |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in order:
        pc, _ = PAPER_D[m]["C"]
        pb, _ = PAPER_D[m]["B"]
        ph, _ = PAPER_D[m]["H"]
        cc, ccs = downstream[m]["C"]
        cb, cbs = downstream[m]["B"]
        ch, chs = downstream[m]["H"]
        lines.append(
            f"| {m} | {pc:.2f} | {cc:.2f}±{ccs:.2f} | {pb:.2f} | {cb:.2f}±{cbs:.2f} | {ph:.2f} | {ch:.2f}±{chs:.2f} |"
        )
    lines += [
        "",
        "## Verdict",
        "- Absolute match to paper: expected **FAIL** without original Keras checkpoints.",
        "- Keep paper NEON tables; use proxy for ranking/feasibility only.",
        "",
    ]
    with open(OUTD, "w") as fp:
        fp.write("\n".join(lines))
    print("Wrote", OUTJ, flush=True)
    print("Wrote", OUTD, flush=True)


if __name__ == "__main__":
    main()
