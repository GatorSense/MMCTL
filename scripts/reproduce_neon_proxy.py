#!/usr/bin/env python3
"""
Fast NEON proxy reproduction using CTL neon_data.npz.

Uses a sampled continuous-triplet objective (not full batch-all O(n^3)) so
it finishes on Apple Silicon. Results are PROXY vs paper tables.
"""

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

from mmctl.embedding_metrics import (  # noqa: E402
    embedding_annotation_alignment_ratio,
    local_smoothness_std,
)
from mmctl.utils import generate_triplet_box_label_normalized  # noqa: E402

NEON_NPZ = "data/neon_data.npz"
OUT_JSON = os.path.join(REPO, "results", "replication_neon.json")
OUT_DELTA = os.path.join(REPO, "results", "replication_neon_delta.md")
EMB_DIR = os.path.join(REPO, "outputs", "neon_proxy_embeddings")

PAPER_STRUCTURE = {
    "RGB": {"C": (-1.66, 0.88), "B": (-0.34, 0.87), "H": (0.48, 0.97)},
    "HSI": {"C": (0.72, 0.11), "B": (0.94, 0.56), "H": (1.49, 0.23)},
    "LiDAR": {"C": (2.60, 0.08), "B": (1.60, 0.72), "H": (-0.35, 0.17)},
    "Concat": {"C": (0.39, 0.15), "B": (0.19, 0.22), "H": (0.36, 0.05)},
    "DS-M": {"C": (-0.27, 0.12), "B": (0.24, 0.29), "H": (-0.27, 0.23)},
    "MMCTL": {"C": (-1.49, 0.23), "B": (-2.07, 0.29), "H": (-1.58, 0.25)},
}
PAPER_DOWNSTREAM = {
    "RGB": {"C": (74.59, 1.90), "B": (0.07, 0.15), "H": (0.34, 0.02)},
    "HSI": {"C": (76.69, 1.39), "B": (0.12, 0.08), "H": (0.40, 0.05)},
    "LiDAR": {"C": (49.66, 1.15), "B": (0.22, 0.01), "H": (0.68, 0.02)},
    "Concat": {"C": (81.69, 1.08), "B": (0.39, 0.03), "H": (0.69, 0.02)},
    "DS-M": {"C": (81.35, 1.18), "B": (0.40, 0.01), "H": (0.61, 0.04)},
    "MMCTL": {"C": (86.95, 1.7), "B": (0.43, 0.04), "H": (0.74, 0.02)},
}


class Cfg:
    emb_dim = 128
    batch_size = 64
    epochs = 25
    lr = 1e-3
    runs = 5
    seed = 42
    n_triplets = 512  # sampled triplets per step
    device = "mps" if torch.backends.mps.is_available() else "cpu"


def minmax_train(x, train_idx, eps=1e-8):
    lo = x[train_idx].min(axis=0)
    hi = x[train_idx].max(axis=0)
    return (x - lo) / (hi - lo + eps)


def build_ann(y_cls, box_feat, height, train_idx, w=(1.0, 1.0, 1.0)):
    n_cls = int(y_cls.max()) + 1
    onehot = np.eye(n_cls, dtype=np.float32)[y_cls]
    box_n = minmax_train(box_feat.astype(np.float32), train_idx)
    h_n = minmax_train(height.astype(np.float32), train_idx)
    l_cls, l_box, l_h = onehot.shape[1], box_n.shape[1], h_n.shape[1]
    l_max = max(l_cls, l_box, l_h)
    return np.concatenate(
        [
            w[0] * onehot * (l_max / l_cls),
            w[1] * box_n * (l_max / l_box),
            w[2] * h_n * (l_max / l_h),
        ],
        axis=1,
    ).astype(np.float32)


def cos_dist_rows(a, b):
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    return (1.0 - (a * b).sum(dim=1)).clamp_min(0.0)


def sampled_ctl_loss(ann, z_r, z_h, z_l, n_triplets=512):
    """Sample continuous triplets over expanded multimodal embeddings."""
    z_all = torch.cat([z_r, z_h, z_l], dim=0)
    y_all = torch.cat([ann, ann, ann], dim=0)
    n = z_all.shape[0]
    device = z_all.device
    # sample anchors / candidates
    a = torch.randint(0, n, (n_triplets,), device=device)
    p = torch.randint(0, n, (n_triplets,), device=device)
    q = torch.randint(0, n, (n_triplets,), device=device)
    dy_ap = cos_dist_rows(y_all[a], y_all[p])
    dy_aq = cos_dist_rows(y_all[a], y_all[q])
    valid = dy_ap < dy_aq
    if valid.sum() == 0:
        return z_all.new_zeros(())
    a, p, q = a[valid], p[valid], q[valid]
    dy_ap, dy_aq = dy_ap[valid], dy_aq[valid]
    dz_ap = cos_dist_rows(z_all[a], z_all[p])
    dz_aq = cos_dist_rows(z_all[a], z_all[q])
    trip = (dz_ap.pow(2) - dz_aq.pow(2) + dy_aq - dy_ap).clamp_min(0.0)
    reg = (dz_ap - dy_ap).pow(2) + (dz_aq - dy_aq).pow(2)
    # same-index cross-modal pull
    align = (
        cos_dist_rows(z_r, z_h).mean()
        + cos_dist_rows(z_r, z_l).mean()
        + cos_dist_rows(z_h, z_l).mean()
    )
    return trip.mean() + reg.mean() + 0.1 * align


class TinyCNN(nn.Module):
    def __init__(self, in_ch, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, emb_dim)

    def forward(self, x_nhwc):
        x = x_nhwc.permute(0, 3, 1, 2).contiguous()
        return self.fc(self.net(x).flatten(1))


class SpectralCNN(nn.Module):
    def __init__(self, n_bands, emb_dim):
        super().__init__()
        self.reduce = nn.Conv2d(n_bands, 32, 1)
        self.cnn = TinyCNN(32, emb_dim)

    def forward(self, x_nhwc):
        x = x_nhwc.permute(0, 3, 1, 2).contiguous()
        x = F.relu(self.reduce(x)).permute(0, 2, 3, 1).contiguous()
        return self.cnn(x)


class MMCTLProxy(nn.Module):
    def __init__(self, emb_dim, n_bands):
        super().__init__()
        self.rgb = TinyCNN(3, emb_dim)
        self.hsi = SpectralCNN(n_bands, emb_dim)
        self.lidar = TinyCNN(1, emb_dim)

    def forward(self, rgb, hsi, lidar):
        zr, zh, zl = self.rgb(rgb), self.hsi(hsi), self.lidar(lidar)
        return torch.cat([zr, zh, zl], 1), zr, zh, zl


def l2(z):
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


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


def zscore_struct(rows, keys=("cls", "box", "height")):
    out = []
    for run in sorted({r["run"] for r in rows}):
        subset = [dict(r) for r in rows if r["run"] == run]
        for src in keys:
            a = np.array([r[f"align_{src}"] for r in subset], float)
            s = np.array([r[f"smooth_{src}"] for r in subset], float)
            za = (a - a.mean()) / (a.std() + 1e-12)
            zs = (s - s.mean()) / (s.std() + 1e-12)
            for i in range(len(subset)):
                subset[i][f"score_{src}"] = float(za[i] + zs[i])
        out.extend(subset)
    return out


def probe(z, y_cls, y_box, y_h, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs, r2b, r2h = [], [], []
    for tr, te in skf.split(z, y_cls):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=4000, solver="saga", n_jobs=1),
        )
        clf.fit(z[tr], y_cls[tr])
        accs.append(accuracy_score(y_cls[te], clf.predict(z[te])))
        rb = make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(1.0)))
        rb.fit(z[tr], y_box[tr])
        pb = rb.predict(z[te])
        r2b.append(float(np.mean([r2_score(y_box[te, d], pb[:, d]) for d in range(y_box.shape[1])])))
        rh = make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(1.0)))
        rh.fit(z[tr], y_h[tr])
        ph = rh.predict(z[te])
        r2h.append(float(np.mean([r2_score(y_h[te, d], ph[:, d]) for d in range(y_h.shape[1])])))
    return {
        "acc_pct": 100.0 * float(np.mean(accs)),
        "r2_box": float(np.mean(r2b)),
        "r2_height": float(np.mean(r2h)),
    }


def normalize_inputs(rgb, hsi, lidar, train_idx):
    rgb = rgb.astype(np.float32) / 255.0
    h = np.clip(hsi.astype(np.float32), -1e4, 1e4)
    lo = h[train_idx].reshape(-1, h.shape[-1]).min(0)
    hi = h[train_idx].reshape(-1, h.shape[-1]).max(0)
    h = (h - lo) / (hi - lo + 1e-8)
    lid = lidar.astype(np.float32)
    lo_l, hi_l = lid[train_idx].min(), lid[train_idx].max()
    lid = (lid - lo_l) / (hi_l - lo_l + 1e-8)
    return rgb, h, lid


def train_one(rgb, hsi, lidar, ann, train_idx, cfg, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(cfg.device)
    model = MMCTLProxy(cfg.emb_dim, hsi.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    rgb_t = torch.from_numpy(rgb[train_idx]).float()
    hsi_t = torch.from_numpy(hsi[train_idx]).float()
    lid_t = torch.from_numpy(lidar[train_idx]).float()
    ann_t = torch.from_numpy(ann[train_idx]).float()
    n = len(train_idx)
    model.train()
    for ep in range(cfg.epochs):
        perm = np.random.permutation(n)
        losses = []
        for i in range(0, n, cfg.batch_size):
            b = perm[i : i + cfg.batch_size]
            if len(b) < 8:
                continue
            opt.zero_grad(set_to_none=True)
            _, zr, zh, zl = model(
                rgb_t[b].to(device), hsi_t[b].to(device), lid_t[b].to(device)
            )
            loss = sampled_ctl_loss(ann_t[b].to(device), zr, zh, zl, cfg.n_triplets)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if ep % 5 == 0 or ep == cfg.epochs - 1:
            print(f"  epoch {ep+1}/{cfg.epochs} loss={np.mean(losses):.4f}", flush=True)
    return model


@torch.no_grad()
def embed_all(model, rgb, hsi, lidar, cfg):
    model.eval()
    device = torch.device(cfg.device)
    joints, uni = [], {"RGB": [], "HSI": [], "LiDAR": []}
    for i in range(0, len(rgb), 128):
        sl = slice(i, i + 128)
        z, zr, zh, zl = model(
            torch.from_numpy(rgb[sl]).float().to(device),
            torch.from_numpy(hsi[sl]).float().to(device),
            torch.from_numpy(lidar[sl]).float().to(device),
        )
        joints.append(z.cpu().numpy())
        uni["RGB"].append(zr.cpu().numpy())
        uni["HSI"].append(zh.cpu().numpy())
        uni["LiDAR"].append(zl.cpu().numpy())
    joint = np.concatenate(joints, 0)
    for k in uni:
        uni[k] = np.concatenate(uni[k], 0)
    return joint, uni


def main():
    cfg = Cfg()
    os.makedirs(EMB_DIR, exist_ok=True)
    print(f"NEON data: {NEON_NPZ}", flush=True)
    print(f"device={cfg.device} emb={cfg.emb_dim} epochs={cfg.epochs} runs={cfg.runs}", flush=True)

    f = np.load(NEON_NPZ, allow_pickle=True)
    rgb0, hsi0, lidar0 = f["rgb"], f["hsi"], f["lidar"]
    y_cls = LabelEncoder().fit_transform(f["class_labels"]).astype(np.int64)
    y_box_raw = f["box_labels"].astype(np.float32)
    height = f["height_info"].astype(np.float32)
    y_str = f["string_label"] if "string_label" in f.files else y_cls.astype(str)
    _, box_feat = generate_triplet_box_label_normalized(y_box_raw, y_str, n_clusters=3)
    box_feat = box_feat.astype(np.float32)

    struct_rows, down_rows = [], []
    for run in range(1, cfg.runs + 1):
        seed = cfg.seed + run - 1
        print(f"\n=== run {run}/{cfg.runs} seed={seed} ===", flush=True)
        counts = np.bincount(y_cls)
        keep = np.isin(y_cls, np.where(counts >= 2)[0])
        idx_keep = np.arange(len(y_cls))[keep]
        tr, te = train_test_split(
            idx_keep, test_size=0.5, stratify=y_cls[keep], random_state=seed
        )
        rgb, hsi, lidar = normalize_inputs(rgb0, hsi0, lidar0, tr)
        ann = build_ann(y_cls, box_feat, height, tr)
        model = train_one(rgb, hsi, lidar, ann, tr, cfg, seed)
        joint, uni = embed_all(model, rgb, hsi, lidar, cfg)
        np.save(os.path.join(EMB_DIR, f"run{run:02d}_mmctl_test.npy"), joint[te])
        for name, arr in uni.items():
            np.save(os.path.join(EMB_DIR, f"run{run:02d}_{name.lower()}_test.npy"), arr[te])

        methods = {
            "RGB": uni["RGB"],
            "HSI": uni["HSI"],
            "LiDAR": uni["LiDAR"],
            "Concat": np.concatenate([uni["RGB"], uni["HSI"], uni["LiDAR"]], 1),
            "DS-M": (uni["RGB"] + uni["HSI"] + uni["LiDAR"]) / 3.0,
            "MMCTL": joint,
        }
        y_te, box_te, h_te = y_cls[te], box_feat[te], height[te]
        for method, z_all in methods.items():
            z = z_all[te]
            m = structure_raw(z, y_te, box_te, h_te, seed=seed)
            struct_rows.append({"run": run, "method": method, **m})
            p = probe(l2(z.astype(np.float32)), y_te, box_te, h_te, seed=seed)
            down_rows.append({"run": run, "method": method, **p})
            print(
                f"  {method:<7} acc={p['acc_pct']:.1f} B={p['r2_box']:.3f} H={p['r2_height']:.3f}",
                flush=True,
            )

    scored = zscore_struct(struct_rows)
    methods_order = ["RGB", "HSI", "LiDAR", "Concat", "DS-M", "MMCTL"]

    def agg_s(method, src):
        vals = [r[f"score_{src}"] for r in scored if r["method"] == method]
        return float(np.mean(vals)), float(np.std(vals))

    def agg_d(method, key):
        vals = [r[key] for r in down_rows if r["method"] == method]
        return float(np.mean(vals)), float(np.std(vals))

    structure = {
        m: {"C": agg_s(m, "cls"), "B": agg_s(m, "box"), "H": agg_s(m, "height")}
        for m in methods_order
    }
    downstream = {
        m: {"C": agg_d(m, "acc_pct"), "B": agg_d(m, "r2_box"), "H": agg_d(m, "r2_height")}
        for m in methods_order
    }

    payload = {
        "protocol": (
            f"PROXY sampled continuous-triplet MMCTL-style on {NEON_NPZ}; "
            f"emb_dim={cfg.emb_dim}, epochs={cfg.epochs}, runs={cfg.runs}, device={cfg.device}. "
            "Not bit-identical to paper Keras pipeline. DS-M=mean pool; DS-A not run."
        ),
        "data_path": NEON_NPZ,
        "structure": structure,
        "downstream": downstream,
        "paper_structure": PAPER_STRUCTURE,
        "paper_downstream": PAPER_DOWNSTREAM,
    }
    with open(OUT_JSON, "w") as fp:
        json.dump(payload, fp, indent=2)

    lines = [
        "# NEON Replication Delta (PROXY)",
        "",
        f"Data: `{NEON_NPZ}`",
        f"Protocol: {payload['protocol']}",
        "",
        "## Structure (lower better)",
        "",
        "| Method | Src | Paper | Proxy | Δmean |",
        "|---|---|---|---|---|",
    ]
    for m in methods_order:
        for src in ("C", "B", "H"):
            pm, ps = PAPER_STRUCTURE[m][src]
            cm, cs = structure[m][src]
            lines.append(f"| {m} | {src} | {pm:.2f}±{ps:.2f} | {cm:.2f}±{cs:.2f} | {cm-pm:+.2f} |")
    lines += [
        "",
        "## Downstream",
        "",
        "| Method | Paper C% | Proxy C% | Paper B | Proxy B | Paper H | Proxy H |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in methods_order:
        pc, _ = PAPER_DOWNSTREAM[m]["C"]
        pb, _ = PAPER_DOWNSTREAM[m]["B"]
        ph, _ = PAPER_DOWNSTREAM[m]["H"]
        cc, ccs = downstream[m]["C"]
        cb, cbs = downstream[m]["B"]
        ch, chs = downstream[m]["H"]
        lines.append(
            f"| {m} | {pc:.2f} | {cc:.2f}±{ccs:.2f} | {pb:.2f} | {cb:.2f}±{cbs:.2f} | {ph:.2f} | {ch:.2f}±{chs:.2f} |"
        )
    lines += [
        "",
        "## Notes",
        "- Keep paper NEON absolute tables unless proxy matches closely.",
        "- Use proxy for ranking / feasibility evidence in response-to-reviewers.",
        "",
    ]
    with open(OUT_DELTA, "w") as fp:
        fp.write("\n".join(lines))
    print(f"Wrote {OUT_JSON}", flush=True)
    print(f"Wrote {OUT_DELTA}", flush=True)


if __name__ == "__main__":
    main()
