#!/usr/bin/env python3
"""Faster sklearn linear probes on released AWIR test embeddings (+ missing modality)."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from mmctl.utils import generate_triplet_box_label_normalized  # noqa: E402

METHOD_SPECS = {
    "RGB": "run{run:02d}_rgb_test.npy",
    "Thermal": "run{run:02d}_thermal_test.npy",
    "MMCTL": "run{run:02d}_mmctl_test.npy",
    "DeepSetMean": "run{run:02d}_deepset_mean_test_fused.npy",
    "DeepSetAttn": "run{run:02d}_deepset_attention_test_fused.npy",
}


def l2(z):
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def convert_box(y_box):
    out = np.zeros((len(y_box), 4), np.float32)
    for i, (xmin, xmax, ymin, ymax) in enumerate(y_box):
        out[i] = [
            ((xmin + xmax) / 2) / 300,
            ((ymin + ymax) / 2) / 300,
            (xmax - xmin) / 300,
            (ymax - ymin) / 300,
        ]
    return out


def mask(z, mode):
    out = z.copy()
    if mode == "drop_thermal":
        out[:, 1024:] = 0
    elif mode == "drop_rgb":
        out[:, :1024] = 0
    return l2(out)


def probe(z, y_cls, y_box, y_pos, seed=42, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, r2b, r2p = [], [], []
    for tr, te in skf.split(z, y_cls):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, multi_class="auto"),
        )
        clf.fit(z[tr], y_cls[tr])
        accs.append(accuracy_score(y_cls[te], clf.predict(z[te])))

        reg_b = make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(alpha=1.0)))
        reg_b.fit(z[tr], y_box[tr])
        pred_b = reg_b.predict(z[te])
        r2b.append(float(np.mean([r2_score(y_box[te, d], pred_b[:, d]) for d in range(y_box.shape[1])])))

        reg_p = make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(alpha=1.0)))
        reg_p.fit(z[tr], y_pos[tr])
        pred_p = reg_p.predict(z[te])
        r2p.append(float(np.mean([r2_score(y_pos[te, d], pred_p[:, d]) for d in range(y_pos.shape[1])])))

    return {
        "acc": float(np.mean(accs)),
        "acc_std_folds": float(np.std(accs)),
        "r2_box_mean": float(np.mean(r2b)),
        "r2_pos_mean": float(np.mean(r2p)),
    }


def main():
    data = "/Users/meilunzhou/Documents/GitHub/CTL/awir_data.npz"
    emb_dir = os.path.join(REPO, "pretrained_embeddings")
    f = np.load(data, allow_pickle=True)
    y_cls_raw = f["class_label"]
    y_box = np.clip(f["box_label"].astype(np.float32), None, 300)
    y_box_cenwh = convert_box(y_box)
    y_trip, box_feat = generate_triplet_box_label_normalized(y_box, y_cls_raw, n_clusters=3)
    box_feat = box_feat.astype(np.float32)
    y_pos = y_box_cenwh[:, :2].astype(np.float32)
    y_cls = LabelEncoder().fit_transform(y_cls_raw).astype(np.int64)
    composite = LabelEncoder().fit_transform([f"{c}_{t}" for c, t in zip(y_cls, y_trip)])
    _, idx_test = train_test_split(
        np.arange(len(y_cls)), test_size=0.5, stratify=composite, random_state=42
    )
    y_cls_t = y_cls[idx_test]
    y_box_t = box_feat[idx_test]
    y_pos_t = y_pos[idx_test]

    rows = []
    for run in range(1, 9):
        z_rgb = np.load(os.path.join(emb_dir, f"run{run:02d}_rgb_test.npy")).astype(np.float32)
        z_th = np.load(os.path.join(emb_dir, f"run{run:02d}_thermal_test.npy")).astype(np.float32)
        for method, pat in METHOD_SPECS.items():
            z = l2(np.load(os.path.join(emb_dir, pat.format(run=run))).astype(np.float32))
            st = probe(z, y_cls_t, y_box_t, y_pos_t)
            rows.append({"run": run, "method": method, "setup": "full", **st})
            print(f"run{run:02d} {method:12s} acc={st['acc']*100:.1f} r2b={st['r2_box_mean']:.3f} r2p={st['r2_pos_mean']:.3f}")
        z_cat = l2(np.concatenate([z_rgb, z_th], 1))
        st = probe(z_cat, y_cls_t, y_box_t, y_pos_t)
        rows.append({"run": run, "method": "Concat", "setup": "full", **st})
        print(f"run{run:02d} {'Concat':12s} acc={st['acc']*100:.1f} r2b={st['r2_box_mean']:.3f} r2p={st['r2_pos_mean']:.3f}")

        z_mm = l2(np.load(os.path.join(emb_dir, f"run{run:02d}_mmctl_test.npy")).astype(np.float32))
        for mode in ["drop_thermal", "drop_rgb"]:
            st = probe(mask(z_mm, mode), y_cls_t, y_box_t, y_pos_t)
            rows.append({"run": run, "method": "MMCTL", "setup": mode, **st})
            print(f"run{run:02d} MMCTL/{mode} acc={st['acc']*100:.1f} r2b={st['r2_box_mean']:.3f} r2p={st['r2_pos_mean']:.3f}")

    def agg(rows, method, setup):
        sub = [r for r in rows if r["method"] == method and r["setup"] == setup]
        return {
            "acc_pct_mean": float(np.mean([r["acc"] for r in sub]) * 100),
            "acc_pct_std": float(np.std([r["acc"] for r in sub], ddof=1) * 100),
            "r2_box_mean": float(np.mean([r["r2_box_mean"] for r in sub])),
            "r2_box_std": float(np.std([r["r2_box_mean"] for r in sub], ddof=1)),
            "r2_pos_mean": float(np.mean([r["r2_pos_mean"] for r in sub])),
            "r2_pos_std": float(np.std([r["r2_pos_mean"] for r in sub], ddof=1)),
            "n_runs": len(sub),
        }

    methods = ["RGB", "Thermal", "Concat", "DeepSetAttn", "DeepSetMean", "MMCTL"]
    aggregate = {m: agg(rows, m, "full") for m in methods}
    missing = {
        "drop_thermal": agg(rows, "MMCTL", "drop_thermal"),
        "drop_rgb": agg(rows, "MMCTL", "drop_rgb"),
        "full": aggregate["MMCTL"],
    }

    paper = {
        "RGB": {"acc": 77.08, "r2_box": 0.34, "r2_pos": -0.13},
        "Thermal": {"acc": 78.96, "r2_box": 0.30, "r2_pos": -0.32},
        "Concat": {"acc": 89.06, "r2_box": 0.32, "r2_pos": -0.15},
        "DeepSetAttn": {"acc": 87.71, "r2_box": 0.36, "r2_pos": -0.18},
        "DeepSetMean": {"acc": 87.40, "r2_box": 0.37, "r2_pos": -0.14},
        "MMCTL": {"acc": 94.06, "r2_box": 0.52, "r2_pos": 0.12},
    }

    payload = {
        "protocol": (
            "Proxy sklearn Linear/Ridge 5-fold CV on released TEST embeddings only "
            "(checkpoints/train embeddings not shipped). Classification is informative for ranking; "
            "regression R² is not expected to match paper absolute values."
        ),
        "per_run": rows,
        "aggregate": aggregate,
        "paper_reference": paper,
        "missing_modality_mmctl": missing,
    }

    out1 = os.path.join(REPO, "results/taskhead_proxy_awir.json")
    with open(out1, "w") as fp:
        json.dump(payload, fp, indent=2)

    # missing modality dedicated file
    mm = {
        "dataset": "AWIR",
        "protocol": payload["protocol"],
        "description": (
            "At test/probe time, zero one half of the 2048-d MMCTL joint embedding "
            "(assumed [RGB|Thermal] 1024+1024) then L2-normalize and re-run frozen probes."
        ),
        "results": missing,
        "delta_vs_full": {
            mode: {
                "acc_pct_drop": missing["full"]["acc_pct_mean"] - missing[mode]["acc_pct_mean"],
                "r2_box_drop": missing["full"]["r2_box_mean"] - missing[mode]["r2_box_mean"],
                "r2_pos_drop": missing["full"]["r2_pos_mean"] - missing[mode]["r2_pos_mean"],
            }
            for mode in ["drop_thermal", "drop_rgb"]
        },
    }
    out2 = os.path.join(REPO, "results/missing_modality.json")
    with open(out2, "w") as fp:
        json.dump(mm, fp, indent=2)

    # update replication json downstream section
    rep_path = os.path.join(REPO, "results/replication_awir.json")
    with open(rep_path) as fp:
        rep = json.load(fp)
    rep["downstream_proxy"] = {
        "protocol": payload["protocol"],
        "aggregate": aggregate,
        "paper_reference": paper,
        "gate": "PROXY_ONLY — do not replace paper downstream table without checkpoints",
    }
    with open(rep_path, "w") as fp:
        json.dump(rep, fp, indent=2)

    print("\n=== Aggregate (proxy) vs paper ===")
    for m in methods:
        a = aggregate[m]
        p = paper[m]
        print(
            f"{m:12s} code C={a['acc_pct_mean']:.2f}±{a['acc_pct_std']:.2f} "
            f"(paper {p['acc']}) | "
            f"B={a['r2_box_mean']:.3f} (paper {p['r2_box']}) | "
            f"P={a['r2_pos_mean']:.3f} (paper {p['r2_pos']})"
        )
    print("Wrote", out1, out2)


if __name__ == "__main__":
    main()
