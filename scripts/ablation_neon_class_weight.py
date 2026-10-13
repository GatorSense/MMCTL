#!/usr/bin/env python3
"""
NEON class-weight ablation (Reviewer R1.4: does the annotation weight matter?).

Uses the released full-set MMCTL embeddings trained with class-label weight in
{0.5, 0.8, 1.0, 1.5} (box/height weights fixed at 1.0):
    neon_mmctl_all_6_{0.5,0.8,1.0,1.5}.npy   (N=592, 3072-d)

For each weight we report:
  * Latent-structure metrics (alignment ratio + local smoothness; lower=better)
    for class / box / height annotations, computed on held-out test folds.
  * Downstream task-head performance (paper protocol: frozen embeddings + MLP
    heads) for classification accuracy, box R^2 and height R^2.

Protocol mirrors scripts/eval_neon_released_embeddings.py: 50/50 stratified
holdout, StratifiedKFold on trainval, paper task heads on the fixed test split.

Outputs:
  results/ablation_neon_class_weight.json
  results/ablation_neon_class_weight.md

Run:
  .venv_exp/bin/python -u scripts/ablation_neon_class_weight.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mmctl.train_taskheads import HeadTrainConfig  # noqa: E402
from mmctl.utils import generate_triplet_box_label_normalized  # noqa: E402
from scripts.eval_neon_released_embeddings import (  # noqa: E402
    neon_index_592,
    paper_task_heads,
    structure_raw,
)

MAIN = "."
WEIGHTS = ["0.5", "0.8", "1.0", "1.5"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main", default=MAIN)
    p.add_argument("--npz_path", default=os.path.join(MAIN, "neon_data.npz"))
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--head_epochs", type=int, default=3000)
    p.add_argument("--head_patience", type=int, default=1000)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.runs, args.head_epochs, args.head_patience = 2, 60, 20

    d = np.load(args.npz_path, allow_pickle=True)
    idx = neon_index_592(args.npz_path)
    assert len(idx) == 592, len(idx)

    y_cls = LabelEncoder().fit_transform(d["class_labels"][idx]).astype(np.int64)
    y_pos = d["box_labels"][idx].astype(np.float32)
    y_box = d["box_info"][idx].astype(np.float32)
    y_h = d["height_info"][idx].astype(np.float32)
    _, box_feat = generate_triplet_box_label_normalized(
        y_pos, np.asarray(y_cls).astype(str), n_clusters=3
    )
    box_feat = box_feat.astype(np.float32)

    head_cfg = HeadTrainConfig(
        epochs=args.head_epochs, batch_size=32, lr=1e-3, weight_decay=1e-4,
        hidden=256, dropout=0.1, early_patience=args.head_patience, grad_clip=5.0,
    )

    results = {}
    for w in WEIGHTS:
        z_all = np.load(os.path.join(args.main, f"neon_mmctl_all_6_{w}.npy")).astype(np.float32)
        assert z_all.shape == (592, 3072), z_all.shape
        print(f"\n===== class weight {w} =====", flush=True)

        (z_tv, z_te, y_tv, y_te, box_tv, box_te, h_tv, h_te,
         bf_tv, bf_te) = train_test_split(
            z_all, y_cls, y_box, y_h, box_feat,
            test_size=0.5, stratify=y_cls, random_state=42,
        )
        skf = StratifiedKFold(n_splits=args.runs, shuffle=True, random_state=42)

        srows, drows = [], []
        for run, (tr, va) in enumerate(skf.split(z_tv, y_tv), start=1):
            s = structure_raw(z_te, y_te, bf_te, h_te, seed=40 + run)
            srows.append(s)
            # Structure box target = k-means triplet box_feat (bf_*);
            # downstream box target = continuous box_info geometry (box_*),
            # matching scripts/eval_neon_released_embeddings.py.
            pr = paper_task_heads(
                z_tv[tr], z_tv[va], z_te,
                y_tv[tr], y_tv[va], y_te,
                box_tv[tr], box_tv[va], box_te,
                h_tv[tr], h_tv[va], h_te,
                head_cfg,
            )
            drows.append(pr)
            print(f"  fold {run}: acc={pr['acc_pct']:.1f} B={pr['r2_box']:.3f} "
                  f"H={pr['r2_height']:.3f}", flush=True)

        def m(rows, k):
            v = np.array([r[k] for r in rows], float)
            return float(np.mean(v)), float(np.std(v))

        results[w] = {
            "structure": {k: m(srows, k) for k in (
                "align_cls", "align_box", "align_height",
                "smooth_cls", "smooth_box", "smooth_height")},
            "downstream": {k: m(drows, k) for k in ("acc_pct", "r2_box", "r2_height")},
        }

    out_json = os.path.join(ROOT, "results", "ablation_neon_class_weight.json")
    with open(out_json, "w") as fp:
        json.dump({
            "protocol": (
                f"Released full-set NEON MMCTL embeddings (N=592) at class weights "
                f"{WEIGHTS} (box/height weights=1.0). 50/50 holdout, {args.runs}-fold "
                f"CV on trainval, paper task heads (epochs={args.head_epochs}, "
                f"patience={args.head_patience})."),
            "results": results,
        }, fp, indent=2)

    lines = [
        "# NEON class-weight ablation (R1.4)",
        "",
        "Released full-set MMCTL embeddings (N=592); class-label weight swept, "
        "box/height weights fixed at 1.0.",
        "",
        "## Downstream (higher better)",
        "",
        "| Class weight | Acc % | Box R² | Height R² |",
        "|---|---|---|---|",
    ]
    for w in WEIGHTS:
        dd = results[w]["downstream"]
        lines.append(
            f"| {w} | {dd['acc_pct'][0]:.2f}±{dd['acc_pct'][1]:.2f} | "
            f"{dd['r2_box'][0]:.3f}±{dd['r2_box'][1]:.3f} | "
            f"{dd['r2_height'][0]:.3f}±{dd['r2_height'][1]:.3f} |")
    lines += [
        "",
        "## Latent structure (alignment ratio + local smoothness, lower better)",
        "",
        "| Class weight | Align cls | Align box | Align hgt | Smooth cls | Smooth box | Smooth hgt |",
        "|---|---|---|---|---|---|---|",
    ]
    for w in WEIGHTS:
        s = results[w]["structure"]
        lines.append(
            f"| {w} | {s['align_cls'][0]:.3f} | {s['align_box'][0]:.3f} | "
            f"{s['align_height'][0]:.3f} | {s['smooth_cls'][0]:.3f} | "
            f"{s['smooth_box'][0]:.3f} | {s['smooth_height'][0]:.3f} |")
    lines += ["", "Figures: `paper_files/figures/neon_ablation_classweight_{boxsize,height}.pdf`.", ""]

    out_md = os.path.join(ROOT, "results", "ablation_neon_class_weight.md")
    with open(out_md, "w") as fp:
        fp.write("\n".join(lines))
    print("\nWrote", out_json, "\n      ", out_md, flush=True)


if __name__ == "__main__":
    main()
