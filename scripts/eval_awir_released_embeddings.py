#!/usr/bin/env python3
"""
Evaluate released full-set AWIR embeddings + stronger fusion baselines.

Embeddings (N=240 = full train+test):
  awir_rgb_all.npy, awir_thermal_all.npy, awir_mmctl_all.npy

Protocol matches paper: 50/50 holdout, 8-fold CV on trainval,
paper task heads (3000 ep / patience 1000), batch 32, grad clip 5.

Baselines: Concat, SupCon, CrossAttn, Transformer (DeepSet removed).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT)

from mmctl.train_taskheads import HeadTrainConfig  # noqa: E402
from mmctl.utils import generate_triplet_box_label_normalized  # noqa: E402
from scripts.eval_neon_released_embeddings import (  # noqa: E402
    apply_supcon,
    apply_task_fusion,
    paper_task_heads,
    structure_raw,
    train_supcon,
    train_task_fusion,
    zscore_struct,
)

PAPER_STRUCTURE = {
    "RGB": {"C": (0.43, 0.28), "B": (-0.41, 0.36), "P": (0.23, 0.38)},
    "Thermal": {"C": (0.54, 0.15), "B": (0.56, 0.14), "P": (0.89, 0.49)},
    "Concat": {"C": (-0.28, 0.17), "B": (-0.19, 0.15), "P": (0.31, 0.27)},
    "SupCon": {"C": (float("nan"), float("nan")), "B": (float("nan"), float("nan")), "P": (float("nan"), float("nan"))},
    "CrossAttn": {"C": (float("nan"), float("nan")), "B": (float("nan"), float("nan")), "P": (float("nan"), float("nan"))},
    "Transformer": {"C": (float("nan"), float("nan")), "B": (float("nan"), float("nan")), "P": (float("nan"), float("nan"))},
    "MMCTL": {"C": (-3.10, 0.04), "B": (-2.93, 0.18), "P": (-3.79, 0.61)},
}
PAPER_DOWN = {
    "RGB": {"C": (77.08, 3.21), "B": (0.34, 0.07), "P": (-0.13, 0.16)},
    "Thermal": {"C": (78.96, 1.59), "B": (0.30, 0.07), "P": (-0.32, 0.34)},
    "Concat": {"C": (89.06, 1.44), "B": (0.32, 0.08), "P": (-0.15, 0.20)},
    "SupCon": {"C": (float("nan"), float("nan")), "B": (float("nan"), float("nan")), "P": (float("nan"), float("nan"))},
    "CrossAttn": {"C": (float("nan"), float("nan")), "B": (float("nan"), float("nan")), "P": (float("nan"), float("nan"))},
    "Transformer": {"C": (float("nan"), float("nan")), "B": (float("nan"), float("nan")), "P": (float("nan"), float("nan"))},
    "MMCTL": {"C": (94.06, 1.86), "B": (0.52, 0.07), "P": (0.12, 0.08)},
}
ORDER = ["RGB", "Thermal", "Concat", "SupCon", "CrossAttn", "Transformer", "MMCTL"]


def convert_box_to_center_width_height(y_box: np.ndarray) -> np.ndarray:
    out = np.zeros((y_box.shape[0], 4), dtype=np.float32)
    for i in range(y_box.shape[0]):
        xmin, xmax, ymin, ymax = y_box[i]
        out[i, 0] = ((xmin + xmax) / 2.0) / 300.0
        out[i, 1] = ((ymin + ymax) / 2.0) / 300.0
        out[i, 2] = (xmax - xmin) / 300.0
        out[i, 3] = (ymax - ymin) / 300.0
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz_path", default="/Users/meilunzhou/Documents/GitHub/CTL/awir_data.npz")
    p.add_argument("--rgb", default=os.path.join(ROOT, "awir_rgb_all.npy"))
    p.add_argument("--thermal", default=os.path.join(ROOT, "awir_thermal_all.npy"))
    p.add_argument("--mmctl", default=os.path.join(ROOT, "awir_mmctl_all.npy"))
    p.add_argument("--runs", type=int, default=8)
    p.add_argument("--fusion_epochs", type=int, default=300)
    p.add_argument("--head_epochs", type=int, default=3000)
    p.add_argument("--head_patience", type=int, default=1000)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.runs = 2
        args.fusion_epochs = 40
        args.head_epochs, args.head_patience = 80, 25

    d = np.load(args.npz_path, allow_pickle=True)
    y_cls = LabelEncoder().fit_transform(d["class_label"])
    y_box_raw = np.clip(d["box_label"].astype(np.float32), None, 300)
    y_box4 = convert_box_to_center_width_height(y_box_raw)
    y_pos = y_box4[:, :2].astype(np.float32)
    _, box_feat = generate_triplet_box_label_normalized(
        y_box_raw, d["class_label"], n_clusters=3
    )
    box_feat = box_feat.astype(np.float32)

    z_rgb = np.load(args.rgb).astype(np.float32)
    z_th = np.load(args.thermal).astype(np.float32)
    z_mm = np.load(args.mmctl).astype(np.float32)
    assert z_rgb.shape == (240, 1024) and z_th.shape == (240, 1024) and z_mm.shape == (240, 2048), (
        z_rgb.shape,
        z_th.shape,
        z_mm.shape,
    )
    assert len(y_cls) == 240

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
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
        f"N=240 mmctl={args.mmctl} runs={args.runs} fusion_ep={args.fusion_epochs} "
        f"heads={args.head_epochs}/{args.head_patience} device={device}",
        flush=True,
    )

    (
        z_rgb_tv,
        z_rgb_te,
        z_th_tv,
        z_th_te,
        z_mm_tv,
        z_mm_te,
        y_tv,
        y_te,
        box_tv,
        box_te,
        pos_tv,
        pos_te,
        bf_tv,
        bf_te,
    ) = train_test_split(
        z_rgb,
        z_th,
        z_mm,
        y_cls,
        y_box4,
        y_pos,
        box_feat,
        test_size=0.5,
        stratify=y_cls,
        random_state=42,
    )

    skf = StratifiedKFold(n_splits=args.runs, shuffle=True, random_state=42)
    struct_rows, down_rows = [], []

    for run, (tr, va) in enumerate(skf.split(z_rgb_tv, y_tv), start=1):
        print(f"\n=== fold {run}/{args.runs} ===", flush=True)
        mods_tr = np.stack([z_rgb_tv[tr], z_th_tv[tr]], axis=1).astype(np.float32)
        mods_va = np.stack([z_rgb_tv[va], z_th_tv[va]], axis=1).astype(np.float32)
        mods_te = np.stack([z_rgb_te, z_th_te], axis=1).astype(np.float32)
        cat_tr = np.concatenate([z_rgb_tv[tr], z_th_tv[tr]], 1)
        cat_va = np.concatenate([z_rgb_tv[va], z_th_tv[va]], 1)
        cat_te = np.concatenate([z_rgb_te, z_th_te], 1)

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
            box_tv[tr],
            box_tv[va],
            pos_tv[tr],
            pos_tv[va],
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
            box_tv[tr],
            box_tv[va],
            pos_tv[tr],
            pos_tv[va],
            args.fusion_epochs,
            32,
            60 + run,
        )
        z_tf_tr = apply_task_fusion(tfm, mods_tr)
        z_tf_va = apply_task_fusion(tfm, mods_va)
        z_tf_te = apply_task_fusion(tfm, mods_te)

        methods = {
            "RGB": (z_rgb_tv[tr], z_rgb_tv[va], z_rgb_te),
            "Thermal": (z_th_tv[tr], z_th_tv[va], z_th_te),
            "Concat": (cat_tr, cat_va, cat_te),
            "SupCon": (z_sc_tr, z_sc_va, z_sc_te),
            "CrossAttn": (z_ca_tr, z_ca_va, z_ca_te),
            "Transformer": (z_tf_tr, z_tf_va, z_tf_te),
            "MMCTL": (z_mm_tv[tr], z_mm_tv[va], z_mm_te),
        }

        for name, (ztr, zva, zte) in methods.items():
            print(f"  eval {name}...", flush=True)
            # structure: class / box-feat / position (P)
            m = structure_raw(zte, y_te, bf_te, pos_te, seed=40 + run)
            struct_rows.append({"run": run, "method": name, **m})
            pr = paper_task_heads(
                ztr,
                zva,
                zte,
                y_tv[tr],
                y_tv[va],
                y_te,
                box_tv[tr],
                box_tv[va],
                box_te,
                pos_tv[tr],
                pos_tv[va],
                pos_te,
                head_cfg,
            )
            # map height slot -> position for AWIR reporting
            pr = {
                "acc_pct": pr["acc_pct"],
                "r2_box": pr["r2_box"],
                "r2_pos": pr["r2_height"],
            }
            down_rows.append({"run": run, "method": name, **pr})
            print(
                f"    {name:<12} acc={pr['acc_pct']:.1f} B={pr['r2_box']:.3f} P={pr['r2_pos']:.3f}",
                flush=True,
            )

    scored = zscore_struct(struct_rows)

    def aggs(method, src):
        # structure_raw uses align_height/smooth_height for the 3rd continuous source (position here)
        key = "height" if src == "pos" else src
        v = [r[f"score_{key}"] for r in scored if r["method"] == method]
        return float(np.mean(v)), float(np.std(v))

    def aggd(method, key):
        v = [r[key] for r in down_rows if r["method"] == method]
        return float(np.mean(v)), float(np.std(v))

    structure = {
        m: {"C": aggs(m, "cls"), "B": aggs(m, "box"), "P": aggs(m, "pos")} for m in ORDER
    }
    downstream = {
        m: {"C": aggd(m, "acc_pct"), "B": aggd(m, "r2_box"), "P": aggd(m, "r2_pos")} for m in ORDER
    }

    out_json = os.path.join(ROOT, "results", "replication_awir_new_baselines.json")
    out_md = os.path.join(ROOT, "results", "replication_awir_new_baselines.md")
    payload = {
        "protocol": (
            "Released full-set AWIR embeddings (N=240); "
            "RGB/Thermal/MMCTL; baselines Concat/SupCon/CrossAttn/Transformer; "
            f"50/50 + {args.runs}-fold; paper task heads "
            f"{args.head_epochs}/{args.head_patience}"
        ),
        "data_path": args.npz_path,
        "n": 240,
        "structure": structure,
        "downstream": downstream,
        "paper_structure": PAPER_STRUCTURE,
        "paper_downstream": PAPER_DOWN,
        "methods": ORDER,
    }
    with open(out_json, "w") as fp:
        json.dump(payload, fp, indent=2)

    lines = [
        "# AWIR Replication (released full-set embeddings)",
        "",
        f"N=240; files: `awir_rgb_all.npy`, `awir_thermal_all.npy`, `awir_mmctl_all.npy`",
        f"Protocol: {payload['protocol']}",
        "",
        "## Structure (lower better)",
        "",
        "| Method | Src | Paper | Code | Δmean |",
        "|---|---|---|---|---|",
    ]
    for m in ORDER:
        for s in ("C", "B", "P"):
            pm, ps = PAPER_STRUCTURE[m][s]
            cm, cs = structure[m][s]
            if np.isnan(pm):
                lines.append(f"| {m} | {s} | — | {cm:.2f}±{cs:.2f} | — |")
            else:
                lines.append(f"| {m} | {s} | {pm:.2f}±{ps:.2f} | {cm:.2f}±{cs:.2f} | {cm - pm:+.2f} |")
    lines += [
        "",
        "## Downstream",
        "",
        "| Method | Paper C% | Code C% | Paper B | Code B | Paper P | Code P |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in ORDER:
        pc, _ = PAPER_DOWN[m]["C"]
        pb, _ = PAPER_DOWN[m]["B"]
        pp, _ = PAPER_DOWN[m]["P"]
        cc, ccs = downstream[m]["C"]
        cb, cbs = downstream[m]["B"]
        cp, cps = downstream[m]["P"]
        pcs = f"{pc:.2f}" if not np.isnan(pc) else "—"
        pbs = f"{pb:.2f}" if not np.isnan(pb) else "—"
        pps = f"{pp:.2f}" if not np.isnan(pp) else "—"
        lines.append(
            f"| {m} | {pcs} | {cc:.2f}±{ccs:.2f} | {pbs} | {cb:.2f}±{cbs:.2f} | {pps} | {cp:.2f}±{cps:.2f} |"
        )
    with open(out_md, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print("Wrote", out_json, flush=True)
    print("Wrote", out_md, flush=True)


if __name__ == "__main__":
    main()
