#!/usr/bin/env python3
"""
Standard embedding-space quality metrics for AWIR and NEON.

Replaces the custom z-scored "structure score" with widely used, directly
interpretable metrics:

  * Class separation  -> silhouette score (cosine), using class labels.
                         Range [-1, 1]; higher = better-separated classes.
  * Continuous-annotation representation -> Spearman rank correlation between
    pairwise embedding (cosine) distances and pairwise annotation (Euclidean)
    distances, computed separately for box geometry and for position/height.
    Range [-1, 1]; higher = embedding distances track annotation differences.

Protocol: StratifiedKFold over the released full-set embeddings (8 folds AWIR,
5 folds NEON). Each fold's held-out split is the evaluation set; the learned
fusion baselines (SupCon, cross-attention, Transformer) are trained on the
remaining folds and applied to the held-out split, so every method (including
the deterministic released embeddings) yields a genuine mean +/- std.

Outputs:
  results/embedding_space_metrics.json
  results/embedding_space_metrics.md

Run:
  .venv_exp/bin/python -u scripts/embedding_space_metrics.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.eval_neon_released_embeddings import (  # noqa: E402
    apply_supcon,
    apply_task_fusion,
    train_supcon,
    train_task_fusion,
)

MAIN = "/Users/meilunzhou/Documents/GitHub/MMCTL"
FUSION_EPOCHS = 200
SEED = 42


def l2(z):
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def silhouette_cls(z, y_cls):
    z = l2(z.astype(np.float32))
    if len(np.unique(y_cls)) < 2:
        return float("nan")
    return float(silhouette_score(z, y_cls, metric="cosine"))


def spearman_dist(z, ann):
    """Spearman rho between pairwise cosine (embedding) and Euclidean (annotation)
    distances. Higher => embedding geometry reflects continuous annotation."""
    z = l2(z.astype(np.float32))
    de = pdist(z, metric="cosine")
    da = pdist(np.nan_to_num(ann.astype(np.float64)), metric="euclidean")
    if np.std(de) < 1e-12 or np.std(da) < 1e-12:
        return float("nan")
    return float(spearmanr(de, da).correlation)


def agg(vals):
    v = np.array([x for x in vals if not np.isnan(x)], float)
    if v.size == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std())


def eval_dataset(name, methods_full, mods_full, y_cls, box_ann, pos_ann,
                 cont_names, n_folds, seed=SEED):
    """methods_full: name -> (N,d) released embeddings (deterministic ones).
    mods_full: (N, n_mod, D) per-modality embeddings for training fusions.
    box_ann/pos_ann: (N, .) continuous annotation matrices.
    cont_names: (box_label, pos_label) column headers, e.g. ("B", "P")."""
    n = y_cls.shape[0]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    method_names = list(methods_full.keys()) + ["SupCon", "CrossAttn", "Transformer"]
    rows = {m: {"sil": [], "box": [], "pos": []} for m in method_names}

    for fold, (tr, te) in enumerate(skf.split(np.zeros(n), y_cls), start=1):
        # Learned fusions: train on tr, evaluate on te.
        cat_tr = mods_full[tr].reshape(len(tr), -1).astype(np.float32)
        sup = train_supcon(cat_tr, y_cls[tr], cat_tr, FUSION_EPOCHS, 128, seed + fold)
        fus = {"SupCon": apply_supcon(sup, mods_full[te].reshape(len(te), -1).astype(np.float32))}
        for kind in ("CrossAttn", "Transformer"):
            mdl = train_task_fusion(
                kind, mods_full[tr].astype(np.float32), mods_full[tr].astype(np.float32),
                y_cls[tr], y_cls[tr], box_ann[tr], box_ann[tr], pos_ann[tr], pos_ann[tr],
                FUSION_EPOCHS, 128, seed + fold,
            )
            fus[kind] = apply_task_fusion(mdl, mods_full[te].astype(np.float32))

        te_emb = {m: methods_full[m][te] for m in methods_full}
        te_emb.update(fus)

        for m, z in te_emb.items():
            rows[m]["sil"].append(silhouette_cls(z, y_cls[te]))
            rows[m]["box"].append(spearman_dist(z, box_ann[te]))
            rows[m]["pos"].append(spearman_dist(z, pos_ann[te]))
        print(f"  [{name}] fold {fold}/{n_folds} done", flush=True)

    out = {}
    for m in method_names:
        out[m] = {
            "silhouette": agg(rows[m]["sil"]),
            "spearman_box": agg(rows[m]["box"]),
            "spearman_pos": agg(rows[m]["pos"]),
        }
    return out, method_names, rows


def wilcoxon_vs_best(rows, metric_key, mmctl="MMCTL"):
    """Paired one-sided Wilcoxon: MMCTL > best-competing baseline on this metric.

    The competitor is the non-MMCTL method with the highest mean. Returns
    (competitor, p_value) or (competitor, nan) if the test is undefined.
    """
    from scipy.stats import wilcoxon
    means = {m: np.nanmean(rows[m][metric_key]) for m in rows if m != mmctl}
    comp = max(means, key=means.get)
    a = np.array(rows[mmctl][metric_key], float)
    b = np.array(rows[comp][metric_key], float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.allclose(a, b):
        return comp, float("nan")
    try:
        p = wilcoxon(a, b, alternative="greater").pvalue
    except ValueError:
        return comp, float("nan")
    return comp, float(p)


def build_awir(args):
    d = np.load(os.path.join(args.main, "awir_data.npz"), allow_pickle=True)
    y_cls = LabelEncoder().fit_transform(d["class_label"]).astype(np.int64)
    box = d["box_label"].astype(np.float64)  # xmin,xmax,ymin,ymax
    w = (box[:, 1] - box[:, 0]) / 300.0
    h = (box[:, 3] - box[:, 2]) / 300.0
    cx = (box[:, 0] + box[:, 1]) / 2 / 300.0
    cy = (box[:, 2] + box[:, 3]) / 2 / 300.0
    box_ann = np.stack([w, h], axis=1)          # box geometry
    pos_ann = np.stack([cx, cy], axis=1)        # position

    z_rgb = np.load(os.path.join(args.main, "awir_rgb_all.npy")).astype(np.float32)
    z_thm = np.load(os.path.join(args.main, "awir_thermal_all.npy")).astype(np.float32)
    z_mm = np.load(os.path.join(args.main, "awir_mmctl_all.npy")).astype(np.float32)
    methods_full = {
        "RGB": z_rgb, "Thermal": z_thm,
        "Concat": np.concatenate([z_rgb, z_thm], 1), "MMCTL": z_mm,
    }
    mods_full = np.stack([z_rgb, z_thm], axis=1)
    order = ["RGB", "Thermal", "Concat", "SupCon", "CrossAttn", "Transformer", "MMCTL"]
    res, _, rows = eval_dataset("AWIR", methods_full, mods_full, y_cls, box_ann, pos_ann,
                                ("B", "P"), n_folds=args.awir_folds)
    sig = {k: wilcoxon_vs_best(rows, k) for k in ("sil", "box", "pos")}
    return {"order": order, "cont": ("B", "P"), "results": res, "wilcoxon": sig}


def build_neon(args):
    d = np.load(os.path.join(args.main, "neon_data.npz"), allow_pickle=True)
    idx = np.load(os.path.join(args.main, "outputs", "neon_emb_index_592.npy"))
    y_cls = LabelEncoder().fit_transform(d["string_label"][idx]).astype(np.int64)
    box_ann = d["box_info"][idx].astype(np.float64)       # box geometry (2-d)
    h_ann = d["height_info"][idx].astype(np.float64)      # canopy height (4-d)

    z_rgb = np.load(os.path.join(args.main, "neon_rgb_all_6.npy")).astype(np.float32)
    z_hsi = np.load(os.path.join(args.main, "neon_hsi_all_6.npy")).astype(np.float32)
    z_lid = np.load(os.path.join(args.main, "neon_lidar_all_6.npy")).astype(np.float32)
    z_mm = np.load(os.path.join(args.main, "neon_mmctl_all_6_1.0.npy")).astype(np.float32)
    methods_full = {
        "RGB": z_rgb, "HSI": z_hsi, "LiDAR": z_lid,
        "Concat": np.concatenate([z_rgb, z_hsi, z_lid], 1), "MMCTL": z_mm,
    }
    mods_full = np.stack([z_rgb, z_hsi, z_lid], axis=1)
    order = ["RGB", "HSI", "LiDAR", "Concat", "SupCon", "CrossAttn", "Transformer", "MMCTL"]
    res, _, rows = eval_dataset("NEON", methods_full, mods_full, y_cls, box_ann, h_ann,
                                ("B", "H"), n_folds=args.neon_folds)
    sig = {k: wilcoxon_vs_best(rows, k) for k in ("sil", "box", "pos")}
    return {"order": order, "cont": ("B", "H"), "results": res, "wilcoxon": sig}


def fmt(pair):
    m, s = pair
    if np.isnan(m):
        return "--"
    return f"{m:.3f}$\\pm${s:.3f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main", default=MAIN)
    p.add_argument("--awir_folds", type=int, default=8)
    p.add_argument("--neon_folds", type=int, default=5)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.awir_folds, args.neon_folds = 3, 3

    print("=== AWIR ===", flush=True)
    awir = build_awir(args)
    print("=== NEON ===", flush=True)
    neon = build_neon(args)

    payload = {
        "protocol": (
            f"StratifiedKFold on released full-set embeddings "
            f"({args.awir_folds} AWIR / {args.neon_folds} NEON); silhouette (cosine, "
            f"class) and Spearman rho between pairwise cosine embedding distances and "
            f"Euclidean annotation distances (box, position/height); learned fusions "
            f"trained per fold."),
        "awir": awir, "neon": neon,
    }
    out_json = os.path.join(ROOT, "results", "embedding_space_metrics.json")
    with open(out_json, "w") as fp:
        json.dump(payload, fp, indent=2)

    lines = ["# Embedding-space quality metrics", "",
             payload["protocol"], "",
             "Silhouette (class separation) and Spearman rho (continuous-annotation "
             "distance preservation); higher is better for all columns.", ""]
    for ds, blk in (("AWIR", awir), ("NEON", neon)):
        bcol, pcol = blk["cont"]
        lines += [f"## {ds}", "",
                  f"| Method | Silhouette (class) | Spearman ρ ({bcol}) | Spearman ρ ({pcol}) |",
                  "|---|---|---|---|"]
        for m in blk["order"]:
            r = blk["results"][m]
            lines.append(f"| {m} | {fmt(r['silhouette'])} | {fmt(r['spearman_box'])} "
                         f"| {fmt(r['spearman_pos'])} |")
        lines.append("")
        sig = blk.get("wilcoxon", {})
        km = {"sil": "Sil", "box": f"rho {bcol}", "pos": f"rho {pcol}"}
        lines.append("Paired one-sided Wilcoxon (MMCTL > best competitor):")
        for k, lab in km.items():
            comp, p = sig.get(k, ("-", float("nan")))
            ptxt = "n/a" if np.isnan(p) else f"p={p:.4f}"
            lines.append(f"- {lab}: vs {comp}, {ptxt}")
        lines.append("")
    out_md = os.path.join(ROOT, "results", "embedding_space_metrics.md")
    with open(out_md, "w") as fp:
        fp.write("\n".join(lines))
    print("\nWrote", out_json, "\n      ", out_md, flush=True)


if __name__ == "__main__":
    main()
