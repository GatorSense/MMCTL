#!/usr/bin/env python3
"""
Per-class continuous-transition metric for the embedding-space zoom figures.

For each class and each continuous label, we measure how smoothly the label
varies within that class's embedding: the Spearman rank correlation between
pairwise cosine distances (embeddings) and pairwise absolute differences of the
continuous label, restricted to samples of that class. Higher = a smoother
within-class transition (the quantity visualized as a color gradient in the
zoom figures). This is the within-class analogue of the distance-preservation
column in Table tab:latent_alignment.

Uses the persisted fusion embeddings in pretrained_embeddings/figures/*.npz
(the exact arrays the zoom figures are drawn from).

Outputs:
  results/perclass_smoothness.json
  results/perclass_smoothness.md

Run:
  .venv_exp/bin/python -u scripts/perclass_smoothness.py
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
EMB_DIR = os.path.join(ROOT, "pretrained_embeddings", "figures")
FUSION = ["Concat", "SupCon", "CrossAttn", "Transformer", "MMCTL"]
MIN_N = 8


def l2(z):
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def within_class_rho(z, scalar):
    z = l2(z.astype(np.float32))
    de = pdist(z, metric="cosine")
    da = pdist(np.asarray(scalar, float).reshape(-1, 1), metric="euclidean")
    if np.std(de) < 1e-9 or np.std(da) < 1e-9:
        return float("nan")
    return float(spearmanr(de, da).correlation)


def build(dataset, cont_specs):
    d = np.load(os.path.join(EMB_DIR, f"{dataset}_fig_embeddings.npz"), allow_pickle=True)
    y = d["class_label"]
    names = list(d["class_names"])
    emb = {m: d[f"emb_{m}"] for m in FUSION}
    out = {}
    for key, label in cont_specs:
        vals = np.asarray(d[key], float)
        # Dataset-wide (all samples together) for this continuous label.
        allrow = {m: within_class_rho(emb[m], vals) for m in FUSION}
        out.setdefault(label, {})["All (dataset-wide)"] = {"n": int(len(y)), "rho": allrow}
        for ci, cn in enumerate(names):
            mask = y == ci
            if mask.sum() < MIN_N:
                continue
            row = {m: within_class_rho(emb[m][mask], vals[mask]) for m in FUSION}
            out[label][cn] = {"n": int(mask.sum()), "rho": row}
    return names, out


def md_table(label, per_class):
    lines = [f"### {label}", "",
             "| Class | n | " + " | ".join(FUSION) + " | Best |",
             "|---|---|" + "|".join(["---"] * (len(FUSION) + 1)) + "|"]
    for cn, rec in per_class.items():
        r = rec["rho"]
        best = max((m for m in FUSION if not np.isnan(r[m])), key=lambda m: r[m], default="-")
        cells = " | ".join(f"**{r[m]:.3f}**" if m == best else f"{r[m]:.3f}" for m in FUSION)
        lines.append(f"| {cn} | {rec['n']} | {cells} | {best} |")
    lines.append("")
    return lines


def main():
    awir_names, awir = build("awir", [("box_area", "box area")])
    neon_names, neon = build("neon", [("box_area", "box area"), ("height", "canopy height")])

    payload = {
        "metric": ("within-class Spearman rho between pairwise cosine embedding "
                   "distance and pairwise |delta continuous label|; higher=smoother"),
        "methods": FUSION,
        "awir": awir, "neon": neon,
    }
    with open(os.path.join(ROOT, "results", "perclass_smoothness.json"), "w") as fp:
        json.dump(payload, fp, indent=2)

    lines = ["# Per-class within-class continuous-transition metric", "",
             payload["metric"], "",
             "## AWIR", ""]
    for label, pc in awir.items():
        lines += md_table(label, pc)
    lines += ["## NEON", ""]
    for label, pc in neon.items():
        lines += md_table(label, pc)
    with open(os.path.join(ROOT, "results", "perclass_smoothness.md"), "w") as fp:
        fp.write("\n".join(lines))
    print("Wrote results/perclass_smoothness.{json,md}")


if __name__ == "__main__":
    main()
