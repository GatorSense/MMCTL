#!/usr/bin/env python3
"""
Embedding-space figures for the MMCTL paper.

Reads the compact per-method embedding archives written by
``scripts/build_figure_embeddings.py`` and renders PCA + t-SNE scatter grids:

  * Points are colored by class label.
  * Point size encodes a continuous geometric target:
      - AWIR : bounding-box area.
      - NEON : one figure sized by box area, one sized by canopy height.
  * Columns are methods (RGB, Thermal/HSI, LiDAR, Concat, SupCon, CrossAttn,
    Transformer, MMCTL); rows are the two projections. DeepSet is excluded.

Also renders the NEON class-weight ablation: MMCTL embeddings at class weights
{0.5, 0.8, 1.0, 1.5}, one grid sized by box area and one sized by height.

The story: MMCTL yields class-separated clusters whose within/between-class
geometry varies smoothly with box size / height, whereas the baselines mix
classes and show no ordered size gradient.

Run:
  .venv_exp/bin/python -u scripts/make_embedding_figures.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

EMB_DIR = os.path.join(ROOT, "pretrained_embeddings", "figures")
PAPER_FIG_DIR = os.path.join(ROOT, "paper_files", "figures")
RESULT_FIG_DIR = os.path.join(ROOT, "results", "figures")

SEED = 42
TSNE_PRE_PCA = 50  # reduce dim before t-SNE for stability/speed


def l2(z: np.ndarray) -> np.ndarray:
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def project(z: np.ndarray, kind: str, seed: int = SEED) -> np.ndarray:
    """2-D projection of L2-normalized embeddings."""
    z = l2(z.astype(np.float32))
    if kind == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(z)
    # t-SNE: pre-reduce high-dim with PCA, then embed.
    d = min(TSNE_PRE_PCA, z.shape[1], z.shape[0] - 1)
    zp = PCA(n_components=d, random_state=seed).fit_transform(z) if z.shape[1] > d else z
    perp = float(min(30, max(5, (z.shape[0] - 1) // 3)))
    ts = TSNE(n_components=2, init="pca", perplexity=perp, random_state=seed,
              learning_rate="auto")
    return ts.fit_transform(zp)


def size_from(values: np.ndarray, smin: float = 25.0, smax: float = 320.0) -> np.ndarray:
    """Map a continuous target to marker areas via robust 5-95 pct scaling."""
    v = np.asarray(values, dtype=np.float64)
    lo, hi = np.percentile(v, 5), np.percentile(v, 95)
    scaled = np.clip((v - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return smin + (smax - smin) * scaled


def subsample_per_class(y_cls: np.ndarray, n_max: int = 20, seed: int = SEED):
    """Balanced subsample: min(n_max, smallest class) points from each class.

    Projections are still fit on the full set; only the plotted points are
    subsampled, so the scatter declutters without distorting the geometry.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y_cls)
    n = int(min(n_max, min((y_cls == c).sum() for c in classes)))
    idx = []
    for c in classes:
        ci = np.where(y_cls == c)[0]
        idx.extend(rng.choice(ci, size=n, replace=False))
    return np.sort(np.array(idx)), n


# Modality layout of the stored embeddings (start:end columns), and per-modality
# marker shapes. Concat / MMCTL / weighted-MMCTL are concatenations in this order.
MOD_LAYOUT = {
    "awir": [("RGB", 0, 1024), ("Thermal", 1024, 2048)],
    "neon": [("RGB", 0, 1024), ("HSI", 1024, 2048), ("LiDAR", 2048, 3072)],
}
MARKERS = {"RGB": "o", "Thermal": "X", "HSI": "^", "LiDAR": "s", "Fused": "D"}
UNIMODAL = {"RGB", "Thermal", "HSI", "LiDAR"}
CONCAT_LIKE = {"Concat", "MMCTL"}


def method_blocks(dataset, name, emb):
    """Return [(modality_label, sub_embedding), ...] for a method panel.

    Unimodal methods -> single block with that modality's marker.
    Concatenation fusions (Concat, MMCTL, weighted MMCTL "w=...") -> split back
    into their per-modality sub-embeddings so cross-modal alignment is visible.
    Learned fusions (SupCon, CrossAttn, Transformer) collapse the modalities
    into one vector and are shown as a single "Fused" block.
    """
    if name in UNIMODAL:
        return [(name, emb)]
    if name in CONCAT_LIKE or name.startswith("w="):
        return [(lab, emb[:, a:b]) for lab, a, b in MOD_LAYOUT[dataset]]
    return [("Fused", emb)]


def project_blocks(blocks, kind, seed):
    """Co-project several equal-dim blocks in one shared 2-D space."""
    Z = np.concatenate([b for _, b in blocks], axis=0)
    xy = project(Z, kind, seed=seed)
    out, s = [], 0
    for lab, b in blocks:
        out.append((lab, xy[s:s + len(b)]))
        s += len(b)
    return out


def _scatter_panel(ax, dataset, name, emb, sub, y_cls, sizes, cmap, kind, seed,
                   title=None):
    blocks = method_blocks(dataset, name, emb)
    for lab, xy in project_blocks(blocks, kind, seed):
        marker = MARKERS.get(lab, "o")
        for c in np.unique(y_cls):
            m = y_cls[sub] == c
            if not np.any(m):
                continue
            ax.scatter(xy[sub][m, 0], xy[sub][m, 1], s=sizes[sub][m],
                       c=[cmap(c)], marker=marker, alpha=0.7, linewidths=0.4,
                       edgecolors="white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#bbbbbb")
    if title:
        ax.set_title(title, fontsize=11, pad=4)
    return [lab for lab, _ in blocks]


def _legends(fig, class_names, cmap, size_label, used_modalities):
    """Class legend (color) bottom-left; modality (marker) bottom-center;
    point-size legend bottom-right."""
    class_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(i),
                            markersize=8, label=str(n))
                     for i, n in enumerate(class_names)]
    ncol = 5 if len(class_handles) > 5 else len(class_handles)
    leg1 = fig.legend(handles=class_handles, loc="lower left", ncol=ncol,
                      frameon=False, fontsize=9, bbox_to_anchor=(0.01, -0.04),
                      title="Class", title_fontsize=9)
    leg1._legend_box.align = "left"
    fig.add_artist(leg1)

    mod_order = [m for m in ("RGB", "Thermal", "HSI", "LiDAR", "Fused")
                 if m in used_modalities]
    mod_handles = [Line2D([0], [0], marker=MARKERS[m], color="w",
                          markerfacecolor="#555555", markeredgecolor="#555555",
                          markersize=8, label=m) for m in mod_order]
    leg2 = fig.legend(handles=mod_handles, loc="lower center", ncol=len(mod_handles),
                      frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04),
                      title="Modality", title_fontsize=9)
    fig.add_artist(leg2)

    size_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markersize=5, label=f"small {size_label}"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markersize=13, label=f"large {size_label}"),
    ]
    fig.legend(handles=size_handles, loc="lower right", ncol=1, frameon=False,
               fontsize=9, bbox_to_anchor=(0.99, -0.04), title="Point size",
               title_fontsize=9)


def grid_figure(dataset, embeddings: dict, methods, y_cls, class_names, size_values,
                size_label, out_stem, seed=SEED):
    """Rows = {PCA, t-SNE}, cols = methods. Saves PDF (paper) + PNG (results)."""
    cmap = plt.get_cmap("tab10")
    sizes = size_from(size_values)
    sub, n_per = subsample_per_class(y_cls, n_max=20, seed=seed)

    ncols = len(methods)
    fig, axes = plt.subplots(2, ncols, figsize=(1.9 * ncols, 4.4), squeeze=False)

    used = set()
    for j, name in enumerate(methods):
        z = embeddings[name]
        for i, kind in enumerate(("pca", "tsne")):
            title = name if i == 0 else None
            labs = _scatter_panel(axes[i][j], dataset, name, z, sub, y_cls, sizes,
                                  cmap, kind, seed, title=title)
            used.update(labs)

    axes[0][0].set_ylabel("PCA", fontsize=12)
    axes[1][0].set_ylabel("t-SNE", fontsize=12)

    _legends(fig, class_names, cmap, size_label, used)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    print(f"    (plotting {n_per} samples/class; modalities: {sorted(used)})")

    os.makedirs(PAPER_FIG_DIR, exist_ok=True)
    os.makedirs(RESULT_FIG_DIR, exist_ok=True)
    pdf = os.path.join(PAPER_FIG_DIR, out_stem + ".pdf")
    png = os.path.join(RESULT_FIG_DIR, out_stem + ".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}\n        {png}")


def load(dataset):
    return np.load(os.path.join(EMB_DIR, f"{dataset}_fig_embeddings.npz"), allow_pickle=True)


def build_awir(seed=SEED):
    d = load("awir")
    methods = [m for m in d["method_order"]]
    emb = {m: d[f"emb_{m}"] for m in methods}
    print("[AWIR] embedding-space figure (sized by box area)")
    grid_figure("awir", emb, methods, d["class_label"], d["class_names"], d["box_area"],
                "box area", "awir_embedding_space", seed=seed)


def build_neon(seed=SEED):
    d = load("neon")
    methods = [m for m in d["method_order"]]
    emb = {m: d[f"emb_{m}"] for m in methods}
    print("[NEON] embedding-space figure (sized by box area)")
    grid_figure("neon", emb, methods, d["class_label"], d["class_names"], d["box_area"],
                "box area", "neon_embedding_space_boxsize", seed=seed)
    print("[NEON] embedding-space figure (sized by canopy height)")
    grid_figure("neon", emb, methods, d["class_label"], d["class_names"], d["height"],
                "height", "neon_embedding_space_height", seed=seed)


def build_neon_ablation(seed=SEED):
    d = load("neon")
    weights = [w for w in d["weight_order"]]
    emb = {f"w={w}": d[f"emb_MMCTL_w{w}"] for w in weights}
    methods = list(emb.keys())
    print("[NEON] class-weight ablation figure (sized by box area)")
    grid_figure("neon", emb, methods, d["class_label"], d["class_names"], d["box_area"],
                "box area", "neon_ablation_classweight_boxsize", seed=seed)
    print("[NEON] class-weight ablation figure (sized by canopy height)")
    grid_figure("neon", emb, methods, d["class_label"], d["class_names"], d["height"],
                "height", "neon_ablation_classweight_height", seed=seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", nargs="+",
                   default=["awir", "neon", "neon_ablation"],
                   choices=["awir", "neon", "neon_ablation"])
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    if "awir" in args.which:
        build_awir(args.seed)
    if "neon" in args.which:
        build_neon(args.seed)
    if "neon_ablation" in args.which:
        build_neon_ablation(args.seed)


if __name__ == "__main__":
    main()
