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


# Only embedding-level fusion methods are shown (unimodal panels removed).
FUSION_METHODS = ["Concat", "SupCon", "CrossAttn", "Transformer", "MMCTL"]
CONT_CMAP = "viridis"


def _save(fig, out_stem):
    os.makedirs(PAPER_FIG_DIR, exist_ok=True)
    os.makedirs(RESULT_FIG_DIR, exist_ok=True)
    pdf = os.path.join(PAPER_FIG_DIR, out_stem + ".pdf")
    png = os.path.join(RESULT_FIG_DIR, out_stem + ".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {pdf}\n        {png}")


def _bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#bbbbbb")


def overall_figure(embeddings, methods, y_cls, class_names, size_values,
                   size_label, out_stem, seed=SEED):
    """Fusion methods only. Rows = {PCA, t-SNE}; color = class; size = continuous
    attribute. One point per sample (no modality splitting)."""
    cmap = plt.get_cmap("tab10")
    sizes = size_from(size_values)
    sub, n_per = subsample_per_class(y_cls, n_max=20, seed=seed)

    ncols = len(methods)
    fig, axes = plt.subplots(2, ncols, figsize=(2.0 * ncols, 4.6), squeeze=False)
    for j, name in enumerate(methods):
        z = embeddings[name]
        for i, kind in enumerate(("pca", "tsne")):
            xy = project(z, kind, seed=seed)
            ax = axes[i][j]
            for c in np.unique(y_cls):
                m = y_cls[sub] == c
                ax.scatter(xy[sub][m, 0], xy[sub][m, 1], s=sizes[sub][m],
                           c=[cmap(c)], alpha=0.75, linewidths=0.3,
                           edgecolors="white")
            _bare(ax)
            if i == 0:
                ax.set_title(name, fontsize=11, pad=4)
    axes[0][0].set_ylabel("PCA", fontsize=12)
    axes[1][0].set_ylabel("t-SNE", fontsize=12)

    class_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(i),
                            markersize=8, label=str(n)) for i, n in enumerate(class_names)]
    ncol = 5 if len(class_handles) > 5 else len(class_handles)
    leg1 = fig.legend(handles=class_handles, loc="lower left", ncol=ncol, frameon=False,
                      fontsize=9, bbox_to_anchor=(0.01, -0.03), title="Class", title_fontsize=9)
    leg1._legend_box.align = "left"
    fig.add_artist(leg1)
    size_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markersize=5, label=f"small {size_label}"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
               markersize=13, label=f"large {size_label}"),
    ]
    fig.legend(handles=size_handles, loc="lower right", ncol=1, frameon=False,
               fontsize=9, bbox_to_anchor=(0.99, -0.03), title="Point size", title_fontsize=9)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    print(f"    (overall: {n_per} samples/class, fusion methods only)")
    _save(fig, out_stem)


def compute_projections(embeddings, methods, seed=SEED):
    """Project each method's full embedding once; reused across all class zooms."""
    proj = {}
    for name in methods:
        z = embeddings[name]
        proj[name] = {kind: project(z, kind, seed=seed) for kind in ("pca", "tsne")}
    return proj


def _safe(name):
    return "".join(c if c.isalnum() else "_" for c in str(name))


def zoom_figure(projections, methods, y_cls, class_names, cont_values, cont_label,
                focus_class, out_stem):
    """Zoom into one class cluster. For each fusion method, show only the focus
    class colored by the continuous attribute so a smooth spatial color gradient
    indicates a smooth continuous transition. Projections are precomputed."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    focus = list(class_names).index(focus_class)
    mask = y_cls == focus
    n = int(mask.sum())
    if n < 5:
        print(f"    (skip zoom class={focus_class}: only {n} samples)")
        return
    cvals = np.asarray(cont_values, float)[mask]
    lo, hi = np.percentile(cvals, 2), np.percentile(cvals, 98)
    if hi - lo < 1e-9:
        lo, hi = cvals.min(), cvals.max() + 1e-9
    norm = Normalize(vmin=lo, vmax=hi)

    ncols = len(methods)
    fig, axes = plt.subplots(2, ncols, figsize=(2.0 * ncols, 4.6), squeeze=False)
    for j, name in enumerate(methods):
        for i, kind in enumerate(("pca", "tsne")):
            xy_f = projections[name][kind][mask]
            ax = axes[i][j]
            ax.scatter(xy_f[:, 0], xy_f[:, 1], c=cvals, cmap=CONT_CMAP, norm=norm,
                       s=70, alpha=0.9, linewidths=0.3, edgecolors="white")
            dx = xy_f[:, 0].ptp() * 0.08 + 1e-6
            dy = xy_f[:, 1].ptp() * 0.08 + 1e-6
            ax.set_xlim(xy_f[:, 0].min() - dx, xy_f[:, 0].max() + dx)
            ax.set_ylim(xy_f[:, 1].min() - dy, xy_f[:, 1].max() + dy)
            _bare(ax)
            if i == 0:
                ax.set_title(name, fontsize=11, pad=4)
    axes[0][0].set_ylabel("PCA", fontsize=12)
    axes[1][0].set_ylabel("t-SNE", fontsize=12)

    sm = ScalarMappable(norm=norm, cmap=CONT_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05,
                        pad=0.08, aspect=40)
    cbar.set_label(f"{cont_label}  (class: {focus_class})", fontsize=10)
    fig.suptitle(f"Within-class continuous transition — {focus_class}", fontsize=12, y=1.02)
    print(f"    (zoom: class={focus_class}, n={n}, color={cont_label})")
    _save(fig, out_stem)


def combined_zoom_figure(specs, out_stem, kind="pca", seed=SEED):
    """One figure: rows = chosen (dataset, class, continuous-label), cols = fusion
    methods. Each panel shows only that class's points colored by the continuous
    label, with a per-row colorbar. specs: list of
    (dataset, class_name, cont_key, cont_label)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    data_cache, proj_cache = {}, {}
    def get(ds):
        if ds not in data_cache:
            d = load(ds)
            data_cache[ds] = d
            emb = {m: d[f"emb_{m}"] for m in FUSION_METHODS}
            proj_cache[ds] = {m: {kind: project(emb[m], kind, seed=seed)} for m in FUSION_METHODS}
        return data_cache[ds], proj_cache[ds]

    nrows, ncols = len(specs), len(FUSION_METHODS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.0 * nrows + 0.4),
                             squeeze=False)
    for r, (ds, cls, ckey, clabel) in enumerate(specs):
        d, proj = get(ds)
        names = list(d["class_names"])
        focus = names.index(cls)
        mask = d["class_label"] == focus
        cvals = np.asarray(d[ckey], float)[mask]
        lo, hi = np.percentile(cvals, 2), np.percentile(cvals, 98)
        if hi - lo < 1e-9:
            lo, hi = cvals.min(), cvals.max() + 1e-9
        norm = Normalize(vmin=lo, vmax=hi)
        for c, m in enumerate(FUSION_METHODS):
            xy = proj[m][kind][mask]
            ax = axes[r][c]
            ax.scatter(xy[:, 0], xy[:, 1], c=cvals, cmap=CONT_CMAP, norm=norm,
                       s=55, alpha=0.9, linewidths=0.3, edgecolors="white")
            dx = xy[:, 0].ptp() * 0.08 + 1e-6
            dy = xy[:, 1].ptp() * 0.08 + 1e-6
            ax.set_xlim(xy[:, 0].min() - dx, xy[:, 0].max() + dx)
            ax.set_ylim(xy[:, 1].min() - dy, xy[:, 1].max() + dy)
            _bare(ax)
            if r == 0:
                ax.set_title(m, fontsize=11, pad=4)
        axes[r][0].set_ylabel(f"{cls}\n({clabel})", fontsize=10)
        sm = ScalarMappable(norm=norm, cmap=CONT_CMAP)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=list(axes[r]), location="right", fraction=0.015,
                            pad=0.01)
        cbar.ax.tick_params(labelsize=7)
    fig.suptitle(f"Within-class continuous transition ({kind.upper()} projection)",
                 fontsize=12, y=1.0)
    print(f"    (combined zoom: {[s[1] for s in specs]}, proj={kind})")
    _save(fig, out_stem)


def zoom_all_classes(embeddings, methods, y_cls, class_names, cont_specs, out_prefix,
                     seed=SEED):
    """Generate a zoom figure for every class, for each continuous attribute.

    cont_specs: list of (values, label, filename_tag).
    """
    proj = compute_projections(embeddings, methods, seed=seed)
    for cls in class_names:
        for values, label, tag in cont_specs:
            zoom_figure(proj, methods, y_cls, class_names, values, label, cls,
                        f"{out_prefix}_zoom_{tag}_{_safe(cls)}")


def load(dataset):
    return np.load(os.path.join(EMB_DIR, f"{dataset}_fig_embeddings.npz"), allow_pickle=True)


def build_awir(seed=SEED):
    d = load("awir")
    emb = {m: d[f"emb_{m}"] for m in FUSION_METHODS}
    print("[AWIR] overall fusion figure (color=class, size=box area)")
    overall_figure(emb, FUSION_METHODS, d["class_label"], d["class_names"],
                   d["box_area"], "box area", "awir_embedding_space", seed=seed)
    print("[AWIR] per-class zooms (color=box area)")
    zoom_all_classes(emb, FUSION_METHODS, d["class_label"], d["class_names"],
                     [(d["box_area"], "box area", "box")], "awir_embedding", seed=seed)


def build_neon(seed=SEED):
    d = load("neon")
    emb = {m: d[f"emb_{m}"] for m in FUSION_METHODS}
    print("[NEON] overall fusion figure (color=class, size=box area)")
    overall_figure(emb, FUSION_METHODS, d["class_label"], d["class_names"],
                   d["box_area"], "box area", "neon_embedding_space_boxsize", seed=seed)
    print("[NEON] overall fusion figure (color=class, size=canopy height)")
    overall_figure(emb, FUSION_METHODS, d["class_label"], d["class_names"],
                   d["height"], "height", "neon_embedding_space_height", seed=seed)
    print("[NEON] per-species zooms (color=box area and canopy height)")
    zoom_all_classes(emb, FUSION_METHODS, d["class_label"], d["class_names"],
                     [(d["box_area"], "box area", "box"),
                      (d["height"], "canopy height", "height")],
                     "neon_embedding", seed=seed)


def build_neon_ablation(seed=SEED):
    d = load("neon")
    weights = [w for w in d["weight_order"]]
    emb = {f"w={w}": d[f"emb_MMCTL_w{w}"] for w in weights}
    methods = list(emb.keys())
    print("[NEON] class-weight ablation figure (sized by box area)")
    overall_figure(emb, methods, d["class_label"], d["class_names"], d["box_area"],
                   "box area", "neon_ablation_classweight_boxsize", seed=seed)
    print("[NEON] class-weight ablation figure (sized by canopy height)")
    overall_figure(emb, methods, d["class_label"], d["class_names"], d["height"],
                   "height", "neon_ablation_classweight_height", seed=seed)


# The four classes featured in the paper's per-class zoom figure.
PAPER_ZOOM_SPECS = [
    ("awir", "horse", "box_area", "box area"),
    ("awir", "cow", "box_area", "box area"),
    ("neon", "PSME", "height", "canopy height"),
    ("neon", "PIPA2", "height", "canopy height"),
]


def build_combined_zoom(seed=SEED):
    print("[combined] per-class zoom (horse, cow, PSME, PIPA2)")
    combined_zoom_figure(PAPER_ZOOM_SPECS, "embedding_zoom_perclass", kind="pca", seed=seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", nargs="+",
                   default=["awir", "neon", "neon_ablation", "combined"],
                   choices=["awir", "neon", "neon_ablation", "combined"])
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    if "awir" in args.which:
        build_awir(args.seed)
    if "neon" in args.which:
        build_neon(args.seed)
    if "neon_ablation" in args.which:
        build_neon_ablation(args.seed)
    if "combined" in args.which:
        build_combined_zoom(args.seed)


if __name__ == "__main__":
    main()
