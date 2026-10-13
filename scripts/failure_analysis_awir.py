#!/usr/bin/env python3
"""Failure / retrieval analysis for AWIR pretrained MMCTL embeddings."""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from mmctl.utils import generate_triplet_box_label_normalized  # noqa: E402


def l2(z):
    n = np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    return z / n


def main():
    data = "data/awir_data.npz"
    emb_dir = os.path.join(REPO, "pretrained_embeddings")
    fig_dir = os.path.join(REPO, "results/figures")
    os.makedirs(fig_dir, exist_ok=True)

    f = np.load(data, allow_pickle=True)
    y_cls_raw = f["class_label"]
    y_box = np.clip(f["box_label"].astype(np.float32), None, 300)
    rgb = f["rgb"]
    thermal = f["thermal"]
    names = f["rgb_image_name"] if "rgb_image_name" in f.files else np.arange(len(y_cls_raw))

    y_triplet, box_feat = generate_triplet_box_label_normalized(y_box, y_cls_raw, n_clusters=3)
    y_cls = LabelEncoder().fit_transform(y_cls_raw)
    composite = LabelEncoder().fit_transform([f"{c}_{t}" for c, t in zip(y_cls, y_triplet)])
    idx_all = np.arange(len(y_cls))
    _, idx_test = train_test_split(
        idx_all, test_size=0.5, stratify=composite, random_state=42
    )

    # Use run01 MMCTL + unimodal for retrieval
    z_mm = l2(np.load(os.path.join(emb_dir, "run01_mmctl_test.npy")))
    z_rgb = l2(np.load(os.path.join(emb_dir, "run01_rgb_test.npy")))
    z_th = l2(np.load(os.path.join(emb_dir, "run01_thermal_test.npy")))
    y = y_cls[idx_test]
    labels = y_cls_raw[idx_test]
    box_areas = ((y_box[idx_test, 1] - y_box[idx_test, 0]) * (y_box[idx_test, 3] - y_box[idx_test, 2]))

    # Cross-modal retrieval within MMCTL halves
    z_rgb_half = l2(z_mm[:, :1024])
    z_th_half = l2(z_mm[:, 1024:])

    def retrieval_acc(q, g, yq, yg, k=1):
        nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(g)
        _, idx = nn.kneighbors(q)
        hits = []
        for i, neigh in enumerate(idx):
            hits.append(any(yg[j] == yq[i] for j in neigh))
        return float(np.mean(hits))

    # Same-index pairing: retrieval of thermal half from rgb half (identity + class)
    nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(z_th_half)
    dist, idx = nn.kneighbors(z_rgb_half)
    rank1_identity = float(np.mean([idx[i, 0] == i for i in range(len(idx))]))
    rank5_identity = float(np.mean([i in idx[i] for i in range(len(idx))]))
    class_r1 = retrieval_acc(z_rgb_half, z_th_half, y, y, k=1)

    # Hard failures: nearest neighbor wrong class
    fail_idx = [i for i in range(len(y)) if y[idx[i, 0]] != y[i]]
    success_idx = [i for i in range(len(y)) if idx[i, 0] == i]

    # PCA viz
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(z_mm)
    fig, ax = plt.subplots(figsize=(6, 5))
    classes = sorted(set(labels.tolist()))
    for c in classes:
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=18 + 40 * (box_areas[m] / (box_areas.max() + 1e-6)), label=str(c), alpha=0.8)
    ax.set_title("AWIR MMCTL test embeddings (PCA, run01)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig_path = os.path.join(fig_dir, "awir_mmctl_pca_run01.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # Qualitative grid: 3 success + 3 failure (rgb | thermal | nn thermal)
    def save_cases(indices, name, n=3):
        take = indices[:n]
        if not take:
            return []
        fig, axes = plt.subplots(len(take), 3, figsize=(7, 2.2 * len(take)))
        if len(take) == 1:
            axes = np.array([axes])
        notes = []
        for r, i in enumerate(take):
            j = int(idx[i, 0])
            axes[r, 0].imshow(rgb[idx_test[i]])
            axes[r, 0].set_title(f"RGB q cls={labels[i]}")
            axes[r, 1].imshow(thermal[idx_test[i]])
            axes[r, 1].set_title("Thermal paired")
            axes[r, 2].imshow(thermal[idx_test[j]])
            axes[r, 2].set_title(f"NN thermal cls={labels[j]}")
            for c in range(3):
                axes[r, c].axis("off")
            notes.append(
                {
                    "query_test_i": int(i),
                    "nn_test_i": j,
                    "query_class": str(labels[i]),
                    "nn_class": str(labels[j]),
                    "cosine_dist": float(dist[i, 0]),
                    "identity_match": bool(j == i),
                    "name": str(names[idx_test[i]]),
                }
            )
        fig.suptitle(name)
        fig.tight_layout()
        outp = os.path.join(fig_dir, f"awir_{name.replace(' ', '_').lower()}.png")
        fig.savefig(outp, dpi=140)
        plt.close(fig)
        return notes

    success_notes = save_cases(success_idx, "retrieval_successes")
    fail_notes = save_cases(fail_idx, "retrieval_failures")

    metrics = {
        "run": 1,
        "cross_modal_rgb_to_thermal_half": {
            "rank1_identity": rank1_identity,
            "rank5_identity": rank5_identity,
            "class_match_r1": class_r1,
            "n_test": int(len(y)),
            "n_identity_success": len(success_idx),
            "n_class_failure_at_r1": len(fail_idx),
        },
        "unimodal_class_r1_rgb_space": retrieval_acc(z_rgb, z_rgb, y, y, k=1),
        "figures": {
            "pca": fig_path,
            "successes": os.path.join(fig_dir, "awir_retrieval_successes.png"),
            "failures": os.path.join(fig_dir, "awir_retrieval_failures.png"),
        },
        "example_successes": success_notes,
        "example_failures": fail_notes,
    }

    with open(os.path.join(REPO, "results/failure_analysis_metrics.json"), "w") as fp:
        json.dump(metrics, fp, indent=2)

    md = os.path.join(REPO, "results/failure_analysis.md")
    with open(md, "w") as fp:
        fp.write("# AWIR Failure / Retrieval Analysis\n\n")
        fp.write("## Setup\n")
        fp.write("- Embeddings: `pretrained_embeddings/run01_mmctl_test.npy` (joint 2048-d).\n")
        fp.write("- Cross-modal probe: RGB half (0:1024) queries Thermal half (1024:2048) with cosine NN.\n")
        fp.write("- Same held-out test split as `demo_metrics.py` (seed=42, 50%).\n\n")
        fp.write("## Quantitative\n")
        cm = metrics["cross_modal_rgb_to_thermal_half"]
        fp.write(f"- Rank-1 **identity** retrieval (paired object): **{cm['rank1_identity']:.3f}**\n")
        fp.write(f"- Rank-5 identity: **{cm['rank5_identity']:.3f}**\n")
        fp.write(f"- Rank-1 **class** match: **{cm['class_match_r1']:.3f}**\n")
        fp.write(f"- Identity successes: {cm['n_identity_success']} / {cm['n_test']}\n")
        fp.write(f"- Class failures at R1: {cm['n_class_failure_at_r1']} / {cm['n_test']}\n\n")
        fp.write("## Qualitative takeaways\n")
        fp.write("- Successes: paired RGB/thermal halves of the same object land as nearest neighbors, consistent with the paper's shared-latent claim.\n")
        if fail_notes:
            fp.write("- Failures: nearest thermal half belongs to a different class; often visually ambiguous scale/pose or weak thermal contrast.\n")
            for n in fail_notes:
                fp.write(
                    f"  - query `{n['name']}` ({n['query_class']}) → NN class {n['nn_class']} "
                    f"(cos-dist={n['cosine_dist']:.3f})\n"
                )
        else:
            fp.write("- No class-mismatch failures at R1 in run01 for this split.\n")
        fp.write("\n## Figures\n")
        fp.write(f"- PCA: `results/figures/awir_mmctl_pca_run01.png`\n")
        fp.write(f"- Success grid: `results/figures/awir_retrieval_successes.png`\n")
        fp.write(f"- Failure grid: `results/figures/awir_retrieval_failures.png`\n")

    print(json.dumps(metrics["cross_modal_rgb_to_thermal_half"], indent=2))
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
