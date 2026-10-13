#!/usr/bin/env python3
"""
Collect and persist per-method embeddings + metadata for the paper's
embedding-space figures (t-SNE / PCA), for both AWIR and NEON.

Why this exists
---------------
The new embedding figure compares *every* fusion method that has been run
(RGB, Thermal/HSI, LiDAR, Concat, SupCon, CrossAttn, Transformer, MMCTL) and
must be regenerated easily. We therefore materialize a single compact ``.npz``
per dataset holding each method's embedding matrix plus the class labels and
the continuous size targets used for scatter-point sizing (box area for AWIR,
box area and canopy height for NEON). ``scripts/make_embedding_figures.py``
reads only these ``.npz`` files, so re-plotting never re-trains anything.

Supervised fusion baselines (SupCon / CrossAttn / Transformer) are learned
transforms of the per-modality embeddings. For a stable, leakage-controlled
visualization we train them on a fixed 50/50 stratified *train* split and apply
them to all points (same protocol family as the quantitative eval scripts).
DeepSet is intentionally excluded.

Outputs (git-tracked, .npy/.npz are force-added under pretrained_embeddings/):
  pretrained_embeddings/figures/awir_fig_embeddings.npz
  pretrained_embeddings/figures/neon_fig_embeddings.npz

Run:
  .venv_exp/bin/python -u scripts/build_figure_embeddings.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Reuse the exact fusion baselines used in the quantitative evaluation.
from scripts.eval_neon_released_embeddings import (  # noqa: E402
    apply_supcon,
    apply_task_fusion,
    train_supcon,
    train_task_fusion,
)

# Data / released embeddings live in the main checkout (npy/npz are gitignored
# in the worktree). Default to the main checkout; override with flags if needed.
MAIN = "."

OUT_DIR = os.path.join(ROOT, "pretrained_embeddings", "figures")

FUSION_EPOCHS = 200
SEED = 42


def _l2(z: np.ndarray) -> np.ndarray:
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def _build_fusion_embeddings(mods_all: np.ndarray, y_cls: np.ndarray, box2: np.ndarray,
                             h2: np.ndarray, seed: int = SEED):
    """Train SupCon / CrossAttn / Transformer on a train split, apply to all.

    mods_all: (N, n_mod, D) stacked per-modality embeddings.
    Returns dict name -> (N, d) embeddings.
    """
    n = mods_all.shape[0]
    idx = np.arange(n)
    tr, _ = train_test_split(idx, test_size=0.5, stratify=y_cls, random_state=seed)

    cat_all = mods_all.reshape(n, -1)  # (N, n_mod*D)

    out = {}

    # SupCon operates on the concatenated feature vector.
    sup = train_supcon(
        z_cat_tr=cat_all[tr].astype(np.float32),
        y_tr=y_cls[tr],
        z_cat_va=cat_all[tr].astype(np.float32),
        epochs=FUSION_EPOCHS,
        batch_size=128,
        seed=seed,
    )
    out["SupCon"] = apply_supcon(sup, cat_all.astype(np.float32))

    # CrossAttn / Transformer operate on the (N, n_mod, D) token set.
    for kind in ("CrossAttn", "Transformer"):
        model = train_task_fusion(
            fusion_kind=kind,
            mods_tr=mods_all[tr].astype(np.float32),
            mods_va=mods_all[tr].astype(np.float32),
            y_cls_tr=y_cls[tr],
            y_cls_va=y_cls[tr],
            y_box_tr=box2[tr],
            y_box_va=box2[tr],
            y_h_tr=h2[tr],
            y_h_va=h2[tr],
            epochs=FUSION_EPOCHS,
            batch_size=128,
            seed=seed,
        )
        out[kind] = apply_task_fusion(model, mods_all.astype(np.float32))
    return out


# ----------------------------------------------------------------------------
# AWIR
# ----------------------------------------------------------------------------
def build_awir(args):
    data = np.load(os.path.join(args.main, "awir_data.npz"), allow_pickle=True)
    cls_raw = data["class_label"]  # strings: cow/deer/horse
    box = data["box_label"].astype(np.float64)  # [xmin, xmax, ymin, ymax], px in [0,300]

    le = LabelEncoder()
    y_cls = le.fit_transform(cls_raw).astype(np.int64)
    class_names = le.classes_.astype(str)

    # Box area as the size target (normalized by image area 300x300).
    w = np.clip(box[:, 1] - box[:, 0], 0, None)
    h = np.clip(box[:, 3] - box[:, 2], 0, None)
    box_area = (w * h) / (300.0 * 300.0)

    z_rgb = np.load(os.path.join(args.main, "awir_rgb_all.npy")).astype(np.float32)
    z_thm = np.load(os.path.join(args.main, "awir_thermal_all.npy")).astype(np.float32)
    z_mm = np.load(os.path.join(args.main, "awir_mmctl_all.npy")).astype(np.float32)
    n = z_rgb.shape[0]
    assert z_thm.shape[0] == n == box_area.shape[0], (z_thm.shape, n, box_area.shape)

    methods = {
        "RGB": z_rgb,
        "Thermal": z_thm,
        "Concat": np.concatenate([z_rgb, z_thm], axis=1),
        "MMCTL": z_mm,
    }

    # Fusion baselines over the two modality embeddings.
    mods_all = np.stack([z_rgb, z_thm], axis=1)  # (N, 2, 1024)
    # AWIR downstream box target for fusion training: center/width/height style
    # (4-d), height target reuses box geometry (no separate height in AWIR).
    box_cenwh = np.stack([
        (box[:, 0] + box[:, 1]) / 2 / 300.0,
        (box[:, 2] + box[:, 3]) / 2 / 300.0,
        w / 300.0,
        h / 300.0,
    ], axis=1).astype(np.float32)
    fusion = _build_fusion_embeddings(mods_all, y_cls, box_cenwh, box_cenwh, seed=SEED)
    methods.update(fusion)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "awir_fig_embeddings.npz")
    save = {f"emb_{k}": v.astype(np.float32) for k, v in methods.items()}
    save.update({
        "method_order": np.array(["RGB", "Thermal", "Concat", "SupCon",
                                  "CrossAttn", "Transformer", "MMCTL"]),
        "class_label": y_cls,
        "class_names": class_names,
        "box_area": box_area.astype(np.float32),
    })
    np.savez_compressed(out_path, **save)
    print(f"[AWIR] N={n} methods={list(methods)} -> {out_path}")
    for k, v in methods.items():
        print(f"   {k:<12} {v.shape}")


# ----------------------------------------------------------------------------
# NEON
# ----------------------------------------------------------------------------
def build_neon(args):
    data = np.load(os.path.join(args.main, "neon_data.npz"), allow_pickle=True)
    idx = np.load(os.path.join(args.main, "outputs", "neon_emb_index_592.npy"))
    assert len(idx) == 592, len(idx)

    string_label = data["string_label"][idx]
    le = LabelEncoder()
    y_cls = le.fit_transform(string_label).astype(np.int64)
    class_names = le.classes_.astype(str)

    box_info = data["box_info"][idx].astype(np.float64)      # (592, 2) ~ [w, h]
    height_info = data["height_info"][idx].astype(np.float64)  # (592, 4)

    box_area = box_info[:, 0] * box_info[:, 1]
    box_area = box_area / (np.max(box_area) + 1e-9)
    height = height_info[:, 0]  # primary canopy height (m)

    z_rgb = np.load(os.path.join(args.main, "neon_rgb_all_6.npy")).astype(np.float32)
    z_hsi = np.load(os.path.join(args.main, "neon_hsi_all_6.npy")).astype(np.float32)
    z_lid = np.load(os.path.join(args.main, "neon_lidar_all_6.npy")).astype(np.float32)

    weight_files = {
        "0.5": "neon_mmctl_all_6_0.5.npy",
        "0.8": "neon_mmctl_all_6_0.8.npy",
        "1.0": "neon_mmctl_all_6_1.0.npy",
        "1.5": "neon_mmctl_all_6_1.5.npy",
    }
    mmctl = {w: np.load(os.path.join(args.main, f)).astype(np.float32)
             for w, f in weight_files.items()}

    n = z_rgb.shape[0]
    assert n == 592 and mmctl["1.0"].shape == (592, 3072), (n, mmctl["1.0"].shape)

    methods = {
        "RGB": z_rgb,
        "HSI": z_hsi,
        "LiDAR": z_lid,
        "Concat": np.concatenate([z_rgb, z_hsi, z_lid], axis=1),
        "MMCTL": mmctl["1.0"],
    }

    mods_all = np.stack([z_rgb, z_hsi, z_lid], axis=1)  # (N, 3, 1024)
    box2 = np.stack([box_info[:, 0], box_info[:, 1]], axis=1).astype(np.float32)
    h2 = height_info[:, :2].astype(np.float32)
    fusion = _build_fusion_embeddings(mods_all, y_cls, box2, h2, seed=SEED)
    methods.update(fusion)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "neon_fig_embeddings.npz")
    save = {f"emb_{k}": v.astype(np.float32) for k, v in methods.items()}
    # Per-weight MMCTL for the class-weight ablation figures.
    for w, arr in mmctl.items():
        save[f"emb_MMCTL_w{w}"] = arr.astype(np.float32)
    save.update({
        "method_order": np.array(["RGB", "HSI", "LiDAR", "Concat", "SupCon",
                                  "CrossAttn", "Transformer", "MMCTL"]),
        "weight_order": np.array(["0.5", "0.8", "1.0", "1.5"]),
        "class_label": y_cls,
        "class_names": class_names,
        "box_area": box_area.astype(np.float32),
        "height": height.astype(np.float32),
        "emb_index_592": idx,
    })
    np.savez_compressed(out_path, **save)
    print(f"[NEON] N={n} methods={list(methods)} + weights={list(mmctl)} -> {out_path}")
    for k, v in methods.items():
        print(f"   {k:<12} {v.shape}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main", default=MAIN, help="Main checkout holding data + released embeddings")
    p.add_argument("--dataset", choices=["awir", "neon", "both"], default="both")
    args = p.parse_args()

    if args.dataset in ("awir", "both"):
        build_awir(args)
    if args.dataset in ("neon", "both"):
        build_neon(args)


if __name__ == "__main__":
    main()
