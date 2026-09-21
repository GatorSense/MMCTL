#!/usr/bin/env python3
"""
Train NEON MMCTL using the user's paper reconstruction stack:
  TripletNetwork_Online.image_reconstruction_multi_modalv1
  + continuous multimodal triplet + RGB/HSI/LiDAR recon losses

Data: /Users/meilunzhou/Documents/GitHub/CTL/neon_data.npz
Paper settings for main tables: class/box/height weights = 1.0, 5 CV folds.

Then extract embeddings and compare to paper NEON tables.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import numpy as np
import tensorflow as tf
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TripletNetwork_Online import image_reconstruction_multi_modalv1  # noqa: E402
from neon_custom_losses import (  # noqa: E402
    hybrid_hsi_loss,
    keras_batch_all_triplet_continuous_loss_multimodal,
    lidar_ssim_loss,
    rgb_ssim_loss,
)
from neon_utilities import (  # noqa: E402
    filter_and_remap_top_classes,
    global_max_normalize,
    remove_bands,
    resampling,
    weigh_features_multi,
)
from mmctl.embedding_metrics import (  # noqa: E402
    embedding_annotation_alignment_ratio,
    local_smoothness_std,
)
from mmctl.train_taskheads import (  # noqa: E402
    HeadTrainConfig,
    standardize_z,
    train_three_heads,
)
from mmctl.utils import generate_triplet_box_label_normalized  # noqa: E402

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

NEON_DEFAULT = "/Users/meilunzhou/Documents/GitHub/CTL/neon_data.npz"
PAPER_STRUCTURE = {
    "RGB": {"C": (-1.66, 0.88), "B": (-0.34, 0.87), "H": (0.48, 0.97)},
    "HSI": {"C": (0.72, 0.11), "B": (0.94, 0.56), "H": (1.49, 0.23)},
    "LiDAR": {"C": (2.60, 0.08), "B": (1.60, 0.72), "H": (-0.35, 0.17)},
    "Concat": {"C": (0.39, 0.15), "B": (0.19, 0.22), "H": (0.36, 0.05)},
    "DS-M": {"C": (-0.27, 0.12), "B": (0.24, 0.29), "H": (-0.27, 0.23)},
    "MMCTL": {"C": (-1.49, 0.23), "B": (-2.07, 0.29), "H": (-1.58, 0.25)},
}
PAPER_DOWN = {
    "RGB": {"C": (74.59, 1.90), "B": (0.07, 0.15), "H": (0.34, 0.02)},
    "HSI": {"C": (76.69, 1.39), "B": (0.12, 0.08), "H": (0.40, 0.05)},
    "LiDAR": {"C": (49.66, 1.15), "B": (0.22, 0.01), "H": (0.68, 0.02)},
    "Concat": {"C": (81.69, 1.08), "B": (0.39, 0.03), "H": (0.69, 0.02)},
    "DS-M": {"C": (81.35, 1.18), "B": (0.40, 0.01), "H": (0.61, 0.04)},
    "MMCTL": {"C": (86.95, 1.7), "B": (0.43, 0.04), "H": (0.74, 0.02)},
}


def center_crop(imgs, size):
    imgs = np.asarray(imgs)
    _, h, w, _ = imgs.shape
    t, l = (h - size) // 2, (w - size) // 2
    return imgs[:, t : t + size, l : l + size, :]


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


def zscore_struct(rows):
    out = []
    for run in sorted({r["run"] for r in rows}):
        sub = [dict(r) for r in rows if r["run"] == run]
        for src in ("cls", "box", "height"):
            a = np.array([r[f"align_{src}"] for r in sub], float)
            s = np.array([r[f"smooth_{src}"] for r in sub], float)
            za = (a - a.mean()) / (a.std() + 1e-12)
            zs = (s - s.mean()) / (s.std() + 1e-12)
            for i in range(len(sub)):
                sub[i][f"score_{src}"] = float(za[i] + zs[i])
        out.extend(sub)
    return out


def paper_task_heads(
    z_tr,
    z_va,
    z_te,
    y_cls_tr,
    y_cls_va,
    y_cls_te,
    y_box_tr,
    y_box_va,
    y_box_te,
    y_h_tr,
    y_h_va,
    y_h_te,
    head_cfg: HeadTrainConfig,
):
    """Paper downstream: frozen embeddings + shallow MLP heads (3000 / patience 1000)."""
    y_box_tr = np.nan_to_num(y_box_tr).astype(np.float32)
    y_box_va = np.nan_to_num(y_box_va).astype(np.float32)
    y_box_te = np.nan_to_num(y_box_te).astype(np.float32)
    y_h_tr = np.nan_to_num(y_h_tr).astype(np.float32)
    y_h_va = np.nan_to_num(y_h_va).astype(np.float32)
    y_h_te = np.nan_to_num(y_h_te).astype(np.float32)

    z_tr, z_va, z_te = standardize_z(
        l2(z_tr.astype(np.float32)),
        l2(z_va.astype(np.float32)),
        l2(z_te.astype(np.float32)),
    )
    stats = train_three_heads(
        cfg=head_cfg,
        z_tr=z_tr,
        z_va=z_va,
        z_te=z_te,
        y_cls_tr=y_cls_tr,
        y_cls_va=y_cls_va,
        y_cls_te=y_cls_te,
        y_box_tr=y_box_tr,
        y_box_va=y_box_va,
        y_box_te=y_box_te,
        y_pos_tr=y_h_tr,
        y_pos_va=y_h_va,
        y_pos_te=y_h_te,
    )
    return {
        "acc_pct": 100.0 * float(stats["acc"]),
        "r2_box": float(stats["r2_box_mean"]),
        "r2_height": float(stats["r2_pos_mean"]),
    }


def build_encoder_extractors(full_model, emb_dim=1024):
    """Return models that map each modality -> 1024-d and joint 3072-d."""
    # full outputs: [concat, rgb_recon, hsi_recon, lidar_recon]
    # inputs: rgb, hsi, lidar
    joint = Model(full_model.inputs, full_model.outputs[0], name="joint_emb")
    # Individual encoders are nested; grab by name if present
    try:
        rgb_enc = full_model.get_layer("rgb_encoder")
        hsi_enc = full_model.get_layer("hsi_encoder")
        lid_enc = full_model.get_layer("lidar_encoder")
        rgb_m = Model(full_model.inputs[0], rgb_enc(full_model.inputs[0]), name="rgb_only")
        hsi_m = Model(full_model.inputs[1], hsi_enc(full_model.inputs[1]), name="hsi_only")
        lid_m = Model(full_model.inputs[2], lid_enc(full_model.inputs[2]), name="lid_only")
    except Exception:
        # fallback: slice joint
        inp = full_model.inputs
        j = full_model.outputs[0]
        rgb_m = Model(inp, j[:, :emb_dim], name="rgb_slice")
        hsi_m = Model(inp, j[:, emb_dim : 2 * emb_dim], name="hsi_slice")
        lid_m = Model(inp, j[:, 2 * emb_dim : 3 * emb_dim], name="lid_slice")
    return joint, rgb_m, hsi_m, lid_m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz_path", default=NEON_DEFAULT)
    p.add_argument("--save_root", default=os.path.join(ROOT, "outputs", "neon_paper_recon"))
    p.add_argument("--emb_dir", default=os.path.join(ROOT, "outputs", "neon_paper_embeddings"))
    p.add_argument("--embedding_dimension", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=1000, help="Recon/MMCTL epochs (paper: 1000)")
    p.add_argument("--iterations", type=int, default=1, help="Outer recon loops (1 × 1000 epochs)")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--rgb_size", type=int, default=60)
    p.add_argument("--hsi_size", type=int, default=6)
    p.add_argument("--cls_weight", type=float, default=1.0)
    p.add_argument("--box_weight", type=float, default=1.0)
    p.add_argument("--height_weight", type=float, default=1.0)
    p.add_argument("--head_epochs", type=int, default=3000)
    p.add_argument("--head_patience", type=int, default=1000)
    p.add_argument("--head_hidden", type=int, default=256)
    p.add_argument("--head_dropout", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--smoke", action="store_true", help="Fast debug: 2 runs, 5 recon epochs, short heads")
    p.add_argument("--eval_only", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.runs, args.iterations, args.epochs = 2, 1, 5
        args.head_epochs, args.head_patience = 50, 15

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    head_cfg = HeadTrainConfig(
        epochs=args.head_epochs,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        hidden=args.head_hidden,
        dropout=args.head_dropout,
        early_patience=args.head_patience,
        grad_clip=args.grad_clip,
        device=device,
    )

    os.makedirs(args.save_root, exist_ok=True)
    os.makedirs(args.emb_dir, exist_ok=True)
    print("npz:", args.npz_path, flush=True)
    print(
        f"recon: runs={args.runs} iter={args.iterations} epochs={args.epochs} "
        f"weights cls/box/hgt={args.cls_weight}/{args.box_weight}/{args.height_weight}",
        flush=True,
    )
    print(
        f"task heads: epochs={head_cfg.epochs} patience={head_cfg.early_patience} "
        f"bs={head_cfg.batch_size} lr={head_cfg.lr} wd={head_cfg.weight_decay} "
        f"clip={head_cfg.grad_clip} hidden={head_cfg.hidden} device={head_cfg.device}",
        flush=True,
    )

    f = np.load(args.npz_path, allow_pickle=True)
    rgb = center_crop(f["rgb"], args.rgb_size)
    hsi = center_crop(f["hsi"], args.hsi_size)
    lidar = center_crop(f["lidar"], args.hsi_size)
    y_cls = f["class_labels"]
    y_pos = f["box_labels"].astype(np.float32)
    y_box = f["box_info"].astype(np.float32)
    y_height = f["height_info"].astype(np.float32)

    rgb, hsi, lidar, y_cls, y_pos, y_box, y_height, class_map = filter_and_remap_top_classes(
        rgb, hsi, lidar, y_cls, y_pos, y_box, y_height, top_X=args.classes
    )
    print("class_map", class_map, "n=", len(y_cls), flush=True)
    hsi = remove_bands(hsi).astype(np.float32)
    print("shapes", rgb.shape, hsi.shape, lidar.shape, flush=True)

    # optional box features from generate_triplet for structure metrics
    y_str = f["string_label"] if "string_label" in f.files else y_cls
    # align string labels to filtered set is hard; use box_info + height for continuous
    _, box_feat_all = generate_triplet_box_label_normalized(
        y_pos, np.asarray(y_cls).astype(str), n_clusters=3
    )
    box_feat_all = box_feat_all.astype(np.float32)

    (
        X_rgb_tv,
        X_rgb_te,
        X_lid_tv,
        X_lid_te,
        X_hsi_tv,
        X_hsi_te,
        y_cls_tv,
        y_cls_te,
        box_tv,
        box_te,
        h_tv,
        h_te,
        bfeat_tv,
        bfeat_te,
    ) = train_test_split(
        rgb,
        lidar,
        hsi,
        y_cls,
        y_box,
        y_height,
        box_feat_all,
        test_size=0.5,
        stratify=y_cls,
        random_state=42,
    )

    X_rgb_tv = global_max_normalize(X_rgb_tv.astype(np.float32))
    X_lid_tv = global_max_normalize(X_lid_tv.astype(np.float32))
    X_hsi_tv = global_max_normalize(X_hsi_tv.astype(np.float32))
    X_rgb_te = global_max_normalize(X_rgb_te.astype(np.float32))
    X_lid_te = global_max_normalize(X_lid_te.astype(np.float32))
    X_hsi_te = global_max_normalize(X_hsi_te.astype(np.float32))

    skf = StratifiedKFold(n_splits=args.runs, shuffle=True, random_state=42)
    folds = list(skf.split(X_rgb_tv, y_cls_tv))

    emb_dim = args.embedding_dimension

    struct_rows, down_rows = [], []

    for run, (tr_idx, va_idx) in enumerate(folds, start=1):
        ckpt = os.path.join(args.save_root, f"w_cls1p00_best_val_model_run_{run}.weights.h5")
        print(f"\n=== fold {run}/{args.runs} ===", flush=True)

        # Always rebuild architecture for this fold's shapes
        sample_rgb = X_rgb_tv[0].shape
        sample_hsi = X_hsi_tv[0].shape
        sample_lid = X_lid_tv[0].shape

        if not args.eval_only:
            X_rgb_tr, X_rgb_va = X_rgb_tv[tr_idx], X_rgb_tv[va_idx]
            X_hsi_tr, X_hsi_va = X_hsi_tv[tr_idx], X_hsi_tv[va_idx]
            X_lid_tr, X_lid_va = X_lid_tv[tr_idx], X_lid_tv[va_idx]
            y_tr, y_va = y_cls_tv[tr_idx], y_cls_tv[va_idx]
            box_tr, box_va = box_tv[tr_idx], box_tv[va_idx]
            h_tr, h_va = h_tv[tr_idx], h_tv[va_idx]

            y_tr, X_rgb_tr, X_lid_tr, X_hsi_tr, box_tr, h_tr = resampling(
                y_tr, X_rgb_tr, X_lid_tr, X_hsi_tr, box_tr, h_tr
            )

            y_tr_oh = tf.keras.utils.to_categorical(y_tr, num_classes=args.classes)
            y_va_oh = tf.keras.utils.to_categorical(y_va, num_classes=args.classes)
            wdict = {
                "class_labels": args.cls_weight,
                "box_features": args.box_weight,
                "height_features": args.height_weight,
            }
            comb_tr = weigh_features_multi(
                {"class_labels": y_tr_oh, "box_features": box_tr, "height_features": h_tr},
                wdict,
            )
            comb_va = weigh_features_multi(
                {"class_labels": y_va_oh, "box_features": box_va, "height_features": h_va},
                wdict,
            )

            model = image_reconstruction_multi_modalv1(
                X_rgb_tr[0].shape, X_hsi_tr[0].shape, X_lid_tr[0].shape, emb_dim
            )
            model.compile(
                optimizer=Adam(1e-4),
                loss=[
                    keras_batch_all_triplet_continuous_loss_multimodal(EMB_DIM=emb_dim, N_MOD=3),
                    rgb_ssim_loss(0.1),
                    hybrid_hsi_loss,
                    lidar_ssim_loss(0.1),
                ],
                loss_weights=[1.0, 1.0, 0.001, 1.0],
            )
            cb = ModelCheckpoint(
                ckpt,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                mode="min",
                verbose=0,
            )
            x_tr = [X_rgb_tr, X_hsi_tr, X_lid_tr]
            y_targets = [comb_tr, X_rgb_tr, X_hsi_tr, X_lid_tr]
            x_va = [X_rgb_va, X_hsi_va, X_lid_va]
            y_va_t = [comb_va, X_rgb_va, X_hsi_va, X_lid_va]

            for it in range(args.iterations):
                print(f"  iteration {it+1}/{args.iterations}", flush=True)
                model.fit(
                    x_tr,
                    y_targets,
                    validation_data=(x_va, y_va_t),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    verbose=2 if args.smoke else 1,
                    callbacks=[cb],
                )
            model.save_weights(os.path.join(args.save_root, f"w_cls1p00_run_{run}_last.weights.h5"))
            K.clear_session()
            del model
            gc.collect()

        weights_path = ckpt if os.path.exists(ckpt) else None
        if weights_path is None or not os.path.exists(weights_path):
            print(f"Missing weights checkpoint {ckpt} (legacy full-model .h5 not supported)", flush=True)
            continue

        print(f"Loading weights from {weights_path}", flush=True)
        model = image_reconstruction_multi_modalv1(sample_rgb, sample_hsi, sample_lid, emb_dim)
        model.load_weights(weights_path)
        joint_m, rgb_m, hsi_m, lid_m = build_encoder_extractors(model, emb_dim)

        def pred_joint(rgb_x, hsi_x, lid_x):
            outs = []
            for i in range(0, len(rgb_x), 64):
                outs.append(
                    joint_m.predict(
                        [rgb_x[i : i + 64], hsi_x[i : i + 64], lid_x[i : i + 64]],
                        verbose=0,
                    )
                )
            return np.concatenate(outs, 0)

        def pred_mod(m, rgb_x, hsi_x, lid_x):
            outs = []
            n_in = len(m.inputs)
            for i in range(0, len(rgb_x), 64):
                if n_in == 1:
                    # single-modality encoder; pick matching slice by name
                    if "rgb" in m.name:
                        x = rgb_x[i : i + 64]
                    elif "hsi" in m.name:
                        x = hsi_x[i : i + 64]
                    else:
                        x = lid_x[i : i + 64]
                    outs.append(m.predict(x, verbose=0))
                else:
                    outs.append(
                        m.predict(
                            [rgb_x[i : i + 64], hsi_x[i : i + 64], lid_x[i : i + 64]],
                            verbose=0,
                        )
                    )
            return np.concatenate(outs, 0)

        def embed_split(rgb_x, hsi_x, lid_x):
            z_rgb = pred_mod(rgb_m, rgb_x, hsi_x, lid_x)
            z_hsi = pred_mod(hsi_m, rgb_x, hsi_x, lid_x)
            z_lid = pred_mod(lid_m, rgb_x, hsi_x, lid_x)
            z_mm = pred_joint(rgb_x, hsi_x, lid_x)
            return {
                "RGB": z_rgb,
                "HSI": z_hsi,
                "LiDAR": z_lid,
                "Concat": np.concatenate([z_rgb, z_hsi, z_lid], 1),
                "DS-M": (z_rgb + z_hsi + z_lid) / 3.0,
                "MMCTL": z_mm,
            }

        emb_tr = embed_split(X_rgb_tv[tr_idx], X_hsi_tv[tr_idx], X_lid_tv[tr_idx])
        emb_va = embed_split(X_rgb_tv[va_idx], X_hsi_tv[va_idx], X_lid_tv[va_idx])
        emb_te = embed_split(X_rgb_te, X_hsi_te, X_lid_te)

        for name in emb_te:
            np.save(os.path.join(args.emb_dir, f"run{run:02d}_{name.lower()}_test.npy"), emb_te[name])
            np.save(os.path.join(args.emb_dir, f"run{run:02d}_{name.lower()}_train.npy"), emb_tr[name])
            np.save(os.path.join(args.emb_dir, f"run{run:02d}_{name.lower()}_val.npy"), emb_va[name])

        y_cls_tr, y_cls_va = y_cls_tv[tr_idx], y_cls_tv[va_idx]
        box_tr, box_va = bfeat_tv[tr_idx], bfeat_tv[va_idx]
        h_tr, h_va = h_tv[tr_idx], h_tv[va_idx]

        for name in ("RGB", "HSI", "LiDAR", "Concat", "DS-M", "MMCTL"):
            print(f"  structure+heads {name} ...", flush=True)
            m = structure_raw(emb_te[name], y_cls_te, bfeat_te, h_te, seed=40 + run)
            struct_rows.append({"run": run, "method": name, **m})
            pr = paper_task_heads(
                emb_tr[name],
                emb_va[name],
                emb_te[name],
                y_cls_tr,
                y_cls_va,
                y_cls_te,
                box_tr,
                box_va,
                bfeat_te,
                h_tr,
                h_va,
                h_te,
                head_cfg,
            )
            down_rows.append({"run": run, "method": name, **pr})
            print(
                f"  {name:<7} acc={pr['acc_pct']:.1f} B={pr['r2_box']:.3f} H={pr['r2_height']:.3f}",
                flush=True,
            )

        K.clear_session()
        del model
        gc.collect()

    if not struct_rows:
        print("No results to aggregate.", flush=True)
        return

    scored = zscore_struct(struct_rows)
    order = ["RGB", "HSI", "LiDAR", "Concat", "DS-M", "MMCTL"]

    def aggs(method, src):
        v = [r[f"score_{src}"] for r in scored if r["method"] == method]
        return float(np.mean(v)), float(np.std(v))

    def aggd(method, key):
        v = [r[key] for r in down_rows if r["method"] == method]
        return float(np.mean(v)), float(np.std(v))

    structure = {m: {"C": aggs(m, "cls"), "B": aggs(m, "box"), "H": aggs(m, "height")} for m in order}
    downstream = {
        m: {"C": aggd(m, "acc_pct"), "B": aggd(m, "r2_box"), "H": aggd(m, "r2_height")} for m in order
    }

    out_json = os.path.join(ROOT, "results", "replication_neon.json")
    out_md = os.path.join(ROOT, "results", "replication_neon_delta.md")
    payload = {
        "protocol": (
            "Paper 3-mod recon (RGB+HSI+LiDAR) + CTL; "
            f"recon epochs={args.epochs}×{args.iterations}, runs={args.runs}; "
            f"task heads Adam lr={head_cfg.lr} wd={head_cfg.weight_decay} "
            f"epochs={head_cfg.epochs} patience={head_cfg.early_patience} "
            f"bs={head_cfg.batch_size} clip={head_cfg.grad_clip} hidden={head_cfg.hidden}; "
            f"data={args.npz_path}; emb={emb_dim}; weights="
            f"{args.cls_weight}/{args.box_weight}/{args.height_weight}"
        ),
        "data_path": args.npz_path,
        "structure": structure,
        "downstream": downstream,
        "paper_structure": PAPER_STRUCTURE,
        "paper_downstream": PAPER_DOWN,
    }
    with open(out_json, "w") as fp:
        json.dump(payload, fp, indent=2)

    lines = [
        "# NEON Replication Delta (paper reconstruction code)",
        "",
        f"Data: `{args.npz_path}`",
        f"Protocol: {payload['protocol']}",
        "",
        "## Structure (lower better)",
        "",
        "| Method | Src | Paper | Code | Δmean |",
        "|---|---|---|---|---|",
    ]
    for m in order:
        for s in ("C", "B", "H"):
            pm, ps = PAPER_STRUCTURE[m][s]
            cm, cs = structure[m][s]
            lines.append(f"| {m} | {s} | {pm:.2f}±{ps:.2f} | {cm:.2f}±{cs:.2f} | {cm-pm:+.2f} |")
    lines += [
        "",
        "## Downstream",
        "",
        "| Method | Paper C% | Code C% | Paper B | Code B | Paper H | Code H |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in order:
        pc, _ = PAPER_DOWN[m]["C"]
        pb, _ = PAPER_DOWN[m]["B"]
        ph, _ = PAPER_DOWN[m]["H"]
        cc, ccs = downstream[m]["C"]
        cb, cbs = downstream[m]["B"]
        ch, chs = downstream[m]["H"]
        lines.append(
            f"| {m} | {pc:.2f} | {cc:.2f}±{ccs:.2f} | {pb:.2f} | {cb:.2f}±{cbs:.2f} | {ph:.2f} | {ch:.2f}±{chs:.2f} |"
        )
    with open(out_md, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print("Wrote", out_json, flush=True)
    print("Wrote", out_md, flush=True)


if __name__ == "__main__":
    main()
