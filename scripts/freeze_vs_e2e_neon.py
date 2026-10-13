#!/usr/bin/env python3
"""
Freeze vs. end-to-end fine-tuning on NEON (Reviewer R1.5).

The paper trains MMCTL to organize a shared latent space, then *freezes* the
encoders and trains shallow task heads on top. Reviewer 1 asked whether that
frozen protocol is justified relative to fine-tuning the encoders end-to-end.

This script runs a controlled A/B using the paper's own reconstruction
checkpoints (outputs/neon_paper_recon/w_cls1p00_best_val_model_run_{run}.weights.h5):

  * FROZEN : encoders frozen (paper protocol); only the task heads train.
  * E2E    : same pretrained init, but encoders are unfrozen and fine-tuned
             jointly with the heads (lower LR).

Both arms share the identical head architecture, initialization, data splits,
standardized regression targets and training budget, so any difference is
attributable to freezing vs. fine-tuning. We report test classification
accuracy, box R^2 and canopy-height R^2, averaged over the 5 paper folds.

Data prep and architecture exactly mirror scripts/run_neon_paper_reconstruction.py.

Outputs:
  results/freeze_vs_e2e_neon.json
  results/freeze_vs_e2e_neon.md

Run (full):
  .venv_exp/bin/python -u scripts/freeze_vs_e2e_neon.py
Run (smoke):
  .venv_exp/bin/python -u scripts/freeze_vs_e2e_neon.py --smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TripletNetwork_Online import image_reconstruction_multi_modalv1  # noqa: E402
from neon_utilities import (  # noqa: E402
    filter_and_remap_top_classes,
    global_max_normalize,
    remove_bands,
    resampling,
)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

MAIN = "."
NEON_DEFAULT = os.path.join(MAIN, "neon_data.npz")
CKPT_DIR = os.path.join(MAIN, "outputs", "neon_paper_recon")


def center_crop(imgs, size):
    imgs = np.asarray(imgs)
    _, h, w, _ = imgs.shape
    t, l = (h - size) // 2, (w - size) // 2
    return imgs[:, t:t + size, l:l + size, :]


def standardize(train, *others, eps=1e-6):
    mu = train.mean(0, keepdims=True)
    sd = np.maximum(train.std(0, keepdims=True), eps)
    return [((a - mu) / sd).astype(np.float32) for a in (train, *others)]


def r2_multi(y_true, y_pred):
    scores = []
    for d in range(y_true.shape[1]):
        if np.std(y_true[:, d]) >= 1e-8:
            scores.append(r2_score(y_true[:, d], y_pred[:, d]))
    return float(np.mean(scores)) if scores else 0.0


def build_task_model(ckpt_path, rgb_shape, hsi_shape, lid_shape, emb_dim,
                     n_classes, d_box, d_h, trainable, hidden=256, dropout=0.1,
                     seed=42):
    """Rebuild AE, load pretrained weights, attach multitask heads.

    trainable=False -> frozen encoders (paper protocol).
    trainable=True  -> end-to-end fine-tuning.
    """
    tf.random.set_seed(seed)
    ae = image_reconstruction_multi_modalv1(rgb_shape, hsi_shape, lid_shape, emb_dim)
    ae.load_weights(ckpt_path)

    # Joint embedding extractor (outputs[0] = concatenated 3*emb_dim embedding).
    joint = Model(ae.inputs, ae.outputs[0], name="joint_emb")
    joint.trainable = trainable  # freeze/unfreeze all encoder weights

    rgb_in = Input(rgb_shape, name="rgb_in")
    hsi_in = Input(hsi_shape, name="hsi_in")
    lid_in = Input(lid_shape, name="lid_in")
    z = joint([rgb_in, hsi_in, lid_in])
    # BatchNorm on the embedding replaces the fixed z-score used by the frozen
    # torch heads, and stays valid when the encoder shifts under E2E.
    z = BatchNormalization(name="emb_bn")(z)

    init = tf.keras.initializers.GlorotNormal(seed=seed)

    def head(name, d_out, activation):
        h = Dense(hidden, kernel_initializer=init, name=f"{name}_fc1")(z)
        h = ReLU()(h)
        h = Dropout(dropout)(h)
        return Dense(d_out, activation=activation, kernel_initializer=init,
                     name=name)(h)

    out_cls = head("cls", n_classes, "softmax")
    out_box = head("box", d_box, None)
    out_hgt = head("hgt", d_h, None)

    model = Model([rgb_in, hsi_in, lid_in], [out_cls, out_box, out_hgt],
                  name=f"tasknet_{'e2e' if trainable else 'frozen'}")
    return model


def run_arm(trainable, ckpt, shapes, splits, targets, args, seed):
    (rgb_tr, hsi_tr, lid_tr), (rgb_va, hsi_va, lid_va), (rgb_te, hsi_te, lid_te) = splits
    (yc_tr_oh, yc_va_oh, yc_te, n_classes,
     box_tr, box_va, box_te, h_tr, h_va, h_te) = targets

    lr = args.e2e_lr if trainable else args.freeze_lr
    model = build_task_model(
        ckpt, shapes[0], shapes[1], shapes[2], args.embedding_dimension,
        n_classes, box_tr.shape[1], h_tr.shape[1], trainable=trainable,
        hidden=args.hidden, dropout=args.dropout, seed=seed,
    )
    model.compile(
        optimizer=Adam(lr),
        loss=["categorical_crossentropy", "mse", "mse"],
        loss_weights=[1.0, 1.0, 1.0],
    )
    es = EarlyStopping(monitor="val_loss", patience=args.patience,
                       restore_best_weights=True, mode="min")
    model.fit(
        [rgb_tr, hsi_tr, lid_tr], [yc_tr_oh, box_tr, h_tr],
        validation_data=([rgb_va, hsi_va, lid_va], [yc_va_oh, box_va, h_va]),
        epochs=args.epochs, batch_size=args.batch_size,
        verbose=2 if args.smoke else 0, callbacks=[es],
    )
    p_cls, p_box, p_hgt = model.predict([rgb_te, hsi_te, lid_te], verbose=0)
    acc = accuracy_score(yc_te, np.argmax(p_cls, 1))
    res = {
        "acc_pct": 100.0 * float(acc),
        "r2_box": r2_multi(box_te, p_box),
        "r2_height": r2_multi(h_te, p_hgt),
    }
    K.clear_session()
    del model
    gc.collect()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz_path", default=NEON_DEFAULT)
    p.add_argument("--ckpt_dir", default=CKPT_DIR)
    p.add_argument("--embedding_dimension", type=int, default=1024)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--rgb_size", type=int, default=60)
    p.add_argument("--hsi_size", type=int, default=6)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--freeze_lr", type=float, default=1e-3)
    p.add_argument("--e2e_lr", type=float, default=1e-4)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.runs, args.epochs, args.patience = 2, 8, 4

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
    hsi = remove_bands(hsi).astype(np.float32)
    print("class_map", class_map, "n=", len(y_cls), "shapes", rgb.shape, hsi.shape, lidar.shape,
          flush=True)

    (X_rgb_tv, X_rgb_te, X_lid_tv, X_lid_te, X_hsi_tv, X_hsi_te,
     y_tv, y_te, box_tv, box_te, h_tv, h_te) = train_test_split(
        rgb, lidar, hsi, y_cls, y_box, y_height,
        test_size=0.5, stratify=y_cls, random_state=42,
    )
    X_rgb_tv = global_max_normalize(X_rgb_tv.astype(np.float32))
    X_lid_tv = global_max_normalize(X_lid_tv.astype(np.float32))
    X_hsi_tv = global_max_normalize(X_hsi_tv.astype(np.float32))
    X_rgb_te = global_max_normalize(X_rgb_te.astype(np.float32))
    X_lid_te = global_max_normalize(X_lid_te.astype(np.float32))
    X_hsi_te = global_max_normalize(X_hsi_te.astype(np.float32))

    shapes = (X_rgb_tv[0].shape, X_hsi_tv[0].shape, X_lid_tv[0].shape)
    skf = StratifiedKFold(n_splits=args.runs, shuffle=True, random_state=42)
    folds = list(skf.split(X_rgb_tv, y_tv))

    rows = {"frozen": [], "e2e": []}
    for run, (tr_idx, va_idx) in enumerate(folds, start=1):
        seed = 42 + run
        ckpt = os.path.join(args.ckpt_dir, f"w_cls1p00_best_val_model_run_{run}.weights.h5")
        if not os.path.exists(ckpt):
            print(f"[run {run}] missing checkpoint {ckpt}; skipping", flush=True)
            continue
        print(f"\n===== run {run}/{args.runs} =====\n  ckpt {ckpt}", flush=True)

        X_rgb_tr, X_rgb_va = X_rgb_tv[tr_idx], X_rgb_tv[va_idx]
        X_hsi_tr, X_hsi_va = X_hsi_tv[tr_idx], X_hsi_tv[va_idx]
        X_lid_tr, X_lid_va = X_lid_tv[tr_idx], X_lid_tv[va_idx]
        yc_tr, yc_va = y_tv[tr_idx], y_tv[va_idx]
        box_tr, box_va = box_tv[tr_idx], box_tv[va_idx]
        hh_tr, hh_va = h_tv[tr_idx], h_tv[va_idx]

        # Class-balance the training fold (paper protocol).
        yc_tr, X_rgb_tr, X_lid_tr, X_hsi_tr, box_tr, hh_tr = resampling(
            yc_tr, X_rgb_tr, X_lid_tr, X_hsi_tr, box_tr, hh_tr
        )

        # Standardize regression targets with train stats (R^2 invariant).
        box_tr, box_va2, box_te2 = standardize(box_tr, box_va, box_te)
        hh_tr, hh_va2, hh_te2 = standardize(hh_tr, hh_va, h_te)

        yc_tr_oh = tf.keras.utils.to_categorical(yc_tr, num_classes=args.classes)
        yc_va_oh = tf.keras.utils.to_categorical(yc_va, num_classes=args.classes)

        splits = (
            (X_rgb_tr, X_hsi_tr, X_lid_tr),
            (X_rgb_va, X_hsi_va, X_lid_va),
            (X_rgb_te, X_hsi_te, X_lid_te),
        )
        targets = (yc_tr_oh, yc_va_oh, y_te, args.classes,
                   box_tr, box_va2, box_te2, hh_tr, hh_va2, hh_te2)

        for arm, trainable in (("frozen", False), ("e2e", True)):
            res = run_arm(trainable, ckpt, shapes, splits, targets, args, seed)
            rows[arm].append(res)
            print(f"  [{arm:<6}] acc={res['acc_pct']:.1f} "
                  f"B={res['r2_box']:.3f} H={res['r2_height']:.3f}", flush=True)

    def agg(arm, k):
        v = np.array([r[k] for r in rows[arm]], float)
        return float(np.mean(v)), float(np.std(v))

    summary = {arm: {k: agg(arm, k) for k in ("acc_pct", "r2_box", "r2_height")}
               for arm in ("frozen", "e2e")}

    payload = {
        "protocol": (
            f"NEON 3-mod MMCTL recon checkpoints ({args.ckpt_dir}); freeze vs E2E; "
            f"runs={args.runs}, epochs={args.epochs}, patience={args.patience}, "
            f"freeze_lr={args.freeze_lr}, e2e_lr={args.e2e_lr}, bs={args.batch_size}, "
            f"hidden={args.hidden}; standardized box/height targets; "
            f"data={args.npz_path}"),
        "per_run": rows,
        "summary": summary,
    }
    out_json = os.path.join(ROOT, "results", "freeze_vs_e2e_neon.json")
    with open(out_json, "w") as fp:
        json.dump(payload, fp, indent=2)

    def line(arm):
        s = summary[arm]
        return (f"| {arm} | {s['acc_pct'][0]:.2f}±{s['acc_pct'][1]:.2f} | "
                f"{s['r2_box'][0]:.3f}±{s['r2_box'][1]:.3f} | "
                f"{s['r2_height'][0]:.3f}±{s['r2_height'][1]:.3f} |")

    md = [
        "# NEON Freeze vs. End-to-End (R1.5)",
        "",
        f"Protocol: {payload['protocol']}",
        "",
        "| Protocol | Acc % | Box R² | Height R² |",
        "|---|---|---|---|",
        line("frozen"),
        line("e2e"),
        "",
        "Both arms start from the same MMCTL-pretrained encoders and share the head "
        "architecture, splits, targets and budget; the only difference is whether the "
        "encoders are frozen or fine-tuned.",
        "",
    ]
    out_md = os.path.join(ROOT, "results", "freeze_vs_e2e_neon.md")
    with open(out_md, "w") as fp:
        fp.write("\n".join(md))
    print("\nWrote", out_json, "\n      ", out_md, flush=True)


if __name__ == "__main__":
    main()
