#!/usr/bin/env python3
# NEON multimodal CTL image reconstruction training
# Sweeps ONE annotation weight at a time:
#   - class_labels OR box_features OR height_features
# while keeping the other weights fixed at 1.0.
# Saves models with filenames that specify which weight is modified.

# Import libraries
from comet_ml import Experiment
import os, sys, time, random, gc, argparse, datetime
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt  # (kept since your original imports include it)
from collections import defaultdict, Counter

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, silhouette_score
from sklearn.utils import resample

import umap
import pickle

from TripletNetwork_Online import (
    image_reconstruction_multi_modal,
    image_reconstruction_multi_modalv1,
    image_reconstruction_multi_modalv2,
    image_reconstruction_multi_modalv1_4x4,
    image_reconstruction_multi_modalv2_4x4
)

from neon_utilities import (
    plot_pca_umap, calculate_iou, calculate_average_iou,
    generate_triplet_box_label_normalized, global_max_normalize,
    calculate_mask_iou, adjust_and_calculate_iou_mask, print_and_log_average_iou,
    generate_triplet_box_label_one_hot, resampling, log_one_image_per_class,
    log_gt_image_per_class, assign_tile_labels, weigh_features_multi, remove_bands,
    center_crop, log_one_hsi_recon_per_class, log_gt_hsi_per_class,
    filter_and_remap_top_classes, center_crop_on_box, filter_npz_in_memory
)

from neon_custom_losses import (
    keras_batch_all_triplet_continuous_loss_multimodal,
    ciou_loss, dice_loss, ssim_loss, mean_squared_loss,
    hybrid_hsi_loss, rgb_ssim_loss, lidar_ssim_loss
)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class CometCallback(keras.callbacks.Callback):
    def __init__(self, experiment, total_epochs):
        super().__init__()
        self.experiment = experiment
        self.global_epoch = 0
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if "loss" in logs:
            self.experiment.log_metric("train_loss", logs["loss"], step=self.global_epoch)
        if "val_loss" in logs:
            self.experiment.log_metric("val_loss", logs["val_loss"], step=self.global_epoch)
        self.global_epoch += 1


def fmt_weight_tag(sweep_key: str, sweep_w: float) -> str:
    key_short = {"class_labels": "cls", "box_features": "box", "height_features": "hgt"}[sweep_key]
    # w_cls1p50
    return f"w_{key_short}{sweep_w:.2f}".replace(".", "p")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--embedding_dimension", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--modality", type=str, default="rgb")
    parser.add_argument("--test_size", type=float, default=0.5)

    # Data / run controls
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--patch_size", type=int, default=6)

    # Comet
    parser.add_argument("--comet_project", type=str, default="neon")
    parser.add_argument("--comet_workspace", type=str, default="zhou-m")
    parser.add_argument("--comet_api_key", type=str, default="xuGNxq43n5AvOfi7zn0JavYDR")

    # Paths
    parser.add_argument(
        "--npz_path",
        type=str,
        default="/blue/azare/zhou.m/commanet/ground_truth_points_HARVARD/NEON_big_dataset_updatedv1.npz"
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,  # if None, will use cwd + your original pattern
    )

    # Sweep config
    parser.add_argument(
        "--weight_values",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
    )
    parser.add_argument(
        "--weight_keys",
        type=str,
        nargs="+",
        default=["class_labels", "box_features", "height_features"],
        choices=["class_labels", "box_features", "height_features"],
    )

    args = parser.parse_args()

    parameters = {
        "epochs": args.epochs,
        "iterations": args.iterations,
        "runs": args.runs,
        "classes": args.classes,
        "patch_size": args.patch_size,
        "embedding_dimension": args.embedding_dimension,
        "margin": args.margin,
        "batch_size": args.batch_size,
        "test_size": args.test_size,
        "modality": args.modality,
        "weight_values": args.weight_values,
        "weight_keys": args.weight_keys,
    }

    print("Training configuration:")
    for k, v in parameters.items():
        print(f"  {k}: {v}")

    # Load the merged data
    f = np.load(args.npz_path)

    modality_chosen = "multimodal"
    print("Modality:", modality_chosen, flush=True)

    # Select indices using RGB crop on box (your original logic)
    test_imgs, test_indices = center_crop_on_box(
        f["rgb"], f["box_labels"] * 10, parameters["patch_size"] * 10
    )
    print("test_img.shape:", test_imgs.shape, "len(test_indices):", len(test_indices))

    filtered_arrays = filter_npz_in_memory(args.npz_path, test_indices)

    # Extract arrays
    y_pos = filtered_arrays["box_labels"]
    y_cls = filtered_arrays["class_labels"]
    y_box = filtered_arrays["box_info"]
    y_height = filtered_arrays["height_info"]

    rgb, _ = center_crop_on_box(
        filtered_arrays["rgb"], filtered_arrays["box_labels"] * 10, parameters["patch_size"] * 10
    )
    hsi, _ = center_crop_on_box(
        filtered_arrays["hsi"], filtered_arrays["box_labels"], parameters["patch_size"]
    )
    lidar, _ = center_crop_on_box(
        filtered_arrays["lidar"], filtered_arrays["box_labels"], parameters["patch_size"]
    )

    print("rgb.shape post cropping:", rgb.shape, "hsi.shape post cropping:", hsi.shape)

    rgb, hsi, lidar, y_cls, y_pos, y_box, y_height, class_map = filter_and_remap_top_classes(
        rgb, hsi, lidar, y_cls, y_pos, y_box, y_height, top_X=parameters["classes"]
    )
    print("New class mapping:", class_map)

    hsi = remove_bands(hsi)
    print("hsi shape after removing bands:", hsi.shape)

    # Train/test split
    (
        X_rgb_trainval_prenorm, X_rgb_test,
        X_lidar_trainval_prenorm, X_lidar_test,
        X_hsi_trainval_prenorm, X_hsi_test,
        y_cls_trainval, y_cls_test,
        pos_feat_trainval, pos_feat_test,
        box_feat_trainval, box_feat_test,
        height_feat_trainval, height_feat_test
    ) = train_test_split(
        rgb, lidar, hsi, y_cls, y_pos, y_box, y_height,
        test_size=parameters["test_size"],
        stratify=y_cls,
        random_state=42
    )

    print("X_rgb_trainval_prenorm min max:", X_rgb_trainval_prenorm.min(), X_rgb_trainval_prenorm.max())
    print("X_hsi_trainval_prenorm min max:", X_hsi_trainval_prenorm.min(), X_hsi_trainval_prenorm.max())
    print("X_lidar_trainval_prenorm min max:", X_lidar_trainval_prenorm.min(), X_lidar_trainval_prenorm.max())

    X_rgb_trainval = global_max_normalize(X_rgb_trainval_prenorm)
    X_lidar_trainval = global_max_normalize(X_lidar_trainval_prenorm)
    X_hsi_trainval = global_max_normalize(X_hsi_trainval_prenorm)

    print("X_rgb_trainval min max after norm:", X_rgb_trainval.min(), X_rgb_trainval.max())
    print("X_hsi_trainval min max after norm:", X_hsi_trainval.min(), X_hsi_trainval.max())
    print("X_lidar_trainval min max after norm:", X_lidar_trainval.min(), X_lidar_trainval.max())

    # CV splits
    skf = StratifiedKFold(n_splits=parameters["runs"], shuffle=True, random_state=42)

    train_splits = []
    val_splits = []
    for train_index, val_index in skf.split(X_rgb_trainval, y_cls_trainval):
        X_rgb_train, X_rgb_val = X_rgb_trainval[train_index], X_rgb_trainval[val_index]
        X_lidar_train, X_lidar_val = X_lidar_trainval[train_index], X_lidar_trainval[val_index]
        X_hsi_train, X_hsi_val = X_hsi_trainval[train_index], X_hsi_trainval[val_index]

        y_cls_train, y_cls_val = y_cls_trainval[train_index], y_cls_trainval[val_index]
        pos_feat_train, pos_feat_val = pos_feat_trainval[train_index], pos_feat_trainval[val_index]
        box_feat_train, box_feat_val = box_feat_trainval[train_index], box_feat_trainval[val_index]
        height_feat_train, height_feat_val = height_feat_trainval[train_index], height_feat_trainval[val_index]

        train_splits.append(
            (X_rgb_train, X_lidar_train, X_hsi_train, y_cls_train, pos_feat_train, box_feat_train, height_feat_train)
        )
        val_splits.append(
            (X_rgb_val, X_lidar_val, X_hsi_val, y_cls_val, pos_feat_val, box_feat_val, height_feat_val)
        )

    # Save root
    if args.save_root is None:
        save_root = os.getcwd() + "/trained_models/image_reconstruction/continuous_triplet/noL2/"
        save_root += f"{modality_chosen}_hsi_rgb/v1_box_cls_height/class_{parameters['classes']}/patch_size_{parameters['patch_size']}/"
        save_root += f"_emb{parameters['embedding_dimension']}_batch{parameters['batch_size']}_margin{parameters['margin']}_test_size{parameters['test_size']}"
    else:
        save_root = args.save_root

    ensure_dir(save_root)
    print("Saving under:", save_root)

    # Losses
    chosen_loss_rgb = rgb_ssim_loss(alpha=0.1)
    chosen_loss_hsi = hybrid_hsi_loss

    # Sweep over one weight at a time
    for sweep_key in parameters["weight_keys"]:
        for sweep_w in parameters["weight_values"]:
            w_tag = fmt_weight_tag(sweep_key, sweep_w)
            sweep_dir = os.path.join(save_root, w_tag)
            ensure_dir(sweep_dir)

            sweep_parameters = dict(parameters)
            sweep_parameters["sweep_key"] = sweep_key
            sweep_parameters["sweep_weight"] = float(sweep_w)

            print(f"\n=== SWEEP: {sweep_key} = {sweep_w} ({w_tag}) ===")

            # Train across folds/runs
            for run in range(parameters["runs"]):
                run_number = run + 1

                experiment = Experiment(
                    api_key=args.comet_api_key,
                    project_name=args.comet_project,
                    workspace=args.comet_workspace
                )
                experiment_name = f"{modality_chosen}_CTL_image_recon_{w_tag}_run{run_number}"
                experiment.set_name(experiment_name)
                experiment.log_parameters(sweep_parameters)

                checkpoint_save_path = os.path.join(
                    sweep_dir,
                    f"{w_tag}_best_val_model_run_{run_number}.h5"
                )

                X_rgb_train, X_lidar_train, X_hsi_train, y_cls_train, pos_feat_train, box_feat_train, height_feat_train = train_splits[run]
                X_rgb_val, X_lidar_val, X_hsi_val, y_cls_val, pos_feat_val, box_feat_val, height_feat_val = val_splits[run]

                print(f"Fold {run_number}:", flush=True)

                print("y_cls_train unique before resampling:", np.unique(y_cls_train, return_counts=True))
                y_cls_train, X_rgb_train, X_lidar_train, X_hsi_train, pos_feat_train, box_feat_train, height_feat_train = resampling(
                    y_cls_train, X_rgb_train, X_lidar_train, X_hsi_train, pos_feat_train, box_feat_train, height_feat_train
                )
                print("y_cls_train unique after resampling:", np.unique(y_cls_train, return_counts=True))

                # One-hot for class labels
                y_train_categorical = tf.keras.utils.to_categorical(y_cls_train)
                y_val_categorical = tf.keras.utils.to_categorical(y_cls_val)

                # Cast
                X_rgb_train = X_rgb_train.astype(np.float32)
                X_hsi_train = X_hsi_train.astype(np.float32)
                X_rgb_val = X_rgb_val.astype(np.float32)
                X_hsi_val = X_hsi_val.astype(np.float32)

                # Build joint annotation vectors
                train_features_dict = {
                    "class_labels": y_train_categorical,
                    "box_features": box_feat_train,
                    "height_features": height_feat_train,
                }
                val_features_dict = {
                    "class_labels": y_val_categorical,
                    "box_features": box_feat_val,
                    "height_features": height_feat_val,
                }

                # Sweep one weight, keep others at 1.0
                weights_dict = {
                    "class_labels": 1.0,
                    "box_features": 1.0,
                    "height_features": 1.0,
                }
                weights_dict[sweep_key] = float(sweep_w)

                combined_train = weigh_features_multi(train_features_dict, weights_dict)
                combined_val = weigh_features_multi(val_features_dict, weights_dict)

                print("weights_dict:", weights_dict)
                print("combined_train.shape:", combined_train.shape, "combined_val.shape:", combined_val.shape)

                # Model dims
                RGB_DIM = X_rgb_train[0].shape
                HSI_DIM = X_hsi_train[0].shape
                EMB_DIM = parameters["embedding_dimension"]

                # Build model (your current choice: v1 with RGB+HSI)
                model = image_reconstruction_multi_modalv1(RGB_DIM, HSI_DIM, EMB_DIM)

                model.compile(
                    optimizer=Adam(0.0001),
                    loss=[
                        keras_batch_all_triplet_continuous_loss_multimodal(EMB_DIM=EMB_DIM),
                        chosen_loss_rgb,
                        chosen_loss_hsi,
                    ],
                    loss_weights=[1, 1, 0.001],
                )

                model_checkpoint_callback = ModelCheckpoint(
                    filepath=checkpoint_save_path,
                    monitor="val_loss",
                    save_best_only=True,
                    mode="min",
                    verbose=0,
                )
                comet_callback = CometCallback(
                    experiment,
                    total_epochs=parameters["iterations"] * parameters["epochs"]
                )

                # Targets / inputs for your current model signature
                target_train = [combined_train, X_rgb_train, X_hsi_train]
                target_val = [combined_val, X_rgb_val, X_hsi_val]
                input_data_train = [X_rgb_train, X_hsi_train]
                input_data_val = [X_rgb_val, X_hsi_val]

                # Train
                for i in range(parameters["iterations"]):
                    model.fit(
                        input_data_train,
                        target_train,
                        validation_data=(input_data_val, target_val),
                        epochs=parameters["epochs"],
                        batch_size=parameters["batch_size"],
                        verbose=0,
                        callbacks=[model_checkpoint_callback, comet_callback],
                    )

                    # Save every 5th iteration
                    if (i + 1) % 5 == 0:
                        save_path = os.path.join(
                            sweep_dir,
                            f"{w_tag}_run_{run_number}_iteration_{i + 1}_model.h5"
                        )
                        model.save(save_path)
                        print(f"Model saved: {save_path}")

                experiment.end()

                # Clean up between folds to reduce memory pressure
                try:
                    K.clear_session()
                except Exception:
                    pass
                del model
                gc.collect()

    print("\nAll sweeps complete.")


if __name__ == "__main__":
    main()