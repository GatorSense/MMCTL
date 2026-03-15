import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from tripletnetwork_online import image_reconstruction_multi_modalv1
from awir_utilities import (
    generate_triplet_box_label_normalized,
    global_max_normalize,
    resampling,
    assign_tile_labels,
    weigh_features_multi,
)
from awir_custom_losses import (
    keras_batch_all_triplet_continuous_loss_multimodal,
    ssim_loss,
)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data.npz")
    parser.add_argument("--output_dir", type=str, default="outputs/mmctl")
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--embedding_dimension", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--modality", type=str, default="rgb")
    parser.add_argument("--test_size", type=float, default=0.5)
    parser.add_argument("--cls_weight", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--runs", type=int, default=8)
    return parser.parse_args()


def build_box_mask(y_box, image_height=300, image_width=300):
    y_mask = np.zeros((y_box.shape[0], image_height, image_width, 1), dtype=np.float32)
    for i, (xmin, xmax, ymin, ymax) in enumerate(y_box):
        y_mask[i, ymin:ymax, xmin:xmax, 0] = 1.0
    return y_mask


def convert_box_to_center_width_height(y_box, image_width=300, image_height=300):
    num_samples = y_box.shape[0]
    y_box_cenwh = np.zeros((num_samples, 4), dtype=np.float32)

    for i in range(num_samples):
        xmin, xmax, ymin, ymax = y_box[i]
        xcenter = (xmin + xmax) / 2.0
        ycenter = (ymin + ymax) / 2.0
        width = xmax - xmin
        height = ymax - ymin

        y_box_cenwh[i, 0] = xcenter / image_width
        y_box_cenwh[i, 1] = ycenter / image_height
        y_box_cenwh[i, 2] = width / image_width
        y_box_cenwh[i, 3] = height / image_height

    return y_box_cenwh


def load_data(data_path):
    f = np.load(data_path)

    rgb = f["rgb"]
    thermal = f["thermal"]
    y_cls = f["class_label"]
    y_box = f["box_label"]

    y_box = np.clip(y_box, None, 300)
    y_mask = build_box_mask(y_box)
    y_box_cenwh = convert_box_to_center_width_height(y_box)

    encoder = LabelEncoder()
    y_cls_encoded = encoder.fit_transform(y_cls)

    y_triplet_box_label, box_features = generate_triplet_box_label_normalized(
        y_box, y_cls, n_clusters=3
    )

    composite_label = [
        f"{cls}_{triplet}"
        for cls, triplet in zip(y_cls_encoded, y_triplet_box_label)
    ]
    composite_label_encoded = LabelEncoder().fit_transform(composite_label)

    rgb_normalized = global_max_normalize(rgb)
    thermal_normalized = global_max_normalize(thermal)
    tile_labels = assign_tile_labels(y_box_cenwh)

    return {
        "rgb": rgb_normalized,
        "thermal": thermal_normalized,
        "y_cls": y_cls,
        "y_box": y_box_cenwh,
        "y_mask": y_mask,
        "y_triplet_box": y_triplet_box_label,
        "box_features": box_features,
        "tile_labels": tile_labels,
        "composite_label_encoded": composite_label_encoded,
    }


def make_splits(data, test_size, runs):
    (
        X_rgb_trainval,
        X_rgb_test,
        X_thermal_trainval,
        X_thermal_test,
        y_cls_trainval,
        y_cls_test,
        y_box_trainval,
        y_box_test,
        y_mask_trainval,
        y_mask_test,
        y_triplet_box_trainval,
        y_triplet_box_test,
        box_feat_trainval,
        box_feat_test,
        tile_label_trainval,
        tile_label_test,
    ) = train_test_split(
        data["rgb"],
        data["thermal"],
        data["y_cls"],
        data["y_box"],
        data["y_mask"],
        data["y_triplet_box"],
        data["box_features"],
        data["tile_labels"],
        test_size=test_size,
        stratify=data["composite_label_encoded"],
        random_state=42,
    )

    skf = StratifiedKFold(n_splits=runs, shuffle=True, random_state=42)

    train_splits = []
    val_splits = []

    for train_index, val_index in skf.split(X_rgb_trainval, y_cls_trainval):
        train_splits.append(
            (
                X_rgb_trainval[train_index],
                X_thermal_trainval[train_index],
                y_cls_trainval[train_index],
                y_box_trainval[train_index],
                y_mask_trainval[train_index],
                y_triplet_box_trainval[train_index],
                tile_label_trainval[train_index],
                box_feat_trainval[train_index],
            )
        )
        val_splits.append(
            (
                X_rgb_trainval[val_index],
                X_thermal_trainval[val_index],
                y_cls_trainval[val_index],
                y_box_trainval[val_index],
                y_mask_trainval[val_index],
                y_triplet_box_trainval[val_index],
                tile_label_trainval[val_index],
                box_feat_trainval[val_index],
            )
        )

    test_data = {
        "X_rgb_test": X_rgb_test,
        "X_thermal_test": X_thermal_test,
        "y_cls_test": y_cls_test,
        "y_box_test": y_box_test,
        "y_mask_test": y_mask_test,
        "y_triplet_box_test": y_triplet_box_test,
        "box_feat_test": box_feat_test,
        "tile_label_test": tile_label_test,
    }

    return train_splits, val_splits, test_data


def prepare_features(y_cls_train, y_cls_val, box_feat_train, box_feat_val, y_box_train, y_box_val, cls_weight):
    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_cls_train)
    y_val_encoded = label_encoder.transform(y_cls_val)

    y_train_categorical = tf.keras.utils.to_categorical(y_train_encoded)
    y_val_categorical = tf.keras.utils.to_categorical(y_val_encoded)

    box_loc_train = y_box_train[:, :2]
    box_loc_val = y_box_val[:, :2]

    train_features_dict = {
        "box_features": box_feat_train,
        "class_labels": y_train_categorical,
        "position_coordinates": box_loc_train,
    }

    val_features_dict = {
        "box_features": box_feat_val,
        "class_labels": y_val_categorical,
        "position_coordinates": box_loc_val,
    }

    weights_dict = {
        "box_features": 1.0,
        "class_labels": cls_weight,
        "position_coordinates": 1.0,
    }

    normalize_dict = {
        "box_features": False,
        "class_labels": False,
        "position_coordinates": False,
    }

    combined_train = weigh_features_multi(train_features_dict, weights_dict, normalize_dict)
    combined_val = weigh_features_multi(val_features_dict, weights_dict, normalize_dict)

    return combined_train, combined_val


def train_one_run(run_idx, train_split, val_split, args):
    (
        X_rgb_train,
        X_thermal_train,
        y_cls_train,
        y_box_train,
        y_mask_train,
        y_triplet_box_train,
        tile_label_train,
        box_feat_train,
    ) = train_split

    (
        X_rgb_val,
        X_thermal_val,
        y_cls_val,
        y_box_val,
        y_mask_val,
        y_triplet_box_val,
        tile_label_val,
        box_feat_val,
    ) = val_split

    y_cls_train, X_rgb_train, X_thermal_train, y_triplet_box_train, tile_label_train, box_feat_train, y_box_train = resampling(
        y_cls_train,
        X_rgb_train,
        X_thermal_train,
        y_triplet_box_train,
        tile_label_train,
        box_feat_train,
        y_box_train,
    )

    X_rgb_train = X_rgb_train.astype(np.float32)
    X_thermal_train = X_thermal_train.astype(np.float32)
    X_rgb_val = X_rgb_val.astype(np.float32)
    X_thermal_val = X_thermal_val.astype(np.float32)

    combined_train, combined_val = prepare_features(
        y_cls_train=y_cls_train,
        y_cls_val=y_cls_val,
        box_feat_train=box_feat_train,
        box_feat_val=box_feat_val,
        y_box_train=y_box_train,
        y_box_val=y_box_val,
        cls_weight=args.cls_weight,
    )

    s1_dim = X_rgb_train[0].shape
    s2_dim = X_thermal_train[0].shape

    model = image_reconstruction_multi_modalv1(
        s1_dim,
        s2_dim,
        args.embedding_dimension,
    )

    model.compile(
        optimizer=Adam(0.0001),
        loss=[
            keras_batch_all_triplet_continuous_loss_multimodal(
                EMB_DIM=args.embedding_dimension
            ),
            ssim_loss(alpha=0.1),
            ssim_loss(alpha=0.1),
        ],
    )

    run_dir = os.path.join(args.output_dir, f"run_{run_idx + 1}")
    os.makedirs(run_dir, exist_ok=True)

    checkpoint_path = os.path.join(run_dir, "best_model.h5")

    checkpoint_cb = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    )

    input_data_train = [X_rgb_train, X_thermal_train]
    input_data_val = [X_rgb_val, X_thermal_val]

    target_train = [combined_train, X_rgb_train, X_thermal_train]
    target_val = [combined_val, X_rgb_val, X_thermal_val]

    history_rows = []

    for iteration in range(args.iterations):
        history = model.fit(
            input_data_train,
            target_train,
            validation_data=(input_data_val, target_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1,
            callbacks=[checkpoint_cb],
        )

        for epoch_idx, (train_loss, val_loss) in enumerate(
            zip(history.history["loss"], history.history["val_loss"])
        ):
            history_rows.append(
                {
                    "run": run_idx + 1,
                    "iteration": iteration + 1,
                    "epoch": epoch_idx + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(os.path.join(run_dir, "train_log.csv"), index=False)

    return checkpoint_path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_data(args.data_path)
    train_splits, val_splits, test_data = make_splits(
        data=data,
        test_size=args.test_size,
        runs=args.runs,
    )

    summary_rows = []

    for run_idx in range(args.runs):
        checkpoint_path = train_one_run(
            run_idx=run_idx,
            train_split=train_splits[run_idx],
            val_split=val_splits[run_idx],
            args=args,
        )

        summary_rows.append(
            {
                "run": run_idx + 1,
                "checkpoint_path": checkpoint_path,
            }
        )

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.output_dir, "run_summary.csv"),
        index=False,
    )

    np.savez(
        os.path.join(args.output_dir, "test_data.npz"),
        X_rgb_test=test_data["X_rgb_test"],
        X_thermal_test=test_data["X_thermal_test"],
        y_cls_test=test_data["y_cls_test"],
        y_box_test=test_data["y_box_test"],
        y_mask_test=test_data["y_mask_test"],
        y_triplet_box_test=test_data["y_triplet_box_test"],
        box_feat_test=test_data["box_feat_test"],
        tile_label_test=test_data["tile_label_test"],
    )


if __name__ == "__main__":
    main()