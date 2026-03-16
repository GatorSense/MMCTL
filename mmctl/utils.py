"""
Minimal utility functions used by train_mmctl.py.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def generate_triplet_box_label_normalized(
    y_box,
    y_cls,
    n_clusters=3,
    eps=1e-12,
    random_state=42,
):
    """
    Build continuous box features and a discrete box-cluster label.

    Returns:
        y_box_triplet_label: shape (N,)
        features: shape (N, 13)

    The first four feature columns are kept in this order for compatibility:
        [norm_width, norm_height, norm_area, norm_symmetric_squareness_deviation]
    """

    x0, x1, y0, y1 = y_box[:, 0], y_box[:, 1], y_box[:, 2], y_box[:, 3]

    width = np.maximum(x1 - x0, 0.0)
    height = np.maximum(y1 - y0, 0.0)

    area = width * height
    perimeter = 2.0 * (width + height)
    diagonal = np.sqrt(width**2 + height**2)

    aspect_ratio = width / (height + eps)
    log_aspect = np.log(aspect_ratio + eps)

    symmetric_squareness_deviation = 1.0 - np.minimum(
        width / (height + eps),
        height / (width + eps),
    )

    thinness_ratio = (4.0 * np.pi * area) / (perimeter**2 + eps)

    ixx = (width * height**3) / 12.0
    iyy = (height * width**3) / 12.0
    log_inertia_ratio = np.log((ixx + eps) / (iyy + eps) + eps)
    polar_moment = ixx + iyy
    radius_gyration_sq = polar_moment / (area + eps)

    log_area = np.log(area + eps)
    shape_angle = np.arctan2(height, width + eps)

    df = pd.DataFrame({
        "class": y_cls,
        "width": width,
        "height": height,
        "area": area,
        "perimeter": perimeter,
        "diagonal": diagonal,
        "aspect_ratio": aspect_ratio,
        "log_aspect": log_aspect,
        "symmetric_squareness_deviation": symmetric_squareness_deviation,
        "thinness_ratio": thinness_ratio,
        "log_inertia_ratio": log_inertia_ratio,
        "radius_gyration_sq": radius_gyration_sq,
        "log_area": log_area,
        "shape_angle": shape_angle,
    })

    numeric_cols = [c for c in df.columns if c != "class"]

    for col in numeric_cols:
        values = df[col].to_numpy(dtype=np.float64)
        vmin = values.min()
        vmax = values.max()
        df[f"norm_{col}"] = (values - vmin) / (vmax - vmin + eps)

    base_feature_order = [
        "norm_width",
        "norm_height",
        "norm_area",
        "norm_symmetric_squareness_deviation",
    ]

    extra_feature_order = [
        "norm_perimeter",
        "norm_diagonal",
        "norm_aspect_ratio",
        "norm_log_aspect",
        "norm_thinness_ratio",
        "norm_log_inertia_ratio",
        "norm_radius_gyration_sq",
        "norm_log_area",
        "norm_shape_angle",
    ]

    features = df[base_feature_order + extra_feature_order].to_numpy(dtype=np.float32)

    clustering_features = df[base_feature_order].to_numpy(dtype=np.float32)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df["y_box_triplet_label"] = kmeans.fit_predict(clustering_features)

    return df["y_box_triplet_label"].to_numpy(), features


def global_max_normalize(data, eps=1e-12):
    """
    Normalize an array using the global maximum magnitude.
    """
    data = np.asarray(data, dtype=np.float32)
    global_max = np.max(data)

    if global_max < eps:
        return data.copy()

    return data / global_max


def assign_tile_labels(y_box_cenwh):
    """
    Assign a 3x3 grid label using normalized box center coordinates.

    Input:
        y_box_cenwh: shape (N, 4)
            columns = [x_center, y_center, width, height]

    Returns:
        tile_labels: shape (N,)
            values in {0, ..., 8}
    """
    x_center = y_box_cenwh[:, 0]
    y_center = y_box_cenwh[:, 1]

    x_tile = np.floor(x_center * 3).astype(int)
    y_tile = np.floor(y_center * 3).astype(int)

    x_tile = np.clip(x_tile, 0, 2)
    y_tile = np.clip(y_tile, 0, 2)

    return y_tile * 3 + x_tile


def weigh_features_multi(features_dict, weights_dict, eps=1e-8):
    """
    Normalize each feature block when needed, apply dimensionality compensation,
    apply user weights, then concatenate.

    Parameters:
        features_dict:
            {feature_name: np.ndarray of shape (n_samples, n_features)}
        weights_dict:
            {feature_name: float}

    Returns:
        np.ndarray of shape (n_samples, total_features)
    """
    scaled_features = []

    max_length = max(arr.shape[1] for arr in features_dict.values())

    for feature_name, x in features_dict.items():
        if feature_name not in weights_dict:
            raise ValueError(f"Missing weight for feature block '{feature_name}'.")

        x = np.asarray(x, dtype=np.float32)

        if x.min() < 0.0 or x.max() > 1.0:
            x_min = x.min(axis=0, keepdims=True)
            x_max = x.max(axis=0, keepdims=True)
            x = (x - x_min) / (x_max - x_min + eps)

        length_scale = max_length / x.shape[1]
        block_weight = float(weights_dict[feature_name]) * length_scale

        scaled_features.append(x * block_weight)

    return np.hstack(scaled_features)


def resampling(y, *arrays):
    """
    Oversample all classes in y to the frequency of the largest class.

    Returns:
        (y_resampled, *arrays_resampled)
    """
    y = np.asarray(y)
    unique_labels, counts = np.unique(y, return_counts=True)
    max_count = int(np.max(counts))

    balanced_indices = []

    for label in unique_labels:
        label_indices = np.where(y == label)[0].tolist()

        if len(label_indices) < max_count:
            repeated = label_indices.copy()
            while len(repeated) < max_count:
                repeated.extend(label_indices)
            label_indices = repeated[:max_count]

        balanced_indices.extend(label_indices)

    balanced_indices = np.asarray(balanced_indices, dtype=int)

    y_balanced = y[balanced_indices]
    resampled_arrays = [np.asarray(arr)[balanced_indices] for arr in arrays]

    return (y_balanced, *resampled_arrays)