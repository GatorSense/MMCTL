"""Minimal neon_utilities for paper reconstruction (adapted from CTL/mmctl)."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import LabelEncoder

# Re-export shared helpers from mmctl
from mmctl.utils import (  # noqa: F401
    assign_tile_labels,
    generate_triplet_box_label_normalized,
    global_max_normalize,
    resampling,
    weigh_features_multi,
)


def remove_bands(array: np.ndarray) -> np.ndarray:
    """Drop noisy HSI bands. Expects NHWC with 426 channels -> 369."""
    x = np.asarray(array)
    axis = -1 if x.ndim == 4 else 0
    x = np.delete(x, np.r_[419:426], axis=axis)
    x = np.delete(x, np.r_[283:315], axis=axis)
    x = np.delete(x, np.r_[192:210], axis=axis)
    return x


def center_crop_spatial(imgs: np.ndarray, out_h: int, out_w: int):
    """Center-crop NHWC images to (out_h, out_w)."""
    imgs = np.asarray(imgs)
    _, h, w, _ = imgs.shape
    if h == out_h and w == out_w:
        return imgs, np.arange(len(imgs))
    if h < out_h or w < out_w:
        raise ValueError(f"Cannot crop {imgs.shape} to {(out_h, out_w)}")
    top = (h - out_h) // 2
    left = (w - out_w) // 2
    return imgs[:, top : top + out_h, left : left + out_w, :], np.arange(len(imgs))


def center_crop_on_box(imgs, box_labels, patch_size):
    """
    Paper script API. For already-patched CTL NEON cubes, center-crop to patch_size.
    RGB uses patch_size (e.g. 60); HSI/LiDAR use patch_size (e.g. 6).
    """
    imgs = np.asarray(imgs)
    return center_crop_spatial(imgs, int(patch_size), int(patch_size))


def filter_npz_in_memory(npz_path: str, indices: np.ndarray) -> dict:
    f = np.load(npz_path, allow_pickle=True)
    idx = np.asarray(indices, dtype=np.int64)
    out = {}
    for k in f.files:
        arr = f[k]
        out[k] = arr[idx] if len(arr) == len(f[f.files[0]]) or (
            hasattr(arr, "shape") and arr.shape and arr.shape[0] == f[f.files[0]].shape[0]
        ) else arr
    # Always index sample-aligned keys
    n = f[f.files[0]].shape[0]
    for k in f.files:
        if hasattr(f[k], "shape") and len(f[k].shape) >= 1 and f[k].shape[0] == n:
            out[k] = f[k][idx]
    return out


def filter_and_remap_top_classes(
    rgb, hsi, lidar, y_cls, y_pos, y_box, y_height, top_X=10
):
    y_cls = np.asarray(y_cls)
    # encode strings if needed
    if y_cls.dtype.kind in ("U", "S", "O"):
        y_int = LabelEncoder().fit_transform(y_cls)
    else:
        y_int = y_cls.astype(np.int64)

    counts = np.bincount(y_int)
    top = np.argsort(counts)[::-1][:top_X]
    keep = np.isin(y_int, top)
    mapping = {int(old): i for i, old in enumerate(top)}
    y_new = np.array([mapping[int(c)] for c in y_int[keep]], dtype=np.int64)

    return (
        rgb[keep],
        hsi[keep],
        lidar[keep],
        y_new,
        y_pos[keep],
        y_box[keep],
        y_height[keep],
        mapping,
    )


# Stubs for unused comet/plot helpers imported by the original script
def plot_pca_umap(*args, **kwargs):
    return None


def calculate_iou(*args, **kwargs):
    return 0.0


def calculate_average_iou(*args, **kwargs):
    return 0.0


def calculate_mask_iou(*args, **kwargs):
    return 0.0


def adjust_and_calculate_iou_mask(*args, **kwargs):
    return 0.0


def print_and_log_average_iou(*args, **kwargs):
    return None


def generate_triplet_box_label_one_hot(*args, **kwargs):
    raise NotImplementedError


def log_one_image_per_class(*args, **kwargs):
    return None


def log_gt_image_per_class(*args, **kwargs):
    return None


def log_one_hsi_recon_per_class(*args, **kwargs):
    return None


def log_gt_hsi_per_class(*args, **kwargs):
    return None
