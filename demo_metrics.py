import os
import numpy as np
import pandas as pd

from embedding_metrics import compute_all_alignment_and_smoothness
from utils import generate_triplet_box_label_normalized


# ------------------------------------------------
# Config
# ------------------------------------------------
RUN = 1
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(REPO_DIR, "data.npz")

EMBED_FILES = {
    "RGB": "run01_rgb_test.npy",
    "Thermal": "run01_thermal_test.npy",
    "Concat": "run01_concat_test.npy",
    "MMCTL": "run01_mmctl_test.npy",
    "DeepSetMean": "run01_deepsetmean_test.npy",
    "DeepSetAttn": "run01_deepsetattn_test.npy",
}

SOURCES = ["cls", "box", "pos"]
K_SMOOTH = 3
LAMBDA = 1


# ------------------------------------------------
# Helpers
# ------------------------------------------------
def l2_normalize_rows(z, eps=1e-12):
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    return z / (norms + eps)


def convert_box_to_center_width_height(y_box):
    image_width = 300
    image_height = 300

    y_box_cenwh = np.zeros((y_box.shape[0], 4))

    for i, (xmin, xmax, ymin, ymax) in enumerate(y_box):
        xcenter = (xmin + xmax) / 2
        ycenter = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin

        y_box_cenwh[i, 0] = xcenter / image_width
        y_box_cenwh[i, 1] = ycenter / image_height
        y_box_cenwh[i, 2] = width / image_width
        y_box_cenwh[i, 3] = height / image_height

    return y_box_cenwh


# ------------------------------------------------
# Load labels
# ------------------------------------------------
f = np.load(DATA_PATH, allow_pickle=True)

y_cls_raw = f["class_label"]
y_box = np.clip(f["box_label"], None, 300)

from sklearn.preprocessing import LabelEncoder

cls_encoder = LabelEncoder()
y_cls_int = cls_encoder.fit_transform(y_cls_raw)

y_box_cenwh = convert_box_to_center_width_height(y_box)
y_pos = y_box_cenwh[:, :2]

_, box_feat = generate_triplet_box_label_normalized(y_box, y_cls_raw, n_clusters=3)


# ------------------------------------------------
# Compute metrics
# ------------------------------------------------
rows = []

for method, file in EMBED_FILES.items():

    path = os.path.join(REPO_DIR, file)
    if not os.path.exists(path):
        print(f"Skipping {method}, file missing: {file}")
        continue

    z = np.load(path)
    z = l2_normalize_rows(z)

    print(f"{method:12s} shape={z.shape}")

    m = compute_all_alignment_and_smoothness(
        z=z,
        y_cls_int=y_cls_int,
        y_box_feat=box_feat,
        y_pos=y_pos,
        seed=42,
        pair_max_n=1200,
        smooth_max_n=None,
    )

    rows.append(
        {
            "method": method,
            "align_cls": m["align_ratio_cls"],
            "align_box": m["align_ratio_box"],
            "align_pos": m["align_ratio_pos"],
            "smooth_cls": m[f"smooth_std_k{K_SMOOTH}_cls"],
            "smooth_box": m[f"smooth_std_k{K_SMOOTH}_box"],
            "smooth_pos": m[f"smooth_std_k{K_SMOOTH}_pos"],
        }
    )

df = pd.DataFrame(rows)


# ------------------------------------------------
# Score calculation
# ------------------------------------------------
eps = 1e-12

for src in SOURCES:
    a = f"align_{src}"
    s = f"smooth_{src}"

    df[f"z_align_{src}"] = (df[a] - df[a].mean()) / (df[a].std(ddof=0) + eps)
    df[f"z_smooth_{src}"] = (df[s] - df[s].mean()) / (df[s].std(ddof=0) + eps)

    df[f"score_{src}"] = df[f"z_align_{src}"] + LAMBDA * df[f"z_smooth_{src}"]


# ------------------------------------------------
# Display results
# ------------------------------------------------
summary = df[["method", "score_cls", "score_box", "score_pos"]]
summary = summary.rename(
    columns={
        "score_cls": "Cls",
        "score_box": "Box",
        "score_pos": "Pos",
    }
)

print("\nEmbedding metric scores (lower is better)\n")
print(summary.to_string(index=False))