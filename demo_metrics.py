import argparse
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from mmctl.embedding_metrics import compute_all_alignment_and_smoothness
from mmctl.utils import generate_triplet_box_label_normalized


# -----------------------------
# Config
# -----------------------------
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(REPO_DIR, "awir_data.npz")
EMB_DIR = os.path.join(REPO_DIR, "pretrained_embeddings")

parser = argparse.ArgumentParser(description="Compute demo metrics from MMCTL embeddings.")
parser.add_argument(
    "--data",
    "-d",
    default=DEFAULT_DATA_PATH,
    help="Path to a dataset NPZ file (default: awir_data.npz in the repo directory).",
)
args = parser.parse_args()
DATA_PATH = args.data

RUNS = range(1, 9)

K_SMOOTH = 3
LAMBDA = 1
DECIMALS = 2
TEST_SIZE = 0.5
SEED = 42

METHOD_SPECS = {
    "RGB": {"type": "file", "pattern": "run{run:02d}_rgb_test.npy"},
    "Thermal": {"type": "file", "pattern": "run{run:02d}_thermal_test.npy"},
    "MMCTL": {"type": "file", "pattern": "run{run:02d}_mmctl_test.npy"},
    "DeepSetMean": {"type": "file", "pattern": "run{run:02d}_deepset_mean_test_fused.npy"},
    "DeepSetAttn": {"type": "file", "pattern": "run{run:02d}_deepset_attention_test_fused.npy"},
    "Concat": {"type": "concat"},
}

SOURCES = ["cls", "box", "pos"]
ALIGN_KEYS = {src: f"align_ratio_{src}" for src in SOURCES}
SMOOTH_KEYS = {src: f"smooth_std_k{K_SMOOTH}_{src}" for src in SOURCES}


# -----------------------------
# Helpers
# -----------------------------
def load_npy(pattern: str, run: int) -> np.ndarray:
    path = os.path.join(EMB_DIR, pattern.format(run=run))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing embedding file: {path}")
    return np.load(path, allow_pickle=True)


def l2_normalize_rows(z: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    return z / (norms + eps)


def convert_box_to_center_width_height(y_box: np.ndarray) -> np.ndarray:
    image_width = 300.0
    image_height = 300.0

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


def get_metrics(z: np.ndarray, y_cls_int: np.ndarray, y_box_feat: np.ndarray, y_pos: np.ndarray):
    return compute_all_alignment_and_smoothness(
        z=z,
        y_cls_int=y_cls_int,
        y_box_feat=y_box_feat,
        y_pos=y_pos,
        seed=SEED,
        pair_max_n=1200,
        smooth_max_n=None,
    )


def pm_str(m, s, decimals=DECIMALS):
    if pd.isna(m) or pd.isna(s):
        return "--"
    return f"{m:.{decimals}f}±{s:.{decimals}f}"


# -----------------------------
# Rebuild the SAME test split
# -----------------------------
f = np.load(DATA_PATH, allow_pickle=True)

y_cls_raw = f["class_label"]
y_box = np.clip(f["box_label"].astype(np.float32), None, 300)

# box -> center, width, height
y_box_cenwh = convert_box_to_center_width_height(y_box)

# class ints
cls_encoder = LabelEncoder()
y_cls_int_all = cls_encoder.fit_transform(y_cls_raw).astype(np.int64)

# box features and triplet labels
y_triplet_box_label, box_feat_all = generate_triplet_box_label_normalized(
    y_box, y_cls_raw, n_clusters=3
)
box_feat_all = box_feat_all.astype(np.float32)

# pos targets
y_pos_all = y_box_cenwh[:, :2].astype(np.float32)

# composite label, same as original
composite_label = [f"{c}_{t}" for c, t in zip(y_cls_int_all, y_triplet_box_label)]
composite_label_encoded = LabelEncoder().fit_transform(composite_label)

# exact same split structure as original first split
idx_all = np.arange(len(y_cls_raw))
_, idx_test = train_test_split(
    idx_all,
    test_size=TEST_SIZE,
    stratify=composite_label_encoded,
    random_state=SEED,
)

y_test_int = y_cls_int_all[idx_test]
box_feat_test = box_feat_all[idx_test]
y_pos_test = y_pos_all[idx_test]


# -----------------------------
# Collect per-run metrics
# -----------------------------
rows = []

for run in RUNS:
    print(f"run={run:02d}")

    z_rgb_raw = load_npy("run{run:02d}_rgb_test.npy", run)
    z_th_raw = load_npy("run{run:02d}_thermal_test.npy", run)

    if z_rgb_raw.shape[0] != z_th_raw.shape[0]:
        raise ValueError(
            f"Concat N mismatch at run={run:02d}: rgb={z_rgb_raw.shape}, thermal={z_th_raw.shape}"
        )

    if z_rgb_raw.shape[0] != len(idx_test):
        raise ValueError(
            f"Test embedding count does not match reconstructed test split at run={run:02d}: "
            f"embeddings={z_rgb_raw.shape[0]}, expected_test={len(idx_test)}"
        )

    for method, spec in METHOD_SPECS.items():
        if spec["type"] == "file":
            z = load_npy(spec["pattern"], run)
            z = l2_normalize_rows(z)

        elif spec["type"] == "concat":
            z = np.concatenate([z_rgb_raw, z_th_raw], axis=1)
            z = l2_normalize_rows(z)

        else:
            raise ValueError(f"Unknown method spec type for {method}")

        print(
            f"  {method:<12} | "
            f"min={z.min():.4f}  max={z.max():.4f}  "
            f"absmax={np.abs(z).max():.4f}  "
            f"mean||z||={np.mean(np.linalg.norm(z, axis=1)):.3f}  "
            f"shape={z.shape}"
        )

        m = get_metrics(z, y_test_int, box_feat_test, y_pos_test)

        missing = [
            k
            for k in list(ALIGN_KEYS.values()) + list(SMOOTH_KEYS.values())
            if k not in m
        ]
        if missing:
            raise KeyError(
                f"Missing keys in metrics for method={method}, run={run}: {missing}"
            )

        rows.append(
            {
                "run": run,
                "method": method,
                **{f"align_{src}": float(m[ALIGN_KEYS[src]]) for src in SOURCES},
                **{f"smooth_{src}": float(m[SMOOTH_KEYS[src]]) for src in SOURCES},
            }
        )

df = pd.DataFrame(rows)

# -----------------------------
# Per-source score
# -----------------------------
eps = 1e-12
for src in SOURCES:
    a = f"align_{src}"
    s = f"smooth_{src}"

    df[f"z_align_{src}"] = (df[a] - df[a].mean()) / (df[a].std(ddof=0) + eps)
    df[f"z_smooth_{src}"] = (df[s] - df[s].mean()) / (df[s].std(ddof=0) + eps)
    df[f"score_{src}"] = df[f"z_align_{src}"] + LAMBDA * df[f"z_smooth_{src}"]

# -----------------------------
# Aggregate
# -----------------------------
long = df.melt(
    id_vars=["run", "method"],
    value_vars=[f"score_{src}" for src in SOURCES],
    var_name="source",
    value_name="score",
)
long["source"] = long["source"].str.replace("score_", "", regex=False)

agg = (
    long.groupby(["method", "source"])["score"]
    .agg(["mean", "std"])
    .reset_index()
)

agg["pm"] = agg.apply(lambda r: pm_str(r["mean"], r["std"]), axis=1)

order = ["RGB", "Thermal", "Concat", "MMCTL", "DeepSetMean", "DeepSetAttn"]
wide = (
    agg.pivot(index="method", columns="source", values="pm")
    .reindex(order)
    .rename(columns={"cls": "Cls", "box": "Box", "pos": "Pos"})
)

print(f"\nPer-source score uses: Z(align_ratio_src) + {LAMBDA} * Z(smooth_std_k{K_SMOOTH}_src)")
print("Lower is better in every column.\n")
print(wide)

# Optional save
raw_csv = os.path.join(REPO_DIR, "demo_metrics_raw.csv")
summary_csv = os.path.join(REPO_DIR, "demo_metrics_summary.csv")

df.to_csv(raw_csv, index=False)
wide.to_csv(summary_csv)

print(f"\nSaved raw per-run metrics to: {raw_csv}")
print(f"Saved summary table to: {summary_csv}")