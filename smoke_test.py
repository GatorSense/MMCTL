import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def make_fake_data(path, n=16, h=300, w=300, c=3):
    rng = np.random.default_rng(42)

    rgb = rng.random((n, h, w, c), dtype=np.float32)
    thermal = rng.random((n, h, w, c), dtype=np.float32)

    class_names = np.array(["cow", "deer", "horse"], dtype="<U10")
    class_label = rng.choice(class_names, size=n)

    box_label = np.zeros((n, 4), dtype=np.int32)
    for i in range(n):
        xmin = rng.integers(0, 180)
        ymin = rng.integers(0, 180)
        bw = rng.integers(30, 100)
        bh = rng.integers(30, 100)
        xmax = min(xmin + bw, w - 1)
        ymax = min(ymin + bh, h - 1)
        box_label[i] = [xmin, xmax, ymin, ymax]

    np.savez(
        path,
        rgb=rgb,
        thermal=thermal,
        class_label=class_label,
        box_label=box_label,
    )


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = tempfile.mkdtemp(prefix="mmctl_smoke_")

    try:
        data_path = os.path.join(tmp_dir, "fake_data.npz")
        out_dir = os.path.join(tmp_dir, "outputs")

        make_fake_data(data_path, n=16)

        cmd = [
            sys.executable,
            os.path.join(repo_root, "train_mmctl.py"),
            "--data_path", data_path,
            "--output_dir", out_dir,
            "--epochs", "1",
            "--iterations", "1",
            "--runs", "2",
            "--batch_size", "4",
            "--test_size", "0.5",
            "--embedding_dimension", "1024",
            "--margin", "0.1",
            "--cls_weight", "0.5",
        ]

        print("Running smoke test...")
        print("Command:", " ".join(cmd))

        result = subprocess.run(cmd, cwd=repo_root, check=False)

        if result.returncode != 0:
            print("\nSmoke test failed.")
            sys.exit(result.returncode)

        expected_files = [
            os.path.join(out_dir, "run_summary.csv"),
            os.path.join(out_dir, "test_data.npz"),
            os.path.join(out_dir, "run_1", "best_model.h5"),
            os.path.join(out_dir, "run_1", "train_log.csv"),
        ]

        missing = [p for p in expected_files if not os.path.exists(p)]

        if missing:
            print("\nSmoke test ran, but some expected files are missing:")
            for p in missing:
                print("  ", p)
            sys.exit(1)

        print("\nSmoke test passed.")
        print("Created outputs in:", out_dir)

    finally:
        # comment this out if you want to inspect temp outputs
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()