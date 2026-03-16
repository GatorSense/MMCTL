import os
import subprocess
import sys


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(repo_root, "data.npz")
    mmctl_dir = os.path.join(repo_root, "smoke_test_outputs")
    out_dir = os.path.join(repo_root, "smoke_test_embedding_metrics_outputs")

    if not os.path.exists(data_path):
        print("ERROR: data.npz not found.")
        sys.exit(1)

    run1_ckpt = os.path.join(mmctl_dir, "run_1", "best_model.h5")
    if not os.path.exists(run1_ckpt):
        print("ERROR: MMCTL smoke test checkpoint not found. Run smoke_test.py first.")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "mmctl/embedding_metrics.py",
        "--data_npz", data_path,
        "--mmctl_dir", mmctl_dir,
        "--out_dir", out_dir,
        "--runs", "1",
        "--test_size", "0.25",
        "--batch_size", "4",
        "--embedding_dimension", "1024",
        "--margin", "0.1",
        "--pair_max_n", "64",
        "--smooth_max_n", "64",
        "--split", "test",
    ]

    print("\nRunning embedding-metrics smoke test\n")
    print("Command:", " ".join(cmd), "\n")

    result = subprocess.run(cmd, cwd=repo_root)

    if result.returncode != 0:
        print("\nEmbedding-metrics smoke test FAILED")
        sys.exit(result.returncode)

    expected_files = [
        os.path.join(out_dir, "embedding_metrics.csv"),
        os.path.join(out_dir, "embedding_metrics.json"),
        os.path.join(out_dir, "embedding_metrics_summary.csv"),
    ]

    missing = [f for f in expected_files if not os.path.exists(f)]
    if missing:
        print("\nEmbedding-metrics smoke test ran but missing files:")
        for f in missing:
            print(" ", f)
        sys.exit(1)

    print("\nEmbedding-metrics smoke test PASSED")
    print("Outputs written to:", out_dir)


if __name__ == "__main__":
    main()