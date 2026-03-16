import os
import subprocess
import sys


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(repo_root, "data.npz")
    output_dir = os.path.join(repo_root, "smoke_test_outputs")

    if not os.path.exists(data_path):
        print("ERROR: data.npz not found in repo directory.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "train_mmctl.py",
        "--data_path", data_path,
        "--output_dir", output_dir,
        "--epochs", "1",
        "--iterations", "1",
        "--runs", "2",
        "--batch_size", "4",
        "--test_size", "0.25",
    ]

    print("\nRunning MMCTL smoke test\n")
    print("Command:", " ".join(cmd), "\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\nSmoke test FAILED")
        sys.exit(result.returncode)

    # Check expected outputs
    expected_files = [
        os.path.join(output_dir, "run_summary.csv"),
        os.path.join(output_dir, "test_data.npz"),
        os.path.join(output_dir, "run_1", "best_model.h5"),
        os.path.join(output_dir, "run_1", "train_log.csv"),
    ]

    missing = [f for f in expected_files if not os.path.exists(f)]

    if missing:
        print("\nSmoke test ran but missing files:")
        for f in missing:
            print(f)
        sys.exit(1)

    print("\nSmoke test PASSED")
    print("Outputs written to:", output_dir)


if __name__ == "__main__":
    main()