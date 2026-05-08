"""
Phase 3: Run all IL experiments.
Saves results to results/results.csv for plotting.

Usage:
    python run_experiments.py              # full sweep
    python run_experiments.py --dry-run    # print planned runs without executing

Requires: iq_learn/, f_irl/ directories (see README for setup)
"""

import argparse
import csv
import itertools
import os
import subprocess
import sys
from datetime import datetime

ENVS   = ["CartPole-v1", "Pendulum-v1"]
ALGOS  = ["iq_learn", "f_irl", "soar_f_irl"]
K_LIST = [1, 5, 20, 100]
SEEDS  = [0, 1, 2]

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")


def planned_runs():
    return list(itertools.product(ENVS, ALGOS, K_LIST, SEEDS))


def run_single(env, algo, K, seed):
    """
    Dispatch a single IL training run.
    Replace the subprocess call with your actual algo runner.
    Returns mean_return (float).
    """
    dataset = f"datasets/{env}_K{K}.npz"

    # ---- IQ-Learn ----
    if algo == "iq_learn":
        cmd = [
            sys.executable, "iq_learn/train.py",
            "--env", env,
            "--dataset", dataset,
            "--seed", str(seed),
            "--output_dir", RESULTS_DIR,
        ]

    # ---- f-IRL ----
    elif algo == "f_irl":
        cmd = [
            sys.executable, "f_irl/train.py",
            "--env", env,
            "--dataset", dataset,
            "--seed", str(seed),
            "--output_dir", RESULTS_DIR,
        ]

    # ---- SOAR + f-IRL ----
    elif algo == "soar_f_irl":
        cmd = [
            sys.executable, "f_irl/train.py",
            "--env", env,
            "--dataset", dataset,
            "--seed", str(seed),
            "--soar",           # flag enabling SOAR augmentation
            "--output_dir", RESULTS_DIR,
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-300:]}")
        return float("nan")

    # Parse mean return from stdout — adapt to your algo's output format
    for line in reversed(result.stdout.splitlines()):
        if "mean_return" in line.lower():
            try:
                return float(line.split(":")[-1].strip())
            except ValueError:
                pass
    return float("nan")


def main(dry_run=False):
    runs = planned_runs()
    print(f"Total runs planned: {len(runs)}")

    if dry_run:
        for env, algo, K, seed in runs:
            print(f"  {env:25s}  algo={algo:12s}  K={K:4d}  seed={seed}")
        return

    fieldnames = ["env", "algo", "K", "seed", "mean_return", "timestamp"]
    write_header = not os.path.exists(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for i, (env, algo, K, seed) in enumerate(runs):
            print(f"\n[{i+1}/{len(runs)}] {env} | {algo} | K={K} | seed={seed}")
            mean_return = run_single(env, algo, K, seed)
            row = dict(env=env, algo=algo, K=K, seed=seed,
                       mean_return=mean_return,
                       timestamp=datetime.now().isoformat())
            writer.writerow(row)
            f.flush()
            print(f"  → mean_return = {mean_return:.2f}")

    print(f"\nDone. Results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
