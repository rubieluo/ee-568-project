"""Plot per-eval training curves from results/training_trace.csv.

Produces one panel per algorithm with one line per K value (colored by K),
plus a combined figure with all algorithms side by side.

Re-run anytime — even mid-sweep — to see current progress.

Usage:
    python plot_training_curves.py                # all seeds averaged (±std shaded)
    python plot_training_curves.py --seed 0       # only seed 0
    python plot_training_curves.py --seed 0 --seed 2   # average over seeds 0 and 2
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DEFAULT_TRACE_CSV   = "results/training_trace.csv"
DEFAULT_FIGURES_DIR = "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":   True,
    "grid.alpha":  0.3,
    "grid.linestyle": "--",
    "figure.dpi":  150,
})

ALGO_LABELS = {
    "iq_learn":   "IQ-Learn",
    "f_irl":      "f-IRL",
    "soar_f_irl": "SOAR+f-IRL",
}
ALGO_ORDER = ["iq_learn", "f_irl", "soar_f_irl"]


def k_color(k, k_values):
    idx = k_values.index(k)
    cmap = cm.get_cmap("viridis")
    if len(k_values) == 1:
        return cmap(0.5)
    return cmap(idx / (len(k_values) - 1))


def plot_algo(df, algo, ax, env_name):
    sub = df[(df["algo"] == algo) & (df["env"] == env_name)].copy()
    if sub.empty:
        ax.text(0.5, 0.5, "(no data yet)", ha="center", va="center",
                transform=ax.transAxes, color="grey")
        ax.set_title(f"{ALGO_LABELS.get(algo, algo)} – {env_name}",
                     fontsize=11, fontweight="bold")
        return

    k_values = sorted(sub["K"].unique().tolist())
    for K in k_values:
        sk = sub[sub["K"] == K].sort_values("step")
        if sk.empty:
            continue
        # average across seeds at each step (if multiple)
        agg = sk.groupby("step")["eval_return"].agg(["mean", "std", "count"]).reset_index()
        c = k_color(K, k_values)
        ax.plot(agg["step"], agg["mean"], marker="o", markersize=4,
                linewidth=1.6, color=c, label=f"K={K}")
        if (agg["count"] > 1).any():
            ax.fill_between(agg["step"],
                            agg["mean"] - agg["std"].fillna(0),
                            agg["mean"] + agg["std"].fillna(0),
                            alpha=0.15, color=c)

    ax.set_xlabel("environment step")
    ax.set_ylabel("eval return")
    ax.set_title(f"{ALGO_LABELS.get(algo, algo)} – {env_name}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.6, title="expert size")


def make_per_env_figure(df, env_name, figures_dir, tag=""):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for ax, algo in zip(axes, ALGO_ORDER):
        plot_algo(df, algo, ax, env_name)
    title = f"Training curves – {env_name}"
    if tag:
        title += f" ({tag})"
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    safe = env_name.replace("-", "_")
    suffix = f"_{tag}" if tag else ""
    for ext in ["png", "pdf"]:
        out = os.path.join(figures_dir, f"training_curves_{safe}{suffix}.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print(f"saved {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default=DEFAULT_TRACE_CSV,
                        help=f"path to training_trace.csv (default: {DEFAULT_TRACE_CSV})")
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES_DIR,
                        help=f"output dir for figures (default: {DEFAULT_FIGURES_DIR})")
    parser.add_argument("--seed", type=int, action="append", default=None,
                        help="only include this seed (repeatable, e.g. --seed 0 --seed 2). omit for all seeds.")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)

    if not os.path.exists(args.trace):
        print(f"no trace file found at {args.trace} — run the training sweep first")
        return
    df = pd.read_csv(args.trace)
    if df.empty:
        print("trace file is empty (training may not have produced an eval yet)")
        return

    tag = ""
    if args.seed is not None:
        df = df[df["seed"].isin(args.seed)]
        if df.empty:
            avail = sorted(pd.read_csv(args.trace)['seed'].unique().tolist())
            print(f"no rows match seed(s) {args.seed} — available: {avail}")
            return
        tag = "seed" + "_".join(str(s) for s in args.seed)

    envs = sorted(df["env"].unique().tolist())
    print(f"trace contains {len(df)} rows across envs {envs}"
          + (f" (filtered to seed(s) {args.seed})" if args.seed is not None else ""))
    print(df.groupby(["env", "algo", "K"]).size().rename("evals"))

    for env_name in envs:
        make_per_env_figure(df, env_name, args.figures_dir, tag=tag)


if __name__ == "__main__":
    main()
