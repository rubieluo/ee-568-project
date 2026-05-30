"""Plot per-eval training curves from results/training_trace.csv.

Produces one panel per algorithm with one line per K value (colored by K),
plus a combined figure with all algorithms side by side.

Re-run anytime — even mid-sweep — to see current progress.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

TRACE_CSV   = "results/training_trace.csv"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

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


def make_per_env_figure(df, env_name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for ax, algo in zip(axes, ALGO_ORDER):
        plot_algo(df, algo, ax, env_name)
    fig.suptitle(f"Training curves – {env_name}", fontsize=12, y=1.02)
    fig.tight_layout()
    safe = env_name.replace("-", "_")
    for ext in ["png", "pdf"]:
        out = os.path.join(FIGURES_DIR, f"training_curves_{safe}.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print(f"saved {out}")
    plt.close(fig)


def main():
    if not os.path.exists(TRACE_CSV):
        print(f"no trace file found at {TRACE_CSV} — run the training sweep first")
        return
    df = pd.read_csv(TRACE_CSV)
    if df.empty:
        print("trace file is empty (training may not have produced an eval yet)")
        return

    envs = sorted(df["env"].unique().tolist())
    print(f"trace contains {len(df)} rows across envs {envs}")
    print(df.groupby(["env", "algo", "K"]).size().rename("evals"))

    for env_name in envs:
        make_per_env_figure(df, env_name)


if __name__ == "__main__":
    main()
