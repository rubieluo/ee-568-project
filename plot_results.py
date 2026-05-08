"""
plot_results.py — Generate all figures for the report.
Usage:
    python plot_results.py

Reads:  results/results.csv
Saves:  figures/  (one PDF per environment, plus a combined figure)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_CSV = "results/results.csv"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

ALGO_STYLE = {
    "iq_learn":    dict(color="#2563EB", marker="o",  linestyle="-",  label="IQ-Learn"),
    "f_irl":       dict(color="#DC2626", marker="s",  linestyle="--", label="f-IRL"),
    "soar_f_irl":  dict(color="#16A34A", marker="^",  linestyle="-.", label="SOAR + f-IRL"),
}


def load_results():
    df = pd.read_csv(RESULTS_CSV)
    # Aggregate over seeds
    agg = (
        df.groupby(["env", "algo", "K"])["mean_return"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    agg["se"] = agg["std"] / np.sqrt(agg["count"])  # standard error for shading
    return agg


def plot_env(agg, env_name, ax, show_ylabel=True):
    data = agg[agg["env"] == env_name]
    K_vals = sorted(data["K"].unique())

    for algo, style in ALGO_STYLE.items():
        sub = data[data["algo"] == algo].sort_values("K")
        if sub.empty:
            continue
        ax.plot(sub["K"], sub["mean"], label=style["label"],
                color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], linewidth=1.8, markersize=6)
        ax.fill_between(sub["K"],
                        sub["mean"] - sub["std"],
                        sub["mean"] + sub["std"],
                        alpha=0.15, color=style["color"])

    ax.set_xscale("log")
    ax.set_xticks(K_vals)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Expert trajectories K", fontsize=11)
    if show_ylabel:
        ax.set_ylabel("Mean episode return", fontsize=11)
    ax.set_title(env_name, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.5)


def make_combined_figure(agg):
    """Single figure with one subplot per environment — for the report."""
    envs = agg["env"].unique()
    fig, axes = plt.subplots(1, len(envs), figsize=(5.5 * len(envs), 4), sharey=False)
    if len(envs) == 1:
        axes = [axes]

    for i, (env, ax) in enumerate(zip(envs, axes)):
        plot_env(agg, env, ax, show_ylabel=(i == 0))

    fig.suptitle("IL algorithm comparison: mean return vs. expert dataset size\n"
                 "(shaded region = ±1 std across 3 seeds)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "combined_results.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")


def make_per_env_figures(agg):
    """One figure per environment — useful for poster / slides."""
    for env in agg["env"].unique():
        fig, ax = plt.subplots(figsize=(5, 3.8))
        plot_env(agg, env, ax)
        fig.tight_layout()
        safe_name = env.replace("-", "_").replace("/", "_")
        path = os.path.join(FIGURES_DIR, f"{safe_name}.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")
        plt.close(fig)


def print_summary_table(agg):
    """Print a LaTeX-ready results table."""
    print("\n% LaTeX results table (paste into report)\n")
    print(r"\begin{tabular}{lllcc}")
    print(r"\toprule")
    print(r"Environment & Algorithm & K & Mean return & Std \\")
    print(r"\midrule")
    for _, row in agg.iterrows():
        algo_label = ALGO_STYLE[row["algo"]]["label"]
        print(f"{row['env']} & {algo_label} & {int(row['K'])} & "
              f"{row['mean']:.1f} & {row['std']:.1f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    agg = load_results()
    make_combined_figure(agg)
    make_per_env_figures(agg)
    print_summary_table(agg)
    print("\nAll figures saved to figures/")
