import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_CSV = "results/all_results.csv"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":    True,
    "grid.alpha":   0.3,
    "grid.linestyle": "--",
    "figure.dpi":   150,
})

ALGO_STYLE = {
    "iq_learn":   dict(color="#2563EB", marker="o", linestyle="-",  label="IQ-Learn"),
    "f_irl":      dict(color="#DC2626", marker="s", linestyle="--", label="f-IRL"),
    "soar_f_irl": dict(color="#16A34A", marker="^", linestyle="-.", label="SOAR+f-IRL"),
}


def load_results(metric="mean_return"):
    df  = pd.read_csv(RESULTS_CSV)
    agg = (
        df.groupby(["env", "algo", "K"])[metric]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    return agg


def plot_env(agg, env_name, ax, show_ylabel=True, title_suffix=""):
    data   = agg[agg["env"] == env_name]
    K_vals = sorted(data["K"].unique())

    for algo, style in ALGO_STYLE.items():
        sub = data[data["algo"] == algo].sort_values("K")
        if sub.empty:
            continue
        ax.plot(sub["K"], sub["mean"],
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.8, markersize=6)
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
    title = env_name
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.5)


def make_main_figure(metric="mean_return", suffix="", filename="combined_results"):
    agg  = load_results(metric)
    envs = ["CartPole-v1", "Pendulum-v1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for i, (env, ax) in enumerate(zip(envs, axes)):
        plot_env(agg, env, ax, show_ylabel=(i == 0), title_suffix=suffix)
    metric_label = "mean return" if metric == "mean_return" else "best return"
    fig.suptitle(
        f"IL algorithm comparison: {metric_label} vs. expert dataset size\n"
        "(shaded region = ±1 std across 3 seeds)",
        fontsize=11, y=1.02
    )
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(FIGURES_DIR, f"{filename}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def make_per_env_figures(metric="mean_return", suffix=""):
    agg  = load_results(metric)
    envs = ["CartPole-v1", "Pendulum-v1"]
    for env in envs:
        fig, ax = plt.subplots(figsize=(5, 3.8))
        plot_env(agg, env, ax, title_suffix=suffix)
        fig.tight_layout()
        safe = env.replace("-", "_").replace("/", "_")
        tag  = f"_{suffix.replace(' ', '_')}" if suffix else ""
        for ext in ["pdf", "png"]:
            path = os.path.join(FIGURES_DIR, f"{safe}{tag}.{ext}")
            fig.savefig(path, bbox_inches="tight")
            print(f"Saved {path}")
        plt.close(fig)


def print_summary_table(metric="mean_return"):
    agg = load_results(metric)
    print(f"\n% LaTeX results table ({metric})\n")
    print(r"\begin{tabular}{llrcc}")
    print(r"\toprule")
    print(r"Environment & Algorithm & $K$ & Mean & Std \\")
    print(r"\midrule")
    for env in ["CartPole-v1", "Pendulum-v1"]:
        for algo, style in ALGO_STYLE.items():
            sub = agg[(agg["env"] == env) & (agg["algo"] == algo)].sort_values("K")
            for _, row in sub.iterrows():
                print(f"{env} & {style['label']} & {int(row['K'])} & "
                      f"{row['mean']:.1f} & {row['std']:.1f} \\\\")
        print(r"\midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    make_main_figure(metric="mean_return", filename="combined_mean_return")
    make_per_env_figures(metric="mean_return")
    make_main_figure(metric="best_return", suffix="best return",
                     filename="combined_best_return")
    make_per_env_figures(metric="best_return", suffix="best return")
    print_summary_table(metric="mean_return")
    print_summary_table(metric="best_return")
    print("\nAll figures saved to figures/")