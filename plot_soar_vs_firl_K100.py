"""Focused plot: SOAR vs f-IRL on CartPole at K=100.

Shows the long-flat-then-takeoff pattern that distinguishes SOAR's
optimism-driven exploration from plain f-IRL at the largest expert
dataset size. Per-seed curves are drawn lightly and the across-seed
mean is drawn bold so the per-seed takeoffs remain visible.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_TRACE = "results_new/results/training_trace.csv"
DEFAULT_OUT   = "results_new/figures"

STYLE = {
    "f_irl":      dict(color="#DC2626", label="f-IRL"),
    "soar_f_irl": dict(color="#16A34A", label="SOAR+f-IRL"),
}

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default=DEFAULT_TRACE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--K", type=int, default=100)
    args = parser.parse_args()

    df = pd.read_csv(args.trace)
    df = df[(df.env == args.env) & (df.K == args.K)
            & df.algo.isin(["f_irl", "soar_f_irl"])]
    if df.empty:
        print("no matching rows")
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for algo, style in STYLE.items():
        sub = df[df.algo == algo]
        if sub.empty:
            continue
        # per-seed (light)
        for seed, g in sub.sort_values("step").groupby("seed"):
            ax.plot(g["step"], g["eval_return"], color=style["color"],
                    alpha=0.25, linewidth=1.0)
        # cross-seed mean (bold)
        agg = sub.groupby("step")["eval_return"].mean().reset_index()
        ax.plot(agg["step"], agg["eval_return"], color=style["color"],
                linewidth=2.2, label=style["label"], marker="o", markersize=4)

    ax.axhline(500, color="grey", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(ax.get_xlim()[1] * 0.99, 505, "max return = 500",
            ha="right", va="bottom", fontsize=8, color="grey")
    ax.set_xlabel("environment step")
    ax.set_ylabel("eval return")
    ax.set_title(f"{args.env}, K={args.K}: SOAR's optimism takeoff vs.\\ plain f-IRL",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9, title="(thin = per seed; bold = mean)")
    fig.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    safe_env = args.env.replace("-", "_")
    for ext in ["png", "pdf"]:
        path = os.path.join(args.out_dir, f"soar_vs_firl_{safe_env}_K{args.K}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
