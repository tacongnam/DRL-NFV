import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def _plot_baseline_results(results: list, out_path: str = None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[Plot] matplotlib not available, skipping.")
        return

    names = [r["name"]  for r in results]
    ar    = [r["ar"]    for r in results]
    cost  = [r["cost"]  for r in results]
    delay = [r["delay"] for r in results]
    x     = np.arange(len(names))
    w     = 0.55
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(names)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Baseline Algorithm Comparison", fontsize=14, fontweight="bold")

    for ax, vals, title, ylabel in [
        (axes[0], ar,    "Acceptance Ratio", "Ratio"),
        (axes[1], cost,  "Total Cost",       "Cost"),
        (axes[2], delay, "Total Delay",      "Delay"),
    ]:
        bars = ax.bar(x, vals, w, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        mx = max(vals) if max(vals) > 0 else 1
        fmt = ".3f" if title == "Acceptance Ratio" else ".1f"
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + mx * 0.01,
                    f"{v:{fmt}}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    save_to = out_path or os.path.join(ROOT_DIR, "baseline_comparison.png")
    plt.savefig(save_to, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {save_to}")
    plt.close()

def _plot_eval_vs_baselines(hrl_results: list, baseline_results: list, out_path: str = None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        return

    all_r  = baseline_results + hrl_results
    names  = [r["name"] for r in all_r]
    ar     = [r["ar"]   for r in all_r]
    cost   = [r["cost"] for r in all_r]
    x      = np.arange(len(names))
    w      = 0.55
    n_base = len(baseline_results)
    colors = ["#5b8dd9"] * n_base + ["#e05c5c"] * len(hrl_results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("HRL-VGAE vs Baselines", fontsize=14, fontweight="bold")

    for ax, vals, title, ylabel in [
        (axes[0], ar,   "Acceptance Ratio", "Ratio"),
        (axes[1], cost, "Total Cost",       "Cost"),
    ]:
        bars = ax.bar(x, vals, w, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        mx  = max(vals) if max(vals) > 0 else 1
        fmt = ".3f" if title == "Acceptance Ratio" else ".1f"
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + mx * 0.01,
                    f"{v:{fmt}}", ha="center", va="bottom", fontsize=8)

    axes[0].legend(handles=[
        mpatches.Patch(color="#5b8dd9", label="Baseline"),
        mpatches.Patch(color="#e05c5c", label="HRL-VGAE"),
    ], loc="lower right", fontsize=9)

    plt.tight_layout()
    save_to = out_path or os.path.join(ROOT_DIR, "hrl_vs_baselines.png")
    plt.savefig(save_to, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {save_to}")
    plt.close()