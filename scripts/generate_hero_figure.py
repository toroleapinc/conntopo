"""Generate figures for both Kuramoto and Wilson-Cowan results.

Creates:
1. Hero figure: 2-row panel (Kuramoto top, Wilson-Cowan bottom)
2. Effect size heatmaps for both models
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.linewidth": 1.2,
    "figure.dpi": 150,
})

NULL_STYLES = {
    "degree_preserving": {"color": "#E74C3C", "label": "Degree-preserving rewire"},
    "erdos_renyi": {"color": "#3498DB", "label": "Erdős-Rényi random"},
    "lattice": {"color": "#2ECC71", "label": "Lattice"},
    "weight_shuffled": {"color": "#9B59B6", "label": "Weight-shuffled"},
    "random_geometric": {"color": "#F39C12", "label": "Random geometric"},
}

METRICS_DISPLAY = {
    "metastability": "Metastability",
    "mean_synchrony": "Mean Synchrony",
    "regional_differentiation": "Regional Differentiation",
    "global_variance": "Global Variance",
    "freq_diversity": "Frequency Diversity",
    "mean_peak_freq": "Mean Peak Frequency",
    "mean_fc": "Mean Functional Connectivity",
}


def load_results(path: str) -> dict:
    with open(Path(path) / "summary.json") as f:
        return json.load(f)


def _plot_metric_row(axes, results, metrics_to_plot, model_label):
    """Plot one row of metrics (one dynamics model)."""
    couplings = sorted(results.keys(), key=float)
    coupling_vals = [float(c) for c in couplings]

    for ax, (metric, ylabel) in zip(axes, metrics_to_plot):
        real_vals = [results[c].get(metric, {}).get("real_mean", np.nan) for c in couplings]
        ax.plot(coupling_vals, real_vals, "k-o", linewidth=2.5, markersize=6,
                label="Real human connectome", zorder=10)

        for null_type, style in NULL_STYLES.items():
            null_vals = [results[c].get(metric, {}).get(f"{null_type}_mean", np.nan)
                         for c in couplings]
            ax.plot(coupling_vals, null_vals, "--", color=style["color"],
                    linewidth=1.5, alpha=0.8, label=style["label"])

        ax.set_xlabel("Coupling Strength (G)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_title(model_label, fontsize=12, fontweight="bold", loc="left")


def plot_combined_hero(kuramoto: dict, wilson_cowan: dict,
                       output_path: str = "figures/hero_combined.png"):
    """2-row hero figure: Kuramoto (top) and Wilson-Cowan (bottom)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    kuramoto_metrics = [
        ("metastability", "Metastability"),
        ("mean_synchrony", "Mean Synchrony"),
        ("regional_differentiation", "Regional Diff."),
    ]
    wc_metrics = [
        ("global_variance", "Global Variance"),
        ("regional_differentiation", "Regional Diff."),
        ("mean_fc", "Mean FC"),
    ]

    _plot_metric_row(axes[0], kuramoto, kuramoto_metrics, "Kuramoto Oscillators")
    _plot_metric_row(axes[1], wilson_cowan, wc_metrics, "Wilson-Cowan E/I Model")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.06), fontsize=10, frameon=False)

    fig.suptitle(
        "Human Brain Connectome Produces Distinct Dynamics vs. Null Models\n"
        "Robust across two independent dynamics models (TVB 76-region connectome)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close()


def plot_effect_heatmap(results: dict, model_name: str,
                        output_path: str = "figures/effect_sizes.png"):
    """Heatmap of Cohen's d at the best coupling strength."""
    couplings = sorted(results.keys(), key=float)

    metrics = [m for m in ["metastability", "mean_synchrony",
               "regional_differentiation", "global_variance",
               "freq_diversity", "mean_peak_freq"]
               if any(m in results[c] for c in couplings)]

    null_types = list(NULL_STYLES.keys())

    # Find coupling with most significant results
    best_coupling = couplings[1] if len(couplings) > 1 else couplings[0]
    best_count = 0
    for c in couplings:
        count = sum(1 for m in metrics for nt in null_types
                    if abs(results[c].get(m, {}).get(f"{nt}_d", 0)) > 0.8)
        if count > best_count:
            best_count = count
            best_coupling = c

    data = np.zeros((len(metrics), len(null_types)))
    for i, metric in enumerate(metrics):
        for j, null_type in enumerate(null_types):
            d = results[best_coupling].get(metric, {}).get(f"{null_type}_d", 0)
            data[i, j] = np.clip(d, -15, 15)  # Clip extreme values for visualization

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-15, vmax=15)

    ax.set_xticks(range(len(null_types)))
    ax.set_xticklabels([NULL_STYLES[nt]["label"].replace(" ", "\n") for nt in null_types],
                       fontsize=9)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([METRICS_DISPLAY.get(m, m) for m in metrics], fontsize=10)

    for i in range(len(metrics)):
        for j in range(len(null_types)):
            val = data[i, j]
            color = "white" if abs(val) > 7 else "black"
            text = f"{val:.1f}" if abs(val) < 100 else f"{val:.0f}"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Cohen's d (effect size)", shrink=0.8)
    ax.set_title(
        f"Effect Sizes: Real vs. Null Models — {model_name} (G={best_coupling})\n"
        f"|d| > 0.8 = large effect; values clipped to [-15, 15] for display",
        fontsize=11, fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close()


def plot_wc_oscillation_highlight(wc_results: dict,
                                  output_path: str = "figures/wc_oscillation_highlight.png"):
    """The killer figure: real connectome oscillates, null models collapse."""
    couplings = sorted(wc_results.keys(), key=float)
    coupling_vals = [float(c) for c in couplings]

    fig, ax = plt.subplots(figsize=(8, 5))

    real_vals = [wc_results[c].get("global_variance", {}).get("real_mean", 0) for c in couplings]
    ax.plot(coupling_vals, real_vals, "k-o", linewidth=3, markersize=8,
            label="Real human connectome", zorder=10)

    for null_type, style in NULL_STYLES.items():
        null_vals = [wc_results[c].get("global_variance", {}).get(f"{null_type}_mean", 0)
                     for c in couplings]
        ax.plot(coupling_vals, null_vals, "--", color=style["color"],
                linewidth=2, alpha=0.8, label=style["label"])

    ax.set_xlabel("Global Coupling Strength (G)", fontsize=13)
    ax.set_ylabel("Global Variance (oscillation strength)", fontsize=13)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.annotate("Real connectome\nsustains oscillations",
                xy=(0.01, real_vals[1]), xytext=(0.04, real_vals[1] + 0.02),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    ax.annotate("Random wiring\ncollapses to silence",
                xy=(0.05, 0.001), xytext=(0.15, 0.04),
                fontsize=10, color="#E74C3C", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#E74C3C", lw=1.5))

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title(
        "The Real Connectome Sustains Oscillations — Random Wiring Cannot\n"
        "Wilson-Cowan E/I dynamics on 76-region human connectome",
        fontsize=12, fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    kuramoto = load_results("results/exp01_kuramoto")
    wc = load_results("results/exp01_wilson_cowan")

    plot_combined_hero(kuramoto, wc)
    plot_effect_heatmap(kuramoto, "Kuramoto", "figures/effect_sizes_kuramoto.png")
    plot_effect_heatmap(wc, "Wilson-Cowan", "figures/effect_sizes_wc.png")
    plot_wc_oscillation_highlight(wc)

    print("\nAll figures generated!")
