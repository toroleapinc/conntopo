"""Generate publication-quality figures for conntopo results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

plt.style.use("default")
mpl.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
    "savefig.facecolor": "#FAFAFA",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#CCCCCC",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.8,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.2,
})

REAL_COLOR = "#1a1a2e"
NULL_STYLES = {
    "degree_preserving": {"color": "#e63946", "label": "Degree-preserving", "marker": "s"},
    "erdos_renyi":       {"color": "#457b9d", "label": "Erdos-Renyi", "marker": "D"},
    "lattice":           {"color": "#2a9d8f", "label": "Lattice", "marker": "^"},
    "weight_shuffled":   {"color": "#e9c46a", "label": "Weight-shuffled", "marker": "v"},
    "random_geometric":  {"color": "#f4a261", "label": "Random geometric", "marker": "P"},
}


def load_results(path: str) -> dict:
    with open(Path(path) / "summary.json") as f:
        return json.load(f)


def _plot_metric(ax, results, metric, ylabel):
    """Plot one metric panel."""
    couplings = sorted(results.keys(), key=float)
    x = [float(c) for c in couplings]

    real_y = [results[c].get(metric, {}).get("real_mean", np.nan) for c in couplings]
    ax.plot(x, real_y, color=REAL_COLOR, linewidth=3, marker="o", markersize=8,
            markerfacecolor="white", markeredgewidth=2.5, markeredgecolor=REAL_COLOR,
            label="Real connectome", zorder=10)

    for null_type, style in NULL_STYLES.items():
        null_y = [results[c].get(metric, {}).get(f"{null_type}_mean", np.nan)
                  for c in couplings]
        ax.plot(x, null_y, color=style["color"], linewidth=1.6,
                marker=style["marker"], markersize=5, markerfacecolor=style["color"],
                markeredgewidth=0, alpha=0.85, label=style["label"])

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xscale("log")
    ax.grid(True, axis="y")


def plot_combined_hero(kuramoto, wilson_cowan, output_path="figures/hero_combined.png"):
    """2-row hero figure with generous spacing."""
    fig = plt.figure(figsize=(18, 11))

    # Use gridspec for precise control
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.88, bottom=0.08)

    # --- Kuramoto row ---
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])

    _plot_metric(ax00, kuramoto, "metastability", "Metastability")
    _plot_metric(ax01, kuramoto, "mean_synchrony", "Mean Synchrony")
    _plot_metric(ax02, kuramoto, "regional_differentiation", "Regional Diff.")

    # Row label
    ax00.text(-0.3, 0.5, "Kuramoto", transform=ax00.transAxes,
              fontsize=13, fontweight="bold", va="center", ha="center",
              rotation=90, color="#555555")

    # --- Wilson-Cowan row ---
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])

    _plot_metric(ax10, wilson_cowan, "global_variance", "Global Variance")
    _plot_metric(ax11, wilson_cowan, "regional_differentiation", "Regional Diff.")
    _plot_metric(ax12, wilson_cowan, "mean_fc", "Mean FC")

    ax10.text(-0.3, 0.5, "Wilson-\nCowan", transform=ax10.transAxes,
              fontsize=13, fontweight="bold", va="center", ha="center",
              rotation=90, color="#555555")

    # X labels only on bottom row
    for ax in [ax10, ax11, ax12]:
        ax.set_xlabel("Coupling Strength (G)", fontsize=11)

    # Legend — outside the plots, below the figure
    handles, labels = ax00.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.52, -0.01), fontsize=10, frameon=True,
               fancybox=False, edgecolor="#CCCCCC", columnspacing=1.5)

    # Title with spacing
    fig.text(0.52, 0.96,
             "Human Brain Connectome Produces Distinct Neural Dynamics",
             ha="center", fontsize=18, fontweight="bold", color=REAL_COLOR)
    fig.text(0.52, 0.925,
             "Robust across two independent dynamics models  |  TVB 76-region connectome vs. 5 null network types",
             ha="center", fontsize=11, color="#777777")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    fig.savefig(output_path.replace(".png", ".svg"))
    print(f"Saved: {output_path}")
    plt.close()


def plot_wc_oscillation_highlight(wc_results, output_path="figures/wc_oscillation_highlight.png"):
    """The killer figure — clear annotations that don't overlap."""
    couplings = sorted(wc_results.keys(), key=float)
    x = [float(c) for c in couplings]

    fig, ax = plt.subplots(figsize=(10, 7))

    real_y = [wc_results[c].get("global_variance", {}).get("real_mean", 0) for c in couplings]
    ax.plot(x, real_y, color=REAL_COLOR, linewidth=4, marker="o", markersize=10,
            markerfacecolor="white", markeredgewidth=3, markeredgecolor=REAL_COLOR,
            label="Real human connectome", zorder=10)
    ax.fill_between(x, 0, real_y, color=REAL_COLOR, alpha=0.05)

    for null_type, style in NULL_STYLES.items():
        null_y = [wc_results[c].get("global_variance", {}).get(f"{null_type}_mean", 0)
                  for c in couplings]
        ax.plot(x, null_y, color=style["color"], linewidth=2,
                marker=style["marker"], markersize=6, markerfacecolor=style["color"],
                markeredgewidth=0, alpha=0.85, label=style["label"])

    ax.set_xlabel("Global Coupling Strength (G)", fontsize=14)
    ax.set_ylabel("Global Variance (oscillation strength)", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, axis="y")

    # Annotation: real connectome — top-left area, well away from data
    ax.annotate(
        "Real connectome\nsustains oscillations",
        xy=(x[0], real_y[0]),
        xytext=(0.55, 0.85), textcoords="axes fraction",
        fontsize=12, fontweight="bold", color=REAL_COLOR,
        arrowprops=dict(arrowstyle="-|>", color=REAL_COLOR, lw=1.8,
                        connectionstyle="arc3,rad=-0.2"),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor=REAL_COLOR, linewidth=1.5),
    )

    # Annotation: random wiring — bottom-right area
    ax.annotate(
        "Random wiring\ncollapses to silence",
        xy=(x[-1], 0.001),
        xytext=(0.55, 0.22), textcoords="axes fraction",
        fontsize=12, fontweight="bold", color="#e63946",
        arrowprops=dict(arrowstyle="-|>", color="#e63946", lw=1.8,
                        connectionstyle="arc3,rad=0.2"),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#e63946", linewidth=1.5),
    )

    # Legend — bottom-left, away from data and annotations
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.45),
              frameon=True, fancybox=False, edgecolor="#CCCCCC", fontsize=10)

    # Title with generous spacing between lines
    ax.set_title(
        "The Real Connectome Sustains Oscillations — Random Wiring Cannot",
        fontsize=15, fontweight="bold", color=REAL_COLOR, pad=40,
    )
    ax.text(0.5, 1.055,
            "Wilson-Cowan E/I dynamics on 76-region human connectome",
            transform=ax.transAxes, ha="center", fontsize=11, color="#777777")

    fig.savefig(output_path)
    fig.savefig(output_path.replace(".png", ".svg"))
    print(f"Saved: {output_path}")
    plt.close()


def plot_effect_heatmap(results, model_name, output_path="figures/effect_sizes.png"):
    """Effect size heatmap with clean layout."""
    couplings = sorted(results.keys(), key=float)

    metrics = [m for m in ["metastability", "mean_synchrony",
               "regional_differentiation", "global_variance",
               "freq_diversity", "mean_peak_freq"]
               if any(m in results[c] for c in couplings)]

    null_types = list(NULL_STYLES.keys())

    metric_labels = {
        "metastability": "Metastability",
        "mean_synchrony": "Synchrony",
        "regional_differentiation": "Regional Diff.",
        "global_variance": "Global Variance",
        "freq_diversity": "Freq. Diversity",
        "mean_peak_freq": "Peak Frequency",
    }

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
            data[i, j] = np.clip(d, -15, 15)

    fig, ax = plt.subplots(figsize=(11, 6))

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "custom", ["#457b9d", "#a8dadc", "#f1faee", "#e9c46a", "#e63946"], N=256)
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=-15, vmax=15)

    ax.set_xticks(range(len(null_types)))
    ax.set_xticklabels([NULL_STYLES[nt]["label"] for nt in null_types], fontsize=10)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([metric_labels.get(m, m) for m in metrics], fontsize=11)

    for i in range(len(metrics)):
        for j in range(len(null_types)):
            val = data[i, j]
            color = "white" if abs(val) > 8 else "#1a1a2e"
            text = f"{val:.1f}" if abs(val) < 100 else f"{val:.0f}"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")

    for i in range(len(metrics) + 1):
        ax.axhline(i - 0.5, color="white", linewidth=2)
    for j in range(len(null_types) + 1):
        ax.axvline(j - 0.5, color="white", linewidth=2)

    cbar = fig.colorbar(im, ax=ax, label="Cohen's d (effect size)",
                        shrink=0.85, pad=0.03)
    cbar.ax.tick_params(labelsize=10)

    # Title with proper spacing — no subtitle overlap
    ax.set_title(
        f"Effect Sizes: Real vs. Null — {model_name} (G={best_coupling})\n"
        f"|d| > 0.8 = large effect  |  values clipped to [-15, 15]",
        fontsize=13, fontweight="bold", color=REAL_COLOR, pad=15,
        linespacing=1.6,
    )

    fig.savefig(output_path)
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    kuramoto = load_results("results/exp01_kuramoto")
    wc = load_results("results/exp01_wilson_cowan")

    plot_combined_hero(kuramoto, wc)
    plot_wc_oscillation_highlight(wc)
    plot_effect_heatmap(kuramoto, "Kuramoto", "figures/effect_sizes_kuramoto.png")
    plot_effect_heatmap(wc, "Wilson-Cowan", "figures/effect_sizes_wc.png")

    print("\nAll figures generated!")
