"""Generate publication-quality figures for conntopo results.

Creates:
1. Hero combined figure (Kuramoto + Wilson-Cowan)
2. The killer figure: oscillation collapse in Wilson-Cowan
3. Effect size heatmaps
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import FancyArrowPatch

# --- Publication-quality style ---
plt.style.use("default")
mpl.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
    "savefig.facecolor": "#FAFAFA",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.labelweight": "medium",
    "axes.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#CCCCCC",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.8,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# Color palette — colorblind-friendly, high contrast
REAL_COLOR = "#1a1a2e"
NULL_STYLES = {
    "degree_preserving": {"color": "#e63946", "label": "Degree-preserving", "marker": "s", "ls": "-"},
    "erdos_renyi":       {"color": "#457b9d", "label": "Erdős-Rényi", "marker": "D", "ls": "-"},
    "lattice":           {"color": "#2a9d8f", "label": "Lattice", "marker": "^", "ls": "-"},
    "weight_shuffled":   {"color": "#e9c46a", "label": "Weight-shuffled", "marker": "v", "ls": "-"},
    "random_geometric":  {"color": "#f4a261", "label": "Random geometric", "marker": "P", "ls": "-"},
}


def load_results(path: str) -> dict:
    with open(Path(path) / "summary.json") as f:
        return json.load(f)


def _plot_metric(ax, results, metric, ylabel, show_legend=False):
    """Plot one metric: real line + null model lines with markers."""
    couplings = sorted(results.keys(), key=float)
    x = [float(c) for c in couplings]

    # Real connectome — thick, prominent
    real_y = [results[c].get(metric, {}).get("real_mean", np.nan) for c in couplings]
    ax.plot(x, real_y, color=REAL_COLOR, linewidth=3.5, marker="o", markersize=9,
            markerfacecolor="white", markeredgewidth=2.5, markeredgecolor=REAL_COLOR,
            label="Real connectome", zorder=10)

    # Null models — thinner, colored, with distinct markers
    for null_type, style in NULL_STYLES.items():
        null_y = [results[c].get(metric, {}).get(f"{null_type}_mean", np.nan) for c in couplings]
        ax.plot(x, null_y, color=style["color"], linewidth=1.8,
                marker=style["marker"], markersize=6, markerfacecolor=style["color"],
                markeredgewidth=0, alpha=0.85, label=style["label"], ls=style["ls"])

    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.grid(True, axis="y")

    if show_legend:
        ax.legend(loc="best", frameon=True, fancybox=False)


def plot_combined_hero(kuramoto: dict, wilson_cowan: dict,
                       output_path: str = "figures/hero_combined.png"):
    """2-row hero figure: Kuramoto top, Wilson-Cowan bottom."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), gridspec_kw={"hspace": 0.35, "wspace": 0.3})

    # Row labels
    axes[0][0].text(-0.25, 0.5, "Kuramoto\nOscillators", transform=axes[0][0].transAxes,
                    fontsize=14, fontweight="bold", va="center", ha="center",
                    rotation=90, color="#555555")
    axes[1][0].text(-0.25, 0.5, "Wilson-Cowan\nE/I Model", transform=axes[1][0].transAxes,
                    fontsize=14, fontweight="bold", va="center", ha="center",
                    rotation=90, color="#555555")

    # Kuramoto row
    _plot_metric(axes[0][0], kuramoto, "metastability", "Metastability")
    _plot_metric(axes[0][1], kuramoto, "mean_synchrony", "Mean Synchrony")
    _plot_metric(axes[0][2], kuramoto, "regional_differentiation", "Regional Differentiation", show_legend=True)

    # Wilson-Cowan row
    _plot_metric(axes[1][0], wilson_cowan, "global_variance", "Global Variance\n(oscillation strength)")
    _plot_metric(axes[1][1], wilson_cowan, "regional_differentiation", "Regional Differentiation")
    _plot_metric(axes[1][2], wilson_cowan, "mean_fc", "Mean Functional\nConnectivity")

    # X labels only on bottom row
    for ax in axes[1]:
        ax.set_xlabel("Coupling Strength (G)")

    fig.suptitle(
        "Human Brain Connectome Produces Distinct Neural Dynamics",
        fontsize=20, fontweight="bold", y=0.98, color="#1a1a2e",
    )
    fig.text(0.5, 0.935,
             "Robust across two independent dynamics models — TVB 76-region connectome vs. 5 null network types",
             ha="center", fontsize=13, color="#666666")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    fig.savefig(output_path.replace(".png", ".svg"), facecolor=fig.get_facecolor())
    print(f"Saved: {output_path}")
    plt.close()


def plot_wc_oscillation_highlight(wc_results: dict,
                                  output_path: str = "figures/wc_oscillation_highlight.png"):
    """The killer figure: real connectome oscillates, random wiring dies."""
    couplings = sorted(wc_results.keys(), key=float)
    x = [float(c) for c in couplings]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Real connectome — bold, prominent
    real_y = [wc_results[c].get("global_variance", {}).get("real_mean", 0) for c in couplings]
    ax.plot(x, real_y, color=REAL_COLOR, linewidth=4, marker="o", markersize=11,
            markerfacecolor="white", markeredgewidth=3, markeredgecolor=REAL_COLOR,
            label="Real human connectome", zorder=10)

    # Fill under real curve
    ax.fill_between(x, 0, real_y, color=REAL_COLOR, alpha=0.06)

    # Null models
    for null_type, style in NULL_STYLES.items():
        null_y = [wc_results[c].get("global_variance", {}).get(f"{null_type}_mean", 0)
                  for c in couplings]
        ax.plot(x, null_y, color=style["color"], linewidth=2.2,
                marker=style["marker"], markersize=7, markerfacecolor=style["color"],
                markeredgewidth=0, alpha=0.85, label=style["label"])

    ax.set_xlabel("Global Coupling Strength (G)", fontsize=14)
    ax.set_ylabel("Global Variance (oscillation strength)", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, axis="y")

    # Annotations
    real_peak_idx = np.argmax(real_y)
    ax.annotate(
        "Real connectome\nsustains oscillations",
        xy=(x[1], real_y[1]),
        xytext=(x[2] * 2, real_y[0] * 0.85),
        fontsize=13, fontweight="bold", color=REAL_COLOR,
        arrowprops=dict(arrowstyle="-|>", color=REAL_COLOR, lw=2),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=REAL_COLOR, alpha=0.9),
    )

    ax.annotate(
        "Random wiring\ncollapses to silence",
        xy=(x[3], 0.001),
        xytext=(x[3] * 1.5, real_y[0] * 0.4),
        fontsize=12, fontweight="bold", color="#e63946",
        arrowprops=dict(arrowstyle="-|>", color="#e63946", lw=2),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#e63946", alpha=0.9),
    )

    ax.legend(loc="upper right", frameon=True, fancybox=False, fontsize=11)

    ax.set_title(
        "The Real Connectome Sustains Oscillations — Random Wiring Cannot",
        fontsize=16, fontweight="bold", color=REAL_COLOR, pad=15,
    )
    ax.text(0.5, 1.02,
            "Wilson-Cowan E/I dynamics on 76-region human connectome",
            transform=ax.transAxes, ha="center", fontsize=12, color="#666666")

    fig.savefig(output_path, facecolor=fig.get_facecolor())
    fig.savefig(output_path.replace(".png", ".svg"), facecolor=fig.get_facecolor())
    print(f"Saved: {output_path}")
    plt.close()


def plot_effect_heatmap(results: dict, model_name: str,
                        output_path: str = "figures/effect_sizes.png"):
    """Heatmap of Cohen's d at the most informative coupling strength."""
    couplings = sorted(results.keys(), key=float)

    metrics = [m for m in ["metastability", "mean_synchrony",
               "regional_differentiation", "global_variance",
               "freq_diversity", "mean_peak_freq"]
               if any(m in results[c] for c in couplings)]

    null_types = list(NULL_STYLES.keys())
    null_labels = [NULL_STYLES[nt]["label"] for nt in null_types]

    metric_labels = {
        "metastability": "Metastability",
        "mean_synchrony": "Synchrony",
        "regional_differentiation": "Regional Diff.",
        "global_variance": "Global Variance",
        "freq_diversity": "Freq. Diversity",
        "mean_peak_freq": "Peak Frequency",
    }

    # Find best coupling
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

    fig, ax = plt.subplots(figsize=(11, 5.5))

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "custom_rdbu", ["#457b9d", "#a8dadc", "#f1faee", "#e9c46a", "#e63946"], N=256
    )
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=-15, vmax=15)

    ax.set_xticks(range(len(null_types)))
    ax.set_xticklabels(null_labels, fontsize=11)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([metric_labels.get(m, m) for m in metrics], fontsize=12)

    for i in range(len(metrics)):
        for j in range(len(null_types)):
            val = data[i, j]
            color = "white" if abs(val) > 8 else "#1a1a2e"
            text = f"{val:.1f}" if abs(val) < 100 else f"{val:.0f}"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")

    # Gridlines between cells
    for i in range(len(metrics) + 1):
        ax.axhline(i - 0.5, color="white", linewidth=2)
    for j in range(len(null_types) + 1):
        ax.axvline(j - 0.5, color="white", linewidth=2)

    cbar = fig.colorbar(im, ax=ax, label="Cohen's d (effect size)",
                        shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(
        f"Effect Sizes: Real Connectome vs. Null Models — {model_name} (G={best_coupling})",
        fontsize=14, fontweight="bold", color=REAL_COLOR, pad=12,
    )
    ax.text(0.5, 1.02,
            "|d| > 0.8 = large effect  •  values clipped to [-15, 15]",
            transform=ax.transAxes, ha="center", fontsize=11, color="#888888")

    fig.savefig(output_path, facecolor=fig.get_facecolor())
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
