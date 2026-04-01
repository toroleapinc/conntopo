"""Experiment 1: Spontaneous Dynamics — Real Connectome vs Null Models

The core scientific question: Does the human brain's macro-connectome topology
produce qualitatively different dynamics compared to null network models?

Protocol:
1. Load real human connectome (TVB 76 regions)
2. Generate 5 types of null models (20 instances each for speed)
3. Run Kuramoto oscillators on each network across coupling strengths
4. Compute metrics: metastability, synchrony, regional differentiation,
   spectral entropy, functional connectivity
5. Statistical comparison: real connectome vs each null model class
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from conntopo.connectome import Connectome
from conntopo.dynamics.kuramoto import KuramotoModel
from conntopo.dynamics.wilson_cowan import WilsonCowanModel
from conntopo.nullmodels.generators import generate_null_ensemble
from conntopo.analysis.metrics import compute_all_metrics


# --- Configuration ---

N_NULL_INSTANCES = 20     # Number of null model instances per type
COUPLING_VALUES_KURAMOTO = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
COUPLING_VALUES_WC = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5]
DURATION = 5000.0         # ms
DT = 0.1                  # ms
TRANSIENT = 1000.0        # ms
SEED = 42

NULL_TYPES = [
    "degree_preserving",
    "erdos_renyi",
    "lattice",
    "weight_shuffled",
]
# random_geometric requires positions — add separately

RESULTS_DIR = Path("results/exp01")  # Will be updated with model_type suffix


def run_single_simulation(
    connectome: Connectome,
    coupling: float,
    model_type: str = "kuramoto",
    seed: int = 42,
) -> dict:
    """Run one simulation and return scalar metrics."""
    if model_type == "kuramoto":
        model = KuramotoModel(
            connectome.weights, global_coupling=coupling,
            freq_mean=10.0, freq_std=2.0,
        )
        result = model.simulate(
            duration=DURATION, dt=DT, transient=TRANSIENT, seed=seed,
        )
        metrics = compute_all_metrics(result, dt=DT, model_type="kuramoto")
    elif model_type == "wilson_cowan":
        model = WilsonCowanModel(
            connectome.weights, global_coupling=coupling,
        )
        result = model.simulate(
            duration=DURATION, dt=DT, transient=TRANSIENT, seed=seed,
        )
        metrics = compute_all_metrics(result, dt=DT, model_type="wilson_cowan")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Return only scalar metrics for comparison
    scalars = {}
    for k, v in metrics.items():
        if isinstance(v, (float, int)):
            scalars[k] = v
    return scalars


def run_experiment(model_type: str = "kuramoto"):
    """Run the full experiment."""
    print("=" * 70)
    print(f"EXPERIMENT 1: SPONTANEOUS DYNAMICS ({model_type.upper()})")
    print("Does human brain connectome topology shape neural dynamics?")
    print("=" * 70)

    # Load real connectome
    connectome = Connectome.from_bundled("tvb76")
    print(f"\nReal connectome: {connectome}")

    # Add random_geometric if positions available
    null_types = list(NULL_TYPES)
    if connectome.positions is not None:
        null_types.append("random_geometric")

    # Generate null models
    print(f"\nGenerating null models ({N_NULL_INSTANCES} instances each)...")
    null_ensembles: dict[str, list[Connectome]] = {}
    for null_type in null_types:
        t0 = time.time()
        null_ensembles[null_type] = generate_null_ensemble(
            connectome, null_type, n_instances=N_NULL_INSTANCES, seed=SEED,
        )
        elapsed = time.time() - t0
        print(f"  {null_type}: {N_NULL_INSTANCES} instances in {elapsed:.1f}s")

    # Select coupling values based on model type
    coupling_values = COUPLING_VALUES_WC if model_type == "wilson_cowan" else COUPLING_VALUES_KURAMOTO

    # Run simulations
    all_results: dict[str, dict] = {}  # coupling -> {network_type -> metrics_list}

    for coupling in coupling_values:
        print(f"\n{'─' * 70}")
        print(f"Coupling G = {coupling}")
        print(f"{'─' * 70}")

        coupling_results: dict[str, list[dict]] = {}

        # Real connectome (run multiple seeds for variability)
        print(f"  Running real connectome...", end=" ", flush=True)
        t0 = time.time()
        real_metrics = []
        for trial in range(5):
            m = run_single_simulation(
                connectome, coupling, model_type=model_type, seed=SEED + trial,
            )
            real_metrics.append(m)
        elapsed = time.time() - t0
        coupling_results["real"] = real_metrics
        print(f"{elapsed:.1f}s")

        # Null models
        for null_type in null_types:
            print(f"  Running {null_type}...", end=" ", flush=True)
            t0 = time.time()
            null_metrics = []
            for i, null_conn in enumerate(null_ensembles[null_type]):
                m = run_single_simulation(
                    null_conn, coupling, model_type=model_type, seed=SEED + i,
                )
                null_metrics.append(m)
            elapsed = time.time() - t0
            coupling_results[null_type] = null_metrics
            print(f"{elapsed:.1f}s")

        all_results[str(coupling)] = coupling_results

    # --- Statistical Analysis ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Get all metric names from first result
    metric_names = list(all_results[str(coupling_values[0])]["real"][0].keys())

    # For each coupling strength and metric, compare real vs each null type
    summary = {}

    for coupling in coupling_values:
        coupling_key = str(coupling)
        real_data = all_results[coupling_key]["real"]

        print(f"\n{'─' * 70}")
        print(f"Coupling G = {coupling}")
        print(f"{'─' * 70}")
        print(f"{'Metric':<30} {'Real':>8} | ", end="")
        for nt in null_types:
            print(f"{nt[:12]:>12}", end=" ")
        print()
        print(f"{'':<30} {'':>8} | ", end="")
        for nt in null_types:
            print(f"{'(p-value)':>12}", end=" ")
        print()
        print("─" * (40 + 13 * len(null_types)))

        coupling_summary = {}

        for metric in metric_names:
            real_values = [d[metric] for d in real_data]
            real_mean = np.mean(real_values)

            row = f"{metric:<30} {real_mean:>8.4f} | "
            metric_summary = {"real_mean": real_mean}

            for null_type in null_types:
                null_data = all_results[coupling_key][null_type]
                null_values = [d[metric] for d in null_data]
                null_mean = np.mean(null_values)

                # Permutation-style p-value: fraction of null instances
                # with metric as extreme as real mean
                if len(null_values) > 1:
                    # Mann-Whitney U test (non-parametric)
                    if np.std(null_values) > 0 and np.std(real_values) > 0:
                        stat, p_value = stats.mannwhitneyu(
                            real_values, null_values, alternative="two-sided"
                        )
                    else:
                        p_value = 1.0

                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        (np.var(real_values) + np.var(null_values)) / 2
                    )
                    if pooled_std > 0:
                        cohens_d = (real_mean - null_mean) / pooled_std
                    else:
                        cohens_d = 0.0
                else:
                    p_value = 1.0
                    cohens_d = 0.0

                # Format with significance markers
                sig = ""
                if p_value < 0.001:
                    sig = "***"
                elif p_value < 0.01:
                    sig = "** "
                elif p_value < 0.05:
                    sig = "*  "

                row += f"{null_mean:>8.4f}{sig:>4}"

                metric_summary[f"{null_type}_mean"] = null_mean
                metric_summary[f"{null_type}_p"] = p_value
                metric_summary[f"{null_type}_d"] = cohens_d

            print(row)
            coupling_summary[metric] = metric_summary

        summary[coupling_key] = coupling_summary

    # --- Key Findings Summary ---
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    sig_count = 0
    total_tests = 0

    for coupling in coupling_values:
        coupling_key = str(coupling)
        for metric in metric_names:
            for null_type in null_types:
                p = summary[coupling_key][metric].get(f"{null_type}_p", 1.0)
                d = summary[coupling_key][metric].get(f"{null_type}_d", 0.0)
                total_tests += 1
                if p < 0.05:
                    sig_count += 1
                    if abs(d) > 0.8:  # Large effect
                        real_m = summary[coupling_key][metric]["real_mean"]
                        null_m = summary[coupling_key][metric][f"{null_type}_mean"]
                        print(
                            f"  G={coupling:>5}, {metric:<28} "
                            f"real={real_m:.4f} vs {null_type}={null_m:.4f} "
                            f"(d={d:+.2f}, p={p:.4f})"
                        )

    print(f"\n  Total significant comparisons (p<0.05): "
          f"{sig_count}/{total_tests}")
    print(f"  Comparisons with large effect (|d|>0.8): listed above")

    # Save results
    results_dir = Path(f"results/exp01_{model_type}")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "summary.json"

    # Convert numpy types for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = json.loads(
        json.dumps(summary, default=to_serializable)
    )
    with open(results_file, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to {results_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="kuramoto", choices=["kuramoto", "wilson_cowan"])
    parser.add_argument("--null-instances", type=int, default=20)
    args = parser.parse_args()
    N_NULL_INSTANCES = args.null_instances
    run_experiment(model_type=args.model)
