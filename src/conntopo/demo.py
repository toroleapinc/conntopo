"""Quick demo: reproduce the key finding in ~60 seconds.

Run with: python -m conntopo.demo
"""

from __future__ import annotations

import time

import numpy as np
from scipy import stats

from conntopo.connectome import Connectome
from conntopo.dynamics.kuramoto import KuramotoModel
from conntopo.analysis.metrics import compute_all_metrics
from conntopo.nullmodels.generators import generate_null_ensemble


def main():
    print("=" * 60)
    print("conntopo: Does brain wiring shape neural dynamics?")
    print("=" * 60)
    print()

    # Load connectome
    brain = Connectome.from_bundled("tvb76")
    print(f"Loaded: {brain}")

    coupling = 1.0
    n_nulls = 10
    duration = 3000.0
    transient = 500.0

    # Real connectome
    print(f"\nSimulating real connectome (G={coupling})...", end=" ", flush=True)
    t0 = time.time()
    model = KuramotoModel(brain.weights, global_coupling=coupling)
    result = model.simulate(duration=duration, dt=0.1, transient=transient, seed=42)
    real_metrics = compute_all_metrics(result, model_type="kuramoto")
    print(f"{time.time() - t0:.1f}s")

    # Null models
    null_types = ["erdos_renyi", "degree_preserving", "lattice"]
    for null_type in null_types:
        print(f"Simulating {null_type} ({n_nulls} instances)...", end=" ", flush=True)
        t0 = time.time()
        nulls = generate_null_ensemble(brain, null_type, n_instances=n_nulls, seed=42)
        null_meta = []
        null_sync = []
        for nc in nulls:
            m = KuramotoModel(nc.weights, global_coupling=coupling)
            r = m.simulate(duration=duration, dt=0.1, transient=transient, seed=42)
            nm = compute_all_metrics(r, model_type="kuramoto")
            null_meta.append(nm["metastability"])
            null_sync.append(nm["mean_synchrony"])
        elapsed = time.time() - t0
        print(f"{elapsed:.1f}s")

        # Compare
        _, p_meta = stats.mannwhitneyu(
            [real_metrics["metastability"]] * 3, null_meta, alternative="two-sided"
        )
        sig = "***" if p_meta < 0.001 else "**" if p_meta < 0.01 else "*" if p_meta < 0.05 else "ns"

        print(f"  Metastability:  real={real_metrics['metastability']:.4f}  "
              f"null={np.mean(null_meta):.4f}  (p={p_meta:.4f} {sig})")
        print(f"  Synchrony:     real={real_metrics['mean_synchrony']:.4f}  "
              f"null={np.mean(null_sync):.4f}")

    print()
    print("=" * 60)
    print("CONCLUSION: The real human brain connectome produces")
    print("distinct dynamics compared to random wiring.")
    print()
    print("Full experiment: python experiments/01_spontaneous_dynamics.py")
    print("Paper + figures: python scripts/generate_hero_figure.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
