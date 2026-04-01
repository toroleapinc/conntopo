"""Find the Wilson-Cowan oscillatory regime.

Sweep E/I balance parameters and global coupling to find parameter
combinations where the model produces sustained oscillations (not
fixed points). Then we can re-run the topology comparison in that regime.
"""

from __future__ import annotations

import numpy as np

from conntopo.connectome import Connectome
from conntopo.dynamics.wilson_cowan import WilsonCowanModel, WilsonCowanParams


def sweep():
    brain = Connectome.from_bundled("tvb76")

    # Key insight: oscillations in Wilson-Cowan emerge when E/I balance
    # is near a Hopf bifurcation. We need to find that regime.
    #
    # Strategy: vary w_ee (excitatory self-coupling) and global_coupling G
    # while keeping other parameters at standard values.

    print("Wilson-Cowan Parameter Sweep: Finding Oscillatory Regime")
    print("=" * 70)
    print(f"{'w_ee':>6} {'w_ei':>6} {'w_ie':>6} {'G':>6} | {'mean_var':>10} {'max_var':>10} {'oscillates':>10}")
    print("-" * 70)

    best_params = None
    best_variance = 0

    # Sweep w_ee and G — these are the two most important parameters
    for w_ee in [10.0, 12.0, 14.0, 16.0, 18.0]:
        for w_ei in [4.0, 6.0, 8.0, 10.0]:
            for G in [0.01, 0.05, 0.1, 0.5, 1.0]:
                params = WilsonCowanParams(
                    w_ee=w_ee, w_ei=w_ei, w_ie=13.0, w_ii=11.0,
                    theta_e=4.0, theta_i=3.7, a_e=1.2, a_i=1.0,
                    noise_sigma=0.02,
                )
                model = WilsonCowanModel(
                    brain.weights, global_coupling=G, params=params,
                )
                result = model.simulate(
                    duration=2000, dt=0.1, transient=500, seed=42,
                )

                E = result["E"]
                var_per_region = np.var(E, axis=0)
                mean_var = float(np.mean(var_per_region))
                max_var = float(np.max(var_per_region))
                oscillates = mean_var > 0.001

                if oscillates:
                    marker = "*** YES ***"
                    if mean_var > best_variance:
                        best_variance = mean_var
                        best_params = (w_ee, w_ei, G, params)
                else:
                    marker = ""

                print(f"{w_ee:>6.1f} {w_ei:>6.1f} {13.0:>6.1f} {G:>6.2f} | "
                      f"{mean_var:>10.6f} {max_var:>10.6f} {marker:>10}")

    print()
    if best_params:
        w_ee, w_ei, G, params = best_params
        print(f"BEST OSCILLATORY REGIME FOUND:")
        print(f"  w_ee={w_ee}, w_ei={w_ei}, w_ie=13.0, G={G}")
        print(f"  Mean variance: {best_variance:.6f}")

        # Now do a finer sweep around the best point
        print(f"\nFine-tuning around best point...")
        for G_fine in [G * 0.5, G * 0.75, G, G * 1.5, G * 2.0]:
            for noise in [0.01, 0.02, 0.05]:
                params_fine = WilsonCowanParams(
                    w_ee=w_ee, w_ei=w_ei, w_ie=13.0, w_ii=11.0,
                    theta_e=4.0, theta_i=3.7, a_e=1.2, a_i=1.0,
                    noise_sigma=noise,
                )
                model = WilsonCowanModel(
                    brain.weights, global_coupling=G_fine, params=params_fine,
                )
                result = model.simulate(
                    duration=3000, dt=0.1, transient=500, seed=42,
                )
                E = result["E"]
                mean_var = float(np.mean(np.var(E, axis=0)))
                print(f"  G={G_fine:.4f} noise={noise:.2f} → var={mean_var:.6f}")
    else:
        print("NO OSCILLATORY REGIME FOUND with these parameters.")
        print("Try wider parameter ranges or different parameter combinations.")
        print()
        print("Alternative: try reducing thresholds or increasing gain:")

        for theta_e in [2.0, 3.0, 4.0]:
            for a_e in [1.0, 1.5, 2.0]:
                for G in [0.01, 0.05, 0.1, 0.5]:
                    params = WilsonCowanParams(
                        w_ee=16.0, w_ei=4.0, w_ie=13.0, w_ii=11.0,
                        theta_e=theta_e, theta_i=3.7, a_e=a_e, a_i=1.0,
                        noise_sigma=0.02,
                    )
                    model = WilsonCowanModel(
                        brain.weights, global_coupling=G, params=params,
                    )
                    result = model.simulate(
                        duration=2000, dt=0.1, transient=500, seed=42,
                    )
                    E = result["E"]
                    mean_var = float(np.mean(np.var(E, axis=0)))
                    if mean_var > 0.001:
                        print(f"  FOUND: theta_e={theta_e} a_e={a_e} G={G} → var={mean_var:.6f}")


if __name__ == "__main__":
    sweep()
