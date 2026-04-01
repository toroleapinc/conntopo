"""Kuramoto phase oscillator model.

Each brain region is a phase oscillator with a natural frequency.
Coupling through the connectivity matrix determines synchronization dynamics.

References:
    Kuramoto (1984). Chemical Oscillations, Waves, and Turbulence.
"""

from __future__ import annotations

import numpy as np


class KuramotoModel:
    """Kuramoto oscillators on a connectivity graph.

    Each region i has a phase theta_i evolving as:
        dtheta_i/dt = omega_i + G * sum_j C_ij * sin(theta_j - theta_i)

    where omega_i are natural frequencies and C_ij is connectivity.
    """

    def __init__(
        self,
        connectivity: np.ndarray,
        global_coupling: float = 1.0,
        freq_mean: float = 10.0,
        freq_std: float = 2.0,
    ):
        if connectivity.ndim != 2 or connectivity.shape[0] != connectivity.shape[1]:
            raise ValueError(f"connectivity must be square 2D, got {connectivity.shape}")
        self.C = connectivity.astype(np.float64)
        self.n = connectivity.shape[0]
        self.G = global_coupling
        self.freq_mean = freq_mean  # Hz
        self.freq_std = freq_std    # Hz

    def simulate(
        self,
        duration: float = 10000.0,
        dt: float = 0.1,
        transient: float = 2000.0,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Run the simulation.

        Args:
            duration: Total simulation time (ms), including transient.
            dt: Integration timestep (ms).
            transient: Initial transient to discard (ms).
            seed: Random seed for reproducibility.

        Returns:
            Dict with keys:
                'theta': Phase angles [timesteps, regions]
                'order_param': Kuramoto order parameter R(t) [timesteps]
                'time': Time vector [timesteps]
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        total_steps = int(duration / dt)
        transient_steps = int(transient / dt)
        record_steps = total_steps - transient_steps

        if record_steps <= 0:
            raise ValueError(
                f"duration ({duration}) must be greater than transient ({transient})"
            )

        n = self.n
        C = self.C * self.G

        # Natural frequencies (Hz → rad/ms)
        omega = rng.normal(self.freq_mean, self.freq_std, size=n)
        omega = omega * 2 * np.pi / 1000.0  # Convert Hz to rad/ms

        # Initialize phases uniformly
        theta = rng.uniform(0, 2 * np.pi, size=n)

        # Pre-allocate
        theta_record = np.zeros((record_steps, n), dtype=np.float64)
        order_record = np.zeros(record_steps, dtype=np.float64)

        record_idx = 0

        for step in range(total_steps):
            # Phase differences: sin(theta_j - theta_i) for all pairs
            diffs = np.sin(theta[np.newaxis, :] - theta[:, np.newaxis])

            # Coupling: sum_j C_ij * sin(theta_j - theta_i)
            coupling = np.sum(C * diffs, axis=1)

            # Euler integration
            dtheta = omega + coupling
            theta = theta + dt * dtheta

            # Wrap to [0, 2*pi]
            theta = theta % (2 * np.pi)

            # Record after transient
            if step >= transient_steps:
                theta_record[record_idx] = theta

                # Kuramoto order parameter R(t)
                complex_phase = np.exp(1j * theta)
                order_record[record_idx] = np.abs(np.mean(complex_phase))

                record_idx += 1

        time = np.arange(record_steps) * dt

        return {
            "theta": theta_record,
            "order_param": order_record,
            "time": time,
        }

    @staticmethod
    def metastability(order_param: np.ndarray) -> float:
        """Compute metastability as std of the order parameter over time."""
        return float(np.std(order_param))

    @staticmethod
    def mean_synchrony(order_param: np.ndarray) -> float:
        """Compute mean synchrony (mean of order parameter)."""
        return float(np.mean(order_param))
