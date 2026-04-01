"""Dynamics analysis metrics for comparing connectome topologies.

Quantifies oscillation properties, synchrony, functional connectivity,
and regional differentiation from simulation time series.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def power_spectral_density(
    timeseries: np.ndarray, dt: float = 0.1, nperseg: int = 1024
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PSD using Welch's method.

    Args:
        timeseries: [timesteps, regions]
        dt: Timestep in ms.
        nperseg: Segment length for Welch.

    Returns:
        freqs: Frequency vector (Hz).
        psd: PSD matrix [freqs, regions].
    """
    fs = 1000.0 / dt  # Convert ms timestep to Hz sampling rate
    nperseg = min(nperseg, timeseries.shape[0])
    freqs, psd = signal.welch(timeseries, fs=fs, nperseg=nperseg, axis=0)
    return freqs, psd


def peak_frequency(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Find peak frequency for each region.

    Args:
        freqs: [n_freqs]
        psd: [n_freqs, regions]

    Returns:
        Peak frequency per region [regions].
    """
    peak_indices = np.argmax(psd, axis=0)
    return freqs[peak_indices]


def spectral_entropy(psd: np.ndarray) -> np.ndarray:
    """Compute spectral entropy for each region (diversity of frequencies).

    Higher entropy = broader frequency content.

    Returns:
        Entropy per region [regions].
    """
    # Normalize PSD to probability distribution per region
    psd_norm = psd / (psd.sum(axis=0, keepdims=True) + 1e-12)
    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12), axis=0)
    return entropy


def functional_connectivity(timeseries: np.ndarray) -> np.ndarray:
    """Compute functional connectivity as Pearson correlation matrix.

    Args:
        timeseries: [timesteps, regions]

    Returns:
        FC matrix [regions, regions].
    """
    return np.corrcoef(timeseries.T)


def metastability(order_param: np.ndarray) -> float:
    """Metastability = standard deviation of the Kuramoto order parameter."""
    return float(np.std(order_param))


def regional_differentiation(timeseries: np.ndarray) -> float:
    """How different are regions from each other?

    Computed as the mean pairwise distance between regional time series
    (1 - correlation), then averaged.
    """
    fc = functional_connectivity(timeseries)
    n = fc.shape[0]
    # Upper triangle, excluding diagonal
    mask = np.triu_indices(n, k=1)
    distances = 1.0 - fc[mask]
    return float(np.mean(distances))


def mean_activity(timeseries: np.ndarray) -> np.ndarray:
    """Mean activity per region [regions]."""
    return np.mean(timeseries, axis=0)


def activity_variance(timeseries: np.ndarray) -> np.ndarray:
    """Activity variance per region [regions]."""
    return np.var(timeseries, axis=0)


def compute_all_metrics(
    sim_result: dict[str, np.ndarray],
    dt: float = 0.1,
    model_type: str = "wilson_cowan",
) -> dict[str, float | np.ndarray]:
    """Compute all metrics from a simulation result.

    Args:
        sim_result: Output from WilsonCowanModel.simulate() or KuramotoModel.simulate().
        dt: Timestep used in simulation.
        model_type: 'wilson_cowan' or 'kuramoto'.

    Returns:
        Dict of metric name -> value.
    """
    metrics = {}

    if model_type == "wilson_cowan":
        ts = sim_result["E"]  # Use excitatory activity
    elif model_type == "kuramoto":
        ts = np.cos(sim_result["theta"])  # Convert phase to amplitude
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # PSD-based metrics
    freqs, psd = power_spectral_density(ts, dt)
    metrics["peak_freqs"] = peak_frequency(freqs, psd)
    metrics["spectral_entropy"] = spectral_entropy(psd)
    metrics["mean_peak_freq"] = float(np.mean(metrics["peak_freqs"]))
    metrics["freq_diversity"] = float(np.std(metrics["peak_freqs"]))

    # FC-based metrics
    fc = functional_connectivity(ts)
    metrics["fc_matrix"] = fc
    metrics["regional_differentiation"] = regional_differentiation(ts)
    metrics["mean_fc"] = float(np.mean(fc[np.triu_indices(fc.shape[0], k=1)]))

    # Activity metrics
    metrics["mean_activity"] = mean_activity(ts)
    metrics["activity_variance"] = activity_variance(ts)
    metrics["global_variance"] = float(np.mean(metrics["activity_variance"]))

    # Kuramoto-specific
    if model_type == "kuramoto" and "order_param" in sim_result:
        metrics["metastability"] = metastability(sim_result["order_param"])
        metrics["mean_synchrony"] = float(np.mean(sim_result["order_param"]))

    return metrics
