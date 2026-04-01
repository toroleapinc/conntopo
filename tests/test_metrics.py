"""Tests for analysis metrics."""

import numpy as np

from conntopo.analysis.metrics import (
    power_spectral_density,
    peak_frequency,
    spectral_entropy,
    functional_connectivity,
    regional_differentiation,
    compute_all_metrics,
)


def _make_oscillating_data(n_regions=10, n_steps=5000, dt=0.1):
    """Create synthetic oscillating data for testing."""
    t = np.arange(n_steps) * dt / 1000.0  # Time in seconds
    rng = np.random.default_rng(42)
    freqs = rng.uniform(5, 20, size=n_regions)
    data = np.zeros((n_steps, n_regions))
    for i in range(n_regions):
        data[:, i] = np.sin(2 * np.pi * freqs[i] * t) + 0.1 * rng.standard_normal(n_steps)
    return data, freqs


def test_psd_shape():
    data, _ = _make_oscillating_data()
    freqs, psd = power_spectral_density(data, dt=0.1)
    assert psd.shape[1] == 10  # n_regions
    assert freqs.shape[0] == psd.shape[0]


def test_peak_frequency_detects_oscillation():
    data, true_freqs = _make_oscillating_data()
    freqs, psd = power_spectral_density(data, dt=0.1)
    peaks = peak_frequency(freqs, psd)
    # Peak should be within 6 Hz of true frequency (limited by PSD resolution)
    assert np.mean(np.abs(peaks - true_freqs)) < 6.0


def test_spectral_entropy_positive():
    data, _ = _make_oscillating_data()
    _, psd = power_spectral_density(data, dt=0.1)
    entropy = spectral_entropy(psd)
    assert np.all(entropy > 0)


def test_fc_shape_and_symmetry():
    data, _ = _make_oscillating_data()
    fc = functional_connectivity(data)
    assert fc.shape == (10, 10)
    np.testing.assert_array_almost_equal(fc, fc.T)


def test_fc_diagonal_is_one():
    data, _ = _make_oscillating_data()
    fc = functional_connectivity(data)
    np.testing.assert_array_almost_equal(np.diag(fc), np.ones(10))


def test_regional_differentiation_bounded():
    data, _ = _make_oscillating_data()
    rd = regional_differentiation(data)
    assert 0.0 <= rd <= 2.0  # 1 - correlation, range [0, 2]


def test_compute_all_metrics_wilson_cowan():
    data, _ = _make_oscillating_data()
    sim_result = {"E": data, "I": data * 0.5, "time": np.arange(data.shape[0]) * 0.1}
    metrics = compute_all_metrics(sim_result, dt=0.1, model_type="wilson_cowan")
    assert "mean_peak_freq" in metrics
    assert "regional_differentiation" in metrics
    assert "global_variance" in metrics


def test_compute_all_metrics_kuramoto():
    rng = np.random.default_rng(42)
    theta = rng.uniform(0, 2 * np.pi, (5000, 10))
    order = np.abs(np.mean(np.exp(1j * theta), axis=1))
    sim_result = {"theta": theta, "order_param": order, "time": np.arange(5000) * 0.1}
    metrics = compute_all_metrics(sim_result, dt=0.1, model_type="kuramoto")
    assert "metastability" in metrics
    assert "mean_synchrony" in metrics
