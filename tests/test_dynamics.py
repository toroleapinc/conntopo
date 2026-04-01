"""Tests for dynamics models."""

import numpy as np

from conntopo.connectome import Connectome
from conntopo.dynamics.wilson_cowan import WilsonCowanModel, WilsonCowanParams
from conntopo.dynamics.kuramoto import KuramotoModel


def _get_connectome():
    return Connectome.from_bundled("toy20")


def test_wilson_cowan_runs():
    c = _get_connectome()
    model = WilsonCowanModel(c.weights, global_coupling=0.5)
    result = model.simulate(duration=500, dt=0.1, transient=100, seed=42)

    assert "E" in result and "I" in result and "time" in result
    assert result["E"].shape[1] == 20
    assert result["I"].shape[1] == 20
    assert result["E"].shape[0] == result["time"].shape[0]


def test_wilson_cowan_bounded():
    c = _get_connectome()
    model = WilsonCowanModel(c.weights, global_coupling=0.5)
    result = model.simulate(duration=500, dt=0.1, transient=100, seed=42)

    assert np.all(result["E"] >= 0) and np.all(result["E"] <= 1)
    assert np.all(result["I"] >= 0) and np.all(result["I"] <= 1)
    assert not np.isnan(result["E"]).any()


def test_wilson_cowan_not_flat():
    """Activity should not be constant — oscillations should emerge."""
    c = _get_connectome()
    model = WilsonCowanModel(c.weights, global_coupling=1.0)
    result = model.simulate(duration=1000, dt=0.1, transient=200, seed=42)

    variance = np.var(result["E"], axis=0)
    assert np.any(variance > 1e-6), "All regions have zero variance — no dynamics"


def test_kuramoto_runs():
    c = _get_connectome()
    model = KuramotoModel(c.weights, global_coupling=0.5)
    result = model.simulate(duration=500, dt=0.1, transient=100, seed=42)

    assert "theta" in result and "order_param" in result and "time" in result
    assert result["theta"].shape[1] == 20
    assert result["order_param"].shape[0] == result["time"].shape[0]


def test_kuramoto_order_param_bounded():
    c = _get_connectome()
    model = KuramotoModel(c.weights, global_coupling=0.5)
    result = model.simulate(duration=500, dt=0.1, transient=100, seed=42)

    assert np.all(result["order_param"] >= 0)
    assert np.all(result["order_param"] <= 1)


def test_kuramoto_metastability():
    c = _get_connectome()
    model = KuramotoModel(c.weights, global_coupling=1.0)
    result = model.simulate(duration=1000, dt=0.1, transient=200, seed=42)

    meta = KuramotoModel.metastability(result["order_param"])
    assert isinstance(meta, float)
    assert meta >= 0


def test_coupling_affects_dynamics():
    """Higher coupling should produce more synchrony in Kuramoto."""
    c = _get_connectome()

    low = KuramotoModel(c.weights, global_coupling=0.01)
    high = KuramotoModel(c.weights, global_coupling=5.0)

    r_low = low.simulate(duration=2000, dt=0.1, transient=500, seed=42)
    r_high = high.simulate(duration=2000, dt=0.1, transient=500, seed=42)

    sync_low = KuramotoModel.mean_synchrony(r_low["order_param"])
    sync_high = KuramotoModel.mean_synchrony(r_high["order_param"])

    assert sync_high > sync_low, (
        f"High coupling synchrony ({sync_high:.3f}) should exceed "
        f"low coupling ({sync_low:.3f})"
    )
