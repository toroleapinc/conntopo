"""Tests for null model generators."""

import numpy as np

from conntopo.connectome import Connectome
from conntopo.nullmodels import (
    degree_preserving_rewire,
    erdos_renyi_null,
    lattice_null,
    weight_shuffled_null,
    generate_null_ensemble,
)


def _get_connectome():
    return Connectome.from_bundled("toy20")


def test_degree_preserving_same_shape():
    c = _get_connectome()
    null = degree_preserving_rewire(c, seed=42)
    assert null.num_regions == c.num_regions


def test_degree_preserving_preserves_density():
    c = _get_connectome()
    null = degree_preserving_rewire(c, seed=42)
    # Edge count should be approximately preserved
    assert abs(null.num_edges - c.num_edges) < c.num_edges * 0.1


def test_erdos_renyi_same_shape():
    c = _get_connectome()
    null = erdos_renyi_null(c, seed=42)
    assert null.num_regions == c.num_regions
    assert null.num_edges > 0


def test_lattice_same_shape():
    c = _get_connectome()
    null = lattice_null(c, seed=42)
    assert null.num_regions == c.num_regions
    assert null.num_edges > 0


def test_weight_shuffled_preserves_topology():
    c = _get_connectome()
    null = weight_shuffled_null(c, seed=42)
    # Same binary topology
    np.testing.assert_array_equal(
        (c.weights > 0).astype(int),
        (null.weights > 0).astype(int),
    )


def test_weight_shuffled_changes_weights():
    c = _get_connectome()
    null = weight_shuffled_null(c, seed=42)
    # Weights should be different (shuffled)
    assert not np.allclose(c.weights, null.weights)


def test_generate_ensemble():
    c = _get_connectome()
    ensemble = generate_null_ensemble(c, "erdos_renyi", n_instances=5, seed=42)
    assert len(ensemble) == 5
    assert all(isinstance(e, Connectome) for e in ensemble)


def test_ensemble_instances_differ():
    c = _get_connectome()
    ensemble = generate_null_ensemble(c, "erdos_renyi", n_instances=3, seed=42)
    # Each instance should be different
    assert not np.allclose(ensemble[0].weights, ensemble[1].weights)


def test_unknown_null_type():
    import pytest
    c = _get_connectome()
    with pytest.raises(ValueError, match="Unknown null type"):
        generate_null_ensemble(c, "nonexistent")
