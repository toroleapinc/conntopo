"""Tests for connectome loading."""

import numpy as np
import pytest

from conntopo.connectome import Connectome


def test_tvb76_loads():
    c = Connectome.from_bundled("tvb76")
    assert c.num_regions == 76
    assert c.weights.shape == (76, 76)


def test_tvb76_valid():
    c = Connectome.from_bundled("tvb76")
    assert not np.isnan(c.weights).any()
    assert (c.weights >= 0).all()
    assert c.num_edges > 0


def test_toy20_loads():
    c = Connectome.from_bundled("toy20")
    assert c.num_regions == 20
