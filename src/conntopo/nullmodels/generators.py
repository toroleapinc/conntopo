"""Null network model generators for controlled comparison.

Each generator takes a real connectome and produces a randomized version
that preserves specific topological properties while destroying others.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from conntopo.connectome.loader import Connectome


def degree_preserving_rewire(
    connectome: Connectome, n_swaps_factor: int = 10, seed: int | None = None
) -> Connectome:
    """Rewire edges while preserving the degree sequence (Maslov-Sneppen algorithm).

    Preserves: degree distribution
    Destroys: clustering, modularity, rich-club, motifs
    """
    rng = np.random.default_rng(seed)
    w = connectome.weights.copy()
    n = w.shape[0]

    # Get all edges
    rows, cols = np.where(w > 0)
    edges = list(zip(rows.tolist(), cols.tolist()))
    weights_list = [float(w[r, c]) for r, c in edges]
    n_edges = len(edges)
    n_swaps = n_swaps_factor * n_edges

    for _ in range(n_swaps):
        # Pick two random edges
        idx1, idx2 = rng.choice(n_edges, size=2, replace=False)
        a, b = edges[idx1]
        c, d = edges[idx2]

        # Avoid self-loops and duplicate edges
        if a == d or c == b:
            continue
        if w[a, d] > 0 or w[c, b] > 0:
            continue

        # Swap: (a→b, c→d) becomes (a→d, c→b)
        w[a, b] = 0
        w[c, d] = 0
        w[a, d] = weights_list[idx1]
        w[c, b] = weights_list[idx2]

        edges[idx1] = (a, d)
        edges[idx2] = (c, b)

    return Connectome.from_numpy(w, list(connectome.labels))


def erdos_renyi_null(
    connectome: Connectome, seed: int | None = None
) -> Connectome:
    """Generate Erdos-Renyi random graph matching edge density.

    Preserves: number of edges (density)
    Destroys: degree distribution, all higher-order structure
    """
    rng = np.random.default_rng(seed)
    n = connectome.num_regions
    n_edges = connectome.num_edges

    # Get weight distribution from real connectome to sample from
    real_weights = connectome.weights[connectome.weights > 0]

    w = np.zeros((n, n), dtype=np.float32)

    # Randomly place edges (no self-loops)
    possible = [(i, j) for i in range(n) for j in range(n) if i != j]
    chosen = rng.choice(len(possible), size=min(n_edges, len(possible)), replace=False)

    for idx in chosen:
        i, j = possible[idx]
        w[i, j] = rng.choice(real_weights)

    return Connectome.from_numpy(w, list(connectome.labels))


def random_geometric_null(
    connectome: Connectome, seed: int | None = None
) -> Connectome:
    """Generate random geometric graph preserving spatial embedding.

    Preserves: spatial layout, distance-dependent connectivity
    Destroys: topological organization beyond geometry
    """
    if connectome.positions is None:
        raise ValueError("Connectome must have positions for geometric null model")

    rng = np.random.default_rng(seed)
    n = connectome.num_regions
    n_edges = connectome.num_edges
    pos = connectome.positions

    # Compute pairwise distances
    diffs = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)  # Exclude self-loops

    # Find distance threshold that gives approximately the right number of edges
    flat_dists = dists[dists < np.inf]
    threshold = np.percentile(flat_dists, 100 * n_edges / (n * (n - 1)))

    # Create connections for pairs within threshold
    real_weights = connectome.weights[connectome.weights > 0]
    w = np.zeros((n, n), dtype=np.float32)

    candidates = np.argwhere(dists <= threshold)
    for idx in range(len(candidates)):
        i, j = candidates[idx]
        w[i, j] = rng.choice(real_weights)

    return Connectome.from_numpy(w, list(connectome.labels))


def lattice_null(
    connectome: Connectome, seed: int | None = None
) -> Connectome:
    """Generate k-nearest-neighbor ring lattice matching edge count.

    Preserves: number of nodes, approximate edge count
    Destroys: all topological heterogeneity
    """
    rng = np.random.default_rng(seed)
    n = connectome.num_regions
    n_edges = connectome.num_edges

    # k-nearest-neighbor ring: each node connects to k/2 neighbors on each side
    k = max(2, n_edges // n)
    real_weights = connectome.weights[connectome.weights > 0]

    w = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for offset in range(1, k // 2 + 1):
            j_fwd = (i + offset) % n
            j_bwd = (i - offset) % n
            w[i, j_fwd] = rng.choice(real_weights)
            w[i, j_bwd] = rng.choice(real_weights)

    return Connectome.from_numpy(w, list(connectome.labels))


def weight_shuffled_null(
    connectome: Connectome, seed: int | None = None
) -> Connectome:
    """Shuffle weights across existing edges, preserving binary topology.

    Preserves: binary topology (which regions connect)
    Destroys: weight-topology correlations
    """
    rng = np.random.default_rng(seed)
    w = connectome.weights.copy()

    # Get all edge weights
    mask = w > 0
    edge_weights = w[mask].copy()
    rng.shuffle(edge_weights)
    w[mask] = edge_weights

    return Connectome.from_numpy(w, list(connectome.labels))


def generate_null_ensemble(
    connectome: Connectome,
    null_type: str,
    n_instances: int = 100,
    seed: int | None = None,
) -> list[Connectome]:
    """Generate an ensemble of null model instances.

    Args:
        connectome: The real connectome to derive null models from.
        null_type: One of 'degree_preserving', 'erdos_renyi',
                   'random_geometric', 'lattice', 'weight_shuffled'.
        n_instances: Number of null instances to generate.
        seed: Base seed (each instance uses seed+i).

    Returns:
        List of Connectome instances.
    """
    generators = {
        "degree_preserving": degree_preserving_rewire,
        "erdos_renyi": erdos_renyi_null,
        "random_geometric": random_geometric_null,
        "lattice": lattice_null,
        "weight_shuffled": weight_shuffled_null,
    }

    if null_type not in generators:
        raise ValueError(
            f"Unknown null type '{null_type}'. "
            f"Available: {list(generators.keys())}"
        )

    gen = generators[null_type]
    results = []
    for i in range(n_instances):
        s = (seed + i) if seed is not None else None
        results.append(gen(connectome, seed=s))

    return results
