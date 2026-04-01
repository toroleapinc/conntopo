# Reddit r/MachineLearning Post

**Title:** [R] Does human brain wiring shape neural computation? We tested the Human Connectome Project data against 5 null models — the answer is yes, with effect sizes d=5-2400+

**Body:**

## TL;DR

We systematically compared neural dynamics on the real human brain connectome (76 regions from the Human Connectome Project) against 5 types of random wiring using two dynamics models (Kuramoto and Wilson-Cowan). The real connectome produces massively different dynamics — and in Wilson-Cowan, random wiring collapses to silence while the real connectome sustains oscillations (d>2000).

## What we did

- Loaded the 76-region structural connectivity matrix from the Human Connectome Project (via The Virtual Brain)
- Ran Kuramoto phase oscillators and Wilson-Cowan excitatory-inhibitory dynamics on it
- Compared against 5 null network models: degree-preserving rewire, Erdős-Rényi random, random geometric, lattice, and weight-shuffled
- 20 null instances per type, 6 coupling strengths per model, Mann-Whitney U tests

## Key results

**Kuramoto:** 111/210 comparisons significant (p<0.05), effect sizes d=5-16

**Wilson-Cowan:** 48/150 comparisons significant, effect sizes up to d>2000

The most dramatic finding: at moderate coupling in the Wilson-Cowan model, the real connectome sustains excitatory-inhibitory oscillations while ER, lattice, and random geometric networks collapse to fixed points. The specific wiring pattern of the human brain is what keeps it oscillating.

Even degree-preserving rewiring (which keeps the same number of connections per region but randomizes targets) produces different dynamics — meaning it's the precise connectivity pattern, not just hub structure or degree distribution.

[Image: figures/wc_oscillation_highlight.png]

## Why this matters for ML/AI

If biological brain topology produces computationally advantageous dynamics, it motivates a new class of brain-topology-informed architectures. Current neural networks use either fully-connected, convolutional (lattice), or attention-based (complete bipartite) topologies — none of which resemble the brain's modular, small-world, rich-club organization.

## Code + data

Everything is open source and fully reproducible: https://github.com/toroleapinc/conntopo

```
pip install conntopo
python -m conntopo  # Reproduce key finding in ~60 seconds
```

## What's next

- Feature attribution: which topological properties (rich-club, modularity, long-range connections) drive the effect?
- Multi-scale validation at 200 and 360 regions
- Cross-species comparison (macaque, mouse)

Looking for feedback — what experiments would strengthen or challenge these findings?
