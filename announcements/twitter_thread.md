# Twitter/X Thread

## Tweet 1 (Hook)

Does human brain wiring actually matter for neural computation?

We tested the real Human Connectome Project data against 5 types of random wiring using two dynamics models.

The answer: yes — and in one model, random wiring literally collapses to silence while the real connectome sustains oscillations.

🧵👇

## Tweet 2 (The killer figure)

[ATTACH: figures/wc_oscillation_highlight.png]

The most dramatic finding: in Wilson-Cowan E/I dynamics, the real human connectome sustains oscillations (black line) while Erdős-Rényi, lattice, and random geometric networks collapse to fixed points.

Effect size: Cohen's d > 2000. Not a typo.

## Tweet 3 (Cross-model validation)

[ATTACH: figures/hero_combined.png]

This isn't model-dependent. We tested with both Kuramoto oscillators (top) and Wilson-Cowan E/I dynamics (bottom).

Both show the real connectome producing distinct dynamics — metastability, synchronization, and regional differentiation all differ from null models.

## Tweet 4 (Why it matters)

Why this matters:

The Drosophila connectome study (Lappalainen et al., Nature 2024) showed wiring predicts function in fruit flies at single-neuron resolution.

We show the same principle holds at the human MACRO scale — the large-scale wiring architecture actively shapes computational dynamics.

## Tweet 5 (It's the specific wiring)

The most surprising part: even degree-preserving rewiring (same number of connections per region, but random targets) produces different dynamics.

It's not just hubs or density. It's the SPECIFIC wiring pattern of the human brain that matters.

## Tweet 6 (Reproducible + open source)

Everything is open source and reproducible:

github.com/toroleapinc/conntopo

pip install conntopo
python -m conntopo

Reproduce the key finding on your machine in 60 seconds.

Kuramoto: 111/210 comparisons significant
Wilson-Cowan: 48/150 significant

## Tweet 7 (The ask)

Looking for feedback from computational neuroscience and NeuroAI communities.

Next steps:
- Feature attribution: which topological properties drive this?
- Multi-scale validation (360 regions)
- Cross-species comparison

What experiments would you want to see?

#neuroscience #NeuroAI #connectome #opensource
