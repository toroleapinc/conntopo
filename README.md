# conntopo

**An open-source toolkit for comparing neural dynamics on brain connectomes vs null network models.**

conntopo simulates Kuramoto and Wilson-Cowan dynamics on the Human Connectome Project structural connectivity data and systematically compares against 5 types of null network models. It reproduces and extends well-established findings in computational connectomics, providing a clean, reproducible Python implementation.

![Hero Figure](figures/hero_combined.png)

## What This Tool Does

- Load real human brain structural connectivity (76 regions from HCP/TVB)
- Simulate two dynamics models: **Kuramoto** phase oscillators and **Wilson-Cowan** excitatory-inhibitory populations
- Generate 5 types of null network models (degree-preserving, Erdős-Rényi, random geometric, lattice, weight-shuffled)
- Compare dynamics across metrics: metastability, synchrony, functional connectivity, regional differentiation, spectral properties
- Statistical analysis with effect sizes and permutation testing

## Context

The finding that real brain connectome topology produces different dynamics than random wiring is well-established in computational neuroscience (see [Honey et al. 2007](https://www.pnas.org/doi/10.1073/pnas.0701519104), [Cabral et al. 2011](https://pubmed.ncbi.nlm.nih.gov/21511044/), [Deco & Kringelbach 2017](https://www.nature.com/articles/s41598-017-03073-5)). For a comprehensive review of null model approaches, see [Váša & Mišić 2022](https://www.nature.com/articles/s41583-022-00601-9).

This toolkit provides a clean, pip-installable Python implementation of this methodology for researchers who want to:
- Reproduce these established results
- Test their own connectome data against null models
- Use it as a starting point for more specific analyses

## Quick Start

```bash
pip install conntopo
python -m conntopo  # Run demo (~60 seconds)
```

```python
from conntopo.connectome import Connectome
from conntopo.dynamics import KuramotoModel
from conntopo.nullmodels import generate_null_ensemble
from conntopo.analysis.metrics import compute_all_metrics

# Load real human brain connectome (76 regions, 1560 connections)
brain = Connectome.from_bundled("tvb76")

# Simulate Kuramoto oscillators
model = KuramotoModel(brain.weights, global_coupling=1.0)
result = model.simulate(duration=5000, dt=0.1, transient=1000, seed=42)
metrics = compute_all_metrics(result, model_type="kuramoto")

# Compare against random wiring
nulls = generate_null_ensemble(brain, "erdos_renyi", n_instances=20, seed=42)
```

## Null Models

| Null Model | Preserves | Destroys |
|---|---|---|
| **Degree-preserving rewire** | Degree distribution | Clustering, modularity, rich-club |
| **Erdős-Rényi random** | Edge density | All structure |
| **Random geometric** | Spatial embedding | Non-spatial topology |
| **Lattice** | Node/edge count | All heterogeneity |
| **Weight-shuffled** | Binary topology | Weight-topology correlations |

## Reproduce Experiments

```bash
git clone https://github.com/toroleapinc/conntopo.git
cd conntopo
pip install -e ".[dev]"
python experiments/01_spontaneous_dynamics.py --model kuramoto
python experiments/01_spontaneous_dynamics.py --model wilson_cowan
python scripts/generate_hero_figure.py
```

## Related Work

For the next step — testing whether region-specific functional roles emerge from topology — see our companion project [encephagen](https://github.com/toroleapinc/encephagen).

## Data

Structural connectivity from [The Virtual Brain](https://www.thevirtualbrain.org/), derived from Human Connectome Project diffusion MRI tractography.

## License

MIT
