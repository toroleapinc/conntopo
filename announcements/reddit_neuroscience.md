# Reddit r/neuroscience Post

**Title:** We compared dynamics on the real human macro-connectome vs 5 null network models — the real topology sustains oscillations that random wiring cannot (open source, fully reproducible)

**Body:**

Inspired by the Drosophila connectome work (Lappalainen et al., Nature 2024) showing that wiring predicts function at the single-neuron level, we asked: does the same principle hold at the human MACRO scale?

## What we did

We loaded the 76-region structural connectivity from the Human Connectome Project (via TVB) and ran two well-characterized dynamics models on it:

- **Kuramoto oscillators** (phase coupling, synchronization)
- **Wilson-Cowan** (excitatory-inhibitory populations)

We compared against 5 null network models, each preserving different graph properties:

| Null Model | Preserves |
|---|---|
| Degree-preserving rewire | Degree distribution |
| Erdős-Rényi | Edge density |
| Random geometric | Spatial embedding |
| Lattice | Node/edge count |
| Weight-shuffled | Binary topology |

20 instances per null type, 6 coupling strengths per model, Mann-Whitney U tests with Cohen's d effect sizes.

## The key finding

At moderate coupling (the biologically relevant regime near criticality), the real connectome produces **qualitatively different dynamics** across both models:

- **Kuramoto:** 111/210 comparisons significant, d=5-16
- **Wilson-Cowan:** The real connectome sustains E/I oscillations while ER, lattice, and geometric networks **collapse to fixed points** (d>2000)

Even degree-preserving rewiring differs from the real connectome — meaning it's not just degree distribution or hub structure, but the **specific wiring pattern**.

## What this suggests

The macro-scale wiring topology of the human brain is not just a substrate — it actively shapes computational dynamics. The modular hierarchy, rich-club hubs, and small-world organization appear to be necessary for sustaining the oscillatory dynamics we observe in EEG/MEG.

## Limitations we're upfront about

- 76-region parcellation is coarse — finer scales (360 regions) are planned
- We haven't yet done feature attribution (which topological features drive this)
- Euler integration; RK4 planned
- No conduction delays yet

## Open source

Everything is reproducible: https://github.com/toroleapinc/conntopo

```
pip install conntopo
python -m conntopo
```

Would love feedback from the computational neuroscience community — especially on methodology and what experiments you'd want to see next.
