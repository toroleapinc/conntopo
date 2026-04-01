# Research Design Proposal: conntopo

## 1. Title & Abstract

**Title:** Does Connectome Topology Shape Neural Dynamics? A Systematic Comparison of Human Brain Wiring Against Null Network Models

**Abstract:**
The relationship between neural wiring topology and emergent computational dynamics remains a central question in neuroscience. While recent work on the *Drosophila* connectome (Lappalainen et al., 2024) demonstrates that structure can predict function at the single-neuron level, and reservoir computing frameworks (Suárez et al., 2024) show that connectome topology constrains computational capacity, no study has systematically isolated the contribution of macro-scale human connectome topology to neural dynamics under controlled conditions. This project addresses that gap. We load human structural connectivity matrices derived from the Human Connectome Project (HCP), instantiate well-characterized dynamical models (Wilson-Cowan and Kuramoto oscillators) on these networks, and compare the resulting dynamics against five classes of null network models: degree-preserving random rewiring, Erdős-Rényi random graphs, random geometric graphs, lattice networks, and weight-shuffled variants. We quantify differences across power spectral density, functional connectivity structure, metastability, transfer entropy, and regional differentiation. Through systematic feature attribution experiments, we identify which topological properties—small-world organization, rich-club structure, modular hierarchy—are necessary and sufficient to reproduce the dynamical signatures of the real connectome. All code, data pipelines, and statistical analyses are designed for full reproducibility on consumer hardware.

## 2. Research Questions

### Primary Question
Does the human brain's macro-connectome topology produce qualitatively and quantitatively different dynamics compared to degree-preserving random rewiring, Erdős-Rényi random graphs, and lattice networks?

### Secondary Questions
1. Which topological features (small-world structure, rich-club organization, modular hierarchy) are individually necessary or jointly sufficient to reproduce the dynamical signatures of the real connectome?
2. Are the observed differences robust across dynamical models (Wilson-Cowan vs. Kuramoto), parcellation scales (76, 180, 360 regions), and parameter regimes?
3. Do the same topological features that shape spontaneous dynamics also govern stimulus-evoked response patterns?

## 3. Background & Motivation

### Structure-Function Relationships in Neural Systems

A foundational hypothesis in network neuroscience holds that the brain's structural wiring constrains and shapes its functional dynamics (Sporns, 2011; Bassett & Sporns, 2017). This hypothesis has gained increasing empirical support across scales:

- **Micro-scale:** Lappalainen et al. (2024) demonstrated in the *Drosophila* connectome that synaptic-resolution wiring diagrams can predict neural function, establishing that structure-to-function mapping is feasible when connectivity is known precisely.

- **Meso-scale:** The conn2res framework (Suárez et al., 2024, *Nature Communications*) showed that connectome topology constrains the computational repertoire available to reservoir computing models, directly linking graph structure to information processing capacity.

- **Macro-scale:** The Virtual Brain project (Sanz Leon et al., 2013) established that large-scale brain network models parameterized by structural connectivity can reproduce empirical features of resting-state functional connectivity, though systematic null model comparisons remain limited.

### Key Topological Features

Three topological properties of brain networks have received particular attention:

1. **Small-world organization** (Watts & Strogatz, 1998): Brain networks exhibit high clustering (like lattices) combined with short path lengths (like random graphs). This architecture supports both local specialization and global integration.

2. **Rich-club organization** (van den Heuvel & Sporns, 2011): A densely interconnected core of high-degree hub regions forms a backbone for long-range communication. Rich-club hubs are disproportionately expensive metabolically and disproportionately affected in neurological disorders.

3. **Modular hierarchy** (Meunier et al., 2010): Brain networks are organized into nested modules at multiple scales, supporting functional segregation while maintaining cross-module integration through connector hubs.

### The Gap

Despite extensive characterization of these topological features and their putative functional roles, no study has systematically and simultaneously:

1. Compared dynamics on the real human macro-connectome against a comprehensive battery of null models,
2. Used multiple well-characterized dynamical models to ensure results are not model-dependent,
3. Performed controlled feature attribution to isolate which topological properties drive observed differences,
4. Tested robustness across parcellation scales.

Individual pieces exist—null model comparisons for specific metrics (Váša et al., 2022), dynamics on connectome-derived networks (Deco et al., 2017), topological characterization studies (Betzel et al., 2016)—but the integrated, controlled comparison framework is missing.

### Why It Matters

If connectome topology alone produces qualitatively different dynamics than random wiring, this has two major implications:

1. **For neuroscience:** It strengthens the case that structural connectivity is not merely a substrate but an active determinant of brain function. The specific topological features responsible would become high-priority targets for understanding both healthy cognition and neurological disease.

2. **For artificial intelligence:** Neural network architectures are overwhelmingly either fully connected, convolutional (lattice), or attention-based (complete bipartite). If biological wiring topology produces computationally advantageous dynamics, this motivates a new class of brain-inspired architectures.

## 4. Methodology

### 4.1 Connectome Data

**Source:** Human Connectome Project (HCP) group-averaged structural connectivity matrices derived from diffusion MRI tractography.

**Parcellation scales:**
- **Desikan-Killiany atlas:** 68 cortical + 8 subcortical = 76 regions (coarse, well-validated)
- **HCP-MMP1.0 (Glasser et al., 2016):** 180 regions per hemisphere = 360 regions (fine-grained, multi-modal parcellation)
- **Schaefer 200:** 200 cortical regions (intermediate, functionally defined)

**Preprocessing:** Matrices will be symmetrized (averaging upper and lower triangles), log-transformed to reduce the heavy-tailed distribution of streamline counts, and thresholded to remove spurious connections (retaining top 20% of connections by weight, with sensitivity analysis on threshold choice). Connection lengths (fiber tract distances) will be retained for conduction delay modeling.

### 4.2 Null Network Models

Five null model classes, each controlling for different aspects of the real connectome:

| Null Model | What It Preserves | What It Destroys | Implementation |
|---|---|---|---|
| **Degree-preserving random rewiring** (Maslov & Sneppen, 2002) | Degree sequence | Clustering, modularity, rich-club, motifs | Edge swap algorithm (10× edges swaps for convergence) |
| **Erdős-Rényi random graph** | Edge density (number of connections) | Degree distribution, all higher-order structure | `networkx.erdos_renyi_graph(n, p)` |
| **Random geometric graph** | Spatial embedding, connection length distribution | Topological organization beyond geometry | Nodes placed at real region coordinates, connected by distance threshold to match density |
| **Lattice/ring network** | Number of nodes and edges | All topological heterogeneity | k-nearest-neighbor ring lattice |
| **Weight-shuffled** | Binary topology (which regions connect) | Weight-topology correlations | Random permutation of edge weights across existing edges |

For each null model class, we generate **N = 100 instances** to establish null distributions for all metrics.

### 4.3 Dynamics Models

We employ two complementary models, chosen for analytical tractability and extensive prior use in computational neuroscience:

#### Wilson-Cowan Oscillators (Wilson & Cowan, 1972)
Each brain region *i* is modeled as a coupled excitatory-inhibitory population:

```
τ_E · dE_i/dt = -E_i + S(w_EE·E_i - w_EI·I_i + Σ_j C_ij·E_j + P_i)
τ_I · dI_i/dt = -I_i + S(w_IE·E_i - w_II·I_i + Q_i)
```

where *C_ij* is the structural connectivity matrix, *S* is a sigmoid transfer function, and *P_i*, *Q_i* are external inputs. This model captures excitation-inhibition balance, oscillatory behavior, and has rich dynamical regimes (fixed points, limit cycles, chaos).

**Parameter sweep:** We systematically vary global coupling strength *G* (scaling factor on *C_ij*) across a range spanning subcritical, critical, and supercritical regimes, as the coupling strength determines the dynamical regime and may interact with topology.

#### Kuramoto Oscillators (Kuramoto, 1984)
Each region *i* is a phase oscillator:

```
dθ_i/dt = ω_i + G · Σ_j C_ij · sin(θ_j - θ_i)
```

where *ω_i* are natural frequencies drawn from a distribution informed by empirical MEG/EEG frequency bands. This model is analytically tractable, well-suited for studying synchronization phenomena, and allows direct comparison with the extensive Kuramoto literature.

**Parameter sweep:** Global coupling *G* varied from desynchronized to fully synchronized regimes.

#### Simulation Parameters
- **Integration:** 4th-order Runge-Kutta, dt = 0.1 ms
- **Duration:** 60 seconds of simulated time (after 10-second transient discard)
- **Trials:** 20 independent runs per condition (different initial conditions, same network)
- **Conduction delays:** Incorporated using fiber tract lengths and assumed conduction velocity (5-20 m/s range)

### 4.4 Analysis Metrics

#### Power Spectral Density (PSD)
- Welch's method on regional time series
- Quantify: peak frequency, bandwidth, spectral entropy, 1/f exponent
- Compare distributions across regions and across network types

#### Functional Connectivity (FC)
- Pearson correlation of band-passed regional time series
- Compare FC matrices: correlation between structural and functional connectivity, FC matrix similarity across conditions
- Edge-level and module-level FC analysis

#### Metastability
- Kuramoto order parameter: *R(t) = |1/N · Σ_j exp(iθ_j(t))|*
- Metastability = standard deviation of *R(t)* over time (Shanahan, 2010)
- Higher metastability indicates richer dynamical repertoire—the system visits many different synchronization configurations

#### Transfer Entropy (TE)
- Directed information transfer between region pairs
- Computed using k-nearest-neighbor estimator (Kraskov et al., 2004)
- Net information flow: TE(i→j) - TE(j→i)
- Aggregate: total information transfer, information flow hierarchy

#### Regional Differentiation
- Coefficient of variation of regional dynamics (time series variance, mean frequency, participation in functional modules)
- Intrinsic ignition framework (Deco et al., 2017): how distinctly does each region respond when driven?
- Entropy of the regional dynamics distribution

### 4.5 Statistical Analysis

- **Permutation testing:** For each metric, compute the metric on the real connectome and on each of the 100 null model instances. The p-value is the fraction of null instances with metric values as extreme as or more extreme than the real connectome.
- **Effect sizes:** Cohen's d and Cliff's delta (non-parametric) for all comparisons.
- **Multiple comparison correction:** Benjamini-Hochberg FDR at q = 0.05 across all metrics and null model comparisons.
- **Bootstrap confidence intervals:** 95% BCa bootstrap CIs for all point estimates.

## 5. Experiments

### Experiment 1: Spontaneous Dynamics
**Question:** Does the real connectome produce different resting-state dynamics than null models?

**Protocol:**
1. Simulate Wilson-Cowan and Kuramoto models on the real connectome and all five null model classes (100 instances each)
2. No external input (P_i = Q_i = 0, or small uniform noise)
3. Sweep global coupling G across 20 values spanning subcritical to supercritical regimes
4. Compute all five metric families for each simulation
5. Statistical comparison: real connectome vs. each null model class

**Expected output:** For each metric × coupling strength × null model class: effect size, p-value, and confidence interval. Summary figure: metric value vs. coupling strength, with null model distributions as shaded bands.

### Experiment 2: Stimulus Response
**Question:** Does connectome topology shape how signals propagate through the network?

**Protocol:**
1. Identify "visual cortex" regions in the parcellation (V1, V2, V3 in Glasser atlas)
2. Inject a brief pulse stimulus into visual regions: P_visual(t) = A · δ(t - t_stim)
3. Measure response propagation: latency to activation in each region, spatial pattern of peak response, temporal profile of network-wide activation
4. Compare propagation patterns: real connectome vs. null models
5. Repeat with stimulation of "motor," "prefrontal," and "auditory" regions

**Expected output:** Propagation maps showing activation spread over time for real vs. null networks. Quantification of propagation speed, spatial specificity, and cross-regional response diversity.

### Experiment 3: Feature Attribution
**Question:** Which topological features drive the dynamical differences?

**Protocol:**
1. **Subtraction approach:** Starting from the real connectome, selectively destroy one feature at a time:
   - Remove rich-club connections (connections between top-10% degree nodes)
   - Rewire to destroy modularity while preserving degree sequence
   - Remove long-range connections (top quartile by fiber length)
   - Randomize weights while preserving binary topology (= weight-shuffled null model)
2. **Addition approach:** Starting from a degree-preserving random rewiring, selectively add one feature at a time:
   - Add rich-club organization (densify connections among hubs)
   - Impose modular structure (rewire to create community structure)
   - Add distance-dependent connectivity (bias connections toward spatial neighbors)
3. For each manipulated network, run full dynamics simulation and metric computation
4. Quantify: what fraction of the real-vs-null difference is explained by each feature?

**Expected output:** Feature attribution table showing the contribution of each topological feature to each dynamical metric. Interaction analysis: do features combine additively or synergistically?

### Experiment 4: Scale Sensitivity
**Question:** Are the results robust across parcellation resolutions?

**Protocol:**
1. Repeat Experiment 1 at three parcellation scales: 76 regions (Desikan-Killiany), 200 regions (Schaefer), 360 regions (Glasser)
2. For each scale, generate matched null models and run identical analyses
3. Compare effect sizes across scales: do the same topological features matter at all resolutions?

**Expected output:** Effect size comparison table across scales. Identification of any resolution-dependent effects (e.g., rich-club effects may be stronger at finer parcellations where hubs are better resolved).

### Experiment 5: Cross-Species Comparison
**Question:** Are the topology-dynamics relationships conserved across species?

**Protocol:**
1. Obtain structural connectivity matrices for macaque (CoCoMac / Markov et al., 2014) and mouse (Allen Mouse Brain Connectivity Atlas)
2. For each species, repeat Experiment 1: real connectome vs. null models
3. Compare the topology-dynamics relationship across species: are the same topological features important? Are effect sizes comparable?

**Contingency:** This experiment depends on data availability and comparability. If cross-species matrices are not directly comparable (different parcellation schemes, different measurement methods), we will restrict to human data and note this as future work.

**Expected output:** Cross-species comparison of which topological features shape dynamics. Test of the hypothesis that topology-dynamics relationships are evolutionarily conserved.

## 6. Expected Outcomes

### Primary Hypothesis
The real human connectome produces **richer dynamics** than null models, specifically:

1. **Greater metastability:** The real connectome will exhibit higher variability in synchronization patterns over time, indicating a richer repertoire of dynamical states. This is predicted by the combination of modular structure (supporting distinct synchronization clusters) and inter-modular hub connections (enabling transitions between states).

2. **Broader frequency spectrum:** Real connectome dynamics will show more diverse oscillation frequencies across regions, reflecting the heterogeneous local circuit properties enabled by the network's hierarchical modular organization.

3. **Higher regional differentiation:** Regions in the real connectome will show more distinct dynamical profiles from each other than in null models, reflecting the topological heterogeneity (degree distribution, clustering heterogeneity) of the real network.

4. **Structured information flow:** Transfer entropy patterns in the real connectome will show hierarchical organization (directed flow from sensory to association areas) absent in null models.

5. **Topology-function specificity:** Feature attribution (Experiment 3) will reveal that rich-club organization primarily supports metastability and information flow hierarchy, while modularity primarily supports regional differentiation and frequency diversity.

### Null Result Scenario
If null models produce dynamics indistinguishable from the real connectome, this is an equally important and publishable finding. It would suggest that:
- Macro-scale topology (at the resolution of diffusion MRI tractography) does not constrain dynamics beyond basic constraints like density and degree distribution
- The structure-function relationship may operate at finer scales (micro-circuitry, synaptic weights) not captured by macro-connectomics
- Current dynamical models may be insufficiently sensitive to topological differences

We will explicitly test for equivalence using two one-sided tests (TOST) to distinguish "no effect" from "insufficient power."

## 7. Timeline

| Week | Milestone | Deliverables |
|------|-----------|-------------|
| **1** | Data acquisition & infrastructure | HCP connectivity matrices loaded and validated; null model generation pipeline; project scaffold with CI/testing |
| **2** | Dynamics engine | Wilson-Cowan and Kuramoto simulators implemented and unit-tested; GPU acceleration if beneficial; parameter sweep infrastructure |
| **3** | Experiment 1 (Spontaneous dynamics) | Full results for real connectome vs. all null models; statistical analysis complete; initial figures |
| **4** | Experiments 2 & 3 (Stimulus response & feature attribution) | Stimulus propagation analysis; feature attribution results; identification of key topological drivers |
| **5** | Experiment 4 & robustness (Scale sensitivity) | Multi-scale results; sensitivity analyses (threshold, parameter, model); cross-validation of findings |
| **6** | Experiment 5 & write-up | Cross-species comparison (if feasible); final figures; manuscript draft; code/data release preparation |

### Weekly Checkpoints
- End of each week: automated report generation summarizing all completed analyses
- Continuous: version-controlled code, reproducible analysis notebooks, documented parameter choices

## 8. Hardware Requirements

**Available hardware:**
- **GPU:** NVIDIA RTX 5070 (12 GB VRAM)
- **RAM:** 16 GB system memory
- **Storage:** Standard SSD

**Feasibility assessment:**

| Component | Memory Estimate | Feasibility |
|---|---|---|
| Connectivity matrix (360 × 360, float64) | ~1 MB | Trivial |
| 100 null model instances | ~100 MB | Trivial |
| Time series (360 regions × 600,000 timesteps × float64) | ~1.7 GB per run | Fits comfortably in RAM; batch if needed |
| Functional connectivity matrix (360 × 360) | ~1 MB | Trivial |
| Transfer entropy computation | CPU-bound, parallelizable | Hours per condition; parallelize across cores |
| GPU acceleration for ODE integration | ~2 GB VRAM for batched simulation | Well within 12 GB; can batch multiple null models simultaneously |

**Conclusion:** All experiments are feasible on the available hardware. The bottleneck will be transfer entropy computation (combinatorial in number of region pairs), which we will address through parallelization and efficient k-NN estimators (e.g., JIDT library or custom implementation).

**Estimated total compute time:** 24-48 hours for all experiments (with parallelization across null model instances and coupling strengths).

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Results depend on dynamical model choice** | Medium | High | Test with both Wilson-Cowan and Kuramoto; if results diverge, this itself is a finding about model sensitivity |
| **Macro-connectome resolution too coarse** | Medium | High | Test at three parcellation scales (76, 200, 360 regions); if effects emerge only at finer scales, this informs the field about resolution requirements |
| **Results depend on coupling strength G** | High | Medium | Systematic parameter sweep; report results across the full range rather than cherry-picking a single G value |
| **Tractography artifacts in connectivity data** | Medium | Medium | Use group-averaged matrices (reduces individual noise); test sensitivity to connection density threshold; compare with published validated matrices |
| **Insufficient statistical power** | Low | High | 100 null model instances per class provides robust null distributions; 20 simulation trials per condition handles dynamical variability; power analysis before running |
| **Transfer entropy computation too slow** | Medium | Low | Use efficient estimators (JIDT); subsample region pairs if needed; parallelize; GPU-accelerate if available |
| **Cross-species data incompatible** | High | Low | Experiment 5 is explicitly contingent; human-only results are independently publishable |

## 10. Publication Target

### Primary Venues

1. **PLoS Computational Biology** — Ideal fit: computational neuroscience, network analysis, reproducible methods. Open access. Impact factor ~4.5.

2. **Network Neuroscience** (MIT Press) — Specialized journal for exactly this type of work: network analysis of brain connectivity and dynamics. Impact factor ~5.0.

3. **NeuroImage** — Broad neuroimaging audience; strong precedent for computational connectomics work. Impact factor ~5.7.

### Secondary Venues

4. **NeurIPS NeuroAI Workshop** — For preliminary results; builds visibility in the computational neuroscience / AI intersection community.

5. **OHBM (Organization for Human Brain Mapping) Annual Meeting** — Poster/talk for early results and community feedback.

### Open Science Commitments

- All code released on GitHub under MIT license
- Processed connectivity matrices and simulation outputs deposited on Zenodo/OSF
- Analysis notebooks fully reproducible (pinned dependencies, containerized environment)
- Pre-registration of hypotheses and analysis plan on OSF prior to running experiments
