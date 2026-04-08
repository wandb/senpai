# SENPAI Research State

- **Date:** 2026-04-08 20:15 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training & Architecture Improvements

## Current Baseline

### Single-Model Baseline (PR #2290, +Re-Stratified Sampling, 2-seed)

| Metric | 2-seed avg | Target to beat |
|--------|-----------|----------------|
| **p_in** | **11.74** | < 11.74 |
| p_oodc | 7.65 | < 7.65 |
| **p_tan** | **27.90** | < 27.90 |
| p_re | 6.40 | < 6.40 |

**Latest merge:** PR #2290 (nezuko) — Re-Stratified Sampling. p_in -1.3%, p_tan -0.8%. p_oodc +1.2%, p_re +0.6% (minor regressions). Net sum -0.24 points.

⚠️ **p_re = 6.40 is now a greater regression vs PR #2213 baseline (6.300).** p_oodc = 7.65 also slightly regressed. Future experiments should target recovery of both.

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

Single-model beats ensemble on p_in (11.74 vs 12.1) and p_tan (27.90 vs 29.1). Ensemble still leads on p_oodc (7.65 vs 6.6) and p_re (6.40 vs 5.8) — large gaps, especially p_oodc.

## Student Status (2026-04-08 19:05 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| nezuko | #2297 | **FV Cell-Area Loss Weighting — 1/sqrt(cell_area) weight on vol loss** | WIP (BOLD) |
| thorfinn | #2298 | **GMSE Gradient-Weighted Pressure Loss — weight by local ∇p magnitude** | WIP (BOLD) |
| alphonse | #2299 | **Potential Flow Residual Loss — Bernoulli-consistency auxiliary signal** | WIP (NEW, BOLD, PARADIGM) |
| fern | #2294 | **Tandem Config Proximity Feature — OOD distance signal for calibration** | WIP |
| askeladd | #2292 | **Flow-Direction Normalization — rotate coords by -AoA to streamwise frame** | WIP |
| frieren | #2291 | **Stagnation Pressure Feature — q_inf = 0.5*Umag² as input channel** | WIP |
| tanjiro | #2300 | **Mirror Symmetry Augmentation — exact y-reflection to double training data** | WIP (NEW, BOLD, DATA AUG) |
| edward | #2301 | **Continuity PDE Loss — mass-conservation ∇·u=0 penalty** | WIP (NEW, BOLD, PARADIGM) |

## PRs Ready for Review

None.

## Latest Reviews (2026-04-08 17:45)

### PR #2290 (nezuko, Re-Stratified Sampling) — MERGED ✅
- p_in -1.3% (11.74), p_tan -0.8% (27.90). p_oodc +1.2%, p_re +0.6% (minor regressions). Net sum improved -0.24 pts.

### PR #2288 (thorfinn, Chord Fraction Feature) — CLOSED ❌
- All metrics worse: p_in +4.7%, p_oodc +4.5%, p_tan +1.5%, p_re +6.2%. Redundant with TE coord frame.

## Most Recent Research Direction from Human Researcher Team

### Issue #1860 — ACTIVE directive (Morgan McGuire, re-raised 2026-04-08)
**"Too many of your current experiments are incremental tweaks. Think BIGGER — radical new full model changes and data aug and data generation."**

**Advisor response (2026-04-08):** Acknowledged. The two newly freed students (nezuko, thorfinn) are assigned physics-grounded, paradigm-level changes:
1. **FV Cell-Area Loss Weighting (#2297)** — addresses fundamental structural bias in unweighted loss on non-uniform meshes. Backed by ICML 2024 paper showing 15-40% improvement.
2. **GMSE Gradient-Weighted Loss (#2298)** — automatically up-weights high-gradient (LE, tandem slot) regions. Backed by arXiv:2411.17059 showing 5-23% improvement.

Next-round assignments (when Round 29 in-flight students complete) will continue bold directions from Round 30 ideas + Round 31 researcher-agent output.

## Current Research Focus and Themes

### Round 29 In-Flight Experiments (6 GPUs)
| Student | PR | Direction | Target |
|---------|-----|-----------|--------|
| fern | #2294 | **Tandem Config Proximity** — KD-tree OOD proximity feature | p_tan |
| askeladd | #2292 | **Flow-Direction Normalization** — rotate (x,y) by -AoA to streamwise frame | p_oodc, p_re |
| frieren | #2291 | **Stagnation Pressure Feature** — q_inf = 0.5*Umag² as input channel | p_in, p_re |
| tanjiro | #2295 | **Surface Curvature Feature** — Menger curvature κ at surface nodes | p_tan, p_in |
| edward | #2296 | **Log-Re Pressure Scaling** — Re-normalize loss for OOD-Re generalization | p_re, p_oodc |
| alphonse | #2293 | **Low-Rank Pressure Loss** — SVD penalty beyond rank-5 on surface error | p_tan, p_in |

### Round 30-31 Bold Experiments (3 GPUs — PARADIGM-LEVEL)
| Student | PR | Direction | Mechanism |
|---------|-----|-----------|-----------|
| nezuko | #2297 | **FV Cell-Area Loss Weighting** | 1/sqrt(cell_area) weights volume loss — FV theory, ICML 2024 |
| thorfinn | #2298 | **GMSE Gradient-Weighted Pressure Loss** | Weight by ∇p magnitude — auto-targets LE/slot high-gradient zones |
| alphonse | #2299 | **Potential Flow Residual Loss** | Bernoulli-consistency auxiliary: couples u,v,p via physics law. PARADIGM. |
| tanjiro | #2300 | **Mirror Symmetry Augmentation** | Exact y-reflection doubles effective training data. DATA AUG. |
| edward | #2301 | **Continuity PDE Loss** | ∇·u=0 penalty on predicted velocity — physics-informed regularization. PARADIGM. |

### Key Mechanistic Insights (accumulated)

1. **Every durable improvement came from: (a) physics-motivated input features, or (b) loss reformulation.**
2. **All architecture replacements failed in Phase 5** (GNOT, Galerkin, GNN, DeepONet, FNO, INR — all 30-60% worse). Transolver is a remarkably strong local optimum.
3. **Tandem p_tan bottleneck:** Largest absolute error gap. High-gradient tandem slot is under-weighted in current loss. GMSE and FV-area weighting both target this.
4. **p_oodc and p_re gaps to ensemble are large:** Single model 7.65 vs ensemble 6.6 (p_oodc), 6.40 vs 5.8 (p_re). These need OOD-focused approaches.
5. **Shared K/V is load-bearing regularization** — per-head projections destroy OOD generalization.
6. **Hard-node mining already captures suction difficulty** — physics-based suction weighting redundant.

### What Works (confirmed and merged)

| Direction | PR | Impact |
|-----------|-----|--------|
| Re-Stratified Sampling | #2290 | p_in -1.3%, p_tan -0.8% |
| Cosine T_max=150 | #2251 | p_in -0.7%, p_oodc -1.1%, p_tan -0.8% |
| Wake deficit feature | #2213 | p_in -4.1%, p_re -1.7% |
| TE coordinate frame | #2207 | p_in -5.4% |
| Pressure-first deep | #2155 | p_in -4.8% |
| DCT frequency loss | #2184 | p_re -1.5%, p_tan -0.3% |
| Gap-stagger spatial bias | #2130 | p_tan -3.0% |
| Residual prediction | #1927 | p_oodc -4.7%, p_tan -1.9% |

### Round 30 Unassigned Ideas (from RESEARCH_IDEAS_2026-04-09_ROUND30.md)

| Priority | Slug | Target | Key bet |
|----------|------|--------|---------|
| 1 | `fv-cell-area-loss-weight` | all | **ASSIGNED to nezuko (#2297)** |
| 2 | `gmse-gradient-loss` | p_tan, p_in | **ASSIGNED to thorfinn (#2298)** |
| 3 | `shortest-vector-feature` | p_tan, p_oodc | FVF-paper: 2D vector to nearest foil surface |
| 4 | `did-feature` | p_oodc, p_re | Streamwise directional integrated distance |
| 5 | `q-criterion-proxy-feature` | p_tan, p_oodc | DSDF-curl × freestream as vortex zone indicator |
| 6 | `wall-layer-bin-feature` | p_in, p_tan | Re-scaled y+ zone bins (sublayer/buffer/log/wake) |
| 7 | `bernoulli-residual-feature` | p_in, p_re | Thin-airfoil Cp estimate as input for residual correction |

Round 31 ideas being generated by researcher-agent (bold/paradigm-level, per human directive).

### What's Exhausted (DO NOT REVISIT)

- **Architecture replacements**: GNOT, Galerkin, Hierarchical, FactFormer, DeepONet, INR — all 30-60% worse
- **Cosine T_max sweep**: T_max=140, 150, 160 tried. T_max=150 is optimal.
- **Slice routing perturbations**: Logit noise, RBF kernel, decoupled tandem — all null/negative
- **Chord-position features**: Chord fraction, surface arc-length PE — all redundant with TE coord frame
- **Augmentation annealing**: All forms hurt OOD metrics
- **Asymmetric surface loss**: Hard-node mining already captures suction difficulty
- **ZCA/PCA whitening**: Near-singular covariance (cond# ~7.9B) makes full decorrelation harmful
- **SRF FiLM/flow-regime conditioning**: Redundant with backbone adaLN
- **Pressure gradient aux head**: FD gradients on unstructured mesh too noisy
- **Foil role embeddings**: Fragile, not robust to config changes
- **Per-foil target whitening**: Structurally hurts p_tan
- **MoE FFN routing**: Hard dispatch starves experts in small-dataset regime
- **Per-head K/V projections**: Shared K/V is load-bearing OOD regularization
- **GNN boundary layer**: Disrupts backbone-to-SRF feature distribution
- **SE(2) canonicalization**: Stats mismatch + DSDF inconsistency
- **All ensemble/distillation approaches**: Head-level diversity insufficient; snapshot cyclic fails
- **Synthetic data (NeuralFoil)**: Geometric inconsistency corrupts geometry-pressure learning
- **Optimizer variants**: SAM, Lookahead, SWA, SOAP, Muon — all worse than Lion+EMA+cosine
- **Inter-foil coupling approaches** (attention, GALE, cross-DSDF, GNN, FNO): 5 failures confirmed
- **Generative/flow-matching surface**: Near-deterministic problem, adds noise to delta function
- **DSDF-derived angle features**: Redundant with model's ability to compose existing inputs
