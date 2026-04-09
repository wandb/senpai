# SENPAI Research State

- **Date:** 2026-04-09 05:30 UTC
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
| nezuko | #2314 | **SE Channel Attention on Slice Tokens** | WIP (NEW, ROUND32, ARCH) |
| thorfinn | #2298 | **GMSE Gradient-Weighted Pressure Loss — weight by local ∇p magnitude** | WIP (BOLD) |
| alphonse | #2312 | **GradNorm Adaptive Loss Weighting — auto-balance surface/volume losses** | WIP (NEW, ROUND32, LOSS) |
| fern | #2311 | **Condition Token Injection — dedicated flow-condition embedding pathway** | WIP (NEW, ROUND32) |
| askeladd | #2308 | **Auxiliary AoA Head — explicit AoA decoding (analogous to Re head PR #780)** | WIP (NEW, ROUND32) |
| frieren | #2304 | **Shortest Vector Feature — 2D displacement to nearest foil surface (FVF)** | WIP (NEW, PHYSICS) |
| tanjiro | #2315 | **Tandem Curriculum Ramp — smooth linear ramp instead of hard cutoff** | WIP (NEW, ROUND32, TRAINING) |
| edward | #2313 | **Multi-Scale Intermediate Skips — FPN-style output from all TransolverBlocks** | WIP (NEW, ROUND32, ARCH) |

## PRs Ready for Review

None.

## Latest Reviews (2026-04-09 04:45)

### PR #2302 (fern, Circulation Lift Feature) — CLOSED ❌ (2026-04-09)
- p_oodc +6.1%, p_re +3.2%. Γ ≈ Re×sin(2α) is a deterministic function of existing inputs — redundant. OOD regressions from misleading extrapolation.

### PR #2305 (nezuko, DID Streamwise Feature) — CLOSED ❌ (2026-04-09)
- All metrics worse: p_in +5.2%, p_oodc +2.1%, p_tan +2.2%, p_re +0.5%. Streamwise position redundant with existing (x,y) + AoA + DSDF.

### PR #2303 (askeladd, Wake Centerline SDF) — CLOSED ❌ (2026-04-09)
- p_in +4.7%, p_tan +3.6% (regressions). p_re -2.0%, p_oodc -0.4%. Net negative. Redundant with wake_deficit_feature (PR #2213). Fixed turbulent spreading model too rigid.

### PR #2300 (tanjiro, Mirror Symmetry Augmentation) — CLOSED ❌ (2026-04-09)
- All metrics worse: p_in +19.7%, p_oodc +9.9%, p_tan +5.3%, p_re +5.2%. Mesh asymmetry breaks the theoretical y-reflection symmetry. Dead end.

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

### Round 31-32 In-Flight Experiments (8 GPUs)
| Student | PR | Direction | Target |
|---------|-----|-----------|--------|
| nezuko | #2314 | **SE on Slice Tokens** — Squeeze-Excite recalibration inside PhysicsAttention | all |
| thorfinn | #2298 | **GMSE Gradient-Weighted Pressure Loss** — weight by ∇p magnitude | p_tan, p_in |
| alphonse | #2312 | **GradNorm Adaptive Loss Weighting** — gradient-norm-based surface/vol balancing | all |
| fern | #2311 | **Condition Token Injection** — additive condition MLP embedding (Unisolver-inspired) | p_oodc, p_re |
| askeladd | #2308 | **Auxiliary AoA Head** — explicit AoA decoding, penultimate block pool | p_tan, p_oodc |
| frieren | #2304 | **Shortest Vector Feature** — 2D displacement to nearest foil surface (FVF) | p_tan, p_oodc |
| tanjiro | #2315 | **Tandem Curriculum Ramp** — smooth epoch 10-30 ramp instead of hard cutoff | p_tan |
| edward | #2313 | **Multi-Scale Intermediate Skips** — FPN-style aggregation from TransolverBlocks | all |

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

| Priority | Slug | Target | Status |
|----------|------|--------|--------|
| 1 | `fv-cell-area-loss-weight` | all | ASSIGNED → nezuko pivoted to #2305 DID |
| 2 | `gmse-gradient-loss` | p_tan, p_in | **ASSIGNED to thorfinn (#2298)** |
| 3 | `shortest-vector-feature` | p_tan, p_oodc | **ASSIGNED to frieren (#2304)** |
| 4 | `did-feature` | p_oodc, p_re | **ASSIGNED to nezuko (#2305)** |
| 5 | `q-criterion-proxy-feature` | p_tan, p_oodc | **ASSIGNED to tanjiro (#2307)** |
| 6 | `wall-layer-bin-feature` | p_in, p_tan | Available |
| 7 | `bernoulli-residual-feature` | p_in, p_re | Related to #2299 (Bernoulli loss variant) |

### Round 32 Queued Ideas (from RESEARCH_IDEAS_2026-04-09_ROUND32.md)

| Priority | Slug | Target | Key bet |
|----------|------|--------|---------|
| 1 | `aux-aoa-head` | p_tan, p_oodc | Auxiliary AoA prediction head (analogous to successful Re head PR #780) |
| 2 | `gqa-2-groups` | all | Grouped Query Attention: 2 K/V groups (middle ground MQA↔MHA) |
| 3 | `condition-token-injection` | p_oodc, p_re | Global condition tokens via additive embedding (Unisolver-inspired) |
| 4 | `gradnorm-adaptive-weighting` | all | GradNorm auto-balancing of surface/volume loss weights |
| 5 | `asymmetric-quantile-cp-loss` | p_in, p_tan | Pinball loss tau=0.65 for suction-side Cp accuracy |
| 6 | `panel-method-cp-feature` | p_in, p_tan | Inviscid Cp as physics baseline input feature |
| 7 | `multi-scale-intermediate-skips` | all | FPN-style skip connections from intermediate TransolverBlocks |
| 8 | `tandem-curriculum-ramp` | p_tan | Smooth linear ramp (epochs 10-30) instead of abrupt reintroduction |

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
- **Mirror symmetry augmentation**: Mesh is asymmetric — y-reflection creates inconsistent mesh-to-flow pairings (p_in +19.7%, all metrics worse)
- **Synthetic data (NeuralFoil)**: Geometric inconsistency corrupts geometry-pressure learning
- **Optimizer variants**: SAM, Lookahead, SWA, SOAP, Muon — all worse than Lion+EMA+cosine
- **Inter-foil coupling approaches** (attention, GALE, cross-DSDF, GNN, FNO): 5 failures confirmed
- **Generative/flow-matching surface**: Near-deterministic problem, adds noise to delta function
- **DSDF-derived angle features**: Redundant with model's ability to compose existing inputs
- **DID streamwise position feature**: Redundant with existing (x,y) + AoA + 8 DSDF channels. AoA-dependent projection creates unhelpful entanglement
- **Wake centerline SDF features**: Redundant with wake_deficit_feature (PR #2213). Fixed spreading model too rigid; 4 zero channels for single-foil create optimization tension
- **GQA K/V groups**: n_head=3, making GQA-2 impossible (3%2≠0). GQA-3=MHA, already confirmed harmful
- **Circulation lift feature (Kutta-Joukowski Γ)**: Γ ≈ Re×sin(2α) is redundant with existing inputs; extrapolates worse under distribution shift (p_oodc +6.1%, p_re +3.2%)
- **Bernoulli residual loss**: 2 iterations, never beats baseline on p_in/p_tan. Interferes with main L1 supervision despite p_re gain in v1
- **Continuity PDE loss (∇·u=0)**: Redundant with joint Ux/Uy supervision. SPH approximation too noisy. All metrics regressed
- **Asymmetric quantile/pinball loss**: Diverges catastrophically — multiplicative instability with adaptive weight stack (surf×tandem×hard-node×PCGrad)
