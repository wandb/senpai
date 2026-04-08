# SENPAI Research State

- **Date:** 2026-04-08 11:00 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training & Architecture Improvements

## Current Baseline

### Single-Model Baseline (PR #2251, +Cosine T_max=150, 2-seed)

| Metric | 2-seed avg | Target to beat |
|--------|-----------|----------------|
| **p_in** | **11.891** | < 11.89 |
| **p_oodc** | **7.561** | < 7.56 |
| **p_tan** | **28.118** | < 28.12 |
| p_re | 6.364 | < 6.36 |

**Latest merge:** PR #2251 (thorfinn) — Cosine T_max=150. p_in -0.7%, p_oodc -1.1%, p_tan -0.8%. p_re regressed +1.0%.

⚠️ **p_re = 6.364 is a regression vs PR #2213 baseline (6.300).** Future experiments should target p_re recovery.

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

Single-model now beats ensemble on p_in (11.891 vs 12.1) and p_tan (28.118 vs 29.1). Ensemble still leads on p_oodc and p_re.

## Student Status (2026-04-08 10:00 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| thorfinn | #2267 | Pressure Gradient Aux Head: dp/dx, dp/dy auxiliary supervision | WIP |
| fern | #2270 | **BOLD: SE(2) Canonicalize — chord-aligned coordinate frame preprocessing** | WIP (NEW) |
| askeladd | #2265 | Per-head K/V projection in Physics_Attention_Irregular_Mesh | WIP |
| frieren | #2269 | GNN Boundary Layer: local mesh message-passing on surface/near-wall nodes | WIP (NEW) |
| tanjiro | #2262 | Foil Role Embedding (v2): fix boundary_id + T_max=150 | WIP (sent back) |
| edward | #2261 | Per-Foil Target Whitening (v2): fore-foil-only + T_max=150 | WIP (sent back) |
| nezuko | #2271 | **BOLD: Flow Matching Surface Head — generative pressure prediction (AlphaFold3-inspired)** | WIP (NEW) |
| alphonse | #2268 | MoE FFN Last Block: tandem-specialized FFN expert in final TransolverBlock | WIP |

## PRs Ready for Review

None.

## Latest Reviews (2026-04-08 10:00)

### PR #2264 (frieren, asymmetric-surface-loss) — CLOSED ❌
- All 4 metrics regressed vs current baseline: p_in +3.3%, p_oodc +2.3%, p_tan +1.9%, p_re +1.7%
- Root cause: asinh transform reduces suction/pressure-side separation; hard-node mining already captures suction difficulty; pred-based side classification is noisy early in training.

### PR #2255 (askeladd, aug-annealing) — CLOSED ❌ (3 trials total)
- No variant beats current baseline on any metric. Best trial (selective AoA stop): p_in=11.914, p_oodc=7.775, p_tan=28.290, p_re=6.414 — all worse than #2251.
- Structural trade-off: clean fine-tuning helps p_in in isolation but augmentation provides essential OOD regularization even at low LR. Cannot be resolved with cutoff tuning.

### PR #2263 (alphonse, attn-logit-noise) — CLOSED ❌ (prior session)
- p_in +3.9%, p_tan +1.0%, p_re +1.3% vs baseline. Confirmed: slice routing is NOT the tandem failure driver.

### PR #2262 (tanjiro, foil-role-embed) — REQUEST CHANGES (sent back)
- Consistent p_oodc improvement, seed 42 p_in=11.76 (beats baseline). Fix: use boundary_id (not saf_norm) for foil identity + T_max=150.

### PR #2261 (edward, per-foil-whiten) — REQUEST CHANGES (sent back)
- p_in=-1.6%, p_oodc=-2.1%, p_re=-1.0% vs baseline. But p_tan=+2.8%. Fix: fore-foil-only whitening (don't normalize aft-foil targets).

## Most Recent Research Direction from Human Researcher Team

### Issue #1860 — ACTIVE directive (Morgan McGuire)
**"Think bigger — radical new full model changes and data aug and data generation, not incremental tweaks."**

Acknowledged and pivoting. Round 27 will include bold architectural additions (GNN boundary layer), data augmentation innovation (geometry consistency distillation), and novel module-level approaches. Researcher-agent running now to generate next batch of bold ideas.

## Current Research Focus and Themes

### Round 27 Active Experiments (8 GPUs occupied)

| Student | PR | Direction | Target |
|---------|-----|-----------|--------|
| thorfinn | #2267 | Physics: Pressure gradient auxiliary head (dp/dx, dp/dy) | p_tan, p_in |
| fern | #2266 | Input rep: ZCA spectral whitening | p_oodc, generalization |
| askeladd | #2265 | Architecture: Per-head K/V projection | p_tan |
| frieren | #2269 | **BOLD: GNN Boundary Layer — local mesh message-passing on surface/near-wall nodes** | p_tan, p_in |
| tanjiro | #2262 | Representation: Foil role embedding (v2) | p_oodc, p_in |
| edward | #2261 | Normalization: Fore-foil-only target whitening (v2) | p_in, p_oodc |
| nezuko | #2260 | Conditioning: FiLM SRF on flow regime (AoA/Umag) | p_tan, p_oodc |
| alphonse | #2268 | MoE FFN Last Block: tandem-specialized FFN | p_tan |

### Key Mechanistic Insights from Round 26-27 Reviews

1. **Slice routing is NOT the tandem failure driver** (attn-logit-noise #2263 confirmed). Input feature limitations more likely.
2. **Hard-node mining already captures suction difficulty** (asymmetric loss #2264 confirmed). Physics-based node weighting redundant when error-based mining is active.
3. **Augmentation is load-bearing at low LR** (aug-annealing #2255 across 3 trials). Cannot trade OOD robustness for ID precision via cutoff — the regularization effect persists throughout training.
4. **Per-foil whitening works for fore-foil** — p_in (-1.6%) and p_oodc (-2.1%). Aft-foil must NOT be normalized.
5. **Foil identity embedding is promising** — consistent p_oodc improvement, seed 42 beats baseline p_in.

### GNN Boundary Layer — Rationale for Bold Pivot

The new frieren assignment (PR #2269) is a genuine architectural departure:
- Previous inter-foil coupling: global attention (fore-aft cross-attention, GALE) — all failed
- GNN boundary layer: LOCAL message passing along wall nodes (2 GraphSAGE rounds, k=4 volume neighbors)
- Motivated by B-GNNs (arXiv:2503.18638, 2025) — 85% error reduction on airfoil meshes via local GNN
- Operates AFTER Transolver backbone, BEFORE SRF head — additive, not replacement
- Inductive bias: boundary layer physics is local propagation, not global attention

### What Works (confirmed and merged)

| Direction | PR | Impact |
|-----------|-----|--------|
| Cosine T_max=150 | #2251 | p_in -0.7%, p_oodc -1.1%, p_tan -0.8% |
| Wake deficit feature | #2213 | p_in -4.1%, p_re -1.7% |
| TE coordinate frame | #2207 | p_in -5.4% |
| Pressure-first deep | #2155 | p_in -4.8% |
| DCT frequency loss | #2184 | p_re -1.5%, p_tan -0.3% |
| Gap-stagger spatial bias | #2130 | p_tan -3.0% |
| PCGrad 3-way | — | OOD separation |
| Residual prediction | #1927 | p_oodc -4.7%, p_tan -1.9% |

### What's Exhausted (DO NOT REVISIT)

- **Architecture replacements**: GNOT, Galerkin, Hierarchical, FactFormer, DeepONet, INR — all 30-60% worse
- **Cosine T_max sweep**: T_max=140, 150, 160 tried. T_max=150 is optimal.
- **Slice routing perturbations**: Logit noise, RBF kernel, decoupled tandem — all null/negative
- **Augmentation annealing**: Hard cutoff at epoch 120 or 140, selective AoA stop — all fail OOD
- **Asymmetric surface loss**: Physics-based node weighting redundant given hard-node mining
- **ZCA/PCA whitening**: Near-singular covariance (cond# ~7.9B) makes full decorrelation harmful (+8-23% regression)
- **SRF FiLM conditioning**: Flow-regime (Re, AoA) modulation on SRF redundant with backbone adaLN
- **Sample-level reweighting**: Focal loss, OHNM — over-correction on top of PCGrad
- **Optimizer variants**: SAM, Lookahead, SWA, SOAP, Muon — all worse than Lion+EMA+cosine

## Potential Next Research Directions (Round 28+)

### In-Flight (Round 25 BOLD ideas)
| Priority | Slug | Target | Status |
|----------|------|--------|--------|
| 1 | `gnn-boundary-layer` | p_tan, p_in | **ASSIGNED to frieren (#2269)** |
| 2 | `cnf-surface-pressure` | p_tan, p_oodc | **ASSIGNED to nezuko (#2271)** |
| 3 | `fno-inter-foil-coupling` | p_tan | Available — 1D FNO spectral convolution in tandem gap region |
| 4 | `geometry-consistency-distill` | p_oodc | Available — Mean Teacher on volume-node-jittered views |

### Fresh Bold Ideas (Round 26 BOLD2 — see `RESEARCH_IDEAS_2026-04-08_BOLD2.md`)
| Priority | Slug | Target | Key bet |
|----------|------|--------|---------|
| 1 | `se2-canonicalize` | p_oodc, p_re | **ASSIGNED to fern (#2270)** |
| 2 | `neuralfoil-synthetic-flood` | p_in, p_oodc, p_re | **DATA GENERATION** — 5000+ NeuralFoil synthetic Cp distributions for single-foil augmentation |
| 3 | `tta-aoa-ensemble` | p_oodc, p_re | Inference-only K=5 AoA rotation TTA, zero training risk |
| 4 | `mae-surface-pretrain` | p_oodc, p_tan | Self-supervised backbone initialization via masked surface node reconstruction |
| 5 | `tandem-difficulty-curriculum` | p_tan | Progressive tandem exposure by gap/stagger difficulty |
| 6 | `surface-arclen-pe` | p_tan, p_in | Per-foil arc-length fraction as 2-channel surface feature |

**Assignment priority for next idle students:**
1. `se2-canonicalize` (simple preprocessing, high information value)
2. `neuralfoil-synthetic-flood` (data generation per human directive #1860)
3. `cnf-surface-pressure` (bold architecture: generative surface head)
