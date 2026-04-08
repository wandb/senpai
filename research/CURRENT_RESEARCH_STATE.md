# SENPAI Research State

- **Date:** 2026-04-08 13:30 UTC
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

## Student Status (2026-04-08 13:30 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| thorfinn | #2280 | **Snapshot Ensemble — cyclic cosine LR with prediction averaging** | WIP (NEW) |
| fern | #2278 | **Surface Arc-Length PE — per-foil chord-wise positional encoding** | WIP (NEW) |
| askeladd | #2276 | **BOLD: MAE Surface Pretrain — masked autoencoder initialization for backbone** | WIP (NEW) |
| frieren | #2277 | **Tandem Difficulty Curriculum — progressive exposure by gap/stagger magnitude** | WIP (NEW) |
| tanjiro | #2273 | **BOLD: Geometry Consistency Self-Distillation — Mean Teacher on augmented mesh** | WIP |
| edward | #2281 | **Multi-Head SRF Ensemble — 3 independent SRF heads, prediction averaging** | WIP (NEW) |
| nezuko | #2279 | **Ensemble Knowledge Distillation — soft targets from 16-seed ensemble** | WIP (NEW) |
| alphonse | #2275 | **BOLD: NeuralFoil Synthetic Data Flooding — single-foil Cp augmentation via neural surrogate** | WIP (NEW) |

## PRs Ready for Review

None.

## Latest Reviews (2026-04-08 13:30)

### PR #2268 (alphonse, MoE FFN Last Block) — CLOSED ❌
- All 4 metrics regressed: p_in +5.3%, p_oodc +6.5%, p_tan +3.3%, p_re +7.4%.
- Root cause: Hard MoE dispatch halves effective data per expert — starves both experts in small-dataset regime.

### PR #2265 (askeladd, Per-head K/V Projection) — CLOSED ❌
- All 4 metrics regressed: p_in +5.9%, p_oodc +18.0%, p_tan +0.7%, p_re +14.4%.
- Root cause: Shared K/V is load-bearing regularization. Per-head K/V destroys OOD generalization.

### PR #2267 (thorfinn, pressure-gradient-aux-head) — CLOSED ❌
- All 4 metrics regressed: p_in +7.1%, p_oodc +11.1%, p_tan +2.1%, p_re +6.1%.

### PR #2262 (tanjiro, foil-role-embed) — CLOSED ❌ (3 iterations)
- Foil role embeddings fragile, not robust to config changes.

### PR #2261 (edward, per-foil-whiten) — CLOSED ❌ (3 iterations)
- Per-foil whitening structurally hurts p_tan.

### PR #2260 (nezuko, FiLM SRF) — CLOSED ❌
- SRF flow-regime conditioning redundant with backbone adaLN.

### PR #2266 (fern, ZCA whitening) — CLOSED ❌
- Near-singular covariance (cond# ~7.9B) makes full decorrelation harmful.

### PR #2264 (frieren, asymmetric-surface-loss) — CLOSED ❌
- Hard-node mining already captures suction difficulty.

### PR #2255 (askeladd, aug-annealing) — CLOSED ❌ (3 trials)
- Structural trade-off: augmentation is load-bearing for OOD metrics.

## Most Recent Research Direction from Human Researcher Team

### Issue #1860 — ACTIVE directive (Morgan McGuire)
**"Think bigger — radical new full model changes and data aug and data generation, not incremental tweaks."**

Acknowledged and pivoting. Round 27 will include bold architectural additions (GNN boundary layer), data augmentation innovation (geometry consistency distillation), and novel module-level approaches. Researcher-agent running now to generate next batch of bold ideas.

## Current Research Focus and Themes

### Round 28 Active Experiments (8 GPUs occupied — ALL BOLD)

| Student | PR | Direction | Target |
|---------|-----|-----------|--------|
| thorfinn | #2280 | **Snapshot Ensemble** — cyclic cosine LR, 3-checkpoint prediction averaging | p_oodc, p_re |
| fern | #2278 | **Surface Arc-Length PE** — per-foil chord-wise sin/cos positional encoding | p_tan, p_in |
| askeladd | #2276 | **BOLD: MAE Surface Pretrain** — masked autoencoder backbone initialization | p_oodc, p_tan |
| frieren | #2277 | **Tandem Difficulty Curriculum** — progressive exposure by gap/stagger magnitude | p_tan |
| tanjiro | #2273 | **BOLD: Geometry Consistency Self-Distillation** — Mean Teacher on jittered mesh | p_oodc |
| edward | #2281 | **Multi-Head SRF Ensemble** — 3 independent heads on shared backbone | p_oodc, p_re |
| nezuko | #2279 | **Ensemble Knowledge Distillation** — soft targets from 16-seed ensemble | p_oodc, p_re |
| alphonse | #2275 | **BOLD: NeuralFoil Synthetic Data Flooding** — single-foil Cp augmentation via neural surrogate | p_in, p_oodc, p_re |

### Key Mechanistic Insights from Rounds 26-27

1. **Slice routing is NOT the tandem failure driver** (attn-logit-noise #2263 confirmed). Input feature limitations more likely.
2. **Hard-node mining already captures suction difficulty** (asymmetric loss #2264 confirmed). Physics-based node weighting redundant when error-based mining is active.
3. **Augmentation is load-bearing at low LR** (aug-annealing #2255 across 3 trials). Cannot trade OOD robustness for ID precision via cutoff.
4. **Shared K/V is essential regularization** (per-head K/V #2265 confirmed). Per-head projections cause catastrophic OOD degradation (+18% p_oodc).
5. **MoE requires more data than we have** (MoE FFN #2268 confirmed). Hard dispatch halves effective data per expert — both experts starved.
6. **SRF conditioning is redundant with adaLN** (FiLM SRF #2260 confirmed). Flow regime already encoded in backbone hidden states.
7. **Full-matrix whitening harmful with near-singular covariance** (ZCA #2266 confirmed). Condition number ~7.9B makes decorrelation destructive.

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
- **Pressure gradient aux head**: FD gradients compete with primary objectives, noisy on unstructured meshes
- **Foil role embeddings**: Fragile, not robust to config changes; saf_norm proxy artifact
- **Per-foil target whitening**: Fore-foil whitening hurts p_tan structurally (3 iterations confirmed)
- **MoE FFN routing**: Hard dispatch halves effective data per expert — both experts starved in small-dataset regime
- **Per-head K/V projections**: Shared K/V is load-bearing regularization; per-head destroys OOD generalization (+18% p_oodc)
- **GNN boundary layer**: Local GNN message-passing disrupts backbone-to-SRF feature distribution; redundant with existing SRF heads (+20-24% regression)
- **SE(2) canonicalization**: Stats mismatch (global frame stats on canonicalized coords) + DSDF gradient inconsistency; TE coordinate frame + AoA augmentation already provide equivalent invariance
- **Flow matching / generative surface head**: CFD pressure is near-deterministic given inputs — generative modeling adds noise to a delta function. 50/50 SRF blend corrupts precise regression predictions (+14-32%)
- **TTA AoA rotation ensemble**: Coordinate rotation ≠ physical AoA change — creates geometry-physics inconsistency with DSDF features. Training augmentation already handles rotation robustness (+12-49%)
- **FNO inter-foil spectral coupling**: 5th inter-foil coupling failure (attention, GALE, cross-DSDF, GNN, FNO all dead). Transolver slice attention already captures inter-foil interactions implicitly
- **Sample-level reweighting**: Focal loss, OHNM — over-correction on top of PCGrad
- **Optimizer variants**: SAM, Lookahead, SWA, SOAP, Muon — all worse than Lion+EMA+cosine

## Potential Next Research Directions (Round 28+)

### All Round 28 BOLD Ideas — Assignment Status
| Slug | Target | Status |
|------|--------|--------|
| `gnn-boundary-layer` | p_tan, p_in | **CLOSED** ❌ — all metrics +6.9% to +24.0% |
| `cnf-surface-pressure` (flow-matching) | p_tan, p_oodc | **CLOSED** ❌ — near-deterministic problem, generative adds noise (+14-32%) |
| `fno-inter-foil-coupling` | p_tan | **CLOSED** ❌ — 5th inter-foil coupling failure, +4.6-13.7% |
| `geometry-consistency-distill` | p_oodc | **ASSIGNED to tanjiro (#2273)** |
| `se2-canonicalize` | p_oodc, p_re | **CLOSED** ❌ — stats mismatch + DSDF inconsistency, +6.6-14.3% |
| `tta-aoa-ensemble` | p_oodc, p_re | **CLOSED** ❌ — coord rotation ≠ AoA change, +12-49% regression |
| `neuralfoil-synthetic-flood` | p_in, p_oodc, p_re | **ASSIGNED to alphonse (#2275)** |
| `mae-surface-pretrain` | p_oodc, p_tan | **ASSIGNED to askeladd (#2276)** |

### Remaining Unassigned Ideas (for Round 29+)
| Priority | Slug | Target | Key bet |
|----------|------|--------|---------|
| 1 | `tandem-difficulty-curriculum` | p_tan | **ASSIGNED to frieren (#2277)** |
| 2 | `surface-arclen-pe` | p_tan, p_in | **ASSIGNED to fern (#2278)** |

**Assignment priority for next idle students:**
1. New ideas from researcher-agent (run when next student becomes idle — idea queue exhausted)
