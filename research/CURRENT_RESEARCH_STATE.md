# SENPAI Research State

- **Date:** 2026-04-08 20:45 UTC
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

## Student Status (2026-04-08 20:45 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| thorfinn | #2280 | **Snapshot Ensemble — cyclic cosine LR with prediction averaging** | WIP |
| fern | #2284 | **Heteroscedastic Loss — learned per-node uncertainty weighting** | WIP |
| askeladd | #2282 | **Point Cloud MixUp — linear interpolation data augmentation** | WIP |
| frieren | #2283 | **Wider SRF Head — surface_refine_hidden 192→384** | WIP |
| tanjiro | #2287 | **Effective AoA Aft Feature — thin-airfoil downwash correction (k sweep)** | WIP (NEW) |
| edward | #2286 | **Velocity Angle Feature — per-node local incidence angle from DSDF gradient** | WIP (NEW) |
| nezuko | #2279 | **Ensemble Knowledge Distillation — soft targets from 16-seed ensemble** | WIP |
| alphonse | #2285 | **Deeper Backbone — 4 Transolver blocks (n_layers 3→4)** | WIP |

## PRs Ready for Review

None.

## Latest Reviews (2026-04-08 20:45)

### PR #2281 (edward, Multi-Head SRF Ensemble) — CLOSED ❌
- All metrics regressed or flat: p_in +2.2%, p_tan +2.2%, p_oodc +0.5%, p_re -0.2% (noise).
- Root cause: Backbone diversity (not head diversity) drives real ensemble benefit. Heads converge to similar solutions from shared features.

### PR #2273 (tanjiro, Geometry Consistency Self-Distillation) — CLOSED ❌
- All metrics regressed: p_oodc +3.8%, p_tan +1.9%, p_re +1.4%, p_in flat.
- Root cause: Consistency loss only active for ~7 of 147 epochs (EMA starts at epoch 140). Existing augmentation suite provides larger perturbations than sigma=0.005 jitter.

### PR #2275 (alphonse, NeuralFoil Synthetic Data Flooding) — CLOSED ❌
- All 4 metrics regressed: p_in +15.8%, p_oodc +24.9%, p_tan +3.7%, p_re +13.9%.
- Root cause: Geometric inconsistency — synthetic samples use template mesh positions (NACA-A geometry) with NeuralFoil Cp for different NACA-B. Model sees contradictory geometry-pressure signals. 30% synthetic dilutes real data heavily.

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
| fern | #2284 | **Heteroscedastic Loss** — learned per-node variance for pressure loss | p_in, p_tan |
| askeladd | #2282 | **Point Cloud MixUp** — linear interpolation data augmentation | p_oodc, p_re |
| frieren | #2283 | **Wider SRF Head** — surface_refine_hidden 192→384, capacity increase | p_in, p_tan |
| tanjiro | #2287 | **Effective AoA Aft Feature** — thin-airfoil downwash correction, k sweep | p_tan, p_re |
| edward | #2286 | **Velocity Angle Feature** — per-node local incidence from DSDF gradient | p_tan, p_in |
| nezuko | #2279 | **Ensemble Knowledge Distillation** — soft targets from 16-seed ensemble | p_oodc, p_re |
| alphonse | #2285 | **Deeper Backbone** — n_layers 3→4, backbone depth capacity test | all metrics |

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
- **MAE surface pretraining**: Catastrophic at small model scale (3-block, 192-dim). Pretraining pushes backbone into reconstruction basin far from flow prediction (+342-1106%)
- **Tandem difficulty curriculum**: Double-gating with tandem_ramp. Withholding tandem diversity during critical early training hurts (+3.9-9.1%)
- **Surface arc-length PE**: Angle-sort ordering unreliable, zero features for volume nodes confuse attention, redundant with TE coord frame (+1.4-7.7%)
- **Sample-level reweighting**: Focal loss, OHNM — over-correction on top of PCGrad
- **Optimizer variants**: SAM, Lookahead, SWA, SOAP, Muon — all worse than Lion+EMA+cosine
- **NeuralFoil synthetic data flooding**: Geometric inconsistency (template mesh positions ≠ NACA encoding) corrupts geometry-pressure learning. Panel-method Cp errors near stagnation. 30% synthetic dilutes real data (+3.7-24.9%)
- **Multi-head SRF ensemble**: Head-level diversity insufficient — backbone diversity is the real ensemble mechanism. 3 heads converge to similar solutions from shared features (+0.5-2.2%)
- **Geometry consistency self-distillation**: Mean Teacher on jittered mesh. Consistency loss only active for ~7 epochs (EMA timing). Existing augmentation already provides larger perturbations (+0.1-3.8%)

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
| `neuralfoil-synthetic-flood` | p_in, p_oodc, p_re | **CLOSED** ❌ — geometric inconsistency, +3.7-24.9% |
| `mae-surface-pretrain` | p_oodc, p_tan | **ASSIGNED to askeladd (#2276)** |

### Remaining Unassigned Ideas (for Round 29+)
| Priority | Slug | Target | Key bet |
|----------|------|--------|---------|
| 1 | `tandem-difficulty-curriculum` | p_tan | **ASSIGNED to frieren (#2277)** |
| 2 | `surface-arclen-pe` | p_tan, p_in | **ASSIGNED to fern (#2278)** |

### Round 28+ Additional Assignments
| Slug | Target | Status |
|------|--------|--------|
| `snapshot-ensemble` | p_oodc, p_re | **ASSIGNED to thorfinn (#2280)** |
| `multi-head-srf` | p_oodc, p_re | **ASSIGNED to edward (#2281)** |
| `pointcloud-mixup` | p_oodc, p_re | **ASSIGNED to askeladd (#2282)** |
| `wider-srf-head` | p_in, p_tan | **ASSIGNED to frieren (#2283)** |
| `heteroscedastic-loss` | p_in, p_tan | **ASSIGNED to fern (#2284)** |
| `ensemble-distillation` | p_oodc, p_re | **ASSIGNED to nezuko (#2279)** |
| `deeper-backbone` | all metrics | **ASSIGNED to alphonse (#2285)** |

**Assignment priority for next idle students (Round 29 ideas — researcher-agent generated):**

| Priority | Slug | Target | Key bet |
|----------|------|--------|---------|
| 1 | `vel-angle-mag-feature` | p_tan, p_in | **ASSIGNED to edward (#2286)** |
| 2 | `effective-aoa-aft-feature` | p_tan, p_re | **ASSIGNED to tanjiro (#2287)** |
| 3 | `chord-fraction-feature` | p_in, p_tan | Chord-wise position ∈[0,1] for each node. Gives SRF explicit "where on chord" signal. No ordering needed (unlike arc-length PE). |
| 4 | `cp-target-normalization` | p_re, p_oodc | Predict Cp instead of raw p. Re-invariant target normalization. Directly targets p_re regression. |
| 5 | `re-stratified-sampling` | p_re, p_oodc | 2× weight for extreme-Re samples via static WeightedRandomSampler. |
| 6 | `stagnation-pressure-feature` | p_in, p_re | q_inf = 0.5*Umag² as input channel. Bernoulli baseline as feature (not loss constraint). |
| 7 | `lowrank-pressure-loss` | p_tan, p_in | SVD penalty on surface pressure: penalize energy beyond rank-5. Orthogonal to DCT freq loss. |
| 8 | `flowdir-anisotropic-norm` | p_oodc, p_re | Rotate (x,y) by -AoA to flow-aligned frame. Different from SE(2) failure: known rotation, DSDF untouched. |
| 9 | `logre-pressure-scaling` | p_re | Normalize pressure residuals by log(Re). Milder than Cp normalization. |
| 10 | `tandem-topo-feature` | p_tan | KD-tree OOD proximity feature — distance to nearest training configs in (gap, stagger, AoA, NACA) space. |

Full details: `/research/RESEARCH_IDEAS_2026-04-08_ROUND29.md`
