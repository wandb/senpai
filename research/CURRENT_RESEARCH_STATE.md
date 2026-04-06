# SENPAI Research State

- **Date:** 2026-04-06 ~14:00 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline

### Single-Model Baseline (PR #2184, DCT Freq Loss w=0.05, 2-seed)

| Metric | 2-seed avg | Target to beat |
|--------|-----------|----------------|
| p_in | **13.21** | < 13.21 |
| p_oodc | **7.82** | < 7.82 |
| **p_tan** | **28.50** | **< 28.50** |
| p_re | **6.45** | < 6.45 |

**Latest merge:** PR #2184 (nezuko) — DCT frequency-weighted auxiliary loss (w=0.05, gamma=2.0, alpha=1.5). Exploits spectral bias theory to force attention to high-frequency leading-edge/TE features. Absolute DCT coefficient difference is numerically stable (unlike failed BSP #2172). W&B: 6yfv5lio (s42, p_tan=28.432), etepxvjc (s73, p_tan=28.572). p_tan -0.3% from prior baseline.

**Key note:** p_in/p_oodc slightly regressed vs prior baseline (PR #2130). All 4 metrics together represent the current Pareto frontier. Priority is p_tan.

**Reproduce current baseline:**
```bash
cd cfd_tandemfoil && python train.py --agent <name> --wandb_name "<name>/baseline-dct-freq" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pcgrad_3way --pcgrad_extreme_pct 0.15 \
  --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --gap_stagger_spatial_bias \
  --dct_freq_loss --dct_freq_weight 0.05 --dct_freq_gamma 2.0 --dct_freq_alpha 1.5
```

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble, Seeds 42-49 + 66-73)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

Note: Current single model (p_tan=28.60) already **BEATS** the 16-seed ensemble (29.1) on p_tan.

## Student Status (~14:00 UTC 2026-04-06)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| fern | #2181 | GEPS Test-Time Low-Rank Adaptation for OOD Tandem | WIP |
| nezuko | #2205 | NOBLE Nonlinear Low-Rank Branches in TransolverBlock FFN (Retry) | WIP |
| alphonse | #2206 | Transolver++ Ada-Temp: per-point adaptive slice temperature + Rep-Slice | WIP |
| thorfinn | #2203 | Muon/Gram-NS Optimizer: orthogonalized gradient updates | WIP |
| frieren | #2199 | Spectral Conditioning of Attention (SCA) to prevent OOD collapse | WIP |
| edward | #2207 | TE Coordinate Frame: trailing-edge-relative input features for wake coupling | WIP |
| tanjiro | #2197 | Geometry-Adaptive Curvature Loss Weighting on Surface Nodes | WIP |
| askeladd | #2202 | Fore-Aft Cross-Attention in AftFoilRefinementHead for Wake Coupling | WIP |

**All 8 students active. Zero idle GPUs.**

## Recently Reviewed

| PR | Student | Experiment | Decision | Key result |
|----|---------|-----------|---------|------------|
| #2201 | edward | Multi-Scale Slice Hierarchy [32,64,96] | **CLOSED** | p_tan +1.8% (29.01 vs 28.50). p_oodc improved -4.6% but primary target regressed. High seed variance (p_tan: 28.41 vs 29.61, Δ=1.20). Fewer slices in early blocks lose gap_stagger_spatial_bias routing resolution. |
| #2200 | alphonse | Local KNN Attention | **CLOSED** | p_tan +14% (32.45 vs 28.50). O(N²) infeasibility for N=120k forced strided anchor fallback; 38% epoch deficit from overhead; uniform anchors miss surface regions. |
| #2190 | nezuko | Laplacian Eigenvector Mesh PE | **CLOSED** | p_tan +3.1% (29.39 vs 28.50), p_oodc +10.3%, p_re +8.4%. Topology-only PE loses spatial coordinates; 16-dim too small vs 32-dim Fourier PE. |
| #2198 | thorfinn | GradNorm Adaptive Loss Weighting | **CLOSED** | p_tan +1.7% (28.979 vs 28.502). GradNorm disrupted PCGrad balance — adaptive weights and gradient surgery are NOT fully orthogonal with EMA. p_in improved -1.0% but primary target regressed. |
| #2195 | askeladd | Inter-Foil Distance Feature in Spatial Bias | **CLOSED** | p_tan +2.4% (29.20 vs 28.50). p_in/p_re improved but primary metric regressed. Per-node distance to foil-2 center overfits spatial routing to training tandem patterns, doesn't transfer to OOD NACA6416. |
| #2191 | thorfinn | SE(2) AoA-Aligned Spatial Bias | **CLOSED** | p_tan +1.8% avg (29.0 vs 28.50). All 4 metrics regressed. AoA ±4° → cos(AoA)≈0.998, rotation is near-identity. Existing aug_full_dsdf_rot already provides invariance. |
| #2183 | frieren | Vorticity Auxiliary Target | **CLOSED** | p_tan +2.5–3.0% (29.20–29.35 vs 28.502). Vorticity auxiliary loss competes with backbone pressure representation. Backbone already captures vorticity implicitly via velocity targets. KNN-based FD targets are noisy on unstructured mesh. Config B improved p_re (-2.8%) but p_tan regressed consistently across all 4 seeds. |
| #2189 | tanjiro | DSDF TTA Feature Alignment | **CLOSED** | p_tan +69% (48.20 vs 28.50). Catastrophic. Double-normalizes DSDF, destroys geometry signal. |
| #2188 | askeladd | MixStyle Tandem Feature Regularization | **CLOSED** | p_tan +18-26%. CFD feature statistics = physics, not nuisance style info. 3rd feature-manipulation failure. |
| #2187 | edward | Normal-Velocity Hard Constraint | **CLOSED** | p_tan +3.0% (29.34 vs 28.50). Multi-foil normal bug + constraint already ~satisfied. |
| #2185 | alphonse | MAE Pretraining (self-supervised geometry init) | **CLOSED** | p_tan +38-67% (2-5x worse). MAE objective conflicts with physics prediction; dataset too small for SSL. |
| #2186 | thorfinn | Panel Cp Residual Target | **CLOSED** | p_tan +349% (catastrophic). Panel solver fails for tandem. |
| #2184 | nezuko | DCT Freq-Weighted Loss (w=0.05) | **MERGED** | p_tan 28.60→28.50 (-0.3%). New baseline. |
| #2182 | tanjiro | Ensemble Distillation (alpha=0.3, 0.5) | **CLOSED** | p_tan +1.6-2.8%. Teacher quality gap. |
| #2175 | askeladd | SWD Tandem Domain Alignment | **CLOSED** | w=0.01 neutral, w=0.05 +3.7%. Distributional diff = real physics. |
| #2180 | edward | Multi-Resolution Hash Grid | **CLOSED** | p_tan +12.2%. Per-sample normalization breaks spatial coherence. |

## Current Research Focus

### Primary target: p_tan = 28.50 → push below 28.0

Single model beats 16-seed ensemble on p_tan (28.50 vs 29.1). More headroom exists — attacking the **NACA6416 representation gap** via input representation, physics constraints, and spectral techniques.

**Confirmed wins (merged into baseline):**
1. `--aft_foil_srf` — dedicated aft-foil SRF head
2. `--aug_gap_stagger_sigma 0.02` — tandem scalar domain randomization
3. `--aug_dsdf2_sigma 0.05` — foil-2 DSDF magnitude aug (p_tan -1.4%)
4. `--pcgrad_3way --pcgrad_extreme_pct 0.15` — 2-way PCGrad gradient surgery (p_tan -2.1%)
5. `--gap_stagger_spatial_bias` — tandem-geometry-aware slice routing (p_tan -3.0%)
6. `--dct_freq_loss --dct_freq_weight 0.05 --dct_freq_gamma 2.0 --dct_freq_alpha 1.5` — DCT spectral auxiliary loss (p_tan -0.3%)

**Active experiments (8 students WIP):**
1. **GEPS Test-Time Adaptation** (fern #2181) — LoRA context params + continuity residual TTA at inference. Zero training change.
2. **NOBLE Nonlinear Low-Rank Branches** (nezuko #2205) — CosNet-activated low-rank residual branches in TransolverBlock FFN layers. Periodic activation physically motivated for pressure fields. Zero-init for safe start. From human team suggestions (issue #1926). Note: PR #2204 was merged prematurely (no student code) — #2205 is the true first run.
3. **Transolver++ Ada-Temp + Rep-Slice** (alphonse #2206) — Per-point adaptive slice temperature (from Transolver++ ICML 2025) replacing global shared temp. Projected per-point offset on top of existing global temp; zero-initialized so baseline-equivalent at epoch 0. Rep-Slice adds Gumbel noise before softmax for sharp discrete routing. Paper reports 62% surface error reduction on aircraft datasets. Highest priority Round 10 idea. Seeds {42, 73}.
4. **TE Coordinate Frame** (edward #2207) — Trailing-edge-relative coordinate features (dx, dy, r for fore-foil TE and aft-foil TE) appended as 6 new input channels. Targets NACA6416 OOD gap: TE location/shape differs significantly from training foils and model has no special TE-relative reference frame. Motivated by GeoMPNN (NeurIPS 2024 ML4CFD Best Student Paper, arXiv 2412.09399) which showed +3–5 OOD score pts from TE frame alone.
5. **Curvature Loss Weighting** (tanjiro #2197) — Per-node curvature-weighted surface loss: `w_i = 1 + alpha * normalize(|kappa_i|)`. Tests alpha={0.5, 1.0, 2.0}. Upweights LE/TE nodes during training; val metric stays uniform.
6. **Muon/Gram-NS Optimizer** (thorfinn #2203) — Replace Lion with Muon optimizer. Applies Newton-Schulz orthogonalization to gradient matrices for 2D+ weight matrices; AdamW for scalar params. lr_muon=0.02, lr_adamw_scalar=3e-4. Previous crash (PR #2006) was a code bug — this retry has correct param group separation. Directly from human researcher team's directive (issue #1926).
7. **Spectral Conditioning of Attention** (frieren #2199) — Learnable diagonal `D = nn.Parameter(torch.ones(n_heads, slice_num))` right-multiplied into attention logits before softmax (~288 params). Applied to all 3 TransolverBlocks. Initialized to identity. Optional condition number regularization via log-variance proxy (`--spectral_attn_conditioning --sac_lambda 0.01`), two seeds {42, 73}.
8. **Fore-Aft Cross-Attention in AftFoilRefinementHead** (askeladd #2202) — Single-head cross-attention (d_attn=64) from aft-foil surface nodes (queries) to fore-foil surface nodes (keys/values) inside the SRF head. Injects direct upstream wake information into the correction head. Zero-init output projection for baseline-equivalent start. Differentiates from failed KNN context (#2127/#2134) by targeting surface nodes only (~300×300 attention vs 15K-node KNN lookup → <1% overhead).

**Key research patterns:**
- **What works:** DSDF magnitude augmentation (foil-2 only), specialized correction heads (aft_srf), gradient surgery (2-way PCGrad), tandem-geometry-aware routing (GSB), geometry-conditioned mechanisms
- **What doesn't work:** Augmentation annealing, foil-1 aug/dropout, fore-foil SRF (4 failures), tandem slice carve-out, 3-way PCGrad, flat-minima-seeking, LR changes, earlier EMA starts, DSDF2 sigma variations, backbone AdaLN, cross-DSDF features, shape similarity bias, differential LR, asymmetric PCGrad, OHEM, wider/deeper SRF, more slices, iterative refinement, tandem surface mixup, BSP spectral loss, foil-1 geometry adapter
- **Confirmed optimal hyperparams:** ema_decay=0.999, ema_start_epoch~140, weight_decay=5e-5, aug_gap_stagger_sigma=0.02, aug_dsdf2_sigma=0.05, lr=2e-4, cosine_T_max=160

## Critical Finding: PCGrad Flag Logic

⚠️ `--pcgrad_3way` in the baseline runs 2-way PCGrad (correct behavior). The flag requires `--disable_pcgrad` to activate true 3-way, which was tested in PR #2147 and FAILED. Baseline is fine as-is.

## Potential Next Research Directions (queue for next idle students)

### Round 5 — Unassigned (from `/research/RESEARCH_IDEAS_2026-04-06_ROUND5.md`)
1. ~~**Normal-Velocity Hard Constraint**~~ → edward #2187
2. ~~**DSDF Test-Time Feature Alignment**~~ → tanjiro #2189
3. ~~**Laplacian Eigenvector Mesh PE**~~ → nezuko #2190
4. **Learned Geometry Tokenizer** — compress foil shape into latent code, inject into backbone.
5. **Stochastic Depth** — randomly drop TransolverBlocks during training. Standard regularizer.
6. **Local KNN Attention** — add local attention alongside global slice attention.
7. **SIREN INR Pressure Decoder** — continuous neural field for pressure prediction. Bold swing.

### Round 6 — Researcher-Agent (2026-04-06) — See `/research/RESEARCH_IDEAS_2026-04-06_ROUND6.md`
1. ~~**Boundary ID 7 Surface Loss Fix**~~ — FALSE ALARM: prepare_multi.py already uses SURFACE_IDS_MULTI=(5,6,7). Comment in train.py:19 is stale.
2. ~~**SE(2) Chord-Aligned Slice Routing**~~ → thorfinn #2191 — DEAD END (p_tan +1.8%)
3. **Hopfield Geometry Memory Bank** — k-NN retrieval: find nearest training geometries at inference, retrieve pressure patterns as SRF prior. Targets NACA6416 distribution shift directly.
4. ~~**Stochastic Depth**~~ → alphonse #2192
5. ~~**Curvature-Conditioned Spatial Bias**~~ → edward #2193 (true Menger curvature, not the existing crude proxy)
6. ~~**Tandem Inter-Foil Distance Feature**~~ → tanjiro #2194
7. ~~**Geometry-Adaptive Curvature Loss Weighting**~~ → askeladd #2196

### Round 8 — Unassigned (from PR #2183 student follow-up analysis)
1. **Vorticity Input Feature** — Pre-compute KNN-based vorticity (ω ≈ curl(v)) from velocity fields and add as per-node input feature (rather than auxiliary loss target). Avoids gradient competition with backbone. Could give the model explicit wake structure information at inference time. Would need ~6th input feature channel.

### Round 7 — Researcher-Agent (2026-04-06) — See `/research/RESEARCH_IDEAS_2026-04-06_ROUND7.md`
1. ~~**Inter-Foil Distance Feature in Spatial Bias**~~ → askeladd #2195 — DEAD END (p_tan +2.4%)
2. ~~**Fore-Aft Cross-Attention in AftFoilRefinementHead**~~ → askeladd #2202
2. ~~**Geometry-Adaptive Curvature Loss Weighting**~~ → tanjiro #2197
3. **Adaptive Boundary Layer Sampling** — Oversample near-wall mesh nodes during training proportional to gradient magnitude. Dense physics there.
4. ~~**GradNorm Adaptive Loss Weighting**~~ → thorfinn #2198
5. **Learned Anisotropic Attention Kernel** — Replace isotropic slice attention with axis-aligned kernel; chord and camber directions have different pressure gradients.
6. ~~**Spectral Conditioning of Attention**~~ → frieren #2199
7. **Transient Conditioning** — Condition model on Reynolds number explicitly; separate normalizations per Re regime.
8. **Hopfield Memory Bank** — k-NN geometry retrieval from training set → pressure prior injection. (Same as Round 6 idea 3, still unassigned.)

### Round 8 — Researcher-Agent (2026-04-06) — See `/research/RESEARCH_IDEAS_2026-04-06_ROUND8.md`
1. **Fore-Aft Cross-Attention in AftFoilRefinementHead** (`fore-aft-crossattn-srf`) — Cross-attend from aft-foil surface nodes to fore-foil hidden states in the SRF head. Directly models the physical wake coupling responsible for p_tan difficulty. Medium-high confidence.
2. **Asinh Scale Progressive Annealing** (`asinh-scale-anneal`) — Anneal asinh scale from 1.5→0.5 during training. Curriculum in prediction space.
3. **Slice Diversity Regularization** (`slice-diversity-reg`) — Gram-matrix orthogonality loss on slice tokens to prevent OOD routing collapse.
4. **Chord-Normalized Coordinates** (`chord-normalized-coords`) — x/c, y/c coordinates as geometry-invariant features.
5. **Wake Deficit Feature** (`wake-deficit-feature`) — Explicit dx/gap, dy/gap from fore-foil TE per node. Makes wake interaction signal explicit.
6. **Kutta Condition Loss** (`kutta-condition-loss`) — Enforce TE pressure continuity as physics constraint.
7. **Tandem-Biased Stochastic Depth** (`tandem-biased-stochastic-depth`) — Conditional drop path: lower for tandem samples to preserve depth for complex interaction.
8. **Iterative SRF (RAFT-style)** (`iterative-srf`) — Run surface refinement head N=3 times, feeding output back as input. Proven in optical flow (RAFT) and protein folding (AlphaFold recycling). Low complexity, high confidence.
9. **Tandem-Conditioned Feature Cross** (`tandem-feature-cross`) — Tandem-specific multiplicative feature interaction before backbone.
10. **Fourier Position Embedding for Spatial Bias** (`fourier-pos-embed`) — Replace raw xy with multi-scale sinusoidal features in spatial_bias MLP.

### Round 10 — Researcher-Agent (2026-04-06) — See `/research/RESEARCH_IDEAS_2026-04-06_ROUND10.md`
1. ~~**Transolver++ Ada-Temp**~~ → alphonse #2206 (per-point adaptive temp + Rep-Slice Gumbel reparameterization)
2. ~~**TE Coordinate Frame**~~ → edward #2207
3. **Arc-Length Surface Loss** (`arclength-surface-loss`) — Reweight surface loss by arc-length element to correct for non-uniform mesh density at LE/TE. ~15 LoC.
4. **Polar Coordinate Bias** (`polar-coord-bias`) — Polar coords relative to foil centroid as spatial_bias MLP inputs. GeoMPNN-motivated.
5. **Domain-Split SRF Norm** (`domain-split-srf-norm`) — Zero-init tandem-specific delta scale/bias in AftFoilRefinementHead LayerNorm.
6. **Attention Register Tokens** (`attention-register-tokens`) — Learnable global register tokens to prevent attention sink formation (arXiv 2309.16588).
7. **Pressure-Conditioned Attn Temp** (`pressure-conditioned-attn-temp`) — Lightweight Ada-Temp variant using spatial_bias output. Only if Idea 1 not in-flight.

### Round 9 — Human Suggestions (Issue #1926) — See `/research/RESEARCH_IDEAS_2026-04-06_ROUND9.md`
1. ~~**NOBLE**~~ → nezuko #2205 (retry; #2204 merged prematurely without student execution)
2. **GeoTransolver/GALE** — Geometry context cross-attention per block.
3. **PirateNets Adaptive Residuals** — Learnable identity-to-transform gating. Prior crash was code bug.
4. **Moon Optimizer** — Muon variant with corrected update order.
5. **mHC Residuals** — Manifold-constrained hyper-connections.

### Human Researcher Directives
- **#1926 (2026-04-06):** Try NOBLE, XSA (retry), Muon/Gram-NS (retry), HyperP, MSA, mHC, PirateNets, Geosolver, HeavyBall variants. Muon retry assigned (#2203). NOBLE assigned (#2204). Researcher-agent generated Round 9 + Round 10 hypotheses for remaining ideas.
- **#1860 (2026-03-27):** Think bigger — radical new full model changes and data aug.
- **#1834 (2026-03-27):** Never use raw data files outside assigned training split.

## Confirmed Dead Ends (Phase 6)

| Direction | PRs | Finding |
|-----------|-----|---------|
| **Multi-Scale Slice Hierarchy [32,64,96]** | **#2201** | **p_tan +1.8% (29.01 vs 28.50). OOD oodc improved -4.6% but primary target regressed. High seed variance on p_tan (Δ=1.20 vs baseline Δ~0.14). Fewer slices in early blocks lose gap_stagger_spatial_bias routing resolution. 96 confirmed optimal across all blocks (PR #2171 also showed 128/144 failed).** |
| **Laplacian Eigenvector Mesh PE** | **#2190** | **p_tan +3.1%, p_oodc +10.3%, p_re +8.4%. Topology-only PE loses spatial coordinates; 16-dim too small vs 32-dim Fourier. Existing walldist+DSDF provide geometry context; Fourier PE provides coordinate info eigenvectors cannot replace.** |
| **GradNorm Adaptive Loss Weighting** | **#2198** | **p_tan +1.7% (28.979 vs 28.502). GradNorm's adaptive scalar weights interfere with PCGrad's gradient direction surgery. EMA smoothing (0.9) insufficient to prevent oscillation. p_in improved but primary target regressed.** |
| Augmentation Annealing | #2152 | p_tan +1.0-2.1%. Constant aug essential for tandem transfer. |
| EMA Start Epoch Earlier (100, 120) | #2151 | Both regress p_tan +3.7-3.8%. Start ~140 optimal. |
| DSDF2 Sigma (0.03, 0.08) | #2150 | σ=0.05 confirmed optimal. |
| Tandem-Slice Carve-Out (K=4,8) | #2131 | Redundant with GSB. |
| Gap/Stagger σ=0 (removal) | #2148 | All metrics worse. |
| Actual 3-Way PCGrad | #2147 | All pct values worse than 2-way. |
| Tail EMA Checkpoint Averaging | #2146 | Null result. |
| Cross-Seed Model Soup | #2142 | Catastrophic. |
| Foil-2 AoA Rotation Aug | #2138 | Target inconsistency. |
| Per-Foil Physics Normalization | #2136 | +5-19% regression. |
| Fore-Foil TE Relative Coords | #2134 | p_tan +2.6%. |
| AftSRF KNN Context Head | #2134,#2127 | +17% overhead → undertrained. |
| Surface Pressure Gradient Aux Loss | #2129 | 3 rounds, diminishing returns. |
| Foil-1 DSDF Magnitude Aug | #2133 | All σ regress p_tan. |
| Tandem DSDF Channel Mixup | #2132 | No geometric diversity. |
| Flat-minima class | #2086,#2094,#2095,#2120,#2137 | ALL DEAD. |
| Loss reformulations | #2112,#2113,#2116 | All worse. |
| Input Feature Noise | #2144 | Catastrophic. |
| DSDF Spatial Dropout | #2143 | Monotonic degradation. |
| Weight Decay 1e-5, 2e-5 | #2145 | 5e-5 confirmed optimal. |
| Learning Rate ±50% (1e-4, 3e-4) | #2149 | lr=2e-4 confirmed optimal. |
| Cosine T_max {140, 180} | #2154 | Both +2.8% p_tan. T_max=160 confirmed optimal. |
| Asymmetric PCGrad | #2158 | p_in +2.7%, all key metrics worse. Symmetric 2-way optimal. |
| Differential LR (mult=2,3) | #2163 | Both regress p_tan. Uniform LR best. |
| Gap/Stagger σ=0.01 | #2140 | Worse than 0.02. |
| Gap/Stagger σ=0.03 | #2153 | Worse: p_tan +3.3%. σ=0.02 confirmed optimal. |
| EMA Decay 0.9995 | #2141 | Regresses with GSB. 0.999 optimal. |
| Reynolds Number Perturbation | #2125 | Null + regression. |
| Tandem Cross-DSDF Features | #2162 | p_tan +4.4%. Hand-crafted features add noise. |
| Foil Shape Similarity Bias (GSB 7D) | #2157 | p_tan +3.7%. Sample-level cosine sim too coarse. |
| Fore-foil SRF (unconditioned) | #2117,#2124 | Worsen p_tan. |
| Aft/Fore-Foil Loss Upweighting | #2121,#2122 | p_oodc mild benefit, p_tan regression. |
| Various Phase 5 architectures | multiple | 5–59% worse. |
| **Backbone-wide AdaLN** | **#2164** | **p_tan +5.9-6.8%. AdaLN disrupts optimized attention routing.** |
| **DSDF-1 Channel Dropout** | **#2156** | **p_tan +5.6-6.3%. Foil-1 channels need exact values.** |
| **Iterative 2-Pass Refinement** | **#2165** | **p_tan +6.6%. 1.3x epoch cost, only 131 epochs. Still converging at wall clock. Correction signal doesn't exist for CFD.** |
| **Tandem Surface Mixup** | **#2167** | **p_tan +5.8-5.9%. Physical inconsistency — aft-foil targets coupled to upstream wake.** |
| **FiLM-Conditioned Fore-Foil SRF** | **#2161** | **p_tan +5.8%. 4th fore-foil SRF failure. Correction norm collapses. Direction exhausted.** |
| **Tandem Pressure Correction MLP** | **#2168** | **p_tan +2.8%. Mixed: p_oodc -2.6%, p_re -3.1% but primary target regressed.** |
| **Wider/Deeper SRF (h=256,384)** | **#2170** | **p_tan +4.0-4.7%. More capacity overfits. h=192 confirmed optimal.** |
| **OHEM Hard Sample Mining** | **#2169** | **p_tan +2.2-2.4%. Redundant with existing 3-layer difficulty system.** |
| **Slice Number 128/144** | **#2171** | **p_tan +3.3/3.8%. More slices = more overfitting, fewer epochs. 96 confirmed.** |
| **BSP Spectral Loss** | **#2172** | **w=0.1 catastrophic collapse, w=0.05 all +3-6%. Spectral bias not the bottleneck.** |
| **Foil-1 Geometry Adapter** | **#2173** | **p_tan +2.1-2.4%. DSDF 4-moment stats too coarse, discard spatial structure.** |
| **Attention Temperature Curriculum** | **#2174** | **p_tan +2.7-4.2%. High initial temp disrupts GSB routing, wastes early epochs.** |
| **Smaller SRF Head (h=128/96)** | **#2178** | **p_tan +3.9-4.5%. h=192 confirmed optimal (full sweep: 96<128<256<384<192).** |
| **Spectral Shaping (k=3 filter)** | **#2176** | **p_tan +2.3% avg. Unstable: s42=28.59 (baseline) vs s73=29.95 (+4.7%).** |
| **Coordinated Tandem Ramp** | **#2177** | **p_tan +2.2% avg. Concurrent schedules interfere during tandem warmup.** |
| **dp/dn=0 Physics Loss (6-seed)** | **#2166** | **p_tan neutral (28.97 vs 28.60, within σ=0.67). Regularizer for p_in/p_re, not p_tan.** |
| **Panel Cp as Input Feature** | **#2179** | **p_tan +3.7%. Single-foil solver lacks tandem interaction. p_oodc/p_re improved.** |
| **Panel Cp Residual Target** | **#2186** | **p_tan +349% (57x worse). asinh mismatch + panel error compounds in tandem. Direction fully exhausted.** |
| **SWD Domain Alignment** | **#2175** | **w=0.01 neutral (+0.5%), w=0.05 all worse (+3.7%). Tandem slice token differences encode real physics — forced alignment counterproductive.** |
| **DCT Frequency-Weighted Loss** | **#2184** | **MERGED. w=0.05 p_tan -0.3% (new baseline). w=0.1 unstable (high seed variance).** |
| **Ensemble Distillation** | **#2182** | **p_tan +1.6-2.8%. Teacher quality gap — ensemble pre-dates GSB/PCGrad, weaker than student.** |
| **Multi-Resolution Hash Grid** | **#2180** | **p_tan +12.2%. Per-sample coord normalization breaks spatial coherence. 1.14M extra params overfit, 20s/epoch overhead.** |
| **MAE Pretraining (SSL geometry init)** | **#2185** | **p_tan +38-67% (2-5x worse). MAE reconstruction conflicts with physics prediction objective. Dataset (1322 samples) too small for SSL benefit. Pretraining wastes 10-20 epochs of supervised budget.** |
| **Normal-Velocity Hard Constraint** | **#2187** | **p_tan +3.0% (29.34 vs 28.50). Multi-foil angle-sorting bug corrupts tandem normals. Constraint already near-satisfied implicitly (|u·n|=0.008 = 0.5% of tangential). Hard constraint removes gradient flexibility.** |
| **DSDF TTA Feature Alignment** | **#2189** | **p_tan +69% (48.20 vs 28.50). Per-sample normalization destroys geometry-specific DSDF info. Double-normalizes with x-standardization.** |
| **MixStyle Tandem Feature Regularization** | **#2188** | **p_tan +18-26% (both configs). Feature statistics encode physics (pressure magnitudes, velocity regimes), not nuisance style. Damage scales with mixing strength.** |
| **⚠️ FEATURE-DISTRIBUTION MANIPULATION** | **#2175,#2189,#2188** | **3 consecutive failures: SWD alignment, raw-input TTA, feature-space MixStyle. ALL catastrophically degrade OOD metrics. Tandem representations are physically meaningful — do NOT perturb at any level.** |
| **SE(2) AoA-Aligned Spatial Bias** | **#2191** | **p_tan avg +1.8% (29.0 vs 28.50). All 4 metrics regressed. AoA range ±4° makes rotation near-identity (cos≈0.998). aug_full_dsdf_rot already provides this invariance.** |
| **Vorticity Auxiliary Target** | **#2183** | **p_tan +2.5–3.0% (29.20–29.35 vs 28.502). Vorticity auxiliary loss competes with backbone pressure representation; backbone already implicitly encodes vorticity via velocity targets (ω=curl(v)). Noisy KNN-FD targets on unstructured mesh add harmful gradients. Config B improved p_re (-2.8%) but primary metric regressed across all 4 seeds.** |
| **Stochastic Depth** | **#2192** | **p_tan +2.1% (29.10 avg). Layer drop regularizer absorbed by EMA + cosine schedule. 3-layer backbone too shallow for drop path benefit (DeiT uses 12+ layers).** |
| **Inter-Foil Distance Feature in Spatial Bias** | **#2195** | **p_tan +2.4% (29.20 avg). Per-node log-distance to foil-2 center overfits spatial routing to training NACA0012 tandem patterns. Seed 73 diverged (29.7). Gap/stagger scalars already capture sufficient tandem config info.** |
| **Curvature-Conditioned Spatial Bias** | **#2193** | **p_tan +2.5% (29.20 avg). Curvature redundant with position — high-κ concentrates at LE/TE, already captured by (x,y). Python loop overhead reduced epoch budget. Only p_in improved (-2.0%).** |

## Ensemble Seed Pool (Complete)

**Total trained: 45 models.** 23-seed evaluation available; defer until single-model improvements land.
