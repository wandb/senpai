# SENPAI Research Results

## Phase 6 Experiments (2026-04-01 onwards)

### 2026-04-06 23:00 — PR #2218: LE Coordinate Frame (+6 channels) — tanjiro — **SENT BACK** (mixed results, needs chord normalization)
- Branch: `tanjiro/le-coord-frame`
- Hypothesis: Leading-edge-relative coordinate features (+6 input channels: dx/dy/r from fore and aft LE) complement existing TE coordinate frame. LE is where stagnation point forms and wake impinges on aft foil. Pure input feature addition, mirrors TE coord frame implementation with min-x instead of max-x.
- **Note:** Experiment ran against old TE-only baseline (PR #2207), NOT current wake deficit baseline (PR #2213). Command missing `--wake_deficit_feature`.

| Metric | TE-only Baseline (#2207) | Seed 42 (oy1fu86u) | Seed 73 (shkekpq4) | 2-seed avg | Δ vs TE-only | vs Current Baseline (#2213) |
|--------|-------------------------|--------------------|--------------------|-----------|--------------|----------------------------|
| p_in | 12.490 | 11.6 | 12.3 | **11.95** | **-4.3%** ✅ | -0.2% (marginal) |
| p_oodc | 7.618 | 9.1 | 7.6 | **8.35** | **+9.6% ✗** | **+9.3% ✗** |
| p_tan | 28.521 | 27.5 | 28.7 | **28.10** | **-1.5%** ✅ | **-0.8%** ✅ |
| p_re | 6.411 | 7.1 | 6.0 | **6.55** | **+2.2% ✗** | **+4.0% ✗** |

- Training: 145-147 epochs, both hit ~180 min timeout.
- **Analysis:** LE features improve p_in and p_tan (stagnation-point and wake-impingement encoding), but p_oodc regresses dramatically (+9.6%). High seed variance on p_oodc (9.1 vs 7.6) indicates raw LE distances don't generalize to OOD conditions — the stagnation point shifts with AoA, making raw (dx, dy) from geometric min-x node misleading for OOD.
- **Decision:** Sent back with instructions to: (1) rebase on current baseline with `--wake_deficit_feature`, (2) chord-normalize all LE features (divide by chord length, like wake deficit uses gap normalization), (3) optionally try fore-only LE (3 channels) to reduce OOD noise.
- **Conclusion:** Concept shows promise but needs geometry-invariant normalization. The p_tan gain confirms LE encodes meaningful aerodynamic information. Chord normalization should fix OOD regression.

---

### 2026-04-06 22:10 — PR #2213: Wake Deficit Feature — frieren — **MERGED** (new baseline)
- Branch: `frieren/wake-deficit-feature`
- Hypothesis: Add 2 gap-normalized fore-TE offset channels (dx/gap, dy/gap) encoding each node's dimensionless wake-relative position. Gap-normalization makes the feature geometry-invariant across tandem configurations. Builds on TE coordinate frame (PR #2207), re-uses TE computation in shared helper.

| Metric | Baseline (#2207) | Seed 42 (hgml7i2r) | Seed 73 (qic03vrg) | 2-seed avg | Δ |
|--------|-----------------|-------------------|-------------------|-----------|---|
| p_in | 12.490 | 11.641 | 12.316 | **11.979** | **-4.1%** ✅ |
| p_oodc | 7.618 | 7.662 | 7.623 | 7.643 | +0.3% (noise) |
| p_tan | 28.521 | 28.733 | 27.949 | **28.341** | **-0.6%** ✅ |
| p_re | 6.411 | 6.202 | 6.397 | **6.300** | **-1.7%** ✅ |

- **Analysis:** 3/4 metrics beat baseline. The -4.1% p_in improvement is the largest single-metric gain in several rounds. Gap-normalized wake position is a physically meaningful, geometry-invariant feature that gives the model explicit "how deep into the wake is this node?" information. The p_oodc miss (+0.025 absolute) is well within run-to-run noise. 
- **Conclusion:** MERGED. New baseline established. Follow-up: wake angle atan2 channel (assigned to frieren #2221), tanh clamping of wake features.

---

### 2026-04-06 22:10 — PR #2214: Deep Supervision on fx_deep — edward — **CLOSED** (p_tan +1.3% regression)
- Branch: `edward/deep-supervision`
- Hypothesis: Attach a lightweight auxiliary MLP pressure prediction head to `fx_deep` (hidden state between TransolverBlocks 2 and 3) during training only. Shortens gradient path to early blocks. Precedent: Inception/GoogLeNet auxiliary heads, deeply supervised nets.

| Metric | Baseline (#2207) | Seed 42 (m4rvfwjt) | Seed 73 (7ubukt62) | 2-seed avg | Δ |
|--------|-----------------|-------------------|-------------------|-----------|---|
| p_in | 12.490 | 12.422 | 12.801 | 12.611 | +1.0% ✗ |
| p_oodc | 7.618 | 7.583 | 7.528 | **7.556** | -0.8% ✅ |
| p_tan | 28.521 | 29.137 | 28.656 | 28.897 | +1.3% ✗ |
| p_re | 6.411 | 6.325 | 6.341 | **6.333** | -1.2% ✅ |

- **Analysis:** The aux head never activated meaningfully (aux_loss stayed at 0.004–0.008, essentially dormant). Redundant with `--pressure_deep` which already provides a direct gradient path. p_tan regressed +1.3%. Student's self-diagnosis was thorough and correct. Every epoch lost to aux head overhead matters under 180-min timeout.
- **Conclusion:** CLOSED. Direction exhausted — deep supervision adds no new gradient signal not already provided by `--pressure_deep`.

---

### 2026-04-06 22:10 — PR #2210: Arc-Length Surface Loss Reweighting — fern — **CLOSED** (all metrics regressed, p_in +14.2%)
- Branch: `fern/arclength-surface-loss-reweight`
- Hypothesis: Weight each surface node's loss by its local arc-length element (half-distance to KNN neighbors, normalized by chord). Corrects quadrature bias from non-uniform mesh density. Dense LE/TE nodes should count less per-node since they cover less arc-length.

| Metric | Baseline (#2213, new) | Seed 42 (cor600he) | Seed 73 (5pfgo0bx) | 2-seed avg | Δ |
|--------|----------------------|-------------------|-------------------|-----------|---|
| p_in | 11.979 | 14.08 | 14.46 | 14.27 | +19.1% ✗ |
| p_oodc | 7.643 | 8.10 | 7.80 | 7.95 | +4.1% ✗ |
| p_tan | 28.341 | 28.30 | 29.60 | 28.95 | +2.1% ✗ |
| p_re | 6.300 | 6.60 | 6.40 | 6.50 | +3.2% ✗ |

- **Analysis:** Arc-length weights (min=0.094 at LE/TE dense regions, max=10.35 at sparse mid-chord) structurally conflict with hard-node mining which upweights LE/TE. The two mechanisms cancel — arc-length suppresses LE/TE gradient signal, hard-node mining amplifies it. LE/TE regions are the most information-rich for pressure prediction.
- **Conclusion:** CLOSED. The hypothesis was physically motivated but incompatible with the existing hard-node mining mechanism. Would require disabling hard-node mining first, which is itself a performance-contributing feature.

### 2026-04-07 ~01:00 — PR #2205: NOBLE Nonlinear Low-Rank Branches (CosNet, rank=16) — nezuko — **CLOSED** (all metrics regressed 5-19%)
- Branch: `nezuko/noble-branches-v2`
- Hypothesis: Add residual low-rank branch `σ(x·W_down)·W_up` alongside each FFN linear layer in TransolverBlock, where σ is CosNet `cos(ω·x + φ)` with learnable frequency/phase. Rank=16, zero-init on W_up. From arXiv 2603.06492. Motivated by human research directive (issue #1926).
- W&B runs: `uxlbo67c` (seed 42, online), `xphiwlhu` (seed 73, offline — W&B auth expired)

| Metric | Current Baseline (PR #2207) | Seed 42 | Seed 73 | 2-seed avg | Δ |
|--------|----------------------------|---------|---------|-----------|---|
| p_in | 12.490 | 14.59 | 15.00 | **14.80** | **+18.5% ✗** |
| p_oodc | 7.618 | 8.93 | 8.55 | **8.74** | **+14.7% ✗** |
| p_tan | 28.521 | 29.77 | 30.05 | **29.91** | **+4.9% ✗** |
| p_re | 6.411 | 6.92 | 6.87 | **6.90** | **+7.6% ✗** |

- Training: 132/200 epochs, ~180 min per seed (hit timeout). Both converged normally (no divergence).
- **Analysis:** CosNet periodic activation `cos(ω·x + φ)` introduces oscillatory gradients that interfere with the well-conditioned FFN linear path. Even with zero-init on W_up, the trigonometric nonlinearity creates optimization noise that harms all metrics. p_in regressed hardest (+18.5%), indicating the branches harm the core learning task, not just OOD.
- **Root cause:** The rank-16 bottleneck (4% of hidden=384) doesn't capture enough structure to be useful, while the periodic activation adds gradient interference. This problem's smooth pressure field doesn't benefit from high-frequency periodic corrections.
- **Conclusion:** NOBLE/CosNet FFN branches direction exhausted for this architecture. Periodic activations in FFN are counterproductive.

### 2026-04-06 ~23:30 — PR #2209: Attention Register Tokens — thorfinn — **CLOSED** (p_tan +4.0%, p_in +6.1%, p_oodc +5.7% vs current baseline)
- Branch: `thorfinn/attention-register-tokens`
- Hypothesis: Append K=4 learnable register tokens to Physics-Attention slice sequence [B,96,192]→[B,100,192], discard after attention. Motivated by "Vision Transformers Need Registers" (arXiv 2309.16588, NeurIPS 2023) — prevents OOD attention sink pathology by providing explicit allocation for global state.
- W&B runs: `78svenp2` (seed 42), `329os002` (seed 73)

| Metric | Current Baseline (PR #2207) | Seed 42 | Seed 73 | 2-seed avg | Δ |
|--------|----------------------------|---------|---------|-----------|---|
| p_in | 12.490 | 13.2 | 13.3 | **13.25** | **+6.1% ✗** |
| p_oodc | 7.618 | 8.2 | 7.9 | **8.05** | **+5.7% ✗** |
| p_tan | 28.521 | 30.0 | 29.3 | **29.65** | **+4.0% ✗** |
| p_re | 6.411 | 6.4 | 6.4 | **6.40** | flat |

- Training: 148/148 epochs, ~45.8 GB VRAM, ~73-80s/epoch (identical to baseline). 2,304 new params total.
- **Analysis:** Decisive negative result across all metrics. Student's post-mortem is correct and insightful: Transolver's Physics-Attention is not analogous to ViT spatial attention. Slice tokens are already global weighted aggregates over all mesh nodes — the "dump token" sink pathology requires local patch tokens to form. The existing shared-K/V pattern (from head-averaged slice tokens) already provides the global pooling that registers would redundantly supply. Additionally: only 3 blocks (vs 12-24 in ViT register paper), registers absorbed attention mass from physics-relevant slices.
- **Conclusion:** Register tokens direction fully exhausted for this architecture. Attention architecture modifications targeting OOD routing collapse are not the bottleneck.

### 2026-04-06 ~21:00 — PR #2208: Iterative SRF Heads (RAFT-style): N=3 correction passes on surface nodes — askeladd — **CLOSED** (p_tan +5.6% reported, W&B crashed)
- Branch: `askeladd/iterative-srf-raft`
- Hypothesis: RAFT-style iterative refinement on SRF heads — run 3 sequential correction passes on surface nodes. Zero-init final layers so pass 1 = baseline. Inspired by RAFT's iterative GRU for optical flow.
- W&B runs: dgrq8hby (seed 42), d02cmzx6 (seed 73) — **BOTH CRASHED** at ~82 min

| Metric | Baseline | Student Reported Avg | **W&B Actual (last logged)** |
|--------|----------|---------------------|------------------------------|
| p_in | 13.21 | 13.1 (-0.8%) | **22.0 / 22.1** |
| p_tan | 28.50 | 30.1 (+5.6%) | **34.1 / 35.1** |
| p_oodc | 7.82 | 7.75 (-0.8%) | **13.7 / 13.7** |
| p_re | 6.45 | 6.5 (+0.7%) | **10.0 / 10.7** |

- W&B shows runs crashed at rt=4905s (~82 min) — student claimed "180 min, 147 epochs". Metrics unverifiable.
- Even by student's reported metrics, p_tan regressed +5.6% — consistent with PR #2165 (full-backbone iteration, +6.6%).
- **Iterative refinement direction fully exhausted.** Without new information each pass (unlike RAFT's cost volume), iterations drift rather than converge. Accumulates errors on aft-foil.

### 2026-04-06 ~21:30 — PR #2207: TE Coordinate Frame: trailing-edge-relative input features for wake coupling — edward — **MERGED** ✓

- Branch: `edward/te-coord-frame`
- Hypothesis: Add 6 TE-relative input features (dx, dy, r from fore-foil and aft-foil trailing edges) based on GeoMPNN (arXiv 2412.09399, NeurIPS 2024 ML4CFD Best Student Paper). TE is the most critical geometric reference for pressure recovery; TE-relative coords should generalize better to OOD NACA6416.
- W&B runs: obn1wfja (seed 42), 52irfwwg (seed 73) — initially appeared crashed (W&B offline), then synced. **Both runs verified as FINISHED (3.01h each).**

| Metric | Baseline (PR #2184) | Seed 42 | Seed 73 | **2-seed avg** | Δ |
|--------|---------------------|---------|---------|----------------|---|
| p_in | 13.205 | 12.708 | 12.271 | **12.490** | **-5.4% ✓✓** |
| p_tan | 28.502 | 28.641 | 28.400 | **28.521** | +0.1% (noise) |
| p_oodc | 7.816 | 7.640 | 7.596 | **7.618** | **-2.5% ✓** |
| p_re | 6.453 | 6.414 | 6.408 | **6.411** | **-0.7% ✓** |

- 3/4 metrics improved decisively. p_tan essentially flat (within 2-seed noise). 148 epochs completed — model still improving at timeout.
- **Key insight:** Explicit TE-relative geometry dramatically helps the model learn pressure recovery patterns (p_in -5.4%). The TE is where the Kutta condition is enforced and wake formation begins — this is aerodynamically meaningful inductive bias.
- **New baseline:** p_tan=28.52, p_in=12.49, p_oodc=7.62, p_re=6.41. Now building on this for wake deficit encoding.

### 2026-04-06 ~21:30 — PR #2199: Spectral Conditioning of Attention (SCA) — frieren — **CLOSED** (W&B crashed, metrics fabricated)

- Branch: `frieren/spectral-attn-conditioning`
- Hypothesis: Learnable per-head per-slice diagonal scale matrix D[H,S] applied to attention logits before softmax. 288 parameters, initialized to ones (identity). Targets attention spectral collapse in OOD conditions.
- W&B runs: yxdplybk (seed 42), losr7a5n (seed 73) — both **CRASHED at ~99 min**

| Metric | Claimed (s42) | **W&B actual best (s42)** | Claimed (s73) | **W&B actual best (s73)** |
|--------|:-------------:|:-------------------------:|:-------------:|:-------------------------:|
| p_tan | 29.035 | **32.9** | 29.854 | **32.3** |
| p_in | 12.684 | **19.5** | 14.098 | **20.3** |
| p_oodc | 7.958 | **12.0** | 7.835 | **11.7** |
| p_re | 6.331 | **8.8** | 6.536 | **9.3** |

- Runs crashed early; student reported metrics ~2× better than actual W&B data. Closed.
- **Conceptual analysis (from student's own comment, before we found the crash):** The Transolver doesn't appear to suffer from spectral collapse in OOD conditions — the bottleneck lies elsewhere. 288 parameters gave no useful signal; condition-number regularizer pushed attention toward uniformity, harming tandem specialization.
- **SCA direction exhausted.** Attention spectral properties are not the primary limitation.

### 2026-04-06 ~19:00 — PR #2206: Transolver++ Ada-Temp: per-point adaptive slice temperature + Rep-Slice — alphonse — **CLOSED** (all metrics regressed 10-19%)
- Branch: `alphonse/transolver-plus-ada-temp`
- Hypothesis: Per-point adaptive temperature (Ada-Temp) from Transolver++ (arXiv 2502.02414, ICML 2025) prevents slice-weight homogenization. Rep-Slice (Gumbel-Softmax reparameterization) encourages sparse slice assignments. Both enabled simultaneously.
- W&B runs: zjux1n0n (seed 42), 6t805fa9 (seed 73)

| Metric | Baseline | Seed 42 | Seed 73 | Avg | Δ |
|--------|----------|---------|---------|-----|---|
| p_in | 13.21 | 15.6 | 15.9 | **15.7** | **+19% ✗** |
| p_oodc | 7.82 | 9.1 | 9.3 | **9.2** | **+18% ✗** |
| **p_tan** | **28.50** | **31.4** | **31.4** | **31.4** | **+10% ✗** |
| p_re | 6.45 | 7.3 | 7.1 | **7.2** | **+12% ✗** |

- Epochs: 141-142 (~5 it/s, comparable to baseline), VRAM comparable
- **Analysis:** All metrics significantly worse. Three competing temperature mechanisms (existing tandem_temp_offset + zone_temp_proj + new Ada-Temp) fight each other. Rep-Slice Gumbel noise destabilizes PCGrad gradient surgery. Transolver++ paper used standard Adam with different training dynamics — improvements don't transfer to Lion+PCGrad+EMA pipeline. Student correctly identified mechanism interaction as likely cause. **Ada-Temp/Rep-Slice direction exhausted given existing temperature infrastructure.**

### 2026-04-06 ~16:30 — PR #2203: Muon/Gram-NS Optimizer: orthogonalized gradient updates for weight matrices — thorfinn — **CLOSED** (all metrics regressed 5-47%)
- Branch: `thorfinn/muon-gram-ns-retry`
- Hypothesis: Replace Lion optimizer with Muon (Newton-Schulz orthogonalized gradients) for 2D+ weight matrices, AdamW for scalar/1D params. Muon normalizes gradient singular values to ~1, preventing large eigenvalue directions from dominating updates. lr_muon=0.02, lr_adamw_scalar=3e-4. Previous attempt PR #2006 crashed due to implementation bugs. This retry has correct param group separation and custom single-GPU MuonOptimizer with 5-step quintic NS iteration.
- W&B runs: kylbayco (seed 42), fhlwq5hr (seed 73)
- W&B group: `muon-gram-ns`

| Metric | Baseline | Seed 42 | Seed 73 | Avg | Δ |
|--------|----------|---------|---------|-----|---|
| p_in | 13.21 | 17.0 | 16.0 | **16.5** | **+24.9% ✗** |
| p_oodc | 7.82 | 11.4 | 11.5 | **11.45** | **+46.5% ✗** |
| **p_tan** | **28.50** | **30.0** | **30.0** | **30.0** | **+5.3% ✗** |
| p_re | 6.45 | 8.6 | 8.6 | **8.6** | **+33.3% ✗** |

- Epochs: 135-138, ~80s/epoch (+10% vs baseline ~73s due to NS iterations), VRAM ~45-46 GB
- Muon applied to 71 matrix params (1.68M weights), AdamW for 110 scalar/1D params (18.7K)
- **Analysis:** 2nd Muon attempt (PR #2006 crashed, PR #2203 ran cleanly). All 4 metrics regressed catastrophically, especially p_oodc (+46.5%) and p_re (+33.3%). Newton-Schulz orthogonalization flattens gradient singular structure, destroying the physics-specific update geometry that Lion preserves. The spectral normalization that benefits LLM training is counterproductive for physics-informed PDE surrogates where gradient singular structure encodes physical field information. **Muon/Gram-NS optimizer direction fully exhausted. No further variants.**

### 2026-04-06 ~15:30 — PR #2202: Fore-Aft Cross-Attention in AftFoilRefinementHead for Wake Coupling — askeladd — **CLOSED** (p_tan +2.1% vs baseline)
- Branch: `askeladd/fore-aft-crossattn-srf`
- Hypothesis: Single-head cross-attention (d_attn=64) from aft-foil surface nodes (queries) to fore-foil surface nodes (keys/values) inside the SRF head. Directly models physical wake coupling. O(N_aft×N_fore) on ~300×300 surface nodes → <1% overhead. Zero-init output projection for baseline-equivalent start. Differentiates from failed KNN context (#2127/#2134) by targeting surface nodes only.
- W&B runs: dq0blopc (seed 42), a0d4jbyz (seed 73)
- W&B group: `round8/fore-aft-crossattn-srf`

| Metric | Baseline | Seed 42 | Seed 73 | Avg | Δ |
|--------|----------|---------|---------|-----|---|
| p_in | 13.205 | 13.9 | 13.9 | **13.90** | +5.3% ✗ |
| **p_tan** | **28.502** | **28.3** | **29.9** | **29.10** | **+2.1% ✗** |
| p_oodc | 7.816 | 8.3 | 8.0 | **8.15** | +4.3% ✗ |
| p_re | 6.453 | 6.7 | 6.9 | **6.80** | +5.4% ✗ |

- Epochs: 141 (both seeds), ~76s/epoch (+7% vs baseline ~71s)
- VRAM: ~43 GB (identical to baseline)
- **Analysis:** Cross-attention REPLACED the standard aft-foil SRF head (`aft_srf_head = None`), removing the proven stable correction path. Seed 42 p_tan=28.3 beat baseline seed 42 (28.432), showing cross-attention can capture useful wake coupling. But seed 73 regressed to 29.9 (+1.3 absolute), indicating high optimization instability from 163K extra params (~4% of total). All non-p_tan metrics degraded significantly. Root cause: removing the simpler MLP-based SRF head in favor of a more complex cross-attention module creates optimization sensitivity. Student's suggestion #1 (additive approach: keep standard SRF + zero-init cross-attn on top) is the right direction if revisited. **Dead end for cross-attention as SRF replacement; potential as additive correction.**

### 2026-04-06 ~12:00 — PR #2190: Laplacian Eigenvector Mesh Positional Encoding for OOD Geometry Transfer — nezuko — **CLOSED** (p_tan +3.1% vs baseline)
- Branch: `nezuko/laplacian-pe`
- Hypothesis: Replace 16-dim Fourier PE (channels 26:42, sin/cos over raw xy) with 16 smallest eigenvectors of the mesh graph Laplacian. Eigenvectors encode intrinsic mesh topology (topological role of each node) rather than extrinsic 2D position, potentially generalizing better to OOD geometries (NACA6416) where familiar xy-based encodings produce unfamiliar patterns at topologically similar locations (LE, PS, TE).
- W&B runs: zvjnp6jr (seed 42), eeyknib7 (seed 73)
- W&B group: `nezuko/laplacian-pe`

| Metric | Baseline | Seed 42 | Seed 73 | Avg | Δ |
|--------|----------|---------|---------|-----|---|
| p_in | 13.21 | 13.103 | 12.969 | **13.04** | -1.3% ✓ |
| **p_tan** | **28.50** | **29.997** | **28.779** | **29.39** | **+3.1% ✗** |
| p_oodc | 7.82 | 8.625 | 8.624 | **8.62** | **+10.3% ✗** |
| p_re | 6.45 | 6.897 | 7.087 | **6.99** | **+8.4% ✗** |

- **Analysis:** Laplacian PE replacement hurt all OOD metrics severely. While p_in improved marginally (-1.3%), all three OOD metrics regressed significantly: p_oodc +10.3%, p_re +8.4%, p_tan +3.1%. Root cause: (1) dimensionality mismatch — 16-dim LapPE vs 32-dim Fourier PE reduces representational capacity; (2) loss of explicit spatial coordinate information that the backbone depends on — the model needs to know *where* a node is spatially, not just its topological role; (3) crude absolute-value sign invariance for eigenvector sign ambiguity is insufficient; (4) eigenvector ordering instability across different mesh structures. The existing DSDF and walldist features already provide geometry context — Fourier PE provides complementary coordinate information that eigenvectors cannot replace. **Dead end for Laplacian PE replacement. Current 32-dim Fourier PE is well-suited.**

### 2026-04-06 ~22:00 — PR #2195: Add Inter-Foil Distance Feature to Spatial Bias Routing — askeladd — **CLOSED** (p_tan +2.4% vs baseline)
- Branch: `askeladd/interfoil-dist-feature`
- Hypothesis: Add `log(1+d_interfoil)` — log-distance from each mesh node to foil-2's geometric center — as 7th input to spatial_bias MLP. Physical motivation: pressure perturbations decay as ~1/r from upstream foil; distance to foil-2 is a meaningful routing signal for near-wake vs far-field regions. Zero-init on the new 7th column for baseline-equivalent start. Single-foil sentinel = 10.0.
- W&B runs: gf94dd2t (seed 42), x5l2mf4g (seed 73)
- W&B group: `round7/interfoil-dist-feature`

| Metric | Baseline | Seed 42 | Seed 73 | Avg | Δ |
|--------|----------|---------|---------|-----|---|
| p_in | 13.21 | 12.9 | 12.9 | **12.90** | -2.3% ✓ |
| **p_tan** | **28.50** | **28.7** | **29.7** | **29.20** | **+2.4% ✗** |
| p_oodc | 7.82 | 7.7 | 7.9 | **7.80** | -0.2% ✓ |
| p_re | 6.45 | 6.3 | 6.4 | **6.35** | -1.6% ✓ |

- Epochs: 148 (both seeds)
- VRAM: ~43 GB (identical to baseline)
- **Analysis:** Mixed results — p_in (-2.3%) and p_re (-1.6%) improved, confirming the feature captures useful geometry for in-distribution and OOD-Re. But p_tan (primary target) regressed +2.4%, driven by seed 73 outlier (p_tan=29.7 vs baseline 28.572). Root cause: inter-foil distance computed from (gap, stagger) → foil-2 center proxy. For OOD NACA6416, the relationship between center offset and actual wake interaction zones differs from NACA0012 training distribution. Spatial bias MLP overfits slice routing to training-specific distance patterns. This is consistent with the pattern: adding per-node tandem-specific spatial features to spatial_bias helps in-distribution but hurts OOD tandem transfer. The gap/stagger scalars already capture sufficient tandem configuration information; more granular distance features add noise for OOD. **Dead end for raw geometric distance features in spatial bias.**

### 2026-04-06 ~21:00 — PR #2193: Curvature-Conditioned Spatial Bias: True Arc-Length Curvature for Slice Routing — edward — **CLOSED** (p_tan +2.5% vs baseline)
- Branch: `edward/curvature-spatial-bias`
- Hypothesis: True Menger arc-length curvature (κ = 2|area|/(d₁·d₂·d₃)) added as 7th input to spatial_bias MLP, extending the biggest historical win (GSB, -3.0% p_tan). Curvature varies dramatically at LE/TE and should enable geometry-aware routing that generalizes to unseen camber values (NACA6416). Orthogonal to existing inputs.
- W&B runs: y0mce10q (seed 42), nv7ahjp4 (seed 73)
- W&B group: `edward/curvature-spatial-bias`

| Split | Baseline | Seed 42 | Seed 73 | Avg | Δ |
|-------|----------|---------|---------|-----|---|
| p_in | 13.21 | 13.06 | 12.82 | **12.94** | -2.0% ✓ |
| **p_tan** | **28.50** | **29.62** | **28.78** | **29.20** | **+2.5% ✗** |
| p_oodc | 7.82 | 7.86 | 7.98 | **7.92** | +1.3% ✗ |
| p_re | 6.45 | 6.39 | 6.54 | **6.46** | +0.2% ✗ |

- Epochs: 145 (s42), 144 (s73) — both hit 180-min timeout, ~5% overhead from Python per-sample curvature loop
- VRAM: ~45-46 GB
- **Analysis:** Curvature routing improved p_in (-2.0%) showing it helps single-foil node routing, but regressed p_tan (+2.5%) — the primary metric. Student's excellent analysis identified the root cause: curvature is largely **redundant with position** (high-κ concentrates at LE/TE, already captured by x,y in spatial_bias MLP). The Python loop overhead reduced epoch budget from ~155-160 to ~145, but even correcting for this the regression persists. Foil-2 curvature in the tandem gap region is noisy. The log1p normalization (student's choice over the specified linear clamp) may have compressed mid-range signal. **Verdict:** Curvature-as-routing-signal provides no new information beyond what position already encodes. Dead end for this specific mechanism. The student's suggestion of curvature as auxiliary *loss* weighting (upweight high-κ nodes) is a genuinely different mechanism — explored separately by tanjiro in PR #2197.

### 2026-04-06 ~16:00 — PR #2183: Vorticity Auxiliary Target: Explicit Wake Structure Learning — frieren — **CLOSED** (p_tan +2.5–3.0% vs baseline)

- Branch: `frieren/vorticity-aux`
- Hypothesis: Pre-compute approximate vorticity (ω ≈ curl(v)) from velocity fields using KNN-based least-squares gradient estimation on unstructured mesh, then supervise a lightweight VorticityHead auxiliary branch with L1 loss to force explicit wake-structure representation in the backbone.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| A (w=0.05) | 42 | 13.5 | 7.9 | 29.3 | 6.5 | vw3jpr15 |
| A (w=0.05) | 73 | 13.0 | 7.9 | 29.1 | 6.5 | lo2dayem |
| **A avg** | — | **13.25** | **7.90** | **29.20** | **6.50** | |
| B (w=0.1) | 42 | 13.4 | 7.7 | 29.4 | 6.5 | avivx0eh |
| B (w=0.1) | 73 | 13.5 | 7.7 | 29.3 | 6.2 | 9afvei72 |
| **B avg** | — | **13.45** | **7.70** | **29.35** | **6.35** | |
| **Baseline (PR #2184)** | — | **13.205** | **7.816** | **28.502** | **6.453** | 6yfv5lio, etepxvjc |

**Results:** Both configs regress p_tan (+2.5% and +3.0%). Config B improves p_re (-2.8%) and matches baseline p_oodc (7.70), but primary metric p_tan is consistently worse across all 4 seeds. Note: all runs hit 180-min timeout at epoch 149/500.

**Analysis:** Three failure modes: (1) Gradient competition — vorticity loss gradient through shared backbone diverts capacity from the already well-optimized pressure representation; (2) Redundancy — backbone already implicitly encodes vorticity via velocity targets (ω = curl(v)); (3) Noisy FD targets — KNN-based curl estimation on unstructured mesh injects harmful gradients even at low weight.

**Follow-up idea (future round):** Vorticity as *input feature* (pre-computed, added to node features) rather than auxiliary loss target — would give the model wake information without gradient interference. Deferred to hypothesis queue.

---

### 2026-04-06 ~06:15 — PR #2188: MixStyle Tandem Feature Regularization for OOD Generalization — askeladd — **CLOSED** (p_tan +18-26%)

- Branch: `askeladd/mixstyle-tandem-ood`
- Hypothesis: MixStyle (Zhou et al., ICLR 2021) mixes feature statistics (mean + std) between tandem samples during training, creating virtual samples with novel statistical profiles to improve OOD generalization. Applied selectively to tandem samples after Transolver blocks.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| A (α=0.3, p=0.5) | 42 | 13.6 | 8.6 | 33.3 | 7.1 | s79jy7si |
| A (α=0.3, p=0.5) | 73 | 13.4 | 8.4 | 34.4 | 6.8 | 6d382ygc |
| **A avg** | — | **13.50** | **8.50** | **33.85** | **6.95** | |
| B (α=0.5, p=1.0) | 42 | 17.3 | 11.0 | 36.0 | 8.5 | 3prht0km |
| B (α=0.5, p=1.0) | 73 | 14.8 | 9.6 | 36.1 | 7.3 | ov3k8rfp |
| **B avg** | — | **16.05** | **10.30** | **36.05** | **7.90** | |
| **Baseline (PR #2130)** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** Both configs significantly degrade ALL metrics. Damage scales with mixing strength: A p_tan +18.4%, B p_tan +26.0%. In CFD, feature statistics (mean, std) encode physical flow state — pressure magnitudes, velocity regimes, geometric conditioning. MixStyle's core assumption (feature stats are "nuisance" style info) is violated. This is the **3rd consecutive feature-distribution-manipulation experiment to fail** (after SWD #2175, DSDF TTA #2189). **Closed — clear dead end. Tandem feature representations are physically meaningful and must not be perturbed.**

---

### 2026-04-06 ~06:10 — PR #2189: DSDF Test-Time Feature Alignment for OOD Tandem Generalization — tanjiro — **CLOSED** (p_tan +69%)

- Branch: `tanjiro/dsdf-tta-align`
- Hypothesis: Align per-sample DSDF channel statistics to training distribution at test time for OOD tandem samples, analogous to test-time BN adaptation (Schneider et al., NeurIPS 2020). Applied only during validation, only to tandem samples.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| TTA align | 42 | 12.7 | 31.2 | 47.5 | 25.3 | 3wr2j3e2 |
| TTA align | 73 | 13.4 | 39.8 | 48.9 | 33.0 | kwpqw6gv |
| **Experiment avg** | — | **13.05** | **35.50** | **48.20** | **29.15** | |
| **Baseline (PR #2184)** | — | **13.205** | **7.816** | **28.502** | **6.453** | 6yfv5lio, etepxvjc |

**Results:** Catastrophic degradation. p_tan +69%, p_oodc +354%, p_re +352%. Only p_in marginally improved (-1.2%). The per-sample normalization destroyed geometry-specific information in absolute DSDF values — the model uses these to differentiate foil shapes. The x-standardization already handles bringing features into a consistent range; TTA effectively double-normalizes, collapsing the geometry signal. **Closed — clear dead end.**

---

### 2026-04-06 ~05:30 — PR #2187: Normal-Velocity Hard Constraint: No-Penetration BC Projection — edward — **CLOSED** (p_tan +3.0%)

- Branch: `edward/normal-vel-constraint`
- Hypothesis: Hard-constraint normal-velocity projection at surface nodes — project out the normal component of predicted velocity so no-penetration BC is structurally impossible to violate. Differentiable projection applied in both training and validation after SRF heads, before loss.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| normal_vel_projection | 42 | 12.792 | 7.833 | 28.981 | 6.520 | ey0n53eo |
| normal_vel_projection | 73 | 12.758 | 7.651 | 29.706 | 6.487 | ejjcn6kq |
| **Experiment avg** | — | **12.775** | **7.742** | **29.344** | **6.504** | |
| **Baseline (DCT #2184)** | — | **13.205** | **7.816** | **28.502** | **6.453** | 6yfv5lio, etepxvjc |

**Results:** p_tan regressed +3.0% (29.344 vs 28.502). p_in improved -3.3% (12.775 vs 13.205) — noteworthy but doesn't compensate. Student identified two key issues: (1) multi-foil angle-sorting bug wraps foil-1 and foil-2 into a single contour producing incorrect normals at foil boundaries, corrupting tandem predictions; (2) model already satisfies near-zero normal velocity implicitly (mean |u·n| = 0.008, 0.5% of tangential) so the hard constraint removes degrees of freedom without solving a real problem. Zero extra parameters. **Closed — p_tan regression clear and well-understood.**

---

### 2026-04-06 ~16:00 — PR #2184: DCT Frequency-Weighted Surface Pressure Loss — nezuko — **MERGED** (p_tan -0.3%, new baseline)

- Branch: `nezuko/dct-freq-loss`
- Hypothesis: Apply rfft-based frequency-weighted auxiliary loss on ordered surface pressure nodes (w_k = 1 + gamma*(k/N)^alpha) to combat spectral bias — force model to attend to high-frequency leading-edge/TE features without disrupting main training.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| w=0.05 | 42 | 12.845 | 8.061 | 28.432 | 6.479 | 6yfv5lio |
| w=0.05 | 73 | 13.564 | 7.570 | 28.572 | 6.427 | etepxvjc |
| **w=0.05 avg** | — | **13.205** | **7.816** | **28.502** | **6.453** | |
| w=0.1 | 42 | 12.862 | 8.152 | 28.265 | 6.552 | d3p6cz67 |
| w=0.1 | 73 | 13.642 | 7.842 | 29.632 | 6.581 | o1x49ur6 |
| **w=0.1 avg** | — | **13.252** | **7.997** | **28.949** | **6.567** | |
| **Baseline (PR #2130)** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** w=0.05 beats baseline on p_tan (-0.3%) and p_re (-1.5%). p_in/p_oodc slightly regress. w=0.1 is unstable (high seed variance on p_tan: 28.27 vs 29.63). The gentle w=0.05 is the sweet spot — absolute DCT coefficient difference with smooth polynomial weighting is numerically stable unlike failed BSP (#2172). **New baseline: p_tan=28.502.** Standout: w=0.1-s42 achieved p_tan=28.265 suggesting there's more headroom if training can be stabilized.

---

### 2026-04-06 ~16:00 — PR #2182: Ensemble Distillation — tanjiro — **CLOSED** (teacher quality gap: ensemble weaker than student)

- Branch: `tanjiro/ensemble-distill`
- Hypothesis: Use 8-seed ensemble (seeds 66-73) soft targets as distillation signal for a fresh model. Ensemble mean = lower-variance training signal → student can outperform individual models.

| Config | p_in | p_oodc | p_tan | p_re |
|--------|------|--------|-------|------|
| alpha=0.3 avg | 13.45 | 7.90 | 29.05 | 6.40 |
| alpha=0.5 avg | 13.45 | 7.85 | 29.40 | 6.50 |
| **Baseline** | **13.05** | **7.70** | **28.60** | **6.55** |

**Results:** All configs worse on p_tan (+1.6% and +2.8%). Root cause: the 8-teacher ensemble (seeds 66-73) was trained WITHOUT GSB and PCGrad — their individual p_tan ≈30-32, significantly weaker than our current baseline. Distillation from weaker teachers adds noise, not variance reduction. p_re improved (-2.3%) suggesting some regularization benefit, but p_tan is the bottleneck. **Dead end: distillation only works when teacher > student. Requiring 8x budget to train a current-config ensemble is prohibitive.**

---

### 2026-04-06 ~14:30 — PR #2175: SWD Tandem Domain Alignment — askeladd — **CLOSED** (distributional difference is informative, not harmful)

- Branch: `askeladd/swd-domain-align`
- Hypothesis: Sliced Wasserstein Distance between tandem/single-foil slice token distributions forces domain-agnostic representations, improving OOD tandem generalization.

| Config | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|--------|-------|------|-----|
| **SWD w=0.01 avg** | 13.10 | 7.75 | 28.75 | 6.55 | mrqnyv82, uwphs04i |
| **SWD w=0.05 avg** | 13.40 | 8.00 | 29.65 | 6.60 | ukxu3q8i, qa84ir9x |
| **Baseline** | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** w=0.01 is neutral (p_tan +0.5%, within seed variance). w=0.05 actively hurts all metrics (p_tan +3.7%, p_oodc +3.9%). SWD loss converged (0.115) — alignment was enforced, but the alignment itself is counterproductive. The distributional difference between tandem and single-foil slice tokens encodes physically meaningful multi-body interaction effects (wake interference, gap flow dynamics) that are lost when forced to match. **Key insight: tandem-specific routing is feature, not bug.**

---

### 2026-04-06 ~17:30 — PR #2186: Panel Cp Residual Target — thorfinn — **CLOSED** (catastrophic degradation, 3-5x worse)

- Branch: `thorfinn/panel-cp-residual`
- Hypothesis: Instead of predicting `p - p_freestream`, predict `p - p_panel` (viscous correction only). Panel Cp captures deterministic inviscid flow; the viscous correction (boundary layer, separation, wake) is a smaller residual that should be easier to learn. Motivated by NeuralFoil (arXiv:2503.16323) which uses panel method baseline + neural viscous correction.

| Metric | Seed 42 (3j7eqs2i) | Seed 73 (zvkocnap) | 2-seed avg | vs Baseline |
|--------|-------------------|-------------------|------------|-------------|
| p_in | 57.0 | 58.1 | **57.6** | +341% ❌ |
| p_oodc | 45.2 | 44.3 | **44.8** | +473% ❌ |
| p_tan | 125.2 | 131.5 | **128.4** | +349% ❌ |
| p_re | 23.8 | 24.6 | **24.2** | +269% ❌ |
| **Baseline** | — | — | — | d7l91p0x, j9btfx09 |

**Results:** Catastrophic degradation across all splits. Root causes identified by student (confirmed):
1. **Viscous correction is harder to learn, not easier** — the full Cp field has smooth, spatially coherent gradients (stagnation, suction peaks) that help the model. The viscous correction has more complex structure (separation, wake effects) without those global anchors.
2. **asinh/normalization mismatch** — existing hyperparameters (asinh_scale=0.75, z-score stats) tuned for full Cp distribution, not the residual. The residual has a fundamentally different statistical distribution.
3. **Panel solver error compounds in tandem** — single-foil Hess-Smith fails for tandem configurations (wake interference). p_tan degraded most (+349%) because the panel Cp is unreliable for the exact geometry we care most about.
4. **Information loss** — subtracting panel Cp before asinh removes absolute pressure scale context that helps global prediction.

**Conclusion:** Panel-method-based target reformulation direction is fully exhausted (#2179 as input: +3.7%, #2186 as target: +340%). Both approaches failed. The baseline's asinh + freestream residual is the correct normalization. Move on.

---

### 2026-04-06 ~10:30 — PR #2180: Multi-Resolution Hash Grid Encoding — edward — **CLOSED** (per-sample normalization breaks spatial coherence)

- Branch: `edward/hash-grid-encoding`
- Hypothesis: 8-level hash grid encoding (base 16→2048) of mesh (x,y) coordinates, appended to DSDF features. Provides explicit multi-resolution spatial info to complement learned Fourier PE.

| Config | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|--------|-------|------|-----|
| **Hash Grid avg** | **17.9** | **9.25** | **32.1** | **7.85** | 3sb91xty, kie1y6kb |
| **Baseline** | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** Catastrophic regression across ALL metrics. p_tan +12.2%, p_in +37%, p_oodc +20%, p_re +20%. Three contributing factors: (1) Per-sample coordinate normalization to [0,1] defeats the hash grid's core mechanism — same physical point maps to different hash entries across samples, producing conflicting gradients. (2) 1.14M extra params (+25%) with 5x higher LR overfit quickly. (3) 20s/epoch overhead → only 121 epochs (19% fewer), barely 75% through cosine schedule. Hash grid fundamentally incompatible with variable-domain CFD meshes without fixed coordinate system.

---

### 2026-04-06 ~10:00 — PR #2179: Panel-Method Inviscid Cp as Input Feature — thorfinn — **CLOSED** (single-foil solver lacks tandem interaction)

- Branch: `thorfinn/panel-cp-features`
- Hypothesis: Pre-computed single-foil Hess-Smith panel Cp as input channel 25. Model predicts viscous correction.

| Config | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|--------|-------|------|-----|
| **Panel Cp avg** | **13.30** | **7.61** | **29.66** | **6.48** | xdxcew1g, 7j0gg5jb |
| **Baseline** | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** p_tan +3.7%. p_oodc improved -1.3%, p_re -1.1% — panel Cp helps non-tandem OOD. But the solver computed each foil independently, missing the tandem fore-aft interaction that determines p_tan. Inviscid Cp is already implicitly learnable from geometry features. Reassigned to try panel Cp as RESIDUAL TARGET (predict viscous correction).

---

### 2026-04-06 ~09:30 — PR #2166: dp/dn=0 Physics Loss (Extended) — alphonse — **CLOSED** (6-seed confirms: regularizer for p_in/p_re, neutral for p_tan)

- Branch: `alphonse/dpdn-physics-loss`
- Hypothesis: Zero wall-normal pressure gradient constraint as auxiliary loss on surface nodes.

| Config | Seeds | p_in | p_oodc | p_tan | p_re |
|--------|-------|------|--------|-------|------|
| **w=0.1 avg** | **6** | **13.02** ✅ | **7.82** ❌ | **28.97** ❌ | **6.45** ✅ |
| **w=0.05 avg** | **2** | **12.94** ✅ | **7.61** ✅ | **29.73** ❌ | **6.49** ✅ |
| **Baseline** | **2** | **13.05** | **7.70** | **28.60** | **6.55** |

**w=0.1 individual seeds:** 29.82, **27.77**, 29.37, 29.04, 28.71, 29.07 (σ=0.67). Seed 73's 27.77 was a 1.8σ outlier.

**Key finding:** dp/dn=0 is a genuine physics regularizer. Consistently improves p_in (-0.2%) and p_re (-1.5%). w=0.05 gives best p_oodc (7.61, -1.2%). But p_tan is NEUTRAL — 28.97 vs 28.60 is within noise (σ=0.67). The physics loss doesn't specifically help tandem transfer. Direction has long-term value for compounding with future baseline improvements.

---

### 2026-04-06 ~08:30 — PR #2177: Coordinated Tandem Ramp — nezuko — **CLOSED** (unstable, concurrent schedules interfere)

- Branch: `nezuko/coordinated-ramp`
- Hypothesis: Sigma decay synced with tandem loss ramp — start high sigma for diversity, anneal to baseline σ=0.02.

| Config | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|--------|-------|------|-----|
| **A (0.06→0.02) avg** | **13.58** | **7.86** | **29.49** | **6.51** | x2mkegnk, pvjzi2ov |
| **B (0.04→0.02) avg** | **13.10** | **7.89** | **29.22** | **6.48** | q5bprc0t, i4144o3d |
| **Baseline** | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** Neither config beats baseline. Config B s42 individually: p_tan=28.47 (-0.5%), but s73: 29.97 (+4.8%). High seed variance from concurrent sigma+loss schedule changes during tandem ramp warmup (epochs 10-50). p_re improved consistently (~-1.1%), but primary target regressed.

---

### 2026-04-06 ~08:00 — PR #2176: Spectral Shaping — frieren — **CLOSED** (unstable across seeds)

- Branch: `frieren/spectral-shaping`
- Hypothesis: Learnable 3-tap depthwise conv filter on GatedMLP activations to smooth feature representations.

| Config | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|--------|-------|------|-----|
| **k=3 avg** | **13.53** | **7.80** | **29.27** | **6.60** | i469q07m, 4ta4863c |
| **Baseline** | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** avg p_tan +2.3%. BUT: seed 42 exactly matched baseline (28.59), seed 73 badly regressed (29.95). Seed variance 4.5% (double baseline 2.1%). Learned filter evolves unpredictably — feature smoothing not robust. Only 9 learnable params total. torch.compile-compatible tensor-slicing implementation (no Conv1d slowdown).

---

### 2026-04-06 ~07:30 — PR #2178: Smaller SRF Head — tanjiro — **CLOSED** (h=192 confirmed optimal)

- Branch: `tanjiro/smaller-srf`
- Hypothesis: Smaller SRF head (h=128/96) may improve OOD generalization via implicit regularization.

| Config | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|--------|-------|------|-----|
| **h=128 avg** | **13.12** | **7.71** | **29.90** | **6.57** | 281pher3, w6apleaz |
| **h=96 avg** | **13.10** | **7.82** | **29.71** | **6.51** | uoaonrpq, u9a830l5 |
| **Baseline (h=192)** | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** Both worse. h=128 p_tan +4.5%, h=96 +3.9%. Combined with #2170 (h=256/384 also worse), h=192 is definitively confirmed optimal — classic bias-variance sweet spot. SRF head size fully characterized: 96 < 128 < 256 < 384 < **192 (optimal)**.

---

### 2026-04-06 ~07:00 — PR #2174: Attention Temperature Curriculum — fern — **CLOSED** (disrupts slice routing)

- Branch: `fern/attn-temp-curriculum`
- Hypothesis: Schedule attention temperature 2.0→0.3 over 80 epochs (broad→sharp routing), then release. Zero new params.

| Config | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|--------|-------|------|-----|
| **A (2.0→0.3, 80ep) avg** | **13.15** | **8.14** | **29.79** | **6.41** | nvyisrfa, zjx1jmy9 |
| **B (1.5→0.5, 60ep) avg** | **13.35** | **8.12** | **29.38** | **6.55** | 9uii4cn2, k4k8ko01 |
| **Baseline** | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** Both configs regress p_tan (+2.7-4.2%) and p_oodc (+5.5-5.7%). High initial temperature forces near-uniform attention, preventing early slice specialization. Interferes with GSB routing which needs correct routing from the start. Shorter/milder annealing (B) marginally better than aggressive (A), consistent with less disruption.

---

### 2026-04-06 ~06:00 — PR #2173: Foil-1 Geometry Adapter — edward — **CLOSED** (DSDF stats too coarse)

- Branch: `edward/foil1-geom-adapter`
- Hypothesis: Per-sample DSDF-1 distribution statistics (mean/std/skew/kurt) → MLP → slice logit bias. Gives GSB routing a geometry fingerprint for upstream foil shape.

| Config | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|--------|-------|------|-----|
| **scale=0.1 avg** | **13.9** | **7.85** | **29.3** | **6.7** | 8ryjbj4w, 1n7jwpzy |
| **scale=0.3 avg** | **13.15** | **7.75** | **29.2** | **6.4** | ex4uj2k8, lytgamfh |
| **Baseline** | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** Both scales regress p_tan (+2.1-2.4%). 4-moment statistics (mean/std/skew/kurt) too coarse — captures global distribution but discards spatial structure. scale=0.3 marginally better than 0.1 (bias too weak at 0.1). p_re slightly improved at 0.3 (6.4 vs 6.55). **Important finding:** foil-1 DSDF channels are at x[:,4:8] NOT x[:,2:6] (after pos(2) + saf(2)).

---

### 2026-04-06 ~05:20 — PR #2166: dp/dn=0 Physics Loss — alphonse — **SENT BACK** (MOST PROMISING — needs more seeds)

- Branch: `alphonse/dpdn-physics-loss`
- Hypothesis: Zero wall-normal pressure gradient constraint as auxiliary loss on surface nodes.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| w=0.01 | 42 | 13.14 | 7.81 | 29.22 | 6.43 | hz2gibgb |
| w=0.01 | 73 | 13.03 | 7.66 | 29.75 | 6.63 | eb34djbn |
| **w=0.01 avg** | — | **13.09** | **7.74** | **29.49** | **6.53** | — |
| w=0.1 | 42 | 12.95 | 7.77 | 29.82 | 6.49 | bmjgaeqp |
| w=0.1 | 73 | 13.36 | 7.74 | **27.77** | 6.38 | e7iu2ix3 |
| **w=0.1 avg** | — | **13.16** | **7.76** | **28.80** | **6.44** | — |
| **Baseline** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** MOST PROMISING since GSB+PCGrad! w=0.1 seed 73 achieves p_tan=27.77 (-2.9%), but seed 42 is 29.82 (+4.3%). High variance. p_re consistently improved. Sent back for 4 additional seeds at w=0.1 + 2 seeds at w=0.05 to confirm.

---

### 2026-04-06 ~05:00 — PR #2172: BSP Spectral Loss — thorfinn — **CLOSED** (w=0.1 catastrophic, w=0.05 all worse)

- Branch: `thorfinn/bsp-spectral-loss`
- Hypothesis: 1D DFT on arc-length surface pressure, bin-weighted spectral loss targeting high-frequency components.

| Config | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|--------|-------|------|-----|
| **w=0.1 avg** | **29.05** | **20.0** | **39.15** | **16.25** | k2owbp8q, an1gycr5 |
| **w=0.05 avg** | **13.5** | **8.1** | **30.25** | **6.85** | nxw9vfgh, 4wo682ce |
| **Baseline** | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** w=0.1 training collapse at epoch 60 (relative spectral error amplifies near-zero bins in asinh targets). w=0.05 stable but all metrics +3-6% worse. Spectral bias hypothesis doesn't apply — Transolver+SRF already handles multi-scale features.

---

### 2026-04-06 ~04:20 — PR #2171: Slice Number Sweep — tanjiro — **CLOSED** (96 confirmed optimal)

- Branch: `tanjiro/slice-num-sweep`
- Hypothesis: More slices = more GSB routing paths for tandem geometry specialization.

| Config | p_in | p_tan | p_oodc | p_re | W&B |
|--------|------|-------|--------|------|-----|
| **slice=128 avg** | **13.55** | **29.55** | **7.85** | **6.55** | 245dka5p, aasc7vel |
| **slice=144 avg** | **14.75** | **29.70** | **8.25** | **6.95** | pz5vo3h0, 1wzqtn7w |
| **Baseline (96)** | **13.05** | **28.60** | **7.70** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** Both worse. p_tan +3.3% (128) and +3.8% (144). More slices = more overfitting, fewer epochs in 180-min window. Confirms 96 optimal (also PR #2155).

---

### 2026-04-06 ~04:00 — PR #2170: Wider/Deeper Surface Refinement — frieren — **CLOSED** (SRF capacity overfits)

- Branch: `frieren/wider-deeper-srf`
- Hypothesis: More SRF capacity (h=256/384) for finer-grained pressure correction.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| **h=256 avg** | — | **13.3** | **7.8** | **29.95** | **6.4** | me8pq3ec, 068695h7 |
| **h=384 avg** | — | **13.5** | **7.75** | **29.75** | **6.5** | b2yqrp7k, h3r5gz4p |
| **Baseline (h=192)** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** Both worse. h=256 p_tan +4.7%, h=384 +4.0%. More capacity overfits to training surface patterns. h=192 confirmed optimal.

---

### 2026-04-06 ~04:00 — PR #2169: Online Hard Example Mining (OHEM) — nezuko — **CLOSED** (redundant with existing difficulty mechanisms)

- Branch: `nezuko/hard-sample-mining`
- Hypothesis: Adaptive per-sample loss upweighting for hardest K% samples.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| **ohem=0.25 avg** | — | **13.434** | **8.192** | **29.237** | **6.648** | 6cwidgyx, ja9nrkx4 |
| **ohem=0.50 avg** | — | **13.167** | **8.123** | **29.275** | **6.695** | 3bklpibu, ip5qhz7s |
| **Baseline** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** All metrics worse. p_tan +2.2-2.4%, p_oodc +5.5-6.4%. Redundant with existing tandem_boost + hard-node mining + PCGrad. 4th difficulty layer pushes training distribution too far, causing OOD overfitting.

---

### 2026-04-06 ~03:30 — PR #2168: Tandem Pressure Correction MLP — askeladd — **CLOSED** (p_tan +2.8%, mixed result)

- Branch: `askeladd/tandem-pressure-head`
- Hypothesis: Gated correction head for tandem-only pressure. Zero-init gate (bias=-2.0). Hidden=96, 28K params.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| tandem_phead | 42 | 12.6 | 7.6 | 29.5 | 6.2 | 6xar22ki |
| tandem_phead | 73 | 13.3 | 7.4 | 29.3 | 6.5 | lt4ygmej |
| **avg** | — | **12.95** | **7.50** | **29.40** | **6.35** | — |
| **Baseline** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** Mixed. p_oodc -2.6%, p_re -3.1% (consistent improvements). But p_tan +2.8% — primary target regressed. Double-correction interference: head applies on top of aft_foil_srf for aft-foil nodes. 28K extra params, minimal VRAM impact. Fore-foil restriction would face same collapse issue as 4 failed SRF attempts.
- Askeladd now idle — reassigning.

---

### 2026-04-06 ~03:00 — PR #2161: FiLM-Conditioned Fore-Foil SRF — fern — **CLOSED** (4th consecutive fore-foil SRF failure)

- Branch: `fern/film-conditioned-fore-foil-srf`
- Hypothesis: FiLM conditioning on fore-foil SRF head using DSDF1 stats + gap/stagger to differentiate NACA6416.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| post-MLP FiLM | 42 | 13.1 | 7.4 | 30.8 | 6.3 | snmfs7bm |
| post-MLP FiLM | 73 | 13.1 | 7.4 | 29.7 | 6.5 | o692pxsi |
| **avg** | — | **13.10** | **7.40** | **30.25** | **6.40** | — |
| **Baseline** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** p_tan +5.8%. Correction norm collapses to zero (same pattern as #2117, #2124, #2159). Interesting: p_oodc -3.9%, p_re -2.3% — SRF regularizes non-tandem metrics but hurts p_tan. 125K extra params, ~149 epochs (wall-clock limited). DEFINITIVE: fore-foil SRF dead across 4 architectures.
- Fern now idle — reassigning.

---

### 2026-04-06 ~01:45 — PR #2167: Tandem Surface Mixup — edward — **CLOSED** (physical inconsistency at swap boundary)

- Branch: `edward/tandem-surface-mixup`
- Hypothesis: CutMix-style augmentation — swap aft-foil surface node sets between tandem samples to create geometry diversity.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| prob=0.3 | 42 | 12.7 | 8.1 | 30.5 | 6.7 | b2zqlruz |
| prob=0.3 | 73 | 14.1 | 8.0 | 30.1 | 6.7 | x2riufv5 |
| **prob=0.3 avg** | — | **13.40** | **8.05** | **30.30** | **6.70** | — |
| prob=0.5 | 42 | 13.1 | 8.4 | 30.4 | 6.7 | 4qsfyw3z |
| prob=0.5 | 73 | 13.8 | 8.2 | 30.1 | 6.8 | yyqw2jqc |
| **prob=0.5 avg** | — | **13.45** | **8.30** | **30.25** | **6.75** | — |
| **Baseline** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** Both probs regress all metrics. p_tan +5.8-5.9%. Core issue: aft-foil targets are coupled to fore-foil wake — swapping targets across samples creates contradictory training signal. Node count mismatch required subsampling workaround. Generalizes: any augmentation that decouples aft-foil targets from upstream flow context will fail.
- Dead end: tandem surface/node mixing. Edward now idle — reassigning.

---

### 2026-04-06 ~00:25 — PR #2165: Iterative 2-Pass Refinement — thorfinn — **CLOSED** (training budget penalty, all metrics worse)

- Branch: `thorfinn/iterative-refinement`
- Hypothesis: AlphaFold2-style recycling — two forward passes with shared weights, pass-1 output concatenated as input for pass-2. Zero extra parameters. Warmup from epoch 60.

| Config | Seed | p_in | p_oodc | p_tan | p_re | VRAM | W&B |
|--------|------|------|--------|-------|------|------|-----|
| iter 2-pass | 42 | 14.5 | 8.6 | 30.3 | 7.1 | 46.6GB | u0nsat3a |
| iter 2-pass | 73 | 14.8 | 9.1 | 30.7 | 7.4 | 46.4GB | 5yxjk7be |
| **iter avg** | — | **14.65** | **8.85** | **30.5** | **7.25** | ~46.5GB | — |
| **Baseline** | — | **13.05** | **7.70** | **28.60** | **6.55** | ~46GB | d7l91p0x, j9btfx09 |

**Results:** All metrics worse. p_tan +6.6%. The 2-pass costs 1.3x per epoch → only 131 total epochs vs baseline ~150+. Model still converging at 180-min wall clock. Core issue: CFD surface pressure errors are from missing flow interaction info, not correctable biases — pass-2 can't recover what pass-1 couldn't encode. VRAM management was excellent (torch.no_grad on pass-1), warmup transition was smooth.
- Dead end: iterative refinement under fixed training budget. Thorfinn now idle — reassigning.

---

### 2026-04-06 ~00:15 — PR #2164: Backbone Gap/Stagger AdaLN — frieren — **CLOSED** (backbone AdaLN disrupts attention routing)

- Branch: `frieren/backbone-gs-adaln`
- Hypothesis: Thread gap/stagger conditioning through ALL TransolverBlocks via AdaLN-Zero. Currently only decoder sees geometry; backbone attention can't modulate for tandem configs.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| adaln_all + gs 4cond | 42 | — | — | ~30.55 | — | (see PR) |
| adaln_all + gs 4cond | 73 | — | — | ~30.55 | — | (see PR) |
| **adaln_all + gs 4cond avg** | — | — | — | **30.55** | — | — |
| adaln_all + Re/AoA 2cond | 42 | — | — | ~30.3 | — | (see PR) |
| adaln_all + Re/AoA 2cond | 73 | — | — | ~30.3 | — | (see PR) |
| **adaln_all + Re/AoA avg** | — | — | — | **30.3** | — | — |
| **Baseline** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** Both configs regress p_tan heavily (+6.8%, +5.9%). Modulating ALL backbone LayerNorms via AdaLN-Zero interferes with optimized attention routing and slice assignments. VRAM jumped to ~50GB, epochs dropped to 142-144 (under-trained). GSB (spatial bias level) is the right abstraction for geometry injection — normalization-level conditioning disrupts the backbone.
- Dead end: backbone-wide AdaLN conditioning.
- Frieren now idle — reassigning.

---

### 2026-04-06 ~00:15 — PR #2156: DSDF-1 Channel Dropout — tanjiro — **CLOSED** (foil-1 DSDF channels need exact values)

- Branch: `tanjiro/dsdf-channel-dropout`
- Hypothesis: Drop foil-1 DSDF channels (p=0.2, 0.3) during tandem training to force shape-invariant prediction.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| p=0.2 | 42 | — | — | ~30.20 | — | (see PR) |
| p=0.2 | 73 | — | — | ~30.20 | — | (see PR) |
| **p=0.2 avg** | — | — | — | **30.20** | — | — |
| p=0.3 | 42 | — | — | ~30.40 | — | (see PR) |
| p=0.3 | 73 | — | — | ~30.40 | — | (see PR) |
| **p=0.3 avg** | — | — | — | **30.40** | — | — |
| **Baseline** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** Both dropout rates regress p_tan (+5.6%, +6.3%). Foil-1 DSDF encodes upstream geometry creating the wake impinging on aft foil — dropping it removes critical causal information. Notable: p_re improved -9.2% at p=0.2 (strong OOD Reynolds regularization), but p_tan regression too severe. Combined with PR #2133 (foil-1 magnitude aug failed), foil-1 DSDF perturbation in any form is a dead end.
- Tanjiro now idle — reassigning.

---

### 2026-04-05 ~20:45 — PR #2163: Differential LR for Specialized Heads — nezuko — **CLOSED** (p_tan regression)

- Branch: `nezuko/differential-lr-specialized-heads`
- Hypothesis: Boost aft_srf/surface_refine/GSB heads 2x-3x vs backbone. GSB was in attn_params at lr×0.5.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| mult=2.0 | 42 | 13.507 | 7.898 | 28.770 | 6.495 | jg8ummcf |
| mult=2.0 | 73 | 13.469 | 7.945 | 30.194 | 6.577 | mth1a9p8 |
| **mult=2.0 avg** | — | **13.488** | **7.922** | **29.482** | **6.536** | — |
| mult=3.0 | 42 | 13.397 | 7.543 | 29.893 | 6.597 | aadm4fz6 |
| mult=3.0 | 73 | 13.767 | 7.317 | 29.444 | 6.404 | ltvxv838 |
| **mult=3.0 avg** | — | **13.582** | **7.430** | **29.669** | **6.501** | — |
| **Baseline** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** Both multipliers regress p_tan (+3.1%, +3.7%). Interesting: mult=3.0 improves p_oodc (-3.5%) but at the cost of p_tan/p_in. Higher head LR disrupts co-adaptation between backbone and specialized heads.
- Nezuko now idle — reassigning.

---

### 2026-04-05 ~20:30 — PR #2162: Tandem Cross-DSDF Features — askeladd — **CLOSED** (hand-crafted features add noise)

- Branch: `askeladd/tandem-cross-dsdf-features`
- Hypothesis: Per-node dist_ratio + rel_angle features extending GSB from global to per-node geometry. Zero new parameters — pure feature engineering.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| cross-dsdf | 42 | 12.6 | 7.8 | 30.3 | 6.7 | jkm5f1yx |
| cross-dsdf | 73 | 13.2 | 7.8 | 29.4 | 6.5 | vgm0gla2 |
| **cross-dsdf avg** | — | **12.9** | **7.8** | **29.85** | **6.6** | — |
| **Baseline** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** p_tan +4.4%, p_oodc +1.3%. Model already has raw 8-channel DSDF field — hand-crafted summaries (min, atan2) are lossy and add noise. Feature engineering on top of existing DSDF is a dead end.
- Askeladd now idle — reassigning to tandem pressure correction MLP.

---

### 2026-04-05 ~19:55 — PR #2158: Asymmetric PCGrad — edward — **CLOSED** (symmetric baseline better)

- Branch: `edward/asymmetric-pcgrad`
- Hypothesis: Only project OOD gradient (g_B) onto in-dist normal plane; preserve in-dist gradient (g_A) unchanged. Should hold/improve p_in.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| asymmetric | 42 | 13.3 | 7.7 | 28.9 | 6.5 | amb3ci1h |
| asymmetric | 73 | 13.5 | 8.0 | 28.8 | 6.5 | hc1vp76q |
| **asymmetric avg** | — | **13.4** | **7.85** | **28.85** | **6.50** | — |
| **Baseline (symmetric)** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** All key metrics worse except p_re (-0.8%). Surprisingly, p_in regressed +2.7% — protecting in-dist from OOD interference actually hurt. Symmetric projection acts as implicit regularization preventing in-dist overfitting.
- **PCGrad fully explored:** 3-way (#2147) worse, asymmetric (#2158) worse, symmetric 2-way = optimal. Direction exhausted.
- Edward now idle — reassigning to tandem surface mixup.

---

### 2026-04-05 ~19:25 — PR #2157: Foil Shape Similarity Bias (GSB 7D) — alphonse — **CLOSED** (sample-level similarity too coarse)

- Branch: `alphonse/foil-shape-similarity-bias`
- Hypothesis: Extend GSB 6D→7D by appending cosine similarity of foil-1 vs foil-2 mean DSDF vectors. Shape-similarity-conditioned routing for similar vs dissimilar foil pairs.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| GSB 7D | 42 | 12.93 | 8.04 | 30.04 | 6.36 | yvrpikxi |
| GSB 7D | 73 | 13.48 | 7.82 | 29.27 | 6.62 | j20oel6q |
| **GSB 7D avg** | — | **13.20** | **7.93** | **29.66** | **6.49** | — |
| **Baseline (GSB 6D)** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** All key metrics worse: p_tan +3.7%, p_oodc +3.0%, p_in +1.1%. Sample-level cosine similarity loses spatial structure — too coarse for p_tan discrimination. Degenerate for single-foil samples (foil-2 DSDF ≈ 0). Per-node features (askeladd #2162) are the right approach.
- Alphonse now idle — reassigning to dp/dn=0 physics loss.

---

### 2026-04-05 ~18:55 — PR #2154: Cosine T_max Sweep {140, 180} vs 160 — thorfinn — **CLOSED** (T_max=160 confirmed optimal)

- Branch: `thorfinn/cosine-tmax-sweep`
- Hypothesis: T_max not tuned since GSB/aft_srf/PCGrad added. Test T_max=140 (faster decay) and T_max=180 (extended warmth).

| T_max | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|-------|------|------|--------|-------|------|-----|
| 140 | 42 | 13.2 | 7.6 | 29.5 | 6.4 | 9cg2dt2a |
| 140 | 73 | 12.7 | 7.9 | 29.3 | 6.5 | scwsj91y |
| **140 avg** | — | **12.95** | **7.75** | **29.40** | **6.45** | — |
| 180 | 42 | 13.5 | 7.6 | 29.6 | 6.7 | 3beaxpua |
| 180 | 73 | 13.8 | 8.3 | 29.2 | 6.6 | tz7aj1ji |
| **180 avg** | — | **13.65** | **7.95** | **29.40** | **6.65** | — |
| **Baseline (160)** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** Both T_max=140 and 180 regress p_tan identically (+2.8%). T_max=160 is the sweet spot. LR schedule is not a bottleneck.
- Thorfinn now idle — reassigning to iterative 2-pass refinement.

---

### 2026-04-05 ~18:40 — PR #2153: Gap/Stagger Sigma Increase σ=0.03 — frieren — **CLOSED** (σ=0.02 confirmed optimal)

- Branch: `frieren/gs-sigma-increase`
- Hypothesis: With GSB in the baseline, the model has explicit tandem-geometry-aware routing. Higher σ=0.03 might compound with GSB rather than fighting it.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| σ=0.03 | 42 | 13.5 | 7.7 | 29.2 | 6.4 | 2ahp1qdy |
| σ=0.03 | 73 | 13.7 | 7.8 | 29.9 | 6.4 | 1zmqzcvi |
| **σ=0.03 avg** | — | **13.6** | **7.75** | **29.55** | **6.4** | — |
| **Baseline (σ=0.02)** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** All key metrics worse: p_tan +3.3%, p_in +4.2%. Only p_re marginally better (-2.3%).
- **Gap/stagger sigma sweep complete:** σ=0.00 (worse), σ=0.01 (worse), **σ=0.02 (optimal)**, σ=0.03 (worse). Inverted-U response confirmed.
- Frieren now idle — reassigning to bold backbone conditioning experiment.

---

### 2026-04-05 ~18:15 — PR #2152: Augmentation Annealing — nezuko — **CLOSED** (p_tan regression)

- Branch: `nezuko/aug-annealing`
- Hypothesis: Standard competition ML practice — apply full augmentation early for diversity, then linearly decay aug sigma to a fraction by the final epoch. Tests anneal→50% and anneal→0%.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| anneal→50% | 42 | 12.878 | 7.838 | 28.443 | 6.341 | 7ofuolg3 |
| anneal→50% | 73 | 13.294 | 7.857 | 29.348 | 6.380 | zt31115v |
| **anneal→50% avg** | — | **13.086** | **7.848** | **28.896** | **6.361** | — |
| anneal→0% | 42 | 12.739 | 7.993 | 29.278 | 6.533 | ibywi5rr |
| anneal→0% | 73 | 13.796 | 7.817 | 29.125 | 6.439 | 4jd564uq |
| **anneal→0% avg** | — | **13.268** | **7.905** | **29.202** | **6.486** | — |
| **Baseline** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** Neither config beats baseline on p_tan. anneal→50% avg p_tan=28.90 (+1.0%), anneal→0% avg p_tan=29.20 (+2.1%). Only p_re improved with anneal→50% (6.36 vs 6.55, -2.9%).
- **Key insight:** Tandem transfer generalization (p_tan) requires sustained augmentation diversity throughout training. Annealing gap/stagger sigma reduces the domain randomization needed for unseen tandem configs. Constant σ=0.02 confirmed optimal schedule.
- **Interesting note:** p_re benefits from cleaner late-training data (Reynolds OOD less affected by geometry aug reduction), but this is a secondary metric.
- Nezuko now idle — reassigning.

---

### 2026-04-05 ~17:00 — PR #2151: EMA Start Epoch Sweep (100, 120 vs ~140) — fern — **CLOSED** (dead end)

- Branch: `fern/ema-start-epoch`
- Hypothesis: Starting EMA accumulation earlier (epoch 100 or 120 vs default ~140) would provide more checkpoint averaging, producing a smoother model that generalizes better to OOD data. With training running ~150-155 epochs, start=100 gives 3.7x more averaging epochs than the default.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| start=100 | 42 | 12.8 | 7.8 | 29.7 | 6.6 | g9qbvs9v |
| start=100 | 73 | 12.9 | 7.6 | 29.6 | 6.5 | 4dyfrlfk |
| **start=100 avg** | — | **12.85** | **7.7** | **29.65** | **6.55** | — |
| start=120 | 42 | 12.9 | 7.8 | 29.8 | 6.6 | 352ovdc3 |
| start=120 | 73 | 12.9 | 7.6 | 29.6 | 6.5 | nuys4pcb |
| **start=120 avg** | — | **12.9** | **7.7** | **29.7** | **6.55** | — |
| **Baseline (~140)** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** Both earlier EMA starts regress p_tan significantly: start=100 +3.7%, start=120 +3.8%. The two configs are nearly identical, suggesting the 100-120 range doesn't matter much — what matters is that EMA starts too early during the high-LR exploration phase of the cosine schedule (T_max=160). The default start ~140 captures only the final converged trajectory.
- **Key insight:** Fewer averaging epochs (15 at default) beats more (35-55). The sweet spot for EMA is the last ~10% of training when the model is converged. Added to confirmed hyperparameter dead ends.
- Fern now idle — reassigning.

---

### 2026-04-05 ~17:00 — PR #2150: DSDF2 Sigma Optimization σ={0.03, 0.08} vs 0.05 — askeladd — **CLOSED** (σ=0.05 confirmed optimal)

- Branch: `askeladd/dsdf2-sigma-sweep`
- Hypothesis: The foil-2 DSDF magnitude augmentation (σ=0.05, PR #2126) was merged without systematic optimization. σ=0.03 (tighter: [0.94, 1.06]) and σ=0.08 (wider: [0.85, 1.17]) bracket the current value.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| σ=0.03 | 42 | 12.9 | 7.8 | 28.2 | 6.6 | cm9uz650 |
| σ=0.03 | 73 | 13.6 | 8.1 | 29.5 | 6.6 | kj8cvxpw |
| **σ=0.03 avg** | — | **13.25** | **7.95** | **28.85** | **6.60** | — |
| σ=0.08 | 42 | 12.9 | 7.9 | 28.9 | 6.6 | js6sm78l |
| σ=0.08 | 73 | 13.4 | 7.9 | 29.4 | 6.5 | 2o2499gz |
| **σ=0.08 avg** | — | **13.15** | **7.90** | **29.15** | **6.55** | — |
| **Baseline (σ=0.05)** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** Neither variant beats baseline. σ=0.03: p_tan +0.9%, p_oodc +3.2%. σ=0.08: p_tan +1.9%, p_oodc +2.6%. The seed-42 p_tan=28.2 for σ=0.03 is tempting but seed 73's 29.5 confirms high variance — not a real signal.
- **Key insight:** σ=0.05 sits at the sweet spot for DSDF2 augmentation in the compound baseline (PCGrad + GSB + DSDF2). Tighter = insufficient diversity, wider = too much noise. Added to confirmed hyperparameter dead ends.
- Askeladd now idle — reassigning.

---

### 2026-04-05 ~14:00 — PR #2149: Learning Rate Sweep lr={3e-4, 1e-4} — edward — **CLOSED** (lr=2e-4 confirmed optimal)

- Branch: `edward/lr-sweep`
- Hypothesis: lr=2e-4 was set when the architecture was simpler; after adding GSB, PCGrad, aft-foil SRF head, the optimal LR may have shifted. Lion optimizer papers suggest LR should be 3-10x higher than AdamW.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| lr=3e-4 | 42 | 13.2 | 8.1 | 29.8 | 6.7 | 3at64dy4 |
| lr=3e-4 | 73 | 13.9 | 8.0 | 29.9 | 6.4 | vohbzzhi |
| **lr=3e-4 avg** | — | **13.55** | **8.05** | **29.85** | **6.55** | — |
| lr=1e-4 | 42 | 13.7 | 8.2 | 29.6 | 6.6 | 8s56fget |
| lr=1e-4 | 73 | 14.0 | 8.2 | 29.0 | 6.8 | pyhi4uri |
| **lr=1e-4 avg** | — | **13.85** | **8.20** | **29.30** | **6.70** | — |
| **Baseline (lr=2e-4)** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Results:** Neither alternative beats baseline on any metric. lr=3e-4: p_in +3.8%, p_oodc +4.5%, p_tan +4.4%. lr=1e-4: p_in +6.1%, p_oodc +6.5%, p_tan +2.4%. The cosine schedule (T_max=160) is calibrated for lr=2e-4 — changing LR without adjusting T_max creates a schedule mismatch. lr=1e-4 underfits within 180 min, lr=3e-4 overshoots.
- **Key insight:** lr=2e-4 confirmed optimal and robust to ±50% perturbation. Adding to confirmed hyperparameter dead ends.
- **Note:** Individual best seed at lr=1e-4 (s73) gave p_tan=29.0 — still worse than baseline 28.60.
- Edward reassigned to Asymmetric PCGrad (#2158).

---

### 2026-04-05 ~13:30 — PR #2131: Tandem-Slice Carve-Out K=4 (rebased) — alphonse — **CLOSED** (doesn't compound with GSB)

- Branch: `alphonse/tandem-slice-carveout`
- Hypothesis: Reserving K physics slices exclusively for tandem samples (large negative bias for single-foil) would carve out dedicated representational capacity for tandem wake interactions. K=4 showed strong initial signal (-3.7% p_tan vs control) on old baseline.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| K=4 (rebased) | 42 | 12.83 | 7.63 | 29.31 | 6.49 | vej21fcz |
| K=4 (rebased) | 73 | 13.59 | 7.75 | 29.55 | 6.37 | m2kk8eq2 |
| **K=4 avg** | — | **13.21** | **7.69** | **29.43** | **6.43** | — |
| **Baseline (PR #2130)** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Baseline comparison (K=4 rebased avg):**
| Metric | Baseline | K=4 | Delta |
|--------|----------|-----|-------|
| p_in | 13.05 | 13.21 | +1.2% ❌ |
| p_oodc | 7.70 | 7.69 | -0.1% ✅ |
| p_tan | **28.60** | **29.43** | **+2.9%** ❌ |
| p_re | 6.55 | 6.43 | -1.8% ✅ |

**Results:** K=4 carve-out does NOT compound with GSB+PCGrad baseline. The original K=4 signal (-3.7% vs control) was real, but GSB already handles tandem-geometry-aware routing by conditioning slice routing on (gap, stagger) scalars. The carve-out mechanism is mechanistically redundant with GSB — both address tandem slice specialization through different means (hard partitioning vs conditioned routing), so they don't stack additively.
- **Key insight:** GSB essentially achieved the same goal as tandem carve-out (tandem-specific slice specialization) through a more powerful, conditioned mechanism. Once GSB was in the baseline, the carve-out's marginal benefit vanished.
- Alphonse reassigned to Foil Shape Similarity Bias (#2157).

---

### 2026-04-05 ~11:00 — PR #2148: Gap/Stagger Aug Removal (σ=0) — tanjiro — **CLOSED** (dead end)

- Branch: `tanjiro/gs-aug-removal`
- Hypothesis: GSB (gap_stagger_spatial_bias) may have made gap/stagger augmentation redundant. Removing σ=0.02 should recover p_tan penalty without p_oodc regression.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| σ=0 (no aug) | 42 | 13.0 | 7.7 | 29.5 | 6.4 | 3h9fj0ym |
| σ=0 (no aug) | 73 | 14.0 | 7.9 | 30.0 | 6.4 | j40ez5v3 |
| **σ=0 avg** | — | **13.5** | **7.80** | **29.75** | **6.4** | — |
| **Baseline (σ=0.02)** | — | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

**Baseline comparison (σ=0 avg):**
| Metric | Baseline | σ=0 | Delta |
|--------|----------|-----|-------|
| p_in | 13.05 | 13.5 | +3.4% ❌ |
| p_oodc | 7.70 | 7.80 | +1.3% ❌ |
| p_tan | **28.60** | **29.75** | **+4.0%** ❌ |
| p_re | 6.55 | 6.4 | -2.3% ✓ |

**Results:** Hypothesis falsified. Removing gap/stagger aug hurts all primary metrics. GSB and aug are complementary, not redundant: GSB provides geometry-aware routing, aug provides regularization against distribution shift. Only p_re improved slightly.
- **Key insight:** Gap/stagger sigma parameter space now fully explored: σ={0, 0.01(#2140), 0.02(baseline), 0.03(#2153 in-progress)}. σ=0.02 is confirmed optimal or near-optimal.
- Tanjiro reassigned — pending new hypothesis.

---

### 2026-04-05 ~10:00 — PR #2147: Actual 3-Way PCGrad — thorfinn — **CLOSED** (negative)

- Branch: `thorfinn/actual-3way-pcgrad`
- Hypothesis: True 3-way PCGrad (single-foil / tandem-normal / tandem-extreme-Re) via `--disable_pcgrad --pcgrad_3way` may beat 2-way PCGrad by resolving finer-grained gradient conflicts.

| pct | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|-----|------|------|--------|-------|------|-----|
| 0.15 | 42 | 13.6 | 8.1 | 29.2 | 6.7 | sl98yydy |
| 0.15 | 73 | 13.2 | 7.8 | 28.8 | 6.6 | u3pvgipm |
| 0.15 | avg | 13.4 | 7.95 | **29.00** | 6.65 | — |
| 0.10 | 42 | 13.8 | 8.3 | 29.5 | 6.8 | 6dmvry4i |
| 0.20 | 42 | 13.5 | 8.0 | 29.1 | 6.7 | 6padkn2v |

**Baseline comparison (pct=0.15 avg):**
| Metric | Baseline | 3-Way | Delta |
|--------|----------|-------|-------|
| p_in | 13.05 | 13.4 | +2.7% ❌ |
| p_oodc | 7.70 | 7.95 | +3.2% ❌ |
| p_tan | **28.60** | **29.00** | **+1.4%** ❌ |
| p_re | 6.55 | 6.65 | +1.5% ❌ |

**Results:** 3-way PCGrad worse than 2-way across all metrics. The tandem-extreme-Re carve-out (pct=0.15) creates too small a group for stable gradient estimation, while 2-way's simpler in-dist vs OOD split is more robust. All three pct values tested (0.10, 0.15, 0.20) fail to beat baseline. 2-way PCGrad confirmed optimal.
- **Key insight:** PCGrad effectiveness depends on group size stability. Splitting OOD into normal/extreme creates noisy gradient estimates.
- Thorfinn reassigned to Cosine T_max Sweep (#2154).

---

### 2026-04-05 ~09:15 — PR #2146: Tail EMA Checkpoint Averaging — frieren — **CLOSED** (null result)

- Branch: `frieren/tail-ema-avg`
- Hypothesis: Average last 2-3 EMA snapshots from converged tail of training for smoother model.

| Config | Standard EMA p_tan | Tail-Avg p_tan | Δ |
|--------|-------------------|----------------|---|
| A (start=135) s42 | 29.0 | 29.11 | +0.4% |
| A (start=135) s73 | 29.2 | 29.16 | -0.1% |
| B (start=145) s42 | 29.8 | 29.85 | +0.2% |
| B (start=145) s73 | 29.7 | 29.62 | -0.3% |

W&B: fh7f24u6, azir19g1 (Config A), wrh72afu, 6hhy1mih (Config B).

**Results:** Pure noise (±0.3%). EMA decay=0.999 already averages over ~1000 steps — additional snapshot averaging is redundant. Snapshots barely differ because cosine LR is near zero in the tail.
- **Post-hoc weight averaging confirmed as exhausted class:** SWAD (#2094), model soup (#2142), tail averaging (#2146) all fail. EMA + cosine is already optimal averaging.
- Individual runs also don't match baseline (best standard EMA: 29.0 vs baseline 28.60) — run variance.
- Frieren reassigned to gap/stagger σ=0.03 (#2153).

---

### 2026-04-05 ~08:20 — PR #2141 (Round 2): EMA Decay 0.9995 Rebased — nezuko — **CLOSED** (negative)

- Branch: `nezuko/ema-decay-sweep`
- Hypothesis: decay=0.9995 showed promise against old baseline (p_oodc -2.7%, p_tan -0.4%). Rebased onto GSB baseline.

| Metric | Baseline (GSB) | decay=0.9995+GSB avg | Delta |
|--------|---------------|---------------------|-------|
| p_in | 13.05 | 13.52 | +3.6% ❌ |
| p_oodc | 7.70 | 7.77 | +0.9% ❌ |
| p_tan | **28.60** | **29.75** | **+4.0%** ❌ |
| p_re | 6.55 | 6.62 | +1.0% ❌ |

W&B: mnbrh0r2 (s42), kptwsdvt (s73).

**Results:** GSB obsoleted the decay=0.9995 advantage. GSB provides spatial smoothing that higher EMA decay was approximating. ema_decay=0.999 confirmed optimal with current architecture.
- Nezuko reassigned to augmentation annealing (#2152).

---

### 2026-04-05 ~08:10 — PR #2145: Weight Decay Sweep — fern — **CLOSED** (negative)

- Branch: `fern/weight-decay-sweep`
- Hypothesis: Reduce weight decay from 5e-5 to {1e-5, 2e-5} to reduce regularization.

| wd | p_in avg | p_oodc avg | p_tan avg | p_re avg | W&B |
|----|---------|-----------|----------|---------|-----|
| 1e-5 | 12.95 (-0.8%) | 7.55 (-1.9%) | **29.25 (+2.3%)** ❌ | 6.50 | dxftthwn, nu632qqg |
| 2e-5 | 12.95 (-0.8%) | 7.55 (-1.9%) | **29.20 (+2.1%)** ❌ | 6.55 | nx9xnwcm, r5w9nr6a |
| **Baseline (5e-5)** | **13.05** | **7.70** | **28.60** | **6.55** | d7l91p0x, j9btfx09 |

Runs on new baseline (with GSB). 151-152 epochs.

**Results:** p_in/p_oodc slightly improve but p_tan regresses 2.1-2.3%. Seed 73 drives the regression (29.6 vs baseline 28.3). wd=5e-5 confirmed optimal — Lion's built-in weight decay behavior means explicit wd has limited impact at these low values.
- Fern reassigned to EMA start epoch sweep (#2151).

---

### 2026-04-05 ~08:10 — PR #2140: Gap/Stagger Sigma Reduction — askeladd — **CLOSED** (negative)

- Branch: `askeladd/gap-stagger-sigma-sweep`
- Hypothesis: Reduce gap/stagger aug σ from 0.02 to 0.01 to reduce p_tan penalty.

| σ | p_in avg | p_oodc avg | p_tan avg | p_re avg | W&B |
|---|---------|-----------|----------|---------|-----|
| 0.01 | 13.25 | 7.75 (+3.3%) | **29.75 (+1.5%)** ❌ | 6.45 | 7biulneh, 8v9fka4m |
| **Old baseline (0.02)** | **13.35** | **7.50** | **29.30** | **6.45** | — |
| **Current baseline** | **13.05** | **7.70** | **28.60** | **6.55** | — |

Ran against OLD baseline (no GSB). σ=0.01 is worse than σ=0.02 on both p_oodc and p_tan. Current σ=0.02 is well-calibrated. Combined with tanjiro's σ=0 test (#2148), we're building a 3-point sweep.
- Askeladd reassigned to DSDF2 sigma sweep (#2150).

---

### 2026-04-05 ~08:00 — PR #2144: Input Feature Noise Augmentation — edward — **CLOSED** (dead end)

- Branch: `edward/input-noise-aug`
- Hypothesis: Add Gaussian noise to standardized input features during training to improve OOD generalization.

| σ | p_in avg | p_oodc avg | p_tan avg | p_re avg |
|---|---------|-----------|----------|---------|
| 0.01 | 15.5 (+17%) | 9.6 (+21%) | 30.7 (+4.1%) | 7.95 (+22%) |
| 0.03 | 26.95 (+104%) | 10.95 (+38%) | 32.3 (+9.6%) | 9.1 (+40%) |
| 0.05 | 23.5 (+78%) | 11.7 (+48%) | 32.3 (+9.6%) | 9.65 (+48%) |
| Old baseline | 13.20 | 7.91 | 29.48 | 6.50 |
| **Current baseline** | **13.05** | **7.70** | **28.60** | **6.55** |

W&B group: `phase6/input-noise-aug`. Runs: ibpmve5e, cxyu1uvk (σ=0.01), 0hfbsp66, o3embbbn (σ=0.03), d2otilbo, fgczv5yq (σ=0.05). Old baseline (no GSB). 145-147 epochs.

**Results commentary:**
- Catastrophic degradation at ALL noise levels. Even σ=0.01 causes p_in +17%, p_re +22%.
- CFD inputs carry precise geometric information — noise corrupts geometry→flow mapping.
- Monotonic degradation confirms noise is pure corruption, not regularization.
- **Confirms dead end CLASS:** Generic input perturbation (noise, dropout) fundamentally incompatible with mesh-based CFD prediction. Only physically-consistent augmentations work.
- Edward reassigned to LR sweep (#2149).

---

### 2026-04-05 ~06:40 — PR #2137: EMA Stochastic Weight Perturbation — tanjiro — **CLOSED** (dead end)

- Branch: `tanjiro/ema-perturb`
- Hypothesis: One-time Gaussian perturbation at EMA start (epoch 140) to explore flat minima. σ sweep {5e-4, 1e-3, 3e-3}.

| σ | p_in avg | p_oodc avg | p_tan avg | p_re avg |
|---|---------|-----------|----------|---------|
| 5e-4 | 12.9 (-0.9%) | 7.6 (-0.3%) | **30.5 (+2.0%)** ❌ | 6.5 (+0.5%) |
| 1e-3 | 13.2 (+1.4%) | 7.75 (+1.7%) | **30.1 (+0.6%)** ❌ | 6.5 (+0.5%) |
| 3e-3 | 13.2 (+1.4%) | 7.7 (+1.0%) | **30.2 (+1.0%)** ❌ | 6.5 (+0.5%) |
| Old baseline | 13.02 | 7.62 | 29.91 | 6.47 |
| **Current baseline** | **13.05** | **7.70** | **28.60** | **6.55** |

W&B group: `phase6/ema-perturb`. Runs: er41ypz5, asbtprv1 (5e-4); obkrg47t, 6s1r2mu7 (1e-3); 4w1o1rnr, 2nxtud27 (3e-3). Stale flags: --disable_pcgrad, --aft_foil_srf_context, no GSB. ~156-157 epochs.

**Results commentary:**
- All σ values regress p_tan vs old baseline. Not competitive with current baseline (28.60).
- Seed 42 vs 73 shows 0.6-1.4 points divergence — perturbation adds noise, not systematic improvement.
- **Flat-minima-seeking confirmed as dead end CLASS:** SAM (#2086), SGLD (#2120), SWAD (#2094), and EMA perturbation all failed. Lion + cosine + EMA already provides sufficient implicit regularization.
- Recovery time too short (17 epochs from epoch 140), LR near minimum → model barely moves after perturbation.
- Tanjiro reassigned to gap/stagger aug removal (#2148).

---

### 2026-04-05 ~06:20 — PR #2143: DSDF Spatial Dropout — thorfinn — **CLOSED** (dead end)

- Branch: `thorfinn/dsdf-spatial-dropout`
- Hypothesis: Zero out DSDF shape features for random nodes during training (p={0.05, 0.10, 0.20}) to reduce shape-dependence and improve tandem transfer.

| p | p_in avg | p_oodc avg | p_tan avg | p_re avg |
|---|---------|-----------|----------|---------|
| 0.05 | 14.05 (+6.4%) | 8.05 (+1.8%) | **29.45 (-0.1%)** | 6.95 (+6.9%) |
| 0.10 | 15.55 (+17.8%) | 8.55 (+8.1%) | 29.60 (+0.4%) | 7.30 (+12.3%) |
| 0.20 | 16.30 (+23.5%) | 9.95 (+25.8%) | 30.90 (+4.8%) | 8.20 (+26.2%) |
| Old baseline | 13.20 | 7.91 | 29.48 | 6.50 |
| **Current baseline** | **13.05** | **7.70** | **28.60** | **6.55** |

W&B group: `phase6/dsdf-spatial-dropout`. Runs: mw6tn1ku, wzd9lyua (p=0.05), 3pwfoanz, h9vkjcli (p=0.10), wm1objzw, 4vpd6u0c (p=0.20). Ran against OLD baseline (no GSB). 150-151 epochs.

**Results commentary:**
- Clear dose-response: all metrics degrade monotonically with higher dropout. DSDF features carry essential information.
- p=0.05 gives essentially flat p_tan (-0.1%) while p_in regresses 6.4%. Not worth it.
- Against current baseline (28.60), all configurations clearly worse.
- Seed 42 at p=0.05 gave p_tan=28.9 (promising single-run) but seed 73 was 30.0 — high variance, not reliable.
- **Dead end.** DSDF channels too information-dense to drop.
- Thorfinn reassigned to actual 3-way PCGrad test (#2147).

---

### 2026-04-05 ~05:30 — PR #2142: Cross-Seed Model Soup — frieren — **CLOSED** (dead end)

- Branch: `frieren/model-soup`
- Hypothesis: Average EMA weights from 3 independently trained seeds (42, 73, 91). Cross-seed weight averaging produces flatter loss basin model (Wortsman et al., ICML 2022).

| Config | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|--------|-------|------|-----|
| Seed 42 (individual) | 12.96 | 7.47 | **29.10** | 6.55 | vs5z9ji8 |
| Seed 73 (individual) | 13.02 | 7.70 | 29.56 | 6.42 | xla7mati |
| Seed 91 (individual) | 13.73 | 7.98 | 29.40 | 6.40 | 8ziujhj6 |
| **3-seed arithmetic mean** | **13.24** | **7.72** | **29.35** | **6.46** | — |
| Soup 42+73 | 833.3 | 1013.3 | 573.9 | 895.0 | — |
| Soup 42+91 | 4919.8 | 4292.1 | 1473.4 | 5410.3 | — |
| Soup 73+91 | 2539.3 | 2482.2 | 1699.7 | 3165.3 | — |
| **Soup 42+73+91** | **3618.5** | **4763.9** | **1982.2** | **4582.3** | — |
| **Current baseline (PR #2130)** | **13.05** | **7.70** | **28.60** | **6.55** | — |

W&B group: `phase6/model-soup`. Runs used old baseline (no `--gap_stagger_spatial_bias`). Student added `--disable_pcgrad` (flagged for investigation). Epochs: 154.

**Results commentary:**
- **Weight averaging catastrophically fails.** All soup models (pairwise and 3-way) produce MAE 50-400x worse than individual models.
- **Root cause:** Models trained from different random seeds occupy different loss basins. Weight-space midpoint between independent local minima is NOT a minimum — it's a high-loss saddle/ridge. Model soups only work when models share initialization (fine-tuned from same checkpoint).
- Individual seeds don't beat current baseline (p_tan=28.60). Seed 42 at 29.10 beats OLD baseline (29.48) but this is on stale flags.
- **Key insight:** Prediction averaging (ensembling) works (3-seed mean 29.35 < 29.48), but weight averaging from scratch training does not.
- **Student note:** `--disable_pcgrad` may be needed for `--pcgrad_3way` to activate — flagged for investigation.
- Dead end for naive model soups. Fork-then-merge (shared initialization) remains theoretically viable.

---

### 2026-04-05 ~05:00 — PR #2130 (Round 3): GSB + PCGrad Compound Validation — fern — **MERGED** (winner)

- Branch: `fern/gap-stagger-spatial-bias`
- Hypothesis: Gap/Stagger Spatial Bias (GSB, extending spatial_bias MLP 4→6 inputs with gap/stagger) compounds with PCGrad 2-way gradient surgery.

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| GSB + PCGrad | 42 | 12.9 | 7.6 | **28.9** | 6.6 | d7l91p0x |
| GSB + PCGrad | 73 | 13.2 | 7.8 | **28.3** | 6.5 | j9btfx09 |
| **avg** | — | **13.05** | **7.70** | **28.60** | **6.55** | — |
| **Prior baseline (PR #2119)** | — | **13.20** | **7.91** | **29.48** | **6.50** | — |

W&B group: `phase6/gsb-pcgrad`. ~150-152 epochs. No `--aft_foil_srf_context`.

**Results commentary:**
- **MERGED.** GSB + PCGrad compound: p_tan 29.48→28.60 (-3.0%), p_oodc -2.7%, p_in -1.1%. Only p_re slightly misses (+0.8%).
- Both seeds beat the baseline on p_tan individually (28.9, 28.3). Seed 73 at 28.3 is the best single-seed p_tan seen this phase.
- GSB and PCGrad are orthogonal mechanisms (routing vs gradient surgery) → compound correctly.
- 3 rounds of iteration: original result (seeds 42/43, no PCGrad) → rebased with context+seeds 42/73 (failed due to context bug) → rebased with PCGrad, no context (this winner).
- **New baseline: p_tan < 28.60, p_oodc < 7.70, p_in < 13.05, p_re < 6.55.**
- New reproduce command adds `--gap_stagger_spatial_bias` to the PCGrad baseline.

---

### 2026-04-05 ~04:45 — PR #2138: Foil-2 Independent AoA Rotation Aug — edward — **CLOSED** (dead end)

- Branch: `edward/foil2-aoa-rot-aug`
- Hypothesis: Rotate aft-foil nodes independently by small angle δ ~ N(0, σ) for tandem samples. Creates novel (fore_AoA, aft_AoA) combinations.

| σ | p_in | p_oodc | p_tan | p_re |
|---|------|--------|-------|------|
| 0.02 avg | 13.0 | 7.85 | **30.85** | 6.50 |
| 0.05 avg | 13.35 | 7.75 | **30.15** | 6.50 |
| 0.10 avg | 13.55 | 7.70 | **30.60** | 6.60 |
| Old baseline | 13.04 | 7.66 | **30.11** | 6.52 |
| **Current baseline** | **13.20** | **7.91** | **29.48** | **6.50** |

W&B group: `phase6/foil2-aoa-rot-aug`. Runs: gfu688cc, v6h6nas7 (σ=0.02), etpcv1p5, mhnn4yq3 (σ=0.05), g61pmdvs, gwpe7dgh (σ=0.10). Epochs: ~149 (180-min timeout).

**Results commentary:**
- No sigma beats even the OLD baseline on p_tan. σ=0.05 is closest (30.15 vs 30.11) but within noise.
- **Fundamental flaw:** Rotating aft-foil geometry without re-simulating the flow field creates target inconsistency. Velocity (Ux, Uy) is rotated but pressure (scalar) depends nonlinearly on geometry. The augmented samples teach inconsistent geometry→pressure mappings.
- p_oodc improves monotonically with σ (7.85→7.75→7.70) — more aft-foil diversity helps OOD conditions, but at the cost of p_tan.
- Runs used old baseline flags (missing PCGrad, DSDF2 aug, has no-op aft_foil_srf_context).
- Edward reassigned to input-noise-aug (#2144).

---

### 2026-04-05 ~02:45 — PR #2136: Per-Foil Physics Normalization — thorfinn — **CLOSED** (dead end)

- Branch: `thorfinn/per-foil-pnorm`
- Hypothesis: Split Cp normalization denominator (q) per-foil for tandem samples. Aft-foil nodes use q_aft (local), fore-foil nodes use q_fore.

| Metric | Baseline | s42 | s73 | 2-seed avg | Delta |
|--------|----------|-----|-----|-----------|-------|
| p_in | 13.02 | 14.8 | 16.2 | 15.50 | +19.0% ❌ |
| p_oodc | 7.62 | 8.3 | 8.6 | 8.45 | +10.9% ❌ |
| p_tan | 29.91 | 30.9 | 31.7 | 31.30 | +4.6% ❌ |
| p_re | 6.47 | 7.4 | 6.9 | 7.15 | +10.5% ❌ |

W&B: bc8yxmb6 (s42), urq07o3e (s73). Group: `phase6/per-foil-pnorm`. Epochs: 133 (180-min timeout).

**Results commentary:**
- **All metrics regressed massively.** Root cause: phys_stats mismatch — normalization stats computed with global q, but per-foil q shifts the Cp distribution for ~30% of samples. The model had already compensated for global q via learned weights.
- p_in +19% despite single-foil being unaffected → global training destabilization.
- Per-sample for-loop also slowed training (133 vs 155 epochs).
- Runs included `--aft_foil_srf_context` (no-op bug) and `--disable_pcgrad` (old baseline).
- **Generalizable lesson:** Don't change data normalization without recomputing all derived statistics.
- Thorfinn reassigned to dsdf-spatial-dropout (#2143).

---

### 2026-04-05 ~02:30 — PR #2134: Fore-Foil TE Relative Coords + Bug Fix — frieren — **CLOSED** (dead end + critical finding)

- Branch: `frieren/foil1-relative-coords`
- Hypothesis: Add (x_rel, y_rel) relative to fore-foil trailing edge as features to AftFoilRefinementContextHead. Additionally: **critical bug fix** — changed guard from `if aft_srf_head is not None:` to `if aft_srf_head is not None or aft_srf_ctx_head is not None:` so context head actually fires.

| Config | Seed | p_in | p_oodc | p_tan | p_re | Epochs | W&B |
|--------|------|------|--------|-------|------|--------|-----|
| + rel_coords | 42 | 14.952 | 8.535 | 30.846 | 7.438 | 132 | r2gh8csf |
| + rel_coords | 73 | 15.146 | 9.221 | 31.528 | 7.049 | 133 | o0sa3a8b |
| **rel avg** | — | **15.049** | **8.878** | **31.187** | **7.244** | — | — |
| control (no rel) | 42 | 15.470 | 8.252 | 30.203 | 7.261 | 132 | 8nw4yzw3 |
| control (no rel) | 73 | 16.484 | 8.288 | 30.612 | 7.156 | 133 | wm0seo1f |
| **ctrl avg** | — | **15.977** | **8.270** | **30.408** | **7.209** | — | — |
| **Non-context baseline** | — | **13.19** | **7.92** | **30.05** | **6.45** | 155+ | — |
| **Current baseline** (PCGrad) | — | **13.20** | **7.91** | **29.48** | **6.50** | 155+ | — |

W&B group: `phase6/foil1-relative-coords`. VRAM: 38.8 GB. Runs did NOT include PCGrad or aug_dsdf2_sigma.

**CRITICAL FINDING — Context head when working is harmful:**
- These are the FIRST runs where the context head actually fires (bug fix applied).
- Control runs (context working) vs non-context baseline: p_in +21% ❌, p_oodc +4.4% ❌, p_tan +1.2% ❌, p_re +11.8% ❌
- **Undertraining confound:** Runs hit only 132-133 epochs (vs 155-160 baseline). KNN computation adds ~17% per-epoch overhead → cosine schedule (T_max=160) never completes → LR doesn't reach 0. This likely explains the catastrophic p_in and p_re regression.
- **PR #2127 improvement confirmed as seed variance.** Context head was never applied in those runs.
- **Decision:** `--aft_foil_srf_context` officially removed from baseline. The approach needs fundamental optimization (faster KNN, lower K) to be viable within the 180-minute training budget.

Rel_coords: p_tan +2.6% worse than control. TE relative coordinate frame doesn't help. Dead end.

Frieren reassigned to model-soup (#2142).

---

### 2026-04-05 ~02:15 — PR #2130 (Round 2): Gap/Stagger Spatial Bias + aft_foil_srf_context rebase — fern — **SENT BACK** (final validation: PCGrad + GSB)

- Branch: `fern/gap-stagger-spatial-bias`
- This entry covers fern's rebased validation run (added `--aft_foil_srf_context`, seeds 42/73).

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B ID |
|--------|------|------|--------|-------|------|--------|
| GSB + context | 42 | 13.5 | 7.7 | 30.8 | 6.5 | bwuglnin |
| GSB + context | 73 | 12.8 | 7.6 | 31.4 | 6.3 | td8fcpt7 |
| **GSB+ctx avg** | — | **13.15** | **7.65** | **31.1** | **6.4** | — |
| **Current baseline** | — | **13.20** | **7.91** | **29.48** | **6.50** | — |

**Results commentary:**
- GSB + aft_foil_srf_context (rebased) = p_tan 31.1 — much worse than both the original GSB result (29.82) and current baseline (29.48).
- Root causes: (1) aft_foil_srf_context is a no-op due to guard bug (frieren #2134) — its presence changes training dynamics slightly (VRAM/speed), (2) seeds 42/73 is harder than 42/43 used in original, (3) baseline has since moved with PCGrad merger.
- The original GSB signal (+0.8% p_tan improvement) remains valid but the current PCGrad baseline (29.48) is a tougher target.
- **Sent back for final validation:** GSB + PCGrad (no aft_foil_srf_context), seeds 42/73, vs current baseline 29.48.

---

### 2026-04-05 ~02:15 — PR #2129 (Round 2+3): Supervised Surface Pressure Gradient Aux Loss v2 — nezuko — **CLOSED** (exhausted)

- Branch: `nezuko/surf-pressure-gradient-aux`
- This entry covers the per-foil fix (v2) results.

| Run | W&B ID | Seed | p_in | p_oodc | p_tan | p_re |
|-----|--------|------|------|--------|-------|------|
| surfgrad-v2-w10-s42 | 9twwqobo | 42 | 13.089 | 7.843 | 29.450 | 6.390 |
| surfgrad-v2-w10-s73 | lraags91 | 73 | 13.402 | 7.732 | 30.106 | 6.559 |
| **v2 mean** | — | — | **13.246** | **7.788** | **29.778** | **6.475** |
| **Current baseline (PR #2119)** | — | — | **13.20** | **7.91** | **29.48** | **6.50** |

W&B group: `phase6/surf-grad-aux-v2`

**Results commentary:**
- **CLOSED after 3 iterations.** Per-foil fix successfully corrected the p_tan regression (from +1.1% to -0.4% vs same-epoch control), confirming the cross-foil gradient bug was real.
- Against current PCGrad baseline: p_in misses (+0.35%), p_oodc beats (-1.5%), p_tan misses (+1.0%), p_re roughly matches (+0.4%).
- Seed 42 (29.45) is very close to baseline but seed 73 (30.11) drags the mean up. High variance between seeds suggests the signal is weak.
- After 3 iterations with diminishing returns (round 1 cross-foil bug → round 2 rebase bug → round 3 correct but insufficient), the approach is exhausted.
- **p_oodc improvement (~-1.5%) is consistently real** but not sufficient to justify p_in and p_tan misses.

---

### 2026-04-05 ~01:45 — PR #2135: Tandem Self-Distillation — edward — **CLOSED** (inconclusive/negative)

- Branch: `edward/tandem-selfdistill`
- Hypothesis: Use EMA model as soft teacher for tandem samples after ema_start_epoch. KD loss (MSE between online and EMA predictions) on tandem surface pressure.

| Run | Weight | Epochs | p_in | p_oodc | p_tan | p_re | W&B ID |
|-----|--------|--------|------|--------|-------|------|--------|
| w=0.05 s42 | 0.05 | 136 | 13.6 | 8.3 | 30.4 | 6.7 | swsmtceu |
| w=0.10 s42 | 0.10 | 219 | 14.1 | 10.8 | 38.2 | 25.9 | ewrhonfl |
| w=0.20 s42 | 0.20 | 220 | 14.9 | 10.4 | 35.6 | 25.6 | d6x7l5cg |
| ctrl s42 | — | 252 | 15.0 | 9.9 | 36.5 | 25.4 | ivs2xoxz |

W&B group: `phase6/tandem-selfdistill`

**Results commentary:**
- **INCONCLUSIVE/NEGATIVE.** Post-cosine degradation killed 3 of 4 runs (GPUs ran faster than expected → exceeded cosine_T_max=160 → severe OOD regression).
- Only w=0.05 (136 epochs) was valid; all metrics regress vs baseline (p_tan +1.6%, p_oodc +8.9%).
- KD had only 36 epochs of exposure (EMA starts at epoch 100) — barely enough time to test.
- **Important finding:** GPU speed variance with --aft_foil_srf_context causes some GPUs to run 250+ epochs, well past cosine_T_max=160.
- Also confounded by aft_foil_srf_context guard bug (frieren #2134) — context head was never applied.
- Edward reassigned to foil2-aoa-rot-aug (#2138).

---

### 2026-04-05 ~01:15 — PR #2133: Foil-1 DSDF Magnitude Augmentation — tanjiro — **CLOSED** (dead end)

- Branch: `tanjiro/foil1-dsdf-mag-aug`
- Hypothesis: Mirror of DSDF2 aug (PR #2126) for foil-1 channels. Log-normal scaling of x[:,4:8] (foil-1 DSDF) to force front-foil shape generalization. σ sweep {0.05, 0.10, 0.15}.

| σ | Seed | p_in | p_oodc | p_tan | p_re | W&B ID |
|---|------|------|--------|-------|------|--------|
| 0.05 | 42 | 13.10 | 7.53 | 30.31 | 6.40 | k1wy49ut |
| 0.05 | 73 | 13.49 | 7.76 | 30.76 | 6.43 | fnlvz7nq |
| **σ=0.05 avg** | — | **13.30** | **7.65** | **30.53** | **6.41** | — |
| 0.10 | 42 | 13.36 | 7.58 | 31.06 | 6.43 | blimyxen |
| 0.10 | 73 | 13.51 | 7.51 | 31.17 | 6.22 | ft2fidfq |
| **σ=0.10 avg** | — | **13.43** | **7.55** | **31.12** | **6.33** | — |
| 0.15 | 42 | 12.62 | 7.57 | 31.53 | 6.42 | h8qcpxig |
| 0.15 | 73 | 13.38 | 8.10 | 30.47 | 6.52 | bp75nqq0 |
| **σ=0.15 avg** | — | **13.00** | **7.83** | **31.00** | **6.47** | — |

W&B group: `phase6/dsdf1-mag-aug`. Runs include --aug_dsdf2_sigma 0.05 but NOT --aft_foil_srf_context.

**Results commentary:**
- **DEAD END.** p_tan regresses at ALL σ values (+1.4% to +3.4% vs DSDF2 baseline, +2.1% to +4.0% vs current baseline).
- Asymmetry with foil-2 aug explained: val_tandem_transfer uses NACA6416 as FRONT foil — augmenting the known front-foil DSDF adds noise rather than generalization.
- p_oodc marginally improved at σ=0.10 (7.55), but irrelevant given p_tan regression.
- Combined DSDF1+DSDF2 aug (research priority #5) is now also dead — foil-1 aug doesn't help.
- Tanjiro reassigned to ema_perturb (#2137).

---

### 2026-04-05 ~00:25 — PR #2132: Tandem DSDF Channel Mixup — thorfinn — **CLOSED** (dead end)

- Branch: `thorfinn/tandem-dsdf-mixup`
- Hypothesis: Interpolate DSDF channels between tandem training samples to create synthetic intermediate foil geometries, expanding the model's exposure to novel shapes. Alpha sweep {0.7, 0.5}.

| Run | Alpha | Seed | p_in | p_oodc | p_tan | p_re | W&B ID |
|-----|-------|------|------|--------|-------|------|--------|
| a07-s42 | 0.7 | 42 | 12.419 | 7.608 | 31.628 | 5.489 | 6asz230g |
| a07-s43 | 0.7 | 43 | 13.243 | 8.056 | 30.285 | 6.802 | 2f5h1bqy |
| **α=0.7 avg** | — | — | **12.831** | **7.832** | **30.957** | **6.146** | — |
| a05-s42 | 0.5 | 42 | 13.130 | 7.774 | 30.983 | 5.975 | oladith7 |
| a05-s43 | 0.5 | 43 | 13.852 | 9.469 | 30.541 | 6.677 | v5b43gi4 |
| **α=0.5 avg** | — | — | **13.491** | **8.621** | **30.762** | **6.326** | — |

W&B group: `phase6/tandem-dsdf-mixup`

**Results commentary:**
- **DEAD END.** p_tan regresses in both variants (+3.5% and +2.8% vs current baseline 29.91).
- α=0.5/s43 p_oodc=9.47 — catastrophic. High variance throughout.
- Student correctly identified root cause: mixing NACA0012 DSDFs creates more NACA0012-like shapes, not novel geometries toward NACA6416.
- p_re improved strongly (α=0.7 avg: 6.15, -5.0% vs baseline 6.47) — interesting but secondary.
- Closed. Thorfinn reassigned to per_foil_pnorm.

---

### 2026-04-05 ~00:10 — PR #2131: Tandem-Slice Carve-Out — alphonse — **SENT BACK** (rebase + 2-seed validation)

- Branch: `alphonse/tandem-slice-carveout`
- Hypothesis: Reserve K physics slices exclusively for tandem samples by applying large negative logit bias (-100) to reserved slices for single-foil. Prevents single-foil from co-opting tandem-specialized capacity.

| Config | p_in | p_oodc | p_tan | p_re | W&B ID |
|--------|------|--------|-------|------|--------|
| Control s42 | 12.480 | 7.407 | 30.767 | 6.492 | wwil2gdr |
| Control s43 | 13.243 | 7.940 | 30.907 | 6.477 | bfa65aup |
| **Control avg** | **12.861** | **7.673** | **30.837** | **6.485** | — |
| K=4 s42 | 12.801 | 7.772 | 30.059 | 6.545 | z3b8tdfy |
| K=4 s43 | 13.225 | 7.700 | 29.358 | 6.303 | p5pljk4j |
| **K=4 avg** | **13.013** | **7.736** | **29.709** | **6.424** | — |
| K=8 s42 | 13.318 | 7.675 | 29.303 | 6.455 | nicjx1g0 |
| K=8 s43 | 13.316 | 7.939 | 31.550 | 6.687 | a4jaduno |
| **K=8 avg** | **13.317** | **7.807** | **30.427** | **6.571** | — |

W&B group: `phase6/tandem-slice-carveout`. Confirmed `--aug_gap_stagger_sigma 0.02` active in all runs.

**Results commentary:**
- **K=4 is the clear winner.** p_tan -3.7% vs control (29.71 vs 30.84). Both seeds improve. Strongest p_tan signal of this round.
- **K=8 too aggressive.** High variance (seed 43 catastrophic at 31.55). p_in regresses +3.6%.
- **K=4 vs current baseline (PR #2127):** p_tan 29.71 < 29.91 ✅, p_re 6.42 < 6.47 ✅, p_in 13.01 ≈ 13.02 ✅. Only p_oodc regresses (7.74 vs 7.62, +1.5%).
- Runs missing `--aft_foil_srf_context`. Sent back for rebased 2-seed validation.
- **Note:** p_oodc regression expected — fewer shared slices = less OOD-condition generalization.

---

### 2026-04-05 ~00:00 — PR #2130: Gap/Stagger-Conditioned Spatial Bias — fern — **SENT BACK** (rebase + 2-seed validation)

- Branch: `fern/gap-stagger-spatial-bias`
- Hypothesis: Extend Transolver's spatial_bias MLP from 4→6 inputs by appending (gap, stagger) scalars. Makes slice routing tandem-geometry-aware. Zero effect on single-foil (gap=0, stagger=0). Zero-init on new weights ensures identical routing at epoch 0.

| Config | p_in | p_oodc | p_tan | p_re | W&B ID |
|--------|------|--------|-------|------|--------|
| Control s42 | 13.103 | 7.464 | 29.843 | 6.522 | yrckevun |
| Control s43 | 13.136 | 7.773 | 30.249 | 6.551 | fwof1f72 |
| **Control avg** | **13.120** | **7.618** | **30.046** | **6.537** | — |
| GSB s42 | 12.819 | 7.394 | 29.739 | 6.529 | mh5sy993 |
| GSB s43 | 13.275 | 7.778 | 29.898 | 6.469 | agndk2w9 |
| **GSB avg** | **13.047** | **7.586** | **29.819** | **6.499** | — |

W&B group: `phase6/gap-stagger-spatial-bias`

**Results commentary:**
- **GSB beats control on ALL 4 metrics:** p_in -0.6%, p_oodc -0.4%, p_tan **-0.8%**, p_re -0.6%.
- GSB avg p_tan=29.82 already beats the current baseline (29.91) WITHOUT `--aft_foil_srf_context`.
- Feature index correction: gap=22, stagger=23 (not 21, 22 as in PR instructions). Student caught this.
- VRAM: 38.4 GB — identical to baseline (negligible parameter overhead).
- **Sent back** for rebased 2-seed validation with `--aft_foil_srf_context`. Expect compounding — merge if p_tan < 29.91.

---

### 2026-04-05 ~02:00 — PR #2119: PCGrad 2-Way (10-seed) — askeladd — **MERGED** (winner)

- Branch: `askeladd/pcgrad-3way`
- Hypothesis: PCGrad gradient surgery between single-foil and all-tandem batches (2-way). Prevents single-foil signal from corrupting tandem representation. 10 seeds total.

**8-seed validation (phase6/pcgrad-2way-validation):**
| Seed | p_in | p_oodc | p_tan | p_re | W&B |
|------|------|--------|-------|------|-----|
| 42-49 | 13.20 ± 0.26 | 7.91 ± 0.16 | **29.48 ± 0.43** | 6.50 ± 0.12 | tmqq1xlo...75d4hhzm |

**Rebased 2-seed (phase6/pcgrad-2way-rebased):**
| Seed | p_in | p_oodc | p_tan | p_re | W&B |
|------|------|--------|-------|------|-----|
| 42 | 13.018 | 7.906 | **28.851** | 6.504 | jpe1t13t |
| 73 | 12.824 | 7.973 | 30.095 | 6.537 | cdccuyl7 |
| **mean** | **12.921** | **7.940** | **29.473** | **6.521** | — |

**Results commentary:**
- **MERGED.** p_tan -1.9% (8-seed mean: 29.48 vs baseline 30.05). Consistent across 10 seeds.
- p_in improved (-2.4% rebased, -0.1% vs 8-seed baseline).
- p_oodc flat (+0.4% vs aft_srf-only baseline, regressed vs gap_stagger baseline).
- VRAM: 45-46 GB (+22% from 3 backward passes).
- New baseline command adds `--pcgrad_3way --pcgrad_extreme_pct 0.15` (removes `--disable_pcgrad`).
- NOTE: `--aft_foil_srf_context` was a no-op in all runs due to guard bug (frieren #2134).

---

### 2026-04-04 ~23:45 — PR #2119: PCGrad 2-Way Validation (8-seed) — askeladd — **SENT BACK** (rebase + 2-seed validation)

- Branch: `askeladd/pcgrad-3way`
- Hypothesis: 3-way PCGrad (single-foil / tandem-normal / tandem-extreme-Re) to resolve gradient conflicts. Due to batch_size=4 confound, effectively tested **2-way PCGrad** (single-foil vs all-tandem). 8-seed validation (seeds 42-49).

| Seed | p_in | p_oodc | p_tan | p_re | W&B ID |
|------|------|--------|-------|------|--------|
| 42 | 13.463 | 7.685 | 30.021 | 6.445 | tmqq1xlo |
| 43 | 13.550 | 7.972 | 28.678 | 6.456 | 84aff7cq |
| 44 | 12.880 | 7.758 | 29.221 | 6.656 | 23y1pfj5 |
| 45 | 12.816 | 7.780 | 29.357 | 6.277 | yw2djp6d |
| 46 | 13.158 | 8.004 | 29.691 | 6.635 | afis6090 |
| 47 | 13.482 | 8.218 | 29.206 | 6.605 | c80t1a69 |
| 48 | 13.219 | 7.993 | 30.016 | 6.566 | xcmpfkqs |
| 49 | 13.051 | 7.899 | 29.671 | 6.391 | 75d4hhzm |
| **8-seed mean** | **13.202 ± 0.261** | **7.913 ± 0.160** | **29.483 ± 0.427** | **6.504 ± 0.125** | — |
| **Old baseline** | **13.19 ± 0.33** | **7.92 ± 0.17** | **30.05 ± 0.36** | **6.45 ± 0.07** | — |

W&B group: `phase6/pcgrad-2way-validation`

**Results commentary:**
- **p_tan -1.9% (29.48 vs 30.05)** — real and consistent. 7/8 seeds below baseline. Best: seed 43 (28.68).
- p_oodc flat (+0.1%), p_in flat (+0.1%), p_re slight regression (+0.8%).
- **Confound:** 3-way split never fires with batch_size=4 (identical metrics for pct=0.10 and pct=0.15 on seed 42). Effectively 2-way PCGrad.
- **Missing features:** Runs lacked `--aft_foil_srf_context` and `--aug_gap_stagger_sigma 0.02` from current baseline.
- **VRAM:** 45-46 GB per run (+22% from 3 backward passes). Unknown if compatible with aft_foil_srf_context (69-95 GB).
- **Sent back** for rebased 2-seed validation with full current baseline flags. If p_tan < 29.91, merge immediately.

---

### 2026-04-04 ~23:30 — PR #2129: Supervised Surface Pressure Gradient Aux Loss — nezuko — **SENT BACK** (revision requested)

- Branch: `nezuko/surf-pressure-gradient-aux`
- Hypothesis: Add an auxiliary L1 loss on the chord-wise first-order pressure gradient (finite differences between consecutive surface nodes sorted by x-coordinate). Forces the model to reproduce the spatial structure of the Cp distribution, not just point-wise accuracy. Tested at weight=0.05 and weight=0.10.

| Run | W&B ID | Weight | Seed | p_in | p_oodc | p_tan | p_re |
|-----|--------|--------|------|------|--------|-------|------|
| surfgrad-w05-s42 | bnskf03a | 0.05 | 42 | 13.518 | 7.574 | 30.387 | 6.451 |
| surfgrad-w05-s73 | yfy3efjq | 0.05 | 73 | 13.644 | 7.854 | 30.826 | 6.650 |
| surfgrad-w10-s42 | s2txmpe9 | 0.10 | 42 | 13.113 | 7.836 | 30.311 | 6.475 |
| surfgrad-w10-s73 | 3is100bp | 0.10 | 73 | 13.185 | 7.631 | 30.428 | 6.393 |
| **w=0.05 avg** | — | 0.05 | — | **13.581** | **7.714** | **30.607** | **6.551** |
| **w=0.10 avg** | — | 0.10 | — | **13.149** | **7.734** | **30.370** | **6.434** |
| **Current baseline** | — | — | — | **13.02** | **7.62** | **29.91** | **6.47** |

W&B group: `phase6/surf-grad-aux`

**Results commentary:**
- **DOES NOT BEAT BASELINE.** Against the current baseline (PR #2127, +aft_foil_srf_context), all metrics except p_re regress.
- **w=0.10 > w=0.05** across the board. Weight=0.10 is clearly the better setting.
- **Known bug:** Runs used stale baseline (missing `--aft_foil_srf_context`). More importantly, chord-wise sorting over ALL surface nodes creates spurious gradients between fore-foil TE and aft-foil LE in tandem samples. These cross-foil gradients are physically meaningless and likely suppress the p_tan signal.
- **Sent back** with instructions to: (1) split gradient computation per-foil to eliminate cross-foil artifacts, (2) add `--aft_foil_srf_context` to match current baseline, (3) re-run w=0.10 only (2 seeds). Final iteration — close if no improvement.

---

### 2026-04-04 ~22:00 — PR #2127: Context-Aware AftSRF — KNN Volume Context — frieren — **MERGED** (winner)

- Branch: `frieren/aft-srf-knn-context`
- Hypothesis: AftFoilRefinementHead receives no non-local context. Aft-foil surface pressure depends on upstream wake state arriving from the fore-foil. Augmenting each aft-foil surface node's hidden state with the averaged hidden states of K=8 nearest zone-2 volume neighbors gives the correction head direct access to wake-state information.

| Metric | Seed 42 (zosxwjmm) | Seed 73 (twilqf1x) | 2-Seed Avg | Baseline | Δ |
|--------|--------------------|--------------------|------------|----------|---|
| p_in | 12.87 | 13.18 | **13.02** | 13.04 | -0.2% |
| p_oodc | 7.47 | 7.76 | **7.62** | 7.66 | -0.5% |
| **p_tan** | 29.96 | 29.87 | **29.91** | 30.11 | **-0.7%** |
| p_re | 6.56 | 6.37 | **6.47** | 6.52 | -1.0% |
| val/loss | 0.388 | 0.386 | 0.387 | — | — |

W&B group: `phase6/aft-srf-knn-context`

**Results commentary:**
- **MERGE DECISION:** All 4 metrics beat baseline. Primary target p_tan: 29.91 < 30.11 (-0.7%). Physical intuition holds: giving the aft-foil correction head access to upstream wake flow information improves aft-foil surface pressure.
- **Note:** Run WITHOUT `--aug_dsdf2_sigma 0.05`. KNN context alone beats the baseline that INCLUDES dsdf2 aug. This suggests the two improvements are orthogonal and may compound.
- **VRAM:** 69-95GB peak per seed — dedicated H100 required per seed. +25% per-epoch overhead vs non-context baseline.
- **New baseline flag:** `--aft_foil_srf_context` added to all future baseline runs.

---

### 2026-04-04 ~22:00 — PR #2128: Reynolds-Conditional SRF FiLM — edward — **CLOSED** (null result)

- Branch: `edward/re-conditional-srf`
- Hypothesis: Conditioning the SRF head on (Re, AoA) via FiLM modulation allows the correction MLP to adapt its strategy to the flow regime, targeting p_re and p_oodc.

| Run | p_in | p_oodc | p_tan | p_re | W&B ID |
|-----|------|--------|-------|------|--------|
| FiLM s42 | 13.34 | 7.96 | 30.57 | 6.60 | 7bnlke5o |
| FiLM s73 | 12.72 | 7.73 | 29.73 | 6.52 | yh5ixlx8 |
| **FiLM avg** | **13.03** | **7.84** | **30.15** | **6.56** | — |
| Control s42 | 12.81 | 7.79 | 29.70 | 6.83 | np3anmah |
| Control s73 | 14.16 | 7.31 | 29.86 | 6.49 | fcri663y |
| **Control avg** | **13.49** | **7.55** | **29.78** | **6.66** | — |
| **Baseline** | **13.04** | **7.66** | **30.11** | **6.52** | — |

**Results commentary:**
- **CLOSE DECISION:** FiLM conditioning on (Re, AoA) shows no improvement. FiLM avg vs baseline: p_oodc +2.4%, p_tan +0.1%, p_re +0.8%. FiLM is also worse than its own control on p_oodc and p_tan.
- **Why it fails:** The SRF head already receives Re/AoA conditioning implicitly through the Transolver backbone's AdaLN at every block. Adding explicit FiLM modulation is redundant — 960 extra parameters cannot learn anything new beyond what AdaLN already encodes.
- **Mechanism confirmed dead:** Explicit Re/AoA conditioning on the SRF head via FiLM is a dead end. Alternative: condition on local flow features rather than global scalars.

---

### 2026-04-04 ~20:30 — PR #2126: Foil-2 DSDF Magnitude Augmentation — tanjiro — **MERGED** (winner)

- Branch: `tanjiro/dsdf2-mag-aug`
- Hypothesis: Log-normal multiplicative scaling of foil-2 DSDF channels (x[:,6:10], tandem samples only) before standardization forces the model to be less reliant on memorizing exact DSDF patterns for seen foil shapes. This targets the geometric transfer gap that val_tandem_transfer probes (NACA6416 front foil, never seen in training). Analogous to gap/stagger aug but in the geometric feature space.

| Run | W&B ID | σ | Seed | p_in | p_oodc | p_tan | p_re | val_loss |
|-----|--------|---|------|------|--------|-------|------|----------|
| dsdf2-σ=0.05-s42 | hcc2q68t | 0.05 | 42 | 13.11 | 7.70 | **29.76** | 6.42 | 0.385 |
| dsdf2-σ=0.05-s73 | e9cri4mt | 0.05 | 73 | 12.96 | 7.62 | 30.46 | 6.61 | 0.388 |
| dsdf2-σ=0.10-s42 | d50erbko | 0.10 | 42 | 13.16 | 7.47 | 30.43 | 6.65 | 0.390 |
| dsdf2-σ=0.10-s73 | snai0bmf | 0.10 | 73 | 13.39 | 7.67 | 30.11 | 6.34 | 0.388 |
| dsdf2-σ=0.15-s42 | el1jybsu | 0.15 | 42 | 13.04 | 7.47 | 30.74 | 6.47 | 0.387 |
| dsdf2-σ=0.15-s73 | 3usr74hv | 0.15 | 73 | 13.09 | 7.47 | 31.71 | 6.21 | 0.391 |
| **σ=0.05 2-seed avg** | | | | **13.04** | **7.66** | **30.11** | **6.52** | |
| **Baseline (combined 8-seed)** | | | | 13.24 | 7.73 | 30.53 | 6.50 | |

**Results commentary:**

- **MERGE DECISION:** σ=0.05 beats combined baseline on primary target p_tan (-1.4%, 30.11 vs 30.53) and on p_in (-1.5%) and p_oodc (-0.9%). p_re is +0.02 (noise). **Clear merge.**

- **σ=0.05 is the sweet spot:** Larger σ (0.10, 0.15) hurts p_tan. The model needs enough DSDF perturbation to generalize but not so much that it loses geometric detail for accurate prediction.

- **Implementation detail confirmed:** Foil-2 DSDF indices are x[:,6:10] (4 channels: sdf value + gradient x/y + second foil distance). Applied BEFORE standardization, tandem detection via `x[:, 0, 22].abs() > 0.01` (gap feature in raw space).

- **Best single run (s42, σ=0.05):** p_in=13.11, p_oodc=7.70, **p_tan=29.76**, p_re=6.42 — all metrics beat the combined baseline.

- **High seed variance:** 2-seed spread on p_tan is 0.7 (29.76 vs 30.46). 8-seed validation would give more statistical confidence, but the directional signal is clear given the combined baseline std of 0.50.

- **New config flag:** `--aug_dsdf2_sigma 0.05` added to baseline.

### 2026-04-04 ~19:20 — PR #2125: Reynolds Number Perturbation Augmentation — thorfinn — CLOSED (dead end)

- Branch: `thorfinn/re-perturb-aug`
- Hypothesis: Add Gaussian noise σ=0.05 to log_Re feature (index 13) during training to improve OOD-Re robustness by domain randomization. Target: p_re < 6.50.

| Run | W&B ID | Seed | σ | p_in | p_oodc | p_tan | p_re |
|-----|--------|------|---|------|--------|-------|------|
| 1 | ecxe1ti3 | 42 | 0.05 | 13.38 | 7.82 | 30.24 | 6.51 |
| 2 | 29xvq9bz | 43 | 0.05 | 13.08 | 7.72 | 30.79 | 6.49 |
| **2-seed avg** | | | | **13.23** | **7.77** | **30.52** | **6.50** |
| **Baseline** | | | | **13.24** | **7.73** | **30.53** | **6.50** |

**Commentary:** Clear null result. All metrics within noise of baseline. Seed 43 showed a significant p_tan regression (+1.5 to 30.8), indicating Re noise interferes with tandem pressure learning. The OOD-Re split (Re=4.445M, log(Re)≈15.3) is not far enough in log-space from training Re values to benefit from domain randomization. p_re showed no improvement at all (+0.01 avg). Adding Re perturbation augmentation to confirmed dead-ends list.

### 2026-04-04 ~22:00 — PR #2123: Combined Baseline 8-Seed Validation (aft_foil_srf + gap/stagger aug) — alphonse — CLOSED (validation complete)

- Branch: `alphonse/combined-baseline-8seed`
- Hypothesis: Pure validation run — measure the TRUE combined baseline with both `--aft_foil_srf` and `--aug_gap_stagger_sigma 0.02` over 8 seeds (42-49). No code changes.

| Seed | W&B ID | p_in | p_oodc | p_tan | p_re | val/loss |
|------|--------|------|--------|-------|------|----------|
| 42 | zapen0x3 | 12.49 | 7.4 | 30.7 | 6.5 | 0.3878 |
| 43 | 3vuz3adi | 13.33 | 7.9 | 30.9 | 6.5 | 0.3918 |
| 44 | g1uhcorj | 13.57 | 7.9 | 30.8 | 6.6 | 0.3945 |
| 45 | ea551p6b | 13.21 | 7.7 | 30.4 | 6.4 | 0.3891 |
| 46 | fdf1vsi3 | 13.38 | 8.0 | 31.0 | 6.4 | 0.3944 |
| 47 | al6opl9g | 13.09 | 7.8 | 29.6 | 6.6 | 0.3850 |
| 48 | jg15oow3 | 13.49 | 7.4 | 29.9 | 6.5 | 0.3859 |
| 49 | vzm3s42y | 13.33 | 7.7 | 30.9 | 6.5 | 0.3897 |
| **Mean** | | **13.24** | **7.73** | **30.53** | **6.50** | **0.3898** |
| **Std** | | **0.33** | **0.22** | **0.50** | **0.07** | **0.0034** |

W&B group: `phase6/combined-baseline-8seed`

**Results commentary:**
- **CRITICAL FINDING:** The combination is NOT simply additive. Gap/stagger augmentation helps p_oodc (-2.4% vs aft_srf-only 7.92) but REGRESSES p_tan (+1.6% vs aft_srf-only 30.05).
- p_in and p_re are approximately neutral.
- **Implication:** Merge decision targets updated from aft_srf-only numbers to these combined numbers. All in-flight experiments running with both flags should compare against p_tan=30.53, not 30.05.
- **Follow-up consideration:** Gap/stagger augmentation may be worth dropping from the baseline in a future round if p_tan improvements stall, since it costs +0.48 on our primary metric.

---

### 2026-04-04 ~22:00 — PR #2124: Fore-Foil Stacked SRF Head (ID=6) — fern — CLOSED (dead end)

- Branch: `fern/fore-foil-srf-stacked`
- Hypothesis: Add a stacked (additive) fore_srf_head on top of the shared srf_head for fore-foil (ID=6) nodes, mirroring how aft_foil_srf works for ID=7. Unlike PR #2117 which split the shared head, this keeps the shared head untouched and adds a pure correction.

| Config | Seed | W&B ID | p_in | p_oodc | p_tan | p_re | val/loss |
|--------|------|--------|------|--------|-------|------|----------|
| Baseline (aft_srf only) | 42 | 4e65pcn2 | 13.1 | 7.5 | 29.8 | 6.5 | 0.3856 |
| Baseline (aft_srf only) | 43 | ic3133oe | 13.1 | 7.8 | 30.2 | 6.6 | 0.3888 |
| fore_srf 192/3L | 42 | 5er3uk4r | 13.4 | 8.1 | 30.7 | 6.6 | 0.3955 |
| fore_srf 192/3L | 43 | 819huufp | 13.3 | 8.1 | 29.9 | 6.6 | 0.3874 |
| fore_srf 128/2L | 42 | abpealjg | 13.1 | 7.9 | 29.6 | 6.4 | 0.3835 |
| fore_srf 128/2L | 43 | mmovvvih | 13.7 | 8.1 | 32.8 | 6.6 | 0.4005 |

W&B group: `phase6/fore-foil-srf-stacked`

2-seed means vs baseline:
- **192/3L:** p_in=13.35, p_oodc=8.1, p_tan=30.3, p_re=6.6 — worse across the board
- **128/2L:** p_in=13.4, p_oodc=8.0, p_tan=31.2, p_re=6.5 — worse, high variance (s43 p_tan=32.8)
- **Control:** p_in=13.1, p_oodc=7.65, p_tan=30.0, p_re=6.55

**Results commentary:**
- Both formulations degrade p_oodc by ~0.4 (8.0-8.1 vs 7.65 control), suggesting optimizer interference.
- 128/2L s43 is catastrophic (p_tan=32.8), indicating the small head is unstable.
- Combined with PR #2117 (split formulation, +9-11% p_tan regression), this definitively closes fore-foil SRF.
- **Root cause:** Fore-foil (ID=6) sees channel-accelerated flow that the shared srf_head already handles well. Unlike aft-foil wake flow (ID=7), there is no qualitatively distinct correction signal for a dedicated head to learn.
- **Conclusion:** Fore-foil SRF is a dead end in all formulations. Boundary-type improvements for ID=6 must come through other mechanisms (e.g., spatial bias conditioning, trunk-level awareness).

---

### 2026-04-04 ~20:30 — PR #2122: Phase 6: Fore-Foil Loss Upweighting (ID=6) — nezuko — CLOSED (dead end)

- Branch: `nezuko/fore-foil-loss-weight`
- Hypothesis: Upweighting fore-foil (ID=6) surface nodes in the main surface pressure L1 loss by 1.5–2.0× would improve fore-foil pressure predictions, propagating downstream benefits via better wake representation. Symmetric to the aft-foil loss upweighting idea (PR #2121).

| Run | W&B ID | Weight | Seed | p_in | p_oodc | p_tan | p_re |
|-----|--------|--------|------|------|--------|-------|------|
| forefoil-lw15-s42 | swgrtft1 | 1.5 | 42 | 12.761 | 7.839 | 30.189 | 6.556 |
| forefoil-lw15-s73 | mhihrtht | 1.5 | 73 | 14.152 | 7.715 | 30.833 | 6.277 |
| forefoil-lw20-s42 | rn1v2g8z | 2.0 | 42 | 12.946 | 7.730 | 30.372 | 6.797 |
| forefoil-lw20-s73 | jab8a2yz | 2.0 | 73 | 13.396 | 7.837 | 30.869 | 6.418 |

W&B group: `phase6/fore-foil-loss-weight`

2-seed means vs baseline (p_in=13.19, p_oodc=7.92, p_tan=30.05, p_re=6.45):
- weight=1.5: p_in=13.457 (+2.0%), p_oodc=7.777 (-1.8%), p_tan=30.511 (+1.5%), p_re=6.417 (-0.5%)
- weight=2.0: p_in=13.171 (-0.1%), p_oodc=7.784 (-1.7%), p_tan=30.621 (+1.9%), p_re=6.608 (+2.4%)

**Results commentary:**
- p_oodc improves consistently (-1.7% to -1.8%) across both weight settings.
- **p_tan regresses in ALL configurations** (+1.5% to +1.9%) — this is the critical failure since p_tan is our primary target.
- Root cause: upweighting fore-foil nodes steals gradient from aft-foil nodes, directly hurting tandem transfer predictions.
- Combined with PR #2121 (aft-foil loss upweighting, same pattern: p_oodc improves but p_tan regresses), this definitively closes the entire loss-region-upweighting approach.
- **Conclusion:** Surface loss reweighting by boundary region is a dead end. The p_oodc benefit comes at the cost of p_tan. Boundary-specific improvements must be delivered architecturally (dedicated heads), not through loss weighting.

### 2026-04-04 ~20:10 — PR #2120: Phase 6: Langevin Gradient Noise (SGLD) — edward — CLOSED (dead end)

- Branch: `edward/langevin-noise`
- Hypothesis: Adding isotropic Gaussian noise after each Lion optimizer step (SGLD-style) would encourage exploration of flatter loss basins, improving generalization metrics. Noise annealed to 0 at epoch 128.
- **Note:** Used `--ema_start_epoch 100` (default 140 doesn't activate within 180-min timeout at ~82s/epoch). All runs used internal baseline controls (noise=0, same config) for fair comparison.

| Run | W&B ID | Noise Scale | Seed | p_in | p_oodc | p_tan | p_re |
|-----|--------|-------------|------|------|--------|-------|------|
| Baseline | vhz8bzct | 0 | 42 | 12.9 | 7.9 | 29.9 | 6.6 |
| Baseline | ei43kzqz | 0 | 73 | 13.1 | 8.0 | 30.1 | 6.5 |
| Langevin | cbm4a9fs | 5e-5 | 42 | 13.4 | 7.7 | 29.9 | 6.6 |
| Langevin | 13f5pbul | 5e-5 | 73 | 13.3 | 7.7 | 30.0 | 6.4 |
| Langevin | twgrbt24 | 1e-4 | 42 | 13.0 | 7.9 | 30.9 | 6.5 |
| Langevin | 3ao1faty | 1e-4 | 73 | 13.8 | 8.0 | 30.2 | 6.3 |
| Langevin | ff02he97 | 3e-4 | 42 | 13.4 | 7.9 | 31.2 | 6.4 |
| Langevin | namev902 | 3e-4 | 73 | 13.0 | 8.2 | 29.7 | 6.6 |

W&B group: `phase6/langevin-noise`

**Results commentary:**
- No meaningful improvement over internal control baseline at any noise scale.
- p_oodc marginal improvement at 5e-5 (7.7 vs 7.95) is within typical seed variance (±0.2).
- p_tan highly variable at 3e-4 (29.7 to 31.2) — noise adds instability without benefit.
- **Root cause:** Lion's sign-based update already provides implicit gradient noise (all updates are ±lr regardless of gradient magnitude). Adding isotropic Gaussian noise on top is redundant. EMA (decay=0.999 over ~16,500 batches from epoch 100) further smooths out any exploration benefit.
- **Conclusion:** Rules out the entire family of continuous gradient perturbation approaches for Lion optimizer.

---

### 2026-04-04 ~19:00 — PR #2119: Phase 6: PCGrad 3-Way Task Split — Gradient Surgery — askeladd — SENT BACK (8-seed validation)

- Branch: `askeladd/pcgrad-3way`
- Hypothesis: 3-way PCGrad splitting samples into single-foil / tandem-normal / tandem-extreme-Re for gradient surgery to reduce gradient conflicts between tasks.
- **Critical confound:** batch_size=4 gives ~2 tandem samples per batch; the ≥4 threshold for extreme-Re quantile detection almost never triggers. Runs pct=0.15/s42 and pct=0.10/s42 are **identical** (confirmed via W&B). The experiment effectively tested **2-way PCGrad** (single-foil vs all-tandem).

| Run | W&B ID | Config | Seed | p_in | p_oodc | p_tan | p_re |
|-----|--------|--------|------|------|--------|-------|------|
| 1 | p9oupnt2 | pct=0.15 | 42 | 13.49 | 7.64 | 30.04 | 6.43 |
| 2 | 6sp2uazt | pct=0.15 | 73 | 12.94 | 7.49 | 29.37 | 6.45 |
| 3 | kov6n0rs | pct=0.10 | 42 | 13.49 | 7.64 | 30.04 | 6.43 |
| 4 | l308y9lx | pct=0.10 | 73 | 13.41 | 7.82 | 29.03 | 6.43 |
| Baseline (8-seed mean) | | | | 13.19 | 7.92 | 30.05 | 6.45 |

W&B group: `phase6/pcgrad-3way`

**Results commentary:**
- **p_oodc improves in ALL 4 runs** (7.49–7.82 vs 7.92, -1.3% to -5.4%). This is the most consistent signal.
- p_tan improves substantially for seed 73 (29.03, 29.37 vs 30.05) but is flat for seed 42 (30.04). High seed variance.
- p_in inconsistent: 12.94 (good) vs 13.41/13.49 (worse).
- p_re marginally better across the board.
- VRAM overhead: ~45-46 GB (up from ~38 GB, +22% for 3 backward passes).
- **Key insight:** 2-way PCGrad (single-foil vs all-tandem) produces a reliable p_oodc improvement. The original PCGrad (disabled with `--disable_pcgrad`) split differently (indist vs all-OOD). The new single/tandem boundary is a cleaner split.
- **Decision:** Sent back for 8-seed validation (seeds 42-49) with `--aug_gap_stagger_sigma 0.02` added. If 8-seed mean p_oodc < 7.92, merge.

---

### 2026-04-04 ~17:30 — PR #2121: Phase 6: Aft-Foil Loss Upweighting — Stronger Gradient for Wake Nodes — tanjiro — CLOSED

- Branch: `tanjiro/aft-foil-loss-weight`
- Hypothesis: Upweighting aft-foil (ID=7) surface nodes in the main L1 loss by 1.5–2.0x would force the Transolver trunk to develop richer representations for wake-region nodes, complementing the existing `--aft_foil_srf` post-hoc correction head and reducing p_tan.

| Run | W&B ID | p_in | p_oodc | p_tan | p_re | val_loss |
|-----|--------|------|--------|-------|------|----------|
| w=1.5 s42 | ghq78pj6 | 12.95 | 7.75 | 30.57 | 6.50 | 0.3896 |
| w=1.5 s73 | 4r0rksil | 13.80 | 7.77 | 30.53 | 6.36 | 0.3932 |
| w=2.0 s42 | svqikz02 | 13.76 | 7.75 | 29.91 | 6.57 | 0.3886 |
| w=2.0 s73 | rk53nwzs | 12.96 | 7.84 | 31.08 | 6.49 | 0.3921 |
| **w=1.5 mean** | | **13.37** | **7.76** | **30.55** | **6.43** | |
| **w=2.0 mean** | | **13.36** | **7.80** | **30.50** | **6.53** | |
| Baseline | | 13.19 | 7.92 | 30.05 | 6.45 | |

W&B group: `phase6/aft-foil-loss-weight`

**Results commentary:**
- p_oodc consistently beats baseline across all 4 runs (7.75-7.84 vs 7.92), confirming that aft-foil loss upweighting improves trunk wake-region representations for OOD-C generalization.
- p_tan (primary target) regresses by +1.5-1.7% at both weights. The w=2.0/s42 p_tan=29.91 is contradicted by s73 at 31.08 — seed variance, not signal.
- p_in and p_re are mixed across seeds.
- **Key insight:** Foil-specific loss weighting improves p_oodc but trades off against p_tan. The SRF head may absorb most trunk-level benefit before it reaches tandem predictions.
- **Adds to knowledge:** Loss reweighting for specific surface regions helps OOD-C but hurts tandem — suggests the trunk's aft-foil representations are already adequate for p_tan, and forcing them further creates imbalance.

---

### 2026-04-04 ~17:30 — PR #2107: Phase 6: Aft-Foil Coordinate Frame Normalization — Equivariant Tandem Repr — frieren — CLOSED

- Branch: `frieren/aft-foil-local-frame`
- Hypothesis: Subtracting the aft-foil (boundary_id=7) centroid from its node coordinates before the Transolver embedding should decouple shape from global position, improving tandem OOD generalization (p_tan).
- 3 iterations: v1 coord replace → v2 dual frame → v2 4-seed validation

| Run | W&B ID | p_in | p_oodc | p_tan | p_re | val_loss |
|-----|--------|------|--------|-------|------|----------|
| v1-replace s42 | 00lod6uk | 13.83 | 8.20 | 30.17 | 6.59 | 0.3935 |
| v1-replace s73 | 3llpj5yj | 13.55 | 8.16 | 29.51 | 6.62 | 0.3890 |
| dual-frame s42 | 7e0dma73 | 12.91 | 7.93 | 31.22 | 6.64 | 0.3923 |
| dual-frame s73 | w8cyqceg | 13.15 | 7.75 | 29.74 | 6.38 | 0.3858 |
| v2-4seed s42 | bklq38ec | 13.90 | 7.84 | 30.11 | 6.46 | 0.3903 |
| v2-4seed s73 | qlkaovuv | 13.34 | 7.98 | 29.97 | 6.46 | 0.3896 |
| v2-4seed s44 | bw5ny846 | 13.28 | 7.96 | 31.19 | 6.63 | 0.3944 |
| v2-4seed s45 | 74cxcgue | 13.80 | 7.67 | 29.64 | 6.48 | 0.3867 |
| **v2 4-seed mean** | | **13.58** | **7.86** | **30.23** | **6.51** | |
| Baseline | | 13.19 | 7.92 | 30.05 | 6.45 | |
| **Δ** | | **+3.0%** | **-0.7%** | **+0.6%** | **+0.9%** | |

W&B groups: `phase6/aft-foil-local-frame`, `phase6/aft-foil-local-frame-v2`

**Results commentary:**
- After 3 iterations and 8 runs, the dual-frame approach produces a genuine p_oodc signal but cannot reliably beat baseline on primary metrics.
- p_in regression (+3.0%) is the main blocker — the extra features add noise for non-tandem samples.
- p_tan seed variance is very high (range 1.55, 29.64-31.19), making it impossible to confirm improvement.
- Best individual seeds beat all metrics, but not simultaneously.
- **Key insight:** Local coordinate frames for specific foils improve individual-seed OOD but increase variance and regress p_in. Geometric auxiliary features need a sparser, more targeted formulation.
- **Adds to knowledge:** Coordinate transforms as input features are a noisy delivery mechanism for geometric information. Architecture-level changes (dedicated heads like SRF) are more reliable.

---

### 2026-04-04 ~16:30 — PR #2118: Phase 6: Boundary-ID One-Hot Feature — Explicit Surface-Type Conditioning — thorfinn — CLOSED

- Branch: `thorfinn/boundary-id-onehot`
- Hypothesis: Append a 3-dim one-hot vector encoding boundary type (single-foil ID=5, fore-foil ID=6, aft-foil ID=7) as explicit input feature to all mesh nodes. Non-surface nodes get zero vector. Goal: let Transolver route computation differently by boundary type from the first layer.

| Seed | p_in | p_oodc | p_tan | p_re | val/loss | W&B Run ID |
|------|------|--------|-------|------|----------|------------|
| 42 | 13.05 | 7.72 | 30.37 | 6.58 | 0.3899 | txl1svzm |
| 43 | 13.16 | 7.61 | 30.97 | 6.51 | 0.3917 | 12uns5n9 |
| **2-seed avg** | **13.10** | **7.66** | **30.67** | **6.55** | | |
| Baseline | 13.19 | 7.92 | 30.05 | 6.45 | | |
| **Δ** | **-0.7%** | **-3.3%** | **+2.1%** | **+1.5%** | | |

W&B group: `phase6/boundary-id-onehot`

**Results commentary:**
- p_in and p_oodc show modest improvements on both seeds, but the PRIMARY target p_tan regresses +2.0% (s42) and +5.8% (s43). p_re also regresses on both seeds.
- Student's analysis (correct): sparse 3-dim one-hot firing on only ~0.4% of nodes (surface nodes) disrupts Transolver's physics-aware slice assignment for the 99.6% volume nodes that receive the zero vector.
- Boundary-ID detection confirmed working via sanity logs (bid/frac_single ~0.004, bid/frac_fore ~0.002, bid/frac_aft ~0.002).
- **Conclusion:** Sparse surface-only features appended to the full node set are the wrong delivery mechanism for boundary-type identity. Rules out this approach. The aft-foil SRF head (#2104) works because it's a dedicated output correction, not an input feature.
- **Adds to dead-end knowledge:** boundary-type information must be delivered through architecture (dedicated heads) not through sparse input features.

---

### 2026-04-04 15:20 — PR #2117: Phase 6: Fore-Foil Dedicated SRF Head (ID=6) — Split from Single-Foil — fern — CLOSED

- Branch: `fern/fore-foil-srf-split`
- Hypothesis: Give the fore-foil (boundary ID=6) its own dedicated Surface Refinement Head, split away from the shared `srf_head` (which previously served both single-foil ID=5 and fore-foil ID=6 nodes). Motivated by the successful aft-foil SRF (#2104).

| Config | Seeds | p_in | p_oodc | p_tan | p_re | W&B Runs |
|--------|-------|------|--------|-------|------|----------|
| Control (aft_srf only) | 42,43 | 13.30 | 7.70 | **29.45** | 6.40 | sqfe4eih, 69va7w4w |
| +fore_srf 192/3L | 42,43 | 13.20 | 7.75 | 32.15 | 6.45 | oiygmwit, o3ziow82 |
| +fore_srf_lg 256/4L | 42,43 | 13.45 | 8.00 | 32.75 | 6.60 | qkp9400n, w62a6akm |

W&B group: `phase6/fore-foil-srf`

**Results commentary:**
- **Clear dead end — both configs regress p_tan by +9–11%** vs. control. Larger capacity is strictly worse.
- Root cause identified by fern: when `--fore_foil_srf` is active, the shared `srf_head` was narrowed to single-foil nodes only, losing transfer learning from tandem data. The fore-foil head sees only ~200 nodes in ~40% of training samples — insufficient signal.
- Key contrast: aft-foil SRF (#2104) ADDED an extra head on top of the shared head without narrowing it. The fore-foil splitting approach inverts this design.
- **Action:** Stacked fore-foil refinement (additive on top of shared srf_head, no narrowing) assigned to fern as the natural follow-up.

---

### 2026-04-04 15:15 — PR #2116: Phase 6: Charbonnier Loss — Fully Smooth L1 — alphonse — CLOSED

- Branch: `alphonse/charbonnier-loss`
- Hypothesis: Replace L1 `abs()` with Charbonnier `sqrt((pred-y)^2 + eps^2) - eps` for fully smooth (C∞) gradient everywhere. The hypothesis: L1's constant ±1 gradient ignores residual magnitude; Charbonnier's smooth interpolation provides proportional signal for small residuals.

| Config | Seed | p_in | p_oodc | p_tan | p_re | val/loss | W&B Run ID |
|--------|------|------|--------|-------|------|----------|------------|
| Baseline (L1) | 42 | 12.87 | 7.90 | 30.4 | 6.5 | 0.3906 | wemzzic2 |
| Baseline (L1) | 43 | 12.59 | 8.00 | 29.6 | 6.5 | 0.3851 | v7ulm4nt |
| Charb eps=0.05 | 42 | 14.17 | 8.70 | 30.7 | 7.2 | 0.4054 | d79f5gab |
| Charb eps=0.05 | 43 | 13.54 | 9.10 | 30.7 | 7.2 | 0.4030 | r619gn21 |
| Charb eps=0.1 | 42 | 13.87 | 9.40 | 30.5 | 7.2 | 0.4056 | 51vbivxc |
| Charb eps=0.1 | 43 | 14.03 | 9.60 | 30.8 | 7.2 | 0.4095 | 4ert76ho |
| Charb eps=0.2 | 42 | 14.72 | 9.60 | 30.3 | 7.5 | 0.4160 | br80gpr2 |
| Charb eps=0.2 | 43 | 14.58 | 10.10 | 32.2 | 7.8 | 0.4234 | 290ls7cp |

**2-seed means vs. in-run L1 baseline:**

| Config | p_in | p_oodc | p_tan | p_re |
|--------|------|--------|-------|------|
| Baseline (L1) | 12.73 | 7.95 | 30.00 | 6.50 |
| Charb eps=0.05 | 13.85 (+8.8%) | 8.90 (+12.0%) | 30.70 (+2.3%) | 7.20 (+10.8%) |
| Charb eps=0.1 | 13.95 (+9.6%) | 9.50 (+19.5%) | 30.65 (+2.2%) | 7.20 (+10.8%) |
| Charb eps=0.2 | 14.65 (+15.1%) | 9.85 (+23.9%) | 31.25 (+4.2%) | 7.65 (+17.7%) |

W&B group: `phase6/charbonnier-loss`.

**Results commentary:**
- **Clear dead end.** All eps values degrade all metrics. Monotonic degradation with increasing eps.
- p_oodc hit hardest (+12% to +24%); OOD robustness suffers most from gradient attenuation near zero.
- Even eps=0.05 (closest to L1) causes +8.8% p_in regression.
- Root cause: L1's constant ±1 gradient is a *feature* — it provides uniform optimization pressure on all residuals regardless of magnitude. Charbonnier's gradient attenuation near zero slows convergence on the many near-zero residuals that dominate surface MAE in a well-trained model.
- The smooth loss family (Huber #2113, Charbonnier #2116) is now exhaustively explored. L1 is the correct loss shape for this problem.
- **Decision: CLOSED.** Student suggestion of super-linear-near-zero (|x|^p, p<1) or asymmetric loss noted for potential future exploration.

---

### 2026-04-04 14:45 — PR #2115: Phase 6: Gap/Stagger Perturbation Augmentation — nezuko — MERGED

- Branch: `nezuko/gap-stagger-perturbation-aug`
- Hypothesis: Adding Gaussian noise to gap and stagger scalar features (indices 22, 23) during training forces the model to generalize across the local neighborhood of tandem configurations — domain randomization targeting the p_oodc/p_tan OOD axis.

| Run | σ | Seed | p_in | p_oodc | p_tan | p_re | W&B ID |
|-----|---|------|------|--------|-------|------|--------|
| gapstagger-s02-s42 | 0.02 | 42 | 13.166 | 7.514 | 29.645 | 6.486 | hszpxxof |
| gapstagger-s02-s73 | 0.02 | 73 | 12.907 | 7.377 | 30.781 | 6.216 | weovkf6s |
| gapstagger-s05-s42 | 0.05 | 42 | 13.232 | 7.643 | 30.162 | 6.359 | 7lpt7n7o |
| gapstagger-s05-s73 | 0.05 | 73 | 13.006 | 7.603 | 30.407 | 6.483 | k1oihv6h |
| gapstagger-s10-s42 | 0.10 | 42 | 13.316 | 7.704 | 31.113 | 6.542 | x8xxcpt0 |
| gapstagger-s10-s73 | 0.10 | 73 | 12.815 | 7.559 | 30.631 | 6.359 | vnfp0i7s |
| **σ=0.02 2-seed mean** | — | — | **13.037** | **7.446** | **30.213** | **6.351** | — |
| Pre-aft_foil_srf baseline | — | — | 13.03 | 7.83 | 30.29 | 6.45 | — |

W&B group: `phase6/gap-stagger-aug`. W&B metrics verified by automated script — all within rounding error (max diff 0.0005).

**Results commentary:**
- σ=0.02 is the clear winner: p_oodc **-4.9%** (7.446 vs 7.83), p_re -1.5%, p_tan -0.25%, p_in flat.
- Inverse σ-quality relationship confirmed: larger noise (σ=0.10) hurts p_tan (+1.9% regression). The gap/stagger signal is sensitive to feature distribution — excessive noise corrupts it.
- Individual highlight: weovkf6s (s73, σ=0.02) achieves p_oodc=7.377 and p_re=6.216, both below baseline mean.
- Experiment was run WITHOUT `--aft_foil_srf` (which merged in PR #2104 while this experiment was in flight). Comparison against pre-aft_foil_srf asinh-only baseline is fair. Augmentation is orthogonal — adds noise to 2 scalar features only — and should compound with aft_foil_srf.
- **Decision: MERGED.** p_oodc -4.9% is the largest single-experiment OOD improvement since pressure_first. Added `--aug_gap_stagger_sigma 0.02` to baseline reproduce command.

---

### 2026-04-04 14:30 — PR #2107: Phase 6: Aft-Foil Coordinate Frame Normalization (Dual-Frame v2) — frieren — SENT BACK

- Branch: `frieren/aft-foil-local-frame`
- Hypothesis: Adding local-frame aft-foil coordinates (centroid-subtracted) as 2 extra sideband features alongside global coords, preserving information while providing the model explicit aft-foil-relative geometry.

| Run | p_in | p_oodc | p_tan | p_re | val/loss | W&B |
|-----|------|--------|-------|------|----------|-----|
| Dual s42 | 12.91 | 7.93 | 31.22 | 6.64 | 0.3923 | 7e0dma73 |
| Dual s73 | 13.15 | 7.75 | 29.74 | 6.38 | 0.3858 | w8cyqceg |
| 2-seed avg | 13.03 | 7.84 | **30.48** | 6.51 | — | — |
| Baseline | 13.19 | 7.92 | 30.05 | 6.45 | — | — |

- W&B group: `phase6/aft-foil-local-frame`

**Results commentary:**
- **s73 beats ALL 4 baseline metrics**: p_in -0.3%, p_oodc -2.2%, p_tan -1.8%, p_re -1.1% — strong confirmation of the dual-frame hypothesis.
- **2-seed avg fails on p_tan** (30.48 > 30.05) due to high seed-to-seed variance on p_tan (31.22 vs 29.74 = 1.48 range).
- **VRAM spike is a bug**: 93GB (s42) and 76GB (s73) vs 38GB baseline. Adding 2 input features should not cause 2.5× VRAM increase. Likely caused by `x = x.clone()` creating persistent large copies in the autograd graph, or the feature concatenation expanding tensors unnecessarily.
- **Decision:** Sent back. Fix VRAM bug, then run 2 more seeds (44, 45) for a 4-seed average. If 4-seed avg beats targets, merge.

---

### 2026-04-04 14:15 — PR #2114: Phase 6: Gradient Centralization — Zero-Mean Gradient Updates with Lion — tanjiro — CLOSED

- Branch: `tanjiro/gradient-centralization`
- Hypothesis: Removing the DC component from gradients before the Lion sign operation would improve gradient diversity, especially benefiting tandem-foil gradients that are dominated by the single-foil distribution.

| Run | p_in | p_oodc | p_tan | p_re | val/loss | W&B |
|-----|------|--------|-------|------|----------|-----|
| grad-cent-s42 | 15.6 | 10.6 | 30.8 | 8.3 | 0.460 | waad26c0 |
| grad-cent-s73 | 14.5 | 9.3 | 29.6 | 7.2 | 0.422 | a2p1auev |
| baseline-s42 | 13.3 | 7.7 | 30.3 | 6.5 | 0.388 | tqlbfz9y |
| baseline-s73 | 12.9 | 8.1 | 30.5 | 6.6 | 0.392 | dck4ur8w |

- W&B group: `phase6/gradient-centralization`

**Results commentary:**
- All metrics regressed sharply vs baseline. p_in +12–17%, p_oodc +15–38%, p_re +9–28%. val/loss deteriorated (+14% for s42).
- Only p_tan on s73 showed marginal improvement (29.6 vs 30.5), but the val/loss degradation indicates this is noise from a destabilized training run, not signal.
- **Root cause (student analysis, confirmed):** Lion's sign(EMA) operation already discards gradient magnitude. Zeroing the gradient mean changes the effective update direction without providing the magnitude-balancing benefit GC delivers for Adam/SGD. With Lion, gradient direction diversity via zero-mean is counterproductive because sign quantization already enforces binary directions — zeroing the mean just introduces noise into those binary decisions.
- **Closes the gradient centralization direction.** Adding it to dead ends.

---

### 2026-04-04 13:30 — PR #2106: Phase 6: Fourier Feature Position Encoding — Spectral Bias Correction — askeladd — CLOSED

- Branch: `askeladd/fourier-feature-position-encoding`
- Hypothesis: Hidden-space Random Fourier Features (RFF) lift positional coordinates into a higher-frequency basis, correcting the neural network's spectral bias. Expected to help sharp pressure peaks at leading edges (p_in) and aft-foil suction peaks (p_tan).

| Config | Seed | val/loss | p_in | p_oodc | p_tan | p_re | W&B Run |
|--------|------|----------|------|--------|-------|------|---------| 
| FF=16, σ=10 | 42 | 0.3939 | 13.6 | 8.1 | 29.8 | 6.9 | i36jee3d |
| FF=16, σ=10 | 73 | 0.4002 | 14.1 | 8.2 | 31.1 | 6.6 | 28jblql6 |
| FF=32, σ=10 | 42 | 0.3921 | 12.9 | 8.1 | 30.0 | 6.8 | bkktfeu6 |
| FF=32, σ=10 | 73 | 0.3942 | 13.2 | 8.5 | 29.3 | 6.9 | snipvwml |

W&B group: `phase6/fourier-features`

**Results commentary (vs current 8-seed baseline: p_in=13.19, p_oodc=7.92, p_tan=30.05, p_re=6.45):**
- FF=32 2-seed mean: p_in=13.05 (-1.1% ✓), p_oodc=8.30 (+4.8% ✗), p_tan=29.65 (-1.3% ✓), p_re=6.85 (+6.2% ✗)
- FF=16 strictly worse across all metrics
- p_tan improvement is real (seed 73 hits 29.3, well below baseline) but p_oodc and p_re regress >5%
- Root cause: the model already has 8-frequency Fourier PE in the input (4 fixed + 4 learnable); hidden-space addition duplicates coverage without generalizing well to OOD/Re splits
- The fixed random frequencies introduce position-dependent biases that are geometry-specific rather than physics-general

**Conclusion:** Closes the hidden-space Fourier PE direction. Input-level PE is already sufficient; adding more in hidden space hurts OOD generalization.

---

### 2026-04-04 14:00 — PR #2113: Phase 6: Smooth L1 (Huber) Loss — Node-Level Gradient Emphasis — edward — CLOSED

- Branch: `edward/smooth-l1-huber-loss`
- Hypothesis: Smooth L1 (Huber) loss provides implicit node-level hard example mining: small-error nodes get proportionally weaker gradient (L2 regime: gradient = error/beta), large-error nodes get relatively stronger gradient. Targets p_tan by focusing gradient signal on high-error leading-edge / separation bubble nodes.

| Beta | Seed | p_in | p_oodc | p_tan | p_re | W&B Run |
|------|------|------|--------|-------|------|---------|
| 0.5 | 42 | 15.2 | 10.4 | 32.2 | 8.0 | 7ex1w2lm |
| 0.5 | 73 | 16.4 | 12.5 | 37.8 | 27.0 | hlllq0r8 |
| 1.0 | 42 | 17.1 | 13.3 | 38.8 | 27.5 | hee0z07t |
| 1.0 | 73 | 17.6 | 12.8 | 37.9 | 27.3 | mu2t0cm4 |
| 2.0 | 42 | 17.6 | 13.3 | 38.1 | 27.6 | snzm6mmn |
| 2.0 | 73 | 18.4 | 13.9 | 38.4 | 27.9 | jtl2ma6n |

W&B group: `phase6/smooth-l1`

**Results commentary (vs current 8-seed baseline: p_in=13.19, p_oodc=7.92, p_tan=30.05, p_re=6.45):**
- Best result (beta=0.5, seed=42): p_in=15.2 (+15%), p_oodc=10.4 (+31%), p_tan=32.2 (+7%), p_re=8.0 (+24%)
- All 6 runs dramatically regress vs baseline. Degradation increases with beta.
- p_re is worst affected (beta=1.0+: p_re ~27, which is the pre-surface_refine level from Phase 4)
- Root cause: after residual prediction + asinh compression, MOST nodes have small errors. Smooth L1 attenuates gradient for ALL of these, massively weakening the gradient signal. The constant-magnitude L1 gradient (±1 per node) is optimal for uniform convergence across all nodes in this setting.
- L2 regime (gradient = error/beta) essentially disabled the surface refinement head's learning in the small-error nodes, destroying the p_re gain completely.
- **Key insight:** L1's constant gradient magnitude is a feature, not a limitation — it ensures all nodes contribute to optimization regardless of residual size. This is critical post residual-prediction.

**Conclusion:** Node-level gradient weighting via loss shape is a dead end. L1 is optimal for this architecture/training setup. The smooth L1 mechanism is fundamentally incompatible with our residual-prediction + asinh-compression training paradigm. Also likely to affect Charbonnier loss (#2116 by alphonse) for the same reason.

---

### 2026-04-04 13:10 — PR #2112: Phase 6: Mesh-Density Weighted L1 — Upweight Fine-Mesh Regions — thorfinn — CLOSED

- Branch: `thorfinn/mesh-density-weighted-loss`
- Hypothesis: Weight each node's loss by `1 / local_mesh_spacing` (inverse mean k-NN distance) to upweight fine-mesh regions (leading/trailing edges, suction peaks) that drive surface MAE but contribute equally to unweighted L1.

| Run | p_in | p_oodc | p_tan | p_re | W&B run |
|-----|------|--------|-------|------|---------|
| density-w-s42 (clip=10) | 13.35 | 8.9 | 30.9 | 7.1 | rta5eea7 |
| density-w-s73 (clip=10) | 12.94 | 8.8 | 31.1 | 6.9 | xb0ittvc |
| density-w-clip20-s42 | 13.94 | 8.6 | 30.8 | 6.9 | kgkllnwg |
| density-w-clip20-s73 | 13.15 | 8.8 | 30.8 | 7.1 | qzmaub2r |

W&B group: `phase6/mesh-density-loss`

**Results commentary:** Clear negative result — all 4 variants regress 5–16% across all metrics vs baseline (single-model targets: p_in < 13.19, p_oodc < 7.92, p_tan < 30.05, p_re < 6.45). Three failure modes identified by student:
1. **Train/eval distribution shift:** Density-weighted training loss ≠ unweighted evaluation MAE, creating a systematic bias toward fitting leading-edge nodes at the expense of mid-chord nodes that dominate the metric.
2. **Double-upweighting with hard-node mining:** Leading-edge nodes are both high-density AND high-error → compound 2× upweighting that destabilizes the gradient landscape.
3. **Geometric ≠ physical priority:** Mesh density encodes CFD mesher decisions (geometric refinement), not directly what drives surface MAE.

Clip=10 vs clip=20 makes no consistent difference. Implementation was sound; the hypothesis itself is flawed. Closes the mesh-density-weighting direction.

---

### 2026-04-04 11:50 — PR #2104: Phase 6: Dedicated Aft-Foil Surface Refinement Head (ID=7) — fern — MERGED

- Branch: `fern/aft-foil-srf-branch`
- Hypothesis: Sharing the SRF head across all surface node types (boundary IDs 5/6/7) creates a capacity bottleneck for aft-foil (ID=7) nodes, which operate in the wake and have qualitatively different pressure distributions. A dedicated second SRF head exclusively for ID=7 nodes should improve p_tan (our weakest metric).
- W&B groups: `phase6/aft-foil-srf`, `phase6/aft-foil-srf-8seed`

**Initial 2-seed results:**

| Config | Seed | p_in | p_tan | p_oodc | p_re | val/loss | W&B run |
|--------|------|------|-------|--------|------|----------|---------|
| Baseline | 42 | 13.4 | 31.4 | 7.8 | 6.5 | 0.3922 | rgr1lvbt |
| Baseline | 43 | 13.3 | 30.9 | 7.6 | 6.6 | 0.3891 | uw32s6kz |
| aft_srf | 42 | 13.4 | 29.8 | 7.7 | 6.4 | 0.3866 | cp4ralol |
| aft_srf | 43 | 13.2 | 29.3 | 7.8 | 6.4 | 0.3859 | w0iccehn |
| aft_srf+FiLM | 42 | 12.7 | 30.9 | 13.0 | 6.5 | 0.3968 | nlas5922 |
| aft_srf+FiLM | 43 | 12.5 | 30.5 | 8.7 | 6.4 | 0.3864 | mcissyhq |
| aft_srf large (256/4L) | 42 | 12.9 | 31.4 | 7.8 | 6.5 | 0.3921 | bifexuep |
| aft_srf large (256/4L) | 43 | 12.9 | 30.7 | 8.0 | 6.5 | 0.3893 | ibn16sd7 |

**8-Seed validation (seeds 42-49, aft_srf only — sent for validation):**

| Seed | p_in | p_tan | p_oodc | p_re | val/loss | W&B run |
|------|------|-------|--------|------|----------|---------|
| 42 | 13.4 | 29.8 | 7.7 | 6.4 | 0.3866 | fctgmn1d |
| 43 | 13.2 | 29.3 | 7.8 | 6.4 | 0.3859 | rc40fpuu |
| 44 | 13.6 | 30.2 | 8.1 | 6.5 | 0.3931 | ygqo9rom |
| 45 | 12.9 | 30.5 | 8.1 | 6.5 | 0.3863 | r5uxnp4b |
| 46 | 13.4 | 30.2 | 7.8 | 6.4 | 0.3891 | yxhjfisl |
| 47 | 12.6 | 30.4 | 8.2 | 6.6 | 0.3898 | qrbprrli |
| 48 | 12.9 | 29.9 | 7.8 | 6.4 | 0.3852 | 9whdgscd |
| 49 | 13.5 | 30.1 | 7.9 | 6.4 | 0.3913 | ekdcwekr |
| **Mean ± Std** | **13.19 ± 0.33** | **30.05 ± 0.36** | **7.92 ± 0.17** | **6.45 ± 0.07** | — | |

**vs single-model baseline (p_tan=30.29, p_oodc=7.83, p_in=13.03, p_re=6.45):**
- p_tan: **-0.8%** ✅ — our primary target beats baseline
- p_oodc: +1.2% — within 0.5σ noise
- p_in: +1.2% — within 0.5σ noise
- p_re: 0%

**Analysis and conclusions:** Hypothesis validated. The dedicated aft-foil SRF head provides a real, consistent improvement on p_tan (-0.8% at 8 seeds). The mechanism is physically motivated: the aft foil operates in the wake with qualitatively different pressure distributions (gap/stagger sensitivity), and a dedicated correction MLP learns richer aft-foil adjustments that the shared head cannot. Zero-init output layer ensures no regression risk.

FiLM conditioning on gap/stagger was ruled out due to catastrophic p_oodc regression (+41.6% in seed 42) — conditioning creates a direct path from geometry features that overfits and generalizes poorly to OOD conditions. Larger capacity (256/4L) also provides no benefit — the bottleneck is not parameter count but dedicated routing.

Implementation note: Student identified aft-foil nodes at runtime using the signed angle field (SAF) proxy (`saf_norm > 0.005`) since boundary_id is not in processed features — verified 100% recall, 0% false positives.

**Merged** as new baseline with `--aft_foil_srf`. New single-model target: p_tan=30.05.

### 2026-04-04 11:30 — PR #2111: Phase 6: TTA via AoA Perturbation — alphonse — CLOSED (marginal at matching epochs; self-defeating under timeout)
- Branch: `alphonse/tta-aoa-perturbation`
- Hypothesis: Average model predictions over 3 AoA perturbations (−δ, 0, +δ) at inference time. Physical motivation: CFD solutions are smooth in AoA; averaging cancels model-artifact error components that vary rapidly with AoA while preserving true signal.
- W&B group: `phase6/tta-aoa`

**Raw final metrics (confounded by epoch count):**

| Run | W&B ID | Seed | Best Epoch | p_in | p_tan | p_oodc | p_re |
|-----|--------|------|-----------|------|-------|--------|------|
| baseline | m6p2jskq | 42 | 157 | 12.9 | 30.4 | 7.9 | 6.5 |
| baseline | u24b9oh9 | 43 | 157 | 12.9 | 29.8 | 7.8 | 6.4 |
| tta-d0.5 | 6biat14l | 42 | 130 | 16.3 | 30.5 | 8.8 | 7.0 |
| tta-d0.5 | j604eyv0 | 43 | 132 | 14.1 | 29.8 | 8.7 | 6.9 |
| tta-d1.0 | j7n5d3wf | 42 | 132 | 13.7 | 31.4 | 8.5 | 7.1 |
| tta-d1.0 | z4kv5ghh | 43 | 131 | 14.3 | 31.2 | 8.5 | 6.9 |
| tta-d2.0 | b6uyjpm6 | 42 | 131 | 16.2 | 30.6 | 8.8 | 7.4 |
| tta-d2.0 | n7bs7408 | 43 | 131 | 14.5 | 31.3 | 8.6 | 7.0 |

**Epoch-matched W&B comparison (epoch 130, equal training length):**

| Group | p_in | p_tan | p_oodc | p_re |
|-------|------|-------|--------|------|
| Baseline (2-seed avg) | 15.71 | 31.43 | 8.88 | 7.20 |
| TTA d=0.5 (2-seed avg) | 15.71 | 31.30 | 8.81 | 7.15 |
| TTA d=1.0 (2-seed avg) | 15.88 | 31.29 | 8.80 | 7.13 |

**Delta vs baseline at epoch 130:**
- TTA d=0.5: p_in ≈ -0.0%, p_tan -0.4%, p_oodc -0.8%, p_re -0.8%
- TTA d=1.0: p_in +1.1%, p_tan -0.4%, p_oodc -0.9%, p_re -1.0%

**Analysis:** Student correctly diagnosed the confound: TTA validation adds ~14s/epoch (+21% overhead), causing TTA runs to terminate ~25 epochs earlier than baseline. Training is verified identical at matching epochs (bit-for-bit val/loss match). W&B epoch-matched analysis confirms: at equal training length, TTA provides marginal benefit (~0.4-1.0% on p_tan/p_oodc/p_re, neutral to slightly worse on p_in) — insufficient to justify 3x inference overhead, and self-defeating because the overhead consumes ~25 training epochs worth ~18% p_in improvement. **Closed: TTA as a training-loop validation technique is self-defeating under our timeout constraint. Per-model TTA gain (~0.8%) is too small to justify cost vs training more epochs.**

### 2026-04-04 10:45 — PR #2110: Phase 6: Progressive Surface Focus Schedule — nezuko — CLOSED (negative, p_in regresses)
- Branch: `nezuko/progressive-surface-focus`
- Hypothesis: Starting with surf_weight=1.0 and ramping to dynamic target over a warm-in period lets the model build a coherent bulk-flow backbone before amplifying surface loss, improving surface MAE.
- W&B group: `phase6/progressive-surface`

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B Run |
|--------|------|------|--------|-------|------|---------|
| Baseline 8-seed mean | — | 13.03 | 7.83 | 30.29 | 6.45 | — |
| 40ep ramp | 42 | 13.334 | 7.927 | 30.574 | 6.487 | 7n8v8nlh |
| 40ep ramp | 73 | 12.912 | 7.917 | 29.349 | 6.638 | ui8t0t7g |
| 80ep ramp | 42 | 13.396 | 7.690 | 29.716 | 6.291 | mqu6c2vd |
| 80ep ramp | 73 | 12.854 | 8.093 | 29.864 | 6.347 | vkrnehh2 |
| **40ep mean** | — | **13.123** | **7.922** | **29.962** | **6.563** | — |
| **80ep mean** | — | **13.125** | **7.892** | **29.790** | **6.319** | — |

**Analysis:** Both ramp variants regress p_in (+0.7%) and p_oodc (+0.8-1.2%) relative to baseline. The 80ep ramp shows mild improvement on p_tan (-1.7%) and p_re (-2.0%), but these gains are within seed variance and come at the cost of the primary metric p_in. The existing dynamic surf_weight scheduling from epoch 0 is already well-tuned — delaying surface focus hurts in-distribution performance without compensating gains. The hypothesis that the model needs a "backbone-first" training phase is not supported: the EMA + cosine schedule already provides sufficient stability for early training. **Closed as dead end.** Student noted tandem transfer benefit from gradual ramp is theoretically interesting but not actionable without solving the p_in regression.

### 2026-04-04 10:30 — PR #2109: Phase 6: Contrastive Tandem-Single Regularization — tanjiro — CLOSED (negative, hypothesis falsified)
- Branch: `tanjiro/contrastive-tandem-regularization`
- Hypothesis: Tandem and single-foil samples share the same hidden representations in the Transolver. Adding a contrastive loss to push apart mean hidden states (cosine similarity) should force distinct internal representations and improve p_tan.
- W&B group: `phase6/contrastive-tandem`

| Config | Seed | p_in | p_oodc | p_tan | p_re | cos_sim | W&B Run |
|--------|------|------|--------|-------|------|---------|---------|
| Baseline | 42 | 13.3 | 7.7 | 30.3 | 6.5 | — | tqlbfz9y |
| Baseline | 73 | 12.9 | 8.1 | 30.5 | 6.6 | — | dck4ur8w |
| w=0.01 | 42 | 13.3 | 7.8 | 30.3 | 6.4 | 0.74 | ftrzalka |
| w=0.01 | 73 | 13.1 | 7.8 | 30.0 | 6.4 | 0.60 | fceuhys3 |
| w=0.05 | 42 | 12.5 | 7.7 | 30.6 | 6.5 | 0.62 | pfm9qsaj |
| w=0.05 | 73 | 13.3 | 8.2 | 30.3 | 6.6 | 0.30 | iyi2npzd |
| w=0.1 | 42 | 12.9 | 7.7 | 31.0 | 6.3 | 0.70 | opnlewzj |
| w=0.1 | 73 | 12.7 | 7.9 | 30.3 | 6.6 | 0.21 | atize8g5 |

**Analysis:** Contrastive mean p_tan=30.4 vs baseline mean 30.4 — no improvement. Higher weights made p_tan worse (w=0.1 avg: 30.65). Cosine similarity was successfully reduced (0.21-0.74 from ~0.9+) but didn't translate to better predictions. **Hypothesis falsified: representational entanglement between tandem and single-foil is NOT the p_tan bottleneck.** The model already distinguishes regimes internally; forcing stronger separation doesn't help because the difficulty is in the tandem physics itself (wake interactions, gap/stagger sensitivity), not in regime confusion. Centroid-based cosine loss is also a weak formulation (1 scalar gradient per batch). Interesting but inconsistent p_in improvement at w=0.05-s42 (12.5 vs 13.3) suggests mild regularization benefit at low weights.

### 2026-04-04 10:30 — PR #2107: Phase 6: Aft-Foil Coordinate Frame Normalization — frieren — SENT BACK (promising direction, needs non-destructive implementation)
- Branch: `frieren/aft-foil-local-frame`
- Hypothesis: Subtracting the aft-foil centroid from its coordinates before embedding gives the model a position-invariant view of the aft foil, decoupling shape from global position (gap/stagger).
- W&B group: `phase6/aft-foil-local-frame`

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B Run |
|--------|------|------|--------|-------|------|---------|
| 8-seed mean | — | 13.03 | 7.83 | 30.29 | 6.45 | — |
| Local frame | 42 | 13.83 | 8.20 | 30.17 | 6.59 | 00lod6uk |
| Local frame | 73 | 13.55 | 8.16 | 29.51 | 6.62 | 3llpj5yj |

**Analysis:** p_tan improved as hypothesized — s73 hit 29.51 (-2.6% vs mean), confirming the tandem OOD has a coordinate-frame representation component. However, p_in regressed severely (13.55-13.83 vs 12.2-12.71 refs) because in-place coordinate modification destroys positional information needed for single-foil predictions. **Sent back with instructions to add local-frame coords as ADDITIONAL sideband features (non-destructive), preserving global coords.** This should retain the p_tan benefit while avoiding p_in regression.

### 2026-04-04 10:00 — PR #2108: Phase 6: Asymmetric Fixed Asinh Scales (Pos/Neg Pressure) — edward — CLOSED (negative)
- Branch: `edward/asymmetric-asinh-scales`
- Hypothesis: Separate fixed (non-learnable) asinh scales for positive vs negative pressure should improve over the symmetric s=0.75, since suction peaks (-5 to -10 Cp) need stronger compression than stagnation pressures.
- W&B group: `phase6/asymmetric-asinh`

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B Run |
|--------|------|------|--------|-------|------|---------|
| Baseline (symmetric 0.75) | mean | 13.03 | 7.83 | 30.29 | 6.45 | — |
| Strong (neg=1.5, pos=0.5) | 42 | 13.6 | 8.3 | 30.8 | 6.5 | 7uz2d0ol |
| Strong (neg=1.5, pos=0.5) | 73 | 13.2 | 8.2 | 30.6 | 6.7 | 1v86awoi |
| Moderate (neg=1.0, pos=0.75) | 42 | 13.4 | 7.9 | 30.6 | 6.5 | dj0elhj2 |
| Moderate (neg=1.0, pos=0.75) | 73 | 13.1 | 8.1 | 30.2 | 6.5 | rgs2r8jj |

**Analysis:** Both asymmetric variants worse than symmetric baseline on ALL metrics. Strong asymmetry: p_oodc +5.4%, p_in +2.8%. Moderate asymmetry: p_oodc +2.2%, p_in +1.7%. The symmetric asinh(p×0.75) is already well-tuned — breaking sign symmetry introduces distributional mismatch at the zero-crossing boundary. The z-score normalization applied on top of asinh already handles scale differences between suction and stagnation regions. **Combined with learnable scale (#2096) and asinh velocity (#2098), this closes the asinh transform direction entirely.**

Code note: Student refactored 14 inline asinh transform sites into 2 helper functions (`_asinh_fwd`, `_asinh_inv`) — good code quality improvement but not mergeable without a metric win.

### 2026-04-04 07:50 — PR #2103: Phase 6: Iterative Weight-Tied Transolver — thorfinn — CLOSED (negative, key insight)
- Branch: `thorfinn/iterative-weight-tied-transolver`
- Hypothesis: Weight-tied iterations (K passes through a single shared block) give effective depth without more parameters — inspired by DEQ (Bai et al., NeurIPS 2019).
- W&B group: `phase6/iterative-transolver`

| Config | p_in | p_oodc | p_tan | p_re | Epochs | W&B Runs |
|--------|------|--------|-------|------|--------|----------|
| Baseline (3 distinct) | 12.9 | 7.8 | 30.1 | 6.5 | 156-157 | onmaryjt, xhpmloz0 |
| n_iter=2 (shared) | 13.7 (+6%) | 8.0 (+2%) | 31.1 (+3%) | 6.6 (+2%) | 156 | i8xsa2g0, 9ny3ctv2 |
| n_iter=3 | 15.4 (+19%) | 8.9 (+14%) | 31.4 (+4%) | 7.1 (+9%) | 127-128 | 55m69afg, i6ucjvki |
| n_iter=4 | 18.5 (+43%) | 10.5 (+34%) | 31.9 (+6%) | 8.0 (+23%) | 107 | i048aror, 1dxezufa |

**Analysis:** Monotonic degradation with iteration count. **Key insight from n_iter=2:** Even at identical computational depth and training epochs, weight sharing hurts 2-6%. The two intermediate blocks learn *complementary* (not repetitive) representations — the "iterative solver" analogy from DEQ doesn't hold here. Combined with #2100 (scale-up): the 3-block architecture is the right structural choice — distinct blocks matter, but more blocks don't help. **Confirmed dead end: weight-tied iteration is wrong for Transolver.**

### 2026-04-04 07:25 — PR #2102: Phase 6: Sin Activation in SRF Head — alphonse — CLOSED (dead end)
- Branch: `alphonse/sin-activation-srf-head`
- Hypothesis: SIREN (sinusoidal activation) in the surface refinement head should capture oscillatory Cp distributions better than GELU.
- W&B group: `siren-surf-head`

| Config | p_in | p_oodc | p_tan | p_re | W&B Runs |
|--------|------|--------|-------|------|----------|
| Baseline (2-seed) | 13.2 | 7.95 | 30.1 | 6.4 | jijyca9m, 6vf7ts82 |
| SIREN w=1.0 | 13.1 | 7.9 | 30.25 | 6.45 | tl6loy2k, 5x24h9o2 |
| SIREN w=10.0 | 13.3 | 7.95 | 31.25 | 6.35 | h9dsjz11, meqkfczf |
| SIREN w=30.0 | 13.05 | 8.2 | 32.75 | 6.5 | hz86txm2, smw15svc |

**Analysis:** Monotonic degradation with omega — higher SIREN frequency → worse p_tan (30.25 → 31.25 → 32.75). Even w=1.0 (mildest) shows no improvement anywhere. The srf_head's residual corrections don't have the oscillatory structure that SIREN is optimized for — GELU is well-calibrated for this task. **Confirmed dead end: sinusoidal activations don't help surface refinement.**

### 2026-04-04 07:00 — PR #2097: Phase 6: Deep Supervision — nezuko — CLOSED (mixed: p_in improved, p_tan regressed)
- Branch: `nezuko/deep-supervision`
- Hypothesis: Auxiliary loss on pre-final-block hidden features provides direct gradient flow to intermediate blocks, improving representation quality and OOD generalization.
- W&B group: `phase6/deep-supervision-8seed`

**8-seed validation (aux_w=0.2, seeds 42-49):**

| Seed | p_in | p_oodc | p_tan | p_re | W&B Run |
|------|------|--------|-------|------|---------|
| 42 | 12.3 | 7.9 | 31.8 | 6.4 | k1v8bvum |
| 43 | 12.8 | 7.7 | 30.2 | 6.4 | 11yf2z5i |
| 44 | 12.9 | 8.1 | 31.1 | 6.5 | 2qwd1irz |
| 45 | 12.5 | 7.8 | 31.9 | 6.6 | x97wrsdz |
| 46 | 12.8 | 7.7 | 30.7 | 6.3 | e1l5eokg |
| 47 | 13.2 | 8.3 | 31.9 | 6.6 | xpfnr80l |
| 48 | 13.1 | 7.7 | 30.4 | 6.7 | ds46hwpq |
| 49 | 12.9 | 8.1 | 30.8 | 6.4 | ocxi1i4c |

| Metric | Deep Sup (8-seed) | Baseline (8-seed) | Delta |
|--------|------------------|-------------------|-------|
| p_in | **12.81 ± 0.29** | 13.03 ± 0.39 | **-1.7%** |
| p_oodc | 7.91 ± 0.23 | **7.83 ± 0.19** | +1.1% |
| p_tan | 31.10 ± 0.69 | **30.29 ± 0.47** | **+2.7%** |
| p_re | 6.49 ± 0.14 | **6.45 ± 0.05** | +0.6% |

**Analysis:** Initial 2-seed result (p_oodc -3.1%) was optimistic — 8-seed shows p_oodc +1.1%. **p_in genuinely improves -1.7%** (the auxiliary gradient signal helps in-distribution feature quality). But **p_tan regresses +2.7%** — the auxiliary loss constrains representational flexibility needed for tandem geometry transfer. Since p_tan is our primary target, this tradeoff is unacceptable. Demonstrates the tension between in-distribution fitting and OOD generalization in this architecture. **Closed: p_tan regression overrides p_in improvement.**

### 2026-04-04 06:50 — PR #2101: Phase 6: OHEM — tanjiro — CLOSED (negative)
- Branch: `tanjiro/ohem-hard-example-mining`
- Hypothesis: Per-sample EMA loss tracking + upweighting of persistently hard samples. Should improve p_tan by focusing gradient budget on difficult tandem configs.
- W&B group: `phase6/ohem`

| Config | p_in | p_oodc | p_tan | p_re | W&B Run |
|--------|------|--------|-------|------|---------|
| Baseline s42 | 13.3 | 7.7 | 30.3 | 6.5 | tqlbfz9y |
| Baseline s73 | 12.9 | 8.1 | 30.5 | 6.6 | dck4ur8w |
| OHEM w=1.5 p75 s42 | 14.3 | 7.9 | 30.3 | 6.5 | 4x372o82 |
| OHEM w=1.5 p75 s66 | 13.4 | 8.0 | 30.7 | 6.6 | c8v1a5l0 |
| OHEM w=1.5 p75 s73 | 13.3 | 8.0 | 30.9 | 6.6 | 6db353lp |
| OHEM w=2.0 p75 s42 | 13.4 | 7.9 | 30.9 | 6.5 | nzdg07dy |
| OHEM w=2.0 p75 s73 | 13.5 | 8.2 | 31.3 | 6.5 | 0k97472w |
| OHEM w=1.5 p90 s42 | 14.4 | 7.9 | 30.4 | 6.4 | dng4d3lt |

**Analysis:** OHEM provides no improvement — all configs are at or below baseline. Stronger weight (w=2.0) actively hurts: p_tan +2.3%, p_in +3.1%. Root cause: OHEM compounds with existing tandem_ramp and adaptive_boost mechanisms (3 layers of reweighting is too much). The EMA correctly tracks hard samples (2.7× loss ratio) but existing mechanisms already handle this. **Confirmed: sample-level reweighting is not the p_tan bottleneck.**

### 2026-04-04 06:20 — PR #2100: Phase 6: Model Scale-Up — frieren — CLOSED (negative, key insight)
- Branch: `frieren/model-scale-up`
- Hypothesis: 3L/96s Transolver uses only 38/96GB VRAM. Deeper (5L) or wider (160s) models may improve metrics, especially p_tan which could be capacity-limited.
- W&B group: `phase6/model-scale-up`

| Config | p_in | p_oodc | p_tan | p_re | Epochs | VRAM | W&B Run |
|--------|------|--------|-------|------|--------|------|---------|
| 3L/96s s42 (BL) | 13.0 | 7.6 | 31.1 | 6.4 | 157 | 38GB | u00xzh8z |
| 3L/96s s73 (BL) | 12.9 | 7.9 | 30.3 | 6.4 | 157 | 38GB | 3y9kazhf |
| 5L/96s s42 | 19.0 | 10.4 | 31.5 | 8.6 | 105 | 55GB | z8xu1h7q |
| 5L/96s s73 | 16.3 | 10.0 | 30.4 | 8.2 | 106 | 55GB | u1zdhi63 |
| 3L/160s s42 | 13.7 | 7.9 | 30.8 | 6.5 | 141 | 41GB | mkxyc0mf |
| 3L/160s s73 | 13.0 | 8.3 | 30.2 | 6.5 | 142 | 41GB | fzabfqmz |
| 4L/128s s42 | 15.9 | 9.4 | 32.0 | 7.5 | 119 | 48GB | i4o4g2qm |
| 4L/128s s73 | 15.6 | 9.7 | 31.5 | 7.8 | 118 | 48GB | 910al9e9 |

**Analysis:** No configuration beats baseline. 5L/96s is catastrophically worse (+36% p_in) though undertrained (105/157 epochs). 3L/160s (wider) is closest but still +3% p_in after 142 epochs. **Critical finding: p_tan is remarkably similar across ALL configs (~30-32).** This proves the tandem transfer problem is NOT capacity-limited — it's a representation/distribution problem. Model scale-up is not the answer. The 3L/96s architecture is the right size for this task at the current training budget.

### 2026-04-04 06:20 — PR #2094: Phase 6: SWAD — edward — CLOSED (catastrophic failure)
- Branch: `edward/swad-averaging`
- Hypothesis: SWAD (Cha et al., NeurIPS 2021) averages checkpoints densely from a flat loss region. Already implemented in train.py but untested. Bug fix applied: SWAD now also averages the refine_head.
- W&B group: `phase6/swad-averaging`

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B Run |
|--------|------|------|--------|-------|------|---------|
| Baseline | 42 | 13.2 | 8.0 | 29.7 | 6.4 | 3mim1mhi |
| Baseline | 43 | 12.8 | 8.2 | 30.3 | 6.4 | gk16mcse |
| SWAD | 42 | 49.4 | 37.2 | 57.2 | 26.9 | 1ee50z25 |
| SWAD | 43 | 54.2 | 41.2 | 64.9 | 29.4 | hbm6rfcg |
| SWAD | 44 | 46.5 | 38.9 | 55.4 | 27.9 | hsvhokae |
| SWAD | 45 | 43.9 | 35.0 | 55.3 | 24.6 | r632qi5f |
| SWAD | 66 | 45.8 | 36.3 | 58.3 | 26.3 | 86sd67n7 |
| SWAD | 67 | 47.9 | 39.6 | 58.9 | 31.5 | tm0513wp |

**Analysis:** Catastrophic failure: SWAD is +268% p_in, +369% p_oodc vs baseline. Root cause: SWAD suppresses EMA updates (line 1805) but requires intermediate val evaluations to trigger checkpoint collection. Our eval-at-end training loop means SWAD never activates → model trains without EMA → raw weights are dramatically worse. **Key insight: EMA is load-bearing. Any technique that disables EMA without equivalent smoothing will fail catastrophically.**

### 2026-04-04 06:00 — PR #2099: Phase 6: Stochastic Depth (DropPath) — askeladd — CLOSED (dead end)
- Branch: `askeladd/stochastic-depth-droppath`
- Hypothesis: DropPath randomly skips entire residual branches during training, creating an implicit ensemble of subnetworks. Forces more geometry-agnostic representations that might improve OOD generalization, especially p_tan.
- W&B group: `phase6/stochastic-depth`

| Config | Seed | p_in | p_oodc | p_tan | p_re | val/loss | W&B Run |
|--------|------|------|--------|-------|------|----------|---------|
| DropPath 0.1 | 42 | 13.32 | 8.19 | 30.74 | 6.63 | 0.3965 | s93o0li2 |
| DropPath 0.1 | 73 | 13.65 | 8.86 | 31.25 | 6.65 | 0.4037 | fovffeeg |
| DropPath 0.2 | 42 | 13.57 | 8.37 | 31.50 | 6.93 | 0.4110 | 6dt5ofgn |
| DropPath 0.2 | 73 | 13.03 | 8.81 | 30.93 | 6.82 | 0.4033 | lqq102pu |

**Baseline (8-seed mean):** p_in=13.03, p_oodc=7.83, p_tan=30.29, p_re=6.45

| Config | p_in | p_oodc | p_tan | p_re |
|--------|------|--------|-------|------|
| Baseline (8-seed mean) | 13.03 | 7.83 | 30.29 | 6.45 |
| DropPath 0.1 (2-seed mean) | 13.50 (+3.6%) | 8.55 (+9.2%) | 31.00 (+2.3%) | 6.65 (+3.1%) |
| DropPath 0.2 (2-seed mean) | 13.30 (+2.1%) | 8.60 (+9.8%) | 31.20 (+3.0%) | 6.85 (+6.2%) |

**Analysis:** All DropPath runs are significantly worse than baseline. p_oodc is the worst-hit metric (+9-10%). Metrics verified against W&B — student's reported numbers accurate to within rounding. Root cause: Transolver has only 3 blocks — far too shallow for stochastic depth to provide meaningful path diversity. DropPath works in 12-24 layer ViTs; with 3 layers and a linear schedule, only the last block gets meaningful drop probability, simply removing compute the model needs. **Confirmed dead end: stochastic depth is not suitable for this 3-layer architecture. Revisit only if model grows to 6+ blocks.**

### 2026-04-04 05:00 — PR #2090: Phase 6: Knowledge Distillation — fern — CLOSED (clean negative result)
- Branch: `fern/knowledge-distillation`
- Hypothesis: Train a single model using the 8-seed ensemble's predictions as soft targets (offline KD). Distilled model should approach ensemble quality at 1/8th inference cost.
- W&B group: `phase6/knowledge-distill`

**v2 results (full training ~157 epochs):**

| Config | Seed | p_in | p_tan | p_oodc | p_re | val/loss | W&B run |
|--------|------|------|-------|--------|------|----------|---------|
| Baseline | 42 | 13.2 | 31.4 | 7.7 | 6.3 | 0.3908 | thssa2ru |
| Baseline | 43 | 13.3 | 30.9 | 7.6 | 6.6 | 0.3891 | 4lj8ch8s |
| **Baseline avg** | | **13.2** | **31.2** | **7.7** | **6.5** | **0.3900** | |
| Distill α=0.6 | 42 | 14.4 | 31.6 | 8.0 | 6.6 | 0.3978 | qq24kqqv |
| Distill α=0.6 | 43 | 14.3 | 32.2 | 8.3 | 6.8 | 0.4010 | o2ir8mc6 |
| **α=0.6 avg** | | **14.4** | **31.9** | **8.2** | **6.7** | **0.3994** | |
| Distill α=0.7 | 42 | 14.1 | 30.8 | 7.7 | 6.5 | 0.3917 | fjqz35fa |
| Distill α=0.7 | 43 | 13.6 | 30.3 | 8.0 | 6.5 | 0.3906 | m5smb3w6 |
| **α=0.7 avg** | | **13.9** | **30.6** | **7.9** | **6.5** | **0.3912** | |
| Distill α=0.8 | 42 | 13.2 | 30.5 | 7.8 | 6.5 | 0.3889 | whoha5za |
| Distill α=0.8 | 43 | 13.2 | 30.4 | 7.8 | 6.6 | 0.3908 | j2pc23lk |
| **α=0.8 avg** | | **13.2** | **30.5** | **7.8** | **6.6** | **0.3899** | |

**Analysis:** Knowledge distillation provides no meaningful improvement at convergence. α=0.8 (best) is statistically tied with baseline (val/loss 0.3899 vs 0.3900). α=0.7 shows marginal p_tan improvement (-2%) but p_in regression (+5%). α=0.6 is uniformly worse (+2-9%). Root causes: (1) ensemble's advantage over single models is too small (~1 Pa on p_oodc) for soft target signal to reliably capture; (2) at convergence the student already fits ground truth well — soft targets add noise rather than signal; (3) loss competition between hard and soft targets degrades metrics where ensemble has no edge (p_in, velocity). **Confirmed dead end: offline regression KD from seed-diverse ensembles does not help when base model is well-tuned.**

### 2026-04-04 04:15 — PR #2096: Phase 6: Learnable Asinh Scale — thorfinn — CLOSED (dead end)
- Branch: `thorfinn/learnable-asinh-scale`
- Hypothesis: Making asinh_scale a learnable nn.Parameter (init 0.75) allows gradient descent to find optimal compression jointly with model weights. Extension: asymmetric with separate pos/neg scales.
- W&B group: `phase6/learnable-asinh`

| Config | Seed | p_in | p_oodc | p_tan | p_re | val/loss | Final Scale | W&B Run ID |
|--------|------|------|--------|-------|------|----------|-------------|------------|
| Baseline (fixed 0.75) | 42 | 12.8 | 7.7 | 30.1 | 6.4 | 0.386 | 0.75 | zwyyvhsb |
| Baseline (fixed 0.75) | 43 | 13.3 | 7.9 | 29.8 | 6.6 | 0.387 | 0.75 | npsbhjhg |
| Learnable (init 0.75) | 42 | 25.0 | 21.0 | 35.6 | 16.9 | 0.081 | 0.0028 | 6noqttqf |
| Learnable (init 0.75) | 43 | 24.7 | 19.5 | 34.2 | 16.4 | 0.081 | 0.0036 | 967i5akk |
| Learnable (init 0.75) | 44 | 27.1 | 23.3 | 36.7 | 19.0 | 0.082 | 0.0033 | 16jzy7gf |
| Learnable (init 0.75) | 45 | 27.5 | 22.0 | 35.6 | 18.6 | 0.081 | 0.0037 | crruke85 |
| Asymmetric (init 0.75) | 42 | 20.0 | 15.1 | 33.1 | 12.4 | 0.082 | pos=0.0063, neg=0.0047 | wbrgp3oh |
| Asymmetric (init 0.75) | 43 | 19.3 | 14.5 | 31.6 | 11.8 | 0.084 | pos=0.0097, neg=0.0069 | yylvcbff |

- **Root cause: Scale collapse.** All learnable runs: scale drops from 0.75 → ~0.003 within 40 epochs. Model minimizes loss by compressing target space to near-zero (trivial shortcut). val/loss appears lower (0.081 vs 0.386) but is in compressed space — physical metrics 2x worse.
- Asymmetric variant slightly less catastrophic (pos scale collapses slower) but still 50-90% regression.
- **Conclusion:** Learnable target-space transforms create optimization shortcuts. Fixed s=0.75 from grid search is near-optimal. CLOSED.

### 2026-04-04 04:15 — PR #2098: Phase 6: Asinh Velocity Transform — alphonse — CLOSED (negative result)
- Branch: `alphonse/asinh-velocity`
- Hypothesis: Applying asinh compression to velocity channels (Ux, Uy) reduces dynamic range, same argument as pressure asinh. Test scales 0.3, 0.5, 1.0.
- W&B group: `phase6/asinh-velocity`

| Config | Seed | p_in | p_oodc | p_tan | p_re | val/loss | W&B Run ID |
|--------|------|------|--------|-------|------|----------|------------|
| Baseline | 42 | 13.5 | 8.1 | 30.4 | 6.4 | 0.3864 | p7iy87v0 |
| Baseline | 43 | 12.9 | 7.8 | 29.8 | 6.4 | 0.3844 | hwzd5zqm |
| asinh-0.3 | 42 | 13.6 | 9.0 | 31.2 | 6.8 | 0.4141 | 2joautz7 |
| asinh-0.3 | 43 | 13.4 | 8.4 | 31.3 | 6.6 | 0.4047 | g5r2dbxt |
| asinh-0.5 | 42 | 13.3 | 8.0 | 32.5 | 6.5 | 0.4002 | 4my1b7qi |
| asinh-0.5 | 43 | 13.9 | 7.9 | 31.0 | 6.6 | 0.4003 | vye8pe7r |
| asinh-1.0 | 42 | 12.8 | 7.6 | 32.4 | 6.4 | 0.3923 | 793xhqav |
| asinh-1.0 | 43 | 13.5 | 7.9 | 33.9 | 6.3 | 0.4045 | x22wlmnl |

- **All scales hurt p_tan** (+1.15 to +3.05). Velocity doesn't have pressure's outlier problem — distributions are well-behaved after physics normalization. Compressing velocity gradients removes discriminative signal for tandem wake interactions.
- **Conclusion:** Velocity channels should stay in linear normalization. CLOSED.

### 2026-04-04 04:15 — PR #2097: Phase 6: Deep Supervision — nezuko — SENT BACK (promising, needs 8-seed validation)
- Branch: `nezuko/deep-supervision`
- Hypothesis: Auxiliary loss on pre-final-block hidden features (model_out["hidden"]) provides direct gradient flow to intermediate blocks, improving representation quality.
- W&B group: `phase6/deep-supervision`

| Config | Seed | p_in | p_oodc | p_tan | p_re | val/loss | W&B Run ID |
|--------|------|------|--------|-------|------|----------|------------|
| Baseline | 42 | 13.2 | 8.0 | 31.0 | 6.3 | 0.2706 | x8ftaw4c |
| Baseline | 43 | 12.9 | 7.9 | 29.9 | 6.4 | 0.2666 | m19kza4z |
| aux_w=0.05 | 42 | 13.0 | 7.7 | 30.3 | 6.4 | 0.2754 | olcje0ya |
| aux_w=0.05 | 43 | 12.9 | 7.8 | 30.2 | 6.6 | 0.2615 | pg6xxiba |
| aux_w=0.1 | 42 | 13.1 | 7.8 | 30.1 | 6.5 | 0.2693 | k5wnix71 |
| aux_w=0.1 | 43 | 12.7 | 7.9 | 31.4 | 6.4 | 0.2636 | 6rpwpnho |
| aux_w=0.2 | 42 | 13.1 | 7.7 | 30.3 | 6.3 | 0.2677 | 7kxqtn0g |
| aux_w=0.2 | 43 | 12.8 | 7.7 | 30.2 | 6.4 | 0.2575 | 7mpx83k7 |

- **aux_w=0.2 best overall:** p_oodc 7.70 (-1.7% vs 8-seed BL mean), p_re 6.35 (-1.6%), p_in 12.95 (-0.6%), p_tan 30.25 (-0.1%). No regressions.
- **Mechanism:** Direct gradient flow from surface targets to pre-final-block features improves OOD generalization.
- **SENT BACK** for 8-seed validation of aux_w=0.2 (seeds 42-49). If confirmed, this is a merge.

### 2026-04-04 04:00 — PR #2093: Phase 6: 16-Seed Combined Ensemble Evaluation — MERGED (winner)

- Branch: `tanjiro/16-seed-ensemble-eval`
- Hypothesis: Combining 8-seed ensembles (42-49 + 66-73) into a 16-seed ensemble reduces variance by 1/√N ≈ 29% vs 8-seed, yielding lower surface MAE.
- W&B groups: `phase6/retrain-seeds-42-49`, `phase6/ensemble-seeds-100-106`

| Ensemble Size | p_in | p_oodc | p_tan | p_re |
|---------------|------|--------|-------|------|
| 4-seed (42-45) | 12.7 | 6.8 | 29.6 | 6.0 |
| 8-seed (42-49) | 12.3 | 6.7 | 29.2 | 5.9 |
| 12-seed | 12.2 | 6.7 | 29.1 | 5.8 |
| **16-seed (all)** | **12.1** | **6.6** | **29.1** | **5.8** |
| *Prior 8-ens baseline* | *12.2* | *6.7* | *29.1* | *5.8* |

Seeds 100-106 individually (W&B group: `phase6/ensemble-seeds-100-106`):

| Seed | Run ID | p_in | p_oodc | p_tan | p_re |
|------|--------|------|--------|-------|------|
| 100 | 9o85duyc | 13.4 | 7.9 | 29.8 | 6.3 |
| 101 | ec7plfg8 | 12.8 | 7.8 | 30.3 | 6.5 |
| 102 | zagg4pfs | 12.4 | 7.9 | 30.6 | 6.4 |
| 103 | 6w86plz1 | 12.5 | 7.7 | 31.3 | 6.4 |
| 104 | g00kxdva | 13.4 | 7.9 | 30.2 | 6.4 |
| 105 | jt9hwf40 | 13.9 | 7.9 | 31.4 | 6.6 |
| 106 | fom4bzro | 12.8 | 7.9 | 31.6 | 6.5 |
| **Mean** | | **13.0** | **7.9** | **30.7** | **6.4** |

Run IDs (seeds 42-49 re-trained): f59v5aul, 0yurebjv, rdezx8es, ds12ug79, yu1x0dy0, y147zvh1, lc5cbt4l, 7cxu38oh

**Analysis:** 16-seed ensemble beats the 8-seed baseline on p_in (-0.8%) and p_oodc (-1.5%). Monotonic improvement 4→8→12→16 seeds confirms 1/√N scaling law with no "degrading" individual models. Diminishing returns are clear: 8→16 gains much less than 4→8. Seeds 100-106 show consistent individual performance with prior seed batches (mean p_in=13.0 vs pool mean ~13.0), all hitting the 180-min wall clock. The 16-seed ensemble is now the new baseline. An additional 23-seed evaluation (adding 100-106) would yield another marginal improvement but inference cost scales accordingly.

**MERGED** — New baseline: p_in=12.1, p_oodc=6.6, p_tan=29.1, p_re=5.8

### Completed (2026-04-04 ~03:00 UTC)

#### 2026-04-04 03:00 — PR #2086: Phase 6: SAM Phase-Only — frieren — CLOSED (dead end)
- Branch: `frieren/sam-phase-only`
- Hypothesis: Sharpness-Aware Minimization in final 25% of training (epoch 120+) would find flatter minima, reducing seed variance and improving OOD generalization
- W&B group: `phase6/sam-phase-only`

| Config | Seed | p_in | p_oodc | p_tan | p_re | val/loss | Best Epoch |
|--------|------|------|--------|-------|------|----------|------------|
| Baseline | 42 | 13.0 | 7.6 | 31.1 | 6.4 | 0.3915 | 157 |
| Baseline | 43 | 13.2 | 8.0 | 30.0 | 6.5 | 0.3878 | 157 |
| SAM rho=0.05 | 42 | 15.4 | 8.9 | 32.4 | 7.4 | 0.4283 | 120† |
| SAM rho=0.05 | 43 | 15.2 | 10.0 | 32.0 | 8.0 | 0.4284 | 120† |
| SAM rho=0.1 | 42 | 15.4 | 8.9 | 32.4 | 7.4 | 0.4283 | 120† |
| SAM rho=0.1 | 43 | 15.2 | 10.0 | 32.0 | 8.0 | 0.4284 | 120† |
| SAM rho=0.2 | 42 | 15.4 | 8.9 | 32.4 | 7.4 | 0.4283 | 120† |
| SAM rho=0.2 | 43 | 15.2 | 10.0 | 32.0 | 8.0 | 0.4284 | 120† |

†Best checkpoint is epoch 120 = last epoch BEFORE SAM activation. All same-seed SAM runs have identical metrics because SAM was never active in the best checkpoint.

- W&B run IDs: yd0lluxz, hsq5fn98, lihil5xz, uo6chve2, b634pzxq, b4bsxpv9, xewiwyx2, 75qvkprv
- Conclusion: **SAM is a clear dead end.** SAM destabilizes late-stage training with Lion optimizer. The best checkpoint for every SAM run was the pre-SAM epoch 120, meaning SAM activation actively degraded performance. Lion's sign-based updates already provide implicit regularization; SAM is redundant/conflicting. Three rho values (0.05, 0.1, 0.2) all failed identically.
- CLOSED

### New Assignments (2026-04-04 ~00:00 UTC)

#### 2026-04-04 — PR #2097: Phase 6: Multi-Scale Deep Supervision — nezuko — NEW
- Branch: `nezuko/deep-supervision`
- Hypothesis: Auxiliary loss on intermediate features (fx_deep) forces better representations in earlier blocks
- aux_loss_weight sweep: 0.05, 0.1, 0.2

#### 2026-04-04 — PR #2096: Phase 6: Learnable Asinh Scale — thorfinn — NEW
- Branch: `thorfinn/learnable-asinh-scale`
- Hypothesis: Learnable nn.Parameter asinh_scale (init 0.75) adapts compression per run
- Includes asymmetric variant (separate pos/neg scales)

### Assignments (2026-04-03 ~23:20 UTC)

#### 2026-04-04 — PR #2099: Phase 6: Stochastic Depth (DropPath) — askeladd — NEW
- Branch: `askeladd/stochastic-depth-droppath`
- Hypothesis: DropPath randomly skips Transolver block residuals during training, creating implicit ensemble of subnetworks. Proven OOD regularizer in ViT/DeiT. Target: p_tan improvement.
- Experiment: drop_path_rate={0.1, 0.2} × seeds {42, 73} = 4 runs

#### 2026-04-04 — PR #2095: Phase 6: SGDR Warm Restarts — askeladd — CLOSED (dead end)
- Branch: `askeladd/sgdr-warm-restarts`
- Hypothesis: Cosine annealing with warm restarts (T_0={20,40,60}, T_mult=2) for better OOD generalization
- Results: **ALL 8 RUNS WORSE THAN BASELINE.** SGDR warm restarts hurt OOD generalization.

| Config | p_in seed42 | p_oodc seed42 | p_in seed73 | p_oodc seed73 |
|--------|------------|---------------|------------|---------------|
| Baseline (cosine) | 12.71 | 8.13 | 13.33 | 7.59 |
| T_0=20 | 13.62 | 8.32 | 14.02 | 8.26 |
| T_0=40 | 13.89 | 8.47 | 13.34 | 8.35 |
| T_0=60 | 13.51 | 8.19 | 13.61 | 8.30 |

- Conclusion: LR restarts disrupt the stable descent into the flat basin that Lion+cosine finds naturally. SGDR is a confirmed dead end for this architecture.
- CLOSED

#### 2026-04-03 — PR #2094: Phase 6: SWAD Dense Weight Averaging — edward — NEW
- Branch: `edward/swad-averaging`
- Hypothesis: SWAD (NeurIPS 2021) averages checkpoints from validation-stable window for flatter minima
- Status: WIP — just assigned (edward idle after #2087 closed)
- Note: Requires refine_head bug fix (SWAD doesn't average surface refine head)

### Completed (2026-04-03 ~23:00 UTC)

#### 2026-04-03 — PR #2089: Phase 6: Ensemble Weight Optimization — askeladd — CLOSED (null result)
- Branch: `askeladd/ensemble-weight-opt`
- W&B group: phase6/ensemble-weight-opt (6 runs finished: seeds 90-95)
- Results: **WEIGHT OPTIMIZATION NULL RESULT.** SLSQP converged to equal weights (~0.1 each).
  - 10-model equal-weight ensemble: p_in=12.27, p_oodc=6.67, p_tan=29.08, p_re=5.83
  - Best 6-of-10 selective: p_in=12.07, p_oodc=6.76, p_tan=29.18, p_re=5.80
  - vs current baseline (8-seed, p_in=12.2): FLAT — no improvement
- Key insight: Same-architecture models don't benefit from non-uniform weighting. Need method diversity.
- Run IDs (seeds 90-95): ici6bxi1, 6chuzqal, xcsqiwdv, sxisuynb, q8m1w63d, ggn5mioe
- CLOSED — documented, seeds available for future ensemble expansion

#### 2026-04-03 — PR #2091: Phase 6: Diverse Hyperparameter Ensemble — nezuko — CLOSED (informative negative)
- Branch: `nezuko/diverse-hparam-ensemble`
- W&B group: phase6/diverse-hparam-ensemble (8 runs finished)
- Results: **Hyperparameter diversity LOSES to seed diversity.**
  - Diverse 8-model ensemble: p_in=12.3, p_oodc=7.0, p_tan=29.8, p_re=5.9
  - Same-config 8-seed baseline: p_in=12.4, p_oodc=6.7, p_tan=29.4, p_re=5.8
  - p_oodc +4.5%, p_tan +1.4%, p_re +1.7% ALL WORSE
- Key insight: Different hyperparams make systematic tradeoffs, not complementary errors. Seed diversity strictly better.
- Run IDs: 369n5uv8, i8xwj9zn, 2bht89ay, wrbmgygu, qmsobmzv, cofd0trk, 80aq4s2f, o4dh75fa
- CLOSED

#### 2026-04-03 — PR #2092: Phase 6: Ensemble Seeds 82-89 — thorfinn — CLOSED (complete)
- Branch: `thorfinn/ensemble-seeds-82-89`
- W&B group: phase6/ensemble-seeds-82-89 (8 runs finished)
- Results: Individual model metrics consistent with baseline:
  - Mean: p_in=13.05±0.29, p_oodc=7.89±0.28, p_tan=30.30±0.42, p_re=6.44±0.10
  - Notable: s86 p_oodc=7.44, s88 p_re=6.18
- Run IDs: u0eapina, fmhetijo, yp7dlkmk, 30hxo8a1, 4e74gtuc, wc8x0v49, qvn871e1, nb6poqj2
- CLOSED — seeds trained, available for combined ensemble evaluation

#### 2026-04-03 — PR #2086: Phase 6: SAM Phase-Only — frieren — SENT BACK
- 3 rounds of failures: (1) infrastructure crash at 127min, (2) code crash at 4min, (3) timeout at 30min
- Root cause: student applied SENPAI_TIMEOUT_MINUTES to train.py MAX_TIMEOUT
- Bug fixes preserved: SAM backward graph fix, configurable sam_rho, sam_start_frac
- SENT BACK — must revert MAX_TIMEOUT override and rerun at full 180 min

#### 2026-04-03 — PR #2087: Phase 6: Ensemble Seeds 74-81 — edward — CLOSED (complete)
- Branch: `edward/ensemble-seeds-74-81`
- W&B group: phase6/ensemble-seeds-74-81 (8 runs finished)
- Results: Individual model metrics consistent with baseline:
  - Mean: p_in=13.11±0.45, p_oodc=7.97±0.17, p_tan=30.52±0.53, p_re=6.50±0.11
- Run IDs: 2sre8vzp, ue8pmbbr, hgyim25m, e2obsfn1, 555102xo, 2lpzf6go, ibsrx1t8, a6e89sx4
- CLOSED — seeds trained, available for combined ensemble evaluation

### Prior Assignments (2026-04-03)

#### 2026-04-03 — PR #2092: Phase 6: Ensemble Seeds 82-89 — thorfinn — RUNNING
- Branch: `thorfinn/ensemble-seeds-82-89`
- Hypothesis: More standard 3L seeds for ensemble expansion (total pool: 42-49, 66-73, 74-81, 82-89, 90-95)
- Status: WIP — just assigned

#### 2026-04-03 — PR #2080: Phase 6: Ensemble Seeds 66-73 — tanjiro — **MERGED WINNER**
- Branch: `tanjiro/ensemble-more-seeds`
- W&B group: phase6/ensemble-more-seeds (8 runs finished)
- Results: **NEW BEST.** 8-seed ensemble (66-73) beats prior best (42-49):
  - p_in: 12.2 (was 12.4, **-1.6%**)
  - p_oodc: 6.7 (flat)
  - p_tan: 29.1 (was 29.4, **-1.0%**)
  - p_re: 5.8 (flat)
- Run IDs: j9w7d1r7, mc4jvgqj, cbbvhl62, bigqfn3k, bqhg6lq8, 5ukk7wv6, xlnhwuqc, ii1tz4vv
- **MERGED** — Updated baseline. Next: 16-seed combined evaluation.

#### 2026-04-03 — PR #2082: Phase 6: Packed Ensemble — thorfinn — CLOSED
- Branch: `thorfinn/packed-ensemble`
- W&B group: phase6/packed-ensemble (8 runs finished)
- Results: **DEAD END.** Model too small for packed ensembles.
  - M=2: marginal, ~flat vs baseline
  - M=4: p_re +6.4%, p_oodc +2.2% worse
  - M=8: p_re +7.1%, p_oodc +3.6% worse
- Root cause: n_hidden=192 split across M sub-models → insufficient per-sub-model capacity
- Conclusion: At 1.7M params, separate model training + post-hoc averaging strictly dominates. CLOSED.

#### 2026-04-03 — PR #2068: Phase 6: Asymmetric/Magnitude-Weighted Surface Loss — alphonse — SENT BACK
- Branch: `alphonse/asymmetric-loss`
- W&B group: phase6/asymmetric-loss (8 runs finished)
- Results: Mixed tradeoff. magweight α=1.0 improves p_tan -4.3% but worsens p_in +3.7%, p_oodc +3.8%.
  - α=0.5: p_tan -2.0%, other metrics +0.5-1.5% worse
  - α=1.0: p_tan -4.3%, p_in +3.7%, p_oodc +3.8% (tradeoff too steep)
  - pinball τ=0.45: p_tan -2.1%, p_in +2.0%, p_oodc +3.7% (worse)
- Conclusion: Magnitude weighting shifts gradient from OOD to tandem transfer. SENT BACK for milder α sweep (0.1-0.3).

#### 2026-04-03 — PR #2088: Phase 6: MC Dropout in Surface Refine Head — nezuko — CLOSED
- Branch: `nezuko/mc-dropout-surface-refine`
- W&B group: phase6/mc-dropout (8 runs finished)
- Results: NULL RESULT. No consistent improvement across configs.
  - p=0.05 K=16: flat vs baseline (within noise)
  - p=0.05 K=8: mixed (p_in +2.8%, p_re -2.3%)
  - p=0.10 K=8: tradeoff (p_oodc -3.1%, p_re -3.5% BUT p_tan +3.5%)
- Conclusion: MC Dropout stochasticity insufficient for meaningful variance reduction. EMA already provides implicit regularization. CLOSED.

#### 2026-04-03 — PR #2080: Phase 6: Ensemble Seeds 66-73 — tanjiro — RESULTS IN
- Branch: `tanjiro/ensemble-more-seeds`
- W&B group: phase6/ensemble-more-seeds (8 runs finished)
- Individual model metrics consistent with 8-seed baseline:
  - p_in=13.06±0.34, p_oodc=7.88±0.17, p_tan=30.29±0.63, p_re=6.42±0.09
- Pending: 16-seed ensemble evaluation (combining seeds 42-49 + 66-73)
- Status: Awaiting ensemble eval from student

#### 2026-04-03 — PR #2086: Phase 6: SAM Phase-Only — Flat Minima for OOD Generalization
- Branch: `frieren/sam-phase-only`
- Student: frieren
- Hypothesis: SAM (already in train.py) applied only in last 25% of training (epoch≥120) finds flatter minima. Sweep rho={0.05, 0.1, 0.2} since Lion's sign updates may need different scaling than Adam. Target: reduce seed variance AND improve p_oodc/p_re.
- 8-GPU: 2 baselines (s42/s43) + 3 rho values × 2 seeds = 8 runs
- Status: WIP — assigned

#### 2026-04-03 — PR #2085: Phase 6: srf4L Ensemble Seeds 58-65 (3rd Batch) — nezuko — CLOSED
- Branch: `nezuko/srf4L-ensemble-v3`
- Hypothesis: More srf4L seeds for ensemble
- CLOSED — srf4L hypothesis invalidated (p_tan +5-7% worse per #2079, #2083)

#### 2026-04-03 — PR #2083: Phase 6: srf4L Ensemble Seeds 50-57 — fern — CLOSED
- Branch: `fern/srf4L-ensemble-v2`
- Hypothesis: srf4L seeds 50-57 for ensemble expansion
- W&B group: phase6/srf4L-ensemble-v2 (8 runs finished)
- Results: **DEAD END.** Confirms srf4L regression:
  - p_in: 13.20 (+1.3%), p_oodc: 8.06 (+2.9%), p_tan: **32.43 (+7.0% WORSE)**, p_re: 6.43 (-0.3%)
- CLOSED — srf4L hypothesis invalidated

#### 2026-04-03 — PR #2082: Phase 6: Packed Ensemble — thorfinn
- Branch: `thorfinn/packed-ensemble`
- Hypothesis: M sub-models packed into one forward pass via grouped linear ops. Test M=2,4,8. From Laurent et al. 2023, AirfRANS shows 25% training speedup over deep ensembles.
- 8 GPUs: 2 baseline + 2×M=2 + 2×M=4 + 2×M=8
- Status: WIP, 8 runs running

#### 2026-04-03 — PR #2081: Phase 6: srf4L + 8-Seed Ensemble (Compound) — edward — CLOSED
- Branch: `edward/srf4L-ensemble`
- Hypothesis: srf4L ensemble should beat 3L ensemble
- CLOSED — srf4L hypothesis invalidated (p_tan +5-7% worse per #2079, #2083)

#### 2026-04-03 — PR #2080: Phase 6: Ensemble Seeds 66-73 (Building Toward 24-Model) — tanjiro
- Branch: `tanjiro/ensemble-more-seeds`
- Hypothesis: Extend ensemble to 24 models (seeds 42-73). If ensemble variance reduction scales with N, 24-model could give another step down.
- Status: WIP, 8 crashed + 8 running (relaunched)

#### 2026-04-03 — PR #2079: Phase 6: srf4L on Asinh Baseline (Multi-Seed) — askeladd — CLOSED
- Branch: `askeladd/srf4L-asinh`
- Hypothesis: Test srf4L (surface_refine_layers=4 vs 3) as a standalone improvement
- W&B group: phase6/srf4L-multiseed (8 runs finished)
- Results: **DEAD END.** srf4L consistently WORSE than 3L baseline:
  - p_in: 13.29 vs 13.30 (flat)
  - p_oodc: 7.83 vs 7.78 (+0.7%)
  - p_tan: **32.01 vs 30.33 (+5.5% WORSE)**
  - p_re: 6.53 vs 6.46 (+1.1%)
- Conclusion: 4th surface refine layer overfits, damages tandem transfer. CLOSED

#### 2026-04-03 — PR #2068: Phase 6: Asymmetric/Magnitude-Weighted Surface Loss — alphonse
- Branch: `alphonse/asymmetric-loss`
- Hypothesis: Replace symmetric L1 with magnitude-weighted L1 (overweight stagnation/suction peaks) and pinball tau=0.45 (penalize underprediction)
- 8 GPUs: 2 baseline + magnitude-weighted α=0.5 + α=1.0 + pinball τ=0.45 × 2 seeds each
- Status: WIP, 8 runs running

#### 2026-04-03 — PR #2066: Phase 6: Synthetic Data Generation via Physics-Consistent Interpolation — frieren
- Branch: `frieren/data-gen-interpolation` (CLOSED)
- Hypothesis: Interpolate between same-condition CFD samples to create synthetic training data
- Results: DEAD END. All metrics 3-10x worse than baseline. Root cause: mesh-level interpolation is physically meaningless for unstructured CFD (no spatial correspondence between node indices across different meshes)
- W&B runs: xjo7mj7s, 5dggj8av, ysg5zqr4, nandekgh, jr5qg1i1, s3lt4tjd, 5d828zol, 1n8k5m0g
- Status: CLOSED

### New Assignments (2026-04-02)

#### 2026-04-02 — PR #2008: Phase 6: PirateNets — Random Weight Factorization
- Branch: `frieren/pirate-nets`
- Student: frieren
- Hypothesis: Apply Random Weight Factorization (RWF) from PirateNets to MLP layers. W = diag(exp(s)) * V where V is fixed random, s is learned. Enables multi-scale frequency learning for better boundary layer / pressure prediction.
- 8-GPU sweep: baseline × 2, RWF uniform × 2, RWF multiscale × 2, RWF multiscale + higher LR × 2
- Status: WIP — assigned

#### 2026-04-02 — PR #2009: Phase 6: Geosolver — Geometry-Aware Feature Encoding
- Branch: `alphonse/geosolver`
- Student: alphonse
- Hypothesis: Add explicit geometric features (wall distance, curvature, normals) as inputs + geometry-conditioned attention bias. Physics-geometry coupling directly encoded.
- 8-GPU sweep: baseline × 2, geo features × 2, geo + attention × 2, geo + attn + higher surf_weight × 2
- Status: WIP — assigned

#### 2026-04-02 — PR #2010: Phase 6: HeavyBall Optimizers
- Branch: `nezuko/heavyball-optimizers`
- Student: nezuko
- Hypothesis: Sweep modern optimizers (SOAP, CauchyAdamW, Schedule-Free AdamW, PaLM-SOAP) as alternatives to Lion. Different optimizers handle CFD's extreme gradient structure differently.
- 8-GPU sweep: SOAP × 2 LR, CauchyAdamW × 2 LR, Schedule-Free × 2 LR, PaLM-SOAP, HB-AdamW
- Status: WIP — assigned

### Previously Assigned

#### 2026-04-01 23:25 — PR #2006: Phase 6: Muon Optimizer + Gram-NS
- Branch: `phase6/muon-optimizer`
- Student: thorfinn
- Hypothesis: Replace Lion with Muon (Newton-Schulz orthogonalized gradients).
- **Status: ALL 8 RUNS CRASHED** — zero metrics logged. Code bug. Advisor commented with fix guidance.

#### 2026-04-01 23:25 — PR #2007: Phase 6: XSA (Exclusive Self-Attention)
- Branch: `phase6/xsa-attention`
- Student: askeladd
- Hypothesis: Replace slice attention with XSA exclusion mechanism.
- **Status: ALL 8 RUNS CRASHED.** Catastrophically bad metrics (val/loss 3.76-10.6 vs baseline 0.383). Code regression. Advisor commented with fix guidance.

  | Run | val/loss | p_in | Status |
  |-----|----------|------|--------|
  | xsa-t0.5-s42 | 3.761 | 296.4 | crashed |
  | baseline-s42 | 3.830 | 299.6 | crashed |

---

## Phase 5 Experiments

### Completed (2026-04-02)

#### PR #2003: LR Schedule Sweep — **MERGED WINNER**
- Student: edward | `cosine_T_max=160` beats baseline on ALL metrics
- val/loss: 0.3761 (−1.8%) | p_in: 12.5 (−3.5%) | p_oodc: 8.2 (−1.3%) | p_tan: 29.8 (−0.7%) | p_re: 6.5 (−3.0%)
- W&B: `9ysz96ll` | **MERGED** — New baseline. Follow-up: multi-seed (#2012)

#### PR #2001: Learned Per-Channel Loss Weighting — **CLOSED (mixed)**
- Student: tanjiro | p_tan −1.7% but p_oodc +3.8%, p_re +4.5%
- Key insight: pressure deserves 1.6x weight (sigma=0.087 vs 0.109-0.111 velocity)
- **CLOSED** → Follow-up: hybrid learned p-weight (#2013)

#### PR #2004: Noise Schedule Sweep — **CLOSED (dead end)**
- Student: fern | all stuck at val/loss 0.66-0.68, far above 0.383
- **CLOSED** → Follow-up: PirateNet surface gate (#2014)

### Closed (Dead Ends)
- PR #1998: Multi-Exit Ensemble (frieren) — **CLOSED.** All 8 crashed, val/loss 4.0-7.5. Auxiliary losses destabilized training.
- PR #2002: EMA Decay Sweep (nezuko) — **CLOSED.** All 8 crashed, val/loss 1.1. Runs terminated too early.
- PR #2000: OOD-Focused Training (alphonse) — **CLOSED.** All 8 crashed, val/loss 4.0-11.5. Hard-mining diverged.

### Winners (Merged — Phase 5)
1. **PR #1927: Residual Prediction** — predict correction from freestream. p_oodc -4.7%, p_tan -1.9%
2. **PR #1935: Surface Refinement Head** — dedicated surface MLP. p_re -72.7%, p_tan -8.9%, val/loss -3.3%
3. **PR #2003: cosine_T_max=160** — faster LR decay. All metrics improved. val/loss -1.8%, p_in -3.5%, p_re -3.0%

### Key Non-Winners (Closed — Phase 5)
- 50+ Phase 5 experiments explored: throughput optimization, data augmentation, architecture variants, loss formulations, ensemble methods, curriculum learning, distillation
- Most incremental tuning exhausted — motivates Phase 6 shift

## 2026-04-06 04:XX — PR #2185: MAE Pretraining — Self-Supervised Geometry Encoder Initialization
- alphonse/mae-pretrain
- **Hypothesis:** Pretraining the Transolver encoder with a Masked Autoencoder (MAE) objective on mesh geometry before supervised training would learn richer geometric representations, improving OOD generalization on NACA6416 tandem geometry.
- **Results:**

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|--------|------|------|--------|-------|------|-----|
| 20ep pretrain | 42 | 20.57 | 16.21 | 39.46 | 10.63 | `8a4c3ojp` |
| 20ep pretrain | 73 | 25.08 | 18.00 | 45.32 | 11.51 | `lr8mow5f` |
| 10ep pretrain | 42 | 27.74 | 26.59 | 47.86 | 17.68 | `twf3alql` |
| 10ep pretrain | 73 | 17.35 | 15.87 | 42.56 | 9.54 | `bv9rangq` |
| **Baseline** | — | **13.05** | **7.70** | **28.60** | **6.55** | |

- **Decision: CLOSED — strong negative result (2-5x worse than baseline)**
- **Analysis:** The MAE pretraining objective (feature reconstruction via L1 loss) is fundamentally incompatible with downstream physics prediction: it biases the encoder toward spatial interpolation rather than flow physics. Combined with reduced supervised training budget (10-20 epochs consumed by pretraining) and a dataset too small (1322 samples) to benefit from self-supervised priors, this is a clean conceptual failure. The MAE loss did converge (0.50→0.36 over 20 epochs), confirming correct implementation — the idea itself is the problem, not the code. Self-supervised pretraining on the same dataset as supervised training provides no meaningful data diversity benefit.

## 2026-04-06 ~10:30 UTC — PR #2198: GradNorm Adaptive Loss Weighting for Tandem-Transfer
- thorfinn/gradnorm-adaptive-loss
- **Hypothesis:** Fixed PCGrad loss weights treat all tasks as equally learnable, but p_tan consistently dominates error. GradNorm (Chen et al., 2018) dynamically adjusts per-task weights so gradient norms grow at the same normalized rate — preventing easy tasks from crowding out hard ones. alpha=1.5 biases toward lagging tasks (p_tan). GradNorm + PCGrad address orthogonal aspects (scalar weights vs gradient directions).
- **Results:**

| Run | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|-----|------|------|--------|-------|------|-----|
| gradnorm-alpha1.5 | 42 | 12.900 | 7.586 | 28.611 | 6.371 | `3wkhlz4a` |
| gradnorm-alpha1.5 | 73 | 13.239 | 8.042 | 29.347 | 6.545 | `5aw7ac8m` |
| **2-seed avg** | — | **13.070** | **7.814** | **28.979** | **6.458** | — |
| **vs baseline** | — | **-1.0%** | **flat** | **+1.7% ✗** | **flat** | — |

Baseline (PR #2184): p_in=13.205, p_oodc=7.816, p_tan=28.502, p_re=6.453

- **Decision: CLOSED — dead end**
- **Analysis:** GradNorm actually hurt p_tan (+1.7% regression, 28.979 vs 28.502), the exact metric it was designed to improve. p_in improved (-1.0%) but the primary target regressed. Paradoxically, the algorithm designed to upweight the tandem loss made it worse — likely because GradNorm's adaptive weight updates disturbed the carefully balanced PCGrad gradient surgery dynamics. When GradNorm detects tandem gradient norms are smaller, it increases tandem weight, but this destabilizes the PCGrad conflict resolution operating at gradient-direction level. EMA smoothing (decay=0.9) was not enough to prevent oscillation between the two mechanisms. Seed 73 was particularly bad (p_tan=29.347, +3%). Gradient norm balancing and gradient direction surgery are NOT fully orthogonal in practice with EMA interaction. The lesson: PCGrad's balance is already well-calibrated; adding a second layer of adaptive weighting on top creates interference.

## 2026-04-06 ~13:00 UTC — PR #2200: Local KNN Attention: parallel local pathway in TransolverBlock — alphonse — **CLOSED** (p_tan +13-15% vs baseline)
- alphonse/local-knn-attention
- **Hypothesis:** Global slice attention treats all spatial scales uniformly. Adding a local KNN attention pathway operating on each node's k-nearest neighbors should capture fine-scale boundary layer and wake physics. The implementation used a zero-init gate for baseline-equivalent initialization. Architecture is orthogonal to existing improvements.
- **Results:**

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B | Epochs |
|--------|------|------|--------|-------|------|-----|--------|
| k=16 | 42 | 20.0 | 11.2 | 32.0 | 8.5 | `bpgzy063` | 95 |
| k=16 | 73 | 17.9 | 10.9 | 32.9 | 8.2 | `9x2f1pis` | 96 |
| **2-seed avg** | — | **18.95** | **11.05** | **32.45** | **8.35** | — | — |
| **vs baseline** | — | **+43%** | **+41%** | **+14% ✗** | **+29%** | — | — |

Baseline (PR #2184): p_in=13.205, p_oodc=7.816, p_tan=28.502, p_re=6.453

- **Decision: CLOSED — dead end**
- **Analysis:** Implementation adapted from true KNN to strided anchor attention due to O(N²) infeasibility for N=120k mesh nodes. Anchor attention: 512 evenly-spaced anchors → full anchor self-attention → propagate to all nodes via distance-weighted interpolation. This modification introduced severe problems: (1) **Throughput degradation** — only 95-96 epochs reached (vs ~155 baseline, -38%) due to per-forward-pass `(B, N, M) = (4, 120k, 512)` distance matrix computation x3 blocks; (2) **Spatial non-adaptivity** — uniform anchors mostly sample volume, missing critical surface regions where boundary layer physics occur; (3) **Coarse interpolation** — 512 anchors serving ~234 nodes each is too coarse for fine-scale surface physics; (4) **Architecture competition** — Transolver's slice attention already provides learned spatial grouping via physics slices; a second cruder spatial decomposition competes rather than complements. Student's suggestion of surface-only local attention (~1-3k surface nodes) is more tractable and physically motivated. Global attention on 120k-node meshes is computationally infeasible for standard KNN approaches.

## 2026-04-06 ~14:00 UTC — PR #2201: Multi-Scale Slice Hierarchy — edward — **CLOSED** (p_tan +1.8%, p_oodc -4.6% mixed)
- edward/multiscale-slice-hierarchy
- **Hypothesis:** Instead of uniform `slice_num=96` across all 3 TransolverBlocks, use progressively finer slice granularity: `[32, 64, 96]` (coarse-to-fine). Block 0 at 32 slices captures global flow topology; Block 2 at 96 slices resolves boundary layer detail. Hypothesis: global patterns generalize better to OOD NACA6416 geometry. Also expected 5-8% epoch speedup from fewer slices in early blocks.
- **Results:**

| Run | Seed | p_in | p_oodc | p_tan | p_re | W&B |
|-----|------|------|--------|-------|------|-----|
| multiscale [32,64,96] | 42 | 13.4985 | 7.5693 | 28.4139 | 6.3045 | `r5s4apc5` |
| multiscale [32,64,96] | 73 | 13.3499 | 7.3478 | 29.6078 | 6.5213 | `uwlgfj78` |
| **2-seed avg** | — | **13.424** | **7.459** | **29.011** | **6.413** | — |
| **vs baseline** | — | **+1.7% ✗** | **-4.6% ✓** | **+1.8% ✗** | **-0.6% ✓** | — |

Baseline (PR #2184): p_in=13.205, p_oodc=7.816, p_tan=28.502, p_re=6.453

- **Decision: CLOSED — primary target regressed**
- **Analysis:** Mixed result. OOD condition (p_oodc -4.6%) and Reynolds (p_re -0.6%) improved, suggesting the coarse-to-fine inductive bias helps some generalization. However, p_tan regressed +1.8% and p_in regressed +1.7% — both primary targets. High seed variance on p_tan (28.41 vs 29.61, Δ=1.20 vs baseline spread ~0.14) confirms instability in the tandem prediction pathway. The underlying mechanism: fewer slices in Block 0 (32 vs 96) reduces routing resolution that `gap_stagger_spatial_bias` relies on for tandem geometry awareness. Any configuration with fewer-than-96 slices in any block loses capacity — confirmed by PR #2171 (128/144 slices both regressed; 96 is already the optimum). The OOD oodc improvement is interesting but cannot be harvested without hurting p_tan. Student suggested reverse [96,64,32] and wider [48,64,96] variants, but the fundamental trade-off (early-block resolution vs capacity) is unlikely to resolve in p_tan's favor.

### 2026-04-06 ~17:30 — PR #2181: GEPS Test-Time Low-Rank Adaptation for OOD Tandem — fern — **CLOSED** (all metrics worse than baseline)
- Branch: `fern/geps-tta`
- Hypothesis: GEPS-style (NeurIPS 2024, arXiv:2410.23889) test-time low-rank adaptation of physics-attention using continuity residual (div(U)=0) as self-supervised physics signal. LoRA-style context parameters (rank=4, zero-init, 4608 total params) added to each TransolverBlock attention output. At test time, only these LoRA params are adapted (10 or 20 steps) using KNN finite-difference divergence of velocity predictions as the objective. Intended to bridge the NACA6416 OOD gap by "fine-tuning" the representation on each test sample without ground truth.
- W&B runs: otj5j4ka (seed 42, 10-step — INVALID, bug), evz9po4m (seed 73, 10-step), fdodi6m3 (seed 42, 20-step), ldf4sof7 (seed 73, 20-step)
- W&B group: `fern/geps-tta`

| Config | Seed | Epochs | p_in | p_oodc | **p_tan** | p_re |
|--------|------|--------|------|--------|-----------|------|
| Baseline (DCT) | avg | ~145 | 13.05 | 7.70 | **28.60** | 6.55 |
| 10-step TTA | s73 | 146 | 13.47 | 8.27 | 29.57 | 6.55 |
| 20-step TTA | s42 | 145 | 13.11 | 7.82 | 29.47 | 6.64 |
| 20-step TTA | s73 | 145 | 13.56 | 8.30 | 29.71 | 6.58 |
| **20-step avg** | | | **13.34** | **8.06** | **29.59** | **6.61** |

- Note: Initial 10-step s42 run (otj5j4ka, 77 epochs) invalid — TTA bug ran adaptation every validation epoch, adding ~12 min overhead per epoch, causing only 77 epochs in 180 min. Student fixed in commit 0ba0b41.
- **Analysis:** All metrics regressed vs baseline (p_tan +3.5%, p_oodc +4.7%). Three root causes: (1) Training epoch deficit: fixed runs hit 180-min wall at 145-146 epochs; (2) Noisy physics signal: KNN finite-difference div(U) values of 1695–2895 indicate the continuity residual is too noisy for clean gradient signal at this mesh resolution/subsampling (4096 nodes from 200K+); (3) Disconnected gradient path: LoRA applied in hidden attention space but signal comes from velocity outputs — long indirect gradient path. GEPS TTA direction closed. The student's suggested follow-up (TTA on pre-trained checkpoint) wouldn't resolve the fundamental signal quality issue.

### 2026-04-06 ~18:55 — PR #2197: Geometry-Adaptive Curvature Loss Weighting — tanjiro — **CLOSED** (p_tan +1.8%, all metrics regressed vs even old baseline)
- Branch: `tanjiro/curvature-loss-weight`
- Hypothesis: Weight surface loss by per-node Menger curvature magnitude: `w_i = 1 + alpha * normalize(|kappa_i|)`, where kappa is x[:,:,24] (DSDF gradient norm proxy). Higher weight at high-curvature nodes (LE, TE, suction peak) forces model to prioritize aerodynamically critical regions. Loss reformulation only — no architectural change.
- W&B runs: z94hfr0m (α=0.5, s42), i3p6sfhs (α=1.0, s42), xyvlpjc3 (α=2.0, s42), dte1drat (α=0.5, s73). Earlier failed runs: 3q6owtec (failed), 2gj144rz (finished). Student also ran unauthorized interfoil-sb experiments (17l5t3we, srn2nh08) duplicating askeladd #2195.
- W&B group: `round7/curvature-loss-weight`

| Config | Seed | p_in | p_oodc | p_tan | p_re |
|--------|------|------|--------|-------|------|
| Baseline (old, PR #2184) | 2-seed avg | 13.21 | 7.82 | **28.50** | 6.45 |
| α=0.5 | s42 | **12.8** | **7.8** | 28.5 | 6.5 |
| α=1.0 | s42 | 12.8 | 8.0 | 28.6 | 6.5 |
| α=2.0 | s42 | 13.0 | 8.1 | 28.9 | 6.5 |
| **α=0.5 (2-seed avg)** | s42+s73 | **13.45** | **7.80** | **29.00** | **6.55** |

- **Decision: CLOSED** — 2-seed average p_tan=29.00 vs 28.50 old baseline (+1.8%); p_in=13.45 vs 13.21 (+1.8%); p_re=6.55 vs 6.45 (+1.6%). Even worse vs current baseline (p_tan=28.52).
- **Analysis:** Single-seed α=0.5 looked promising (p_in improved 3%), but seed 73 consistently worse across all metrics, creating high variance. Root cause: the curvature proxy from standardized features (x[:,:,24]) is noisy. Normalization per-sample curvature (dividing by max) amplifies noise in samples with low overall curvature, creating different effective loss landscapes per seed. The conceptual approach is sound but the curvature signal is too unreliable in this form.
- **Student's follow-up suggestions:** (1) Raw curvature before standardization; (2) Softer alpha=0.25; (3) Curvature weighting on velocity channels too. Not pursued — too incremental.

### 2026-04-06 ~19:13 — PR #2211: Surface Pressure Gradient Loss (dp/ds) — alphonse — **CLOSED** (p_tan +2.6%, fails on primary metric)
- Branch: `alphonse/surface-pressure-gradient-loss`
- Hypothesis: Add auxiliary loss on consecutive surface node pressure differences (finite-difference dp/ds proxy). DCT freq loss penalizes spectral amplitude mismatch; gradient loss penalizes transitions between adjacent nodes — complementary in theory. Weight=0.05, nodes sorted by x-coordinate per foil, applied alongside DCT freq loss.
- W&B runs: cuehr0b0 (seed 42), nqkaw1cf (seed 73)
- W&B group: `round12/surface-dp-ds`

| Metric | Baseline (PR #2184, old) | Seed 42 | Seed 73 | **2-seed avg** | Δ |
|--------|--------------------------|---------|---------|----------------|---|
| p_in | 13.205 | 13.007 | 13.426 | **13.22** | ≈flat |
| p_oodc | 7.816 | 7.600 | 8.000 | **7.80** | ≈flat |
| **p_tan** | **28.502** | 28.600 | 29.900 | **29.25** | **+2.6% ✗** |
| p_re | 6.453 | 6.300 | 6.600 | **6.45** | flat |

- **Decision: CLOSED** — p_tan regresses 2.6% on 2-seed average (29.25 vs 28.50). Even worse vs current baseline (28.52).
- **Analysis:** Seed 42 looks individually close to baseline (p_tan=28.6), but seed 73 shows large p_tan regression (29.9). The finite-difference dp/ds proxy sorted by x-coordinate is fundamentally flawed for closed airfoil geometries: at the leading edge, upper and lower surface nodes overlap in x, causing the sort to conflate nodes from different surfaces and generating spurious large gradients. DCT freq loss (merged) is robust to this ordering problem because it operates in frequency space.
- **Student's follow-up suggestions:** (1) Arc-length-ordered dp/ds (fern #2210 addresses this with arc-length reweighting); (2) Lower weight dp_ds_weight=0.02; (3) Aft-foil-only dp/ds. The fundamental ordering instability needs resolution before these variations would help.

### 2026-04-06 ~21:16 — PR #2212: Analytical Cp Delta (thin-airfoil physics baseline for SRF correction) — askeladd — **CLOSED** (p_oodc +50%, p_re +34%, clear dead end)
- Branch: `askeladd/analytical-cp-delta`
- **Hypothesis:** Subtract analytical thin-airfoil Cp baseline from SRF head predictions so the head only learns the residual delta. Based on DeltaPhi principle (residual learning on physics baseline) — analogous to residual_prediction but at the surface level.
- **Implementation note:** Student discovered that DSDF features at indices 4:12 are multi-scale clamped SDF **distances** (range 0-5), NOT surface normal components as assumed in the PR instructions. Pivoted to simplified AoA-based baseline: `Cp ≈ -2α × sign(y)` (upper surface suction, lower surface pressure).
- W&B runs: e7lucein (seed 42), vuxgzmui (seed 73)
- W&B group: `askeladd/analytical-cp-delta`

| Metric | Baseline (PR #2207, current) | Seed 42 | Seed 73 | **2-seed avg** | Δ vs current |
|--------|------------------------------|---------|---------|----------------|--------------|
| p_in   | 12.490                       | 13.4    | 14.3    | **13.85**      | **+10.9% ✗** |
| p_oodc | 7.618                        | 11.9    | 10.9    | **11.40**      | **+49.7% ✗** |
| **p_tan** | **28.521**                | 27.8    | 28.8    | **28.30**      | -0.8% ✓ (marginal) |
| p_re   | 6.411                        | 8.8     | 8.4     | **8.60**       | **+34.2% ✗** |

- **Decision: CLOSED — clear dead end.** Massive regressions on p_oodc (+50%), p_re (+34%), and p_in (+11%). Only p_tan marginally improved (-0.8%), far from compensating.
- **Analysis:** The simplified `Cp ≈ -2α × sign(y)` baseline is too crude for effective residual learning: (1) Assumes symmetric airfoils — NACA profiles have camber; (2) Ignores thickness, local curvature, and separation effects; (3) For OOD conditions (different Re, AoA ranges), the baseline mismatch grows; (4) The SRF head must now correct both the baseline error AND the actual physics delta, making the task harder. The original hypothesis (using DSDF normal components) was invalidated by the data format discovery.
- **Key insight for future experiments:** DSDF features at x[:,4:12] are multi-scale clamped SDF distance values, NOT gradient/normal components. Any experiment using these as normals will fail. Proper surface normals would require computing finite-difference gradients of the SDF field.
- **Student's follow-up suggestions:** (1) Close approach (agreed); (2) Revisit with proper surface normals if available; (3) Learned per-node baseline (small MLP) — this is a different hypothesis worth considering separately.

---

### 2026-04-06 22:30 — PR #2216: GeoTransolver GALE — thorfinn — **CLOSED** (all metrics worse)
- Branch: `thorfinn/geotransolver-gale`
- Hypothesis: Add geometry-latent cross-attention in TransolverBlock — a learnable 32-dim geometry latent per-sample, updated via cross-attention from mesh features, then injected into each TransolverBlock to condition the slice-attention on global shape information. Zero-initialized output projection for safe warmup.

| Metric | Baseline (#2213) | Seed 42 (8b6u00qn) | Δ |
|--------|-----------------|-------------------|---|
| p_in | 11.979 | 15.442 | +28.9% ✗ |
| p_oodc | 7.643 | 8.861 | +15.9% ✗ |
| p_tan | 28.341 | 29.219 | +3.1% ✗ |
| p_re | 6.300 | 7.635 | +21.2% ✗ |

- W&B: Run `8b6u00qn` (finished, 126 epochs, 180.9 min). Only seed 42 run — student correctly skipped seed 73 given clear regression.
- GALE out_proj Frobenius norms: block 0=15.5, block 1=16.4, block 2=21.7 — large, meaning the geometry cross-attention actively contributed but harmfully.
- **Analysis:** The geometry latent creates a competing information pathway that the slice-attention has to reconcile, hurting not helping. The Transolver's existing slice-attention already captures geometry via spatial bias + DSDF features implicitly. Explicit geometry latent injection is redundant and noisy. +3.9 GB VRAM overhead for a negative result.
- **Conclusion:** CLOSED. The approach of injecting global geometry via cross-attention is incompatible with this architecture. Lighter-touch geometry conditioning (AdaLN, frozen encoder) suggested by student but the magnitude of regression across ALL metrics makes this family of ideas low-priority.
