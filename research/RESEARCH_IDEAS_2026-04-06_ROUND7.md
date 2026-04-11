# SENPAI Research Ideas — Round 7
# Date: 2026-04-06 ~07:00 UTC
# Author: researcher-agent
# Baseline: p_in=13.21, p_oodc=7.82, p_tan=28.50 (PRIMARY TARGET), p_re=6.45

## Context

130 PRs merged, 1500+ experiments run. The primary target is p_tan=28.50 (NACA6416 tandem transfer, unseen geometry).
The 6 active students are covering: Stochastic Depth (#2192), Curvature Spatial Bias (#2193), SE(2) AoA Routing (#2191),
Laplacian Eigenvector PE (#2190), Vorticity Auxiliary Target (#2183), GEPS TTA LoRA (#2181).

**Hard constraint:** Do NOT manipulate feature distributions (SWD, MixStyle, raw-input TTA all catastrophically failed — 3 consecutive disasters).

**Unassigned from prior rounds (still viable):**
- Round 6: Hopfield Geometry Memory Bank, Tandem Inter-Foil Distance Feature, Geometry-Adaptive Curvature Loss Weighting
- Round 5: Learned Geometry Tokenizer, Local KNN Attention, SIREN INR Pressure Decoder

---

## IDEA 1 (HIGH PRIORITY — ASSIGN IMMEDIATELY): Tandem Inter-Foil Distance Feature

### Mechanism
Add a single scalar feature per mesh node: `log(1 + d_interfoil)` where `d_interfoil` is the minimum Euclidean distance from node `(x, y)` to any surface node of the *opposite* foil. For fore-foil nodes, this is distance to aft-foil surface; for aft-foil nodes, distance to fore-foil surface; for volume nodes, distance to the nearest of either foil.

This is a pure input enrichment — no architecture changes needed. The feature encodes aerodynamic coupling strength: aft-foil nodes in the inter-foil gap (the highest-error region in p_tan) have small d_interfoil and strong coupling to fore-foil wake; nodes far away have large d_interfoil and decouple. The spatial_bias MLP already takes `(x, y, sdf, tandem_flag, gap, stagger)` — simply add this 7th channel.

### Why this addresses p_tan
The NACA6416 tandem transfer error arises because the model hasn't seen this foil shape in training. The inter-foil distance encodes *topology of coupling* rather than foil shape — it tells the model how strongly each node is influenced by the opposite foil, regardless of what that foil looks like. This is geometry-agnostic coupling information, generalizable across unseen geometries.

### Implementation
In `prepare_multi.py` data loading (or augmentation hook in `train.py`), compute:
```python
# For each sample: foil_1_xy = surface nodes with surface_id in {5,6} (approx)
#                  foil_2_xy = surface nodes with surface_id in {7} (approx)
# For each volume node i with position (xi, yi):
#   d_to_foil1 = min(sqrt((xi - x1j)^2 + (yi - y1j)^2)) for j in foil_1_nodes
#   d_to_foil2 = min(sqrt((xi - x2j)^2 + (yi - y2j)^2)) for j in foil_2_nodes
#   d_interfoil[i] = min(d_to_foil1, d_to_foil2)  -- distance to opposite foil
#   feature[i] = log1p(d_interfoil[i])
```
Due to mesh sizes (~3k nodes), brute-force pairwise CPU is feasible (or use scipy.spatial.cKDTree). Normalize by gap distance for physical consistency.

Add as a new feature channel appended to the existing 24-dim DSDF feature vector (making it 25-dim), adjusted in `train.py`'s input dimension handling.

### Risk
Low-moderate. If spatial_bias already implicitly encodes gap/stagger such that the model infers coupling, this will be redundant (null result). The key question is whether explicit per-node coupling distance adds signal beyond the scalar gap/stagger values.

### Suggested command addition
`--interfoil_dist_feature` flag — compute and append log(1+d_interfoil) to input features.

---

## IDEA 2 (HIGH PRIORITY — ASSIGN IMMEDIATELY): Geometry-Adaptive Curvature Loss Weighting

### Mechanism
Upweight the surface loss at high-curvature nodes (leading edge, trailing edge) by a factor proportional to local arc-length curvature `κ_i`. The loss becomes:

```
L_surface = mean_i( w_i * |pred_i - true_i| )
w_i = 1 + α * normalize(κ_i)   # κ_i = Menger arc-length curvature at surface node i
```

This is the *spatial analog* of the merged DCT frequency-weighted loss (PR #2184, p_tan -0.3%), but operating in physical space rather than Fourier space. High curvature = high frequency = high aerodynamic sensitivity. The current L1 loss treats TE and mid-chord surface nodes identically despite the LE/TE being responsible for >80% of pressure errors.

### Why this addresses p_tan
For NACA6416 in tandem: the TE/LE curvature distributions differ from NACA0012/2412/4412. The model currently sees the LE/TE as just another surface node — same weight as a mid-chord node — despite these being the regions where pressure prediction fails most. By concentrating loss signal at high-κ nodes, we force the SRF head to specialize on the geometrically-critical locations regardless of foil shape.

### Implementation
Curvature already computed in PR #2193 (edward, Curvature Spatial Bias) — if that merges first, the curvature computation code is available. Otherwise:

```python
# In train.py surface loss computation:
# x_surf = surface node positions (coords available in batch['x'])
# Compute curvature: for consecutive triple (p_{i-1}, p_i, p_{i+1}) on surface
# κ_i = 2*|AB × AC| / (|AB| * |BC| * |AC|)  (Menger curvature)
# Apply weight: w_i = 1 + curvature_alpha * (κ_i / κ_max)
# curvature_alpha is a hyperparameter (try 0.5, 1.0, 2.0)
```

Key implementation detail: curvature is computed per-sample per-epoch (geometry varies by sample). Cache on the batch or compute once at dataset load time.

Surface nodes ordering: requires that surface nodes are ordered consecutively along the chord. Verify this from `prepare_multi.py` before implementing.

### Suggested flags
`--curvature_loss_weight --curvature_alpha 1.0`

Try alpha in {0.5, 1.0, 2.0}. Use `--wandb_group curvature-loss-alpha` if iterating.

### Note on relation to edward's #2193
Edward's PR uses curvature to condition the *spatial_bias routing* (which slice a node routes to). This idea uses curvature to weight the *loss*. Orthogonal mechanisms — if both work, they stack.

---

## IDEA 3: Transolver++ Eidetic States — Persistent Memory Tokens

### Mechanism
Add a small set of "eidetic state" persistent vectors to each TransolverBlock that participate in slice attention as additional memory tokens. During forward pass, K=8 learned vectors `M ∈ R^{K×d}` are concatenated with the slice tokens before the self-attention:

```python
# In Physics_Attention_Irregular_Mesh forward:
# Standard: slice_tokens = einsum('bnd,knd->bkd', x, slice_weights)  [B, S, d]
# + eidetic: tokens = cat([slice_tokens, M.expand(B, K, d)], dim=1)  [B, S+K, d]
#            tokens = self_attention(tokens)                           [B, S+K, d]
#            slice_tokens = tokens[:, :S, :]                          [B, S, d]  (discard M-rows)
```

The M vectors are global persistent parameters (learned, not input-dependent). They provide a "scratchpad" — the model can route information through persistent memory across layers, preventing the pathological collapse where all slice tokens converge to an average physical state. Paper claims 13% mean improvement over 6 physics benchmarks.

### Why this addresses p_tan
The core failure mode for NACA6416 tandem: attention slices may collapse to an average over training geometries, erasing the distinctive features of an unseen foil shape. Eidetic states give the model persistent storage that doesn't collapse — the M vectors can specialize as "tandem-interference mode" or "foil-wake-coupling mode" stable across all inputs.

### Source
Transolver++ paper (2025). "Eidetic states" = persistent fixed-size memory visible to all attention heads. Analogous to Universal Transformers' global workspace.

### Implementation notes
- K=8 is the paper recommendation for PDE problems. Try K=4 (one per layer) and K=16 (richer memory).
- Memory matrix M is `nn.Parameter` per layer (not shared across layers).
- No change to output dimension since M-rows are discarded after attention.
- torch.compile compatible — no dynamic shapes.
- 8*192*3 = 4.6K extra params (negligible). No VRAM impact.
- Critical: the M-rows must be masked out from the output aggregation. Do NOT pass M outputs back to nodes.

### Suggested flags
`--eidetic_states --eidetic_k 8`

---

## IDEA 4: GradNorm Adaptive Multi-Task Loss Weighting

### Mechanism
Replace the fixed training loss weights with GradNorm dynamic balancing. The idea: track the gradient norm of each task's loss with respect to the shared backbone parameters, then adjust task weights so that all gradient norms grow at the same rate (normalized by the initial ratio).

```python
# After each backward pass:
# G_task = ||grad_backbone||_2 for each of {in_dist, ood, tandem, re}
# W_target = G_task / mean(G_all) * mean_initial_G
# loss_weights = loss_weights * (W_target / G_task) ** alpha_gradnorm
```

Current approach: fixed PCGrad with `pcgrad_extreme_pct=0.15`. GradNorm is not a replacement for PCGrad gradient surgery — it's a *complementary* automatic weighting layer on top. The two address different problems: PCGrad reduces gradient *conflict*, GradNorm balances gradient *magnitude*.

### Why this addresses p_tan
The p_tan loss gradient is likely much smaller in absolute magnitude than p_in (tandem samples are rarer, harder, with lower-frequency errors) — so the optimizer systematically under-weights it even with equal loss coefficients. GradNorm automatically compensates for this imbalance, giving p_tan proportionally more weight whenever it falls behind.

### Source
GradNorm (Chen et al., ICML 2018). Widely used in multi-task learning; Kaggle winners regularly apply this for cross-domain tasks.

### Implementation notes
- alpha_gradnorm = 1.5 is the paper recommendation (higher = more aggressive equalization). Try 1.0, 1.5, 2.0.
- Gradient computation for GradNorm only touches the last shared layer (e.g., the final TransolverBlock output norm) — do NOT compute full-network grad norms (2x backward, expensive).
- Can be combined with PCGrad: PCGrad happens at the per-gradient conflict level, GradNorm at the loss weighting level.
- Requires retaining the task-separated losses at each step — this requires separate forward passes per task (or masking within a single pass, which is already done via domain splits).
- EMA-smooth the weight updates with decay 0.9 to avoid oscillation.

### Suggested flags
`--gradnorm --gradnorm_alpha 1.5`

---

## IDEA 5: Modern Hopfield Geometry Memory Bank

### Mechanism
A differentiable associative memory that, at inference, retrieves the pressure pattern from the K nearest training geometries and uses it as a conditioning signal for the aft-foil SRF head.

**Training phase:** For each training sample, compute a geometry embedding vector `g = MLP(dsdf_2_stats)` (8-dim: 4 moments × 2 stats for chord/camber/thickness characterization from DSDF). Store all `(g_i, p_surface_i)` pairs in a memory matrix H.

**Inference phase:** For a test sample (e.g., NACA6416), compute `g_test`, retrieve top-K training patterns via modern Hopfield attention:
```python
# M = [g_1 | ... | g_N]  (N training embeddings, N×8)
# V = [p_1 | ... | p_N]  (N surface pressure vectors, N×n_surf)
# attn_weights = softmax(β * g_test @ M.T)  (β = temperature)
# p_retrieved = attn_weights @ V            (retrieved pressure prior)
# aft_srf_input = cat([srf_features, p_retrieved], dim=-1)
```

### Why this addresses p_tan
NACA6416 is an unseen foil. The closest training foils (NACA4412, NACA2412) have known pressure distributions. The Hopfield bank retrieves these and provides a "warm start" for the SRF head — the model doesn't have to predict from scratch, it refines a retrieved prior. This directly addresses the generalization gap without touching the backbone or feature distributions.

### Source
Modern Hopfield Networks (Ramsauer et al., ICLR 2021). Used in protein structure prediction (ProtTrans) and chemistry (CHEMBERT) to retrieve similar training examples. The key insight: exponential storage capacity (2^(n/2) patterns in n-dim space) allows exact retrieval of similar patterns even with small memory.

### Implementation notes
- Memory H is computed on the training set during epoch 0 (or cached), updated each epoch with EMA.
- The retrieval is differentiable — gradients flow back to the geometry embedding MLP.
- CRITICAL: Memory contains training *targets*, not *predictions* — otherwise you get train-set overfitting with no OOD benefit. Use ground truth p_surface in H.
- K=5 nearest neighbors in geometry space. β=0.1 (soft, not hard retrieval).
- Extra params: geometry MLP (8→8, ~100 params) + memory projection. Negligible.
- torch.compile compatible if memory is stored as registered buffer.
- Edge case: for in-distribution samples (NACA0012), the retrieval will be exact — the model may learn to rely on it. Add retrieval dropout (p=0.2 during training) to force robustness.

### Suggested flags
`--hopfield_geometry_memory --hopfield_k 5 --hopfield_beta 0.1`

---

## IDEA 6: Spectral Conditioning of Attention (SCA) — Drop-in OOD Stabilizer

### Mechanism
Replace the standard softmax attention in each TransolverBlock with Spectral Conditioning of Attention (SCA). The method adds a learnable diagonal matrix D that right-multiplies the attention weight matrix before softmax, reducing the Jacobian condition number of the attention map:

```python
# Standard:  A = softmax(QK^T / sqrt(d))
# SCA:        A = softmax((QK^T * D) / sqrt(d))
# where D = diag(sigma(d_1), ..., d_S) for S slice tokens, learned per head
```

The matrix D is `nn.Parameter` of shape `[n_heads, slice_num]`, initialized to all-ones. D acts as a per-head, per-slice attention scaling that is learned to minimize condition number during training.

### Why this addresses p_tan
OOD performance degrades when attention maps become ill-conditioned — a few slices dominate, others collapse to near-zero. For NACA6416, the model routes to familiar in-distribution slices and misses the tandem-coupling-relevant slices. SCA systematically prevents this collapse by reducing condition number as a training objective.

### Source
NeurIPS 2025: "Spectral Conditioning of Attention for Improved OOD Generalization in Neural Operators" (title reconstructed from search results). The paper reports consistent OOD improvement across 8 physics benchmarks with no in-distribution regression.

### Implementation notes
- This is a drop-in — D is initialized to ones, so the model starts as standard attention. Training gradually learns the optimal D.
- The condition number regularization term: `L_cond = lambda * log(cond(A_mean))` where A_mean is the mean attention weight matrix across the batch. lambda=0.01.
- Alternatively (simpler version): just add the D matrix with no explicit condition number loss and let the optimizer learn it freely. Try this simpler variant first.
- torch.compile compatible.
- No VRAM overhead. ~O(n_heads * slice_num) = 3 * 96 = 288 extra params.

### Suggested flags
`--spectral_attn_conditioning --sac_lambda 0.01`

---

## IDEA 7: Local KNN Attention Augmentation

### Mechanism
Add a local attention pathway alongside the existing global slice attention. For each surface node, compute attention over its K=16 nearest mesh neighbors (by Euclidean distance). Combine local and global outputs:

```python
# Global path: existing slice attention → g_global  [N, d]
# Local path:  for each node i, attend over knn_i = K nearest nodes
#   local_q = W_q @ x_i, local_k = W_k @ x_knn_i, local_v = W_v @ x_knn_i
#   g_local_i = softmax(local_q @ local_k.T / sqrt(d)) @ local_v
# Combined: x_out = x_in + gate * g_global + (1-gate) * g_local
# gate = sigmoid(W_gate @ x_in)  (learned per-node gating)
```

The local attention is only computed for *surface* nodes (the high-value targets), not volume nodes, to keep compute tractable. For volume nodes, fallback to global-only.

### Why this addresses p_tan
The current slice attention is global — each node attends to all other nodes through slice routing. For pressure prediction at the TE of NACA6416, the most physically relevant information is in the immediate vicinity (neighboring surface nodes, nearby wake nodes). The global routing may miss this local structure. Adding explicit local KNN attention for surface nodes provides a direct local gradient path.

### Source
Local+global attention used in ViT-Neighborhood (2024), weather forecasting (Pangu-Weather uses both), molecular dynamics (DimeNet++). The pattern of combining local graph attention with global attention is the key contribution of GraphFormer.

### Implementation notes
- Precompute KNN graph once at data load time (K=16 per node, stored as edge_index).
- Local attention is only applied at surface nodes (tagged in the batch).
- Gating network W_gate: 192→1, scalar per node. Use sigmoid gate (not hard switch).
- torch.compile: use `torch.ops.aten.index_select` for KNN gather, avoid dynamic shapes.
- VRAM: KNN attention for ~150 surface nodes × 16 neighbors × 3 layers ≈ 3K attention entries — negligible.
- Risk: if local KNN neighbors are predominantly same-foil nodes, this may hurt tandem coupling (reduces cross-foil attention). Test on validation before full training.

### Suggested flags
`--local_knn_attn --knn_k 16 --knn_surface_only`

---

## IDEA 8: SIREN INR Pressure Decoder — Bold Architectural Swing

### Mechanism
Replace the SRF MLP (which takes a fixed-size feature vector and outputs pressure) with a SIREN (Sinusoidal Representation Network) Implicit Neural Representation. The SIREN takes coordinates `(x, y, sdf)` as input (not node features) and outputs pressure at any continuous location:

```python
# Standard SRF: pressure = MLP(node_features)  [feature-space]
# SIREN:        pressure(x,y,s) = SIREN_θ(x, y, sdf)  [physical-space INR]
# where SIREN_θ is a SIREN conditioned on global backbone features via FiLM:
#   h_0 = sin(W_0 @ [x, y, sdf] + b_0)
#   for each layer l: h_l = sin(W_l @ h_{l-1} + b_l + gamma_l * z + beta_l)
#   where (gamma_l, beta_l) = CondMLP(backbone_output)  [FiLM conditioning]
```

The backbone still processes all mesh nodes via slice attention. The SIREN replaces the final per-node MLP for surface pressure only, reading coordinates as "query" positions rather than feature vectors.

### Why this addresses p_tan
The NACA6416 has different chord length and camber than training foils. When the SRF takes `(x, y)` as coordinates (via SIREN) rather than processing DSDF features, it learns a coordinate-space pressure map that is shape-agnostic. The FiLM conditioning from the backbone encodes what the foil *is* globally; the SIREN maps where pressure *is* locally. This decoupling of global conditioning from local spatial mapping may generalize better to unseen foil shapes.

### Source
SIREN (Sitzmann et al., NeurIPS 2020). Applied to fluid simulation: "Implicit Neural Representations for Fluid Simulation" (Wandel et al., 2021). FiLM conditioning: Perez et al. (2018). The combination is used in NeRF++, NeRF-based fluid simulation, and protein structure (AlphaFold2's IPA module uses coordinate-conditioned MLP with FiLM-style conditioning).

### Implementation notes
- SIREN initialization is critical: W_0 should use U(-1/d, 1/d) * omega_0 with omega_0=30, W_l uses U(-sqrt(6/n), sqrt(6/n)) / omega_0. Wrong init kills training.
- FiLM conditioning: backbone outputs a 192-dim vector, CondMLP maps it to 2*layers*d = 2*3*64 = 384 (gamma+beta pairs). 
- Replace ONLY the aft-foil SRF (the component most responsible for p_tan). Keep forward SRF as standard MLP.
- SIREN depth: 3 layers, width 64. Activation: `sin`. This is a `<10K param` module.
- Coordinate normalization: normalize (x, y) to [-1, 1] by chord length before SIREN input.
- torch.compile: SIREN uses only sin/matmul — fully compatible.
- Risk: SIREN training can be finicky. The FiLM conditioning must be strong enough to break degeneracy (SIREN is underdetermined without strong conditioning). May need omega_0 tuning.

### Suggested flags
`--siren_aft_srf --siren_omega0 30 --siren_hidden 64 --siren_layers 3`

---

## Priority Ranking for Immediate Assignment

### For tanjiro (assign now):
**IDEA 2 — Geometry-Adaptive Curvature Loss Weighting**

Rationale: Pure loss function change, no architecture risk. The DCT freq loss (p_tan -0.3%) proved spectral weighting helps; curvature loss is the spatial equivalent targeting LE/TE physically. If edward's #2193 merges first, curvature computation is already available. Low risk, clear mechanism, 30-40 line implementation.

### For askeladd (assign now):
**IDEA 1 — Tandem Inter-Foil Distance Feature**

Rationale: Direct input enrichment, no distribution manipulation (adding a new feature, not rescaling existing ones). The gap/stagger scalars already help route slices — per-node coupling distance extends this to the full field. The physics motivation is clear: inter-foil gap nodes experience the strongest fore-aft coupling, and this feature encodes that explicitly. Low complexity.

---

## Notes on Ideas NOT to prioritize immediately (covered by active PRs or higher-risk):

- **Eidetic States (Idea 3)**: Wait for stochastic depth (#2192) to resolve first — both modify TransolverBlock internals, should not be assigned simultaneously. Assign after #2192 closes.
- **GradNorm (Idea 4)**: Wait for the 6 active PRs to land — GradNorm's effectiveness depends on which tasks are currently under-optimized, which changes as each PR merges.
- **Hopfield Memory (Idea 5)**: Higher implementation complexity. Best assigned to a student with strong Python skills after the simpler ideas are exhausted.
- **SCA (Idea 6)**: Needs paper access to verify exact implementation. Reasonable next-round candidate.
- **Local KNN (Idea 7)**: Round 5 carry-over. Moderate risk (KNN edge cases). Good candidate if Laplacian PE (#2190) fails.
- **SIREN (Idea 8)**: Bold swing. Assign when in a plateau — it's a significant rearchitecting of the SRF. The SIREN init is fussy; if the student doesn't get it right it will catastrophically fail. Reserve for a student who's demonstrated careful implementation.
