# Radical Research Ideas — 2026-04-02

## Context

Current baseline: val/loss 0.3761, p_in 12.5, p_oodc 8.2, p_tan 29.8, p_re 6.5
Architecture: Transolver (3 blocks, n_hidden=192, 96 slices), Lion optimizer, cosine T_max=160, EMA, residual prediction, surface refinement MLP, ~1.7M parameters, 38GB VRAM, ~155 epochs in 180-min timeout.

Plateau status: 5+ phases of optimization with no major gains in Phase 6. All hyperparameter/optimizer/attention tweaks exhausted.

This document is a literature survey of RADICAL new directions — meaning ideas that work at a different level of abstraction than anything tried so far.

---

## What Has Already Been Tried (Do Not Repeat)

From reviewing 1640 PRs:

**Architecture replacements (all failed or never ran):**
- Full INR/neural field replacement (PR #1876): GELU variant was 41% worse (val/loss 0.571 vs 0.405). SIREN variants catastrophically unstable.
- Full Galerkin Transformer (PR #1872): val/loss 0.646 — much worse.
- GNOT (PR #1871): val/loss 0.402 — marginally worse, never cleanly confirmed.
- DeepONet with POD basis (PR #1874): never ran (implementation issue).
- Latent Neural Operator with encode-process-decode (PR #1870): val/loss 0.4362 — worse.
- Full AB-UPT (PR #1919): never ran.
- Hybrid Anchor-Decoder (PR #1906): val/loss 0.610 — much worse.
- Local k-NN graph attention prepend (PR #1928): only 5 epochs in 3h (k-NN construction OOM/slow).
- MAE pre-training (PR #1862): catastrophic — best result val/loss 3.3, 189 p_in.
- FNO-Hybrid blocks (PR #1825): val/loss 0.3994 — same as baseline.
- Mamba SSM replacement (PR #1826): val/loss 0.3994.
- Sparse MoE in TransolverBlocks (PR #1831): val/loss 0.3994.

**Physics features (partial/stalled):**
- Inviscid Cp panel method (PR #1865): implementation started but died — panel method was 3s/batch before caching. The experiment with caching was NEVER completed (students re-launched but results never posted to advisor). This idea has NEVER been properly tested.
- Surface curvature (PR #1911): merged, +modest improvement.
- Distance-to-surface continuous feature (PR #1473): merged.

**Input encoding:**
- Wavelet-Fourier PE (PR #1147): merged, major improvement.
- Learnable Fourier frequencies (PR #1051): merged.

**Optimizers:** Muon, SOAP, HeavyBall, CauchyAdamW, Schedule-Free — ALL worse than Lion.

---

## TOP 5 RADICAL IDEAS

---

### IDEA 1: LinearNO — Replace Physics-Attention with Asymmetric Linear Attention

**Confidence: HIGH. Directly applicable, same code structure, proven results.**

**What it is:**
LinearNO (arXiv:2511.06294, AAAI 2026) is a theoretical reanalysis of Transolver's physics-attention that shows: (a) the slice attention between slices *hurts* performance, (b) what actually works is the slice and deslice operations (projecting nodes to slices, computing per-slice features, projecting back). The paper replaces the softmax slice-to-slice attention with asymmetric linear projections and proves this is a Monte Carlo approximation of the integral kernel operator.

**The equation:**
```
LinearNO(H_N) = phi(Q) * (psi^T(K) * V)
```
where phi and psi are *asymmetric* softmax projections (phi normalizes along the M/slice dimension, psi normalizes along the N/node dimension). There is NO attention matrix between slices — the intermediate self-attention is dropped entirely.

**Why it should help here:**
- Our current Transolver uses slice-to-slice attention. AAAI 2026 paper shows removing this improves performance on AirfRANS by 60%+ on lift coefficient, with 40% fewer parameters and 36% fewer FLOPs.
- LinearNO outperforms Transolver on ALL 6 standard PDE benchmarks AND AirfRANS.
- 40% fewer parameters = we could scale up n_hidden with the same VRAM budget.
- On the Airfoil benchmark: LinearNO 0.0049 vs Transolver 0.0067 (27% better relative L2 error).
- Ablation confirms: removing slice attention under asymmetric projections consistently improves performance.

**Key implementation detail:**
The change is surgical — only the attention computation inside `Physics_Attention_Irregular_Mesh` changes:

1. Make Q and K projections asymmetric (separate learned weights, NOT tied):
   - `phi_Q = softmax(Linear_Q(H), dim=M)` — normalize along slice dimension
   - `psi_K = softmax(Linear_K(H), dim=N)` — normalize along node dimension
2. Drop the `to_q`, `to_k`, `to_v` projections and the slice-to-slice attention matrix
3. Output: `phi_Q @ (psi_K.T @ Linear_V(H))`
4. Keep everything else (DomainLayerNorm, SE blocks, surface refinement, MLP blocks, etc.)

**Critical gotcha:** The existing code has `in_project_x` and `in_project_fx` as separate projections — these can serve as the asymmetric K and V respectively. The Q projection (`in_project_slice`) currently projects to slice dimension. The change is to make Q normalization per-slice (dim=M) and K normalization per-node (dim=N), then multiply directly without intermediate attention.

**Suggested experiment design:**
- Implement LinearNO attention in `Physics_Attention_Irregular_Mesh` as a flag (`linear_no_attention=True`).
- Keep n_hidden=192, 96 slices, everything else fixed.
- Test with M=96 slices (same as current) and M=256 (since LinearNO is more efficient, we can afford more slices).
- Also try: with the 40% parameter saving, increase n_hidden to 256 or 288.

**Reference:** https://arxiv.org/abs/2511.06294

---

### IDEA 2: Precomputed Inviscid Cp as Input Feature (Panel Method Physics Prior)

**Confidence: VERY HIGH. The strongest physics-informed feature idea in the literature, never cleanly tested here.**

**What it is:**
Use a panel method (thin airfoil theory / Hess-Smith) to compute the inviscid pressure coefficient Cp_inviscid at every surface node for each training sample, then feed this as an additional input feature. The neural network then learns only the RESIDUAL CORRECTION from inviscid to viscous Cp — a much simpler function.

**Why this is not just a repeat of PR #1865:**
PR #1865 was assigned to alphonse who tried to compute the panel method ONLINE during training (3 seconds/batch) and then pivoted to batch precomputation but the experiment was NEVER completed — the student reported launching with caching but no results were ever submitted to the advisor. This idea has LITERALLY NEVER BEEN TESTED on our pipeline.

**Why the literature is overwhelming here:**
1. B-GNN paper (arXiv:2503.18638, March 2025): Using inviscid Cp from a panel method code as an input feature achieves **88% error reduction on OOD pressure prediction** compared to purely geometry-based inputs, and **83% model size reduction** to achieve the same in-distribution accuracy. This is the single most dramatic feature engineering result in recent airfoil ML literature.
2. "Inviscid Information-Embedded Machine Learning for Airfoil Inverse Mapping" (ScienceDirect 2025): Using Cp_inviscid as an intermediate bridge reduces prediction error by 25-40% average, with up to 500% error reduction at the leading edge.
3. NeurIPS 2024 ML4CFD competition: ALL top 4 solutions used explicit physics priors.
4. NeuralFoil (arXiv:2503.16323, 2025): Panel method-derived features are "the #1 most impactful feature engineering change" for airfoil aerodynamics ML.

**Physics intuition (critically important for our tandem problem):**
- For single foils: inviscid Cp captures ~80% of the surface pressure signal. The model only learns the viscous correction (boundary layer effects, 20% of the signal). This dramatically reduces the learning problem.
- For TANDEM foils (our worst metric, p_tan=29.8): The inviscid interference between foil 1 and foil 2 creates the large-scale pressure variation the model struggles to learn. A panel method naturally captures the multi-body interference, giving the model the right starting point for the correction.
- For OOD Reynolds (p_re=6.5): Inviscid Cp is Re-INDEPENDENT. The model only needs to learn the Re-dependent viscous correction, making it naturally more generalizable.

**Implementation (OFFLINE PRECOMPUTATION — the key fix over PR #1865):**
1. Write a preprocessing script that runs a panel method for every training/val sample ONCE and saves Cp_inviscid per surface node to disk (e.g., as an HDF5 file keyed by sample ID).
2. In the data loader, load the precomputed Cp_inviscid and append it as an extra feature to the surface node input.
3. For volume nodes, set Cp_inviscid = 0 (or use a SDF-interpolated value, but 0 is fine as a first attempt).

**Panel method options (fast, in Python):**
- **NeuralFoil** (5ms per sample, pip-installable): `from neuralfoil import get_aero_from_kulfan_parameters` — computes viscous AND inviscid Cp distributions. Already handles NACA 4-digit parameterization which matches our dataset.
- **Hess-Smith panel method**: ~50ms per sample in pure Python, gives exact potential flow Cp.
- For TANDEM configurations: use a multi-element panel method or run two separate single-element solves and superimpose (potential flow is linear, so superposition is valid for incompressible inviscid flow).

**Critical detail on feature dimension:**
The current code has X_DIM from `prepare_multi.py`. Adding Cp_inviscid as a new feature column increases X_DIM by 1 (surface nodes get the computed value, volume nodes get 0). This requires a config change (`x_dim=X_DIM+1`) but is fully compatible with existing code.

**Suggested experiment design:**
- Step 1: Write preprocessing script offline (no GPU needed). Run once, save to disk.
- Step 2: Modify DataLoader to append Cp_inviscid feature.
- Step 3: Train with identical config to current baseline (lr=5e-4, Lion, T_max=160, etc.) — only the input feature changes.
- Target: We expect the largest gain on p_tan and p_oodc, since these are the cases where inviscid interference is most important.

**Reference:** https://arxiv.org/abs/2503.18638, https://arxiv.org/abs/2503.16323

---

### IDEA 3: All-to-All Surface Attention — Dedicated Surface Node Communication Layer

**Confidence: HIGH. Directly addresses the known weakness of our model on surface metrics.**

**What it is:**
Add a dedicated attention layer that processes ONLY surface nodes with full all-to-all attention (not sliced). This is motivated by the incompressible flow constraint: because pressure satisfies an elliptic PDE (Laplace/Poisson equation), the pressure at any surface point depends on ALL other surface points globally. The current slice attention approximates this global communication via soft clustering — but clustering introduces locality bias that misses the exact global pressure coupling.

**Why this hasn't been tried before:**
PR #1878 ("Surface All-to-All Attention Layer") was assigned but the results show it either never ran properly or the implementation had issues (results show only CLA bot responses, no metric results). The idea was NEVER properly validated.

**The physics argument (from B-GNN paper):**
"The governing fluid equations are elliptic due to the incompressibility constraint, meaning every point on the boundary will affect the pressure at every other point — requiring all-to-all communication in the prediction pipeline."

The Transolver's 96-slice approximation of all-to-all attention may not capture the exact long-range pressure coupling between the leading edge of foil 1 and the trailing edge of foil 2 in tandem configurations — which is exactly where p_tan=29.8 fails.

**Implementation:**
Our mesh has ~200-300 surface nodes per sample (much smaller than the ~85K total nodes). Full O(N²) attention on just the surface nodes is trivial:
- ~200 surface nodes × 200 surface nodes = 40,000 attention weights per head — negligible.

```python
class SurfaceAllToAllAttention(nn.Module):
    """Full attention on surface nodes only. O(S^2) where S ~ 200-300."""
    def __init__(self, n_hidden, n_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(n_hidden, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(n_hidden)
    
    def forward(self, fx, surf_mask):
        # fx: [B, N, H], surf_mask: [B, N] boolean
        # Extract surface nodes, apply full attention, scatter back
        B, N, H = fx.shape
        # ... extract surface nodes, attend, put back
```

Position: Insert AFTER the last Transolver block, BEFORE the surface refinement MLP. The idea is: Transolver does the global field reconstruction, then the surface nodes do a dedicated refinement communication pass.

**Why this addresses p_tan specifically:**
In tandem configurations, the pressure distribution on foil 2 is strongly influenced by the wake of foil 1. The all-to-all surface attention allows foil 2 surface nodes to directly attend to foil 1 surface nodes, capturing the exact interference pattern. The current model can only communicate this through the shared slice tokens — an indirect path.

**Key parameters to tune:**
- Number of heads: 4-8 (8 is current model's standard).
- Whether to use standard softmax attention or linear attention (since S is small, softmax is fine here).
- Insert before or after surface refinement MLP.
- Include volume nodes near surface (k=1 nearest volume neighbor per surface node) or surface-only.

**Connection to LinearNO (Idea 1):**
These two ideas are ORTHOGONAL and COMPOSABLE. LinearNO improves the global field processing efficiency; the surface all-to-all layer adds exact local communication for the surface physics. The combination could be particularly powerful.

**Reference:** https://arxiv.org/abs/2503.18638 (B-GNN paper motivation), plus our existing experiment infrastructure.

---

### IDEA 4: Conditional Flow Matching / Rectified Flow Surrogate (Probabilistic Architecture)

**Confidence: MEDIUM-HIGH. Strong evidence from adjacent domains; untested here.**

**What it is:**
Replace the deterministic regression with a flow matching / rectified flow generative model. During training, learn a time-dependent velocity field f(x_t, t, condition) that transforms a Gaussian distribution into the target pressure/velocity field distribution. During inference, integrate the ODE from t=0 (noise) to t=1 (prediction) using ~8 steps (fast rectified flow variant).

**Why this is radical for our problem:**
- All previous experiments are DETERMINISTIC regressors: given geometry+conditions → single field prediction.
- Flow matching is a GENERATIVE model: learns the distribution of flow fields given conditions.
- For tandem configurations specifically, the interference patterns create multi-modal uncertainty in the pressure field — a single deterministic prediction averages over modes, giving high MAE. A generative model can learn to sample the correct mode.
- Recent work (arXiv:2601.03030, Jan 2026): Flow matching PointNet for steady incompressible flow past cylinders with varying geometry outperforms regression-based surrogates (CNNs, FNOs, DeepONets) and crucially provides **uncertainty quantification**.
- arXiv:2506.03111 (Rectified Flows for Multiscale Fluid Flow): Achieves diffusion-class posterior accuracy with only 8 ODE steps by learning near-straight trajectories.
- PBFM (arXiv:2506.08604, June 2025): Physics-Based Flow Matching embeds PDE residuals directly into the flow matching objective, achieving 8x improvement in physical residual accuracy.

**Key insight for our setting:**
We are predicting STEADY-STATE flow, not time evolution. The flow matching "time" t is a FICTITIOUS parameter (the diffusion time) not the physical flow time. This is exactly the setting of arXiv:2601.03030. The geometry and flow conditions serve as the conditioning signal.

**Architecture:**
- Backbone: Keep the Transolver architecture but add a time embedding (sinusoidal or learned scalar t → hidden dim) that modulates each block via FiLM conditioning.
- Input: noisy field x_t = (1-t) * x_0 + t * x_1, where x_0 ~ N(0,1) (noise) and x_1 is the target flow field.
- Output: velocity field (dx/dt) at time t.
- Inference: 8-step ODE solve from t=0 to t=1 using RK4.

**Why this addresses the tandem problem:**
The key insight is that p_tan=29.8 is likely high because the model predicts a single value that is wrong for some samples — not because it's consistently wrong. A generative model with uncertainty quantification would identify WHICH samples are uncertain and could average over samples to reduce MAE. The 8x inference cost (8 ODE steps) is manageable given we have 96GB VRAM.

**Implementation complexity:** MODERATE. The main change is:
1. Add time embedding to the Transolver.
2. Change the training loop to sample random t, construct x_t, and supervise the predicted velocity field.
3. Add inference ODE integration.

**Critical hyperparameter:** Number of ODE steps at inference (8-16 for rectified flows, 100+ for score-based diffusion — prefer rectified flows for efficiency).

**Gotcha:** Flow matching is typically used for generative tasks (images, proteins) where you want diverse samples. For a regression task, you want the MEAN of the distribution (which matches the deterministic regressor in the infinite-samples limit). The benefit comes from: (1) potentially better training dynamics (the score function formulation avoids the mode-averaging problem), (2) the ability to ensemble 8-step inference paths, and (3) uncertainty quantification.

**Reference:** https://arxiv.org/abs/2601.03030, https://arxiv.org/abs/2506.03111, https://arxiv.org/abs/2506.08604

---

### IDEA 5: MARIO-Style Latent Geometry Encoding — Two-Stage Training with SDF Autoencoder

**Confidence: MEDIUM. Strong results in aerodynamics domain; significant architectural change.**

**What it is:**
MARIO (arXiv:2505.14704, May 2025) is a neural field surrogate that won 3rd place in the NeurIPS 2024 ML4CFD competition. The key innovation: it separates geometry encoding from flow prediction into two stages:
1. Stage 1: Train an SDF autoencoder to compress the airfoil geometry (SDF field) into a compact latent code z of dimension 8.
2. Stage 2: Train a conditional neural field where the flow prediction at position x is f(γ(x), z, AoA, Re), where γ(x) is a Fourier embedding of spatial coordinates and z modulates the network via a hypernetwork.

MARIO achieves an ORDER OF MAGNITUDE improvement over GNN baselines on AirfRANS, with drag coefficient error of 0.794% vs 3.5-11.5% for baselines.

**Why this is different from the failed INR attempt (PR #1876):**
PR #1876 tried a naive coordinate-conditioned MLP (FiLM conditioning on global parameters). MARIO's key insight is that **the geometry encoding must be LEARNED SEPARATELY** from the flow prediction. The hypernetwork generates layer-wise modulation vectors specific to each geometry — not just a global conditioning vector.

Additionally, MARIO uses:
- Explicit **boundary layer mask** σ_bl (which nodes are in the near-wall region) — directly targeting the most important prediction region.
- **Fourier features** (σ=1 Gaussian, 64 features) for spatial coordinate encoding — not fixed sinusoidal.
- **Meta-learning (CAVIA-based)** for geometry encoding: optimize latent code z via 3 inner gradient steps for each new geometry.

**Why the naive INR failed and MARIO wouldn't:**
The INR in PR #1876 had no geometry-specific latent encoding — it tried to use the raw geometric input features (AoA, Re, NACA, gap, stagger) as conditioning. For tandem configurations with 2 airfoils, the geometry information is implicit in the mesh coordinates and SDF — not easily captured by 5 global parameters. MARIO explicitly encodes each geometry's SDF into a compact latent code, capturing the exact shape information.

**Implementation for our setting:**
Our dataset has NACA airfoils, so the geometry is parameterized. We can use the existing SDF (dsdf feature) as the geometry representation. The adaptation:

1. Stage 1: Small autoencoder (5-layer FC, 256 hidden, 64 Fourier features) that encodes the per-sample SDF field → 8-dim latent z.
2. Stage 2: Replace Transolver backbone with a modulated neural field:
   - Hypernetwork takes z and outputs modulation vectors for each layer.
   - Coordinate network maps (x, y, σ_bl) + Fourier(x,y) → hidden → prediction, modulated by hypernetwork.
3. Training: Joint optimization of both stages.

**Key architectural differences from naive INR:**
- Hypernetwork generates LAYER-WISE modulation (not just FiLM conditioning on the output).
- 3 inner gradient steps for geometry-specific adaptation at inference (meta-learning).
- Boundary layer mask σ_bl explicitly tells the model which nodes are near-wall.
- Fourier features with σ=1 Gaussian (64 features) for better high-frequency representation.

**Expected speedup/benefit:**
MARIO trains on 10% of mesh nodes per epoch (random subsampling) — which means for our 85K-node meshes, we process 8,500 nodes/epoch instead of 85,000. This gives ~10x more effective samples per epoch within the time budget.

**Known failure mode:**
MARIO paper notes degradation on unusual airfoil geometries and at extremes of parameter space. For our OOD Reynolds split, this could be a problem — but our inviscid Cp feature (Idea 2) could mitigate this by providing a physics prior.

**Reference:** https://arxiv.org/abs/2505.14704, GitHub: https://github.com/giovannicatalani/MARIO

---

## HONORABLE MENTIONS (Ideas 6-10)

### 6. LinearNO + Inviscid Cp Compound
Combine Idea 1 (LinearNO attention) and Idea 2 (inviscid Cp feature). These are ORTHOGONAL changes — one is architectural, one is a feature. Expected: compound gains.

### 7. All-to-All Surface Attention on Boundary Conditions Only
Instead of using the full surface node count, apply attention only to the ~50 HIGHEST-CURVATURE surface nodes (leading edge, maximum thickness, trailing edge) — the nodes where pressure variation is largest. This gives a sparse but physically motivated global communication graph with negligible overhead.

### 8. Offline Ensemble Knowledge Distillation (Properly Done)
PR #1979 (offline distillation) failed because of implementation issues. The CORRECT approach: train 8 models, compute ensemble predictions for ALL TRAINING samples offline, save to disk, then train the distillation student on the soft targets (interpolating between hard labels and ensemble soft predictions with weight α=0.5). This avoids the 8x inference overhead at training time that killed the previous attempt.

### 9. Recurrent / Iterative Transolver (Fixed-Point Iteration)
Run the Transolver backbone for K iterations, using the output of iteration k as the input to iteration k+1 (analogous to deep equilibrium networks or Gauss-Seidel iteration for linear systems). The physical intuition: incompressible flow satisfies an elliptic PDE and is naturally solved by iterative methods. With K=3-4 iterations and weight-tied blocks, this is equivalent to a much deeper model with the same parameter count — but critically uses the same VRAM budget. Physical analogy: each iteration refines the pressure field using the velocity field from the previous iteration (pressure-velocity coupling).

### 10. Pressure-Poisson Soft Constraint (Fixed Version)
Physics losses have failed before due to WLS gradient instability on unstructured meshes. The correct approach: discretize the Laplacian using the FINITE DIFFERENCE stencil for EACH NODE (using the k-nearest neighbors as the stencil), precompute the stencil coefficients OFFLINE, and apply them as a fixed sparse matrix at training time. This is computationally trivial once precomputed. The constraint: Lap(p) = -rho * div(u) * grad(u) (simplified for incompressible RANS). The soft loss penalizes violation of this relationship between predicted p and predicted velocity.

---

## Implementation Priority Ranking

| Rank | Idea | Rationale |
|------|------|-----------|
| 1 | **LinearNO (Idea 1)** | Surgical code change, AAAI 2026 paper proves it improves Transolver on AirfRANS, 40% fewer params enables width scaling |
| 2 | **Inviscid Cp Feature (Idea 2)** | Never properly tested, literature shows 88% OOD error reduction, directly targets our worst metric (p_tan) |
| 3 | **Surface All-to-All Attention (Idea 3)** | Addresses elliptic pressure physics, never properly tested, minimal overhead (200 nodes), composable with Idea 1 |
| 4 | **LinearNO + Inviscid Cp Compound (Idea 6)** | If Ideas 1+2 individually show improvement, compound them |
| 5 | **MARIO Geometry Encoding (Idea 5)** | Highest risk/reward — complete architectural change, but MARIO has strongest published results on AirfRANS |

---

## Key Unresolved Questions for Future Research

1. **Why does p_tan=29.8 remain so high?** The tandem transfer metric has barely improved despite 1600+ experiments. The inviscid Cp feature is our best hypothesis for breaking through this, as it directly encodes the foil-foil interference.

2. **Is the bottleneck data (1322 samples) or architecture?** The B-GNN paper achieves 88% OOD error reduction with 87% FEWER training samples using physics features. This suggests data scarcity is being amplified by poor features — not that we need more data.

3. **Is the surface refinement MLP enough, or do surface nodes need direct global communication?** Our model has surface refinement but no dedicated surface-to-surface attention. The all-to-all surface attention (Idea 3) tests this directly.

4. **Has LinearNO been validated at small dataset scale (1322 samples)?** The AAAI paper tested on AirfRANS (200 train samples) — even smaller than ours. This is strong evidence it works at our scale.

---

## Sources

- LinearNO (AAAI 2026): https://arxiv.org/abs/2511.06294
- B-GNN Boundary Graph NN with Panel Method Features (March 2025): https://arxiv.org/abs/2503.18638
- MARIO Neural Field Aerodynamics (May 2025): https://arxiv.org/abs/2505.14704
- Flow Matching PointNet for CFD (Jan 2026): https://arxiv.org/abs/2601.03030
- Rectified Flows for Multiscale Fluid (June 2025): https://arxiv.org/abs/2506.03111
- Physics-Based Flow Matching (June 2025): https://arxiv.org/abs/2506.08604
- NeuralFoil Panel Method Surrogate (March 2025): https://arxiv.org/abs/2503.16323
- NeurIPS 2024 ML4CFD Competition Retrospective: https://arxiv.org/abs/2506.08516
- Transolver-3 Scaling (Feb 2026): https://arxiv.org/abs/2602.04940
- GeoMPNN Surface-to-Volume Message Passing (NeurIPS 2024 competition)
- MoE Gating for External Aerodynamics (August 2025): https://arxiv.org/abs/2508.21249
