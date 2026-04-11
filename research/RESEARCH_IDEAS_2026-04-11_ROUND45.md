<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Round 45 Research Ideas — Bold, Paradigm-Shifting Directions
**Date:** 2026-04-11
**Advisor:** noam
**Baseline:** p_in=11.872, p_oodc=7.459, p_tan=26.319, p_re=6.229

---

## Context and Motivation

Recent rounds (40-44) have been too incremental — auxiliary heads, gradient surgery variants, attention tweaks. The current best architecture (Transolver + SRF + AftSRF + PCGrad + physics features) is well-optimized locally. Further progress requires working at a **different level of abstraction**.

The following 12 ideas are drawn from a systematic review of 1995 PRs (141 merged) plus targeted literature searches across: neural operators for CFD (2024-2026), physics-informed data augmentation, synthetic data generation, GNNs for fluid dynamics, and physics-informed ML for aerodynamics. Each idea represents a **paradigm shift**, not a hyperparameter sweep.

**Priority tiers:**
- **P0** (assign immediately): Ideas with direct literature evidence targeting our hardest metrics (p_tan, p_re)
- **P1** (assign next round): Strong theoretical basis, moderate implementation risk
- **P2** (reserve): High upside but higher implementation risk or uncertainty

---

## P0 IDEAS — Assign Immediately

---

### Idea 1: Inviscid-to-Viscous Residual Learning (DeltaPhi)
**Priority:** P0 — Assign to: **alphonse**
**Student slot:** After PirateNet SRF finishes

**Title:** DeltaPhi Residual Prediction: Learn the Gap Between Panel Theory and CFD

**Hypothesis:**
The network currently predicts absolute flow fields from geometry + conditions. But we already compute panel-method Cp (inviscid physics) as an input feature. The key insight from DeltaPhi (NeurIPS 2025, Zhejiang Univ) is: instead of predicting the full field, predict only the *residual* between the inviscid (analytically cheap) solution and the true viscous CFD solution. This residual is much smaller in magnitude and smoother — easier to learn, especially in data-limited regimes. For tandem foils, the dominant viscous effect is wake-induced separation, which the panel method completely misses; making that residual the target focuses the model's capacity exactly where CFD adds value over potential flow theory.

**Boldness Justification:**
This is a fundamental change to what the network is predicting. It's not a loss weight or architecture tweak — it's changing the prediction target itself. DeltaPhi showed this approach is particularly effective when training data is limited (exactly our tandem wake distribution) and when a physics prior exists (we have panel-method Cp already computed as a feature).

**Implementation Sketch:**
1. The panel-method Cp feature is already computed and available as input. Extend `prepare_multi.py`-style logic to also produce a panel-method Ux/Uy estimate (thin airfoil theory or existing panel code).
2. In `train.py`: compute the inviscid baseline field `y_inviscid` for each sample at data-load time (or precompute and store).
3. Change network output targets from `[Ux, Uy, p]` to `[dUx, dUy, dp]` = `[Ux_cfd - Ux_inv, Uy_cfd - Uy_inv, p_cfd - p_inv]`.
4. At inference: final prediction = inviscid baseline + network residual.
5. The inviscid baseline for pressure is already panel-method Cp (scaled by 0.5*rho*V^2), so `dp = p_cfd - p_inv` is the target for the pressure head.
6. Keep all other components (SRF, AftSRF, PCGrad) unchanged — they now learn residuals instead of absolute values.
7. Loss: same L1/Huber, but now on residuals. Asinh transform on dp may still help but likely needs scale re-tuning (dp is smaller magnitude).

**Expected Impact:** p_tan reduction of 10-20% — wake residuals are the dominant challenge, and the inviscid solution already captures the attached-flow regions.

**Literature Reference:** DeltaPhi (NeurIPS 2025, Zhejiang University): "Physics-Residual Learning for Data-Efficient PDE Surrogate Modeling." Shows 30-40% error reduction in data-limited flow regimes when predicting residuals from a physics prior.

---

### Idea 2: Physics-Informed Data Augmentation via Chord-Scaling (AdaField-PIDA)
**Priority:** P0 — Assign to: **edward**
**Student slot:** After Hard Kutta TE v2 finishes

**Title:** PIDA: Physics-Consistent Chord-Scaling Augmentation for OOD Generalization

**Hypothesis:**
Our OOD metrics (p_oodc, p_re) are significantly better than p_tan but still lag in-distribution. AdaField (arxiv 2601.07139, Jan 2026) introduced Physics-Informed Data Augmentation (PIDA): rather than augmenting geometry (FFD — which failed due to label mismatch), augment the *flow conditions* using known scaling laws. Specifically, potential flow theory says that if you scale chord length by factor `s`, pressure scales as `Cp` (dimensionless) is invariant — meaning `p_new = 0.5 * rho * (V_new)^2 * Cp`. This lets us generate valid new training samples by: (1) taking an existing solution, (2) applying chord-scale or velocity-scale transform, (3) computing the correctly-transformed output field. Unlike FFD, there is no label mismatch because we derive the new labels analytically from the scaling.

**Boldness Justification:**
FFD geometry augmentation failed due to label mismatch (PR #2275 area). PIDA sidesteps this by augmenting along physics-invariant directions where we can compute exact labels — this is entirely different from naive interpolation or noise injection. AdaField showed 15-25% improvement in cross-domain generalization on surface pressure with this approach.

**Implementation Sketch:**
1. In the training loop (or data loader in `train.py`): with probability `p_aug=0.3`, apply a PIDA transform to a batch sample.
2. **Reynolds scaling:** Given a sample at Re=Re0, V=V0, generate a new sample at Re1=Re0*(1+eps) by: adjusting V (V1 = V0 * Re1/Re0 if chord/viscosity fixed), and scaling p by (V1/V0)^2. Ux, Uy scale as V1/V0.
3. **AoA interpolation:** Between two existing samples at AoA1 and AoA2, generate an intermediate AoA via small-angle potential flow interpolation: p_interp = p1 + (AoA_interp - AoA1)/(AoA2-AoA1) * (p2 - p1) for the inviscid Cp component; keep viscous residual fixed.
4. Apply transforms on-the-fly during training, using the existing data sources as the base pool.
5. Augmented samples are only used for training, not validation.

**Expected Impact:** p_re and p_oodc improvement of 5-15%. May also help p_tan by effectively expanding the training distribution of wake configurations.

**Literature Reference:** AdaField (arxiv 2601.07139, Jan 2026): "Adaptive Field Learning with Physics-Informed Data Augmentation for Cross-Domain Aerodynamic Surrogate Modeling." PIDA section shows physics-consistent augmentation along invariant manifolds avoids label corruption.

---

### Idea 3: Ollivier-Ricci Graph Rewiring for Wake Long-Range Dependencies (PIORF)
**Priority:** P0 — Assign to: **thorfinn**
**Student slot:** After Arc-Length PE finishes

**Title:** PIORF: Ricci-Flow Mesh Rewiring to Connect Wake-Interacting Nodes Directly

**Hypothesis:**
The tandem foil problem (p_tan) is essentially a long-range dependency problem: pressure on the aft foil surface depends on the upstream foil's wake state, but in the mesh graph, these nodes are many hops apart. Existing message-passing or attention mechanisms suffer from "over-squashing" — information from the fore-foil wake gets diluted by the time it reaches aft-foil surface nodes. PIORF (ICLR 2025) uses Ollivier-Ricci curvature to identify bottleneck edges in the graph (negative curvature = information bottleneck) and rewires the graph by adding direct edges between high-curvature node pairs. Applied to our mesh, this would directly connect fore-foil trailing edge / wake nodes to aft-foil leading edge / surface nodes — exactly the physically relevant long-range connections.

**Boldness Justification:**
No existing experiment has touched graph topology. All attention mechanisms (including Transolver's slice attention) operate on fixed node sets. PIORF is a pre-processing step that changes what nodes can communicate, not how they communicate — orthogonal to all previous work.

**Implementation Sketch:**
1. Install `torch-geometric` and the `GraphRicciCurvature` library (or implement Ollivier-Ricci from scratch — it's ~50 lines for the discrete version).
2. Compute Ollivier-Ricci curvature for each edge in the mesh graph at data-load time (once per sample, cached).
3. Identify edges with curvature < threshold (e.g., -0.5) — these are bottlenecks.
4. For each bottleneck edge, add "shortcut edges" between the most curvature-negative node pairs, up to a budget (e.g., top-k=50 edges per sample).
5. In `train.py`: augment the node feature matrix with a flag `is_rewired_edge` so the model knows which connections are shortcuts.
6. The Transolver already uses attention over node sets, so the new shortcut nodes just appear as additional neighbors in the attention pool.
7. Alternative simpler version: physicistically motivated — just add edges between all aft-foil surface nodes and the 10 nearest fore-foil trailing-edge nodes. This is the "physics PIORF" without computing curvature.

**Expected Impact:** p_tan reduction of 15-30% — the primary bottleneck for tandem cases is precisely this long-range wake dependency.

**Literature Reference:** PIORF (ICLR 2025): "Physics-Informed Ollivier-Ricci Flow for Graph Neural Networks in Fluid Dynamics." Demonstrates 18% improvement on wake interaction prediction in tandem cylinder cases by solving over-squashing via curvature-guided rewiring.

---

### Idea 4: Local Boundary Layer Reynolds Number Feature (B-GNN Re_x)
**Priority:** P0 — Assign to: **askeladd**
**Student slot:** After Multi-Scale Hierarchical Attention finishes

**Title:** Local Re_x Feature: Arc-Length Reynolds Number for Boundary Layer State Estimation

**Hypothesis:**
B-GNN (arxiv 2503.18638, Mar 2025) showed that a single additional feature — local Reynolds number Re_x = (Re * x/c) where x is the arc-length from the leading edge and c is chord length — allows the model to implicitly estimate the boundary layer state (laminar/transitional/turbulent) at each surface node. This is deeply physical: the boundary layer momentum thickness and displacement thickness, which control surface pressure via viscous-inviscid interaction, are functions of Re_x. Our current features include global Re but not local Re_x. Adding Re_x gives the model direct access to the length scale at which viscous effects dominate at each point.

**Boldness Justification:**
B-GNN achieved 83% model compression and 87% training data reduction using local physics features. Re_x is the key feature they identified. We already have arc-length PE (#2389 in-flight), so we have the arc-length values available — Re_x is just (Re * s/c) where s is the node's arc-length coordinate. This is nearly free to add but physically highly informative.

**Implementation Sketch:**
1. In `train.py` or `prepare_multi.py` (read-only — so in `train.py`): compute `Re_x = Re * (arc_length / chord)` for each surface node.
2. For volume nodes, use the distance from the nearest surface point as a proxy for the local boundary layer thickness scale.
3. Add `Re_x` as an additional input feature (dimension +1, from 24 to 25 dims).
4. Also add `log(Re_x)` to capture the log-scale variation (Re_x spans ~0 to ~1e6 across a chord).
5. Normalize `log(Re_x)` to zero mean unit variance over the training set.
6. May synergize strongly with arc-length PE (#2389) — combine them if that PR merges before this one starts.

**Expected Impact:** p_re improvement of 10-20% (Reynolds OOD is exactly the regime where local Re_x matters), moderate p_tan improvement.

**Literature Reference:** B-GNN (arxiv 2503.18638, Mar 2025): "Boundary-Only Graph Neural Networks for Aerodynamic Surrogate Modeling." Local Re_x feature identified as key contributor to BL state estimation, enabling 83% model size reduction.

---

## P1 IDEAS — Assign Next Round

---

### Idea 5: Denoising Diffusion Probabilistic Model Surface Head (DDPM-SRF)
**Priority:** P1 — Assign to: **fern**
**Student slot:** After Bernoulli constraint finishes

**Title:** Diffusion-Based Surface Pressure Head: Stochastic Generative Decoding for p_tan

**Hypothesis:**
The tandem wake interaction creates multimodal pressure distributions — a given aft-foil configuration can have qualitatively different pressure profiles depending on whether it's in the fore-foil's wake separation bubble or attached flow region. A deterministic MLP decoder (SRF/AftSRF) predicts the conditional mean, which may be a blurred compromise between these modes. A denoising diffusion model (DDPM) conditions on the Transolver latent and iteratively denoises a surface pressure signal — it can represent multimodal distributions by sampling different "modes" during inference. At test time, use the mean of K=10 samples as the prediction. Qiang Liu & Nils Thuerey (Thuerey group, 2024) showed DDPM surrogates outperform deterministic models on multimodal airfoil flows.

**Boldness Justification:**
No previous experiment has used generative models as a decoder. All decoders have been deterministic (MLP, Mamba SSM). A diffusion head is a fundamentally different class of model — it generates rather than regresses. The ensemble mean of diffusion samples provides uncertainty quantification as a bonus.

**Implementation Sketch:**
1. Replace the AftSRF MLP head with a small DDPM (T=50 denoising steps, U-Net backbone operating over the 1D arc-length sequence of aft surface nodes).
2. Conditioning: the Transolver latent representations of aft surface nodes are used as the conditioning signal (cross-attention to the U-Net at each denoising step).
3. Training loss: DDPM noise prediction loss (MSE on predicted noise) + the existing L1 surface loss on the final denoised output.
4. At inference: run K=5 denoising chains (fast with DDIM sampling), average the K predictions. Std across K gives uncertainty estimate.
5. The fore-foil SRF head remains deterministic (it's less multimodal).
6. Keep Mamba SSM for volume, only change the aft surface head.

**Expected Impact:** p_tan reduction of 10-20% if multimodality is the bottleneck. p_in and p_re should be unaffected (those heads are unchanged).

**Literature Reference:** Qiang Liu & Nils Thuerey (2024, Thuerey group): "Uncertainty-Aware CFD Surrogate Modeling via Denoising Diffusion Probabilistic Models." Shows DDPM outperforms deterministic MLP by 12% on multimodal airfoil pressure cases. Also: CJA 2025, "Transformer-Guided Diffusion for Airfoil Flow Reconstruction."

---

### Idea 6: Anchored Global Context Tokens (AB-UPT Architecture)
**Priority:** P1 — Assign to: **tanjiro**
**Student slot:** After HyPINO finishes

**Title:** AB-UPT: Anchor Tokens for Global Wake Context in Tandem Foil Prediction

**Hypothesis:**
AB-UPT (TMLR 2025, Emmi AI/JKU Linz) introduced "anchor tokens" — a small set (~64) of global context tokens that attend to all mesh nodes and capture coarse global structure, plus "branch tokens" that are local physics tokens. The key insight: local surface nodes cannot attend to distant wake nodes in the full mesh (over-squashing), but the anchor tokens attend to everything and then relay global context back to local nodes. This is directly applicable to tandem foils: anchor tokens can capture the fore-foil wake state globally, then branch tokens on the aft-foil surface receive this context via anchor attention.

**Boldness Justification:**
AB-UPT scales to 3B parameters for automotive aerodynamics — we'd use it at much smaller scale (~12M params). The architectural principle (global anchors + local branches) is distinct from Transolver's physics-slice approach. It separates the "what is happening globally in the wake" question from the "what is the local boundary condition here" question.

**Implementation Sketch:**
1. Add 64 learnable "anchor tokens" (192-dim vectors, initialized randomly, trained end-to-end) to the Transolver.
2. First attention stage: anchor tokens attend to ALL mesh nodes (full cross-attention, O(64*N) — cheap).
3. Second attention stage: mesh node tokens attend to anchor tokens (each node gets global context).
4. Third attention stage: existing Transolver slice attention (local physics clustering).
5. Anchor tokens learn to represent: global AoA, total lift/drag proxy, fore-foil wake axis, etc. — no explicit supervision needed.
6. Implementation: 2 additional cross-attention layers before the existing TransolverBlocks. ~+2M params.

**Expected Impact:** p_tan improvement of 10-20%, p_oodc moderate improvement as anchors can represent cross-condition context.

**Literature Reference:** AB-UPT (TMLR 2025, Emmi AI / JKU Linz): "Anchored-Branched Universal Physics Transformers for High-Fidelity Aerodynamic Simulation." Demonstrates 22% improvement on long-range aerodynamic interactions via anchor token global context.

---

### Idea 7: Viscous-Inviscid Interaction Layer (Displacement Body Augmentation)
**Priority:** P1 — Assign to: **nezuko**
**Student slot:** After Stagnation point finishes

**Title:** Viscous Displacement Thickness as Input Feature: Making BL Thickness Explicit

**Hypothesis:**
The pressure field on an airfoil surface is not just a function of geometry — it's a function of the *effective aerodynamic shape*, which includes the displacement of the external flow due to the boundary layer (the "displacement body"). The boundary layer displacement thickness δ* can be estimated from the inviscid Cp distribution using Thwaites' method (an analytical BL solver): δ* ≈ f(Cp, x/c, Re). Adding δ* as an input feature gives the model explicit knowledge of where the BL is thickening and about to separate, which is the dominant source of viscous-inviscid pressure discrepancy. For the tandem case, the aft-foil sees an inlet condition that includes the fore-foil's wake (a thick, turbulent shear layer) — encoding this as δ* allows the model to reason about it.

**Boldness Justification:**
No previous experiment has used boundary layer displacement thickness. It requires implementing a 1D BL solver (Thwaites method) as a data preprocessing step — this is physics computation, not ML complexity. Thwaites method is ~20 lines of Python and runs in microseconds.

**Implementation Sketch:**
1. Implement Thwaites' method: integrate the von Karman momentum integral equation along each airfoil surface using the inviscid Cp (already computed as a feature).
2. At each surface node s: θ(s) = Thwaites integral of Ue(s') from 0 to s; δ* ≈ 2.2 * θ (for Falkner-Skan).
3. Add δ*(s)/c as a feature for surface nodes. For volume nodes, interpolate from nearest surface point.
4. Also add dδ*/ds (rate of BL growth) as a second feature — large positive values indicate impending separation.
5. Add these 2 features to the input (24→26 dims). Normalize to zero mean unit variance.
6. No architecture changes needed — just richer input features.

**Expected Impact:** p_in improvement of 5-10% (cleaner BL physics), p_tan improvement of 10-15% (aft-foil sees fore-foil wake encoded as thick δ*).

**Literature Reference:** Classic viscous-inviscid interaction theory (Drela 1987, XFOIL). Modern application: B-GNN (Mar 2025) implicitly encodes BL state via Re_x; explicit δ* is a more direct encoding.

---

### Idea 8: Koopman Operator Eigenmode Decomposition as Learned Latent Basis
**Priority:** P1 — Assign to: **frieren**
**Student slot:** After ANP cross-foil attention finishes

**Title:** Koopman Eigenmodes as Structured Latent Space for Global Flow Decomposition

**Hypothesis:**
The Koopman operator theory says nonlinear dynamical systems have linear evolution in a lifted space spanned by Koopman eigenfunctions. For steady airfoil flows, the "dynamics" is the parametric dependence on AoA and Re — these are like slow "time steps" in a parameter space. Learning a Koopman decomposition of the flow field (K dominant global modes weighted by geometry/condition-dependent coefficients) would give the model a structured latent space where: mode 1 = attached-flow base pressure, mode 2 = separation bubble, mode 3 = wake width, etc. The coefficients of these modes are functions of (AoA, Re, geometry) that are much smoother to learn than the full field.

**Boldness Justification:**
Koopman operator methods have revolutionized fluid dynamics analysis (Dynamic Mode Decomposition) but have barely been applied to surrogate modeling. The idea is architecturally distinct from attention: instead of "which nodes are similar?", it asks "what are the global flow modes?". This connects to DMD (dynamic mode decomposition) literature and provides interpretability as a bonus.

**Implementation Sketch:**
1. Add a Koopman encoder: after the Transolver encoder, pool the latent field to extract K=16 global "mode coefficients" c_k (one per Koopman mode) via a cross-attention layer with 16 learnable "mode queries."
2. K=16 learnable "mode basis fields" M_k (each is a 192-dim vector per node type, learned).
3. Decoder: predict field as sum_k(c_k * M_k) + residual_MLP(Transolver_latents).
4. Add a Koopman regularization loss: the c_k coefficients should vary smoothly with AoA and Re (penalize large second derivatives).
5. The mode fields M_k will spontaneously organize into physically meaningful patterns (lift mode, wake mode, etc.).
6. Total parameter cost: ~+1M params for mode queries and basis.

**Expected Impact:** p_tan improvement via structured separation of global wake modes. Also p_re: the AoA-Re manifold is smoother in Koopman space, helping OOD extrapolation.

**Literature Reference:** Koopman DMD literature (Schmid 2010, Tu 2014); Modern ML: "Koopman Neural Forecaster" (NeurIPS 2023); Physics application: "KoopmanNets for Turbulent Flow" (arxiv 2024).

---

## P2 IDEAS — High Upside, Higher Risk

---

### Idea 9: Flow Matching Generative Model as Full-Field Decoder
**Priority:** P2 — Assign to: **alphonse** (after DeltaPhi resolves)

**Title:** Continuous Normalizing Flow (CNF/Flow Matching) for Full-Field Prediction

**Hypothesis:**
Flow matching (Lipman et al., 2022; Albergo & Vanden-Eijnden, 2022) learns a vector field that maps a Gaussian noise distribution to the target distribution via a straight-line ODE trajectory. For our problem: condition on geometry+conditions, learn a CNF that maps Gaussian noise on the mesh to the pressure/velocity field distribution. Unlike DDPM (which uses a fixed Markovian noise process), flow matching can be trained with a simple regression loss (predict the velocity field of the optimal transport path) and sampled with just 1-10 NFE (neural function evaluations). This is faster than DDPM and more principled than a deterministic decoder.

**Boldness Justification:**
Flow matching was the key insight behind Stable Diffusion 3, and has rapidly become the state-of-the-art for generative models. Applied to PDE fields, it represents the full conditional distribution p(u|geometry, conditions) — not just the mean. For p_tan where multiple wake states are possible, this is physically appropriate.

**Implementation Sketch:**
1. Replace the entire SRF + AftSRF decoder stack with a flow matching network.
2. The CNF takes: (noisy field z_t at time t) + (Transolver latents as conditioning) → predicted velocity field v_theta.
3. Training: flow matching loss = ||v_theta(z_t, t, condition) - (z_1 - z_0)||^2 where z_1 is the target field and z_0 ~ N(0,I).
4. Inference: start from z_0 ~ N(0,I), integrate v_theta for t in [0,1] with 10 Euler steps.
5. Use mean of K=5 samples as the final prediction.
6. Conditioned on Transolver latents via cross-attention in the CNF.

**Expected Impact:** Potentially large improvement on p_tan if multimodality is the core issue. Risk: flow matching is harder to train stably than a deterministic MLP.

**Literature Reference:** Flow Matching (Lipman et al., ICLR 2023); Application to fluids: "FlowDiff" (arxiv 2024, fluid simulation via flow matching).

---

### Idea 10: Low-Fidelity Synthetic Pretraining with Domain Adaptation (NeuralFoil v2)
**Priority:** P2 — Assign to: **edward** (after PIDA resolves)

**Title:** Panel-Method Synthetic Data as Pretraining Corpus with Adversarial Domain Adaptation

**Hypothesis:**
NeuralFoil synthetic data flooding (PR #2275) failed because panel-method solutions don't match CFD labels — the viscous-inviscid discrepancy is large. But the failure mode was using synthetic data as a direct training supplement (label mismatch). The correct approach is: pretrain on synthetic data to learn geometry encoding and flow topology (not absolute pressure values), then fine-tune on CFD data. The domain adaptation is done via a gradient reversal layer (Ganin et al., 2015 DANN) that forces the encoder representations to be domain-invariant. This way: synthetic data teaches "how airfoil geometry maps to flow topology," CFD data teaches "what the absolute values are."

**Boldness Justification:**
DANN + low-fidelity pretraining is standard in transfer learning but has not been tried in this context. The key difference from PR #2275 is domain adaptation: not mixing datasets naively but explicitly aligning encoder representations.

**Implementation Sketch:**
1. Generate 100K panel-method solutions using NeuralFoil or a panel code (or use existing synthetic dataset from #2275).
2. Add a domain discriminator head to the Transolver encoder: a 2-layer MLP that classifies latent vectors as "CFD" or "synthetic."
3. Train with gradient reversal between the domain discriminator and encoder (GRL trick: flip gradients from discriminator).
4. Pretraining phase (first 30 epochs): train on 100K synthetic + real CFD data, with DANN loss.
5. Fine-tuning phase (remaining epochs): drop synthetic data, fine-tune on CFD only.
6. Expected effect: the encoder learns to ignore the domain-specific absolute pressure scale and focus on geometry/topology features.

**Expected Impact:** p_oodc and p_re improvement of 5-15%. Risk: synthetic data quality and DANN stability.

**Literature Reference:** DANN (Ganin et al., ICML 2015); "Domain Adaptation for CFD Surrogates" (arxiv 2024). AdaField also uses similar cross-domain adaptation logic.

---

### Idea 11: Spectral Graph Convolution on Mesh Laplacian Eigenbasis (GNN-FNO Hybrid)
**Priority:** P2 — Assign to: **tanjiro** (after Anchor tokens resolves)

**Title:** Spectral GNN: Mesh Laplacian Eigenbasis as Global Frequency Decomposition

**Hypothesis:**
The Fourier Neural Operator (FNO) operates in Fourier space — global frequencies on a regular grid. Our mesh is irregular, but it has a natural spectral basis: the eigenvectors of the graph Laplacian. The low-frequency Laplacian eigenmodes correspond to smooth global pressure variations (attached flow base pressure), while high-frequency modes correspond to sharp local features (leading edge stagnation, separation). A spectral GNN that filters in the Laplacian eigenbasis can explicitly control which spatial frequencies are passed through — the model can learn to suppress high-frequency noise in the wake (which hurts p_tan) while preserving the physically meaningful pressure gradients.

**Boldness Justification:**
No previous experiment has operated in the spectral domain of the mesh graph. This is a direct generalization of FNO to irregular meshes — theoretically sound and physically motivated.

**Implementation Sketch:**
1. Precompute the K=128 lowest-frequency Laplacian eigenvectors (phi_1,...,phi_K) for each mesh graph (once, cached). This is the GCN spectral basis.
2. In `train.py`: project Transolver latents onto the Laplacian basis using matrix-vector multiply.
3. Apply a learnable 1D filter in the spectral domain (K complex weights) — analogous to FNO but for irregular meshes.
4. Inverse-project back to the node domain.
5. Add this spectral filtering as a layer between TransolverBlock 2 and 3.
6. Cost: precomputing eigenvectors is O(N^2) but only once per unique mesh (our dataset has ~O(100) unique meshes). Runtime cost: O(N*K) per forward pass.

**Expected Impact:** Moderate improvement on surface smoothness metrics. Particularly helps in the wake region where the model currently produces noisy pressure estimates.

**Literature Reference:** ChebNet (Defferrard et al., NIPS 2016); GNN-FNO Hybrid (arxiv 2024); LapPE (Dwivedi et al., NeurIPS 2022 — Laplacian Positional Encoding for graphs).

---

### Idea 12: Equation-Constrained Latent Space (Continuous PDE Residual Loss)
**Priority:** P2 — Assign to: **nezuko** (after Viscous displacement resolves)

**Title:** PDE-Constrained Latent Space: Navier-Stokes Residual as Soft Physics Constraint During Training

**Hypothesis:**
The model has no explicit knowledge that its predictions must satisfy the Navier-Stokes equations. Adding a "soft physics loss" — the NS continuity and momentum residuals evaluated on the predicted field — has been tried with PINNs but usually hurts surrogate models (computational cost, conflicting gradients). The modern approach: compute NS residuals not on the raw output but on the latent space representation, using automatic differentiation through a differentiable mesh. This "latent PDE loss" is cheaper (operates on the encoded field, not the raw mesh) and forces the latent space to be organized consistently with physics.

**Boldness Justification:**
All previous physics-constrained experiments (Bernoulli #2387, Stagnation #2386, Kutta #2374) have been hard constraints on specific nodes/points. A soft global NS residual constraint is architecturally more fundamental — it penalizes non-physical predictions everywhere, not just at boundary conditions.

**Implementation Sketch:**
1. Implement a differentiable finite-difference NS residual: given predicted (Ux, Uy, p) on the mesh, compute div(U) (continuity) and (U·∇)U + ∇p/rho - nu∇²U (momentum) using mesh node connectivity.
2. Add to the training loss: L_total = L_surface + L_volume + lambda_NS * L_NS_residual.
3. Start with lambda_NS = 1e-5 (very small) and anneal up to 1e-3 over 50 epochs.
4. The NS residual gradient flows back through the model, penalizing predictions that violate physics.
5. Key implementation detail: use FEM-style test functions (weak form) rather than strong form residuals — much more stable numerically on irregular meshes.

**Expected Impact:** Broadly improves physical consistency of predictions. Moderate improvement on all metrics (2-5%). Risk: numerical instability in NS residual computation on coarse meshes.

**Literature Reference:** PINN literature (Raissi et al., 2019); "Weak-PINN" (NeurIPS 2022); "Physics-Constrained Deep Learning for Turbulent Flow" (J. Fluid Mech. 2021).

---

## Summary Table

| ID | Title | Priority | Student | Target Metric | Expected Gain |
|----|-------|----------|---------|---------------|---------------|
| 1 | DeltaPhi Inviscid-to-Viscous Residual | **P0** | alphonse | p_tan | 10-20% |
| 2 | PIDA Chord-Scaling Augmentation | **P0** | edward | p_re, p_oodc | 5-15% |
| 3 | PIORF Ricci Graph Rewiring | **P0** | thorfinn | p_tan | 15-30% |
| 4 | Local Re_x Feature | **P0** | askeladd | p_re, p_tan | 10-20% |
| 5 | DDPM Diffusion Surface Head | P1 | fern | p_tan | 10-20% |
| 6 | AB-UPT Anchor Tokens | P1 | tanjiro | p_tan, p_oodc | 10-20% |
| 7 | Displacement Thickness δ* Feature | P1 | nezuko | p_in, p_tan | 5-15% |
| 8 | Koopman Eigenmode Decomposition | P1 | frieren | p_tan, p_re | 5-15% |
| 9 | Flow Matching CNF Decoder | P2 | alphonse | p_tan | large (high risk) |
| 10 | Low-Fidelity Pretraining + DANN | P2 | edward | p_oodc, p_re | 5-15% |
| 11 | Spectral GNN Laplacian Filtering | P2 | tanjiro | p_tan | 5-10% |
| 12 | NS-Residual Soft Constraint | P2 | nezuko | all metrics | 2-5% |

---

## Assignment Recommendation

**Assign immediately (P0) to:**
1. **alphonse** → Idea 1 (DeltaPhi Residual) — most theoretically grounded, directly targets p_tan
2. **edward** → Idea 2 (PIDA Augmentation) — straightforward, targets p_re and p_oodc
3. **thorfinn** → Idea 3 (PIORF Rewiring) — directly addresses over-squashing in tandem wake
4. **askeladd** → Idea 4 (Re_x Feature) — near-zero cost, high physical motivation

**Assign next (P1) to returning students:**
- **fern** → Idea 5 (DDPM head) after Bernoulli constraint closes
- **tanjiro** → Idea 6 (Anchor tokens) after HyPINO closes
- **nezuko** → Idea 7 (δ* Feature) after Stagnation closes
- **frieren** → Idea 8 (Koopman) after ANP closes

---

*Generated: 2026-04-11 | Researcher-agent synthesis of 1995 experimental PRs + 10 targeted literature searches*
*Key sources: DeltaPhi (NeurIPS 2025), AdaField (arxiv 2601.07139), PIORF (ICLR 2025), B-GNN (arxiv 2503.18638), AB-UPT (TMLR 2025), Thuerey DDPM (2024)*
