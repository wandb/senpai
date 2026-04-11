# Round 38 Research Ideas — 2026-04-10

Current baseline (PR #2319): p_in=11.709, p_oodc=7.544, p_tan=27.402, p_re=6.481

In-flight (avoid duplicating): #2340, #2341, #2344, #2345, #2346, #2347, #2348, #2349

---

## Idea 1: Joukowski Cp Superposition — Camber-Thickness Correction to Panel Cp

**What it is.** The current panel Cp (PR #2319) uses a flat-plate approximation: Cp_upper(x) = -2·sin(α)/√(x·(1-x)). Real airfoil Cp departs from this by a thickness and camber correction. Using the thin-airfoil Fourier series (A0=α-camber_integral, A1..An from camber slope) gives a corrected Cp that accounts for the actual NACA profile shape rather than a flat plate.

**Why it might help here.** The current panel Cp is geometry-agnostic — every airfoil with the same AoA gets identical Cp. Adding the A0/A1 camber terms (computable analytically from the NACA 4-digit/5-digit parameterization already in the feature set) makes the input feature geometry-aware. The aft foil sees a different effective AoA due to wake interference; a camber-corrected Cp would capture that geometry×flow interaction more precisely than the flat-plate version. This is the next logical step from the cp_panel_scale=0.1 success.

**Literature grounding.**
- Thin airfoil theory (Kutta-Joukowski): standard aerodynamics textbook (Anderson, "Fundamentals of Aerodynamics"). The Fourier cosine series A0..An are deterministic from camber line slope.
- NeuralFoil (arXiv:2503.16323, Mar 2025): uses panel method as a physics oracle; the paper discusses why thin-airfoil Fourier coefficients capture camber effects better than flat-plate approximations.

**Implementation complexity.** ~40-60 LoC in train.py. Compute NACA camber line slope from chord fraction x and the 4-digit NACA parameters (m, p already derivable from geometry features); compute A0 = α - (1/π)∫camber_slope·dθ; add correction ΔCp = -(A0/2 + A1·cos(θ)) to flat-plate Cp. Gate with `--cp_joukowski` flag, keep `--cp_panel` behavior unchanged. Scale = 0.1 initially (same as cp_panel_scale).

**Metrics targeted.** p_in, p_oodc, p_tan — all metrics that benefit from more accurate surface pressure physics hints.

**Suggested experiment.** Add `--cp_joukowski` flag that computes camber-corrected Cp using NACA 4-digit parameterization. Start with `--cp_panel_scale 0.1` unchanged. Compare against baseline (PR #2319). Expect biggest gain on p_tan where wake-modified effective AoA makes the camber correction most significant.

**Confidence.** Moderate. The flat-plate panel feature already helps; adding geometry information to it is the natural extension. Risk: if the backbone already learns the camber correction implicitly from DSDF features, the marginal gain may be small.

---

## Idea 2: Re-Conditioned Surface Refinement Head via FiLM

**What it is.** The surface refinement head (SRF) and aft-foil SRF receive backbone hidden states but have no explicit conditioning on Reynolds number. FiLM (Feature-wise Linear Modulation) injects log(Re) as a conditioning signal via learned scale and shift applied to the SRF hidden activations: h_conditioned = γ(log_Re) · h + β(log_Re), where γ,β are 2-layer MLPs.

**Why it might help here.** The p_re metric (Reynolds generalization) is the hardest to beat — it represents OOD flow regimes. The SRF head learns a fixed correction function; at extreme Re, the boundary layer physics change qualitatively (laminar vs. turbulent separation, stagnation point shift). FiLM conditioning allows the SRF to apply different correction strategies at different Re without a separate head. This directly targets the p_re weakness without adding a new loss term. Domain FiLM has been shown to work well in neural operators for PDEs with varying coefficients (e.g., meta-learning neural operators).

**Literature grounding.**
- FiLM: "FiLM: Visual Reasoning with a General Conditioning Layer" (Perez et al., 2018, AAAI). Standard technique, well-validated.
- "Meta-Learning Neural Operator Surrogates for Physics Simulations" (ICLR 2025, various): FiLM-style conditioning on PDE parameters for cross-Re generalization.
- Domain-conditioned operator learning: standard in fluid ML (see CORAL, arXiv:2306.03030).

**Implementation complexity.** ~50-70 LoC. In SurfaceRefinementHead.__init__, add two linear layers (2-layer MLP, hidden=32) producing γ, β of shape [hidden_dim]. In forward, after each Linear+GELU block, apply h = γ(log_re) * h + β(log_re). Zero-init β output layer so conditioning starts as identity. Add `--srf_film_conditioning` flag.

**Metrics targeted.** p_re primarily; p_in and p_oodc secondarily.

**Suggested experiment.** Add FiLM conditioning to both SRF and AftFoilSRF heads. Use log(Re)/log(1e6) as normalized conditioning input. Hidden dim for γ,β MLP = 32. Compare against PR #2319 baseline. Flag: `--srf_film_conditioning`.

**Confidence.** Moderate-high. FiLM is well-validated for parameter-conditional network adaptation. The connection to Re-generalization is direct. Risk: the backbone already receives Re via input features; if those are sufficient, the SRF conditioning may be redundant.

---

## Idea 3: Vortex-Panel Induced Velocity as Per-Node Input Feature

**What it is.** Instead of just scalar Cp at surface nodes, compute the 2D velocity field induced by a discrete vortex-panel representation of both foils (using the Biot-Savart kernel in 2D: u = Γ/(2π) · dy/r², v = -Γ/(2π) · dx/r²). For each mesh node, sum contributions from all panels. This gives (u_induced, v_induced) as two additional input channels that encode the full inviscid velocity field from panel theory, not just the surface pressure.

**Why it might help here.** Panel Cp gives a surface-only physics hint; induced velocity extends the physics oracle to every mesh node (volume and surface). For tandem configurations, the aft-foil induced velocity captures the wake interference effect analytically — the fore foil's vortex sheet creates a downwash at the aft foil that shifts the effective AoA. This is a stronger physics signal than the wake deficit feature (which only encodes distance to the trailing edge) because it encodes the actual velocity direction and magnitude from inviscid theory.

**Literature grounding.**
- Classical panel method: Hess & Smith (1966), standard aerodynamics. The O(N_panels × N_nodes) computation is tractable (N_panels ~100, N_nodes ~85K = 8.5M multiplications per sample, negligible on GPU).
- "Predicting airfoil pressure distribution using boundary graph neural networks" (arXiv:2503.18638, Mar 2025): uses boundary-only inputs for field prediction; vortex-induced velocity is the natural analytical complement.
- NeuralFoil (arXiv:2503.16323): uses panel method as oracle; induced velocity is the velocity analog.

**Implementation complexity.** ~80-100 LoC in train.py (or prepare.py — but that's read-only, so compute in train.py collation). Precompute panel control points from surface node coordinates (boundary_id=6,7); compute vortex strengths Γ from flat-plate approximation; broadcast Biot-Savart kernel over all mesh nodes. Cache per-sample. Gate with `--vortex_panel_velocity` flag. Scale independently.

**Metrics targeted.** p_in, p_oodc, p_tan — the metrics that reflect pressure prediction quality.

**Suggested experiment.** Compute (u_ind, v_ind) from a 64-panel vortex sheet for each foil. Add as 4 additional input channels (u_fore, v_fore, u_aft, v_aft where aft channels are zero for single-foil). Scale by 0.1 initially. Flag: `--vortex_panel_velocity --vortex_panel_scale 0.1`. This is a richer physics feature than cp_panel and more likely to generalize across the tandem configuration space.

**Confidence.** Moderate. The physics motivation is strong. Main risk: computation cost in the training loop (O(NxM) per sample), but with N=100 panels and M=85K nodes, 8.5M multiply-adds per sample is fast on GPU. Second risk: the induced velocity from inviscid flat-plate theory may diverge significantly from viscous reality at high AoA, making it noisy.

---

## Idea 4: Spectral Regularization on FFN Weights (Lip-SRF)

**What it is.** Add spectral norm regularization as a soft penalty on the FFN weight matrices in each Transolver block. Specifically, add λ·∑_l σ_max(W_l)² to the loss, where σ_max is the largest singular value of each FFN weight matrix. This penalizes large Lipschitz constants, biasing the model toward smoother and more generalizable function approximations.

**Why it might help here.** The p_re (Reynolds OOD) and p_oodc (OOD conditions) metrics reflect generalization to unseen flow regimes. High Lipschitz constants in the FFN layers are a known predictor of poor OOD generalization — they allow the model to memorize training distributions by amplifying small input differences. Spectral regularization is the theoretically-grounded way to control this, and it was specifically validated for continual learning generalization (ICLR 2025). The current model uses weight_decay=5e-5 which is a crude L2 proxy; spectral regularization targets the same objective more precisely.

**Literature grounding.**
- "Spectral regularization for continual learning" (ICLR 2025): shows spectral norm regularization improves OOD generalization in neural networks with structured weight matrices.
- "Spectral Normalization for Generative Adversarial Networks" (Miyato et al., 2018, ICLR): proved that bounding σ_max stabilizes training and improves generalization.
- POET optimizer (ICLR 2026 under review): uses spectral stability as an optimization objective.

**Implementation complexity.** ~30-40 LoC. In the loss computation, add: `spec_loss = sum(torch.linalg.norm(W, ord=2)**2 for W in ffn_weights)`. Use `named_parameters()` to collect FFN weight matrices. Multiply by λ=1e-4. Backprop through this — spectral norm of a matrix is differentiable via torch.linalg. Gate with `--spectral_reg --spectral_reg_lambda 1e-4` flag. No architecture change required.

**Metrics targeted.** p_re, p_oodc — the OOD generalization metrics.

**Suggested experiment.** Add `--spectral_reg --spectral_reg_lambda 1e-4` to the baseline. Try λ ∈ {1e-5, 1e-4, 1e-3} — the `--wandb_group` should group these as `round38/spectral-reg`. The right λ is critical: too small = no effect, too large = underfitting.

**Confidence.** Moderate. Spectral regularization is well-validated for generalization; the question is whether the FFN weights are actually the bottleneck in this model's OOD behavior. The Re-stratified sampling already helps p_re; spectral reg targets a different mechanism (smoothness vs. data distribution).

---

## Idea 5: Two-Stage Surface Refinement — Velocity First, Pressure Second

**What it is.** Split the SRF into two sequential MLPs: Stage 1 predicts (ΔUx, ΔUy) corrections from backbone hidden states; Stage 2 receives both the backbone hidden states AND the Stage 1 velocity correction, and predicts Δp. This exploits Bernoulli physics: pressure is a function of velocity, so having an intermediate velocity estimate as context should improve pressure predictions.

**Why it might help here.** The current single SRF head predicts (ΔUx, ΔUy, Δp) simultaneously, but pressure and velocity corrections are not independent — Bernoulli's equation links them. By forcing the model to commit to a velocity correction before predicting the pressure correction, Stage 2 can use physical consistency as an implicit constraint. This is motivated by the `--pressure_deep` and `--pressure_first` flags already in the baseline, which reflect the observation that pressure prediction benefits from explicit architectural separation.

**Literature grounding.**
- "PirateNets: Adaptive Residuals for Physics-Informed Learning" (arXiv:2402.00265, NeurIPS 2024): motivates sequential residual corrections for PDE quantities.
- Bernoulli-informed sequential prediction: implicit in the design of "pressure-velocity coupling" solvers (SIMPLE algorithm in CFD), where pressure is solved from the velocity field in a corrector step.
- The `--pressure_first` and `--pressure_deep` flags in the current codebase reflect empirical validation of this physics-informed architectural ordering.

**Implementation complexity.** ~60-80 LoC. Replace SurfaceRefinementHead with TwoStageSRF: Stage1 = MLP(hidden→[Ux, Uy] corrections), Stage2 = MLP(hidden + Ux_pred + Uy_pred → p correction). Both stages are 3-layer MLPs with hidden=192. Apply the same to AftFoilSRF. Gate with `--srf_two_stage` flag.

**Metrics targeted.** p_in, p_oodc, p_tan (pressure metrics); the velocity metrics may also improve slightly.

**Suggested experiment.** Add `--srf_two_stage` flag replacing the current SRF forward pass. Stage 2 input: concat(backbone_hidden, Ux_correction, Uy_correction). Keep hidden=192, layers=3 for both stages. Compare against PR #2319 baseline.

**Confidence.** Moderate. The physical motivation is sound. Risk: the current single-stage SRF may already learn this decomposition implicitly; explicit staging adds ~2x parameters for the SRF heads (small absolute cost) but may not improve over implicit joint prediction.

---

## Idea 6: Wake Angle Feature — atan2(dy/gap, dx/gap)

**What it is.** The existing wake deficit feature (PR #2213) encodes (dx/gap, dy/gap) — the signed normalized offset from each node to the fore foil trailing edge. Adding atan2(dy/gap, dx/gap) as a 3rd channel gives the signed angle of each node relative to the wake centerline. This polar representation complements the Cartesian (dx, dy) features and is invariant to gap scaling.

**Why it might help here.** Nodes upstream vs. downstream of the wake center behave very differently — upstream nodes see undisturbed flow, while downstream nodes see the velocity deficit and pressure recovery. The (dx, dy) features encode this as a Cartesian offset, but the angular coordinate encodes the "wake sector" more directly. This is analogous to the TE coordinate frame (PR #2207) which uses both distance and direction for trailing-edge features. Given that the wake deficit feature gave -4.1% on p_in, the angle refinement might compound that gain.

**Literature grounding.**
- Polar coordinate representations for fluid wakes: standard in experimental fluid dynamics (hot-wire anemometry traverses are typically done in polar coordinates relative to the wake center).
- GeoMPNN (arXiv:2412.09399, NeurIPS 2024 ML4CFD Best Student Paper): uses directional features for geometry-aware message passing.

**Implementation complexity.** ~15-20 LoC. In the wake deficit feature computation, add `wake_angle = torch.atan2(dy_gap, dx_gap + 1e-8)` and append as a 3rd channel. Adjust input_dim by +1. Gate with `--wake_angle_feature` flag that requires `--wake_deficit_feature`. Normalize by dividing by π to keep in [-1, 1].

**Metrics targeted.** p_in primarily (the biggest wake deficit gain was here); p_tan secondarily.

**Suggested experiment.** Add `--wake_angle_feature` flag (requires `--wake_deficit_feature`). Single seed first to validate direction, then 2-seed if positive. This is a very cheap change and a natural follow-on to PR #2213.

**Confidence.** Moderate. The angular feature captures directional information already implicitly present in (dx, dy) but in a more rotation-invariant form. Risk: if the backbone already extracts this from the Cartesian features, the gain may be negligible.

---

## Idea 7: Surface-Normal Local Coordinate Frame for SRF Input

**What it is.** For each surface node, compute the inward surface normal direction and reproject the backbone hidden-state gradient (or the velocity prediction) into the (tangential, normal) coordinate frame aligned with the surface. Feed this reprojected representation as additional input to the SRF head, alongside the standard backbone hidden state.

**Why it might help here.** The SRF correction head receives global coordinate backbone features. But pressure gradients are highest in the surface-normal direction (no-slip condition means velocity is zero at the wall; the boundary layer pressure gradient ∂p/∂n ≈ 0 by thin boundary layer assumption, but the tangential pressure gradient drives the flow). A surface-normal coordinate system makes these physics transparent to the SRF MLP, potentially allowing it to learn wall-normal corrections more easily. This is complementary to the TE frame (which is TE-relative rather than surface-normal).

**Literature grounding.**
- "Neural fields for rapid aircraft aerodynamics simulations" (Nature Scientific Reports, Oct 2024): uses surface-normal projection for boundary condition enforcement.
- GeoMPNN (arXiv:2412.09399): uses geometric coordinate frames for physics-aware message passing on meshes.
- Boundary layer theory (Prandtl): the surface-normal direction is the physically privileged direction for wall-bounded flows.

**Implementation complexity.** ~50-70 LoC. Precompute surface normals from the mesh connectivity (finite-difference or cross-product of adjacent edge vectors). For each surface node, build a 2×2 rotation matrix R=[tangent, normal]. Rotate the node's (dx, dy) backbone features into this frame. Append as +2 channels to SRF input. Gate with `--srf_normal_frame` flag.

**Metrics targeted.** p_in, p_oodc, p_tan — primarily the surface pressure metrics where the SRF head has most influence.

**Suggested experiment.** Compute surface normals from the boundary node connectivity using adjacent node vectors. Add `--srf_normal_frame` flag that appends the normal-frame-reprojected position features to the SRF input. Keep SRF hidden=192 unchanged (the extra 2 channels are absorbed by the first linear layer).

**Confidence.** Moderate. The physical motivation is clear. Risk: surface normals require mesh connectivity computation that may not be straightforward in the current data format — need to verify that boundary node ordering in the dataset is consistent enough for finite-difference normal estimation.

---

## Idea 8: Learnable Cp Panel Scale (Per-Domain)

**What it is.** The current cp_panel feature uses a fixed scale=0.1 for all samples. Replace this with a learned scalar parameter per domain (tandem vs. single-foil), initialized to 0.1, that is jointly optimized with the rest of the model. This allows the model to find the optimal Cp signal strength rather than requiring the advisor to hand-tune it.

**Why it might help here.** The cp_panel_scale=0.1 was found empirically by PR #2319 after trying 1.0 (too strong, hurt p_in by 5.2%) and 0.0 (no physics hint). A learnable scale allows the model to self-calibrate: if the Cp signal is too strong at the current weight, gradient descent will reduce the scale; if it's too weak, it will increase it. Per-domain learnable scales allow the tandem domain (where Cp hint matters more due to wake interference) to use a different effective scale than single-foil samples.

**Literature grounding.**
- Learnable input scaling: standard practice in multi-modal fusion (learned modality weights in vision-language models).
- "NeuralFoil: Machine Learning for Airfoil Aerodynamics" (arXiv:2503.16323): discusses adaptive weighting of physics-based features vs. learned representations.

**Implementation complexity.** ~20-30 LoC. Replace the fixed `cp_panel_scale` multiplication with `x[:, cp_idx] * torch.sigmoid(self.cp_scale_tandem) * 2` for tandem samples and a separate `cp_scale_single` for single-foil (or just zero). Initialize log_scale = log(0.1) ≈ -2.3 so sigmoid * 2 ≈ 0.1 at init. Register as nn.Parameter. Gate with `--cp_panel_learnable_scale` flag.

**Metrics targeted.** p_in, p_oodc, p_tan — all metrics where cp_panel contributed.

**Suggested experiment.** Add `--cp_panel_learnable_scale` flag. Initialize per-domain scale parameters at 0.1. Monitor the learned scale values via W&B (log them as scalars). Compare against PR #2319 fixed-scale baseline.

**Confidence.** Moderate-high. This is a low-risk, low-complexity change that removes one hyperparameter from the search space. The question is whether self-calibration actually improves over the hand-tuned 0.1.

---

## Idea 9: Mixture-of-Experts Routing in Transolver Slices (Domain-Conditioned)

**What it is.** Replace the single FFN after Transolver attention with two expert FFNs — one specialized for tandem-regime physics, one for single-foil physics — with a learned soft router that conditions expert weights on the gap/stagger scalars (for tandem) or a single-foil indicator. The router uses `gap * stagger` interaction features to compute mixing weights between experts.

**Why it might help here.** The Transolver backbone processes single-foil, in-distribution tandem, and OOD tandem geometries identically through the same FFN weights. Tandem aerodynamics is fundamentally different from single-foil: wake interference, effective AoA modification, and pressure recovery interactions are all domain-specific. Two expert FFNs allow the model to develop tandem-specific and single-foil-specific representations without competing for the same weight space. This directly targets the p_tan metric.

**Literature grounding.**
- "Mixture of Neural Operator Experts for Boundary Conditions" (ICLR 2026 under review): MoE routing conditioned on domain indicators for physics operators.
- "MoE Operator Transformer for Large-Scale PDE Pre-Training" (NeurIPS 2025): shows MoE routing improves cross-domain PDE generalization.
- NESTOR (arXiv:2602.22059, Feb 2026): nested MoE for neural operators; shows consistent gains over single-FFN baselines.

**Implementation complexity.** ~80-100 LoC. In TransolverBlock.forward, replace `self.ffn(x)` with: `w = sigmoid(router(gap_stagger_feature)); x = w * expert1(x) + (1-w) * expert2(x)`. Router is a 2-layer MLP (input=2, hidden=16, output=1). Single-foil samples use w=0 (expert2 only). Initialize both experts with the same weights as the original FFN. Gate with `--moe_domain_experts` flag.

**Metrics targeted.** p_tan primarily (the tandem-specific metric); p_in and p_oodc secondarily.

**Suggested experiment.** Add `--moe_domain_experts` with soft routing conditioned on gap/stagger. Apply MoE to all 3 Transolver blocks. Use W&B group `round38/moe-domain`. Monitor routing weights via W&B to verify the router learns a meaningful split.

**Confidence.** Moderate. MoE in transformer blocks has strong empirical support; the domain-conditioned routing is well-motivated by the data heterogeneity. Risk: with only 2 experts and soft routing, the model may collapse to near-uniform mixing (both experts learn the same thing). Fix: add load-balancing auxiliary loss.

---

## Idea 10: Stochastic Weight Averaging with Cosine Annealing Restarts (SWALR)

**What it is.** SWALR (SWA + cyclic LR restarts) runs multiple short cosine annealing cycles during the second half of training, averaging the weights at each cycle minimum. Unlike the current single cosine schedule (T_max=150), this gives multiple weight snapshots at low-LR convergence points and averages them, which provably flattens the loss basin.

**Why it might help here.** PR #2344 (in-flight) tests basic SWA. SWALR is the more principled extension: instead of a fixed constant LR for SWA collection, it uses mini-cycles (T_swa=10-20 epochs each) to explore a larger region of the loss surface before averaging. The theoretical result (Izmailov et al., 2018) is that SWA finds wider minima; SWALR explores those minima more systematically. This is different from the in-flight PR #2344 which likely implements basic SWA.

**Literature grounding.**
- "Stochastic Weight Averaging" (Izmailov et al., 2018, UAI): foundational SWA paper.
- "Loss of Plasticity in Deep Continual Learning" (ICML 2024): shows SWA-like averaging helps in non-stationary settings similar to curriculum learning.
- "Averaging Weights Leads to Wider Optima and Better Generalization" (Izmailov et al., 2018).

**Implementation complexity.** ~40-60 LoC. After epoch `swa_start` (=75), begin cyclic LR with T_swa=15 epochs using `CosineAnnealingLR(optimizer, T_max=15, eta_min=lr*0.1)`. Accumulate SWA model at each cycle end. Average at training end. Use `torch.optim.swa_utils.AveragedModel` and `update_bn()`. Gate with `--swalr --swalr_T 15 --swa_start 75` flags. This is complementary to (and different from) whatever PR #2344 implements.

**Metrics targeted.** p_oodc, p_re (the OOD metrics that most benefit from wider optima).

**Suggested experiment.** Only assign if PR #2344 results are negative or inconclusive. If #2344 tests basic SWA, this tests the cyclic-LR variant. Flag: `--swalr --swalr_T 15 --swa_start 75`. Compare against PR #2319 baseline.

**Confidence.** Moderate. SWA is well-validated; the question is whether the 30-minute training budget leaves enough cycles for meaningful weight averaging. With T_swa=15 and ~75 epochs remaining after swa_start, we get ~5 averaging snapshots — the minimum needed for SWA to work.

---

## Idea 11: Log-Re-Conditioned Panel Cp (Physics-Adaptive Inviscid Correction)

**What it is.** The current panel Cp uses only the angle of attack (AoA) from flat-plate theory. In reality, as Reynolds number changes, the boundary layer thickness and stagnation point location change, modifying the effective pressure distribution. Add log(Re) as a conditioning input to a learned scaling of the panel Cp: Cp_effective = Cp_panel * σ(MLP(log_Re, AoA)), where MLP is a tiny 2-layer network (2→16→1) that learns to re-weight the physics hint based on flow regime.

**Why it might help here.** The cp_panel feature improved p_tan and p_oodc but caused p_re regression (+1.0%). This suggests the flat-plate Cp (which ignores Re effects) is misleading the model at extreme Reynolds numbers — at high Re, the inviscid approximation is better (thinner boundary layer); at low Re, viscous effects dominate and Cp deviates more from inviscid theory. A Re-conditioned scale allows the model to downweight the physics hint at low Re (where it's least accurate) and upweight it at high Re. This directly addresses the p_re regression introduced by PR #2319.

**Literature grounding.**
- Viscous-inviscid interaction theory (Drela, "XFOIL: An analysis and design system for low Reynolds number airfoils", 1989): documents how Re changes the departure of real Cp from inviscid theory.
- FiLM conditioning (Perez et al., 2018): the same mechanism as Idea 2 but applied to the input feature rather than the SRF head.

**Implementation complexity.** ~30-40 LoC. Replace `x[:, cp_idx] *= cp_panel_scale` with `x[:, cp_idx] *= self.re_cp_scale(log_re) * cp_panel_scale` where `re_cp_scale` is a `nn.Sequential(Linear(1,16), GELU, Linear(16,1), Sigmoid)` network that outputs a multiplier in (0,1). Gate with `--cp_panel_re_cond` flag. Initialize output layer bias to sigmoid^{-1}(1.0)=inf so it starts at 1.0 (identity).

**Metrics targeted.** p_re (fix the regression from PR #2319), p_oodc.

**Suggested experiment.** Add `--cp_panel_re_cond` flag. The tiny MLP is jointly trained; monitor the learned Re→scale mapping via W&B (log the scale at Re_min, Re_max training values). Compare against PR #2319 to see if p_re regression is recovered while maintaining other metric gains.

**Confidence.** Moderate-high. The hypothesis is specific: the p_re regression in PR #2319 is caused by the fixed-scale Cp being inaccurate at extreme Re. If true, Re-conditioning should restore p_re while preserving the tandem gains. If false (i.e., the regression is random seed noise), this will have neutral effect.

---

## Idea 12: Arc-Length Positional Encoding on Surface Nodes

**What it is.** For the ~5K surface nodes, compute the cumulative arc-length distance along each airfoil's surface contour from the leading edge. Use this as a 1D positional embedding (sinusoidal or learned) appended to the surface node input features. Unlike Cartesian coordinates, arc-length is intrinsic to the surface geometry and invariant to rigid-body rotations.

**Why it might help here.** The surface nodes are processed by both the backbone (globally) and the SRF head (as surface-only). Neither has an explicit notion of the sequential ordering along the surface contour. Arc-length encodes "how far along the surface" a node is — the leading edge (arc=0), upper surface, trailing edge (arc=0.5·chord·perimeter), lower surface. Pressure distribution has a characteristic shape as a function of arc-length (suction peak near LE, recovery toward TE) that the model must currently infer from Cartesian x,y. Explicit arc-length encoding makes this implicit ordering explicit.

**Literature grounding.**
- Surface parametrization in geometric deep learning: arc-length is the canonical 1D coordinate for curve/surface learning.
- "Predicting airfoil pressure distribution using boundary graph neural networks" (arXiv:2503.18638, Mar 2025): uses boundary-aligned coordinates; arc-length is the natural 1D version.
- GeoMPNN (arXiv:2412.09399): geometry-aware features for mesh learning.

**Implementation complexity.** ~50-70 LoC. For each sample, identify the surface node sequence (boundary_id=6,7), compute pairwise distances between adjacent nodes ordered by angle from centroid, accumulate to get arc-length s ∈ [0, perimeter]. Normalize by chord length. Append sin(2πk·s/L) and cos(2πk·s/L) for k=1..4 (8 channels) as additional features for surface nodes; zero for volume nodes. Gate with `--arc_length_encoding --arc_length_freqs 4` flag.

**Metrics targeted.** p_in, p_oodc, p_tan — all surface pressure metrics.

**Suggested experiment.** Add `--arc_length_encoding` flag. Use 4 frequency pairs (8 additional channels). For non-surface nodes, set these to zero. Compare against PR #2319 baseline.

**Confidence.** Moderate. The arc-length feature is physically well-motivated and untried in this codebase. Risk: the DSDF (distance-to-surface) features already give an approximate arc-length proxy; the true arc-length may be redundant if DSDF is computed correctly.

---

## Idea 13: Pressure Recovery Ratio as Input Feature

**What it is.** For each mesh node in the volume, compute its location relative to the "pressure recovery region" between the two foils and downstream of both trailing edges. Specifically: given the freestream dynamic pressure q∞ = ½ρU∞², compute a dimensionless "recovery potential" r = (x - x_fore_TE) / (x_aft_LE - x_fore_TE), clamped to [0,1] in the inter-foil gap and 0/1 outside. This feature encodes how far through the inter-foil pressure recovery the node is located.

**Why it might help here.** The inter-foil gap is the region of highest physics complexity in tandem configurations: the fore-foil wake mixes with the aft-foil leading-edge stagnation, creating a complex pressure recovery that drives the p_tan error. The current wake deficit feature encodes distance to the fore trailing edge, but not the normalized position within the recovery region. A recovery ratio r ∈ [0,1] explicitly tells the model "this node is 30% through the inter-foil gap" — a continuous label for the recovery stage.

**Literature grounding.**
- "Direct numerical simulation of aerodynamic interactions of tandem wings" (ScienceOpen, 2024): documents the inter-foil pressure recovery as the primary source of prediction error in tandem CFD surrogates.
- Tandem airfoil aerodynamics theory: the inter-foil pressure recovery region has known analytical structure (exponential recovery with decay constant ~gap/chord).

**Implementation complexity.** ~30-40 LoC. Compute fore_TE_x from the surface nodes (max x of boundary_id=6). Compute aft_LE_x (min x of boundary_id=7). For each node: r = clamp((x - fore_TE_x) / (aft_LE_x - fore_TE_x + 1e-6), 0, 1). Append as 1 additional channel; zero for single-foil samples. Gate with `--pressure_recovery_feature`.

**Metrics targeted.** p_tan primarily (the inter-foil gap pressure recovery is the dominant source of tandem-specific error).

**Suggested experiment.** Add `--pressure_recovery_feature` flag. This is a single scalar channel per node with near-zero implementation cost. Monitor via W&B whether p_tan improves. If positive, extend with sin/cos encoding.

**Confidence.** Moderate. The feature is domain-specific and well-motivated for p_tan. Risk: the existing gap_stagger_spatial_bias and wake_deficit_feature may already encode this implicitly; the marginal gain could be small.

---

## Idea 14: Global Aerodynamic State Embedding (Cl/Cd as Intermediate Representation)

**What it is.** Add an auxiliary head that predicts global aerodynamic coefficients (Cl, Cd, Cm) as intermediate representations, computed by integrating the predicted pressure field over the airfoil surface. The embedding from the global prediction head is then concatenated to the SRF input — giving the local surface correction access to global aerodynamic context. Unlike a standalone auxiliary Cl loss (which has been tried), this uses the Cl/Cd *embedding vector* (not just the loss) as a conditioning signal.

**Why it might help here.** The SRF correction is local — it corrects node-by-node. But lift and drag are integral constraints: ΣCp·n·dA = Cl, Cd. A network that can predict Cl/Cd embeds global self-consistency information that is fundamentally different from local pressure correction. Feeding the Cl/Cd embedding into the SRF allows the local corrections to be globally consistent — a property the current architecture cannot guarantee. This is distinct from PR #2315 (global aerodynamic state embedding, in-flight as #2347) — check the exact implementation: if #2347 adds Cl as an auxiliary loss only, this idea adds the embedding as an SRF conditioning signal.

**Literature grounding.**
- "Global-information-enhanced GNN for physical field reconstruction" (Springer, Jul 2025): shows global physical quantities as conditioning signals improve local field predictions.
- Neural operators for integral constraints: standard in physics-informed ML (e.g., "Physics-Informed Neural Operators", ICLR 2023).

**Implementation complexity.** ~70-90 LoC. Add a GlobalAeroHead that reduces surface nodes to (Cl, Cd, Cm) via pressure-weighted integration. Feed the hidden state of this head as a 16-dim embedding into SRF input. Use stop-gradient on the embedding path to prevent the global head from destabilizing local predictions. Gate with `--global_aero_embedding` flag.

**Metrics targeted.** p_in, p_oodc, p_tan — all surface pressure metrics that benefit from integral consistency.

**Suggested experiment.** Only assign if PR #2347 uses a fundamentally different mechanism (auxiliary loss only). Add `--global_aero_embedding` flag. Use a 16-dim embedding. Stop-gradient between the global head and the SRF conditioning path. Monitor Cl/Cd prediction accuracy separately in W&B.

**Confidence.** Moderate. The mechanism is different from auxiliary loss — it's about embedding reuse, not gradient shaping. Risk: overlap with #2347 if that PR already implements embedding conditioning.

---

## Idea 15: Input Consistency Regularization (Dropout-Based Self-Distillation)

**What it is.** During training, forward each sample twice: once with standard dropout and once with a different dropout mask (or with a small Gaussian noise perturbation to the input features). Add a consistency loss: MSE between the two predictions. This forces the model to produce stable predictions regardless of stochastic activation patterns — a form of self-distillation that is orthogonal to target noise (which was PR #2332).

**Why it might help here.** The current model uses EMA (decay=0.999) for weight averaging, which improves test-time stability. Input consistency regularization attacks a different source of instability: the stochasticity introduced by dropout in the model itself. By penalizing prediction inconsistency across two forward passes with different dropout masks, the model is pushed toward more robust internal representations. This is closely related to R-Drop (NeurIPS 2021) and π-model (temporal ensembling for semi-supervised learning), both of which showed consistent gains for robust generalization.

**Literature grounding.**
- R-Drop: "R-Drop: Regularized Dropout for Neural Networks" (NeurIPS 2021): consistency loss between two dropout forward passes, consistent gains across diverse tasks.
- Mean Teacher / π-model (Tarvainen & Valpola, NeurIPS 2017): consistency regularization between augmented views for semi-supervised learning.
- Distinct from target noise (PR #2332): that adds noise to y; this adds noise to the forward pass stochasticity.

**Implementation complexity.** ~30-40 LoC. After computing the main loss, run a second forward pass (with train mode / different dropout mask active), compute consistency_loss = MSE(pred1, pred2.detach()) on surface nodes only, add λ·consistency_loss to total loss. Set λ=0.1. Gate with `--input_consistency_reg --consistency_weight 0.1` flag. Note: this doubles forward pass cost, similar to PCGrad overhead — check VRAM budget (~45GB currently).

**Metrics targeted.** p_oodc, p_re — OOD generalization metrics that benefit most from robust internal representations.

**Suggested experiment.** Add `--input_consistency_reg --consistency_weight 0.1` flag. The second forward pass shares the same batch but uses a different dropout mask (just calling model.forward twice in train mode achieves this). Apply consistency loss to surface node predictions only to target the most important metric. Compare against PR #2319.

**Confidence.** Moderate. R-Drop is well-validated on diverse tasks; the extension to mesh regression is straightforward. Main uncertainty: the current model may use very little dropout (check train.py), making the consistency loss degenerate (identical predictions from two forward passes). Verify dropout rate in the architecture before implementing.

---

## Summary Table

| # | Idea | Complexity (LoC) | Primary Target | Confidence | Novelty vs. Existing PRs |
|---|------|-----------------|----------------|------------|--------------------------|
| 1 | Joukowski camber-corrected Cp | 40-60 | p_in, p_tan | Moderate | New (extends #2319) |
| 2 | Re-conditioned SRF (FiLM) | 50-70 | p_re, p_oodc | Moderate-high | New |
| 3 | Vortex-panel induced velocity | 80-100 | p_in, p_oodc, p_tan | Moderate | New |
| 4 | Spectral regularization on FFN | 30-40 | p_re, p_oodc | Moderate | New |
| 5 | Two-stage SRF (vel→pressure) | 60-80 | p_in, p_tan | Moderate | New |
| 6 | Wake angle feature (atan2) | 15-20 | p_in, p_tan | Moderate | New (extends #2213) |
| 7 | Surface-normal coordinate frame | 50-70 | p_in, p_oodc | Moderate | New (different from TE frame) |
| 8 | Learnable cp_panel scale | 20-30 | p_in, p_oodc, p_tan | Moderate-high | New (extends #2319) |
| 9 | MoE domain-expert FFN | 80-100 | p_tan | Moderate | New |
| 10 | SWALR cyclic averaging | 40-60 | p_oodc, p_re | Moderate | Distinct from #2344 |
| 11 | Log-Re-conditioned panel Cp | 30-40 | p_re | Moderate-high | New (fixes #2319 regression) |
| 12 | Arc-length positional encoding | 50-70 | p_in, p_oodc | Moderate | New |
| 13 | Pressure recovery ratio feature | 30-40 | p_tan | Moderate | New |
| 14 | Global aero embedding for SRF | 70-90 | p_in, p_tan | Moderate | Distinct from #2347 |
| 15 | Input consistency regularization | 30-40 | p_oodc, p_re | Moderate | New (distinct from #2332) |

**Top priority for immediate assignment:**
1. Idea 11 (Log-Re-conditioned Cp): directly addresses the p_re regression from the last merged PR — highest specificity hypothesis.
2. Idea 6 (Wake angle feature): cheapest change with clear physical motivation.
3. Idea 2 (Re-conditioned SRF via FiLM): targets p_re with a well-validated mechanism.
4. Idea 8 (Learnable cp_panel scale): low complexity, removes a hyperparameter.
5. Idea 3 (Vortex-panel induced velocity): highest potential upside if the inviscid oracle generalizes.
