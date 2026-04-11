<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# Round 48 Research Ideas — BOLD & RADICAL

Generated: 2026-04-11
Mandate: Radical new model families, synthetic data generation, physics-grounded augmentation, and problem reformulation. NOT incremental ANP tweaks.

ALL experiments must include `--anp_srf`.

Current baseline (PR #2379, 2-seed avg):
- p_in=3.561, p_oodc=3.847, p_tan=10.825, p_re=7.232
- val_loss=0.5426

Top priority: p_re (7.232, +16% regression vs pre-ANP baseline). Secondary: p_tan.

---

## Idea 1: PIDA Chord-Scaling Augmentation

**Bold claim**: Reynolds-similarity-exact chord scaling will directly fix the p_re regression by teaching the model the physics of Reynolds number variation through exact kinematic similarity, at zero additional data cost.

**Mechanism**: For a chord-scaled sample, multiply the chord length by a random scale factor s ~ Uniform(0.5, 2.0). By kinematic similarity (Re = V·c/ν), this is equivalent to changing Re by the same factor s while holding Ux, Uy, and Cp (pressure coefficient) exactly constant — the physics is exact, not approximate. The model sees "Re=5e5 with c=0.2" and "Re=1e6 with c=0.4" as variations of the same physical flow, which teaches Re-invariant representations grounded in dimensional analysis rather than memorization. This is categorically different from Re interpolation because it creates genuine Reynolds similarity pairs — the augmented sample IS physically correct at the scaled Reynolds number. Apply to 50% of training batches. This directly addresses p_re because the model's current failure mode is almost certainly a lack of Re-covariant representations learned from sparse Re coverage in training data.

**Target metrics**: p_re (primary), p_in, p_oodc

**Implementation**:
```bash
cd cfd_tandemfoil && python train.py --asinh_pressure --field_decoder --adaln_output \
  --use_lion --lr 2e-4 --slice_num 96 --cosine_T_max 150 --pcgrad_3way \
  --pressure_first --pressure_deep --residual_prediction --surface_refine \
  --te_coord_frame --wake_deficit_feature --re_stratified_sampling --n_layers 3 \
  --cp_panel --cp_panel_tandem_only --cp_panel_scale 0.1 --wake_angle_feature \
  --vortex_panel_velocity --vortex_panel_scale 0.1 --vortex_panel_n 64 --anp_srf \
  --pida_chord_aug --pida_aug_prob 0.5 --pida_scale_min 0.5 --pida_scale_max 2.0
```

**Confidence**: High. Reynolds similarity is exact physics, not an approximation. The mechanism directly targets the feature space where p_re is known to fail. DeepAirfoil (AIAA 2024) and NeuralFoil both use chord-scaling as a core augmentation strategy. The only risk is implementation fidelity — if the Re feature in the input is updated consistently with the scaling, this should work.

**Risk**: Medium-low. The student must ensure that when chord is scaled, the Re input feature is also scaled (Re *= s), and that all geometry coordinates are scaled consistently. If Re feature is not updated, this becomes misleading rather than helpful.

---

## Idea 2: Flip-Symmetry Augmentation (Negative AoA Mirror)

**Bold claim**: Reflecting samples across the chord axis (negating AoA and Uy, preserving Ux and p structure) doubles the effective training set with physically exact labels and forces learning of a symmetry-respecting representation that generalizes better to unseen conditions.

**Mechanism**: For any single-foil sample at AoA=α, the physically correct flow at AoA=-α is obtained by flipping y-coordinates, negating Uy, and negating AoA in the input features — Ux and pressure (and Cp) remain correct by symmetry. This is not an approximation for symmetric airfoils; for cambered airfoils it is only approximate, but even approximate symmetry is a useful inductive bias. Apply with probability 0.5 to all non-tandem samples. This regularizes the model against overfitting to the sign of AoA, which likely accounts for some OOD generalization failures. NeuralFoil (which failed as a feature source in #2275) uses this augmentation as its primary strategy for AoA generalization — the augmentation itself may work even though NeuralFoil predictions didn't. For tandem samples, symmetry is more complex (both foils must be flipped) but could still be applied.

**Target metrics**: p_oodc (primary), p_in, p_re

**Implementation**:
```bash
cd cfd_tandemfoil && python train.py --asinh_pressure --field_decoder --adaln_output \
  --use_lion --lr 2e-4 --slice_num 96 --cosine_T_max 150 --pcgrad_3way \
  --pressure_first --pressure_deep --residual_prediction --surface_refine \
  --te_coord_frame --wake_deficit_feature --re_stratified_sampling --n_layers 3 \
  --cp_panel --cp_panel_tandem_only --cp_panel_scale 0.1 --wake_angle_feature \
  --vortex_panel_velocity --vortex_panel_scale 0.1 --vortex_panel_n 64 --anp_srf \
  --flip_symmetry_aug --flip_aug_prob 0.5
```

**Confidence**: High. Symmetry augmentation is among the most validated techniques in aerodynamics ML. This is pure data augmentation — no architectural risk. XFOIL (which computes symmetric airfoils analytically) and virtually every aerodynamics ML paper uses some form of this. The failure mode is only if cambered airfoil asymmetry causes the approximately-flipped labels to hurt rather than help, which can be diagnosed by checking if the effect is positive on symmetric-airfoil splits.

**Risk**: Low. Pure data augmentation with no side effects. Clear failure diagnosis available.

---

## Idea 3: Massive Multi-Fidelity Panel-Method Pretraining

**Bold claim**: Pretraining the Transolver backbone on 100k+ panel-method solutions (free to generate) before fine-tuning on CFD data will teach Re-dependent flow structures at a scale the CFD dataset alone cannot support, directly addressing the sparse Re coverage that causes p_re failures.

**Mechanism**: Generate 100k–500k panel-method solutions using the existing vortex-panel solver already in the codebase (it's already integrated as a feature source). These solutions are inexact (inviscid) but capture the dominant pressure distribution, circulation, and Reynolds-number trends. Use them to pretrain the full Transolver backbone with a standard MSE loss, then fine-tune on the real CFD training set. This is the DeepAirfoil (AIAA 2024) strategy: 6.17M XFOIL + 575K HAM2D CFD. The key insight is that even imperfect physics-based pretraining provides a rich initialization that the sparse CFD dataset can then correct. Unlike using panel-method predictions as input features (which failed in #2275 via NeuralFoil), this uses panel solutions as pretraining TARGETS — the model learns to predict panel flows, then adapts to predict CFD flows. The residual to learn from CFD is then just the viscous/separation correction rather than the full flow structure.

**Target metrics**: p_re (primary), p_tan, p_oodc

**Implementation**:
```bash
# Phase 1: Generate panel-method pretraining dataset (student implements)
# Phase 2: Pretrain
cd cfd_tandemfoil && python train.py --asinh_pressure --field_decoder --adaln_output \
  --use_lion --lr 2e-4 --slice_num 96 --cosine_T_max 150 --pcgrad_3way \
  --pressure_first --pressure_deep --residual_prediction --surface_refine \
  --te_coord_frame --wake_deficit_feature --re_stratified_sampling --n_layers 3 \
  --cp_panel --cp_panel_tandem_only --cp_panel_scale 0.1 --wake_angle_feature \
  --vortex_panel_velocity --vortex_panel_scale 0.1 --vortex_panel_n 64 --anp_srf \
  --panel_pretrain --panel_pretrain_samples 100000 --panel_pretrain_epochs 50
```

**Confidence**: Medium-High. The mechanism is well-validated by DeepAirfoil and the general pretraining literature. The risk is engineering complexity — the student needs to generate the panel dataset and implement the two-phase training loop. But the panel solver is already in the codebase, so data generation is relatively straightforward.

**Risk**: Medium. Implementation complexity is high. The panel-method solutions use the same vortex-panel approximation that's already in the feature set — if the model has already internalized those features, the pretraining signal may be redundant. Monitor whether pretraining loss decreases meaningfully before committing to fine-tuning.

---

## Idea 4: Flow Matching Generative Surface Head

**Bold claim**: Replacing the deterministic surface decoder with a flow matching (rectified flow) generative head will produce sharper, more calibrated surface pressure predictions by learning the full distribution over Cp rather than its conditional mean, achieving better extrema prediction at leading edge and trailing edge where MAE is highest.

**Mechanism**: PR #2271 proposed this but NEVER RAN. Now we have the ANP backbone (#2379) as a strong conditioning signal. The architecture: the ANP encoder produces surface context tokens; a 3-block DiT (Diffusion Transformer, d=128, 4 heads) denoises a noisy surface field conditioned on those tokens using flow matching (straight-line ODE trajectories from noise to signal). Inference uses 4-step Euler integration. This avoids the blurriness inherent in MSE-trained deterministic decoders — flow matching heads produce sharper, more physically plausible predictions at discontinuities (suction peaks, TE pressure recovery). The conditioning mechanism is straightforward: concatenate ANP context embeddings to the DiT's cross-attention keys/values. Use rectified flow (Lipman et al. 2022) rather than DDPM for speed and stability.

**Target metrics**: p_in (primary — suction peak accuracy), p_oodc, p_re

**Implementation**:
```bash
cd cfd_tandemfoil && python train.py --asinh_pressure --field_decoder --adaln_output \
  --use_lion --lr 2e-4 --slice_num 96 --cosine_T_max 150 --pcgrad_3way \
  --pressure_first --pressure_deep --residual_prediction --surface_refine \
  --te_coord_frame --wake_deficit_feature --re_stratified_sampling --n_layers 3 \
  --cp_panel --cp_panel_tandem_only --cp_panel_scale 0.1 --wake_angle_feature \
  --vortex_panel_velocity --vortex_panel_scale 0.1 --vortex_panel_n 64 --anp_srf \
  --flow_matching_head --flow_matching_steps 4 --flow_matching_hidden 128 \
  --flow_matching_blocks 3
```

**Confidence**: Medium. The theory is strong (flow matching consistently outperforms DDPM in similar settings), but CFD surface pressure has sharp gradients that may be hard to denoise accurately in 4 steps. The PR #2271 body contains the prior design — review it carefully before implementing. Key hyperparameter: the number of flow matching steps during inference (try 4, 8, 16).

**Risk**: Medium-High. Training a generative head alongside a regression backbone requires careful loss balancing. The flow matching loss must be weighted relative to the volume MAE loss. Recommend lambda_fm=0.5 as a starting point, with the backbone frozen for the first 20 epochs.

---

## Idea 5: AeroDiT Diffusion Head with Reynolds Classifier-Free Guidance

**Bold claim**: A diffusion-based surface field decoder with explicit Re-conditioning via classifier-free guidance (CFG) will recover the p_re regression by treating different Reynolds regimes as distinct distributions and using guidance to sharpen Re-specific predictions at inference time.

**Mechanism**: Prior attempt #2349 failed because it lacked ANP conditioning and Re-CFG. The new design: (1) ANP encoder provides surface context embeddings; (2) a DiT surface decoder (3 blocks, d=128) is conditioned on both ANP tokens AND a Re embedding; (3) during training, 10% of samples drop the Re conditioning (replaced with a null embedding); (4) at inference, use CFG: epsilon_guided = epsilon_uncond + gamma * (epsilon_cond - epsilon_uncond) with gamma=3.0. The Re-specific guidance dramatically sharpens predictions for high-Re and low-Re regimes by amplifying the Re-signal direction in the latent space. Use DDIM with 4 steps for fast inference. Key difference from #2349: the ANP backbone provides a strong geometry-and-flow-structure prior; the diffusion head only needs to predict residuals on top of that. This is much easier than predicting the full flow from scratch.

**Target metrics**: p_re (primary), p_oodc

**Implementation**:
```bash
cd cfd_tandemfoil && python train.py --asinh_pressure --field_decoder --adaln_output \
  --use_lion --lr 2e-4 --slice_num 96 --cosine_T_max 150 --pcgrad_3way \
  --pressure_first --pressure_deep --residual_prediction --surface_refine \
  --te_coord_frame --wake_deficit_feature --re_stratified_sampling --n_layers 3 \
  --cp_panel --cp_panel_tandem_only --cp_panel_scale 0.1 --wake_angle_feature \
  --vortex_panel_velocity --vortex_panel_scale 0.1 --vortex_panel_n 64 --anp_srf \
  --diffusion_head --diffusion_steps 4 --diffusion_cfg_scale 3.0 \
  --diffusion_hidden 128 --diffusion_blocks 3 --diffusion_re_dropout 0.1
```

**Confidence**: Medium. AeroDiT (NeurIPS 2024 workshop) demonstrated this mechanism works for RANS prediction with explicit Re conditioning. The CFG mechanism for Re is well-motivated: the p_re regression suggests the model conflates different Re regimes, and CFG forces explicit Re-conditioned sampling. The risk is training instability with a diffusion head on top of a regression backbone.

**Risk**: Medium. CFG scale gamma=3.0 is a strong starting point but may require tuning. If the diffusion head trains stably but CFG doesn't improve over gamma=1.0, the issue is elsewhere. Recommend ablating with gamma in [1.0, 2.0, 3.0, 5.0] and reporting all.

---

## Idea 6: GeoMPNN — Bipartite Surface-to-Volume Graph Message Passing

**Bold claim**: Replacing the ANP volume decoder with a bipartite message-passing network (surface nodes as sources, volume nodes as targets) will improve volume MAE and indirectly surface MAE by enforcing geometric consistency between surface boundary conditions and interior flow via explicit graph structure.

**Mechanism**: ML4CFD 2024 4th-place solution used bipartite graph neural networks to propagate boundary information into the volume. Architecture: build a k-NN bipartite graph where each volume node receives messages from its k=8 nearest surface nodes; edge features encode [Euclidean distance, wall-normal dot product, log(distance), estimated BL thickness from Re and chord]; 2 rounds of message passing update volume node embeddings before the final MLP decoder. The key advantage over the current volume decoder (which ignores the surface→volume geometric relationship) is that boundary conditions are enforced by construction: a high-pressure stagnation point on the surface explicitly propagates to the near-wall volume region. This is particularly relevant for p_tan (tandem wake) where the downstream foil receives flow that originated on the upstream foil's surface.

**Target metrics**: p_tan (primary), p_in

**Implementation**:
```bash
cd cfd_tandemfoil && python train.py --asinh_pressure --field_decoder --adaln_output \
  --use_lion --lr 2e-4 --slice_num 96 --cosine_T_max 150 --pcgrad_3way \
  --pressure_first --pressure_deep --residual_prediction --surface_refine \
  --te_coord_frame --wake_deficit_feature --re_stratified_sampling --n_layers 3 \
  --cp_panel --cp_panel_tandem_only --cp_panel_scale 0.1 --wake_angle_feature \
  --vortex_panel_velocity --vortex_panel_scale 0.1 --vortex_panel_n 64 --anp_srf \
  --bipartite_mpnn --bipartite_k 8 --bipartite_rounds 2 \
  --bipartite_edge_features dist,wall_normal,log_dist,bl_thickness
```

**Confidence**: Medium. The ML4CFD competition result validates this for a related task (3D CFD). The extension to our irregular mesh with tandem geometry is non-trivial but the mechanism is clear. The main uncertainty is whether 2 rounds of message passing are sufficient for the long-range wake effects in tandem configurations.

**Risk**: Medium. k-NN graph construction at training time adds preprocessing overhead. The student must ensure the bipartite graph is recomputed per sample (geometries vary). Memory overhead with k=8 and ~10k volume nodes is manageable. Try k=4 first if memory is an issue.

---

## Idea 7: Divergence-Free Hard Constraint via Helmholtz Projection

**Bold claim**: Projecting predicted velocity fields onto the divergence-free subspace (∇·u=0) as a differentiable post-processing step will improve surface pressure consistency by enforcing incompressible flow physics that the neural network currently violates, creating a hard constraint that cannot be circumvented by gradient descent.

**Mechanism**: The Helmholtz decomposition states any vector field u = u_div_free + ∇φ. For incompressible flow, u_div_free is the physically correct component. After the network predicts (Ux, Uy), solve a discrete Poisson equation ∇²φ = ∇·u_pred on the mesh, then correct: u_corrected = u_pred - ∇φ. This is exact (not approximate) for the discrete mesh. The correction layer is differentiable — gradients flow through the Poisson solve (using conjugate gradient or a learned Poisson preconditioner). The surface pressure is intimately connected to velocity divergence via the pressure Poisson equation (∇²p = -ρ ∇u:∇u^T), so reducing divergence error directly reduces pressure error. This addresses a fundamental physics violation that MSE training cannot prevent.

**Target metrics**: p_in (primary — highest surface pressure accuracy stakes), p_oodc

**Implementation**:
```bash
cd cfd_tandemfoil && python train.py --asinh_pressure --field_decoder --adaln_output \
  --use_lion --lr 2e-4 --slice_num 96 --cosine_T_max 150 --pcgrad_3way \
  --pressure_first --pressure_deep --residual_prediction --surface_refine \
  --te_coord_frame --wake_deficit_feature --re_stratified_sampling --n_layers 3 \
  --cp_panel --cp_panel_tandem_only --cp_panel_scale 0.1 --wake_angle_feature \
  --vortex_panel_velocity --vortex_panel_scale 0.1 --vortex_panel_n 64 --anp_srf \
  --div_free_constraint --div_free_weight 0.1 --div_free_solver cg --div_free_iters 10
```

**Confidence**: Medium. Helmholtz projection is a classical numerical method; differentiable versions have been validated in physics-informed ML (e.g., Jacobson et al. 2023). The key uncertainty is whether CFD meshes (unstructured, irregular) support efficient discrete divergence computation. If the mesh has the right connectivity structure, this is straightforward. If not, the student may need to use a GNN-based Poisson solver.

**Risk**: Medium-High. Unstructured mesh Poisson solves are non-trivial to implement efficiently. The div_free_weight=0.1 is a soft constraint — consider starting with a penalty formulation before attempting a hard Helmholtz projection. If the Poisson solve fails to converge within 10 CG iterations, fall back to the soft penalty form.

---

## Idea 8: Multi-Resolution ANP Context (Hierarchical Arc-Zone Tokens)

**Bold claim**: Replacing the ANP's flat surface token set with a hierarchical 3-level structure (coarse zone tokens + fine surface nodes + micro TE tokens) will improve p_re and p_tan by giving the cross-attention mechanism explicit access to multi-scale surface structure — global pressure distribution (C_L-scale), local suction peak shape, and TE boundary condition simultaneously.

**Mechanism**: The current ANP SRF encodes all surface nodes at equal resolution. But aerodynamic surfaces have a natural hierarchy: global Cp distribution (10-20 zone tokens), local suction peak and LE stagnation (50-100 tokens), TE recovery and Kutta condition (5-10 tokens at the trailing edge). The hierarchical design: pool surface nodes into Z=16 arc-length zones (max-pool of node embeddings within each zone), creating coarse zone tokens; keep the full surface node tokens; add 5 dedicated TE tokens from the final 5% chord. The decoder then attends to all three levels simultaneously via multi-head cross-attention with level-specific learned positional encodings. This is analogous to the AB-UPT "anchor token" idea but applied specifically to the surface representation. The re-weighting of different scales directly addresses p_re: coarse zone tokens encode global Re effects while fine tokens encode local viscous corrections.

**Target metrics**: p_re (primary), p_tan, p_in

**Implementation**:
```bash
cd cfd_tandemfoil && python train.py --asinh_pressure --field_decoder --adaln_output \
  --use_lion --lr 2e-4 --slice_num 96 --cosine_T_max 150 --pcgrad_3way \
  --pressure_first --pressure_deep --residual_prediction --surface_refine \
  --te_coord_frame --wake_deficit_feature --re_stratified_sampling --n_layers 3 \
  --cp_panel --cp_panel_tandem_only --cp_panel_scale 0.1 --wake_angle_feature \
  --vortex_panel_velocity --vortex_panel_scale 0.1 --vortex_panel_n 64 --anp_srf \
  --anp_hierarchical --anp_zone_tokens 16 --anp_te_tokens 5
```

**Confidence**: Medium. Hierarchical attention is well-validated in vision transformers (multi-scale ViT, Swin) and protein structure prediction (AlphaFold2's multi-track attention). The mapping to surface arc-length zones is natural. The risk is increased attention complexity and potential for the coarse tokens to dominate the fine tokens.

**Risk**: Medium. The student must implement hierarchical pooling carefully — the gradient must flow from coarse tokens back to the original surface nodes. Max-pooling is not differentiable at argmax; use soft (log-sum-exp) pooling or average pooling. Alternatively, add a lightweight 1D convolutional striding layer over arc-length-sorted nodes to produce multi-scale tokens.

---

## Idea 9: Tandem Wake Superposition Prior (Panel-Initialized Decoder)

**Bold claim**: Initializing the ANP tandem decoder with a panel-method Cp prior — specifically, computing the superposition of the two foils' panel solutions as an initial state before cross-attention refinement — will dramatically improve p_tan by providing the correct first-order wake interaction physics as a starting point, leaving only the CFD correction to be learned.

**Mechanism**: For tandem configurations, the ANP cross-attention must discover wake interactions from scratch. The panel method already provides an inviscid approximation of how upstream foil wake affects downstream foil Cp via vortex superposition — this is exactly the physics our network struggles with (p_tan=10.825). Design: (1) compute panel-method Cp for each foil independently (already available as cp_panel features); (2) compute the induced velocity at downstream foil from upstream foil's vortex sheet (using the existing vortex_panel_velocity feature, already in the baseline); (3) use these as the initial query embeddings in the ANP tandem decoder, rather than learning them from scratch. The cross-attention then only needs to predict the viscous correction δCp = Cp_CFD - Cp_panel_superposition. This is a form of the "residual prediction" idea (#2392 DeltaPhi is currently in-flight) but applied specifically to the tandem wake interaction rather than the single-foil viscous correction.

**Target metrics**: p_tan (primary), p_oodc

**Implementation**:
```bash
cd cfd_tandemfoil && python train.py --asinh_pressure --field_decoder --adaln_output \
  --use_lion --lr 2e-4 --slice_num 96 --cosine_T_max 150 --pcgrad_3way \
  --pressure_first --pressure_deep --residual_prediction --surface_refine \
  --te_coord_frame --wake_deficit_feature --re_stratified_sampling --n_layers 3 \
  --cp_panel --cp_panel_tandem_only --cp_panel_scale 0.1 --wake_angle_feature \
  --vortex_panel_velocity --vortex_panel_scale 0.1 --vortex_panel_n 64 --anp_srf \
  --tandem_wake_prior --tandem_prior_scale 1.0
```

**Confidence**: Medium-High. The mechanism is directly motivated by the physics of the failure mode (p_tan is worst). The vortex_panel_velocity feature is already computed and proven useful (merged in #2357). Using it as a decoder prior rather than just an encoder input feature is a natural extension. The main risk is that the panel method is inviscid and separation effects dominate in the tandem configuration — the correction to be learned may be larger than the prior itself.

**Risk**: Medium. The student must carefully implement the initialization of decoder queries from panel features without breaking gradient flow. The panel prior should be applied as an additive bias to the initial query embeddings, not as a hard initialization (which would slow early convergence).

---

## Idea 10: Displacement Thickness delta* Feature (Thwaites BL Method)

**Bold claim**: Computing the boundary-layer displacement thickness δ*(s; Re, AoA) analytically via Thwaites' integral method for each surface node and adding it as an input feature will provide the model with a Re-dependent viscous correction signal that the current inviscid panel features cannot capture, directly addressing the p_re regression.

**Mechanism**: The displacement thickness δ* represents how much the effective airfoil shape is "thickened" by viscous effects: the pressure distribution sees a slightly different airfoil than the geometric one due to the BL displacing streamlines outward. Thwaites' method computes δ*(s) from the edge velocity distribution u_e(s) (available from panel method) and Re via the integral: θ(s) = (0.45ν/u_e^6) ∫₀ˢ u_e^5 ds, δ* = 2.59 θ. This gives a per-node, Re-dependent feature that captures where viscous effects are strongest (near TE, in separated regions) and how strongly they scale with Re. Note: this is distinct from the Local Re_x feature in PR #2391, which is a purely local Reynolds number estimate. δ* is a global integral that encodes the history of the boundary layer from leading edge to each surface node — exactly the non-local effect the model struggles to learn from local features alone.

**Target metrics**: p_re (primary), p_oodc

**Implementation**:
```bash
cd cfd_tandemfoil && python train.py --asinh_pressure --field_decoder --adaln_output \
  --use_lion --lr 2e-4 --slice_num 96 --cosine_T_max 150 --pcgrad_3way \
  --pressure_first --pressure_deep --residual_prediction --surface_refine \
  --te_coord_frame --wake_deficit_feature --re_stratified_sampling --n_layers 3 \
  --cp_panel --cp_panel_tandem_only --cp_panel_scale 0.1 --wake_angle_feature \
  --vortex_panel_velocity --vortex_panel_scale 0.1 --vortex_panel_n 64 --anp_srf \
  --disp_thickness_feature --disp_thickness_scale 10.0
```

**Confidence**: Medium. Thwaites' method is a classical BL approximation (works well for attached flows, degrades near separation). The feature is physically motivated and Re-dependent — exactly what p_re needs. The risk is that near-stall conditions (high AoA, high Re separation) violate Thwaites' assumptions and the feature may be noisy or misleading there.

**Risk**: Medium. The student must implement the arc-length integral along the surface for each sample. This is a loop over surface nodes sorted by arc length, computable in PyTorch with cumsum. The Re-scaling must be applied correctly (ν = ν_air / Re if normalized by chord and freestream). Recommend comparing computed δ* against published NACA 0012 BL data before using.

---

## Idea 11: Separated ANP Heads (Single-Foil vs. Tandem Specialization)

**Bold claim**: Training two separate ANP surface decoders — one for single-foil samples and one for tandem samples — with a learned mixing gate to handle intermediate cases, will improve p_tan by allowing the tandem decoder to specialize on wake interaction patterns that the single-foil decoder is never exposed to, eliminating the representation interference currently forced by the shared decoder architecture.

**Mechanism**: The current ANP SRF uses a single cross-attention head for all configurations. In tandem configurations, the surface context must simultaneously encode (a) the upstream foil's flow, (b) the downstream foil's flow, and (c) the wake interaction between them — fundamentally different from single-foil attention patterns. Training a single head on both creates representational interference. The proposed design: two parallel ANP decoders of identical architecture; during training, route single-foil samples to Head A and tandem samples to Head B; at inference, for the rare case of partial tandem configurations, use a learned mixing gate g = sigmoid(MLP(tandem_flag)) ∈ [0,1]. Total parameter overhead: 2× the ANP head parameters, which is modest (the ANP head is small relative to the backbone). This is the "mixture of experts" idea but applied specifically to the geometry-conditional specialization problem rather than capacity scaling.

**Target metrics**: p_tan (primary), p_in (ensure no regression)

**Implementation**:
```bash
cd cfd_tandemfoil && python train.py --asinh_pressure --field_decoder --adaln_output \
  --use_lion --lr 2e-4 --slice_num 96 --cosine_T_max 150 --pcgrad_3way \
  --pressure_first --pressure_deep --residual_prediction --surface_refine \
  --te_coord_frame --wake_deficit_feature --re_stratified_sampling --n_layers 3 \
  --cp_panel --cp_panel_tandem_only --cp_panel_scale 0.1 --wake_angle_feature \
  --vortex_panel_velocity --vortex_panel_scale 0.1 --vortex_panel_n 64 --anp_srf \
  --anp_split_heads --anp_tandem_head_separate
```

**Confidence**: Medium. The logic is clean — different geometries benefit from specialized attention patterns. The main risk is data imbalance: if tandem samples are rarer, the tandem head may be undertrained. This is mitigated by the re_stratified_sampling baseline flag (which presumably balances Re, but check if it also balances tandem vs single-foil).

**Risk**: Medium. The student must ensure the two heads have gradient flow balanced appropriately. If the tandem dataset is small, the tandem head may need a higher learning rate or explicit upsampling. Monitor tandem head vs single-foil head validation loss separately.

---

## Idea 12: Spectral Normalization of ANP Cross-Attention Weights

**Bold claim**: Applying spectral normalization to the ANP cross-attention query/key/value projection matrices will stabilize the attention mechanism across different Reynolds number regimes by bounding the Lipschitz constant of the attention operator, preventing the gradient explosions that likely cause the p_re regression when ANP encounters out-of-distribution Re values.

**Mechanism**: The p_re regression (+16%) after adding the ANP head is a strong signal that the attention mechanism is Re-sensitive in a way that degrades OOD performance. Spectral normalization bounds the spectral norm of each weight matrix: W_SN = W / sigma(W) where sigma(W) is the largest singular value, computed via power iteration. This limits the Lipschitz constant of each linear layer, which bounds the sensitivity of the output to input perturbations — including Re-induced input distribution shifts. Unlike dropout or weight decay (which are stochastic or penalize magnitude), spectral normalization is a deterministic constraint on the operator geometry. Apply to Q, K, V, and output projection matrices in the ANP cross-attention layers only (not the backbone Transolver layers, which are already well-tuned).

**Target metrics**: p_re (primary), p_oodc

**Implementation**:
```bash
cd cfd_tandemfoil && python train.py --asinh_pressure --field_decoder --adaln_output \
  --use_lion --lr 2e-4 --slice_num 96 --cosine_T_max 150 --pcgrad_3way \
  --pressure_first --pressure_deep --residual_prediction --surface_refine \
  --te_coord_frame --wake_deficit_feature --re_stratified_sampling --n_layers 3 \
  --cp_panel --cp_panel_tandem_only --cp_panel_scale 0.1 --wake_angle_feature \
  --vortex_panel_velocity --vortex_panel_scale 0.1 --vortex_panel_n 64 --anp_srf \
  --anp_spectral_norm --anp_spectral_norm_targets qkvo
```

**Confidence**: Medium-Low. Spectral normalization is well-validated for discriminators (GANs) and Lipschitz networks, but its application to cross-attention in regression settings is less studied. The theoretical motivation is sound, but empirical evidence in similar surrogate settings is limited. This is a lower-confidence idea than the data augmentation or architectural ideas above.

**Risk**: Low-Medium. Spectral normalization is a simple wrapper on existing layers. The main risk is over-constraining the attention weights, reducing expressiveness. If this hurts in-distribution performance (p_in), the normalization is too tight — try only normalizing Q and K (not V and O).

---

## Priority Ranking

| Rank | Idea | Confidence | Target | Rationale |
|------|------|-----------|--------|-----------|
| 1 | PIDA Chord-Scaling Aug (#1) | High | p_re | Exact physics, zero new data, direct target |
| 2 | Flip-Symmetry Aug (#2) | High | p_oodc | Proven mechanism, pure data aug, zero risk |
| 3 | AeroDiT Re-CFG (#5) | Medium | p_re | NeurIPS-validated for Re conditioning |
| 4 | Tandem Wake Prior (#9) | Medium-High | p_tan | Directly targets worst metric, motivated physics |
| 5 | Flow Matching Head (#4) | Medium | p_in | #2271 never ran; ANP conditioning now available |
| 6 | Displacement Thickness δ* (#10) | Medium | p_re | Re-dependent integral feature, distinct from #2391 |
| 7 | Multi-Resolution ANP (#8) | Medium | p_re, p_tan | Hierarchical → multi-scale Re effects |
| 8 | GeoMPNN (#6) | Medium | p_tan | ML4CFD 4th place, geometric consistency |
| 9 | Multi-Fidelity Pretraining (#3) | Medium-High | p_re | Most expensive but highest ceiling |
| 10 | Divergence-Free Constraint (#7) | Medium | p_in | Hard physics constraint, implementation risk |
| 11 | Separated ANP Heads (#11) | Medium | p_tan | Clean logic, tandem specialization |
| 12 | Spectral Norm ANP (#12) | Medium-Low | p_re | Theoretical motivation, limited empirical evidence |

## Notes for Assignment

- Assign ideas 1 and 2 together to a single student (both are pure data augmentation, complementary, can be combined in one experiment).
- Ideas 3 (multi-fidelity pretraining) should go to a student with extra time — it requires two-phase training and dataset generation.
- Ideas 4 and 5 (flow matching and diffusion head) are architecturally similar but mechanistically different — assign to different students to test in parallel.
- Wait for results from #2391 (Local Re_x BL Feature) before assigning Idea 10 (δ*) — they share BL motivation and running both simultaneously creates interpretation ambiguity.
- Wait for results from #2392 (DeltaPhi residual) before assigning Idea 9 (Wake Prior) — similar residual learning mechanism.
