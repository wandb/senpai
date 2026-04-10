# Round 40 Bold Research Ideas — 2026-04-10

## Context

All Round 39 experiments failed. All incremental physics-feature tweaks are exhausted. This document proposes 10 genuinely novel directions for Round 40, each attacking the problem from a fundamentally different angle: new backbone architectures, different prediction targets, pretraining/transfer strategies, and data augmentation. None of these overlap with existing PRs (#2270-#2278 or any of the 1966 prior experiments).

**Current baseline (2-seed avg, PR #2350):**
- p_in = 11.90 | p_oodc = 7.35 | p_tan = 27.20 | p_re = 6.40

**Baseline command:**
```
python train.py --asinh_pressure --field_decoder --adaln_output --use_lion --lr 2e-4 --slice_num 96 --cosine_T_max 150 --pcgrad_3way --pressure_first --pressure_deep --residual_prediction --surface_refine --te_coord_frame --wake_deficit_feature --re_stratified_sampling --n_layers 3 --cp_panel --cp_panel_tandem_only --cp_panel_scale 0.1 --wake_angle_feature
```

---

## Idea 1: Viscous Residual Prediction Reformulation

**Hypothesis:** Instead of predicting raw pressure p, predict the *viscous correction* `delta_p = p_CFD - p_panel` where `p_panel` is the inviscid panel-method Cp already available via `--cp_panel`. The model only needs to learn the viscous correction (boundary layer separation, wake viscous effects), which is smoother, smaller in magnitude, and more structured than the full pressure field. This is analogous to residual prediction in physics-informed networks.

**Scientific justification:** The panel Cp is already a first-order approximation of the pressure field. The CFD pressure differs from this mainly through: (1) boundary layer displacement effect, (2) trailing-edge separation, (3) wake viscous interaction. These corrections are localized, have predictable structure, and are much smaller in dynamic range than the full pressure. Predicting a smaller-magnitude, more structured residual is fundamentally an easier regression problem — equivalent to preconditioning the output space. This is the same insight behind multigrid methods and hierarchical prediction in image super-resolution. The `--residual_prediction` flag already exists for a different purpose (velocity field residual from a different base), so the infrastructure pattern is understood.

**Papers:** 
- "Residual-based physics-informed machine learning" (arXiv:2205.09025, 2022) — direct inspiration
- "Physics-informed neural networks" (Raissi et al. 2019) — residual from physics baseline as target

**Implementation guidance:**
1. In `train.py`, add a `--viscous_residual_target` flag
2. When active: subtract `cp_panel * q_inf` from the ground-truth pressure labels before computing loss
3. At inference: add back `cp_panel * q_inf` to get final pressure predictions
4. The panel Cp feature is already computed and available as an input feature — using it as the subtraction baseline is a natural extension
5. Keep `--cp_panel --cp_panel_tandem_only --cp_panel_scale 0.1` active for the *input feature* side; the new flag changes the *prediction target*
6. Importantly, this is only applied to surface nodes (where Cp is defined); volume nodes use standard prediction

**Expected impact:** Potentially large improvement in p_tan (tandem surface pressure) and p_in (single-foil in-distribution). The model no longer needs to learn the global pressure level or the dominant inviscid contribution — only the correction. This directly attacks the hardest part of pressure prediction.

**Confidence:** Strong theoretical basis. Has not been tried in any prior PR. High priority.

---

## Idea 2: Surface-Only Boundary GNN (B-GNN)

**Hypothesis:** Replace the volume Transolver + SRF architecture with a graph neural network that operates *exclusively on surface mesh nodes*, using all-to-all message passing with panel-method Cp as edge/node initialization. This eliminates volume-to-surface information transfer bottlenecks and forces the model to learn aerodynamics from surface-intrinsic quantities only.

**Scientific justification:** Jena et al. (arXiv:2503.18638, 2025) showed that Boundary GNNs operating exclusively on surface meshes achieve 83% model size reduction and require 87% less training data than full-field models on the airFRANS benchmark — matching full-field model accuracy. The key insight: for engineering quantities (lift, drag, surface pressure), the surface contains all the information. The volume field is a consequence of the boundary conditions, not a cause. Our current SRF head is a step toward this, but it still depends on volume-interpolated backbone features. A pure surface GNN would receive: surface coordinates, AoA, Re, panel Cp, wake angle — and directly predict surface p, Ux, Uy without any volume backbone.

**Paper:** "Boundary Graph Neural Networks for 3D Simulations" → arXiv:2503.18638, Jena et al., TU Delft, 2025

**Implementation guidance:**
1. Build a standalone `SurfaceGNN` module in `train.py`
2. Nodes: surface mesh points with features [x, y, n_x, n_y, cp_panel, s/c (arc-length fraction), wake_deficit, wake_angle, log(Re), AoA]
3. Edges: k-nearest neighbors on the surface (k=8), plus "global" edges via attention-weighted all-to-all (like a surface Transformer)
4. Layers: 4-6 GraphTransformer layers (use `torch_geometric.nn.TransformerConv`)
5. Output: per-node [Ux, Uy, p] — surface only
6. For volume prediction: keep the Transolver backbone for volume nodes, discard SRF, use the surface GNN output as boundary condition anchors for volume interpolation
7. Or: run as surface-only model, skip volume prediction entirely (metrics only care about surface)
8. Start with surface-only prediction: `python train.py --surface_only_gnn` (new flag)

**Expected impact:** If surface-only B-GNN matches full-field Transolver, we gain a much smaller, faster model. If it beats it, we validate that volume information is noise for surface prediction. Either outcome is scientifically valuable.

**Confidence:** Strong empirical evidence in analogous domain (airFRANS). Implementation requires new module but no fundamental changes to training loop.

---

## Idea 3: DPOT-Style Denoising Pretraining

**Hypothesis:** Pretrain a Transolver backbone on diverse PDE datasets (heat equation, Navier-Stokes, Burgers, wave equation) using denoising as the pretraining objective, then fine-tune on TandemFoilSet. The pretrained model develops universal PDE representations that transfer to our specific CFD setting, reducing the effective amount of labeled CFD data needed and improving generalization (especially p_re OOD).

**Scientific justification:** DPOT (Hao et al., ICML 2024) demonstrated that auto-regressive denoising pretraining on diverse PDE families (using FNO/Transolver-style architectures) produces neural operator backbones that transfer strongly to new PDE tasks with fewer labeled examples. The pretraining teaches the model to denoise corrupted PDE solutions — forcing it to learn the underlying PDE structure rather than memorizing input-output maps. This is directly analogous to BERT/MAE pretraining in NLP/vision. For our p_re metric (the Re-OOD split), the model likely overfits to the training Re range — a pretrained backbone with universal PDE representations would generalize across Re values better.

**Papers:**
- "DPOT: Auto-Regressive Denoising Operator Transformer for Large Scale PDE Pre-Training" (Hao et al., ICML 2024, arXiv:2403.03542)
- GitHub: HaoZhongkai/DPOT (pretrained checkpoints available)
- "Strategies for Pretraining Neural Operators" (Subramanian et al., TMLR 09/2024, CMU)

**Implementation guidance:**
1. Download DPOT pretrained Transolver checkpoint from HaoZhongkai/DPOT GitHub
2. In `train.py`, add `--pretrained_backbone <path>` flag to load pretrained weights into the Transolver backbone
3. Freeze backbone for first 20 epochs (warm-up), then unfreeze with 10× lower LR for backbone vs. head
4. The DPOT Transolver may use different hidden dimensions — add a projection layer to adapt if needed
5. Key hyperparameters: backbone LR = 2e-5 (10× lower than head), unfreeze after epoch 20, weight decay 1e-4 on backbone
6. Alternatively: use DPOT checkpoint as initialization and fine-tune all layers from epoch 0 with discriminative LRs
7. Use `--wandb_group dpot_pretrain` to group multiple LR-ratio experiments

**Expected impact:** Most likely improvement in p_re (Re-OOD generalization). Possible improvement in p_oodc. May not help p_tan (tandem-specific geometry) since pretrained data has no tandem foils.

**Confidence:** Strong evidence from DPOT paper (FNO/Transolver architectures). Main risk: checkpoint compatibility. Medium-high priority.

---

## Idea 4: Biot-Savart Cross-Foil Attention Bias

**Hypothesis:** Use the vortex-panel method to compute the velocity induced by foil 1 at foil 2's surface nodes (and vice versa), and inject this as an attention bias in the Transolver's cross-node attention. This gives the model exact inviscid aerodynamic interaction physics between the two foils as a structural prior in the attention mechanism.

**Scientific justification:** In tandem airfoil aerodynamics, the primary aerodynamic coupling between fore and aft foil is through the vortex wake: the fore foil sheds vorticity that modifies the inflow to the aft foil. The Biot-Savart law gives the induced velocity at any point due to a vortex sheet. This is the same physics that the panel method computes to get the pressure distribution. By injecting Biot-Savart-derived induced velocities as attention biases (like ALiBi positional encoding but physics-based), we give the attention mechanism explicit knowledge of which nodes are aerodynamically "connected" across foils. This is the cross-foil analogue of what the wake_deficit feature does for single-node features — but now as a structural attention prior.

**Papers:**
- Classic potential flow theory: Katz & Plotkin "Low-Speed Aerodynamics" (2001) — Biot-Savart kernel
- "Relative Positional Encoding via Kernel Regression" (arXiv:2212.10271) — kernel-based attention bias
- ALiBi (Press et al., 2022) — additive attention biases from non-learned signals

**Implementation guidance:**
1. At data preprocessing time (in `prepare.py` or as a runtime feature): for each tandem configuration, compute the panel vortex distribution on foil 1, then evaluate Biot-Savart-induced (Ux, Uy) at all foil 2 surface nodes (and vice versa)
2. This gives a (N_surface_nodes × 2) tensor per foil of cross-foil induced velocity
3. In Transolver attention: add this as an additive bias to attention logits between foil-1 and foil-2 node pairs — nodes from the same foil get zero bias, cross-foil pairs get the Biot-Savart magnitude as the bias
4. Implementation in attention: `attention_logits += biot_savart_bias.unsqueeze(0)` where bias is (N1, N2) for cross-foil pairs
5. The vortex-panel induced velocity experiment (#2357) by askeladd already computes the per-node velocity oracle — this experiment extends it to be used as an *attention bias* rather than a node feature
6. Flag: `--biot_savart_attention_bias`

**Expected impact:** Directly targets p_tan (tandem OOD) by giving the model structural knowledge of cross-foil coupling. Should also help p_oodc where tandem configurations appear in the OOD split.

**Confidence:** Strong aerodynamic motivation. Novel in the neural operator literature (most attention biases are positional, not physics-derived). Medium priority — depends on askeladd's #2357 result.

---

## Idea 5: Geometry-Augmented Training via Free-Form Deformation

**Hypothesis:** Apply random Free-Form Deformation (FFD) to training airfoil geometries to create novel synthetic geometries, then compute their panel Cp using XFoil/NeuralFoil, and use these augmented samples during training. This multiplies the effective training set size and forces the model to generalize to a wider distribution of airfoil shapes.

**Scientific justification:** TandemFoilSet contains a fixed set of airfoil geometries. The model may overfit to the specific shapes in the training set. FFD is a standard aerodynamic shape parameterization (used in adjoint-based aerodynamic optimization) that smoothly deforms airfoil geometry using a lattice of control points. Small FFD perturbations (displacement < 2% chord) create realistic airfoil shapes that are physically plausible and not in the training set. NeuralFoil (the fast neural network Cp predictor already referenced in the codebase) can compute approximate Cp for these synthetic geometries in milliseconds. The augmented samples then serve as additional training points with approximate but useful labels. This is a data augmentation strategy with physics consistency built in — unlike random noise augmentation, FFD augmentation creates new geometries that lie on the manifold of aerodynamically plausible shapes.

**Papers:**
- "FuncGenFoil: Function-Space Generative Model for Airfoil Geometry" (arXiv:2502.10712, Feb 2025) — generative model for novel airfoil geometry in function space
- "Free-Form Deformation of Solid Geometric Models" (Sederberg & Parry, SIGGRAPH 1986) — the original FFD paper
- NeuralFoil GitHub (peterdsharpe/NeuralFoil) — fast Cp predictor

**Implementation guidance:**
1. At training time, for each airfoil in the batch, apply random FFD with 4×2 control point lattice, displacement amplitude sampled from N(0, 0.01c) clipped to [-0.02c, 0.02c]
2. Compute NeuralFoil Cp for the deformed geometry (fast, < 1ms per airfoil)
3. Use the deformed geometry + NeuralFoil Cp as a new training sample with a reduced-weight loss (e.g., 0.3× normal weight) to reflect label noise
4. Only augment "single foil" configurations to avoid the complexity of tandem FFD
5. Flag: `--ffd_augmentation --ffd_weight 0.3 --ffd_amplitude 0.01`
6. Implement FFD in < 50 lines of pure PyTorch: B-spline lattice deformation applied to (x, y) coordinates

**Expected impact:** Primarily improves p_in and p_oodc (single-foil in-distribution and OOD). The model sees more shape diversity during training, improving its generalization to unseen geometries. Tandem metrics may not benefit directly.

**Confidence:** Medium. FFD augmentation is well-established in aerodynamic shape optimization but novel in neural CFD surrogates. Risk: NeuralFoil Cp may be insufficiently accurate for the reduced-weight augmented samples to help. Start with `--ffd_weight 0.1` to be conservative.

---

## Idea 6: Multiphysics Pretraining via The Well Dataset

**Hypothesis:** Pretrain the Transolver backbone on "The Well" (a large-scale collection of diverse physics simulation datasets spanning fluid dynamics, MHD, wave equations, etc.) using a masked-field reconstruction objective, then fine-tune on TandemFoilSet. The diverse pretraining develops representations that generalize across Reynolds numbers and flow regimes.

**Scientific justification:** "Towards Universal Neural Operators through Multiphysics Pretraining" (arXiv:2511.10829, Nov 2025) demonstrated that pretraining on The Well dataset (15 diverse physics simulations, 15TB of data) with a codomain attention mechanism yields neural operators that transfer strongly to held-out PDE families with minimal fine-tuning. "Pretraining Codomain Attention Neural Operators for Solving Multiphysics PDEs" (NeurIPS 2024, McCabe et al.) showed similar results with codomain attention. The key: diverse PDE data teaches the model universal discretization-invariant representations, which is exactly what we need for Re-OOD generalization.

**Papers:**
- "Towards Universal Neural Operators through Multiphysics Pretraining" (arXiv:2511.10829, Nov 2025)
- "Pretraining Codomain Attention Neural Operators for Solving Multiphysics PDEs" (McCabe et al., NeurIPS 2024)
- "The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning" (Ohana et al., NeurIPS 2024)
- "DPOT: Auto-Regressive Denoising Operator Transformer" (Hao et al., ICML 2024, arXiv:2403.03542)

**Implementation guidance:**
1. Download DPOT pretrained checkpoint (HaoZhongkai/DPOT on GitHub) — this already includes Transolver architecture pretraining on diverse PDE families
2. Add `--pretrained_checkpoint <path>` flag to `train.py`
3. Load checkpoint weights selectively (backbone only, not output head)
4. Discriminative learning rates: backbone 2e-5, SRF head and output layers 2e-4
5. Add `--backbone_freeze_epochs 15` to freeze backbone for first 15 epochs while output head adapts
6. After unfreezing: cosine annealing with T_max=135 (remaining epochs from 150 total)
7. The Transolver architecture in DPOT uses identical attention mechanisms to our baseline — weight loading should be straightforward if hidden_dim matches (check DPOT config)

**Expected impact:** Target metric: p_re (Re-OOD). Secondary: p_oodc. This is the most principled approach to improving OOD generalization that hasn't been tried. The pretraining provides a strong prior over PDE solution structure.

**Confidence:** Strong evidence from DPOT/The Well papers. Main risk: checkpoint dimension mismatch requiring projection layer.

---

## Idea 7: MoE Domain-Expert FFN Routing

**Hypothesis:** Replace the Transolver's position-wise FFN layers with a Mixture of Experts (MoE) where routing is conditioned on domain type: one set of expert FFN weights for single-foil configurations, another for tandem configurations, with a third for transition regions. Unlike the Gumbel MoE tried in Round 26 (which failed), this routing is deterministic and conditioned on a global geometry flag, not learned per-token.

**Scientific justification:** The Gumbel MoE trial failed because sparse routing added instability and the routing itself had to be learned. The key failure mode was PCGrad incompatibility and routing collapse. A deterministic, globally-conditioned MoE avoids both: we explicitly know whether a sample is single-foil or tandem, and we use this as a hard routing signal. The tandem configuration has fundamentally different aerodynamic physics (wake interaction, circulation modification) that may benefit from separate FFN weights. This is equivalent to training two separate models and combining them — but sharing the attention backbone (which captures geometry regardless of foil count).

**Papers:**
- "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (Shazeer et al., 2017)
- "Switch Transformers" (Fedus et al., 2022) — simplified MoE routing
- Domain-conditioned routing: implicit in many multi-task learning works

**Implementation guidance:**
1. Add `--domain_moe` flag
2. In each Transolver FFN layer, instantiate 2 expert FFNs: `ffn_single`, `ffn_tandem`
3. Routing: check if `is_tandem` batch flag (already in data) → hard route all tokens to the appropriate expert
4. No learned router, no gating, no auxiliary loss — purely deterministic based on configuration type
5. Parameter count doubles in FFN layers only — other layers shared. Total parameter increase: ~40-60% depending on FFN fraction
6. To keep training efficient: use LoRA-style delta experts — `ffn_tandem = ffn_shared + delta_ffn_tandem` where `delta_ffn_tandem` has rank=32. This limits extra parameters to < 5% of total.
7. Flag: `--domain_moe --domain_moe_rank 32`

**Expected impact:** Primary target: p_tan (tandem-specific surface MAE). The tandem-specific FFN expert can specialize in wake interaction physics. Secondary: p_oodc. May not affect p_in (single-foil in-distribution since shared expert handles it).

**Confidence:** Medium. The LoRA-style delta expert design avoids the Gumbel MoE failure mode. Deterministic routing is simpler and more stable. Worth trying.

---

## Idea 8: Global Cl/Cd Embedding as SRF Conditioning

**Hypothesis:** Compute predicted Cl and Cd from the model's current surface pressure prediction (via numerical integration), use these as conditioning signals for the SRF head refinement, and train with an auxiliary Cl/Cd supervision loss. The SRF head is then conditioned on global aerodynamic performance, allowing it to consistently enforce integral aerodynamic constraints.

**Scientific justification:** The current SRF head operates node-by-node without awareness of global aerodynamic integrals. Yet the surface pressure distribution must integrate to give physically consistent Cl and Cd. By explicitly conditioning SRF on predicted Cl/Cd, we inject a global constraint that acts like a Lagrange multiplier — forcing local pressure refinements to be globally consistent. This is different from the Cl/Cd auxiliary loss (fern #2340) which only adds a loss term. Here, Cl/Cd *enters as input to SRF*, making the refinement aware of the global aerodynamic state. A model that can condition on "this is a high-lift configuration" (Cl=1.8) vs. "this is a drag-dominated configuration" (Cd=0.05) can make more accurate local corrections.

**Papers:**
- "Physically consistent neural networks for aerodynamic coefficient prediction" (various, see: Liu et al. 2021)
- "Integral constraints in neural PDE surrogates" — implicit in many physics-informed works
- HyperNetworks (Ha et al., 2017) — using one network's output to condition another

**Implementation guidance:**
1. After the first pass of pressure prediction, numerically integrate surface pressure to get Cl, Cd estimates: `Cl = integrate(p * n_y, ds)` (trapezoidal rule on surface arc length)
2. Encode [Cl, Cd] through a 2-layer MLP to get a 64-dim conditioning vector
3. Use AdaLN (already available via `--adaln_output`) to inject this conditioning into SRF head layers
4. Add auxiliary supervision: `loss_cl = MSE(pred_Cl, true_Cl)` with weight 0.1 — this requires true Cl/Cd in the dataset (check if available in `prepare.py`)
5. Flag: `--clcd_srf_conditioning --clcd_aux_weight 0.1`
6. Two-pass forward: (1) backbone → coarse pressure → integrate to get Cl/Cd, (2) SRF conditioned on Cl/Cd → refined pressure
7. Gradient flows through both passes

**Expected impact:** Targets p_tan (tandem configurations have distinct Cl/Cd signatures) and p_in. The global conditioning allows SRF to make context-aware corrections. Low implementation risk since `--adaln_output` infrastructure already exists.

**Confidence:** Medium-high. The mechanism is clean and specific. Risk: if true Cl/Cd is not in the dataset, must use predicted Cl/Cd only (noisier signal). Check `prepare.py` for available features.

---

## Idea 9: LoRA Adaptation from Pretrained Neural Operator

**Hypothesis:** Load a pretrained neural operator backbone (DPOT or CORAL pretrained on diverse CFD data), freeze all parameters, and train only low-rank LoRA adapters (rank=8, alpha=16) per Transolver layer on TandemFoilSet. The frozen backbone captures universal PDE structure; the LoRA adapters specialize to our tandem airfoil domain with minimal overfitting.

**Scientific justification:** LoRA (Hu et al., 2022) has proven highly effective for adapting large pretrained models to specific domains while preventing catastrophic forgetting. "LoRA for PDE Surrogates" (arXiv:2502.00782, 2025) demonstrated that freezing a pretrained neural operator backbone and adapting only LoRA modules achieves comparable performance to full fine-tuning with 90% fewer trainable parameters — and significantly better OOD generalization because the pretrained representations are preserved. For our setting: the DPOT pretrained Transolver has seen diverse fluid dynamics problems (Navier-Stokes, turbulence, channel flow). Freezing its backbone and using LoRA to specialize on tandem airfoil CFD would preserve the universal fluid dynamics representations while adapting to our specific geometry/Re/AoA distribution.

**Papers:**
- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2022, arXiv:2106.09685)
- "Low-Rank Adaptation for Neural Operators" (arXiv:2502.00782, 2025)
- "DPOT: Auto-Regressive Denoising Operator Transformer" (Hao et al., ICML 2024, arXiv:2403.03542)

**Implementation guidance:**
1. Start with DPOT pretrained Transolver checkpoint (same as Idea 3)
2. Load checkpoint, freeze all backbone weights: `backbone.requires_grad_(False)`
3. Add LoRA to all Q, K, V, and FFN weight matrices in the Transolver: use `loralib` or manual implementation
4. LoRA parameters: rank=8, alpha=16, dropout=0.1
5. Trainable components: LoRA adapters (< 2M params), SRF head, output projection
6. LR: 5e-4 for LoRA + heads (higher than standard since backbone is frozen)
7. Flag: `--lora_pretrained --lora_rank 8 --lora_alpha 16 --pretrained_checkpoint <path>`
8. This is distinct from full fine-tuning (Idea 3) — here the backbone is FROZEN. Compare both to understand representation transfer vs. feature extraction tradeoffs.

**Expected impact:** Strongest expected improvement in p_re (Re-OOD) and p_oodc (geometry OOD). The frozen pretrained backbone maintains universal PDE representations that generalize across conditions. Risk of underfitting if LoRA rank is too small — try rank=16 if rank=8 underperforms.

**Confidence:** Strong theoretical basis + paper directly in this setting. Main risk: pretrained checkpoint compatibility.

---

## Idea 10: DSDF-Weighted Physics Features + Adaptive Feature Fusion

**Hypothesis:** Scale all physics input features (panel Cp, wake deficit, wake angle, induced velocity) by a learned distance-to-surface decay function, and use a small learned fusion network to adaptively weight each physics feature's contribution per node rather than using fixed scale factors. Features near the surface are amplified; far-field nodes see attenuated physics hints.

**Scientific justification:** Currently, physics features like `--cp_panel` use a fixed global scale (0.1). But panel Cp is most accurate near the foil surface and increasingly inaccurate in the wake and far field (potential flow assumption breaks down). A Signed Distance Field (SDF) or unsigned distance to the nearest surface node gives a natural weighting: `weight(x) = exp(-d(x) / lambda)` where d is distance-to-surface and lambda is a learned decay length. This ensures physics features are trusted most where they are most accurate, and contributes minimally where they are unreliable. The "learned fusion" extension uses a per-feature gating network: a small 2-layer MLP per physics feature that outputs a scalar weight given node position and local geometry, replacing fixed hyperparameter scales with adaptive weighting.

**Papers:**
- "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation" (Park et al., 2019) — SDF computation
- "Feature-wise Linear Modulation" (Perez et al., 2018) — adaptive feature scaling
- "Geometric deep learning" (Bronstein et al., 2021) — geometry-aware feature processing

**Implementation guidance:**
1. Precompute distance-to-nearest-surface for all mesh nodes at data loading time (already possible from node coordinates and surface node mask)
2. Compute `sdf_weight = exp(-dist / lambda)` where lambda is a learnable scalar per physics feature type (initialized to median distance in dataset)
3. Multiply each physics feature by its sdf_weight before concatenating to node features
4. Flag: `--dsdf_weighted_features --dsdf_lambda_init 0.1`
5. For adaptive fusion: replace fixed `--cp_panel_scale 0.1` with a learned scale: `cp_panel_effective = cp_panel * sigma(linear(node_features))` where sigma is sigmoid scaled to [0, 1]
6. Keep `--cp_panel_tandem_only` constraint to avoid regression on previously-merged behavior
7. Regularize learned lambdas with L2 to prevent collapse to zero (physics features disappearing)

**Expected impact:** Primary improvement in volume MAE (physics features are currently over-weighted in far field). Surface MAE improvement in p_oodc if the adaptive weighting generalizes better across geometry distributions. This is a relatively low-risk change since it subsumes the existing fixed-scale behavior.

**Confidence:** Medium. The SDF weighting has clear theoretical motivation. The learned fusion adds complexity — if it doesn't help, fall back to fixed SDF weighting without the adaptive component.

---

## Priority Ranking for Round 40

Prioritized by: (1) theoretical strength, (2) implementation feasibility, (3) non-redundancy with current WIP experiments

| Priority | Idea | Target Metric | Complexity | Expected Impact |
|----------|------|--------------|------------|-----------------|
| 1 | Viscous Residual Prediction (Idea 1) | p_tan, p_in | Low | High — fundamental reformulation |
| 2 | DPOT Pretraining (Idea 3) | p_re, p_oodc | Medium | High — well-validated transfer |
| 3 | LoRA from Pretrained Operator (Idea 9) | p_re, p_oodc | Medium | High — addresses OOD directly |
| 4 | Biot-Savart Attention Bias (Idea 4) | p_tan, p_oodc | Medium | High — novel cross-foil physics |
| 5 | Surface B-GNN (Idea 2) | p_in, p_oodc | High | High — architectural rethink |
| 6 | Global Cl/Cd SRF Conditioning (Idea 8) | p_tan, p_in | Low | Medium — global-local consistency |
| 7 | MoE Domain Expert FFN (Idea 7) | p_tan | Low | Medium — tandem specialization |
| 8 | FFD Geometry Augmentation (Idea 5) | p_in, p_oodc | Medium | Medium — data diversity |
| 9 | Multiphysics Pretraining (Idea 6) | p_re | Medium | Medium — similar to Idea 3 |
| 10 | DSDF-Weighted Features (Idea 10) | p_oodc | Low | Low-Medium — incremental |

**Immediate assignments:** Ideas 1, 3, 9, 4, 8 are the top 5 picks for idle students. Ideas 2, 6, 7 as secondary assignments.

**Note:** Ideas 3 and 9 share the DPOT pretraining infrastructure — assign to different students and run in parallel to compare full fine-tune vs. LoRA-only approaches directly.
