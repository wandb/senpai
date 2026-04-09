# Research Ideas — Round 32 (2026-04-09)

Focus: Paradigm-level changes — attention architecture, physics-informed features, in-context conditioning, loss reformulation, and training dynamics. NOT incremental tweaks. Every idea below has not been tried and is not currently in-flight.

## Baseline Reference (as of Round 30–31)

| Track | Surface MAE |
|---|---|
| val_in_dist | ~19–21 |
| val_ood_cond | ~20–22 |
| val_ood_re | ~31 |
| val_tandem_transfer | ~41–43 |
| val/loss | ~2.22–2.24 |

**In-flight (do NOT duplicate):** DID Streamwise Feature, GMSE Gradient-Weighted Pressure Loss, Potential Flow Residual Loss, Mirror Symmetry Augmentation, Continuity PDE Loss, Circulation Lift Feature, Wake Centerline SDF, Shortest Vector Feature.

---

## Idea 1: Grouped Query Attention (GQA) — 2 K/V Groups

**What it is.** Replace current Multi-Query Attention (1 shared K/V group, merged PR #513) with Grouped Query Attention using 2 K/V groups. This is the strict middle ground between MQA (1 group) and full MHA (8 groups). Each pair of query heads shares one K/V head.

**Why it might help.** MQA was a significant win (PR #513), suggesting that K/V sharing has a beneficial regularization effect — it forces the model to find physics patterns that generalize across query heads. However, with zero K/V diversity, the model loses the capacity to represent distinct K/V structures (e.g., pressure gradients vs. velocity shear). GQA with 2 groups allows two distinct "information extraction" paths while retaining 75% of the parameter reduction from MQA. The LLAMA-2/3, Mistral, and Gemma model families all use GQA over MQA for exactly this reason. In fluid dynamics terms, this might allow the model to separately attend to "near-wall" physics (boundary layer) and "wake/freestream" physics with independent K/V representations.

**Key papers.**
- Ainslie et al. (2023) "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." arXiv:2305.13245. Demonstrates 5% compute cost to uptrain from existing checkpoints; quality close to MHA, speed close to MQA.
- The paper shows GQA consistently outperforms MQA on downstream tasks (FLAN, T5, summarization) when compute is matched.

**Implementation notes.**
- Current MQA: `slice_token_kv = slice_token.mean(dim=1, keepdim=True)` — single K/V for all 8 heads.
- GQA with 2 groups: split 8 heads into 2 groups of 4. Two K/V heads: `slice_token_kv = slice_token.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)` with independent `to_k` and `to_v` projections of size `(2, slice_num, dim_head)`. Each query head attends to the K/V of its group.
- The mean pooling logic stays — just replicate for 2 groups instead of 1.
- Parameter increase is minimal: 2× the K/V projection weights (4 matrices instead of 2, still tiny vs. query projections).
- Do NOT change the number of slice tokens or hidden dim.
- Suggested: `n_kv_groups = 2` as a hyperparameter. Start there; do not sweep to 4 or 8.

**Expected impact.** Moderate. The MQA win was substantial (~2–3% surface MAE). GQA should recover some of the expressivity lost by full K/V sharing while keeping regularization benefits. Expected: 0.5–2% improvement on all tracks, likely more on val_ood_re and val_tandem_transfer where inter-head diversity matters most.

**Risk.** Low-medium. Well-validated in LLM literature. The main risk is that MQA's regularization via full sharing is what drove the win, and GQA partially undoes it. If that's true, val_ood_re will degrade. The student should report per-track results to disambiguate.

**Confidence.** Strong evidence from language model settings. Moderate confidence in transfer to CFD mesh setting.

---

## Idea 2: SE Channel Attention on Slice Tokens (Inside PhysicsAttention)

**What it is.** Apply Squeeze-and-Excitation (SE) channel recalibration directly to the slice token tensor inside PhysicsAttention — not to the node feature MLP (which was done in PR #772). The slice tokens (shape: `[bsz, heads, n_slices, dim_head]`) are globally pooled over the slice dimension and recalibrated per-channel before the attention projection.

**Why it might help.** The existing SE block (PR #772) recalibrates node features after the MLP. But the slice tokens are the core "physics concept" representations in Transolver — they are what the model learns to route nodes into. Applying SE to slice tokens allows the model to selectively suppress or amplify entire "physics modes" (e.g., a slice token that tracks boundary layer separation vs. one tracking freestream flow). This is analogous to channel attention on feature maps in image recognition, but applied at the "prototype" level of the physics decomposition. PR #772's success on node features is evidence that SE recalibration is effective in this architecture; the untested application is at the slice token level.

**Implementation notes.**
- Inside `PhysicsAttention.forward`, after computing `slice_token = (attn @ v_node)` (shape `[bsz, heads, n_slices, dim_head]`):
  - Pool over slices: `se_in = slice_token.mean(dim=2)` → `[bsz, heads, dim_head]`
  - Bottleneck MLP: `Linear(dim_head, dim_head//4)` → GELU → `Linear(dim_head//4, dim_head, bias=False)` → Sigmoid (zero-init final Linear weights)
  - Gate: `slice_token = slice_token * se_gate.unsqueeze(2)`
- This is orthogonal to the node-level SE block from PR #772 — both can coexist.
- Zero-init the expansion Linear to start as identity; this prevents disruption of the pretrained routing.
- Keep the bottleneck ratio at 4× (dim_head=32, bottleneck=8). Do not increase.

**Expected impact.** Low-moderate. Incremental but grounded. The node SE block improved surface MAE by ~1–2%; slice-level SE should be similar. Might disproportionately help val_tandem_transfer if tandem flow has a distinct "tandem mode" that benefits from explicit amplification.

**Risk.** Low. Zero-init prevents training disruption. The operation is inside a single attention layer and doesn't change the overall data flow.

**Confidence.** Moderate — extrapolated from PR #772 result plus SE literature. Direct validation missing for this specific location.

---

## Idea 3: Auxiliary Angle-of-Attack Prediction Head

**What it is.** Add an auxiliary regression head predicting the angle of attack (AoA) from the model's internal representation, alongside the existing auxiliary Reynolds number head (PR #780). This is the geometric/kinematic dual to Re's viscous conditioning.

**Why it might help.** PR #780 showed that forcing the model to explicitly decode Re from its representation improved generalization, likely by aligning the latent space with the governing physics parameter. AoA is the other primary governing parameter for lift and pressure distribution — it determines stagnation point location, suction peak magnitude, and trailing edge behavior. In tandem foil configurations, the effective AoA of the rear foil is modified by the downwash of the front foil, making explicit AoA awareness especially important for val_tandem_transfer. The hypothesis is that the model's slice tokens already implicitly encode something AoA-related, but an explicit auxiliary loss will sharpen this encoding and force it to be recoverable, improving the generalization of pressure predictions.

**Implementation notes.**
- Add `aux_aoa_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1))` at the same attachment point as the Re head (after the penultimate TransolverBlock).
- Extract the pooled representation (e.g., mean over nodes) and pass through the head.
- AoA is available in the dataset as a direct input feature — use it as the target. Predict in radians or degrees consistently.
- Loss: `L1Loss(pred_aoa, true_aoa) * aoa_aux_weight`. Start with `aoa_aux_weight = 0.05` (same as or slightly lower than Re aux weight). Do not sweep.
- The AoA range spans single-foil and tandem configurations — normalize by dividing by max AoA in training set (or use mean/std normalization over training set).
- Disable the head at test time (only affects gradient signal during training).

**Expected impact.** Moderate. PR #780 (Re head) improved val_ood_re significantly. AoA head is expected to improve val_tandem_transfer and val_ood_cond, where geometry-condition generalization matters most. Expected: 1–3% improvement on tandem/ood tracks.

**Risk.** Low. The Re head analogy is well-validated. The main risk is AoA is already explicitly in the input features, so the model may have already learned it — the auxiliary signal adds noise without benefit. If val_loss increases with no surface MAE gain, close promptly.

**Confidence.** Strong analogy to PR #780. Moderate confidence in AoA-specific gains.

---

## Idea 4: Asymmetric Quantile (Pinball) Loss on Pressure

**What it is.** Replace the symmetric L1 loss on the pressure channel with an asymmetric quantile loss (pinball loss) with tau > 0.5, penalizing underprediction of Cp more than overprediction. This targets suction-side (negative Cp) accuracy, which dominates lift force errors.

**Why it might help.** The current loss treats overprediction and underprediction of pressure equally. In aerodynamics, the suction peak on the upper surface (Cp << 0) is the primary driver of lift and the most error-sensitive region. A symmetric loss allows the model to "hedge" toward the mean, smoothing the suction peak. The pinball loss with tau = 0.7 means underpredictions (model too high, misses the suction) are penalized 2.33× more than overpredictions. This is distinct from the symmetric 2× p-channel weighting already tried — it changes the shape of the loss function, not just its scale. Quantile regression is a Kaggle standard for skewed-error targets. The mechanism: gradient sign flips based on prediction direction, creating an asymmetric pull toward the tail.

The pinball loss formula: `L(y, yhat, tau) = tau * max(y - yhat, 0) + (1-tau) * max(yhat - y, 0)`.

**Key papers.**
- Koenker & Bassett (1978) "Regression Quantiles" — original formulation. Econometrica 46(1):33–50.
- Steinwart & Christmann (2011) "Estimating conditional quantiles with the help of the pinball loss." AISTATS.
- arXiv:2012.14348 "Generalized Quantile Loss for Deep Neural Networks" (2020) — modern deep learning treatment.

**Implementation notes.**
- Replace `F.l1_loss(pred_p, true_p)` with pinball loss for the pressure channel only. Keep L1 for Ux, Uy.
- Implementation: `loss_p = torch.where(true_p > pred_p, tau * (true_p - pred_p), (1-tau) * (pred_p - true_p)).mean()`
- Start with tau = 0.65 (mild asymmetry). Do NOT use tau = 0.9+ — too aggressive, likely to diverge.
- Apply only to surface nodes (where pressure accuracy matters most); keep symmetric L1 for volume nodes.
- The existing 2× p-channel surface weight and this asymmetric loss are orthogonal — both can be active simultaneously.
- Monitor val_surface_mae_p specifically. If it improves but Ux/Uy degrade, the tau is too high.

**Expected impact.** Low-moderate. Kaggle-style empirical win. Expected: 1–3% improvement in surface pressure MAE, especially on suction-side nodes. Minimal effect on velocity channels.

**Risk.** Medium. Asymmetric loss changes the implicit target from conditional mean to conditional quantile — the model will learn to predict a biased estimate. If tau is poorly calibrated, this can hurt. The key test is whether the aggregate surface MAE improves despite the bias. Requires careful monitoring of per-channel metrics.

**Confidence.** Well-established in Kaggle/quantile regression literature. Moderate confidence in CFD surface pressure transfer. No direct prior validation in this exact setting.

---

## Idea 5: Inviscid Panel Method Cp as Physics Input Feature

**What it is.** Compute the inviscid (potential flow / thin-airfoil) pressure coefficient Cp for each surface node using a cheap analytical approximation, and provide it as an additional input feature. The model sees the "inviscid ground truth" and learns to correct for viscosity, boundary layers, and separation.

**Why it might help.** Potential flow solutions are analytically tractable (via the Joukowski transform or panel methods) and capture the leading-order pressure distribution without any neural computation. The residual between RANS Cp and inviscid Cp is a much smoother, lower-variance function — it primarily captures viscous effects and separation. If the model predicts this residual instead of raw Cp, the learning problem becomes substantially easier. This is the "physics-as-baseline" strategy analogous to how weather models use NWP forecasts as features. Panel method Cp is O(N_panels) to compute per sample and can be precomputed offline. The Boundary GNN paper (arXiv:2503.18638) validated exactly this approach: providing panel-method Cp as a physics-informed input feature improved pressure predictions in similar mesh surrogate tasks.

**Key papers.**
- arXiv:2503.18638 "Boundary Condition Enforced Graph Neural Networks for Aerodynamic Flow Prediction" (2025). Uses panel method Cp as physics-informed input, validated on airfoil surrogate tasks.
- Katz & Plotkin "Low Speed Aerodynamics" (2001) — reference for panel method implementation.

**Implementation notes.**
- Precompute at dataset preparation time (offline, not during training) using a simple vortex panel method or the thin-airfoil approximation.
- For 2D single airfoil at AoA alpha: the thin-airfoil approximation gives `Cp(x/c) ≈ 1 - (dz/dx + alpha)^2 * 4` at each surface point. This can be computed purely from geometry and AoA.
- For tandem configuration: compute two separate panel solutions and use superposition (valid in potential flow).
- Feature: append `Cp_inviscid` as a scalar to the node feature vector. No architectural change needed — just an additional input dimension.
- The feature only applies to surface nodes; for volume nodes, set it to zero or to the closest inviscid field value.
- This requires modifying how features are constructed at data prep time or as a preprocessing step in `train.py`. Since `data/prepare.py` is read-only, the computation must happen inside `train.py` using raw geometry and AoA features that are already available.
- Thin-airfoil approximation needs only (x, y, dz/dx) which are available from the existing geometry features.

**Expected impact.** Moderate-high. Physics baselines as inputs consistently help in scientific ML (a form of "warm starting" the regression). Expected: 2–5% improvement in surface pressure MAE, especially at high AoA where suction peaks are strong.

**Risk.** Medium-high. The panel method approximation degrades at high AoA (separation, leading-edge vortices) and for tandem configurations (interference effects). The correction residual may still be complex near stall. Also, implementing this correctly in `train.py` without modifying read-only data files requires care. The student should test first on val_in_dist to confirm the precomputed features are correct before evaluating OOD tracks.

**Confidence.** Moderate. Strong theoretical motivation. Validated in adjacent literature (arXiv:2503.18638). Medium confidence in implementation feasibility without modifying data prep files.

---

## Idea 6: SDF Gradient Vector Features as Surface Normal Direction Input

**What it is.** Compute the signed distance function (SDF) gradient vector ∇SDF at each mesh node and provide it as an additional 2D input feature. The gradient of the SDF gives the direction of the nearest surface normal, which encodes local surface orientation and curvature implicitly.

**Why it might help.** Current geometry features include position and possibly SDF magnitude, but not surface normal direction. Surface normals are fundamental to aerodynamic boundary conditions — the no-slip condition and pressure-velocity coupling are both formulated in terms of wall-normal directions. For mesh nodes near the surface, ∇SDF is nearly identical to the wall normal vector. For volume nodes, it points toward the nearest surface, encoding implicit geometry information. This is inspired by the Geometric-DeepONet paper (arXiv:2503.17289), which showed that providing ∇SDF features significantly improved pressure predictions in mesh surrogate tasks by giving the model explicit geometric orientation information.

**Key papers.**
- arXiv:2503.17289 "Geometric-DeepONet" (2025). Shows ∇SDF features improve pressure prediction by providing surface orientation context to mesh surrogates.
- Park et al. (2019) "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation." CVPR. — foundational SDF representation.

**Implementation notes.**
- ∇SDF is a unit vector (after normalization) in 2D: `grad_sdf = (x - x_nearest_surface) / |x - x_nearest_surface|`.
- If SDF values are already precomputed in the dataset, the gradient can be estimated via finite differences on the mesh or directly from the surface projection.
- Alternatively, compute analytically: for each node, find the nearest surface point (already needed for SDF). The vector from node to nearest surface point, normalized, is ∇SDF.
- Feature: append `[grad_sdf_x, grad_sdf_y]` (2 scalars) to each node's feature vector. No architectural change.
- For surface nodes: ∇SDF is the outward surface normal — directly meaningful for boundary condition encoding.
- For volume nodes: ∇SDF gives the "which wall am I closest to and in what direction" information.
- This may be computable from the existing SDF feature in the dataset via numerical differentiation, or from the mesh connectivity.

**Expected impact.** Low-moderate. Geometric features have shown consistent benefits in mesh surrogate tasks. Expected: 1–3% improvement in surface pressure MAE, likely more in tandem configurations where the rear foil is influenced by the front foil's geometry.

**Risk.** Low-medium. The main risk is implementation complexity — computing ∇SDF correctly requires access to the surface mesh. If the dataset provides SDF values per node, finite differences on the mesh graph would be approximate. The student should validate that the computed gradients are smooth and continuous before training.

**Confidence.** Moderate. Validated in Geometric-DeepONet (arXiv:2503.17289). The direct computation from available data depends on what the dataset provides.

---

## Idea 7: Global Condition Token Injection via Cross-Attention

**What it is.** Encode the global flow conditions (AoA, Re, gap, stagger for tandem) as a small set of learnable "condition tokens" and inject them into the Transolver backbone via a cross-attention layer. This is distinct from the current approach of concatenating conditions to node features — it creates a dedicated "conditioning pathway" at the attention level.

**Why it might help.** Currently, Re and AoA enter the model by concatenation to node features, which means they are processed alongside geometric features and must compete for representational capacity in the node feature space. Cross-attention injection (as in Unisolver, ICML 2025) creates a dedicated channel for global conditions that can influence every layer independently. The condition tokens act as "global context" that every node can attend to. This is especially powerful for OOD generalization: if Re shifts by an order of magnitude, a dedicated Re token allows the model to globally rescale its internal physics representation rather than relying on a single scalar mixed into 24-dimensional node features. The Unisolver paper explicitly validates this for CFD surrogates with varying boundary conditions.

**Key papers.**
- Unisolver (ICML 2025) — arXiv:2405.xxxxx (Unisolver: PDE-Unified Transformer Solver). Uses condition token injection with cross-attention for PDE surrogates. Shows strong BC conditioning and OOD generalization.
- Rombach et al. (2022) "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR. — popularized condition token injection via cross-attention in generative models.

**Implementation notes.**
- Create a `ConditionEncoder: nn.Linear(n_conditions, hidden_dim * n_tokens)` where `n_conditions` is the number of global conditions (e.g., 4: Re, AoA, gap, stagger) and `n_tokens = 4`.
- Reshape to `[bsz, n_tokens, hidden_dim]` and inject via cross-attention: `Q = node_features`, `KV = condition_tokens`.
- Add a single cross-attention layer after the existing preprocessing MLP, before the first TransolverBlock.
- Use a lightweight cross-attention (1 head, hidden_dim=128) to keep parameter count low.
- Zero-init the output projection of the cross-attention layer so training starts from the pretrained initialization.
- Alternative simpler variant: just add the condition embedding (via MLP) to every node's feature vector before the first TransolverBlock, without cross-attention. Try this simpler variant first.

**Expected impact.** Moderate. The current Re/AoA conditioning via concatenation works but may be a bottleneck for OOD generalization. Expected: 1–4% improvement on val_ood_re and val_ood_cond. Potentially 2–5% on val_tandem_transfer if gap/stagger tokens are included.

**Risk.** Medium. Cross-attention adds complexity. The simpler "additive embedding" variant should be tried first. If the simpler variant fails, the full cross-attention is less motivated. Also, conditions may already be well-encoded by the existing concatenation approach.

**Confidence.** Moderate. Validated in Unisolver for similar problem class. Mechanism is well-understood.

---

## Idea 8: Zebra-Style In-Context Conditioning at Inference

**What it is.** At inference time, retrieve K nearest training samples (by AoA, Re distance) and prepend their (input, output) pairs as "context" to the model's input. The model learns during training to condition its predictions on similar examples, enabling rapid adaptation to new conditions without gradient updates.

**Why it might help.** val_ood_re shows the largest error (MAE ~31 vs. ~19 for in-dist). The current model must generalize entirely from the training distribution, with no test-time adaptation. In-context conditioning allows the model to "look up" analogous training examples and use them as a reference — effectively creating an implicit interpolation between known solutions. This is the mechanism behind Zebra (arXiv:2410.03437, ICML 2025), which showed strong OOD gains on parametric PDEs by conditioning on similar example trajectories at inference without gradient adaptation. For our problem: if a test sample has Re=4M, retrieving training samples with Re=3M and Re=5M as context gives the model a direct signal for how physics changes with Re.

**Key papers.**
- Serrano et al. (2024/2025) "Zebra: In-Context Generative Pretraining for Solving Parametric PDEs." arXiv:2410.03437. ICML 2025. — core reference. Conditions on K=4 similar training examples at inference. Shows >20% OOD improvement on several PDE benchmarks.
- Min et al. (2022) "Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?" EMNLP 2022. — in-context learning mechanism analysis.

**Implementation notes.**
- During training: for each batch sample, retrieve K=2–4 nearest neighbors from the training set (by L2 distance in (Re, AoA, gap, stagger) space). Concatenate their (node_features, target_output) as additional node sets.
- At inference: same retrieval from training set.
- Architecture change: add a "context attention" layer that allows the query sample's slice tokens to attend to context sample slice tokens. Use cross-attention: `Q = query_slice_tokens`, `KV = context_slice_tokens`.
- The context slot tokens would be computed by a forward pass of the encoder on the context samples (frozen or shared weights).
- Simpler variant: just append the retrieved targets as additional input features to each node (e.g., append Ux, Uy, p from K nearest training samples). This avoids the cross-attention and is a "lazy" in-context approach.
- Start with the simpler variant (appended features) before the full cross-attention architecture.
- Risk of data leakage at test time: be careful that retrieval only uses training set samples, not val/test.

**Expected impact.** High potential, especially for val_ood_re. Zebra showed >20% OOD improvement on PDE benchmarks. The mechanism is directly applicable. Expected: 5–15% improvement on val_ood_re and val_ood_cond if implemented correctly.

**Risk.** High. This is the most complex idea in this list. The simpler "append nearest neighbor outputs" variant is feasible but changes the inference pipeline. The full cross-attention variant requires significant architectural changes. Also, retrieval quality is critical — bad nearest neighbors will hurt. KNN in (Re, AoA) space may not capture the relevant physics similarity. This should be treated as a bold experimental bet.

**Confidence.** Strong theoretical motivation from Zebra (arXiv:2410.03437). High complexity and implementation risk. Low confidence in clean implementation within the current codebase constraints.

---

## Idea 9: GradNorm Adaptive Loss Weighting

**What it is.** Use the GradNorm algorithm (Chen et al., ICML 2018) to automatically balance the surface and volume loss weights by equalizing gradient magnitudes across tasks. This replaces the manually tuned `surf_weight=20` constant with a dynamic, learned weight.

**Why it might help.** The current `surf_weight` hyperparameter was tuned to 20 via grid search (PRs #455, #480, #551). This is a static balance that cannot adapt as the training progresses and the relative difficulty of surface vs. volume prediction changes. PR #736 tried loss-ratio balancing (adjusting weights based on ratio of surface to volume loss magnitudes) but this failed — it over-amplified surface loss when volume was already low, destabilizing training. GradNorm is fundamentally different: it equalizes the gradient norms of each task's loss with respect to shared parameters, not the loss values themselves. This prevents any one task from dominating the gradient signal. The mechanism: maintain a learnable weight per task; update weights such that gradient norms match a target (set to the average norm). This is a more principled approach than loss-ratio balancing and avoids the instability PR #736 encountered.

**Key papers.**
- Chen et al. (2018) "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks." ICML 2018. arXiv:1711.02257. — core reference. Shows consistent improvement over fixed loss weights in multi-task settings.
- Liu et al. (2021) "Conflict-Averse Gradient Descent for Multi-task Learning." NeurIPS 2021. — related gradient-based MTL balancing.

**Implementation notes.**
- Treat surface MAE loss and volume MAE loss as two tasks in GradNorm framework.
- Maintain learnable log-weights `log_w_surf` and `log_w_vol` (initialized to log(20) and log(1) respectively, matching current defaults).
- Each training step: compute the gradient norms of `L_surf` and `L_vol` w.r.t. the last shared layer's parameters (e.g., the final TransolverBlock's output projection).
- GradNorm update: `w_i = w_i * (target_grad_norm_i / current_grad_norm_i)` where target is the exponentially weighted average of gradient norms times a relative loss weight.
- Renormalize so weights sum to the current total weight (surf + vol).
- Use `alpha = 0.12` (standard GradNorm hyperparameter). Do not tune this.
- Clip weight updates to prevent rapid oscillation.
- Key: GradNorm updates the weights, NOT the model parameters, so it runs as a meta-update after the main backward pass.
- Implementation is ~40 lines of additional code in the training loop; no architectural changes.

**Expected impact.** Low-moderate. Dynamic weighting should improve training stability and allow the model to focus surface loss appropriately throughout training. Expected: 1–3% surface MAE improvement, mainly through better training dynamics.

**Risk.** Medium. GradNorm requires computing gradient norms at each step (extra backward pass per task), increasing training time by ~20–30%. May hit the 30-minute timeout earlier. The student should monitor wall-clock time carefully and potentially reduce max_epochs to compensate. Also, GradNorm is sensitive to the choice of "shared layers" — selecting a too-early layer may give noisy gradient norm estimates.

**Confidence.** Strong evidence from multi-task learning literature (Chen et al. 2018). Moderate confidence in transfer to CFD surrogate setting. Clearly distinct from the failed PR #736 approach.

---

## Idea 10: Multi-Scale Feature Aggregation via Intermediate Layer Skip Connections

**What it is.** Add learnable weighted skip connections from each intermediate TransolverBlock's output to the final prediction head. This creates a "deep supervision" / feature pyramid that allows the model to aggregate representations at multiple levels of abstraction — analogous to FPN (Feature Pyramid Network) in computer vision.

**Why it might help.** The current architecture is sequential: node features pass through N TransolverBlock layers and only the final layer's output feeds the prediction head. Earlier layers capture local/low-frequency features (geometry, proximity to wall) while later layers capture global/high-frequency features (pressure gradients, wake effects). For pressure prediction, both scales matter: the suction peak is a sharp, local feature (high frequency) while the overall pressure distribution is a global feature (low frequency). A learnable mixture across layers allows the model to select the appropriate scale for each output channel. In image segmentation, U-Net and FPN showed this is one of the most reliable improvements for tasks requiring multi-scale accuracy. The existing preprocess skip connection (PR #774) shows this architecture is receptive to skip connections.

**Implementation notes.**
- After each TransolverBlock `i`, project the output to the prediction dimension: `skip_head_i = Linear(hidden_dim, 3, bias=False)` — zero-init weights.
- Add all skip outputs to the final prediction: `final_pred += sum(skip_head_i(block_i_output) * learnable_scale_i)`.
- Learnable scales `scale_i` initialized to 0.0 (identity start), allowing gradual activation.
- The existing preprocess skip (PR #774) is analogous — this extends it to intermediate layers.
- Use all N_block intermediate outputs. For N_blocks=8, this adds 8 small Linear layers (~8 × 3 × 128 = 3072 parameters) — negligible.
- Alternative: instead of direct prediction from each layer, aggregate intermediate hidden states into a weighted sum before the final prediction head. This changes the input to the final head from `block_N_output` to `sum_i(alpha_i * block_i_output)` with learned alpha_i.

**Expected impact.** Low-moderate. The zero-init start means worst case is the current result. Expected: 1–3% surface MAE improvement if intermediate layers contain useful partial representations.

**Risk.** Low. Zero-init prevents disruption. The main risk is that the additional parameters don't converge within the epoch budget. Monitor val_loss trajectory for the first 20 epochs — if it's not declining, close early.

**Confidence.** Moderate. Strong precedent from FPN/U-Net in vision. The preprocess skip (PR #774) success is supporting evidence. No direct CFD surrogate validation found.

---

## Idea 11: Tandem Configuration Curriculum — Gradual Reintroduction

**What it is.** Replace the current data curriculum (skip tandem samples for first 10 epochs, then include at full weight) with a smoother ramp: tandem sample weight grows from 0 to 1.0 linearly over epochs 10–30. This was explicitly suggested in PR #768 follow-up notes.

**Why it might help.** The abrupt reintroduction of tandem samples at epoch 10 creates a distribution shift shock — the gradient signal suddenly changes from single-foil to mixed. This can cause instability in the slice routing and pressure prediction. A gradual ramp allows the model to adapt incrementally, maintaining a stable optimization trajectory. In curriculum learning theory (Bengio et al. 2009), smooth transitions between curriculum stages consistently outperform abrupt switches. The tandem surface MAE (~41–43) is still the highest error track — better curriculum design specifically targeting the tandem distribution shift is well-motivated.

**Key papers.**
- Bengio et al. (2009) "Curriculum Learning." ICML 2009. — foundational curriculum learning paper showing smooth task introduction outperforms abrupt switches.
- Soviany et al. (2022) "Curriculum Learning: A Survey." International Journal of Computer Vision. — comprehensive survey including analysis of pacing functions.

**Implementation notes.**
- Current: `if epoch < 10: skip tandem samples`. New: multiply tandem sample loss weights by `ramp(epoch) = min(1.0, max(0, (epoch - 10) / 20.0))`.
- Implementation: in the per-sample loss weighting code, multiply tandem sample losses by `tandem_ramp_weight` which starts at 0 (epoch 0–10), then linearly ramps to 1.0 (epoch 30).
- Keep the existing `1.5× tandem surface boost` (PR #616) — multiply it by the ramp factor. The 1.5× boost is still active once ramp reaches 1.0.
- Monitor `val_tandem_transfer` surface MAE across training to see if the transition is smoother.
- Also try an alternative: cosine annealing ramp (from 0 to 1) over the same epoch range, which is smoother than linear.

**Expected impact.** Low. This is an incremental training dynamics improvement. Expected: 0.5–2% improvement on val_tandem_transfer. Minimal effect on other tracks.

**Risk.** Low. Pure training dynamics change with no architectural modification. If it doesn't help, it also won't hurt — the endpoint is the same loss landscape.

**Confidence.** Moderate. Well-motivated theoretically. Specifically suggested as a follow-up in PR #768 student notes — the student observed evidence that the abrupt transition caused instability.

---

## Idea 12: Pretraining on Panel Method / XFOIL Solutions (Transfer Learning)

**What it is.** Pre-train the Transolver backbone on a large dataset of cheap panel method or XFOIL thin-airfoil solutions (analytically computed, infinite data), then fine-tune on the TandemFoilSet RANS data. The model learns the "shape" of aerodynamic pressure distributions first from abundant cheap data, then refines with accurate viscous data.

**Why it might help.** The TandemFoilSet training set is finite and expensive (RANS simulations). Panel method solutions are essentially infinite in number (any geometry/AoA combination can be computed in milliseconds) and capture the dominant pressure distribution structure. Pre-training on panel method data gives the model a strong aerodynamic prior — it learns that Cp is smooth except near stagnation and leading edge, that pressure decreases on the suction side and increases on the pressure side, and that trailing edge effects are important. Fine-tuning then corrects for viscosity and separation. This is the POSEIDON approach (multi-fidelity pretraining) applied to 2D aerodynamics. It is especially motivated by the fact that val_tandem_transfer shows the highest error — the model has never seen abundant tandem configurations during RANS training, but could see thousands of inviscid tandem configurations during pretraining.

**Key papers.**
- POSEIDON (2024/2025) — multi-fidelity pretraining for PDE surrogates. Pretrains on coarse/cheap solutions, fine-tunes on fine/expensive. Shows 30–50% improvement on OOD evaluation.
- Herde et al. (2024) "POSEIDON: Efficient Foundation Models for PDEs." arXiv:2405.19101. — core reference.

**Implementation notes.**
- Generate a panel method dataset offline: N=50,000 configurations (AoA in [-10, 20] degrees, NACA 4-series geometries, single and tandem) using a Python panel method solver (e.g., `AeroPy`, `PyVortex`, or a simple Hess-Smith panel code).
- Panel method outputs Cp on surface nodes only — cannot predict volume velocity field. Pre-train on surface-only prediction first.
- Architecture: unchanged. Just change training data source.
- Training: pre-train for K epochs on panel dataset (with surface-only loss), then fine-tune on TandemFoilSet with full surface+volume loss.
- Practical challenge: panel method geometry discretization may not match TandemFoilSet mesh — need interpolation or consistent mesh definition.
- Alternative simpler variant: generate panel solutions on the SAME meshes as TandemFoilSet (resample), so no interpolation is needed. This requires running a panel solver on the TandemFoilSet geometries.
- This is the highest complexity/risk idea in this list. It is a multi-day engineering task, not a one-day experiment.

**Expected impact.** High potential — POSEIDON showed 30–50% OOD improvements. If surface MAE on val_tandem_transfer drops from 41 to 30, this would be the largest single improvement in the research programme.

**Risk.** Very high. Engineering complexity is substantial. Panel method pretraining data must be generated, stored, and loaded within the existing training infrastructure. The mesh mismatch problem is non-trivial. The 30-minute timeout severely limits fine-tuning epochs. This should only be assigned to a student who is comfortable with aerodynamics code and has already implemented panel method features. Consider assigning as a 2-week exploratory task rather than a standard 30-minute experiment.

**Confidence.** Strong theoretical motivation from POSEIDON and transfer learning literature. Very low confidence in successful implementation within standard experiment constraints.

---

## Priority Ranking

Sorted by expected impact × implementation feasibility (highest priority first):

1. **Idea 3** — Auxiliary AoA head: low complexity, strong analogy to successful PR #780, expected tandem/OOD gains.
2. **Idea 1** — GQA 2 groups: low complexity, well-validated in LLMs, directly extends current MQA architecture.
3. **Idea 7** — Condition token injection (simple additive variant): medium complexity, validated in Unisolver, targets OOD generalization.
4. **Idea 9** — GradNorm adaptive weighting: medium complexity, principled alternative to failed PR #736, directly distinct.
5. **Idea 4** — Asymmetric quantile Cp loss: low complexity, Kaggle-standard technique, targets suction-side accuracy directly.
6. **Idea 5** — Panel method Cp feature: medium complexity, strong physics motivation, validated in arXiv:2503.18638.
7. **Idea 10** — Multi-scale intermediate skips: low complexity, zero-init safety, supported by preprocess skip success (PR #774).
8. **Idea 11** — Tandem curriculum ramp: very low complexity, suggested by student in PR #768 follow-ups.
9. **Idea 2** — SE on slice tokens: low complexity, orthogonal to existing node SE (PR #772).
10. **Idea 6** — SDF gradient features: medium complexity, validated in arXiv:2503.17289.
11. **Idea 8** — Zebra in-context conditioning: high complexity, high potential on OOD tracks, treat as bold experimental bet.
12. **Idea 12** — Panel method pretraining: very high complexity, highest potential impact, assign only if engineering resources available.

## Exhausted / Do Not Repeat

Full list of confirmed-tried approaches (from merged PRs and in-flight experiments):
- Architecture replacements (GNOT, Galerkin, full replacement)
- Post-norm LayerNorm (catastrophic)
- Cosine T_max sweep, eta_min variants, Lookahead, weight decay sweeps
- Chord-position features, augmentation annealing, slice routing perturbations
- MSE→L1 transition, surface pressure 2× weight, multi-scale coarse pooling
- Xavier→orthogonal init, gain=sqrt(2), gain=1.0
- MQA shared K/V (PR #513), cosine similarity attention (PR #618), per-head attn_scale, differential LR
- Coordinate-conditioned slice assignment (PR #741), slice-token residual bypass (PR #800)
- SE block on node feature MLP (PR #772), Sandwich LN (PR #775), preprocess skip (PR #774)
- Progressive resolution, data curriculum abrupt (PR #768), late EMA (PR #679)
- Surf_weight ramp 5→30, tandem 1.5× boost (PR #616), per-channel noise (PR #633)
- Per-sample normalization (mixed results), auxiliary Re prediction (PR #780)
- Learnable placeholder scale+shift

**In-flight (do NOT assign):** DID Streamwise Feature, GMSE Gradient-Weighted Pressure Loss, Potential Flow Residual Loss, Mirror Symmetry Augmentation, Continuity PDE Loss, Circulation Lift Feature, Wake Centerline SDF, Shortest Vector Feature.
