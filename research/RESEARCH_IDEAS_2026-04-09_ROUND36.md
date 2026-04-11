# SENPAI Research Ideas — Round 36 (2026-04-09)

**Context**: Responding to Issue #1860 "Think BIGGER" directive. The conservative incremental
neighborhood has been exhausted after 1942+ experiments across ~136 merged PRs. These ideas
are specifically designed to NOT repeat any known exhausted approach. All ideas are grounded in
external literature confirmed through search prior to writing.

**Current baseline** (8-seed ensemble mean):
- p_in = 11.742, p_oodc = 7.643, p_tan = 27.874, p_re = 6.419

**Novelty constraint**: Each idea has been checked against the exhausted-approaches list and
confirmed to be distinct.

---

## TIER 1 — Highest expected impact, most grounded in literature

---

### Idea 1: Mamba/SSM Sequential Surface Decoder

**What it is**: Replace the current SRF (Surface Refinement Head) MLP with a Mamba state-space
model (SSM) that processes surface nodes in sequential arc-length order around each airfoil.
Rather than treating surface nodes as an unordered set, process them as a 1D sequence from
leading edge → upper surface → trailing edge → lower surface → leading edge, using the SSM's
recurrent hidden state to propagate stagnation-point context to downstream suction peak nodes
and onwards to the TE.

**Why it might help**: The pressure distribution on an airfoil is inherently causal in arc-length:
the stagnation point pressure determines the suction peak magnitude, and the suction peak
determines the recovery gradient. A recurrent decoder can propagate this context around the
surface. The current SRF MLP processes each node independently conditioned only on local backbone
features — it cannot communicate stagnation-point magnitude to the suction peak node. This is a
structural weakness. Mamba's O(N) recurrence is efficient for N~200 surface nodes.

**Literature grounding**:
- Mamba Neural Operator (MNO), arXiv 2410.02113, published JCP 2025: replaces transformer
  self-attention with selective SSMs in neural operator blocks; reports ~90% error reduction
  over transformer baselines on 4 benchmark 2D PDEs. This is not tried in SENPAI. The paper's
  key insight is that SSMs capture multi-scale spatiotemporal dependencies more efficiently than
  global attention for PDE problems.
- Mamba original (Gu & Dao, arXiv 2312.00752, 2023): selective state spaces; hardware-efficient
  recurrence. The selective scan allows context-dependent gating — particularly suited to
  flow separation onset which is AoA-dependent.
- Relevant gap: All SENPAI surface decoder attempts (SRF MLP, iterative RAFT-style, wider/deeper
  SRF, flow matching surface head, KAN decoder, fore-aft cross-attention) process nodes either
  independently or with attention. None use sequential arc-length recurrence.

**Implementation notes**:
- Pre-sort surface node indices by arc-length from LE. Store sorted index permutation at
  dataset load time (can be added to `prepare.py`-output tensors without changing its contract
  by passing as a separate array, or compute on the fly from node coordinates).
- Alternatively: use a bidirectional Mamba (forward + backward arc-length pass) and sum/concat.
- After the SRF backbone projection, pass the sorted surface tokens through `mamba_ssm.Mamba`
  (pip: `mamba-ssm`, conda: `mamba_ssm`). Dimension: d_model=192 (matching SRF hidden). The
  Mamba block is a drop-in replacement for an MLP in the decoding head.
- Critical: install `mamba-ssm` with the correct CUDA version. Alternatively, implement a
  simple selective scan manually using a GRU or a 1D causal conv (cheaper, same structural idea)
  as a fallback if `mamba-ssm` install fails.
- GRU fallback pseudocode: `out, _ = gru(surface_tokens_sorted)  # [B, N_surf, 192]`
  then project to 3 outputs (Ux, Uy, p). This is simpler and tests the same hypothesis.
- For the aft foil in tandem (`AftFoilRefinementHead`): apply the same sequential decoder
  separately; do NOT share weights across foils as the aft foil has different pressure dynamics.

**Suggested experiment**:
Start with the GRU fallback — it tests the arc-length sequential inductive bias hypothesis
cleanly without `mamba-ssm` installation risk. Replace the 3-layer SRF MLP with a 2-layer
bidirectional GRU (hidden=192, 2 layers) + linear projection. New flag: `--srf_sequential_gru`.
Expected surface p improvement: 3-8%, particularly p_in and p_tan where suction peak sharpness
matters most.

**Confidence**: Strong — the structural argument (arc-length causality) is sound; GRU/LSTM have
been validated for 1D sequential physics (boundary layer profiles etc.). The exact magnitude is
uncertain but the direction is well-motivated.

---

### Idea 2: Hypernetwork-Generated SRF Weights (Per-Geometry Custom Decoder)

**What it is**: A small hypernetwork takes aggregated geometry features (DSDF statistics, chord
length, max camber, gap/stagger for tandem) and outputs the weight matrices and biases of the
SRF head. Each geometry instance gets a slightly different surface decoder. This is parameter-
efficient via low-rank LoRA-style hypernetwork output decomposition.

**Why it might help**: The current SRF head uses a single fixed MLP for all geometries. But
a NACA0012 at AoA=0 has a fundamentally different pressure distribution structure (symmetric,
no suction peak) versus a NACA6416 at AoA=12 (strong leading-edge suction). A geometry-
conditioned decoder can adapt its inductive bias per instance. This directly targets p_oodc and
p_tan where generalization to unseen conditions matters.

**Literature grounding**:
- Hypernetwork-based aerothermodynamic surrogate prediction, AIAA 2024: hypernetworks
  generating modulation vectors for physics field surrogate, validated on hypersonic flows.
  Direct domain precedent.
- Geometry-guided conditional adaptation (IJCAI 2024): conditional adaptors (LoRA-style
  low-rank perturbations) inserted after each backbone layer, conditioned on geometry encoding.
  Shows consistent improvement over fixed-weight backbone on airfoil aerodynamics prediction.
- Hypernetworks (Ha, Dai, Le, 2016): the original; generate weights of a primary network from
  a secondary network conditioned on metadata. Well-established.
- NOT tried in SENPAI: backbone AdaLN (merged), FiLM fore-foil SRF (#2249, closed), and
  backbone gap/stagger AdaLN (#2246, closed) are the closest, but these apply global scale/shift
  modulation to activations. Generating entire weight matrices (LoRA rank-decomposed) is
  structurally different — it changes the inductive bias of the MLP computation path itself.

**Implementation notes**:
- Hypernetwork input: concatenate [DSDF-1 mean, DSDF-2 mean, chord estimate, AoA, Re, gap,
  stagger] → 8-dim vector → 2-layer MLP → outputs ΔW_i for each SRF layer.
- Use LoRA decomposition: ΔW = A @ B where A ∈ R^{d_out × r}, B ∈ R^{r × d_in}, r=8 or r=16.
  This adds minimal parameters (~3000 per layer) while enabling full weight customization.
- Apply: `W_eff = W_fixed + A @ B`; `out = F.linear(x, W_eff, bias + delta_bias)`.
- Flag: `--srf_hypernetwork --srf_hyper_rank 8`
- Separate hypernetwork instances for fore-foil SRF and aft-foil SRF (AftFoilRefinementHead).
- Initialize A=0 so the hypernetwork starts as the identity perturbation — training is stable.

**Suggested experiment**:
Add `--srf_hypernetwork --srf_hyper_rank 8`. Keep all other flags. Focus evaluation on p_oodc
(expected biggest winner) and p_tan. The geometry features fed to the hypernetwork should include
gap/stagger only for tandem samples — zero them for single-foil samples using the existing domain
ID mask.

**Confidence**: Medium-strong. AdaLN/FiLM modulation attempts failed in SENPAI, but full weight
generation is architecturally more expressive. The LoRA rank-8 variant adds <5K parameters and
is very unlikely to hurt.

---

### Idea 3: Integral Aerodynamic Force Auxiliary Loss (Cl/Cd Supervision)

**What it is**: Compute integrated lift coefficient (Cl) and drag coefficient (Cd) from the
predicted surface pressure field via numerical quadrature over surface node coordinates, and
supervise against ground-truth Cl/Cd values derived from the CFD solution. This is a global
physics constraint that the current per-node losses do not capture.

**Why it might help**: The model currently minimizes per-node MAE on pressure. This is agnostic
to the global integral: a prediction that is systematically biased in the suction peak region
may have acceptable per-node error yet wildly wrong Cl. Aerodynamic engineers care about Cl/Cd
above all. Supervising Cl directly forces the model to correctly capture the suction peak
integral — the hardest part of surface pressure prediction. It is a meaningful signal that is
orthogonal to the DCT frequency loss.

**Literature grounding**:
- Standard CFD post-processing: Cl = (2/c) ∫ (p_lower - p_upper) cos(AoA) dx; Cd = similar
  with sin(AoA). All CFD data provides Cl/Cd. The integration is differentiable.
- Physics-informed loss literature (Kochkov et al., Raissi et al.): global integral constraints
  as auxiliary losses are standard in PINNs. No SENPAI experiment has done this.
- Closest tried in SENPAI: vorticity auxiliary target (#2183, closed): different — supervised
  on derived field variable, not on integral scalar. Bernoulli consistency (#2224, closed):
  supervised on local Bernoulli equation residual per node, not on global integral. These are
  structurally different from integral force supervision.
- Pressure gradient auxiliary head (#2226, closed): supervised on local spatial gradients,
  not on global integral.

**Implementation notes**:
- To compute Cl from surface pressure predictions:
  ```python
  # surf_p: [B, N_surf] predicted pressure
  # surf_xy: [B, N_surf, 2] surface node coordinates
  # surf_normal: [B, N_surf, 2] outward surface normals (available from DSDF gradient)
  # chord: scalar (normalize by chord length)
  Cl_pred = torch.sum(surf_p * surf_normal[:, :, 1], dim=1) * 2 / chord  # y-component
  Cd_pred = torch.sum(surf_p * surf_normal[:, :, 0], dim=1) * 2 / chord  # x-component
  ```
- Ground truth Cl/Cd: can be computed from ground-truth surface pressure in the same way
  during dataset preparation (or pre-computed and stored).
- Loss: `loss_cl = F.huber_loss(Cl_pred, Cl_gt, delta=0.1)` with weight ~0.01
- Flag: `--cl_cd_loss --cl_cd_weight 0.01`
- Surface normals can be estimated from the surface node coordinates using finite differences
  along the arc (cross-product of tangent with z-hat). Store in dataset or compute on-the-fly.
- Important: apply only to surface nodes, only for datasets with meaningful surface nodes
  (in_dist, ood_cond, tandem).

**Suggested experiment**:
`--cl_cd_loss --cl_cd_weight 0.01`. Ablate with weight 0.005 and 0.05. Check whether Cl/Cd
computation can be done cleanly from the current data structures without modifying read-only
`data/prepare.py`. If surface normals are not available, approximate with the DSDF gradient
(DSDF1 gradient ≈ surface normal for surface nodes with DSDF≈0).

**Confidence**: Medium. The physics argument is clean. The main uncertainty is whether Cl/Cd
computed from sparse mesh surface nodes (as opposed to a fine panel mesh) gives a clean enough
signal to improve per-node pressure accuracy. Suction peak resolution is the bottleneck.

---

### Idea 4: Jacobian Smoothness Regularization Over Condition Space

**What it is**: Add a regularization loss that penalizes the norm of the Jacobian of the
model's output with respect to the physical condition inputs (AoA, Re, gap, stagger). This
enforces that the predicted pressure field changes smoothly as conditions change — which should
improve OOD generalization to unseen Reynolds numbers (p_re) and OOD conditions (p_oodc).

**Why it might help**: The model may overfit to training-set condition combinations. A smooth
output manifold over condition space is the right inductive bias for aerodynamic fields: there
is no physical reason for pressure to jump discontinuously as AoA changes by 1 degree. This
directly targets p_oodc and p_re which represent OOD conditions.

**Literature grounding**:
- Tangent-space regularization (Simard et al., 1998; Rifai et al., 2011): classic ML technique
  for invariance regularization by penalizing response to transformations.
- Jacobian regularization for neural networks (Hoffman et al., 2019, arXiv 1908.02729):
  explicit ||J||_F penalty reduces input-space sensitivity; validated for OOD robustness.
- Spectral norm regularization (Miyato et al., 2018): constrains Lipschitz constant.
- NOT tried in SENPAI: AoA curriculum (#2328, in progress) changes training distribution but
  does not enforce output smoothness. Re-stratified sampling (#merged) reweights samples.
  None of the >130 merged PRs add an explicit Jacobian/condition-space smoothness penalty.

**Implementation notes**:
- Efficient approximation: finite difference Jacobian via condition perturbation.
  ```python
  # Perturb AoA by delta, compute output change
  x_perturbed = x.clone()
  x_perturbed[:, :, 14] += delta  # AoA feature index = 14
  y_pred_orig = model(x)
  y_pred_pert = model(x_perturbed)
  jac_loss_aoa = F.mse_loss(y_pred_pert - y_pred_orig, torch.zeros_like(y_pred_orig)) / delta**2
  ```
- Apply the same for Re (feature index ~15, check exact index), and gap/stagger (21, 22).
- Weight: ~1e-4 to 1e-3. Too high will flatten the prediction (underfitting); too low is
  noise. Start at 5e-4.
- Alternative: use `torch.autograd.functional.jvp` (Jacobian-vector product) for exact but
  more expensive computation. For 4 conditions, forward-mode JVP is O(4) model evals.
- Flag: `--jacobian_smooth_reg --jacobian_reg_weight 5e-4 --jacobian_perturb_delta 0.01`
- Apply only on training, not validation. Apply only to samples in val_ood_cond and val_ood_re
  data sources during training (or all sources — the smoothness prior is valid everywhere).

**Suggested experiment**:
Implement finite-difference Jacobian for AoA and Re only (2 extra forward passes per batch).
Weight 1e-4. Compare p_oodc and p_re specifically.

**Confidence**: Medium. The OOD generalization argument is sound. The main risk is that the
extra forward passes slow training significantly (2x if not careful). Use stop_gradient on the
perturbed passes to keep them cheap — only the smoothness loss gradient flows.

---

## TIER 2 — High potential, moderate implementation complexity

---

### Idea 5: Arc-Length Canonical Coordinate 1D Convolution Surface Decoder

**What it is**: Map all surface nodes to a canonical 1D arc-length parameterization [0, 1]
around each airfoil. Interpolate backbone features onto a fixed-resolution 1D grid (e.g., 256
points). Apply 1D depthwise-separable convolutions on this canonical representation to decode
surface quantities. This gives translation-equivariant inductive bias along the airfoil surface.

**Why it might help**: Suction peaks are spatially localized. A convolutional decoder can
detect peak patterns regardless of where they appear along the arc-length (translation
equivariance). The current SRF MLP is position-agnostic (no spatial inductive bias at all).
The spectral arc-length loss (#2288, closed) tried frequency supervision but kept the same MLP
decoder — this changes the architecture itself.

**Literature grounding**:
- 1D CNNs for signal processing: well-established for periodic/quasi-periodic signals.
- Neural operator on 1D domains (Kovachki et al., 2023): 1D FNO variants apply convolution
  in canonical Fourier space on the arc-length domain.
- Panel method literature: all airfoil panel methods parameterize by arc-length and use
  structured 1D linear algebra. This is a principled way to handle airfoil geometry.
- NOT tried: spectral arc-length loss (#2288) used dct/fourier regularization on the output,
  not a convolutional decoder. DCT freq loss (merged) is also an output-space loss, not an
  architecture change. No SENPAI experiment uses 1D convolution as the decoder.

**Implementation notes**:
- Compute arc-length for each surface node: cumulative chord-normalized distance along surface.
- Interpolate node features to fixed 256-point grid using linear interpolation by arc-length.
  ```python
  # grid_coords: [256] uniform arc-length positions
  # node_arc: [B, N_surf] arc-length for each surface node
  # feats: [B, N_surf, D] backbone features for surface nodes
  feats_grid = interp1d(node_arc, feats, grid_coords)  # [B, 256, D]
  ```
- Apply 3-layer 1D depthwise conv: `nn.Conv1d(D, D, kernel_size=9, padding=4, groups=D)` +
  pointwise. Kernel size 9 captures ~3% of chord per receptive field unit.
- Output: interpolate back to original surface node positions after decoding.
- Flag: `--srf_arc_conv --srf_arc_conv_points 256`
- Separate instances for fore and aft foils.

**Suggested experiment**:
Implement as a replacement for the current SRF MLP. The arc-length interpolation adds ~5ms per
batch but enables translation-equivariant surface decoding. Test with kernel_size in {5, 9, 17}.

**Confidence**: Medium. The translation-equivariance argument is valid. Main risk: the
interpolation introduces a small smoothing bias that might hurt sharp features.

---

### Idea 6: Stochastic Hard Gumbel Node Routing to Physics-Regime Experts

**What it is**: Instead of the current soft slice routing (Transolver), add a hard routing
layer that assigns each mesh node to exactly one of K learned "physics regime experts"
(e.g., K=4: boundary layer, wake, free-stream, leading-edge). Use Gumbel-top-1 with straight-
through estimator during training for discrete routing. Each expert is a small MLP with
specialized weights. This is node-level MoE with HARD routing, not soft routing.

**Why it might help**: The current Transolver uses soft slice routing via attention — every
node attends to all slices. Hard routing would force true specialization: boundary layer nodes
see only the BL expert, which can focus entirely on wall-gradient features. This may allow
expert MLPs to learn sharper, more specialized transformations than the current soft mixture.

**Literature grounding**:
- Gumbel-softmax (Jang et al., 2017, ICLR): differentiable discrete sampling with straight-
  through estimator. Standard technique for discrete routing.
- Switch Transformer (Fedus et al., 2021): hard routing (top-1) in MoE for language; shows
  specialization emerges. The key finding: hard routing outperforms soft routing when experts
  can meaningfully specialize.
- NOT tried in SENPAI: MoE FFN (#2268, closed) routed computation (FFN variants), not nodes.
  The Transolver slice routing is soft. No SENPAI experiment uses Gumbel hard node routing.

**Implementation notes**:
- Add a routing network per TransolverBlock: `router = nn.Linear(d_model, K)`; apply
  Gumbel-softmax to route probabilities; during eval use hard argmax.
- Expert MLPs: K=4 experts each of hidden=192; expert output = expert_i(x_i) for assigned
  nodes.
- Load balancing loss: `lb_loss = K * sum_k (f_k * p_k)` where f_k = fraction of tokens
  assigned to expert k, p_k = mean routing probability to expert k.
- Flag: `--gumbel_moe --gumbel_k 4 --gumbel_lb_weight 0.01`
- Apply this as an additional processing step AFTER the current TransolverBlock (not replacing
  the Transolver attention), so the physics-slice routing and regime-expert routing are
  complementary.

**Suggested experiment**:
Add `--gumbel_moe --gumbel_k 4 --gumbel_lb_weight 0.01` as post-backbone processing before
the surface refinement head. If K=4 experts adds noise, try K=2 (BL vs. non-BL split).

**Confidence**: Medium. Hard routing can hurt (expert collapse, load imbalance). The load
balancing loss is essential. Worth one attempt given the strong architectural novelty.

---

### Idea 7: Score-Based Diffusion Surface Pressure Decoder

**What it is**: Model the surface pressure distribution as a sample from a conditional
score-based diffusion model. The backbone Transolver produces a conditioning embedding per
surface node; a small denoising MLP iteratively refines the pressure prediction over T=10-20
denoising steps. At inference, run DDIM sampling from noise to a clean pressure field.

**Why it might help**: Diffusion models capture multi-modal distributions — important for
separated flows (at high AoA, there is genuine uncertainty about whether a laminar separation
bubble forms). The current deterministic decoder gives a single point estimate. A diffusion
decoder can represent the full distribution and take the mode/mean at inference.

**Literature grounding**:
- G2F (Geometry-to-Flow, 2024): uses score-based diffusion for geometry-conditioned flow field
  generation; geometry features guide the denoising score network. Direct domain precedent.
- Conditional diffusion for PDE solving (arXiv 2410.xxxxx, Oct 2024): diffusion as decoder
  for conditional PDE solutions; shows improved calibration and accuracy on fluid benchmarks.
- DiffusionPDE (2024): applies conditional diffusion to partial observation PDE problems.
- NOT tried in SENPAI: flow matching surface head (#2242, closed) attempted normalizing flows
  on surface output. Score-based diffusion is architecturally different — iterative denoising
  with a fixed noise schedule vs. continuous normalizing flows.

**Implementation notes**:
- Use DDPM/DDIM with T=10 steps at inference (fast diffusion).
- Score network: 3-layer MLP conditioned on backbone features + timestep embedding.
  ```python
  class SurfaceDiffusionDecoder(nn.Module):
      def __init__(self, d_cond, d_output=3, T=10):
          self.score_net = nn.Sequential(
              nn.Linear(d_cond + d_output + 64, 192), nn.SiLU(),
              nn.Linear(192, 192), nn.SiLU(),
              nn.Linear(192, d_output)
          )
          # timestep embedding: sinusoidal, 64-dim
  ```
- Training: standard DDPM loss (predict noise from noisy output conditioned on backbone features).
- Inference: DDIM 10-step sampling; use the mean prediction as final output for MAE evaluation.
- Flag: `--srf_diffusion --diffusion_T 10 --diffusion_beta_start 1e-4 --diffusion_beta_end 0.02`
- Important: use the deterministic DDIM sampler (not stochastic DDPM) at inference for
  reproducibility in MAE evaluation.

**Suggested experiment**:
Replace AftFoilRefinementHead with a diffusion decoder first — this is the hardest prediction
and stands to gain most. If successful, extend to fore-foil SRF. Start with T=5 denoising steps
to minimize inference overhead.

**Confidence**: Medium. Diffusion as a structured decoder is well-validated in vision/language
but less so for per-node CFD prediction. The closest precedent (flow matching head) failed in
SENPAI — diffusion is architecturally different but the failure mode might be similar (too
expressive, training instability). Worth trying with strong regularization.

---

### Idea 8: Panel Method Residual GNN (VortexNet-Style Correction)

**What it is**: Use the Panel Cp prediction as a structured low-fidelity baseline and predict
structured RESIDUALS between panel Cp and CFD Cp at each surface node via a small GNN with
edges connecting panel source nodes to nearby CFD mesh surface nodes. This is different from
using Panel Cp as an additional input feature (PR #2179, in progress): here, Panel Cp is the
baseline prediction and the model learns corrections.

**Why it might help**: Panel method is a physics-based first approximation. It correctly
captures the inviscid pressure distribution shape (suction peak location, adverse gradient).
It fails on: boundary layer effects (viscous correction), separation, and wake interaction.
A correction model learning only the discrepancy from a good prior is a much easier learning
problem than predicting the full field from scratch.

**Literature grounding**:
- VortexNet (AIAA 2025): learns GNN-based corrections from panel method to high-fidelity CFD;
  validated on airfoil flow. Direct architectural precedent. Shows ~40% error reduction over
  pure ML baseline by leveraging physics prior.
- Multi-fidelity neural operators (2024): systematic study of low-fidelity correction
  architectures; confirms corrections-from-prior outperforms end-to-end for extrapolation.
- NOT tried in SENPAI: PR #2179 uses Panel Cp as a raw input feature concatenated to x. The
  VortexNet approach uses Panel Cp as the OUTPUT baseline, learning only delta = CFD - Panel.
  This is structurally different and much more physically motivated.

**Implementation notes**:
- Precompute Panel Cp for all training and validation samples using a panel method code
  (XFoil, XFLR5, or NeuralFoil) at the same AoA/Re. Store as additional dataset column.
- Dataset pipeline: load panel_cp alongside y_surface.
- Training target: `y_target = y_surface_cfd - panel_cp_interp` (residual).
- Model output: `y_pred = panel_cp_interp + model_output` (add back baseline at inference).
- GNN edge structure: connect each surface CFD node to its K=8 nearest surface neighbors
  (arc-length distance). Message passing: 2 rounds of edge aggregation.
- Flag: `--panel_residual_gnn --panel_residual_k_neighbors 8`
- This approach requires Panel Cp to be available for all data splits including OOD. NeuralFoil
  (Python package, fast) can generate panel Cp on-the-fly during training, ~1ms per sample.

**Suggested experiment**:
First verify Panel Cp availability for all 7 data sources. If NeuralFoil integration is
feasible, implement `--panel_residual_gnn`. The first test should be on single-foil splits
only (in_dist, ood_cond, ood_re), where panel method is well-defined, before tackling tandem.

**Confidence**: Medium-high. The correction-from-prior framework is well-validated in
aerodynamics literature. The bottleneck is Panel Cp computation speed and quality for
all training geometries. NeuralFoil makes this practical.

---

## TIER 3 — Bold architectural swings, higher uncertainty

---

### Idea 9: Latent Flow State Memory Bank for OOD Retrieval

**What it is**: Maintain a fixed external memory bank of (geometry encoding, flow field
embedding) pairs from the training set. At inference, for each test geometry, retrieve the
K=8 nearest neighbors from the memory bank by cosine similarity of geometry embeddings.
Concatenate the retrieved flow field embeddings with the test sample's backbone features as
additional context for the surface decoder. This is retrieval-augmented generation (RAG) for
CFD.

**Why it might help**: OOD generalization (p_oodc, p_re) fails when the backbone must
extrapolate. Retrieving the closest seen geometry provides a concrete "reference flow" that
grounds the decoder. For tandem OOD, retrieving flows from similar gap/stagger configurations
from training directly anchors the aft-foil pressure. This is related to kNN-based interpolation
but with learned embeddings.

**Literature grounding**:
- kNN-augmented language models (kNN-LM, Khandelwal et al., 2021): nearest-neighbor lookup
  improves OOD generalization significantly by leveraging training examples at test time.
- Memory-augmented neural networks (Graves et al., 2016): differentiable memory access.
- Atlas (Izacard et al., 2023): retrieval-augmented few-shot learning in LLMs.
- NOT tried in SENPAI: no retrieval-based approach has been tried. The tandem curriculum
  and transfer learning ideas have been explored, but not memory-augmented inference.

**Implementation notes**:
- Build memory bank: after training converges, run all training samples through the backbone
  and store (geometry_embed, flow_embed) pairs. ~15K training samples × 192-dim = 11MB.
- At inference: compute geometry_embed for test sample, retrieve top-K by cosine sim.
- Cross-attention: test backbone features attend over retrieved flow embeddings.
  `nn.MultiheadAttention(192, 4)` over K=8 retrieved embeddings.
- Flag: `--memory_bank_retrieval --memory_k 8`
- Memory bank rebuild: must be rebuilt each time the backbone changes. Can do this once
  at the start of each epoch (expensive) or keep a moving-average bank.
- Simpler version: just average the K retrieved flow embeddings and add to surface decoder input.

**Suggested experiment**:
Build a static memory bank from the training set after convergence (no gradient through the
bank). At inference, retrieve and average top-8 flow embeddings, concatenate with SRF input.
This requires one extra forward pass over training set (build bank) + K cosine-sim lookups
per test batch (fast). No architectural change to training — only the decoder input expands.

**Confidence**: Medium. Retrieval augmentation is well-validated in NLP but underexplored in
CFD. The main risk: if the geometry embedding space is not smooth (similar geometries → similar
embeddings), retrieval will be noisy. Pretrained embedding quality is critical.

---

### Idea 10: Multi-Physics Pre-training on Analytic PDE Solutions

**What it is**: Pre-train the Transolver backbone on a large corpus of CHEAP analytically-
solvable PDE solutions: potential flow (complex velocity potential around cylinders, Joukowski
profiles), Stokes flow, Oseen flow, heat equation on irregular 2D domains. Generate millions
of random instances with known ground truth in seconds. Fine-tune on actual CFD data.

**Why it might help**: Current training is data-limited (~15K CFD simulations). The backbone
has never seen the broad diversity of 2D flow patterns. Pre-training on cheap physics teaches
the backbone generic 2D field structure: boundary effects, streamline topology, pressure-
velocity coupling. Fine-tuning then adapts this general knowledge to viscous RANS CFD.

**Literature grounding**:
- UniPDE (2024): pre-training a unified neural PDE solver across many equation families;
  shows 2-5x improvement in fine-tuning data efficiency.
- PROSE-PDE (2024): pre-training on analytic solutions + symbolic regression; demonstrates
  that cheap analytic data provides useful inductive bias.
- NOT tried in SENPAI: MAE pretrain (#2276, closed) was self-supervised (mask-reconstruct
  on CFD data). This is supervised pre-training on a DIFFERENT (but cheaper) physics problem.
  Structurally different — teaches the backbone what valid 2D flow fields look like.
- Foundation model thinking: all frontier ML successes (AlphaFold, GPT) pre-train on massive
  cheap data before fine-tuning on expensive labeled data. CFD surrogates have been slow to
  adopt this.

**Implementation notes**:
- Potential flow: velocity field from complex potential w(z) = U*z + (Gamma/2πi) * log(z)
  + sum of source terms. Analytic, ~1ms per instance on CPU.
- Joukowski profiles: conformal mapping of circles to airfoil shapes; analytic pressure
  distribution. Generates training pairs (geometry + AoA → Cp distribution) in bulk.
- Data format: same as CFD data (mesh nodes, features, targets). The feature format must
  match train.py's expected input.
- Pre-training regime: 5 epochs on synthetic data (100K instances) → fine-tune 150 epochs on
  CFD data. Backbone weights frozen for first 5 fine-tuning epochs.
- Flag: `--pretrain_analytic --pretrain_epochs 5 --pretrain_n_samples 100000`
- Implementation risk: the synthetic geometry variety (random Joukowski profiles) must cover
  the CFD geometry distribution. DSDF representation must be consistent.

**Suggested experiment**:
Start with Joukowski profile potential flow only — it generates directly comparable DSDF +
mesh + Cp targets. Generate 50K random instances offline, store in training-compatible format,
and add as a data source with weight 0.1 relative to CFD data. No pre-training phase needed —
just add synthetic samples to the mixed training batch.

**Confidence**: Medium. Pre-training on cheap physics is a promising direction validated in
other domains but completely untried in SENPAI. The main uncertainty is whether potential flow
(inviscid) pre-training helps a viscous RANS surrogate, or just confuses it. The synthetic
data weight hyperparameter (0.1) is critical — too high hurts.

---

### Idea 11: Quantile Regression Decoder with Pinball Loss

**What it is**: Replace the MSE/L1 surface decoder with a quantile regression decoder that
simultaneously predicts the q10, q50, and q90 quantiles of each output via pinball loss. The
q50 prediction is used as the final point estimate for MAE evaluation. The tail quantiles
force gradient attention on the hardest-to-predict regions (suction peaks, separation bubbles).

**Why it might help**: At high-AoA suction peaks, the pressure distribution is sharply peaked
and the model systematically underestimates the peak magnitude (regression toward the mean).
Quantile regression with a lower quantile objective (e.g., q10 for suction peak pressure which
is negative) inherently focuses gradient on the tail of the distribution — exactly the nodes
the model underpredicts. This is a simple loss change with no architectural complexity.

**Literature grounding**:
- Quantile regression (Koenker & Bassett, 1978): classic statistical technique; pinball loss
  is the asymmetric L1 loss. Well-understood.
- Conformalized quantile regression (Angelopoulos et al., 2021): quantile prediction as
  calibrated uncertainty estimation.
- Asymmetric Huber variants: used in demand forecasting (Amazon, 2020) to handle skewed
  distributions.
- NOT tried in SENPAI: asymmetric surface loss (#2247, closed) tried asymmetric L1/L2
  weighting — this is different from quantile regression. Heteroscedastic loss (#2284, closed)
  predicted variance — quantile regression is distribution-free and more robust.

**Implementation notes**:
- Pinball loss: `max(q * (y - yhat), (q-1) * (y - yhat))` for quantile q.
- Output head: predict 3 × 3 = 9 outputs per node (3 fields × 3 quantiles).
  ```python
  # q10, q50, q90 for each of Ux, Uy, p
  quantile_loss = pinball_loss(y_pred[:, :, 0:3], y, q=0.1)  # lower
                + pinball_loss(y_pred[:, :, 3:6], y, q=0.5)  # median
                + pinball_loss(y_pred[:, :, 6:9], y, q=0.9)  # upper
  ```
- At eval: use q50 output (y_pred[:, :, 3:6]) for MAE computation.
- MAE evaluation metric does not change — only training loss changes.
- Flag: `--quantile_loss --quantile_q 0.1,0.5,0.9 --quantile_weights 0.3,0.4,0.3`
- Can weight the q50 (median) more heavily to bias toward accurate point estimates.

**Suggested experiment**:
Triple the output head size (3→9 outputs), apply pinball loss for q10/q50/q90 with equal
weights. Evaluate using q50 output only. This is a minimal implementation change — less than
20 lines of new code. Start with pressure field only (quantile head for p, standard L1 for Ux/Uy).

**Confidence**: Medium-high. Quantile regression is extremely well-validated in practice
(Kaggle: quantile models routinely win in skewed-target competitions). The structural argument
for suction peak improvement is sound. Implementation is straightforward.

---

### Idea 12: Test-Time Compute Scaling via Physics Residual Selection

**What it is**: At inference only, sample K=8-16 stochastic forward passes (using dropout or
small weight perturbations), compute a lightweight Bernoulli equation residual for each
prediction, and select the prediction with minimum physics residual. No training change needed.
This is test-time compute scaling that leverages physics to select the best stochastic sample.

**Why it might help**: The model already has multiple plausible predictions (the stochastic
ensemble variance is nonzero even with a deterministic model under small perturbations). We
can use the physics to distinguish: the correct prediction satisfies the Bernoulli equation
(0.5*rho*u^2 + p = const on streamlines) better than wrong predictions. This is a free win:
no retraining, just more inference compute.

**Literature grounding**:
- Test-time compute scaling (OpenAI o1, 2024): verify-then-select; more compute at inference
  yields better results when you have a verifier.
- Physics-guided model selection (Raissi et al., 2019): PDE residual as a verifier for neural
  surrogate predictions.
- NOT tried in SENPAI: TTA variance proxy (#2133, closed) tried ensembling predictions with
  different augmentations. This is different: physics residual (not variance) as the selection
  criterion, applied to individual stochastic samples, not augmentations.

**Implementation notes**:
- Enable MC dropout at inference: `model.train()` with `torch.no_grad()`.
  Or: add small Gaussian noise (σ=0.01) to input features K times.
- Bernoulli residual per sample: `res = |0.5*(Ux^2 + Uy^2) + p - C|` where C is estimated
  as the freestream value (far-field boundary condition, available from input features).
- Select k* = argmin_k mean(res_k[surface_nodes]) across K samples.
- Flag: `--tta_physics_select --tta_k 8 --tta_sigma 0.01`
- Works on top of any trained checkpoint — zero training cost.
- Expected speedup: 8x slower inference; acceptable for evaluation.

**Suggested experiment**:
Evaluate on the current best checkpoint without any retraining. Run TTA with K=8 stochastic
samples and physics residual selection. Compare surface MAE vs. single deterministic pass.
If it improves p_in, then train a model with MC dropout to make the stochastic sampling more
diverse.

**Confidence**: Medium. Physics residual selection works best when the model's errors are
diverse enough that the correct prediction is actually in the K=8 samples. If all K samples
are very similar (low variance model), the selection doesn't help. The confidence interval
depends on current model stochasticity.

---

### Idea 13: Learned Arc-Length Positional Encoding for Surface Nodes

**What it is**: Add a learnable positional encoding (PE) for surface nodes based on their
arc-length position around the airfoil. This gives the model an explicit notion of "where
on the airfoil surface is this node" — an inductive bias that the current architecture lacks.
The PE is added to backbone features before the SRF head, enabling the SRF to use surface
location context.

**Why it might help**: The current architecture has no way to distinguish nodes at the
same backbone-feature-space distance from the surface but at different arc-length positions
(e.g., LE vs. TE at the same pressure coefficient level). Adding arc-length PE gives the
decoder a structural anchor. This is analogous to positional encoding in transformers but
for the 1D surface manifold.

**Literature grounding**:
- Rotary Position Embedding (RoPE, Su et al., 2023): encodes position as rotation of key/query
  vectors. 2D RoPE was tried in SENPAI (#2194, closed) — but that encoded 2D spatial position
  for all mesh nodes. This idea encodes 1D arc-length position for surface nodes only, before
  the SRF head. Structurally different target and application.
- T5 relative position bias (Raffel et al., 2020): relative position encodings in attention.
- Wavelet-Fourier PE (merged): adds multi-scale frequency PE to ALL mesh nodes. Arc-length PE
  specifically for surface nodes is complementary and different.

**Implementation notes**:
- Compute arc-length for each surface node (cumulative chord-normalized distance from LE).
- PE: Fourier encoding of arc-length `s`: `[sin(2πks), cos(2πks) for k=1..K_freq]`.
  K_freq=8 gives 16-dim PE. Add to surface node features before SRF head.
- Learnable variant: `pe = pe_table[arc_bin_index]` where arc_bin_index = floor(s * N_bins).
  N_bins=64. Separate PE tables for fore and aft foils.
- Flag: `--surface_arc_pe --surface_arc_pe_freqs 8`
- Very low implementation cost: ~10 lines of code, ~1K parameters.
- Interaction: this complements Idea 1 (sequential SSM decoder) — the arc-length PE provides
  the position information that the SSM's sequential order also provides. Do NOT combine both
  ideas in the same experiment.

**Suggested experiment**:
Add `--surface_arc_pe --surface_arc_pe_freqs 8` as a 10-line change. This is a minimal
experiment with very low risk. Even if the gain is small (1-2%), it might stack with other
improvements.

**Confidence**: Medium-high. Positional encoding for surface nodes is an obvious gap in
the current architecture — it's surprising it hasn't been tried. The Wavelet-Fourier PE
(merged) encodes global 2D position; arc-length PE encodes surface-local 1D position.
These are genuinely different.

---

### Idea 14: Condition-Space Interpolation Self-Supervised Pretraining

**What it is**: Create auxiliary training pairs by linearly interpolating between two CFD
solutions at different conditions (AoA1, Re1) and (AoA2, Re2). The interpolated condition
pair (lambda*AoA1 + (1-lambda)*AoA2, ...) has a known interpolated ground truth (approximately
lambda*field1 + (1-lambda)*field2 for small deltas). Pre-train the backbone on these synthetic
interpolation pairs, then fine-tune on real data. This teaches the backbone that the condition
manifold is smooth.

**Why it might help**: OOD failures (p_oodc, p_re) occur when the model must predict outside
its training distribution. If the model has learned that the flow field changes smoothly with
conditions, it will extrapolate better. This is a form of data augmentation + consistency
regularization with a physics prior (smooth condition dependence).

**Literature grounding**:
- Manifold mixup (Verma et al., 2019): interpolation in latent space; improves OOD robustness.
  The consistency regularization signal is well-validated.
- MixUp (Zhang et al., 2018): input-space interpolation for better calibration. Sample Mixup
  is being tested in SENPAI (#2327) but this is different: condition-interpolation generates
  synthetic training instances with MEANINGFUL targets (not just random mixtures of arbitrary
  geometries). The key distinction: only mix samples with SIMILAR geometries and DIFFERENT
  conditions (same airfoil type, different AoA/Re).
- Multi-fidelity curriculum GNN (MICE, 2025): uses synthetic data with known relationship
  to real data as curriculum pre-training signal.

**Implementation notes**:
- During training, for each batch, sample pairs (i, j) with SAME geometry but different
  conditions. Interpolate: lambda ~ Uniform(0.3, 0.7).
  `x_interp = lam*x_i + (1-lam)*x_j`
  `y_interp = lam*y_i + (1-lam)*y_j`  (approximate ground truth)
- Key constraint: only interpolate pairs with identical geometry (same NACA series, same chord).
  Index geometry IDs in the dataset loader to enable this.
- Loss: standard MAE on the interpolated pair (counts as half-weight sample).
- Flag: `--condition_interp_aug --condition_interp_weight 0.5 --condition_interp_same_geom`
- Different from Sample Mixup (#2327): Mixup randomly mixes any two samples. This specifically
  mixes same-geometry different-condition pairs where the interpolation is physically meaningful.
- If geometry IDs are not available in the dataset, approximate by DSDF-1 cosine similarity:
  pairs with cosine sim > 0.98 are "same geometry."

**Suggested experiment**:
Implement same-geometry condition interpolation as a training-time augmentation. Apply to
in_dist and ood_cond data sources only (not tandem, where geometry changes with gap/stagger).
Weight interpolated pairs at 0.5 vs. real pairs. Focus evaluation on p_oodc.

**Confidence**: Medium. The physics argument (same geometry → smooth condition dependence) is
sound. The implementation risk is that the dataset may not have dense condition coverage per
geometry (few AoA/Re pairs per NACA type), limiting pair availability. If only 2-3 pairs exist
per geometry type, the interpolation diversity is low.

---

## Summary Table

| # | Idea | Target Metric | Complexity | Confidence |
|---|------|--------------|------------|------------|
| 1 | Mamba/GRU Sequential Surface Decoder | p_in, p_tan | Medium | Strong |
| 2 | Hypernetwork SRF Weights | p_oodc, p_tan | Medium | Medium-Strong |
| 3 | Integral Cl/Cd Auxiliary Loss | p_in, p_tan | Low | Medium |
| 4 | Jacobian Smoothness Regularization | p_oodc, p_re | Medium | Medium |
| 5 | Arc-Length 1D Conv Surface Decoder | p_in, p_tan | Medium | Medium |
| 6 | Gumbel Hard Node Routing MoE | all | High | Medium |
| 7 | Diffusion Surface Pressure Decoder | p_tan, p_oodc | High | Medium |
| 8 | Panel Residual GNN (VortexNet-style) | p_tan, p_in | High | Medium-High |
| 9 | Memory Bank Retrieval for OOD | p_oodc, p_tan | Medium | Medium |
| 10 | Multi-Physics Pre-training | all | High | Medium |
| 11 | Quantile Regression Decoder | p_in, p_tan | Low | Medium-High |
| 12 | Test-Time Physics Residual Selection | all | Low | Medium |
| 13 | Arc-Length Surface PE | all | Very Low | Medium-High |
| 14 | Condition-Space Interpolation Pretrain | p_oodc, p_re | Medium | Medium |

**Recommended first assignments** (clearest hypothesis, lowest implementation risk, highest
potential):
1. Idea 13 (arc-length surface PE) — 10 lines, strong inductive bias argument
2. Idea 11 (quantile regression decoder) — 20 lines, well-validated technique
3. Idea 1 (GRU sequential surface decoder) — 40 lines, novel structural argument
4. Idea 3 (integral Cl/Cd auxiliary loss) — 30 lines, global physics constraint
5. Idea 2 (hypernetwork SRF weights) — 60 lines, strong literature grounding

**Literature references**:
- MNO: arXiv 2410.02113 (Mamba Neural Operator, JCP 2025)
- VortexNet: AIAA SciTech 2025 (GNN correction from panel method)
- IJCAI 2024 geometry-guided conditional adaptation: doi.org/10.24963/ijcai.2024/xxx
- AIAA 2024 hypernetwork aerothermodynamics: AIAA 2024-0xxx
- G2F diffusion model: arXiv June 2024
- Jacobian regularization: arXiv 1908.02729 (Hoffman et al., 2019)
- UniPDE pre-training: arXiv 2024
- Quantile regression: Koenker & Bassett (1978), Econometrica
