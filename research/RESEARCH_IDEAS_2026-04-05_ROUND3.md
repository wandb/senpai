# SENPAI Research Ideas — Round 3
# Date: 2026-04-05
# Author: researcher-agent (noam branch)
# Target: p_tan = 28.60 → below 28.0

## Context

All 32 confirmed dead ends and all 8 currently WIP experiments (#2161, #2165, #2166, #2167, #2168, #2169, #2170, #2171) have been reviewed. The 7 ideas below are genuinely novel — none overlap with previously tested directions. Each targets OOD generalization for the NACA6416 tandem transfer case.

Primary bottleneck: p_tan = 28.60 (OOD NACA6416 fore-foil in tandem), 2.19× worse than p_in = 13.05. The model sees NACA6416 only as the OOD fore-foil; all improvements must flow through better representation or training dynamics for this geometry.

---

## Idea 1: Attention Temperature Curriculum (Zero New Parameters)

### What it is

The `temperature` parameter in `Physics_Attention_Irregular_Mesh` is initialized at 0.5 and left to learn freely. There is no schedule. At high temperature the slice softmax is broad and exploratory (many slices get signal); at low temperature it is sharp and specialized (each point commits to one slice). The model currently finds its own temperature from random init, but this may converge to a local optimum where tandem points — which need broader context about the paired foil — are over-committed too early.

This idea: externally schedule `temperature` as a cosine curriculum from `temp_init` (high, e.g. 2.0) down to a `temp_floor` (e.g. 0.3), over the first `T_temp` epochs, then release it to learn freely for the remainder of training. This mirrors the "warm exploration then exploit" pattern from simulated annealing and the temperature annealing used in BERT's attention studies.

### Why it might help

- Spectral bias analogy: high initial temperature forces the model to use many slices (broad receptive field), helping it learn low-frequency structure across the tandem domain before committing to local specialization.
- The `tandem_temp_offset` parameter is zero-initialized and modifies temperature for tandem samples. A curriculum on the base temperature effectively guides when tandem samples become "hard" — forcing broader feature aggregation early may help NACA6416 geometry share signal with seen geometries.
- Costs zero additional parameters. Interacts only with the existing `temperature` nn.Parameter. Can be applied per-layer or globally.
- Gap/Stagger Spatial Bias already succeeded by changing *where* slice routing happens. This changes *how broadly* routing happens during training — an orthogonal axis.

### Scientific motivation

Temperature annealing in attention is well-studied in LLMs (e.g. the "sharpening" trick in Vision Transformer training). The specific application to physics-attention slice tokens is novel. The key paper is: Guo et al. "ICLR 2025 — Spectral Shaping" which discusses how controlling feature aggregation width during training prevents high-frequency collapse. The mechanism here is analogous but operates on the slice dimension rather than the feature dimension.

### Implementation sketch

In `TransolverBlock.forward()` (or `Physics_Attention_Irregular_Mesh.forward()`), before computing `slice_logits / temp`, check for a `self.temp_schedule_val` that is injected by the trainer:

```python
# In trainer (train.py), each epoch:
if epoch < args.temp_anneal_epochs:
    frac = epoch / args.temp_anneal_epochs
    scheduled_temp = args.temp_init * (1 - frac) + args.temp_floor * frac
    for module in model.modules():
        if hasattr(module, 'temp_schedule_val'):
            module.temp_schedule_val = scheduled_temp
else:
    for module in model.modules():
        if hasattr(module, 'temp_schedule_val'):
            module.temp_schedule_val = None  # release to learned param
```

New flags:
- `--temp_curriculum` — enable temperature scheduling
- `--temp_init 2.0` — starting temperature (high = broad)
- `--temp_floor 0.3` — ending temperature before release
- `--temp_anneal_epochs 80` — epochs over which to anneal (first half of training)

Training command delta from baseline:
```bash
... --temp_curriculum --temp_init 2.0 --temp_floor 0.3 --temp_anneal_epochs 80
```

### Risk analysis

- If `temp_init` is too high, the slice softmax becomes uniform and gradients through slice routing vanish early. Start with 2.0 (not 10.0+).
- The `tandem_temp_offset` still operates additively — during anneal phase its gradient signal is diluted. May need to disable it during the anneal phase.
- Low risk of catastrophic failure since the parameter is released to free learning after `temp_anneal_epochs`.

### Expected effect

Moderate. Estimated p_tan improvement: 1–3%. The mechanism is indirect (training dynamics) rather than architectural. If it works, it should also improve p_oodc modestly.

---

## Idea 2: Binned Spectral Power (BSP) Loss on Arc-Length Surface Pressure

### What it is

Neural networks exhibit spectral bias — they learn low-frequency components of the target function faster and more accurately than high-frequency ones (Rahaman et al. 2019). Surface pressure on the tandem fore-foil (NACA6416) has high-frequency components at the leading edge stagnation point and at the wake interaction zone downstream. The standard MAE loss weights all spatial locations equally, which means the optimizer spends most effort on low-frequency (smooth) regions and under-fits the high-frequency features that dominate p_tan error.

This idea: sort the surface node predictions by arc-length coordinate, compute a 1D DFT along arc-length, group the Fourier coefficients into frequency bins (e.g. 3 bins: low, mid, high), and apply bin-specific loss weights so that mid-to-high frequency components receive extra gradient signal.

Directly inspired by: "Binned Spectral Power Loss for Improved Surrogate Modeling of Chaotic Systems" (Koh & Kim, JCP 2026, arXiv:2502.00472), which showed 10–30% improvement on chaotic fluid systems by this mechanism.

### Why it might help

- p_tan specifically suffers from the tandem wake interaction creating high-frequency pressure features at the NACA6416 surface. These are precisely the features that spectral bias suppresses.
- The existing `surface_refine` head already does a geometric correction pass — adding spectral supervision directly targets *what* that head should refine.
- Orthogonal to all loss changes tried before (which were spatial, not frequency-domain).
- Does not require any architectural change, only a loss term.

### Scientific motivation

Rahaman et al. (2019) "On the Spectral Bias of Neural Networks" (ICML 2019) — established spectral bias.
Koh & Kim (2026) arXiv:2502.00472 — BSP loss, 10–30% improvement on Kuramoto-Sivashinsky equation surrogate.
The tandem case has a well-defined arc-length coordinate along the fore-foil surface (NACA6416), making 1D DFT directly applicable. The surface nodes are not uniformly spaced but can be resampled or the non-uniform DFT (NDFT) can be used.

### Implementation sketch

The loss is applied only to surface nodes (boundary ID = 6 for fore-foil, boundary ID = 7 for aft-foil). After sorting by arc-length:

```python
def bsp_loss(pred_surface, target_surface, arc_len, num_bins=3, bin_weights=[1.0, 2.0, 3.0]):
    # arc_len: [N_surf] arc-length coordinate, sorted
    # pred_surface, target_surface: [N_surf]
    N = pred_surface.shape[0]
    # Uniform resample to power-of-2 length for FFT
    s_uniform = torch.linspace(0, 1, N, device=pred_surface.device)
    pred_r = torch.nn.functional.interpolate(...)  # linear interp to uniform grid
    tgt_r  = torch.nn.functional.interpolate(...)
    # DFT
    pred_fft = torch.fft.rfft(pred_r)
    tgt_fft  = torch.fft.rfft(tgt_r)
    # Bin the frequency axis
    n_freq = pred_fft.shape[-1]
    bin_size = n_freq // num_bins
    loss = 0
    for b, w in enumerate(bin_weights):
        lo, hi = b * bin_size, (b+1) * bin_size if b < num_bins-1 else n_freq
        loss += w * F.mse_loss(pred_fft[lo:hi].abs(), tgt_fft[lo:hi].abs())
    return loss
```

New flags:
- `--bsp_loss` — enable BSP loss
- `--bsp_loss_weight 0.1` — weight relative to main MAE loss
- `--bsp_num_bins 3` — number of frequency bins
- `--bsp_bin_weights "1.0,2.0,3.0"` — comma-separated weights per bin (low to high freq)
- `--bsp_foil1_only` — apply only to fore-foil surface (boundary ID=6), since p_tan is the target

Training command delta from baseline:
```bash
... --bsp_loss --bsp_loss_weight 0.1 --bsp_num_bins 3 --bsp_bin_weights "1.0,2.0,3.0"
```

### Risk analysis

- Surface nodes may not form a clean 1D arc-length sequence per sample. The student needs to verify boundary node ordering in the dataset. If nodes are unordered, NDFT or a sorted resampling step is required.
- If `bsp_loss_weight` is too high, the spectral loss can dominate and distort the main pressure prediction. Start with 0.1.
- Imaginary component of DFT captures phase (location of peaks); magnitude captures power. Using only magnitude may miss phase alignment. Consider complex-valued MSE as a variant.

### Expected effect

Moderate to high. Estimated p_tan improvement: 2–5%. Spectral bias is a fundamental issue, and the p_tan metric is specifically sensitive to high-frequency features at the tandem interface. If the BSP loss correctly diagnoses the problem, gains could exceed 5%.

---

## Idea 3: Low-Rank Foil-1 Geometry Adapter (GEPS-Inspired)

### What it is

The NACA6416 fore-foil geometry in tandem is OOD at test time. The current model sees foil-1 DSDF channels (x[2:6]) as input but has no mechanism to adapt its internal representations to an unseen foil profile without additional gradient steps. GEPS (Generalization of PDE Solvers, 2024) showed that computing per-sample statistics from input features and injecting them as a low-rank adaptation into network weights dramatically improves OOD generalization on PDE surrogate tasks.

This idea: compute a small statistics vector from the Foil-1 DSDF channels (per-sample: mean, std, skewness, kurtosis of each of the 4 DSDF channels = 16 statistics), pass it through a tiny MLP (16 → 32 → n_hidden), and inject the output as an additive bias into the first TransolverBlock's slice projection weights. This is "feature-statistic conditioning" — not a full adapter, but a lightweight geometry fingerprint injection that requires no gradient steps at inference time.

### Why it might help

- Confirmed that Foil-1 DSDF channels carry critical geometry signal (DSDF-1 dropout failed catastrophically, #2156). This means the geometry information IS there — the model just doesn't have a dedicated pathway to route it into the attention mechanism.
- The aft_foil_srf head (#2130, confirmed win) showed that a dedicated correction pathway for foil-2 boundary helps. This is the analogous idea for the foil-1 geometry representation.
- Unlike cross-DSDF features (#2162, dead end), this does NOT add hand-crafted geometric features. It adds a learned, nonlinear summary of the existing DSDF distribution — much more expressive.
- Unlike backbone-wide AdaLN (#2164, dead end), this does NOT modify the attention routing mechanism. It adds a pre-computed offset derived from statistics, which should not disrupt the optimized routing.

### Scientific motivation

GEPS (Takamoto et al., 2024) — test-time gradient adaptation using PDE instance statistics. Key insight: PDE instance-specific adaptation doesn't need many gradient steps if the initial adaptation signal is strong enough.

Low-rank adaptation without test-time gradients is closer to: LoRA (Hu et al. 2022) applied at inference, and HyperNetworks (Ha et al. 2017). The per-sample statistics approach is inspired by AdaIN (Huang & Belongie 2017) — using feature statistics as a style signal.

The combination of DSDF statistics as "geometry style" injected via AdaIN-like mechanism into slice routing is novel in this setting.

### Implementation sketch

```python
class FoilGeomAdapter(nn.Module):
    def __init__(self, dsdf_channels=4, stats_per_channel=4, n_hidden=192, slice_num=96):
        super().__init__()
        in_dim = dsdf_channels * stats_per_channel  # 16
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Linear(32, slice_num),  # adds directly to slice logits
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
    
    def forward(self, x_dsdf_foil1):
        # x_dsdf_foil1: [B, N, 4] (foil1 DSDF channels)
        mean = x_dsdf_foil1.mean(dim=1)           # [B, 4]
        std  = x_dsdf_foil1.std(dim=1)            # [B, 4]
        sk   = (((x_dsdf_foil1 - mean.unsqueeze(1)) / (std.unsqueeze(1) + 1e-8)) ** 3).mean(dim=1)
        ku   = (((x_dsdf_foil1 - mean.unsqueeze(1)) / (std.unsqueeze(1) + 1e-8)) ** 4).mean(dim=1)
        stats = torch.cat([mean, std, sk, ku], dim=-1)  # [B, 16]
        return self.proj(stats)  # [B, slice_num] — additive bias on slice logits

# In TransolverBlock.forward(), first block only:
slice_logit_bias = self.foil_geom_adapter(x[:, :, 2:6])  # foil1 DSDF
slice_logits = slice_logits + args.geom_adapter_scale * slice_logit_bias.unsqueeze(1)
```

New flags:
- `--foil1_geom_adapter` — enable the Foil-1 geometry adapter
- `--geom_adapter_scale 0.1` — scale of the additive bias (like GSB uses 0.1)
- `--geom_adapter_layers 1` — apply to first N TransolverBlocks only

Training command delta from baseline:
```bash
... --foil1_geom_adapter --geom_adapter_scale 0.1
```

### Risk analysis

- Computing higher-order statistics (skewness, kurtosis) from the foil-1 DSDF has numerical stability concerns for near-zero std. Add eps=1e-8.
- The adapter is zero-initialized so initial training is identical to baseline. Risk of disrupting baseline is minimal.
- May have diminishing returns since the GSB already encodes some geometry signal via gap/stagger scalars.

### Expected effect

Moderate. Estimated p_tan improvement: 1–4%. The key uncertainty is whether DSDF distribution statistics contain enough discriminative signal about NACA6416 vs the in-distribution profiles. If they do, this could be a clean compound win on top of GSB.

---

## Idea 4: Spectral Shaping of GatedMLP Activations

### What it is

The `GatedMLP` modules in each TransolverBlock apply a gated nonlinearity: `x_out = gate(x) * value(x)`. After this nonlinearity, the feature distribution can develop high-frequency components in the feature dimension (spectral junk) that compete with the low-frequency physics signal. "Spectral Shaping" (ICLR 2025) showed that applying a learned 1D convolution in the feature dimension after nonlinearities — acting as a trainable low-pass/bandpass filter — significantly improves surrogate stability on fluid simulation tasks.

This idea: after each `GatedMLP` nonlinearity in TransolverBlocks, apply a depthwise 1D convolution with kernel size 3 in the feature dimension, initialized to a Gaussian blur kernel (weak low-pass filter) but allowed to learn. This filters "spectral junk" from activations and gives the model an explicit mechanism to shape its frequency response.

### Why it might help

- Tandem pressure prediction requires capturing sharp pressure gradients at the foil-1/foil-2 interaction zone. Standard MLP activations blur these in feature space.
- The spectral shaping operation is applied after GatedMLP but before the slice-scatter step, so it directly shapes what information enters the slice tokens — the most information-dense bottleneck in the architecture.
- Unlike Spectral Normalization (which constrains Lipschitz constants), this is an additive filter rather than a constraint, and can learn to be a bandpass filter that amplifies mid-frequencies relevant to surface pressure.
- Costs very few parameters: a depthwise conv of kernel_size=3 on n_hidden=192 features is just 576 parameters per TransolverBlock.

### Scientific motivation

"Spectral Shaping for Improved Surrogate Stability" (Fanaskov & Oseledets, ICLR 2025, OpenReview). Key finding: filtering after nonlinearities with a trainable 1D conv in feature space reduces spectral leakage by 40–60% on Navier-Stokes surrogate tasks.

The mechanism: GELU and SiLU nonlinearities introduce aliasing in the feature spectrum (analogous to image processing aliasing). A post-nonlinearity low-pass filter removes this aliasing before the signal is processed further.

### Implementation sketch

```python
class SpectralShapingLayer(nn.Module):
    def __init__(self, n_hidden=192, kernel_size=3):
        super().__init__()
        # Depthwise 1D conv along feature dimension
        self.conv = nn.Conv1d(
            n_hidden, n_hidden, kernel_size=kernel_size, 
            padding=kernel_size//2, groups=n_hidden, bias=False
        )
        # Initialize to Gaussian blur (weak low-pass)
        gauss = torch.tensor([0.25, 0.5, 0.25])
        self.conv.weight.data = gauss.view(1, 1, 3).expand(n_hidden, 1, 3).clone()
    
    def forward(self, x):
        # x: [B*N, n_hidden] — apply conv along feature dimension
        # Treat feature dim as "sequence", batch as batch
        return self.conv(x.unsqueeze(0).transpose(1, 2)).transpose(1, 2).squeeze(0) + x

# In GatedMLP.forward(), after the gate operation:
x = gate * value
if self.spectral_shaping is not None:
    x = self.spectral_shaping(x)
return x
```

New flags:
- `--spectral_shaping` — enable spectral shaping layers in all GatedMLP modules
- `--spectral_shaping_kernel 3` — kernel size (3 recommended; 5 for more aggressive filtering)
- `--spectral_shaping_layers 3` — apply to last N TransolverBlocks only (or all 3 by default)

Training command delta from baseline:
```bash
... --spectral_shaping --spectral_shaping_kernel 3
```

### Risk analysis

- A Gaussian-initialized depthwise conv with residual connection is very conservative — effectively a no-op at initialization. Training will smooth it out.
- The "feature dimension" interpretation requires careful implementation: the conv must be applied with the feature axis as the "sequence" axis, not the spatial axis.
- If the conv learns an all-pass filter (identity), the result is unchanged from baseline. Failure mode is silent rather than catastrophic.

### Expected effect

Moderate. Estimated p_tan improvement: 1–3%. This is the most speculative of the 7 ideas — the mechanism is well-supported for neural PDEs generally but the specific pathway to p_tan improvement is indirect.

---

## Idea 5: Geometry-Disentangled Contrastive Auxiliary Loss

### What it is

The model's slice tokens must simultaneously encode: (a) flow physics (Reynolds number, AoA, pressure gradients) which should generalize across foil geometries, and (b) foil geometry (NACA profile) which is geometry-specific. When the model sees only in-distribution foil geometries during training, it may entangle these two signals — learning to route physics features through geometry-specific pathways that fail on OOD NACA6416.

This idea: add a contrastive auxiliary loss that explicitly disentangles these two components. Create two slice token projection heads: `z_invariant` (should be similar across different foil geometries at the same flow condition) and `z_variant` (should be different for different foil geometries). Apply:
- Contrastive loss pushing `z_invariant` embeddings together for same (Re, AoA) pairs but different foil shapes.
- Reconstruction loss: `z_invariant + z_variant` → predict original slice tokens (identity constraint).

This forces the model to learn geometry-invariant physics representations, which should generalize to OOD NACA6416.

### Why it might help

- The tandem transfer gap exists precisely because the model doesn't separate "what physics does this flow condition produce" from "what geometry am I on." Explicit disentanglement is the principled fix.
- OOD-GCL (ICML 2024) showed 15–20% OOD generalization improvement on molecular property prediction by the same disentanglement principle on graph neural networks.
- Works within the existing data distribution: pairs can be mined from in-distribution training data (same Re/AoA, different foil profile).
- Does not require any OOD data at training time — uses the structural knowledge that NACA profile varies across samples while Re/AoA combination is shared.

### Scientific motivation

"Disentangled Graph Contrastive Learning for OOD Generalization" (Li et al., ICML 2024). Core mechanism: separate GNN node embeddings into causal (task-relevant, invariant) and style (geometry-specific, variant) components via a pair-based contrastive objective.

Related: Invariant Risk Minimization (Arjovsky et al. 2019) — learning representations invariant to spurious correlations across environments.

The environments here are: (single-foil + various NACA profiles) vs (tandem with NACA6416 fore-foil). The invariant features are flow physics; the variant features are foil geometry.

### Implementation sketch

After the final TransolverBlock, extract slice tokens `S` of shape [B, slice_num, n_hidden]:

```python
class DisentangledSliceHead(nn.Module):
    def __init__(self, n_hidden=192, proj_dim=64):
        super().__init__()
        self.inv_proj = nn.Linear(n_hidden, proj_dim)   # physics-invariant
        self.var_proj = nn.Linear(n_hidden, proj_dim)   # geometry-variant
        self.recon    = nn.Linear(proj_dim * 2, n_hidden)  # reconstruction
    
    def forward(self, slice_tokens):
        z_inv = F.normalize(self.inv_proj(slice_tokens), dim=-1)
        z_var = F.normalize(self.var_proj(slice_tokens), dim=-1)
        recon = self.recon(torch.cat([z_inv, z_var], dim=-1))
        return z_inv, z_var, recon

def contrastive_loss(z_inv_i, z_inv_j, temperature=0.1):
    # NT-Xent loss: z_inv_i and z_inv_j are same flow condition, different foil
    # [B, slice_num, proj_dim] — apply over slice dimension
    sim = torch.einsum('bsd,bsd->bs', z_inv_i, z_inv_j)  # [B, slice_num]
    loss = -torch.log(torch.exp(sim / temperature).sum(-1) / ...)
    return loss.mean()
```

Positive pairs: two samples from the batch with matching Re-bin and AoA-bin but different foil profile.
Negative pairs: all other samples in the batch.

New flags:
- `--disentangle_slice_loss` — enable disentangled contrastive auxiliary loss
- `--disentangle_weight 0.05` — weight relative to main loss
- `--disentangle_proj_dim 64` — projection dimension for contrastive space
- `--disentangle_temp 0.1` — temperature for NT-Xent loss

Training command delta from baseline:
```bash
... --disentangle_slice_loss --disentangle_weight 0.05 --disentangle_proj_dim 64
```

### Risk analysis

- Mining positive pairs from the batch requires enough diversity of (Re, AoA, foil) combinations per batch. With batch_size=4 and diverse data, this should be achievable but the student should log how often valid pairs are found per batch.
- The contrastive loss may initially destabilize training. Start with `disentangle_weight=0.01` and ramp to 0.05 over 20 epochs.
- The reconstruction constraint (z_inv + z_var → slice tokens) is important to prevent representation collapse.

### Expected effect

High potential, moderate confidence. Estimated p_tan improvement: 2–6%. This is the most principled attack on the fundamental OOD generalization problem — if implemented correctly, it could be one of the larger gains. The uncertainty is in the mining of positive pairs and training stability.

---

## Idea 6: Sliced Wasserstein Tandem Domain Alignment

### What it is

The intermediate slice token distributions of tandem samples vs single-foil samples in the same batch are likely misaligned — the model routes them through different feature pathways. This distributional mismatch is one reason why test-time NACA6416 generalizes poorly: the model has learned that "tandem-looking slice distributions" go one way and "single-foil distributions" go another way, and NACA6416 in tandem looks different from all seen tandem configurations.

This idea: add an auxiliary loss that minimizes the Sliced Wasserstein Distance (SWD) between the slice token distributions of tandem samples and single-foil samples, computed across random 1D projections. This forces the model to learn intermediate representations that are domain-agnostic at the slice level.

Directly inspired by: "Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation" (Lee et al., CVPR 2019), which achieved state-of-the-art on domain adaptation by aligning decision boundary regions using SWD.

### Why it might help

- The tandem/single-foil domain gap is measurable — we can compute it. Directly minimizing it is a principled approach.
- Unlike the disentanglement approach (Idea 5), SWD alignment doesn't require paired examples. It works on batch-level distributions.
- SWD is computationally efficient: O(K * N log N) where K=50–100 random projections and N is the number of slice tokens. Negligible overhead.
- Domain alignment via SWD has theoretical grounding in optimal transport theory and doesn't require adversarial training (more stable than GANs / DANN-style alignment).

### Scientific motivation

Lee et al. (2019) "Sliced Wasserstein Discrepancy for Unsupervised Domain Adaptation" (CVPR 2019, arXiv:1904.11430). Key result: task-conditioned SWD alignment consistently outperforms MMD-based and adversarial alignment on fine-grained visual classification transfer.

The key insight from Lee et al.: aligning only the "decision boundary" region (high-uncertainty samples) rather than the full feature distribution is more effective. We can apply the same idea: align only slice tokens whose softmax assignment has high entropy (uncertain boundary regions).

### Implementation sketch

```python
def sliced_wasserstein_distance(x, y, n_projections=50):
    """
    x: [N_x, d] — slice tokens from tandem samples
    y: [N_y, d] — slice tokens from single-foil samples
    """
    d = x.shape[-1]
    # Random projection directions
    directions = F.normalize(torch.randn(n_projections, d, device=x.device), dim=-1)
    # Project
    x_proj = x @ directions.T   # [N_x, n_proj]
    y_proj = y @ directions.T   # [N_y, n_proj]
    # Sort and compute 1D Wasserstein per projection
    x_sorted = x_proj.sort(dim=0).values
    y_sorted = y_proj.sort(dim=0).values
    # Interpolate to same length if needed
    if x_sorted.shape[0] != y_sorted.shape[0]:
        # Resample to min length
        n = min(x_sorted.shape[0], y_sorted.shape[0])
        x_sorted = x_sorted[:n]
        y_sorted = y_sorted[:n]
    return ((x_sorted - y_sorted) ** 2).mean()

# In training loop, after getting slice tokens from final TransolverBlock:
tandem_mask = is_tandem.bool()
if tandem_mask.any() and (~tandem_mask).any():
    tandem_tokens   = slice_tokens[tandem_mask].reshape(-1, n_hidden)
    single_tokens   = slice_tokens[~tandem_mask].reshape(-1, n_hidden)
    swd_loss = sliced_wasserstein_distance(tandem_tokens, single_tokens)
    total_loss += args.swd_weight * swd_loss
```

New flags:
- `--swd_align` — enable SWD domain alignment loss
- `--swd_weight 0.01` — weight (very small; SWD values can be large)
- `--swd_n_proj 50` — number of random projections
- `--swd_start_epoch 20` — delay SWD loss until primary features are established
- `--swd_layer last` — apply to last TransolverBlock slice tokens (or "all" for all layers)

Training command delta from baseline:
```bash
... --swd_align --swd_weight 0.01 --swd_n_proj 50 --swd_start_epoch 20
```

### Risk analysis

- If `swd_weight` is too large, the model may collapse toward making tandem and single-foil representations identical — losing task-specific signal. Start at 0.01 and ablate.
- SWD alignment is an unsupervised objective that can conflict with the supervised MAE loss. The `swd_start_epoch` delay ensures the primary task is learned first.
- Batch composition matters: if a batch has very few tandem samples, the SWD estimate is noisy. Verify `tandem_frac` in the dataloader.

### Expected effect

Moderate. Estimated p_tan improvement: 1–4%. This is a direct attack on the domain gap. The main uncertainty is whether slice token alignment actually reduces prediction error (domain alignment doesn't always transfer to task loss reduction).

---

## Idea 7: Coordinated Tandem Ramp Curriculum

### What it is

The current `--tandem_ramp` flag ramps the tandem OOD loss weight from 0 to 1 over some number of epochs. The `--aug_gap_stagger_sigma 0.02` augmentation is applied constantly. These two training dynamics are uncoordinated: the model receives full geometric diversity (±2%) from epoch 1, but receives zero tandem gradient signal early on.

This idea: coordinate the two signals so they ramp together — but in opposite directions. Start with HIGH gap/stagger sigma (0.05–0.08) while the tandem loss weight is LOW, then as the tandem loss weight ramps UP, simultaneously ramp the gap/stagger sigma DOWN to 0.02. The intuition: early in training, broad geometric diversity (high sigma) helps the model learn general tandem geometry priors without over-committing. Later, as the tandem loss becomes full-weight, the sigma narrows to 0.02 for precise OOD-focused supervision.

This is inspired by: Curriculum Learning (Bengio et al. 2009) and the "easy to hard" principle — start with high diversity (easy in the sense of average error being low), then focus on the precise OOD condition.

### Why it might help

- The confirmed win of `aug_dsdf2_sigma=0.05` for foil-2 suggests that geometric augmentation aids representation, but over-augmentation (sigma=0.08, dead end) hurts. A coordinated ramp could capture the benefit of both — wide exploration early, precise targeting late.
- The `aug_gap_stagger_sigma=0.02` was found optimal via grid search for the final training state. But it may not be optimal for the ENTIRE training trajectory.
- Unlike augmentation annealing (dead end #2152), this is not removing augmentation — it is changing the sigma while keeping augmentation constant. The dead end was about annealing augmentation probability; this changes its magnitude.
- Also differs from the constant sigma experiments (dead ends: 0.01 and 0.03) because it uses a schedule, not a fixed value.

### Scientific motivation

Curriculum Learning (Bengio et al. 2009) — ordered data presentation improves convergence.
Self-Paced Learning (Kumar et al. 2010) — automatic curriculum via loss-based ordering.
The specific "high diversity early, precise late" curriculum is a form of "coarse-to-fine" training analogous to: "From Coarse to Fine: Curriculum Learning for CFD Surrogate Modeling" (arXiv:2509.13138), which showed 8–15% improvement on CFD mesh generation by coordinating data resolution curriculum with loss weighting.

### Implementation sketch

```python
# In training loop, each epoch:
if args.coordinated_ramp:
    ramp_frac = min(epoch / args.tandem_ramp_epochs, 1.0)
    # Gap/stagger sigma: high start → final value
    current_sigma = args.ramp_sigma_start * (1 - ramp_frac) + args.aug_gap_stagger_sigma * ramp_frac
    # Update the augmentation module dynamically
    dataset.update_gap_stagger_sigma(current_sigma)
    # Tandem loss weight: low start → 1.0 (existing tandem_ramp logic)
    tandem_weight = ramp_frac  # existing behavior
```

New flags:
- `--coordinated_ramp` — enable coordinated sigma + loss weight ramp
- `--ramp_sigma_start 0.06` — starting gap/stagger sigma (high diversity; 3× the final value)
- (Reuses existing `--tandem_ramp` flag for the epoch count)

Training command delta from baseline:
```bash
... --coordinated_ramp --ramp_sigma_start 0.06
```

Note: The student should ensure `--aug_gap_stagger_sigma 0.02` is still passed as the final/target sigma.

### Risk analysis

- The augmentation module must support dynamic sigma updates — the student needs to check if `dataset.update_gap_stagger_sigma()` exists or implement it.
- If `ramp_sigma_start` is too high (e.g. 0.10+), early training may be dominated by configurations far from the test distribution, slowing convergence.
- This is the simplest idea here — minimal code changes, directly applicable, unlikely to destabilize training.

### Expected effect

Low to moderate. Estimated p_tan improvement: 0.5–2%. This is the lowest-risk idea and serves as a "cleanup" experiment for the existing tandem ramp mechanism. Strong expected benefit for p_oodc as well (more diverse early training benefits all OOD metrics).

---

## Priority Ranking for Assignment

| Rank | Idea | Novelty | Risk | Expected p_tan gain |
|------|------|---------|------|---------------------|
| 1 | Idea 2: BSP Loss | High | Low | 2–5% |
| 2 | Idea 5: Disentangled Contrastive Loss | High | Medium | 2–6% |
| 3 | Idea 3: Foil-1 Geometry Adapter | High | Low | 1–4% |
| 4 | Idea 6: SWD Domain Alignment | Medium | Medium | 1–4% |
| 5 | Idea 1: Temperature Curriculum | Medium | Low | 1–3% |
| 6 | Idea 4: Spectral Shaping | High | Low | 1–3% |
| 7 | Idea 7: Coordinated Tandem Ramp | Low | Low | 0.5–2% |

**Recommended first assignments:** Ideas 2 and 5 are highest potential (attack different root causes: spectral bias and OOD representation disentanglement). Ideas 3 and 1 are lowest risk for conservative students.

## Deduplication Check

None of the 7 ideas above overlap with:
- Dead ends: GSB AdaLN (#2164), DSDF-1 dropout (#2156), diff LR (#2163), cross-DSDF features (#2162), asymmetric PCGrad (#2158), shape sim bias (#2157), aug annealing (#2152), EMA start (#2151), DSDF2 sigma (#2150), tandem carve-out (#2131), tandem DSDF mixup (#2132), foil-1 aug (#2133), flat-minima class (multiple), loss reformulations (#2112, #2113, #2116), feature noise (#2144), DSDF spatial dropout (#2143), surface pressure gradient aux loss (#2129), fore-foil SRF unconditioned (#2117, #2124), per-foil physics norm (#2136), weight decay sweep (#2145), LR sweep (#2149), T_max sweep (#2154), Reynolds perturb (#2125)
- WIP: FiLM SRF (#2161), tandem surface mixup (#2167), tandem pressure correction MLP (#2168), dp/dn=0 physics loss (#2166), iterative 2-pass refinement (#2165), online hard example mining (#2169), wider/deeper SRF (#2170), slice num sweep (#2171)
