# Research Ideas — 2026-04-06 (Thorfinn Round)

Generated after PR #2209 (register tokens) closure. Focus: push p_tan below 28.0.
Current baseline: p_in=12.490, p_oodc=7.618, p_tan=28.521, p_re=6.411.

---

## Idea 1: PirateNets Gated Adaptive Residuals (`pirate-residuals`)

**Confidence:** HIGH (LOW risk, strong theoretical basis, zero-init safe)

### Hypothesis

Standard residual connections in Transolver use fixed `x = x + f(x)` blending. PirateNets (Schiassi et al., 2023; Bora et al., 2024 ICML) replace this with a learnable gate `x = (1 - tanh(s)) * x + tanh(s) * f(x)` where `s` is a scalar initialized to 0. At init, `tanh(0) = 0`, so the full pass is identity — guaranteed baseline-equivalent start. As training progresses, each block learns how much to mix residual vs. transformed representation. This is directly motivated by the spectral bias literature: standard residuals force equal weighting of all frequency components, while gated residuals let the model adaptively weight high-frequency corrections (which dominate p_tan error near the aft-foil leading edge). The DCT frequency loss success (in baseline) already confirms high-frequency spectral bias is a real problem here.

**Target**: p_tan — aft-foil pressure is where high-frequency wake interactions occur; adaptive gating may help the deeper blocks focus on these corrections.

### Implementation

Add flag `--pirate_residuals` (boolean). In `TransolverBlock.forward`, replace:
```python
x = x + self.ffn(x)
```
with:
```python
if self.cfg.pirate_residuals:
    gate = torch.tanh(self.residual_gate)  # scalar ∈ (-1, 1), init near 0
    x = (1.0 - gate) * x + gate * self.ffn(x)
else:
    x = x + self.ffn(x)
```

In `TransolverBlock.__init__`, add:
```python
if cfg.pirate_residuals:
    self.residual_gate = nn.Parameter(torch.zeros(1))
```

Apply to ALL 3 TransolverBlocks (each gets its own scalar gate).

Also apply the same gating to the attention branch:
```python
# For attn branch:
if self.cfg.pirate_residuals:
    gate_attn = torch.tanh(self.attn_gate)
    x = (1.0 - gate_attn) * x + gate_attn * attn_out
else:
    x = x + attn_out
```
with `self.attn_gate = nn.Parameter(torch.zeros(1))` at init.

Total new parameters: 6 scalars (2 gates × 3 blocks). Negligible.

### Training Command

```bash
cd cfd_tandemfoil && python train.py --agent thorfinn --wandb_name "thorfinn/pirate-residuals" \
  --wandb_group "pirate-residuals" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pcgrad_3way --pcgrad_extreme_pct 0.15 \
  --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --gap_stagger_spatial_bias \
  --dct_freq_loss --dct_freq_weight 0.05 --dct_freq_gamma 2.0 --dct_freq_alpha 1.5 \
  --te_coord_frame \
  --pirate_residuals \
  --seed 42
```

Run seed 42 first. If p_tan < 28.52 at seed 42, run seed 73.

### Risk Assessment

**LOW.** Zero-init guarantees exact baseline behavior at step 0. Lion optimizer handles scalar parameters fine (sign updates on near-zero scalars will just flip between small positive and negative, converging quickly). PCGrad surgery unaffected (gates are not task-specific). EMA unaffected. ~10 lines of new code.

**Watch for:** Gates that saturate to ±1 (check histogram). If all gates converge to tanh=1, it means the model prefers fully transformed representation — consider constraining with L2 regularization on gate values if this causes instability.

---

## Idea 2: GeoTransolver GALE — Geometry-Latent Cross-Attention (`geotransolver-gale`)

**Confidence:** MEDIUM-HIGH (MEDIUM risk, directly targets OOD shape generalization)

### Hypothesis

The core OOD failure on p_tan is that NACA6416 (test) has different camber, chord and thickness distributions from NACA0012 (training). The backbone sees dsdf features that encode proximity to geometry, but no explicit geometric "shape identity" signal is injected into the global slice tokens. GeoTransolver (arXiv 2412.14171) showed that injecting a geometry latent into the Transolver attention blocks via cross-attention significantly improves OOD generalization on airfoil shape families.

Mechanism: pool surface node features → fixed-size geometry latent vector (per foil) → cross-attend slice tokens against geometry latent at each TransolverBlock. The slice tokens now carry explicit shape conditioning, so the physics-attention routing adapts to the geometry being processed rather than relying solely on node-local dsdf features.

The TE coordinate frame (PR #2207) confirmed that adding explicit geometric reference signals helps OOD. This is the next level: instead of fixed geometric offsets, inject a learned geometric representation into the global routing mechanism.

**Target**: p_tan primarily (NACA6416 OOD), secondarily p_oodc.

### Implementation

Add flag `--geotransolver_gale` (boolean) and `--gale_latent_dim 32` (int, default 32).

Add a `GeometryEncoder` module:
```python
class GeometryEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
    def forward(self, surface_x, surface_mask):
        # surface_x: [B, N_surf, D], surface_mask: [B, N_surf] bool
        masked = surface_x * surface_mask.unsqueeze(-1).float()
        pooled = masked.sum(1) / surface_mask.float().sum(1, keepdim=True).clamp(min=1)
        return self.proj(pooled)  # [B, latent_dim]
```

Separately encode fore-foil and aft-foil surface nodes, concatenate → `geom_latent` of dim `2 * latent_dim`.

In each `TransolverBlock`, add a cross-attention layer:
```python
# slice_tokens: [B, slice_num, D]
# geom_latent: [B, 2*latent_dim]
geom_q = self.geom_q_proj(slice_tokens)   # [B, S, D]
geom_k = self.geom_k_proj(geom_latent.unsqueeze(1))  # [B, 1, D]
geom_v = self.geom_v_proj(geom_latent.unsqueeze(1))  # [B, 1, D]
cross_out = F.scaled_dot_product_attention(geom_q, geom_k, geom_v)
# Zero-init output projection to start as identity
cross_out = self.geom_out_proj(cross_out)  # init weights=0
slice_tokens = slice_tokens + cross_out   # additive
```

Pass `geom_latent` through the TransolverBlock forward call. Use `in_dim = n_hidden = 192`.

Total new parameters: ~3 × (4 × 192 × 32 + ...) ≈ ~75K params. Small relative to backbone.

### Training Command

```bash
cd cfd_tandemfoil && python train.py --agent thorfinn --wandb_name "thorfinn/geotransolver-gale" \
  --wandb_group "geotransolver-gale" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pcgrad_3way --pcgrad_extreme_pct 0.15 \
  --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --gap_stagger_spatial_bias \
  --dct_freq_loss --dct_freq_weight 0.05 --dct_freq_gamma 2.0 --dct_freq_alpha 1.5 \
  --te_coord_frame \
  --geotransolver_gale --gale_latent_dim 32 \
  --seed 42
```

### Risk Assessment

**MEDIUM.** The cross-attention is zero-init (additive, baseline-equivalent at step 0). Main risk: the geometry encoder may not pool meaningful signals from surface nodes on the first pass — use a warmup LR schedule to let geometry encoder learn before the gates open. The 32-dim latent is small enough that overfitting is unlikely. If GALE improves p_tan but hurts p_in, consider applying GALE only to aft-foil-related slice tokens (masking).

**Known gotcha**: Surface node indices must be correctly separated into fore vs. aft foil masks — use the existing `is_tandem`, `foil2_mask` flags already in the data pipeline.

---

## Idea 3: Domain-Conditional LayerNorm in AftSRF Only (`domain-split-srf-norm`)

**Confidence:** MEDIUM-HIGH (LOW-MEDIUM risk, targeted at the OOD-sensitive component)

### Hypothesis

The AftFoilRefinementHead (AftSRF) is the component most sensitive to OOD geometry because it directly predicts aft-foil surface pressure. Currently it uses standard LayerNorm. The hypothesis: replace LayerNorm in AftSRF with domain-conditional LayerNorm (separate scale/bias embeddings indexed by `is_tandem`). This lets the SRF head learn domain-specific normalization statistics — in-distribution tandem (NACA0012 fore) vs. OOD tandem (NACA6416 fore).

This is EXPLICITLY different from the failed PR #2164 (domain AdaLN applied to ALL backbone blocks), which was too disruptive. This targets only AftSRF — the 3-layer MLP that already specializes in tandem aft-foil predictions. The backbone remains unchanged.

Implementation is zero-init: the delta embeddings start at zero, so at step 0 behavior is identical to standard LayerNorm. The model only deviates when it has learned domain-specific statistics.

**Target**: p_tan and p_oodc (both involve tandem configurations with potentially OOD fore geometries).

### Implementation

Add flag `--domain_split_srf_norm` (boolean).

In `AftFoilRefinementHead.__init__`, replace:
```python
self.norms = nn.ModuleList([nn.LayerNorm(n_hidden) for _ in range(n_layers)])
```
with:
```python
if cfg.domain_split_srf_norm:
    self.norms = nn.ModuleList([nn.LayerNorm(n_hidden) for _ in range(n_layers)])
    # Delta embeddings: [2, 2*n_hidden] — first n_hidden = scale delta, second = bias delta
    self.domain_delta = nn.Embedding(2, 2 * n_hidden)
    nn.init.zeros_(self.domain_delta.weight)
else:
    self.norms = nn.ModuleList([nn.LayerNorm(n_hidden) for _ in range(n_layers)])
```

In `AftFoilRefinementHead.forward`, for each layer:
```python
h = self.norms[i](h)
if self.cfg.domain_split_srf_norm:
    # domain_idx: [B] LongTensor, 1 for tandem, 0 for single
    delta = self.domain_delta(domain_idx)  # [B, 2*n_hidden]
    d_scale = delta[:, :n_hidden].unsqueeze(1)  # [B, 1, n_hidden]
    d_bias  = delta[:, n_hidden:].unsqueeze(1)  # [B, 1, n_hidden]
    h = h * (1.0 + d_scale) + d_bias
```

The `domain_idx` flag is already tracked in the training loop for tandem vs. single cases.

Total new parameters: 2 × 2 × 192 = 768 scalars per layer × 3 layers = 4608 params. Negligible.

### Training Command

```bash
cd cfd_tandemfoil && python train.py --agent thorfinn --wandb_name "thorfinn/domain-split-srf-norm" \
  --wandb_group "domain-split-srf-norm" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pcgrad_3way --pcgrad_extreme_pct 0.15 \
  --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --gap_stagger_spatial_bias \
  --dct_freq_loss --dct_freq_weight 0.05 --dct_freq_gamma 2.0 --dct_freq_alpha 1.5 \
  --te_coord_frame \
  --domain_split_srf_norm \
  --seed 42
```

### Risk Assessment

**LOW-MEDIUM.** Zero-init delta embeddings guarantee baseline-equivalent start. The only risk is that `domain_idx` batches may not always be pure single or pure tandem — verify that the training loop correctly assigns domain identity per sample before applying delta. If batches mix domains (which is expected), indexing must be per-sample, not per-batch.

**Distinction from dead-end PR #2164**: That PR applied domain conditioning to ALL backbone TransolverBlocks. This applies ONLY to the 3-layer AftSRF MLP, which is already domain-specific by design (gated to run only for tandem cases). The failure mode of #2164 was backbone-wide disruption; this is localized and cannot cause that failure.

---

## Idea 4: Slice Diversity Regularization (`slice-diversity-reg`)

**Confidence:** MEDIUM (MEDIUM risk, addresses routing collapse hypothesis for OOD)

### Hypothesis

The register tokens experiment (PR #2209) was motivated by the hypothesis that OOD inputs (NACA6416) cause attention routing collapse — all nodes assigned to the same few slice "supernodes," losing representational diversity. Thorfinn confirmed register tokens don't help (expected: Transolver slices are already globally aggregated, unlike ViT patches).

BUT: the underlying routing collapse concern may still be valid even if register tokens aren't the fix. A Gram matrix orthogonality penalty on the slice attention weights directly encourages slice diversity without adding architectural parameters. If slices 0–95 are encouraged to attend to diverse subsets of the domain, OOD inputs get a richer multi-scale representation.

Mechanism: after computing the soft assignment matrix A ∈ R^{B × N × S} (nodes × slices), enforce:
`L_diversity = λ * ||A^T A / N - I_S||_F^2`

This penalizes slices from co-activating. At training time, this regularizes the routing to be more diverse; at test time on OOD inputs, the routing has been pre-conditioned to use all S=96 slices rather than collapsing to a few.

**Target**: p_tan (OOD routing diversity). Secondary benefit: may improve p_oodc.

### Implementation

Add flags `--slice_diversity_reg` (boolean) and `--slice_diversity_weight 0.01` (float).

In the Transolver attention forward pass, after computing assignment matrix `A`:
```python
if self.cfg.slice_diversity_reg and self.training:
    # A: [B, N, S] — soft assignment weights (post-softmax)
    A_flat = A.view(B * N, S)  # merge batch and node dims
    A_norm = A_flat / (A_flat.norm(dim=0, keepdim=True) + 1e-8)  # normalize per-slice
    G = torch.mm(A_norm.t(), A_norm)  # [S, S] Gram matrix
    I = torch.eye(S, device=G.device)
    diversity_loss = self.cfg.slice_diversity_weight * ((G - I) ** 2).mean()
    # Accumulate into total loss (return alongside main output)
```

The diversity loss must be accumulated across all 3 TransolverBlocks and added to the total training loss. Either pass it back through the return value or use a module-level accumulator.

Alternative: apply only to the middle block (block index 1) where routing collapse is most detrimental. Start with all 3 blocks, ablate if needed.

**Start value**: `--slice_diversity_weight 0.01`. If p_tan improves but surface MAE degrades, reduce to 0.005. If no effect, try 0.05.

### Training Command

```bash
cd cfd_tandemfoil && python train.py --agent thorfinn --wandb_name "thorfinn/slice-diversity-reg" \
  --wandb_group "slice-diversity-reg" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pcgrad_3way --pcgrad_extreme_pct 0.15 \
  --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --gap_stagger_spatial_bias \
  --dct_freq_loss --dct_freq_weight 0.05 --dct_freq_gamma 2.0 --dct_freq_alpha 1.5 \
  --te_coord_frame \
  --slice_diversity_reg --slice_diversity_weight 0.01 \
  --seed 42
```

### Risk Assessment

**MEDIUM.** The Gram penalty can conflict with PCGrad's gradient surgery if it produces gradients that dominate the p_tan task gradient. Monitor the `diversity_loss` magnitude relative to total loss — it should be < 5% of total. The weight 0.01 is conservative. Key failure mode: if slice diversity is already high (no collapse), this penalty wastes optimization budget. Check slice entropy histogram in W&B to verify collapse is occurring on OOD inputs before committing to this direction.

**Distinction from register tokens (#2209)**: Register tokens ADD capacity for sinks. Diversity regularization CONSTRAINTS existing capacity to stay spread. These are complementary hypotheses; the fact that register tokens failed doesn't invalidate the diversity concern.

---

## Idea 5: Additive Fore→Aft Cross-Attention in AftSRF (`additive-fore-aft-crossattn-srf`)

**Confidence:** MEDIUM (MEDIUM risk, targeted retry of PR #2202 with correct additive formulation)

### Hypothesis

PR #2202 tested fore-foil → aft-foil cross-attention as a REPLACEMENT for the existing SRF MLP path. It caused instability (p_tan +2.1%) and was closed. The post-mortem identified the failure mechanism: replacing (not augmenting) the existing path removed the stable MLP prediction, leaving only the cross-attention output which had not yet learned a meaningful representation.

The correct formulation is ADDITIVE: keep the existing AftSRF MLP path unchanged, add a parallel cross-attention branch with zero-init output projection, and sum the outputs. At step 0, the cross-attention branch contributes exactly zero (zero-init), so behavior is identical to the current baseline. As training proceeds, the branch learns to extract physical information from the fore-foil surface hidden states — specifically, wake deficit patterns that are geometrically meaningful for aft-foil pressure prediction.

Physical motivation: each aft-foil surface node's pressure is directly influenced by the local velocity deficit from the fore-foil wake. Cross-attention from aft-foil surface nodes to fore-foil surface nodes (using their hidden states) gives the AftSRF access to the most geometrically proximate fore-foil flow information. The TE coordinate frame features already encode spatial relationships; this adds dynamic wake interaction information encoded in the hidden states.

**Target**: p_tan primarily (aft-foil pressure directly governed by fore-foil wake interaction).

### Implementation

Add flag `--additive_fore_aft_crossattn` (boolean) and `--fore_aft_crossattn_heads 4` (int).

In `AftFoilRefinementHead.__init__`, add:
```python
if cfg.additive_fore_aft_crossattn:
    n_heads = cfg.fore_aft_crossattn_heads  # default 4
    self.fore_aft_crossattn = nn.MultiheadAttention(
        embed_dim=n_hidden,
        num_heads=n_heads,
        batch_first=True
    )
    # Zero-init output projection to start as identity correction
    nn.init.zeros_(self.fore_aft_crossattn.out_proj.weight)
    nn.init.zeros_(self.fore_aft_crossattn.out_proj.bias)
```

In `AftFoilRefinementHead.forward`, after computing the base SRF output `h`:
```python
if self.cfg.additive_fore_aft_crossattn and fore_surface_h is not None:
    # aft_h: [B, N_aft, n_hidden] — aft foil surface hidden states
    # fore_surface_h: [B, N_fore, n_hidden] — fore foil surface hidden states (passed in)
    cross_out, _ = self.fore_aft_crossattn(
        query=aft_h,          # [B, N_aft, n_hidden]
        key=fore_surface_h,   # [B, N_fore, n_hidden]
        value=fore_surface_h, # [B, N_fore, n_hidden]
    )
    h = h + cross_out  # additive; cross_out=0 at step 0 due to zero-init out_proj
```

The fore-foil surface hidden states must be passed through from the backbone output. These are already extracted for other purposes (e.g., gap_stagger_spatial_bias uses foil-2 features). Ensure `fore_surface_h` is detached or not — experiment with both; detached is safer initially.

Total new parameters: MHA with 4 heads, embed=192 → ~4 × 192² / 64 × 4 ≈ ~150K params. Reasonable.

### Training Command

```bash
cd cfd_tandemfoil && python train.py --agent thorfinn --wandb_name "thorfinn/additive-fore-aft-crossattn-srf" \
  --wandb_group "additive-fore-aft-crossattn-srf" \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pcgrad_3way --pcgrad_extreme_pct 0.15 \
  --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --gap_stagger_spatial_bias \
  --dct_freq_loss --dct_freq_weight 0.05 --dct_freq_gamma 2.0 --dct_freq_alpha 1.5 \
  --te_coord_frame \
  --additive_fore_aft_crossattn --fore_aft_crossattn_heads 4 \
  --seed 42
```

### Risk Assessment

**MEDIUM.** The zero-init out_proj guarantees baseline-equivalent start. Main implementation risk: correctly passing fore-foil surface hidden states to the AftSRF forward method — this requires threading them through the model forward pass. Fore-foil and aft-foil surface masks are already tracked. If the fore-foil hidden states come from the backbone output before AftSRF, they are frozen relative to the cross-attention pass (no circular dependency).

**Key difference from PR #2202**: That PR REPLACED the SRF computation. This ADDS to it. The zero-init out_proj is non-negotiable — without it, the initial cross-attention output is random noise that corrupts the existing SRF predictions.

---

## Summary Ranking

| Rank | Slug | Confidence | Risk | p_tan Mechanism |
|------|------|-----------|------|----------------|
| 1 | pirate-residuals | HIGH | LOW | Adaptive spectral weighting per block |
| 2 | geotransolver-gale | MEDIUM-HIGH | MEDIUM | OOD shape geometry conditioning |
| 3 | domain-split-srf-norm | MEDIUM-HIGH | LOW-MEDIUM | Domain-conditional normalization in AftSRF |
| 4 | additive-fore-aft-crossattn-srf | MEDIUM | MEDIUM | Dynamic wake interaction via cross-attention |
| 5 | slice-diversity-reg | MEDIUM | MEDIUM | Routing diversity for OOD slice assignment |
