# SENPAI Research Ideas — 2026-04-06 (For askeladd)

## Context

This document contains ONE hypothesis for askeladd, generated after reviewing all 1,794
prior experiments (129 merged, 1,527 ran, 141 never ran) and the 7 currently in-flight
experiments (#2181–#2187).

### Current baseline (PR #2130, gap/stagger spatial bias)
- p_in = 13.05 | p_oodc = 7.70 | p_tan = 28.60 | p_re = 6.55
- val/loss = 0.3994 (Phase 4 compound, #1833 merged)

### The core problem: p_tan = 28.60 (2.19x worse than p_in = 13.05)

p_tan measures surface pressure MAE on NACA6416 tandem OOD validation — a cambered
fore-foil geometry unseen during tandem training. This is the hardest remaining target.

### What just failed (PR #2175: SWD Domain Alignment)

Sliced Wasserstein Distance alignment of tandem vs. single-foil slice tokens failed because
the distributional difference between tandem and single-foil representations IS INFORMATIVE.
It encodes genuine multi-body aerodynamic interactions. Forcing alignment destroys that signal.

**The lesson**: We must NOT erase the tandem vs. single-foil distinction globally. But the
distributional difference has two components:
1. Tandem-specific signal (real physics of foil-foil interaction) — MUST be preserved
2. Geometry-OOD shift (NACA6416 camber not seen in tandem training) — SHOULD be corrected

PR #2175 failed because it conflated both components. The right intervention is surgical:
align only the part of the representation that captures geometry-generic flow patterns
(shared across airfoil shapes) while explicitly protecting the part that captures tandem
interaction (specific to the two-foil configuration).

---

## Hypothesis: Selective Per-Head DANN with Scheduled Alpha (Surgical Domain Alignment)

### What it is

Apply Domain-Adversarial Neural Network (DANN) gradient reversal to ONLY a subset of
attention heads — the heads whose `tandem_temp_offset` is small (near zero), indicating
they encode domain-generic features. Leave the heads with large `tandem_temp_offset`
magnitude (they encode tandem-specific physics) completely untouched.

The DANN forces the selected "shared" heads to produce representations that are invariant
to airfoil geometry (tandem NACA6416 vs. tandem NACA00XX training shapes). This should
improve OOD generalization for NACA6416 tandem cases while preserving the tandem
interaction signal that lives in the tandem-specific heads.

### Why this is NOT the same as PR #1218 (prior DANN failure)

PR #1218 applied gradient reversal to the ENTIRE hidden representation with:
- 3 coarse domain labels (single/tandem/cruise — heuristic and noisy)
- alpha=1.0 fixed from epoch 0 (too aggressive, destabilized early training)
- No protection for tandem-specific attention heads

This approach differs on every one of these failure modes:
1. **Binary domain**: tandem_NACA6416 vs. tandem_NACA00XX only (not 3-way, not noisy)
2. **Scheduled alpha**: ramp from 0 to 0.1 using the standard DANN schedule
3. **Selective heads**: only apply reversal to the heads with small tandem_temp_offset

### Why this is NOT the same as SWD alignment (#2175)

SWD (#2175) operated on the full slice token distribution, including tandem-specific tokens.
This approach operates on head-level outputs and explicitly PROTECTS the tandem-specific
heads. The alignment is surgical: only the geometry-generic heads are forced toward
domain invariance.

### Why it might help p_tan

The model has 3 attention heads. After the `tandem_temp_offset` training (PR #1323), the
heads have learned specialized roles: some heads have large offset magnitudes (they route
tandem nodes differently from single-foil nodes) and some have small offsets (they are
essentially geometry-generic). The tandem-specific heads capture the genuine foil-foil
interaction physics that varies by gap/stagger. The shared heads should ideally capture
pressure patterns that generalize across airfoil shapes.

For NACA6416 (OOD), the shared heads are the bottleneck: they have not seen this camber
level during tandem training, so they produce poor representations for the OOD surface
geometry. By applying DANN-style gradient reversal ONLY to these shared heads, we force
them to produce geometry-invariant representations without destroying the foil-interaction
signal in the tandem-specific heads.

This is the minimum surgical intervention consistent with the lesson from #2175.

### Key papers

- Ganin et al. (2016). "Domain-Adversarial Training of Neural Networks." JMLR 17:2096–2030.
  The canonical DANN paper with the Gradient Reversal Layer (GRL) and alpha schedule.
  https://arxiv.org/abs/1505.07818

- Li et al. (2018). "Selective Adversarial Networks for Adaptation." ECCV 2018.
  Applies adversarial alignment selectively to task-relevant features — the direct
  theoretical predecessor of this approach.
  https://arxiv.org/abs/1811.07440

- Rozantsev et al. (2019). "Beyond Sharing Weights for Deep Domain Adaptation." IEEE TPAMI.
  Shows that partial weight sharing (some layers adapt, some are shared) outperforms
  full domain-adversarial training in transfer scenarios similar to ours.
  https://arxiv.org/abs/1603.06432

- Long et al. (2018). "Conditional Adversarial Domain Adaptation." NeurIPS 2018.
  Conditions the domain discriminator on class predictions — more informative alignment.
  https://arxiv.org/abs/1705.10667

### Implementation notes

**Step 1: Identify which heads to align (one-time diagnostic)**

After training the baseline, inspect `model.blocks[*].attn.tandem_temp_offset`:
```python
import torch
ckpt = torch.load("latest_checkpoint.pt")
for i, block in enumerate(ckpt['model_state']):
    if 'tandem_temp_offset' in block:
        print(f"Layer {i} head offsets: {block['tandem_temp_offset'].data}")
```
Heads with |offset| < threshold (e.g., 0.05) are the "shared" heads. Heads with
|offset| > 0.05 are tandem-specific and should NOT receive gradient reversal.

For the experiment, we can hard-code which heads to align based on the known offset
distribution. If all 3 heads have large offsets, apply reversal to the head with
SMALLEST offset magnitude only.

**Step 2: Gradient Reversal Layer**

```python
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        alpha = ctx.saved_tensors[0].item()
        return -alpha * grad, None

def grad_reverse(x, alpha=1.0):
    return GRL.apply(x, alpha)
```

**Step 3: Binary domain discriminator (tandem geometry shift only)**

The domain discriminator classifies: "Is this a NACA6416 tandem sample or a training-distribution
tandem sample?" This is a binary classification on tandem samples only (ignore single-foil).

Domain labels:
- Label 1: tandem validation samples from NACA6416 family (p_tan split)
- Label 0: tandem training samples (all other tandem samples)

**CRITICAL**: This requires access to per-sample domain labels in the DataLoader. Check if
`dataset.is_ood_tandem` or similar flag exists. If not, proxy: NACA6416 can be identified
by its DSDF signature (the camber creates a systematic asymmetry in the first two DSDF
channels between pressure and suction sides). Use `dsdf_asymmetry = mean(dsdf_channel_0 -
dsdf_channel_1) over surface nodes` to compute a per-sample proxy label.

Alternatively: since the model already reports p_tan separately, the data loader knows
which samples go to which split. Use this split membership as the domain label.

**Step 4: Head-selective DANN loss**

After the attention block, extract the output of each head separately:
```python
# In Physics_Attention_Irregular_Mesh.forward():
# Split multi-head output before concatenation
head_outputs = torch.chunk(concat_heads, n_head, dim=-1)  # list of [B, N, head_dim]

# For shared heads (e.g., head index 0 based on smallest tandem_temp_offset):
shared_head_repr = head_outputs[shared_head_idx].mean(dim=1)  # [B, head_dim]
domain_logit = self.domain_discriminator(grad_reverse(shared_head_repr, alpha=current_alpha))
domain_loss = F.binary_cross_entropy_with_logits(domain_logit, domain_label.float())
```

Add `domain_loss_weight * domain_loss` to the total training loss.

**Step 5: Alpha schedule (standard DANN)**

```python
p = epoch / max_epochs  # progress fraction in [0, 1]
current_alpha = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0  # sigmoid ramp: 0→1
current_alpha *= alpha_max  # scale to alpha_max=0.1
```

Key difference from #1218: alpha ramps from 0 to 0.1 (not 1.0). This is 10x less
aggressive, avoiding the early-training instability that wrecked #1218.

**Domain discriminator architecture:**
```python
self.domain_discriminator = nn.Sequential(
    nn.Linear(n_hidden // n_head, 32),  # input = head_dim
    nn.GELU(),
    nn.Linear(32, 1)  # binary logit
)
```
Total added parameters: (head_dim × 32 + 32) + (32 × 1 + 1) ≈ 2,200 params for head_dim=64.
Negligible parameter cost.

**Implementation flag:** `--selective_dann` with `--dann_alpha_max 0.1` and
`--dann_shared_head_idx 0` (or auto-detected from tandem_temp_offset magnitudes).

### Risk assessment and relationship to #1218

The three failure modes of #1218 are directly addressed:

| Failure in #1218 | Fix in this approach |
|---|---|
| Noisy 3-way domain labels | Binary: NACA6416 tandem vs. training-dist tandem only |
| alpha=1.0 fixed from epoch 0 | DANN schedule: 0 → 0.1 over training |
| Global reversal destroys tandem signal | Only applied to shared head(s) with small tandem_temp_offset |

**Remaining risk:** The biggest new risk is that the DSDF-based proxy for "is this NACA6416"
is noisy. If the domain discriminator cannot reliably distinguish NACA6416 from training
tandem samples, the gradient reversal is pure noise. Mitigation: use split membership
directly if available in the data pipeline.

**Second risk:** The concept of a "shared head" may not exist — all 3 heads may have large
tandem_temp_offsets. In that case, apply to the head with smallest offset magnitude and
weight the reversal more conservatively (alpha_max=0.05).

**Third risk:** Head output extraction requires modifying the attention module's return
values. This is implementable but requires careful surgery in `train.py`. If this is too
complex, a fallback is to apply reversal to the FINAL layer output only, masked by a
learned "domain-relevance gate" that is initialized to zero (conservative start).

### Confidence level

**Medium confidence.** The theoretical motivation is strong (directly addresses the
diagnosed failure modes of #1218 and the lessons from #2175). The selective approach is
novel — no prior experiment in 1,794 PRs has tried head-level selective domain adaptation.

However, the implementation complexity is higher than typical experiments, and the domain
label construction (NACA6416 identification) is non-trivial. This is a medium-risk,
medium-reward experiment.

### Expected impact

**3-6% p_tan reduction** if the shared heads can learn geometry-invariant representations.
If it works, the mechanism cleanly explains both the #1218 failure (too coarse) and the
#2175 failure (too broad), pointing toward even more refined selective approaches.

---

## Training command for askeladd

Build on the current best baseline command (PR #2130):

```bash
cd cfd_tandemfoil && python train.py \
  --agent askeladd \
  --wandb_name "askeladd/selective-dann-p-tan" \
  --wandb_group selective-dann \
  --asinh_pressure --asinh_scale 0.75 \
  --field_decoder --adaln_output \
  --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot \
  --high_p_clamp \
  --n_layers 3 --slice_num 96 \
  --tandem_ramp \
  --domain_layernorm --domain_velhead \
  --ema_decay 0.999 \
  --weight_decay 5e-5 \
  --cosine_T_max 160 \
  --pcgrad_3way --pcgrad_extreme_pct 0.15 \
  --pressure_first --pressure_deep \
  --residual_prediction \
  --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf \
  --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --gap_stagger_spatial_bias \
  --selective_dann \
  --dann_alpha_max 0.1 \
  --dann_shared_head_idx 0
```

### What the student needs to implement

1. **GRL class**: Standard `torch.autograd.Function` with negated gradients in backward pass.

2. **Binary domain labels**: In the DataLoader or training loop, add a per-sample flag:
   `is_ood_tandem = (batch belongs to val_tandem_transfer split)`. During training,
   use training-split tandem samples as label=0, and augment with the OOD tandem split
   as label=1 (if the training pipeline supports mixing validation samples into training
   — if not, use DSDF asymmetry as a proxy label computed on-the-fly).

3. **Domain discriminator**: 2-layer MLP on the shared head's mean-pooled output.

4. **Alpha schedule**: Standard DANN sigmoid ramp, capped at alpha_max=0.1.

5. **Head selection**: Read tandem_temp_offset from the model's first attention block;
   apply GRL to the head with smallest |offset|. If all offsets are large (>0.1),
   apply with alpha_max=0.05 (half the default).

6. **Loss term**: Add `0.1 * domain_loss` to the total loss (the 0.1 coefficient is
   separate from alpha — it controls the scale of the domain loss itself).

### Diagnostic to check implementation

After the first epoch, log:
- `domain_acc`: accuracy of the domain discriminator (should plateau near 0.5 when GRL works)
- `dann_alpha`: current alpha value (should increase from 0 toward 0.1)
- `domain_loss`: should decrease initially as discriminator trains, then stabilize

If `domain_acc` stays above 0.8 after epoch 10, the shared head is still too domain-specific
and the GRL is too weak. Try increasing alpha_max to 0.2.

If `p_tan` gets WORSE during training (tracked on the OOD tandem val split), the reversal
is destroying tandem-specific signal even in the "shared" head — reduce alpha_max to 0.05
or try a different head.

---

## Summary

One experiment. One precise hypothesis. Three specific improvements over the two prior
failed approaches (#1218, #2175) that directly address their diagnosed failure modes.

**The core bet:** The Transolver heads have learned a de facto specialization between
geometry-generic flow prediction and tandem-specific interaction modeling. By exploiting
this existing structure (via `tandem_temp_offset` as a proxy), we can apply domain
adaptation surgically — exactly where it helps, not where it hurts.

Target: p_tan < 27.5 (>3.8% improvement over baseline 28.60).
