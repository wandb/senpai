# Research Ideas — 2026-04-04 22:00

Generated for Phase 6. All 8 students active. These ideas are ranked by expected p_tan impact and implementation confidence.

## Context

Current single-model baseline: p_in=13.24±0.33, p_oodc=7.73±0.22, **p_tan=30.53±0.50**, p_re=6.50±0.07.
Ensemble floor: p_tan=29.1.
PRIMARY TARGET: p_tan.

Dead ends confirmed: fore-foil SRF (all formulations), loss reweighting by surface region, FiLM on gap/stagger (catastrophe), Langevin noise, Charbonnier/Huber/SmoothL1 loss, Fourier features.
Critical interaction: gap/stagger aug (σ=0.02) hurts p_tan +1.6% but helps p_oodc -2.4%.

In-flight (do NOT duplicate): #2119 PCGrad 2-way, #2125 Re-perturb aug, #2126 Foil-2 DSDF magnitude aug, #2127 Context-aware AftSRF, #2128 Reynolds-conditional SRF, #2129 Surf-grad aux loss, #2130 Gap/stagger-conditioned spatial bias, #2131 Tandem slice carve-out.

---

## Idea 1 — Per-Foil Physics Normalization (per_foil_pnorm)

### What it is

The current `_umag_q` function computes a single domain-mean Umag from ALL mesh nodes (fore + aft + wake). For tandem samples the aft-foil sits deep in the fore-foil's wake, where local velocities are significantly lower than freestream. This means the q=0.5*Umag^2 denominator used for aft-foil Cp is physically incorrect — it overcounts the local dynamic pressure for the aft foil. The fix: for tandem samples, split the normalization so fore-foil nodes use q_fore (computed from fore-foil nodes only) and aft-foil nodes use q_aft (computed from aft-foil nodes only). Single-foil samples are completely unaffected.

### Why it might help p_tan

This is a physically motivated bias correction. The model currently sees aft-foil Cp values that are inflated by the wrong denominator relative to what the physics implies. This asymmetry is unique to tandem — for single-foil there is no wake interference. The model has to implicitly learn to undo this normalization artifact, which costs representational capacity. Giving it the correct normalization should make the aft-foil pressure targets more learnable and reduce systematic error.

### Code change locations

**`_umag_q` and `_phys_norm` / `_phys_denorm`** (lines 1028–1080):

```python
# ADD a new function:
def _umag_q_perfoil(y, mask, aft_mask):
    """For tandem: separate q for fore and aft nodes."""
    # fore = mask & ~aft_mask, aft = mask & aft_mask
    fore_mask = mask & ~aft_mask
    aft_mask_valid = mask & aft_mask
    Umag_fore, q_fore = _umag_q(y, fore_mask)
    Umag_aft, q_aft = _umag_q(y, aft_mask_valid)
    return Umag_fore, q_fore, Umag_aft, q_aft
```

In `Config` (line ~950): add `per_foil_pnorm: bool = False`

In training loop (line ~1589, after `_aft_foil_mask` is computed, before `Umag, q = _umag_q(y, mask)`):
```python
if cfg.per_foil_pnorm and _aft_foil_mask is not None:
    _is_tandem_norm = (x[:, 0, 21].abs() > 0.01)
    # ... split Umag/q per foil for tandem samples, use single for non-tandem
    # build per-node Umag/q tensors [B, 1, 1] → [B, N, 1] by node masking
```

Apply split q in `_phys_norm` call:
- For tandem samples: `y_phys[:, fore_nodes] = _phys_norm(y[:, fore_nodes], Umag_fore, q_fore)`
- For tandem samples: `y_phys[:, aft_nodes] = _phys_norm(y[:, aft_nodes], Umag_aft, q_aft)`
- For non-tandem: `y_phys = _phys_norm(y, Umag, q)` (unchanged)

Must also apply in `_phys_denorm` in the validation loop (same split logic).

**Note on aft-foil mask availability**: `_aft_foil_mask` is computed from raw (pre-normalization) x at line 1562–1566 and re-computed in the validation loop at line 2090–2095. Both sites already have the correct mask.

### Hyperparameters and risk

- No new hyperparameters
- Risk: MEDIUM-LOW. If Umag_aft is close to Umag_fore (low-wake cases), the split degenerates cleanly to the current behavior. Edge case: if aft-foil nodes have very low velocity (tight gap, strong wake), q_aft could be very small → pressure normalization blows up. Need a minimum clamp, e.g., `q_aft = q_aft.clamp(min=0.1 * q_fore)`.
- Sanity check: print mean(Umag_fore) vs mean(Umag_aft) across a few tandem batches early in training to confirm the expected gap.

### Expected impact

-2% to -5% p_tan. Physically motivated — the model has been fighting an incorrect normalization from day one for aft-foil nodes.

### LoC estimate

~25 LoC in train.py (new function + training loop insertion + val loop insertion + Config flag).

---

## Idea 2 — EMA Stochastic Weight Perturbation (ema_perturb)

### What it is

At the epoch when EMA starts accumulating (cfg.ema_start_epoch=140), inject a small Gaussian perturbation into the live model weights. The EMA then averages the recovery trajectory from this perturbation. This is a form of "flat minima seeking via stochastic perturbation": a model that recovers quickly is in a flatter basin. The trick comes from the Stochastic Weight Averaging literature (Izmailov et al., 2018) combined with EMA recovery — similar in spirit to the "perturbation + sharpness" work but simpler to implement than full SAM.

The key insight: EMA decay=0.998 means the EMA essentially accumulates over ~500 gradient steps. A single perturbation at ema_start_epoch kicks the model into a slightly different location; if the loss basin is flat (good generalization), the EMA averages the perturbed and recovered models, effectively sampling more of the flat manifold.

### Why it might help p_tan

p_tan is an out-of-distribution metric (NACA6416, unseen foil shape). Models in flatter loss basins generalize better to distribution shifts. The current setup doesn't explicitly encourage flatness. The perturbation should help explore the flat manifold without the complexity of SAM (which is incompatible with Lion).

### Code change locations

`Config` (line ~950): add `ema_perturb_sigma: float = 0.0`

In training loop, at the EMA initialization point (line ~1928-1931):
```python
if epoch >= cfg.ema_start_epoch and not cfg.swad and ...:
    if ema_model is None:
        # NEW: perturb live model once at EMA start
        if cfg.ema_perturb_sigma > 0.0:
            with torch.no_grad():
                for p in _base_model.parameters():
                    p.data.add_(torch.randn_like(p.data) * cfg.ema_perturb_sigma)
        ema_model = deepcopy(_base_model)
    else:
        # ... existing EMA update
```

### Hyperparameters

- `ema_perturb_sigma` sweep: {5e-4, 1e-3, 3e-3}
- Run 2 seeds per sigma value (seeds 42, 73) for signal detection

### Risk

MEDIUM-LOW. The perturbation only fires once. If sigma is too large, the model degrades and EMA picks up the damaged weights — but EMA decay=0.998 means the initial perturbation only has ~0.2% weight at any given step afterward. The risk is small.

### Expected impact

-1% to -3% p_tan (flat minima exploration). Less certain than B3 because it's not physically motivated — it's a generalization argument.

### LoC estimate

~12 LoC (Config flag + 5 lines at EMA init).

---

## Idea 3 — Foil-2 Independent AoA Rotation Augmentation (foil2_aoa_rot_aug)

### What it is

For tandem samples only: rotate the aft-foil nodes independently by a small angle delta_aoa2 ~ N(0, sigma) in addition to any global AoA augmentation. This requires:
1. Identifying aft-foil node indices (already done via `_aft_foil_mask` logic)
2. Rotating their (x, y) coordinates around the aft-foil centroid
3. Rotating the paired DSDF gradient channels: (2,3), (4,5), (6,7), (8,9) — same rotation matrix
4. Rotating the velocity targets (Ux, Uy) for those nodes
5. Adjusting gap/stagger scalars to remain consistent (gap changes with vertical displacement, stagger changes with horizontal)

This augmentation creates novel (fore_AoA, aft_AoA) combinations not present in the training data, significantly expanding the effective tandem training distribution.

### Why it might help p_tan

p_tan is surface MAE on NACA6416 in tandem configurations. The current training data fixes the aft-foil AoA (it's mechanically linked to the fore-foil in the dataset). Independently perturbing the aft AoA forces the model to disentangle fore-foil and aft-foil flow features — a more general representation that should transfer better to the unseen foil geometry.

The existing `aoa_perturb` augmentation (cfg.aug = "aoa_perturb") rotates ALL nodes globally. This new augmentation is aft-foil-only and tandem-only — much more targeted.

### Code change locations

In Config (line ~950): add `aug_foil2_aoa_sigma: float = 0.0`

In the gap/stagger aug block (line ~1541-1554), add after it:
```python
# Foil-2 AoA rotation augmentation (tandem only)
if cfg.aug_foil2_aoa_sigma > 0.0 and _aft_foil_mask is not None:
    _is_tan = (x[:, 0, 21].abs() > 0.01)  # pre-normalization gap
    if _is_tan.any():
        _delta = torch.randn(B, device=x.device) * cfg.aug_foil2_aoa_sigma
        _delta = _delta * _is_tan.float()  # zero for non-tandem
        _cos_d = torch.cos(_delta)  # [B]
        _sin_d = torch.sin(_delta)  # [B]
        # rotate aft-foil xy coords around aft-foil centroid
        _aft = _aft_foil_mask  # [B, N] — computed before normalization
        for b in range(B):
            if not _is_tan[b] or not _aft[b].any():
                continue
            _cx = x[b, _aft[b], 0].mean()
            _cy = x[b, _aft[b], 1].mean()
            _dx = x[b, _aft[b], 0] - _cx
            _dy = x[b, _aft[b], 1] - _cy
            x[b, _aft[b], 0] = _cos_d[b] * _dx - _sin_d[b] * _dy + _cx
            x[b, _aft[b], 1] = _sin_d[b] * _dx + _cos_d[b] * _dy + _cy
            # rotate DSDF gradient pairs
            for _xi, _yi in [(2,3),(4,5),(6,7),(8,9)]:
                _dg = x[b, _aft[b], _xi].clone()
                _eg = x[b, _aft[b], _yi].clone()
                x[b, _aft[b], _xi] = _cos_d[b] * _dg - _sin_d[b] * _eg
                x[b, _aft[b], _yi] = _sin_d[b] * _dg + _cos_d[b] * _eg
            # rotate velocity targets for aft nodes
            _ux = y[b, _aft[b], 0].clone()
            _uy = y[b, _aft[b], 1].clone()
            y[b, _aft[b], 0] = _cos_d[b] * _ux - _sin_d[b] * _uy
            y[b, _aft[b], 1] = _sin_d[b] * _ux + _cos_d[b] * _uy
```

**Important gotcha**: this augmentation must happen BEFORE `_aft_foil_mask` is recomputed (i.e., after the mask is already available but before normalization). Check line ordering carefully — `_aft_foil_mask` is computed at line 1562, and normalization `x = (x - stats["x_mean"]) / stats["x_std"]` is at line 1568. The aug block at line 1542 runs before normalization, so this fits cleanly.

**Second gotcha**: gap/stagger scalars at x[:,:,22:24] are not rotated but the aft-foil centroid shifts slightly — this inconsistency is small (delta_aoa is ~0.5 degrees max) and should be acceptable.

### Hyperparameters

- `aug_foil2_aoa_sigma` sweep: {0.003, 0.006, 0.010} rad (0.17°, 0.34°, 0.57°)
- Start with sigma=0.006 (middle value)
- Can run with `--aug_gap_stagger_sigma 0.0` to isolate effect on p_tan (avoid the negative p_tan interaction)

### Risk

MEDIUM-HIGH due to implementation complexity (~30 LoC in augmentation, per-sample loop). Primary failure mode: rotation radius too large for tight-gap tandem cases creates physically impossible geometries. Mitigate with small sigma. Student should validate that aft-foil DSDF values post-augmentation look physically plausible.

### Expected impact

-2% to -4% p_tan. More uncertain than B3 but targets the representational diversity gap directly.

### LoC estimate

~35 LoC.

---

## Idea 4 — Tandem DSDF Channel Mixup (tandem_dsdf_mixup)

### What it is

Between two tandem training samples, linearly interpolate only the DSDF feature channels (x[:,:,2:10]) using a Beta(0.7, 0.4) lambda. Keep position coordinates, Re/AoA scalars, gap/stagger, and targets fixed. This creates synthetic tandem samples with "intermediate" foil shapes while preserving the physical flow conditions. The asymmetric Beta pushes lambda toward 1.0 (minor perturbation) 70% of the time, making most augmented samples close to one real sample.

### Why it might help p_tan

p_tan is a transfer metric (NACA6416 is unseen during training). Training sees NACA0012 and similar profiles in tandem. DSDF channel mixup creates smooth interpolations between known foil shapes — synthetic tandem geometries that sit "between" the training distribution. This may improve interpolation generalization to NACA6416 which sits between the training profiles in foil-shape space.

This is different from global mixup (cfg.aug = "mixup") which mixes positions and targets too — corrupting the physical geometry-flow correspondence.

### Code change locations

In Config (line ~950): add `aug_tandem_dsdf_mixup: bool = False`, `aug_tandem_dsdf_mixup_alpha: float = 0.7`

After the gap/stagger aug block (line ~1554):
```python
if cfg.aug_tandem_dsdf_mixup:
    _is_tan_mixup = (x[:, 0, 21].abs() > 0.01)  # pre-normalization
    if _is_tan_mixup.sum() >= 2:
        _tan_idx = _is_tan_mixup.nonzero(as_tuple=False).squeeze(1)
        _perm = _tan_idx[torch.randperm(len(_tan_idx), device=x.device)]
        _beta = torch.distributions.Beta(
            torch.tensor(cfg.aug_tandem_dsdf_mixup_alpha),
            torch.tensor(0.4)
        )
        _lam = _beta.sample((_tan_idx.shape[0],)).to(x.device).view(-1, 1, 1)
        # Mix DSDF channels only (2:10), keep all other features and targets fixed
        x[_tan_idx, :, 2:10] = (
            _lam * x[_tan_idx, :, 2:10] +
            (1 - _lam) * x[_perm, :, 2:10]
        )
```

### Hyperparameters

- Beta(0.7, 0.4): asymmetric toward lambda~1 (minor perturbation)
- Can also try Beta(0.5, 0.5) for more aggressive mixup
- Apply only to tandem samples (single-foil already well-handled)

### Risk

LOW. Only DSDF channels are mixed — positions and targets are untouched. Worst case is that interpolated DSDF values are physically inconsistent (no foil actually has that shape), and the model ignores the augmentation. No catastrophic failure mode.

### Expected impact

-1% to -3% p_tan (improved foil-shape interpolation). Less certain than B3 but very low risk.

### LoC estimate

~20 LoC.

---

## Idea 5 — Aft-Foil Chord-Wise Pressure Smoothness Regularization (aft_foil_tv_loss)

### What it is

Add a chord-wise total variation (TV) regularization term on the predicted pressure for aft-foil surface nodes. For each tandem sample, sort aft-foil surface nodes by x-coordinate (chord position) and penalize |p_i - p_{i+1}| across adjacent chord positions. This acts as a physics-informed smoothness prior: pressure distributions on airfoils are smooth along the chord (no artificial discontinuities). 

This is distinct from nezuko's #2129 (supervised surface pressure gradient aux loss, which uses ground-truth pressure gradients as supervision signal). This TV loss is unsupervised — it only penalizes roughness in the prediction, not deviation from a target gradient.

### Why it might help p_tan

If the model produces noisy chord-wise pressure predictions on the aft foil (which is harder due to wake effects), TV regularization forces smoother predictions that better match the physical behavior. This is particularly relevant for NACA6416 (p_tan) which has a different camber from training foils — a smooth prior should generalize better than a memorized rough pattern.

### Code change locations

In Config (line ~950): add `aft_tv_loss_weight: float = 0.0`

After the main loss computation (line ~1798), before PCGrad block:
```python
if cfg.aft_tv_loss_weight > 0.0 and _aft_foil_mask is not None:
    _aft_surf_mask = _aft_foil_mask & surf_mask  # aft-foil surface nodes only
    if _aft_surf_mask.any():
        # Sort aft-foil nodes by x-coordinate for each sample
        _aft_tv = torch.tensor(0.0, device=device)
        _n_tv = 0
        for b in range(B):
            _node_idx = _aft_surf_mask[b].nonzero(as_tuple=False).squeeze(1)
            if _node_idx.numel() < 3:
                continue
            _x_coord = x[b, _node_idx, 0]  # chord position
            _sort_idx = _x_coord.argsort()
            _p_sorted = pred[b, _node_idx[_sort_idx], 2]  # sorted pressure predictions
            _tv = (_p_sorted[1:] - _p_sorted[:-1]).abs().mean()
            _aft_tv = _aft_tv + _tv
            _n_tv += 1
        if _n_tv > 0:
            _aft_tv = _aft_tv / _n_tv
            loss = loss + cfg.aft_tv_loss_weight * _aft_tv
```

### Hyperparameters

- `aft_tv_loss_weight` sweep: {0.01, 0.05, 0.1}
- Per-sample loop is acceptable for typical batch sizes (~4-8 tandem samples per batch)

### Risk

MEDIUM. TV regularization can over-smooth pressure predictions and cause p_tan to lose important high-frequency features near the suction peak. If weight is too large, the model will under-predict peak suction. Start with 0.01. 

Known interaction: nezuko's #2129 (surf-grad aux loss) is in-flight. If that shows a signal, the two should be compared — they are different mechanisms targeting the same failure mode. Do NOT assign this until #2129 has results.

### Expected impact

-1% to -2% p_tan. Conservative estimate because TV regularization is known to help with ood generalization in physics problems but can over-smooth.

### LoC estimate

~25 LoC.

---

## Priority Ranking

1. **B3 — Per-Foil Physics Normalization** (HIGHEST PRIORITY): Directly corrects a physical error in the normalization that affects every aft-foil Cp computation in every tandem sample. The bug has been present from the start. Low risk, no new hyperparameters, physically motivated. Expected -2% to -5% p_tan.

2. **Idea 3 — Foil-2 AoA Rotation Aug**: Targets representational diversity for the aft foil. Creates unseen (fore_AoA, aft_AoA) combinations. Higher complexity but addresses a genuine data diversity gap. Expected -2% to -4% p_tan. Assign after B3 is confirmed running.

3. **B2 — EMA Stochastic Weight Perturbation**: Low complexity, no physical motivation but solid theoretical basis (flat minima = better OOD). Can be stacked with any other improvement. Assign to a student not already running normalization changes.

4. **Idea 4 — Tandem DSDF Mixup**: Very low risk, moderate complexity. Assign if a student slot opens and higher-priority ideas are already covered.

5. **Idea 5 — Aft-Foil TV Loss**: Wait for nezuko #2129 results first (same failure mode, different mechanism). If #2129 shows no signal, assign this as the alternative.

---

## Notes on Interaction with In-Flight Experiments

- **B3 interacts with gap/stagger aug**: if per-foil normalization removes the systematic Cp error on the aft foil, the gap/stagger aug (which adds noise to those scalars) may become less destructive. Consider pairing B3 with gap/stagger aug once B3 is validated.
- **Idea 3 and fern's #2130 (gap/stagger-conditioned spatial bias)**: these target different mechanisms (data augmentation vs. slice routing). Can be stacked.
- **Idea 2 (EMA perturb) is fully orthogonal** to all architecture ideas. Safe to run in parallel with anything.
- **Idea 5 should not be assigned while #2129 is active** — they are the same failure mode hypothesis.
