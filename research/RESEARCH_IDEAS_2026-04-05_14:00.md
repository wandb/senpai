# SENPAI Research Ideas — 2026-04-05 14:00

Generated after reviewing train.py architecture, 1,765 experiment PRs, and the current research state.

**Current baseline (PR #2130, GSB + PCGrad, 2-seed avg):**
p_in=13.05, p_oodc=7.70, **p_tan=28.60**, p_re=6.55

**Primary target:** push p_tan below 28.0 (NACA6416 OOD tandem transfer gap).

**Active WIP (do NOT duplicate):**
- askeladd #2150 — DSDF2 Sigma (σ={0.03, 0.08})
- tanjiro #2156 — DSDF-1 Channel Dropout (p={0.2, 0.3})
- fern #2151 — EMA Start Epoch ({100, 120})
- alphonse #2157 — Foil Shape Similarity Bias (GSB 6D→7D)
- nezuko #2152 — Augmentation Annealing
- thorfinn #2154 — Cosine T_max Sweep ({140, 180})
- frieren #2153 — Gap/Stagger Sigma Increase (σ=0.03)
- edward #2158 — Asymmetric PCGrad

---

## Hypothesis 1: film-conditioned-fore-foil-srf

**Slug:** `film-conditioned-fore-foil-srf`

**What it is:** An additive correction MLP for fore-foil surface nodes (boundary proxy: `_raw_saf_norm <= 0.005` AND `_is_tandem`) that is FiLM-conditioned on the raw DSDF1 magnitude statistics (mean and std of `x[:, :, 2:6].norm(dim=-1)` over foil-1 surface nodes). The conditioning signal encodes the shape fingerprint of foil-1 — distinct from the failed unconditional fore-foil SRF attempts (#2117, #2124) which used unconditioned additive corrections and worsened p_tan.

**Why it helps p_tan:** All prior fore-foil SRF attempts failed because they applied a single learned correction without knowing which foil shape (NACA6416 vs NACA0015 etc.) is present. FiLM conditioning on DSDF1 surface statistics encodes a compact shape fingerprint: NACA6416 has a specific DSDF1 gradient profile on its surface that differs from NACA0015. This allows the fore-foil correction head to specialize to the NACA6416 shape — exactly the OOD geometry responsible for the p_tan gap. The aft-foil SRF head (already merged, using FiLM on gap/stagger 2D) validated the pattern: conditioning the correction on geometry context unlocks the gain that unconditioned correction could not achieve.

**Key differentiator from dead-ends:** #2117 and #2124 failed because they used unconditioned corrections that learned a fixed per-foil-position bias, which caused overfitting to training shapes and worsened NACA6416 transfer. FiLM conditioning on shape fingerprint (DSDF1 statistics) introduces shape-aware routing rather than fixed bias.

**Code location:** In `train.py`:

1. Define `ForeAftFoilRefinementHead` class mirroring `AftFoilRefinementHead` (lines ~440-520), but:
   - FiLM conditioning input: 4D vector `[dsdf1_mean, dsdf1_std, gap, stagger]` instead of 2D gap/stagger
   - The 4D conditioning signal: `_fore_dsdf1_stats = x[:, :, 2:6].norm(dim=-1)` → compute mean/std over fore-foil surface nodes only, concatenate with `_raw_gap_stagger`

2. Build fore-foil mask alongside `_aft_foil_mask` (line ~1664):
   ```python
   _fore_foil_mask = is_surface & (_raw_saf_norm <= 0.005) & _is_tandem.unsqueeze(1)
   ```

3. Compute shape fingerprint for FiLM conditioning:
   ```python
   # Fore-foil DSDF1 magnitude stats as shape fingerprint
   _dsdf1_mag = x[:, :, 2:6].norm(dim=-1)  # [B, N] — raw BEFORE standardization
   _fore_dsdf1_mean = (_dsdf1_mag * _fore_foil_mask.float()).sum(dim=1) / _fore_foil_mask.float().sum(dim=1).clamp(min=1)
   _fore_dsdf1_std  = ((_dsdf1_mag - _fore_dsdf1_mean.unsqueeze(1)) ** 2 * _fore_foil_mask.float()).sum(dim=1)
   _fore_dsdf1_std  = (_fore_dsdf1_std / _fore_foil_mask.float().sum(dim=1).clamp(min=1)).sqrt()
   _fore_film_cond  = torch.stack([_fore_dsdf1_mean, _fore_dsdf1_std, _raw_gap_stagger[:, 0], _raw_gap_stagger[:, 1]], dim=-1)  # [B, 4]
   ```

4. Apply correction in training loop (after aft-foil correction block, ~line 1835):
   ```python
   if fore_srf_head is not None and model.training and _fore_foil_mask is not None:
       fore_idx = _fore_foil_mask.nonzero(as_tuple=False)
       if fore_idx.numel() > 0:
           fore_hidden = hidden[fore_idx[:, 0], fore_idx[:, 1]]
           fore_pred   = pred[fore_idx[:, 0], fore_idx[:, 1]]
           _fore_cond  = _fore_film_cond[fore_idx[:, 0]]
           with torch.amp.autocast("cuda", dtype=torch.bfloat16):
               fore_correction = fore_srf_head(fore_hidden, fore_pred, _fore_cond).float()
           pred = pred.clone()
           pred[fore_idx[:, 0], fore_idx[:, 1]] += fore_correction
   ```

5. New flags: `--fore_foil_srf_film` (bool), `--fore_foil_srf_hidden 192`, `--fore_foil_srf_layers 3`. Add to Config dataclass.

**Concrete hyperparams:**
- `fore_foil_srf_hidden=192`, `fore_foil_srf_layers=3` (matches aft-foil SRF baseline)
- FiLM input dim=4, hidden=32 (matches `AftFoilRefinementHead.film_mlp` design at ~line 458)
- Zero-initialize output layer weight (same as `AftFoilRefinementHead`)
- Same `base_lr` as aft-foil head (no differential LR needed at this stage)

**Risk:** MEDIUM. Core mechanism validated by aft-foil SRF success; the unconditional version failed but the conditioned variant is materially different. Implementation is ~60 LoC mirroring existing pattern.

**Estimated LoC:** ~65 (class definition ~35 + training loop integration ~30, mostly copy-paste from `AftFoilRefinementHead` with 4D conditioning).

**Expected p_tan gain:** -2 to -5% (i.e., 28.60 → 27.2 to 28.0). Conservative because fore-foil surface error is smaller than aft-foil error, so headroom is lower.

**Confidence:** Moderate-strong. Mechanism directly addresses the NACA6416 OOD gap via shape-conditional correction; prior failures were due to lack of conditioning, not the correction concept itself.

---

## Hypothesis 2: tandem-cross-dsdf-features

**Slug:** `tandem-cross-dsdf-features`

**What it is:** Compute two explicit inter-foil geometry features from existing raw DSDF channels and inject them as additional scalar inputs to every mesh node in tandem samples: (a) `dsdf_dist_ratio = dsdf2_min / (dsdf1_min + dsdf2_min + 1e-4)` — the relative proximity of each node to foil-2 vs foil-1, ranging [0,1]; (b) `dsdf_relative_angle = atan2(dsdf2_gy_mean, dsdf2_gx_mean) - atan2(dsdf1_gy_mean, dsdf1_gx_mean)` — the angular difference of gradient directions between the two foil SDFs, encoding their geometric relationship. These are zero for non-tandem samples.

**Why it helps p_tan:** The current GSB (Gap-Stagger Spatial Bias, confirmed -3.0% p_tan) injects gap and stagger as scalar inputs. But gap and stagger describe the global tandem configuration — not the local relationship of each node to each foil. The dist_ratio feature encodes "how much of this node's neighborhood is influenced by foil-2 vs foil-1" at node granularity, which directly correlates with the inter-foil pressure channel problem. The relative angle encodes foil orientation alignment — a key predictor of interference lift. These are explicit physics-motivated features that the model could only learn indirectly from DSDF values alone.

**Code location:** In `train.py`, immediately before standardization (after the `_raw_gap_stagger` extraction, ~line 1668):

```python
# Tandem cross-DSDF features (explicit inter-foil geometry, zero for non-tandem)
if cfg.tandem_cross_dsdf:
    _dsdf1_min = x[:, :, 2:6].abs().min(dim=-1, keepdim=True).values   # [B, N, 1]
    _dsdf2_min = x[:, :, 6:10].abs().min(dim=-1, keepdim=True).values  # [B, N, 1]
    _dist_ratio = _dsdf2_min / (_dsdf1_min + _dsdf2_min + 1e-4)        # [B, N, 1] in [0,1]
    # Gradient direction difference: atan2(dsdf2_gy, dsdf2_gx) - atan2(dsdf1_gy, dsdf1_gx)
    _dsdf1_angle = torch.atan2(x[:, :, 3:4], x[:, :, 2:3])  # foil-1 gradient direction
    _dsdf2_angle = torch.atan2(x[:, :, 7:8], x[:, :, 6:7])  # foil-2 gradient direction
    _rel_angle = _dsdf2_angle - _dsdf1_angle  # [B, N, 1], range ~[-pi, pi]
    # Zero out for non-tandem samples
    _tandem_mask_f = (x[:, 0, 22].abs() > 0.01).float().view(-1, 1, 1)
    _dist_ratio = _dist_ratio * _tandem_mask_f
    _rel_angle  = _rel_angle  * _tandem_mask_f
    # Append to x before standardization
    x = torch.cat([x, _dist_ratio, _rel_angle], dim=-1)   # X_DIM becomes 26
```

After this, standardization (`x = (x - stats["x_mean"]) / stats["x_std"]`) runs over the new dim. The stats dict must accommodate the extra channels — use running statistics or a static zero/one initialization for the new dims (simplest: append zeros to `stats["x_mean"]` and ones to `stats["x_std"]` so new channels are already normalized; the model will learn their scale from data).

New flag: `--tandem_cross_dsdf` (bool, default False). Requires adjusting `fun_dim` by +2. Since `fun_dim` is computed as `x.shape[-1]` inside the loop, the model projection layer auto-adapts if the model is instantiated AFTER the first batch processes the new dim — but for `torch.compile`, the input shape must be static. The cleanest approach: compute the new X_DIM at model construction time using the same flag and pass it via `model_config`.

**Concrete hyperparams:**
- New dim: +2 (dist_ratio, rel_angle)
- No additional hyperparams needed
- Run 2 seeds (s42, s73) as standard

**Risk:** LOW-MEDIUM. Feature computation is pure tensor arithmetic, no new parameters. The only implementation risk is the static input-dim requirement for `torch.compile` — resolve by conditionally setting X_DIM=26 when flag is set.

**Estimated LoC:** ~25 (feature computation ~15 + config/model-dim bookkeeping ~10).

**Expected p_tan gain:** -2 to -4%. GSB proved that explicit tandem geometry scalars help; this extends the idea from global to per-node geometry.

**Confidence:** Moderate. The feature design is physically motivated and directly computable from existing inputs. The question is whether the model can't already learn these features implicitly from the 8 DSDF channels — explicit injection removes that burden.

---

## Hypothesis 3: differential-lr-specialized-heads

**Slug:** `differential-lr-specialized-heads`

**What it is:** Apply a higher learning rate multiplier to the specialized heads (aft-foil SRF head, surface refinement head, GSB spatial bias MLP) relative to the Transolver backbone. Concretely: backbone attn params get `lr * 0.5` (unchanged), backbone other params get `lr * 1.0` (unchanged), but specialized head params get `lr * 2.0` or `lr * 3.0`.

**Why it helps p_tan:** The aft-foil SRF head and surface refinement head were added to an existing baseline with a globally tuned LR of 2e-4. These specialized heads are additive correction modules initialized to near-zero (zero-init output layers). With the default LR, they take many epochs to "wake up" and contribute meaningful corrections. A higher LR specifically for these heads accelerates their convergence without destabilizing the backbone's learning — a standard transfer-learning / fine-tuning trick. The GSB spatial bias MLP uses `spatial_bias` parameters currently in the `attn_params` group (LR = 1e-4) — bumping these to `2e-4` or `4e-4` may extract more from the GSB mechanism that just delivered -3.0% p_tan.

**Code location:** In `train.py`, optimizer parameter group setup (~lines 1427-1460):

```python
# Current structure:
attn_params = [p for n, p in model.named_parameters() if any(k in n for k in ['Wqkv', 'temperature', 'slice_weight', 'attn_scale', 'spatial_bias'])]
other_params = [p for n, p in model.named_parameters() if not any(k in n for k in [...])]

# CHANGE: extract spatial_bias from attn_params and put in a boosted group
backbone_attn = [p for n, p in model.named_parameters() if any(k in n for k in ['Wqkv', 'temperature', 'slice_weight', 'attn_scale'])]
backbone_gsb  = [p for n, p in model.named_parameters() if 'spatial_bias' in n]
backbone_other = [p for n, p in model.named_parameters() if not any(k in n for k in ['Wqkv', 'temperature', 'slice_weight', 'attn_scale', 'spatial_bias'])]

_head_lr_mult = cfg.head_lr_mult  # new flag, default 2.0
base_opt = Lion([
    {'params': backbone_attn,  'lr': _base_lr * 0.5},
    {'params': backbone_other, 'lr': _base_lr},
    {'params': backbone_gsb,   'lr': _base_lr * _head_lr_mult},  # GSB gets boosted LR
], weight_decay=cfg.weight_decay)
# Then when adding refinement/aft-srf head params:
base_opt.add_param_group({'params': _refine_params, 'lr': _base_lr * _head_lr_mult})
base_opt.add_param_group({'params': _aft_params,    'lr': _base_lr * _head_lr_mult})
```

New flags: `--head_lr_mult 2.0` (sweep: {2.0, 3.0}).

**Concrete hyperparams:**
- `head_lr_mult=2.0` for initial run (so aft-srf/refine/GSB get lr=4e-4 vs backbone 2e-4)
- `head_lr_mult=3.0` as a second seed variant
- Keep `cosine_T_max=160`, `ema_decay=0.999` unchanged

**Risk:** LOW. No new architecture — pure optimizer configuration. The worst case is that high LR for heads destabilizes them (visible as p_in regression), in which case lower `head_lr_mult` can be tried. The mechanism is well-understood.

**Estimated LoC:** ~15 (parameter group restructuring + config flag).

**Expected p_tan gain:** -1 to -3%. Conservative estimate given that specialized heads are already converged somewhat from the long training run. Larger gain possible if heads are currently learning-rate-starved.

**Confidence:** Moderate. Differential LRs are standard practice in transfer learning and fine-tuning. The specific application to additive correction heads in surrogate models is novel but the principle is well-established.

---

## Hypothesis 4: tandem-surface-mixup

**Slug:** `tandem-surface-mixup`

**What it is:** A novel, geometry-aware augmentation: during training, for tandem-foil samples only, randomly swap the surface-node feature-target pairs between the fore-foil and aft-foil of two different tandem samples in the same batch. Concretely: for sample A (gap=0.3, stagger=0.1) and sample B (gap=0.4, stagger=0.2), swap the raw x/y/target data for aft-foil surface nodes of sample A with those from sample B. The volume nodes and global conditions (gap, stagger, Re, AoA) are kept from the original sample. This creates a "chimeric" sample: aft-foil geometry from B embedded in the flow field of A.

**Why it helps p_tan:** The primary p_tan bottleneck is NACA6416 aft-foil prediction — the model sees NACA6416 only in tandem OOD configurations. This augmentation forces the model to predict aft-foil surface behavior for geometrically mismatched configurations (aft-foil from one case, flow field from another). This trains the model to be invariant to the specific aft-foil geometry while attending to the gap/stagger and flow conditions — exactly the generalization needed for NACA6416 OOD transfer. It is conceptually related to domain randomization: by randomizing which aft-foil shape appears in which flow field, the model cannot memorize geometry-specific pressure patterns and must learn the underlying physics relationship.

This differs from DSDF Channel Mixup (#2132, dead end) which mixed DSDF channel values within a sample — a non-physical operation. Tandem Surface Mixup swaps entire surface node sets between samples, which corresponds to a physically plausible chimeric geometry.

**Code location:** In `train.py`, augmentation block (~line 1629), after gap/stagger perturbation:

```python
if cfg.tandem_surface_mixup and model.training and epoch >= cfg.aug_start_epoch:
    _tandem_idx = is_tandem_batch.nonzero(as_tuple=True)[0]  # indices of tandem samples in batch
    if len(_tandem_idx) >= 2:
        # Randomly pair tandem samples
        _perm = _tandem_idx[torch.randperm(len(_tandem_idx), device=device)]
        # Only swap if unpaired (even number); skip last if odd
        _n_swaps = len(_tandem_idx) // 2
        for _si in range(_n_swaps):
            _a, _b = _tandem_idx[_si * 2], _perm[_si * 2]
            if _a == _b:
                continue
            # Identify aft-foil surface nodes for sample _a and _b
            # Use same proxy as _aft_foil_mask: saf_norm > 0.005
            _saf_a = x[_a, :, 2:4].norm(dim=-1)  # [N] — after standardization
            _saf_b = x[_b, :, 2:4].norm(dim=-1)
            _aft_a = is_surface[_a] & (_saf_a > (0.005 / stats["x_std"][2:4].norm()))
            _aft_b = is_surface[_b] & (_saf_b > (0.005 / stats["x_std"][2:4].norm()))
            # Swap aft surface nodes (x-coords, targets)
            if _aft_a.sum() == _aft_b.sum() and _aft_a.sum() > 0:
                _xa_aft = x[_a, _aft_a].clone()
                _ya_aft = y_norm[_a, _aft_a].clone()
                x[_a, _aft_a] = x[_b, _aft_b]
                y_norm[_a, _aft_a] = y_norm[_b, _aft_b]
                x[_b, _aft_b] = _xa_aft
                y_norm[_b, _aft_b] = _ya_aft
```

Note: This must be applied AFTER standardization (since saf norm threshold is in raw space — use adjusted threshold) and AFTER y_norm computation. The swap condition `_aft_a.sum() == _aft_b.sum()` guards against mesh topology mismatch (different node counts for different geometries). For initial implementation, use `cfg.tandem_surface_mixup_prob` to control activation probability (0.3 is a reasonable start).

New flags: `--tandem_surface_mixup`, `--tandem_surface_mixup_prob 0.3`.

**Concrete hyperparams:**
- `tandem_surface_mixup_prob=0.3` (activation probability per batch)
- `aug_start_epoch=0` (apply from start)
- Run 2 seeds

**Risk:** MEDIUM. The swap validity condition (equal node counts) may filter out most swaps for irregular meshes; if so, effectiveness is limited. The chimeric label may also confuse the model if the x-coordinate ranges of swapped aft-foil nodes are out-of-distribution relative to the host sample's volume field. A preliminary check: log the fraction of eligible swaps per batch to confirm non-trivially many swaps occur.

**Estimated LoC:** ~40 (swap logic ~30 + config ~10).

**Expected p_tan gain:** -2 to -5% if swap fraction is high enough. The mechanism is novel for CFD surrogates; analogy to domain randomization in robotics (which shows consistent 5-15% gains in zero-shot transfer) suggests the upper end is achievable.

**Confidence:** Moderate. Novel for this setting. The dead-end DSDF Channel Mixup (#2132) failed for a different reason (non-physical channel mixing within a sample); this is between-sample geometric augmentation — a physically distinct operation. Implement carefully with eligibility logging.

---

## Hypothesis 5: pressure-separate-mlp-tandem

**Slug:** `pressure-separate-mlp-tandem`

**What it is:** Add a tandem-specific pressure decoding pathway — a separate small MLP that takes the hidden state and the gap/stagger condition as inputs, and outputs a tandem-specific pressure correction added on top of the main pressure prediction. This is distinct from `pressure_sep_mlp` (which separates velocity and pressure in the main trunk) and from the aft-foil SRF head (which corrects the aft-foil surface only). This is a tandem-conditioned pressure head that applies to ALL nodes of tandem samples, weighted by a learned gate that learns to specialize for the inter-foil channel region.

**Why it helps p_tan:** The tandem p_tan metric is 2.19x worse than p_in (28.60 vs 13.05). Most of this gap comes from pressure prediction in the inter-foil channel — a physics regime that doesn't exist in single-foil flow. The current model uses the same pressure decoder for single-foil and tandem, with only the gap/stagger spatial bias (GSB) as tandem-specific routing. A tandem-dedicated pressure MLP forces a separate learned representation for inter-foil pressure patterns, conditioned on the specific gap/stagger geometry. The gate (sigmoid scalar per node, initialized to 0 for stability) ensures this MLP starts inert and only activates as it proves useful.

This differs from `pressure_first` + `pressure_deep` (already in baseline — sequential decoding) by operating in parallel as a residual, conditioned on tandem geometry context not available to the main trunk at pressure decoding time.

**Code location:** In `train.py`, after the model forward pass, analogous to `aft_srf_head` pattern (~line 1823):

```python
# New class TandemPressureHead (add near AftFoilRefinementHead class):
class TandemPressureHead(nn.Module):
    def __init__(self, n_hidden, hidden=96, gs_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_hidden + gs_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),  # pressure correction only
        )
        self.gate = nn.Sequential(nn.Linear(n_hidden + gs_dim, 1), nn.Sigmoid())
        # Zero-init output to start inert
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.constant_(self.gate[0].bias, -2.0)  # gate starts near-zero
    
    def forward(self, h, gs_cond):
        # h: [M, n_hidden], gs_cond: [M, 2] (gap, stagger per node)
        inp = torch.cat([h, gs_cond], dim=-1)
        corr = self.mlp(inp)   # [M, 1]
        gate = self.gate(inp)  # [M, 1]
        return corr * gate     # [M, 1]
```

In training loop, after main pred, for tandem samples:
```python
if tandem_p_head is not None and model.training:
    _is_tandem_b = is_tandem_batch  # [B]
    if _is_tandem_b.any():
        _tan_node_mask = _is_tandem_b.unsqueeze(1).expand(-1, x.shape[1])  # [B, N]
        _tan_node_mask = _tan_node_mask & mask  # restrict to valid nodes
        _tan_idx = _tan_node_mask.nonzero(as_tuple=False)  # [M, 2]
        if _tan_idx.numel() > 0:
            _h_tan = hidden[_tan_idx[:, 0], _tan_idx[:, 1]]  # [M, n_hidden]
            _gs_tan = x[_tan_idx[:, 0], 0:1, 22:24].squeeze(1)  # [M, 2] — per-node but broadcast from sample-level
            # Correct: gs is sample-level, broadcast to all nodes of that sample
            _gs_tan = x[_tan_idx[:, 0], 0, 22:24]  # [M, 2]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _p_corr = tandem_p_head(_h_tan, _gs_tan).float()  # [M, 1]
            pred = pred.clone()
            pred[_tan_idx[:, 0], _tan_idx[:, 1], 2:3] += _p_corr
```

New flags: `--tandem_pressure_head` (bool), `--tandem_pressure_head_hidden 96`.

**Concrete hyperparams:**
- `tandem_pressure_head_hidden=96` (smaller than aft-srf 192, since it applies to all nodes not just surface)
- Zero-init output + gate bias=-2.0 for inert start
- Same `base_lr` as other specialized heads
- Run 2 seeds (s42, s73)

**Risk:** MEDIUM. The gated structure prevents catastrophic failure. Main risk: the correction is applied to ALL tandem nodes (including volume), which may introduce correlated error patterns that confuse the pressure-first/pressure-deep pipeline. Mitigation: apply correction only to surface nodes of tandem samples if full-mesh application regresses p_in.

**Estimated LoC:** ~55 (class ~30 + integration ~25).

**Expected p_tan gain:** -2 to -4%. The gap/stagger conditional gives this head access to geometry context that the standard trunk (which only sees GSB spatial bias through the attention mechanism) processes more implicitly.

**Confidence:** Moderate. Tandem-specific pressure pathways are a natural extension of the confirmed wins (aft-foil SRF, GSB). The gated design ensures low downside risk. The interaction with `pressure_first`+`pressure_deep` (the detached pressure-first pass in the baseline) is the main unknown.

---

## Summary Table

| Rank | Slug | Risk | LoC | Expected p_tan gain | Priority |
|------|------|------|-----|---------------------|----------|
| 1 | film-conditioned-fore-foil-srf | MEDIUM | ~65 | -2 to -5% | HIGH |
| 2 | tandem-cross-dsdf-features | LOW-MED | ~25 | -2 to -4% | HIGH |
| 3 | differential-lr-specialized-heads | LOW | ~15 | -1 to -3% | MEDIUM-HIGH |
| 4 | tandem-surface-mixup | MEDIUM | ~40 | -2 to -5% | MEDIUM |
| 5 | pressure-separate-mlp-tandem | MEDIUM | ~55 | -2 to -4% | MEDIUM |

**Recommended priority order for assignment:**
1. `tandem-cross-dsdf-features` — lowest risk, shortest LoC, direct extension of confirmed GSB win
2. `differential-lr-specialized-heads` — trivial to implement, may unlock latent capacity in existing heads
3. `film-conditioned-fore-foil-srf` — highest expected gain but most implementation complexity
4. `pressure-separate-mlp-tandem` — strong mechanism, medium complexity
5. `tandem-surface-mixup` — novel augmentation, requires eligibility-fraction logging to validate

**Key research patterns informing these choices:**
- Confirmed wins share a pattern: geometry-conditioned specialization (GSB, aft-foil SRF with FiLM, DSDF2 aug)
- Failed experiments share a pattern: unconditioned corrections (fore-foil SRF #2117/#2124), non-physics augmentations (#2132), global weight changes (#2121/#2122)
- The NACA6416 OOD gap is a geometry identity problem — the model doesn't know which foil shape it's seeing. All 5 hypotheses above address this from different angles.
