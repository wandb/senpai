---
date: 2026-04-05T10:39
advisor: noam
target: p_tan < 28.60 (current single-model baseline, PR #2130)
context: Phase 6 — all 8 students active on hyperparameter sweeps. Two idle GPUs (tanjiro, alphonse) available for immediate assignment.
---

# Research Ideas — 2026-04-05 10:39

## Context

**Current baseline (PR #2130, 2-seed):**
- p_in: 13.05, p_oodc: 7.70, **p_tan: 28.60**, p_re: 6.55

**What has worked (Phase 6):**
1. `--aft_foil_srf` — additive correction head for aft-foil surface nodes (boundary ID=7)
2. `--aug_gap_stagger_sigma 0.02` — scalar domain randomization for gap/stagger
3. `--aug_dsdf2_sigma 0.05` — log-normal foil-2 DSDF channel scaling (shape uncertainty)
4. `--pcgrad_3way --pcgrad_extreme_pct 0.15` — 2-way gradient surgery (in-dist vs OOD)
5. `--gap_stagger_spatial_bias` — extends spatial_bias MLP from 4D→6D with (gap, stagger); -3.0% p_tan

**Confirmed dead ends (relevant to new ideas):**
- DSDF Spatial Dropout (p=0.05, 0.1, 0.2 all worse: p_in +6.4% monotonic) — NOTE: spatial dropout drops individual node positions; the new Channel Dropout idea below drops entire semantic feature channels — fundamentally different mechanism
- Fore-Foil SRF unconditional (split or stacked): p_tan +1-11%
- Input Feature Noise: catastrophic at σ=0.01 — NOTE: targets any input feature; channel dropout is purely on DSDF geometry channels, not physics features
- DSDF Channel Mixup: no geometric diversity for NACA0012→NACA6416 generalization
- FiLM on gap/stagger scalars: catastrophic p_oodc +41.6%
- Actual 3-way PCGrad (with `--disable_pcgrad --pcgrad_3way`): all pct values worse

**In-flight experiments (do NOT duplicate):**
- askeladd #2150: DSDF2 sigma sweep (σ=0.03, 0.08)
- tanjiro #2155: slice_num sweep (64, 128)
- fern #2151: EMA start epoch sweep (100, 120)
- alphonse #2131: Tandem-Slice Carve-Out K=4 (rebasing)
- nezuko #2152: Augmentation Annealing (linear σ decay)
- thorfinn #2154: Cosine T_max sweep (140, 180)
- frieren #2153: Gap/Stagger sigma increase (σ=0.03)
- edward #2149: LR sweep (1e-4, 3e-4)

---

## Hypothesis 1: DSDF Channel Dropout (for tanjiro)

**Slug:** `dsdf-channel-dropout`

**Hypothesis:**
The model's failure to generalize to unseen NACA6416 geometry (p_tan=28.60, far above p_in=13.05) is a representation gap: the model has overfit to the specific DSDF signatures of NACA0012. If we randomly zero out individual foil-1 DSDF channels (indices 0:5) during training with probability p=0.2, we force the model to make good predictions even when some geometric shape information is missing — producing a more shape-agnostic encoder that can interpolate to novel geometries. This is fundamentally different from DSDF Spatial Dropout (#2143, which dropped node positions), and from Input Feature Noise (#2144, which added Gaussian noise to all inputs). Channel dropout zeros entire semantic feature channels (e.g., "entire DSDF channel 2 is zeroed for this forward pass"), mimicking the uncertainty about which geometric features are most informative for an unseen foil shape. The mechanism is analogous to feature-level dropout used in tabular learning and multi-modal fusion, where dropping full modalities forces robust cross-modal representations.

**Code changes:**
In `train.py`, in the training loop (before the forward pass), add a per-sample channel dropout:
```python
# DSDF channel dropout (foil-1 channels 0:5, p=0.2, training only)
if cfg.dsdf_channel_dropout > 0 and model.training:
    mask = (torch.rand(x.shape[0], 5, device=x.device) > cfg.dsdf_channel_dropout)
    x[:, :, 0:5] = x[:, :, 0:5] * mask.unsqueeze(1)
```
Add flag: `--dsdf_channel_dropout 0.2` (default 0.0).
Also try: `--dsdf_channel_dropout 0.3` as second run in same group.

**Expected impact:** -2% to -5% p_tan, minimal effect on p_in/p_oodc (those don't test NACA6416 transfer). LOW risk: worst case the foil-1 features are too important and p_tan regresses slightly — easily reverted.

**Risk:** LOW. 12 lines of code. The foil-1 channels are redundant (mesh nodes already encode position) — the DSDF primarily adds semantic shape info. Dropping channels randomly teaches the model which combinations matter rather than memorizing the exact NACA0012 DSDF signature.

**Suggested W&B runs:**
- `--wandb_group phase6/dsdf-channel-dropout`
- Run 1: `--dsdf_channel_dropout 0.2 --seed 42`
- Run 2: `--dsdf_channel_dropout 0.2 --seed 73`
- Run 3 (optional): `--dsdf_channel_dropout 0.3 --seed 42`
- Compare p_tan vs baseline p_tan=28.60

**Full command (seed 42):**
```bash
cd cfd_tandemfoil && python train.py --agent tanjiro \
  --wandb_name "tanjiro/dsdf-channel-dropout-p02-s42" \
  --wandb_group phase6/dsdf-channel-dropout \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --pcgrad_3way --pcgrad_extreme_pct 0.15 --gap_stagger_spatial_bias \
  --dsdf_channel_dropout 0.2 --seed 42
```

---

## Hypothesis 2: Foil Shape Similarity Bias (for alphonse)

**Slug:** `foil-shape-similarity-bias`

**Hypothesis:**
GSB (Gap/Stagger Spatial Bias, PR #2130) extended the spatial_bias MLP from 4D→6D by appending (gap, stagger) scalars, making slice routing tandem-geometry-aware and producing -3.0% p_tan. A natural extension is to also provide the routing MLP with a measure of the **geometric relationship between the two foils** — specifically, the cosine similarity between the foil-1 and foil-2 DSDF mean vectors. When the two foils are geometrically similar (e.g., NACA0012 tandem with itself), this similarity is high; when they differ (NACA6416 with NACA0012), it is low. This scalar tells the routing MLP explicitly whether it is in an interpolation scenario (similar foils) or an extrapolation scenario (dissimilar foils), allowing it to allocate slice capacity differently. This extends GSB's raw_xy from 6D→7D with one semantically rich scalar — zero-init on the new column ensures identical routing at epoch 0 (zero risk of regression). The mechanism is complementary to GSB since GSB encodes where the foils are, while shape similarity encodes how different they look.

**Code changes:**
In `train.py`, in the GSB computation section, add:
```python
# Shape similarity scalar: cosine sim between foil-1 and foil-2 DSDF mean vectors
dsdf1_mean = x[:, :, 0:5].mean(dim=1)  # (B, 5)
dsdf2_mean = x[:, :, 6:10].mean(dim=1)  # (B, 4) — use first 4 dims for compatibility
dsdf1_norm = F.normalize(dsdf1_mean[:, :4], dim=-1)
dsdf2_norm = F.normalize(dsdf2_mean, dim=-1)
shape_sim = (dsdf1_norm * dsdf2_norm).sum(dim=-1, keepdim=True)  # (B, 1)
# Append to existing gap/stagger scalars in raw_xy
raw_xy = torch.cat([raw_xy, shape_sim], dim=-1)  # 6D→7D
```
Also zero-init the new input column's corresponding weight in the spatial_bias MLP.
Add flag: `--foil_shape_sim_bias` (boolean, default False).

**Expected impact:** -1% to -3% p_tan. Targeted at NACA6416 transfer generalization (p_tan only). LOW risk.

**Risk:** LOW. ~30 lines of code. Purely additive on top of existing GSB. Zero-init ensures no regression at epoch 0. Worst case: scalar is noisy / collinear with gap/stagger and has no effect.

**Suggested W&B runs:**
- `--wandb_group phase6/foil-shape-sim-bias`
- Run 1: `--foil_shape_sim_bias --seed 42`
- Run 2: `--foil_shape_sim_bias --seed 73`
- Primary metric: p_tan vs baseline 28.60

**Full command (seed 42):**
```bash
cd cfd_tandemfoil && python train.py --agent alphonse \
  --wandb_name "alphonse/foil-shape-sim-bias-s42" \
  --wandb_group phase6/foil-shape-sim-bias \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --pcgrad_3way --pcgrad_extreme_pct 0.15 --gap_stagger_spatial_bias \
  --foil_shape_sim_bias --seed 42
```

---

## Hypothesis 3: Asymmetric PCGrad (for next idle student)

**Slug:** `asymmetric-pcgrad`

**Hypothesis:**
The current 2-way PCGrad (PR #2119) performs symmetric gradient surgery: when the in-dist and OOD gradients conflict, BOTH gradients are projected to remove the conflicting component. This means the in-dist gradient is also modified when OOD gradients conflict with it — sacrificing in-dist accuracy to reduce inter-task conflict. Asymmetric PCGrad inverts this: only the OOD gradient is projected onto the plane orthogonal to the in-dist gradient. The in-dist gradient is never modified. This preserves the in-dist (p_in, p_oodc, p_re) signal at full strength while still preventing OOD from actively harming in-dist learning. The physical intuition: single-foil and known-tandem predictions are the "anchor" tasks; NACA6416 transfer is the "adapting" task that should learn from the anchors without destabilizing them.

**Code changes:**
In `train.py`, in the PCGrad section, change from symmetric to asymmetric projection:
```python
# Symmetric (current): both grads projected
# Asymmetric: only project g_ood onto g_indist normal plane
if g_ood.dot(g_indist) < 0:
    g_ood = g_ood - (g_ood.dot(g_indist) / g_indist.dot(g_indist)) * g_indist
    # Do NOT touch g_indist
```
Add flag: `--asymmetric_pcgrad` (boolean, default False). When set, only OOD gradient is projected, never in-dist.

**Expected impact:** -1% to -3% across all OOD metrics (p_tan, p_oodc). VERY LOW risk.

**Risk:** VERY LOW. 3-line change within existing PCGrad logic. No new parameters. Worst case: null result (symmetric is already optimal). Could also slightly improve p_in since in-dist gradient is now fully preserved.

**Suggested W&B runs:**
- `--wandb_group phase6/asymmetric-pcgrad`
- Run 1: `--asymmetric_pcgrad --seed 42`
- Run 2: `--asymmetric_pcgrad --seed 73`

**Full command (seed 42):**
```bash
cd cfd_tandemfoil && python train.py --agent <student> \
  --wandb_name "<student>/asymmetric-pcgrad-s42" \
  --wandb_group phase6/asymmetric-pcgrad \
  --asinh_pressure --asinh_scale 0.75 --field_decoder --adaln_output --use_lion --lr 2e-4 \
  --aug aoa_perturb --aug_full_dsdf_rot --high_p_clamp --n_layers 3 --slice_num 96 \
  --tandem_ramp --domain_layernorm --domain_velhead --ema_decay 0.999 --weight_decay 5e-5 \
  --cosine_T_max 160 --pressure_first --pressure_deep \
  --residual_prediction --surface_refine --surface_refine_hidden 192 --surface_refine_layers 3 \
  --aft_foil_srf --aug_gap_stagger_sigma 0.02 --aug_dsdf2_sigma 0.05 \
  --pcgrad_3way --pcgrad_extreme_pct 0.15 --gap_stagger_spatial_bias \
  --asymmetric_pcgrad --seed 42
```

---

## Hypothesis 4: Tandem Cross-DSDF Features (for next idle student)

**Slug:** `tandem-cross-dsdf-features`

**Hypothesis:**
The model currently receives foil-1 DSDF (channels 0:5) and foil-2 DSDF (channels 6:10) as independent feature vectors, but the network has no explicit signal about the **relative geometry** between the two foils beyond the gap/stagger scalars in GSB. For tandem transfer, the key unknown is: how does the aft-foil pressure distribution change when the fore-foil is NACA6416 instead of NACA0012? Adding explicit cross-foil geometric features can bridge this gap. Specifically: (1) element-wise difference DSDF1 - DSDF2 (4 channels — the geometric delta), (2) element-wise product DSDF1 * DSDF2 (4 channels — shared shape features), (3) L2 norm of the difference (1 scalar — geometric dissimilarity). These 9 additional features are appended to each node's input vector before the first Transolver layer. All are derived from existing DSDF channels — no new data. The geometric delta and product encode the specific asymmetry of the tandem configuration in a spatially explicit way.

**Code changes:**
In `train.py`, in the input preparation section:
```python
# Cross-DSDF features for tandem samples
if cfg.tandem_cross_dsdf and is_tandem:
    dsdf1 = x[:, :, 0:4]  # (B, N, 4) — first 4 channels align with foil-2
    dsdf2 = x[:, :, 6:10]  # (B, N, 4)
    dsdf_diff = dsdf1 - dsdf2  # geometric delta
    dsdf_prod = dsdf1 * dsdf2  # shared features
    dsdf_dist = dsdf_diff.norm(dim=-1, keepdim=True)  # scalar dissimilarity
    x = torch.cat([x, dsdf_diff, dsdf_prod, dsdf_dist], dim=-1)
```
Update the input projection layer to handle the expanded feature dimension. For non-tandem samples, pad with zeros.
Add flag: `--tandem_cross_dsdf` (boolean, default False).

**Expected impact:** -2% to -4% p_tan (targeted), minimal effect on p_in/p_oodc/p_re. MEDIUM risk (requires input projection dimension change, but zero-padding for non-tandem is clean).

**Risk:** LOW-MEDIUM. ~20 lines of code. Main risk: input dimension change requires careful handling of the input projection layer. However, zero-padding for non-tandem samples and zero-init on new projection columns ensures safe initialization.

**Suggested W&B runs:**
- `--wandb_group phase6/tandem-cross-dsdf`
- Run 1: `--tandem_cross_dsdf --seed 42`
- Run 2: `--tandem_cross_dsdf --seed 73`

---

## Hypothesis 5: Fork-then-Merge Model Soup (for next idle student, multi-GPU)

**Slug:** `fork-merge-soup`

**Hypothesis:**
Cross-seed model soup (#2142) failed catastrophically because independently trained models from different random seeds exist in separate loss basins separated by high barriers — naive weight averaging produces a model in "no-man's land". The fix is to exploit the **linear mode connectivity** property that holds when models share initialization: if two models start from the same checkpoint, their loss barrier is near-zero, and their weights can be averaged directly. The protocol: (1) train a single model for 100 epochs to convergence to a good basin, (2) branch into 3 copies with different random seeds for the remaining 60 epochs (each explores a slightly different region of the local basin), (3) average the 3 final checkpoints. This should give a cheap ensemble-like effect (variance reduction) without independent training from scratch. Key constraint: the branching point at epoch 100 must be past the main EMA warmup and cosine annealing has fully kicked in.

**Code changes:**
This is a training protocol change, not a model change. The student needs to:
1. Train the baseline model with `--save_checkpoint_at 100` (new flag) to save a checkpoint at epoch 100
2. Load that checkpoint 3 times with different seeds (`--load_checkpoint <path> --seed {42,73,99}`) for epochs 100-160
3. Average the 3 final checkpoints with `eval_soup.py` (or inline code)
This requires minimal code: one `save_checkpoint_at` flag and one `load_checkpoint` flag.

**Expected impact:** -1% to -3% across all metrics (variance reduction effect). The benefit is proportional to seed diversity in the final 60 epochs vs the shared first 100. MEDIUM risk: if the branched models don't diverge meaningfully, the soup has no benefit over a single run.

**Risk:** MEDIUM. Protocol complexity: 4 training runs + averaging. But each individual component (checkpoint save/load) is simple. The main risk is that 60-epoch divergence is insufficient for meaningful diversity. Start with 2 branches before 3.

**Suggested W&B runs:**
- `--wandb_group phase6/fork-merge-soup`
- Phase 1: Train base model with `--save_checkpoint_at 100`
- Phase 2: 2 branches from epoch 100: seed 42, seed 73
- Phase 3: Average and evaluate

---

## Assignment Recommendations

### For tanjiro (immediate): Hypothesis 1 — DSDF Channel Dropout
- **Why tanjiro:** Low risk, 12 LoC, directly attacks the NACA6416 representation gap. No prior history for this student relevant to this idea (tanjiro just completed gap/stagger aug removal #2148 and is currently doing slice sweep #2155 — but the task says tanjiro is idle). Fast to implement, results interpretable.
- **Expected results in:** 2 seeds (p=0.2) → 2-seed avg vs baseline 28.60

### For alphonse (immediate): Hypothesis 2 — Foil Shape Similarity Bias
- **Why alphonse:** Natural extension of GSB (alphonse is already familiar with the spatial_bias MLP from #2131 tandem-slice work). Low risk, 30 LoC, zero-init ensures safety. Directly targets p_tan via improved routing awareness of geometric dissimilarity.
- **Expected results in:** 2 seeds → 2-seed avg vs baseline 28.60

### Rationale for prioritizing Hypotheses 1 and 2 over 3, 4, 5:
- **H3 (Asymmetric PCGrad)** is very small (8 LoC) and low-risk but expected smaller impact. Best held for a student coming off a sweep experiment who can implement it quickly.
- **H4 (Cross-DSDF Features)** is medium risk due to input dimension change — better suited for a student with more implementation bandwidth.
- **H5 (Fork-then-Merge Soup)** requires multi-GPU coordination and protocol complexity — not ideal for immediate assignment without full context.
