# Round 2 Research Ideas — 2026-04-20

Generated after reviewing all Round 1 assignments on the radford branch (16 PRs, no results yet),
100 historical PRs on the noam branch, and full program documentation.

**Key assumptions driving Round 2 design:**
- Round 1 is still running; none have reported results yet. Round 2 ideas are designed to be
  independent of Round 1 outcomes, or to clearly state what Round 1 result they depend on.
- Defaults: `optimizer=lion`, `lr=3e-4`, `use_lookahead=True`, `use_ema=True`, `ema_decay=0.999`,
  `ema_start_step=50`, `cosine_t_max=150`, `batch_size=2`, `weight_decay=1e-4`,
  `model_layers=3`, `model_hidden_dim=192`, `model_heads=3`, `model_slices=96`,
  `model_mlp_ratio=4`, `model_dropout=0.0`, `surface_refine=True`,
  `surface_refine_hidden=128`, `surface_refine_layers=2`, `anp_srf=False`,
  `asinh_pressure=False`, `residual_prediction=False`, `re_stratified_sampling=False`.
- noam branch lesson: batch_size=8 **hurt OOD** performance — be cautious with batch scaling.
- noam branch lesson: wider hidden dim (256) was slower and worse — 192 seems right.
- noam branch lesson: mlp_ratio=4 (wide FFN) was worse — but that was an older stack.
- noam branch lesson: fewer slices (32 vs 64) helped via faster epochs — worth testing on radford.

---

## HIGH PRIORITY

### 1. cosine-schedule-sweep-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** The default `cosine_t_max=150` may be poorly matched to the actual number of
training epochs run under the SENPAI timeout. If T_max greatly exceeds actual epochs, the LR
never bottoms out and the model never exploits the final convergence phase. Trying T_max=50 and
T_max=100 forces a steeper final LR decay within the available compute budget, potentially
improving late-training convergence and surface pressure MAE.
**Key changes:**
- Run A: `--cosine_t_max 50`
- Run B: `--cosine_t_max 100`
- Run C (control): `--cosine_t_max 150` (default, to establish exact epoch count for comparison)
- Use `--wandb_group cosine-schedule-sweep-tandem` for all three runs.
**Expected outcome:** One of T_max=50 or T_max=100 beats T_max=150 by improving final-phase
convergence within the actual compute envelope. Surface pressure MAE improvement of 3–8%.

---

### 2. cosine-schedule-sweep-airfrans
**Dataset:** AirfRANS
**Hypothesis:** Same motivation as above, applied to AirfRANS. The `airfrans_task=full` has
different epoch dynamics than TandemFoil; the cosine schedule may need separate tuning per
dataset. A T_max misaligned with actual epoch count could leave 5–10% MAE improvement on the
table.
**Key changes:**
- Run A: `--cosine_t_max 50`
- Run B: `--cosine_t_max 100`
- Run C: `--cosine_t_max 150` (default control)
- `--dataset airfrans --airfrans_task full`
- `--wandb_group cosine-schedule-sweep-airfrans`
**Expected outcome:** One of T_max=50 or T_max=100 produces lower `val_primary/surface_mse`.

---

### 3. ema-decay-sweep-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** The default `ema_decay=0.999` is relatively slow, requiring ~1000 gradient steps
before the EMA model diverges meaningfully from the live model. With Lion's aggressive sign-update
steps and a limited epoch budget, this might mean the EMA shadow is too close to the noisy
last-batch model. Trying `ema_decay=0.9995` (slower, smoother) and `ema_decay=0.99` (faster,
more responsive) explores whether a tighter or looser EMA averaging improves the held-out metric.
**Key changes:**
- Run A: `--ema_decay 0.9995 --ema_start_step 50`
- Run B: `--ema_decay 0.99 --ema_start_step 50`
- Run C: `--ema_decay 0.999 --ema_start_step 25` (earlier EMA onset, keep decay)
- `--wandb_group ema-decay-sweep-tandem`
**Expected outcome:** `ema_decay=0.9995` provides a more stable shadow model that better
captures the long-run optimum; expected 2–5% reduction in surface pressure MAE.

---

### 4. re-stratified-sampling-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** The TandemFoilSet includes an OOD Reynolds-number split (`val_re_rand`) that
directly tests generalization to unseen Re. The default uniform sampler over-represents
training-distribution Re values, potentially starving the model of rare-Re examples during
training. Enabling `re_stratified_sampling=True` reweights training batches by Reynolds number
bucket, giving rare-Re cases equal representation and improving OOD performance on `val_re_rand`
while maintaining or improving overall `val_eq4` surface pressure MAE.
**Key changes:**
- `--re_stratified_sampling True`
- All other settings default (including `lr=3e-4`, `optimizer=lion`)
**Expected outcome:** Improvement on `val_re_rand/surface_pressure_mae` of 5–15% and measurable
improvement in the equal-weighted average `val_eq4/surface_pressure_mae`.

---

### 5. anp-surface-decoder-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** The ANP cross-foil surface decoder (`anp_srf=True`) was designed to model
inter-foil pressure interactions in tandem configurations, which are precisely the cases where
the standard decoder struggles (the foil wakes interact). Enabling it adds an attention pathway
that can reason about fore-foil influence on aft-foil pressure. This may be especially valuable
for the tandem-transfer validation splits.
**Key changes:**
- `--anp_srf True`
- All other settings default
**Expected outcome:** Reduction in surface pressure MAE on tandem-specific splits; measurable
improvement on `val_eq4/surface_pressure_mae` overall. This feature was explicitly pinned at
`origin/frieren/anp-surface-decoder@7999a2e` and is considered high-priority by the program.

---

### 6. disable-lookahead-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** Lookahead is ON by default (`use_lookahead=True`), which wraps the Lion optimizer
with a two-level update scheme (fast weights + slow weights synchronized every k=5 steps). While
Lookahead generally helps with generalization, it also introduces a lag in following gradient
signals. With Lion's sign-gradient updates already providing implicit regularization, Lookahead
may be redundant or even harmful. Disabling it provides a cleaner signal for whether Lookahead
is pulling its weight on this stack.
**Key changes:**
- `--use_lookahead False`
- All other settings default
**Expected outcome:** If Lookahead is redundant, disabling it speeds up training (more effective
steps per epoch) and possibly improves OOD generalization. If it helps, keeping it confirmed.
Either outcome is informative for all future experiments.

---

### 7. weight-decay-sweep-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** Weight decay `1e-4` is a light regularizer. Given the model has ~192 hidden dim
with 3 layers and likely fewer than 1M parameters trained on a fixed finite dataset, stronger
L2 regularization (`5e-4` or `1e-3`) might reduce overfitting to in-distribution patterns.
Conversely, `1e-5` almost zero regularization lets the model memorize training more but might
generalize better if the inductive bias of the architecture is already strong.
**Key changes:**
- Run A: `--weight_decay 1e-5`
- Run B: `--weight_decay 5e-4`
- Run C: `--weight_decay 1e-3`
- `--wandb_group weight-decay-sweep-tandem`
**Expected outcome:** A U-shaped curve where one of {1e-5, 5e-4, 1e-3} outperforms the default
`1e-4`. Expect `5e-4` to be the best bet given the dataset size and OOD evaluation.

---

### 8. airfrans-task-scarce
**Dataset:** AirfRANS
**Hypothesis:** The `scarce` task variant restricts training data to a small labeled set (simulating
low-data regimes). This is a challenging evaluation condition that may expose model weaknesses
not visible in the `full` task. Establishing a strong result on `scarce` is strategically
important for the ICML submission because it demonstrates generalization from few examples —
a key differentiator vs larger supervised methods. Round 1 only tests `full`; scarce is untested.
**Key changes:**
- `--dataset airfrans --airfrans_task scarce`
- All other settings default (`lr=3e-4`, `optimizer=lion`, `use_lookahead True`)
**Expected outcome:** Establishes baseline `val_primary/surface_mse` on the scarce task.
Provides the foundation for Round 3 improvements on the low-data regime.

---

## MEDIUM PRIORITY

### 9. airfrans-task-reynolds
**Dataset:** AirfRANS
**Hypothesis:** The `reynolds` task tests generalization to OOD Reynolds numbers — directly
analogous to the `val_re_rand` split in TandemFoilSet. Since TandemFoil Re-stratified sampling
is an active research direction, establishing how the model performs on AirfRANS Reynolds OOD
provides a parallel data point and enables cross-dataset comparison of generalization strategies.
**Key changes:**
- `--dataset airfrans --airfrans_task reynolds`
- All other settings default
**Expected outcome:** Establishes AirfRANS Reynolds-OOD baseline. Likely higher error than
`full` task, guiding whether Re-stratified sampling should be ported to AirfRANS.

---

### 10. airfrans-task-aoa
**Dataset:** AirfRANS
**Hypothesis:** The `aoa` task tests generalization to OOD angles of attack. This directly tests
the geometric generalization capacity of the Transolver's physics-attention mechanism. High error
here suggests the model is memorizing flow pattern templates rather than learning the underlying
aerodynamic mapping. This is important for ICML: a model that generalizes to unseen AoA is
publishable, one that doesn't isn't.
**Key changes:**
- `--dataset airfrans --airfrans_task aoa`
- All other settings default
**Expected outcome:** Establishes AirfRANS AoA-OOD baseline. Informs whether any
geometry-encoding or augmentation tricks are needed for robust AoA generalization.

---

### 11. dropout-regularization-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** Model dropout is 0.0 by default — no stochastic regularization during training.
On a fixed dataset of finite CFD cases, a small dropout rate (0.05–0.1) applied to the
Transolver's internal attention and FFN paths may improve OOD generalization by preventing
co-adaptation of attention heads to specific training case signatures. The `noam` branch had
no equivalent test; this is uncharted on the new stack.
**Key changes:**
- Run A: `--model_dropout 0.05`
- Run B: `--model_dropout 0.1`
- `--wandb_group dropout-sweep-tandem`
**Expected outcome:** Dropout of 0.05 provides mild regularization benefit on OOD splits
without significantly hurting in-distribution performance. Net improvement on `val_eq4`.

---

### 12. surface-refine-hidden-256-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** The surface refinement head has `surface_refine_hidden=128, surface_refine_layers=2`.
Doubling the hidden width to 256 gives the post-processing head more capacity to correct
systematic errors in the main decoder's surface pressure predictions. This is especially
relevant for complex tandem-wake interaction regions where the main decoder may produce
structured errors the wider refine head can learn to correct.
**Key changes:**
- `--surface_refine_hidden 256`
- All other settings default
**Expected outcome:** 2–5% improvement in surface pressure MAE, most notably on tandem-split
validation cases where wake interaction errors are largest.

---

### 13. surface-refine-layers-3-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** Increasing `surface_refine_layers` from 2 to 3 adds an extra nonlinear correction
stage to the surface refinement head, allowing it to model more complex residual error patterns.
Combined with the existing zero-initialization, a 3-layer refine head starts identically to the
2-layer version but has additional representational capacity to capture structured error fields.
**Key changes:**
- `--surface_refine_layers 3`
- All other settings default
**Expected outcome:** Marginal improvement (~2%) on surface pressure MAE on both in-distribution
and OOD splits as the deeper refine head captures nonlinear correction patterns.

---

### 14. fourier-features-airfrans
**Dataset:** AirfRANS
**Hypothesis:** Fourier position encoding (`enable_fourier=True`) augments each point's spatial
coordinates with sinusoidal features at multiple frequencies, providing the model with explicit
multi-scale spatial information. On AirfRANS, where flow gradients vary sharply near the
airfoil surface and smoothly in the far field, Fourier features may help the Transolver
differentiate near-surface and far-field physics without relying solely on coordinate magnitude.
This feature is currently off by default and untested on AirfRANS.
**Key changes:**
- `--enable_fourier True`
- `--dataset airfrans --airfrans_task full`
**Expected outcome:** Reduction in `val_primary/surface_mse` from improved near-surface
spatial resolution encoding. Estimated 3–7% improvement.

---

### 15. fewer-slices-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** On the noam branch, reducing slices from 64 to 32 was a **merged winner** —
it allowed faster epoch cycling within the compute budget, effectively showing more training
iterations. The radford branch defaults to 96 slices. Reducing to 64 or 48 may yield the
same benefit: faster per-epoch time → more epochs → better convergence within the SENPAI
timeout, without significantly hurting the quality of the physics-attention mechanism.
**Key changes:**
- Run A: `--model_slices 64`
- Run B: `--model_slices 48`
- `--wandb_group slices-sweep-tandem`
**Expected outcome:** Faster epoch cycling leads to more epochs within the timeout, resulting
in better final surface pressure MAE. Based on noam precedent, expect 5–10% improvement.

---

### 16. fewer-slices-airfrans
**Dataset:** AirfRANS
**Hypothesis:** Same motivation as above, applied to AirfRANS. With `model_slices=96` as default,
the physics-attention partitioning may be finer than needed for the AirfRANS geometry complexity
(single airfoil vs tandem). Reducing slices allows more epochs per SENPAI training run.
**Key changes:**
- Run A: `--model_slices 64`
- Run B: `--model_slices 48`
- `--dataset airfrans --airfrans_task full`
- `--wandb_group slices-sweep-airfrans`
**Expected outcome:** More effective training iterations within the compute budget → lower
`val_primary/surface_mse`.

---

### 17. residual-prediction-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** The `residual_prediction=True` flag subtracts the freestream baseline from
normalized targets before training, making the model predict the departure from freestream
rather than the absolute field. For TandemFoil, where pressure variations from freestream are
small compared to absolute magnitudes, learning the residual should be an easier function to
fit — the model can focus representational capacity on the physically interesting pressure
fluctuations rather than the large mean offset.
**Key changes:**
- `--residual_prediction True`
- All other settings default
**Expected outcome:** Improved surface pressure MAE, especially on OOD splits where freestream
extrapolation is needed. Estimated 3–8% improvement.

---

### 18. asinh-pressure-scale-sweep-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** The `asinh_pressure=True` flag applies `asinh(p * scale)` to the pressure target
before training, compressing extreme pressure values and making the loss more uniform across
the pressure range. The default `asinh_scale=0.75` was not tuned on the radford stack. Trying
`asinh_scale=0.5` (less compression) and `asinh_scale=1.0` (more compression) explores whether
the default scale is optimal. Round 1 already tests enabling asinh; this tests the scale
sensitivity.
**Key changes:**
- Run A: `--asinh_pressure True --asinh_scale 0.5`
- Run B: `--asinh_pressure True --asinh_scale 1.0`
- `--wandb_group asinh-scale-sweep-tandem`
**Expected outcome:** One of the two scale values outperforms `asinh_scale=0.75` depending on
the distribution of extreme pressure values in the training set.

---

### 19. combined-physics-features-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** Round 1 ablates individual physics features. If TE coord frame and Cp panel
individually improve things, combining them may compound the benefit. The feature set is
additive — enabling both simultaneously costs only input dimensionality. Given noam-branch
evidence that physics features helped the model understand aerodynamic physics, the combination
is likely better than either alone. This experiment doesn't depend on Round 1 results since we
can stack the most theoretically motivated features.
**Key changes:**
- `--enable_te_coord_frame True --enable_cp_panel True --enable_wake_deficit True`
- All other settings default
**Expected outcome:** Combining the three most physically meaningful feature groups (TE frame
provides orientation, Cp panel provides inviscid pressure prior, wake deficit captures
downstream effects) produces a synergistic improvement over any single feature.

---

### 20. adamw-with-tuned-lr-airfrans
**Dataset:** AirfRANS
**Hypothesis:** Round 1 tests AdamW on TandemFoil. If AdamW wins there, the optimal LR likely
differs from Lion's optimal LR (AdamW scales as 1/sqrt(step) via second moment; Lion uses
sign gradient, so LR of 3e-4 for Lion ≈ LR of 1e-3 for AdamW in terms of effective step size).
Testing AdamW with `lr=1e-3` on AirfRANS tests whether the optimizer-dataset interaction
differs from TandemFoil.
**Key changes:**
- `--optimizer adamw --lr 1e-3`
- `--dataset airfrans --airfrans_task full`
**Expected outcome:** AdamW with higher LR may achieve faster early convergence on AirfRANS
where the loss landscape is smoother (single airfoil, no tandem interaction complexity).

---

## LOWER PRIORITY / SPECULATIVE

### 21. ema-disabled-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** EMA is ON by default. While EMA smoothing generally helps, it has a cost:
the checkpoint used for evaluation lags the current model weights by ~1/ema_decay steps.
For small models with fast convergence, the EMA might track a worse point than the final
live weights. Disabling EMA to test this is cheap and provides a clean controlled comparison.
**Key changes:**
- `--use_ema False`
- All other settings default
**Expected outcome:** If EMA is critical, this will confirm it and shut down further
investigation. If EMA hurts, disabling it is a free win. Informative either way.

---

### 22. batch-size-4-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** Batch size 2 (default) processes only 2 samples per gradient step. Doubling to
4 increases gradient quality and VRAM utilization but reduces the number of gradient steps per
epoch. On the 96GB GPU, batch_size=4 should fit comfortably for TandemFoil. The noam branch
showed batch_size=8 hurt OOD, but batch_size=4 was never tested — a more conservative upscale
might capture the gradient quality benefit without the OOD penalty.
**Key changes:**
- `--batch_size 4`
- (Keep `lr=3e-4` — do NOT scale LR, per linear scaling rule guidance for small batch sizes)
**Expected outcome:** Slightly better in-distribution performance from higher quality gradients.
OOD impact unknown but smaller than batch_size=8's regression (which was severe on noam).

---

### 23. cp-panel-scale-sweep-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** The Cp panel feature is appended with a scale factor `cp_panel_scale=1.0` by
default. If the Cp panel feature has a much larger or smaller magnitude than other input features,
the model may over-weight or under-weight it. Testing `cp_panel_scale=0.5` and `cp_panel_scale=2.0`
explores whether the default scale is already well-calibrated or needs adjustment.
**Key changes:**
- Requires `--enable_cp_panel True` to be active
- Run A: `--enable_cp_panel True --cp_panel_scale 0.5`
- Run B: `--enable_cp_panel True --cp_panel_scale 2.0`
- `--wandb_group cp-panel-scale-sweep`
**Expected outcome:** One of the two scales outperforms `cp_panel_scale=1.0` by better
calibrating the contribution of the thin-airfoil pressure estimate relative to geometric features.

---

### 24. vortex-panel-scale-sweep-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** The vortex panel velocity features use `vortex_panel_scale=0.1` by default.
This is a physics-derived velocity estimate from Biot-Savart integration; its magnitude relative
to freestream velocity matters. Trying `vortex_panel_scale=0.05` (halved) and `vortex_panel_scale=0.2`
(doubled) tests whether 0.1 was chosen correctly or was a guess.
**Key changes:**
- Requires `--enable_vortex_panel_velocity True`
- Run A: `--enable_vortex_panel_velocity True --vortex_panel_scale 0.05`
- Run B: `--enable_vortex_panel_velocity True --vortex_panel_scale 0.2`
- `--wandb_group vortex-scale-sweep`
**Expected outcome:** Scale calibration improvement leads to 2–5% better surface pressure MAE.

---

### 25. combined-best-round1-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** If Round 1 identifies two or more independent improvements (e.g., a better
learning rate AND a useful physics feature), combining them should compound. This experiment
waits for Round 1 results and then stacks the 2–3 winning modifications into a single
"best-of-round-1" run. This is a mandatory Round 2 experiment once Round 1 results arrive.
**Key changes:** (PLACEHOLDER — to be filled in after Round 1 results)
- If lr=5e-4 wins in Round 1: `--lr 5e-4`
- If TE coord frame wins: `--enable_te_coord_frame True`
- If asinh wins: `--asinh_pressure True`
- Stack all winning modifications together
**Expected outcome:** Multiplicative improvement from orthogonal modifications. Target:
surface pressure MAE at least 10% below the single-modification best.

---

### 26. pressure-prior-addition-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** The `enable_pressure_prior_addition=True` flag adds a physics-derived pressure
estimate directly to the model's output prediction rather than using it only as an input feature.
This is an output-side prior: the model predicts the residual departure from the inviscid
pressure estimate, effectively learning to correct a physics-based baseline. This is a
fundamentally different approach to incorporating prior knowledge than feature concatenation.
**Key changes:**
- `--enable_pressure_prior_addition True --enable_cp_panel True`
  (pressure prior addition likely requires Cp panel to be enabled as the prior source)
- All other settings default
**Expected outcome:** If the inviscid Cp estimate is accurate within 10–20% on average, the
model only needs to learn the viscous correction — a simpler function — potentially improving
OOD generalization.

---

### 27. model-layers-4-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** Round 1 tests a 4-layer model. This experiment tests the interaction between
depth and the physics features. A 4-layer model with TE coord frame features enabled may be
qualitatively different from a 4-layer model without — deeper networks may better utilize
physics-informed inputs to build hierarchical aerodynamic representations. Combine the
depth increase with the most theoretically motivated physics feature.
**Key changes:**
- `--model_layers 4 --enable_te_coord_frame True`
- All other settings default
**Expected outcome:** The combination of extra depth AND TE coord frame provides a synergy
not present in either alone, improving surface pressure MAE by 5–10%.

---

### 28. ema-start-step-sweep-tandem
**Dataset:** TandemFoilSet
**Hypothesis:** EMA averaging starts at step 50 by default (`ema_start_step=50`). If training
runs for only a limited epoch count, step 50 might represent a significant fraction of total
steps, meaning the EMA model begins tracking before the live model has properly warmed up.
Starting EMA earlier (step 10) or later (step 100) tests whether the default onset is optimal.
**Key changes:**
- Run A: `--ema_start_step 10`
- Run B: `--ema_start_step 100`
- `--wandb_group ema-start-sweep-tandem`
**Expected outcome:** Optimizing EMA onset provides 1–3% improvement in surface pressure MAE.

---

## Summary Table

| Priority | # | Slug | Dataset | Key Change |
|---|---|---|---|---|
| HIGH | 1 | cosine-schedule-sweep-tandem | TandemFoil | cosine_t_max ∈ {50, 100, 150} |
| HIGH | 2 | cosine-schedule-sweep-airfrans | AirfRANS | cosine_t_max ∈ {50, 100, 150} |
| HIGH | 3 | ema-decay-sweep-tandem | TandemFoil | ema_decay ∈ {0.99, 0.999, 0.9995} |
| HIGH | 4 | re-stratified-sampling-tandem | TandemFoil | re_stratified_sampling=True |
| HIGH | 5 | anp-surface-decoder-tandem | TandemFoil | anp_srf=True |
| HIGH | 6 | disable-lookahead-tandem | TandemFoil | use_lookahead=False |
| HIGH | 7 | weight-decay-sweep-tandem | TandemFoil | weight_decay ∈ {1e-5, 5e-4, 1e-3} |
| HIGH | 8 | airfrans-task-scarce | AirfRANS | airfrans_task=scarce |
| MED | 9 | airfrans-task-reynolds | AirfRANS | airfrans_task=reynolds |
| MED | 10 | airfrans-task-aoa | AirfRANS | airfrans_task=aoa |
| MED | 11 | dropout-regularization-tandem | TandemFoil | model_dropout ∈ {0.05, 0.1} |
| MED | 12 | surface-refine-hidden-256-tandem | TandemFoil | surface_refine_hidden=256 |
| MED | 13 | surface-refine-layers-3-tandem | TandemFoil | surface_refine_layers=3 |
| MED | 14 | fourier-features-airfrans | AirfRANS | enable_fourier=True |
| MED | 15 | fewer-slices-tandem | TandemFoil | model_slices ∈ {64, 48} |
| MED | 16 | fewer-slices-airfrans | AirfRANS | model_slices ∈ {64, 48} |
| MED | 17 | residual-prediction-tandem | TandemFoil | residual_prediction=True |
| MED | 18 | asinh-pressure-scale-sweep-tandem | TandemFoil | asinh_scale ∈ {0.5, 1.0} |
| MED | 19 | combined-physics-features-tandem | TandemFoil | TE+Cp+wake all enabled |
| MED | 20 | adamw-with-tuned-lr-airfrans | AirfRANS | optimizer=adamw, lr=1e-3 |
| LOW | 21 | ema-disabled-tandem | TandemFoil | use_ema=False |
| LOW | 22 | batch-size-4-tandem | TandemFoil | batch_size=4 |
| LOW | 23 | cp-panel-scale-sweep-tandem | TandemFoil | cp_panel_scale ∈ {0.5, 2.0} |
| LOW | 24 | vortex-panel-scale-sweep-tandem | TandemFoil | vortex_panel_scale ∈ {0.05, 0.2} |
| LOW | 25 | combined-best-round1-tandem | TandemFoil | stack Round 1 winners |
| LOW | 26 | pressure-prior-addition-tandem | TandemFoil | enable_pressure_prior_addition=True |
| LOW | 27 | model-layers-4-with-te-tandem | TandemFoil | model_layers=4 + enable_te_coord_frame=True |
| LOW | 28 | ema-start-step-sweep-tandem | TandemFoil | ema_start_step ∈ {10, 100} |

**Total: 28 hypotheses (8 High, 12 Medium, 8 Lower/Speculative)**
