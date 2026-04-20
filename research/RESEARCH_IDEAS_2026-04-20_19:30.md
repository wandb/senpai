# SENPAI Research Ideas — 2026-04-20 19:30

Generated after reviewing all 25 PRs on the `radford` branch plus literature search
covering SpiderSolver, GeoTransolver, Transolver++, LinearNO, AB-UPT, and MARIO.

## Context

**Current bests:**
- TandemFoil: `val_primary/surface_pressure_mae` = 262.82 (PR #2414, core physics features)
- AirfRANS: `val_primary/surface_mse` = 0.3308 (PR #2423, AdamW lr=5e-4)
- DrivAerML: no baseline yet (target < 3.71%)

**Key gap:** AirfRANS is 77x worse than SpiderSolver SOTA (0.331 vs 0.0043) and 41x worse
than Transolver literature (0.0080). Root causes identified:
1. Timeout budget allows only 2-6 epochs vs SpiderSolver's 398 epochs
2. SpiderSolver uses boundary-guided patch tokenization — our slice assignment is not geometry-aware
3. Pressure channel (p) dominates: surface_mse_p ≈ 1.28; nut has extreme dynamic range
4. No asinh normalization on AirfRANS (already helps TandemFoil)
5. No wall-distance or boundary-layer features

---

## Ideas by Theme

---

### AIRFRANS — Close the 77x Gap

#### Idea 1: AirfRANS asinh-pressure + nut log-normalization
**Hypothesis:** The dominant error on AirfRANS is the pressure channel (surface_mse_p ≈ 1.28) and the turbulent viscosity field (nut), which spans several orders of magnitude. Applying `--asinh-pressure` (already validated on TandemFoil, +improvement on PR #2414) and a log or asinh transform specifically for nut should compress the dynamic range and dramatically reduce MSE on both channels. This is the single cheapest intervention with the highest expected per-epoch improvement.

**Flags to add (over PR #2423 config):**
```
--asinh-pressure \
--asinh-scale 0.75
```
Note: nut normalization may require a code path if not already wired; if `--asinh-pressure` only applies to p, also test whether a separate nut-transform flag exists.

**Primary metric:** `val_primary/surface_mse`, especially `surface_mse_p` and `surface_mse_nut`

---

#### Idea 2: AirfRANS residual prediction from freestream baseline
**Hypothesis:** AirfRANS fields (Ux, Uy, p, nut) have strong freestream baselines — far from the airfoil, Ux = U_inf, Uy = 0, p = 0, nut = 0. Predicting the deviation (residual) from this analytical freestream baseline instead of absolute values should dramatically reduce the prediction range and improve convergence. Already validated on TandemFoil (#2414). This is the same `--residual-prediction` flag applied to AirfRANS.

**Flags to add (over PR #2423 config):**
```
--residual-prediction
```

**Primary metric:** `val_primary/surface_mse` + `val_primary/volume_mse`

---

#### Idea 3: AirfRANS asinh-pressure + residual-prediction combined
**Hypothesis:** Both asinh normalization and residual prediction are individually promising for AirfRANS. Their combination, which was the winning formula on TandemFoil (#2414 used both), should stack additively. This tests whether the TandemFoil recipe transfers directly to AirfRANS.

**Flags:**
```
--asinh-pressure --asinh-scale 0.75 --residual-prediction
```

**Primary metric:** `val_primary/surface_mse`

---

#### Idea 4: AirfRANS cosine T_max reduction to match actual epoch budget
**Hypothesis:** With cosine_t_max=150 (default) but only 2-6 epochs completing within the 30-min timeout, the learning rate never anneals — the model never reaches the warm part of the schedule. Reducing T_max to 10-20 means the LR anneals to its minimum within the actual run budget. This is free throughput — same epochs, better schedule alignment. Combine with the winning AdamW lr=5e-4.

**Command change:**
```
--cosine-t-max 10
```
(try 10, 20, 30 as a sweep if budget allows; 10 is safest given 2-6 epoch budget)

**Primary metric:** `val_primary/surface_mse`

---

#### Idea 5: AirfRANS model_slices reduction for more epochs per budget
**Hypothesis:** Reducing `--model-slices` from 96 to 64 or 48 should cut per-step compute, enabling 30-50% more epochs within the 30-minute timeout. Given the extreme epoch-budget constraint (2-6 epochs vs literature's 398), more epochs at lower slice count should beat fewer epochs at higher slice count. This trades slice resolution for epoch count.

**Config change:**
```
--model-slices 64 --cosine-t-max 10
```

**Primary metric:** `val_primary/surface_mse`

---

#### Idea 6: AirfRANS wall-distance boundary layer feature
**Hypothesis:** The dominant error source on AirfRANS is the near-wall region (boundary layer, y+ ≈ 1). Neither the input features nor the model architecture currently encode wall distance or boundary-layer proximity. Adding wall distance as a normalized input feature (d_wall / chord) would give the model an explicit signal about where the steep gradients are. This mirrors the MARIO paper's boundary layer weighting function σbl(x). If `--enable-te-coord-frame` provides distance to trailing edge, the same principle applied globally gives wall-distance encoding.

**Check:** Does `--enable-te-coord-frame` on AirfRANS (not just tandemfoil) add distance-to-surface features? If so, test:
```
--enable-te-coord-frame
```
This needs investigation in the AirfRANS data path — the flag may be tandem-only.

**Primary metric:** `val_primary/surface_mse`

---

#### Idea 7: AirfRANS OOD tasks (scarce + reynolds) with winning config
**Hypothesis:** PR #2432 plans this but hasn't run. After establishing a strong `full` task baseline, test the winning full-task configuration on `--airfrans-task scarce` and `--airfrans-task reynolds`. These test generalization to data-scarce and Reynolds-extrapolation regimes — two of the three hardest tasks. This is straightforward task-transfer with no config changes beyond `--airfrans-task`.

**Commands (two separate runs):**
```
--airfrans-task scarce --optimizer adamw --lr 5e-4 --asinh-pressure --residual-prediction
--airfrans-task reynolds --optimizer adamw --lr 5e-4 --asinh-pressure --residual-prediction
```

---

### TANDEMFOIL — Push from 262.82

#### Idea 8: TandemFoil ANP cross-foil decoder (--anp-srf)
**Hypothesis:** The ANP decoder (`--anp-srf`) is explicitly flagged as HIGH PRIORITY in `tandemfoil/program.md`, sourced from `origin/frieren/anp-surface-decoder@7999a2e`. It has never been tested on the `radford` branch. This decoder enables cross-foil surface prediction using attentive neural processes, directly targeting the hardest OOD generalization modes (p_oodc, p_tan). This is the most important untested idea on TandemFoil.

**Flags:**
```
--anp-srf \
--optimizer adamw --lr 5e-4 \
--enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --cp-panel-scale 1.0 \
--enable-pressure-prior-addition --asinh-pressure --residual-prediction
```
(Layer the ANP decoder on top of the winning core physics config from #2414)

**Primary metric:** `val_primary/surface_pressure_mae`, with emphasis on `legacy_noam/p_oodc` and `legacy_noam/p_tan`

---

#### Idea 9: TandemFoil AdamW + core physics stack (best of #2416 + #2414 combined)
**Hypothesis:** PR #2414 (core physics, Lion lr=3e-4) = 262.82. PR #2416 (AdamW lr=5e-4, no physics) = 269.32. Neither combined AdamW with core physics features. This is the obvious composition experiment: the two individually winning changes should stack.

**Flags:**
```
--optimizer adamw --lr 5e-4 \
--enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --cp-panel-scale 1.0 \
--enable-pressure-prior-addition --asinh-pressure --residual-prediction
```

**Primary metric:** `val_primary/surface_pressure_mae`

Note: This is in-flight as `tanjiro` Round 2, so do NOT re-assign. Listed here for completeness.

---

#### Idea 10: TandemFoil deeper model (6L/192d) with core physics + AdamW
**Hypothesis:** The current 3L/192d model may be capacity-limited on TandemFoil's multi-foil geometry. SpiderSolver uses deeper dual-attention stacks. Doubling depth to 6 layers with same width should improve the model's ability to learn cross-foil interactions without quadrupling parameter count. Combine with the winning physics config.

**Flags:**
```
--model-layers 6 --model-hidden-dim 192 --model-heads 3 --model-slices 96 \
--optimizer adamw --lr 5e-4 \
--enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --cp-panel-scale 1.0 \
--enable-pressure-prior-addition --asinh-pressure --residual-prediction
```

**Primary metric:** `val_primary/surface_pressure_mae`

---

#### Idea 11: TandemFoil Fourier position encoding with physics features
**Hypothesis:** `--enable-fourier` adds Fourier positional encoding to point coordinates. For TandemFoil, the two-airfoil geometry has periodically-varying pressure fields (interference patterns, wake interactions). Fourier encoding provides a natural basis for these. Never tested in combination with the core physics stack.

**Flags (on top of winning #2414 config):**
```
--enable-fourier \
--enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --cp-panel-scale 1.0 \
--enable-pressure-prior-addition --asinh-pressure --residual-prediction \
--optimizer adamw --lr 5e-4
```

**Primary metric:** `val_primary/surface_pressure_mae`

---

#### Idea 12: TandemFoil cp-panel-scale sweep (0.1, 0.5, 1.0, 2.0)
**Hypothesis:** PR #2414 used `--cp-panel-scale 1.0` without ablating. The scale controls how strongly the analytical panel-method Cp is injected into the model. Too high can dominate the learned features; too low is ineffective. A sweep from 0.1 to 2.0 identifies the optimal injection strength. This is a 4-point hyperparameter sweep on a single well-isolated knob.

**Commands (4 runs):**
```
--cp-panel-scale 0.1
--cp-panel-scale 0.5
--cp-panel-scale 1.0   # already ran
--cp-panel-scale 2.0
```
All other flags identical to #2414.

**Primary metric:** `val_primary/surface_pressure_mae`

---

#### Idea 13: TandemFoil Reynolds-stratified sampling + core physics
**Hypothesis:** PR #2436 tests `--re-stratified-sampling` without physics features. The more natural test is `--re-stratified-sampling` combined with the winning core physics stack — if Reynolds generalization (p_re) is the bottleneck, stratified sampling should directly improve it. The physics prior (panel Cp) is Reynolds-number-aware, so the two should reinforce each other.

**Flags:**
```
--re-stratified-sampling \
--enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only --cp-panel-scale 1.0 \
--enable-pressure-prior-addition --asinh-pressure --residual-prediction \
--optimizer adamw --lr 5e-4
```

**Primary metric:** `val_primary/surface_pressure_mae`, especially `legacy_noam/p_re`

---

### DRIVAERML — Establish First Baseline and Beat 3.71%

#### Idea 14: DrivAerML AdamW lr=5e-4 with cosine T_max=20 (fast iteration baseline)
**Hypothesis:** Before any tuning, establish whether AdamW (which won on AirfRANS by 38%) also outperforms Lion on DrivAerML. This is the most important baseline experiment — zero DrivAerML results exist. Use the lesson from TandemFoil: reduce cosine_t_max to 20 so the schedule actually anneals within the budget. Use geometry_supernodes=4096 and surface_anchor_points=8000 to start.

**Command:**
```
--dataset drivaerml \
--optimizer adamw --lr 5e-4 --weight-decay 1e-4 \
--cosine-t-max 20 \
--geometry-supernodes 4096 --surface-anchor-points 8000 \
--use-ema --ema-decay 0.999 --ema-start-step 50 \
--use-lookahead \
--surface-refine --surface-refine-hidden 128 --surface-refine-layers 2 \
--model-layers 3 --model-hidden-dim 192 --model-heads 3 --model-slices 96
```

**Primary metric:** `val_primary/surface_rel_l2_pct` (target < 3.71%)

---

#### Idea 15: DrivAerML geometry supernodes uplift (4096 → 8192)
**Hypothesis:** GeoTransolver achieves 2.86% (vs our 3.71% target) using multi-scale geometry context. AB-UPT uses 16384 supernodes. Our default is 4096 — 4x fewer geometry tokens than literature best. Doubling to 8192 while keeping the 96GB VRAM budget is likely feasible. This directly tests the geometry resolution hypothesis. Pair with AdamW (already established as the better optimizer) and cosine_t_max=20.

**Command:**
```
--dataset drivaerml \
--optimizer adamw --lr 5e-4 \
--cosine-t-max 20 \
--geometry-supernodes 8192 \
--surface-anchor-points 8000
```

**Primary metric:** `val_primary/surface_rel_l2_pct`

---

#### Idea 16: DrivAerML surface anchor points uplift (8000 → 16000)
**Hypothesis:** AB-UPT and GeoTransolver both use 16384 surface anchor points; our default is 8000. Surface anchor density directly controls how finely the surface pressure field is sampled during training. Doubling anchors to 16000 (within 96GB VRAM) should directly reduce surface_rel_l2_pct, especially for high-curvature regions on the car body.

**Command:**
```
--dataset drivaerml \
--optimizer adamw --lr 5e-4 \
--cosine-t-max 20 \
--geometry-supernodes 4096 \
--surface-anchor-points 16000
```

**Primary metric:** `val_primary/surface_rel_l2_pct`

---

#### Idea 17: DrivAerML deeper model (6L/256d) for complex 3D geometry
**Hypothesis:** DrivAerML is a 3D automotive body — far more geometrically complex than 2D airfoils. The literature's best result (GeoTransolver 2.86%) used 20 layers with GALE attention. Our baseline is 3 layers. Even within standard Transolver, increasing to 6 layers and 256d should provide substantially more model capacity for the complex 3D geometry without requiring architecture changes. GeoTransolver shows a clear depth scaling law: 6→12→20 layers = 3.52%→3.18%→2.86%.

**Command:**
```
--dataset drivaerml \
--model-layers 6 --model-hidden-dim 256 --model-heads 4 --model-slices 128 \
--optimizer adamw --lr 5e-4 \
--cosine-t-max 20 \
--geometry-supernodes 4096 --surface-anchor-points 8000
```

**Primary metric:** `val_primary/surface_rel_l2_pct`

---

#### Idea 18: DrivAerML combined uplift (supernodes + anchors + depth)
**Hypothesis:** Based on GeoTransolver's results, the three strongest levers for DrivAerML are: (1) geometry resolution, (2) surface sampling density, and (3) model depth. This experiment tests all three together — the combination most likely to beat the 3.71% target in a single shot. Use the VRAM budget aggressively.

**Command:**
```
--dataset drivaerml \
--model-layers 6 --model-hidden-dim 256 --model-heads 4 --model-slices 96 \
--optimizer adamw --lr 5e-4 \
--cosine-t-max 20 \
--geometry-supernodes 8192 \
--surface-anchor-points 16000
```

**Primary metric:** `val_primary/surface_rel_l2_pct`

---

### CROSS-CUTTING — Optimizer and Training Improvements

#### Idea 19: Adaptive EMA decay based on training progress (EMA warm start)
**Hypothesis:** With only 2-6 epochs completing, early EMA checkpoints (ema_start_step=50) may never be reached, or the EMA model may be dominated by poor early weights. A shorter ema_start_step (e.g., 10 or 20 steps) ensures the EMA model is built from more training data. Alternatively, removing EMA entirely may be better when epochs are so few — the standard model at step N may already be the best checkpoint. Test EMA-off on TandemFoil and AirfRANS to confirm whether EMA helps at all within our budget.

**Ablation (for any benchmark):**
```
# Remove EMA scaffold entirely
# (omit --use-ema --use-lookahead)
```

**Note:** PR #2431 (AirfRANS EMA ablation) plans this but hasn't run. Combine with the best physics config.

---

#### Idea 20: Asymmetric Q/K projection for Transolver slices (LinearNO insight)
**Hypothesis:** The LinearNO paper shows that standard Transolver slice attention (where Q and K use the same projection) creates uniform slice weights, reducing effective capacity. Using asymmetric Q and K projections — where the slice assignment is computed from a different linear transform than the slice readout — fixes the uniformity problem and improves performance without any architecture change beyond the projection matrices. If this is controllable via a flag in train.py (check `--model-asymmetric-qk` or similar), it should be tested immediately. If not, this is a one-line code change worth requesting as a student task.

**Investigation needed:** Check whether `train.py` has a flag for asymmetric Q/K or Transolver++ adaptive temperature (τ per point). If available:
```
--transolver-asymmetric-qk  # or --transolver-ada-temp
```

---

## Prioritization (Top 7 for immediate assignment)

Ranked by expected impact × ease of implementation:

1. **Idea 1** (AirfRANS asinh-pressure) — proven on TandemFoil, directly addresses dominant error channel
2. **Idea 3** (AirfRANS asinh + residual combined) — combines two validated improvements
3. **Idea 4** (AirfRANS cosine T_max=10) — free improvement, no risk
4. **Idea 14** (DrivAerML AdamW baseline) — first DrivAerML result ever needed immediately
5. **Idea 8** (TandemFoil ANP decoder) — HIGH PRIORITY per program.md, never tested
6. **Idea 9** (TandemFoil AdamW + core physics) — natural composition, in-flight as tanjiro
7. **Idea 17** (DrivAerML deeper model) — literature shows clear depth scaling law

---

## Notes on What NOT to Try

- Do NOT re-assign: TandemFoil physics+AdamW LR sweep (tanjiro), ANP decoder experiments (shinji), DrivAerML AdamW baseline sweep (shoya), DrivAerML surface-points budget sweep (shouko), DrivAerML model capacity sweep (mitsuha), DrivAerML cosine T_max sweep (taki), DrivAerML anchor token budget sweep (nezuko)
- Do NOT try full physics stack without ablating — PR #2413 tests this; wait for results before layering more features
- Do NOT increase model_slices above 96 without first establishing the per-epoch compute budget
