<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
-->

# Research Ideas — Round 37 (2026-04-21)

## Context and baselines

| Dataset | Metric | Current best | PR |
|---|---|---|---|
| TandemFoil | `val_primary/surface_pressure_mae` | 44.72 | #2724 |
| AirfRANS | `val_primary/surface_mse` | 0.001479 | #2771 |
| DrivAerML | `val_primary/surface_rel_l2_pct` | 4.619% | #2691 |

External targets:
- AirfRANS: SpiderSolver Surf MSE=0.0043 (BEATEN), Vol MSE=0.0017 (in range but not primary)
- DrivAerML: Transolver-3 3.71% (need ~1% improvement on our packaged split)
- TandemFoil: no external leaderboard

Key recent insights that shape this round:
1. **Width-dominant architecture** (3L/256d, PR #2771): 3L/256d beats 4L/256d on AirfRANS (0.001479 vs 0.00277). This is the single most important unvalidated structural finding. It has not been tested on TandemFoil or DrivAerML.
2. **Phase transition / gc=0.5 interaction** (PR #2774, #2755): AirfRANS shows stochastic divergence ~ep200 with T_max=5; best model found just before divergence. gc=0.5 extended stable training and gave AirfRANS=0.00277 before the 3L/256d discovery. That means 3L/256d + gc=0.5 compound has not been tested anywhere.
3. **TandemFoil LR monotone trend**: 3e-4 (52.81) → 2e-4 (45.07) → 1.5e-4 (44.72). Optimal LR is likely at or below 1e-4. No run has tried lr=1e-4 or lr=8e-5 on TandemFoil, nor any run at 3L/256d for TandemFoil.
4. **DrivAerML gc=0.5 unconfirmed**: In-flight as PR #2805 (4L/640d) and #2803 (5L/512d). But neither tests gc=0.5 at 3L/256d or 4L/512d directly.

In-flight experiments excluded from this round (do not duplicate):
#2805 DrivAerML 4L/640d, #2803 DrivAerML 5L/512d, #2802 AirfRANS pressure-weighted 4L/256d lr=3e-4, #2799 AirfRANS 5L/256d, #2797 DrivAerML lr=3e-4 4L/512d, #2795 DrivAerML T_max=20 4L/512d, #2791 AirfRANS gc=1.5 no-WD, #2790 DrivAerML gc=1.0+WD 4L/320d, #2783 DrivAerML WD=1e-2+T_max=20 4L/320d, #2781 DrivAerML T_max=10 4L/320d, #2780 AirfRANS 4L/320d, #2779 DrivAerML lr=3e-4 4L/320d, #2776 AirfRANS lr=1e-3+gc=1.5, #2775 TandemFoil 5L/256d, #2773 TandemFoil T_max=5, #2772 TandemFoil 4L/256d, #2771 AirfRANS 3L/256d (already merged), #2769 AirfRANS T_max=3, #2768 AirfRANS lr=5e-4, #2767 AirfRANS gc=1.5+WD+T_max=10, #2766 AirfRANS gc=1.5+T_max=5, #2765 AirfRANS gc=2.0, #2764 AirfRANS lr=1e-3, #2761 DrivAerML gc=1.5+WD 4L/320d, #2759 DrivAerML 4L/256d throughput, #2758 AirfRANS 5L/256d (dupe of #2799)

---

## Hypotheses

### H1: Width-dominant architecture transfer — TandemFoil 3L/256d at lr=1.5e-4

**Dataset**: TandemFoil  
**Priority**: CRITICAL — highest expected impact single experiment

**Rationale**: The width-dominant finding (3L/256d beats 4L/256d) emerged on AirfRANS. The current TandemFoil best uses 3L/192d. Widening to 256d without adding depth may give the same ~47% lift seen on AirfRANS. Combined with the best known TandemFoil LR (1.5e-4), this is the most likely path to a large TandemFoil gain. The TandemFoil current best is 44.72 at 3L/192d — if the 3L/256d vs 3L/192d ratio on AirfRANS (0.001479 / 0.007264 = 0.20) approximates the TandemFoil ratio, a 3L/256d TandemFoil run at lr=1.5e-4 could reach the mid-30s or lower.

**Exact command**:
```bash
cd target/icml2026
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset tandemfoil \
  --model-layers 3 --model-hidden-dim 256 --model-heads 4 \
  --optimizer lion --lr 1.5e-4 \
  --cosine-t-max 10 \
  --no-use-ema \
  --enable-fourier \
  --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only \
  --asinh-pressure --residual-prediction \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --wandb-group radford-tandemfoil-3l256d \
  --wandb-name tandemfoil-3l256d-lr1.5e4
```

**Expected outcome**: val_primary/surface_pressure_mae in the 30–42 range (best case: ~30, floor: ~42 if width doesn't transfer). If result is >44.72 the 3L/256d architecture does not transfer to TandemFoil.

---

### H2: Width-dominant architecture transfer — DrivAerML 3L/256d

**Dataset**: DrivAerML  
**Priority**: HIGH — second most likely large gain

**Rationale**: The current DrivAerML best uses 4L/512d (val=4.619%). Neither 3L/256d nor any shallower-wider variant has been tested on DrivAerML. The 3L/256d finding on AirfRANS suggests that depth scaling past 3L may be counterproductive. DrivAerML with 3L/256d reduces parameter count but increases per-layer width-to-depth ratio, which may be better suited to surface-only CFD prediction. Also runs faster, giving more epochs.

**Exact command**:
```bash
cd target/icml2026
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset drivaerml \
  --model-layers 3 --model-hidden-dim 256 --model-heads 4 \
  --optimizer adamw --lr 5e-4 \
  --cosine-t-max 30 \
  --no-use-ema \
  --enable-fourier \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --wandb-group radford-drivaerml-3l256d \
  --wandb-name drivaerml-3l256d-gc1.0
```

**Expected outcome**: val_primary/surface_rel_l2_pct below 4.619%. Best case: 3.5–4.0% if the width-dominant effect transfers. The faster epoch time means more training steps in the same wall-clock budget, which also helps.

---

### H3: TandemFoil lower LR bracket — lr=1e-4 and lr=8e-5 at 3L/192d

**Dataset**: TandemFoil  
**Priority**: HIGH — the LR sweep is a monotone trend with no sign of stopping

**Rationale**: Every LR reduction on TandemFoil has improved the result: 3e-4 (52.81) → 2e-4 (45.07) → 1.5e-4 (44.72). The curve has not yet inflected. Lion optimizer with cosine annealing typically shows this behavior at moderate LRs — the optimal is likely still lower. Testing lr=1e-4 and lr=8e-5 at the current best 3L/192d architecture will confirm whether the trend continues and establish the true optimal LR for this architecture. This experiment is a prerequisite for knowing whether H1 is finding architecture improvement vs just operating at a better LR.

**Exact command** (two trials on separate GPUs):
```bash
# Trial 1 — lr=1e-4
cd target/icml2026
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset tandemfoil \
  --model-layers 3 --model-hidden-dim 192 --model-heads 3 \
  --optimizer lion --lr 1e-4 \
  --cosine-t-max 10 \
  --no-use-ema \
  --enable-fourier \
  --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only \
  --asinh-pressure --residual-prediction \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --wandb-group radford-tandemfoil-lr-sweep \
  --wandb-name tandemfoil-3l192d-lr1e4

# Trial 2 — lr=8e-5
CUDA_VISIBLE_DEVICES=1 python train.py \
  --dataset tandemfoil \
  --model-layers 3 --model-hidden-dim 192 --model-heads 3 \
  --optimizer lion --lr 8e-5 \
  --cosine-t-max 10 \
  --no-use-ema \
  --enable-fourier \
  --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only \
  --asinh-pressure --residual-prediction \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --wandb-group radford-tandemfoil-lr-sweep \
  --wandb-name tandemfoil-3l192d-lr8e5
```

**Expected outcome**: At lr=1e-4, val_primary/surface_pressure_mae below 44.72. At lr=8e-5 the LR may be too low for a cosine T_max=10 schedule (effective floor too small to make progress). If lr=1e-4 beats baseline, it gives a clean lower LR optimum and informs the 3L/256d run in H1.

---

### H4: AirfRANS 3L/256d + gc=0.5 — compound the two biggest wins

**Dataset**: AirfRANS  
**Priority**: HIGH — untested compound of two orthogonal wins

**Rationale**: The 3L/256d win (PR #2771, val=0.001479) used gc=1.0. The gc=0.5 win (PR #2774, val=0.00277) used 4L/256d. These two improvements have never been combined. On AirfRANS the phase-transition phenomenon benefits from aggressive gradient clipping (gc=0.5) by extending stable training before divergence. At 3L/256d, which may have a different gradient dynamics profile (shallower network = smaller gradient norm at depth), gc=0.5 may stabilize training even further, potentially enabling more epochs before the phase transition.

**Exact command**:
```bash
cd target/icml2026
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset airfrans --airfrans-task full \
  --model-layers 3 --model-hidden-dim 256 --model-heads 4 \
  --optimizer adamw --lr 7e-4 \
  --cosine-t-max 5 \
  --no-use-ema \
  --enable-fourier \
  --weight-decay 1e-2 \
  --grad-clip 0.5 \
  --wandb-group radford-airfrans-3l256d-gc05 \
  --wandb-name airfrans-3l256d-gc0.5-T5
```

**Expected outcome**: val_primary/surface_mse below 0.001479. Best case: 0.0010–0.0013 if gradient clipping and the width-dominant effect compound. This is the most likely near-term path to a new AirfRANS state-of-the-art.

---

### H5: DrivAerML gc=0.5 at 4L/512d — stability transfer from AirfRANS

**Dataset**: DrivAerML  
**Priority**: HIGH — in-flight experiments test at different architectures; this targets the current best

**Rationale**: In-flight PRs #2805 and #2803 test gc=0.5 at larger architectures (4L/640d and 5L/512d) that have never been validated. No run has tested gc=0.5 at the current best DrivAerML architecture (4L/512d, val=4.619%). On AirfRANS, gc=0.5 at the baseline architecture gave a large win (0.00277 vs 0.007264 at 4L/256d). Testing gc=0.5 directly at the known-good 4L/512d config is a controlled test of whether the gc=0.5 stabilization benefit transfers to DrivAerML without introducing an architecture variable.

**Exact command**:
```bash
cd target/icml2026
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset drivaerml \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 \
  --optimizer adamw --lr 5e-4 \
  --cosine-t-max 30 \
  --no-use-ema \
  --enable-fourier \
  --weight-decay 1e-4 \
  --grad-clip 0.5 \
  --wandb-group radford-drivaerml-gc05 \
  --wandb-name drivaerml-4l512d-gc0.5
```

**Expected outcome**: val_primary/surface_rel_l2_pct below 4.619%. If gc=0.5 helps DrivAerML similarly to AirfRANS, expect 3.8–4.3%. This also gives a controlled comparison point for the in-flight larger-architecture gc=0.5 experiments.

---

### H6: AirfRANS vol metrics push — 3L/256d + longer run (T_max=10)

**Dataset**: AirfRANS  
**Priority**: MEDIUM-HIGH — AirfRANS surface already beats external target; vol is the new frontier

**Rationale**: AirfRANS Surf MSE=0.001479 already demolishes the SpiderSolver surface target (0.0043). The volume MSE external target is 0.0017 (SpiderSolver). Current best vol MSE on AirfRANS is not explicitly tracked as a primary metric but the paper will need it. The T_max=5 config drives the phase transition fast but may sacrifice volume quality (the vol nodes see different flow regimes). A T_max=10 config at 3L/256d gives the model more time at stable LR before the cosine trough, potentially improving volume fidelity. This is exploratory but the paper needs a strong vol number to tell the full AirfRANS story.

**Exact command**:
```bash
cd target/icml2026
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset airfrans --airfrans-task full \
  --model-layers 3 --model-hidden-dim 256 --model-heads 4 \
  --optimizer adamw --lr 7e-4 \
  --cosine-t-max 10 \
  --no-use-ema \
  --enable-fourier \
  --weight-decay 1e-2 \
  --grad-clip 1.0 \
  --wandb-group radford-airfrans-3l256d-vol \
  --wandb-name airfrans-3l256d-T10-vol
```

**Expected outcome**: val_primary/surface_mse near or below 0.001479 (primary), and val_primary/volume_mse potentially approaching 0.0017 (SpiderSolver vol target). If surface degrades significantly vs T_max=5, the T_max tradeoff between surf and vol becomes visible and informs the paper.

---

### H7: TandemFoil weight decay sweep — WD=1e-3 and WD=5e-3 at 3L/192d

**Dataset**: TandemFoil  
**Priority**: MEDIUM — WD=1e-2 was huge for AirfRANS but TandemFoil uses Lion with WD=1e-4

**Rationale**: Every AirfRANS win with WD=1e-2 was with AdamW. TandemFoil uses Lion, where weight decay semantics differ (Lion applies normalized gradient updates, so WD is more like a momentum-scaled penalty). No TandemFoil run has tested WD above 1e-4. On AirfRANS, WD=1e-2 gave a large improvement even with Lion variants. On TandemFoil, the physics features (TE coordinate frame, Cp panel) provide strong structural inductive bias that may reduce the need for high WD, but WD=1e-3 at minimum is worth testing. Run two trials in parallel: WD=1e-3 and WD=5e-3 at the current best config.

**Exact command** (two trials):
```bash
# Trial 1 — WD=1e-3
cd target/icml2026
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset tandemfoil \
  --model-layers 3 --model-hidden-dim 192 --model-heads 3 \
  --optimizer lion --lr 1.5e-4 \
  --cosine-t-max 10 \
  --no-use-ema \
  --enable-fourier \
  --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only \
  --asinh-pressure --residual-prediction \
  --weight-decay 1e-3 \
  --grad-clip 1.0 \
  --wandb-group radford-tandemfoil-wd-sweep \
  --wandb-name tandemfoil-3l192d-wd1e3

# Trial 2 — WD=5e-3
CUDA_VISIBLE_DEVICES=1 python train.py \
  --dataset tandemfoil \
  --model-layers 3 --model-hidden-dim 192 --model-heads 3 \
  --optimizer lion --lr 1.5e-4 \
  --cosine-t-max 10 \
  --no-use-ema \
  --enable-fourier \
  --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only \
  --asinh-pressure --residual-prediction \
  --weight-decay 5e-3 \
  --grad-clip 1.0 \
  --wandb-group radford-tandemfoil-wd-sweep \
  --wandb-name tandemfoil-3l192d-wd5e3
```

**Expected outcome**: WD=1e-3 may give a small improvement over WD=1e-4 (current). WD=5e-3 is a larger jump and may overregularize given the physics feature inductive bias. Baseline is val=44.72.

---

### H8: DrivAerML Lion optimizer at 4L/512d — transfer the TandemFoil optimizer win

**Dataset**: DrivAerML  
**Priority**: MEDIUM — Lion beats AdamW on TandemFoil; never properly tested on DrivAerML at current best arch

**Rationale**: Early DrivAerML experiments (PR #2440) swept AdamW vs Lion at the first architecture (4L/256d, before current best was established). All subsequent DrivAerML experiments use AdamW. The current best architecture is 4L/512d. Lion has different gradient dynamics (sign-based update, no second moment) that may be advantageous for DrivAerML's large-scale surface geometry. The TandemFoil + Lion combination has been consistently better than AdamW there. Testing Lion at the current DrivAerML best architecture is a clean controlled test that has never been done.

**Exact command**:
```bash
cd target/icml2026
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset drivaerml \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 \
  --optimizer lion --lr 3e-4 \
  --cosine-t-max 30 \
  --no-use-ema \
  --enable-fourier \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --wandb-group radford-drivaerml-lion \
  --wandb-name drivaerml-4l512d-lion-lr3e4
```

Note: Lion typically uses a lower LR than AdamW (roughly 3-10x lower). AdamW baseline uses lr=5e-4, so lr=3e-4 for Lion is a conservative starting point. If this diverges, fall back to lr=1e-4.

**Expected outcome**: val_primary/surface_rel_l2_pct below 4.619%. If Lion transfers the TandemFoil pattern, expect 3.8–4.3%. If DrivAerML is more AdamW-friendly (like AirfRANS), expect neutral or slightly worse.

---

### H9: AirfRANS T_max=5 extended run at 3L/256d — exploit phase transition at golden-config LR

**Dataset**: AirfRANS  
**Priority**: MEDIUM — the phase transition at T_max=5 was the key to PR #2774 and #2755; 3L/256d + T_max=5 untested

**Rationale**: The extended 180-min run on AirfRANS (PR #2755) found val=0.003904 at 4L/256d with T_max=5, right before divergence at ep205. PR #2771 (3L/256d, val=0.001479) used golden config with the default T_max. The specific interaction of 3L/256d + T_max=5 + extended time has not been tested. The 3L/256d model has smaller gradient norms (shallower) and may stay stable longer with T_max=5, potentially reaching a deeper minimum before divergence. This is a natural extension to lock in the best possible AirfRANS number.

**Exact command**:
```bash
cd target/icml2026
SENPAI_TIMEOUT_MINUTES=180 CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset airfrans --airfrans-task full \
  --model-layers 3 --model-hidden-dim 256 --model-heads 4 \
  --optimizer adamw --lr 7e-4 \
  --cosine-t-max 5 \
  --no-use-ema \
  --enable-fourier \
  --weight-decay 1e-2 \
  --grad-clip 1.0 \
  --wandb-group radford-airfrans-3l256d-extended \
  --wandb-name airfrans-3l256d-T5-180min
```

**Expected outcome**: val_primary/surface_mse potentially reaching 0.0010–0.0013 if the phase transition at 3L/256d occurs later or at a deeper basin than at 4L/256d. Monitor the W&B run for divergence — save the best checkpoint before it occurs.

---

### H10: Cross-benchmark pressure-weighted loss at 3L/256d — transfer the AirfRANS loss insight

**Dataset**: TandemFoil AND DrivAerML  
**Priority**: MEDIUM — pressure-weighted loss gave a large AirfRANS win at 3L/192d but only at the old architecture; not yet tested on TandemFoil or DrivAerML

**Rationale**: PR #2802 (in-flight) tests pressure-weighted loss at 4L/256d on AirfRANS. The earlier result (at 3L/192d, val=0.00435) showed the concept worked before the golden config was established. For TandemFoil, pressure fidelity is the primary metric (`surface_pressure_mae`) — a 20x pressure loss weight should directly improve the optimization target. For DrivAerML, `surface_cp` is the only target, so any loss reweighting is between field magnitude contributions and won't have the same multi-field interpretation. Still worth testing both.

**TandemFoil command**:
```bash
cd target/icml2026
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset tandemfoil \
  --model-layers 3 --model-hidden-dim 256 --model-heads 4 \
  --optimizer lion --lr 1.5e-4 \
  --cosine-t-max 10 \
  --no-use-ema \
  --enable-fourier \
  --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only \
  --asinh-pressure --residual-prediction \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --pressure-loss-weight 20.0 \
  --wandb-group radford-tandemfoil-ploss \
  --wandb-name tandemfoil-3l256d-ploss20
```

**DrivAerML command**:
```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --dataset drivaerml \
  --model-layers 3 --model-hidden-dim 256 --model-heads 4 \
  --optimizer adamw --lr 5e-4 \
  --cosine-t-max 30 \
  --no-use-ema \
  --enable-fourier \
  --weight-decay 1e-4 \
  --grad-clip 1.0 \
  --pressure-loss-weight 5.0 \
  --wandb-group radford-drivaerml-ploss \
  --wandb-name drivaerml-3l256d-ploss5
```

Note: If `--pressure-loss-weight` is not yet a CLI argument in train.py, the student should check whether it exists or needs to be added. The AirfRANS version (#2802) confirms the flag exists.

**Expected outcome**: TandemFoil: val below 44.72 (this is a direct optimization target alignment so expect meaningful improvement). DrivAerML: the effect is less clear since it's a single-channel target; expect neutral or mild improvement.

---

### H11: AirfRANS WD=1e-2 at 3L/256d — confirm regularization benefit at golden arch

**Dataset**: AirfRANS  
**Priority**: MEDIUM — WD=1e-2 was part of the golden config for 4L/256d (PR #2709) but PR #2771's 3L/256d result used what WD?

**Rationale**: PR #2771 (3L/256d, val=0.001479) achieved the current best. The PR body describes it as "3L/256d with golden config" but the command needs verification. If WD=1e-2 was already included, this is moot. If WD=1e-2 was NOT included in #2771, then adding it to 3L/256d is an untested compound. Check the W&B run for PR #2771 to confirm the WD used. If WD was 1e-4 (default), this experiment should be prioritized.

**Command to check first** (verify PR #2771 config, then if WD=1e-4 was used):
```bash
cd target/icml2026
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset airfrans --airfrans-task full \
  --model-layers 3 --model-hidden-dim 256 --model-heads 4 \
  --optimizer adamw --lr 7e-4 \
  --cosine-t-max 5 \
  --no-use-ema \
  --enable-fourier \
  --weight-decay 1e-2 \
  --grad-clip 1.0 \
  --wandb-group radford-airfrans-3l256d-wd \
  --wandb-name airfrans-3l256d-wd1e2
```

**Expected outcome**: If WD was 1e-4 in #2771, adding WD=1e-2 could give a similar ~30% improvement seen when WD was added to 4L/256d (PR #2709). If WD was already 1e-2 in #2771, this is a pure replication and likely redundant.

---

### H12: TandemFoil gc=0.5 at 3L/192d — check if AirfRANS gc=0.5 gain transfers

**Dataset**: TandemFoil  
**Priority**: MEDIUM — gc=0.5 gave huge AirfRANS win; never tested on TandemFoil

**Rationale**: gc=0.5 is the single biggest AirfRANS lever after architecture (gave 0.00277 vs 0.007264 at 4L/256d). No TandemFoil run has tested gc below 1.0. TandemFoil uses Lion instead of AdamW — Lion's sign-based update has inherently clipped gradient direction but not magnitude in the usual sense, so gc=0.5 may interact differently. Still, gradient norm stability has been empirically important across benchmarks and testing gc=0.5 at the current best TandemFoil config is straightforward.

**Exact command**:
```bash
cd target/icml2026
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset tandemfoil \
  --model-layers 3 --model-hidden-dim 192 --model-heads 3 \
  --optimizer lion --lr 1.5e-4 \
  --cosine-t-max 10 \
  --no-use-ema \
  --enable-fourier \
  --enable-te-coord-frame --enable-cp-panel --enable-cp-panel-tandem-only \
  --asinh-pressure --residual-prediction \
  --weight-decay 1e-4 \
  --grad-clip 0.5 \
  --wandb-group radford-tandemfoil-gc05 \
  --wandb-name tandemfoil-3l192d-gc0.5
```

**Expected outcome**: val_primary/surface_pressure_mae below 44.72 if gradient clipping generalizes. If neutral or worse, this rules out gc=0.5 as a TandemFoil lever and narrows the focus to architecture and LR.

---

### H13: AirfRANS SGDR warm restarts — replace cosine annealing with cosine warm restarts

**Dataset**: AirfRANS  
**Priority**: MEDIUM-LOW — fundamentally different LR schedule, untried on any benchmark

**Rationale**: All experiments so far use `CosineAnnealingLR` with a fixed T_max. SGDR (Loshchilov & Hutter, 2016) uses `CosineAnnealingWarmRestarts` with T_0 period and T_mult multiplier. The phase transition phenomenon on AirfRANS is essentially the model being driven by the cosine LR into a sharp basin just before divergence. SGDR with T_mult=2 gives geometrically increasing cycle lengths, which could allow the model to find increasingly deep basins across restarts without requiring a single lucky phase transition. This is a structurally different exploration strategy that has not been tested. The risk is that it prevents the single deep basin dive that has been working.

**Exact command** (check if `--cosine-warm-restarts` or `--sgdr` flag exists; if not, student should implement via T_0/T_mult params):
```bash
cd target/icml2026
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset airfrans --airfrans-task full \
  --model-layers 3 --model-hidden-dim 256 --model-heads 4 \
  --optimizer adamw --lr 7e-4 \
  --cosine-t-max 10 --cosine-t-mult 2 \
  --no-use-ema \
  --enable-fourier \
  --weight-decay 1e-2 \
  --grad-clip 0.5 \
  --wandb-group radford-airfrans-sgdr \
  --wandb-name airfrans-3l256d-sgdr-T10-T2
```

Note: `--cosine-t-mult` may not yet be a CLI arg. If not, student should check if `CosineAnnealingWarmRestarts` is supported in train.py and add a minimal `--cosine-t-mult` argument defaulting to 1.0 (which recovers standard cosine annealing).

**Expected outcome**: If SGDR restarts help, we should see multiple progressive improvements at each restart boundary in the loss curve. Best case: matches or beats the phase-transition result without requiring the lucky divergence timing.

---

### H14: DrivAerML WD=1e-2 + gc=0.5 at 4L/512d — compound the two strongest AirfRANS levers

**Dataset**: DrivAerML  
**Priority**: MEDIUM-LOW — WD=1e-2 catastrophically diverged on DrivAerML alone (#2790 was gc=1.0+WD which diverged), but gc=0.5 may stabilize it

**Rationale**: WD=1e-2 alone catastrophically diverged on DrivAerML (grad norms 231x in #2790). However, on AirfRANS, WD=1e-2 + gc=1.0 was the golden config that enabled the phase transition. The key enabling condition may be gc. On DrivAerML, gc=0.5 (H5 above) may stabilize WD=1e-2 that would otherwise diverge. Testing WD=1e-2 + gc=0.5 at 4L/512d directly tests whether the AirfRANS golden config compound (WD+gc) transfers when both components are present simultaneously. This is higher risk than H5 but higher reward if it works.

**Exact command**:
```bash
cd target/icml2026
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset drivaerml \
  --model-layers 4 --model-hidden-dim 512 --model-heads 8 \
  --optimizer adamw --lr 3e-4 \
  --cosine-t-max 30 \
  --no-use-ema \
  --enable-fourier \
  --weight-decay 1e-2 \
  --grad-clip 0.5 \
  --wandb-group radford-drivaerml-wd1e2-gc05 \
  --wandb-name drivaerml-4l512d-wd1e2-gc0.5
```

Note: Lower lr to 3e-4 (from 5e-4) as extra insurance against divergence. If this run diverges within the first epoch, close it immediately — the WD+gc compound does not transfer.

**Expected outcome**: If stable, val_primary/surface_rel_l2_pct below 4.619%, potentially 3.5–4.0% if the full golden-config compound works. If it diverges early, that closes this direction and confirms DrivAerML needs WD kept at 1e-4.

---

## Ranked summary for 7 idle students

| Rank | Hypothesis | Dataset | Expected gain | Risk |
|---|---|---|---|---|
| 1 | H1: TandemFoil 3L/256d at lr=1.5e-4 | TandemFoil | Large (potentially 30-40% over 44.72) | Low (direct architecture transfer) |
| 2 | H4: AirfRANS 3L/256d + gc=0.5 | AirfRANS | Large (potentially new SOTA below 0.001479) | Low (two validated win components) |
| 3 | H2: DrivAerML 3L/256d | DrivAerML | Large (4.619% → potentially ~3.5%) | Medium (architecture transfer from 2D to 3D) |
| 4 | H3: TandemFoil LR sweep lr=1e-4 and lr=8e-5 | TandemFoil | Medium (44.72 → potentially ~42) | Low (monotone trend continuation) |
| 5 | H5: DrivAerML gc=0.5 at 4L/512d | DrivAerML | Medium (4.619% → potentially ~4.0%) | Low (isolated variable, stable config) |
| 6 | H12: TandemFoil gc=0.5 at 3L/192d | TandemFoil | Medium (44.72 → potentially ~42) | Low (isolated gc change) |
| 7 | H9: AirfRANS 3L/256d + T_max=5 extended 180-min | AirfRANS | Medium (pushing current SOTA further) | Medium (phase transition timing) |

Secondary ideas for when the top-7 slots free up: H6 (AirfRANS vol push), H7 (TandemFoil WD sweep), H8 (DrivAerML Lion), H10 (cross-benchmark pressure-weighted loss), H11 (AirfRANS WD confirm at 3L/256d), H13 (SGDR), H14 (DrivAerML WD+gc compound).
