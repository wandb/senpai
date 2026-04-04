# SENPAI Research Results

## Phase 6 Experiments (2026-04-01 onwards)

### 2026-04-04 10:30 — PR #2109: Phase 6: Contrastive Tandem-Single Regularization — tanjiro — CLOSED (negative, hypothesis falsified)
- Branch: `tanjiro/contrastive-tandem-regularization`
- Hypothesis: Tandem and single-foil samples share the same hidden representations in the Transolver. Adding a contrastive loss to push apart mean hidden states (cosine similarity) should force distinct internal representations and improve p_tan.
- W&B group: `phase6/contrastive-tandem`

| Config | Seed | p_in | p_oodc | p_tan | p_re | cos_sim | W&B Run |
|--------|------|------|--------|-------|------|---------|---------|
| Baseline | 42 | 13.3 | 7.7 | 30.3 | 6.5 | — | tqlbfz9y |
| Baseline | 73 | 12.9 | 8.1 | 30.5 | 6.6 | — | dck4ur8w |
| w=0.01 | 42 | 13.3 | 7.8 | 30.3 | 6.4 | 0.74 | ftrzalka |
| w=0.01 | 73 | 13.1 | 7.8 | 30.0 | 6.4 | 0.60 | fceuhys3 |
| w=0.05 | 42 | 12.5 | 7.7 | 30.6 | 6.5 | 0.62 | pfm9qsaj |
| w=0.05 | 73 | 13.3 | 8.2 | 30.3 | 6.6 | 0.30 | iyi2npzd |
| w=0.1 | 42 | 12.9 | 7.7 | 31.0 | 6.3 | 0.70 | opnlewzj |
| w=0.1 | 73 | 12.7 | 7.9 | 30.3 | 6.6 | 0.21 | atize8g5 |

**Analysis:** Contrastive mean p_tan=30.4 vs baseline mean 30.4 — no improvement. Higher weights made p_tan worse (w=0.1 avg: 30.65). Cosine similarity was successfully reduced (0.21-0.74 from ~0.9+) but didn't translate to better predictions. **Hypothesis falsified: representational entanglement between tandem and single-foil is NOT the p_tan bottleneck.** The model already distinguishes regimes internally; forcing stronger separation doesn't help because the difficulty is in the tandem physics itself (wake interactions, gap/stagger sensitivity), not in regime confusion. Centroid-based cosine loss is also a weak formulation (1 scalar gradient per batch). Interesting but inconsistent p_in improvement at w=0.05-s42 (12.5 vs 13.3) suggests mild regularization benefit at low weights.

### 2026-04-04 10:30 — PR #2107: Phase 6: Aft-Foil Coordinate Frame Normalization — frieren — SENT BACK (promising direction, needs non-destructive implementation)
- Branch: `frieren/aft-foil-local-frame`
- Hypothesis: Subtracting the aft-foil centroid from its coordinates before embedding gives the model a position-invariant view of the aft foil, decoupling shape from global position (gap/stagger).
- W&B group: `phase6/aft-foil-local-frame`

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B Run |
|--------|------|------|--------|-------|------|---------|
| 8-seed mean | — | 13.03 | 7.83 | 30.29 | 6.45 | — |
| Local frame | 42 | 13.83 | 8.20 | 30.17 | 6.59 | 00lod6uk |
| Local frame | 73 | 13.55 | 8.16 | 29.51 | 6.62 | 3llpj5yj |

**Analysis:** p_tan improved as hypothesized — s73 hit 29.51 (-2.6% vs mean), confirming the tandem OOD has a coordinate-frame representation component. However, p_in regressed severely (13.55-13.83 vs 12.2-12.71 refs) because in-place coordinate modification destroys positional information needed for single-foil predictions. **Sent back with instructions to add local-frame coords as ADDITIONAL sideband features (non-destructive), preserving global coords.** This should retain the p_tan benefit while avoiding p_in regression.

### 2026-04-04 10:00 — PR #2108: Phase 6: Asymmetric Fixed Asinh Scales (Pos/Neg Pressure) — edward — CLOSED (negative)
- Branch: `edward/asymmetric-asinh-scales`
- Hypothesis: Separate fixed (non-learnable) asinh scales for positive vs negative pressure should improve over the symmetric s=0.75, since suction peaks (-5 to -10 Cp) need stronger compression than stagnation pressures.
- W&B group: `phase6/asymmetric-asinh`

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B Run |
|--------|------|------|--------|-------|------|---------|
| Baseline (symmetric 0.75) | mean | 13.03 | 7.83 | 30.29 | 6.45 | — |
| Strong (neg=1.5, pos=0.5) | 42 | 13.6 | 8.3 | 30.8 | 6.5 | 7uz2d0ol |
| Strong (neg=1.5, pos=0.5) | 73 | 13.2 | 8.2 | 30.6 | 6.7 | 1v86awoi |
| Moderate (neg=1.0, pos=0.75) | 42 | 13.4 | 7.9 | 30.6 | 6.5 | dj0elhj2 |
| Moderate (neg=1.0, pos=0.75) | 73 | 13.1 | 8.1 | 30.2 | 6.5 | rgs2r8jj |

**Analysis:** Both asymmetric variants worse than symmetric baseline on ALL metrics. Strong asymmetry: p_oodc +5.4%, p_in +2.8%. Moderate asymmetry: p_oodc +2.2%, p_in +1.7%. The symmetric asinh(p×0.75) is already well-tuned — breaking sign symmetry introduces distributional mismatch at the zero-crossing boundary. The z-score normalization applied on top of asinh already handles scale differences between suction and stagnation regions. **Combined with learnable scale (#2096) and asinh velocity (#2098), this closes the asinh transform direction entirely.**

Code note: Student refactored 14 inline asinh transform sites into 2 helper functions (`_asinh_fwd`, `_asinh_inv`) — good code quality improvement but not mergeable without a metric win.

### 2026-04-04 07:50 — PR #2103: Phase 6: Iterative Weight-Tied Transolver — thorfinn — CLOSED (negative, key insight)
- Branch: `thorfinn/iterative-weight-tied-transolver`
- Hypothesis: Weight-tied iterations (K passes through a single shared block) give effective depth without more parameters — inspired by DEQ (Bai et al., NeurIPS 2019).
- W&B group: `phase6/iterative-transolver`

| Config | p_in | p_oodc | p_tan | p_re | Epochs | W&B Runs |
|--------|------|--------|-------|------|--------|----------|
| Baseline (3 distinct) | 12.9 | 7.8 | 30.1 | 6.5 | 156-157 | onmaryjt, xhpmloz0 |
| n_iter=2 (shared) | 13.7 (+6%) | 8.0 (+2%) | 31.1 (+3%) | 6.6 (+2%) | 156 | i8xsa2g0, 9ny3ctv2 |
| n_iter=3 | 15.4 (+19%) | 8.9 (+14%) | 31.4 (+4%) | 7.1 (+9%) | 127-128 | 55m69afg, i6ucjvki |
| n_iter=4 | 18.5 (+43%) | 10.5 (+34%) | 31.9 (+6%) | 8.0 (+23%) | 107 | i048aror, 1dxezufa |

**Analysis:** Monotonic degradation with iteration count. **Key insight from n_iter=2:** Even at identical computational depth and training epochs, weight sharing hurts 2-6%. The two intermediate blocks learn *complementary* (not repetitive) representations — the "iterative solver" analogy from DEQ doesn't hold here. Combined with #2100 (scale-up): the 3-block architecture is the right structural choice — distinct blocks matter, but more blocks don't help. **Confirmed dead end: weight-tied iteration is wrong for Transolver.**

### 2026-04-04 07:25 — PR #2102: Phase 6: Sin Activation in SRF Head — alphonse — CLOSED (dead end)
- Branch: `alphonse/sin-activation-srf-head`
- Hypothesis: SIREN (sinusoidal activation) in the surface refinement head should capture oscillatory Cp distributions better than GELU.
- W&B group: `siren-surf-head`

| Config | p_in | p_oodc | p_tan | p_re | W&B Runs |
|--------|------|--------|-------|------|----------|
| Baseline (2-seed) | 13.2 | 7.95 | 30.1 | 6.4 | jijyca9m, 6vf7ts82 |
| SIREN w=1.0 | 13.1 | 7.9 | 30.25 | 6.45 | tl6loy2k, 5x24h9o2 |
| SIREN w=10.0 | 13.3 | 7.95 | 31.25 | 6.35 | h9dsjz11, meqkfczf |
| SIREN w=30.0 | 13.05 | 8.2 | 32.75 | 6.5 | hz86txm2, smw15svc |

**Analysis:** Monotonic degradation with omega — higher SIREN frequency → worse p_tan (30.25 → 31.25 → 32.75). Even w=1.0 (mildest) shows no improvement anywhere. The srf_head's residual corrections don't have the oscillatory structure that SIREN is optimized for — GELU is well-calibrated for this task. **Confirmed dead end: sinusoidal activations don't help surface refinement.**

### 2026-04-04 07:00 — PR #2097: Phase 6: Deep Supervision — nezuko — CLOSED (mixed: p_in improved, p_tan regressed)
- Branch: `nezuko/deep-supervision`
- Hypothesis: Auxiliary loss on pre-final-block hidden features provides direct gradient flow to intermediate blocks, improving representation quality and OOD generalization.
- W&B group: `phase6/deep-supervision-8seed`

**8-seed validation (aux_w=0.2, seeds 42-49):**

| Seed | p_in | p_oodc | p_tan | p_re | W&B Run |
|------|------|--------|-------|------|---------|
| 42 | 12.3 | 7.9 | 31.8 | 6.4 | k1v8bvum |
| 43 | 12.8 | 7.7 | 30.2 | 6.4 | 11yf2z5i |
| 44 | 12.9 | 8.1 | 31.1 | 6.5 | 2qwd1irz |
| 45 | 12.5 | 7.8 | 31.9 | 6.6 | x97wrsdz |
| 46 | 12.8 | 7.7 | 30.7 | 6.3 | e1l5eokg |
| 47 | 13.2 | 8.3 | 31.9 | 6.6 | xpfnr80l |
| 48 | 13.1 | 7.7 | 30.4 | 6.7 | ds46hwpq |
| 49 | 12.9 | 8.1 | 30.8 | 6.4 | ocxi1i4c |

| Metric | Deep Sup (8-seed) | Baseline (8-seed) | Delta |
|--------|------------------|-------------------|-------|
| p_in | **12.81 ± 0.29** | 13.03 ± 0.39 | **-1.7%** |
| p_oodc | 7.91 ± 0.23 | **7.83 ± 0.19** | +1.1% |
| p_tan | 31.10 ± 0.69 | **30.29 ± 0.47** | **+2.7%** |
| p_re | 6.49 ± 0.14 | **6.45 ± 0.05** | +0.6% |

**Analysis:** Initial 2-seed result (p_oodc -3.1%) was optimistic — 8-seed shows p_oodc +1.1%. **p_in genuinely improves -1.7%** (the auxiliary gradient signal helps in-distribution feature quality). But **p_tan regresses +2.7%** — the auxiliary loss constrains representational flexibility needed for tandem geometry transfer. Since p_tan is our primary target, this tradeoff is unacceptable. Demonstrates the tension between in-distribution fitting and OOD generalization in this architecture. **Closed: p_tan regression overrides p_in improvement.**

### 2026-04-04 06:50 — PR #2101: Phase 6: OHEM — tanjiro — CLOSED (negative)
- Branch: `tanjiro/ohem-hard-example-mining`
- Hypothesis: Per-sample EMA loss tracking + upweighting of persistently hard samples. Should improve p_tan by focusing gradient budget on difficult tandem configs.
- W&B group: `phase6/ohem`

| Config | p_in | p_oodc | p_tan | p_re | W&B Run |
|--------|------|--------|-------|------|---------|
| Baseline s42 | 13.3 | 7.7 | 30.3 | 6.5 | tqlbfz9y |
| Baseline s73 | 12.9 | 8.1 | 30.5 | 6.6 | dck4ur8w |
| OHEM w=1.5 p75 s42 | 14.3 | 7.9 | 30.3 | 6.5 | 4x372o82 |
| OHEM w=1.5 p75 s66 | 13.4 | 8.0 | 30.7 | 6.6 | c8v1a5l0 |
| OHEM w=1.5 p75 s73 | 13.3 | 8.0 | 30.9 | 6.6 | 6db353lp |
| OHEM w=2.0 p75 s42 | 13.4 | 7.9 | 30.9 | 6.5 | nzdg07dy |
| OHEM w=2.0 p75 s73 | 13.5 | 8.2 | 31.3 | 6.5 | 0k97472w |
| OHEM w=1.5 p90 s42 | 14.4 | 7.9 | 30.4 | 6.4 | dng4d3lt |

**Analysis:** OHEM provides no improvement — all configs are at or below baseline. Stronger weight (w=2.0) actively hurts: p_tan +2.3%, p_in +3.1%. Root cause: OHEM compounds with existing tandem_ramp and adaptive_boost mechanisms (3 layers of reweighting is too much). The EMA correctly tracks hard samples (2.7× loss ratio) but existing mechanisms already handle this. **Confirmed: sample-level reweighting is not the p_tan bottleneck.**

### 2026-04-04 06:20 — PR #2100: Phase 6: Model Scale-Up — frieren — CLOSED (negative, key insight)
- Branch: `frieren/model-scale-up`
- Hypothesis: 3L/96s Transolver uses only 38/96GB VRAM. Deeper (5L) or wider (160s) models may improve metrics, especially p_tan which could be capacity-limited.
- W&B group: `phase6/model-scale-up`

| Config | p_in | p_oodc | p_tan | p_re | Epochs | VRAM | W&B Run |
|--------|------|--------|-------|------|--------|------|---------|
| 3L/96s s42 (BL) | 13.0 | 7.6 | 31.1 | 6.4 | 157 | 38GB | u00xzh8z |
| 3L/96s s73 (BL) | 12.9 | 7.9 | 30.3 | 6.4 | 157 | 38GB | 3y9kazhf |
| 5L/96s s42 | 19.0 | 10.4 | 31.5 | 8.6 | 105 | 55GB | z8xu1h7q |
| 5L/96s s73 | 16.3 | 10.0 | 30.4 | 8.2 | 106 | 55GB | u1zdhi63 |
| 3L/160s s42 | 13.7 | 7.9 | 30.8 | 6.5 | 141 | 41GB | mkxyc0mf |
| 3L/160s s73 | 13.0 | 8.3 | 30.2 | 6.5 | 142 | 41GB | fzabfqmz |
| 4L/128s s42 | 15.9 | 9.4 | 32.0 | 7.5 | 119 | 48GB | i4o4g2qm |
| 4L/128s s73 | 15.6 | 9.7 | 31.5 | 7.8 | 118 | 48GB | 910al9e9 |

**Analysis:** No configuration beats baseline. 5L/96s is catastrophically worse (+36% p_in) though undertrained (105/157 epochs). 3L/160s (wider) is closest but still +3% p_in after 142 epochs. **Critical finding: p_tan is remarkably similar across ALL configs (~30-32).** This proves the tandem transfer problem is NOT capacity-limited — it's a representation/distribution problem. Model scale-up is not the answer. The 3L/96s architecture is the right size for this task at the current training budget.

### 2026-04-04 06:20 — PR #2094: Phase 6: SWAD — edward — CLOSED (catastrophic failure)
- Branch: `edward/swad-averaging`
- Hypothesis: SWAD (Cha et al., NeurIPS 2021) averages checkpoints densely from a flat loss region. Already implemented in train.py but untested. Bug fix applied: SWAD now also averages the refine_head.
- W&B group: `phase6/swad-averaging`

| Config | Seed | p_in | p_oodc | p_tan | p_re | W&B Run |
|--------|------|------|--------|-------|------|---------|
| Baseline | 42 | 13.2 | 8.0 | 29.7 | 6.4 | 3mim1mhi |
| Baseline | 43 | 12.8 | 8.2 | 30.3 | 6.4 | gk16mcse |
| SWAD | 42 | 49.4 | 37.2 | 57.2 | 26.9 | 1ee50z25 |
| SWAD | 43 | 54.2 | 41.2 | 64.9 | 29.4 | hbm6rfcg |
| SWAD | 44 | 46.5 | 38.9 | 55.4 | 27.9 | hsvhokae |
| SWAD | 45 | 43.9 | 35.0 | 55.3 | 24.6 | r632qi5f |
| SWAD | 66 | 45.8 | 36.3 | 58.3 | 26.3 | 86sd67n7 |
| SWAD | 67 | 47.9 | 39.6 | 58.9 | 31.5 | tm0513wp |

**Analysis:** Catastrophic failure: SWAD is +268% p_in, +369% p_oodc vs baseline. Root cause: SWAD suppresses EMA updates (line 1805) but requires intermediate val evaluations to trigger checkpoint collection. Our eval-at-end training loop means SWAD never activates → model trains without EMA → raw weights are dramatically worse. **Key insight: EMA is load-bearing. Any technique that disables EMA without equivalent smoothing will fail catastrophically.**

### 2026-04-04 06:00 — PR #2099: Phase 6: Stochastic Depth (DropPath) — askeladd — CLOSED (dead end)
- Branch: `askeladd/stochastic-depth-droppath`
- Hypothesis: DropPath randomly skips entire residual branches during training, creating an implicit ensemble of subnetworks. Forces more geometry-agnostic representations that might improve OOD generalization, especially p_tan.
- W&B group: `phase6/stochastic-depth`

| Config | Seed | p_in | p_oodc | p_tan | p_re | val/loss | W&B Run |
|--------|------|------|--------|-------|------|----------|---------|
| DropPath 0.1 | 42 | 13.32 | 8.19 | 30.74 | 6.63 | 0.3965 | s93o0li2 |
| DropPath 0.1 | 73 | 13.65 | 8.86 | 31.25 | 6.65 | 0.4037 | fovffeeg |
| DropPath 0.2 | 42 | 13.57 | 8.37 | 31.50 | 6.93 | 0.4110 | 6dt5ofgn |
| DropPath 0.2 | 73 | 13.03 | 8.81 | 30.93 | 6.82 | 0.4033 | lqq102pu |

**Baseline (8-seed mean):** p_in=13.03, p_oodc=7.83, p_tan=30.29, p_re=6.45

| Config | p_in | p_oodc | p_tan | p_re |
|--------|------|--------|-------|------|
| Baseline (8-seed mean) | 13.03 | 7.83 | 30.29 | 6.45 |
| DropPath 0.1 (2-seed mean) | 13.50 (+3.6%) | 8.55 (+9.2%) | 31.00 (+2.3%) | 6.65 (+3.1%) |
| DropPath 0.2 (2-seed mean) | 13.30 (+2.1%) | 8.60 (+9.8%) | 31.20 (+3.0%) | 6.85 (+6.2%) |

**Analysis:** All DropPath runs are significantly worse than baseline. p_oodc is the worst-hit metric (+9-10%). Metrics verified against W&B — student's reported numbers accurate to within rounding. Root cause: Transolver has only 3 blocks — far too shallow for stochastic depth to provide meaningful path diversity. DropPath works in 12-24 layer ViTs; with 3 layers and a linear schedule, only the last block gets meaningful drop probability, simply removing compute the model needs. **Confirmed dead end: stochastic depth is not suitable for this 3-layer architecture. Revisit only if model grows to 6+ blocks.**

### 2026-04-04 05:00 — PR #2090: Phase 6: Knowledge Distillation — fern — CLOSED (clean negative result)
- Branch: `fern/knowledge-distillation`
- Hypothesis: Train a single model using the 8-seed ensemble's predictions as soft targets (offline KD). Distilled model should approach ensemble quality at 1/8th inference cost.
- W&B group: `phase6/knowledge-distill`

**v2 results (full training ~157 epochs):**

| Config | Seed | p_in | p_tan | p_oodc | p_re | val/loss | W&B run |
|--------|------|------|-------|--------|------|----------|---------|
| Baseline | 42 | 13.2 | 31.4 | 7.7 | 6.3 | 0.3908 | thssa2ru |
| Baseline | 43 | 13.3 | 30.9 | 7.6 | 6.6 | 0.3891 | 4lj8ch8s |
| **Baseline avg** | | **13.2** | **31.2** | **7.7** | **6.5** | **0.3900** | |
| Distill α=0.6 | 42 | 14.4 | 31.6 | 8.0 | 6.6 | 0.3978 | qq24kqqv |
| Distill α=0.6 | 43 | 14.3 | 32.2 | 8.3 | 6.8 | 0.4010 | o2ir8mc6 |
| **α=0.6 avg** | | **14.4** | **31.9** | **8.2** | **6.7** | **0.3994** | |
| Distill α=0.7 | 42 | 14.1 | 30.8 | 7.7 | 6.5 | 0.3917 | fjqz35fa |
| Distill α=0.7 | 43 | 13.6 | 30.3 | 8.0 | 6.5 | 0.3906 | m5smb3w6 |
| **α=0.7 avg** | | **13.9** | **30.6** | **7.9** | **6.5** | **0.3912** | |
| Distill α=0.8 | 42 | 13.2 | 30.5 | 7.8 | 6.5 | 0.3889 | whoha5za |
| Distill α=0.8 | 43 | 13.2 | 30.4 | 7.8 | 6.6 | 0.3908 | j2pc23lk |
| **α=0.8 avg** | | **13.2** | **30.5** | **7.8** | **6.6** | **0.3899** | |

**Analysis:** Knowledge distillation provides no meaningful improvement at convergence. α=0.8 (best) is statistically tied with baseline (val/loss 0.3899 vs 0.3900). α=0.7 shows marginal p_tan improvement (-2%) but p_in regression (+5%). α=0.6 is uniformly worse (+2-9%). Root causes: (1) ensemble's advantage over single models is too small (~1 Pa on p_oodc) for soft target signal to reliably capture; (2) at convergence the student already fits ground truth well — soft targets add noise rather than signal; (3) loss competition between hard and soft targets degrades metrics where ensemble has no edge (p_in, velocity). **Confirmed dead end: offline regression KD from seed-diverse ensembles does not help when base model is well-tuned.**

### 2026-04-04 04:15 — PR #2096: Phase 6: Learnable Asinh Scale — thorfinn — CLOSED (dead end)
- Branch: `thorfinn/learnable-asinh-scale`
- Hypothesis: Making asinh_scale a learnable nn.Parameter (init 0.75) allows gradient descent to find optimal compression jointly with model weights. Extension: asymmetric with separate pos/neg scales.
- W&B group: `phase6/learnable-asinh`

| Config | Seed | p_in | p_oodc | p_tan | p_re | val/loss | Final Scale | W&B Run ID |
|--------|------|------|--------|-------|------|----------|-------------|------------|
| Baseline (fixed 0.75) | 42 | 12.8 | 7.7 | 30.1 | 6.4 | 0.386 | 0.75 | zwyyvhsb |
| Baseline (fixed 0.75) | 43 | 13.3 | 7.9 | 29.8 | 6.6 | 0.387 | 0.75 | npsbhjhg |
| Learnable (init 0.75) | 42 | 25.0 | 21.0 | 35.6 | 16.9 | 0.081 | 0.0028 | 6noqttqf |
| Learnable (init 0.75) | 43 | 24.7 | 19.5 | 34.2 | 16.4 | 0.081 | 0.0036 | 967i5akk |
| Learnable (init 0.75) | 44 | 27.1 | 23.3 | 36.7 | 19.0 | 0.082 | 0.0033 | 16jzy7gf |
| Learnable (init 0.75) | 45 | 27.5 | 22.0 | 35.6 | 18.6 | 0.081 | 0.0037 | crruke85 |
| Asymmetric (init 0.75) | 42 | 20.0 | 15.1 | 33.1 | 12.4 | 0.082 | pos=0.0063, neg=0.0047 | wbrgp3oh |
| Asymmetric (init 0.75) | 43 | 19.3 | 14.5 | 31.6 | 11.8 | 0.084 | pos=0.0097, neg=0.0069 | yylvcbff |

- **Root cause: Scale collapse.** All learnable runs: scale drops from 0.75 → ~0.003 within 40 epochs. Model minimizes loss by compressing target space to near-zero (trivial shortcut). val/loss appears lower (0.081 vs 0.386) but is in compressed space — physical metrics 2x worse.
- Asymmetric variant slightly less catastrophic (pos scale collapses slower) but still 50-90% regression.
- **Conclusion:** Learnable target-space transforms create optimization shortcuts. Fixed s=0.75 from grid search is near-optimal. CLOSED.

### 2026-04-04 04:15 — PR #2098: Phase 6: Asinh Velocity Transform — alphonse — CLOSED (negative result)
- Branch: `alphonse/asinh-velocity`
- Hypothesis: Applying asinh compression to velocity channels (Ux, Uy) reduces dynamic range, same argument as pressure asinh. Test scales 0.3, 0.5, 1.0.
- W&B group: `phase6/asinh-velocity`

| Config | Seed | p_in | p_oodc | p_tan | p_re | val/loss | W&B Run ID |
|--------|------|------|--------|-------|------|----------|------------|
| Baseline | 42 | 13.5 | 8.1 | 30.4 | 6.4 | 0.3864 | p7iy87v0 |
| Baseline | 43 | 12.9 | 7.8 | 29.8 | 6.4 | 0.3844 | hwzd5zqm |
| asinh-0.3 | 42 | 13.6 | 9.0 | 31.2 | 6.8 | 0.4141 | 2joautz7 |
| asinh-0.3 | 43 | 13.4 | 8.4 | 31.3 | 6.6 | 0.4047 | g5r2dbxt |
| asinh-0.5 | 42 | 13.3 | 8.0 | 32.5 | 6.5 | 0.4002 | 4my1b7qi |
| asinh-0.5 | 43 | 13.9 | 7.9 | 31.0 | 6.6 | 0.4003 | vye8pe7r |
| asinh-1.0 | 42 | 12.8 | 7.6 | 32.4 | 6.4 | 0.3923 | 793xhqav |
| asinh-1.0 | 43 | 13.5 | 7.9 | 33.9 | 6.3 | 0.4045 | x22wlmnl |

- **All scales hurt p_tan** (+1.15 to +3.05). Velocity doesn't have pressure's outlier problem — distributions are well-behaved after physics normalization. Compressing velocity gradients removes discriminative signal for tandem wake interactions.
- **Conclusion:** Velocity channels should stay in linear normalization. CLOSED.

### 2026-04-04 04:15 — PR #2097: Phase 6: Deep Supervision — nezuko — SENT BACK (promising, needs 8-seed validation)
- Branch: `nezuko/deep-supervision`
- Hypothesis: Auxiliary loss on pre-final-block hidden features (model_out["hidden"]) provides direct gradient flow to intermediate blocks, improving representation quality.
- W&B group: `phase6/deep-supervision`

| Config | Seed | p_in | p_oodc | p_tan | p_re | val/loss | W&B Run ID |
|--------|------|------|--------|-------|------|----------|------------|
| Baseline | 42 | 13.2 | 8.0 | 31.0 | 6.3 | 0.2706 | x8ftaw4c |
| Baseline | 43 | 12.9 | 7.9 | 29.9 | 6.4 | 0.2666 | m19kza4z |
| aux_w=0.05 | 42 | 13.0 | 7.7 | 30.3 | 6.4 | 0.2754 | olcje0ya |
| aux_w=0.05 | 43 | 12.9 | 7.8 | 30.2 | 6.6 | 0.2615 | pg6xxiba |
| aux_w=0.1 | 42 | 13.1 | 7.8 | 30.1 | 6.5 | 0.2693 | k5wnix71 |
| aux_w=0.1 | 43 | 12.7 | 7.9 | 31.4 | 6.4 | 0.2636 | 6rpwpnho |
| aux_w=0.2 | 42 | 13.1 | 7.7 | 30.3 | 6.3 | 0.2677 | 7kxqtn0g |
| aux_w=0.2 | 43 | 12.8 | 7.7 | 30.2 | 6.4 | 0.2575 | 7mpx83k7 |

- **aux_w=0.2 best overall:** p_oodc 7.70 (-1.7% vs 8-seed BL mean), p_re 6.35 (-1.6%), p_in 12.95 (-0.6%), p_tan 30.25 (-0.1%). No regressions.
- **Mechanism:** Direct gradient flow from surface targets to pre-final-block features improves OOD generalization.
- **SENT BACK** for 8-seed validation of aux_w=0.2 (seeds 42-49). If confirmed, this is a merge.

### 2026-04-04 04:00 — PR #2093: Phase 6: 16-Seed Combined Ensemble Evaluation — MERGED (winner)

- Branch: `tanjiro/16-seed-ensemble-eval`
- Hypothesis: Combining 8-seed ensembles (42-49 + 66-73) into a 16-seed ensemble reduces variance by 1/√N ≈ 29% vs 8-seed, yielding lower surface MAE.
- W&B groups: `phase6/retrain-seeds-42-49`, `phase6/ensemble-seeds-100-106`

| Ensemble Size | p_in | p_oodc | p_tan | p_re |
|---------------|------|--------|-------|------|
| 4-seed (42-45) | 12.7 | 6.8 | 29.6 | 6.0 |
| 8-seed (42-49) | 12.3 | 6.7 | 29.2 | 5.9 |
| 12-seed | 12.2 | 6.7 | 29.1 | 5.8 |
| **16-seed (all)** | **12.1** | **6.6** | **29.1** | **5.8** |
| *Prior 8-ens baseline* | *12.2* | *6.7* | *29.1* | *5.8* |

Seeds 100-106 individually (W&B group: `phase6/ensemble-seeds-100-106`):

| Seed | Run ID | p_in | p_oodc | p_tan | p_re |
|------|--------|------|--------|-------|------|
| 100 | 9o85duyc | 13.4 | 7.9 | 29.8 | 6.3 |
| 101 | ec7plfg8 | 12.8 | 7.8 | 30.3 | 6.5 |
| 102 | zagg4pfs | 12.4 | 7.9 | 30.6 | 6.4 |
| 103 | 6w86plz1 | 12.5 | 7.7 | 31.3 | 6.4 |
| 104 | g00kxdva | 13.4 | 7.9 | 30.2 | 6.4 |
| 105 | jt9hwf40 | 13.9 | 7.9 | 31.4 | 6.6 |
| 106 | fom4bzro | 12.8 | 7.9 | 31.6 | 6.5 |
| **Mean** | | **13.0** | **7.9** | **30.7** | **6.4** |

Run IDs (seeds 42-49 re-trained): f59v5aul, 0yurebjv, rdezx8es, ds12ug79, yu1x0dy0, y147zvh1, lc5cbt4l, 7cxu38oh

**Analysis:** 16-seed ensemble beats the 8-seed baseline on p_in (-0.8%) and p_oodc (-1.5%). Monotonic improvement 4→8→12→16 seeds confirms 1/√N scaling law with no "degrading" individual models. Diminishing returns are clear: 8→16 gains much less than 4→8. Seeds 100-106 show consistent individual performance with prior seed batches (mean p_in=13.0 vs pool mean ~13.0), all hitting the 180-min wall clock. The 16-seed ensemble is now the new baseline. An additional 23-seed evaluation (adding 100-106) would yield another marginal improvement but inference cost scales accordingly.

**MERGED** — New baseline: p_in=12.1, p_oodc=6.6, p_tan=29.1, p_re=5.8

### Completed (2026-04-04 ~03:00 UTC)

#### 2026-04-04 03:00 — PR #2086: Phase 6: SAM Phase-Only — frieren — CLOSED (dead end)
- Branch: `frieren/sam-phase-only`
- Hypothesis: Sharpness-Aware Minimization in final 25% of training (epoch 120+) would find flatter minima, reducing seed variance and improving OOD generalization
- W&B group: `phase6/sam-phase-only`

| Config | Seed | p_in | p_oodc | p_tan | p_re | val/loss | Best Epoch |
|--------|------|------|--------|-------|------|----------|------------|
| Baseline | 42 | 13.0 | 7.6 | 31.1 | 6.4 | 0.3915 | 157 |
| Baseline | 43 | 13.2 | 8.0 | 30.0 | 6.5 | 0.3878 | 157 |
| SAM rho=0.05 | 42 | 15.4 | 8.9 | 32.4 | 7.4 | 0.4283 | 120† |
| SAM rho=0.05 | 43 | 15.2 | 10.0 | 32.0 | 8.0 | 0.4284 | 120† |
| SAM rho=0.1 | 42 | 15.4 | 8.9 | 32.4 | 7.4 | 0.4283 | 120† |
| SAM rho=0.1 | 43 | 15.2 | 10.0 | 32.0 | 8.0 | 0.4284 | 120† |
| SAM rho=0.2 | 42 | 15.4 | 8.9 | 32.4 | 7.4 | 0.4283 | 120† |
| SAM rho=0.2 | 43 | 15.2 | 10.0 | 32.0 | 8.0 | 0.4284 | 120† |

†Best checkpoint is epoch 120 = last epoch BEFORE SAM activation. All same-seed SAM runs have identical metrics because SAM was never active in the best checkpoint.

- W&B run IDs: yd0lluxz, hsq5fn98, lihil5xz, uo6chve2, b634pzxq, b4bsxpv9, xewiwyx2, 75qvkprv
- Conclusion: **SAM is a clear dead end.** SAM destabilizes late-stage training with Lion optimizer. The best checkpoint for every SAM run was the pre-SAM epoch 120, meaning SAM activation actively degraded performance. Lion's sign-based updates already provide implicit regularization; SAM is redundant/conflicting. Three rho values (0.05, 0.1, 0.2) all failed identically.
- CLOSED

### New Assignments (2026-04-04 ~00:00 UTC)

#### 2026-04-04 — PR #2097: Phase 6: Multi-Scale Deep Supervision — nezuko — NEW
- Branch: `nezuko/deep-supervision`
- Hypothesis: Auxiliary loss on intermediate features (fx_deep) forces better representations in earlier blocks
- aux_loss_weight sweep: 0.05, 0.1, 0.2

#### 2026-04-04 — PR #2096: Phase 6: Learnable Asinh Scale — thorfinn — NEW
- Branch: `thorfinn/learnable-asinh-scale`
- Hypothesis: Learnable nn.Parameter asinh_scale (init 0.75) adapts compression per run
- Includes asymmetric variant (separate pos/neg scales)

### Assignments (2026-04-03 ~23:20 UTC)

#### 2026-04-04 — PR #2099: Phase 6: Stochastic Depth (DropPath) — askeladd — NEW
- Branch: `askeladd/stochastic-depth-droppath`
- Hypothesis: DropPath randomly skips Transolver block residuals during training, creating implicit ensemble of subnetworks. Proven OOD regularizer in ViT/DeiT. Target: p_tan improvement.
- Experiment: drop_path_rate={0.1, 0.2} × seeds {42, 73} = 4 runs

#### 2026-04-04 — PR #2095: Phase 6: SGDR Warm Restarts — askeladd — CLOSED (dead end)
- Branch: `askeladd/sgdr-warm-restarts`
- Hypothesis: Cosine annealing with warm restarts (T_0={20,40,60}, T_mult=2) for better OOD generalization
- Results: **ALL 8 RUNS WORSE THAN BASELINE.** SGDR warm restarts hurt OOD generalization.

| Config | p_in seed42 | p_oodc seed42 | p_in seed73 | p_oodc seed73 |
|--------|------------|---------------|------------|---------------|
| Baseline (cosine) | 12.71 | 8.13 | 13.33 | 7.59 |
| T_0=20 | 13.62 | 8.32 | 14.02 | 8.26 |
| T_0=40 | 13.89 | 8.47 | 13.34 | 8.35 |
| T_0=60 | 13.51 | 8.19 | 13.61 | 8.30 |

- Conclusion: LR restarts disrupt the stable descent into the flat basin that Lion+cosine finds naturally. SGDR is a confirmed dead end for this architecture.
- CLOSED

#### 2026-04-03 — PR #2094: Phase 6: SWAD Dense Weight Averaging — edward — NEW
- Branch: `edward/swad-averaging`
- Hypothesis: SWAD (NeurIPS 2021) averages checkpoints from validation-stable window for flatter minima
- Status: WIP — just assigned (edward idle after #2087 closed)
- Note: Requires refine_head bug fix (SWAD doesn't average surface refine head)

### Completed (2026-04-03 ~23:00 UTC)

#### 2026-04-03 — PR #2089: Phase 6: Ensemble Weight Optimization — askeladd — CLOSED (null result)
- Branch: `askeladd/ensemble-weight-opt`
- W&B group: phase6/ensemble-weight-opt (6 runs finished: seeds 90-95)
- Results: **WEIGHT OPTIMIZATION NULL RESULT.** SLSQP converged to equal weights (~0.1 each).
  - 10-model equal-weight ensemble: p_in=12.27, p_oodc=6.67, p_tan=29.08, p_re=5.83
  - Best 6-of-10 selective: p_in=12.07, p_oodc=6.76, p_tan=29.18, p_re=5.80
  - vs current baseline (8-seed, p_in=12.2): FLAT — no improvement
- Key insight: Same-architecture models don't benefit from non-uniform weighting. Need method diversity.
- Run IDs (seeds 90-95): ici6bxi1, 6chuzqal, xcsqiwdv, sxisuynb, q8m1w63d, ggn5mioe
- CLOSED — documented, seeds available for future ensemble expansion

#### 2026-04-03 — PR #2091: Phase 6: Diverse Hyperparameter Ensemble — nezuko — CLOSED (informative negative)
- Branch: `nezuko/diverse-hparam-ensemble`
- W&B group: phase6/diverse-hparam-ensemble (8 runs finished)
- Results: **Hyperparameter diversity LOSES to seed diversity.**
  - Diverse 8-model ensemble: p_in=12.3, p_oodc=7.0, p_tan=29.8, p_re=5.9
  - Same-config 8-seed baseline: p_in=12.4, p_oodc=6.7, p_tan=29.4, p_re=5.8
  - p_oodc +4.5%, p_tan +1.4%, p_re +1.7% ALL WORSE
- Key insight: Different hyperparams make systematic tradeoffs, not complementary errors. Seed diversity strictly better.
- Run IDs: 369n5uv8, i8xwj9zn, 2bht89ay, wrbmgygu, qmsobmzv, cofd0trk, 80aq4s2f, o4dh75fa
- CLOSED

#### 2026-04-03 — PR #2092: Phase 6: Ensemble Seeds 82-89 — thorfinn — CLOSED (complete)
- Branch: `thorfinn/ensemble-seeds-82-89`
- W&B group: phase6/ensemble-seeds-82-89 (8 runs finished)
- Results: Individual model metrics consistent with baseline:
  - Mean: p_in=13.05±0.29, p_oodc=7.89±0.28, p_tan=30.30±0.42, p_re=6.44±0.10
  - Notable: s86 p_oodc=7.44, s88 p_re=6.18
- Run IDs: u0eapina, fmhetijo, yp7dlkmk, 30hxo8a1, 4e74gtuc, wc8x0v49, qvn871e1, nb6poqj2
- CLOSED — seeds trained, available for combined ensemble evaluation

#### 2026-04-03 — PR #2086: Phase 6: SAM Phase-Only — frieren — SENT BACK
- 3 rounds of failures: (1) infrastructure crash at 127min, (2) code crash at 4min, (3) timeout at 30min
- Root cause: student applied SENPAI_TIMEOUT_MINUTES to train.py MAX_TIMEOUT
- Bug fixes preserved: SAM backward graph fix, configurable sam_rho, sam_start_frac
- SENT BACK — must revert MAX_TIMEOUT override and rerun at full 180 min

#### 2026-04-03 — PR #2087: Phase 6: Ensemble Seeds 74-81 — edward — CLOSED (complete)
- Branch: `edward/ensemble-seeds-74-81`
- W&B group: phase6/ensemble-seeds-74-81 (8 runs finished)
- Results: Individual model metrics consistent with baseline:
  - Mean: p_in=13.11±0.45, p_oodc=7.97±0.17, p_tan=30.52±0.53, p_re=6.50±0.11
- Run IDs: 2sre8vzp, ue8pmbbr, hgyim25m, e2obsfn1, 555102xo, 2lpzf6go, ibsrx1t8, a6e89sx4
- CLOSED — seeds trained, available for combined ensemble evaluation

### Prior Assignments (2026-04-03)

#### 2026-04-03 — PR #2092: Phase 6: Ensemble Seeds 82-89 — thorfinn — RUNNING
- Branch: `thorfinn/ensemble-seeds-82-89`
- Hypothesis: More standard 3L seeds for ensemble expansion (total pool: 42-49, 66-73, 74-81, 82-89, 90-95)
- Status: WIP — just assigned

#### 2026-04-03 — PR #2080: Phase 6: Ensemble Seeds 66-73 — tanjiro — **MERGED WINNER**
- Branch: `tanjiro/ensemble-more-seeds`
- W&B group: phase6/ensemble-more-seeds (8 runs finished)
- Results: **NEW BEST.** 8-seed ensemble (66-73) beats prior best (42-49):
  - p_in: 12.2 (was 12.4, **-1.6%**)
  - p_oodc: 6.7 (flat)
  - p_tan: 29.1 (was 29.4, **-1.0%**)
  - p_re: 5.8 (flat)
- Run IDs: j9w7d1r7, mc4jvgqj, cbbvhl62, bigqfn3k, bqhg6lq8, 5ukk7wv6, xlnhwuqc, ii1tz4vv
- **MERGED** — Updated baseline. Next: 16-seed combined evaluation.

#### 2026-04-03 — PR #2082: Phase 6: Packed Ensemble — thorfinn — CLOSED
- Branch: `thorfinn/packed-ensemble`
- W&B group: phase6/packed-ensemble (8 runs finished)
- Results: **DEAD END.** Model too small for packed ensembles.
  - M=2: marginal, ~flat vs baseline
  - M=4: p_re +6.4%, p_oodc +2.2% worse
  - M=8: p_re +7.1%, p_oodc +3.6% worse
- Root cause: n_hidden=192 split across M sub-models → insufficient per-sub-model capacity
- Conclusion: At 1.7M params, separate model training + post-hoc averaging strictly dominates. CLOSED.

#### 2026-04-03 — PR #2068: Phase 6: Asymmetric/Magnitude-Weighted Surface Loss — alphonse — SENT BACK
- Branch: `alphonse/asymmetric-loss`
- W&B group: phase6/asymmetric-loss (8 runs finished)
- Results: Mixed tradeoff. magweight α=1.0 improves p_tan -4.3% but worsens p_in +3.7%, p_oodc +3.8%.
  - α=0.5: p_tan -2.0%, other metrics +0.5-1.5% worse
  - α=1.0: p_tan -4.3%, p_in +3.7%, p_oodc +3.8% (tradeoff too steep)
  - pinball τ=0.45: p_tan -2.1%, p_in +2.0%, p_oodc +3.7% (worse)
- Conclusion: Magnitude weighting shifts gradient from OOD to tandem transfer. SENT BACK for milder α sweep (0.1-0.3).

#### 2026-04-03 — PR #2088: Phase 6: MC Dropout in Surface Refine Head — nezuko — CLOSED
- Branch: `nezuko/mc-dropout-surface-refine`
- W&B group: phase6/mc-dropout (8 runs finished)
- Results: NULL RESULT. No consistent improvement across configs.
  - p=0.05 K=16: flat vs baseline (within noise)
  - p=0.05 K=8: mixed (p_in +2.8%, p_re -2.3%)
  - p=0.10 K=8: tradeoff (p_oodc -3.1%, p_re -3.5% BUT p_tan +3.5%)
- Conclusion: MC Dropout stochasticity insufficient for meaningful variance reduction. EMA already provides implicit regularization. CLOSED.

#### 2026-04-03 — PR #2080: Phase 6: Ensemble Seeds 66-73 — tanjiro — RESULTS IN
- Branch: `tanjiro/ensemble-more-seeds`
- W&B group: phase6/ensemble-more-seeds (8 runs finished)
- Individual model metrics consistent with 8-seed baseline:
  - p_in=13.06±0.34, p_oodc=7.88±0.17, p_tan=30.29±0.63, p_re=6.42±0.09
- Pending: 16-seed ensemble evaluation (combining seeds 42-49 + 66-73)
- Status: Awaiting ensemble eval from student

#### 2026-04-03 — PR #2086: Phase 6: SAM Phase-Only — Flat Minima for OOD Generalization
- Branch: `frieren/sam-phase-only`
- Student: frieren
- Hypothesis: SAM (already in train.py) applied only in last 25% of training (epoch≥120) finds flatter minima. Sweep rho={0.05, 0.1, 0.2} since Lion's sign updates may need different scaling than Adam. Target: reduce seed variance AND improve p_oodc/p_re.
- 8-GPU: 2 baselines (s42/s43) + 3 rho values × 2 seeds = 8 runs
- Status: WIP — assigned

#### 2026-04-03 — PR #2085: Phase 6: srf4L Ensemble Seeds 58-65 (3rd Batch) — nezuko — CLOSED
- Branch: `nezuko/srf4L-ensemble-v3`
- Hypothesis: More srf4L seeds for ensemble
- CLOSED — srf4L hypothesis invalidated (p_tan +5-7% worse per #2079, #2083)

#### 2026-04-03 — PR #2083: Phase 6: srf4L Ensemble Seeds 50-57 — fern — CLOSED
- Branch: `fern/srf4L-ensemble-v2`
- Hypothesis: srf4L seeds 50-57 for ensemble expansion
- W&B group: phase6/srf4L-ensemble-v2 (8 runs finished)
- Results: **DEAD END.** Confirms srf4L regression:
  - p_in: 13.20 (+1.3%), p_oodc: 8.06 (+2.9%), p_tan: **32.43 (+7.0% WORSE)**, p_re: 6.43 (-0.3%)
- CLOSED — srf4L hypothesis invalidated

#### 2026-04-03 — PR #2082: Phase 6: Packed Ensemble — thorfinn
- Branch: `thorfinn/packed-ensemble`
- Hypothesis: M sub-models packed into one forward pass via grouped linear ops. Test M=2,4,8. From Laurent et al. 2023, AirfRANS shows 25% training speedup over deep ensembles.
- 8 GPUs: 2 baseline + 2×M=2 + 2×M=4 + 2×M=8
- Status: WIP, 8 runs running

#### 2026-04-03 — PR #2081: Phase 6: srf4L + 8-Seed Ensemble (Compound) — edward — CLOSED
- Branch: `edward/srf4L-ensemble`
- Hypothesis: srf4L ensemble should beat 3L ensemble
- CLOSED — srf4L hypothesis invalidated (p_tan +5-7% worse per #2079, #2083)

#### 2026-04-03 — PR #2080: Phase 6: Ensemble Seeds 66-73 (Building Toward 24-Model) — tanjiro
- Branch: `tanjiro/ensemble-more-seeds`
- Hypothesis: Extend ensemble to 24 models (seeds 42-73). If ensemble variance reduction scales with N, 24-model could give another step down.
- Status: WIP, 8 crashed + 8 running (relaunched)

#### 2026-04-03 — PR #2079: Phase 6: srf4L on Asinh Baseline (Multi-Seed) — askeladd — CLOSED
- Branch: `askeladd/srf4L-asinh`
- Hypothesis: Test srf4L (surface_refine_layers=4 vs 3) as a standalone improvement
- W&B group: phase6/srf4L-multiseed (8 runs finished)
- Results: **DEAD END.** srf4L consistently WORSE than 3L baseline:
  - p_in: 13.29 vs 13.30 (flat)
  - p_oodc: 7.83 vs 7.78 (+0.7%)
  - p_tan: **32.01 vs 30.33 (+5.5% WORSE)**
  - p_re: 6.53 vs 6.46 (+1.1%)
- Conclusion: 4th surface refine layer overfits, damages tandem transfer. CLOSED

#### 2026-04-03 — PR #2068: Phase 6: Asymmetric/Magnitude-Weighted Surface Loss — alphonse
- Branch: `alphonse/asymmetric-loss`
- Hypothesis: Replace symmetric L1 with magnitude-weighted L1 (overweight stagnation/suction peaks) and pinball tau=0.45 (penalize underprediction)
- 8 GPUs: 2 baseline + magnitude-weighted α=0.5 + α=1.0 + pinball τ=0.45 × 2 seeds each
- Status: WIP, 8 runs running

#### 2026-04-03 — PR #2066: Phase 6: Synthetic Data Generation via Physics-Consistent Interpolation — frieren
- Branch: `frieren/data-gen-interpolation` (CLOSED)
- Hypothesis: Interpolate between same-condition CFD samples to create synthetic training data
- Results: DEAD END. All metrics 3-10x worse than baseline. Root cause: mesh-level interpolation is physically meaningless for unstructured CFD (no spatial correspondence between node indices across different meshes)
- W&B runs: xjo7mj7s, 5dggj8av, ysg5zqr4, nandekgh, jr5qg1i1, s3lt4tjd, 5d828zol, 1n8k5m0g
- Status: CLOSED

### New Assignments (2026-04-02)

#### 2026-04-02 — PR #2008: Phase 6: PirateNets — Random Weight Factorization
- Branch: `frieren/pirate-nets`
- Student: frieren
- Hypothesis: Apply Random Weight Factorization (RWF) from PirateNets to MLP layers. W = diag(exp(s)) * V where V is fixed random, s is learned. Enables multi-scale frequency learning for better boundary layer / pressure prediction.
- 8-GPU sweep: baseline × 2, RWF uniform × 2, RWF multiscale × 2, RWF multiscale + higher LR × 2
- Status: WIP — assigned

#### 2026-04-02 — PR #2009: Phase 6: Geosolver — Geometry-Aware Feature Encoding
- Branch: `alphonse/geosolver`
- Student: alphonse
- Hypothesis: Add explicit geometric features (wall distance, curvature, normals) as inputs + geometry-conditioned attention bias. Physics-geometry coupling directly encoded.
- 8-GPU sweep: baseline × 2, geo features × 2, geo + attention × 2, geo + attn + higher surf_weight × 2
- Status: WIP — assigned

#### 2026-04-02 — PR #2010: Phase 6: HeavyBall Optimizers
- Branch: `nezuko/heavyball-optimizers`
- Student: nezuko
- Hypothesis: Sweep modern optimizers (SOAP, CauchyAdamW, Schedule-Free AdamW, PaLM-SOAP) as alternatives to Lion. Different optimizers handle CFD's extreme gradient structure differently.
- 8-GPU sweep: SOAP × 2 LR, CauchyAdamW × 2 LR, Schedule-Free × 2 LR, PaLM-SOAP, HB-AdamW
- Status: WIP — assigned

### Previously Assigned

#### 2026-04-01 23:25 — PR #2006: Phase 6: Muon Optimizer + Gram-NS
- Branch: `phase6/muon-optimizer`
- Student: thorfinn
- Hypothesis: Replace Lion with Muon (Newton-Schulz orthogonalized gradients).
- **Status: ALL 8 RUNS CRASHED** — zero metrics logged. Code bug. Advisor commented with fix guidance.

#### 2026-04-01 23:25 — PR #2007: Phase 6: XSA (Exclusive Self-Attention)
- Branch: `phase6/xsa-attention`
- Student: askeladd
- Hypothesis: Replace slice attention with XSA exclusion mechanism.
- **Status: ALL 8 RUNS CRASHED.** Catastrophically bad metrics (val/loss 3.76-10.6 vs baseline 0.383). Code regression. Advisor commented with fix guidance.

  | Run | val/loss | p_in | Status |
  |-----|----------|------|--------|
  | xsa-t0.5-s42 | 3.761 | 296.4 | crashed |
  | baseline-s42 | 3.830 | 299.6 | crashed |

---

## Phase 5 Experiments

### Completed (2026-04-02)

#### PR #2003: LR Schedule Sweep — **MERGED WINNER**
- Student: edward | `cosine_T_max=160` beats baseline on ALL metrics
- val/loss: 0.3761 (−1.8%) | p_in: 12.5 (−3.5%) | p_oodc: 8.2 (−1.3%) | p_tan: 29.8 (−0.7%) | p_re: 6.5 (−3.0%)
- W&B: `9ysz96ll` | **MERGED** — New baseline. Follow-up: multi-seed (#2012)

#### PR #2001: Learned Per-Channel Loss Weighting — **CLOSED (mixed)**
- Student: tanjiro | p_tan −1.7% but p_oodc +3.8%, p_re +4.5%
- Key insight: pressure deserves 1.6x weight (sigma=0.087 vs 0.109-0.111 velocity)
- **CLOSED** → Follow-up: hybrid learned p-weight (#2013)

#### PR #2004: Noise Schedule Sweep — **CLOSED (dead end)**
- Student: fern | all stuck at val/loss 0.66-0.68, far above 0.383
- **CLOSED** → Follow-up: PirateNet surface gate (#2014)

### Closed (Dead Ends)
- PR #1998: Multi-Exit Ensemble (frieren) — **CLOSED.** All 8 crashed, val/loss 4.0-7.5. Auxiliary losses destabilized training.
- PR #2002: EMA Decay Sweep (nezuko) — **CLOSED.** All 8 crashed, val/loss 1.1. Runs terminated too early.
- PR #2000: OOD-Focused Training (alphonse) — **CLOSED.** All 8 crashed, val/loss 4.0-11.5. Hard-mining diverged.

### Winners (Merged — Phase 5)
1. **PR #1927: Residual Prediction** — predict correction from freestream. p_oodc -4.7%, p_tan -1.9%
2. **PR #1935: Surface Refinement Head** — dedicated surface MLP. p_re -72.7%, p_tan -8.9%, val/loss -3.3%
3. **PR #2003: cosine_T_max=160** — faster LR decay. All metrics improved. val/loss -1.8%, p_in -3.5%, p_re -3.0%

### Key Non-Winners (Closed — Phase 5)
- 50+ Phase 5 experiments explored: throughput optimization, data augmentation, architecture variants, loss formulations, ensemble methods, curriculum learning, distillation
- Most incremental tuning exhausted — motivates Phase 6 shift
