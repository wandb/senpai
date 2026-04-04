# SENPAI Research State

- **Date:** 2026-04-04 ~14:55 UTC
- **Advisor branch:** noam
- **Phase:** Phase 6 — Beyond Ensemble: Training Improvements

## Current Baseline

### Single-Model Baseline (PRs #2104 + #2115 merged — +aft_foil_srf +aug_gap_stagger σ=0.02, 8-seed mean, seeds 42-49)

| Metric | 8-seed mean | Target to beat |
|--------|-------------|----------------|
| p_in | 13.19 ± 0.33 | < 13.19 |
| p_oodc | 7.92 ± 0.17 | < 7.92 |
| p_tan | **30.05 ± 0.36** | **< 30.05** |
| p_re | 6.45 ± 0.07 | < 6.45 |

**Note:** Numeric targets from PR #2104 8-seed run. PR #2115 (gap/stagger aug) merged; its validated p_oodc improvement (-4.9%) will be reflected in next 8-seed re-run. All new experiments must include `--aft_foil_srf --aug_gap_stagger_sigma 0.02`.

### Ensemble Baseline (PR #2093 — 16-Seed Ensemble, Seeds 42-49 + 66-73)

| Metric | 16-Ensemble |
|--------|------------|
| p_in | **12.1** |
| p_oodc | **6.6** |
| p_tan | **29.1** |
| p_re | **5.8** |

## Student Status (~14:55 UTC)

| Student | PR | Experiment | Status |
|---------|-----|-----------|--------|
| nezuko | #2122 | Fore-Foil Loss Upweighting (ID=6) — Symmetric to Aft-Foil Weight | WIP — just assigned |
| fern | #2117 | Fore-Foil Dedicated SRF Head (ID=6) — Split from Single-Foil | WIP |
| tanjiro | #2121 | Aft-Foil Loss Upweighting — Stronger Gradient for Wake Nodes (weight 1.5–2.0×) | WIP |
| edward | #2120 | Langevin Gradient Noise (SGLD) — Stochastic Exploration for Lion | WIP |
| askeladd | #2119 | PCGrad 3-Way Task Split — Gradient Surgery (single/tandem-normal/tandem-extreme) | WIP |
| frieren | #2107 | Aft-Foil Coordinate Frame Normalization (dual-frame v2) — VRAM fix + 2 more seeds | WIP — sent back |
| alphonse | #2116 | Charbonnier Loss — Fully Smooth L1 (eps sweep: 0.05, 0.1, 0.2) | WIP |
| thorfinn | #2118 | Boundary-ID One-Hot Feature — Explicit Surface-Type Conditioning | WIP |

**All 8 students active. Zero idle GPUs.**

## Recently Reviewed / Merged (2026-04-04)

| PR | Student | Experiment | Decision | Key result |
|----|---------|-----------|---------|------------|
| #2115 | nezuko | Gap/Stagger Perturbation Aug (σ=0.02) | **MERGED** | p_oodc -4.9% (7.446 vs 7.83); p_re -1.5%; p_tan flat; largest p_oodc gain since pressure_first |
| #2114 | tanjiro | Gradient Centralization (GC-Lion) | CLOSED | p_in +12-17%, p_oodc +15-38%. GC incompatible with Lion sign operation |
| #2107 | frieren | Aft-Foil Local Frame (dual-frame v2) | SENT BACK | s73 beats all 4 metrics (p_tan -1.8%) but 2-seed avg fails; VRAM bug; needs fix + 2 more seeds |
| #2106 | askeladd | Fourier Feature Position Encoding | CLOSED | p_tan -1.3% (FF=32) but p_oodc +4.8%, p_re +6.2% |
| #2112 | thorfinn | Mesh-Density Weighted L1 | CLOSED | All metrics regressed 5–16% |
| #2104 | fern | Dedicated Aft-Foil SRF Head (ID=7) | **MERGED** | p_tan -0.8% (30.29→30.05) |
| #2113 | edward | Smooth L1 (Huber) Loss | CLOSED | Catastrophic: p_in +15%, p_tan +7% |

## Current Research Focus

### Primary target: p_tan = 30.05 → push toward 29.1 (ensemble floor)

**Confirmed wins (merged):**
1. `--aft_foil_srf`: dedicated aft-foil SRF head (ID=7) — p_tan -0.8%
2. `--aug_gap_stagger_sigma 0.02`: domain randomization on tandem features — p_oodc -4.9%, p_re -1.5%
3. `--surface_refine`, `--residual_prediction`, `--pressure_first`, `--pressure_deep`, `--asinh_pressure 0.75`, etc.

**Active experiments (8 students):**
1. **Fore-Foil Loss Upweighting (ID=6)** (nezuko #2122) — upweight fore-foil nodes 1.5–2.0× in surface loss; symmetric to tanjiro's aft-foil upweighting; orthogonal to fern's SRF head
2. **Fore-Foil SRF Head (ID=6)** (fern #2117) — dedicated SRF head for fore-foil; natural extension of aft-foil SRF
3. **Aft-Foil Loss Upweighting (ID=7)** (tanjiro #2121) — upweight aft-foil nodes 1.5–2.0× in main surface loss
4. **Dual-Frame Coordinate Features** (frieren #2107) — VRAM bug fix + 2 more seeds; high confidence in improvement once variance resolved
5. **Boundary-ID One-Hot Feature** (thorfinn #2118) — append 3-dim one-hot (ID=5/6/7) to all nodes
6. **PCGrad 3-Way Task Split** (askeladd #2119) — gradient surgery across single/tandem-normal/tandem-extreme-Re
7. **Langevin Gradient Noise / SGLD** (edward #2120) — Gaussian noise after Lion step; sweep {5e-5, 1e-4, 3e-4}
8. **Charbonnier Loss** (alphonse #2116) — fully smooth L1; eps sweep {0.05, 0.1, 0.2}; watch for Smooth L1 failure mode

**Confirmed bottleneck findings:**
- NOT capacity-limited (scale-up #2100)
- NOT sample-level reweighting (OHEM #2101)
- NOT representational entanglement (contrastive #2109)
- NOT progressive surface weight scheduling (#2110)
- NOT gradient centralization (GC-Lion incompatible: #2114)
- HAS coordinate-frame component (dual-frame #2107: s73 beats all 4 metrics; pending VRAM fix)
- HAS dedicated-head component (aft-foil SRF #2104: p_tan -0.8%)
- HAS OOD domain randomization benefit (gap/stagger aug #2115: p_oodc -4.9%)

## Potential Next Research Directions (not yet assigned)

**Priority queue (assign to next idle students):**
1. **Precomputed Pressure-Poisson Soft Constraint** — finite-diff Laplacian stencil as auxiliary physics loss; targets p_tan/p_oodc; ~65 LoC; MEDIUM-HIGH risk; highest remaining novel idea
2. **8-seed combined baseline validation** — re-run 8 seeds with aft_foil_srf + aug_gap_stagger combined to get accurate numeric baseline for future merge decisions; defer until slot opens
3. **Learnable per-boundary asinh scale** — different compression per boundary type (ID=5/6/7); builds on gap/stagger aug insight about OOD robustness

**Deferred pending current results:**
- Expand ensemble to 23 seeds (seeds 100-106 already trained) — do after single-model improvements land

## Key Research Insights

1. **Architecture changes are absorbed by EMA+Lion** — Phase 5: 6 new architectures all failed
2. **Target-space transforms cannot be learnable** — Learnable asinh scale collapses
3. **Ensembling is near-saturating** — 16→23 seeds gains diminish; single-model improvements priority
4. **Deep supervision improves p_in but hurts p_tan** — (#2097) auxiliary loss constrains intermediate reps
5. **Dedicated SRF heads work for boundary-specific geometry** — #2104 confirms per-boundary-type correction
6. **Model capacity is NOT the p_tan bottleneck** — scale-up (#2100) falsified
7. **Contrastive rep separation is NOT the bottleneck** — (#2109) cleanly falsified
8. **p_tan has a coordinate-frame component** — (#2107) local-frame improved p_tan -2.6% with correct implementation
9. **FiLM conditioning on gap/stagger overfits** — (#2104) catastrophic p_oodc regression with conditioning

## Human Researcher Directives

- **#1860 (2026-03-27):** Think bigger — radical new full model changes and data aug. Phase 5 tested 6 new architectures (all failed); current focus on loss reformulation, coordinate representation, training dynamics.
- **#1834 (2026-03-27):** Never use raw data files outside assigned training split. Confirmed.

## Ensemble Seed Pool (Complete)

| Batch | Seeds | Status | Run IDs |
|-------|-------|--------|---------|
| Batch 1 | 42-49 | ✓ BASELINE (aft_srf) | fctgmn1d, rc40fpuu, ygqo9rom, r5uxnp4b, yxhjfisl, qrbprrli, 9whdgscd, ekdcwekr |
| Batch 2 | 66-73 | ✓ ENSEMBLE | j9w7d1r7, mc4jvgqj, cbbvhl62, bigqfn3k, bqhg6lq8, 5ukk7wv6, xlnhwuqc, ii1tz4vv |
| Batch 3 | 74-81 | ✓ Trained | 2sre8vzp, ue8pmbbr, hgyim25m, e2obsfn1, 555102xo, 2lpzf6go, ibsrx1t8, a6e89sx4 |
| Batch 4 | 82-89 | ✓ Trained | u0eapina, fmhetijo, yp7dlkmk, 30hxo8a1, 4e74gtuc, wc8x0v49, qvn871e1, nb6poqj2 |
| Batch 5 | 90-95 | ✓ Trained | ici6bxi1, 6chuzqal, xcsqiwdv, sxisuynb, q8m1w63d, ggn5mioe |
| Batch 6 | 100-106 | ✓ Trained | 9o85duyc, ec7plfg8, zagg4pfs, 6w86plz1, g00kxdva, jt9hwf40, fom4bzro |

**Total trained: 45 models.** 23-seed evaluation (42-49 + 66-73 + 100-106) available; defer until single-model improvements land.

## Confirmed Dead Ends (all time)

| Direction | PRs | Finding |
|-----------|-----|---------|
| Fourier Feature Position Encoding (hidden-space) | #2106 | p_tan -1.3% (FF=32) but p_oodc +4.8%, p_re +6.2%; input-level PE already sufficient |
| Mesh-Density Weighted L1 | #2112 | All metrics regressed 5–16%; train/eval distribution shift; compounds with hard-node mining |
| TTA via AoA Perturbation (training-loop) | #2111 | Self-defeating under timeout; marginal at matching epochs |
| Progressive surface focus (curriculum ramp) | #2110 | p_in regresses +0.7%; dynamic surf_weight already optimal |
| Contrastive tandem-single regularization | #2109 | Hypothesis falsified: rep entanglement NOT bottleneck |
| Asymmetric Asinh Scales (pos/neg) | #2108 | All metrics worse; closes asinh direction |
| Weight-Tied Iterative Transolver | #2103 | Monotonic degradation |
| SIREN Activation in SRF Head | #2102 | Monotonic degradation |
| Deep Supervision (aux loss) | #2097 | p_tan +2.7% regression |
| OHEM (hard example mining) | #2101 | No improvement |
| Model Scale-Up (5L, 160s, 4L/128s) | #2100 | NOT capacity-limited |
| SWAD | #2094 | Catastrophic (+268% p_in) |
| Stochastic Depth (DropPath) | #2099 | All metrics worse |
| Knowledge Distillation | #2090 | No improvement |
| SGDR warm restarts | #2095 | All T_0 values worse |
| SAM Phase-Only | #2086 | Destabilizes Lion |
| srf4L (deeper SRF) | #2079-2085 | p_tan +5-7% WORSE |
| Mesh interpolation | #2066 | Physically invalid |
| SOAP/HeavyBall | #2010,2018-2023 | 2-6% WORSE |
| Muon | #2006 | 30-70% worse |
| Gradient Centralization (GC-Lion) | #2114 | p_in +12-17%, p_oodc +15-38%; GC changes direction which Lion's sign-op treats as noise; incompatible |
| Smooth L1 / Huber Loss | #2113 | Catastrophic: gradient attenuation for small errors incompatible with residual-prediction+asinh; L1 constant gradient is optimal |
| FiLM on gap/stagger | #2104 | p_oodc catastrophe (+41.6%) |
| Foil-2 Loss Upweighting | #1893 | Marginal (still shared head); retesting as #2121 WITH aft-foil SRF head in baseline |
| Ada-Temp (per-node) | #1879,#1793,#1615 | Null across multiple phases |
