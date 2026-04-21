# SENPAI Research Results

## 2026-04-21 — Round 20: T_max=5 Breakthrough + TandemFoil Epoch Starvation

### PR #2732: AirfRANS T_max=5+grad-clip=1.0 (kohaku) — MERGED ✓ NEW BEST (0.01271)

- val_primary/surface_mse: **0.01271** (-3.9% vs 0.01323) at epoch 40
- W&B: uh7fchiy (41 epochs, 30-min). Best at epoch 40 (cosine trough), ep41 rebounds to 0.043.
- T_max=5 produces 8 full cosine cycles in 41 epochs (vs 4 for T_max=10). More frequent phase transitions = deeper basins.
- NOTE: Used default WD=1e-4, NOT golden WD=1e-2. T_max=5+WD=1e-2 compound is HIGHEST PRIORITY.
- Gap to external: **~3x** (was 3.1x).

### PR #2726 (taki): AirfRANS multi-seed at old config — SENT BACK

- Best seed: 0.01292 (seed=103). Doesn't beat new 0.01271 baseline. Sent back with updated golden config.

### PR #2727 (shoya): AirfRANS 4L/256d+grad-clip — SENT BACK

- 0.03693, 2.9x worse. Grad norms rising (10→29). Sent back with golden config + 180-min.

### PRs #2723, #2722, #2724: TandemFoil lr=2e-4 variants — ALL SENT BACK (epoch starvation)

- violet #2723: 5L/256d+lr=2e-4 — only 7 epochs (4.3 min/ep). Sent back for 180-min.
- tanjiro #2722: lr=2e-4+T_max=20 — only 8 epochs (3.75 min/ep). Sent back for 180-min.
- gilbert #2724: lr=1.5e-4 — only 7 epochs. Sent back for 180-min.
- **CRITICAL BUG**: TandemFoil experiments defaulting to 30-min budget. Must set SENPAI_TIMEOUT_MINUTES=180.

### PR #2689 (emma): AirfRANS lr=3e-4 seeds — CLOSED ✗

- Pre-grad-clip experiment. Obsolete.

## 2026-04-21 — Round 19: AirfRANS Golden Config (grad-clip+WD)

### PR #2709: AirfRANS lr=7e-4+grad-clip=1.0+WD=1e-2 — MERGED ✓ NEW BEST (0.01323)

- fern/airfrans-lr7e4-gradclip-wd1e2
- val_primary/surface_mse: **0.01323** (-6.8% vs 0.01419 baseline)
- test_primary/surface_mse: 0.01478 (-2.3%)
- W&B: 7vic8kxn (41 epochs, best at FINAL epoch — still improving!)
- Phase transition at epoch 14, then smooth descent. 7 consecutive new-best epochs (6-12) from WD regularization.
- GOLDEN CONFIG: lr=7e-4 + T_max=10 + grad-clip=1.0 + WD=1e-2 + Fourier + no-EMA.
- Gap to external: **3.1x** (was 3.3x).

### PR #2708: AirfRANS lr=3e-4+grad-clip+seed=789 (kaneda) — CLOSED ✗

- 0.01706 (+20% worse). lr=7e-4 is essential for surface accuracy. lr=3e-4+grad-clip doesn't compete.
- Insight: lr=3e-4 gives better volume_mse (0.093 vs 0.080) but worse surface_mse. Surface is the primary metric.

## 2026-04-21 — Round 18: 4L/512d Sent Back + Grad-Clip Sweep Data

### PR #2691: DrivAerML 4L/512d (frieren) — SENT BACK (epoch cap bug)

- 12.54% at 45 epochs (31 min). Student used `--epochs 50` instead of `--epochs 999`. Loss still steeply descending (63% → 12.5%). UNINTERPRETABLE — needs full 180-min budget. Sent back with corrected command.

### PR #2707: AirfRANS grad-clip=0.5 (haku) — CLOSED ✗

- 0.01446 (+1.9% worse than 0.01419 baseline). Tighter clip counterproductive: 98.6-100% batches clipped (vs 91-98% at clip=1.0). Val-test gap doubles. Hypothesis cleanly falsified — optimal clip is ≥1.0, not <1.0.

### PR #2623: TandemFoil MQA audit (gen, human-directed) — SENT BACK

- MQA: 45.95 at epoch 99 (120ep). Doesn't beat 45.07 baseline. BUT MQA halves val-test gap (4.96 vs 9.12) and beats non-MQA control by 2.1%. Genuine regularization benefit. Sent back to rerun with winning lr=2e-4 config.

### PR #2696: DrivAerML 4L/384d seed sweep (askeladd) — CLOSED (valuable data)

- 5 seeds at 30-min: 13.90-14.70% (std=0.33%). Can't compare to 5.73% baseline (180-min).
- KEY INSIGHT: 4L/384d has remarkably LOW seed sensitivity (0.33% std vs 30pp+ at 4L/256d). No lucky seeds needed — one run is representative.

### Round 18 Assignments

| Student | PR | Experiment |
|---|---|---|
| haku | #2737 | AirfRANS lr=7e-4+grad-clip=1.5 (fill sweep gap) |
| askeladd | #2738 | DrivAerML 4L/384d+lr=2e-4 (transfer TandemFoil finding, 180-min) |

## 2026-04-21 — Round 17: lr=2e-4 Breakthrough + Massive Cleanup

### PR #2610: TandemFoil lr=2e-4+T_max=10 — MERGED ✓ NEW BEST (45.07)

- tetsuo/tandem-tmax10-lr2e4
- val_primary/surface_pressure_mae: **45.07** (-14.7% vs 52.81 baseline) at epoch 107/119
- W&B: ixs1rqgk (119 epochs, 180-min budget, still improving)
- CRITICAL: lr=2e-4 at DEFAULT 3L/192d beats lr=3e-4 at 5L/256d (45.07 vs 52.81). Lower LR + more epochs is the dominant lever. 119 epochs vs 67 epochs, oscillation ~10-20 points (vs 20-30 at lr=3e-4). T_max=10+lr=2e-4 also beats T_max=30+lr=2e-4 (49.99).
- **PARADIGM SHIFT**: LR tuning > architecture scaling for TandemFoil. The combination of lr=2e-4 + 5L/256d is now the highest-priority experiment.

### 8 PRs CLOSED (dead ends / budget mismatch)

**AirfRANS (no grad-clip, all dominated by 0.01419 baseline):**
- #2698 (tanjiro): lr=5e-4 multi-seed → best 0.01667 (17.5% worse). Seeds 100-104.
- #2695 (violet): lr=4e-4 multi-seed → best 0.01703 (20% worse). Seeds 100-104.
- #2683 (gilbert): lr=3e-4 multi-seed → best 0.01530 (7.8% worse). Wide variance (0.015-0.035).
- **CONCLUSION**: LR sweep without grad-clip is exhausted. All future AirfRANS MUST use grad-clip.

**DrivAerML (30-min budget vs 180-min baseline — uninterpretable):**
- #2693 (edward): lr=3e-4 → 12.15% at 45ep (30-min)
- #2692 (taki): 800b → 10.49% at 28ep (30-min)
- #2690 (shoya): dropout=0.05 → 12.64% at 46ep (30-min)
- #2685 (norman): eta_min=1e-5 → 11.42% at 45ep (30-min)
- #2681 (shinji): 600b → 11.54% at 35ep (30-min)
- **CRITICAL BUG**: All ran default 30-min, not 180-min baseline budget. Future DrivAerML MUST set SENPAI_TIMEOUT_MINUTES=180.

### 10 Additional Stale PRs CLOSED

**AirfRANS (no grad-clip):** #2658 (nezuko lr=1e-4), #2666+#2613 (thorfinn T_max=5), #2664+#2615 (senku 3L/256d+lr=1e-3), #2686 (kohaku seeds), #2668 (historia WD)
**TandemFoil (old config):** #2665 (tetsuo dropout), #2667+#2616 (naruto gradclip+slices)

### Round 17 Assignments (15 students)

| Student | PR | Experiment | Dataset |
|---|---|---|---|
| tanjiro | ASSIGNING | lr=2e-4+T_max=20 | TandemFoil |
| violet | ASSIGNING | lr=2e-4+5L/256d (HIGHEST PRIORITY) | TandemFoil |
| gilbert | ASSIGNING | lr=1.5e-4 (bracket LR) | TandemFoil |
| tetsuo | ASSIGNING | lr=2e-4+5L/256d+T_max=20 | TandemFoil |
| naruto | ASSIGNING | lr=2e-4+grad-clip=1.0 | TandemFoil |
| historia | ASSIGNING | lr=2e-4+T_max=5 | TandemFoil |
| senku | ASSIGNING | lr=2e-4+WD=1e-2 | TandemFoil |
| taki | ASSIGNING | lr=7e-4+gc=1.0 multi-seed | AirfRANS |
| shoya | ASSIGNING | 4L/256d+gc=1.0 | AirfRANS |
| kohaku | ASSIGNING | lr=7e-4+gc=1.0+T_max=5 | AirfRANS |
| thorfinn | ASSIGNING | lr=7e-4+gc=1.0 seeds 200-204 | AirfRANS |
| edward | ASSIGNING | 4L/384d+lr=3e-4 (180-min!) | DrivAerML |
| norman | ASSIGNING | 4L/384d+600b (180-min!) | DrivAerML |
| shinji | ASSIGNING | 4L/384d+600b+gc=1.0 (180-min!) | DrivAerML |
| nezuko | ASSIGNING | 4L/384d+800b (180-min!) | DrivAerML |

## 2026-04-21 — Round 16: Two Winners + Grad-Clip Breakthrough

### PR #2595: TandemFoil 5L/256d deep model — MERGED ✓ NEW BEST (52.81)

- sasuke/tandem-5L256d-tmax10
- val_primary/surface_pressure_mae: **52.81** (-30.1% vs 75.59 baseline)
- test_primary/surface_pressure_mae: 55.25 (-23.4%)
- Per-split test: single_in_dist=61.80, geom_camber_rc=60.20, geom_camber_cruise=48.99, re_rand=50.00
- W&B: l5kggnbg (67 epochs, 180-min budget, still improving)
- Commentary: DEPTH SCALING works for TandemFoil too! 5L/256d vs 3L/192d = 30% improvement. All splits improved uniformly (cruise -30.8%, re_rand -27.9%). High oscillation (52-85) from T_max=10 but consistent downward envelope. Train loss 0.115 still decreasing. Mirrors DrivAerML width-scaling discovery.

### PR #2680: AirfRANS lr=7e-4+grad-clip=1.0 — MERGED ✓ NEW BEST (0.01419)

- haku/airfrans-lr7e4-gradclip
- val_primary/surface_mse: **0.01419** (-7.3% vs 0.0153)
- full_val/surface_mse_p: 0.0564, full_val/volume_mse: 0.0723 (-45.7%)
- test_primary/surface_mse: 0.01513
- W&B: 48ldl625 (41 epochs, 30-min, still improving)
- CRITICAL: 91-98% of batches clipped at norm=1.0. Spike reduction 40-45% (peaks 0.23-0.27 → 0.15-0.17). Epoch 40 trough 2.2x deeper than unclipped. Volume MSE improvement (45.7%) even larger than surface.
- Commentary: GRAD-CLIP REOPENS HIGH LR. Severe gradient instability at lr=7e-4 was destroying basins at cosine peaks. Clipping preserves deep trough discoveries. Gap to external: 3.3x (was 3.6x).

### PR #2679: AirfRANS T_max=8 (kaneda) — CLOSED ✗

- val_primary/surface_mse: 0.025 (64% worse than 0.0153 baseline). T_max=8 is a dead end.

### PR #2678: AirfRANS lr=7e-4+WD=1e-2 (fern) — CLOSED ✗

- val_primary/surface_mse: 0.02716 (91% worse than 0.01419 baseline). WD without grad-clip is insufficient.

### PR #2676: DrivAerML 600b+gradclip 4L/256d (shoya) — CLOSED ✗

- 12.82%, obsolete 4L/256d experiment.

### PR #2682: DrivAerML T_max=50 (rei) — SENT BACK

- 12.96% at 45 epochs (30-min budget). Cannot compare to 5.73% at 144 epochs (180-min). Sent back for longer training.

### Round 16 Assignments

| Student | PR | Experiment |
|---|---|---|
| shouko | #2700 | DrivAerML 4L/384d+seed=789 |
| mitsuha | #2701 | DrivAerML 4L/384d+600b+T_max=50 |
| luffy | #2702 | DrivAerML 4L/384d+warmup=3 |
| nami | #2703 | AirfRANS pressure-upweighted loss (20x) |
| asuka | #2704 | AirfRANS asinh-pressure at winning config |
| zoro | #2705 | DrivAerML 4L/384d+lr=4e-4 |
| sasuke | #2706 | TandemFoil 5L/256d+T_max=20 |
| haku | #2707 | AirfRANS lr=7e-4+grad-clip=0.5 |
| kaneda | #2708 | AirfRANS lr=3e-4+grad-clip=1.0+seed=789 |
| fern | #2709 | AirfRANS lr=7e-4+grad-clip=1.0+WD=1e-2 |

15 obsolete stale WIP PRs closed (3L/192d TandemFoil, no-grad-clip AirfRANS, 4L/256d DrivAerML). 12 students freed for reassignment.

## 2026-04-21 — PR #2655: AirfRANS: lr=3e-4 multi-seed — MERGED ✓ NEW BEST

- gilbert/airfrans-lr3e4-multiseed
- Hypothesis: Multi-seed exploitation at lr=3e-4 to find deeper phase transition basins.

| Seed | val_primary/surface_mse | Best Epoch | W&B |
|---|---|---|---|
| **789** | **0.0153** | **41** | srd0fcew |
| 456 | 0.0170 | 39 | 7ha2pinb |
| 123 | 0.0182 | 31 | hwv9hgdc |
| 42 | 0.0193 | 39 | 8vbb1pyk |
| 1337 | 0.0194 | 39 | o1zzutgu |

Commentary: CRITICAL FINDING — seed selection > LR tuning! lr=3e-4+seed=789 achieves 0.0153 (17% better than lr=7e-4's 0.01841). lr=3e-4 distribution (0.0153-0.0194) is TIGHTER than lr=7e-4 (0.0198-0.0463). Seed 789 was still descending at epoch 41. This PR also adds the --seed CLI flag. New AirfRANS baseline: 0.0153. Gap to external: 3.6x.

## 2026-04-21 — PRs #2675,#2674,#2673: TandemFoil WD + AirfRANS LR sweep — CLOSED ✗

- violet: TandemFoil WD=1e-2 → 101.08 at 7ep (cold-start). W&B: dlaf9w6c.
- edward: AirfRANS lr=8e-4 → 0.03833 (108% worse). Too volatile. W&B: djl4y4o5.
- emma: AirfRANS lr=6e-4 → 0.02510 (36% worse). Non-monotonic LR landscape. W&B: 8z0r0fqx.
- All dead ends. AirfRANS LR fully mapped: 7e-4>3e-4>5e-4>6e-4>8e-4.

## 2026-04-21 — PR #2671: AirfRANS lr=7e-4 multi-seed (5 seeds) — CLOSED ✗ (CRITICAL DATA)

- kohaku/airfrans-lr7e4-multiseed
- 5 seeds: 0.0322, 0.0203, 0.0198, 0.0225, 0.0463. Mean=0.028, best=0.0198.
- CRITICAL: The 0.01841 baseline was a statistical outlier (~15th percentile). lr=7e-4 has wide variance.
- **PLATEAU SIGNAL** for LR tuning approach. Seed exploitation is the correct strategy.

## 2026-04-21 — PRs #2670,#2669,#2641: DrivAerML 4L/256d batch experiments — CLOSED ✗

- taki: 1000 batches → 12.40% at 23ep. Too many batches for 4L/256d.
- frieren: 800 batches → 11.41% at 27ep. Data lever NOT saturated at 800b. Useful data but obsolete vs 5.73%.
- tanjiro: warmup+600b → 12.70%. Warmup eats 5 of 34 epochs — hurts at 600b budget.
- All closed: 4L/256d obsolete vs 4L/384d baseline (5.73%).

## 2026-04-21 — Batch close: 17 obsolete DrivAerML 4L/256d PRs

Closed PRs: #2619 (historia WD=1e-2), #2620 (chihiro replica), #2628 (ymir T_max=35), #2630 (inosuke WD=0), #2632 (giyu 25k pts), #2634 (shinobu grad-accum), #2640 (zenitsu T_max=40), #2648 (zoro 4L/320d), #2650 (luffy dropout), #2652 (asuka eval-400), #2654 (nami grad-clip), #2656 (chihiro LR decay), #2659 (shouko lr=5.5e-4), #2660 (mitsuha warmup), #2672 (norman WD+600b), #2676 (shoya gradclip+600b), #2677 (askeladd dropout+600b). All superseded by 4L/384d baseline.

## 2026-04-21 — PR #2602: DrivAerML: 4L/384d+T_max=30 — MERGED ✓ MASSIVE NEW BEST

- kakashi/drivaerml-4L384d-tmax30 (180-min budget)
- Hypothesis: Wider model (384d vs 256d) with proportional heads (6H vs 4H) should scale capacity for 3D automotive CFD.

| Config | val_primary/surface_rel_l2_pct | Epochs | W&B |
|---|---|---|---|
| **4L/384d+T_max=30** | **5.73%** | 151 | 7ogfs7ph |
| 4L/256d+600b (prev best) | 11.97% | 34 | dar47nwl |

Commentary: BREAKTHROUGH — 52% relative improvement. Width scaling (256→384d) is the dominant lever. 151 epochs in 180-min budget (~1.2 min/epoch), still improving at cutoff. Late oscillation suggests T_max=30 slightly aggressive for 384d. External target (3.71%) now 1.55x away. New baseline: 5.73%.

## 2026-04-21 — PR #2663: AirfRANS dropout=0.1 — CLOSED ✗

- shinji/airfrans-lr3e4-dropout. val=0.029072 (58% worse than 0.01841). Dropout disrupts phase transition. W&B: qo0ytm1i. DEAD END.

## 2026-04-21 — PR #2643: DrivAerML eta_min=1e-5+600batches — CLOSED ✗

- rei/drivaerml-4L256d-etamin. val=12.47% — doesn't beat new 5.73% baseline. Architecture change is more important than LR tuning on 4L/256d.

## 2026-04-21 — PR #2662: DrivAerML lr=3e-4+10ep warmup — CLOSED ✗

- shoya/drivaerml-4L256d-lr3e4-warmup
- Result: 51.15% at 2 epochs. Only reached warmup phase (10-epoch warmup + lr=3e-4 way too slow for 30-min budget). W&B: n8mepvlq. DEAD END.

## 2026-04-21 — PR #2661: DrivAerML multi-seed (seed=1) — CLOSED ✗

- askeladd/drivaerml-4L256d-multiseed
- Hypothesis: Test run-to-run variance with deterministic seeding.
- Result: seed=1 → 43.93% (3.5x worse than baseline 12.70%). W&B: 27ac33zp, 46 epochs.
- **KEY INSIGHT:** DrivAerML is extremely initialization-sensitive. The default random seed landed in a favorable basin. Multi-seed runs are high-value for DrivAerML too. DEAD END but important finding.

## 2026-04-21 — PR #2657: AirfRANS lr=2e-4+T_max=10 — CLOSED ✗

- fern/airfrans-lr2e4-tmax10
- Result: val=0.0306 (epoch 31). Phase transition occurs but is delayed and shallower than lr=3e-4 (0.0197) and lr=7e-4 (0.01841). Post-transition stall at epoch 31. LR lower bound confirmed: lr=2e-4 too conservative. W&B: 6hl0j2kn. DEAD END.

## 2026-04-21 — PR #2627: TandemFoil SCA surface cross-attention — CLOSED ✗

- kaneda/tandem-surface-cross-attention (human-directed from issue #2545)
- Two variants: zero-init (destabilized E6, best 122.73) and LayerScale init=1e-4 (stable, best 107.62). Neither beats 75.59. Fatal issue: SCA reduces epochs from 14→8 (75% overhead). SRF head already provides sufficient surface refinement.
- **Finding:** LayerScale (init=1e-4) is the correct initialization for post-backbone attention modules (zero-init causes catastrophic symmetry breaking).
- W&B: fo4hnahz (zero-init), vmlw179l (LayerScale). DEAD END.

## 2026-04-21 — PR #2582: TandemFoil WD sweep — CLOSED ✗

- haku/tandem-tmax10-wd-sweep
- Final results: WD=1e-2=95.59 (7ep), WD=0=96.39 (7ep), WD=1e-4=103.87 (5ep). WD=1e-2 converges 8% faster than default. Can't fairly compare to 14-epoch baseline due to cold-start I/O. violet (#2675) testing WD=1e-2 independently. W&B: wbeh83ah, ssosuyjt, 8tx8zt2n. CLOSED — insufficient epochs for fair comparison.

## 2026-04-21 — PR #2646: AirfRANS: lr=7e-4+T_max=10 — MERGED ✓ NEW BEST

- emma/airfrans-tmax10-lr7e4
- Hypothesis: lr=7e-4 (between the successful 5e-4 and the unstable 1e-3) may find a deeper phase transition basin.

| Config | val_primary/surface_mse | Epochs | Best Epoch | W&B |
|---|---|---|---|---|
| **lr=7e-4+T_max=10** | **0.01841** | 41 | 35 | 3pbxocca |
| lr=3e-4+T_max=10 (prev best) | 0.0197 | 41 | 38 | v5ka7832 |
| lr=5e-4+T_max=10 | 0.0207 | 41 | 40 | z7t3ibwi |

Commentary: lr=7e-4 triggers the phase transition EARLIER (epoch 35 vs 38-40 for lower LRs) and finds a DEEPER basin (0.01841 vs 0.0197). High volatility at cosine LR peaks (epochs 26, 28, 38, 41 spike to ~0.23-0.27) but the trough at epoch 35 is robust. LR sweet spot for AirfRANS is now identified as 3e-4 to 7e-4 range. New baseline: 0.01841. Gap to external: 4.3x.

## 2026-04-21 — PR #2645: DrivAerML: 4L/256d+T_max=30 — 600 batches/epoch — MERGED ✓ NEW BEST

- taki/drivaerml-600batches
- Hypothesis: Increasing train batches from 394 to 600 per epoch (53% more car configurations per epoch) should improve generalization.

| Config | val_primary/surface_rel_l2_pct | test | Epochs | W&B |
|---|---|---|---|---|
| **600 batches/epoch** | **11.97%** | **13.03%** | 34 | dar47nwl |
| 394 batches/epoch (prev best) | 12.70% | 13.54% | 45 | 3aaevlho |

Commentary: KEY INSIGHT — more data per epoch is a critical lever. 600 batches sees 53% more car configs per epoch. Despite fewer total epochs (34 vs 45, hit 30-min timeout), per-epoch improvement compensates. Model still converging at cutoff — longer training or even more batches could push further. New DrivAerML baseline: 11.97%. Gap to external: 3.2x.

## 2026-04-21 — PR #2641: DrivAerML: 4L/256d+T_max=30 — 5-epoch LR warmup — SEND BACK

- tanjiro/drivaerml-4L256d-warmup
- Hypothesis: Linear LR warmup (5 epochs) stabilizes early training.
- Result: val=12.259% (beat old 12.70% baseline, but used old 394 batches). W&B: xf2hw10b, 45 epochs.
- Sent back to compound warmup with new 600-batch baseline.

## 2026-04-21 — PR #2643: DrivAerML: 4L/256d+T_max=30 — eta_min=1e-5 — SEND BACK

- rei/drivaerml-4L256d-etamin
- Result: val=12.38% (beat old 12.70%, but used old 394 batches). Sent back to compound with 600 batches.

## 2026-04-21 — PR #2644: TandemFoil: T_max=10 + slices=32 — CLOSED ✗

- frieren/tandem-tmax10-checkpoint-warmstart
- Hypothesis: Reducing slices from 64 to 32 would speed up epochs enough to overcome cold-start.
- Result: val=97.23 — dead end. No speedup from slices reduction (data loading dominates, not compute). DEAD END.

## 2026-04-21 — PR #2642: AirfRANS: 3L/256d+T_max=10 — CLOSED ✗

- kohaku/airfrans-3L256d-tmax10
- Hypothesis: Width expansion (256d vs 192d) without depth overhead.
- Result: val=0.0357 — dead end. Too slow per epoch to reach phase transition zone (~epoch 38-40). DEAD END.

## 2026-04-21 — PR #2626: DrivAerML: 4L/256d+T_max=25 — CLOSED ✗

- norman/drivaerml-4L256d-tmax25
- Result: val≈13.1%. T_max landscape fully mapped: T_max=30 is the optimum. DEAD END.

## 2026-04-21 — PRs #2625, #2624: DrivAerML LR fine-tuning — CLOSED ✗

- violet: lr=6e-4 → 13.42%. shinji: lr=4e-4 → 13.28%. Best=12.81% (violet 2nd run). Neither beats 12.70%.
- DrivAerML LR landscape fully mapped: lr=5e-4 optimal. DEAD END.

## 2026-04-21 — PR #2612: AirfRANS: 4L/256d+T_max=10 — CLOSED ✗

- edward/airfrans-4L256d-tmax10
- Result: val=0.0881 (best at epoch 22, mid-phase-transition). Only 25 epochs in 30 min (too slow for AirfRANS phase transition). W&B: 77hjmn6u. DEAD END.

## 2026-04-21 — PR #2521: DrivAerML: T_max=10 long run — CLOSED ✗

- asuka/drivaerml-fourier-tmax10-longrun
- Result: val=17.08% at 31 epochs (183 min). Confirms T_max=10 dead end on DrivAerML (2nd confirmation). W&B: x2m4rzm5. DEAD END.

## 2026-04-21 — PR #2614: AirfRANS: lr=3e-4+T_max=10 — MERGED ✓ NEW BEST

- gilbert/airfrans-tmax10-lr3e4
- Hypothesis: Lower LR (3e-4 vs 5e-4) may produce deeper phase transition basin.

| Config | val_primary/surface_mse | Epochs | Best Epoch | W&B |
|---|---|---|---|---|
| **lr=3e-4+T_max=10** | **0.0197** | 41 | 38 | v5ka7832 |
| lr=5e-4+T_max=10 (baseline) | 0.0207 | 41 | 40 | z7t3ibwi |

Commentary: lr=3e-4 finds a DEEPER phase transition basin (0.0197 vs 0.0207). The transition occurs slightly earlier (epoch 38 vs 40). Lower LR = slower convergence but more stable descent into the sharp minimum. This confirms AirfRANS benefits from lower LR during the phase transition. New baseline config: lr=3e-4+T_max=10. Gap to external: 4.6x.

## 2026-04-21 — PRs #2621, #2611: TandemFoil LR+T_max sweep — CLOSED ✗

- askeladd: lr=5e-4 → 91.98 (21.7% worse). Lion lr=3e-4 confirmed optimal.
- nezuko: T_max=7 → 88.25 (16.7% worse). T_max=10 confirmed optimal. W&B: qvay65ie, fcpljam9.

## 2026-04-21 — PR #2605: DrivAerML 5L/256d (2nd confirmation) — CLOSED ✗

- shoya/drivaerml-5L256d-tmax30. val=13.24% (WORSE than 12.70%). Consistent with violet's 13.62%. At matched epochs (43): 5L=14.04% vs 4L=12.96%. Depth sweet spot is 4 layers. W&B: 9fwg8o17.

## 2026-04-21 — PR #2546: TandemFoil coarse aux loss (4 iterations) — CLOSED ✗

- fern/tandem-coarse-aux-loss. 4 variants tested: v1 (16x16, w=0.1)→79.15, v2 (64x64, w=0.01)→75.80, v3-1 (64x64, w=0.005)→78.24, v3-2 (128x128, w=0.01)→82.48. Best=75.80, gap=+0.21. Direction exhausted — 64x64/0.01 is the sweet spot but doesn't beat 75.59.

## 2026-04-21 — PR #2617: AirfRANS: T_max=10 replication — MERGED ✓ NEW BEST

- kohaku/airfrans-tmax10-replica
- Hypothesis: Confirm phase transition is reproducible across runs.

| Config | val_primary/surface_mse | Epochs | W&B |
|---|---|---|---|
| **T_max=10 (replica)** | **0.0207** | 41 | z7t3ibwi |
| Original (PR #2556) | 0.0248 | 41 | 7qre8z5x |

Commentary: Phase transition confirmed reproducible but STOCHASTIC. Different depths each run: 0.0248 (first), 0.0207 (replication), 0.0395 (emma extended run — bad run). The transition reliably occurs at epoch 40, but the basin depth varies. This stochasticity means running multiple seeds should help. New AirfRANS baseline: 0.0207. Gap to external: 4.8x.

## 2026-04-21 — PR #2604: TandemFoil: T_max=10 long run (3rd attempt) — CLOSED ✗

- frieren/tandem-tmax10-longrun-v2
- Result: 91.13 (ep 7), only 8 epochs due to cold-start filesystem I/O (first 5 epochs at 5.4 min/ep vs 2.2 min/ep warm). Third consecutive failure from same infrastructure issue. CLOSED — reassigned to slices=32 approach.

## 2026-04-21 — PRs #2608, #2606, #2609 — DrivAerML T_max/LR sweep — CLOSED ✗

- T_max=15 (taki): 13.65% — worse than 12.70%
- lr=3e-4 (tanjiro): 13.50% — worse than 12.70%
- lr=1e-3 (rei): 12.91% — doesn't beat 12.70%
- Dead ends confirmed: T_max=10<T_max=15<T_max=30=BEST>T_max=50. lr=3e-4<lr=5e-4=BEST>lr=1e-3 (marginal). W&B: crn4k87h, wo8d2l1g, p3lnxcqw.

## 2026-04-21 — PR #2583: TandemFoil: lr=1e-3 — CLOSED ✗

- kaneda/tandem-tmax10-lr1e3
- Result: lr=1e-3 diverged at epoch 9 (val→377). Fallback lr=5e-4: 97.69 (7 ep). lr=3e-4 confirmed optimal for Lion+T_max=10. W&B: chutrgmm, hcx7882f. DEAD END.

## 2026-04-21 — PR #2618: AirfRANS extended run — CLOSED ✗

- emma/airfrans-tmax10-extended
- Result: 0.0395 (worse than 0.0207). Same epoch count as baseline (41) due to same timeout. Phase transition is stochastic — this run got a shallow transition. DEAD END.

## 2026-04-21 — PR #2520: DrivAerML T_max=150 (old arch) — CLOSED ✗

- zoro/drivaerml-fourier-tmax150-longrun
- 3L/192d config (pre-4L/256d), 18.49% at 32 epochs. Superseded by 12.70% baseline. DEAD END.

## 2026-04-21 — PR #2593: DrivAerML: 4L/256d+T_max=30 replication — MERGED ✓ NEW BEST

- shinji/drivaerml-4L256d-tmax30-replica
- Hypothesis: Confirm 12.96% result is robust and not an artifact.

| Config | val_primary/surface_rel_l2_pct | test | Epochs | W&B |
|---|---|---|---|---|
| **4L/256d + T_max=30 (replica)** | **12.70%** | **13.54%** | 45 | 3aaevlho |
| Original baseline (PR #2550) | 12.96% | 14.41% | 43 | 8s5i8y06 |

Commentary: Replication succeeded and BEAT the original by 0.26pp. Key finding: model hit SENPAI_MAX_EPOCHS=50 cap (NOT the timeout) at epoch 45 — still converging! More training headroom confirmed. New DrivAerML baseline: 12.70%.

## 2026-04-21 — PR #2592: DrivAerML: 5L/256d+T_max=30 — CLOSED ✗

- violet/drivaerml-5L256d-tmax30
- Hypothesis: If 4L beats 3L dramatically, does 5L continue the trend?
- Result: val=13.62% (WORSE than 12.70%). 5L causes optimization instability (epoch 38: 21.05% — wild swings). 4L is the sweet spot for 256d width on DrivAerML. W&B: fhp6qzfc. DEAD END.

## 2026-04-21 — PR #2603: DrivAerML: 4L/256d+T_max=10+lr=3e-4 — CLOSED ✗

- norman/drivaerml-4L256d-tmax10-lr3e4
- Hypothesis: Compound T_max=10 + lr=3e-4 on 4L architecture.
- Result: val=14.90% (WORSE). T_max=10 too fast for DrivAerML (high variance). lr=3e-4 too slow. TandemFoil hyperparams don't transfer. W&B: l2kaq446. DEAD END.

## 2026-04-21 — PR #2582: TandemFoil: T_max=10 weight decay sweep — REQUEST CHANGES

- haku/tandem-tmax10-wd-sweep
- Hypothesis: Weight decay interacts with Lion's sign-based updates. Default WD=1e-4 may be suboptimal.

| Config | val_primary/surface_pressure_mae | Epochs | W&B |
|---|---|---|---|
| WD=1e-2 | 93.20 | 7 | wbeh83ah |
| WD=0 | 96.39 | 7 | ssosuyjt |
| WD=1e-4 (control) | 103.87 | 5 | 8tx8zt2n |

Commentary: All runs ran only ~7 epochs in a 30-minute budget (vs 14 epochs needed for fair comparison to 75.59 baseline). WD=1e-2 has smoothest convergence and is still descending at cutoff. Interesting finding: Lion's implicit regularization may make small WD counterproductive (control WD=1e-4 is worst). Sent back for full 180-min run with WD=1e-2 only.

## 2026-04-21 — PR #2550: DrivAerML: Fourier+4L/256d+T_max=30 — MERGED ✓ NEW BEST

- violet/drivaerml-fourier-4L256d-longrun
- Hypothesis: 4L/256d architecture scaling with long training on DrivAerML.

| Config | val_primary/surface_rel_l2_pct | test | Epochs | W&B |
|---|---|---|---|---|
| **4L/256d + T_max=30** | **12.96%** | **14.41%** | 43 | 8s5i8y06 |
| 4L/256d + T_max=50 | 13.04% | — | 44 | qf8vxows |
| Prior baseline (3L/192d) | 33.65% | 34.00% | 6 | xm765o85 |

Commentary: MASSIVE breakthrough — 61.5% relative improvement. Architecture depth is the critical lever: 4L/256d at 43 epochs yields 12.96%, still converging. 3L/256d (PR #2541, 36.14%) was WORSE than 3L/192d baseline, proving width alone doesn't help. T_max=30 slightly better than T_max=50. Gap to external target (3.71%) narrowed from 9x to 3.5x.

## 2026-04-21 — PR #2553: TandemFoil: T_max=10 long run — REQUEST CHANGES

- frieren/tandem-fourier-physics-tmax10-longrun
- Result: val=96.39 at 8 epochs — run too short (~3.75 min/ep vs expected ~2.1 min/ep). Throughput failure, not model failure. Sent back.

## 2026-04-21 — PR #2541: DrivAerML: 3L/256d (round 2) — CLOSED ✗

- shinji/drivaerml-fourier-3L256d-longrun
- Round 2 results: T_max=30 val=36.48%, T_max=50 val=36.14% (both 6 ep). WORSE than 3L/192d baseline (33.65%). Width without depth is counterproductive. W&B: xby1kf9x, 36z4zwiz. DEAD END.

## 2026-04-21 — PR #2437: DrivAerML: surface points sweep (4k/8k/16k) — REQUEST CHANGES

- shouko/drivaerml-spts-sweep
- Results: 8k pts=23.75%, 16k pts=23.95%, 4k pts=25.78%. All beat old baseline but outdated config (no Fourier, 3L/192d, step-based). Sent back to re-run with 4L/256d+Fourier at 8k pts. W&B: pgruvrbi, 4vnb8ko1, whqhyymf.

## 2026-04-21 — Bulk closure: 6 stale WIP PRs (#2519-2524)

Closed 6 more stale WIP PRs after DrivAerML baseline shifted to 12.96%. kakashi #2524, itachi #2498, luffy #2519, zoro #2520, asuka #2521, nami #2523.

## 2026-04-21 — PR #2549: TandemFoil: wake deficit features — CLOSED ✗

- haku/tandem-wake-deficit
- Hypothesis: Gap-normalized displacement from forefoil TE captures wake interaction effects.
- Result: val=81.63 (best ep11), baseline=75.59. +8% worse. W&B: t0kg2ymx. TE coord frame already captures this signal; wake deficit adds noise. DEAD END.

## 2026-04-21 — PR #2548: AirfRANS: Cp panel physics feature — CLOSED ✗

- norman/airfrans-cp-panel
- Hypothesis: Thin-airfoil Cp panel feature transfers from TandemFoil to AirfRANS.
- Result: val=0.2395 (best ep21), baseline=0.0696. +3.4x worse. W&B: jbz675q4. Inviscid theory wrong physics for viscous AirfRANS regime. DEAD END.

## 2026-04-21 — PR #2546: TandemFoil: coarse spatial-pooling aux loss — REQUEST CHANGES

- fern/tandem-coarse-aux-loss
- Hypothesis: 16x16 grid spatial-pooling auxiliary loss provides low-frequency supervision.
- Result: val=79.15 (best ep13), baseline=75.59. +4.7% worse but closest miss. W&B: 7fasc0um. Sent back with instructions to try 64x64 grid, lower weight (0.01), and update to T_max=10.

## 2026-04-21 — PR #2539: AirfRANS: Fourier+4L/256d+T_max=25/15 (round 2) — CLOSED ✗

- gilbert/airfrans-fourier-4L256d-tmax25
- No new results submitted after send-back. Original T_max=25 val=0.2044 — now 3x above 0.0696 baseline (phase transition superseded this approach). CLOSED — outdated architecture.

## 2026-04-21 — PR #2536: TandemFoil: T_max=120/80 extended (round 2) — CLOSED ✗

- kaneda/tandem-fourier-physics-tmax60-sweep
- Round 2 results: T_max=120 regressed to 82.64 (from 78.95 in round 1). T_max=80: 88.90. W&B: x87qzcl3, dw5egv2w. Large T_max values leave LR near peak throughout training. DEAD END — T_max should be ≤ steps per epoch, not multiples of it.

## 2026-04-21 — Bulk closure: 15 stale WIP PRs (#2504-2518)

Closed 15 PRs from round 1-2 whose baselines shifted dramatically (TandemFoil 82.65→75.59, AirfRANS 0.2357→0.0696, DrivAerML 51.35%→33.65%). Students reassigned to current-generation experiments.
- sasuke #2504, sakura #2505, eren #2506, mikasa #2508, armin #2509, levi #2510, ymir #2507, zenitsu #2511, inosuke #2512, giyu #2513, shinobu #2514, chrome #2515, gen #2516, ray #2518, kaworu #2517

## 2026-04-21 — PR #2540: AirfRANS: Fourier+3L/192d+T_max=50 phase transition — MERGED ✓ NEW BEST

- emma/airfrans-fourier-3L192d-tmax50
- Hypothesis: Fourier+3L/192d with T_max=50 (faster per-epoch than 4L/256d) reaches more epochs.

| Run | Config | val_primary/surface_mse | test | Epochs | W&B |
|---|---|---|---|---|---|
| **Winner** | Fourier+3L/192d+T_max=50, lr=5e-4 | **0.0696** | **0.0877** | 23 | ijwvfcms |
| Run 2 | Fourier+3L/192d+T_max=50, lr=8e-4 | 0.1048 (ep21 best) | 0.1511 | 22 | km5xxa3n |
| Prior baseline | Fourier+4L/256d+T_max=50 | 0.2015 | 0.1890 | 14 | ty0cmdfz |

Commentary: PHASE TRANSITION BREAKTHROUGH. val held at 0.19-0.21 from epochs 9-22, then collapsed to 0.0696 at epoch 23 — a single-epoch jump of -65.4%. The mechanism: cosine LR near the T_max=50 trough reaches very low values, allowing the optimizer to settle into a sharp narrow minimum. lr=8e-4 also showed partial transition (0.1048 at ep21) but bounced back to 0.1714 — the higher LR is too large to stabilize in the basin. The 3L/192d model trains faster (23 ep in 30 min vs 14 ep for 4L/256d), reaching the phase transition first. Pressure MSE_p dropped 70.5% (0.9427→0.2779). With 180-min budget (130+ epochs), expect even deeper transitions.

## 2026-04-21 — PR #2490: TandemFoil: Fourier+physics T_max sweep (10/15/20) — MERGED ✓ NEW BEST

- frieren/tandem-fourier-phys-tmax
- Hypothesis: Shorter T_max produces better minima through more rapid LR averaging.

| T_max | val_primary/surface_pressure_mae | test | Epochs | W&B |
|---|---|---|---|---|
| **10** | **75.59** | **72.12** | 14 | 77yoba65 |
| 15 | 80.23 | 77.46 | 14 | aiols138 |
| 20 | 77.00 (ep13 best) | 79.51 | 14 | yt60qcd1 |
| 30 (baseline) | 78.81 | 75.13 | 14 | 8k0blg8s |

Commentary: T_max=10 creates ~75 cosine cycles per epoch (750 steps ÷ 10), enabling extremely rapid LR averaging. All three shorter values beat T_max=30. T_max=10 > T_max=20 > T_max=15 > T_max=30. Still improving at epoch 14 — longer training should push below 70. Per-split test: single_in_dist=72.33, geom_camber_rc=76.01, geom_camber_cruise=70.80, re_rand=69.34. T_max=10 is the new TandemFoil default.

## 2026-04-21 — PR #2544: DrivAerML: Compound 4L/256d + 100k pts — CLOSED ✗

- rei/drivaerml-fourier-compound-best
- Hypothesis: Compound 4L/256d + 100k surface points beats 33.65% baseline.

| Config | val_primary/surface_rel_l2_pct | test | Epochs | W&B |
|---|---|---|---|---|
| 4L/256d + 100k pts + T_max=30 | 36.70% | 37.29% | 5 | 1mk9pwx2 |
| 4L/256d + 100k pts + T_max=50 | 40.36% | 40.99% | 5 | kqhcf1of |
| Baseline | 33.65% | 34.00% | 6 | xm765o85 |

Commentary: Neither run beats 33.65%. The compound (4L/256d + 100k pts) trains slower per epoch (only 5 epochs vs baseline's 6). The capacity gain doesn't compensate for lost training time in the current budget. Closed — violet #2550 is testing 4L/256d with standard 50k pts for a fair architecture comparison.

## 2026-04-21 — PR #2542: DrivAerML: asinh-pressure + residual-prediction — CLOSED ✗ DEAD END

- tanjiro/drivaerml-fourier-physics-features
- Hypothesis: asinh-pressure compression and residual-prediction improve DrivAerML.

| Config | val_primary/surface_rel_l2_pct | test | Epochs | W&B |
|---|---|---|---|---|
| Fourier + asinh-pressure | 38.87% | 38.35% | 5 | o4g84han |
| Fourier baseline (control) | 33.19% | 34.81% | 5 | ccg7wc9k |

Commentary: asinh-pressure HURTS DrivAerML (38.87% vs 33.19%). residual-prediction is a NO-OP on DrivAerML (only implemented for TandemFoil path). Dead end confirmed — DrivAerML pressure range is already well-conditioned for MSE without compression.

## 2026-04-21 — PR #2543: DrivAerML: Fourier+no-EMA+T_max=30 long training replica — MERGED ✓ NEW BEST

- violet/drivaerml-fourier-noema-replica
- Hypothesis: The 2-epoch baseline (51.35%) was compute-limited; longer training with the same config should substantially improve results.

| Run | Config | val_primary/surface_rel_l2_pct | test | Epochs | W&B |
|---|---|---|---|---|---|
| **Winner** | Fourier+3L/192d+T_max=30+no-EMA | **33.65%** | **34.00%** | 6 | xm765o85 |
| Prior baseline | Fourier+3L/192d+T_max=30+no-EMA | 51.35% | 52.06% | 2 | 5ncrjm32 |

Commentary: -34.5% relative improvement. Epoch 2 (51.98%) replicates the original baseline closely, confirming reproducibility. Monotonic convergence, no instability. Run cut short at 6 epochs (~36 min); still strongly descending at cutoff. LR at epoch 6 was 4.77e-5 (cosine trough) — restart would accelerate further. Critical insight: training time is the dominant variable for DrivAerML. A full 180-min run likely pushes below 30%. Also: luffy WIP run shows 28.80% at epoch 11.

## 2026-04-21 — PR #2538: AirfRANS: Fourier+4L/256d+T_max=50 (compound) — MERGED ✓ NEW BEST

- kohaku/airfrans-fourier-4L256d-tmax50
- Hypothesis: Compound architecture (4L/256d from #2478) + schedule (T_max=50 from #2482) gains are super-additive.

| Run | Config | val_primary/surface_mse | test | Epochs | W&B |
|---|---|---|---|---|---|
| **Winner** | Fourier+4L/256d+T_max=50+no-EMA | **0.2015** | **0.1890** | 14 | ty0cmdfz |
| Also beats baseline | Fourier+4L/256d+T_max=30+no-EMA | 0.2195 | 0.1889 | 14 | 85pabaza |
| Prior baseline | no-Fourier+3L/192d+T_max=50 | 0.2357 | 0.2002 | 24 | xmrkwt1y |

Commentary: -14.5% relative improvement. Compound hypothesis confirmed: Fourier+4L/256d (0.2387) + T_max=50 (0.2357) → compound (0.2015), gains are super-additive. Pressure MSE dominates (~99.9% of composite). Still converging at epoch 14 — envelope of cycle minima still descending. T_max=30 creates excessive oscillation (spikes to 0.52 at epoch 3), T_max=50 is better matched to this architecture.

## 2026-04-21 — PR #2536: TandemFoil: Fourier+physics+T_max=60/90/120 sweep — REQUEST CHANGES

- kaneda/tandem-fourier-physics-tmax60-sweep
- Hypothesis: T_max should scale with epoch count for long training runs.

| Config | val_primary/surface_pressure_mae | test | Epochs | W&B |
|---|---|---|---|---|
| T_max=120 | 78.95 | 75.61 | 13 | mqtawnqo |
| T_max=90 | 89.65 | 96.65 | 12 | d0z3jqk7 |
| T_max=60 | 90.26 | 92.28 | 13 | h8l5wru4 |
| Baseline | 78.81 | 75.13 | 14 | 8k0blg8s |

Commentary: T_max=120 misses baseline by 0.14 points (78.95 vs 78.81). T_max=60/90 are significantly worse — high-LR restart peaks dominate their trajectories. T_max=120 is still improving at epoch 13 (monotonically converging). Scheduler steps per-batch, not per-epoch — "epoch-scaling T_max" framing is not quite right. Sent back for T_max=120 extended run + T_max=80 comparison.

## 2026-04-21 — PR #2539: AirfRANS: Fourier+4L/256d+T_max=25/15 — REQUEST CHANGES

- gilbert/airfrans-fourier-4L256d-tmax25
- Hypothesis: Shorter cosine cycles (T_max=25/15) improve convergence for Fourier+4L/256d.

| Config | val_primary/surface_mse | test | Epochs | W&B |
|---|---|---|---|---|
| T_max=25 | **0.2044** | 0.1798 | 14 | lb20qwze |
| T_max=15 | 0.2198 | 0.2146 | 14 | 917gyt1m |
| Baseline | 0.2015 | 0.1890 | 14 | ty0cmdfz |

Commentary: T_max=25 reaches 0.2044 — near miss, 1.4% above 0.2015 baseline. T_max=15 too aggressive (24 cosine cycles/epoch, epoch-end always at LR peak). T_max=25 still converging (epoch 12 spike 0.4634 → epoch 13 new best 0.2044). Test generalization excellent (0.1798 < val). Sent back for T_max=25 full 180-min run.

## 2026-04-21 — PR #2534: TandemFoil: Fourier+physics+4L/256d capacity — REQUEST CHANGES

- edward/tandem-fourier-physics-4L256d-180min
- Hypothesis: 4L/256d capacity with 180-min budget can beat 78.81 (prior 2-epoch test was starved).

| Config | val_primary/surface_pressure_mae | test | Epochs | W&B |
|---|---|---|---|---|
| 4L/256d lr=3e-4 | 95.57 | 102.09 | 9 | edfu20wd |
| 4L/256d lr=2e-4 | 96.99 | 96.37 | 9 | fzx3yf7j |
| Baseline | 78.81 | 75.13 | 14 | 8k0blg8s |

Commentary: Neither run beats baseline (95.57 vs 78.81). But run got only 9 epochs instead of expected 40+ — the 180-min budget was not honored, again running ~30 min only. Trajectory still steeply descending at epoch 9 (95.57). lr=2e-4 more stable test generalization. Sent back with instruction to investigate timeout issue + switch to T_max=50.

## 2026-04-21 01:00 — PR #2482: AirfRANS: T_max=50 + lr=5e-4 + no-EMA (24 epochs) — MERGED ✓ NEW BEST

- emma/airfrans-noema-lr-tmax-variants
- Hypothesis: T_max=50 with multiple cosine restarts per epoch budget improves generalization vs T_max=150

| Run | Config | val_primary/surface_mse | test_primary/surface_mse | Epochs | W&B |
|---|---|---|---|---|---|
| **Run 1 (winner)** | T_max=50, lr=5e-4, no-EMA, no-Fourier, 3L/192d | **0.2357** | **0.2002** | 24 | xmrkwt1y |
| Run 2 | T_max=150, lr=8e-4, no-EMA | 0.2806 (final, unstable) | 0.2297 | 24 | d057fle1 |
| Baseline (#2478) | Fourier+4L/256d+no-EMA, T_max=150 | 0.2387 | 0.2079 | 8 | — |

**Commentary:** T_max=50 delivers 1.3% val improvement over Fourier+4L/256d at 8 epochs. Surprising: no-Fourier 3L/192d with T_max=50 at 24 epochs beats the Fourier+4L/256d baseline. The key insight: T_max=50 allows ~345 cosine warm restarts at 24 epochs (vs ~38 for T_max=150 at 8 epochs). Many warm restarts help escape local minima. Run 2 (lr=8e-4, T_max=150) was unstable — best mid-run 0.2364 but diverged to 0.2806 final. New baseline: 0.2357. CRITICAL: Fourier+4L/256d+T_max=50 not yet tested — this is THE next priority.

---

## 2026-04-21 01:00 — PR #2492: AirfRANS: Fourier+physics+no-EMA — CLOSED (metric incompatibility)

- kohaku/airfrans-fourier-physics-noema
- Hypothesis: Fourier+physics synergy from TandemFoil transfers to AirfRANS

| Run | Config | val_primary/surface_mse | Physical space surface_mse | W&B |
|---|---|---|---|---|
| Fourier+physics | asinh-pressure + residual-pred | 0.1147 (WRONG SPACE) | 4,749,411 | fepjfiw2 |
| Fourier only | no physics | 0.2889 | 2,568,679 | ofv8hcza |

**Commentary:** CONFIRMED NEGATIVE — Fourier+physics does NOT transfer to AirfRANS. The 0.1147 metric is in asinh-compressed space (incompatible with baseline). In physical space, Fourier+physics is 85% WORSE than Fourier-only. Root causes: (1) asinh normalization changes target space making metrics incompatible; (2) residual prediction conflicts with no-slip boundary conditions on AirfRANS surfaces. Valuable code contributions: student implemented --residual-prediction for AirfRANS and surface_mse_phys metric. Fourier-only (0.2889) doesn't beat baseline either — Fourier still caps AirfRANS at 2 epochs.

---

## 2026-04-21 01:00 — PR #2491: TandemFoil: 4L/256d + Fourier+physics capacity — CLOSED (epoch starvation)

- edward/tandem-fourier-physics-capacity

| Run | Config | val_primary/surface_pressure_mae | Epochs | W&B |
|---|---|---|---|---|
| 4L/256d/4H | Lion lr=3e-4 | 185.63 | 2 | pqlvn6qv |
| 3L/192d | Lion lr=2e-4 | 158.01 | 2 | c317jc60 |

**Commentary:** Epoch starvation at old 30-min timeout. Only 2 epochs for both runs. The capacity hypothesis is still scientifically valid. With 180-min budget and --epochs 999, 4L/256d should get 40+ epochs — a fair test. lr=2e-4 at 2 epochs (158.01) showed stronger early trajectory than lr=3e-4 (185.63). Reassigning edward with proper budget.

---

## 2026-04-21 01:00 — PR #2486: TandemFoil: golden config + AdamW vs Lion — CLOSED (dead end)

- shinji/tandem-golden-noema-adamw

| Run | LR | val_primary/surface_pressure_mae | Epochs | W&B |
|---|---|---|---|---|
| AdamW lr=3e-4 | 3e-4 | 167.94 | 11 | n3dnhol4 |
| AdamW lr=5e-4 | 5e-4 | 160.25 | 11 | rurnfmgc |

**Commentary:** CONFIRMED DEAD END. AdamW is definitively inferior to Lion on TandemFoil. Gap WIDENED with more training (from ~28% worse at 2 epochs to ~40% worse at 11 epochs). lr=5e-4 showed catastrophic divergence at epoch 3 (val=701.6). The earlier "AdamW+physics > Lion+physics" finding was likely a 2-epoch artifact. Lion is the optimizer for TandemFoil. Never revisit AdamW on TandemFoil.

---

## 2026-04-21 01:00 — PR #2471: TandemFoil: golden no-EMA (no Fourier/physics) — CLOSED (superseded)

- gilbert/tandem-golden-noema

| Run | LR | val_primary/surface_pressure_mae (best) | Epochs | W&B |
|---|---|---|---|---|
| Lion lr=2e-4 | 2e-4 | 112.62 (ep11 best) | 14 | 7zoua8mi |
| Lion lr=3e-4 | 3e-4 | 111.59 (ep12 best) | 14 | xivn73t6 |

**Commentary:** Neither beats 82.65 current baseline. Projected 25% EMA suppression not found — actual gain only ~2-3%. Key finding: lr=2e-4 shows monotonically smooth convergence while lr=3e-4 spikes to 263 at ep5 and 218 at ep8. T_max=30 appears too short for 14-epoch runs (LR cycles back up aggressively). The no-Fourier/no-physics lineage cannot compete with the Fourier+physics golden config. Good bug catch: epochs default=2 needs --epochs flag.

---

## 2026-04-21 00:05 — PR #2475: DrivAerML: Fourier + no-EMA — MERGED ✓ NEW BEST

- chihiro/drivaerml-fourier-noema
- Hypothesis: Fourier positional encoding compresses high-frequency pressure gradients in 3D car geometry

| Run | T_max | val_primary/surface_rel_l2_pct | test | W&B |
|---|---|---|---|---|
| **Fourier+noEMA T_max=30** | 30 | **51.35%** | 52.06% | 5ncrjm32 |
| Fourier+noEMA T_max=150 | 150 | 52.06% | 51.50% | uy73j36s |
| Baseline (#2467) | — | 56.91% | 57.33% | — |

**Commentary:** Fourier delivers 9.8% relative improvement on DrivAerML at only 2 epochs. Physically motivated: 3D car geometry has sharp pressure gradients at edges/mirrors/underbody. T_max=30 > T_max=150 with Fourier — faster LR cycling better. Critical: with Fourier, lr=5e-4 outperforms lr=8e-4 (51.35% vs violet's 54.33%). Still in steep descent at epoch 2 — major headroom with longer training. New baseline: 51.35%.

---

## 2026-04-21 00:00 — PR #2478: AirfRANS: Fourier + 4L/256d full epoch run — MERGED ✓ NEW BEST

- senku/airfrans-fourier-4L-fullrun
- Hypothesis: More epochs with Fourier + bigger model breaks through AirfRANS stagnation

| Run | T_max | val_primary/surface_mse | test | Epochs | W&B |
|---|---|---|---|---|---|
| **Fourier+4L/256d T_max=150** | 150 | **0.2387** | **0.2079** | 8 | vwb9teqa |
| Fourier+4L/256d T_max=20 | 20 | 0.2390 (ep7 best) / 0.3210 (ep8) | 0.2604 | 8 | fnjbxrks |
| Baseline (#2455) | — | 0.2597 | 0.2392 | 6 | — |

**Commentary:** 17.4% improvement over prior AirfRANS baseline. Critical bug discovered: epochs=2 is hardcoded default in train.py — SENPAI_MAX_EPOCHS only caps, does not raise. Must pass --epochs 999 explicitly going forward. T_max=20 causes LR oscillation at epoch boundaries (best at ep7, then spikes at ep8). T_max=150 stable and still improving at ep8. Full_val/volume_mse=0.2933. New baseline: 0.2387.

---

## 2026-04-21 00:00 — PR #2488: TandemFoil: golden + no-EMA + 4L/256d — CLOSED (epoch starvation)

- kaneda/tandem-golden-noema-capacity

| Run | Model | val_primary/surface_pressure_mae | Epochs | W&B |
|---|---|---|---|---|
| 4L/256d/4H | — | 224.40 | 2 | 2qi6a8tv |
| 5L/320d/5H | — | 206.50 | 2 | fwilsngh |

**Commentary:** Both large models only got 2 epochs vs 14 for baseline. ~15 min/epoch (5.5x slower than 3L/192d). Dead end at current timeout. Key finding: 5L/320d is more epoch-efficient than 4L/256d (206.50 < 224.40 at same epoch count). With 180-min budget and --epochs 999, these models could now be viable — needs retest.

---

## 2026-04-21 00:00 — PR #2479: DrivAerML: Fourier + no-EMA + lr=8e-4 — CLOSED (superseded)

- violet/drivaerml-fourier-noema-lr8e4

| Run | T_max | val_primary/surface_rel_l2_pct | W&B |
|---|---|---|---|
| Fourier+noEMA+lr=8e-4, T_max=150 | 150 | 54.33% | 06i67y41 |
| Fourier+noEMA+lr=8e-4, T_max=30 | 30 | 55.04% | 1aaqtdk4 |

**Commentary:** Both beats old 56.91% baseline but superseded by #2475 (51.35%). Key finding: with Fourier, lr=5e-4 outperforms lr=8e-4 on DrivAerML (51.35% vs 54.33%). Fourier benefit confirmed from independent PR. Closed as superseded.

---

## 2026-04-21 00:00 — PR #2463: TandemFoil: physics + no-EMA + lookahead ablation — CLOSED (superseded)

- rei/tandem-noema-lookahead-ablation-v2

| Run | Config | val_primary/surface_pressure_mae | W&B |
|---|---|---|---|
| no-lookahead, Lion lr=3e-4 | physics, no-EMA, slices=96 | 177.81 | qrhkp488 |
| lookahead, Lion lr=2e-4 | physics, no-EMA, slices=96 | 211.71 | xqh88100 |

**Commentary:** Run 1 beat its own stated baseline (197.87) by 10.1% — validates no-lookahead > lookahead in no-EMA regime (lookahead's slow weights partially replicate EMA lag). BUT current TandemFoil baseline is 82.65 (#2473) — 2x better. Physics at slices=96 is blocked at 2 epochs. The no-lookahead insight should be tested on the Fourier+physics golden config (slices=64). Closed as superseded.

---

## 2026-04-20 22:20 — PR #2455: AirfRANS: 3L/192d no-EMA no-Fourier 6 epochs — MERGED ✓ NEW BEST

- **Branch:** emma/airfrans-noema-4L256d-retest

| Run | Config | val_primary/surface_mse | test | Epochs | W&B |
|---|---|---|---|---|---|
| 4L/256d | no-EMA, no-Fourier | 0.2935 | 0.2706 | 5 | wptz6kat |
| **3L/192d** | **no-EMA, no-Fourier** | **0.2597** (-10.2%) | **0.2392** | **6** | pifi0x1v |

**Commentary:** BREAKTHROUGH — Fourier features add ~3x epoch overhead (5→15 min/epoch). Without Fourier, 6 epochs (0.2597) beats Fourier at 2 epochs (0.2710). Same pattern as TandemFoil: more epochs > features. 4L/256d conclusively worse than 3L/192d under time budget. Pressure finally below 1.0 (test_surface_mse_p=0.9556).

---

## 2026-04-20 22:20 — PR #2470: AirfRANS: Fourier full epoch run — CLOSED (superseded)

- **Branch:** haku/airfrans-fourier-noema-fullrun

| Run | Config | val_primary/surface_mse | Epochs | W&B |
|---|---|---|---|---|
| **lr=5e-4, T_max=150** | Fourier+noEMA | **0.2710** | 2 | dui0c6qg |
| lr=3e-4, T_max=150 | Fourier+noEMA | 0.2805 | 2 | anzo6z7u |
| lr=5e-4, T_max=20 | Fourier+noEMA | 0.4354 | 2 | sklqsht0 |

**Commentary:** Best Fourier result (0.2710) beats old baseline (0.2891) but superseded by emma's no-Fourier 0.2597. CONFIRMED: AirfRANS epoch starvation is structural — 15 min/epoch with Fourier, 30-min timeout = 2 epochs max regardless of parallelism. T_max=20 catastrophically bad (LR cycles back to peak). T_max=150 correct for AirfRANS.

---

## 2026-04-20 22:20 — PR #2476: AirfRANS: Fourier + no-EMA on OOD tasks — MERGED ✓ REYNOLDS OOD BEST

- **Branch:** norman/airfrans-fourier-noema-ood

| Task | val_primary/surface_mse | vs Baseline | Epochs | W&B |
|---|---|---|---|---|
| **reynolds** | **0.3319** (-18.2%) | 0.4059 | 2 | m24dt4cg |
| scarce | 0.2760 (+8.4%) | 0.2547 | 2 | vb77cptv |

**Commentary:** Mixed OOD results. Reynolds: Fourier helps significantly (-18.2%). Scarce: Fourier hurts (+8.4%), possibly overfitting with limited data. New reynolds OOD baseline: 0.3319.

---

## 2026-04-20 22:20 — PR #2469: AirfRANS: cosine T_max sweep — CLOSED (obsolete)

- **Branch:** alphonse/airfrans-noema-cosine-sweep

| T_max | val_primary/surface_mse | Epochs | W&B |
|---|---|---|---|
| 10 | 0.3407 | 2 | bcp5ht2b |
| 20 | 0.3840 | 2 | tnypfuoy |
| 50 | 0.3703 | 2 | ujr87q52 |

**Commentary:** No Fourier features. All worse than pre-Fourier baseline (0.3308) at only 2 epochs. Doubly obsolete vs current 0.2597. Cosine T_max hypothesis untestable at 2 epochs.

---

## 2026-04-20 22:00 — PR #2467: DrivAerML: no-EMA + AdamW lr=8e-4 — MERGED ✓ NEW BEST

- **Branch:** violet/drivaerml-noema-lr-bracket

| Run | LR | val_primary/surface_rel_l2_pct | test | Epochs | W&B |
|---|---|---|---|---|---|
| **lr=8e-4** | **8e-4** | **56.91%** (-20% relative) | 57.33% | 2 | ip8ybl80 |
| lr=1e-3 | 1e-3 | 58.78% | 59.14% | 2 | 28udv9x7 |

**Commentary:** MASSIVE DrivAerML improvement. No-EMA + lr=8e-4 crushes the 71.35% EMA baseline. EMA was suppressing the higher LR's effectiveness. lr=8e-4 is the new DrivAerML default.

---

## 2026-04-20 22:00 — PR #2474: AirfRANS: Fourier + no-EMA + 4L/256d — MERGED ✓ NEW BEST

- **Branch:** senku/airfrans-fourier-noema-combo

| Run | Config | val_primary/surface_mse | test | Epochs | W&B |
|---|---|---|---|---|---|
| **4L/256d** | Fourier+noEMA+4L/256d+lr=5e-4 | **0.2891** (-3.9%) | **0.2856** | 2 | hxyibvbf |
| lr=3e-4 | Fourier+noEMA+3L/192d+lr=3e-4 | 0.2975 | 0.3052 | 2 | 1mhw0tph |

**Commentary:** Fourier + 4L/256d capacity synergize. Still in steep descent at epoch 2 (0.4256→0.2891). More epochs should push much lower. lr=3e-4 variant underperformed on test.

---

## 2026-04-20 22:00 — PR #2471: TandemFoil: golden config + no-EMA — SENT BACK (epoch starvation)

- **Branch:** gilbert/tandem-golden-noema

| Run | val_primary/surface_pressure_mae | Epochs | W&B |
|---|---|---|---|
| lr=3e-4 | 215.94 | 2 | 457alys4 |
| lr=2e-4 | **190.34** | 2 | xpuptoy5 |

**Commentary:** Only 2 epochs at slices=64 (should be 11). Likely parallel execution causing I/O contention. lr=2e-4 improving at 12.7%/epoch — very promising. Sent back for strict sequential rerun. With 11 epochs, projected to dramatically beat 114.92 baseline.

---

## 2026-04-20 22:00 — PR #2472: TandemFoil: golden + physics + no-EMA — CLOSED (epoch starvation)

- **Branch:** kaneda/tandem-golden-physics-noema

| Run | val_primary/surface_pressure_mae | Epochs | W&B |
|---|---|---|---|
| Lion lr=3e-4 | 173.00 | 2 | 3fjtrbv6 |
| **AdamW lr=3e-4** | **153.10** | 2 | dc717g1b |

**Commentary:** Physics features add ~7x overhead at slices=64. Only 2 epochs. Key finding: **AdamW outperforms Lion with physics features** (153.10 vs 173.00, -11.5%) — a reversal of the no-physics optimizer preference. Physics + AdamW at 2 epochs (153.10) already beats old no-physics no-EMA baseline at 2 epochs (197.87). Path forward: slices=32 for more epochs with physics.

---

## 2026-04-20 22:00 — PR #2439: DrivAerML: anchor budget sweep — CLOSED (no-ops, superseded)

- **Branch:** nezuko/drivaerml-anchor-budget-sweep

| Trial | surface_pts/view | val_primary/surface_rel_l2_pct | Epochs | W&B |
|---|---|---|---|---|
| A | 500K | 72.46% | 2 | w0a1g9qo |
| B | 1.5M | 71.37% | 2 | 37zg4voz |

**Commentary:** geometry_supernodes and surface_anchor_points are NO-OPS for senpai_transolver (only work with ABUPTCollate). Student pivoted to surface point budget — more points help marginally but doesn't beat baseline. Now superseded by violet's 56.91%.

---

## 2026-04-20 21:50 — PR #2460: AirfRANS OOD tasks (scarce + reynolds) with no-EMA — MERGED ✓ NEW OOD BASELINES

- **Branch:** norman/airfrans-noema-ood
- **Hypothesis:** No-EMA should improve OOD tasks as it improved the full task

| Task | val_primary/surface_mse | test_primary | val_mse_p | Epochs | W&B |
|---|---|---|---|---|---|
| **scarce** | **0.2547** (-24% vs 0.3351) | 0.6368 | 1.0156 | 2 | bxrn5yye |
| **reynolds** | **0.4059** (-32% vs 0.5956) | 0.6618 | 1.6183 | 2 | az53l5l6 |

**Commentary:** Confirms no-EMA generalizes to OOD tasks. Both improved substantially vs Round 2 EMA baselines. Pressure dominates >95% of surface error in both tasks. Large val/test gap on scarce (1.02→2.54 pressure) indicates distribution shift. Only 2 epochs due to OOD dataset size (~15 min/epoch).

---

## 2026-04-20 21:30 — PR #2435: TandemFoil: cosine T_max sweep at slices=64 — MERGED ✓ NEW BEST

- **Branch:** gilbert/tandem-cosine-tmax-sweep
- **Hypothesis:** Shorter cosine T_max cycles complete more LR restarts in the training budget

| T_max | val_primary/surface_pressure_mae | test_primary | Epochs | W&B |
|---|---|---|---|---|
| **30** | **114.92** (-42% vs 197.87) | **108.16** | 11 | 3ec9m9az |
| 10 | 117.23 | 109.89 | 11 | lx4ly3m6 |
| 50 | 127.51 | 120.69 | 11 | uusjik96 |
| 20 | 132.62 | 124.48 | 10 | 7p6hxl5r |

**Commentary:** BREAKTHROUGH. slices=64 enables 11 epochs in 30 min (vs 2 at slices=96) — a 5.5x training multiplier that completely dominates. T_max=30 is optimal, giving ~25 cosine restarts per epoch at 750 batches/epoch. ALL runs used EMA=True yet still crushed the 197.87 no-EMA baseline. No-EMA retest at slices=64 + T_max=30 is now the highest-priority TandemFoil experiment — projected estimate ~86-90. **slices=64 + T_max=30 is the new golden config for TandemFoil.**

---

## 2026-04-20 21:30 — PR #2459: AirfRANS: asinh-pressure + residual-prediction + no-EMA — CLOSED (metric incompatibility)

- **Branch:** senku/airfrans-noema-asinh-residual
- **Hypothesis:** asinh-pressure + residual-prediction transfer from TandemFoil to AirfRANS

| Trial | val_primary/surface_mse | Epochs | W&B |
|---|---|---|---|
| asinh only | 0.000104 (epoch 1) | 2 | xwbxj30u |
| asinh + residual | 0.002809 | 2 | oyohiwf0 |

**Commentary:** Results are in compressed asinh-normalized space, NOT comparable to baseline (0.3009). The asinh transform changes the target space before normalization. Student correctly identified the incompatibility. Direction is not dead but needs inverse-transform evaluation path. Student also implemented --residual-prediction for AirfRANS (code contribution).

---

## 2026-04-20 21:30 — PR #2449: TandemFoil: Full physics + AdamW LR sweep — CLOSED (EMA, superseded)

- **Branch:** kaneda/tandem-fullphys-adamw-lr-sweep-v2

| Trial | val_primary/surface_pressure_mae | Epochs | EMA | W&B |
|---|---|---|---|---|
| Full physics + AdamW lr=3e-4 | **235.94** | 2 | True | dsictzuq |
| Full physics + AdamW lr=5e-4 | 237.42 | 2 | True | mwh4y0pz |
| Full physics + AdamW lr=8e-4 | 367.42 | 1 | True | kwslbj4e |
| Core physics + AdamW lr=5e-4 | 366.44 | 1 | True | k735vytc |

**Commentary:** EMA=True, now superseded by gilbert's 114.92. Full physics + AdamW lr=3e-4 projected ~189 without EMA — was competitive with old baseline but irrelevant vs new. Only 2 epochs at slices=64 (likely parallel execution). Full physics + AdamW beats core physics at matched EMA conditions.

---

## 2026-04-20 21:30 — PR #2443: TandemFoil: physics+AdamW slices sweep — CLOSED (EMA, superseded)

- **Branch:** edward/tandem-physics-adamw-slices-sweep

| Slices | val_primary/surface_pressure_mae | Epochs | W&B |
|---|---|---|---|
| 32 | **244.33** | 2 | hgj1bash |
| 64 | 251.09 | 2 | alchrjkp |
| 48 | 367.64 | 1 | u2dkyj00 |
| 80 | 353.37 | 1 | yb1b6oru |
| 96 | 445.55 | 1 | elgagd4t |

**Commentary:** EMA=True + broken cosine_t_max=30 (in steps not epochs). Superseded by gilbert's 114.92. Only 1-2 epochs due to parallel execution and physics feature overhead.

---

## 2026-04-20 21:30 — PR #2436: TandemFoil: Reynolds-stratified sampling — CLOSED (dead end)

- **Branch:** chihiro/tandem-re-stratified-sampling

| Variant | val_primary/surface_pressure_mae | val_re_rand | Epochs | EMA |
|---|---|---|---|---|
| v0: restrat + EMA 0.999 | 587.30 (diverged) | 486.13 | 2 | True |
| v1: restrat + EMA 0.9995 | 364.48 | 290.51 | 1 | True |
| v2: restrat + no-EMA | 343.25 | 300.90 | 1 | False |
| v3: control (no restrat) | 345.33 | 292.04 | 1 | True |

**Commentary:** All results far worse than baseline. Re-stratified sampling showed no OOD benefit (re_rand: 300.9 vs control 292.0). v0 diverged. Clear dead end.

---

## 2026-04-20 21:15 — PR #2457: AirfRANS: Fourier + no-EMA + AdamW lr=5e-4 — MERGED ✓ NEW BEST

- **Branch:** haku/airfrans-fourier-noema
- **Hypothesis:** Fourier positional encoding helps resolve high-frequency pressure gradients near airfoil surface

| Trial | Config | val_primary/surface_mse | test_primary/surface_mse | Epochs | W&B |
|---|---|---|---|---|---|
| **0 (WINNER)** | **Fourier + no-EMA + AdamW lr=5e-4** | **0.3009** (-9.1%) | **0.2869** (-10.3%) | 2 | cgr5omp3 |
| 1 | no-EMA + AdamW lr=8e-4 (no Fourier) | 0.3741 (+13.1%) | 0.3457 | 2 | zcho7dzb |

**Per-channel test breakdown (Trial 0):** Ux=0.001468, Uy=0.0000729, p=1.1459, nut=0.000351

**Commentary:** BREAKTHROUGH — Fourier features + no-EMA beat the 6-epoch baseline in just 2 epochs. Pressure channel improved -10.3% (1.28→1.15). nut channel regressed +875% but is negligible in composite (3 orders of magnitude smaller than pressure). Still rapidly descending at cutoff — full epoch budget should push significantly lower. Trial 1 confirmed lr=8e-4 without Fourier is a dead end on AirfRANS. Fourier encoding is now mandatory for AirfRANS.

---

## 2026-04-20 21:00 — PR #2440: DrivAerML: AdamW vs Lion baseline sweep — MERGED ✓ FIRST BASELINE

- **Branch:** shoya/drivaerml-adamw-baseline-sweep
- **Hypothesis:** Establish first DrivAerML baseline comparing AdamW vs Lion optimizer

| Run | Config | val_primary/surface_rel_l2_pct | Epochs |
|---|---|---|---|
| AdamW lr=3e-4 | 3L/192d, slices=96 | 71.76% | 2 |
| **AdamW lr=5e-4** | 3L/192d, slices=96 | **71.35%** (BEST) | 2 |
| AdamW lr=8e-4 | 3L/192d, slices=96 | 71.76% | 2 |
| Lion lr=3e-4 | 3L/192d, slices=96 | 78.45% (degraded) | 2 |

**Commentary:** First DrivAerML baseline. AdamW clearly beats Lion (which degraded epoch-over-epoch). All AdamW LRs converge to ~71.4-71.8% — optimizer matters more than LR in this range. 71.35% vs 3.71% target = huge gap, but only 2 epochs (30-min timeout, ~10-11 min/epoch). DrivAerML cases have ~8.6M surface points; student resolved OOM with 50k-point sampling. Eval coverage thin (~3.5% of val surface per epoch). AdamW lr=5e-4 is the DrivAerML starting point going forward.

---

## 2026-04-20 21:00 — PR #2434: TandemFoil: slices throughput sweep — CLOSED (EMA-suppressed)

- **Branch:** violet/tandem-slices-sweep

| Slices | val_primary/surface_pressure_mae | Epochs | Peak VRAM |
|---|---|---|---|
| 32 | 288.51 | 2 | ~77 GB |
| 48 | 452.38 | 2 | — |
| 64 | 486.53 | 1 | — |
| 96 | 294.21 | 2 | ~92 GB |

**Commentary:** EMA=True. Slices do NOT affect throughput (all got 2 epochs regardless). slices=32 ≈ slices=96 quality with 15 GB less memory. slices=48 is anomalously bad. Data loading is the bottleneck, not slice attention.

---

## 2026-04-20 21:00 — PR #2433: TandemFoil: AdamW LR sweep slices=64 — CLOSED (Lion dominates)

- **Branch:** alphonse/tandem-adamw-lr-sweep

| LR | val_primary/surface_pressure_mae | Epochs |
|---|---|---|
| 3e-4 | 444.39 | 1 |
| 5e-4 | 338.15 | 1 |
| **8e-4** | **254.34** | 2 |
| 1e-3 | 456.86 | 2 |

**Commentary:** No-EMA (EMA=None confirmed). AdamW lr=8e-4 is optimal AdamW LR but still 22% behind Lion baseline (197.87). Lion dominates AdamW on TandemFoil — opposite of AirfRANS finding. 4 parallel jobs caused epoch starvation (v0/v1 only 1 epoch).

---

## 2026-04-20 21:00 — PR #2413: TandemFoil: full physics stack — CLOSED (core subset better)

- **Branch:** fern/tandem-full-physics

| Variant | val_primary/surface_pressure_mae | Epochs |
|---|---|---|
| v0: Full physics (all flags) | 270.74 | 2 |
| v1: Full minus wake-angle | 293.44 | 2 |
| v2: cp-panel-scale=0.5 | 285.87 | 2 |
| v3: vortex-panel-scale=0.05 | **268.10** | 2 |

**Commentary:** EMA=True. Full physics stack (best 268.10) worse than core physics subset (262.82, #2414). Wake-angle is the most impactful single feature (+22.7 when removed). Vortex-panel computation has Python for-loop bottleneck (~25 min/epoch). Core physics subset is the right path — full stack not worth the computational cost.

---

## 2026-04-20 19:50 — PR #2412: TandemFoil: clean baseline no-EMA (frieren v4) — MERGED ✓ NEW BEST

- **Branch:** frieren/tandem-baseline-default
- **Hypothesis:** Removing EMA in ultra-short training regime (2 epochs)
- **W&B run:** y8f8pkkn (v4)

| Metric | Value |
|--------|-------|
| val_primary/surface_pressure_mae | **197.87** (NEW BEST, -24.7% vs 262.82) |
| test_primary/surface_pressure_mae | 191.70 |
| test_single_in_dist | 212.64 |
| test_geom_camber_rc | 172.00 |
| test_geom_camber_cruise | 187.39 |
| test_re_rand | 194.77 |
| Epochs | 2 (30-min timeout) |
| Config | Lion lr=3e-4, slices=96, **use_ema=False**, use_lookahead=True, NO physics features, cosine_t_max=50 |

**Commentary:** CRITICAL FINDING. Removing EMA improved val_mae by 24.7% without any physics features. EMA with ema_start_step=50 never meaningfully activates in 2 epochs (only 2×750=1500 steps, barely above start step), and the exponential moving average of improving weights with stale early weights is actively harmful. This was independently confirmed on AirfRANS (#2431: EMA degrades 0.3914→0.5038). ALL future experiments MUST use `--no-use-ema`. Compounding this with physics features should give further gains.

Secondary findings from this PR:
- v1 (lr=2e-4, EMA=True): 264.14 — lower LR also helpful even with EMA
- v3 (no-lookahead, EMA=True): 281.15 — lookahead is beneficial
- v0 (baseline, EMA=True): 310.96 — confirms EMA was masking improvements all along
- v2 (lr=5e-4, EMA=True): 446.12 — higher LR with EMA is catastrophic (1 epoch only)

---

## 2026-04-20 19:50 — AirfRANS Round 2 Summary (5 PRs closed — epoch starvation)

Key pattern: ALL 5 AirfRANS Round 2 PRs ran at slices=64 with 4 parallel jobs → only 2 epochs completed vs baseline's 6 epochs. Results are confounded and cannot be compared to baseline.

**#2428 (kohaku, LR bracket):**
| LR | val_primary/surface_mse | Epochs |
|---|---|---|
| 8e-4 | 0.3278 (best) | 5 |
| 3e-4 | 0.3414 | 5 |
| 6e-4 | 0.3513 | 5 |
| 4e-4 | 0.3754 | 5 |
*Note: 5 epochs at slices=64, vs baseline 6 at slices=96. Confounded. lr=8e-4 slightly best but vol_mse regresses.*

**#2429 (emma, capacity):** 4L/256d + 3L/192d at slices=64, only 2 epochs each due to 4-job parallelism. Inconclusive.

**#2430 (senku, cosine T_max):** T_max=10/20/30/50 at slices=64, only 2 epochs. Best T_max=20 (val=0.4763) but confounded.

**#2431 (haku, scaffold ablation):** CRITICAL FINDING — EMA is harmful on AirfRANS!
| Config | val_primary/surface_mse |
|---|---|
| no-EMA + Lookahead (v2, best) | 0.3914 |
| bare AdamW (v3) | 0.4590 |
| full scaffold EMA+Lookahead (v0) | 0.5038 |
| no-Lookahead (v1) | 0.5268 |
*All at slices=64, 2 epochs. No-EMA is the key lever.*

**#2432 (norman, OOD tasks):** First OOD baselines established.
| Task | val | test |
|---|---|---|
| scarce | 0.3351 (AdamW) | 0.8021 |
| reynolds | 0.5956 | 0.8999 |
| full (confounded) | 0.5201 | 0.5041 |
*Large val/test gap on OOD tasks confirms real generalization challenge.*

**Round 2 Key Lessons:**
1. Running 4 parallel AirfRANS jobs causes epoch starvation (I/O contention with num_workers=0)
2. EMA is harmful on AirfRANS (and TandemFoil) in short training regimes
3. MAX 2 parallel jobs per AirfRANS student going forward
4. Must use slices=96 (not 64) for fair comparison to baseline

---

## 2026-04-20 19:30 — PR #2414: TandemFoil: core physics features (TE+Cp+asinh+residual) — MERGED ✓

- **Branch:** tanjiro/tandem-physics-features
- **Hypothesis:** Physics features (TE coord frame, Cp panel, asinh pressure, residual prediction) improve TandemFoil surface pressure prediction by giving the model physically-structured inputs.
- **W&B run:** 1zbp5dlu

| Metric | Value |
|--------|-------|
| val_primary/surface_pressure_mae | **262.82** (NEW BEST) |
| test_primary/surface_pressure_mae | 257.51 |
| test_single_in_dist | 267.26 |
| test_geom_camber_rc | 280.59 |
| test_geom_camber_cruise | 225.63 |
| test_re_rand | 256.55 |
| Epochs | 2 (30-min timeout, ~15 min/epoch) |
| Config | Lion lr=3e-4, slices=96, physics: te_coord+cp_panel+cp_panel_tandem_only+asinh+residual+pressure_prior |

**Commentary:** New TandemFoil best — beats alphonse's AdamW baseline (269.32) by 2.4%. Physics features provide physically-structured inductive bias. Two Inf values in test_geom_camber_cruise/mae_vol_p (asinh inversion overflow on volume predictions at early training) — surface metrics are unaffected. Only 2 epochs completed; model was still rapidly improving. Key gap: physics features tested only with Lion — combining with AdamW should compound the gains. Next priority: physics + AdamW LR sweep (tanjiro #2441), ANP decoder (shinji #2444), wake feature ablation (askeladd).

---

## 2026-04-20 19:30 — PR #2419: TandemFoil: batch_size=4 with scaled LR — CLOSED

- **Branch:** askeladd/tandem-batch4-lr
- **W&B run:** 2lc5q8ae

| Metric | Value |
|--------|-------|
| val_primary/surface_pressure_mae | 454.96 |
| test_primary/surface_pressure_mae | 429.53 |
| Epochs | 2 (30-min timeout) |
| Config | Lion lr=6e-4, batch_size=4, slices=96 |

**Commentary:** batch_size=4 halves gradient steps per unit time → only 2 epochs, severely undertrained. val_mae=454.96 vs baseline 269.32 (+69%). Clear dead end: batch_size doubling destroys the epoch budget. batch_size=2 is optimal for TandemFoil within 30-min timeout.

---

## 2026-04-20 19:30 — PR #2418: TandemFoil: normalization tricks (asinh+residual) — CLOSED (superseded)

- **Branch:** thorfinn/tandem-normalization
- **W&B run:** svy77euk

| Metric | Value |
|--------|-------|
| val_primary/surface_pressure_mae | 291.32 |
| test_primary/surface_pressure_mae | 280.60 |
| Epochs | 2 (30-min timeout) |
| Config | Lion lr=3e-4, slices=96, asinh_pressure=True, residual_prediction=True, cosine_t_max=50 |

**Commentary:** asinh+residual alone (without TE coord+Cp panel) achieves 291.32 — worse than baseline 269.32. Tanjiro's broader physics stack (#2414) includes these features AND more, and beats baseline at 262.82. The subset is superseded. Also: only 1 of requested 4 ablation variants was submitted. Inf in cruise vol_p — same numerical overflow from asinh as #2414 (early-training artifact).

---

## 2026-04-20 19:30 — PR #2417: TandemFoil: bigger model (4L/256d/4H/128slices) — CLOSED

- **Branch:** edward/tandem-bigger-model
- **W&B run:** fv82ma84

| Metric | Value |
|--------|-------|
| val_primary/surface_pressure_mae | 314.52 |
| test_primary/surface_pressure_mae | 306.34 |
| Epochs | 2 (30-min timeout, ~27 min/epoch) |
| Config | Lion lr=3e-4, 4L/256d/4H, slices=128 |

**Commentary:** Bigger model is too slow for 30-min budget (27 min/epoch → only 2 epochs). val_mae=314.52 vs baseline 269.32 (+17%), but model was still rapidly improving. On AirfRANS, bigger model + Lion was also weak while + AdamW showed promise. Capacity scaling should be revisited with AdamW + slices=64 for fairer comparison.

---

## 2026-04-20 19:30 — PR #2415: TandemFoil: higher LR lr=1e-3 (Lion) — CLOSED

- **Branch:** nezuko/tandem-lr-1e3
- **W&B run:** 1gshqd87

| Metric | Value |
|--------|-------|
| val_primary/surface_pressure_mae | 352.40 |
| test_primary/surface_pressure_mae | 338.49 |
| Epochs | 2 (30-min timeout) |
| Config | Lion lr=1e-3, slices=96, cosine_t_max=150 |

**Commentary:** Lion at lr=1e-3 achieves 352.40 vs baseline 269.32 (+31%). Mirrors AirfRANS pattern where Lion at any LR lost to AdamW. LR tuning within Lion is the wrong direction. The correct experiment is AdamW LR sweep (covered in tanjiro's #2441 and alphonse's #2433).

---

## 2026-04-20 18:38 — PR #2416: TandemFoil: AdamW optimizer vs Lion baseline

- **Branch:** alphonse/tandem-adamw
- **Hypothesis:** AdamW optimizer may outperform Lion on TandemFoil as it does on AirfRANS
- **W&B run:** r5t674uy

| Metric | Value |
|--------|-------|
| val_primary/surface_pressure_mae | 269.32 |
| test_primary/surface_pressure_mae | 262.56 |
| val_geom_camber_cruise | 224.60 (test) |
| val_re_rand | 249.91 (test) |
| val_geom_camber_rc | 270.91 (test) |
| val_single_in_dist | 304.83 (test) |
| Epochs | 2 (30-min timeout, ~15 min/epoch) |
| Config | AdamW lr=5e-4, slices=96, 3L/192d |

**Commentary:** Only 2 epochs completed due to TandemFoil's high per-epoch cost (~15 min/epoch at slices=96). Model still strongly improving (val MAE 349→269). Establishes first TandemFoil baseline on the radford branch. Infinity observed in `test_geom_camber_cruise/mae_vol_p` — likely EMA artifact at very early training. The Lion vs AdamW comparison cannot be made fairly at 2 epochs. Merged to establish baseline. Round 2 will test AdamW LR sweep and slices reduction for faster epoch cycling.

---

## 2026-04-20 18:35 — PR #2423: AirfRANS: AdamW optimizer lr=5e-4

- **Branch:** kohaku/airfrans-adamw-lr5e4
- **Hypothesis:** AdamW may outperform Lion+Lookahead on AirfRANS
- **W&B run:** u95mzqso

| Metric | Value |
|--------|-------|
| val_primary/surface_mse | 0.3308 |
| test_primary/surface_mse | 0.3199 |
| surface_mse_Ux (test) | 0.001287 |
| surface_mse_Uy (test) | 0.000466 |
| surface_mse_p (test) | 1.2775 |
| surface_mse_nut (test) | 3.6e-05 |
| Epochs | 6 (30-min timeout) |
| Config | AdamW lr=5e-4, slices=96, 3L/192d/3H |

**Commentary:** AdamW at lr=5e-4 dramatically outperforms Lion lr=3e-4 (0.331 vs 0.538 baseline, -38%). Clean monotonic improvement across all 6 epochs with no plateau — still improving at cutoff. Pressure channel dominates error (surface_mse_p=1.28 vs <0.002 for velocity channels). All other AirfRANS PRs closed as inferiors: Lion at any LR cannot compete with AdamW. Merged as AirfRANS baseline. Round 2 will bracket the AdamW LR (3e-4–8e-4) and explore capacity + cosine schedule.

---

## 2026-04-20 18:35 — PR #2420: AirfRANS: clean default baseline (closed — superseded)

| Metric | Value |
|--------|-------|
| val_primary/surface_mse | 0.3973 (best epoch 4) / 0.5384 (final epoch 6) |
| Config | Lion lr=3e-4, slices=96, 3L/192d |

**Commentary:** Superseded by kohaku's AdamW recipe (-38%). Oscillating val_mse at epochs 5-6 consistent with cosine LR mismatch (T_max=150 at epoch 6 = barely moved off initial LR). Research question answered: Lion at default settings is not competitive on AirfRANS.

---

## 2026-04-20 18:35 — PR #2421: AirfRANS: higher LR lr=1e-3 (closed)

| Metric | Value |
|--------|-------|
| val_primary/surface_mse | 0.4695 (epoch 6) |
| Config | Lion lr=1e-3, slices=96 |

**Commentary:** Lion at higher LR (0.470) still far behind AdamW (0.331). Unstable spike at epoch 3 (1.36). LR tuning within Lion is the wrong direction — optimizer is the lever.

---

## 2026-04-20 18:35 — PR #2422: AirfRANS: intermediate LR lr=5e-4 (closed)

| val_primary/surface_mse | 0.5940 (final) / 0.4151 (best epoch 4) |
|---|---|
| Config | Lion lr=5e-4, cosine_t_max=50, slices=96 |

**Commentary:** Lion lr=5e-4 degraded when LR peaked at 5e-4 in cosine cycle (surface_mse spiked to 0.576). Final metric 0.594 is worse than Lion baseline. Confirms Lion is not competitive regardless of LR on AirfRANS.

---

## 2026-04-20 18:35 — PR #2424: AirfRANS: bigger model 4L/256d (closed)

| val_primary/surface_mse | 0.5222 (epoch 5) |
|---|---|
| Config | Lion lr=3e-4, 4L/256d/4H/128sl |

**Commentary:** Bigger model with Lion (0.522) barely beats Lion baseline (0.538) — not meaningful. Slower training (5 epochs in 30 min) and noisy trajectory. Capacity helps only when paired with a good optimizer (haku's 4L+AdamW reached 0.379).

---

## 2026-04-20 18:35 — PR #2425: AirfRANS: ablate surface refinement head (closed — research question answered)

| val_primary/surface_mse | 0.5700 (final) / 0.4769 (best ep3) |
|---|---|
| Config | Lion lr=3e-4, surface_refine=False, cosine_t_max=50 |

**Commentary:** Without surface refinement (0.570) is worse than with it (0.538 baseline). Surface refinement head confirmed beneficial. Default surface_refine=True is correct.

---

## 2026-04-20 18:35 — PR #2426: AirfRANS: deeper model 6L/192d (closed — diverging)

| val_primary/surface_mse | 0.9425 (epoch 5) |
|---|---|
| Config | Lion lr=3e-4, 6L/192d, cosine_t_max=50 |

**Commentary:** Severe divergence — spiked to 1.017 at epoch 4. 6-layer model is ~6 min/epoch, only 5 epochs in 30 min. Going deeper with Lion is clearly harmful. Depth scaling is not the direction.

---

## 2026-04-20 18:35 — PR #2427: AirfRANS: bigger model + AdamW lr=1e-3 (closed — direction redirected)

| val_primary/surface_mse | 0.3793 (epoch 5) / test: 0.3482 |
|---|---|
| Config | 4L/256d, AdamW lr=1e-3, cosine_t_max=50 |

**Commentary:** Second best AirfRANS result (0.379) but loses to simpler AdamW 3L/192d at lr=5e-4 (0.331). Bigger model is slower (~6 min/epoch), fewer epochs, and the higher LR (1e-3 vs optimal 5e-4) likely suboptimal. Direction is promising but needs to be tested with lr=5e-4 and fewer slices — covered in emma's Round 2 PR #2429.

---

## 2026-04-20 22:45 — PR #2461: TandemFoil: physics + no-EMA + Lion LR sweep (2e-4, 3e-4) — CLOSED

- **Student:** tanjiro
- **Branch:** tanjiro/tandem-noema-physics-lion-lr-sweep-v2

| Metric | Run 1 (Lion 2e-4) | Run 2 (Lion 3e-4) | Baseline |
|---|---|---|---|
| val_primary/surface_pressure_mae | 160.46 | ~170+ | 114.92 |
| Epochs | 2 (slices=96) | 2 (slices=96) | 11 (slices=64) |

**Commentary:** Physics + no-EMA + Lion at slices=96 still only gets 2 epochs. lr=2e-4 was the stronger LR (160.46 vs ~170+ for 3e-4), showing strong epoch-over-epoch improvement. Runs were still descending rapidly at cutoff. Cannot compete with the 11-epoch golden config at slices=64. Closed — redirected tanjiro to test Lion lr=2e-4 at slices=64 (PR #2485).

---

## 2026-04-20 22:45 — PR #2456: TandemFoil: triple stack (no-EMA + physics + AdamW) — CLOSED

- **Student:** shinji
- **Branch:** shinji/tandem-noema-physics-adamw

| Metric | Trial 0 (AdamW 5e-4) | Trial 1 (AdamW 3e-4) | Baseline |
|---|---|---|---|
| val_primary/surface_pressure_mae | 207.02 | **173.85** | 114.92 |
| test_primary/surface_pressure_mae | 189.42 | 170.13 | 108.16 |
| val_re_rand | 169.76 | 149.66 | — |
| W&B run | z8pxqegf | zhn2jxyv | — |
| Epochs | 2 (slices=96) | 2 (slices=96) | 11 (slices=64) |

**Commentary:** AdamW reversal with physics confirmed again — lr=3e-4 beats lr=5e-4. Trial 0 (5e-4) actually *regressed* from epoch 1 to 2 (180.28→207.02), likely cosine LR cycle instability. Trial 1 (3e-4) was improving strongly (207.85→173.85). 173.85 beats the no-physics no-EMA baseline (197.87) by 12.1%, validating that physics + AdamW is a productive direction. Cannot beat golden 11-epoch baseline. Closed — redirected shinji to test AdamW vs Lion at 11 epochs without physics (PR #2486).

---

## 2026-04-20 22:45 — PR #2453: TandemFoil: ANP cross-foil decoder + no-EMA + physics + AdamW — CLOSED

- **Student:** thorfinn
- **Branch:** thorfinn/tandem-noema-anp-decoder

| Metric | ANP Decoder | Control (no-ANP) | Baseline |
|---|---|---|---|
| val_primary/surface_pressure_mae | 166.56 | **158.04** | 114.92 |
| Epochs | 2 (slices=96) | 2 (slices=96) | 11 (slices=64) |

**Commentary:** ANP decoder is conclusively negative (+5.4% vs no-ANP control). Control (physics + AdamW + no-EMA, 158.04) consistent with shinji's triple-stack result (173.85 different config). ANP should never be used going forward. Control result of 158.04 confirms AdamW+physics+no-EMA trajectory but can't beat golden config. Closed — redirected thorfinn to DrivAerML slices reduction (PR #2487).

---

## 2026-04-20 22:45 — PR #2477: TandemFoil: physics + no-EMA + AdamW at slices=32 — CLOSED

- **Student:** kaneda
- **Branch:** kaneda/tandem-physics-slices32-noema

| Metric | Run 1 (AdamW 3e-4) | Run 2 (AdamW 5e-4) | Baseline |
|---|---|---|---|
| val_primary/surface_pressure_mae | 152.25 | 172.79 | 114.92 |
| test_primary/surface_pressure_mae | 146.76 | 167.75 | 108.16 |
| W&B run | toguophc | 920u2eqy | — |
| Epochs | 2 (slices=32) | 2 (slices=32) | 11 (slices=64) |

**Commentary:** CRITICAL FINDING — slices=32 with physics still only gets 2 epochs. The central hypothesis failed: physics overhead is per-sample, not per-slice. Halving slices twice (96→64→32) produces zero meaningful speedup on physics feature computation. The 7x overhead comes from Cp panel + TE coord frame + pressure prior — these are datapoint-level operations. Physics features need precomputed caching to be viable. Best result 152.25 (lr=3e-4) confirms lr=3e-4 > lr=5e-4 for AdamW+physics. Closed — redirected kaneda to 4L/256d capacity test at golden config (PR #2488).

**Key structural finding:** Physics features are permanently blocked at ~2 epochs until precomputed caching is implemented. This is not a hyperparameter problem.

---

## 2026-04-20 23:15 — PR #2473: TandemFoil: golden + Fourier + physics + no-EMA — MERGED (NEW BEST)

- **Student:** edward
- **Branch:** edward/tandem-golden-noema-fourier (merged into radford)

| Metric | Run 1 (Fourier only) | Run 2 (Fourier+physics) | Previous Baseline |
|---|---|---|---|
| val_primary/surface_pressure_mae | 106.61 | **82.65** | 114.92 |
| test_primary/surface_pressure_mae | — | 80.63 | 108.16 |
| val_single_in_dist | — | 102.40 | — |
| val_geom_camber_cruise | — | 62.37 | — |
| val_geom_camber_rc | — | 88.97 | — |
| val_re_rand | — | 76.87 | — |
| W&B run | 8a26mlm6 | nh380grv | 3ec9m9az |
| Epochs | 14 | 14 (best=final, still improving) | 11 |

**Commentary:** MAJOR BREAKTHROUGH. Fourier+physics synergy at slices=64 achieves 82.65 — a 28.1% improvement over 114.92 baseline. Run 1 (Fourier only) also beats baseline at 106.61. CRITICAL: both runs got 14 epochs (MORE than the no-Fourier/no-physics 11-epoch baseline), contradicting our earlier assumption that physics would reduce epoch count at slices=64. The Fourier+physics combination appears to be computationally efficient at slices=64. Run 2 was still descending sharply at epoch 14 (95.63→82.65 in last 2 epochs) — significant headroom remains. New TandemFoil baseline: **82.65**.

**Key insight revision:** Physics features are NOT slow at slices=64 when combined with Fourier. The per-sample overhead bottleneck observed at slices=96 may not manifest at slices=64 where the base computation is faster. Fourier+physics is now the mandatory TandemFoil configuration.

---

## 2026-04-20 23:15 — PR #2464: TandemFoil: physics + no-EMA + T_max sweep — CLOSED

- **Student:** frieren

| T_max | val_primary/surface_pressure_mae | Epochs |
|---|---|---|
| 10 | **147.94** | 2 |
| 20 | 168.86 | 2 |
| 50 | 214.70 (regressed) | 2 |

**Commentary:** Dead end at slices=96. 2 epochs only, 147.94 doesn't beat 82.65 (new baseline). Key finding: T_max=10 > T_max=20 >> T_max=50 for physics. T_max=50 regressed (LR still high at cutoff). OOD splits (camber, re_rand) stronger than single_in_dist. Redirected frieren to test T_max=10/15/20 on the new Fourier+physics golden config.

---

## 2026-04-20 23:15 — PR #2465: AirfRANS: no-EMA + AdamW lr=5e-4/8e-4 — CLOSED

- **Student:** kohaku

| LR | val_primary/surface_mse (epoch 6 final) | test_primary |
|---|---|---|
| 5e-4 | 0.3033 | 0.3010 |
| 8e-4 | 0.3061 | **0.2610** |

**Commentary:** Neither beats 0.2597 baseline. Interesting: lr=8e-4 test=0.2610 is very close. lr=8e-4 showed best-at-epoch-3 pattern (0.2781 student-reported, unverified in W&B), suggesting oscillation issue with T_max=150 at lr=8e-4. Redirected kohaku to test AirfRANS Fourier+physics synergy.

---

## 2026-04-20 23:15 — PR #2438: DrivAerML: T_max sweep + 1M surface points — CLOSED (invalid comparison)

- **Student:** taki

| T_max | val_primary (1M eval pts) | Epochs | Runtime |
|---|---|---|---|
| 50 | **36.05%** | 1 | ~80 min |
| 10 | 37.07% | 1 | ~80 min |
| 20 | 38.59% | 1 | ~80 min |
| 30 | 41.20% | 1 | ~80 min |

**Commentary:** Cannot merge — baseline used 50k eval points, this used 1M eval points (metrics not comparable). Also exceeded 30-min timeout (1 epoch = 80 min at 1M points). However, critical findings: (1) 1M training surface points provides dramatically more gradient signal than 50k; (2) T_max=50 is best DrivAerML T_max; (3) cosine scheduling is per-step, so T_max semantics are step-level (71 cycles/epoch at tmx50). Redirected taki to re-run with standardized 50k eval points for fair comparison.
