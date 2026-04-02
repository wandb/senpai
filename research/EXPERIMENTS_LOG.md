# SENPAI Research Results

## Phase 6 Experiments (2026-04-01 onwards)

### Assigned — Awaiting Results

#### 2026-04-01 23:25 — PR #2006: Phase 6: Muon Optimizer + Gram-NS
- Branch: `phase6/muon-optimizer`
- Student: thorfinn
- Hypothesis: Replace Lion optimizer with Muon (Newton-Schulz orthogonalized gradients) for better optimization landscape navigation. Also test Gram-NS faster variant.
- 8-GPU sweep: Muon at lr=[0.02, 0.01, 0.005], NS steps=[5, 7], Gram-NS variant, + 2 Lion baselines
- **Status: ALL 8 RUNS CRASHED** — zero metrics logged. Likely code bug (shape mismatch in NS iterations or missing parameter routing). Advisor commented with fix guidance. Sent back for debugging.

#### 2026-04-01 23:25 — PR #2007: Phase 6: XSA (Exclusive Self-Attention)
- Branch: `phase6/xsa-attention`
- Student: askeladd
- Hypothesis: Replace standard slice attention with XSA exclusion mechanism to encourage head specialization (different heads → different physical regions).
- 8-GPU sweep: XSA at temperatures=[0.5, 1.0, 2.0], 4 seeds at t=1.0, + 2 baselines
- **Status: ALL 8 RUNS CRASHED.** Some logged partial metrics — catastrophically bad (val/loss 3.76-10.6 vs baseline 0.383). Even baseline runs showed broken metrics (~3.83 val/loss), indicating a code regression that breaks training entirely. Advisor commented with fix guidance. Sent back for debugging.
  
  | Run | val/loss | p_in | Status |
  |-----|----------|------|--------|
  | xsa-t0.5-s42 | 3.761 | 296.4 | crashed |
  | baseline-s42 | 3.830 | 299.6 | crashed |
  | xsa-t1.0-s42 | 4.318 | 322.1 | crashed |
  | xsa-t2.0-s42 | 5.990 | 324.7 | crashed |

---

## Phase 5 Experiments (Still Running)

### In Progress

- PR #1998: Multi-Exit Ensemble (frieren) — average predictions from intermediate Transolver blocks
- PR #2004: Noise Schedule Sweep (fern) — re-verify noise schedules on current baseline
- PR #2003: Warmup & LR Schedule Sweep (edward) — re-tune LR/warmup after architecture changes
- PR #2002: EMA Decay Sweep (nezuko) — sweep EMA decay values
- PR #2001: Learned Per-Channel Loss Weighting (tanjiro) — Kendall uncertainty weighting
- PR #2000: OOD-Focused Training (alphonse) — hard example mining

---

## Phase 5 Experiments (Completed — Summary of Key Results)

### Winners (Merged)
1. **PR #1927: Residual Prediction** — predict correction from freestream. p_oodc -4.7%, p_tan -1.9%
2. **PR #1935: Surface Refinement Head** — dedicated surface MLP. p_re -72.7%, p_tan -8.9%, val/loss -3.3%

### Key Non-Winners (Closed)
- 50+ Phase 5 experiments explored: throughput optimization, data augmentation, architecture variants, loss formulations, ensemble methods, curriculum learning, distillation
- Most incremental tuning has been exhausted — diminishing returns on Phase 5 approaches
- This motivates the shift to Phase 6 (radical new ideas)
