<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: skills
-->

# Training Curve Intuition — A Researcher's Reading Guide

Use this as reference material while interpreting training curves. It captures the gut judgments experienced ML researchers apply when they look at loss curves, LR schedules, grad norms, and gradient histograms.

Recommended order: confirm the real `step_key`, compute features with `training_diagnostics.py`, open the PNGs from `curve_plots.py`, then use the heuristics below to write a verdict with step-indexed evidence.

---

## 1. What a healthy loss curve looks like

- **Monotone-ish decrease**: small fluctuations around a falling trend.
- **Diminishing returns**: fast early drop, slow tail — the curve "bends" on log-y.
- **Smooth tail**: rolling std small relative to the run's range (`smoothness < ~0.02`).
- **Grad-norm slowly decays** alongside the loss; no late spikes.
- **Monotonicity_pct** ≥ 55–60% for loss-type metrics (consecutive decreases dominate).

**Red flag signals to look for even when the final number looks good:**
- Fat noise band across the whole run = LR too high or batch-size too small.
- Smooth but flat = already converged OR plateau; check slope at the last 20%.

## 2. Instability patterns

- **Co-occurring spikes**: a loss spike at step S that aligns with a grad-norm spike at the same step is a near-certain sign the optimizer took a bad step (too-high LR, bad batch, fp16 overflow). One or two can be harmless if recovered; more than a handful means instability.
- **Divergence after LR peak**: loss spikes right after warmup ends → warmup too short or peak LR too high.
- **NaN mid-run**: hard instability. Check `nan_inf_count` and the first step; the fix is almost never in the loss — it's in the optimizer, precision, or data pipeline.
- **Escalating spike frequency**: if the spike rate rises in the last 30% of training, the optimizer is losing stability — usually needs gradient clipping or a lower LR floor.

## 3. Overfitting signatures

- Train loss keeps decreasing; val loss **turns upward** in the final 20–30%.
- Train/val gap **widens** in the tail (`val_tail - train_tail` grows).
- A shape with val loss forming a shallow U — the bottom of the U is the right stopping point.
- If a run ends at "lowest final val loss" but earlier val loss was lower, the run **overshot** and should be stopped earlier in subsequent runs.

## 4. Plateau vs convergence

Both look like a flat tail. To distinguish:

- **Convergence**: tail slope ≈ 0, tail std small, loss near its run-wide min. Verdict: done.
- **Plateau**: tail slope ≈ 0, tail std small, but loss is still far from reasonable targets. Verdict: stuck — try LR bump, data quality, curriculum.

Look at `checkpoint_slopes`. If the last 3 segments all sit near zero and the value is below a known "good" threshold, call it **converged**. If it's flat but far from target, call it **plateaued**.

## 5. LR schedule reading

- **Warmup**: roughly 1–10% of training steps for most setups; very long warmup (>20%) only if training is very unstable.
- **Cosine decay**: smooth sinusoidal fall from peak to near-zero. Verify `decay_shape == "cosine"`.
- **Linear decay**: straight line from peak to final. Verify `decay_shape == "linear"`.
- **Flat tail** (LR pinned at a small value for the final segment): often good; lets the model settle.
- **Restarts**: `restart_steps` non-empty means LR went back up after its global peak. Intentional (SGDR) or a bug — ask before assuming.
- **Mismatch with loss**: if loss starts diverging right at the LR peak step, the peak is too high. If loss flatlines as soon as LR enters the decay segment, decay may be too aggressive.

## 6. Grad-norm intuition

- Early grad-norm **high and dropping** is normal — the model is adjusting large weights.
- Stable mid-run grad-norm is good.
- **Late-training grad-norm spikes** indicate instability. A single spike that recovers is fine; repeated spikes in the last 30% → lower LR or clip more aggressively.
- **Sudden drop to ~zero** (`dead_flag` in `grad_norm_features`) often means a layer died (ReLU collapse) or the schedule drove LR to zero prematurely.
- **Kurtosis** > ~3 indicates heavy-tailed update distribution — often a sign of rare catastrophic updates that gradient clipping would catch.

## 7. Reading gradient histogram heatmaps

The heatmap is (layer × step) → a summary stat (default `mean_abs`). A few patterns:

- **Horizontal bands** (one row much brighter/darker than its neighbors) are usually a **layer-type artefact**, not a pathology — embedding layers, LayerNorms, and biases naturally sit at different magnitudes than the dense weights around them. Don't raise an alarm just from banding.
- **Row collapsing to near-zero mid-training** while neighbors stay bright = **dead layer**. A big deal; usually a bad init or a ReLU saturation problem.
- **Late-training widening** (kurtosis rising, `max_abs` blowing out) in several rows = **optimizer instability**.
- **Entire column dark** = grads were ~0 that step — check if the run logged during a no-op step (warmup? eval? clipped grads?).
- **Rows near the output darker than input rows** across all steps = **vanishing gradient** signature (deep models, poor init). Rows near the input dark means exploding gradients elsewhere.

When viewing the heatmap PNG, first pick out structural patterns (bands, dark corridors, columns), then zoom to step ranges of interest using the numeric `grad_histogram_features` DataFrame to drill in.

## 8. Comparing runs

When overlaying runs, weight the verdict by **all four** of:

1. **Final metric** (final_10pct_mean): the obvious ranking dimension.
2. **Smoothness**: a noisy winner is often a worse run — variance in eval samples masks what's happening.
3. **Slope at 80–100%**: if one run is still falling while another flattens, the still-falling run would probably win with more steps.
4. **Spike count**: a clean run generalizes better than a noisy one at the same final metric.

Rules of thumb when ranking:

- Prefer **smoother + slightly higher final** over **noisier + slightly lower final** when the gap is within 2× the final-10%-std.
- A run that is still descending at 100% is evidence to **train longer**, not evidence that the hyperparameters are bad.
- If two runs tie on final metric but one has 10× more spikes, the stable one is the better policy for future work.
- A run with lower peak-grad-norm at the same final metric is generally more robust.

Present the verdict as: **winner → why**, **runner-up → why**, **losers → how they failed** (divergence / overfit / plateau / noise), and a concrete **next action** (train longer, drop LR, add clipping, etc.).
