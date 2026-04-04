# Research Ideas — 2026-04-04 10:30

Generated for: frieren (idle) and tanjiro (currently running #2113 Huber loss — ideas below are queued for next assignment)

Primary target: **p_tan** (tandem OOD pressure MAE, currently 29.1 on 16-seed ensemble / 30.29 on single-model mean — 2.5x worse than p_in=12.1).

## Background and Motivation

Phase 6 experiments have systematically ruled out:
- Capacity bottleneck (#2100 scale-up: p_tan ~30–32 regardless of model size)
- Loss-magnitude issues (#2101 OHEM: no improvement)
- Representational entanglement (#2109 contrastive: no improvement)
- Iterative weight-tied refinement (#2103: −3–6% due to parameter sharing + fewer epochs)
- Per-node adaptive temperature (#2105: catastrophic failure, −22.6% p_tan)

What HAS worked for p_tan:
- Dedicated aft-foil SRF head with surface ID=7 (#2104: p_tan −5.1%)
- PCGrad gradient projection between OOD and in-dist splits (merged earlier)
- Per-head tandem temperature offset (#1168, #1323)
- AoA perturbation augmentation (merged — geometric augmentation precedent)

**Structural insight**: p_tan is an OOD generalization problem. The tandem val split is not a representation of what the model saw in training to the same depth as in-dist data. The two most promising levers are: (1) further surface-specific capacity targeted at the tandem foil configuration, and (2) direct augmentation over the configuration-space axes (gap/stagger) that define the tandem OOD distribution.

---

## Idea 1: Separate SRF Heads for Fore-Foil (ID=6) and Single-Foil (ID=5)

**Priority**: HIGH — for frieren

### Hypothesis

The current surface refinement head (`srf_head`) trained jointly on boundary IDs 5 and 6 must serve two aerodynamically different contexts:
- ID=5: single-foil pressure/suction side (no tandem interaction)
- ID=6: fore-foil in tandem configuration (subject to aft-foil downwash, wake interaction, modified pressure recovery)

Forcing a single MLP to specialize for both surface types creates a conflicted gradient signal. Separating ID=5 and ID=6 into dedicated SRF heads — mirroring the already-proven separation of ID=7 (#2104) — should further reduce the error on tandem configurations by allowing the fore-foil head to specialize on tandem aerodynamics.

### Why This Targets p_tan

The fore-foil (ID=6) sees fundamentally different flow from the single-foil (ID=5). In tandem, the fore-foil's pressure recovery is modified by the presence of the aft foil. Specifically:
- Circulation redistribution between foils
- Modified stagnation point location on the fore foil due to tandem gap/stagger
- Suction peak broadening from aft-foil upwash

A dedicated MLP for ID=6 can learn these tandem-specific surface corrections. This is directly analogous to why the ID=7 aft-foil head helped in #2104 (−5.1% p_tan) — the same principle applies to the fore-foil.

### Connection to Literature

Multi-task airfoil surrogate (Physics of Fluids 2025) validates the principle: dedicated heads per surface region resolve inherent prediction conflicts. ML4CFD NeurIPS 2024 competition also found that region-specific output heads for surface vs. volume are the most consistent gain across architectures.

### Implementation Sketch

In `train.py`, the current `srf_head` applies to nodes where `boundary_id in {5, 6}`. The change is:

1. Create two separate SRF MLPs: `srf_head_5` (same architecture as current, for boundary 5) and `srf_head_6` (for boundary 6).
2. Keep the existing `srf_head` (which the merged #2104 renamed to `srf_head_7`) for boundary ID=7.
3. At forward pass: apply `srf_head_5` to ID=5 nodes, `srf_head_6` to ID=6 nodes, `srf_head_7` to ID=7 nodes.
4. The three heads share the same hyperparameters: 3 layers, hidden=192 (same as current best from #2104 ablation).

New flags needed:
- `--surface_refine_split_fore`: boolean, enables separate head for ID=6 vs ID=5
- No change to existing `--surface_refine`, `--surface_refine_hidden`, `--surface_refine_layers` flags

**Estimated LoC**: ~30 lines (copy srf_head init block, add a second head, split the forward pass mask)

**Interaction with #2104**: This experiment requires #2104 (aft-foil ID=7 head) to be the baseline — the student must confirm they are branching from main (which includes #2104's merge) not from an older checkpoint.

### Risk Assessment

**Low risk.** This is a direct extension of a proven positive result (#2104, −5.1% p_tan). The mechanism is well-understood. The main failure mode is that the fore-foil (ID=6) gradient signal is already adequate given shared parameters — in which case performance should be neutral rather than regressive.

One caution: if the ID=6 head has too few training examples (tandem data is a smaller fraction of training than single-foil), the dedicated head might overfit. Mitigate by using the same dropout as current `srf_head`. If student sees divergence on `val_in_dist`, fall back to shared ID=5+6 head (i.e., revert to #2104 baseline).

### Suggested Training Command

```bash
python train.py \
  --surface_refine \
  --surface_refine_hidden 192 \
  --surface_refine_layers 3 \
  --surface_refine_split_fore \
  --surface_refine_id7 \
  [all other current best hyperparameters]
```

The student should confirm the flag name matches their implementation. Key thing to verify in results: p_tan AND p_in — we want to ensure the single-foil (ID=5) is not degraded by the split.

---

## Idea 2: Gap/Stagger Perturbation Augmentation

**Priority**: HIGH — for tanjiro (next assignment after #2113 completes)

### Hypothesis

The tandem OOD validation split (`val_tandem_transfer`) tests generalization to gap/stagger configurations not seen (or underrepresented) in training. The model has never learned to be invariant to small perturbations in these configuration parameters. By augmenting each training sample with small random perturbations to the gap (x[:,0,21]) and stagger (x[:,0,22]) features, we expose the model to a broader distribution of tandem configurations at training time, directly reducing the distribution gap between training and the OOD tandem eval set.

This is the same principle as the already-merged AoA perturbation augmentation (`--aug aoa_perturb`), extended to the configuration-space dimensions that specifically define tandem OOD.

### Why This Targets p_tan

Gap and stagger are the two primary axes along which the tandem OOD distribution differs from training. The tandem transfer set tests configurations at the edges of (or outside) the training distribution of (gap, stagger) pairs. By adding ±10–20% Gaussian perturbations to these features during training, we:
1. Reduce the effective distribution shift between training and p_tan eval
2. Force the network to produce smooth predictions across small configuration changes (implicit Lipschitz regularization)
3. Do not require new simulation data — the augmentation is purely on the input feature vector

This is supported by 4+ literature sources on domain randomization for configuration-space OOD in aerodynamic surrogates (Cambridge 2025, SIMSHIFT 2025, ML4CFD 2024 best practices, TandemFoilSet ICLR 2026 curriculum learning). The domain randomization approach is identified as the single strongest technique for OOD robustness in coordinate-based aerodynamic surrogates.

### Important Caveat

The perturbation is applied to the feature vector only — the mesh geometry is NOT changed. This is a form of feature-space augmentation (input noise), not geometric augmentation. The model is being asked to predict flow for a slightly different gap/stagger while seeing the mesh of the original configuration. This is deliberately a soft consistency regularization rather than a ground-truth simulation of the perturbed configuration.

The key assumption: for small perturbations (5–15%), the flow field changes smoothly and the model's prediction should also change smoothly. Large perturbations would be physically incorrect (different mesh, different flow). Start with ±10%.

### Implementation Sketch

In `train.py`, in the data augmentation / collation step (where `--aug aoa_perturb` is applied):

1. Add a new aug mode: `gap_stagger_perturb`
2. During training (not validation), for each batch: sample `delta_gap ~ N(0, sigma_gap)` where `sigma_gap = gap_mean * perturb_frac`, add to `x[:,0,21]`; similarly for stagger at index 22.
3. Clip to valid range (gap > 0, stagger within dataset bounds) to avoid out-of-distribution feature values.
4. Apply only to tandem samples (where gap/stagger features are non-zero) — single-foil samples have gap=0 and should not be perturbed.

New flags:
- `--aug gap_stagger_perturb`: enables this augmentation
- `--gap_stagger_perturb_frac 0.1`: perturbation fraction (default 10%)

**Estimated LoC**: ~25–35 lines (augmentation hook, flag parsing, mask for tandem-only samples)

**Interaction with existing augmentation**: Can be combined with `--aug aoa_perturb` since they perturb different feature dimensions. Recommend combining both — the student should run:
- Trial A: `--aug gap_stagger_perturb` only (isolates the effect)
- Trial B: `--aug aoa_perturb --aug gap_stagger_perturb` (combined, likely better)

Use `--wandb_group gap_stagger_aug` to group both trials.

### Risk Assessment

**Medium risk.** The main risk is that feature-space perturbation without corresponding geometry change creates an inconsistent training signal (the model sees features for gap=1.1c but mesh for gap=1.0c). This inconsistency could hurt in-distribution performance.

Mitigation: start with small perturbation fraction (5–10%), monitor p_in carefully. If p_in degrades by more than 2%, reduce `perturb_frac` to 0.05 or disable for single-foil batches (single-foil already can't generalize gap/stagger they never see).

Alternative if feature-space perturbation proves inconsistent: the student could instead implement a "tandem configuration noise" augmentation that adds the same delta to all tandem nodes' gap/stagger feature columns simultaneously (as a batch-level rather than per-sample perturbation), preserving within-sample consistency.

### Suggested Training Command

Trial A (baseline comparison):
```bash
python train.py \
  --aug gap_stagger_perturb \
  --gap_stagger_perturb_frac 0.10 \
  --wandb_group gap_stagger_aug \
  [all other current best hyperparameters]
```

Trial B (combined with AoA):
```bash
python train.py \
  --aug aoa_perturb \
  --aug gap_stagger_perturb \
  --gap_stagger_perturb_frac 0.10 \
  --wandb_group gap_stagger_aug \
  [all other current best hyperparameters]
```

---

## Prioritization

| Rank | Idea | Target Student | Risk | Expected p_tan Gain | LoC | Novelty |
|------|------|----------------|------|---------------------|-----|---------|
| 1 | Separate SRF heads for ID=5 vs ID=6 (fore-foil split) | frieren | Low | −3 to −6% (analogous to #2104's −5.1%) | ~30 | Incremental extension of proven idea |
| 2 | Gap/stagger perturbation augmentation | tanjiro (after #2113) | Medium | −5 to −10% (literature support, novel for this problem) | ~35 | Novel — no prior attempt in this codebase |

**Idea 1** should be assigned first: lowest risk, builds directly on proven #2104 result, minimal implementation complexity.

**Idea 2** is potentially higher upside but carries the feature/geometry inconsistency risk. If AoA perturbation (already merged) is any guide, configuration-space augmentation should help — and gap/stagger are more directly the OOD axes than AoA.

Both ideas are orthogonal to the currently-running experiments (#2112 mesh-density weight for frieren, #2113 Huber loss for tanjiro) — they can be assigned as follow-ons or in parallel if students become idle.

## Dead-End Avoidance

The following ideas have been tried or are structurally similar to failures:
- Larger backbone / more capacity: #2100 ruled out
- OHEM / hard example mining: #2101 ruled out
- Weight-tied iterative refinement: #2103 ruled out
- FiLM conditioning on global features: attempted in #2104, caused p_oodc catastrophe — do NOT revisit
- Contrastive representation loss: #2109 ruled out
- Per-node adaptive temperature (linear): #2105 catastrophic — do NOT revisit without GELU MLP

If both Idea 1 and Idea 2 fail or are running, the next tier to explore would be:
- **GELU-MLP per-node temperature** (fix for #2105's failure mode — use softplus/exp activation instead of linear to guarantee positive temperature)
- **Tandem-conditional global FiLM** at the Transolver slice level (not per-node, just one scale/shift pair per sample conditioned on gap+stagger scalar — much simpler than per-node and avoids #2104's catastrophe)
- **Residual pre-training**: train on single-foil first, then fine-tune on tandem (inspired by TandemFoilSet ICLR 2026 paper's residual pre-training finding)
