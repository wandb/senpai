# Next Batch Research Ideas — 2026-04-03

**Prepared by:** research agent  
**Context:** Baseline is Transolver + Lion + asinh(p, s=0.75) + T_max=160, surface_refine, residual_prediction, pressure_first.  
**Baseline (8-seed):** p_in 13.03, p_oodc 7.83, p_tan 30.29, p_re 6.45  
**Status:** Strong local optimum. All architecture changes, optimizer swaps, and physics loss additions have failed. The two successful levers this phase were target-space compression (asinh) and schedule tuning.

---

## How to read this document

Each idea is evaluated on:
- **Mechanism** — one sentence, no jargon inflation
- **Why it might help** — specific connection to a known weakness
- **Key evidence** — papers with ablations, not just best-case results
- **Implementation notes** — the things that aren't in the abstract
- **Suggested experiment design** — minimal test of the hypothesis
- **Confidence** — how well-supported the evidence is

---

## Idea 1: Packed Ensembles for Inference-Time Variance Reduction

### What it is

Packed-Ensembles (PE) embeds M independent sub-networks inside a single forward pass using grouped linear operations. At inference, the M predictions are averaged. This gives ensemble-quality variance reduction at significantly lower cost than training M separate models.

### Why it might help here

Our 8-seed characterization shows the model has high seed variance: p_in spans 12.2–13.6 (std 0.39), p_tan spans 29.5–31.2 (std 0.47). This variance is larger than the gains from many of our recent experiments. If a 4-member packed ensemble can collapse that variance, the mean prediction should be measurably better than any single seed. The competition most directly comparable to ours — NeurIPS ML4CFD on AirfRANS — specifically validates this: the PE(8,4,1) configuration beat a standard Deep Ensemble on surface pressure MAE with 25% faster training time.

Critically, this is not the same as the snapshot ensemble, SWA, or checkpoint soup that we already tried. Those approaches interpolate parameters along a single training trajectory. Packed Ensembles train truly independent models (different random initializations) but pack them into one network with grouped operations, so their prediction errors are much less correlated.

### Key papers

- **Packed-Ensembles for Efficient Uncertainty Estimation** (Laurent et al., 2023) — arXiv:2312.13403. Validated on AirfRANS specifically (RANS airfoil surface pressure, the closest analogue to our task). PE(8,4,1) outperforms Deep Ensemble with 25% training speedup.
- **A Unified Theory of Diversity in Ensemble Learning** (JMLR 2023) — proves diversity is a hidden term in the bias-variance decomposition for squared-error regression. Lower diversity = higher variance of the mean predictor.
- **Deep Ensembles Secretly Perform Empirical Bayes** (arXiv:2501.17917, 2025) — explains mechanistically why multi-seed averaging reduces expected loss.
- **On Local Posterior Structure in Deep Ensembles** (Jordahn et al., NeurIPS 2025) — plain DEs outperform BNN-augmented DEs on in-distribution regression. Keep it simple.

### Implementation notes

The core trick is replacing `nn.Linear(in, out)` with `nn.Linear(in * M, out * M, groups=M)` (or equivalently, a batched matmul across M heads). Input features are tiled M times; outputs are split and averaged. This works for both linear and attention layers, though attention typically requires a separate implementation per head group.

For our Transolver: the physics_attention slices and the MLP layers are the main targets. The key hyperparameter is M (number of sub-models). The original paper used M=8 with alpha=4 (each sub-model uses 1/4 the channels of the base model, so total params ≈ 2x the single model, not 8x). For our n_hidden=192, M=4 with alpha=4 means each sub-model uses n_hidden=192/4=48 channels — likely too narrow. More practical: M=4, keep per-submodel n_hidden=96 (total effective params ≈ 2x baseline). This must fit in 96 GB VRAM.

The simplest viable implementation that doesn't require architectural surgery: train 4 separate models with different seeds (already trivial with our infrastructure), then average their predictions at inference time. This is not a packed ensemble but achieves the same variance reduction with zero code change. The packed version just makes this faster to train.

**Recommended minimal test:** Train 4 seeds with identical hyperparameters, average predictions at inference for the validation splits. Compare the averaged p_in/p_oodc/p_tan/p_re against the 8-seed mean of single-model results. This is the simplest possible ensemble and requires only a prediction-averaging script, not any architecture changes.

If that works, move to a true packed ensemble implementation to make the inference-time cost deterministic.

### Suggested experiment design

Stage 1 (zero code change): Run 4 seeds of the baseline, save the per-sample predictions at best checkpoint for each val split, average across the 4 seeds. Report the ensemble MAE vs single-seed mean.

Stage 2 (if Stage 1 validates): Implement packed ensemble by using 4 independent prediction heads sharing the Transolver trunk, trained with different random seeds for each head (achieved by splitting the random seed for head initialization only). This preserves the existing training loop.

Stage 3 (if Stage 2 validates): True grouped-linear Packed Ensemble for M=4, alpha=2 (each sub-model uses n_hidden=96, total params slightly less than 2x single model).

### Confidence

Strong evidence from a directly comparable task (AirfRANS, same RANS surface pressure problem). The variance-reduction mechanism is theoretically guaranteed by the bias-variance-diversity decomposition. Stage 1 can be validated with zero code changes. High confidence that Stage 1 gives some gain; moderate confidence the gain is large enough to beat baseline significantly.

---

## Idea 2: Geometry-Conditioned Input Augmentation via AoA/Scale Interpolation (Mixup in physics space)

### What it is

Standard Mixup interpolates training samples in feature space. Physics-space Mixup specifically interpolates between CFD samples that share similar geometry but differ only in angle of attack (AoA) or freestream velocity magnitude. The interpolated sample is physically plausible (a real-valued AoA between the two input samples corresponds to a real flow state), unlike random Mixup in feature space.

### Why it might help here

Our data augmentation currently uses `aoa_perturb` (small AoA jitter on single samples) and `aug_full_dsdf_rot` (rotation of DSDF gradient pairs). These create variations around each sample individually. Physics-space Mixup between two samples at different AoAs creates genuinely new flow states that lie on the physical manifold — unlike feature-space Mixup which creates unphysical interpolations. This is particularly valuable for p_oodc (OOD conditions: extreme AoA/gap/stagger), where the model must generalize to conditions between and beyond the training samples.

The failed Mixup in PR #2031 and PR #1997 was feature-space Mixup (interpolate x, interpolate y) — unphysical for flow fields. Physics-space Mixup only interpolates within a physically plausible subspace (same geometry, different conditions). This is a fundamentally different operation.

From B-GNN paper (arXiv:2503.18638): panel method inviscid Cp provides a physically consistent interpolation baseline — if we have panel method solutions at AoA1 and AoA2, the interpolated solution at (AoA1+AoA2)/2 is close to the true RANS solution at that angle. This gives us a principled way to generate synthetic targets for the interpolated input.

### Key papers

- **Boundary Graph Neural Networks for Airfoil Pressure** (Jena et al., arXiv:2503.18638, 2025) — panel method Cp as physics-consistent input feature achieves 85-88% error reduction vs volumetric models on surface pressure. The key insight is that inviscid flow is an excellent first-order predictor of viscous pressure, especially for attached flow.
- **NeurIPS ML4CFD winner analysis** (arXiv:2506.08516, 2025) — OB-GNN uses 8x surface oversampling; top methods use geometry-aware inductive biases. The competition context validates geometry-conditioned approaches.
- **NeuralFoil** (arXiv:2503.16323, 2025) — panel method + physics augmentation gives 8-1000x speedup over XFoil with competitive accuracy. Demonstrates that inviscid flow solutions are powerful priors.
- **Data-Augmented Few-Shot Neural Emulator** (arXiv:2508.19441, 2025) — generates synthetic training samples via space-filling sampling of local stencil states. Reduces spatial redundancy.

### Implementation notes

The most practical implementation is not full Mixup but **AoA interpolation augmentation**:

1. At each training step, randomly select two samples from the same geometry family (same foil shape, similar Reynolds number) but different AoA.
2. Interpolate their input features at weight lambda (e.g., lambda uniform on [0.3, 0.7]).
3. For the target y: interpolate the asinh-transformed pressure targets at the same lambda.
4. The interpolated x is a plausible new AoA condition; the interpolated y is an approximation of the RANS solution at that intermediate AoA.

The key constraint: only interpolate between samples where the flow topology is similar (both attached, or both separated). Do not interpolate across stall. The existing `aoa_perturb` flag gives a template for where to inject this.

An even simpler version: use the existing `aoa_perturb` augmentation but increase its range from the current small jitter to ±5 degrees, and also interpolate the y targets proportionally. Currently `aoa_perturb` only perturbs x without adjusting y, which is inconsistent. Fixing this inconsistency is the minimal experiment.

**Critical gotcha:** The failed Mixup experiments (PRs #2031, #1997) used random pairs without any physics constraint. The hypothesis here is that physics-constrained pairing (same geometry, similar conditions) produces useful synthetic samples, while random pairing does not. This is a testable distinction.

### Suggested experiment design

Minimal test: Modify `aoa_perturb` to also scale the pressure y-targets by the AoA change ratio (delta_AoA / nominal_AoA). This is a one-line change and tests whether making the augmentation y-consistent helps. Expected cost: zero throughput impact.

Full test: During data loading, build a dictionary of samples grouped by geometry. At each batch, with probability p_aug=0.3, replace one sample with an interpolation between two same-geometry samples. The interpolation weight lambda is drawn uniform from [0.3, 0.7]. Use `--aug phys_mixup --phys_mixup_prob 0.3` as the flag.

### Confidence

Moderate. The physics-Mixup idea is well-motivated but not directly validated on CFD surface pressure. The B-GNN paper's success with panel method features is evidence that physics-constrained inputs help generalization, but that's a different mechanism (features vs. augmentation). The minimal test (consistent AoA perturbation in x and y) is low-risk.

---

## Idea 3: Inference-Time Ensemble via Stochastic Forward Passes (Monte Carlo Dropout at Prediction)

### What it is

Train with standard dropout (p=0.1) kept active at inference time, run K=8-16 stochastic forward passes per sample, average the predictions. This approximates Bayesian model averaging at near-zero training cost overhead and gives a free variance-reduction signal without requiring multiple models to be trained.

### Why it might help here

This is a very different mechanism from the ensembles we have tried. We have tried: EMA (parameter averaging along a single trajectory), SWA (weight averaging at specific checkpoints), snapshot ensemble (checkpoint averaging). All of these average parameters, not predictions. MC Dropout averages predictions from stochastic forward passes, which are more diverse because dropout creates exponentially many implicit sub-models.

The theoretical justification is strong: Gal & Ghahramani (2016) showed MC Dropout is equivalent to approximate variational inference in a deep Gaussian process. The practical justification: for test-time averaging, even 8 stochastic passes with p=0.1 dropout reduce prediction variance substantially on regression tasks.

The key question for our setting: does MC Dropout introduce enough diversity to reduce our seed variance, or does it create too much noise? This depends on the dropout probability. For our ~300K parameter model, p=0.05 is likely safer than p=0.1 — we want a small amount of stochasticity, not large feature dropout.

There is a specific version that works well in our setting: **only apply dropout inside the surface refinement head**, not in the Transolver backbone. The backbone computes shared representations; the surface refinement head is where sample-specific noise is most useful. This also limits the downside risk — if MC Dropout hurts, it only affects the 3-layer surface refinement MLP, not the whole model.

### Key papers

- **Gal & Ghahramani, "Dropout as a Bayesian Approximation"** (ICML 2016) — foundational. Shows K=50 MC passes gives calibrated uncertainty for regression.
- **Divergent Ensemble Networks (DEN)** (arXiv:2412.01193, 2024) — shared backbone + independent stochastic branches. Reduces parameter redundancy vs. full independence while maintaining diversity. Directly applicable to surface refinement head.
- **On Local Posterior Structure in Deep Ensembles** (arXiv:2503.13296, 2025) — plain DEs outperform BNN-augmented DEs for in-distribution regression when ensemble is large enough. Suggests MC Dropout may not beat seed diversity, but it is much cheaper to implement.

### Implementation notes

Training change: Add `nn.Dropout(p=0.05)` after each hidden layer in `SurfaceRefinementHead`. Keep backbone dropout-free.

Inference change: At validation/test time, call `model.train()` only for the surface refinement head, run K=8 forward passes, average the predictions. Total inference cost: 8x the surface refinement head cost (not the backbone cost, which dominates).

Key hyperparameters: p in {0.03, 0.05, 0.1}; K in {4, 8, 16}. Start with p=0.05, K=8.

Important: **do not apply dropout to the pressure prediction pathway** (the channels that feed into the asinh-space loss). The asinh transform amplifies small errors in the extreme pressure range; stochastic dropout on the pressure channel will create large spikes in the transformed space that hurt training stability.

Alternative implementation that is even simpler: Add `nn.Dropout(p=0.05)` to the existing `SurfaceRefinementHead` class, add an `--mc_dropout` flag, and in the validation loop run K passes when the flag is set. This is ~15 lines of code total.

### Suggested experiment design

Baseline: train with `surface_refine` as-is (no dropout). Evaluate single-pass MAE.

Experiment: train with dropout p=0.05 in surface_refine head, evaluate at K=1 and K=8 forward passes. If K=8 MAE < K=1 MAE, the dropout is adding useful variance. Also compare K=8 MC Dropout MAE vs. K=8 independent seed ensemble MAE (from Idea 1) to understand which strategy gives better variance reduction per GPU-hour.

Flag: `--mc_dropout --mc_dropout_p 0.05 --mc_dropout_K 8`

### Confidence

Moderate confidence that MC Dropout in the surface refinement head adds some variance reduction. Low confidence that it beats independent seed ensembles. The value here is cheapness — if it provides even 30% of the benefit of a 4-seed ensemble at essentially zero cost, it is worth deploying as a default.

---

## Idea 4: SAM Optimizer for the Final Training Phase (Already Partially Implemented — Activate It Properly)

### What it is

Sharpness-Aware Minimization (SAM) perturbs model parameters in the gradient direction, then computes gradients at the perturbed point and steps in that direction. This seeks flat minima rather than sharp ones, which generalize better.

**Critical observation: SAM is already implemented in train.py (`cfg.adaln_sam`, class `SAM` at line 1164, rho=0.05).** The flag `--adaln_sam` triggers it. The question is whether it has been properly explored.

### Why it might help here

We are at a strong local optimum. The bias-variance analysis (our 8-seed characterization) shows p_in spans ±0.39 std and p_oodc spans ±0.19 std. A sharp minimum explains high seed variance: different seeds converge to different sharp basins. SAM specifically addresses this by seeking wider, flatter basins where initialization matters less.

SAM's documented OOD improvement is 4.76% average over Adam across domain shift benchmarks (arXiv:2412.05169, 2024). Our p_oodc and p_re are OOD metrics — this is exactly the scenario where SAM's generalization improvement is largest.

However, SAM applied throughout training doubles gradient computation cost, costing ~30% of our epoch budget. The existing `adaln_sam` flag likely applies it to all epochs — this is expensive. A better strategy: **run SAM only in the final 20-30% of training** (after the model has converged to a region near the optimum), then SAM "polishes" the minimum by seeking the flattest point in that neighborhood. This is the approach of ESAM (Efficient SAM) and is much cheaper.

### Key papers

- **SAM for Physics-Informed Neural Networks** (IJCAI 2024) — most directly applicable. Shows SAM helps PDE training, with the key insight that PDE residual loss landscapes (with differential operators) require domain-specific rho tuning. Results: SAM + PINNs improves over Adam + PINNs consistently.
- **SAM OOD Generalization Benchmark** (arXiv:2412.05169, 2024) — 4.76% average OOD improvement. Key: FisherSAM and FriendlySAM variants are best; rho is not very sensitive (rho in [0.01, 0.2] all work, rho=0.05 is fine).
- **Momentum-SAM (MSAM)** (OpenReview Oct 2025) — achieves SAM benefits at near-zero overhead by using Nesterov momentum for the perturbation step rather than a separate gradient computation. Avoids the 2x gradient cost entirely.
- **SAMPa** (arXiv:2410.10683, 2024) — parallelizes SAM's two gradient steps across two devices, achieving 2x wall-clock speedup.
- **SAM Approximation Effects** (arXiv:2411.01714, 2024) — more precise SAM approximation actually degrades performance. The first-order Taylor approximation is the mechanism, not a bug. Standard SAM is fine.

### Implementation notes

The existing `SAM` class in train.py wraps Lion. This is correct — SAM wraps any base optimizer. The current rho=0.05 is within the validated range. The issue is when it's applied.

**Phase-only SAM schedule:** Apply SAM only when `epoch >= 0.75 * total_epochs`. For T_max=160, this means SAM from epoch 120 onward. The model has already converged; SAM "refines" the minimum. Training cost: 2x gradient per step for the last 25% of epochs = 1.25x total cost.

```python
# In train loop, replace optimizer.step() with:
use_sam = cfg.adaln_sam and (epoch >= 0.75 * cfg.cosine_T_max)
if use_sam:
    loss.backward()
    sam_optimizer.perturb()
    optimizer.zero_grad()
    # recompute loss at perturbed point
    pred2 = model(data)
    loss2 = compute_loss(pred2, target)
    loss2.backward()
    sam_optimizer.restore_and_step()
else:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Key hyperparameter: the epoch threshold (0.75 is a reasonable start; 0.5 or 0.6 may also work).

The existing implementation uses `rho=0.05`. For Lion optimizer specifically, the effective scale of parameter updates is different from Adam. Consider rho in {0.02, 0.05, 0.1} as a small sweep.

**Important gotcha:** SAM with Lion may require lower rho than SAM with Adam because Lion's update direction (sign gradient) has norm 1 per parameter, not the Adam-scaled norm. The perturbation `e_w = rho * grad / grad_norm` may be under-scaled. Try rho=0.1 or rho=0.2 as a starting point for Lion.

### Suggested experiment design

Flag: `--adaln_sam --sam_start_frac 0.75 --sam_rho 0.05`

Run 4 seeds. Primary comparison: same-seed runs with and without SAM final phase. If SAM reduces the p_in/p_oodc std, that is success even if the mean is similar.

Also sweep rho: {0.02, 0.05, 0.1, 0.2} in single-seed runs to calibrate.

Expected throughput impact: ~25% slower in the last 25% of training = ~6% total slowdown. Acceptable.

### Confidence

Moderate-high. SAM is already in the codebase (class `SAM`, line 1164), which means it has been contemplated. The IJCAI 2024 paper gives direct evidence for PDE settings. The OOD improvement (4.76% average over Adam) is directly relevant to our p_oodc/p_re metrics. The main uncertainty is whether it interacts well with Lion, which is a sign-based optimizer.

---

## Idea 5: Asymmetric Loss Formulation for the Surface Refinement Head

### What it is

Replace the symmetric L1 loss in the surface refinement head with an asymmetric loss that penalizes under-prediction of peak pressures more than over-prediction. CFD pressure fields have skewed error distributions: the tail events (stagnation point, leading-edge peak pressure) dominate the MAE but are rare. A loss function that overweights these events will push the model to improve on the worst predictions.

This is not just "higher surface weight" (which we have tried). It is changing the loss *shape* to be asymmetric, inspired by quantile regression (pinball loss) and log-cosh loss behavior on heavy-tailed distributions.

### Why it might help here

Our current baseline uses asinh pressure compression to reduce the dynamic range of targets. This helps the OOD splits (p_oodc, p_re) because it down-weights extreme pressures in the loss signal. But for p_in and p_tan (in-distribution), the surface refinement head still struggles with stagnation-point pressure accuracy — these are the highest-magnitude nodes and they dominate the MAE.

The pressure distribution on an airfoil surface has a specific structure: most surface nodes have moderate pressure (suction side, trailing edge), but the leading-edge stagnation point and upper-surface peak suction have much larger magnitudes. If we under-predict these peaks, they contribute disproportionately to MAE. An asymmetric loss that penalizes under-prediction of pressure peaks (negative pressure errors more than positive) will concentrate gradient signal on these nodes.

From the ML4CFD competition (arXiv:2506.08516): OB-GNN used 8x surface oversampling specifically to address the imbalance between surface nodes and volume nodes. An asymmetric loss achieves a similar effect at the node level (overweighting high-pressure nodes) without the computational overhead of changed sampling.

### Key papers

- **Huber loss vs L1 vs MSE for regression** (standard ML knowledge) — Huber is less sensitive to outliers than MSE but more so than L1. For our task, outliers in pressure space (stagnation points) are physically meaningful, not noise — we should NOT smooth them away.
- **Quantile regression / pinball loss** (Koenker & Bassett, 1978; extended in ML to Koenker 2005) — asymmetric loss function: `L(u, tau) = tau * max(u, 0) + (1-tau) * max(-u, 0)`. For tau > 0.5, over-prediction is penalized more; for tau < 0.5, under-prediction is penalized more. For pressure, if we want to accurately predict peaks, use tau=0.4 (penalize under-prediction of high pressure more).
- **Asymmetric loss functions for imbalanced regression** (Steinwart & Christmann 2011; extended in "Is Your Loss Function Weakly Symmetric?" 2024) — asymmetric losses provably improve calibration on right-skewed targets.
- **Log-cosh loss** (practical ML) — similar to Huber but differentiable everywhere. `log(cosh(x))` behaves like MSE for small x and L1 for large x. Avoids the discontinuous gradient at the Huber threshold.

### Implementation notes

For the surface refinement head, the loss is computed as:
```python
surf_loss = F.l1_loss(pred_surf, target_surf)
```

Replace with an asymmetric version for pressure only:
```python
# Asymmetric L1 (pinball loss) for pressure channel
pred_p = pred_surf[:, :, 2:3]   # pressure channel
targ_p = target_surf[:, :, 2:3]
residual = pred_p - targ_p
tau = 0.45  # slightly penalize under-prediction more
asym_p_loss = torch.where(residual >= 0, tau * residual.abs(), (1 - tau) * residual.abs()).mean()

# Standard L1 for velocity channels
vel_loss = F.l1_loss(pred_surf[:, :, :2], target_surf[:, :, :2])
surf_loss = asym_p_loss + vel_loss
```

**Important interaction with asinh:** The asinh transform compresses large pressure values. An asymmetric loss in **transformed space** (asinh space) is equivalent to a different asymmetric loss in **original space** — the exact shape depends on the scale. To target original-space peaks, apply the asymmetric loss **after** the inverse asinh transform (i.e., compute MAE in original pressure units). This is already what the surface refinement head does (it computes the correction to the main model's output in original units, then applies asinh later). Confirm this interaction carefully.

**Alternative: magnitude-weighted L1** — weight each node's loss by `1 + |target_p| / mean_target_p`. This directly overweights high-pressure nodes without asymmetry. Simpler to implement and likely more stable.

Key hyperparameter: tau in {0.4, 0.45, 0.5}. tau=0.5 recovers standard L1 (ablation baseline). Try tau=0.45 first (mild asymmetry).

### Suggested experiment design

Flag: `--asym_surf_loss --asym_tau 0.45`

Also test `--mag_weighted_surf_loss` (magnitude-weighting, simpler baseline for the same idea).

Run 2 seeds for each: standard L1, pinball tau=0.45, magnitude-weighted L1. Compare p_in (primary beneficiary of peak-pressure accuracy improvement) and val/loss.

### Confidence

Moderate. The asymmetric loss is well-motivated theoretically for right-skewed targets (pressure distributions are right-skewed near stagnation). The interaction with asinh transform requires careful implementation. The magnitude-weighted variant is lower-risk and simpler. Neither approach has been directly tested in this setting — the Huber loss experiment (PR #2030) failed, but Huber is symmetric; this idea is specifically asymmetric, which is different.

---

## Summary Table

| Rank | Idea | Expected Gain | Code Complexity | Confidence |
|------|------|---------------|-----------------|------------|
| 1 | Packed Ensembles / Prediction Averaging | Medium-High (variance collapse) | Low (Stage 1 zero code change) | High |
| 2 | Physics-Space AoA Interpolation Augmentation | Medium | Low-Medium | Moderate |
| 3 | MC Dropout in Surface Refinement Head | Low-Medium | Very Low (~15 lines) | Moderate |
| 4 | SAM Final-Phase Optimization | Medium | Low (SAM class already exists) | Moderate-High |
| 5 | Asymmetric/Magnitude-Weighted Surface Loss | Low-Medium | Very Low | Moderate |

---

## Implementation Priority Ordering

**Run in parallel (all are independent):**

1. **Idea 1 Stage 1** (4-seed prediction averaging): Run immediately with zero code change. Tells us definitively whether ensemble variance is exploitable.

2. **Idea 4** (SAM final phase): The SAM class is already in train.py. Just add `--adaln_sam --sam_start_frac 0.75` and test.

3. **Idea 3** (MC Dropout in surface_refine head): ~15 lines of code. Fast to implement and test.

4. **Idea 5** (asymmetric surface loss): ~10 lines of code. Test in parallel with Idea 3.

5. **Idea 2** (physics-space augmentation): More involved. Test after validating results from 1-4.

---

## What NOT to Attempt (Based on Full Literature + Experiment History)

The following have been confirmed ineffective in this setting and should not be revisited:

- Feature-space Mixup (PRs #2031, #1997, #1990) — physically inconsistent interpolations
- Snapshot ensembles / SWA (PR #1807) — failed because checkpoints at different training stages have incompatible representations
- Checkpoint soup / weight-space averaging (PR #1987, #1996) — failed due to permutation symmetry
- SAM applied throughout training (`--adaln_sam` in earlier phases) — likely tested but expensive; the phase-limited version is new
- Any approach requiring MORPH/OmniFluids foundation model pretraining — these require structured grids, incompatible with our unstructured mesh
- DINOZAUR — only applies to FNO tensor multipliers, not Transolver attention

---

## Known Theoretical Floor

With p_in=13.03 and std=0.39 across 8 seeds, if perfect ensemble averaging eliminated ALL seed variance, the irreducible bias component is approximately (13.03^2 - 0.39^2)^0.5 ≈ 13.00. This is a rough lower bound from variance elimination alone. True improvement requires reducing the bias term, which requires better generalization — hence the focus on ideas 2, 4, 5 which improve the learned function rather than just averaging it.

---

## Literature Gaps / Open Questions

1. **Asinh on velocity channels** (suggested by PR #2054 student): `--asinh_velocity` — untested. If velocity distributions are also skewed, asinh compression could help p_tan (which is dominated by velocity errors in the wake region).

2. **Learnable asinh scale**: Let `s` in `asinh(p * s)` be a learned parameter updated by gradient descent. This adapts the compression to the training data distribution, not just hand-tuned.

3. **Asymmetric asinh** (separate scales for positive and negative pressure): Stagnation pressure is large-positive; suction peak is large-negative. A symmetric asinh treats both identically; an asymmetric version could compress each tail differently.

4. **Test-time augmentation with physical symmetries**: Predict at AoA+eps and AoA-eps, average predictions. Unlike y-flip TTA (which failed because the model has no y-flip equivariance), AoA perturbation TTA exploits the smoothness of the physical solution with respect to AoA — a real symmetry.
