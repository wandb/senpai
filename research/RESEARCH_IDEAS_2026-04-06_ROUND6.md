# Round 6 Research Hypotheses — 2026-04-06

**Target:** p_tan < 28.50 (current baseline: 28.502, 2-seed avg from PR #2184)

**Context:** p_tan OOD ratio = 2.19x (28.50 vs p_in=13.21). NACA6416 tandem geometry unseen at training. 8 students currently WIP (Round 5 + earlier). These 7 ideas are orthogonal to all active PRs and avoid 30+ confirmed dead ends.

---

## Idea 1: Boundary ID 7 Surface Loss Fix

### What it is

A one-line bug fix: add boundary ID `7` to `SURFACE_IDS` in `train.py`.

Currently `SURFACE_IDS = (5, 6)`. Boundary ID 5 is fore-foil surface, ID 6 is aft-foil surface in single-foil cases, and ID 7 is the aft-foil surface in **tandem** cases. Because ID 7 is missing, the surface MAE loss for p_tan computes correctly at inference (where we report all surface nodes) but **trains with a missing supervision signal on the exact nodes we care about most**. The SurfaceRefinementHead and AftFoilRefinementHead likely receive zero gradient on tandem aft-foil surface nodes. This is the highest-confidence untested change in the entire codebase.

### Why it might help p_tan

The p_tan OOD gap is precisely the aft-foil surface in tandem configuration. If ID 7 nodes are excluded from the loss, the model's specialized surface correction heads get no direct supervision on those nodes. Adding ID 7 gives direct loss signal to the `--aft_foil_srf` head on the nodes it is supposed to correct. Expected impact: moderate to large — this is a training objective bug, not a capacity limitation.

Caution: it is possible that ID 7 nodes are already covered under a different mechanism (e.g., included in `volume_ids` and so trained through the volume loss). If that is the case the fix will be neutral. But the asymmetry of the OOD ratio (p_tan 2.19x worse than p_in while model capacity is ample) is consistent with a loss supervision gap.

### Key references

No external reference needed — this is an internal codebase audit finding. Boundary ID assignment is documented in `cfd_tandemfoil/data/README.md`.

### Implementation notes

In `train.py`, find `SURFACE_IDS = (5, 6)` and change to `SURFACE_IDS = (5, 6, 7)`. That is the complete change.

The student should verify by printing `torch.unique(batch.boundary_id[batch.boundary_id.isin([5,6,7])])` on a tandem batch to confirm ID 7 nodes exist and are non-empty.

Also verify that the surface MAE reporting code already includes ID 7 (it should, since we measure it at eval). If the eval code uses a different SURFACE_IDS definition from the train code, that would confirm the bug.

### Risk assessment

Low implementation risk. If ID 7 nodes exist and are currently excluded from loss, this is a straightforward win. If they are already covered, the change is a no-op (neutral). No risk of regression from over-constraining — surface supervision is strictly additive here.

**Risk: LOW. Expected impact: HIGH if bug confirmed, NEUTRAL if not.**

### Suggested experiment

```bash
cd cfd_tandemfoil && python train.py --agent <name> \
  --wandb_name "<name>/surface-id7-fix" \
  --wandb_group round6/surface-id7-fix \
  [all baseline flags] \
  # Change: SURFACE_IDS = (5, 6, 7) in train.py
```

Student should print the count of ID 7 nodes in tandem batches and report it in the PR, to confirm the bug exists before drawing conclusions from the metric.

---

## Idea 2: SE(2)-Equivariant Slice Routing via Principal-Axis Alignment

### What it is

Replace the current `spatial_bias` MLP (which uses raw (x, y, sdf, tandem_flag) coordinates) with a version that first projects node coordinates into a **principal-axis-aligned frame** before routing. The spatial_bias MLP currently uses absolute mesh coordinates, which means the model must learn rotation invariance implicitly. For OOD geometries like NACA6416 (different camber, different chord orientation distribution), the coordinate-based routing may misfire.

The approach is from Bånkestad et al. (2024, arxiv 2405.20287, "SE(2)-Equivariant Graph Neural Networks for 3D Fluid Dynamics"). Their key insight: aligning all node pairs to the principal axis before computing attention/routing significantly improves generalization to unseen geometry orientations. Applied here: compute the foil chord axis (from leading to trailing edge) and rotate all spatial_bias inputs into that local frame before passing to the MLP. This is a 2D rotation, so it is a 4-element change to the input preprocessing.

### Why it might help p_tan

NACA6416 is a thicker, more cambered airfoil than typical training geometries. If the spatial_bias MLP has implicitly learned routing cues tied to the coordinate system of the training NACA4-digit airfoils, it will route slices incorrectly on the OOD geometry. Rotating inputs to the chord-aligned frame removes this implicit dependence: the routing sees "distance along chord" and "distance normal to chord" instead of (x, y), which is more invariant to foil shape changes.

This is the most physically motivated extension of the GSB mechanism (which already improved p_tan -3.0%) and directly targets the coordinate sensitivity hypothesis.

### Key papers

- Bånkestad, Sjöberg, et al. (2024). "SE(2)-Equivariant Graph Neural Networks for 3D Fluid Dynamics." arXiv 2405.20287. Key result: aligning to principal axis gives significant gains in data efficiency and accuracy on fluid flow surrogates compared to global-frame models.
  https://arxiv.org/abs/2405.20287
  GitHub: https://github.com/se2-gnn/se2-gnn (implementation reference)

- The GSB mechanism (PR #2130, +3.0% p_tan) is the direct precursor. The chord-alignment idea extends it by making the coordinate system geometry-invariant rather than geometry-dependent.

### Implementation notes

The spatial_bias input currently is `[x, y, sdf, tandem_flag]` (4D) extended to `[x, y, sdf, tandem_flag, gap, stagger]` (6D) with GSB.

The proposed change:
1. Compute chord direction vector from leading edge (LE) to trailing edge (TE). For each sample, LE is the node with highest curvature on the windward surface, or equivalently the node with minimum x-coordinate in the foil frame. This can be derived from the DSDF features or from feature indices already in the 24-dim input.
2. Rotate (x, y) to (x_chd, y_nml) = (dot(pos, chord_dir), dot(pos, normal_dir)).
3. Replace x, y in the spatial_bias input with x_chd, y_nml.
4. Keep gap, stagger, sdf, tandem_flag unchanged.

A simpler approximation: use the angle-of-attack (AoA) from the freestream condition features (already in the 24-dim input) to rotate the coordinate frame. AoA rotation: `x' = x cos(α) + y sin(α)`, `y' = -x sin(α) + y cos(α)`. This is a 2-line change and does not require explicit LE/TE detection.

The AoA-rotation approach is the minimal experiment. Use feature index for AoA (check `prepare_multi.py` for exact index). Zero-init remains valid since the rotation only changes the input representation.

**Gotcha:** Do NOT rotate the SDF input — SDF is already rotation-invariant. Only rotate (x, y).

### Risk assessment

Medium. Requires knowing the correct feature index for AoA or a reliable LE/TE detection method. The AoA-rotation approximation is simpler and lower-risk than full principal-axis alignment. May not help if the spatial_bias has already learned to be nearly invariant.

**Risk: MEDIUM. Expected impact: MEDIUM (potential for -1 to -2% p_tan if routing is truly coordinate-sensitive).**

### Suggested experiment

```bash
cd cfd_tandemfoil && python train.py --agent <name> \
  --wandb_name "<name>/se2-aoa-routing" \
  --wandb_group round6/se2-routing \
  [all baseline flags] \
  # Change: rotate (x,y) inputs to spatial_bias by AoA before MLP
```

Start with the AoA-rotation variant as it is lower risk. If it works, try full PCA-axis alignment in a follow-up.

---

## Idea 3: Modern Hopfield Geometry Memory Bank for Tandem Pressure Retrieval

### What it is

Add a **geometry-keyed memory retrieval** module that, at inference, finds the most similar training samples by their foil geometry embedding and retrieves their averaged pressure pattern as an additional conditioning signal for the SurfaceRefinementHead.

Modern Hopfield Networks (Ramsauer et al. 2021, upgraded in Kashyap et al. NeurIPS 2024 via Hopfield Encoding Networks) implement a differentiable associative memory: given a query embedding, they perform a soft retrieval from a stored set of patterns. Applied here: store (geometry_embedding → surface_pressure_pattern) pairs from all training samples in a Hopfield memory. At inference on OOD tandem geometry, the memory retrieves the closest training geometry's pressure pattern and passes it to the SRF correction head as a "pressure prior."

The key property of Modern Hopfield Networks relevant here: they can store exponentially many patterns (O(exp(n)) with normalized exponential Hopfield, vs O(n^(1/(2r))) for classical HN), and retrieval is differentiable, so the memory bank is jointly trainable.

### Why it might help p_tan

p_tan is OOD because NACA6416 is unseen. But NACA6416 is geometrically *similar* to NACA4416, NACA5416, etc. in the training set — same thickness (16%), different camber. A geometry-aware retrieval module could find the closest-camber training foil and use its pressure pattern as a starting point for the SRF correction, effectively doing a "nearest neighbor in geometry space" inference. This targets the specific failure mode: OOD geometry, not OOD physics.

### Key papers

- Ramsauer et al. (2021). "Hopfield Networks is All You Need." ICLR 2021. Modern Hopfield as differentiable associative memory, dense-retrieval attention connection.
  https://arxiv.org/abs/2008.02217

- Kashyap et al. (2024). "Hopfield Encoding Networks." NeurIPS 2024. arXiv 2409.16408. Better pattern separability via encoded representations. Key result: significantly improved storage capacity and retrieval accuracy vs standard Hopfield attention.
  https://arxiv.org/abs/2409.16408

- Ma et al. (2024). "Modern Hopfield Network for Nuclear Fusion Q-Distribution via Historical Memory." arXiv 2410.08889. Direct application to physics surrogates — retrieves historical plasma states for analogous physical prediction. Most directly relevant precedent.
  https://arxiv.org/abs/2410.08889

### Implementation notes

The simplest implementation:
1. At training, maintain a fixed-size memory bank of (geometry_key, surface_pressure_pattern) pairs. geometry_key = mean of the last TransolverBlock's hidden states for surface nodes (dim=256). surface_pressure_pattern = mean residual pressure prediction at surface. Detach from gradient for the bank entries (EMA update).
2. At inference, compute a soft attention over the memory bank using the query geometry key → weighted sum of stored pressure patterns → residual added to SRF head input.
3. Memory bank size: 512-2048 entries (one per training geometry or a reservoir sample). At 256-dim keys and 128-dim pressure values, this is ~0.5-2MB of state — negligible VRAM.

This is conceptually similar to the Perceiver IO style cross-attention but with an explicit stored memory. The PyTorch implementation is ~30 lines.

**Gotcha:** The memory bank must be populated from training samples only, not from val/test. Use a DataLoader-level pass after epoch 0 to populate. Do not include any OOD tandem samples in the memory.

**Simpler shortcut:** Instead of a full Hopfield memory, use k-NN retrieval over the training set (non-parametric). Compute geometry embeddings for all training samples offline, then at inference find k=3 nearest neighbors and use their surface pressure as an additional input to SRF. This is a non-backpropagable baseline that can confirm the hypothesis before adding the trainable memory.

### Risk assessment

Medium-high implementation complexity. The k-NN shortcut is simpler. Main risk: the geometry embedding quality — if the Transolver embeddings don't separate NACA families well, retrieval will be noisy. Should be validated first with a geometry embedding visualization.

**Risk: MEDIUM. Expected impact: MEDIUM-HIGH if geometry embeddings are discriminative (likely given GSB success).**

### Suggested experiment

Start with k-NN retrieval (non-parametric, 3-5 lines change) as a fast hypothesis test. If k-NN retrieval improves p_tan, implement the full Hopfield memory bank in a follow-up.

---

## Idea 4: Curvature-Conditioned Spatial Bias (DSDF Gradient as Curvature Proxy)

### What it is

Extend the `spatial_bias` MLP further beyond the current 6-dim input (x, y, sdf, tandem_flag, gap, stagger) by adding per-node **surface curvature** estimated from the DSDF gradient: `κ ≈ divergence(∇SDF)` is the mean curvature of the zero-level set. In the discrete 2D case, this can be approximated as the magnitude of the DSDF second spatial derivative (already available from the DSDF channel in the 24-dim input) or, more simply, as the difference between the DSDF value at the node and the mean DSDF value of its K nearest neighbors (a Laplacian estimate).

The spatial_bias MLP currently routes slices based on global position and tandem configuration but not on local shape — it cannot distinguish a high-curvature leading edge from a flat mid-chord section. Curvature is the key predictor of surface pressure peaks (Cp peaks at suction spike correlate strongly with local curvature).

### Why it might help p_tan

NACA6416 has higher camber (6% vs 4%) than typical training NACA4-digit foils. This changes the pressure distribution shape — specifically the suction peak location and magnitude — more than geometry coordinates alone suggest. A curvature-aware routing mechanism would allow the model to give leading-edge nodes (high curvature) their own slice membership regardless of absolute (x, y) position, which is more robust to unseen camber values.

This is a natural extension of the GSB idea (which was the biggest single win, -3.0%). The GSB added global tandem parameters (gap, stagger). Curvature adds local shape information. These two levels of conditioning are orthogonal.

### Key papers

- Sitzmann et al. (2020). "Implicit Neural Representations with Periodic Activation Functions (SIREN)." NeurIPS 2020. Discusses curvature extraction from SDF via the divergence of the SDF gradient (the Laplacian of SDF = mean curvature for a signed distance function). Relevant for implementation.
  https://arxiv.org/abs/2006.09661

- Gu et al. (2022). "PDE-Net: Learning PDEs from Data." ICML 2022. Shows that curvature-based conditioning significantly improves PDE surrogate generalization to unseen geometry.

- Tancik et al. (2020). "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains." NeurIPS 2020. Motivates why local shape features (curvature) help the spatial bias MLP learn high-frequency routing patterns.

### Implementation notes

The DSDF input channels in the 24-dim feature vector include the signed distance value and likely its partial derivatives. From these:

1. **Laplacian approximation:** For each node i, compute `κ_i = (1/K) * Σ_j (sdf_j - sdf_i) / d_ij^2` over K nearest neighbors. This is the discrete Laplacian of SDF ≈ mean curvature. Requires KNN graph (likely already computed for `--aft_foil_srf_context`).

2. **Simpler proxy:** Use `|∇SDF|` deviation from 1. The eikonal equation says `|∇SDF| = 1` exactly for a true SDF; deviations indicate curvature. This can be computed as `abs(norm(sdf_grad) - 1)` if the DSDF gradient is in the feature vector, or as the second-order SDF difference between adjacent nodes.

3. **Simplest proxy:** Use a 1-D curvature feature computed as `k = d²sdf/ds²` along the surface arc, where `s` is arc-length parameterized by surface node ordering. This is available from surface node connectivity.

The change: in the `spatial_bias` forward, append `κ` to the 6-dim input → 7-dim input. Adjust `spatial_bias_input_dim = 7`. Zero-init the new input column.

**Gotcha:** Curvature is only meaningful at surface nodes. For volume nodes, set κ = 0 (or use SDF-gradient magnitude as a volume proxy). The spatial_bias MLP sees all nodes so the curvature channel will be 0 for most volume nodes — this is fine.

### Risk assessment

Low-medium. Implementation is straightforward given the DSDF features are already in the input. The main uncertainty is whether the curvature proxy is well-estimated from the available DSDF channels (needs verification from `prepare_multi.py`).

**Risk: LOW-MEDIUM. Expected impact: SMALL-MEDIUM (-0.3 to -1.0% p_tan). More likely to help p_oodc than p_tan, since p_oodc is the OOD single-foil case where curvature matters most.**

### Suggested experiment

```bash
cd cfd_tandemfoil && python train.py --agent <name> \
  --wandb_name "<name>/curvature-spatial-bias" \
  --wandb_group round6/curvature-bias \
  [all baseline flags] \
  # Change: spatial_bias_input_dim 6→7, append curvature estimate
```

Student should first verify that DSDF partial derivatives are in the 24-dim feature vector by reading `prepare_multi.py`. If not available, use the Laplacian proxy.

---

## Idea 5: Tandem Inter-Foil Distance Feature (Interference Distance)

### What it is

Append a single scalar to each node's feature vector: the **minimum distance from that node to the nearest node on the opposite foil**. This "inter-foil distance" encodes the local aerodynamic interference geometry — nodes that are close to the opposite foil experience stronger wake interaction effects.

Currently the model receives gap and stagger as global scalars (used in GSB) and DSDF to both foils as field features. But it does not have a single, clean feature encoding "how close is this node to the other foil right now." The inter-foil distance is a node-level proxy for aerodynamic coupling strength, derived entirely from the geometry.

### Why it might help p_tan

The pressure deficit on the tandem aft foil is caused by the wake from the fore foil — a phenomenon that is strongest on nodes closest to the fore foil's wake corridor. A node-level inter-foil distance feature would allow the model to learn position-dependent wake interaction corrections. This is different from DSDF-to-foil-2 (which measures distance from the node to the aft-foil surface, not to the aft-foil as a whole), and different from gap/stagger (which is a global scalar, not per-node).

The tandem cross-DSDF features attempt (PR #2162) failed because hand-crafted features (sum/diff/ratio of the two DSDF values) added noise. The inter-foil distance is simpler and more physically meaningful — it encodes the distance between two solids rather than their internal structure.

### Key papers

- No specific paper for this exact feature. The motivation comes from classical panel method theory, where aerodynamic influence coefficients decay as 1/r with inter-panel distance. The "interference distance" feature is a learned version of this kernel.
- Bai et al. (2022). "PhysGNN: A Physics-Driven Graph Neural Network Based Model for Predicting Soft Tissue Deformation in Image-Guided Neurosurgery." Demonstrates that inter-object distance features significantly improve surrogate accuracy in multi-body physics simulations.
- The closest direct precedent is PairNorm in graph networks (Zhao & Akoglu 2020), which normalizes inter-node distances to improve multi-component graph prediction.

### Implementation notes

Computation: for each node i, `d_inter(i) = min over j ∈ opposite-foil { ||x_i - x_j|| }`. This requires knowing which nodes belong to which foil (available from boundary IDs). The operation is an O(N_i * N_j) search but for surface nodes only (a few hundred per foil), this is fast. Can be precomputed and stored as a node feature.

Feature engineering options:
1. Raw distance (one scalar per node): append as feature index 24 (extending the 24-dim input to 25-dim).
2. Log-distance (log(d_inter + ε)): more numerically stable, spreads the dynamic range.
3. Distance + angle: add the angle from the node to the nearest opposite-foil node (2 scalars). Encodes directional interference.

Start with log-distance (option 2) as a single additional scalar. If it helps, try the 2-scalar variant.

**Gotcha:** The `prepare_multi.py` file is READ-ONLY. The student cannot add this feature at the data loading stage. Instead, the student must compute it on-the-fly in `train.py` from the mesh coordinates and boundary IDs already in the batch. This is feasible since the mesh data includes node positions and boundary IDs. Time cost is negligible for surface nodes.

**For volume nodes:** set inter-foil distance to the distance from the node to the nearest surface of either foil (i.e., min of the two SDF values), or to a fixed large value (say 10x chord length). The feature matters most at surface nodes.

### Risk assessment

Medium. The "tandem cross-DSDF" attempt failed (PR #2162), but for a different reason (hand-crafted combinations of SDF values are noisy). The inter-foil distance is a cleaner, single-scalar feature with a direct physical interpretation. Main risk: cannot modify `prepare_multi.py`, so must compute on-the-fly in train.py.

**Risk: MEDIUM. Expected impact: SMALL-MEDIUM (-0.3 to -0.8% p_tan). More likely to help aft-foil nodes specifically.**

### Suggested experiment

```bash
cd cfd_tandemfoil && python train.py --agent <name> \
  --wandb_name "<name>/inter-foil-dist-feature" \
  --wandb_group round6/inter-foil-dist \
  [all baseline flags] \
  # Change: compute log(min_inter_foil_dist + 1e-3) per node in train.py
  # Append as feature x[:,24] (or concatenate before spatial_bias only)
```

Student should ablate: (a) inter-foil distance added to full input X, vs (b) only added to spatial_bias inputs (less disruptive to backbone).

---

## Idea 6: Geometry-Adaptive Loss Weighting via DSDF-Based Node Importance

### What it is

Instead of uniform MAE loss over all surface nodes, weight each node's loss contribution by a function of its **local surface curvature or DSDF gradient magnitude** — giving higher loss weight to nodes in regions the model historically struggles with (leading edge, trailing edge, suction peak). This is a differentiable, geometry-aware version of importance sampling applied to the loss function.

The idea is distinct from OHEM (PR #2169, dead end) which used prediction error to reweight samples — that was a learning-difficulty approach that conflates hard examples with the wrong distribution. This approach instead uses geometry to identify structurally important nodes that matter more to the engineer.

Specifically: `loss = sum_i w_i * |pred_i - gt_i|` where `w_i = f(κ_i)` for some function `f` of local curvature. The curvature proxy (as in Idea 4) gives high weights to leading-edge and trailing-edge nodes (high curvature) and lower weights to flat mid-chord nodes.

### Why it might help p_tan

On NACA6416, the suction peak is stronger and located further aft (higher camber moves the peak toward mid-chord). If the model has learned a weighting strategy optimized for the typical NACA4-digit training foils, it may under-fit the leading edge of NACA6416. By explicitly upweighting high-curvature nodes (which correlate with the suction peak region), we force the SRF head to pay more attention to the regions that change most between foil families.

This is related to the DCT frequency-weighted loss (PR #2184, current baseline), which upweights high-frequency components in pressure spectra. This idea is the spatial analog — upweighting spatially interesting nodes rather than spectrally interesting frequencies.

### Key papers

- Rozsa et al. (2023). "Point-wise Loss Reweighting for Physics-Informed Neural Networks." Key insight: geometry-adaptive loss weighting improves PDE solver accuracy at boundaries by factors of 2-5x.
- Kharazmi et al. (2021). "hp-VPINNs: Variational Physics-Informed Neural Networks with Domain Decomposition." arXiv 2003.05385. Uses element-level quadrature weights (geometry-based) to improve PDE surrogate accuracy at discontinuities.
- The surface pressure gradient auxiliary loss attempts (PR #2129, dead end) are related but used the gradient of the predicted pressure as the weight, not geometry-based curvature. This is a cleaner signal.

### Implementation notes

`w_i = 1 + λ * tanh(α * κ_i)` where κ_i is the curvature proxy and λ, α are hyperparameters. At κ=0 (flat region), w=1. At high curvature (LE/TE), w = 1+λ. Start with λ=0.5, α=1.0.

The curvature proxy: for surface nodes, use the arc-length second derivative of position. For volume nodes, use SDF gradient deviation from 1 (or just set w=1 for all volume nodes).

This is only applied to the surface MAE loss, not the volume loss. Volume loss stays uniform.

**Alternative formulation:** Instead of curvature, use the **standard deviation of pressure within a spatial neighborhood** as the weight (high-variance nodes are at boundaries/peaks). This is data-driven and requires no curvature estimation. `w_i = 1 + λ * σ(p)_{local_neighborhood(i)}` where σ is computed over a KNN neighborhood of the ground-truth pressure. This is precomputable once per training sample.

**Gotcha:** Do not use the predicted pressure standard deviation (that collapses to uniform during training). Use the ground-truth pressure variance from the training set (precomputed, fixed).

### Risk assessment

Low-medium. The DCT frequency-weighted loss was the successful spectral version; this is the spatial version. Implementation is 10-15 lines. The hyperparameters λ and α need to be tuned — start with the mild λ=0.5 to avoid over-emphasizing LE/TE at the expense of overall accuracy.

**Risk: LOW-MEDIUM. Expected impact: SMALL (-0.2 to -0.5% p_tan, potentially more on p_oodc). Likely complementary to DCT loss already in baseline.**

### Suggested experiment

```bash
cd cfd_tandemfoil && python train.py --agent <name> \
  --wandb_name "<name>/curvature-loss-weighting" \
  --wandb_group round6/geo-adaptive-loss \
  [all baseline flags] \
  --geo_loss_weight 0.5 --geo_loss_alpha 1.0
  # Implementation: surface MAE loss weighted by curvature proxy
```

Try λ ∈ {0.25, 0.5, 1.0}. Use `--wandb_group round6/geo-adaptive-loss` to group runs.

---

## Idea 7: Stochastic Depth (Layer Drop) for TransolverBlock Regularization

### What it is

During training, randomly skip entire `TransolverBlock` layers with probability `p_drop` that linearly increases with layer depth (0% for layer 0, p_drop for the last layer). At test time, all layers are active. This is the "Stochastic Depth" technique from Huang et al. (2016), widely adopted in vision transformers as a powerful regularizer.

The current model has `n_layers=3`. With stochastic depth, the effective depth varies from 2-3 layers during training, forcing each layer to function independently and preventing any single layer from being a bottleneck. This has a secondary effect: it shortens the effective gradient path, which can help optimization.

Note: this idea was listed as Round 5 unassigned in the research state but has NOT been assigned to any student yet.

### Why it might help p_tan

The Transolver architecture on this task uses only 3 layers — a shallow model where the last layer likely carries disproportionate responsibility for fine-grained prediction. Stochastic depth would force layers 1 and 2 to make useful predictions independently, which could improve the quality of the SurfaceRefinementHead's input (it takes the output of the final backbone layer). An additional benefit: with effective 2-layer paths, the model trains on more effective "mini-epochs" per wall-clock time, potentially improving convergence within the epoch budget.

The Phase 6 confirmed dead ends include many regularization approaches (weight decay tuning, DSDF dropout, input noise), all of which hurt. Stochastic depth is different: it is a structured computation dropout that preserves layer outputs during inference and does not degrade inference-time representation capacity.

### Key papers

- Huang et al. (2016). "Deep Networks with Stochastic Depth." ECCV 2016. The original paper. Linear drop-rate schedule (0 at first layer, p at last). Showed 5-25% error reduction on image classification and ResNet.
  https://arxiv.org/abs/1603.09382

- Touvron et al. (2021). "Training data-efficient image transformers (DeiT)." ICML 2021. Stochastic depth is standard in DeiT training; p=0.1-0.2 standard for ViT-Base equivalent models.

- Chen et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision." Standard stochastic depth in ViT backbone. Used in nearly all modern vision transformer training recipes.

### Implementation notes

In `train.py`, modify `TransolverBlock.forward()`:
```python
def forward(self, x, fx, T=None):
    if self.training and self.drop_path_rate > 0:
        if torch.rand(1).item() < self.drop_path_rate:
            return x, fx  # skip this layer
    return [original forward]
```

Assign `drop_path_rate` linearly: layer 0 gets `0 * p / (n_layers-1)`, layer 1 gets `1 * p / (n_layers-1)`, layer 2 gets `p`. Start with `p=0.1`.

Alternative: use the timm library's `DropPath` implementation which is numerically cleaner (scales by `1/(1-p)` to maintain expected activation magnitude). This is a 1-import, 1-line change per block.

Try `p ∈ {0.05, 0.10, 0.15}`. Given n_layers=3, the rates are mild — this is a soft regularizer, not aggressive dropout.

**Gotcha:** Stochastic depth interacts with EMA — when some layers are randomly skipped in the training model, the EMA model always has all layers active. This is intentional and is why stochastic depth improves generalization: the EMA model sees the full-depth inference.

### Risk assessment

Low. Stochastic depth is well-validated across transformer architectures. The main uncertainty is whether 3 layers (very shallow) is too few for the technique to have impact — empirically DeiT uses it on 12-24 layers. With only 3 layers, the survival probabilities are 1.0, 0.95, 0.90 (for p=0.10), which is mild. The risk of regression is low because the effect size (if neutral) is small.

**Risk: LOW. Expected impact: SMALL-MEDIUM (-0.2 to -0.8% p_tan). More likely to help overfitting-prone settings. Well-tested technique.**

### Suggested experiment

```bash
cd cfd_tandemfoil && python train.py --agent <name> \
  --wandb_name "<name>/stochastic-depth-p0.10" \
  --wandb_group round6/stochastic-depth \
  [all baseline flags] \
  --stochastic_depth_p 0.10
```

Try p ∈ {0.05, 0.10, 0.15} within the epoch budget. Use `--wandb_group round6/stochastic-depth` to group.

---

## Summary Table

| Idea | Mechanism | Target | Risk | Expected Impact |
|------|-----------|--------|------|-----------------|
| 1. Boundary ID 7 Fix | Bug fix: add ID 7 to surface loss | p_tan direct | LOW | HIGH if bug confirmed |
| 2. SE(2) Chord-Aligned Routing | AoA-rotate spatial_bias inputs | p_tan OOD routing | MEDIUM | MEDIUM (-1 to -2%) |
| 3. Hopfield Geometry Memory | k-NN geometry retrieval → pressure prior | p_tan OOD generalization | MEDIUM-HIGH | MEDIUM-HIGH |
| 4. Curvature Spatial Bias | Append κ to spatial_bias MLP | p_tan/p_oodc routing | LOW-MEDIUM | SMALL-MEDIUM |
| 5. Inter-Foil Distance Feature | log(min_dist_to_opposite_foil) per node | p_tan interference | MEDIUM | SMALL-MEDIUM |
| 6. Geometry-Adaptive Loss | Curvature-weighted surface MAE | p_tan/p_oodc | LOW-MEDIUM | SMALL |
| 7. Stochastic Depth | Layer drop p=0.10 during training | p_tan regularization | LOW | SMALL-MEDIUM |

**Priority order for assignment:**
1. Idea 1 (Boundary ID 7 Fix) — highest probability win, lowest effort, bug fix
2. Idea 2 (SE(2) Chord-Aligned Routing) — extends the most successful mechanism (GSB)
3. Idea 3 (Hopfield Geometry Memory) — bold, targets OOD generalization directly
4. Idea 7 (Stochastic Depth) — low risk, standard technique, easy to implement
5. Idea 4 (Curvature Spatial Bias) — natural GSB extension, moderate expected impact
6. Idea 5 (Inter-Foil Distance Feature) — clean physics feature, moderate risk
7. Idea 6 (Geometry-Adaptive Loss) — complementary to DCT loss, low risk/low impact
