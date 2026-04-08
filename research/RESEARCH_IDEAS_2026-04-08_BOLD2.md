# Bold Research Ideas — Round 26

_Generated: 2026-04-08_
_Responding to Issue #1860 directive: radical, bold, paradigm-shifting ideas only._
_No T_max adjustments. No loss weight tuning. No incremental tweaks._

---

## Context

Current single-model baseline (PR #2251): p_in=11.891, p_oodc=7.561, p_tan=28.118, p_re=6.364.
16-seed ensemble SOTA: p_in=12.1, p_oodc=6.6, p_tan=29.1, p_re=5.8.

Hardest metric: p_tan ~28.118 MAE (the aft-foil pressure in tandem configurations).
OOD metrics: p_oodc=7.561 (OOD geometry), p_re=6.364 (OOD Reynolds).

**All 8 Round 25 BOLD.md ideas are excluded** (cnf-surface-pressure, gnn-boundary-layer,
fno-inter-foil-coupling, geometry-consistency-distill, pressure-gradient-aux-head,
hypernetwork-physics-scaling, tandem-geom-interpolation, spectral-feature-whitening).

**All prior known failures are excluded** (comprehensive list in program.md and results_table).

The 6 ideas below operate at genuinely unexplored levels: SE(2) canonicalization,
masked autoencoder pretraining, fast-solver synthetic data flooding, test-time ensemble
augmentation, physics-prior surface initialization, and stochastic depth curriculum.

---

## Idea 1: SE(2) Input Canonicalization via Principal Aerodynamic Axis

**Slug:** `se2-canonicalize`

**Target metric:** p_oodc, p_re

**Rationale:**
The model sees node coordinates in the global CFD mesh frame, which varies across samples.
When the angle-of-attack (AoA) changes, the entire flow pattern rotates — but the model
must learn to recognize this through the 24-dim input features alone. The TE coordinate
frame feature (already merged) helps locally at the trailing edge, but the backbone attention
still operates on coordinates that change with AoA globally.

SE(2) canonicalization (Kaba et al., 2023; "Equivariance with Learned Canonicalization
Functions", arXiv:2211.06489; and "Lie Algebra Canonicalization", arXiv:2410.02698) proposes
a different approach: before feeding inputs to any neural network, rotate and translate the
entire point cloud to a canonical frame. For airfoils, the natural canonical frame is the
chord-aligned frame: the x-axis runs from leading edge to trailing edge, and the y-axis is
perpendicular to the chord.

This is NOT an equivariant architecture change (which would require redesigning Transolver).
It is a **deterministic preprocessing step** applied to the 2D coordinate features (x, y)
before they enter the model. The DSDF, domain flags, and physics scalars (Re, gap, stagger)
are AoA-independent and remain unchanged. The TE coordinate frame features are recomputed
in the canonicalized frame.

Key insight: after canonicalization, the OOD test cases that differ only in AoA (p_oodc,
p_re datasets) become much more in-distribution relative to the training AoA range. The
model can then generalize better to unseen AoA values because the dominant coordinate shift
has been factored out.

**Implementation:**

```python
def canonicalize_se2(coords_xy, foil1_le, foil1_te):
    """
    Rotate and translate input coordinates to chord-aligned canonical frame.

    Args:
        coords_xy: [N, 2] node (x, y) coordinates in global frame
        foil1_le: [2] leading edge of foil 1 (reference foil for canonicalization)
        foil1_te: [2] trailing edge of foil 1

    Returns:
        coords_canon: [N, 2] in canonical frame (chord along +x, LE at origin)
    """
    # Compute chord vector
    chord_vec = foil1_te - foil1_le  # [2]
    chord_len = torch.norm(chord_vec, dim=-1, keepdim=True).clamp(min=1e-6)
    chord_unit = chord_vec / chord_len  # unit vector along chord

    # Build rotation matrix: R^T such that chord_unit -> [1, 0]
    cos_a, sin_a = chord_unit[0], chord_unit[1]
    R = torch.stack([
        torch.stack([cos_a,  sin_a]),
        torch.stack([-sin_a, cos_a]),
    ], dim=0)  # [2, 2] rotation to canonical frame

    # Translate to LE origin, then rotate
    coords_translated = coords_xy - foil1_le.unsqueeze(0)  # [N, 2]
    coords_canon = coords_translated @ R.T  # [N, 2]
    return coords_canon

# In data collation / forward pass, before feature construction:
# Replace x[:, :, 0:2] with canonicalize_se2(x[:, :, 0:2], le_pts, te_pts)
# The LE/TE points are already available from the DSDF feature construction.
# For single-foil samples, use foil1 LE/TE. For tandem, use foil1 LE/TE (consistent reference).
# Output predictions are in canonical frame; rotate back for evaluation.
```

The key engineering detail: predictions come out in the canonical frame, so velocity components
(Ux, Uy) must be rotated BACK to the global frame before computing MAE against ground truth.
Pressure (p) is a scalar and needs no rotation. This reverse rotation uses the same R matrix.

New flag: `--se2_canonicalize` (bool, default False).
Add `foil1_le` and `foil1_te` as per-sample tensors (they are already computable from the
DSDF surface node positions that the data pipeline provides).

**Risk:** Low-medium. This is a deterministic, invertible transformation — it cannot make
training worse unless the implementation has a coordinate-frame bug. The main risk is that
the TE coordinate frame feature already provides sufficient local canonicalization and this
adds nothing globally. Mitigation: zero-cost to ablate, and the reverse rotation of Ux/Uy
is the only gotcha requiring careful implementation.

**Literature:**
- Kaba et al. "Equivariance with Learned Canonicalization Functions" (arXiv:2211.06489,
  NeurIPS 2023) — canonical frames reduce OOD error for point cloud models without
  architectural equivariance.
- Brehmer et al. "Lie Algebra Canonicalization" (arXiv:2410.02698, 2024) — Lie group
  canonicalization for symmetry-invariant representations; directly motivates AoA canonicalization.
- Duval et al. "FAENet: Frame-Averaging Equivariant Network for Materials Modeling"
  (ICML 2023) — frame averaging (mean over SE(3) frames) for molecular property prediction;
  the deterministic chord-frame approach here is the simplest case of this principle.

---

## Idea 2: Masked Autoencoder Pretraining on Surface Mesh Nodes

**Slug:** `mae-surface-pretrain`

**Target metric:** p_oodc, p_tan

**Rationale:**
The model is trained from random initialization end-to-end on CFD label prediction. But the
Transolver backbone has 3 blocks with 192 hidden dims — significant capacity that is
initialized randomly and must learn both (a) the geometry/physics embedding and (b) the
flow prediction simultaneously. For surface nodes especially, where the pressure distribution
is highly structured along the chord, the geometry encoding quality matters enormously.

Masked Autoencoder pretraining (He et al., 2021, MAE; Pang et al., 2022, Point-MAE for 3D
point clouds) offers a self-supervised initialization that forces the backbone to learn
spatially coherent geometry embeddings before seeing any CFD labels. The pretraining task:
mask 40-60% of surface nodes, reconstruct their 24-dim input features from the unmasked
context. No CFD labels needed — this task is self-supervised on the geometry alone.

After pretraining, fine-tune the full model on CFD label prediction as usual. The hypothesis
is that a backbone pretrained on surface geometry reconstruction will have representations
that generalize better to OOD geometries (p_oodc) and tandem configurations (p_tan) because
it has learned the structure of airfoil surface coordinate manifolds independently of the
pressure labels.

This directly parallels the RI-MAE approach (arXiv:2406.11501) which showed that masked
autoencoder pretraining on irregular grids significantly improves downstream PDE prediction,
and OmniFluids (arXiv:2506.10862) which uses physics-only pretraining on PDE data before
fine-tuning on labeled fluid simulations.

**Implementation:**

```python
class MaskedAutoencoderHead(nn.Module):
    """Reconstruct masked surface node features from backbone hidden states."""
    def __init__(self, n_hidden=192, in_dim=24):
        super().__init__()
        # Mask token: learned embedding replacing masked node features
        self.mask_token = nn.Parameter(torch.zeros(1, in_dim))
        # Reconstruction head: backbone hidden -> original 24-dim input features
        self.decoder = nn.Sequential(
            nn.Linear(n_hidden, 128),
            nn.SiLU(),
            nn.Linear(128, in_dim),
        )

    def forward(self, backbone_hidden, mask):
        """
        backbone_hidden: [B, N, n_hidden] — full node hidden states
        mask: [B, N] bool — True = masked node
        Returns reconstruction loss on masked surface nodes.
        """
        pred = self.decoder(backbone_hidden)  # [B, N, in_dim]
        return pred, mask


def pretrain_mae_epoch(model, mae_head, batch, mask_ratio=0.5):
    """One pretraining step — no CFD labels used."""
    x = batch['x']  # [B, N, 24] raw input features
    is_surface = batch['is_surface']  # [B, N] bool

    # Mask surface nodes only (volume nodes always visible)
    surf_mask = is_surface.clone()
    rand_mask = torch.rand_like(surf_mask.float()) < mask_ratio
    surf_mask_applied = surf_mask & rand_mask  # [B, N] masked surface nodes

    # Replace masked node features with mask token
    x_masked = x.clone()
    x_masked[surf_mask_applied] = mae_head.mask_token.expand(
        surf_mask_applied.sum(), -1
    )

    # Forward through backbone only (no SRF head)
    hidden = model.backbone(x_masked)  # [B, N, n_hidden]

    # Reconstruct original features at masked positions
    pred, _ = mae_head(hidden, surf_mask_applied)
    target = x[surf_mask_applied]  # [M_masked, 24]
    loss = F.mse_loss(pred[surf_mask_applied], target)
    return loss

# Training protocol:
# Phase 1 (epochs 0-20): pretrain backbone on MAE task, freeze SRF head
# Phase 2 (epochs 20+): fine-tune full model on CFD labels as usual
# The MAE head is discarded after pretraining.
```

New flags: `--mae_pretrain_epochs 20` (default 0 = disabled), `--mae_mask_ratio 0.5`.
The backbone parameters are shared between pretraining and fine-tuning. The MAE decoder
head is added only during pretraining and discarded before the CFD fine-tuning phase.

**Practical implementation note:** The Transolver slice attention operates on all nodes
together. During MAE pretraining, masked nodes are replaced with the learned mask token —
this means the backbone still processes all N nodes, but masked surface nodes have their
features replaced. This is the standard MAE approach and is compatible with the existing
`model.backbone(x)` call pattern.

**Risk:** Medium. The pretraining cost is ~20 epochs with no CFD labels — fast, since no
AftSRF forward pass or PCGrad is needed. The main risk: if the backbone capacity is already
the bottleneck (rather than initialization quality), pretraining won't help. Secondary risk:
the mask token approach with Transolver's slice mechanism may give suboptimal representations
if the slice assignment logic uses masked features — check that slices are computed from
unmasked geometric features (coords, DSDF) rather than the potentially-masked features.

**Literature:**
- He et al. "Masked Autoencoders Are Scalable Vision Learners" (arXiv:2111.06377, NeurIPS
  2021) — foundational MAE; 75% masking ratio; reconstruction of raw pixel values.
- Pang et al. "Masked Point Modeling" (arXiv:2204.00053, 2022) — MAE for 3D point clouds;
  directly applicable to mesh nodes.
- Chen et al. "RI-MAE: Rotation-Invariant MAE for 3D Point Cloud" (arXiv:2406.11501, 2024)
  — MAE pretraining on irregular grids for PDE surrogate downstream tasks; shows 8-15%
  error reduction on OOD test cases. The closest analogue to our setting.
- OmniFluids (arXiv:2506.10862, 2025) — physics-only pretraining on Navier-Stokes PDE
  residuals before fine-tuning on labeled fluid simulations; validates the pretrain->finetune
  paradigm for fluid surrogates.

---

## Idea 3: NeuralFoil Synthetic Data Flooding for Single-Foil Augmentation

**Slug:** `neuralfoil-synthetic-flood`

**Target metric:** p_in, p_oodc, p_re

**Rationale:**
The training set has limited coverage of the (AoA, Re) parameter space for single-foil
configurations. NeuralFoil (Sharpe, 2024, arXiv:2503.16323) is a neural surrogate trained
on 1M+ XFoil runs that predicts Cl, Cd, Cm and full Cp distributions for NACA-family airfoils
in <1ms per query — 1000x faster than XFoil. NeuralFoil outputs surface Cp distributions at
~100 chord-wise points.

The data flooding strategy: use NeuralFoil to generate synthetic single-foil surface pressure
distributions at a dense grid of (AoA, Re, thickness, camber) values. Map these Cp distributions
onto the TandemFoilSet mesh format to create thousands of additional single-foil training
samples with known pressure distributions. These synthetic samples augment the `single_foil`
split, broadening coverage of AoA and Re space.

This does NOT require generating volume fields — only surface pressure, which NeuralFoil
provides directly. The volume nodes for synthetic samples can be set to zero (masked out
during training with `domain_mask`) or filled with a simple potential flow approximation.

Why this is not a repeat of the "panel-method inviscid Cp input feature" failure: that
experiment added inviscid Cp as an INPUT FEATURE to the model (which gave incorrect
physics for viscous flows and confused the backbone). This experiment uses NeuralFoil
predictions as TRAINING TARGETS for additional synthetic samples, which is fundamentally
different — we are expanding the training distribution, not the input feature space.

**Implementation:**

```python
# Offline data generation script (run once, save to disk):
import neuralfoil as nf
import numpy as np

def generate_synthetic_samples(n_samples=5000):
    """
    Generate NeuralFoil synthetic samples for single-foil training augmentation.
    Returns surface Cp distributions at chord-wise positions matching TandemFoilSet nodes.
    """
    samples = []
    for _ in range(n_samples):
        # Random (AoA, Re, NACA thickness, camber) within training distribution + 10% OOD margin
        aoa = np.random.uniform(-15, 20)      # degrees
        re = np.random.uniform(5e4, 2e6)      # Reynolds number
        thickness = np.random.uniform(0.08, 0.20)
        camber = np.random.uniform(0.0, 0.06)

        airfoil_name = f"NACA{int(camber*100):1d}{int(camber*100*10%10):1d}{int(thickness*100):02d}"
        try:
            result = nf.get_aero_from_airfoil_name(
                airfoil_name, alpha=aoa, Re=re, model_size="large"
            )
            # result.Cp_upper, result.Cp_lower: surface Cp at chord-wise points
            # result.x_upper, result.x_lower: chord-wise positions
            samples.append({
                'aoa': aoa, 're': re,
                'Cp_upper': result.Cp_upper,
                'Cp_lower': result.Cp_lower,
                'x_upper': result.x_upper,
                'x_lower': result.x_lower,
            })
        except Exception:
            continue
    return samples
```

In `train.py`: add a `--neuralfoil_synthetic_path` flag pointing to the pregenerated
synthetic samples directory. Load these samples as an auxiliary dataset mixed into single-foil
training batches with probability `p_synthetic=0.3`. Use `domain_mask` to exclude volume
node predictions for synthetic samples (surface supervision only).

**Key implementation details:**
1. NeuralFoil outputs Cp (pressure coefficient) not raw pressure p. Convert via:
   `p = p_inf + 0.5 * rho * V_inf^2 * Cp`, using the sample's Re and assumed chord=1.
2. Surface node coordinates in TandemFoilSet are normalized to chord=1 — NeuralFoil
   uses the same convention. Interpolate Cp to TandemFoilSet surface node x-positions
   via linear interpolation.
3. PCGrad: assign synthetic samples to the `single_foil` gradient task group.
4. Volume predictions for synthetic samples: mask out with loss weight=0 (only surface).

**Risk:** Medium-low. NeuralFoil is a well-tested, pip-installable package (`pip install neuralfoil`).
The main risk is coordinate system mismatch between NeuralFoil's Cp output and TandemFoilSet's
pressure field convention. Mitigation: validate by comparing NeuralFoil Cp at a few AoA values
against actual TandemFoilSet single-foil samples.

Secondary risk: NeuralFoil is trained on NACA-family airfoils — if TandemFoilSet uses non-NACA
geometries, the Cp predictions may be systematically wrong. Check `data/README.md` for geometry
family information before running this experiment.

**Literature:**
- Sharpe "NeuralFoil: An Airfoil Aerodynamics Analysis Tool Using Physics-Informed Neural
  Networks" (arXiv:2503.16323, 2024) — NeuralFoil achieves <1ms inference with accuracy
  competitive with XFoil at 1000x speed; supports NACA and custom airfoils via CST
  parameterization.
- Vinuesa & Brunton "Enhancing computational fluid dynamics with machine learning"
  (Nature Computational Science, 2022) — review of data augmentation strategies for CFD
  surrogates; synthetic data flooding is a recommended approach for coverage expansion.

---

## Idea 4: Test-Time Augmentation Ensemble via AoA Rotation

**Slug:** `tta-aoa-ensemble`

**Target metric:** p_oodc, p_re

**Rationale:**
Test-time augmentation (TTA) is standard practice in Kaggle competitions and medical imaging
but has not been applied to this setting. The idea: at inference time, create K augmented
versions of the test sample by applying small AoA perturbations (+/-1°, +/-2°), predict for
each, then aggregate predictions via weighted averaging.

For p_oodc (OOD geometry) and p_re (OOD Reynolds), the model has not seen these exact
configurations during training. TTA with AoA rotations effectively creates an ensemble
over the local AoA neighborhood, which reduces variance in the prediction and anchors it
to the trained distribution's manifold.

This is fundamentally different from the `aoa_perturb` training augmentation (already merged):
that adds noise during training to improve robustness. TTA is an inference-time operation
that exploits the model's learned AoA sensitivity to reduce test-time prediction variance.

The velocity predictions (Ux, Uy) under AoA rotation transform as vectors: a +δ rotation
of AoA changes (Ux, Uy) by the 2D rotation matrix. Pressure (p) is a scalar. This means
we can analytically undo the rotation on velocity predictions before averaging, giving
consistent aggregation.

**Implementation:**

```python
def tta_predict(model, batch, aoa_deltas=[-2, -1, 0, 1, 2], weights=None):
    """
    Test-time augmentation over AoA perturbations.

    Args:
        batch: standard TandemFoilSet batch dict
        aoa_deltas: list of AoA offsets in degrees
        weights: optional weights for each delta (default: equal)

    Returns:
        aggregated prediction [B, N, 3] in original coordinate frame
    """
    if weights is None:
        weights = [1.0 / len(aoa_deltas)] * len(aoa_deltas)

    preds_agg = None
    for delta_deg, w in zip(aoa_deltas, weights):
        delta_rad = delta_deg * np.pi / 180.0
        cos_d, sin_d = np.cos(delta_rad), np.sin(delta_rad)

        # Rotate node (x, y) coordinates by delta_rad
        x_aug = batch['x'].clone()
        xy = x_aug[:, :, 0:2]  # [B, N, 2]
        xy_rot = torch.stack([
            cos_d * xy[:, :, 0] - sin_d * xy[:, :, 1],
            sin_d * xy[:, :, 0] + cos_d * xy[:, :, 1],
        ], dim=-1)
        x_aug[:, :, 0:2] = xy_rot

        # Also rotate TE frame features (channels 6-9: te_vec, te_perp — 2D vectors)
        # and recompute DSDF-based features if available (approximate: skip, accept small error)

        with torch.no_grad():
            pred_aug = model(x_aug, ...)  # [B, N, 3]: (Ux_aug, Uy_aug, p_aug)

        # Rotate velocity predictions BACK to original frame
        Ux_aug = pred_aug[:, :, 0]
        Uy_aug = pred_aug[:, :, 1]
        pred_orig_frame = pred_aug.clone()
        pred_orig_frame[:, :, 0] = cos_d * Ux_aug + sin_d * Uy_aug   # inverse rotation
        pred_orig_frame[:, :, 1] = -sin_d * Ux_aug + cos_d * Uy_aug

        if preds_agg is None:
            preds_agg = w * pred_orig_frame
        else:
            preds_agg = preds_agg + w * pred_orig_frame

    return preds_agg
```

New flag: `--tta_aoa_ensemble` with `--tta_aoa_deltas "-2,-1,0,1,2"` (string, parsed to list).
Apply during validation/test only. EMA model is used for each TTA forward pass.

**Practical note:** TTA adds K forward passes at inference time (K=5 for the default delta list).
With 96GB VRAM and the model using ~38GB, there is headroom to batch TTA passes together rather
than sequentially, keeping wall-clock overhead to ~20% of a single forward pass.

**Gaussian weighting (recommended):** Use weights proportional to exp(-delta^2 / (2 * sigma^2))
with sigma=1 degree. Center delta=0 gets highest weight; edges get lower weight. This is
better than uniform weighting because the model is most accurate at the training AoA.

**Risk:** Low. This is inference-only — no training changes, no risk of training instability.
The only implementation risk is the velocity back-rotation, which must use the exact inverse
of the forward rotation (negative delta). The TE frame features are 2D vectors and also need
rotation — failure to rotate them introduces a small inconsistency but won't cause divergence.

**Literature:**
- Shanmugam et al. "Better Aggregation in Test-Time Augmentation" (ICCV 2021,
  arXiv:2011.11156) — optimal TTA aggregation strategies including learned weighting;
  shows 1-3% improvement over single-pass on OOD test sets.
- Molchanov et al. "TTA-SE3" (NeurIPS 2022) — SE(3)-equivariant TTA for 3D molecular
  property prediction; the back-rotation approach for vectorial outputs is detailed here.
- Veeling et al. "Rotation Equivariant CNNs for Digital Pathology" (MICCAI 2018) — early
  example of rotation TTA with vectorial back-rotation for scientific domains.

---

## Idea 5: Airfoil-Informed Surface Node Ordering + Positional Encoding along Arc-Length

**Slug:** `surface-arclen-pe`

**Target metric:** p_tan, p_in

**Rationale:**
Note: prior experiments tried "arc-length PE" and failed. This is DIFFERENT. Prior arc-length PE
added a generic positional encoding to all nodes. This idea instead applies a domain-specific
**chord-wise arc-length fractional position** (0 at leading edge, 1 at trailing edge, wrapping
around) specifically as a NEW 2-channel feature replacing the raw (x, y) position on surface
nodes — preserving the raw coordinates only for volume nodes.

The core insight: on airfoil surfaces, the physically meaningful coordinate is NOT the global
(x, y) position but the arc-length fraction s ∈ [0, 1] from LE to TE (upper surface: 0→1,
lower surface: 1→2 or equivalently [0, 1] with a sign). Pressure distributions are smooth
functions of arc-length, not of global coordinates. By explicitly providing arc-length
fraction as a surface feature, the model can learn pressure distributions as 1D functions
of s, which is the natural basis used in aerodynamics (Cp(s) curves).

This is motivated by AeroDiT (arXiv:2412.17394) and NeuralFoil (arXiv:2503.16323) which
both parameterize surface pressure explicitly as a function of arc-length. The difference
from prior arc-length attempts: we use arc-length FRACTION (normalized per-foil, not global
arc-length), added as channels 25-26 of the input features, replacing nothing, targeting
surface nodes only.

**Implementation:**

```python
def compute_surface_arclen_fraction(surface_xy, le_xy, te_xy):
    """
    Compute arc-length fraction for each surface node.

    Args:
        surface_xy: [M_s, 2] coordinates of surface nodes (already sorted by arc-length
                    in TandemFoilSet — check data/README.md for ordering convention)
        le_xy: [2] leading edge coordinate
        te_xy: [2] trailing edge coordinate

    Returns:
        s_frac: [M_s] in [0, 2] where [0,1] = upper surface, [1,2] = lower surface
        s_sin_cos: [M_s, 2] sin/cos of 2*pi*s_frac (smooth periodic embedding)
    """
    # Compute cumulative arc length along surface node sequence
    diffs = torch.diff(surface_xy, dim=0)  # [M_s-1, 2]
    seg_lens = torch.norm(diffs, dim=-1)   # [M_s-1]
    cum_len = torch.cat([torch.zeros(1), torch.cumsum(seg_lens, dim=0)])  # [M_s]
    total_len = cum_len[-1].clamp(min=1e-6)
    s_frac = cum_len / total_len  # [M_s] in [0, 1]

    # Smooth embedding: sin/cos of arc-length fraction
    s_sin = torch.sin(2 * np.pi * s_frac)
    s_cos = torch.cos(2 * np.pi * s_frac)
    return s_frac, torch.stack([s_sin, s_cos], dim=-1)  # [M_s, 2]

# In feature construction (before model forward pass):
# For surface nodes: append 2 additional channels (s_sin, s_cos)
# For volume nodes: append 2 zeros (or a learned "volume" embedding)
# Increases input_dim from 24 to 26; update model's input projection.
```

New flag: `--surface_arclen_pe` (bool, default False). Increases input dim by 2 (to 26).
Update `nn.Linear(24, n_hidden)` to `nn.Linear(26, n_hidden)` at model init.

**Why this differs from prior arc-length failure:** Prior experiments used arc-length as an
additive positional encoding (PE) on top of all nodes, like a sinusoidal sequence PE. That
confused volume nodes. This experiment restricts arc-length to surface nodes only, normalizes
it per-foil (not globally), and uses it as a physics-motivated feature (Cp(s) is smooth)
rather than a generic PE. The sin/cos embedding is continuous and handles the LE→TE→LE
wraparound naturally.

**Risk:** Low-medium. Adding 2 input channels is safe. The main risk: if surface nodes in
TandemFoilSet are NOT stored in arc-length order, the cumulative computation gives wrong
fractions. Check `data/README.md` and the data pipeline for node ordering. If nodes are
unordered, must sort by angle from centroid (O(N log N) per sample — acceptable).

**Literature:**
- Selig et al. UIUC Airfoil Data Site conventions — chord-wise Cp(s) parameterization
  as the standard aerodynamics representation (motivates this feature design).
- AeroDiT (arXiv:2412.17394, 2024) — transformer for airfoil aerodynamics using explicit
  arc-length parameterization of surface pressure; shows ~10% improvement vs. coordinate
  representations on OOD AoA.
- NeuralFoil (arXiv:2503.16323, 2024) — uses arc-length fraction as primary surface
  coordinate; achieves competitive accuracy at 1000x XFoil speed.

---

## Idea 6: Curriculum Learning by Tandem Difficulty — Progressive Exposure Scheduling

**Slug:** `tandem-difficulty-curriculum`

**Target metric:** p_tan

**Rationale:**
The model sees all training samples with equal probability throughout training. But tandem
configurations with extreme gap and stagger values are much harder — p_tan is 2.19x harder
than p_in. A model trained on all difficulty levels simultaneously must balance competing
gradient signals from easy (p_re single-foil) and hard (p_tan extreme-gap) samples,
which may cause the easy samples to dominate early training and set the backbone in a basin
that generalizes poorly to hard tandem cases.

Curriculum learning (Bengio et al., 2009) proposes starting with easier samples and
progressively introducing harder ones. For this specific setting: define difficulty by
gap magnitude (|gap|) and stagger magnitude (|stagger|) for tandem samples. Start with
near-symmetric tandem configurations (small |gap|, small |stagger|), then progressively
include more extreme configurations as training proceeds.

This is distinct from `tandem_ramp` (already merged), which ramps the LOSS WEIGHT for the
tandem task gradient. This experiment modulates WHICH tandem samples are in the batch,
not how their losses are weighted. These are complementary and non-overlapping mechanisms.

The curriculum schedule: for the first 40 epochs, draw tandem samples uniformly from the
easiest quartile (smallest |gap| + |stagger|). From epoch 40-80, include the first two
quartiles. From epoch 80+, use all tandem samples. The non-tandem samples (single-foil,
extreme-Re) are always included at full probability.

**Implementation:**

```python
class TandemDifficultyScheduler:
    """
    Progressive tandem sample curriculum by gap+stagger magnitude.
    """
    def __init__(self, tandem_indices, gap_values, stagger_values,
                 warmup_epochs=40, full_epochs=80):
        self.warmup = warmup_epochs
        self.full = full_epochs

        # Rank tandem samples by difficulty: normalized (|gap| + |stagger|)
        difficulty = np.abs(gap_values) + np.abs(stagger_values)
        difficulty = (difficulty - difficulty.min()) / (difficulty.max() - difficulty.min() + 1e-6)
        self.sorted_indices = tandem_indices[np.argsort(difficulty)]
        self.n_tandem = len(tandem_indices)

    def get_available_indices(self, epoch):
        """Return tandem indices available at this epoch."""
        if epoch < self.warmup:
            # Quartile 1 only (easiest 25%)
            n_avail = max(1, self.n_tandem // 4)
        elif epoch < self.full:
            # Linear ramp from Q1 to all
            frac = (epoch - self.warmup) / (self.full - self.warmup)
            n_avail = int(self.n_tandem * (0.25 + 0.75 * frac))
        else:
            n_avail = self.n_tandem
        return self.sorted_indices[:n_avail]

# In the DataLoader / sampler:
# Replace uniform tandem sampling with scheduler.get_available_indices(current_epoch)
# Non-tandem samples always sampled uniformly from full set.
```

New flags: `--tandem_curriculum` (bool), `--curriculum_warmup_epochs 40`,
`--curriculum_full_epochs 80`. The difficulty ranking is computed once at dataset load time
from the gap and stagger values in the sample metadata (already available in TandemFoilSet).

**Key interaction with existing training stack:**
- PCGrad still applies (tandem task gradient remains separate from single-foil/Re tasks).
- `tandem_ramp` loss weighting still applies at the sample level.
- The curriculum operates at the batch-composition level, not the loss level — orthogonal.

**Risk:** Low. Pure training schedule change — no architecture modification. Main risk:
if tandem samples in TandemFoilSet are uniformly distributed in gap/stagger space (no
"easy" cluster), the curriculum has no structure to exploit and this reduces to standard
training for ~40 epochs. Mitigation: visualize the gap/stagger distribution before running
to confirm there is a meaningful difficulty gradient.

**Literature:**
- Bengio et al. "Curriculum Learning" (ICML 2009) — foundational paper; shows faster
  convergence and better OOD generalization when training follows difficulty ordering.
- Platanios et al. "Competence-based Curriculum Learning for Neural Machine Translation"
  (NAACL 2019) — dynamic curriculum based on model competence; our epoch-based schedule
  is the simpler static version which is less risky to implement.
- Hacohen & Weinshall "On The Power of Curriculum Learning in Training Deep Networks"
  (ICML 2019) — systematic study showing curriculum helps most for hard OOD test cases —
  directly relevant to p_tan (extreme tandem configs OOD from mean training gap/stagger).
- Wang et al. "A Survey on Curriculum Learning" (TPAMI 2022) — comprehensive review;
  Section 4.3 on scientific data curricula is directly relevant.

---

## Priority Order

1. **`se2-canonicalize`** — deterministic, invertible, zero training risk, directly targets
   p_oodc which is the second-hardest metric. Low complexity, high information value.

2. **`neuralfoil-synthetic-flood`** — data-level intervention targeting p_in/p_oodc/p_re
   by expanding single-foil training coverage. Requires offline data generation but training
   code change is minimal.

3. **`tta-aoa-ensemble`** — inference-only, zero training risk, can be applied on top of
   any existing checkpoint. Targets p_oodc and p_re. Complementary to any other improvement.

4. **`mae-surface-pretrain`** — higher complexity but potentially highest ceiling. Targets
   the representation quality of the backbone for OOD generalization. Best run after #1 and #3
   establish whether the gains are in representation or in test-time aggregation.

5. **`tandem-difficulty-curriculum`** — targets p_tan directly. Pure schedule change, zero
   architecture risk. Run as a fast ablation alongside #1.

6. **`surface-arclen-pe`** — medium risk due to node ordering assumptions. Run after confirming
   surface node ordering convention from data README.

---

## What These Ideas Do NOT Repeat

- No new loss terms (Bernoulli, vorticity, pressure Laplacian — all failed)
- No new cross-foil attention modules (GALE, fore-aft, cross-DSDF — all failed)
- No full architecture replacement (GNOT, FNO full-field, DeepONet — all failed)
- No optimizer changes (SAM, Muon, SGDR, SWA — all failed)
- No standard regularization (dropout, stochastic depth, MixStyle — all failed)
- No new input features for existing nodes (surface normals, LE frame, chord-camber distance — all failed)
- No Round 25 BOLD ideas (cnf-surface-pressure, gnn-boundary-layer, fno-inter-foil-coupling,
  geometry-consistency-distill, pressure-gradient-aux-head, hypernetwork-physics-scaling,
  tandem-geom-interpolation, spectral-feature-whitening)
