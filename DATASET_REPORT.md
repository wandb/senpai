<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# TandemFoilSet Dataset Report

**Source:** <https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/KTXSCU>
**Location:** `/mnt/new-pvc/datasets/tandemfoil/`
**Total size:** ~178 GB across 26 pickle files
**Format:** Lists of `torch_geometric.data.Data` objects (loaded via `torch.load(..., weights_only=False)`)

---

## 1. File Inventory

### 1.1 Cruise — Random Fields (tandem foils, variable Re)

| File | Samples | Nodes (mean) | Size |
|------|---------|-------------|------|
| `cruise_randomFields_ive_Part1.pickle` | 300 | 209,073 | 6.0 GB |
| `cruise_randomFields_ive_Part2.pickle` | 300 | 207,732 | 6.0 GB |
| `cruise_randomFields_ive_Part3.pickle` | 300 | 207,436 | 6.0 GB |
| `cruise_randomFields_mgn_Part1.pickle` | 300 | 209,073 | 6.0 GB |
| `cruise_randomFields_mgn_Part2.pickle` | 300 | 207,732 | 6.0 GB |
| `cruise_randomFields_mgn_Part3.pickle` | 300 | 207,436 | 6.0 GB |
| `cruise_randomFields_mgn_extrap_Part1.pickle` | 300 | 209,073 | 6.8 GB |
| `cruise_randomFields_mgn_extrap_Part2.pickle` | 300 | 207,732 | 6.7 GB |
| `cruise_randomFields_mgn_extrap_Part3.pickle` | 300 | 207,436 | 6.7 GB |

**Subtotal: 2,700 samples (900 unique simulations x 3 file variants)**

Parts 1/2/3 correspond to **different Reynolds numbers**: Re=1,475,000 / Re=4,445,000 / Re=802,000. The `ive`, `mgn`, and `mgn_extrap` variants share identical CFD data but attach different pre-computed baseline model predictions (see Section 3.7).

### 1.2 Cruise — Re=500 (tandem foils, fixed Re, fixed AoA)

| File | Samples | Nodes (mean) | Size |
|------|---------|-------------|------|
| `cruise_Re500_aoa0_ive_Part{1,2,3}.pickle` | 261+261+262 | ~349,000 | 9.8 GB each |
| `cruise_Re500_aoa0_mgn_Part{1,2,3}.pickle` | 261+261+262 | ~349,000 | 9.8 GB each |
| `cruise_Re500_aoa5_ive_Part{1,2,3}.pickle` | 261+261+262 | ~349,000 | 9.8 GB each |
| `cruise_Re500_aoa5_mgn_Part{1,2,3}.pickle` | 261+261+262 | ~349,000 | 9.8 GB each |

**Subtotal: 3,136 samples (784 unique simulations x 2 AoA x 2 file variants)**

Parts 1/2/3 sweep **different NACA leading-foil families**: 0006 / 1408 / 2418. The `aoa0` vs `aoa5` variants differ only in angle of attack (0° vs 5° for both foils). The `ive` vs `mgn` variants share identical CFD data but attach different baseline predictions (see Section 3.7).

### 1.3 RaceCar — Single Element (single foil, variable Re)

| File | Samples | Nodes (mean) | Size |
|------|---------|-------------|------|
| `raceCar_single_randomFields.pickle` | 899 | 85,964 | 6.5 GB |

**Subtotal: 899 samples**

### 1.4 RaceCar — Tandem (dual foils, variable Re)

| File | Samples | Nodes (mean) | Size |
|------|---------|-------------|------|
| `raceCar_randomFields_mgn_Part1.pickle` | 300 | 127,154 | 3.7 GB |
| `raceCar_randomFields_mgn_Part2.pickle` | 300 | 129,507 | 3.7 GB |
| `raceCar_randomFields_mgn_Part3.pickle` | 300 | 125,637 | 3.6 GB |

**Subtotal: 900 samples**

Parts 1/2/3 correspond to **different rear-foil chord ratios** (`resize`): 0.35 / 0.45 / 0.50, and different fixed front-foil NACA profiles: 2412 / 6416 / 9412.

---

## 2. Grand Totals

| Subset | Unique simulations | Total files | Total samples (with variants) |
|--------|-------------------|-------------|-------------------------------|
| Cruise randomFields | 900 | 9 | 2,700 |
| Cruise Re500 | 784 | 12 | 3,136 |
| RaceCar single | 899 | 1 | 899 |
| RaceCar tandem | 900 | 3 | 900 |
| **Total** | **3,483** | **26** | **7,635** |

> **Important:** The `ive`/`mgn`/`mgn_extrap` variants for the same Part share identical CFD ground truth (`y`, `pos`, `edge_index`, etc.). They only differ in the attached pre-computed prediction keys. For downstream training you only need **one variant per simulation** (the ground truth is the same).

**Deduplicated unique simulations: 3,483**

---

## 3. Per-Sample Data Schema

Each sample is a `torch_geometric.data.Data` graph object with the following fields:

### 3.1 Core Mesh Fields

| Field | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `pos` | `(N, 2)` | float32 | Node coordinates in 2D (x, z) |
| `edge_index` | `(2, E)` | int64 | Graph connectivity (undirected edges) |
| `boundary` | `(N,)` | uint8 | Boundary condition type per node |
| `zoneID` | `(N,)` | float32 | Mesh zone identifier |

### 3.2 Target Field (Ground Truth from CFD)

| Field | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `y` | `(N, 3)` | float16 | **Target: [Ux, Uy, p]** — velocity components and kinematic pressure (p/ρ, m²/s²) |

### 3.3 Input Features (Geometric Encodings)

| Field | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `saf` | `(N, 2)` | float16 | Signed arc-length features — encodes each node's position relative to the airfoil surface(s) |
| `dsdf` | `(N, 8)` | float16 | Distance-based shape descriptor field — 8-channel encoding of distance to airfoil geometry features |

### 3.4 Flow Condition Metadata

| Field | Type | Description |
|-------|------|-------------|
| `flowState` | dict (24–25 keys) | Freestream flow conditions (see Section 4) |
| `AoA` | float or list[2] | Angle(s) of attack in degrees (per foil) |
| `NACA` | list[str] (len 1 or 2) | NACA 4-digit airfoil profile code(s) |

### 3.5 Geometry Metadata

| Field | Type | Present in | Description |
|-------|------|-----------|-------------|
| `af_pos` | Tensor `(n_foils, 2)` float32 | All | Airfoil reference position(s) |
| `gap` | float | Tandem only | Chordwise gap between foils |
| `stagger` | float | Tandem only | Cross-stream stagger between foils |
| `height` | float | RaceCar only | Ground clearance height |
| `hc_net` | float | RaceCar only | Height-to-chord ratio (net) |
| `hcb_net` | float | RaceCar tandem | Rear foil height-to-chord ratio |
| `scb_net` | float | RaceCar tandem | Rear foil stagger-to-chord ratio |
| `resize` | float | RaceCar tandem | Rear foil chord ratio (0.35/0.45/0.50) |

### 3.6 Pre-computed Baseline Predictions

| Field | Shape | dtype | Present in |
|-------|-------|-------|-----------|
| `y_est_model_ive_*` | `(N, 3)` | float32 | `ive` files |
| `y_est_model_mgn_*` | `(N, 3)` | float32 | `mgn` files |
| `y_est_model_*_extrapRE` | `(N, 3)` | float32 | `mgn_extrap` files |
| `y_est_model_*_extrapAOA` | `(N, 3)` | float32 | `mgn_extrap` files |

These are predictions from pre-trained single-element surrogate models applied to the tandem configurations. Useful as baselines or as input features for transfer learning.

### 3.7 File Variant Suffixes: IVE vs MGN vs MGN_extrap

The `ive`, `mgn`, and `mgn_extrap` suffixes denote which **baseline surrogate model's predictions** are attached to the file. The underlying CFD ground truth (`y`, `pos`, `edge_index`, etc.) is identical across variants for the same Part — only the `y_est_*` prediction field and some extra metadata differ.

| Suffix | Baseline model | Mesh resolution (typical) | Extra fields |
|--------|---------------|--------------------------|--------------|
| `ive` | **IVE** — Implicit Volume Estimator (paper's own method) | ~209K–349K nodes | — |
| `mgn` | **MGN** — MeshGraphNet (GNN baseline from DeepMind) | ~114K–209K nodes | `hc_net`, `hcb_net`, `scb_net`, `height`, `resize` |
| `mgn_extrap` | **MGN** with extrapolation predictions | Same as `mgn` | Same as `mgn`, plus `y_est_*_extrapRE` and `y_est_*_extrapAOA` |

**Key differences:**
- **IVE files have denser meshes** than MGN files for the same simulation (e.g., cruise: ~209K vs ~114K nodes). The two variants use different mesh resolutions tailored to each baseline model.
- **MGN files include extra geometry metadata** (`hc_net`, `hcb_net`, `scb_net`, `height`, `resize`) that the IVE files omit.
- **MGN_extrap files** additionally store predictions from a model trained on a different Re or AoA range, enabling extrapolation analysis.

> **For downstream training:** Pick one variant per Part (e.g., always `mgn` for smaller meshes, or `ive` for denser meshes). You only need the ground truth `y` — the `y_est_*` fields are optional baselines.

---

## 4. FlowState Dictionary

The `flowState` dict encodes the freestream boundary conditions:

| Key | Type | Example (cruise randomFields) | Description |
|-----|------|------|-------------|
| `Re` | float | 1,475,000 | Reynolds number |
| `Umag` | float | 21.55 | Freestream velocity magnitude (m/s) |
| `Ux` | float | 21.55 | x-velocity component |
| `Uy` | float | 0.0 | y-velocity component |
| `Uz` | float | 0.0 | z-velocity component |
| `Uinlet` | list[3] | [21.55, 0.0, 0.0] | Inlet velocity vector |
| `nu` | float | 1.461e-5 | Kinematic viscosity (m²/s) |
| `nuT` | float | 1.461e-8 | Turbulent viscosity |
| `rhoTotal` | float | 1.225 | Density (kg/m³) |
| `pTotal` | float | 0.0 | Total pressure |
| `c` | float | 1.0 | Chord length (single-element) |
| `c_f` | float | — | Front foil chord (tandem raceCar only) |
| `c_b` | float | — | Back foil chord (tandem raceCar only) |
| `c_eff` | float | — | Effective combined chord (tandem raceCar only) |
| `AOA` | float | 0.0 | Reference AoA (always 0 — actual AoA is in `AoA` field) |
| `radAOA` | float | 0.0 | AoA in radians |
| `omega` | float | 2.155 | Specific dissipation rate (freestream) |
| `omegaInf` | float | 2.155 | Same as omega |
| `omega_wall` | float | 29,220,000 | Wall omega boundary condition |
| `k` | float | 3.148e-8 | Turbulent kinetic energy |
| `kInf` | float | 3.148e-8 | Same as k |
| `y_l` | float | 2e-5 | First cell height (boundary layer) |
| `Re_L` | float | 29,500,000 | Reynolds number based on domain length |
| `domain` | int | 20 | Domain size (chord lengths) |
| `maxLength` | int | 20 | Max domain extent |
| `cutX` | float | 1.25 | Domain cut x-position |
| `cutZ` | float | 0.0 | Domain cut z-position |

---

## 5. Boundary Condition Types

The `boundary` field (uint8) encodes 8 distinct boundary types:

| Value | Likely meaning |
|-------|---------------|
| 0 | Interior / field nodes |
| 1 | Inlet |
| 2 | Outlet |
| 3 | Top wall |
| 4 | Bottom wall |
| 5 | Airfoil surface (foil 1, upper/main) |
| 6 | Airfoil surface (foil 1, lower/trailing edge) |
| 7 | Airfoil surface (foil 2, tandem only) |

> Note: Single-element files use values 0–6; tandem files use 0–7.

---

## 6. Zone IDs and Overset (Chimera) Mesh Structure

| Value | Description |
|-------|-------------|
| 0 | Background mesh (coarse, covers full domain) |
| 1 | Foil 1 refinement patch (dense, around first airfoil) |
| 2 | Foil 2 refinement patch (dense, around second airfoil — tandem only) |

Single-element files: `{0, 1}`. Tandem files: `{0, 1, 2}`.

### 6.1 Overset Mesh Layout

All datasets use an **overset (chimera) mesh** where the refinement zones spatially overlap with the background mesh. The CFD solver computes on each zone independently and interpolates at overlap boundaries.

```
┌─────────────────────────────────────────────────┐
│  Zone 0 — coarse background (full domain)       │
│                                                   │
│       ┌──────────────┐   ┌──────────────┐        │
│       │  Zone 1       │   │  Zone 2       │       │
│       │  (dense,      │   │  (dense,      │       │
│       │  foil 1)      │   │  foil 2)      │       │
│       └──────────────┘   └──────────────┘        │
│                                                   │
└─────────────────────────────────────────────────┘
```

**This means points from different zones overlap in space.** Zone 0 has coarse background points underneath the dense zone 1/2 patches. The overlap counts per dataset:

| Subset | Zones | Zone 0 | Zone 1 | Zone 2 | Zone 0 pts in Zone 1 bbox | Zone 0 pts in Zone 2 bbox |
|--------|-------|--------|--------|--------|--------------------------|--------------------------|
| RaceCar single | 0, 1 | 20,780 | 59,692 | — | 6,057 | — |
| RaceCar tandem (mgn) | 0, 1, 2 | 31,327 | 61,432 | 21,292 | 7,280 | 4,084 |
| Cruise randomFields | 0, 1, 2 | 101,017 | 61,104 | 61,104 | 8,458 | 8,462 |
| Cruise Re500 | 0, 1, 2 | 318,854 | 15,300 | 15,300 | 4,996 | 4,990 |

> **Implication for training:** The overlapping points mean two nodes at nearly the same spatial location can have different field values (one from the background solve, one from the refinement solve). This is normal for overset CFD — the refinement zone values are authoritative near the airfoil. For NN surrogates, the model must learn to predict consistent values for both overlapping nodes.
>
> **Implication for visualization:** Naively triangulating all points together creates artifacts at zone boundaries due to the density jump and spatial overlap. For clean plots, either (a) remove zone 0 points inside zone 1/2 bounding boxes, or (b) triangulate each zone separately.

---

## 7. Value Ranges and Statistics

### 7.1 Target Field `y` — [Ux, Uy, p]

| Subset | Re range | y min | y max | y mean | y std |
|--------|----------|-------|-------|--------|-------|
| Cruise randomFields Part1 | 1.475M | -1,278 | 233 | 1.9 | 55 |
| Cruise randomFields Part2 | 4.445M | -2,360 | 2,118 | 5.3 | 304 |
| Cruise randomFields Part3 | 802K | -300 | 69 | 2.1 | 17 |
| Cruise Re500 | 500 | -0.054 | 0.155 | 0.045 | 0.066 |
| RaceCar single | ~700K–2M | -874 | 467 | -23 | 141 |
| RaceCar tandem | ~700K–2M | -4,277 | 668 | -65 | 235 |

> **Key insight:** The `y` values span vastly different magnitudes across Re regimes. The omega (vorticity) channel dominates the range. **Normalization is essential** — ideally per-Re or per-sample.

### 7.2 Node Counts

| Subset | Min nodes | Max nodes | Mean nodes | Std |
|--------|-----------|-----------|------------|-----|
| Cruise randomFields | 179,524 | 242,577 | ~208,000 | ~13,500 |
| Cruise Re500 | 347,047 | 349,617 | ~349,000 | ~300 |
| RaceCar single | 74,782 | 90,173 | 85,964 | 1,996 |
| RaceCar tandem | 88,925 | 163,642 | ~127,000 | ~17,000 |

All meshes are **variable-size** (no two samples have the same node count), making this naturally suited for graph-based methods.

### 7.3 Edge Counts (approximate)

| Subset | Edges per sample |
|--------|-----------------|
| Cruise randomFields | ~370K–458K |
| Cruise Re500 | ~700K |
| RaceCar single | ~165K |
| RaceCar tandem | ~235K–255K |

Average degree ≈ 2E/N ≈ 3.5–4.1 (consistent with triangular mesh connectivity).

---

## 8. Parameter Space Coverage

### 8.1 Reynolds Number

| Subset | Re values |
|--------|-----------|
| Cruise randomFields | 802K, 1.475M, 4.445M (one per Part) |
| Cruise Re500 | 500 (fixed) |
| RaceCar single | ~700K–2M (varies per sample via `flowState.Re`) |
| RaceCar tandem | ~700K–2M (varies per sample) |

### 8.2 NACA Profiles

- **Cruise randomFields:** 2-foil combos, leading foil fixed per Part (0006 / 2408 / 4408), trailing foil varies
- **Cruise Re500:** 2-foil combos sweeping systematic NACA pairs (0006–0024 / 1408–1424 / 2418–4424)
- **RaceCar single:** Single NACA, sweeps 2205–2209+ range
- **RaceCar tandem:** 2-foil combos, front foil fixed per Part (2412 / 6416 / 9412)

### 8.3 Angle of Attack

| Subset | AoA range |
|--------|-----------|
| Cruise randomFields | ~[-8°, +8°] (random per sample, same for both foils) |
| Cruise Re500 aoa0 | 0° (fixed) |
| Cruise Re500 aoa5 | 5° (fixed) |
| RaceCar single | ~[-10°, +10°] (random per sample) |
| RaceCar tandem | Two independent AoAs per sample |

### 8.4 Tandem Foil Geometry

| Parameter | Cruise randomFields | RaceCar tandem |
|-----------|-------------------|----------------|
| gap | ~[-0.8, +0.5] | ~[0.4, 1.3] |
| stagger | ~[0.7, 2.0] | ~[0.7, 1.0] |
| resize | — | 0.35, 0.45, 0.50 |

---

## 9. Recommendations for NN-CFD Surrogate Training

### 9.1 Deduplication

The `ive`/`mgn`/`mgn_extrap` file variants contain **identical ground truth**. For training, load only one variant per Part (e.g., always use `mgn`). The `y_est_*` predictions can optionally be used as:
- Baseline comparison metrics
- Input features for residual/correction learning
- Teacher predictions for distillation

### 9.2 Recommended Input Features

| Feature | Source | Notes |
|---------|--------|-------|
| Node position | `pos` (N, 2) | Primary spatial encoding |
| Signed arc-length | `saf` (N, 2) | Airfoil-relative position encoding |
| Distance SDF | `dsdf` (N, 8) | Rich shape descriptor |
| Boundary type | `boundary` (N,) | One-hot encode (8 classes) |
| Zone ID | `zoneID` (N,) | One-hot encode (3 classes) |
| Reynolds number | `flowState['Re']` | Global conditioning (normalize in log-space) |
| Angle of attack | `AoA` | Global conditioning |
| NACA code | `NACA` | Encode as 4 digits or embed |
| Gap/stagger | `gap`, `stagger` | Global conditioning (tandem only) |
| Graph connectivity | `edge_index` | For message passing |

### 9.3 Target

- `y` (N, 3): [Ux, Uy, p] — velocity field + kinematic pressure (p/ρ)
- Consider predicting **normalized residuals** relative to freestream: `(y - y_freestream) / scale`
- The pressure channel has much larger magnitude and variance — consider separate loss weighting

### 9.4 Data Splits

Suggested approach for generalization testing:

| Split | Strategy |
|-------|----------|
| In-distribution | Random 80/10/10 within each subset |
| NACA generalization | Hold out specific NACA profiles (e.g., Part3 leading foils) |
| Re generalization | Train on 2 Re values, test on held-out Re (cruise randomFields Part splits are natural for this) |
| AoA generalization | Train on aoa0, test on aoa5 (cruise Re500) |
| Single→Tandem transfer | Train on `raceCar_single`, evaluate on `raceCar_randomFields_mgn` |

### 9.5 Architecture Considerations

- **Variable mesh sizes** → Use graph neural networks (MeshGraphNet, GNN, etc.) or point cloud methods
- **Multi-scale features** → `dsdf` already provides multi-scale distance encoding; consider multi-resolution message passing
- **Global conditioning** → Inject Re, AoA, NACA via FiLM layers or global node features
- **Single→Tandem transfer** → The dataset explicitly supports this with single-element and tandem pairs sharing the same flow regime
- **Large graphs** (up to 350K nodes) → Consider graph sampling, graph pooling, or patch-based approaches for memory efficiency

### 9.6 Memory Considerations

- Each cruise Re500 sample is ~350K nodes → ~2.7M floats for `y` alone
- Loading a full Part (261 samples x 350K nodes) requires ~50+ GB RAM
- Recommend lazy loading or memory-mapped approaches for training
- The float16 targets save 2x memory but may need casting to float32 for loss computation
