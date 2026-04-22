# data

Data preparation, benchmark splits, and normalization stats for the packaged
TandemFoilSet parity target inside `target/icml2026`.

The shared trainer lives at `../../train.py`.

This README documents the **active `tandemfoil/` contract**:

- split manifest: `split_manifest_tandemfoilset_v2.json`
- metrics: denormalized surface-pressure MAE on the balanced `kagent` split
  family

It is intentionally **not** the original TandemFoilSet paper’s Experiment 4 MSE
contract. For that literature-facing benchmark, see `../../tandemfoil_paper/`.

---

## Why this split design?

The active `tandemfoil/` target uses the public `kagent` TandemFoilSet split
design (v2), which was built to answer four concrete questions with balanced
validation and test tracks.

All val/test tracks are balanced:

- `100` validation cases per track
- `200` test cases per track
- paper-facing summary = equal-weight average surface-pressure MAE across the four tracks

### Validation / test tracks

| Split | Source | Selection | What it tests |
|-------|--------|-----------|---------------|
| `val_single_in_dist` / `test_single_in_dist` | file `0` | random holdout | Sanity check on single-foil interpolation |
| `val_geom_camber_rc` / `test_geom_camber_rc` | file `2` | full file holdout | Race-car tandem geometry generalization to unseen front-foil camber `M=6-8` |
| `val_geom_camber_cruise` / `test_geom_camber_cruise` | file `5` | full file holdout | Cruise tandem geometry generalization to unseen front-foil camber `M=2-4` |
| `val_re_rand` / `test_re_rand` | files `1, 3, 4, 6` | every 4th sample after sorting by `Re` | Cross-regime Reynolds-number generalization across the tandem training domains |

### File allocation

| File | Name | Train | Val | Test | Rationale |
|------|------|------:|----:|-----:|-----------|
| `0` | raceCar single | `599` | `100` | `200` | random in-distribution sanity check |
| `1` | raceCar tandem P1 | `~225` | `~25` | `~50` | contributes to stratified `Re` holdout |
| `2` | raceCar tandem P2 | `0` | `100` | `200` | full geometry holdout: unseen front-foil camber `M=6-8` |
| `3` | raceCar tandem P3 | `~225` | `~25` | `~50` | contributes to stratified `Re` holdout |
| `4` | cruise tandem P1 | `~225` | `~25` | `~50` | contributes to stratified `Re` holdout |
| `5` | cruise tandem P2 | `0` | `100` | `200` | full geometry holdout: unseen front-foil camber `M=2-4` |
| `6` | cruise tandem P3 | `~225` | `~25` | `~50` | contributes to stratified `Re` holdout |

Split totals:

- train: `1499`
- validation: `400`
- test: `800`

### Why these holdouts?

- Full-file geometry holdouts give the cleanest separation for camber
  generalization because the held-out front-foil families do not appear in
  training at all.
- Stratified Reynolds-number holdout tests whether one model works across the
  full `Re` range instead of only at one corner of the distribution.
- The random single-foil holdout acts as a sanity check so the harder tandem
  tracks are interpreted against an easier baseline.

### Training sampler

The active manifest also keeps balanced domain sampling via three train-domain
groups:

- `racecar_single`
- `racecar_tandem`
- `cruise`

A `WeightedRandomSampler` gives those groups equal expected minibatch weight so
the largest domain does not dominate the loss.

---

## Metric contract

Primary harness metric:

- `val_primary/surface_pressure_mae`
- `test_primary/surface_pressure_mae`

Definition:

- per split, compute
  `surface_pressure_mae = mean |p_hat - p_true|`
- use the pressure / `C_p` channel only
- evaluate **after full denormalization back to the original target space**
- aggregate globally over all valid surface nodes in the split, not per case

Summary metric:

- `val_eq4/surface_pressure_mae`
- `test_eq4/surface_pressure_mae`

This is the equal-weight mean of the four active split-specific surface-pressure
MAEs:

- `single_in_dist`
- `geom_camber_rc`
- `geom_camber_cruise`
- `re_rand`

Secondary diagnostics:

- `mae_surf_Ux`, `mae_surf_Uy`, `mae_surf_p`
- `mae_vol_Ux`, `mae_vol_Uy`, `mae_vol_p`
- per-split validation loss

Historical note:

- the old `split_manifest.json` and legacy names `p_in`, `p_oodc`, `p_tan`,
  `p_re` are still useful for historical lineage
- the active `tandemfoil/` target does **not** rank runs on that legacy split
  family anymore
- do not compare `surface_pressure_mae` on the v2 manifest directly against the
  paper’s Table 6 `field_mse` numbers

---

## Files

| File | Purpose |
|------|---------|
| `split_tandemfoilset_v2.py` | normalize the public `kagent` competition manifest into this repo’s schema |
| `split_manifest_tandemfoilset_v2.json` | committed active train/val/test manifest |
| `split_stats.json` | committed normalization stats over the active training split |
| `prepare_multi.py` | extended preprocessing: 24-dim `x`, foil-2 features, boundary ID `7` |
| `split.py` | older legacy split generator kept for historical reference |
| `split_manifest.json` | older legacy structured manifest kept for historical reference |

---

## Running

```bash
# Standard run (manifest and stats default to the committed active files)
cd target/icml2026 && python train.py --dataset tandemfoil --agent <your-name> --wandb_name "<your-name>/<description>"
```

**W&B project:** `wandb-applied-ai-team / senpai-v1`

---

## Regenerating the manifest

Only needed if the public competition split is updated or the active TandemFoil
parity contract changes:

```bash
cd target/icml2026
python tandemfoil/data/split_tandemfoilset_v2.py
```

The legacy `split.py` flow is retained only for historical experiments.

---

## Launching via launch.py

```bash
python k8s/launch.py \
  --tag <research-tag> \
  --wandb_project senpai-v1 \
  [--n_students 4]
```

`launch.py` defaults to `--repo_branch main`. Once this branch is merged to
main, student pods will clone main and get `train.py`, the committed manifest,
and the stats file automatically.

---

# TandemFoilSet Dataset Report

**Source:** <https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/KTXSCU>
**Location:** `/mnt/new-pvc/datasets/tandemfoil/`
**Total size:** ~178 GB across 26 pickle files
**Format:** Lists of `torch_geometric.data.Data` objects (loaded via `torch.load(..., weights_only=False)`)

---

## File Inventory

### Cruise — Random Fields (tandem foils, variable Re)

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

Parts 1/2/3 correspond to **different Reynolds numbers**: Re=1,475,000 / Re=4,445,000 / Re=802,000. The `ive`, `mgn`, and `mgn_extrap` variants share identical CFD data but attach different pre-computed baseline model predictions (see file variant suffixes below).

### Cruise — Re=500 (tandem foils, fixed Re, fixed AoA)

| File | Samples | Nodes (mean) | Size |
|------|---------|-------------|------|
| `cruise_Re500_aoa0_ive_Part{1,2,3}.pickle` | 261+261+262 | ~349,000 | 9.8 GB each |
| `cruise_Re500_aoa0_mgn_Part{1,2,3}.pickle` | 261+261+262 | ~349,000 | 9.8 GB each |
| `cruise_Re500_aoa5_ive_Part{1,2,3}.pickle` | 261+261+262 | ~349,000 | 9.8 GB each |
| `cruise_Re500_aoa5_mgn_Part{1,2,3}.pickle` | 261+261+262 | ~349,000 | 9.8 GB each |

**Subtotal: 3,136 samples (784 unique simulations x 2 AoA x 2 file variants)**

### RaceCar — Single Element (single foil, variable Re)

| File | Samples | Nodes (mean) | Size |
|------|---------|-------------|------|
| `raceCar_single_randomFields.pickle` | 899 | 85,964 | 6.5 GB |

### RaceCar — Tandem (dual foils, variable Re)

| File | Samples | Nodes (mean) | Size |
|------|---------|-------------|------|
| `raceCar_randomFields_mgn_Part1.pickle` | 300 | 127,154 | 3.7 GB |
| `raceCar_randomFields_mgn_Part2.pickle` | 300 | 129,507 | 3.7 GB |
| `raceCar_randomFields_mgn_Part3.pickle` | 300 | 125,637 | 3.6 GB |

### Grand Totals

| Subset | Unique simulations | Total files | Total samples (with variants) |
|--------|-------------------|-------------|-------------------------------|
| Cruise randomFields | 900 | 9 | 2,700 |
| Cruise Re500 | 784 | 12 | 3,136 |
| RaceCar single | 899 | 1 | 899 |
| RaceCar tandem | 900 | 3 | 900 |
| **Total** | **3,483** | **26** | **7,635** |

> The `ive`/`mgn`/`mgn_extrap` variants for the same Part share identical CFD ground truth. They only differ in attached pre-computed prediction keys. For training you only need **one variant per simulation**.

---

## Per-Sample Data Schema

Each sample is a `torch_geometric.data.Data` graph object.

### Core Mesh Fields

| Field | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `pos` | `(N, 2)` | float32 | Node coordinates in 2D (x, z) |
| `edge_index` | `(2, E)` | int64 | Graph connectivity (undirected edges) |
| `boundary` | `(N,)` | uint8 | Boundary condition type per node |
| `zoneID` | `(N,)` | float32 | Mesh zone identifier |

### Target Field (Ground Truth from CFD)

| Field | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `y` | `(N, 3)` | float16 | **Target: [Ux, Uy, p]** — velocity components and kinematic pressure (p/ρ, m²/s²) |

### Input Features (Geometric Encodings)

| Field | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `saf` | `(N, 2)` | float16 | Signed arc-length features |
| `dsdf` | `(N, 8)` | float16 | Distance-based shape descriptor field |

### Flow Condition Metadata

| Field | Type | Description |
|-------|------|-------------|
| `flowState` | dict (24–25 keys) | Freestream flow conditions (see FlowState section) |
| `AoA` | float or list[2] | Angle(s) of attack in degrees (per foil) |
| `NACA` | list[str] (len 1 or 2) | NACA 4-digit airfoil profile code(s) |

### Geometry Metadata

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

### File Variant Suffixes: IVE vs MGN vs MGN_extrap

| Suffix | Baseline model | Mesh resolution (typical) | Extra fields |
|--------|---------------|--------------------------|--------------|
| `ive` | IVE — Implicit Volume Estimator | ~209K–349K nodes | — |
| `mgn` | MGN — MeshGraphNet | ~114K–209K nodes | `hc_net`, `hcb_net`, `scb_net`, `height`, `resize` |
| `mgn_extrap` | MGN with extrapolation predictions | Same as `mgn` | Same as `mgn`, plus `y_est_*_extrapRE` and `y_est_*_extrapAOA` |

> Pick one variant per Part (e.g., always `mgn` for smaller meshes). The `y_est_*` fields are optional baselines.

---

## Boundary Condition Types

| Value | Meaning |
|-------|---------|
| 0 | Interior / field nodes |
| 1 | Inlet |
| 2 | Outlet |
| 3 | Top wall |
| 4 | Bottom wall |
| 5 | Airfoil surface (foil 1, upper/main) |
| 6 | Airfoil surface (foil 1, lower/trailing edge) |
| 7 | Airfoil surface (foil 2, tandem only) |

---

## Zone IDs and Overset Mesh

| Value | Description |
|-------|-------------|
| 0 | Background mesh (coarse, covers full domain) |
| 1 | Foil 1 refinement patch (dense, around first airfoil) |
| 2 | Foil 2 refinement patch (dense, around second airfoil — tandem only) |

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

Overlapping points from different zones at the same spatial location is normal for overset CFD — the refinement zone values are authoritative near the airfoil.

---

## Value Ranges

### Target Field `y` — [Ux, Uy, p]

| Subset | Re range | y min | y max | y mean | y std |
|--------|----------|-------|-------|--------|-------|
| Cruise randomFields Part1 | 1.475M | -1,278 | 233 | 1.9 | 55 |
| Cruise randomFields Part2 | 4.445M | -2,360 | 2,118 | 5.3 | 304 |
| Cruise randomFields Part3 | 802K | -300 | 69 | 2.1 | 17 |
| Cruise Re500 | 500 | -0.054 | 0.155 | 0.045 | 0.066 |
| RaceCar single | ~700K–2M | -874 | 467 | -23 | 141 |
| RaceCar tandem | ~700K–2M | -4,277 | 668 | -65 | 235 |

### Node Counts

| Subset | Min nodes | Max nodes | Mean nodes |
|--------|-----------|-----------|------------|
| Cruise randomFields | 179,524 | 242,577 | ~208,000 |
| Cruise Re500 | 347,047 | 349,617 | ~349,000 |
| RaceCar single | 74,782 | 90,173 | 85,964 |
| RaceCar tandem | 88,925 | 163,642 | ~127,000 |

---

## Parameter Space

### Reynolds Number

| Subset | Re values |
|--------|-----------|
| Cruise randomFields | 802K, 1.475M, 4.445M (one per Part) |
| Cruise Re500 | 500 (fixed) |
| RaceCar single | ~700K–2M (varies per sample) |
| RaceCar tandem | ~700K–2M (varies per sample) |

### NACA Profiles

- **Cruise randomFields:** 2-foil combos, leading foil fixed per Part (0006 / 2408 / 4408), trailing foil varies
- **Cruise Re500:** 2-foil combos sweeping systematic NACA pairs (0006–0024 / 1408–1424 / 2418–4424)
- **RaceCar single:** Single NACA, sweeps 2205–2209+ range
- **RaceCar tandem:** 2-foil combos, front foil fixed per Part (2412 / 6416 / 9412)

### Angle of Attack

| Subset | AoA range |
|--------|-----------|
| Cruise randomFields | ~[-8, +8] degrees (random per sample) |
| Cruise Re500 aoa0 | 0 degrees (fixed) |
| Cruise Re500 aoa5 | 5 degrees (fixed) |
| RaceCar single | ~[-10, +10] degrees (random per sample) |
| RaceCar tandem | Two independent AoAs per sample |

### Tandem Foil Geometry

| Parameter | Cruise randomFields | RaceCar tandem |
|-----------|-------------------|----------------|
| gap | ~[-0.8, +0.5] | ~[0.4, 1.3] |
| stagger | ~[0.7, 2.0] | ~[0.7, 1.0] |
| resize | — | 0.35, 0.45, 0.50 |

---

## FlowState Dictionary

The `flowState` dict encodes freestream boundary conditions:

| Key | Type | Example | Description |
|-----|------|---------|-------------|
| `Re` | float | 1,475,000 | Reynolds number |
| `Umag` | float | 21.55 | Freestream velocity magnitude (m/s) |
| `Ux` / `Uy` / `Uz` | float | 21.55 / 0.0 / 0.0 | Velocity components |
| `nu` | float | 1.461e-5 | Kinematic viscosity (m²/s) |
| `rhoTotal` | float | 1.225 | Density (kg/m³) |
| `c` | float | 1.0 | Chord length |
| `c_f` / `c_b` / `c_eff` | float | — | Front/back/effective chord (tandem only) |
| `omega` | float | 2.155 | Specific dissipation rate |
| `k` | float | 3.148e-8 | Turbulent kinetic energy |
| `y_l` | float | 2e-5 | First cell height (boundary layer) |
| `domain` | int | 20 | Domain size (chord lengths) |
