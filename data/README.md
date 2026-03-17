# data

Data preparation, benchmark splits, and normalization stats for the multi-domain TandemFoilSet experiment track.

The training script lives at the repo root (`train.py`).

---

## Why a structured benchmark?

A naive random 90/10 split mostly tests **interpolation**: the model sees the same
airfoil families, nearby (Re, α, gap, stagger) values, and often the same front/rear
shapes in both train and test. That hides the paper's central question — whether
single-airfoil pretraining actually helps on unseen tandem configurations.

The structured split instead tests four orthogonal failure modes:

| Track | What it tests | Why it matters |
|-------|--------------|----------------|
| `val_in_dist` | Interpolation on seen shapes and conditions | Sanity check; should be easy |
| `val_tandem_transfer` | Tandem pair with a front foil (NACA6416) absent from tandem training | Core claim: does single-foil data transfer to unseen tandem front shapes? |
| `val_ood_cond` | Cruise cases in the frontier 20% of the joint (AoA, gap, stagger) space | Condition extrapolation — far from the training distribution centroid |
| `val_ood_re` | All cruise Part2 cases (Re=4.445M, entirely outside training Re range) | Reynolds-number extrapolation — a clean OOD physics shift |

These four tracks map directly onto the paper's evaluation axes: **composition**,
**transfer**, **condition OOD**, and **physics OOD**. Reporting each separately
prevents a single global MSE from masking failures on harder sub-tasks (e.g. near-wall
regions, wake prediction on novel pairs).

### Split assignment rules

The seven pickle files are divided as follows:

| File | Samples | Assignment |
|------|---------|------------|
| `raceCar_single` (file 0) | 899 | 70% subsample → 90% train, 10% `val_in_dist` |
| `raceCar_tandem Part1` (file 1, front=NACA2412) | 300 | 70% subsample → train |
| `raceCar_tandem Part2` (file 2, front=NACA6416) | 300 | 70% subsample → `val_tandem_transfer` |
| `raceCar_tandem Part3` (file 3, front=NACA9412) | 300 | 70% subsample → train |
| `cruise Part1` (file 4, Re=1.475M) | 300 | 70% subsample → interior 80% train, frontier 20% `val_ood_cond` |
| `cruise Part2` (file 5, Re=4.445M) | 300 | 70% subsample → `val_ood_re` |
| `cruise Part3` (file 6, Re=802K) | 300 | 70% subsample → interior 80% train, frontier 20% `val_ood_cond` |

**Key design choices:**

- **Part2 files go entirely to val.** raceCar tandem Part2 uses NACA6416 as the front
  foil, which does not appear as a tandem front foil in any training file. This makes
  `val_tandem_transfer` a true held-out shape test rather than a re-split of seen pairs.
  Similarly, cruise Part2's Re=4.445M is above the training ceiling (~1.5M), giving a
  clean physics-OOD track.

- **Frontier detection for `val_ood_cond`.** Cruise Parts 1+3 are split by distance from
  the centroid in normalized (AoA, gap, stagger) space. The outermost 20% of cases —
  those with the most extreme combined conditions — are held out. This is better than
  holding out tails of a single variable, which would leave near-duplicates in train.

- **30% subsampling** (SAMPLE_FRACTION=0.70) is applied proportionally to every source
  before splitting, so no domain is disproportionately thinned.

**1,889 samples** retained (70% of 2,699, balanced across sources): **1,322 train** + 567 val (63 + 210 + 84 + 210).

### Training sampler

Even after structured splitting, raw sample counts are unbalanced (raceCar single has
~4× more training samples than cruise). A `WeightedRandomSampler` gives each of the
three domain groups — `racecar_single`, `racecar_tandem`, `cruise` — equal expected
weight per minibatch, preventing the largest domain from dominating the loss and
obscuring transfer performance on the smaller tandem/cruise groups.

Features:
- 24-dim input features (adds foil-2 NACA/AoA, gap, stagger)
- `SURFACE_IDS` includes boundary ID 7 (foil-2 surface nodes)
- Balanced domain sampler (racecar_single / racecar_tandem / cruise equally weighted)

---

## Files

| File | Purpose |
|------|---------|
| `split.py` | One-time script to regenerate the manifest and stats |
| `prepare_multi.py` | Extended preprocessing: 24-dim x, foil-2 features, boundary ID 7 |
| `split_manifest.json` | Committed train/val indices (run `split.py` to regenerate) |
| `split_stats.json` | Committed normalization stats over training set |

---

## Running

```bash
# Standard run (manifest and stats default to the committed files)
python train.py --agent <your-name> --wandb_name "<your-name>/<description>"

# Debug run (6 train samples, 2 per val split, 3 epochs)
python data/split.py --quick   # fast manifest, no data loading
python train.py --debug
```

**W&B project:** `wandb-applied-ai-team / senpai-v1`

---

## Regenerating the manifest

Only needed if you change `SAMPLE_FRACTION`, the file list, or the split logic:

```bash
# Full run (~30-60 min — loads all 7 pickle files twice for two-pass stats)
python data/split.py

# Quick debug manifest (instant — no data loading, identity normalization)
python data/split.py --quick
```

---

## Launching via launch.py

```bash
python k8s/launch.py \
  --tag <research-tag> \
  --wandb_project senpai-v1 \
  [--n_students 4]
```

`launch.py` defaults to `--repo_branch main`. Once this branch is merged to main,
student pods will clone main and get `train.py`, the committed manifest,
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
