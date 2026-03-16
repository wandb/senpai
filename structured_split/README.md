# structured_split

Benchmark split and training for the multi-domain TandemFoilSet experiment track.

The root `train.py` is a simpler single-dataset script for earlier experiments.
**This directory is the active research track.**

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

Also improves over the root `train.py` by:
- 24-dim input features (adds foil-2 NACA/AoA, gap, stagger)
- `SURFACE_IDS` includes boundary ID 7 (foil-2 surface nodes)
- Balanced domain sampler (racecar_single / racecar_tandem / cruise equally weighted)
- 30-minute training timeout

---

## Files

| File | Purpose |
|------|---------|
| `structured_train.py` | **Main training script** — run this instead of root `train.py` |
| `split.py` | One-time script to regenerate the manifest and stats |
| `prepare_multi.py` | Extended preprocessing: 24-dim x, foil-2 features, boundary ID 7 |
| `split_manifest.json` | Committed train/val indices (run `split.py` to regenerate) |
| `split_stats.json` | Committed normalization stats over training set |

---

## Running

```bash
# Standard run (manifest and stats default to the committed files)
python structured_split/structured_train.py \
  --agent <your-name> --wandb_name "<your-name>/<description>"

# Debug run (6 train samples, 2 per val split, 3 epochs)
python structured_split/split.py --quick   # fast manifest, no data loading
python structured_split/structured_train.py --debug
```

**W&B project:** `wandb-applied-ai-team / senpai-v1`

---

## Regenerating the manifest

Only needed if you change `SAMPLE_FRACTION`, the file list, or the split logic:

```bash
# Full run (~30-60 min — loads all 7 pickle files twice for two-pass stats)
python structured_split/split.py

# Quick debug manifest (instant — no data loading, identity normalization)
python structured_split/split.py --quick
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
student pods will clone main and get `structured_train.py`, the committed manifest,
and the stats file automatically.
