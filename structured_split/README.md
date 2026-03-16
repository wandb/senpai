# structured_split

Benchmark split and training for the multi-domain TandemFoilSet experiment track.

The root `train.py` is a simpler single-dataset script for earlier experiments.
**This directory is the active research track.**

---

## What it does

Instead of a random 90/10 split on one dataset, this implements a structured benchmark
across all 7 data sources with four distinct validation tracks that test different
types of generalization:

| Val split | Source | Tests |
|-----------|--------|-------|
| `val_in_dist` | raceCar single (10% holdout) | Interpolation sanity |
| `val_tandem_transfer` | raceCar tandem Part2 (front=NACA6416) | Single→tandem transfer |
| `val_ood_cond` | cruise Part1+3 frontier 20% | Condition extrapolation (extreme AoA/gap/stagger) |
| `val_ood_re` | cruise Part2 Re=4.445M | Reynolds number extrapolation |

**1,889 samples** retained (70% of 2,699, balanced across sources): **1,322 train** + 567 val (63 + 210 + 84 + 210).

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

**W&B project:** `wandb-applied-ai-team / senpai-testing`

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
  --wandb_project senpai-testing \
  [--n_students 4]
```

`launch.py` defaults to `--repo_branch main`. Once this branch is merged to main,
student pods will clone main and get `structured_train.py`, the committed manifest,
and the stats file automatically.
