# Paper TandemFoil Tasks

This directory defines a paper-faithful benchmark variant for the high-Re
TandemFoilSet experiments in Experiment 4 of the TandemFoilSet paper.

## Supported Tasks

The following task names are available through:

```bash
cd target/icml2026
python train.py --dataset tandemfoil_paper --tandemfoil-paper-task <task>
```

Supported tasks:

- `cruise_random_uniform`
- `cruise_random_aoa_extrap`
- `cruise_random_re_extrap`
- `cruise_random_stagger_extrap`
- `cruise_random_gap_extrap`
- `racecar_uniform`

## Split Rules

Uniform tasks:

- deterministic `80/10/10` train/val/test split
- RNG seed `42`

Extrapolation tasks:

- lowest `5%` and highest `5%` of the selected variable become `test`
- the middle `90%` is split uniformly into:
  - `train = 80%` of the full dataset
  - `val = 10%` of the full dataset
- RNG seed `42` for the middle-90% train/val shuffle

## Artifacts

The generator writes:

- `split_manifest_tandemfoil_paper_experiment4.json`
- `split_stats_tandemfoil_paper_experiment4.json`

These files are auto-materialized on first use if they are missing and the raw
TandemFoil pickles are available under the mounted PVC.

Manual generation:

```bash
cd target/icml2026
python tandemfoil_paper/data/split_paper_experiment4.py
```

## Implementation Note

This benchmark uses the shared tandem pickle loader and the same 24-feature
dual-foil preprocessing as `tandemfoil/`. The difference is the split contract
and the metric contract: this benchmark evaluates normalized full-field MSE,
matching the paper’s Experiment 4 reporting style.

