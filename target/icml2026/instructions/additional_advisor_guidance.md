# Additional Advisor Guidance For Large-Fleet ICML Relaunches

This file is intended to be passed via `k8s/launch.py --extra_instructions ...`
for large-fleet ICML sprint relaunches built on `target/icml2026`.

## Scientific goal

The goal is not three unrelated benchmark-specific wins. The goal is a shared
recipe whose core changes more or less work across:

- `target/icml2026/tandemfoil/`
- `target/icml2026/airfrans/`
- `target/icml2026/drivaerml/`

It is fine if some hyperparameters differ by dataset. It is not fine if the
story only works because every dataset needs a completely different core idea.

## Benchmark-facing priorities

- `tandemfoil`
  - paper-facing summary metric: `test_primary/surface_pressure_mae`
  - no exact external leaderboard scalar for the packaged target
  - current sprint anchor at time of writing (`2026-04-21`):
    - `val_primary/surface_pressure_mae = 44.72`
    - reported test from that lane: `50.77`
- `airfrans`
  - paper-facing metric: `test_primary/surface_mse`
  - external targets:
    - `Surf MSE = 0.0043`
    - `Vol MSE = 0.0017`
  - current clean reported starting point at time of writing:
    - `test_primary/surface_mse = 0.01478`
- `drivaerml`
  - paper-facing metric: `test_primary/surface_rel_l2_pct`
  - strongest nominal external reference:
    - `3.71` from `Transolver-3`
  - current reported starting point at time of writing:
    - `test_primary/surface_rel_l2_pct = 6.24`

Always keep the target or reference beside the reported test metric.

## Evaluation discipline

- Validation is for steering.
- Test is for paper-facing comparison.
- Evaluate test at the best validation checkpoint whenever possible.
- Do not rely on final-epoch test alone when long runs can improve and then
  deteriorate late.

## Assignment guidance

When a large fleet is available, default to assigning a hypothesis family across
all three datasets to the same student.

- A student has `8` GPUs.
- Use those GPUs as a matrix across datasets and nearby variants.
- A good default is:
  - at least one run per dataset
  - remaining GPUs used for the most decision-critical nearby variants
- The resulting PR should report metrics across all three datasets so the same
  student can judge whether the idea transferred or was too dataset-specific.

Single-dataset assignments are still appropriate for:

- frontier closure on the current best line
- best-checkpoint test recovery
- a dataset-specific failure analysis that is clearly blocking progress

## Throughput guidance

If a large relaunch uses something like `50` worker nodes with `8` GPUs each,
think in terms of GPU saturation, not just student saturation.

- One PR can and often should contain multiple related runs.
- Idle GPUs are a real failure mode.
- Broad-but-interpretable grouped experiment families are better than a narrow
  queue that leaves a large fraction of the fleet idle.

## Scope guardrails

- This is not an architecture-invention phase by default.
- Prefer simple improvements and transferable mechanisms over benchmark-specific
  hacks.
- If an old `RESEARCH_STATE` or `CURRENT_RESEARCH_STATE` file reflects an older
  narrower focus, it is fine to overwrite it with the current ICML
  multi-dataset focus.
