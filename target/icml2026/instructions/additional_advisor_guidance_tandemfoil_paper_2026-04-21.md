# Additional Advisor Guidance For The TandemFoil Paper-Calibration Expansion

This file is intended to be passed via `k8s/launch.py --extra_instructions ...`
for the next advisor restart or fleet expansion after adding the fourth
benchmark under `target/icml2026/tandemfoil_paper/`.

## Why The Fourth Benchmark Exists

We currently have two TandemFoilSet views:

- `target/icml2026/tandemfoil/`
  - the main parity / ICML sprint benchmark
  - useful for noam-lineage progress and the shared sprint story
- `target/icml2026/tandemfoil_paper/`
  - a paper-faithful high-Re Experiment 4 benchmark
  - useful for answering a simpler question:
    - are we actually good relative to the published TandemFoilSet paper?

Do not confuse them.

## Scientific Goal

The goal is still a shared recipe, not a collection of unrelated per-benchmark
tricks.

The core question is now whether one broad recipe can survive across:

- `target/icml2026/tandemfoil/`
- `target/icml2026/tandemfoil_paper/`
- `target/icml2026/airfrans/`
- `target/icml2026/drivaerml/`

It is fine if LR, schedule length, or regularization strength differ somewhat.
It is not fine if every benchmark needs a different core idea.

## Benchmark-Facing Priorities

- `tandemfoil`
  - paper-facing summary metric: `test_primary/surface_pressure_mae`
  - this is the ICML sprint parity target, not the original paper contract
- `tandemfoil_paper`
  - paper-facing metric: `test_primary/field_mse`
  - published Experiment 4 references:
    - `cruise_random_uniform = 0.10`
    - `cruise_random_aoa_extrap = 0.18`
    - `cruise_random_re_extrap = 0.36`
    - `cruise_random_stagger_extrap = 0.13`
    - `cruise_random_gap_extrap = 0.14`
    - `racecar_uniform = 0.21`
- `airfrans`
  - paper-facing metric: `test_primary/surface_mse`
  - targets:
    - `surface_mse = 0.0043`
    - `volume_mse = 0.0017`
- `drivaerml`
  - paper-facing metric: `test_primary/surface_rel_l2_pct`
  - nominal reference:
    - `3.71`

Always keep the reported test metric beside the target or reference.

## Current Strategic Read

- AirfRANS is already strong enough that it should stay narrow.
- DrivAerML is still the weakest benchmark and remains the main rescue lane.
- The new `tandemfoil_paper` benchmark is not a license to reopen huge Tandem
  local sweeps.
- Its job is calibration:
  - tell us whether our Tandem-related ideas are merely improving the internal
    parity target
  - or are also competitive against the original paper contract

## Assignment Guidance

When a student is assigned a cross-benchmark hypothesis family, treat the `8`
GPUs as a small matrix.

Strong default:

- `1` run on `airfrans`
- `2-3` runs on `drivaerml`
- `1` run on `tandemfoil`
- `1` run on `tandemfoil_paper` if the idea is relevant to TandemFoil
- remaining GPUs used for the most decision-critical nearby variants

Use both Tandem benchmarks together when the hypothesis is about:

- Tandem transfer
- Tandem generalization
- whether a Tandem-side change is only helping the noam/parity contract
- whether a Tandem-side change also helps the original paper-style benchmark

Single-dataset work is still appropriate for:

- best-checkpoint cleanup
- preserving the AirfRANS frontier
- a clearly justified DrivAerML rescue lane
- one-off Tandem paper calibration checks

## What To Emphasize

- DrivAerML-centered ideas that can also be checked on AirfRANS and both Tandem
  views
- simple transferable recipe changes
- hypotheses that tell us whether an idea is globally useful or too
  dataset-specific
- test metrics at the best validation checkpoint

## What To De-Emphasize

- broad AirfRANS local mapping
- broad Tandem local mapping on only one Tandem benchmark
- speculative code-change branches unless they are clearly high value
- architecture proliferation without a strong transfer rationale

## Operational Reminder

- The run budget is inherited from the pod environment.
- Do not hardcode obsolete `180`-minute run budgets in PR bodies.
- If a stale `CURRENT_RESEARCH_STATE.md` on the branch conflicts with this
  benchmark expansion, overwrite it immediately.

## Sharp Summary

The fourth benchmark is a calibration tool, not the new center of gravity.

Use it to answer:

- is our Tandem recipe actually paper-competitive?

But keep the overall queue aimed at the harder global question:

- can one shared recipe work across AirfRANS, DrivAerML, the Tandem parity
  target, and the Tandem paper-style target?
