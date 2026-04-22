# Additional Advisor Guidance For The Final ICML Benchmark Sprint

## Background And Why This Guidance Changed

We have already submitted an abstract that makes concrete empirical claims about
SENPAI on multiple CFD surrogate benchmarks.

At this point, the job is **not** to keep searching for one elegant shared
recipe across all datasets.

The job now is to produce the strongest, cleanest, benchmark-faithful evidence
we can for each benchmark individually, in the limited time remaining before
ICML deadlines force the paper story to harden.

That means the strategy has changed:

- no more broad cross-dataset comparison work by default
- no more trying to make every experiment teach us about transfer
- no more spending large amounts of compute on generic novelty families
- instead: aggressively optimize each benchmark on its own terms

Treat this as the final benchmark-closure phase, not the broad discovery phase.

## Direct Mission

Your mission is to maximize benchmark-specific paper evidence for:

- `airfrans`
- `drivaerml`
- `tandemfoil_paper`
- and secondarily `tandemfoil`

If an experiment does not materially improve one of those benchmark-specific
evidence packages, it is probably not worth doing right now.

## Abstract Claims We Must Support Or Honestly Bound

These are the practical claims that matter most for the paper story.

### 1. AirfRANS

Claim in the abstract:

- our final recipe outperforms the reported Transolver baseline on surface-MSE

What this means now:

- surface is already strong enough to support this claim
- but our paper-facing AirfRANS reporting must always include the official full
  pair:
  - `Surf MSE`
  - `Vol MSE`
- therefore the urgent remaining AirfRANS problem is **volume improvement**

Direct instruction:

- stop spending meaningful compute on AirfRANS surface-only exploration
- prioritize experiments that improve `Vol MSE` while preserving already-good
  surface performance

### 2. TandemFoil paper benchmark

By `TFP` we mean the **TandemFoil paper-faithful benchmark**:

- `target/icml2026/tandemfoil_paper/`

This is **not** the same thing as the repo parity benchmark:

- `target/icml2026/tandemfoil/`

Claim in the abstract:

- we are competitive with the normalized full-field MSE reported in the
  TandemFoilSet paper benchmark

What this means now:

- we need a clean `test_primary/field_mse` row
- on the correct paper-style split
- with clean provenance
- and reported as normalized full-field MSE, not surface error

Direct instruction:

- getting a clean TFP test result is one of the most urgent remaining paper
  tasks

### 3. DrivAerML

Claim in the abstract:

- preliminary runs show surface pressure relative-L2 nearing reported
  Transolver-family references

What this means now:

- this is still our weakest benchmark story
- we need a stronger full-eval test result
- this should receive the largest share of new compute

Direct instruction:

- DrivAerML is the main empirical benchmark gap
- optimize it directly and aggressively

### 4. Tandem parity benchmark

This is:

- `target/icml2026/tandemfoil/`

Its role now:

- a useful internal anchor
- a sanity check that we have not broken our strongest internal Tandem line

Direct instruction:

- preserve it
- improve it if it is cheap
- do not let it absorb a large share of the fleet

## Non-Negotiable Benchmark Contracts

Use these contracts in all advisor reasoning, PR bodies, status updates, and
paper-facing summaries.

### `tandemfoil`

- repo parity contract only
- paper-facing metric: `test_primary/surface_pressure_mae`
- split contract: `kagent` v2 parity split family

### `tandemfoil_paper`

- original TandemFoilSet-paper contract
- paper-facing metric: `test_primary/field_mse`
- split contract: Experiment 4 / Table 6 style high-Re split family
- do not call this surface-MSE
- do not mix this with the parity `tandemfoil` contract

### `airfrans`

- official full-task benchmark contract
- paper-facing reporting must always include both:
  - `Surf MSE`
  - `Vol MSE`
- if training uses any extra target transform, decode predictions back to raw
  target space and rescore with the official train-stat normalization
- surface-only numbers are auxiliary only and must never replace the full pair

### `drivaerml`

- paper-facing metric: `test_primary/surface_rel_l2_pct`
- split contract: repaired public `400 / 34 / 50` split
- paper-facing claims must use full evaluation, not truncated eval

Validation is for steering.
Test is for the paper.
Best-checkpoint evaluation is mandatory whenever possible.

## Current Benchmark Read

### Tandem parity

- best reportable parity test anchor: `24.581`
- healthy enough that it is no longer the main blocker

### TandemFoil paper benchmark

- strongest current internal steering anchor:
  - `val_primary/field_mse = 0.002383`
- still missing:
  - a clean, citation-ready `test_primary/field_mse` row

### AirfRANS

- current benchmark-faithful pair to beat internally:
  - `Surf MSE = 0.003`
  - `Vol MSE = 0.00764`
- published full-task comparison:
  - SpiderSolver `0.0043 / 0.0017`
- surface-only `0.002999` is real but auxiliary only

Interpretation:

- surface is already good
- volume is now the urgent AirfRANS problem

### DrivAerML

- current best paper-facing test anchor:
  - `6.244%`
- strongest published comparison band:
  - `3.71%` from `Transolver-3`
  - `3.82%` from `AB-UPT`
- relevant older Transolver-family comparison:
  - around `4.81%` on surface pressure
- strong internal steering anchor:
  - `val_primary/surface_rel_l2_pct = 3.997%`

Interpretation:

- this remains the main benchmark gap
- this benchmark should receive the largest share of the fleet

## Effective Strategy Change: Optimize Benchmarks Individually

Effective immediately:

- stop framing the main queue around shared-recipe transfer
- stop assigning broad cross-dataset experiment families by default
- stop using each student as a mini cross-dataset matrix unless there is a very
  specific infrastructure reason

Default mode now:

- each PR should usually target **one benchmark**
- each student should usually optimize **one benchmark-specific family**
- each result should be judged first by whether it improves that benchmark's
  paper-facing evidence

Cross-dataset work is still allowed only when:

- it is infrastructure that directly affects benchmark-faithful reporting
- or it is a very high-value code path such as best-checkpoint behavior

But benchmark-specific optimization is now the default.

## What We Should Focus On Right Now

If time is limited, prioritize in exactly this order.

### 1. DrivAerML full-eval test improvement

This is the most important remaining research problem.

We need:

- a better benchmark-facing DrivAerML test row
- under full evaluation
- with clean best-checkpoint provenance

Prefer:

- exact-champion local refinements
- LR refinement
- `T_max` refinement
- sampling / surface-point refinement
- objective refinement near the current best line
- best-checkpoint recovery

Actively target:

- `#3043`
- `#3044`
- `#3045`
- `#3046`
- `#3047`
- `#3048`
- `#3051`
- `#3060`

Concrete target:

- move materially closer to the Transolver-family comparison band
- if possible, get below `5%`
- if not, produce the strongest honest preliminary result with full-eval
  provenance

### 2. TandemFoil paper benchmark clean test result

This is the most important remaining result for supporting the TandemFoil paper
claim in the abstract.

We need:

- a clean `test_primary/field_mse`
- on the correct paper-style split
- with clean best-checkpoint provenance
- explicitly reported as normalized full-field MSE

Prefer:

- `#2947`
- `#2948`
- `#2949`
- `#3056`

### 3. AirfRANS volume improvement under the official full contract

This is the key AirfRANS instruction:

- **surface is good enough**
- **volume is not**

Therefore:

- AirfRANS work should now primarily target `Vol MSE`
- the goal is to lower `Vol MSE` materially while preserving strong `Surf MSE`

Only do AirfRANS work that helps one of:

- full `Surf / Vol` pair closure
- best-checkpoint recovery
- official rescoring / provenance cleanup
- a narrowly justified frontier line that still reports the full pair

Do not spend meaningful compute on:

- AirfRANS surface-only local mapping
- new broad AirfRANS neighborhood sweeps that are not clearly volume-oriented

### 4. Preserve Tandem parity as a healthy anchor

Use `tandemfoil` for:

- transfer sanity checks only if needed
- quick anchor replications
- keeping one strong Tandem row alive in the paper package

Do not spend a large share of the fleet on new Tandem-only local mapping.

## Resource Allocation Guidance

Default compute allocation for the current phase:

- about `50-60%` of new capacity to `drivaerml`
- about `20-30%` to `tandemfoil_paper`
- about `10-20%` to narrow AirfRANS work, mostly volume-oriented
- minimal extra `tandemfoil` work beyond anchor checks

If forced to choose between:

- a broad novelty family
- a local Tandem sweep
- another AirfRANS surface-only refinement
- a DrivAerML closure lane
- or a TandemFoil-paper clean-test lane

choose the DrivAerML closure lane or the TandemFoil-paper clean-test lane.

## What To Stop Doing

Do **not** keep the queue wide just to keep students busy.

Broad novelty work is lower priority than the paper-critical closure items
above, especially:

- Pre-LN / RMSNorm / warmup / gradient centralization
- normals / curvature features
- divergence / mass-conservation aux loss
- MoE FFN layers
- AdaFactor
- generic dropout / spectral norm / MQA / SWA / LayerScale families
- broad Fourier / slice-temperature / normalization-side explorations
- old duplicated scheduler or regularization themes that are already superseded
- broad cross-dataset matrices whose main value is “transfer evidence”

If a lane is not helping the benchmark-specific evidence package, close it.

## Assignment Guidance

Default to narrow, benchmark-specific assignments.

- one student should usually own one benchmark family
- one PR should usually target one benchmark
- do not assign broad cross-dataset novelty by default
- preserve active high-EV lanes and avoid duplicate assignments against live PRs

Good assignments now:

- one student owning a DrivAerML family with `3-6` nearby variants
- one student owning a TandemFoil paper-benchmark calibration family
- one student on AirfRANS volume-focused full-pair cleanup
- one student on best-checkpoint or evaluation infrastructure if it directly
  improves paper-facing reporting

Bad assignments now:

- another generic architecture novelty branch
- another broad AirfRANS local neighborhood map
- another Tandem-only local sweep without paper relevance
- another cross-dataset transfer matrix just because it fits the old strategy

## Mandatory Reporting Rules

Every advisor-generated PR and every summary should make the benchmark contract
explicit.

### AirfRANS

- always report both `Surf MSE` and `Vol MSE`
- if mentioning surface-only progress, label it auxiliary
- explicitly say whether the work improved volume, since that is now the main
  unresolved AirfRANS need

### TandemFoil paper benchmark

- report `field_mse`
- explicitly say normalized full-field MSE
- do not call it surface-MSE
- if abbreviated, define `TFP` the first time as:
  - `tandemfoil_paper`, the TandemFoil paper-faithful benchmark

### DrivAerML

- report `surface_rel_l2_pct`
- say whether eval was full or truncated
- paper-facing claims require full eval

### Tandem parity

- report `surface_pressure_mae`
- do not mix it with the TandemFoil paper contract

### All datasets

- keep the external reference beside the reported test metric
- prefer best-checkpoint test reporting
- do not let validation-only numbers masquerade as paper evidence

## Operational Rules

- The pod environment already provides:
  - `SENPAI_TIMEOUT_MINUTES=360`
  - `SENPAI_MAX_EPOCHS=999`
- Do not hardcode stale `180`-minute assumptions.
- Do not hardcode `9999`-epoch overrides in PR bodies unless you are
  intentionally changing the budget and you explain why.
- If `CURRENT_RESEARCH_STATE.md` or any local status memo is stale, overwrite it
  immediately rather than letting it steer the next wave.
- The benchmark reference docs already exist in `analysis/`.
  Do not spend unnecessary time re-deriving baseline provenance that is already
  documented there.

## Sharp Summary

The remaining ICML-phase job is not:

- explore broadly
- maximize PR count
- prove a shared recipe
- or keep every GPU busy on whatever is available

The remaining ICML-phase job is:

- strengthen the DrivAerML test story
- produce a clean TandemFoil paper-benchmark test row
- improve AirfRANS volume while keeping its strong surface performance
- preserve a strong Tandem parity anchor
- and spend compute only where it helps defend the abstract we already wrote
