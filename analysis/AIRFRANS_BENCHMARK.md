# AirfRANS Benchmark History and Comparison Contract

This note is aligned with the repo's current `STATUS_*` files and `analysis/ICML_2026_ABSTRACTS_PLAN.md`: for a paper-facing AirfRANS claim, the safe comparison contract is the official normalized-target `Surf MSE` plus `Vol MSE` on the official test split, not a surface-only number.

## Executive Summary

For a new paper on the official AirfRANS `full` task, the clean post-Transolver comparison set is narrower than a generic "AirfRANS papers" list, but it is not just two rows. The safe full-task apples-to-apples chain is `Transolver -> GeoANF -> SpiderSolver`.

| Paper | Venue | Split / eval contract | Metric family | Surface reported? | Volume reported? | Apples-to-apples to official `Surf MSE` / `Vol MSE`? | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Transolver | ICML 2024 | Official `full` task; code follows AirfRANS split logic and scores on official test set | Official AirfRANS MSE family | Yes | Yes | Yes | Original published row is `Vol = 0.0037`, `Surf = 0.0142`. Repo README corrects the metric name: these are MSE, not Relative L2. |
| SpiderSolver | NeurIPS 2025 | Official `full` task; code inherits the same split / test protocol | Official AirfRANS MSE family | Yes | Yes | Yes | Strongest clearly documented full-task result I found: `Vol = 0.0017`, `Surf = 0.0043`. |
| GeoANF | Appl. Sci. 2024 | Explicitly says data configuration and metrics are aligned with AirfRANS; reports `full`, `reynolds`, and `aoa` tables | Official AirfRANS-style MSE family | Yes | Yes | Yes, with one wording caveat | Table 5 gives a directly comparable `full` row: `Vol = 0.0062`, `Surf = 0.0089`. The abstract also quotes `volume flow field MSE = 0.0038`, which should not be mixed with the aggregate `Vol MSE` row. |
| GLOBE | arXiv 2025 | AirfRANS `full` and `scarce`, but headline tables are on validation set | Per-field z-score MSE, plus physically nondimensionalized MAE | Surface pressure only | Per-field volume only | No | Strong paper, but not a leaderboard-style `Surf MSE` / `Vol MSE` comparison. |
| MARIO | 2025 | AirfRANS `scarce` task | Per-field MSE on normalized outputs | Surface pressure only | Per-field volume only | No for `full` benchmark; partial for `scarce` per-field comparisons | Commonly cited, but not the same benchmark target as full-task `Surf MSE` / `Vol MSE`. |
| FLOW-GLIDE | Appl. Sci. 2025 | AirfRANS full / OOD tasks | Relative-L2 for `Volume` and `Surface` | Yes | Yes | No | Explicitly changes metric family away from official AirfRANS MSE. |

If we want the safest apples-to-apples comparison in our paper, the main full-task table should include:

- `Transolver`: `Surf MSE = 0.0142`, `Vol MSE = 0.0037` as the original published row.
- `GeoANF`: `Surf MSE = 0.0089`, `Vol MSE = 0.0062` from the paper's AirfRANS-aligned full-task table.
- `SpiderSolver`: `Surf MSE = 0.0043`, `Vol MSE = 0.0017`.

If we want to mention the stronger `Transolver = 0.0080 / 0.0025` pair that appears in later citations and in our local AirfRANS notes, we should label it as a later cited / corrected Transolver baseline, not as the original ICML 2024 published row.

## Official Benchmark Contract

These are the details that have to match before two AirfRANS numbers are really comparable.

### Tasks and splits

From the official AirfRANS paper / repo:

- `full`: 800 train, 200 test.
- `scarce`: 200 train, same test set as `full`.
- `reynolds`: train on mid-range Reynolds numbers, test on out-of-range Reynolds numbers.
- `aoa`: train on mid-range angles of attack, test on out-of-range angles of attack.

From the official code and the later Transolver / SpiderSolver codebases:

- Training starts from the official task-specific train manifest.
- The last 10% of that train manifest is carved out as validation.
- Evaluation uses the official task test split.
- Special case: `scarce` evaluates on `full_test`.

So for the `full` task, the effective training recipe used by AirfRANS-style code is usually:

- 720 training cases
- 80 validation cases
- 200 official test cases

### Normalization

The official AirfRANS `dataset.py` computes z-score normalization coefficients on the training data only:

- input normalization: `mean_in`, `std_in`
- output normalization: `mean_out`, `std_out`

Then validation / test targets are normalized with those training-set coefficients.

This matters because several later papers still use z-score-normalized targets, but not always with the same reporting format.

### Metric aggregation

The official AirfRANS `metrics.py` computes:

- MSE separately on surface nodes and non-surface volume nodes
- mean over output channels
- then average over cases in the test loader

So the official benchmark-style scalar metrics are:

- `Surf MSE`: surface-node MSE on normalized targets
- `Vol MSE`: non-surface-node MSE on normalized targets

One source of historical confusion is that the original AirfRANS literature often reports per-field normalized MSEs such as `u_x`, `u_y`, `p`, `nu_t`, and `p_s`, while Transolver / SpiderSolver-style practical-design tables compress this into aggregate `Vol` and `Surf` scalars. Those two reporting styles should not be mixed without saying so explicitly.

This is the contract our own repo notes are implicitly enforcing when they warn that surface-only numbers are not enough for a full benchmark claim.

## Directly Comparable Post-Transolver Results

These are the rows I would treat as genuinely valid full-task benchmark comparators.

### 1. Transolver (ICML 2024)

- Source: Transolver paper / AirfRANS repo.
- Split / protocol: follows the AirfRANS task manifests, 10% validation carve-out, official test scoring.
- Metric family: official AirfRANS MSE family.
- Surface + volume: yes.

Original published AirfRANS full-task row:

- `Vol = 0.0037`
- `Surf = 0.0142`
- `C_L error = 0.1030`
- `rho_L = 0.9978`

Important caveat:

- The Transolver AirfRANS README explicitly says the paper has a typo: the physics-field metrics are `MSE`, not `Relative L2`.

### 2. GeoANF (Applied Sciences 2024)

- Source: Applied Sciences 2024 paper.
- Split / protocol: the paper states that the data configuration and metrics are fully aligned with the AirfRANS benchmark.
- Metric family: official AirfRANS-style normalized-field MSE family.
- Surface + volume: yes.

The directly comparable full-task row in Table 5 is:

- `Vol = 0.0062`
- `Surf = 0.0089`
- `C_L error = 0.1042`
- `rho_L = 0.9998`

Important caveat:

- The abstract also quotes `volume flow field MSE = 0.0038` together with `surface pressure MSE = 0.0089`.
- For benchmark history, the safer row to use is the paper's AirfRANS-aligned `Volume / Surface` table, not the abstract shorthand.

### 3. SpiderSolver (NeurIPS 2025)

- Source: NeurIPS 2025 poster plus released code.
- Split / protocol: same AirfRANS-style split logic as Transolver, including the `scarce -> full_test` convention.
- Metric family: official AirfRANS MSE family.
- Surface + volume: yes.

AirfRANS full-task row:

- `Vol = 0.0017`
- `Surf = 0.0043`
- `C_L error = 0.0741`
- `rho_L = 0.9988`

This is the strongest clean full-task `Surf` / `Vol` result I found with clearly matching benchmark semantics.

## Partially Comparable or Not Directly Comparable Papers

These papers are still useful to mention in related work, but I would not mix their numbers into the main full-task `Surf MSE` / `Vol MSE` leaderboard without a caveat.

### GLOBE (arXiv 2025 / PhysicsNeMo example)

What matches:

- Uses AirfRANS `full`, `scarce`, `reynolds`, and `aoa`.
- Uses z-score-normalized field MSE in headline tables.

Why it is not directly comparable to the Transolver / SpiderSolver full-task row:

- The headline AirfRANS tables are validation-set tables, not official test-set leaderboard rows.
- Reporting is per field: `u_x`, `u_y`, `p`, and `p_s`.
- The paper also introduces physically nondimensionalized MAE tables, which are more interpretable but again a different contract.

How I would use it:

- Strong recent related work.
- Do not place it in the same table as official full-test `Surf MSE` / `Vol MSE`.
- It is reasonable to build a separate auxiliary table for `surface pressure` or per-field AirfRANS metrics, but only if we compare like-for-like:
  - same split family
  - same normalization
  - same field (`p_s` or surface pressure only)
  - same aggregation rule

One concrete auxiliary comparison we can now justify is:

| Method | Split / reporting contract | `p_s` z-score MSE | Safe to place in a GLOBE-style auxiliary table? | Notes |
| --- | --- | --- | --- | --- |
| GLOBE | `full` validation; per-field z-score MSE | `0.0039` | Yes | From GLOBE Table 3. |
| Our repo, PR `#2771` / run `q4hytsr6` | `full` validation at best-val epoch; per-field normalized `p_s` MSE | `0.00588` | Yes | The PR comment explicitly labels this as the best-validation checkpoint surface-pressure breakdown. |
| Transolver | `full` validation; per-field z-score MSE | `1.4200` | Yes | From the same GLOBE Table 3. |

Important caveat:

- this auxiliary table is useful for a `surface pressure only` comparison
- it is **not** a substitute for the official full-test `Surf MSE` / `Vol MSE` table
- the newer PR `#2824` comment also reports `surface_mse_p = 6.53e-3`, but its split labeling is less explicit than PR `#2771`, so `0.00588` is the cleaner field-level citation unless we re-pull the exact local logs summary

### AB-UPT, Transolver++, and Transolver-3

These are relevant CFD baselines, but not additional AirfRANS benchmark rows.

- `AB-UPT` benchmarks `ShapeNet-Car`, `AhmedML`, and `DrivAerML`, then later `SHIFT-SUV` and `SHIFT-Wing`; I did not find an AirfRANS benchmark in the original paper or the automotive / aerospace follow-up.
- `Transolver++` improves Transolver on six standard PDE benchmarks plus two industrial datasets, but the listed benchmarks are things like `Elasticity`, `Plasticity`, `Airfoil`, `Pipe`, `NS2d`, and `Darcy`, not AirfRANS.
- `Transolver-3` benchmarks `NASA-CRM`, `AhmedML`, and `DrivAerML`, and compares against `AB-UPT` and `Transolver++` there, not on AirfRANS.

### MARIO (2025)

What matches:

- Uses AirfRANS.
- Uses MSE on normalized outputs.

Why it is not a full-task benchmark comparator:

- The main AirfRANS comparison is on the `scarce` task, not `full`.
- Reporting is per field (`u_x`, `u_y`, `p`, `nu_t`, `p_s`), not official aggregate `Surf MSE` / `Vol MSE`.

Useful recovered scarce-task values:

- MARIO: `u_x = 0.152e-2`, `u_y = 0.113e-2`, `p = 0.240e-2`, `p_s = 2.700e-2`
- Transolver in the same scarce-style table: `u_x = 2.105e-2`, `u_y = 2.108e-2`, `p = 4.434e-2`, `p_s = 23.670e-2`

How I would use it:

- Good for a scarce-task subsection.
- Not valid as a replacement for full-task `Surf` / `Vol` benchmark rows.

### FLOW-GLIDE (Applied Sciences 2025)

Why I would exclude it from official benchmark comparison:

- The paper explicitly defines `Volume` and `Surface` as relative-L2 errors.
- That is not the official AirfRANS MSE family.

How I would use it:

- Mention only in a separate "different metric family" paragraph, or omit from a strict comparison table.
- Do not try to back-compute official `Surf MSE` from the published relative-L2 number alone. That conversion is not identifiable without extra information such as raw predictions or the exact target norms and aggregation statistics used in evaluation.

## Transolver Number Drift

I found two different Transolver AirfRANS full-task rows in the literature:

1. Original Transolver published row:
   - `Vol = 0.0037`
   - `Surf = 0.0142`

2. Later cited row that appears in SpiderSolver and in our local AirfRANS notes:
   - `Vol = 0.0025`
   - `Surf = 0.0080`

What I could verify directly:

- The Transolver README corrects the metric type from `Relative L2` to `MSE`.
- I did not find a first-party Transolver artifact in this pass that explains the numeric jump from `0.0037 / 0.0142` to `0.0025 / 0.0080`.

Recommendation:

- For a strict historical citation, use the original published Transolver row.
- If we also mention the later `0.0025 / 0.0080` row because it appears in follow-on work and in our repo notes, we should label it explicitly as a later cited / corrected Transolver baseline.

## Recommendation for Our Paper

If our model is evaluated on the official AirfRANS `full` task and we compute the official normalized-target benchmark metrics, then:

- headline comparison should use both `Surf MSE` and `Vol MSE`
- the safest external baselines are `Transolver`, `GeoANF`, and `SpiderSolver`
- `GLOBE`, `MARIO`, and `FLOW-GLIDE` should not be mixed into the same main comparison table without an explicit note about different split / aggregation / metric contracts

Concretely, the cleanest paper table for us is probably:

- `Our model`
- `Transolver`
- `GeoANF`
- `SpiderSolver`
- optional footnote for later cited `Transolver = 0.0080 / 0.0025`

And the repo's current status notes are right about one important thing:

- a surface-only score is not enough to claim a full AirfRANS benchmark win

## Are Our Repo Metrics Comparable?

Short answer:

- **split comparability:** yes
- **metric-family comparability:** yes on the current AirfRANS scorer
- **current scientific status:** yes on `Surf MSE`, no on the full `Surf MSE + Vol MSE` pair

Why the split is comparable:

- `target/icml2026/airfrans/data/split_airfrans.py` is generated from the official AirfRANS `manifest.json`
- it preserves the official task train/test lists
- it uses the same deterministic tail-10% validation carve-out as the official AirfRANS / Transolver / SpiderSolver code
- it preserves the benchmark rule `scarce_test = full_test`

Why the metric calculation is comparable:

- `target/icml2026/airfrans/data/prepare_airfrans.py` computes target stats from the AirfRANS training split only
- `target/icml2026/train.py` now evaluates AirfRANS in an explicit official-metric space:
  - model outputs are decoded back to raw target space
  - benchmark metrics are then recomputed with the official train-stat z-score transform
  - per-case channel means are averaged into `surface_mse` and `volume_mse`
- that means paper-facing AirfRANS metrics stay benchmark-faithful even if training uses a different target transform

Important repo caveat, now addressed in code:

- the shared trainer supports an optional `asinh_pressure` transform
- training may still use that flag
- benchmark reporting should not
- the repo scorer now enforces this separation for AirfRANS evaluation, so `surface_mse` and `volume_mse` stay on the official contract
- for historical runs logged before this fix, we should still verify the config before citing a number in the paper

Follow-up verification on the cited frontier run:

- PR `#2824`'s published command for run `3e0ce368` does **not** include `--asinh_pressure`, so that specific cited run was already on the benchmark-faithful target path
- the same PR comment also exposes a per-field row at the best checkpoint, including `surface_mse_p = 6.53e-3`
- that means the repo does have the ingredients for a GLOBE-style auxiliary `surface pressure` table
- however, the status memos still headline aggregate `surface_mse`, not the exact validation-side `surface_mse_p` number needed for a fully like-for-like GLOBE comparison table

## Current Repo Numbers

The latest status memo family gives two slightly different things:

- the **newest status memo** gives the latest surface headline
- the **most recent status memo that still reports the full pair** gives the latest explicit `surface + volume` comparison

| Source | Provenance | `Surf MSE` | `Vol MSE` | Fully comparable to official full-task literature table? | Interpretation |
| --- | --- | --- | --- | --- | --- |
| Repo latest surface headline | `STATUS_2026-04-22-0923` and `STATUS_2026-04-22-0008`, run `3e0ce368` / PR `#2824` | `0.002999` | not reported in newest memo | Surface-only: yes. Full pair: not enough evidence from that memo alone. | Strong latest surface result; better than SpiderSolver's `0.0043` on surface alone. |
| Repo latest explicit full pair | `STATUS_2026-04-21-1759`, PR `#2824` | `0.003` | `0.00764` | Yes, assuming default AirfRANS target transform was used. | Beats the external surface target but misses the external volume target badly. |
| Repo multi-seed follow-up | `STATUS_2026-04-21-1759`, PR `#2831` | `0.00333 / 0.00668 / 0.00857` | `0.00886 / 0.00901 / 0.01709` | Yes for metric family, again assuming default transform. | Confirms the surface win is real, but also confirms volume is still the blocker. |

So the current paper-safe summary of our own AirfRANS position is:

- **surface:** competitive and apparently better than the published `0.0043` reference
- **volume:** still far from the published `0.0017` reference
- **full benchmark claim:** not yet

If we want one compact comparison table for the paper draft right now, this is the honest version:

| Method | `Surf MSE` | `Vol MSE` | Status |
| --- | --- | --- | --- |
| Transolver (published row) | `0.0142` | `0.0037` | baseline |
| SpiderSolver | `0.0043` | `0.0017` | strongest clean published full-task reference |
| Ours (latest surface headline) | `0.002999` | `—` | surface-only headline, not enough for full-pair claim |
| Ours (latest explicit full pair) | `0.003` | `0.00764` | better on surface, much worse on volume |

## Sources

- Local repo context:
  - `analysis/ICML_2026_ABSTRACTS_PLAN.md`
  - `analysis/STATUS_*`
  - `target/icml2026/airfrans/program.md`
- Official AirfRANS:
  - [AirfRANS README](https://github.com/Extrality/AirfRANS)
  - [AirfRANS paper (arXiv)](https://arxiv.org/abs/2212.07564)
  - [AirfRANS NeurIPS / paper PDF](https://papers.nips.cc/paper_files/paper/2022/file/94ab7b23a345f93333eac8748a66c763-Paper-Datasets_and_Benchmarks.pdf)
  - [AirfRANS main.py](https://raw.githubusercontent.com/Extrality/AirfRANS/main/main.py)
  - [AirfRANS dataset.py](https://raw.githubusercontent.com/Extrality/AirfRANS/main/dataset.py)
  - [AirfRANS metrics.py](https://raw.githubusercontent.com/Extrality/AirfRANS/main/metrics.py)
- Transolver:
  - [Transolver paper](https://arxiv.org/abs/2402.02366)
  - [Transolver AirfRANS README](https://raw.githubusercontent.com/thuml/Transolver/main/Airfoil-Design-AirfRANS/README.md)
  - [Transolver AirfRANS main.py](https://raw.githubusercontent.com/thuml/Transolver/main/Airfoil-Design-AirfRANS/main.py)
- SpiderSolver:
  - [SpiderSolver README](https://raw.githubusercontent.com/Kai-Qi/SpiderSolver/main/README.md)
  - [SpiderSolver poster](https://neurips.cc/media/PosterPDFs/NeurIPS%202025/116641.png)
  - [SpiderSolver AirfRANS training script](https://raw.githubusercontent.com/Kai-Qi/SpiderSolver/main/AirfRANS/main_SpiderSolver_Airfoil.py)
  - [SpiderSolver AirfRANS evaluation script](https://raw.githubusercontent.com/Kai-Qi/SpiderSolver/main/AirfRANS/main_evaluation.py)
- GeoANF:
  - [GeoANF paper](https://www.mdpi.com/2076-3417/14/22/10685)
- MARIO:
  - [MARIO paper](https://arxiv.org/abs/2505.14704)
  - [ar5iv HTML](https://ar5iv.labs.arxiv.org/html/2505.14704v1)
- GLOBE:
  - [GLOBE paper](https://arxiv.org/abs/2511.15856)
  - [GLOBE HTML](https://arxiv.org/html/2511.15856)
  - [PhysicsNeMo AirFRANS example](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/cfd/external_aerodynamics/globe/airfrans/README.html)
- FLOW-GLIDE:
  - [FLOW-GLIDE paper](https://www.mdpi.com/2076-3417/15/19/10834)
- Other related CFD surrogate families:
  - [AB-UPT paper](https://arxiv.org/abs/2502.09692)
  - [AB-UPT for Automotive and Aerospace Applications](https://arxiv.org/abs/2510.15808)
  - [Transolver++ paper](https://arxiv.org/abs/2502.02414)
  - [Transolver-3 paper](https://arxiv.org/abs/2602.04940)
