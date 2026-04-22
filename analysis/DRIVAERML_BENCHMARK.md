# DrivAerML Benchmark History and Comparison Contract

This note mirrors `AIRFRANS_BENCHMARK.md`, but DrivAerML needs a stricter caveat up front:
there is no single canonical public scalar in the literature analogous to AirfRANS
`Surf MSE` / `Vol MSE`. Instead, papers report a family of per-field relative-L2 metrics,
usually on surface pressure, wall shear stress, velocity, and volume pressure.

For our current repo, the only paper-safe direct comparison object is:

- **surface pressure relative-L2 (%)**
- averaged **per case**
- on the public `400 train / 34 effective val / 50 test` split family
- computed on **unnormalized** predictions and targets

That means DrivAerML should currently be written as a **surface-pressure transfer benchmark**,
not as a full multi-field benchmark win.

## Executive Summary

The clean literature history is not a single ladder. It breaks into three tiers:

1. **Early / broader DrivAerML context**
   - `X-MeshGraphNet`
   - `DoMINO`
   - `A Benchmarking Framework for AI models in Automotive Aerodynamics`
2. **Direct public-split surface-pressure comparators**
   - `AB-UPT`
   - `Transolver-3`
3. **Our current repo position**
   - best provenanced test: `6.244%`
   - best provenanced validation: `4.6190%`

If we want the safest paper table right now, it should be:

| Method | `p_s` relative-L2 (%) | Directly comparable to our current repo? | Notes |
| --- | --- | --- | --- |
| AB-UPT | `3.82` | Yes | First strong public-split benchmark row. |
| Transolver-3 | `3.71` | Yes | Strongest directly comparable published `p_s` row I found. |
| Ours | `6.244` | Yes | Current best provenanced test result in local status / PR history. |

If we want extra historical context, we can add later baseline rows recovered from
the later `AB-UPT` / `Transolver-3` comparison family:

- `Transolver`: `4.81`
- `Transolver++`: `4.12`

But those two should be labeled as **secondary-provenance baseline rows**, not as the
canonical standalone DrivAerML benchmark publications.

## Literature Map

| Paper / result | Venue | Split / eval contract | Surface pressure reported? | Surface + volume coverage? | Directly comparable to our current repo `surface_rel_l2_pct`? | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| X-MeshGraphNet | arXiv 2024 | DrivAerML surface-field study, but not a clean public `400 / 34 / 50` benchmark row | not a stable paper-facing scalar to cite | surface only | No | Important early DrivAerML surrogate paper, but not the row I would use in a comparison table. |
| DoMINO | arXiv 2025 | Relative-L2 on DrivAerML, with a different split story and strong OOD emphasis | Yes: `p_s = 15.05%` (`11.81%` area-weighted) | Yes | No | First strong quantitative surface+volume DrivAerML field baseline I found. |
| Benchmarking Framework | arXiv 2025 | common validation benchmark inside PhysicsNeMo-CFD; different split and metric suite | Yes | Yes | No | Best apples-to-apples multi-model paper, but not the same public `400 / 34 / 50` test contract. |
| Transolver baseline inside AB-UPT | TMLR 2025 | public `400 train / 34 effective val / 50 test`; mean per-case relative-L2 on unnormalized targets | Yes: `p_s = 4.81%` | partial in main table | Yes | Best direct provenance I found for a non-Transolver-paper DrivAerML `Transolver` row. |
| AB-UPT | TMLR 2025 | public `400 train / 34 effective val / 50 test`; mean per-case relative-L2 on unnormalized targets | Yes: `p_s = 3.82%` | Yes | Yes | First strong directly comparable public-split row. |
| Transolver-3 | arXiv 2026 | same `400 / 34 / 50` public contract; mean per-case relative-L2 on unnormalized targets | Yes: `p_s = 3.71%` | Yes | Yes | Strongest directly comparable surface-pressure reference I found. |
| Ours, latest best reported test | local status / PR ledger | repaired public split; mean per-case relative-L2 on unnormalized `surface_cp` | Yes: `6.244%` | No benchmark-comparable volume story | Yes, on surface pressure only | Still behind the published public-split benchmark band. |

## What Counts As A Valid Comparison?

These pieces need to line up before two DrivAerML numbers are honestly comparable.

### Split protocol

From the later public-split literature used by `AB-UPT` and `Transolver-3`:

- total cases: `500`
- training: `400`
- test: `50`
- validation: `50`, but `16` are hidden / unavailable
- effective public validation: `34`

Our packaged public processed split now matches that repaired public contract:

- `400 train`
- `34 val`
- `50 test`
- `0` excluded repaired public cases

Relevant local files:

- `target/icml2026/drivaerml/data/split_drivaerml.py`
- `target/icml2026/drivaerml/data/split_manifest_drivaerml.json`
- `target/icml2026/core/datasets.py`

### Metric family

The directly relevant DrivAerML papers use:

- per-case relative-L2 on **unnormalized** targets and predictions
- averaged across cases
- reported on the percent scale

Formally:

- `rel_l2_case = 100 * ||Y_hat - Y||_2 / ||Y||_2`
- dataset score = arithmetic mean of `rel_l2_case` over the evaluation split

Important details:

- this is **not** an MSE benchmark like AirfRANS
- this is **not** a pooled-all-points global relative-L2
- chunked inference is allowed only if the full case is reconstructed before computing
  the case-level ratio

### Target family

The literature is broader than our current repo target.

Common published DrivAerML quantities:

- surface pressure `p_s`
- wall shear stress `tau`
- volume velocity `u`
- volume pressure `p_v`
- sometimes vorticity `omega`

Our current repo target is narrower:

- packaged `surface_cp.npy`
- surface-first by default
- optional small processed volume subset only

So the paper-safe statement is:

- **surface pressure comparison:** yes
- **full multi-field DrivAerML comparison:** not yet

## Literature History

### Is there an original Transolver DrivAerML benchmark?

Not as a clean standalone benchmark row that I would cite.

What I could verify:

- the original `Transolver` paper says the method excels on large-scale industrial
  simulations including car design
- but I did **not** find a published DrivAerML table row in that original paper
- the original automotive benchmark associated with `Transolver` is the earlier car-design
  family, not the later public DrivAerML benchmark contract

So for DrivAerML specifically, the correct citation pattern is:

- **not** "original Transolver paper benchmarked DrivAerML"
- **yes** "later papers trained / evaluated Transolver on DrivAerML as a baseline"

### 1. X-MeshGraphNet (arXiv 2024)

This is the earliest DrivAerML surrogate paper in the chain I found.

Why it matters:

- it establishes DrivAerML as a serious ML surrogate benchmark
- it predicts surface pressure and wall-shear-related fields

Why I would not use it as a main numeric baseline:

- I did not find a clean public-split `400 / 34 / 50` benchmark row to cite in the same
  paper-facing way as `AB-UPT` or `Transolver-3`
- it is better treated as early context than as the benchmark anchor for our table

### 2. DoMINO (arXiv 2025)

This is the first strong quantitative DrivAerML **surface + volume** field baseline I found.

Reported test metrics include:

- surface pressure `p_s = 0.1505` (`15.05%`)
- area-weighted surface pressure `0.1181` (`11.81%`)
- volume pressure `p_v = 0.2193` (`21.93%`)
- volume velocity:
  - `u_x = 0.2397`
  - `u_y = 0.5025`
  - `u_z = 0.4567`

Why it is only partial context for us:

- it is not the same clean public benchmark split story used later by `AB-UPT` and
  `Transolver-3`
- `AB-UPT` explicitly notes that the DoMINO comparison split contains a `20%` OOD subset
  chosen by drag-force range

So I would cite DoMINO as:

- the first strong multi-field DrivAerML baseline
- **not** the main apples-to-apples comparator for our current repo table

### 3. A Benchmarking Framework for AI models in Automotive Aerodynamics (arXiv 2025)

This is the best multi-model DrivAerML benchmarking paper I found.

What it adds:

- common evaluation of `DoMINO`, `X-MeshGraphNet`, and `FIGConvNet`
- both surface and volume metrics
- additional engineering metrics such as drag / lift correlations

Representative validation metrics from the paper:

- surface pressure relative-L2:
  - `X-MeshGraphNet = 0.14`
  - `FIGConvNet = 0.21`
  - `DoMINO = 0.10`
- area-weighted surface pressure relative-L2:
  - `X-MeshGraphNet = 0.14`
  - `FIGConvNet = 0.14`
  - `DoMINO = 0.08`
- volume pressure relative-L2 for `DoMINO`:
  - `0.1042`

Why it is not our main table:

- it uses a different split / benchmark harness (`436 train / 48 validation` in the paper)
- it is a **validation benchmark**, not the later public `400 / 34 / 50` test contract

So I would use it as:

- the best broader-literature context section
- not the primary direct comparison table for our current repo

### 4. AB-UPT (TMLR 2025)

This is the first strong DrivAerML paper that lines up well with our current repo metric.

Importantly for your question, `AB-UPT` does **not** only compare against abstract
families; it explicitly benchmarks a `Transolver` baseline on DrivAerML.

Public DrivAerML row:

- `p_s = 3.82`
- `u = 5.93`
- `omega = 35.1`
- `tau = 7.29`
- `p_v = 6.08`

Why it is directly relevant:

- split is the public `400 / 34 / 50` family
- metric is mean per-case relative-L2 on unnormalized predictions / targets
- it reports the exact surface pressure quantity we can currently compare

Important nuance:

- AB-UPT is a **full surface + volume** model
- our current repo is only directly comparable on `p_s`

AB-UPT also provides the cleanest direct-provenance `Transolver` baseline row I found on
DrivAerML:

- `Transolver` on DrivAerML:
  - `p_s = 4.81`
  - `u = 6.78`
  - `omega = 38.4`

Why this matters:

- this is not just a row copied from another paper; AB-UPT states that it benchmarks
  `Transolver` against the other neural surrogates on the same public-split DrivAerML
  contract
- AB-UPT also states that it does **not** include `Transolver++` there because of
  reproducibility issues

So if we want to include a `Transolver` DrivAerML row in our paper, the strongest
provenance is:

- cite `AB-UPT` for the `Transolver` baseline
- label it as a later benchmarked baseline, not as an original Transolver-paper result

### 5. Transolver-3 (arXiv 2026)

This is the strongest directly comparable DrivAerML surface-pressure reference I found.

DrivAerML Table 4 row:

- `p_s = 3.71`
- `u = 4.14`
- `tau = 5.85`
- `p_v = 5.72`

Why it matters:

- same public `400 / 34 / 50` split family
- same per-case relative-L2 on unnormalized targets
- strongest surface-pressure row among the directly comparable papers I found

It also gives useful secondary-provenance baseline rows:

| Method | `p_s` | `u` | `tau` | `p_v` | How to use it |
| --- | --- | --- | --- | --- | --- |
| Transolver | `4.81` | `6.78` | `8.95` | `7.74` | Historical context only; same surface-pressure / velocity baseline family, with additional field coverage in the later paper. |
| Transolver++ | `4.12` | `4.70` | `6.42` | `6.70` | Historical context only; recovered from later comparison table. |

This gives us a clean provenance story:

- `Transolver` baseline on DrivAerML exists in `AB-UPT`
- `Transolver-3` later extends the comparison family and adds a broader field table
- `Transolver++` only appears in the later `Transolver-3` comparison table, not in the
  original `AB-UPT` benchmark table

## Are Our Repo Metrics Comparable?

Short answer:

- **split comparability:** yes
- **metric comparability:** yes, for surface pressure
- **full-task comparability:** no, because our repo is still surface-first on DrivAerML

Why the split is comparable:

- `target/icml2026/drivaerml/data/split_drivaerml.py` enforces the repaired public
  `400 / 34 / 50` contract
- `target/icml2026/core/datasets.py` validates the manifest against the restored public
  case IDs and requires `excluded_case_count = 0`

Why the metric is comparable:

- `target/icml2026/train.py` computes DrivAerML paper-facing metrics on
  **unnormalized** targets and predictions
- when point-limited evaluation is enabled, the trainer stores numerator /
  denominator pieces per case and reconstructs the exact full-case relative-L2 before
  averaging
- that matches the `AB-UPT` / `Transolver-3` metric contract for `p_s`

Where the repo still diverges from the broader literature:

- the packaged sprint path is surface-first
- the processed volume subset is tiny (`15 train / 1 val`) and has no benchmark-grade
  public test contract
- so we should not imply a multi-field DrivAerML comparison yet

## Current Repo Position

The latest `STATUS_*` files and PR ledger give a consistent picture:

- best provenanced validation: `4.618963354953193`
  - run `k8qtsxxz`
  - PR `#2691`
- best provenanced test: `6.244070498131545`
  - run `qx7z7if3`
  - PR `#2648`

There is also a same-day advisor issue draft that mentions a stronger approximate line
around `3.997%` validation and `5.93%` final test for a "no-compile AdamW" family, but
it does not currently have the same run-level provenance, so I would **not** use that in
the paper table without confirming the exact run / PR first.

| Source | Provenance | `surface_rel_l2_pct` | Directly comparable to published `p_s` rows? | Interpretation |
| --- | --- | --- | --- | --- |
| Repo best reported test | `STATUS_2026-04-22-0923` and `STATUS_2026-04-22-0008`, run `qx7z7if3`, PR `#2648` | `6.244%` | Yes | Current best reportable test result, but still well above `3.71%` / `3.82%`. |
| Repo best reported validation | `STATUS_2026-04-22-0008`, run `k8qtsxxz`, PR `#2691` | `4.6190%` | Not directly, because it is validation not test | Shows the stack can enter the mid-4% regime, but the test story did not move with it. |
| PR `#2691` best-checkpoint test | run `k8qtsxxz` | `6.29%` | Yes | Better validation than older runs, but not a new test breakthrough. |

So the honest current paper summary is:

- **surface pressure:** benchmark-comparable metric, but still behind the published frontier
- **volume / wall shear / integrated quantities:** not yet a valid repo-level comparison story
- **DrivAerML role in the paper today:** transfer evidence with caveats, not a benchmark win

## Recommendation For Our Paper

If we write DrivAerML with the same discipline as AirfRANS, the safest approach is:

- compare **only** `surface pressure relative-L2 (%)`
- use `AB-UPT` and `Transolver-3` as the main published anchors
- optionally add `Transolver` and `Transolver++` as later-table historical baselines
- keep `DoMINO` and the `Benchmarking Framework` in a broader-context subsection, not in
  the same main benchmark table
- state explicitly that our current repo target is **surface-first**, not the full
  multi-field DrivAerML task

The cleanest direct-comparison table right now is:

| Method | `p_s` relative-L2 (%) | Note |
| --- | --- | --- |
| Transolver baseline (reported by AB-UPT) | `4.81` | later benchmarked baseline, not an original Transolver-paper DrivAerML row |
| AB-UPT | `3.82` | first strong public-split benchmark anchor |
| Transolver-3 | `3.71` | strongest directly comparable published row I found |
| Ours | `6.244` | current best provenanced test |

## Sources

- Local repo context:
  - `analysis/STATUS_2026-04-22-0008_radford_post_restart_metric_update.md`
  - `analysis/STATUS_2026-04-22-0923_radford_live_status_after_cross_dataset_wave.md`
  - `analysis/ADVISOR_ISSUE_2026-04-22_drivaerml_sampling_and_tandem_debugging.md`
  - `target/icml2026/drivaerml/program.md`
- DrivAerML dataset:
  - [DrivAerML paper](https://arxiv.org/abs/2408.11969)
- Early / broader DrivAerML surrogate papers:
  - [X-MeshGraphNet](https://arxiv.org/abs/2411.17164)
  - [DoMINO](https://arxiv.org/abs/2501.13350)
  - [A Benchmarking Framework for AI models in Automotive Aerodynamics](https://arxiv.org/abs/2507.10747)
- Direct public-split comparators:
  - [AB-UPT (OpenReview)](https://openreview.net/forum?id=nwQ8nitlTZ)
  - [AB-UPT (arXiv)](https://arxiv.org/abs/2502.09692)
  - [Transolver-3](https://arxiv.org/abs/2602.04940)
