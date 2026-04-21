# SENPAI Research State

- **Date:** 2026-04-21 (post-prune relaunch refocus)
- **Branch:** radford
- **Fleet status:** 50 live students after the relaunch reset
- **Queue hygiene:** 29 open PRs from retired students were closed on 2026-04-21; treat those lanes as historical only
- **Current relaunch budget:** inherit the pod env defaults
  - `SENPAI_TIMEOUT_MINUTES=360`
  - `SENPAI_MAX_EPOCHS=999`
  - do not hardcode `180` minute runs
  - do not hardcode `9999` epoch caps

## Paper-Facing Snapshot

| Dataset | Paper-facing metric | Current best | Target / reference | Current read |
|---|---|---|---|---|
| TandemFoil | `test_primary/surface_pressure_mae` | **33.88** from #2810 | no single packaged external scalar | strong enough that Tandem is no longer the main bottleneck |
| AirfRANS | `test_primary/surface_mse` | **0.003** from #2824 | `0.0043` | surface test target is already beaten |
| AirfRANS full contract | `surface_mse + volume_mse` view | surface good, volume still around **0.0076+** | `volume_mse < 0.0017` | still not clean on the full published surface+volume pair |
| DrivAerML | `test_primary/surface_rel_l2_pct` | **6.24%–6.29%** | `3.71%` nominal reference | still the weakest benchmark and the main empirical gap |

## Steering Anchors

These are still useful for choosing the next experiments, even though test is the paper-facing number.

| Dataset | Steering metric | Current anchor |
|---|---|---|
| TandemFoil | `val_primary/surface_pressure_mae` | **30.10** from #2810 |
| AirfRANS | `val_primary/surface_mse` | **0.001095** pending in #2823 |
| DrivAerML | `val_primary/surface_rel_l2_pct` | **4.619%** from #2691 |

## Main Scientific Goal

The goal is **not** three unrelated benchmark-specific wins.

The goal is a shared recipe whose core changes more or less work across:

- `target/icml2026/tandemfoil/`
- `target/icml2026/airfrans/`
- `target/icml2026/drivaerml/`

It is fine if LR, scheduler, WD, or point-budget differ by dataset.
It is not fine if the abstract only works because every dataset needs a different core idea.

Because AirfRANS surface test is already over the line and Tandem is healthy, the next wave should be:

- **DrivAerML-weighted**
- **cross-dataset by default**
- **test-aware when reporting**

## Mandatory Config Rules

- `--no-use-ema` is mandatory everywhere
- `--epochs 999` is mandatory because the default epoch count is too small
- inherit `SENPAI_TIMEOUT_MINUTES=360` from the pod env unless there is a clearly justified shorter run
- inherit `SENPAI_MAX_EPOCHS=999` from the pod env unless there is a clearly justified override
- for DrivAerML keep:
  - `--batch-size 1`
  - `--drivaerml-train-surface-points 50000`
  - `--drivaerml-eval-surface-points 50000`
  - `--max-train-batches 394`
  - `--max-eval-batches 200`
- when reporting benchmark-facing results, always keep the reported **test** metric beside its target or reference

## Default Assignment Pattern

When a student is free, default to a **single hypothesis family across all three datasets** in one PR.

Use the student's 8 GPUs roughly like this:

- `1` TandemFoil run
- `1` AirfRANS run
- `2-4` DrivAerML runs
- remaining GPUs for the most decision-critical nearby variants

The resulting PR should let the same student answer:

- did this transfer broadly?
- did it help AirfRANS and TandemFoil but fail on DrivAerML?
- is it too dataset-specific to support the abstract story?

Single-dataset PRs are still acceptable only for:

- AirfRANS benchmark cleanup or best-checkpoint / test cleanup
- a clearly justified DrivAerML rescue lane
- minimal Tandem anchor locking

## Keep And Prioritize

### 1. Cross-dataset transfer evidence

These are the clearest matches to the abstract story and should stay live:

- `#2834` askeladd — no-Lookahead ablation
- `#2825` levi — 3L architecture transfer

New PRs should look more like these.

### 2. DrivAerML rescue lanes

These are the highest-value single-benchmark lanes because DrivAerML is now the main gap:

- `#2873` chrome — LR headroom above `5e-4`
- `#2868` norman — `2L/512d` and `3L/512d`
- `#2867` historia — `3L/256d` and `3L/384d`
- `#2855` eren — seed replication
- `#2853` zenitsu — `T_max` neighborhood
- `#2851` shinobu — WD ablation
- `#2849` rei — conservative LR neighborhood
- `#2814` taki — mild regularization (`WD=1e-3 + gc=0.5`) if still behaving sanely

### 3. Minimal Tandem anchor lanes

Keep only a small Tandem lane alive:

- `#2864` senku — 2-layer depth frontier
- `#2840` alphonse — `lr=1e-4` multi-seed replication
- `#2837` fern — `3L/256d` width check
- `#2842` tanjiro — compound lane only if it is already near a useful readout

### 4. Minimal AirfRANS lane

AirfRANS should now be narrow:

- `#2823` kakashi — preserve / rebase / merge the current best validation anchor
- `#2801` edward — only if it yields a broadly useful transferable lesson rather than another benchmark-specific trick

## Retask Or De-Emphasize

These open lanes are **not** central to the new plan and should be retasked if they are not already very close to a useful answer:

- `#2820` haku — AirfRANS local stability line
- `#2786` thorfinn — AirfRANS `T_max=7`
- `#2770` hinata — AirfRANS `WD=5e-3`
- `#2801` edward — de-emphasize unless it is near review and clearly transferable
- `#2842` tanjiro — de-emphasize if it is turning into another Tandem-only local sweep
- `#2857` shouko — DrivAer MLP-ratio lane is lower value than LR / depth / seeds / scheduler

Do **not** recreate the following families by default:

- broad AirfRANS WD / LR / MLP-ratio neighborhoods around already-strong lines
- broad Tandem LR / WD / `T_max` neighborhood mapping
- DrivAerML Lion sweeps
- DrivAerML clipping-heavy compounds that repeat already weak signals

## Queue Notes

- 29 PRs owned by retired students were closed during this refocus; if old notes mention them, they are no longer active
- do not assign work to retired students
- if many newly relaunched students are idle, assign cross-dataset matrices first, not more one-benchmark local sweeps

## Immediate Priorities

1. Keep DrivAerML as the main destination for fresh capacity
2. Prefer cross-dataset PRs over single-benchmark PRs whenever possible
3. Preserve the minimal AirfRANS and Tandem anchors without letting them dominate the queue
4. Report test metrics beside SOTA targets / references when summarizing progress

## Most Recent Human Guidance

The current human direction is:

- close stale dead-student PRs
- focus the next wave on DrivAerML and cross-dataset evidence
- keep pushing toward an ICML abstract story based on a shared recipe, not three disconnected wins
