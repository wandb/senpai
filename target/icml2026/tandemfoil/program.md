<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# TandemFoilSet

TandemFoilSet is the load-bearing benchmark inside the ICML 2026 sprint target.

## Problem

We are training machine-learning surrogates for computational fluid dynamics. The general task is full-field flow prediction: given geometry, operating conditions, and other problem-specific inputs, predict physically meaningful flow quantities such as velocity and pressure over a mesh, graph, or point cloud.

Benchmark details for the tandemfoil dataset live in `data/README.md`. Shared
training now happens via `../train.py`, while the tandemfoil-specific pipeline
stays under `data/`.

## Codebase

- `../train.py` — shared ICML sprint trainer. **Primary editable entrypoint.**
- `data/prepare.py` — tandemfoil dataset loading and collation. **Read-only during normal experiment PRs.**
- `data/prepare_multi.py` — tandemfoil-specific preprocessing and feature engineering. **Read-only during normal experiment PRs.**
- `data/utils.py` — visualization. **Read-only.**
- `data/README.md` — benchmark splits, dataset assumptions, and problem-local documentation.

## Reference Parity

TandemFoilSet is the only dataset in this target that is currently pinned to a
specific historical frontier recipe rather than a purely clean-sheet baseline.

- historical report anchor:
  - `PR #2319`, the best merged single-seed result in the finished-run cache
- clean-target parity goal:
  - `PR #2379`, the later merged two-seed baseline with the ANP cross-foil
    decoder
- merged lineage that this target now mirrors:
  - `#2319 -> #2350 -> #2357 -> #2379`

Pinned source refs:

- transform, denorm, residual, legacy metric contract, and merged feature stack:
  - `origin/noam@d743ba27eb1c561750f55daeefadcbe41e2b8421`
- ANP cross-foil decoder implementation:
  - `origin/frieren/anp-surface-decoder@7999a2e`

## Metrics

**This benchmark does not have a published external leaderboard metric like AirfRANS or DrivAerML.** The source-of-truth contract is:
- the `kagent` competition split design in `target/icml2026/tandemfoil/data/split_manifest_tandemfoilset_v2.json`
- the historical `origin/noam` validation path that produced the paper-facing `p_*` numbers
- the merged `#2379` TandemFoil recipe described above, with ANP code sourced
  from `origin/frieren/anp-surface-decoder`

Use the following metric definitions exactly:

- **Primary merge / ranking score for this ICML sprint**
  - Equal-weight mean of the four split-specific **surface-pressure MAEs** on:
    - `val_single_in_dist`
    - `val_geom_camber_rc`
    - `val_geom_camber_cruise`
    - `val_re_rand`
  - The `kagent` split document defines the balanced validation tracks and says they are summarized by equal-weight average surface MAE.
  - For this paper sprint, we pin that surface metric to the **pressure channel** because the historical `senpai` frontier and abstract use split-specific surface-pressure MAE as the paper-facing scalar.
  - In the shared trainer this scalar is logged as `val_eq4/surface_pressure_mae`, and mirrored as `val_primary/surface_pressure_mae` for harness ranking.

- **Per-split scalar**
  - For any validation split `S`, compute:
  - `surface_pressure_mae(S) = mean_{all valid surface nodes in S} |p_hat - p_true|`
  - `p` is the pressure / `C_p` channel **after full denormalization back to the original target space**.
  - Aggregation is **global over all valid surface nodes in the split**, not per-case then averaged.

- **Secondary diagnostics**
  - `mae_surf_Ux`, `mae_surf_Uy`, `mae_surf_p`
  - `mae_vol_Ux`, `mae_vol_Uy`, `mae_vol_p`
  - combined validation loss per split

- **Historical note**
  - Earlier `origin/noam` frontier numbers use the legacy names `p_in`, `p_tan`, `p_oodc`, and `p_re`.
  - Those are also **surface-pressure MAEs after denormalization**, but they were measured on an older legacy split manifest.
  - Do **not** compare raw values between the legacy `p_*` contract and the new `kagent` v2 contract without restating the split definition.
  - The shared trainer logs these separately as:
    - `legacy_noam/*`
    - `icml2026_v2/*`

Lower is better. For the paper sprint, the decision-driving quantity is **surface pressure MAE on the explicit validation tracks**, with the equal-weight 4-way average used as the benchmark summary.

For held-out reporting after model selection, the trainer also computes the
analogous `test_eq4/surface_pressure_mae` summary when the four matching test
tracks are available in the split manifest.

## Transform and Residual Semantics

For TandemFoilSet, the clean trainer now mirrors the noam target-space contract
instead of using the generic shared transform path:

- compute per-sample `Umag` and `q` from the masked flow targets
- apply `_phys_norm` / `_phys_denorm` equivalents in velocity and pressure
- apply `asinh_pressure` in `C_p` space, not in raw pressure space
- compute paper-facing errors only after full denormalization back to physical
  units

`residual_prediction` has the noam meaning here:

- subtract the freestream baseline from the normalized targets before loss
- add the freestream baseline back before denormalization and metric logging

This is intentionally separate from the model-side pressure-prior addition path.
That mechanism is retained as its own option and is no longer described as
"residual prediction."

## Frontier Mechanisms

The clean target now carries the merged TandemFoil mechanisms that bridge
`#2319` to `#2379`:

- TE coordinate frame
- wake-deficit feature
- panel-method `C_p` feature with tandem-only scaling
- wake-angle feature
- vortex-panel induced-velocity feature
- zero-init surface refinement head
- optional ANP cross-foil surface decoder
- noam-style TandemFoil denorm and legacy metric accumulation

AirfRANS and DrivAerML do not inherit these TandemFoil-specific mechanisms.

## Sources

- `kagent` split design: <https://github.com/tcapelle/kagent/blob/main/cfd-competition/organizer/SPLITS.md>
- historical metric path: `origin/noam:cfd_tandemfoil/train.py`

**VRAM**: GPUs have 96GB.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it.

**Timeout**: Each training run is capped by time or epochs. Do not override this.

## Roles

Research is coordinated through GitHub PRs with an advisor/student model. GitHub Issues are used for communication with the human researcher team.
