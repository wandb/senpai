# TandemFoil2 — program context for senpai agents

This is the active SENPAI problem package: a clean Transolver-based implementation seeded from the `[ANON_FLAT_AGENT_REPO]` TandemFoil competition. See `KAGENT_SOURCE.md` for provenance.

## Task

Predict the 3D airflow velocity field around F1 front wings (tandem-foil geometries) given mesh-node features. The TandemFoilSet has seven pickle files spanning `raceCar` (land-inverted) and `cruise` (aerial-freestream) domains with structured holdouts on front-foil camber and Reynolds number (see `organizer/SPLITS.md`).

## Data contract

- Input `x`: `(N, 24)` float32 per mesh node — geometric + flow features (see `data.py`). **Do not edit `data.py`.**
- Target `y`: `(N, 3)` float32 — velocity components `(u, v, w)`.
- Mask `is_surface`: `(N,)` bool — mesh nodes on the airfoil surface. Velocity is zero there (no-slip).
- `N` varies per sample, up to ~300K points. Each sample is ~100K 3D points on average.
- Splits materialised under `/mnt/<pvc-mount>/datasets/tandemfoil/splits_v2/{train, val_single_in_dist, val_geom_camber_rc, val_geom_camber_cruise, val_re_rand}/` as individual `.pt` files. Runner: `organizer/prepare_splits.py` (one-off, reads pickles + manifest, writes .pt shards).

## Primary metric

- **`val/l2_error`** — mean L2 velocity error across all four val splits, equal-weighted. Lower is better.
- Tracked per split: `val/<split_name>/l2_error`.
- Ranking metric for the leaderboard.

## Constraints

- **96GB VRAM ceiling** per student pod. 100K-point samples are large — subsample, chunk, or use memory-efficient attention.
- **Timeout** is wall-clock, controlled by `SENPAI_TIMEOUT_MINUTES` (default 30 min for smoke runs). Honor it — training should checkpoint and exit cleanly.
- **Max epochs** via `SENPAI_MAX_EPOCHS` (default 50; current run uses 999). Early-stop on validation plateau.

## Files you may edit

- `train.py` — the model, optimizer, training loop, logging. Primary target of experimentation.
- `predict.py` — inference / prediction assembly. Edit when you change model I/O shape.
- `EXPERIMENT_JOURNAL.md` — append one entry per attempted experiment, success or failure. Failed experiments are the most valuable.

## Files that are read-only

- `data.py` — data loader contract. Changing this breaks comparability across students.
- `organizer/*` — split manifest, scoring, and materialisation are the ground-truth contract.
- `KAGENT_SOURCE.md` — attribution record.
- `README.md`, `program.md`, `KAGGLER_AGENT.md` — description and rules.

## Git workflow (critical)

This repo is pulled into the SENPAI runner as a **git submodule** under `target/tandemfoil2/`. All your commits, branches, and PRs live in THIS repo (`[ANON_TANDEMFOIL_REPO]`) on branch `kagent_royal_rumble` — **not in `[ANON_HARNESS_REPO]`**. Your entrypoint script sets this up automatically; when in doubt verify `git remote -v` points at `[ANON_TANDEMFOIL_REPO].git`.

- Base branch: `kagent_royal_rumble`.
- Your working branch: `$RESEARCH_TAG/student-$STUDENT_NAME` (set by the entrypoint).
- PR target: `gh pr create --repo [ANON_TANDEMFOIL_REPO] --base kagent_royal_rumble`.
- Never `git commit` in the parent senpai repo — that repo is read-only from the agent's perspective.
