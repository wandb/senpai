<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# autoresearch

This is an experiment to have the LLM do its own research on NN-CFD surrogates.

## Context

We are training a neural network surrogate for CFD (computational fluid dynamics) on the TandemFoilSet dataset. The task is full-field flow prediction: given airfoil geometry and flow conditions, predict velocity (Ux, Uy) and pressure (p) at every mesh node.

Key files:
- `prepare.py` — **read-only**. Dataset loading, preprocessing, collation. Do not modify.
- `train.py` — training script. Hyperparameters, optimizer, training loop, loss formulation. **You can modify this.**
- `transolver.py` — model architecture (Transolver with physics attention). **You can modify this.**
- `utils.py` — visualization. Do not modify (not relevant to metrics).
- `DATASET_REPORT.md` — dataset documentation. Read for context.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar12`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `prepare.py` — fixed data prep, dataset class, collation.
   - `train.py` — training script with Config dataclass. This is where you tune.
   - `transolver.py` — model architecture. This is where you experiment with the model.
   - `DATASET_REPORT.md` — dataset schema, value ranges, overset mesh structure.
4. **Verify data exists**: Check that `/mnt/new-pvc/datasets/tandemfoil/raceCar_single_randomFields.pickle` exists.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script stops after **10 epochs or 5 minutes** (whichever comes first).

**What you CAN do:**
- Modify `train.py` — hyperparameters, optimizer, loss formulation, batch size, learning rate schedule, loss weighting, normalization strategy, etc.
- Modify `transolver.py` — model architecture, attention mechanism, number of layers, hidden dimensions, activation functions, positional encoding, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed dataset, preprocessing, and data loading.
- Modify `utils.py`. Visualization is not part of the experiment.
- Install new packages or add dependencies beyond what's in `pyproject.toml`.

**The goal is simple: get the lowest validation losses.** We track two primary metrics:
- **Surface MAE** — mean absolute error on airfoil surface nodes (Ux, Uy, p). This is the most important metric for engineering applications.
- **Volume MAE** — mean absolute error on volume (field) nodes.
- **val/loss** — the combined validation loss (vol_loss + surf_weight * surf_loss).

Lower is better for all metrics. Surface accuracy matters most — these are the quantities engineers care about (forces, pressure distributions on the airfoil).

**VRAM** is a soft constraint. The GPUs have 96GB. Some increase is acceptable for meaningful metric gains, but it should not OOM.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
======================================================================
TRAINING COMPLETE
======================================================================
Best model at epoch 48
  Val total loss: 15.2345
  Volume  MAE:  Ux=4.38  Uy=1.71  p=113.7
  Surface MAE:  Ux=1.19  Uy=0.55  p=83.2
```

You can extract the key metrics from the log file:

```
grep "Val total loss\|Volume  MAE\|Surface MAE" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 7 columns:

```
commit	val_loss	surf_mae_Ux	surf_mae_p	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val/total_loss (e.g. 15.2345) — use 0.0 for crashes
3. surface MAE for Ux (e.g. 1.19) — use 0.0 for crashes
4. surface MAE for p (e.g. 83.2) — use 0.0 for crashes
5. peak memory in GB, round to .1f — use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short text description of what this experiment tried

Example:

```
commit	val_loss	surf_mae_Ux	surf_mae_p	memory_gb	status	description
a1b2c3d	15.2345	1.19	83.2	54.0	keep	baseline
b2c3d4e	12.1000	0.95	71.5	54.2	keep	increase surf_weight to 20
c3d4e5f	18.5000	1.45	95.0	54.0	discard	switch to L1 loss
d4e5f6g	0.0	0.0	0.0	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar12`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Formulate a hypothesis and modify `train.py` and/or `transolver.py`
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "Val total loss\|Volume  MAE\|Surface MAE" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If metrics improved (lower surface MAE / val_loss), you "advance" the branch, keeping the git commit
9. If metrics are equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Ideas to explore** (non-exhaustive):
- Loss formulation: surface weight, per-channel weighting, L1 vs MSE, gradient-based losses
- Learning rate schedule: warmup, cosine annealing, OneCycleLR
- Model architecture: number of layers, hidden dim, number of heads, slice count, MLP ratio
- Attention mechanism: different slice projections, local attention, multi-scale
- Input features: encoding improvements, positional encoding enhancements
- Normalization: per-sample vs global, different normalization strategies
- Data augmentation: if applicable to CFD meshes
- Optimizer: AdamW vs Adam, weight decay, gradient clipping
- Multi-scale or hierarchical approaches

**Timeout**: Default is 10 epochs / 5 minutes. Don't change this, it's your budget. You have 8 GPUs, so you can run this workflow up to 8 time sin parallel.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read the dataset report for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.
