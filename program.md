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

To set up a new research session, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar12`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the base branch**: `git checkout -b autoresearch/<tag>` from current main. This branch represents the "current best" — experiments that improve on it get merged back in.
3. **Read the in-scope files**: Read these files for full context:
   - `prepare.py` — fixed data prep, dataset class, collation.
   - `train.py` — training script with Config dataclass. This is where you tune.
   - `transolver.py` — model architecture. This is where you experiment with the model.
   - `DATASET_REPORT.md` — dataset schema, value ranges, overset mesh structure.
4. **Verify data exists**: Check that `/mnt/new-pvc/datasets/tandemfoil/raceCar_single_randomFields.pickle` exists.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row in the main worktree. This file stays untracked by git.
6. **Run baseline**: Run `train.py` as-is on GPU 0 to establish baseline metrics. Record in `results.tsv`.
7. **Confirm and go**: Confirm setup looks good, then begin parallel experimentation.

Once you get confirmation, kick off the experimentation using the worktree-based parallel workflow described below.

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

## GPU pool and worktrees

You have **8 GPUs** (indices 0–7). Each GPU can run one experiment at a time. GPUs are a shared pool — allocate them flexibly based on what ideas need exploration.

Each experiment runs in its own **git worktree**, which gives it an isolated copy of the repo. This means multiple experiments can modify `train.py` and `transolver.py` simultaneously without conflicts.

### Worktree lifecycle

```bash
# Create a worktree for an idea (branch from wherever makes sense)
git worktree add ../senpai-<name> -b exp/<name>

# Run an experiment in that worktree on a specific GPU
cd ../senpai-<name>
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py > run.log 2>&1 &

# When done, collect results, then clean up losers
git worktree remove ../senpai-<name>  # also deletes the branch if merged
```

### Branching strategy

Branch from wherever your idea needs:
- **New idea from scratch**: branch from `autoresearch/<tag>` (the current best)
- **Iterating on an idea**: keep working in the same worktree, committing as you go
- **Variant of a previous experiment**: branch from that experiment's branch

An idea might take 1 run or 5 iterative runs. The worktree persists as long as the idea is being explored. You can dedicate 4 GPUs to deeply exploring one architectural idea while running 4 independent smaller experiments on the others.

### Promoting a winner

When an experiment produces better results than the current best:
1. Merge its branch into `autoresearch/<tag>`: `git checkout autoresearch/<tag> && git merge exp/<name>`
2. This advances the baseline. Future experiments can branch from this new state.
3. Clean up the worktree.

When an experiment doesn't improve, just delete the worktree and branch.

## The experiment loop

LOOP FOREVER:

1. **Survey the state**: Check which GPUs are free, review `results.tsv` for the current best. Query W&B to see what other agents have tried (use the wandb skill's `wandb_helpers.runs_to_dataframe()` to check recent runs in the "senpai" project). Avoid duplicating experiments that are already running or completed.
2. **Plan**: Decide how to allocate free GPUs. Consider:
   - Is there an active idea that needs more iterations? Give it another GPU run.
   - Are there new ideas worth exploring? Spin up new worktrees.
   - Can you run independent ideas in parallel on available GPUs?
3. **For each experiment**:
   a. Create a worktree (or reuse an existing one for iterations on the same idea)
   b. Modify `train.py` and/or `transolver.py` in that worktree
   c. Git commit in the worktree
   d. Run: `cd ../senpai-<name> && CUDA_VISIBLE_DEVICES=<gpu_id> python train.py --wandb_group <idea-name> --wandb_name <run-description> > run.log 2>&1 &`
   e. Track which GPU is running what
4. **Collect results** as experiments finish:
   a. `grep "Val total loss\|Volume  MAE\|Surface MAE" ../senpai-<name>/run.log`
   b. If grep is empty, the run crashed — `tail -n 50 ../senpai-<name>/run.log` for the traceback
   c. Record in `results.tsv` (in the main worktree — this file stays untracked)
   d. Decide: keep (merge) or discard (delete worktree)
5. **Iterate**: Use results to inform the next batch of experiments. Combine near-misses, go deeper on promising directions, abandon dead ends.

### Exploring ideas in depth

Some ideas need multiple iterations before they show results. For example, a new attention mechanism might need:
- Run 1: basic implementation (might crash or perform poorly)
- Run 2: fix bugs, tune hyperparams for the new mechanism
- Run 3: combine with the best loss formulation
- Run 4: final tuning

This is fine. Keep the worktree alive, keep iterating on the same branch. Use as many GPUs as the idea deserves. Not every GPU needs to explore a different idea.

Use `--wandb_group` to tie related runs together. All iterations on the same idea should share a group name (e.g. `--wandb_group multi-scale-attn`). Use `--wandb_name` to label each individual run (e.g. `--wandb_name "v1-basic"`, `--wandb_name "v2-fix-norm"`). This makes it easy to compare iterations in the W&B UI.

### Managing GPU allocation

Keep a mental model of GPU allocation:
```
GPU 0: exp/multi-scale-attention (running, started 2min ago)
GPU 1: exp/channel-weighted-loss (running, started 4min ago)
GPU 2: free
GPU 3: exp/larger-model-v2 (running, iteration 3)
...
```

Check if a GPU is free: `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader` or check if the background process finished.

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

**Timeout**: Default is 10 epochs / 5 minutes. Don't change this, it's your budget.

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run in the same worktree. If the idea itself is fundamentally broken, log "crash" in the tsv, delete the worktree, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read the dataset report for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.
