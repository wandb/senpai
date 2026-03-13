<!--
SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
SPDX-License-Identifier: Apache-2.0
SPDX-PackageName: senpai
-->

# senpai

Autonomous neural network research on CFD surrogates, coordinated through GitHub PRs.

## Context

We are training a neural network surrogate for CFD (computational fluid dynamics) on the TandemFoilSet dataset. The task is full-field flow prediction: given airfoil geometry and flow conditions, predict velocity (Ux, Uy) and pressure (p) at every mesh node.

Key files:
- `prepare.py` — **read-only**. Dataset loading, preprocessing, collation. Do not modify.
- `train.py` — training script. Hyperparameters, optimizer, training loop, loss formulation. **You can modify this.**
- `transolver.py` — model architecture (Transolver with physics attention). **You can modify this.**
- `utils.py` — visualization. Do not modify (not relevant to metrics).
- `DATASET_REPORT.md` — dataset documentation. Read for context.

## Constraints

**What you CAN do:**
- Modify `train.py` — hyperparameters, optimizer, loss formulation, batch size, learning rate schedule, loss weighting, normalization strategy, etc.
- Modify `transolver.py` — model architecture, attention mechanism, number of layers, hidden dimensions, activation functions, positional encoding, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Modify `utils.py`. Visualization is not part of the experiment.
- Install new packages or add dependencies beyond what's in `pyproject.toml`.

## Goal and metrics

**The goal is simple: get the lowest validation losses.** We track:
- **Surface MAE** — mean absolute error on airfoil surface nodes (Ux, Uy, p). **Most important** — these are the quantities engineers care about.
- **Volume MAE** — mean absolute error on volume (field) nodes.
- **val/loss** — the combined validation loss (vol_loss + surf_weight * surf_loss).

Lower is better for all metrics. Surface accuracy matters most.

**VRAM** is a soft constraint. The GPUs have 96GB. Some increase is acceptable for meaningful metric gains, but it should not OOM.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome.

## Training output

The script prints a summary when done:

```
======================================================================
TRAINING COMPLETE
======================================================================
Best model at epoch 48
  Val total loss: 15.2345
  Volume  MAE:  Ux=4.38  Uy=1.71  p=113.7
  Surface MAE:  Ux=1.19  Uy=0.55  p=83.2
```

Extract key metrics:
```bash
grep "Val total loss\|Volume  MAE\|Surface MAE" run.log
```

**Timeout**: Default is 10 epochs / 5 minutes. Don't change this.

**Crashes**: If a run crashes, check `tail -n 50 run.log`. If it's a typo or import error, fix and re-run. If the idea is fundamentally broken, report it and move on.

## W&B logging

All runs log to the shared W&B project. Always pass:
- `--agent <your-name>` — stored in config and as a tag for filtering
- `--wandb_name "<your-name>/<description>"` — human-readable run name
- `--wandb_group "<idea-name>"` — only to group iterations on the same idea

## Roles

Research is coordinated through GitHub PRs with an advisor/student model. See `advisor.md` and `student.md` for role-specific workflows.
