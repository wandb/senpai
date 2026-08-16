---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

name: plot-experiment-charts
description: >
  Operator-side guide for generating a target-specific training curve chart
  for a legacy experiment PR. Use when auditing or presenting historical runs.
---

# Plot Experiment Charts

This guide is not installed into live advisor or student runtimes. Target
repositories that require experiment charts should provide their own project
skill with the correct metrics and plotting code.

Generate a comparison chart so a reviewer can see the training dynamics at a
glance—not just the final numbers, but how the experiment got there. A bolded
best-run line and a properly scaled y-axis make the story immediately readable,
even if some runs diverged.

This skill takes about 30 seconds. It's worth it.

## What you need

- **Baseline W&B run ID**: in the PR body under `## Baseline`, look for the `W&B run: \`xxxxxxxx\`` line.
- **Your own run IDs**: the 8-character W&B IDs of the runs you just completed. Find them in the W&B run URLs or in the training output (the run ID is printed at launch).
- **W&B credentials**: `WANDB_ENTITY` and `WANDB_PROJECT` in the current environment.

## Step 1 — Run the script

```bash
uv run .agents/skills/plot-experiment-charts/scripts/plot_training_curves.py \
  --baseline <baseline-run-id> \
  --runs <your-run-id-1>,<your-run-id-2>,...
```

The script automatically:
- Downloads the full training history for each run using W&B's `beta_scan_history` (fast, batched)
- Detects which of your runs achieved the lowest final `val_in_dist/mae_surf_p` and **bolds it**
- Clamps the y-axis so the baseline curve stays readable even if some runs diverged
- Saves `training_curves.png` in the current directory
- Prints the GitHub raw URL to embed

If you want to override which run gets bolded, pass `--bold <run-id>`.

**Full flag reference:**

| Flag | Default | Notes |
|---|---|---|
| `--baseline` | required | 8-char W&B run ID of the current best baseline |
| `--runs` | required | Comma-separated run IDs for your experiments (1–8) |
| `--bold` | auto | Override which run gets the bold treatment |
| `--output` | `training_curves.png` | Output filename |
| `--entity` | `$WANDB_ENTITY` | W&B entity |
| `--project` | `$WANDB_PROJECT` | W&B project |

## Step 2 — Commit the chart alongside train.py

```bash
git add train.py training_curves.png
git commit -m "<your experiment description>"
git push origin <branch>
```

The chart lives on the experiment branch. It's visible during review — which is the only time it matters. After the PR is squash-merged and the branch deleted, the image in the archived PR body will show as broken, but by then the advisor has already reviewed it.

## Step 3 — Embed in the PR body

The script prints a raw GitHub URL when it finishes. Copy it and add this to the `## Results` section of the PR body:

```markdown
## Results

![Training curves](https://raw.githubusercontent.com/owner/repo/branch/training_curves.png)

| Metric | Baseline | This run |
| ... |
```

Put the chart *before* the metrics table so the advisor sees the curves first, then the numbers.

## Reading the chart

Two panels, side by side:

- **Left**: `val_in_dist/mae_surf_p` — surface pressure MAE on in-distribution data. This is the primary metric. Lower is better.
- **Right**: `val/loss` — combined validation loss across all splits. Lower is better.

The **black dashed line** is the baseline. Your runs are colored lines. The **bold colored line** is your best run — the one that would be a candidate for merging.

The y-axis is clamped at 3× the baseline's best value. If a run diverges far above that, it's clipped — that's fine, the advisor just needs to see "this diverged" rather than the exact value. The important region (near the baseline) stays readable.

## If a run crashed or never logged metrics

The script skips runs with no history and prints a warning. Include the missing run IDs in your PR results section with a note about why they crashed — don't silently omit them.

## Troubleshooting

**"Run not found"** — Double-check the run ID (8 alphanumeric chars, case-sensitive). The run ID is in the W&B run URL: `wandb.ai/{entity}/{project}/runs/{run-id}`.

**"No data for key val_in_dist/mae_surf_p"** — The run may have crashed before the first validation epoch. Check the run logs. Still include it in the chart call — the script handles empty runs gracefully.

**Script not found** — Run from the repo root: `uv run .agents/skills/plot-experiment-charts/scripts/plot_training_curves.py ...`. If the helper script is not present in this repo checkout, generate the comparison chart manually from W&B history instead of blocking on the helper.
