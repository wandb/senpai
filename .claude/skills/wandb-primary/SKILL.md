---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

name: wandb-primary
description: Comprehensive primary skill for agents working with Weights & Biases. Covers both the W&B SDK (training runs, metrics, artifacts, sweeps) and the Weave SDK (GenAI traces, evaluations, scorers). Includes helper libraries, gotcha tables, and data analysis patterns. Use this skill whenever the user asks about W&B runs, Weave traces, evaluations, training metrics, loss curves, model comparisons, or any Weights & Biases data — even if they don't say "W&B" explicitly.
---

# W&B Primary Skill

## Python environment detection (DO THIS FIRST)

Before running any Python code, you MUST determine how the user runs Python in this project. Check for clues in this order:

1. Look for `pyproject.toml`, `uv.lock`, `poetry.lock`, `Pipfile`, `requirements.txt`, `setup.py`, `setup.cfg`, `Makefile`, or `docker-compose.yml` in the project root
2. Check if there is an activated virtual environment (`$VIRTUAL_ENV`, `.venv/`, `venv/`)
3. Check for a `Dockerfile` or CI config that reveals the runtime
4. If the user has explicitly told you how they run Python, use that

Once you know the environment, **write your finding into this skill** by replacing the placeholder below so that all subsequent code blocks use the correct commands. If you cannot determine the environment from any of the above, default to `uv`.

<!-- AGENT: Replace the content between the ENVIRONMENT markers with the detected environment -->
<!-- ENVIRONMENT_START -->
**Detected Python environment:** _not yet detected_

```
# Run command: <not yet detected>
# Install command: <not yet detected>
```
<!-- ENVIRONMENT_END -->

**Examples of what to write here:**

| Environment | Run command | Install command |
|---|---|---|
| uv | `uv run script.py` | `uv pip install pandas` |
| poetry | `poetry run python script.py` | `poetry add pandas` |
| conda | `conda run python script.py` | `conda install pandas` |
| bare venv | `python script.py` (with venv activated) | `pip install pandas` |
| docker | `docker exec <ctr> python script.py` | `docker exec <ctr> pip install pandas` |

**If you cannot determine the environment, write this:**

```
# Run command: uv run script.py        # always use uv run, never bare python
# Install command: uv pip install <pkg>
```

---

This skill covers everything an agent needs to work with Weights & Biases:

- **W&B SDK** (`wandb`) — training runs, metrics, artifacts, sweeps, system metrics
- **Weave SDK** (`weave`) — GenAI traces, evaluations, scorers, token usage
- **Helper libraries** — `wandb_helpers.py` and `weave_helpers.py` for common operations

## When to use what

| I need to... | Use |
|---|---|
| Query training runs, loss curves, hyperparameters | **W&B SDK** (`wandb.Api()`) — see `references/WANDB_SDK.md` |
| Query GenAI traces, calls, evaluations | **Weave SDK** (`weave.init()`, `client.get_calls()`) — see `references/WEAVE_SDK.md` |
| Convert Weave wrapper types to plain Python | **`weave_helpers.unwrap()`** |
| Build a DataFrame from training runs | **`wandb_helpers.runs_to_dataframe()`** |
| Extract eval results for analysis | **`weave_helpers.eval_results_to_dicts()`** |
| Need low-level Weave filtering (CallsFilter, Query) | **Raw Weave SDK** (`weave.init()`, `client.get_calls()`) — see `references/WEAVE_SDK.md` |
| Judge curve shape (spikes, smoothness, slope, overfit) | **`training_diagnostics` + `curve_plots`** — see `references/TRAINING_DIAGNOSTICS.md` and the "Training curve analysis workflow" section below |

---

## Bundled files

### Helper libraries

```python
import sys
sys.path.insert(0, "skills/wandb-primary/scripts")

# Weave helpers (traces, evals, GenAI)
from weave_helpers import (
    unwrap,                  # Recursively convert Weave types -> plain Python
    get_token_usage,         # Extract token counts from a call's summary
    eval_results_to_dicts,   # predict_and_score calls -> list of result dicts
    pivot_solve_rate,        # Build task-level pivot table across agents
    results_summary,         # Print compact eval summary
    eval_health,             # Extract status/counts from Evaluation.evaluate calls
    eval_efficiency,         # Compute tokens-per-success across eval calls
)

# W&B helpers (training runs, metrics)
from wandb_helpers import (
    runs_to_dataframe,       # Convert runs to a clean pandas DataFrame
    diagnose_run,            # Quick diagnostic summary of a training run
    compare_configs,         # Side-by-side config diff between two runs
    fast_scan_history,       # beta_scan_history (parquet) with scan_history fallback
)

# X-axis (step metric) detection — ALWAYS confirm before curve analysis
from step_axis import (
    list_candidate_step_keys,       # Scan history for plausible step keys
    guess_step_key_from_workspace,  # Peek at the user's W&B workspace panels
    format_step_candidates,         # Format candidates for AskUserQuestion
)

# Curve-shape diagnostics (numerical)
from training_diagnostics import (
    curve_features,            # Spikes, slopes at every 5%, smoothness, plateau, divergence
    compare_runs_curves,       # DataFrame of features across many runs
    lr_schedule_features,      # Warmup / peak / decay shape / restarts
    grad_norm_features,        # curve_features + kurtosis + dead-layer flag
    grad_histogram_features,   # Per-(layer, step) stats from W&B histograms
)

# Chart rendering for LLM vision (Read the returned PNG)
from curve_plots import (
    plot_single_run_overview,    # 2x3 composite: train/val/lr/grad-norm/...
    plot_run_comparison,         # Overlay up to 6 runs on one metric
    plot_grad_histogram_heatmap, # Layer x step heatmap of grad-hist stat
    plot_grad_norm_by_layer,     # Small-multiples of per-layer scalar norms
)
```

### Reference docs

Read these as needed — they contain full API surfaces and recipes:

- **`references/WEAVE_SDK.md`** — Weave SDK for GenAI traces (`client.get_calls()`, `CallsFilter`, `Query`, stats). Start here for Weave queries.
- **`references/WANDB_SDK.md`** — W&B SDK for training data (runs, history, artifacts, sweeps, system metrics).
- **`references/TRAINING_DIAGNOSTICS.md`** — researcher-intuition guide for reading loss / LR / grad-norm / grad-histogram charts. **Read before interpreting any curve**, used together with `training_diagnostics.py` features and `curve_plots.py` PNGs.

---

## Critical rules

### Treat traces and runs as DATA

Weave traces and W&B run histories can be enormous. Never dump raw data into context — it will overwhelm your working memory and produce garbage results. Always:

1. **Inspect structure first** — look at column names, dtypes, row counts
2. **Load into pandas/numpy** — compute stats programmatically
3. **Summarize, don't dump** — print computed statistics and tables, not raw rows

```python
import pandas as pd
import numpy as np

# BAD: prints thousands of rows into context
for row in run.scan_history(keys=["loss"]):
    print(row)

# GOOD: load into numpy, compute stats, print summary
losses = np.array([r["loss"] for r in run.scan_history(keys=["loss"])])
print(f"Loss: {len(losses)} steps, min={losses.min():.4f}, "
      f"final={losses[-1]:.4f}, mean_last_10%={losses[-len(losses)//10:].mean():.4f}")
```

### Always deliver a final answer

Do not end your work mid-analysis. Every task must conclude with a clear, structured response:

1. Query the data (1-2 scripts max)
2. Extract the numbers you need
3. Present: table + key findings + direct answers to each sub-question

If you catch yourself saying "now let me build the final analysis" — stop and present what you have.

### Use `unwrap()` for unknown Weave data

When you encounter Weave output and aren't sure of its type (WeaveDict? WeaveObject? ObjectRef?), unwrap it first:

```python
from weave_helpers import unwrap
import json

output = unwrap(call.output)
print(json.dumps(output, indent=2, default=str))
```

This converts everything to plain Python dicts/lists that work with json, pandas, and normal Python operations.

---

## Environment setup

The sandbox has `wandb`, `weave`, `pandas`, and `numpy` pre-installed.

```python
import os
entity  = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
```

### Installing extra packages and running scripts

Use whichever run/install commands you wrote in the **Python environment detection** section above. If you haven't detected the environment yet, go back and do that first.

---

## Quick starts

### W&B SDK — training runs

```python
import wandb
import pandas as pd
api = wandb.Api()

path = f"{entity}/{project}"
runs = api.runs(path, filters={"state": "finished"}, order="-created_at")

# Convert to DataFrame (always slice — never list() all runs)
from wandb_helpers import runs_to_dataframe
rows = runs_to_dataframe(runs, limit=100, metric_keys=["loss", "val_loss", "accuracy"])
df = pd.DataFrame(rows)
print(df.describe())
```

For full W&B SDK reference (filters, history, artifacts, sweeps), read `references/WANDB_SDK.md`.

### Weave — SDK

```python
import weave
client = weave.init(f"{entity}/{project}")  # positional string, NOT keyword arg
calls = client.get_calls(limit=10)
```

For raw SDK patterns (CallsFilter, Query, advanced filtering), read `references/WEAVE_SDK.md`.

---

## Key patterns

### Weave eval inspection

Evaluation calls follow this hierarchy:

```
Evaluation.evaluate (root)
  ├── Evaluation.predict_and_score (one per dataset row x trials)
  │     ├── model.predict (the actual model call)
  │     ├── scorer_1.score
  │     └── scorer_2.score
  └── Evaluation.summarize
```

Extract per-task results into a DataFrame:

```python
from weave_helpers import eval_results_to_dicts, results_summary

# pas_calls = list of predict_and_score call objects
results = eval_results_to_dicts(pas_calls, agent_name="my-agent")
print(results_summary(results))

df = pd.DataFrame(results)
print(df.groupby("passed")["score"].mean())
```

### Eval health and efficiency

```python
from weave_helpers import eval_health, eval_efficiency

health = eval_health(eval_calls)
df = pd.DataFrame(health)
print(df.to_string(index=False))

efficiency = eval_efficiency(eval_calls)
print(pd.DataFrame(efficiency).to_string(index=False))
```

### Token usage

```python
from weave_helpers import get_token_usage

usage = get_token_usage(call)
print(f"Tokens: {usage['total_tokens']} (in={usage['input_tokens']}, out={usage['output_tokens']})")
```

### Cost estimation

```python
call_with_costs = client.get_call("id", include_costs=True)
costs = call_with_costs.summary.get("weave", {}).get("costs", {})
```

### Run diagnostics

```python
from wandb_helpers import diagnose_run

run = api.run(f"{path}/run-id")
diag = diagnose_run(run)
for k, v in diag.items():
    print(f"  {k}: {v}")
```

### Error analysis — open coding to axial coding

For structured failure analysis on eval results:

1. **Understand data shape** — use `project.summary()`, `calls.input_shape()`, `calls.output_shape()`
2. **Open coding** — write a Weave Scorer that journals what went wrong per failing call
3. **Axial coding** — write a second Scorer that classifies notes into a taxonomy
4. **Summarize** — count primary labels with `collections.Counter`

See `references/WEAVE_SDK.md` for the full SDK reference.

### W&B Reports

Install `wandb[workspaces]` using the install command from the **Python environment detection** section.

```python
from wandb.apis import reports as wr
import wandb_workspaces.expr as expr

report = wr.Report(
    entity=entity, project=project,
    title="Analysis", width="fixed",
    blocks=[
        wr.H1(text="Results"),
        wr.PanelGrid(
            runsets=[wr.Runset(entity=entity, project=project)],
            panels=[wr.LinePlot(title="Loss", x="_step", y=["loss"])],
        ),
    ],
)
# report.save(draft=True)  # only when asked to publish
```

Use `expr.Config("lr")`, `expr.Summary("loss")`, `expr.Tags().isin([...])` for runset filters — not dot-path strings.

---

## Training curve analysis workflow

This is the workflow for judging how a training run actually went — the "look at the charts" step that an experienced researcher does. Use it any time the user asks whether a run is healthy, what went wrong, which of N runs is best on more than just final metrics, or asks for an eval of a student's experiment.

Pair three sources of truth: (a) numerical features from `training_diagnostics.py`, (b) PNGs from `curve_plots.py` read back via the Read tool, (c) heuristics from `references/TRAINING_DIAGNOSTICS.md`. Each catches what the others miss.

### Step 1 — Confirm the x-axis step metric (MANDATORY, every session)

Never assume `_step`. Different training stacks log different step keys (`global_step`, `trainer/global_step`, `epoch`, `train/step`, or custom). A wrong axis silently misaligns every single chart and feature.

```python
from step_axis import (
    list_candidate_step_keys,
    guess_step_key_from_workspace,
    format_step_candidates,
)

candidates = list_candidate_step_keys(run)
workspace_guess = guess_step_key_from_workspace(entity, project)
options = format_step_candidates(candidates, workspace_guess)
# `options` is [(label, description), ...] — hand it to AskUserQuestion.
```

Then **ask the user via `AskUserQuestion`**, showing the options with the workspace guess (if any) marked `(Recommended)`. Use the confirmed value as `step_key` in all subsequent calls. Shortcut: if there is exactly one candidate and it matches the workspace guess, you may proceed without asking — but say which one you picked.

### Step 2 — Read the intuition guide

Read `references/TRAINING_DIAGNOSTICS.md` once at the start of the task. It encodes the researcher heuristics you will apply in Step 5.

### Step 3 — Extract numeric features

```python
import pandas as pd
from training_diagnostics import (
    curve_features, lr_schedule_features, grad_norm_features, grad_histogram_features,
)

history = list(fast_scan_history(run, keys=[step_key, "loss", "val_loss", "lr", "grad_norm"]))
df = pd.DataFrame(history)

loss_feats   = curve_features(df["loss"].dropna(), df.loc[df["loss"].notna(), step_key],
                              direction="decreasing")
val_feats    = curve_features(df["val_loss"].dropna(), df.loc[df["val_loss"].notna(), step_key],
                              direction="decreasing")
lr_feats     = lr_schedule_features(df["lr"].dropna(), df.loc[df["lr"].notna(), step_key])
gnorm_feats  = grad_norm_features(df["grad_norm"].dropna(),
                                  df.loc[df["grad_norm"].notna(), step_key])

# If histograms logged (wandb.watch with log="gradients"):
hist_df = grad_histogram_features(run, layer_prefix="gradients/", step_key=step_key)
```

Print a compact table of `spike_count`, `smoothness`, `final_10pct_mean`, `monotonicity_pct`, `divergence.detected`, `lr_feats.decay_shape`, `gnorm_feats.kurtosis`, `gnorm_feats.dead_flag`. Don't print the raw spike/slope lists unless you're drilling in.

### Step 4 — Render plots and view them with the Read tool

```python
from curve_plots import plot_single_run_overview, plot_grad_histogram_heatmap

overview = plot_single_run_overview(run, step_key=step_key)
print(f"overview PNG: {overview}")

# Only if grad histograms are logged:
heatmap = plot_grad_histogram_heatmap(run, step_key=step_key)
print(f"grad-hist heatmap: {heatmap}")
```

Then **use the Read tool on the printed paths** so Claude views them as images. This is the "see the curves" step — without it you're flying blind. The PNGs default to `/tmp/wandb_plots/<run-id>/`.

### Step 5 — Synthesize

Combine:
- What the numbers say (final, smoothness, spikes, slopes at each 5% checkpoint, plateaus, divergence).
- What the image looks like (use the patterns from `TRAINING_DIAGNOSTICS.md` — healthy-shape, instability, overfit, plateau vs convergence).
- Where the three sources disagree (often the interesting part).

Produce a short verdict in this shape:

```
Verdict: <healthy | unstable | overfit | plateaued | diverged | converged>
Evidence:
  - <specific step range> — <what the curves show>
  - <specific step range> — ...
Next actions:
  - <concrete hyperparameter / logging / code change>
```

### Multi-run comparison variant

```python
from training_diagnostics import compare_runs_curves
from curve_plots import plot_run_comparison

feats_df = compare_runs_curves(runs, metric="val_loss", step_key=step_key)
print(feats_df.sort_values("final_10pct_mean").to_string())
if feats_df.attrs.get("warning"):
    print(feats_df.attrs["warning"])

# Cap 6 runs per overlay; above that, pick top-k first.
best_name = feats_df["final_10pct_mean"].idxmin()
png = plot_run_comparison(runs[:6], metric="val_loss", step_key=step_key,
                          highlight=best_name)
# Read the PNG, then synthesize using TRAINING_DIAGNOSTICS §8 (comparing runs).
```

### Per-layer gradient view

```python
from curve_plots import plot_grad_norm_by_layer, plot_grad_histogram_heatmap

# Scalar per-layer grad norms (if logged as scalars):
png1 = plot_grad_norm_by_layer(run, step_key=step_key, layer_prefix="parameters/")

# Per-layer histogram stat (if logged as histograms via wandb.watch):
png2 = plot_grad_histogram_heatmap(run, step_key=step_key, metric="mean_abs")
```

Read the PNGs and apply `TRAINING_DIAGNOSTICS.md` §7 (reading heatmaps): scan for banding, collapsed rows (dead layers), widening tails (instability), dark columns (no-op steps).

---

## Gotchas

### Weave API

| Gotcha | Wrong | Right |
|--------|-------|-------|
| weave.init args | `weave.init(project="x")` | `weave.init("x")` (positional) |
| Parent filter | `filter={'parent_id': 'x'}` | `filter={'parent_ids': ['x']}` (plural, list) |
| WeaveObject access | `rubric.get('passed')` | `getattr(rubric, 'passed', None)` |
| Nested output | `out.get('succeeded')` | `out.get('output').get('succeeded')` (output.output) |
| ObjectRef comparison | `name_ref == "foo"` | `str(name_ref) == "foo"` |
| CallsFilter import | `from weave import CallsFilter` | `from weave.trace.weave_client import CallsFilter` |
| Query import | `from weave import Query` | `from weave.trace_server.interface.query import Query` |
| Eval status path | `summary["status"]` | `summary["weave"]["status"]` |
| Eval success count | `summary["success_count"]` | `summary["weave"]["status_counts"]["success"]` |
| When in doubt | Guess the type | `unwrap()` first, then inspect |

### WeaveDict vs WeaveObject

- **WeaveDict**: dict-like, supports `.get()`, `.keys()`, `[]`. Used for: `call.inputs`, `call.output`, `scores` dict
- **WeaveObject**: attribute-based, use `getattr()`. Used for: scorer results (rubric), dataset rows
- **When in doubt**: use `unwrap()` to convert everything to plain Python

### W&B API

| Gotcha | Wrong | Right |
|--------|-------|-------|
| Summary access | `run.summary["loss"]` | `run.summary_metrics.get("loss")` |
| Loading all runs | `list(api.runs(...))` | `runs[:200]` (always slice) |
| History — all fields | `run.history()` | `run.history(samples=500, keys=["loss"])` |
| scan_history — no keys | `scan_history()` | `scan_history(keys=["loss"])` (explicit) |
| Raw data in context | `print(run.history())` | Load into DataFrame, compute stats |
| Metric at step N | iterate entire history | `scan_history(keys=["loss"], min_step=N, max_step=N+1)` |
| Cache staleness | reading live run | `api.flush()` first |

### Package management

| Gotcha | Details |
|--------|---------|
| Using the wrong runner | Always use the run/install commands from the **Python environment detection** section — never guess |
| Bare `python` when env unknown | If you haven't detected the environment yet, default to `uv run script.py` (never bare `python`) |

### Weave logging noise

Weave prints version warnings to stderr. Suppress with:

```python
import logging
logging.getLogger("weave").setLevel(logging.ERROR)
```

---

## Quick reference

```python
# --- Weave: Init and get calls ---
import weave
client = weave.init(f"{entity}/{project}")
calls = client.get_calls(limit=10)

# --- W&B: Best run by loss ---
best = api.runs(path, filters={"state": "finished"}, order="+summary_metrics.loss")[:1]
print(f"Best: {best[0].name}, loss={best[0].summary_metrics.get('loss')}")

# --- W&B: Loss curve to numpy ---
losses = np.array([r["loss"] for r in run.scan_history(keys=["loss"])])
print(f"min={losses.min():.6f}, final={losses[-1]:.6f}, steps={len(losses)}")

# --- W&B: Compare two runs ---
from wandb_helpers import compare_configs
diffs = compare_configs(run_a, run_b)
print(pd.DataFrame(diffs).to_string(index=False))
```
