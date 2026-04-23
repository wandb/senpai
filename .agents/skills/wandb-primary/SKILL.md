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
**Detected Python environment:** `uv` with a local `.venv`

```
# Run command: uv run script.py
# Install command: uv pip install <pkg>
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
| Judge curve shape (spikes, smoothness, slope, overfit) | **`training_diagnostics` + `curve_plots`** — use the workflow below, then load `references/TRAINING_DIAGNOSTICS.md` for the heuristics |

---

## Bundled files

### Helper libraries

```python
import sys
sys.path.insert(0, ".agents/skills/wandb-primary/scripts")

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
    format_step_candidates,         # Format candidate choices for user confirmation
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

### One-shot curve diagnostics CLI

Use `scripts/curve_diagnostics_cli.py` when you want the default training-curve workflow without hand-writing a temporary script. It confirms the `step_key` (or refuses if ambiguous), prints helper-derived tables, and writes the standard PNGs.

Run it with the command runner from the environment section above.

```bash
# Replace `python` with the runner from the environment section above if needed.
python .agents/skills/wandb-primary/scripts/curve_diagnostics_cli.py \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --run <run_id>

python .agents/skills/wandb-primary/scripts/curve_diagnostics_cli.py \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --run <run_id_a> \
  --run <run_id_b> \
  --metric <metric_key>
```

### Reference docs

Read these as needed — they contain full API surfaces and recipes:

- **`references/WEAVE_SDK.md`** — Weave SDK for GenAI traces (`client.get_calls()`, `CallsFilter`, `Query`, stats). Start here for Weave queries.
- **`references/WANDB_SDK.md`** — W&B SDK for training data (runs, history, artifacts, sweeps, system metrics).
- **`references/TRAINING_DIAGNOSTICS.md`** — reference heuristics for reading loss / LR / grad-norm / grad-histogram charts. Load this when you are actively interpreting training curves.

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

The sandbox often has `wandb`, `weave`, `numpy`, and plotting/data packages available, but you should verify imports in the detected project environment before relying on them. In this repo's current `uv` environment, `pandas` was missing during audit-time verification even though the diagnostics helpers import it.

```python
import os
import importlib

entity  = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

for module_name in ["wandb", "weave", "numpy", "matplotlib", "pandas"]:
    importlib.import_module(module_name)
```

### Installing extra packages and running scripts

Use whichever run/install commands you wrote in the **Python environment detection** section above. If you haven't detected the environment yet, go back and do that first.

If one of the imports above fails, install the missing package before using `training_diagnostics`, `curve_plots`, or the CLI wrapper. The diagnostics stack imports `pandas` and `matplotlib` at module import time, so a missing dependency will fail fast.

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

Use this when the user asks whether a run is healthy, why training diverged, whether a run overfit, or which run has the best training dynamics.

This workflow is the default for curve-health questions. A raw W&B query plus hand-written curve narration is not enough unless the user explicitly asked for a quick scalar-only check.

Keep the inline workflow short and load detail on demand:

1. Confirm `step_key` before doing any curve work. Never assume `_step`.
2. Compute features with the bundled helpers instead of hand-rolling spike or slope logic.
3. Render PNGs and inspect them visually.
4. Load `references/TRAINING_DIAGNOSTICS.md` while you interpret the results.
5. End with a verdict, evidence tied to step ranges, and concrete next actions.

### Stop signs

If you catch yourself writing any of these phrases before running the helpers, stop and switch to the workflow or the CLI:

- "trough envelope plateaued/regressed"
- "gradient storm"
- "restart disruption"
- "spikes at cosine peaks"
- "still descending at cutoff"
- "best checkpoint not final"

Those are exactly the cases the diagnostics helpers are meant to standardize.

### Completion gate

A curve-analysis answer is incomplete until it includes all of:

1. The chosen `step_key` and why it was chosen.
2. At least one helper-derived table (`curve_features`, `compare_runs_curves`, `lr_schedule_features`, or `grad_norm_features`).
3. At least one rendered PNG (`plot_single_run_overview`, `plot_run_comparison`, `plot_grad_histogram_heatmap`, or `plot_grad_norm_by_layer`) unless the run truly lacks the required metrics.
4. A verdict tied to specific steps, step ranges, or checkpoints.

If you skip one of these, say why.

### Required sequence

Use `list_candidate_step_keys()`, `guess_step_key_from_workspace()`, and `format_step_candidates()` to confirm the x-axis. If there is one obvious candidate and it matches the workspace guess, say which `step_key` you picked. Otherwise, ask the user to choose before plotting or comparing runs.

For a single run, compute a compact feature table from the metrics that actually exist, then render `plot_single_run_overview(run, step_key=step_key)`. If gradient histograms or per-layer scalar norms are logged, add `plot_grad_histogram_heatmap()` or `plot_grad_norm_by_layer()`.

For multi-run comparisons, use `compare_runs_curves()` for the ranking table and `plot_run_comparison()` for the overlay. Keep overlays to at most 6 runs; if there are more, rank first and then plot the shortlist.

### Phrase-to-helper mapping

| If you're about to say... | Compute / inspect first |
|---|---|
| "trough envelope plateaued/regressed" | `curve_features(...)[["final_10pct_mean", "final_10pct_std", "smoothness", "checkpoint_slopes"]]` |
| "still descending at cutoff" | `curve_features` tail slope / final segment + `compare_runs_curves` ranking table |
| "gradient storm" or "spikes at cosine peaks" | `grad_norm_features`, `lr_schedule_features`, and the LR + grad-norm panels in `plot_single_run_overview` |
| "restart disruption" | `lr_schedule_features()["restart_steps"]` plus the comparison plot around the restart window |
| "best checkpoint not final" or "late overfit" | `curve_features`, then the overfitting section of `references/TRAINING_DIAGNOSTICS.md` |

### Fastest default path

If the user wants the standard curve-analysis loop and you already know the run IDs, use the CLI above instead of building a throwaway script. It exists to lower the activation energy from "I know the heuristics" to "I actually ran the helpers".

### Output shape

Do not dump raw history rows or the full spike/slope payloads unless you are drilling into a specific anomaly. Summaries should stay compact.

Use this response shape:

```text
Verdict: <healthy | unstable | overfit | plateaued | diverged | converged>
Evidence:
- <specific step range> — <what the metrics and plot show>
- <specific step range> — <what changed and why it matters>
Next actions:
- <concrete hyperparameter, logging, or code change>
```

Load `references/TRAINING_DIAGNOSTICS.md` for the interpretation heuristics, especially when the numbers and the image disagree.

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
