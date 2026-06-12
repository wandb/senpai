---
name: wbagent
description: "Use this skill for querying and analyzing Weights & Biases projects through the W&B SDK and the local `wandb_helpers.py` query helpers. Covers run discovery, filtered run-table queries, selected summary metrics, config comparisons, exact run counts, artifacts, sweeps, reports, and bounded history scans. This is the preferred way to query W&B, both standalone and from the wandb-driven-dev skill."
---

# W&B Query Skill

This is the standalone W&B query surface. Use it whenever a task needs live W&B
data — run metadata, summary metrics, config values, histories, artifacts,
sweeps, or Reports SDK details. The `wandb-driven-dev` skill builds on it, but
it works on its own for any W&B querying.

Do not write broad ad hoc W&B loops first. Start with
`scripts/wandb_helpers.py`, then use direct W&B SDK calls only when the helper
does not cover the query shape.

## Available Files

- `scripts/wandb_helpers.py`: primary helper module for W&B querying.
- `references/WANDB_SDK.md`: W&B SDK usage notes and query patterns.
- `references/WANDB_CONCEPTS.md`: entity/project/run/config/history concepts.
- `references/REPORTS.md`: W&B Reports SDK details.

## Standard Preamble

Use this import pattern in one-off scripts:

```python
import sys
sys.path.insert(0, ".claude/skills/wbagent/scripts")

from wandb_helpers import (
    get_api,
    probe_project,
    build_filters,
    fetch_runs,
    fetch_run_summaries,
    count_runs,
    runs_to_dataframe,
    compare_configs,
    scan_history,
    scan_history_until_step,
    compare_runs_at_step,
)
```

Create clients with `get_api()` rather than bare `wandb.Api()` unless there is
a specific reason. It sets a larger timeout for real projects.

## Query Rules

- Use server-side filters for run searches: `state`, `display_name`, `tags`,
  `config.KEY`, `summary_metrics.KEY`, `created_at`, `group`, and `job_type`.
- Use `fetch_runs()` for run tables. It uses selected GraphQL
  `summaryMetrics(keys=...)` and avoids materializing wide SDK run objects.
- Use `count_runs()` for exact counts. It uses lazy `api.runs(..., per_page=1)`
  and does not load all runs.
- Use `fetch_run_summaries()` for a small set of known run IDs when setup or
  review needs selected summary keys.
- Always pass explicit `keys=[...]` to history scans. Never call
  `run.history()` or `run.scan_history()` without keys on large projects.
- Use `scan_history_until_step()` or `compare_runs_at_step()` for at-budget
  comparisons so scans stop once the selected step key passes the target.
- Use pandas/numpy for analysis. Do not dump raw histories or long run lists
  into the response.
- For unfamiliar projects, call `probe_project(api, "entity/project")` before
  guessing metric names.

## Common Recipes

### Exact Counts

```python
api = get_api()
path = "entity/project"

total = count_runs(api, path)
finished = count_runs(api, path, {"state": "finished"})
crashed = count_runs(api, path, {"state": "crashed"})

print({"total": total, "finished": finished, "crashed": crashed})
```

### Filtered Run Table

```python
api = get_api()
rows = fetch_runs(
    api,
    "entity/project",
    metric_keys=["val/loss", "train/global_step"],
    filters={
        "state": "finished",
        "config.model.name": "baseline",
        "summary_metrics.val/loss": {"$lt": 0.2},
    },
    config_keys=["model.name", "lr", "batch_size"],
    order="+summary_metrics.val/loss",
    limit=20,
)
```

`fetch_runs()` returns flat rows with `id`, `name`, `display_name`, `state`,
`created_at`, requested metrics, and requested config values such as
`config.model.name`.

### CLI-Friendly Filters

```python
filters = build_filters(
    [
        "config.max_steps=20000",
        "summary_metrics.val/loss<0.2",
        "created_at>=2026-05-01",
    ],
    default_state="finished",
)
rows = fetch_runs(api, "entity/project", metric_keys=["val/loss"], filters=filters)
```

### Known Run Summaries

```python
rows = fetch_run_summaries(
    api,
    "entity/project",
    run_ids=["abc123", "def456"],
    summary_keys=["val/loss", "train/global_step"],
)
```

### At-Step Comparison

```python
comparison = compare_runs_at_step(
    api,
    "entity/project",
    run_ids=["abc123", "def456"],
    step=10_000,
    step_key="train/global_step",
    metrics=["train/loss", "val/loss"],
)
```

This groups metrics by namespace and retries sparse metrics one by one when a
combined scan misses rows.

### Direct History Scan

```python
run = api.run("entity/project/abc123")
rows = scan_history(
    run,
    keys=["_step", "train/global_step", "train/loss"],
    max_rows=10_000,
)
```

`scan_history()` automatically uses `beta_scan_history()` for large runs when
available.

## Output Guidance

When answering a W&B data question, report:

- the project path queried
- filters and metric/config keys used
- sample size or count method
- compact tables or aggregates
- any missing metrics or incomplete runs

For large analyses, save reusable query scripts near the experiment context
when useful, but keep final answers concise and number-backed.

## When To Read References

- Read `WANDB_SDK.md` for SDK method signatures, filters, history scans,
  artifacts, and sweeps.
- Read `WANDB_CONCEPTS.md` when entity/project/run/history/summary semantics
  matter.
- Read `REPORTS.md` for W&B Reports SDK details.

Do not read all references by default. Use only the one needed for the current
question.
