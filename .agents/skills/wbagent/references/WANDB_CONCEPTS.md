# W&B Concepts & Nomenclature

Reference for W&B Models / experiment tracking concepts used by this plugin.
Use it when interpreting user requests, resolving run identifiers, or deciding
where a metric/config/artifact lives.

---

## Scope

This skill queries W&B Models data with the `wandb` Python SDK and
`wandb.Api()`: runs, projects, configs, histories, summaries, artifacts,
sweeps, registries, and reports. If a project uses an external command to start
training, `wandb-driven-dev` treats that as a user-provided launcher and this
skill queries the resulting W&B runs.

---

## Core Hierarchy

```
Entity (username or team)
  +-- Project (groups related training/evaluation runs)
      +-- Runs
          +-- Config         (input hyperparameters)
          +-- History        (time-series metrics)
          +-- Summary        (final/best metric values)
          +-- Artifacts      (versioned datasets, checkpoints, models, files)
          +-- System Metrics (GPU, CPU, memory)

Organization
  +-- Registry
      +-- Collections
          +-- Linked Artifact Versions
```

- **Entity**: A username or team namespace.
- **Project**: A group of related runs under an entity.
- **Run**: One execution of training/evaluation code. Created by
  `wandb.init()` in user code, queried later with `wandb.Api()`.
- **Run path**: `<entity>/<project>/<run_id>`, the canonical Public API path.
- **Registry**: Organization-level curated artifact collections above projects.

---

## Run Names And IDs

When a user says "find run X", determine which identifier they mean:

- **`run.id`**: Unique short hash. Access directly with
  `api.run("entity/project/run_id")`.
- **`run.name` / display name**: Human-friendly and not guaranteed unique.
  Search with `filters={"display_name": "X"}` or a regex filter.

In many SDK objects, `run.name` is the short ID and `run.display_name` is the
UI name. In GraphQL run rows, this plugin returns both `name` and
`display_name` from `fetch_runs()`.

---

## Config, History, Summary

| What | Set by | When | Query with |
| --- | --- | --- | --- |
| **Config** | `wandb.init(config=...)` or `wandb.config.update()` | At run start | `run.config`, `config.KEY` filters |
| **History** | `run.log({...})` | Repeatedly during a run | `run.history(keys=...)`, `run.scan_history(keys=...)` |
| **Summary** | Last logged value, manual summary writes, or `define_metric(summary=...)` | End/current run state | `run.summary_metrics`, `summary_metrics.KEY` filters |

Rules:

- Config values are experiment inputs. They should not change during training.
- History values are time series. Always pass `keys=[...]` when reading them.
- Summary values are the dashboard/run-table values. Check summaries first for
  final metrics, then scan history only if the needed aggregate is absent or
  summary semantics are wrong.
- Avoid dots in raw config keys because W&B uses dotted paths for nested keys.

Common mistake: "best loss" might mean a summary value if the project used
`define_metric("loss", summary="min")`; otherwise compute the minimum from
history.

---

## Public API vs Training SDK

| | Training SDK | Public API |
| --- | --- | --- |
| Import | `import wandb` | `wandb.Api()` |
| Purpose | Log a live run | Query existing runs and artifacts |
| Creates runs? | Yes, via `wandb.init()` | No |
| Typical use here | User training code only | Agent analysis and reports |

As an agent, prefer the Public API. Do not create fake runs with `wandb.init()`
to answer analysis questions.

---

## Filtering Runs

Use server-side filters instead of loading all runs:

```python
api = wandb.Api(timeout=120)
path = "entity/project"

runs = api.runs(path, filters={"state": "finished"})
runs = api.runs(path, filters={"display_name": {"$regex": ".*baseline.*"}})
runs = api.runs(path, filters={"config.lr": {"$lt": 1e-3}})
runs = api.runs(path, filters={"summary_metrics.val/loss": {"$lt": 0.2}})
runs = api.runs(path, filters={"tags": {"$in": ["exp/my-slug"]}})
```

Common filter keys:

`state`, `display_name`, `name`, `config.KEY`, `summary_metrics.KEY`, `tags`,
`created_at`, `group`, `job_type`, `sweep`.

Use `order="+summary_metrics.loss"` for lowest-first metric sorting and
`order="-summary_metrics.accuracy"` for highest-first sorting.

---

## Artifacts

Artifacts are versioned data objects logged by runs.

- Types include `dataset`, `model`, `checkpoint`, `code`, and custom strings.
- Versions are `v0`, `v1`, ...
- Aliases are mutable labels such as `latest`, `best`, or `production`.
- Reference format: `entity/project/artifact_name:version_or_alias`.

```python
api = wandb.Api(timeout=120)

artifact = api.artifact("entity/project/my-dataset:latest")
artifact = api.artifact("entity/project/model-weights:v3")

local_path = artifact.download()
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()

for art in run.logged_artifacts():
    print(art.name, art.type, art.aliases)
```

Registry-linked artifacts use paths like
`wandb-registry-Models/collection-name:latest`.

---

## Sweeps

Sweeps are W&B-managed hyperparameter searches. Sweep runs are still normal
runs in a project.

```python
api = wandb.Api(timeout=120)
project = api.project("project", entity="entity")
sweeps = list(project.sweeps(per_page=50))

runs = api.runs("entity/project", filters={"sweep": "sweep_id"})
```

Use `include_sweeps=False` for ordinary run-table counts unless the user asks
for sweep runs explicitly.

---

## Tables

W&B Tables are interactive tabular data logged by runs. They can include rich
media. Query table artifacts through the run/artifact APIs rather than dumping
large table contents into context.

```python
table = wandb.Table(columns=["input", "prediction", "ground_truth"])
table.add_data("hello", "world", "world")
run.log({"predictions": table})
```

---

## Runtime Freshness

`wandb.Api()` caches network requests. For running jobs or watcher loops, call
`api.flush()` when you need fresh data. Finished runs rarely require cache
flushes.

---

## Logging And Query Limits

These affect how agents should design logging and analysis:

| Resource | Limit |
| --- | ---: |
| Distinct metrics per project | 10,000 |
| Single `run.log()` payload | 25 MB |
| Config size | 10 MB |
| Scalar data points per metric | 100,000 |
| Media data points | 50,000 |
| Histogram data points | 10,000 |
| Runs per project (SaaS) | 100,000 |
| Table rows per single log call | 10,000 |
| Config key nesting | max 3 dots |
| Summary key nesting | max 4 dots |

Large projects require selected queries: selected summary keys, server-side
filters, and explicit history keys.

---

## `define_metric`

`define_metric()` controls custom x-axes and summary aggregation:

```python
run.define_metric("val/loss", step_metric="epoch")
run.define_metric("train/*", step_metric="train/global_step")
run.define_metric("val/loss", summary="min")
run.define_metric("val/acc", summary="max")
```

Without `define_metric`, metrics use W&B `_step` as the x-axis and summary
stores the last logged value.

---

## Distributed Training

Common W&B logging patterns:

| Pattern | When | How |
| --- | --- | --- |
| Rank-0 only | Most distributed training | Only rank 0 calls `wandb.init()` |
| Process-per-run | Need per-GPU metrics | Each process logs a separate grouped run |
| Shared mode | Unified multi-process run, SDK 0.19.9+ | `wandb.init(mode="shared")` |

Always call `run.finish()` to avoid hanging runs.

---

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `WANDB_API_KEY` | Authentication |
| `WANDB_ENTITY` | Default entity |
| `WANDB_PROJECT` | Default project |
| `WANDB_BASE_URL` | Self-hosted/server URL |
| `WANDB_MODE` | `online`, `offline`, `disabled` |
| `WANDB_DIR` | Local W&B storage |
| `WANDB_SILENT` / `WANDB_QUIET` | Suppress output |
| `WANDB_CONFIG_PATHS` | YAML configs loaded into `wandb.config` |
| `WANDB_IGNORE_GLOBS` | Files excluded from code saving |
| `WANDB_DISABLE_GIT` | Skip git metadata |
| `WANDB_DISABLE_CODE` | Skip code saving |

---

## Quick Disambiguation

| User says... | They probably mean... | Query |
| --- | --- | --- |
| "my runs" | Training/evaluation runs | `api.runs(path)` |
| "my experiments" | Runs or groups of runs | `api.runs(path)` with filters/tags |
| "run X" short hash | Run ID | `api.run("entity/project/X")` |
| "run X" phrase/name | Display name | `api.runs(path, filters={"display_name": "X"})` |
| "this run's config" | Hyperparameters | `run.config` |
| "this run's metrics" | Summary or history metrics | `run.summary_metrics`, `scan_history()` |
| "best loss" | Summary min or history minimum | Check summary semantics, then scan history |
| "my model" | Model artifact | `api.artifact("entity/project/name:alias")` |
| "my dataset" | Dataset artifact | `api.artifact("entity/project/name:alias")` |
