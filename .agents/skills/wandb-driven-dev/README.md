# wandb-driven-dev

A Claude Code **skill** that enforces **experiment-driven development** for ML
codebases. Every empirical claim is backed by a W&B run, a baseline to compare
against, and a falsifier written before the run started. Nothing merges on vibes.

This used to be a bundled plugin; it has been exploded into two standalone
project skills plus two project agents, all auto-discovered from `.claude/`.

## Layout

```
.claude/
├── skills/
│   ├── wandb-driven-dev/          # this skill — the methodology (Phases 0–6)
│   │   ├── SKILL.md
│   │   ├── scripts/{wdd_helpers.py, setup_project.py, curve_analysis.py,
│   │   │            create_report.py, watch_runs.py, bootstrap_experiment.sh}
│   │   └── templates/wandb-driven-dev.local.md.template
│   └── wbagent/                   # standalone W&B query skill (also usable on its own)
│       ├── SKILL.md
│       ├── scripts/wandb_helpers.py
│       ├── references/{WANDB_SDK.md, WANDB_CONCEPTS.md, REPORTS.md}
│       └── .upstream-commit
└── agents/
    ├── wandb-query.md             # off-thread W&B analysis
    └── reviewer.md                # Phase-5 experiment verdict drafting
```

## The two skills

| Skill | Trigger | Purpose |
|---|---|---|
| `wandb-driven-dev` | `/wandb-driven-dev` (also auto-triggers on "experiment", "ablation", "is A better than B", setup/reconfigure, and W&B Report requests) | The methodology — Phases 0–6 from setup to cleanup, with project-local launcher config, training entrypoint, worktree bootstrap, smoke gates, launch, ETA-aware watcher, review, and experiment report helpers. |
| `wbagent` | Auto-triggered on W&B queries | Standalone toolkit for querying W&B runs, summaries, configs, histories, artifacts, sweeps, and reports through `wandb_helpers.py`. |

`wandb-driven-dev` builds on `wbagent` for all W&B querying — its scripts import
`wbagent/scripts/wandb_helpers.py` and its docs point at
`wbagent/references/REPORTS.md`. `wbagent` has no dependency on
`wandb-driven-dev` and can be used on its own for any W&B query.

Cross-skill paths are referenced relative to the repo root
(`.claude/skills/<skill>/...`). The Python scripts also self-resolve the sibling
`wbagent` location via `__file__`, so imports work regardless of the caller's
`sys.path`.

## Agents

| Agent | When | Purpose |
|---|---|---|
| `wandb-query` | On-demand | Off-thread analysis of a W&B project or run. Frees the main thread from large query outputs. |
| `reviewer` | Spawned by the `wandb-driven-dev` skill in Phase 5 | Reads the staged result, validates numbers against fresh W&B summaries, drafts the verdict and merge recommendation. |

## Quick start

```
/wandb-driven-dev setup
```

Claude interviews you for the W&B project, repo-specific experiment launcher
command, training entrypoint, reproduction model, GPU budgets, and
decision/health metrics, then writes `.claude/wandb-driven-dev.local.md`.
Subsequent invocations read it.

For a new experiment:

```
/wandb-driven-dev
```

Claude walks you through hypothesis → design → smoke → launch → review →
cleanup, gating each phase. Wandb runs use the `exp/<slug>` tag and
`exp-<slug>-<role>` name convention so they're trivially filterable.

## Prerequisites

- `wandb` and `pandas` Python packages on the Python you're running
- A W&B account with API key configured (`wandb login` or `WANDB_API_KEY`)
- For remote training: access to the runner, scheduler, or cluster used by the
  launcher command recorded in project config

## Project config schema

`.claude/wandb-driven-dev.local.md` (per-project, gitignored):

```markdown
---
wandb_project: entity/project
launcher:
  # Project-specific command to start/submit training; not W&B Launch.
  command: uv run python scripts/train.py
  reproduction: working_tree   # working_tree | clone | shared_fs | image
training:
  # Underlying training entrypoint used for --help flag validation.
  script: scripts/train.py
  config_dir: configs/
gpus:
  smoke: 1
  full: 8
metrics:
  decision: [val/loss, val/accuracy]
  health: [train/loss, train/grad_norm]
curves:
  # W&B default. Override per metric/namespace when the project logs semantic
  # step metrics such as train/global_step or stage_4e/epoch.
  default_step_key: _step
  metric_step_keys:
    train/*: train/global_step
    val/*: train/global_step
  candidate_step_keys: [_step, train/global_step]
wandb_metadata: {}
---

# Free-form project notes

The body of this file is read by the agents as context. Use it for things that
don't fit the structured schema: dataset quirks, known-good baseline run IDs,
project-specific gotchas.
```

## wbagent provenance

`wbagent` is vendored from the upstream W&B core repository at
`services/wb_agent/src/agent_repository/context_content/production/wbagent/skills/wbagent`.
The original upstream base commit is recorded in `wbagent/.upstream-commit`, but
this copy intentionally carries local query-helper extensions in
`wbagent/scripts/wandb_helpers.py`. Do not overwrite `wbagent/` with a blind
upstream sync; port local improvements to upstream manually and then reconcile
the vendored copy deliberately.
