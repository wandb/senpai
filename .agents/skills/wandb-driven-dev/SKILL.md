---
name: wandb-driven-dev
description: "Enforce a systematic, reproducible approach to empirical questions — hypothesis first, baseline + variant, smoke before full, wandb as source of truth. Use whenever the user asks 'does X work?', 'is A better than B?', 'what's the best N?', or says 'experiment', 'ablation', 'benchmark', 'sweep'. Also use for first-run setup/reconfigure of a repo's experiment launcher command, training entrypoint, reproduction model, GPU budgets, W&B metrics, and for W&B Reports from experiment runs."
user-invocable: true
argument-hint: "[setup | reconfigure | review experiment <slug> | <empty>]"
allowed-tools: Read, Write, Edit, Bash, Grep, Glob, Agent
model: opus
---

# Experiment-Driven Development

Any empirical claim is backed by a wandb run, a baseline to compare against, and a falsifier written before the run started. Nothing merges on vibes. If a required piece is missing (hypothesis, baseline, passing smoke), stop and surface the gap — don't work around it.

## When to invoke

Any empirical question. Also invoke for W&B Report, dashboard, or publishable
run-comparison requests. Skip if the answer is already in existing code, logs,
or wandb runs — check first.

## Setup mode

If invoked as `setup` (or `reconfigure`), run **Phase 0a** to interview the user and write the project config. Otherwise proceed normally; if config is missing, run Phase 0a first.

## Project config

`.claude/wandb-driven-dev.local.md` holds the per-project preferences gathered once and reused for every experiment. The file is YAML frontmatter (structured fields: wandb project, launcher command, default metrics, GPU counts, training entrypoint, curve step keys, and W&B project metadata) followed by a markdown body with free-form project notes the agents read verbatim. Schema lives in `scripts/wdd_helpers.py:default_config`. Read it first thing on every invocation:

```python
import sys
sys.path.insert(0, ".claude/skills/wandb-driven-dev/scripts")
from wdd_helpers import read_config, curve_step_keys
read_config()  # None if not yet configured
```

W&B always has `_step`; keep that as the generic default. If a project logs
semantic step metrics, persist them under `curves`. `curve_analysis.py compare`
reads these mappings automatically; pass an explicit `--step-key` only for
one-off at-step/debug commands.

The launch fields are the repo's own experiment-starting contract, not W&B
Launch:

- `launcher.command`: exact command to copy-paste when starting or submitting a
  training run (`uv run python scripts/train.py`, `sbatch ...`, `k8s/launch.py`,
  etc.).
- `launcher.reproduction`: how that command gets the code onto compute
  (`working_tree`, `clone`, `shared_fs`, or `image`).
- `training.script`: the underlying training entrypoint used for `--help` flag
  validation and plan citations. It may differ from `launcher.command` when the
  launcher is a wrapper or cluster submit script.

```yaml
curves:
  default_step_key: _step
  metric_step_keys:
    train/*: train/global_step
    val/*: train/global_step
  candidate_step_keys: [_step, train/global_step]
```

## Wandb querying delegated to wbagent

Don't reinvent wandb queries. Generic query helpers live in
`.claude/skills/wbagent/scripts/wandb_helpers.py`; experiment
gate helpers live in `scripts/wdd_helpers.py`. Standard preamble:

```python
import sys
sys.path.insert(0, ".claude/skills/wbagent/scripts")
sys.path.insert(0, ".claude/skills/wandb-driven-dev/scripts")
from wandb_helpers import (
    get_api, build_filters, fetch_runs, fetch_run_summaries, count_runs,
    compare_configs, scan_history, scan_history_until_step, compare_runs_at_step,
)
from wdd_helpers import (
    find_runs_by_config, verify_required_metrics,
    find_run_by_name, runtime_estimate, validate_flags,
)
```

wandb-driven-dev helpers (config IO and experiment report construction) live in `scripts/wdd_helpers.py`.

## W&B query recipes via wbagent

For recurring query shapes, use the `wbagent` helper module directly. It keeps
queries bounded with selected GraphQL fields, lazy counts, and explicit history
keys:

```python
import sys
sys.path.insert(0, ".claude/skills/wbagent/scripts")
from wandb_helpers import (
    get_api, build_filters, fetch_runs, count_runs, compare_runs_at_step,
)

api = get_api()
project = "milieu/drivaerml"

# Exact count, server-side, no run materialization.
count = count_runs(
    api,
    project,
    build_filters(["summary_metrics.train/global_step=100000"]),
)

# Top-k run lookup with selected summary/config fields.
rows = fetch_runs(
    api,
    project,
    metric_keys=["val/surface_rel_l2", "train/global_step"],
    filters=build_filters([
        "config.model.model_class=abupt",
        "config.max_steps=20000",
    ]),
    config_keys=["max_steps", "model"],
    order="+summary_metrics.val/surface_rel_l2",
    limit=2,
)

# Compare pinned runs at the latest logged step <= target step.
comparison = compare_runs_at_step(
    api,
    project,
    run_ids=["srehoxzc", "bo2blqjb"],
    step=15305,
    step_key="train/global_step",
    metrics=[
        "train/loss",
        "val/surface_rel_l2",
        "val/volume_rel_l2",
        "val/u_rel_l2",
        "val/loss",
    ],
)
```

Use these helpers before delegating to `wandb-query` for: "best run matching
filters", "how many runs match X", and "compare these runs at step N". Delegate
only when the question needs broader interpretation after the helper query
returns.

## Curve analysis primitives

When the human question is really "what does the plot say?", use
`scripts/curve_analysis.py` instead of eyeballing W&B charts or dumping full
history. It fetches only selected runs/metrics, converts rows to pandas frames,
and returns curve characteristics: value at target steps, local slope, percent
movement over the slope window, trend, best run by value, and best run by slope.
Slope/trend are noise-aware: by default the script uses a trailing rolling
median over 5 logged points, fits a line across the selected window, and reports
`noise_std`, `noise_to_signal`, and `trend_confidence`.

Curve questions must choose a training stage:

- `--stage early`: launch/smoke monitoring. Checks that the metric has enough
  points, starts moving in the right direction, has acceptable directional
  consistency, and does not show bad-direction spikes.
- `--stage progress`: mid/late training comparison. Uses local slope,
  slope-shift versus the previous window, spike detection, and noise/confidence
  fields to decide whether a run is improving, plateauing, or regressing.
- `--stage auto`: uses `--early-step-threshold` to route small target steps to
  early checks and later target steps to progress checks.

```bash
uv run python .claude/skills/wandb-driven-dev/scripts/setup_project.py
```

Step-key discovery is setup-time work and must go through
`scripts/setup_project.py`, not `curve_analysis.py`. That script reads the
local config, uses configured decision/health metrics, samples a few recent
finished runs when explicit run IDs are not supplied, then writes `curves` and
`wandb_metadata.preflight`. Store keys and decisions only; do not persist run
summaries or metric values in the local config.

If summary coverage is ambiguous, rerun setup discovery with
`--validate-history`; that slower path follows `wbagent` large-project rules by
using `get_api(timeout=120)`, explicit `scan_history(keys=...)`, bounded
`--max-rows`, and targeted metric+candidate scans only when sparse metrics need
coverage confirmation. Do not write ad hoc `run.history()` or all-candidate
history loops for step-key discovery.

`curve_analysis.py compare` is intentionally config-driven. In the normal path,
do not pass project, step key, smoothing, worker, or window arguments; setup
already stored the project and metric step-key mapping. The script groups
metrics by configured step key internally.

```bash
uv run python .claude/skills/wandb-driven-dev/scripts/curve_analysis.py \
  compare \
  --runs srehoxzc,bo2blqjb \
  --metrics train/loss,val/surface_rel_l2,val/volume_rel_l2,val/u_rel_l2,val/loss \
  --steps 15305
```

Use this for review/kill decisions such as "is this run going bad?", "which run
is improving faster?", and "which validation curve is better at this budget?".
The script uses latest logged point at or before each requested step; this
handles train/validation metrics that log on different cadences.

For 2-3 curves, this should return in a few seconds on normal W&B latency. For
larger comparisons, the implementation supports up to ~20 selected curves by
fetching runs in parallel with separate W&B API clients per worker.

When curves are noisy, prefer `smoothed_value` and `trend_confidence` for kill
or keep-going decisions. Treat a `low` confidence worsening slope as "watch
longer / inspect more context" unless the raw value is already outside the
falsifier threshold.

For just-launched runs, use `stage_analysis[*].status` from the early stage
instead of ranking by raw value; a brand-new run can be "healthy" before it is
competitive. For mature comparisons, use the progress stage fields
`current.slope_per_1k`, `effective_slope_shift_per_1k`, and
`spikes.bad_spike_count` alongside the best-run ranking.

## Reports

For any request to create or update a W&B Report, dashboard, or publishable run
comparison, use `scripts/wdd_helpers.py` for experiment-specific report
construction. For the authoring recipe (query strategy → narrative arc → chat
reply), a runnable skeleton, Runset filters, ordering, explicit run selection,
the `RunComparer` panel, and the rendering gotchas that crash the report UI,
read `.claude/skills/wbagent/references/REPORTS.md`. Reports use
the supported `wandb_workspaces.reports.v2` API (not the deprecated
`wandb.apis.reports` alias).

Do not delegate experiment comparison report creation to `wbagent`; use
`wbagent` only to query/discover runs and metrics.

## Off-thread analysis

For any analysis that touches more than ~5 runs or scans history, delegate to
the `wandb-query` agent via the Agent tool. It runs queries off the main
thread and returns a structured summary. Use it for:

- "What runs match this filter and how do their metrics compare?"
- "Has anyone in this project hit metric X > Y?"
- "Diagnose why this run crashed/diverged."

Don't delegate single-run lookups or anything you can answer in one
`run.summary_metrics.get(...)` call — the round-trip overhead isn't worth it.

## Run identification (mandatory)

Every wandb run launched by this skill must be findable by a single workspace filter:

- **Tag:** every launch passes `--wandb_tags exp/<slug>` (smoke runs additionally tag `smoke`; variants additionally tag `variant`).
- **Name:** wandb run name is `exp-<slug>-<role>`, where `<role>` ∈ `{smoke-baseline, smoke-variant, baseline, variant, variant-<id>}`. If the launcher names the underlying job, use the same string.

If the launcher recorded in `cfg.launcher.command` doesn't thread `--wandb_tags`/`--wandb_name` through to the training script, fix the launcher *before* running experiments — don't bypass identification to ship faster.

## Phases

Each has a gate; do not advance until met.

### 0a. First-run setup (gate: config written)

Skip if `.claude/wandb-driven-dev.local.md` exists. Otherwise interview the user via `AskUserQuestion`, then write the config:

1. **wandb project** (e.g. `entity/project`).
2. **Experiment launcher command** — exact string: a local invocation (`uv run python scripts/train.py …`) or a cluster submit (`uv run python k8s/launch.py …`, `sbatch slurm/train.sh …`). This is the repo's launcher, not W&B Launch; the skill only cares how to reproduce it.
3. **Reproduction model** — how does the launcher get the code? `working_tree` (local, runs uncommitted edits as-is), `clone` (submitter clones origin/HEAD; needs clean tree + pushed commit), `shared_fs` (compute mounts the working tree), `image` (built/pushed before submit).
4. **Training entrypoint + config dir** — the script that actually parses training flags, plus config directory if any. Use this for `--help` validation even when `launcher.command` is a wrapper.
5. **GPU defaults** — smoke (usually 1) and full (project-typical).
6. **Decision metrics** — exact wandb keys driving pass/fail. Probe the project before asking, so the question is grounded:
   ```python
   from wandb_helpers import get_api, probe_project
   probe_project(get_api(), "<entity>/<project>")
   ```
   If the project has zero finished runs, surface that — usually a typo'd slug.
7. **Health metrics** — keys watched for divergence/NaN.
8. **Curve step keys** — default to `_step`; if the project logs semantic
   steps such as `train/global_step` or `stage_4e/epoch`, run setup discovery
   after decision/health metrics are known:
   ```bash
   uv run python .claude/skills/wandb-driven-dev/scripts/setup_project.py
   ```
   This reads `.claude/wandb-driven-dev.local.md`, samples recent finished runs,
   pulls selected summaries, and writes stable mappings under
   `curves.metric_step_keys` plus metadata under `wandb_metadata.preflight`.
9. **Project notes (optional)** — anything you'd want future-you to know that doesn't fit the schema (dataset quirks, known-good baseline IDs, launcher gotchas). Goes into the markdown body of `.local.md`.

Write with `wdd_helpers.write_config(cfg, notes=...)` and confirm the file path back to the user.

### 0b. Worktree (gate: on `exp/<slug>` branch with plan.md scaffold)

Experiments never run from the main checkout. Bootstrap:

```bash
WT="$(.claude/skills/wandb-driven-dev/scripts/bootstrap_experiment.sh <slug> | tail -1)"
cd "$WT"
```

Slug convention: `YYYYMMDD-<short-kebab>`. Script refuses to clobber an existing branch or worktree, and runs `uv sync` automatically if a `pyproject.toml` is present. If already in a worktree on `exp/*`, keep working there.

### 1. Hypothesis (gate: user approval of plan.md)

Most context comes from the project config. Per-experiment, ask only:

- **Baseline status:** to-be-created (default), fresh re-run on this branch, or pinned existing wandb run. For pinned baselines, verify the chosen decision metrics already exist on it:
  ```python
  run = get_api().run("<entity>/<project>/<pinned-id>")
  {k: run.summary_metrics.get(k) for k in cfg["metrics"]["decision"]}
  ```
- **Per-experiment metric overrides** — only if this experiment cares about something beyond `cfg.metrics`.

Write `experiments/<slug>/plan.md`:

```markdown
# <slug>

**Date:** YYYY-MM-DD
**Question:** one sentence — what are we trying to learn?

## Hypothesis
What we expect to see, stated as a prediction.

## Falsifier
The concrete observation that would prove the hypothesis wrong.
Measurable on wandb: metric key + threshold + which runs.

## Success criteria
Quantitative bar for "variant wins".

## Baseline
- **Status:** to-be-created | fresh re-run | pinned
- **Wandb run:** URL or `TBD — recorded in Phase 4`
- **Why this is the right control:** one line.

## Variant
The change(s) from baseline. Default: one knob, two runs (baseline + variant). For multi-variant benchmarks (≥3 runs comparing one named dimension like architecture or hidden_dim), list each variant; verdict becomes a ranking table. Same data/eval/budget across runs.

## Metrics
Inherited from `.claude/wandb-driven-dev.local.md` unless overridden here.
- **Decision:** ...
- **Health:** ...

## Report Columns
<!-- Optional. Bullet lines only; prose is ignored by the parser. -->
<!-- List config inputs changed and summary outputs you care about. -->
<!-- Example: -->
<!-- - config.lr, config.model.depth, val/loss -->

## Design        (filled in Phase 2)
## Smoke         (filled in Phase 3)
## Runs          (filled in Phase 4)
## Result        (filled in Phase 5)
```

Show plan.md and wait for explicit approval.

### 2. Design (gate: user approval of budget + commands)

Pick the **smallest step budget** that beats the falsifier with comfortable margin — don't default to the config's `max_steps`.

| Question type | Typical budget |
| --- | --- |
| LR / warmup / optimizer stability | 500–2k |
| Scaling laws | 1k–3k |
| Loss / target scheme | 2k–5k |
| Architecture change | 5k–10k |
| Final paper-number claim | full `max_steps` |

Write into `## Design`:

- launch command for baseline (exact, copy-pasteable, with `--max_steps` override if needed) — uses `cfg.launcher.command`
- launch command(s) for variant(s) — same budget, only the named dimension changes
- step budget + one-line rationale
- expected wall-clock + GPU count

**Flag validation gate (mandatory before writing the commands).** `simple_parsing` flattens nested dataclasses, so YAML's `model: {slice_num: ...}` is exposed as `--slice_num`, NOT `--model.slice_num`. A typo here costs a smoke round. Validate every flag against the training script's `--help` first:

```python
from wdd_helpers import validate_flags
proposed = ["--max_steps", "--slice_num", "--n_layers"]  # whatever your variant overrides
result = validate_flags(f"uv run {cfg['training']['script']}", proposed)
assert not result["missing"], f"Phase 2 flag gate failed: {result['missing']}"
```

If any flag is missing, fix the command before writing it into `## Design` — don't paper over it.

Optionally use `runtime_estimate(api, project, name_pattern, target_steps, min_steps=...)` for the wall-clock projection. Pass `min_steps` (e.g. `target_steps // 10`) to keep smoke runs out of the throughput sample.

### 3. Smoke (gate: clean exit AND every metric in `## Metrics` shows up in wandb)

Shrunk config for both baseline and variant: `max_steps: 50–200`, `num_gpus: cfg.gpus.smoke`, `val_every_n_steps` small enough to fire once. Always pass `--wandb_tags smoke,exp/<slug>` and `--wandb_name exp-<slug>-smoke-{baseline,variant}` through the launcher.

Each smoke must:
1. exit 0
2. log ≥1 train step and ≥1 val step
3. not NaN / diverge on first val

**Verify from training-process logs, not the launcher's success signal** — many launchers report success while the underlying training crashed. Capture stdout/stderr to `experiments/<slug>/logs/` and grep for failure markers (`Traceback`, `CUDA out of memory`, `NaN`, `ChildFailedError`).

**Verify metric logging via wandb.** Use the wbagent helper:

```python
required = cfg["metrics"]["decision"] + cfg["metrics"]["health"]  # plus any per-experiment additions
result = verify_required_metrics(get_api(), [
    "<entity>/<project>/<smoke-baseline-id>",
    "<entity>/<project>/<smoke-variant-id>",
], required)
assert all(not missing for missing in result.values()), f"Phase 3 metric gate failed: {result}"
```

If a required metric is missing: instrument the log call, fix eval cadence, or — only with explicit user acknowledgment — drop the metric and re-smoke.

### 4. Launch (fire-and-forget)

Submit baseline and variant with the approved commands. Use `--wandb_tags exp/<slug>,<role>` and `--wandb_name exp-<slug>-<role>`. If the baseline is pinned, only launch the variant.

Create the live dashboard via the hardcoded report workflow. It resolves run
URLs from W&B and `plan.md`, validates every decision/health metric, creates a
draft report (pass `--publish` to save non-draft), and splices run/report URLs
into `## Runs`. The report x-axis defaults to `cfg.curves.default_step_key`
(falling back to `_step`); override with `--x-axis` if needed:

```bash
uv run python .claude/skills/wandb-driven-dev/scripts/create_report.py <slug>
```

To show only the experiment inputs and outputs you care about in the focused
comparison table, either fill `## Report Columns` in `plan.md` or pass them:

```bash
uv run python .claude/skills/wandb-driven-dev/scripts/create_report.py \
    <slug> --columns config.lr,config.model.depth,val/loss
```

The report always includes the RunComparer panel. Focused columns add a compact
table above the plots; they do not replace the full diff panel.

If the Reports SDK grows a RunComparer column allowlist, wire `--columns` into
that API directly. Until then, keep RunComparer visible and use the focused
table as the curated view.

For pinned-baseline or multi-variant experiments, record any pinned run URL in
`plan.md` first, or pass explicit required roles:

```bash
uv run python .claude/skills/wandb-driven-dev/scripts/create_report.py \
    <slug> --roles baseline,variant-a,variant-b
```

The script prints JSON with `report_url`, `runs`, metric-check output, and
`plan_updated`. Treat nonzero exit as a Phase 4 gate failure.

**Spawn the watcher** (optional, only when expected runtime fits in ~1 hour). `scripts/watch_runs.py` projects ETA from each run's `train/global_step` + `_runtime`, sleeps until ~70% of the slowest ETA, then enters an **adaptive** poll loop where each subsequent sleep is `0.5 × current max ETA` (floored at 60 s). The wait shrinks naturally as runs converge — typically ~3–5 polls total instead of a constant 2-minute interval. Hard `--max_wait_min` deadline. Burns ~zero LLM tokens (Python loop, not an agent).

```bash
SLUG=<slug>
LOGS=/path/to/worktree/experiments/$SLUG/logs
uv run python .claude/skills/wandb-driven-dev/scripts/watch_runs.py \
    $SLUG --logs_dir $LOGS \
    --target_steps <budget> --max_wait_min 60
```

Use `Bash(run_in_background=True)` so it survives between conversation turns. Watcher writes:
- `logs/05-watch.json` — live status (overwritten each poll).
- `logs/05-staged-result.md` — draft `## Result` block once all runs are terminal. Phase 5 reads this; it never edits plan.md directly.

Skip the watcher if the run budget is multi-hour (it'll hit the deadline) or if you'd rather rely solely on wandb account alerts at https://wandb.ai/settings → Alerts. Both are valid; wandb alerts also cover crash/divergence per-run.

Then exit:

> Launched. Watcher running; say `review experiment <slug>` when you're ready, or you'll see the staged verdict next time we talk.

Don't try to *block* in-conversation — Bash backgrounding and subagent lifetimes both cap below the runtime of a multi-hour training job. The watcher pattern works because it's bounded; multi-hour experiments still need wandb alerts.

### 5. Review (delegate to reviewer agent)

When the user types `review experiment <slug>`, **delegate to the `reviewer` agent** via the Agent tool. The agent:

1. Reads `experiments/<slug>/plan.md`, `experiments/<slug>/logs/05-staged-result.md` (if present), and `.claude/wandb-driven-dev.local.md`.
2. Pulls fresh wandb summaries for the experiment runs.
3. Validates that the only config difference between baseline and variant is the named dimension (`compare_configs`).
4. Drafts the `## Result` block (verdict, key numbers, against-hypothesis, merge recommendation).
5. Returns the proposed `## Result` markdown — does NOT auto-write it.

```python
# Example invocation from the main thread:
Agent(
    description=f"Review experiment {slug}",
    subagent_type="reviewer",
    prompt=(
        f"Review experiment '{slug}'. The plan is at "
        f"experiments/{slug}/plan.md (in the current worktree). "
        f"The staged result, if any, is at experiments/{slug}/logs/05-staged-result.md. "
        f"Project config is at .claude/wandb-driven-dev.local.md. "
        f"Return the proposed ## Result block as markdown — do not write it."
    ),
)
```

Then **show the proposed verdict to the user** and ask for approval before splicing into `plan.md`. After approval:

- **Pass** → splice `## Result`, push branch, `gh pr create`, link plan.md + wandb URLs. After merge, run Phase 6.
- **Fail** → splice `## Result`, run Phase 6. Optionally cherry-pick plan.md into main's `experiments/` as a record.
- **Inconclusive** → splice `## Result`, stay in worktree, propose smallest next run, loop back to Phase 1. Skip Phase 6 until terminal.

### 6. Cleanup (gate: worktree removed, branch resolved)

From the **main checkout**:

```bash
# pass — branch merged via PR
git worktree remove ../<repo>-exp/<slug>
git branch -d exp/<slug>
git push origin --delete exp/<slug>  # only if pushed

# fail / abandoned
git worktree remove ../<repo>-exp/<slug>
git branch -D exp/<slug>
git push origin --delete exp/<slug>  # only if pushed
```

If `git worktree remove` refuses, investigate uncommitted changes before `--force` — they may be unsaved log captures.

## Output capture

Training-process output is ephemeral. Tee every command into `experiments/<slug>/logs/NN-<name>.txt` (numbered for chronological replay): smoke submit, smoke run logs, baseline submit, variant submit, etc. Commit `experiments/<slug>/` at each phase boundary — the commit history is the audit trail.

## Anti-patterns (call them out)

- "I'll just run it and see" — write the falsifier first.
- "Compare to last week's numbers" — re-run the baseline on the current branch unless pinned and justified.
- "Skip smoke, I'm confident" — 10 min of smoke saves 6h of wasted GPU.
- "Change optimizer AND LR AND batch size in one variant" — that's an uncontrolled change. Either name a single dimension or make it an explicit multi-variant benchmark.
- "It looks worse, let it finish" — surface it, let the user decide.
- "Use the config's max_steps for everything" — pick the smallest budget that answers the question.

If the user insists on skipping a gate, state the trade plainly ("I can skip smoke but we won't know if the 6h run crashes on step 1 — proceed?") and require explicit override.

## Iterating on this skill

After a successful invocation, look for patterns worth lifting into `wbagent/scripts/wandb_helpers.py` (generic) or `scripts/wdd_helpers.py` (wandb-driven-dev-specific). Slow paths or missing checks belong in code; phase logic belongs here.
