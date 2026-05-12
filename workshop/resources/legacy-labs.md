# Optional Paper Labs

These group exercises are retained from the earlier workshop design. They are useful when credentials or network access are unavailable, or when a facilitator wants discussion-based labs instead of runnable notebooks.

## Lab 1: Turn A Plausible Idea Into A Falsifiable Hypothesis

Scenario:

> Try a larger Transolver with cosine learning-rate decay and a surface-pressure-weighted loss.

Task:

- Name the hypothesis.
- Name the mechanism.
- Name the primary metric.
- Name the split or benchmark contract.
- Name allowed files.
- Name the falsifier.

Expected insight:

- A plausible idea is not research evidence.
- A GPU-worthy assignment needs metric, split, allowed files, logging, and falsifier.

Related notebook: `workshop/notebooks/01_llm_calls_to_hypotheses.py`

## Lab 2: Raw Command Or Workflow Wrapper?

Compare:

```bash
gh pr ready 1842
gh pr edit 1842 --remove-label status:wip --add-label status:review
```

with:

```bash
source "${CLAUDE_PLUGIN_ROOT}/scripts/senpai-gh.sh"
mark_ready_for_review 1842
```

Expected insight:

- The wrapper requires a terminal `SENPAI-RESULT`.
- The wrapper uses safe label transition semantics.
- The raw command can mark unverified work as ready.

Related notebook: `workshop/notebooks/02_tool_calls_and_contracts.py`

## Lab 3: Write A Student Result Handoff

Given:

- Baseline `test_primary/surface_pressure_rel_l2_pct = 6.24`
- Run `driv-ema-042` has test `6.31`
- Validation improved, test regressed

Task:

- Write a result marker.
- Decide if it is terminal.
- Explain whether advisor should merge.

Expected insight:

- Validation-only gain does not merge when held-out test is the merge contract.

Related notebook: `workshop/notebooks/03_single_agent_student_loop.py`

## Lab 4: Route A PR Queue

Given PRs with labels:

- `student:ws-fern`, `status:wip`
- `student:ws-fern`, `status:wip`
- `student:ws-robin`, `status:review`
- missing advisor branch label

Task:

- Identify duplicate assignments.
- Identify review-ready PRs.
- Identify malformed PRs.
- Decide advisor next action.

Expected insight:

- Labels are queue infrastructure.
- Duplicate WIP assignments are distributed-system bugs.

Related notebook: `workshop/notebooks/04_multiagent_advisor_student_flow.py`

## Lab 5: Reconcile PR, W&B, And Weave

Given:

- PR result prose says the run is promising.
- W&B test metric regresses.
- Weave trace shows advisor verified W&B.

Task:

- Identify authoritative evidence for metric, code, workflow, and agent behavior.
- Decide merge, request changes, or close.

Expected insight:

- W&B is metric truth.
- Weave is decision-path truth.
- PR comments are workflow truth.

Related notebook: `workshop/notebooks/05_wandb_and_weave_as_memory.py`

## Lab 6: Inspect A Dry-Run Launch

Given a dry-run manifest, classify:

- research identity
- routing
- resources
- data access
- observability
- secrets

Expected insight:

- A launch command establishes research identity, queue state, GPU economics, and observability wiring.

Related notebook: `workshop/notebooks/06_k8s_autoresearch_dry_run.py`
