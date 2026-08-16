---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

name: senpai-status-check
description: Produce a fresh, read-only status report for a Senpai research track from GitHub, W&B, and the current agent's local controller state.
---

# Senpai status check

Report progress against the configured `program.md` and fleet activity now. Keep scientific progress separate from infrastructure liveness, and distinguish observed evidence from inference.

## Establish scope

Resolve these values from the environment before collecting evidence:

- `GH_REPO`
- `ADVISOR_BRANCH`
- `RESEARCH_TAG`
- `WANDB_ENTITY`
- `WANDB_PROJECT`
- `TARGET_WORKDIR`
- `SENPAI_OPENHANDS_STATE_DIR`
- current UTC time

If a required value is absent, record an evidence gap. Never substitute a
remembered repository, branch, cluster, project, dataset, or metric.

Read the `program.md` identified in the system prompt for its goals, metric contracts, benchmarks, training constraints, and permitted reporting paths. Derive every metric and benchmark in the report from that file or live evidence.

## Collect bounded evidence

1. **Metrics:** use the `wandb-primary` skill against
   `$WANDB_ENTITY/$WANDB_PROJECT`. Start with run IDs linked from active PRs,
   then inspect only the recent runs needed to establish the test frontier.
2. **GitHub:** use `get_prs` for the configured `$GH_REPO` and
   `$ADVISOR_BRANCH`. Check assignment, workflow state, update time, linked W&B
   runs, and results. Keep the query bounded to relevant open and recently
   completed work.
3. **Current controller and training:** inspect
   `$SENPAI_OPENHANDS_STATE_DIR/training/*.json` for supervised training state.
   Treat this as local evidence for the current advisor or student only. Do not
   infer another node's process state; its GitHub transitions and W&B run state
   are the portable evidence boundary.

Cross-check timestamps, branch assignments, terminal training records, W&B run
state, and PR claims. State contradictions explicitly. Missing access or absent
records are evidence gaps, not evidence that nothing happened. Senpai uses no
cross-node RPC or cluster API for status collection.

This workflow is observational. Do not mutate GitHub, Kubernetes, W&B, local
state, or agent sessions.

## Report

Return a compact report with:

1. **Scope and evidence gaps:** exact repo, branch, research tag, W&B project,
   collection time, and anything unavailable.
2. **Executive read:** the scientific frontier and the main operational risk.
3. **Test metric frontier:** metric contract, best verified test value,
   benchmark or target, gap, W&B run, PR, and evaluation caveat.
4. **PR queue and routing:** active work, assignment integrity, stale work, and
   evidence-backed blockers.
5. **Runtime health:** current assignment, persisted local training and monitor
   state, W&B run liveness, stale GitHub transitions, and any
   branch-to-assignment mismatch.
6. **Contradictions and confidence:** where GitHub, W&B, persisted state, logs,
   or local controller state disagree.
7. **Next actions:** the one to three highest-value actions, clearly labeled as
   recommendations rather than changes already made.

Put paper-facing test metrics before validation metrics unless `program.md`
defines a different publication contract. Pair every claimed result with its
source and caveat; never promote a validation-only or partial-evaluation result
to a test result.

If `program.md` permits a repository status artifact and the user requested
one, write:

```text
analysis/STATUS_<YYYY-MM-DD-HHMM>_<branch>_fleet.md
```

Otherwise, return the report in the conversation without creating files.
