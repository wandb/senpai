---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

name: monitor-jobs
description: Configure durable, context-silent monitoring for an active long-running job. Use after launching or discovering training, evaluation, inference, or another W&B-backed job when completion, failure, staleness, or a decision-changing metric boundary should resume the conversation without manual polling.
---

# Monitor jobs

Use `monitor_job`; never poll, sleep, or tail logs while waiting.

Monitor terminal state and at most three metrics. For each metric, use its exact
W&B key, `min` or `max` direction, only gates that would change the next action,
and a stale timeout longer than its expected logging interval. Use `lte` and
`gte` for absolute boundaries. `improved_by` and `regressed_by` compare with
the policy's first observed sample and require `direction`. Omit metric
policies when completion is enough.

Students pass the job ID returned by `run_job` and, for metric policies, the
exact associated `wandb_run_id`. It may be omitted only when the job has one
unambiguous associated W&B run; the selected run is validated and durably bound
when the policy is registered. Advisors pass a configured-project W&B run ID as
`job_id`; this grants no launch, immediate-status, or cancellation authority.

After registration, continue unrelated work or finish the turn. Ordinary
checks run in the background and add nothing to model context. A deduplicated
gate, staleness, monitor error, or terminal state becomes a priority event for
the next safe turn; inspect only the evidence needed, act, then revise the
policy only when its decision boundary changes.
