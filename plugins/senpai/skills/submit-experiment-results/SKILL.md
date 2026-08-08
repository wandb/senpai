---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: submit-experiment-results
description: >
  Commit and submit a terminal experiment result for advisor review through the
  typed Senpai GitHub result tool.
argument-hint: "<pr-number> <problem-dir>"
model: claude-sonnet-4-6
effort: high
---

# Submit experiment results

First make the target worktree clean by committing only the assigned change.
Collect the current local commit SHA and the current remote assignment-branch
SHA. Build the strict `ExperimentResult` required by
`submit_experiment_result`:

- the assignment repository, PR, assignment ID, revision ID, student, and
  current expected head SHA;
- terminal status, hypothesis, and bounded summary;
- every W&B run ID, URL, and terminal state;
- the primary metric comparison; and
- the same local commit SHA.

Call `submit_experiment_result` with exactly this operation-specific payload. Do not add PR-body
text, per-run metrics, hyperparameters, or aliases such as `head_sha`,
`previous_head_sha`, `success`, or `min`; put those details in the bounded
`summary` instead.

```json
{
  "branch": "student/experiment",
  "remote_branch_sha_before_push": "REMOTE_SHA_BEFORE_PUSH",
  "result": {
    "assignment": {
      "repo": "owner/repo",
      "pr_number": 123,
      "assignment_id": "assignment-id",
      "revision_id": "LATEST_REVISION_ID",
      "expected_head_sha": "LOCAL_COMMIT_SHA",
      "student": "student-name"
    },
    "status": "succeeded",
    "hypothesis": "The falsifiable hypothesis tested.",
    "summary": "The conclusion, evidence, caveats, and important per-run metrics (maximum 4,000 characters).",
    "runs": [
      {
        "run_id": "wandb-run-id",
        "url": "https://wandb.ai/entity/project/runs/wandb-run-id",
        "state": "finished"
      }
    ],
    "primary_metric": {
      "name": "validation/metric",
      "direction": "minimize",
      "baseline": 1.23,
      "candidate": 1.10,
      "delta": -0.13
    },
    "commit_sha": "LOCAL_COMMIT_SHA"
  }
}
```

Use one of `succeeded`, `failed`, `inconclusive`, or `cancelled` for result
status; `minimize` or `maximize` for metric direction; and `finished`,
`failed`, `crashed`, or `killed` for each run. `primary_metric` may be omitted
when no finite comparison exists. Refresh the PR first and use its latest
assignment `revision_id`; a mid-turn advisor revision supersedes an earlier
one.

That single tool call derives the PR and local result head from `result`,
lease-pushes the clean assignment branch, verifies the new PR head, upserts the
authenticated structured result, marks the PR ready,
and reconciles `status:review`. That label is the durable advisor notification.
Exact replay of the same result is safe. Once result evidence is published for
that revision and commit, changed evidence requires a new commit or assignment
revision; the tool will not rewrite the reviewed result in place.
Do not run `git push`, edit labels, write result markers, or call `gh pr ready`
yourself.

If any run is still active or could change the conclusion, keep the assignment
in progress. Start long-running work with `run_job`; it registers terminal-state
monitoring automatically. Use `monitor_job` only when an active W&B run also
benefits from a metric threshold or staleness policy. Do not submit a terminal
result until the evidence is terminal.
