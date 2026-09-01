---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: review-experiment
description: >
  Review one terminal experiment and choose the next step: merge a reproducible
  winner, close a useful negative or dead end, or request missing evidence.
argument-hint: "<pr-number> <problem-dir>"
---

# Review an experiment

Retrieve the complete PR with `get_prs`. Verify its assignment, current head
SHA, terminal structured result, W&B evidence, metric direction, and scientific
conclusion. Then choose the appropriate next step:

- merge a reproducible improvement;
- close a completed control, clean negative, invalid candidate, or bounded dead
  end with a clear reason; or
- request a revision when the evidence is incomplete or another bounded run is
  required.

## Merge a winner

```json
{
  "assignment": {
    "pr_number": 123,
    "assignment_id": "assignment-id",
    "revision_id": "current-revision-id",
    "expected_pr_head_sha": "CURRENT_PR_HEAD_SHA"
  },
  "expected_current_base_sha": "CURRENT_BASE_SHA",
  "merge_method": "squash"
}
```

Call `merge_experiment`. It refuses drafts, missing or foreign results, stale
heads, blocking labels, unknown mergeability, and conflicts. It compares the
assignment's base SHA with the live base branch immediately before merging. If
the research base changed, reassess the exact terminal result against the
event's `current_base_sha`.

If the conclusion still holds, record that decision before merging:

```json
{
  "assignment": {
    "pr_number": 123,
    "assignment_id": "assignment-id",
    "revision_id": "current-revision-id",
    "expected_pr_head_sha": "CURRENT_PR_HEAD_SHA"
  },
  "expected_current_base_sha": "CURRENT_BASE_SHA",
  "reason": "Why this exact result remains valid against the current research base."
}
```

Call `accept_result_on_current_base` with that payload. Its durable acceptance
is bound to the assignment, revision, result head, canonical structured result,
and exact current base SHA. If any of them changes, reassess. If the conclusion
no longer holds, request a new revision instead. Never invent a SHA or call
`gh pr merge`.

## Close a completed non-winner

```json
{
  "assignment": {
    "pr_number": 123,
    "assignment_id": "assignment-id",
    "revision_id": "current-revision-id",
    "expected_pr_head_sha": "CURRENT_PR_HEAD_SHA"
  },
  "reason": "Concise scientific reason for closing this experiment."
}
```

Call `close_experiment`. Distinguish a useful negative from an invalid or
incomplete run. Do not edit labels, write protocol markers, or close the PR with
`gh`.

## Request a revision

```json
{
  "assignment": {
    "pr_number": 123,
    "assignment_id": "assignment-id",
    "revision_id": "current-revision-id",
    "expected_pr_head_sha": "CURRENT_PR_HEAD_SHA"
  },
  "new_revision_id": "new-revision-id",
  "required_base_sha": "CURRENT_BASE_SHA",
  "comment": "Exact missing evidence and the bounded next run required."
}
```

Call `request_assignment_revision`. Use a new stable revision ID and the exact
research base SHA the next revision must use. State one concrete change or
experiment and its acceptance evidence; do not close an experiment that can
still answer the assigned question with one bounded correction.

## Record the outcome

Update the baseline or research log in the format prescribed by `program.md`, including the PR, metrics, run IDs and links, reproduction command, and conclusion. Commit that advisor-owned change and publish it only through:

```json
{
  "remote_branch_sha_before_push": "REMOTE_SHA_BEFORE_PUSH",
  "local_commit_sha": "LOCAL_COMMIT_SHA"
}
```

Call `publish_advisor_branch`; it publishes only the configured advisor branch.

Review multiple candidates strongest-first and refresh the baseline between
each decision.
