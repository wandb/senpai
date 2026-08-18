---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: assign-experiment
description: >
  Create a typed assignment branch and draft PR for one student. Use when the
  advisor has a concrete hypothesis and the student has no open `status:wip`
  or `status:review` assignment.
argument-hint: "<student-name> <hypothesis-slug> <problem-dir>"
model: claude-sonnet-4-6
effort: high
---

# Assign an experiment

Read the current baseline and `program.md`, then write one complete assignment:

- a falsifiable hypothesis and mechanism;
- exact files and changes in scope;
- baseline metrics and W&B evidence;
- commands, run limits, metrics, and stopping conditions; and
- one student, one branch, and one experiment.

Fetch the configured advisor branch and record its exact remote SHA. Call
`create_assignment` with this operation-specific payload; the runtime supplies
and enforces the configured advisor branch and student allowlist:

```json
{
  "assignment_id": "stable-assignment-id",
  "revision_id": "initial-revision-id",
  "student": "student-name",
  "expected_base_sha": "FETCHED_REMOTE_ADVISOR_SHA",
  "head_branch": "student-name/hypothesis-slug",
  "title": "Concise hypothesis title",
  "body": "Complete actionable experiment brief"
}
```

The tool creates an empty assignment commit without changing the advisor
worktree, pushes the branch with a lease, creates or reconciles one draft PR,
adds the exact routing labels, embeds the typed assignment marker, verifies the
result, and rejects a second active assignment for the student.

Do not create the branch, PR, labels, or assignment marker through shell
commands. If the tool reports stale state, refresh the named SHA and
re-evaluate rather than bypassing the precondition.
