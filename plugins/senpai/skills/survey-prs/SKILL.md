---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: survey-prs
description: >
  Survey all experiment PRs on a branch and return a structured status
  report: which students are idle, which PRs await review, which are WIP.
  This is the heartbeat query — use it to understand the current state of
  the research track. Triggers for: "survey state", "check PR status",
  "who's idle", "any PRs ready for review", "what's the current state".
context: fork
model: claude-sonnet-4-6
effort: high
allowed-tools: Bash(gh *), Bash(source *), Bash(python3 *)
---

# survey-prs

Get a snapshot of the current state of the research track — who's working on what, what's ready for review, and who needs an assignment.

Uses `$ADVISOR_BRANCH` and `$STUDENT_NAMES` from the environment (set by the k8s ConfigMap).

## Steps

1. **Source the library and query:**

```bash
source "${CLAUDE_PLUGIN_ROOT}/scripts/senpai-gh.sh"

# All open PRs on the branch
list_all_prs "$ADVISOR_BRANCH"

# Just the review-ready ones
list_ready_for_review_prs "$ADVISOR_BRANCH"

# Who's idle
list_idle_students "$STUDENT_NAMES" "$ADVISOR_BRANCH"
```

2. **Categorize** each PR by its status labels:
   - `status:review` — ready for advisor review
   - `status:wip` — student is working on it (note which student from the `student:*` label)
   - Draft with no status — may be newly created or stalled

3. **Return a structured summary** in this format:

```markdown
## Current Experiment PR State — <branch>

### Review-ready
- <pr-number> "<pr-title>" (student:<student-name>)
- <pr-number> "<pr-title>" (student:<student-name>)

### Work in progress
- <pr-number> "<pr-title>" (student:<student-name>) — WIP

### Idle students (no status:wip PR)
- <student-name> (last PR <pr-number> is in review)

### Summary
Review-ready: <number-of-review-ready-prs> | WIP: <number-of-wip-prs> | Idle: <number-of-idle-students>
```

Keep it compact. The parent agent uses this to decide what to do next.
