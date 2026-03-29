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
argument-hint: "<branch> <student-names-csv>"
context: fork
allowed-tools: Bash(gh *), Bash(source *), Bash(python3 *)
---

# survey-prs

Get a structured snapshot of where things stand — who's working on what, what's ready for review, and who needs an assignment.

## Arguments

- `$0` — The advisor branch name (e.g. `noam`)
- `$1` — Comma-separated student names (e.g. `frieren,fern,stark`)

## Steps

1. **Source the library and query:**

```bash
source "${CLAUDE_PLUGIN_ROOT}/scripts/senpai-gh.sh"

# All open PRs on the branch
senpai_list_all_prs "$0"

# Just the review-ready ones
senpai_list_review_prs "$0"

# Who's idle
senpai_idle_students "$1" "$0"
```

2. **Categorize** each PR by its status labels:
   - `status:review` — ready for advisor review
   - `status:wip` — student is working on it (note which student from the `student:*` label)
   - Draft with no status — may be newly created or stalled

3. **Return a structured summary** in this format:

```markdown
## PR Survey — <branch>

### Review-ready
- #1842 "Cosine annealing with warm restarts" (student:frieren)
- #1855 "Spectral normalization on decoder" (student:fern)

### Work in progress
- #1860 "Gradient-weighted loss" (student:stark) — WIP

### Idle students (no status:wip PR)
- fern (last PR #1855 is in review)

### Summary
Review-ready: 2 | WIP: 1 | Idle: 1
```

Keep it compact. The parent agent uses this to decide what to do next.
