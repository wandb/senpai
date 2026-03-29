---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: senpai-gh
description: >
  GitHub CLI primitives for the senpai research workflow — label swaps,
  send-back, close, mark-review, issue checks, PR queries. Use this skill
  whenever you need to manipulate PR labels, send a PR back to a student,
  close a dead-end experiment, mark a PR for review, or query the current
  state of PRs and issues. Also triggers for: "swap labels", "send back to
  student", "close this PR", "mark for review", "check human issues",
  "list review-ready PRs", "idle students".
user-invocable: false
allowed-tools: Bash(gh *), Bash(source *), Bash(python3 *)
---

# senpai-gh

A bash library of GitHub operations shared by both advisor and student agents. Source it once, then call the functions you need.

The library lives at `${CLAUDE_PLUGIN_ROOT}/scripts/senpai-gh.sh`. Source it before using any function:

```bash
source "${CLAUDE_PLUGIN_ROOT}/scripts/senpai-gh.sh"
```

## Why this exists

GitHub's `gh pr edit --remove-label X --add-label Y` silently strips **all other labels** from the PR. The only safe way to swap a single label is two REST API calls: DELETE the old one, POST the new one. This library wraps that pattern so you never have to remember it.

## Available functions

### Label operations

| Function | What it does |
|---|---|
| `senpai_label_swap <pr#> <remove> <add>` | Atomically swap one label for another. Safe — won't error if the old label is already gone. |

### Compound actions (advisor)

| Function | What it does |
|---|---|
| `senpai_send_back <pr#> <comment>` | Comment on the PR, convert back to draft, swap `status:review` → `status:wip`. |
| `senpai_close_pr <pr#> <reason>` | Comment with reason, close the PR, delete the remote branch. |

### Compound actions (student)

| Function | What it does |
|---|---|
| `senpai_mark_review <pr#>` | Mark the PR as ready + swap `status:wip` → `status:review`. |

### Queries

| Function | What it does |
|---|---|
| `senpai_repo` | Print `owner/repo` from the git remote (cached). |
| `senpai_check_issues <role_label>` | List open human issues for a role label + team issues, deduplicated. Returns JSON array. |
| `senpai_list_review_prs <branch>` | List PRs with `status:review` on a branch. Returns JSON array. |
| `senpai_list_all_prs <branch>` | List all open PRs on a branch (any status). Returns JSON array. |
| `senpai_poll_work <student_name>` | List WIP PRs assigned to a student. Returns JSON array. |
| `senpai_idle_students <names_csv> <branch>` | Print names of students with no `status:wip` PR, one per line. |

## Usage examples

```bash
source "${CLAUDE_PLUGIN_ROOT}/scripts/senpai-gh.sh"

# Advisor sends a PR back for revision
senpai_send_back 1842 "ADVISOR: Promising direction but didn't beat baseline. Try lr=1e-3 with cosine annealing."

# Student marks PR ready for review
senpai_mark_review 1842

# Advisor closes a dead end
senpai_close_pr 1850 "results 12% worse than baseline with no promising signal"

# Check who's idle
senpai_idle_students "frieren,fern,stark" "noam"

# List PRs awaiting review
senpai_list_review_prs "noam"
```
