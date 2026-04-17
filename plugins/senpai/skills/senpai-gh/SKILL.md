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
model: claude-opus-4-6
effort: high
---

# senpai-gh

A bash library of GitHub operations shared by both advisor and student agents. Source it once, then call the functions you need.

The library lives at `${CLAUDE_PLUGIN_ROOT}/scripts/senpai-gh.sh`. Source it before using any function:

```bash
source "${CLAUDE_PLUGIN_ROOT}/scripts/senpai-gh.sh"
```

## Available functions

### Label operations

| Function | What it does |
|---|---|
| `swap_gh_pr_label <pr#> <remove-label> <add-label>` | Atomically swap one label for another. Safe — won't error if the old label is already gone. |

#### The `gh pr edit` footgun

GitHub's `gh pr edit --remove-label X --add-label Y` silently strips **all other labels** from the PR. The only safe way to swap a single label is two REST API calls: DELETE the old one, POST the new one. This library wraps that pattern so you never have to remember it.

### Advisor actions

| Function | What it does |
|---|---|
| `send_pr_back_to_student_with_comment <pr#> <comment>` | Send a PR back to the student with feedback. Comment on the PR, convert back to draft, swap `status:review` → `status:wip`. |
| `close_pr_with_comment <pr#> <reason>` | Close a dead-end PR with a comment explaining why. Comment with reason, close the PR, delete the remote branch. |
| `comment_pr_with_file <pr#> <body-file>` | Comment on a PR using a markdown body file. Uses the shared `gh_retry` policy. |

### Student actions

| Function | What it does |
|---|---|
| `mark_ready_for_review <pr#>` | Mark a PR as ready for advisor review. Mark the PR as ready + swap `status:wip` → `status:review`. |

### Queries (both roles)

| Function | What it does |
|---|---|
| `print_gh_repo` | Print `owner/repo` from the git remote (cached). |
| `current_pr_number` | Print the open PR number for the current branch. |
| `check_gh_issues <role_label>` | List open human issues for a role label + team issues, deduplicated. Returns JSON array. |
| `list_ready_for_review_prs <branch>` | List PRs with `status:review` on a branch. Returns JSON array. |
| `list_all_prs <branch>` | List all open PRs on a branch (any status). Returns JSON array. |
| `student_poll_for_work <student_name>` | List WIP PRs assigned to a student. Returns JSON array. |
| `list_idle_students <names_csv> <branch>` | Print names of students with no `status:wip` PR, one per line. |

### Git state helpers

| Function | What it does |
|---|---|
| `require_clean_tracked_worktree` | Fail if the current worktree has tracked changes. |
| `require_pushed_head` | Fail if the current HEAD commit is not pushed to the branch upstream. |

## Usage examples

```bash
source "${CLAUDE_PLUGIN_ROOT}/scripts/senpai-gh.sh"

# Advisor sends a PR back for revision
send_pr_back_to_student_with_comment 1842 "ADVISOR: Promising direction but didn't beat baseline. Try lr=1e-3 with cosine annealing."

# Student marks PR ready for review
mark_ready_for_review 1842

# Advisor closes a dead end
close_pr_with_comment 1850 "results 12% worse than baseline with no promising signal"

# Check who's idle
list_idle_students "frieren,fern,stark" "noam"

# List PRs awaiting review
list_ready_for_review_prs "noam"
```
