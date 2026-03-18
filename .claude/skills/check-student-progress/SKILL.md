---
name: check-student-progress
description: Use this skill to survey the current research state and review student PRs. Queries W&B for baseline metrics, lists open PRs, identifies idle students, and reviews PRs marked status:review (merge winners, request changes on promising directions, close dead ends).
---

# Check Student Progress

## Environment variables

These are set via the ConfigMap and available in all shell commands:
- `$ADVISOR_BRANCH` — the branch you work on (PRs target this as base)
- `$STUDENT_NAMES` — comma-separated list of student names
- `$WANDB_ENTITY` / `$WANDB_PROJECT` — W&B coordinates
- `$RESEARCH_TAG` — current research tag

The `gh` CLI auto-detects the repo owner/name when run inside the clone — no need to hardcode `{owner}/{repo}`.

## 1. Survey current state

Query W&B for the best metrics so far. Identify the current baseline.

List all open PRs:
```bash
gh pr list --label "$ADVISOR_BRANCH" --json number,title,state,labels,headRefName,isDraft
```

Identify:
- Which students are idle (no `status:wip` PR)
- Which PRs are awaiting review (`status:review`)
- Which PRs are in progress (`status:wip`)

Report a concise summary before proceeding to reviews.

## 2. Review completed PRs (`status:review`)

Review **each PR individually** — never batch-close an entire round.

### a. Rank by metrics

Rank all review-ready PRs by `best_mae_surf_p` (lower is better). Check the W&B run for each PR — the student's reported metrics in the PR body may be stale or incomplete.

### b. Merge winners sequentially, best first

A PR is a winner if its `best_mae_surf_p` is lower than the current baseline. Merge aggressively — even small improvements compound.

For each winner, starting with the best:
```bash
gh pr merge <number> --squash
```

Update your baseline immediately. Pull the updated advisor branch before the next merge:
```bash
git checkout $ADVISOR_BRANCH && git pull origin $ADVISOR_BRANCH
```

If the next winner has **merge conflicts** (branched before a previous merge), send it back for rebase:
```bash
gh pr comment <number> --body "Rebasing needed: $ADVISOR_BRANCH was updated after merging PR #<merged>. Please rebase onto $ADVISOR_BRANCH, re-run the experiment to verify the improvement still holds, and resubmit."
gh pr ready <number> --undo
gh api repos/$(gh repo view --json nameWithOwner -q .nameWithOwner)/issues/<number>/labels/status:review --method DELETE
gh api repos/$(gh repo view --json nameWithOwner -q .nameWithOwner)/issues/<number>/labels -f "labels[]=status:wip" --method POST
```

### c. Request changes on promising directions

PRs that didn't beat baseline but show an interesting direction — leave specific feedback on what variation to try next:
```bash
gh pr ready <number> --undo
gh api repos/$(gh repo view --json nameWithOwner -q .nameWithOwner)/issues/<number>/labels/status:review --method DELETE
gh api repos/$(gh repo view --json nameWithOwner -q .nameWithOwner)/issues/<number>/labels -f "labels[]=status:wip" --method POST
```

### d. Close dead ends

Results significantly worse than baseline (>5% regression) or fundamentally broken:
```bash
gh pr close <number> --delete-branch
```

## Decision criteria

- **Merge** if `best_mae_surf_p` < current baseline (even small amounts — they compound)
- **Request changes** if promising direction but didn't beat baseline — suggest a variation
- **Close** only if clearly worse (>5% regression) or fundamentally broken (diverged, crashed)
- When in doubt between merge and close, **merge**

## Label management

**IMPORTANT:** Never use `gh pr edit --remove-label --add-label` — it strips other labels. Always use the API calls above to swap status labels individually.

## Output

After completing reviews, report:
- Which PRs were merged (and updated baseline)
- Which PRs were sent back with feedback
- Which PRs were closed
- Which students are now idle and need new work
