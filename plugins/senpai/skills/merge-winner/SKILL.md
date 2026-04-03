---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: merge-winner
description: >
  Squash-merge a winning experiment PR and update the baseline. Handles
  the merge, BASELINE.md update, commit, push, and branch pull. Also
  handles merge conflicts by sending the PR back for rebase. Use this
  skill to: merge a winning PR, update baseline, squash merge experiment.
  Triggers for: "merge winner", "merge this PR", "update baseline after
  merge", "squash merge".
argument-hint: "<pr-number>"
model: claude-sonnet-4-6
effort: high
---

# merge-winner

A PR beat the baseline — merge it and record the new best metrics in BASELINE.md. Comment on the PR saying your are merging it and why if there isn't already a comment.

## Arguments

- **$0** — The PR number (e.g. `1842`)

Note: `$ADVISOR_BRANCH` is available as an environment variable.

## Steps

1. **Squash-merge the PR:**

```bash
gh pr merge $0 --squash
```

If this fails due to merge conflicts, send the PR back for rebase instead:

```bash
source "${CLAUDE_PLUGIN_ROOT}/scripts/senpai-gh.sh"
send_pr_back_to_student_with_comment $0 "ADVISOR: Rebasing needed — the advisor branch was updated after merging a previous winner. Please rebase onto $ADVISOR_BRANCH, re-run the experiment to verify the improvement still holds, and resubmit."
```

Then stop — don't proceed with baseline update.

2. **Pull the updated branch:**

```bash
git checkout "$ADVISOR_BRANCH" && git pull origin "$ADVISOR_BRANCH"
```

3. **Update BASELINE.md** by appending the new baseline entry. Read the winning metrics from the PR comments and the W&B run metrics. The entry should include:

```markdown
## <YYYY-MM-DD HH:MM> — PR #<number>: <title>

- **Surface MAE:** Ux=X.XXXX, Uy=X.XXXX, p=X.XXXX
- **val/loss:** X.XXX
- **W&B run:** <run-id>
- **Reproduce:** `cd cfd_tandemfoil && python train.py <full command>`
```

4. **Commit and push the baseline update:**

```bash
git add BASELINE.md
git commit -m "Update baseline: <short description> (PR #$0)"
git push origin "$ADVISOR_BRANCH"
```

## After merging

All subsequent PR reviews in this round should compare against the **newly updated baseline**, not the old one. If you're reviewing multiple winners in a batch, merge them best-first and update the baseline after each one.
