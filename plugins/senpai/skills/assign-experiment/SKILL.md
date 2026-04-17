---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: assign-experiment
description: >
  Create a branch and draft PR to assign a hypothesis to a student.
  Handles branch creation, PR creation with the correct template and
  labels, and ensures the student gets a well-structured assignment.
  Use this skill to: assign an experiment, create a hypothesis PR,
  give a student work.
argument-hint: "<student-name> <hypothesis-slug>"
model: claude-sonnet-4-6
effort: high
---

# assign-experiment

Create a branch and draft PR that assigns a hypothesis to a student. The student will pick it up on their next poll cycle.

## Arguments

- **$0** — The student to assign (e.g. `fern`)
- **$1** — A short kebab-case slug for the hypothesis (e.g. `cosine-annealing`)

The hypothesis details, instructions, and baseline metrics come from your own reasoning — this skill handles the git/GitHub mechanics.

## Steps

1. **Start from the latest advisor branch:**

```bash
ADVISOR_BRANCH="${ADVISOR_BRANCH}"  # from environment
git checkout "$ADVISOR_BRANCH" && git pull origin "$ADVISOR_BRANCH"
```

2. **Create the experiment branch:**

```bash
BRANCH="$0/$1"
git checkout -b "$BRANCH"
git push -u origin "$BRANCH"
```

3. **Create the draft PR** with the template below. Replace the placeholders with your actual hypothesis, instructions, and baseline data:

```bash
gh pr create --draft \
    --title "<Hypothesis title — clear, specific, under 70 chars>" \
    --body "$(cat <<'PREOF'
## Hypothesis
<What we think will improve metrics and why. For non-trivial changes,
include links to papers or code that support the hypothesis.>

## Instructions
<Specific changes to make to cfd_tandemfoil/train.py — be concrete.
"Try a higher learning rate" is vague. Change lr from 5e-4 to 1e-3 and add cosine annealing with T_max=epochs" is actionable.>

## Baseline
<Current best metrics from BASELINE.md:
- val/loss: X.XXX
- Surface MAE metrics: p_in | p_oodc | p_tan | p_re
- Baseline W&B run: <run-id> (<wandb-link>)
- Reproduce command: `cd cfd_tandemfoil && python train.py ...`>
PREOF
)" \
    --label "$ADVISOR_BRANCH" \
    --label "student:$0" \
    --label "status:wip" \
    --base "$ADVISOR_BRANCH" \
    --head "$BRANCH"
```

## Important details

- **Read BASELINE.md** before creating the PR take the most recent metrics from the file. The student needs concrete metrics to compare against.
- **Be specific in instructions.** The student implements exactly what you write. Vague instructions waste GPU time.
- **Use `--wandb_group`** in instructions when a hypothesis needs multiple iterations (e.g. "try surface weight 5, 10, 20") so related runs are grouped in W&B. The student's harness will add its own `--wandb_tag` when it backgrounds runs.
- **One hypothesis per PR.** Bundling multiple changes makes it impossible to attribute what worked.
- If the PR body is too long for `gh pr create`, put the core info in the body and add supplementary details as a follow-up comment.
