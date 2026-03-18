---
name: assign-work
description: Use this skill to distribute research hypotheses to idle students by creating experiment branches and draft PRs. For each idle student, creates a branch from the advisor branch, pushes it, and opens a labeled draft PR with the hypothesis, instructions, and baseline metrics.
---

# Assign Work to Students

## Environment variables

These are set via the ConfigMap and available in all shell commands:
- `$ADVISOR_BRANCH` — the branch you work on (PRs target this as base)
- `$STUDENT_NAMES` — comma-separated list of student names
- `$WANDB_ENTITY` / `$WANDB_PROJECT` — W&B coordinates
- `$RESEARCH_TAG` — current research tag

For each idle student (no `status:wip` PR), create a branch and draft PR assigning them a hypothesis.

## Steps per assignment

1. Ensure you're on the latest advisor branch:
```bash
git checkout $ADVISOR_BRANCH && git pull origin $ADVISOR_BRANCH
```

2. Create an experiment branch:
```bash
git checkout -b $ADVISOR_BRANCH/<hypothesis-name>
git push -u origin $ADVISOR_BRANCH/<hypothesis-name>
```

3. Create a labeled draft PR:
```bash
gh pr create --draft \
  --title "<hypothesis title>" \
  --body "<PR body — use template below>" \
  --label "$ADVISOR_BRANCH" --label "student:<name>" --label "status:wip" \
  --base $ADVISOR_BRANCH --head $ADVISOR_BRANCH/<hypothesis-name>
```

4. Return to the advisor branch before the next assignment:
```bash
git checkout $ADVISOR_BRANCH
```

## PR body template

Every PR must follow this structure:

```markdown
## Hypothesis
<what we think will improve metrics and why>

## Instructions
<specific changes to make to train.py — be concrete>

## Baseline
<current best metrics for reference>

---

## Results
_To be filled by student_
```

## Writing good instructions

- **Be specific.** "Try a higher learning rate" is vague. "Change lr from 5e-4 to 1e-3 and add cosine annealing with T_max=epochs" is actionable.
- **Always include baseline metrics.** Students need a concrete target to compare against.
- **Use `--wandb_group`** in instructions when a hypothesis needs multiple iterations (e.g., trying several values of the same hyperparameter).
- **One hypothesis per PR.** Bundling changes makes it impossible to attribute what worked.

## Matching hypotheses to students

- If there are more hypotheses than idle students, pick the most promising ones.
- If there are more idle students than hypotheses, note which students still need work — the advisor should generate more ideas.
- Avoid assigning the same idea to multiple students. Check what's already in-flight.
