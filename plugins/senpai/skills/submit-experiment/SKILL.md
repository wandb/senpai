---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: submit-experiment
description: >
  Submit experiment results for advisor review. Commits changes, pushes
  the branch, marks the PR as ready, and swaps the status label from
  wip to review. Use this skill when you've finished running experiments
  and posted your results comment. Triggers for: "submit for review",
  "mark PR ready", "send results to advisor", "submit experiment".
argument-hint: "<pr-number>"
allowed-tools: Bash(git *), Bash(gh *), Bash(source *)
---

# submit-experiment

You've run the experiment, posted a results comment on the PR — now wrap it up and hand it to the advisor.

## Arguments

- `$ARGUMENTS` — The PR number (e.g. `1842`)

## Before you call this

Make sure you've already:
1. Posted a results comment on the PR with metrics, W&B run ID, analysis, and suggested follow-ups
2. Run `/plot-experiment-charts` if applicable

## Steps

1. **Stage and commit your changes:**

```bash
git add cfd_tandemfoil/train.py
# Also add any other files you modified (pyproject.toml if you added packages, etc.)
git commit -m "<concise description of what you changed>"
```

2. **Push the branch:**

```bash
git push origin "$(git branch --show-current)"
```

3. **Mark ready and swap labels:**

```bash
source "${CLAUDE_PLUGIN_ROOT}/scripts/senpai-gh.sh"
senpai_mark_review $ARGUMENTS
```

That's it. The advisor will pick it up in their next review cycle.
