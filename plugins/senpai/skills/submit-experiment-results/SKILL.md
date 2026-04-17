---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: submit-experiment-results
description: >
  Submit experiment results for advisor review. Commits changes, pushes
  the branch, marks the PR as ready, and swaps the status label from
  wip to review, and clears any in-flight background-run tracking for the
  PR. Use this skill when you've finished running experiments and posted
  your results comment. Triggers for: "submit for review",
  "mark PR ready", "send results to advisor", "submit experiment results".
argument-hint: "<pr-number>"
model: claude-sonnet-4-6
effort: high
---

# submit-experiment-results

You've run the experiment, posted a results comment on the experiment PR — now wrap it up and hand it to the advisor.

## Arguments

- **$0** — The experiment PR number (e.g. `1842`)

## Before you call this

Make sure you've already posted a results comment on the experiment PR with metrics, W&B run ID, analysis, and suggested follow-ups

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

3. **Mark the PR as ready for advisor review and swap labels:**

```bash
source "${CLAUDE_PLUGIN_ROOT}/scripts/senpai-gh.sh"
mark_ready_for_review $0
```

4. **Clear any remaining harness tracking for this PR:**

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/in_flight.py" clear --pr $0
```

That's it. The advisor will pick it up in their next review cycle.
