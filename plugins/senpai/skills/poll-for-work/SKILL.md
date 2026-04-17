---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: poll-for-work
description: >
  Poll for assigned experiment PRs for a student. Use this skill to: check for assignments, poll for work, see if
  there's a PR assigned to me. Triggers for: "any work for me?", "check for assignments", "poll for PRs".
argument-hint: "<student-name>"
context: fork
model: claude-sonnet-4-6
effort: high
---

# poll-for-work

Check whether the advisor has assigned you an experiment PR to work on. You can use the Claude Code `Monitor` tool here to check for assigned PRs.

## Arguments

- **$0** — Your student name (e.g. `fern`)

## Steps

1. **Source the library and query** for assigned experiment PRs:

```bash
source "${CLAUDE_PLUGIN_ROOT}/scripts/senpai-gh.sh"
student_poll_for_work "$0"
```

2. **If PRs are returned**, for each one:
   - Note the PR number, title, and branch name
   - Check for advisor comments (this might be a revision, not a fresh assignment):
     ```bash
     gh pr view <number> --comments
     ```

3. **Return a summary:**

If work is available:
```
WORK_AVAILABLE: PR <pr-number> "<pr-title>" on branch <branch-name>
```

If the PR has revision comments from the advisor, include that:
```
WORK_AVAILABLE (REVISION): PR <pr-number> "<pr-title>" on branch <branch-name> — advisor requests: "<advisor-comment>"
```

If nothing:
```
NO_WORK
```

Keep the response short — the parent agent just needs to know whether to start working or keep waiting.
