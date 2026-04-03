---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: check-human-issues
description: >
  Check and respond to GitHub Issues from the human researcher team.
  Runs in a forked context (no access to main conversation). Use this skill whenever you need to: check for human
  messages, respond to human issues, poll for team communications,
  check GitHub issues. Also triggers for: "any human messages?",
  "check issues", "respond to humans".
argument-hint: "<name> <ADVISOR|STUDENT>"
context: fork
model: claude-opus-4-6
effort: high
---

# check-human-issues

Check GitHub Issues tagged `human` for messages from the research team, and respond to any that need a reply.

## Arguments

- **$0** — Your name/label for gh issue filtering (e.g. `noam`, `fern`)
- **$1** — Either `ADVISOR` or `STUDENT`

The comment prefix is: if role is `ADVISOR` → `ADVISOR:`, if role is `STUDENT` → `STUDENT $0:`.

## How it works

Human researchers communicate with agents through GitHub Issues. Issues are tagged with `human` plus either your name or `team` (for broadcast messages). Your job is to check them, respond to new ones, and skip ones you've already handled.

## Steps

1. **Source the senpai-gh library** and list issues using **name** as the label filter:

```bash
source "${CLAUDE_PLUGIN_ROOT}/scripts/senpai-gh.sh"
check_gh_issues "$0"
```

This returns a deduplicated JSON array of issues addressed to you and the whole team.

2. **For each issue**, read the full body and comments:

```bash
gh issue view <number> --json body,comments
```

3. **Decide whether to respond:**
   - If you haven't commented on this issue yet → respond.
   - If you have commented, check if the human posted a new comment *after* your last response. If so → respond to the new message. If not → skip, you're waiting for the human.

4. **Respond** with your role prefix:

```bash
# ADVISOR example:
gh issue comment <number> --body "ADVISOR: <your response>"
# STUDENT example:
gh issue comment <number> --body "STUDENT $0: <your response>"
```

5. **Never close human issues.** Only the human does that.

## Return format

When you're done, return a structured summary of the issues you checked and responded to:

### New research directives from the human researcher team

If there are research directives in the issues, include them in detail in your summary so the parent agent can incorporate them into planning.
