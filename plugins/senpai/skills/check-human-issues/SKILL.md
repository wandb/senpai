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
model: claude-opus-4-8
effort: high
---

# check-human-issues

Check GitHub Issues tagged `human` for messages from the research team, and respond to any that need a reply.

## Arguments

- **$0** — The configured advisor branch for `ADVISOR`, or student name for
  `STUDENT`
- **$1** — Either `ADVISOR` or `STUDENT`

## How it works

Human researchers communicate with agents through GitHub Issues. Issues are
tagged with `human` plus `team` for a broadcast, the configured advisor branch
for an advisor, or `student:<student-name>` for a student. Your job is to check
messages routed to your exact role, respond to new ones, and skip ones you've
already handled.

## Steps

1. **Read the current `human_issue` event.** The controller supplies the issue
   identity and the exact human message ID that triggered the wake. It polls
   GitHub for issues addressed to you or the whole team.

2. **Decide whether to respond:**
   - If you haven't commented on this issue yet → respond.
   - If you have commented, check if the human posted a new comment *after* your last response. If so → respond to the new message. If not → skip, you're waiting for the human.
   - Record the exact numeric `id` of the issue body or human comment you are
     answering. Never substitute the issue number for a comment ID.

3. **Respond** through `respond_to_human_issue` with the issue number, the exact
   `human_message_id`, and the response text without a role prefix. This verified,
   idempotent operation refuses closed issues, pull requests, missing `human`
   labels, stale message IDs, messages authored by the agent identity, and
   issues not addressed to this configured advisor branch or student.

```json
{
  "issue_number": 123,
  "human_message_id": 987654,
  "response": "<your response>"
}
```

Never mutate the issue through `gh` or `curl`.

4. **Never close human issues.** Only the human does that.

## Return format

When you're done, return a structured summary of the issues you checked and responded to:

### New research directives from the human researcher team

If there are research directives in the issues, include them in detail in your summary so the parent agent can incorporate them into planning.
