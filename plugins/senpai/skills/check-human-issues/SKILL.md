---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: check-human-issues
description: >
  Open GitHub Issues for human input and respond to the researcher team.
  Use this skill whenever you need to handle a human_issue event, respond to
  human issues, ask humans a question, or check team communications. Also triggers
  for: "any human messages?", "check issues", "respond to humans".
argument-hint: "<name> <ADVISOR|STUDENT>"
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

## Open an issue for human input

Use `create_human_issue` when you need a research decision or help with a blocker.
Provide a stable `issue_id`, a concise `title`, and the complete `body` without a
role prefix. Reuse the same ID, title, and body if the call needs a retry. Changed
content requires a new ID.

The runtime adds `human` and your configured audience label. It infers human
maintainers from the target repository's GitHub permissions and mentions them in
the initial issue body. Do not add maintainer mentions yourself. Later replies
do not repeat these automatic mentions.

```json
{
  "issue_id": "confirm-next-seed-budget",
  "title": "Confirm the budget for another seed",
  "body": "The first two seeds disagree. Can we use the remaining budget for a third seed?"
}
```

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
   labels, stale message IDs, Senpai protocol messages from the agent identity, and
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

When you're done, record a structured summary of the issues you checked and
responded to in the current conversation:

### New research directives from the human researcher team

If there are research directives in the issues, include them in detail so they
remain available for subsequent planning in this conversation.
