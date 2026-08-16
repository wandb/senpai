---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: delegate-subagents
description: >
  Delegate bounded research, inspection, synthesis, planning, review, or command
  execution when spawn_agents is available. Use when independent work can run
  in parallel or a fresh specialist perspective would improve a decision.
---

# Delegate subagents

Batch independent tasks in one `spawn_agents` call with a stable `batch_key`. Give each child a self-contained assignment and ask for a compact, evidence-linked result. Normally use `include_context=false`.

Choose:

- `explore` for local code, data, artifacts, or history;
- `search_general_web` for current public sources;
- `search_research_publications` for scholarly literature and primary papers;
- `bash-runner` with `model=fast` for tests, builds, and bounded commands; and
- `general-purpose` for mixed analysis, planning, review, or implementation.

Use `fast` for mechanical work, `smart` for subtle synthesis, and `frontier` for the hardest judgment. For a fresh independent perspective, use `model="frontier"`, `agent="general-purpose"`, and `include_context=false`; ask for research, critique, ideas, or a plan rather than edits. The search agent and its search skills own source selection and search mechanics.

`spawn_agents` returns task IDs immediately. Continue useful work, then use bounded `await_agents` calls with `all`, `first`, `quorum`, or `change` and a timeout of at most 300 seconds. Use `agent_status` for one non-blocking snapshot and `cancel_agents` when work is no longer useful; do not poll.
