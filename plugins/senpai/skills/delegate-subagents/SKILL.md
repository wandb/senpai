---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: senpai

name: delegate-subagents
description: >
  Choose, configure, launch, and collect bounded subagents for research and
  engineering decisions. Root agents and delegation-capable subagents should
  read this before delegating: every task requires an explicit model tier, and
  high-leverage work such as research ideation, round planning, plateau pivots,
  large research reviews, hard optimization, disputed evidence, and expensive
  experiment portfolios requires frontier judgment.
---

# Delegate subagents

Batch independent tasks in one `spawn_agents` call with a stable `batch_key`. Give each child a self-contained assignment and ask for a compact, evidence-linked result.

Every task must explicitly set `model` to `fast`, `smart`, or `frontier`; there is no implicit model tier. Choose the least expensive tier that provides the judgment the task requires:

- Use `model="fast"` for mechanical, easily verified work: locating files, extracting facts, counting records, formatting results, running bounded commands, and reporting test or build failures.
- Use `model="smart"` for ordinary implementation and review, literature retrieval, bounded synthesis, standard failure diagnosis, and debugging code that is not already heavily optimized.
- Use `model="frontier"` for quality-first research judgment. This includes fresh research ideation, planning a new research round, changing direction after a plateau, reviewing a large research or experiment history, difficult debugging or optimization of code that is already highly optimized, reconciling conflicting evidence or disagreement between local and external evaluation, and selecting a portfolio that will consume substantial GPU time or external-evaluation budget.

Keep frontier tasks focused. When using more than one for the same decision, give them distinct questions or perspectives. Do not spend frontier capacity on routine monitoring, simple retrieval, formatting, or other work whose answer is cheap to verify. Treat every child result as advice: inspect its evidence before acting on it.

## Choose an agent and context

Choose:

- `explore` for local code, data, artifacts, or history;
- `search_general_web` for current public sources;
- `search_research_publications` for scholarly literature and primary papers;
- `bash-runner` with `model=fast` for tests, builds, and bounded commands; and
- `general-purpose` for mixed analysis, planning, review, or implementation.

Agent specialization and model tier are independent. For first-principles synthesis, critique, diagnosis, or planning, use `agent="general-purpose"`. Use `search_research_publications` when the task is to find and compare primary papers, and `search_general_web` for current public sources. The search agent and its search skills own source selection and search mechanics.

Normally set `include_context=false` and provide a self-contained task with exact evidence paths. This gives the child a fresh perspective while preserving access to the merged system prompt and searchable parent history. Set `include_context=true` only when the complete model-visible conversation is necessary and cannot be summarized reliably. For research judgment, ask for research, critique, diagnosis, ideas, or a plan rather than edits.

## Collect results

`spawn_agents` returns task IDs immediately. Continue useful work, then use bounded `await_agents` calls with `all`, `first`, `quorum`, or `change` and a timeout of at most 300 seconds. Use `agent_status` for one non-blocking snapshot and `cancel_agents` when work is no longer useful; do not poll.
