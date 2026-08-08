---
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: skills

name: senpai-tool-telemetry
description: Audit recent tool use across local Senpai OpenHands root and recursively delegated conversations without exposing tool arguments. Use for provider comparisons, tool-loop diagnosis, status-poll analysis, error rates, and latency reports.
---

# Senpai tool telemetry

Use this developer-only skill when inspecting a copied or mounted Senpai
OpenHands state tree. It is deliberately excluded from the skills installed
into advisor and student runtimes.

Run the read-only analyzer from the Senpai repository:

```bash
uv run python tools/senpai_tool_telemetry.py \
  /path/to/openhands_state \
  --hours 12 \
  --json /tmp/senpai-tool-telemetry.json
```

Pass multiple state roots to compare deployments in one report. The command
recursively discovers root and delegated-child `events/` directories. It
prints a compact table and writes structured JSON containing per
source/root-or-child-depth/role/model/tool calls, successes, errors, pending
calls, retained tool latency, explicit status-check volume, and identical-call
repetition signals.

Tool arguments are never included in the report. The analyzer creates a
short-lived in-memory fingerprint only to identify repeated identical calls.
Do not copy raw event bodies into a report merely to recover arguments.

Interpret the output carefully:

- `get_training_status` and `get_job_status` counts are model-issued tool
  calls. They do not include cheap controller-internal monitor polls.
- Latency is action-to-observation wall time when both timestamps survived.
- Model attribution comes from each conversation's current
  `base_state.json`. If a conversation changed models in place, older calls
  cannot be attributed more precisely from the event store alone.
- A pending call has no matching persisted observation. It may be genuinely
  active or evidence of an interrupted turn.
- A rapid repeat means the same conversation called the same tool with the
  same arguments inside the configured repetition window. It is a diagnostic
  signal, not automatically a bug.

Report evidence before recommendations. Separate provider behavior from
runtime failures such as dead terminal sessions, missing observations, or
schema errors.
